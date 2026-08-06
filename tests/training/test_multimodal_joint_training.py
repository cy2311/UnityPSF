from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from unity_psf.models import UnityPSF
from unity_psf.training.multimodal_joint import (
    DUAL_MODALITY_DUAL_CHANNEL_INSTANCE_KEYS,
    ExpertParallelPlan,
    ExpertTrainingUnit,
    MultimodalTrainingPlan,
    RoutedTrainingBatch,
    commit_joint_training_checkpoint,
    train_round_robin_epoch,
)


class _TrainableEmitter(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.tensor(value))

    def forward(self, images: torch.Tensor, conditions: torch.Tensor | None = None) -> torch.Tensor:
        return self.value * torch.ones((images.shape[0], 10, *images.shape[-2:]))


class _TrainableAstigmatism(nn.Module):
    condition_dim = 1

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.tensor(value))

    def forward(self, images: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        return (self.value + conditions[:, :1].reshape(-1, 1, 1, 1)) * torch.ones(
            (images.shape[0], 10, *images.shape[-2:])
        )


def _loss(output: torch.Tensor, batch: RoutedTrainingBatch) -> torch.Tensor:
    return (output[:, 0].mean() - batch.target).square()


def _model() -> UnityPSF:
    return UnityPSF(
        {
            "emitter_2d:main": _TrainableEmitter(1.0),
            "astigmatism:left": _TrainableAstigmatism(2.0),
            "astigmatism:right": _TrainableAstigmatism(3.0),
        }
    )


def _batch(target: float, *, astigmatism: bool = False) -> RoutedTrainingBatch:
    return RoutedTrainingBatch(
        images=torch.zeros(1, 3, 4, 4),
        conditions=torch.zeros(1, 1) if astigmatism else None,
        target=torch.tensor(target),
    )


def test_dual_modality_plan_has_one_2d_and_two_independent_astig_routes() -> None:
    plan = MultimodalTrainingPlan.dual_modality(
        step_budgets={
            "emitter_2d:main": 2,
            "astigmatism:left": 3,
            "astigmatism:right": 3,
        }
    )

    assert plan.instance_keys == (
        "emitter_2d:main",
        "astigmatism:left",
        "astigmatism:right",
    )
    assert plan.round_robin_schedule().count("emitter_2d:main") == 2
    assert plan.round_robin_schedule().count("astigmatism:left") == 3
    assert plan.round_robin_schedule().count("astigmatism:right") == 3
    assert ExpertParallelPlan.one_instance_per_rank(plan).rank_for("astigmatism:right") == 2


def test_dual_modality_dual_channel_plan_groups_each_modality_on_one_rank() -> None:
    plan = MultimodalTrainingPlan.dual_modality_dual_channel(
        step_budgets={key: 2 for key in DUAL_MODALITY_DUAL_CHANNEL_INSTANCE_KEYS}
    )

    parallel = ExpertParallelPlan.one_modality_per_rank(plan)

    assert plan.instance_keys == (
        "emitter_2d:left",
        "emitter_2d:right",
        "astigmatism:left",
        "astigmatism:right",
    )
    assert parallel.assignments == {
        0: ("emitter_2d:left", "emitter_2d:right"),
        1: ("astigmatism:left", "astigmatism:right"),
    }
    assert parallel.rank_for("emitter_2d:right") == 0
    assert parallel.rank_for("astigmatism:left") == 1


def test_round_robin_updates_only_selected_expert_and_records_separate_losses() -> None:
    model = _model()
    before_left = model.experts["astigmatism:left"].value.detach().clone()
    plan = MultimodalTrainingPlan.dual_modality(
        step_budgets={
            "emitter_2d:main": 1,
            "astigmatism:left": 0,
            "astigmatism:right": 0,
        }
    )
    units = {
        "emitter_2d:main": ExpertTrainingUnit(
            optimizer=torch.optim.SGD(model.experts["emitter_2d:main"].parameters(), lr=0.1),
            batches=(_batch(0.0),),
            loss_fn=_loss,
        ),
        "astigmatism:left": ExpertTrainingUnit(
            optimizer=torch.optim.SGD(model.experts["astigmatism:left"].parameters(), lr=0.1),
            batches=(),
            loss_fn=_loss,
        ),
        "astigmatism:right": ExpertTrainingUnit(
            optimizer=torch.optim.SGD(model.experts["astigmatism:right"].parameters(), lr=0.1),
            batches=(),
            loss_fn=_loss,
        ),
    }

    result = train_round_robin_epoch(model=model, plan=plan, units=units, epoch=1)

    assert result.step_counts == {
        "emitter_2d:main": 1,
        "astigmatism:left": 0,
        "astigmatism:right": 0,
    }
    assert model.experts["emitter_2d:main"].value.item() < 1.0
    assert torch.equal(model.experts["astigmatism:left"].value, before_left)
    assert model.activation_audit() == {"emitter_2d:main": 1}
    assert result.losses_by_instance["emitter_2d:main"] == (1.0,)


def test_round_robin_advances_all_three_routes_with_independent_optimizers() -> None:
    model = _model()
    plan = MultimodalTrainingPlan.dual_modality(
        step_budgets={key: 1 for key in (
            "emitter_2d:main",
            "astigmatism:left",
            "astigmatism:right",
        )}
    )
    units = {
        "emitter_2d:main": ExpertTrainingUnit(
            optimizer=torch.optim.SGD(model.experts["emitter_2d:main"].parameters(), lr=0.1),
            batches=(_batch(0.0),),
            loss_fn=_loss,
        ),
        "astigmatism:left": ExpertTrainingUnit(
            optimizer=torch.optim.SGD(model.experts["astigmatism:left"].parameters(), lr=0.1),
            batches=(_batch(0.0, astigmatism=True),),
            loss_fn=_loss,
        ),
        "astigmatism:right": ExpertTrainingUnit(
            optimizer=torch.optim.SGD(model.experts["astigmatism:right"].parameters(), lr=0.1),
            batches=(_batch(0.0, astigmatism=True),),
            loss_fn=_loss,
        ),
    }

    result = train_round_robin_epoch(model=model, plan=plan, units=units, epoch=1)

    assert result.step_counts == {key: 1 for key in plan.instance_keys}
    assert set(result.losses_by_instance) == set(plan.instance_keys)
    assert model.activation_audit() == {key: 1 for key in sorted(plan.instance_keys)}


def test_expert_parallel_plan_rejects_missing_required_rank() -> None:
    plan = MultimodalTrainingPlan.dual_modality(step_budgets={key: 1 for key in (
        "emitter_2d:main",
        "astigmatism:left",
        "astigmatism:right",
    )})
    parallel = ExpertParallelPlan.one_instance_per_rank(plan)

    with pytest.raises(ValueError, match="missing completed ranks"):
        parallel.validate_completed_ranks({0: "complete", 1: "complete"})


def test_joint_commit_rejects_incomplete_training_result(tmp_path: Path) -> None:
    model = _model()
    plan = MultimodalTrainingPlan.dual_modality(step_budgets={key: 1 for key in (
        "emitter_2d:main",
        "astigmatism:left",
        "astigmatism:right",
    )})

    with pytest.raises(ValueError, match="required instance"):
        commit_joint_training_checkpoint(
            tmp_path / "unitypsf_joint.ckpt",
            model=model,
            plan=plan,
            completed_instances=("emitter_2d:main", "astigmatism:left"),
            optimizers={},
            role="release",
        )
