from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from unity_psf.localization.conditioning import ConditioningProviderStore
from unity_psf.training.high_fidelity.diagnostics import path_token
from unity_psf.training.high_fidelity.physical_state import write_current_physical_state
from unity_psf.training.loop_types import TrainingRunEpochResult, TrainingStepResult


@dataclass(frozen=True)
class DomainPeakLayout:
    base: Any
    domain_name: str

    @property
    def run_dir(self):
        return self.base.run_dir

    @property
    def logs_dir(self):
        return self.base.logs_dir

    @property
    def cache_dir(self):
        return self.base.cache_dir

    @property
    def metadata_dir(self):
        return self.base.metadata_dir

    @property
    def checkpoints_dir(self):
        return self.base.checkpoints_dir

    @property
    def metrics_dir(self):
        return self.base.metrics_dir

    @property
    def artifacts_dir(self):
        return self.base.artifacts_dir

    def stage_dir(self, stage: str) -> Path:
        if str(stage) == "peak":
            return self.base.stage_dir("peak") / path_token(self.domain_name)
        return self.base.stage_dir(stage)


class DeferredGammaFeedbackCommitter:
    """Publish multi-domain physical feedback once per update cycle."""

    def __init__(
        self,
        *,
        layout,
        condition_store: ConditioningProviderStore,
        latest_coeff_maps: dict[str, str],
    ) -> None:
        self.layout = layout
        self.condition_store = condition_store
        self.latest_coeff_maps = latest_coeff_maps
        self.pending_domains: set[str] = set()
        self.last_result: TrainingRunEpochResult | TrainingStepResult | None = None
        self.last_store_entries: tuple[dict[str, str], ...] = ()
        self.committed_this_cycle = False

    def stage(
        self,
        *,
        entries: tuple[tuple[str, str], ...],
        result: TrainingRunEpochResult | TrainingStepResult,
    ) -> tuple[dict[str, str], ...]:
        for name, path in entries:
            self.latest_coeff_maps[str(name)] = str(path)
            self.pending_domains.add(str(name))
        self.last_result = result
        self.last_store_entries = tuple(
            {"name": name, "coeff_maps_npz": path}
            for name, path in sorted(self.latest_coeff_maps.items())
        )
        self.committed_this_cycle = False
        return self.last_store_entries

    def commit(self) -> dict[str, object]:
        if not self.pending_domains:
            return {
                "feedback_deferred_commit_skipped": True,
                "feedback_deferred_pending_domain_count": 0,
                "condition_store_version": self.condition_store.version,
            }
        if self.last_result is None:
            raise RuntimeError("deferred gamma feedback has pending domains but no trigger result")
        if not self.last_store_entries:
            raise RuntimeError("deferred gamma feedback has pending domains but no coeff-map entries")
        self.condition_store.update_from_coeff_maps(self.last_store_entries)
        state_path = write_current_physical_state(
            self.layout,
            coeff_maps=self.last_store_entries,
            source="gamma_feedback",
            epoch=int(self.last_result.epoch),
            global_step=int(self.last_result.global_step),
            condition_store_version=self.condition_store.version,
        )
        committed = sorted(self.pending_domains)
        self.pending_domains.clear()
        self.committed_this_cycle = True
        return {
            "feedback_deferred_commit": True,
            "feedback_deferred_committed_domains": committed,
            "feedback_deferred_committed_domain_count": len(committed),
            "feedback_deferred_pending_domain_count": 0,
            "feedback_coeff_maps_all_domains": {
                str(item["name"]): str(item["coeff_maps_npz"])
                for item in self.last_store_entries
            },
            "condition_store_version": self.condition_store.version,
            "physical_state_path": state_path,
        }
