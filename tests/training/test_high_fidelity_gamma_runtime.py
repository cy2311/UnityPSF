from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from unity_psf.training.high_fidelity.gamma_runtime import (
    DeferredGammaFeedbackCommitter,
    gamma_hook_bindings,
    gamma_update_config,
    heldout_monitor_metrics,
    merge_zernike_delta_maps,
)


def test_gamma_hook_bindings_preserve_epoch_and_batch_routes() -> None:
    hook = object()

    assert gamma_hook_bindings(None, {}) == (None, None)
    assert gamma_hook_bindings(hook, {}) == (hook, None)
    assert gamma_hook_bindings(hook, {"start_batch": 3}) == (None, hook)


def test_gamma_update_config_preserves_defaults_and_explicit_overrides() -> None:
    defaults = gamma_update_config({}, {"epochs": 9})
    overrides = gamma_update_config(
        {
            "start_epoch": 2,
            "stop_epoch": 7,
            "interval_epochs": 3,
            "lr": 0.2,
            "steps": 4,
            "start_batch": 8,
            "stop_batch": 14,
            "update_interval_batches": 2,
            "optimizer": "sgd",
        },
        {"epochs": 9},
    )

    assert (defaults.start_epoch, defaults.stop_epoch, defaults.update_interval_epochs, defaults.lr, defaults.steps) == (1, 9, 1, 0.01, 1)
    assert (overrides.start_epoch, overrides.stop_epoch, overrides.update_interval_epochs, overrides.lr, overrides.steps) == (2, 7, 3, 0.2, 4)
    assert (overrides.start_batch, overrides.stop_batch, overrides.update_interval_batches, overrides.optimizer) == (8, 14, 2, "sgd")


def test_merge_zernike_delta_maps_merges_by_mode_not_position() -> None:
    merged = merge_zernike_delta_maps(
        np.asarray([[[1.0]], [[10.0]]], dtype=np.float32),
        base_mode_order=((2, 0), (2, 2)),
        delta_maps=torch.tensor([[[3.0]], [[4.0]]]),
        delta_mode_order=((2, 2), (2, 0)),
    )

    assert np.array_equal(merged, np.asarray([[[5.0]], [[13.0]]], dtype=np.float32))


def test_deferred_feedback_without_pending_domains_is_an_explicit_noop() -> None:
    committer = DeferredGammaFeedbackCommitter(
        layout=SimpleNamespace(),
        condition_store=SimpleNamespace(version=7),
        latest_coeff_maps={},
    )

    assert committer.commit() == {
        "feedback_deferred_commit_skipped": True,
        "feedback_deferred_pending_domain_count": 0,
        "condition_store_version": 7,
    }


def test_heldout_unavailable_metrics_keep_the_complete_schema() -> None:
    metrics = heldout_monitor_metrics(
        None,
        objective=object(),
        gamma_before=torch.zeros(1),
        gamma_after=torch.zeros(1),
        unavailable_mode="not_configured",
    )

    assert metrics["heldout_available"] is False
    assert metrics["heldout_monitor_mode"] == "not_configured"
    assert metrics["heldout_initial_loss"] is None
    assert metrics["heldout_poisson_nll_full_roi"] is None
