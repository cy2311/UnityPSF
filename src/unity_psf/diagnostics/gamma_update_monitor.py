from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable


_EPOCH_RE = re.compile(r"epoch_(\d+)")
_STEP_RE = re.compile(r"step_(\d+)")


def find_gamma_summary_paths(*, run_dir: Path | None = None, update_dir: Path | None = None) -> tuple[Path, ...]:
    roots: list[Path] = []
    if update_dir is not None:
        roots.append(update_dir)
    if run_dir is not None:
        roots.extend(path for path in run_dir.glob("roi_bank_gamma*") if path.is_dir())

    paths: set[Path] = set()
    for root in roots:
        if root.name.startswith("domain_") and (root / "gamma_alternation_summary.json").is_file():
            paths.add(root / "gamma_alternation_summary.json")
        paths.update(root.glob("epoch_*/gamma_alternation/domain_*/gamma_alternation_summary.json"))
        paths.update(root.glob("gamma_alternation/domain_*/gamma_alternation_summary.json"))
        paths.update(root.glob("step_*/source_*/domain_*/gamma_alternation_summary.json"))
    return tuple(sorted(path.resolve() for path in paths))


def summarize_gamma_update(summary_path: Path) -> dict[str, Any]:
    summary = _read_json(summary_path)
    config = dict(summary.get("config") or {})
    heldout_initial = summary.get("heldout_initial_loss")
    heldout_delta = summary.get("heldout_loss_delta")
    heldout_metrics = dict(summary.get("heldout_metrics_final") or {})

    return {
        "epoch": _epoch_from_path(summary_path),
        "global_step": _step_from_path(summary_path),
        "domain": summary.get("domain_name"),
        "best_step": _as_int(summary.get("best_step")),
        "steps_completed": _as_int(summary.get("steps_completed")),
        "selected_step": _as_int(summary.get("selected_step")),
        "selected_poisson_nll": _as_float(summary.get("selected_poisson_nll")),
        "selected_sample_count": _as_int(summary.get("selected_sample_count")),
        "selected_sampled_emitter_count": _as_int(summary.get("selected_sampled_emitter_count")),
        "selected_projected_photons": _as_float(summary.get("selected_projected_photons")),
        "selected_background_mean": _as_float(summary.get("selected_background_mean")),
        "heldout_available": bool(summary.get("heldout_available", False)),
        "heldout_monitor_mode": summary.get("heldout_monitor_mode"),
        "heldout_roi_count": _as_int(summary.get("heldout_roi_count")),
        "heldout_sample_count": _as_int(summary.get("heldout_sample_count")),
        "heldout_sampled_emitter_count": _as_int(summary.get("heldout_sampled_emitter_count")),
        "heldout_initial_loss": _as_float(heldout_initial),
        "heldout_final_loss": _as_float(summary.get("heldout_final_loss")),
        "heldout_loss_delta": _as_float(heldout_delta),
        "heldout_loss_delta_percent": _loss_delta_percent(heldout_initial, heldout_delta),
        "heldout_poisson_nll": _as_float(heldout_metrics.get("roi_bank_nll")),
        "heldout_poisson_nll_full_roi": _as_float(heldout_metrics.get("roi_bank_nll_full_roi")),
        "gamma_before_norm": _as_float(summary.get("gamma_before_norm")),
        "gamma_after_norm": _as_float(summary.get("gamma_after_norm")),
        "gamma_delta_norm": _as_float(summary.get("gamma_delta_norm")),
        "roi_size_px": _as_int(config.get("roi_size_px")),
        "over_cut_px": _as_int(config.get("over_cut_px")),
        "gamma_steps": _as_int(config.get("steps")),
        "gamma_lr": _as_float(config.get("lr")),
        "num_posterior_samples": _as_int(config.get("num_posterior_samples")),
        "target_projected_emitters": _as_int(config.get("target_projected_emitters")),
        "projection_sample_batch_size": _as_int(config.get("roi_bank_projection_sample_batch_size")),
        "checkpoint_path": summary.get("checkpoint_path"),
        "report_path": summary.get("report_path"),
        "summary_path": str(summary_path),
    }


def build_monitor_payload(summary_paths: Iterable[Path]) -> dict[str, Any]:
    updates = [summarize_gamma_update(path) for path in summary_paths]
    updates.sort(
        key=lambda item: (
            -1 if item["global_step"] is None else int(item["global_step"]),
            -1 if item["epoch"] is None else int(item["epoch"]),
            str(item["domain"]),
        ),
        reverse=True,
    )
    latest_by_domain: dict[str, dict[str, Any]] = {}
    for item in updates:
        domain = str(item.get("domain"))
        if domain not in latest_by_domain:
            latest_by_domain[domain] = item
    return {
        "update_count": int(len(updates)),
        "domains": sorted(latest_by_domain),
        "latest_by_domain": latest_by_domain,
        "updates": updates,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    rows = []
    for domain, item in sorted(dict(payload.get("latest_by_domain") or {}).items()):
        rows.append(
            "| {domain} | {epoch} | {best_step} | {mode} | {heldout} | {pct} | {gamma} | {nll} | {emitters} |".format(
                domain=domain,
                epoch=item.get("global_step") if item.get("global_step") is not None else item.get("epoch", "n/a"),
                best_step=item.get("best_step", "n/a"),
                mode=item.get("heldout_monitor_mode") or "n/a",
                heldout=_format_number(item.get("heldout_loss_delta"), 3),
                pct=_format_number(item.get("heldout_loss_delta_percent"), 5),
                gamma=_format_number(item.get("gamma_delta_norm"), 5),
                nll=_format_number(item.get("selected_poisson_nll"), 4),
                emitters=item.get("selected_sampled_emitter_count") or "n/a",
            )
        )
    if not rows:
        rows.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
    return "\n".join(
        [
            "# Gamma Update Monitor",
            "",
            f"- update_count: {payload.get('update_count', 0)}",
            f"- domains: {', '.join(str(v) for v in payload.get('domains', [])) or 'n/a'}",
            "",
            "| Domain | Epoch | Best step | Held-out mode | Held-out delta | Held-out delta % | Gamma delta norm | Selected NLL | Selected emitters |",
            "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
            *rows,
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize ROI-bank gamma update monitor metrics.")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--update-dir", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.run_dir is None and args.update_dir is None:
        parser.error("Provide --run-dir or --update-dir.")
    payload = build_monitor_payload(find_gamma_summary_paths(run_dir=args.run_dir, update_dir=args.update_dir))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _loss_delta_percent(initial: Any, delta: Any) -> float | None:
    initial_value = _as_float(initial)
    delta_value = _as_float(delta)
    if initial_value is None or delta_value is None or initial_value == 0.0:
        return None
    return float(delta_value / initial_value * 100.0)


def _epoch_from_path(path: Path) -> int | None:
    for part in path.parts:
        match = _EPOCH_RE.fullmatch(part)
        if match:
            return int(match.group(1))
    return None


def _step_from_path(path: Path) -> int | None:
    for part in path.parts:
        match = _STEP_RE.fullmatch(part)
        if match:
            return int(match.group(1))
    return None


def _format_number(value: Any, digits: int = 4) -> str:
    number = _as_float(value)
    if number is None:
        return "n/a"
    return f"{number:.{digits}f}"


if __name__ == "__main__":
    raise SystemExit(main())
