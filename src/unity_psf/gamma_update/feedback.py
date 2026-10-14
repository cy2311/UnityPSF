from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class FeedbackMap:
    coefficients_nm: torch.Tensor
    metadata: dict[str, object]


def save_feedback_map(feedback: FeedbackMap, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "coefficients_nm": feedback.coefficients_nm.detach().cpu(),
            "metadata": dict(feedback.metadata),
        },
        output,
    )
    return output


def load_feedback_map(path: str | Path) -> FeedbackMap:
    payload = torch.load(Path(path), map_location="cpu")
    return FeedbackMap(
        coefficients_nm=torch.as_tensor(payload["coefficients_nm"], dtype=torch.float32),
        metadata=dict(payload.get("metadata", {})),
    )
