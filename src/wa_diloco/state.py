from __future__ import annotations

import math
from pathlib import Path
from typing import Any


def torch_module():
    import torch

    return torch


def load_state(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    torch = torch_module()
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def save_state(path: str | Path, state: dict[str, Any]) -> None:
    torch = torch_module()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(target)


def compute_delta(base: dict[str, Any], updated: dict[str, Any]) -> tuple[dict[str, Any], float]:
    torch = torch_module()
    delta: dict[str, Any] = {}
    norm_sq = 0.0
    for key, base_value in base.items():
        updated_value = updated[key]
        if not torch.is_tensor(base_value) or not torch.is_floating_point(base_value):
            continue
        diff = (updated_value.detach().to("cpu") - base_value.detach().to("cpu")).to(torch.float32)
        norm_sq += float(torch.sum(diff * diff))
        delta[key] = diff.to(torch.bfloat16)
    return delta, math.sqrt(norm_sq)


def apply_delta(base: dict[str, Any], delta: dict[str, Any], scale: float = 1.0) -> dict[str, Any]:
    torch = torch_module()
    merged: dict[str, Any] = {}
    for key, value in base.items():
        if key in delta and torch.is_tensor(value) and torch.is_floating_point(value):
            merged[key] = (
                value.detach().to("cpu").to(torch.float32) + delta[key].to(torch.float32) * scale
            ).to(dtype=value.dtype)
        else:
            merged[key] = value
    return merged


def merge_deltas(
    base: dict[str, Any],
    weighted_delta_paths: list[tuple[str | Path, float]],
    max_update_norm: float | None = None,
) -> dict[str, Any]:
    torch = torch_module()
    if not weighted_delta_paths:
        raise ValueError("cannot merge zero learner updates")

    total_weight = sum(max(0.0, weight) for _, weight in weighted_delta_paths)
    if total_weight <= 0:
        total_weight = float(len(weighted_delta_paths))
        weighted_delta_paths = [(path, 1.0) for path, _ in weighted_delta_paths]

    accum: dict[str, Any] = {}
    for path, weight in weighted_delta_paths:
        delta = load_state(path, map_location="cpu")
        scaled = max(0.0, weight) / total_weight
        for key, value in delta.items():
            value_f32 = value.to(torch.float32) * scaled
            if key not in accum:
                accum[key] = value_f32
            else:
                accum[key] += value_f32

    if max_update_norm is not None:
        norm_sq = 0.0
        for value in accum.values():
            norm_sq += float(torch.sum(value * value))
        norm = math.sqrt(norm_sq)
        if norm > max_update_norm and norm > 0:
            shrink = max_update_norm / norm
            for key in list(accum):
                accum[key] *= shrink

    return apply_delta(base, accum)


def freshness_weight(staleness_sec: float, half_life_sec: float) -> float:
    if half_life_sec <= 0:
        return 1.0
    return 0.5 ** (max(0.0, staleness_sec) / half_life_sec)

