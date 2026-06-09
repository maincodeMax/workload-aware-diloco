from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Protocol

from wa_diloco.config import PolicyConfig


@dataclass(frozen=True)
class ClusterPressure:
    network: float = 0.0
    inference: float = 0.0
    checkpoint: float = 0.0
    serving_sync_cost: float = 0.0

    @classmethod
    def from_mapping(cls, raw: dict) -> "ClusterPressure":
        return cls(
            network=_clamp01(float(raw.get("network_pressure", raw.get("network", 0.0)))),
            inference=_clamp01(float(raw.get("inference_pressure", raw.get("inference", 0.0)))),
            checkpoint=_clamp01(float(raw.get("checkpoint_pressure", raw.get("checkpoint", 0.0)))),
            serving_sync_cost=_clamp01(
                float(
                    raw.get(
                        "serving_sync_cost",
                        raw.get("sync_serving_cost", raw.get("expected_sync_cost", 0.0)),
                    )
                )
            ),
        )


@dataclass(frozen=True)
class LearnerReport:
    learner_id: int
    control_cycle: int
    global_round: int
    local_steps: int
    local_steps_since_sync: int
    tokens_since_sync: int
    loss_start: float | None
    loss_end: float | None
    update_norm: float
    wall_time_sec: float
    staleness_sec: float
    delta_path: str
    local_state_path: str
    domain: str = "general"

    @property
    def loss_delta(self) -> float:
        if self.loss_start is None or self.loss_end is None:
            return 0.0
        return max(0.0, self.loss_start - self.loss_end)


@dataclass(frozen=True)
class SyncDecision:
    sync_now: bool
    next_local_steps: int
    reason: str
    score: float = 0.0


class SyncPolicy(Protocol):
    def decide(self, report: LearnerReport, pressure: ClusterPressure) -> SyncDecision:
        ...


def build_policy(config: PolicyConfig) -> SyncPolicy:
    policies: dict[str, SyncPolicy] = {
        "fixed_h": FixedHPolicy(config),
        "adaptive_tokens": AdaptiveTokensPolicy(config),
        "adaptive_load": AdaptiveLoadPolicy(config),
        "pressure_gate": PressureGatePolicy(config),
        "random_deferral": RandomDeferralPolicy(config),
        "wa_calibrated": WorkloadAwarePolicy(config),
        "wa_diloco": WorkloadAwarePolicy(config),
    }
    try:
        return policies[config.name]
    except KeyError as exc:
        raise ValueError(f"unknown sync policy {config.name!r}; expected {sorted(policies)}") from exc


class FixedHPolicy:
    def __init__(self, config: PolicyConfig) -> None:
        self.config = config

    def decide(self, report: LearnerReport, pressure: ClusterPressure) -> SyncDecision:
        del pressure
        return SyncDecision(
            sync_now=report.local_steps_since_sync >= self.config.fixed_h,
            next_local_steps=self.config.fixed_h,
            reason=f"fixed_h_{self.config.fixed_h}",
        )


class AdaptiveTokensPolicy:
    def __init__(self, config: PolicyConfig) -> None:
        self.config = config

    def decide(self, report: LearnerReport, pressure: ClusterPressure) -> SyncDecision:
        del pressure
        forced = report.local_steps_since_sync >= self.config.max_h
        enough_tokens = report.tokens_since_sync >= self.config.min_tokens
        if forced:
            return SyncDecision(True, self.config.min_h, "max_h_forced")
        if enough_tokens:
            return SyncDecision(True, self.config.min_h, "token_threshold")
        return SyncDecision(False, self.config.min_h, "need_more_tokens")


class AdaptiveLoadPolicy:
    def __init__(self, config: PolicyConfig) -> None:
        self.config = config

    def decide(self, report: LearnerReport, pressure: ClusterPressure) -> SyncDecision:
        forced = report.local_steps_since_sync >= self.config.max_h
        load_penalty = (
            self.config.network_pressure_weight * pressure.network
            + self.config.inference_pressure_weight * pressure.inference
            + self.config.checkpoint_pressure_weight * pressure.checkpoint
        )
        if forced:
            return SyncDecision(True, self.config.min_h, "max_h_forced", score=-load_penalty)
        if report.local_steps_since_sync < self.config.min_h:
            return SyncDecision(False, self.config.min_h, "below_min_h", score=-load_penalty)
        if load_penalty > 0.6:
            return SyncDecision(False, self.config.min_h, "load_deferral", score=-load_penalty)
        return SyncDecision(True, self.config.min_h, "load_clear", score=-load_penalty)


class PressureGatePolicy:
    def __init__(self, config: PolicyConfig) -> None:
        self.config = config

    def decide(self, report: LearnerReport, pressure: ClusterPressure) -> SyncDecision:
        forced = report.local_steps_since_sync >= self.config.max_h
        if forced:
            return SyncDecision(True, self.config.min_h, "max_h_forced")
        if report.local_steps_since_sync < self.config.min_h:
            return SyncDecision(False, self.config.min_h, "below_min_h")

        max_pressure = max(pressure.network, pressure.inference, pressure.checkpoint)
        if max_pressure > self.config.pressure_threshold:
            return SyncDecision(False, self.config.min_h, "pressure_gate_deferral", score=-max_pressure)
        return SyncDecision(True, self.config.min_h, "pressure_gate_clear", score=-max_pressure)


class RandomDeferralPolicy:
    def __init__(self, config: PolicyConfig) -> None:
        self.config = config

    def decide(self, report: LearnerReport, pressure: ClusterPressure) -> SyncDecision:
        del pressure
        forced = report.local_steps_since_sync >= self.config.max_h
        if forced:
            return SyncDecision(True, self.config.min_h, "max_h_forced")
        if report.local_steps_since_sync < self.config.min_h:
            return SyncDecision(False, self.config.min_h, "below_min_h")

        rng = random.Random(
            self.config.random_seed
            + 1_000_003 * report.learner_id
            + 10_007 * report.control_cycle
            + report.global_round
        )
        if rng.random() < self.config.deferral_probability:
            return SyncDecision(False, self.config.min_h, "random_deferral")
        return SyncDecision(True, self.config.min_h, "random_clear")


class WorkloadAwarePolicy:
    def __init__(self, config: PolicyConfig) -> None:
        self.config = config

    def decide(self, report: LearnerReport, pressure: ClusterPressure) -> SyncDecision:
        forced = report.local_steps_since_sync >= self.config.max_h
        if forced:
            return SyncDecision(True, self.config.min_h, "max_h_forced", score=float("inf"))
        if report.local_steps_since_sync < self.config.min_h:
            return SyncDecision(False, self.config.min_h, "below_min_h")

        progress_score = (
            self.config.loss_weight * _soft_cap(report.loss_delta, scale=0.05)
            + self.config.token_weight * _soft_cap(report.tokens_since_sync, scale=1_000_000.0)
        )
        pressure_score = (
            self.config.network_pressure_weight * pressure.network
            + self.config.inference_pressure_weight * pressure.inference
            + self.config.checkpoint_pressure_weight * pressure.checkpoint
            + self.config.serving_sync_cost_weight
            * _soft_cap(pressure.serving_sync_cost, scale=self.config.serving_sync_cost_scale)
            + self.config.staleness_penalty * _soft_cap(report.staleness_sec, scale=1800.0)
        )
        score = progress_score - pressure_score
        if score > self.config.threshold:
            return SyncDecision(True, self.config.min_h, "value_exceeds_pressure", score=score)
        return SyncDecision(False, self.config.min_h, "pressure_exceeds_value", score=score)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _soft_cap(value: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return min(1.0, max(0.0, value / scale))
