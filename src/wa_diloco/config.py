from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    name_or_path: str
    seq_len: int = 1024
    dtype: str = "bf16"


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    split: str = "train"
    text_column: str = "text"
    streaming: bool = True
    shuffle_seed: int = 17
    domain_name: str = "general"


@dataclass(frozen=True)
class LearnerConfig:
    count: int = 8
    min_ready: int = 8
    gpus_per_learner: int = 1
    per_device_batch_size: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0


@dataclass(frozen=True)
class PolicyConfig:
    name: str = "fixed_h"
    fixed_h: int = 256
    min_h: int = 128
    max_h: int = 1024
    min_tokens: int = 1_000_000
    max_wall_delay_sec: int = 900
    threshold: float = 0.0
    loss_weight: float = 0.5
    token_weight: float = 0.5
    network_pressure_weight: float = 0.5
    inference_pressure_weight: float = 0.5
    checkpoint_pressure_weight: float = 0.25
    serving_sync_cost_weight: float = 0.0
    serving_sync_cost_scale: float = 1.0
    staleness_penalty: float = 0.5
    pressure_threshold: float = 0.6
    deferral_probability: float = 0.5
    random_seed: int = 0


@dataclass(frozen=True)
class MergeConfig:
    token_weighted: bool = True
    staleness_half_life_sec: float = 1800.0
    max_update_norm: float | None = None


@dataclass(frozen=True)
class RuntimeConfig:
    workspace: Path
    max_global_rounds: int = 3
    max_control_cycles: int = 64
    report_timeout_sec: int = 1800
    poll_interval_sec: float = 2.0
    pressure_file: Path | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    model: ModelConfig
    dataset: DatasetConfig
    learners: LearnerConfig
    policy: PolicyConfig
    merge: MergeConfig
    runtime: RuntimeConfig
    tags: dict[str, str] = field(default_factory=dict)


def _require_mapping(data: Any, path: Path) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return data


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"config section {key!r} must be a mapping")
    return value


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open() as f:
        raw = _require_mapping(yaml.safe_load(f), config_path)

    runtime_raw = _section(raw, "runtime")
    workspace = Path(runtime_raw.get("workspace", f"workspaces/{raw['name']}")).expanduser()
    if not workspace.is_absolute():
        workspace = (config_path.parent.parent / workspace).resolve()

    pressure_file_raw = runtime_raw.get("pressure_file")
    pressure_file = None
    if pressure_file_raw:
        pressure_file = Path(pressure_file_raw).expanduser()
        if not pressure_file.is_absolute():
            pressure_file = (workspace / pressure_file).resolve()

    return ExperimentConfig(
        name=str(raw["name"]),
        model=ModelConfig(**_section(raw, "model")),
        dataset=DatasetConfig(**_section(raw, "dataset")),
        learners=LearnerConfig(**_section(raw, "learners")),
        policy=PolicyConfig(**_section(raw, "policy")),
        merge=MergeConfig(**_section(raw, "merge")),
        runtime=RuntimeConfig(
            workspace=workspace,
            max_global_rounds=int(runtime_raw.get("max_global_rounds", 3)),
            max_control_cycles=int(runtime_raw.get("max_control_cycles", 64)),
            report_timeout_sec=int(runtime_raw.get("report_timeout_sec", 1800)),
            poll_interval_sec=float(runtime_raw.get("poll_interval_sec", 2.0)),
            pressure_file=pressure_file,
        ),
        tags={str(k): str(v) for k, v in _section(raw, "tags").items()},
    )


def ensure_workspace(config: ExperimentConfig) -> None:
    for child in [
        "assignments",
        "checkpoints",
        "events",
        "learners",
        "pressure",
        "reports",
        "updates",
    ]:
        (config.runtime.workspace / child).mkdir(parents=True, exist_ok=True)
