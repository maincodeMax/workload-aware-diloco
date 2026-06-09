#!/usr/bin/env python3
from __future__ import annotations

import copy
from pathlib import Path

import yaml


SEEDS = [23, 37, 41]

BASE = {
    "model": {
        "name_or_path": "HuggingFaceTB/SmolLM2-1.7B",
        "seq_len": 1024,
        "dtype": "bf16",
    },
    "dataset": {
        "name": "HuggingFaceFW/fineweb-edu",
        "split": "train",
        "text_column": "text",
        "streaming": True,
        "shuffle_seed": 0,
        "domain_name": "general",
    },
    "learners": {
        "count": 8,
        "min_ready": 8,
        "gpus_per_learner": 1,
        "per_device_batch_size": 1,
        "learning_rate": 1.0e-5,
        "weight_decay": 0.1,
        "max_grad_norm": 1.0,
    },
    "policy": {
        "name": "wa_diloco",
        "fixed_h": 128,
        "min_h": 128,
        "max_h": 512,
        "min_tokens": 1_000_000,
        "max_wall_delay_sec": 900,
        "threshold": 0.0,
        "loss_weight": 0.5,
        "token_weight": 0.5,
        "network_pressure_weight": 0.5,
        "inference_pressure_weight": 0.75,
        "checkpoint_pressure_weight": 0.25,
        "staleness_penalty": 0.25,
        "pressure_threshold": 0.6,
    },
    "merge": {
        "token_weighted": True,
        "staleness_half_life_sec": 1800,
        "max_update_norm": None,
    },
    "runtime": {
        "workspace": "",
        "max_global_rounds": 4,
        "max_control_cycles": 24,
        "report_timeout_sec": 7200,
        "poll_interval_sec": 2,
        "pressure_file": "pressure/latest.json",
    },
    "tags": {
        "platform": "b200",
        "accelerator": "b200",
        "stage": "9",
        "model_size": "1.7b",
    },
}


VARIANTS = {
    "fixed_h_256": {
        "workspace_name": "fixed-h-256",
        "patch": {
            "policy": {
                "name": "fixed_h",
                "fixed_h": 256,
                "min_h": 256,
                "max_h": 256,
            }
        },
    },
    "pressure_gate": {
        "workspace_name": "pressure-gate",
        "patch": {
            "policy": {
                "name": "pressure_gate",
                "fixed_h": 128,
                "min_h": 128,
                "max_h": 512,
                "pressure_threshold": 0.6,
            }
        },
    },
    "wa_current": {
        "workspace_name": "wa-current",
        "patch": {},
    },
}


def main() -> int:
    config_paths: list[Path] = []
    for variant, spec in VARIANTS.items():
        for seed in SEEDS:
            config_paths.append(
                _write_config(
                    variant=variant,
                    workspace_prefix=spec["workspace_name"],
                    seed=seed,
                    patch=spec["patch"],
                )
            )

    manifest = Path("configs/stage9/configs.txt")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(str(path) for path in config_paths) + "\n")
    print(f"wrote {len(config_paths)} configs")
    print(f"manifest={manifest}")
    return 0


def _write_config(variant: str, workspace_prefix: str, seed: int, patch: dict) -> Path:
    config = copy.deepcopy(BASE)
    _deep_update(config, patch)
    workspace_name = f"{workspace_prefix}-s{seed}"
    config["name"] = f"stage9-{workspace_name}"
    config["dataset"]["shuffle_seed"] = seed
    config["runtime"]["workspace"] = f"../workspaces/stage9/{workspace_name}"
    config["tags"].update({"mode": variant, "seed": str(seed)})

    path = Path("configs/stage9") / f"{variant}_s{seed}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def _deep_update(target: dict, patch: dict) -> None:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


if __name__ == "__main__":
    raise SystemExit(main())
