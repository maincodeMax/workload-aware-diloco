#!/usr/bin/env python3
from __future__ import annotations

import copy
from pathlib import Path

import yaml


SEEDS = [23, 29, 31, 37, 41]
SWEEP_SEEDS = [23, 37, 41]


BASE = {
    "model": {
        "name_or_path": "HuggingFaceTB/SmolLM2-135M",
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
        "per_device_batch_size": 2,
        "learning_rate": 2.0e-5,
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
    },
    "merge": {
        "token_weighted": True,
        "staleness_half_life_sec": 1800,
        "max_update_norm": None,
    },
    "runtime": {
        "workspace": "",
        "max_global_rounds": 8,
        "max_control_cycles": 32,
        "report_timeout_sec": 3600,
        "poll_interval_sec": 2,
        "pressure_file": "pressure/latest.json",
    },
    "tags": {
        "platform": "b200",
        "accelerator": "b200",
    },
}


def main() -> int:
    stage7 = {
        "full_wa": {},
        "no_slo": {"policy": {"inference_pressure_weight": 0.0}},
        "no_staleness": {"policy": {"staleness_penalty": 0.0}},
        "no_network": {"policy": {"network_pressure_weight": 0.0}},
        "no_loss": {"policy": {"loss_weight": 0.0}},
        "no_token": {"policy": {"token_weight": 0.0}},
        "pressure_only": {
            "policy": {"loss_weight": 0.0, "token_weight": 0.0, "threshold": -0.6}
        },
        "progress_only": {
            "policy": {
                "network_pressure_weight": 0.0,
                "inference_pressure_weight": 0.0,
                "checkpoint_pressure_weight": 0.0,
            }
        },
        "no_checkpoint": {"policy": {"checkpoint_pressure_weight": 0.0}},
        "random_matched": {
            "policy": {
                "name": "random_deferral",
                "fixed_h": 128,
                "min_h": 128,
                "max_h": 512,
                "deferral_probability": 0.7,
            }
        },
        "pressure_gate_matched": {
            "policy": {
                "name": "pressure_gate",
                "fixed_h": 128,
                "min_h": 128,
                "max_h": 512,
                "pressure_threshold": 0.45,
            }
        },
    }

    stage8 = {
        "slo_w025": {"policy": {"inference_pressure_weight": 0.25}},
        "slo_w050": {"policy": {"inference_pressure_weight": 0.50}},
        "slo_w100": {"policy": {"inference_pressure_weight": 1.00}},
        "slo_w150": {"policy": {"inference_pressure_weight": 1.50}},
        "stale_w050": {"policy": {"staleness_penalty": 0.50}},
        "stale_w100": {"policy": {"staleness_penalty": 1.00}},
        "min64_max256": {"policy": {"fixed_h": 64, "min_h": 64, "max_h": 256}},
        "min64_max512": {"policy": {"fixed_h": 64, "min_h": 64, "max_h": 512}},
        "min256_max512": {"policy": {"fixed_h": 256, "min_h": 256, "max_h": 512}},
        "min256_max1024": {"policy": {"fixed_h": 256, "min_h": 256, "max_h": 1024}},
    }

    config_paths: list[Path] = []
    for variant, patch in stage7.items():
        for seed in SEEDS:
            config_paths.append(_write_config("stage7", variant, seed, patch))
    for variant, patch in stage8.items():
        for seed in SWEEP_SEEDS:
            config_paths.append(_write_config("stage8", variant, seed, patch))

    manifest = Path("configs/stage7/configs.txt")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(str(path) for path in config_paths) + "\n")
    print(f"wrote {len(config_paths)} configs")
    print(f"manifest={manifest}")
    return 0


def _write_config(stage: str, variant: str, seed: int, patch: dict) -> Path:
    config = copy.deepcopy(BASE)
    _deep_update(config, patch)
    config["name"] = f"{stage}-{variant.replace('_', '-')}-s{seed}"
    config["dataset"]["shuffle_seed"] = seed
    config["runtime"]["workspace"] = f"../workspaces/{stage}/{variant.replace('_', '-')}-s{seed}"
    config["tags"].update({"stage": stage[-1], "mode": variant, "seed": str(seed)})
    if config["policy"]["name"] == "random_deferral":
        config["policy"]["random_seed"] = seed

    path = Path("configs") / stage / f"{variant}_s{seed}.yaml"
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
