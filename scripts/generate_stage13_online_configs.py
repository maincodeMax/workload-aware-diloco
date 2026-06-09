#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import yaml


SEEDS = [23, 29, 31, 37, 41]


def main() -> int:
    output_dir = Path("configs/stage13")
    output_dir.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        base_path = _base_config(seed)
        with base_path.open() as f:
            config = yaml.safe_load(f)

        config["name"] = f"stage13-wa-calibrated-s{seed}"
        config["policy"]["name"] = "wa_calibrated"
        config["policy"]["serving_sync_cost_weight"] = 1.0
        config["policy"]["serving_sync_cost_scale"] = 1.0
        config["runtime"]["workspace"] = f"../workspaces/stage13/wa-calibrated-s{seed}"
        config.setdefault("tags", {})
        config["tags"].update(
            {
                "mode": "wa_calibrated",
                "stage": "13",
                "seed": str(seed),
                "serving_cost": "bursty_online",
            }
        )

        output_path = output_dir / f"wa_calibrated_s{seed}.yaml"
        with output_path.open("w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        print(output_path)
    return 0


def _base_config(seed: int) -> Path:
    if seed in {23, 29, 31}:
        return Path(f"configs/stage2/wa_current_s{seed}.yaml")
    if seed in {37, 41}:
        return Path(f"configs/stage5/wa_current_s{seed}.yaml")
    raise KeyError(seed)


if __name__ == "__main__":
    raise SystemExit(main())
