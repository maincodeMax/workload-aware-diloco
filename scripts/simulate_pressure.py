#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", help="Experiment workspace directory")
    parser.add_argument("--period-sec", type=float, default=300.0)
    parser.add_argument("--interval-sec", type=float, default=5.0)
    parser.add_argument("--mode", choices=["clean", "mixed"], default="mixed")
    args = parser.parse_args()

    target = Path(args.workspace) / "pressure" / "latest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    while True:
        elapsed = time.time() - start
        if args.mode == "clean":
            network = inference = checkpoint = 0.0
        else:
            phase = (math.sin(2 * math.pi * elapsed / args.period_sec) + 1.0) / 2.0
            network = 0.2 + 0.6 * phase
            inference = 0.1 + 0.8 * max(0.0, phase - 0.35) / 0.65
            checkpoint = 0.8 if int(elapsed // args.period_sec) % 4 == 3 else 0.0
        tmp = target.with_suffix(".json.tmp")
        with tmp.open("w") as f:
            json.dump(
                {
                    "ts": time.time(),
                    "network_pressure": network,
                    "inference_pressure": inference,
                    "checkpoint_pressure": checkpoint,
                },
                f,
                indent=2,
            )
            f.write("\n")
        tmp.replace(target)
        time.sleep(args.interval_sec)


if __name__ == "__main__":
    raise SystemExit(main())

