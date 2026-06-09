from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def utc_ms() -> int:
    return int(time.time() * 1000)


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"ts_ms": utc_ms(), **record}
    with target.open("a") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def write_json(path: str | Path, record: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(record, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(target)


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as f:
        return json.load(f)

