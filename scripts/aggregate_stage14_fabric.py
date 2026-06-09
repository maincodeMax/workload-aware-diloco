from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(
    os.environ.get(
        "WA_DILOCO_STAGE14_RESULT_ROOT",
        Path(__file__).resolve().parents[1] / "results/stage14-fabric",
    )
)


def _read_metadata(path: Path) -> dict[str, str]:
    metadata = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            metadata[key] = value
    return metadata


def main() -> None:
    rows = []
    for run_dir in sorted(ROOT.glob("*n")):
        meta_path = run_dir / "run-metadata.txt"
        probe_path = run_dir / "fabric-probe.jsonl"
        if not meta_path.exists() or not probe_path.exists():
            continue
        meta = _read_metadata(meta_path)
        for line in probe_path.read_text().splitlines():
            record = json.loads(line)
            if record.get("event") != "measurement":
                continue
            if record.get("rank") != 0:
                continue
            rows.append(
                {
                    "run": run_dir.name,
                    "nodes": int(meta["nnodes"]),
                    "world_size": int(record["world_size"]),
                    "size_mib": int(record["size_mib"]),
                    "avg_ms": float(record["avg_ms"]),
                    "algbw_gbps": float(record["algbw_gbps"]),
                    "busbw_gbps": float(record["busbw_gbps"]),
                }
            )

    if not rows:
        existing_md = ROOT / "stage14-fabric-summary.md"
        if existing_md.exists():
            print(existing_md.read_text())
            return

    output_json = ROOT / "stage14-fabric-summary.json"
    output_md = ROOT / "stage14-fabric-summary.md"
    output_json.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Stage 14 MI355X Fabric Probe",
        "",
        "| nodes | world | payload MiB/rank | avg ms | algbw GB/s | busbw GB/s |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {nodes} | {world_size} | {size_mib} | {avg_ms:.3f} | "
            "{algbw_gbps:.3f} | {busbw_gbps:.3f} |".format(**row)
        )
    output_md.write_text("\n".join(lines) + "\n")
    print(output_md.read_text())


if __name__ == "__main__":
    main()
