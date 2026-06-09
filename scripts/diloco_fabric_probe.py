from __future__ import annotations

import datetime as dt
import json
import os
import socket
import time
from pathlib import Path

import torch
import torch.distributed as dist


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_sizes() -> list[int]:
    raw = os.environ.get("FABRIC_BENCH_SIZES_MIB", "64,256,1024")
    return [int(item) for item in raw.split(",") if item.strip()]


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    result_dir = Path(os.environ.get("FABRIC_RESULT_DIR", "/results"))
    result_dir.mkdir(parents=True, exist_ok=True)

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=dt.timedelta(seconds=_env_int("FABRIC_DIST_TIMEOUT", 1800)),
    )

    host = socket.gethostname()
    gpu_name = torch.cuda.get_device_name(local_rank)
    sizes_mib = _env_sizes()
    warmup = _env_int("FABRIC_BENCH_WARMUP_ITERS", 3)
    iters = _env_int("FABRIC_BENCH_MEASURE_ITERS", 8)

    if rank == 0:
        meta = {
            "event": "start",
            "host": host,
            "world_size": world_size,
            "sizes_mib": sizes_mib,
            "warmup_iters": warmup,
            "measure_iters": iters,
            "master_addr": os.environ.get("MASTER_ADDR"),
            "master_port": os.environ.get("MASTER_PORT"),
            "nccl_socket_ifname": os.environ.get("NCCL_SOCKET_IFNAME"),
            "nccl_ib_hca": os.environ.get("NCCL_IB_HCA"),
            "gpu_name": gpu_name,
            "ts": time.time(),
        }
        (result_dir / "fabric-probe.jsonl").write_text(json.dumps(meta, sort_keys=True) + "\n")

    print(
        f"rank={rank} local_rank={local_rank} host={host} world={world_size} gpu={gpu_name}",
        flush=True,
    )

    for size_mib in sizes_mib:
        numel = (size_mib * 1024 * 1024) // 4
        tensor = torch.full((numel,), float(rank + 1), device=device, dtype=torch.float32)

        for _ in range(warmup):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor.mul_(1.0 / world_size)

        dist.barrier()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor.mul_(1.0 / world_size)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        avg_s = elapsed / iters
        bytes_per_rank = numel * 4
        algbw_gbps = bytes_per_rank / avg_s / 1e9
        busbw_gbps = algbw_gbps * (2.0 * (world_size - 1.0) / world_size)
        sample = float(tensor[0].item())
        expected = (world_size + 1.0) / 2.0
        ok = abs(sample - expected) <= 0.01

        row = {
            "event": "measurement",
            "rank": rank,
            "world_size": world_size,
            "size_mib": size_mib,
            "bytes_per_rank": bytes_per_rank,
            "iters": iters,
            "avg_ms": avg_s * 1000.0,
            "algbw_gbps": algbw_gbps,
            "busbw_gbps": busbw_gbps,
            "sample": sample,
            "expected": expected,
            "ok": ok,
        }
        print(json.dumps(row, sort_keys=True), flush=True)
        if rank == 0:
            with (result_dir / "fabric-probe.jsonl").open("a") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        if not ok:
            raise SystemExit(f"sample mismatch rank={rank} size_mib={size_mib}")

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        with (result_dir / "fabric-probe.jsonl").open("a") as f:
            f.write(json.dumps({"event": "stop", "ts": time.time()}, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
