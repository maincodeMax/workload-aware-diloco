#!/usr/bin/env bash
set -euo pipefail

INTERVAL_SEC="${1:-120}"
REPO_ROOT="${WA_DILOCO_REPO:-$PWD}"
PYTHON_BIN="${WA_DILOCO_PYTHON:-}"
MANIFEST="${WA_DILOCO_STAGE7_MANIFEST:-}"
LOG="${WA_DILOCO_WATCHDOG_LOG:-}"
ONCE="${WA_DILOCO_WATCHDOG_ONCE:-0}"

cd "${REPO_ROOT}"

if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    PYTHON_BIN="$(command -v python)"
  fi
fi

if [[ -z "${MANIFEST}" ]]; then
  MANIFEST="$(ls -t results/stage7/requeued-*.tsv results/stage7/submitted-*.tsv 2>/dev/null | head -n 1)"
fi

if [[ -z "${LOG}" ]]; then
  mkdir -p results/stage7
  LOG="results/stage7/watchdog-$(date -u +%Y%m%dT%H%M%SZ).log"
fi

snapshot() {
  {
    echo "===== $(date -Is) ====="
    echo "repo=${REPO_ROOT}"
    echo "manifest=${MANIFEST}"
    echo
    echo "--- queue ---"
    squeue -u "${USER}" -o "%.18i %.8T %.10M %.36j %R" || true
    echo
    echo "--- gpu apps ---"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader \
      | while IFS=, read -r pid pname mem; do
          [[ -z "${pid:-}" ]] && continue
          owner="$(ps -o user= -p "${pid}" 2>/dev/null | tr -d ' ' || true)"
          printf "%s user=%s process=%s mem=%s\n" "${pid}" "${owner:-?}" "${pname}" "${mem}"
        done || true
    echo
    echo "--- eta ---"
    "${PYTHON_BIN}" - "${MANIFEST}" <<'PY'
import glob
import json
import math
import pathlib
import statistics
import subprocess
import sys
from datetime import datetime, timedelta, timezone

manifest = pathlib.Path(sys.argv[1])
rows = []
if manifest.exists():
    for line in manifest.read_text().splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 4:
            rows.append({"job_id": parts[0], "kind": parts[1], "config": parts[2], "workspace": parts[3]})

training = [r for r in rows if r["kind"] == "training"]
aggregate = [r for r in rows if r["kind"] == "aggregate"]

completed = []
durations = []
for row in training:
    matches = glob.glob(f"results/summary-*-{row['job_id']}.json")
    if not matches:
        continue
    completed.append(row)
    try:
        data = json.loads(pathlib.Path(matches[0]).read_text())
        dur = data.get("coordinator_duration_sec")
        if isinstance(dur, (int, float)) and dur > 0:
            durations.append(float(dur))
    except Exception:
        pass

try:
    sq = subprocess.check_output(
        ["squeue", "-u", subprocess.check_output(["id", "-un"], text=True).strip(), "-h", "-o", "%i %T"],
        text=True,
    ).splitlines()
except Exception:
    sq = []

states = {}
for line in sq:
    bits = line.split()
    if len(bits) >= 2:
        states[bits[0]] = bits[1]

running = [r for r in training if states.get(r["job_id"]) == "RUNNING"]
pending = [r for r in training if states.get(r["job_id"]) == "PENDING"]
remaining = len(training) - len(completed)

if durations:
    avg = statistics.mean(durations)
    med = statistics.median(durations)
else:
    avg = med = 600.0

# The first guarded full-WA runs are longer than the first no-SLO reruns, so
# use a conservative floor for the estimate until the ablation campaign fills in.
eta_per_job = max(avg, 600.0)
eta_sec = remaining * eta_per_job
now = datetime.now(timezone.utc)
finish = now + timedelta(seconds=eta_sec)

print(f"training_total={len(training)} completed={len(completed)} running={len(running)} pending={len(pending)} remaining={remaining}")
if aggregate:
    print(f"aggregate_job={aggregate[-1]['job_id']} state={states.get(aggregate[-1]['job_id'], 'not-in-queue')}")
if running:
    r = running[0]
    print(f"running_job={r['job_id']} config={r['config']}")
if durations:
    print(f"completed_duration_avg={avg/60:.2f}min median={med/60:.2f}min n={len(durations)}")
else:
    print("completed_duration_avg=unknown")
print(f"eta_per_job_used={eta_per_job/60:.2f}min")
print(f"eta_remaining={eta_sec/3600:.2f}h")
print(f"eta_finish_utc={finish.isoformat(timespec='seconds')}")
print(f"eta_finish_local={(finish + timedelta(hours=10)).isoformat(timespec='seconds')} Australia/Melbourne")
PY
    echo
  } | tee -a "${LOG}"
}

echo "watchdog_log=${LOG}"
while true; do
  snapshot
  if [[ "${ONCE}" == "1" ]]; then
    exit 0
  fi
  sleep "${INTERVAL_SEC}"
done
