#!/usr/bin/env bash
#SBATCH --job-name=wa-stage13-agg
#SBATCH --partition=resident
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=results/stage13/aggregate-%j.out
#SBATCH --error=results/stage13/aggregate-%j.err

set -euo pipefail

REPO_ROOT="${WA_DILOCO_REPO:-$PWD}"
cd "${REPO_ROOT}"
mkdir -p results/stage13
python3 scripts/aggregate_stage13_online.py
