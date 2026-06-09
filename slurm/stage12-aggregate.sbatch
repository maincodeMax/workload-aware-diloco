#!/usr/bin/env bash

#SBATCH --job-name=wa-stage12-agg
#SBATCH --partition=resident
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=results/stage12/aggregate-%j.out
#SBATCH --error=results/stage12/aggregate-%j.err

set -euo pipefail

REPO_ROOT="${WA_DILOCO_REPO:-$PWD}"
cd "${REPO_ROOT}"
python3 scripts/aggregate_stage12_bursty.py
