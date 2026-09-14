#!/bin/bash
# Generic SGE wrapper for any dataset prep script in scripts/data/.
#
#   qsub scripts/proc_data.sge.sh scripts/data/prepare_lm1b_memmap.sh
#   qsub scripts/proc_data.sge.sh scripts/data/prepare_lm1b_memmap.sh --overwrite
#
# Submit from the repo root: -cwd makes that the job's working directory and
# where the .o log lands. Requires the project venv at .venv.
#
# Prep is CPU-bound tokenization that parallelises almost linearly, so the job
# asks for a big SMP slice; prep scripts pick up NSLOTS as their worker count.
# Override per dataset on the command line rather than editing this file --
# qsub flags win over the #$ directives:
#
#   qsub -pe smp 64 -l tmem=1.5G,h_vmem=1.5G -l h_rt=24:00:00 -N proc-owt \
#        scripts/proc_data.sge.sh scripts/data/prepare_owt_memmap.sh
#
# If your site has dedicated CPU-only nodes, target them with -q (queue names
# are site-specific; `qstat -g c` lists them):
#
#   qsub -q cpu.q scripts/proc_data.sge.sh scripts/data/prepare_lm1b_memmap.sh
# qstat / qstat -f -j <job-ID> / qdel <job-ID> to monitor and cancel.
#
# Keep this block free of blank lines: GE stops reading #$ directives at the
# first line not starting with '#', and a dropped -l tmem leaves the job queued
# forever with no error.
#$ -S /bin/bash
#$ -N proc-data
#$ -cwd
#$ -j y
# Scheduler logs land in logs/<job-name>.o<job-ID>, relative to the submit dir.
# GE opens this file before the job starts and will NOT create the directory:
# run `mkdir -p logs` once after cloning, or submits land in Eqw.
#$ -o logs/
#$ -V
# Tokenization is CPU-only: no -l gpu=true here.
#$ -l h_rt=12:00:00
# SMP slice for the tokenizer worker pool. -R y reserves the cores, without
# which a wide request can starve behind smaller jobs.
#$ -pe smp 32
#$ -R y
# PER SLOT, and multiplied by the smp count: 2G x 32 = 64G total. Both keys are
# required and must match. Workers stream shards of a memory-mapped Arrow table,
# so each needs little; raising this is what makes a wide job queue for hours.
#$ -l tmem=2G,h_vmem=2G

set -euo pipefail

# Identifies the node and start time if the job fails for cluster reasons.
hostname
date

if [ "$#" -lt 1 ]; then
  echo "usage: qsub $0 <scripts/data/prepare_*.sh> [args...]" >&2
  exit 2
fi

prep_script="$1"
shift

# Paths resolve against the -cwd working directory, not $0: SGE runs the job
# from a spooled copy of this file, so its own location tells us nothing.
if [ ! -f "$prep_script" ]; then
  echo "No such prep script: $prep_script (submit from the repo root)" >&2
  exit 2
fi

source scripts/env.sh
echo "FLM_DATA_DIR=$FLM_DATA_DIR"
case "$FLM_DATA_DIR" in
  "$HOME"/*) echo "WARNING: writing datasets under \$HOME. Fine locally, but the" >&2
             echo "         cluster home quota is strict -- set FLM_ROOT to project space." >&2 ;;
esac

# shellcheck disable=SC1091
source .venv/bin/activate
python --version

# One tokenizer process per reserved core. BLAS stays single-threaded so the
# worker pool is the only source of parallelism.
export OMP_NUM_THREADS=1
echo "NSLOTS=${NSLOTS:-1}"

echo "Running: $prep_script $*"
bash "$prep_script" "$@"

date
echo "Done."
