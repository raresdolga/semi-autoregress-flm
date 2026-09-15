#!/bin/bash
# ---------------------------------------------------------------------------------------
# Generic SGE wrapper for any training script in scripts/train/.
# Instructions:
# 1. Submit from the repo root: -cwd makes that the job's working directory.
# 2. Requires the project venv at .venv.
# 3. Set FLM_ROOT in your shell (or .bashrc) to project space; -V imports it.
# 4. run `mkdir -p logs` after cloning
# Train scripts pin a stable run directory and enable
# resume, so re-submitting the identical qsub continues from the last
# checkpoint. -r y also lets GE restart the job itself after a node failure.
#
# Usage:
#   qsub scripts/train.sge.sh scripts/train/train_flm_lm1b.sh
#   qsub -pe gpu 8 -l tmem=16G -l h_rt=240:00:00 -N flm-owt \
#        scripts/train.sge.sh scripts/train/train_flm_owt.sh trainer.max_steps=1000
#
# Monitoring:
#   qstat / qstat -f -j <job-ID> / qdel <job-ID> to monitor and cancel.
# --------------------------------------------------------------------------------------
#
#$ -S /bin/bash
#$ -N flm-train
#$ -cwd
#$ -j y
#$ -o logs/
#$ -V
#$ -r y
#$ -l h_rt=240:00:00
#$ -l gpu=true
#$ -l gpu_type=a100
#$ -pe gpu 2
#$ -R y
# tmem is HOST RAM, per GPU, multiplied by -pe gpu: 16G x 2 = 32G.
# h_vmem is deliberately absent: it breaks GPU allocation for PyTorch jobs.
#$ -l tmem=16G

set -euo pipefail

# Identifies the node and start time if the job fails for cluster reasons.
hostname
date

if [ "$#" -lt 1 ]; then
  echo "usage: qsub $0 <scripts/train/train_*.sh> [hydra overrides...]" >&2
  exit 2
fi

train_script="$1"
shift

# Paths resolve against the -cwd working directory, not $0: SGE runs the job
# from a spooled copy of this file, so its own location tells us nothing.
if [ ! -f "$train_script" ]; then
  echo "No such train script: $train_script (submit from the repo root)" >&2
  exit 2
fi

# Sourced here so the job fails fast on a bad FLM_ROOT rather than after the
# queue wait. Train scripts source it too; it is idempotent.
source scripts/env.sh
echo "FLM_DATA_DIR=$FLM_DATA_DIR"
echo "FLM_OUTPUT_DIR=$FLM_OUTPUT_DIR"

source .venv/bin/activate
python --version

# CUDA_VISIBLE_DEVICES is set by the scheduler; the site guide is explicit that
# jobs must not modify it or pin device indices. Just report what we were given.
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
python -c "import torch; print('torch', torch.__version__, 'gpus', torch.cuda.device_count())"

echo "Running: $train_script $*"
bash "$train_script" "$@"

date
echo "Done."
