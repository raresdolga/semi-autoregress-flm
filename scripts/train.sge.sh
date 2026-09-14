#!/bin/bash
# Generic SGE wrapper for any training script in scripts/train/.
#
#   qsub scripts/train.sge.sh scripts/train/train_flm_lm1b.sh
#   qsub scripts/train.sge.sh scripts/train/train_flm_lm1b.sh trainer.max_steps=1000
#
# Submit from the repo root: -cwd makes that the job's working directory and
# where the .o log lands. Requires the project venv at .venv.
#
# GPU count comes from -pe gpu; trainer.devices resolves to
# torch.cuda.device_count(), so the scheduler's allocation is picked up with no
# override. Per-device batch and grad accumulation then derive from
# loader.global_batch_size. Override on the command line -- qsub flags win over
# the #$ directives below:
#
#   qsub -pe gpu 8 -l tmem=16G -l h_rt=240:00:00 -N flm-owt \
#        scripts/train.sge.sh scripts/train/train_flm_owt.sh
#
# Defaults are 2 A100s; -pe gpu N and -l gpu_type=... override both.
#
# Set FLM_ROOT in your shell (or .bashrc) to project space; -V imports it.
#
# Long runs outlive h_rt. Train scripts pin a stable run directory and enable
# resume, so re-submitting the identical qsub continues from the last
# checkpoint. -r y also lets GE restart the job itself after a node failure.
#
# qstat / qstat -f -j <job-ID> / qdel <job-ID> to monitor and cancel.
#
#$ -S /bin/bash
#$ -N flm-train
#$ -cwd
#$ -j y
# Scheduler logs land in logs/<job-name>.o<job-ID>, relative to the submit dir.
# GE opens this file before the job starts and will NOT create the directory:
# run `mkdir -p logs` once after cloning, or submits land in Eqw.
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
