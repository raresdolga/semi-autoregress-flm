#!/bin/bash
# FLM pretraining on LM1B -- the Table 10 replication (arXiv 2602.16813).
# Target: Gen. PPL 96.91, entropy 4.29 at 1024 sampling steps.
#
#   qsub scripts/train.sge.sh scripts/train/train_flm_lm1b.sh
#
# Requires the memmap stream from scripts/data/prepare_lm1b_memmap.sh.
#
#
# RESUME: the run directory is pinned by FLM_RUN_NAME rather than a timestamp,
# save_last writes checkpoints/last.ckpt, and resume_from_ckpt picks it up.
# Re-submitting the identical qsub therefore continues rather than restarting,
# which is what makes a 1M-step run survive h_rt limits. main.py only resumes if
# the checkpoint exists, so the first submission starts cleanly.
# Use a fresh FLM_RUN_NAME to start a genuinely new run.
set -euo pipefail

source "$(dirname "$0")/../env.sh"

# torch.compile is decided at *import* time: models/dit.py reads DIT_USE_COMPILE
# when the module loads, long before Hydra parses the config, so it cannot be a
# Hydra override. Off by default; untested at this scale.
export DIT_USE_COMPILE="${DIT_USE_COMPILE:-0}"

RUN_NAME="${FLM_RUN_NAME:-flm_lm1b_repro}"
RUN_DIR="$FLM_OUTPUT_DIR/lm1b/$RUN_NAME"

# Per-device batch. 4 GPUs x 128 = the paper's 512 with no accumulation; with
# fewer GPUs accumulate_grad_batches makes up the difference automatically, so
# the effective batch stays 512. Lower this if a GPU OOMs.
BS="${FLM_BS:-128}"

python -u -m main \
  hydra.run.dir="$RUN_DIR" \
  data=lm1b-memmap \
  algo=flm \
  algo.double_temb=False \
  model=small \
  model.length=128 \
  loader.global_batch_size=512 \
  loader.batch_size="$BS" \
  loader.eval_batch_size="$BS" \
  optim.lr=3e-4 \
  trainer.precision=bf16 \
  trainer.max_steps=1000000 \
  trainer.val_check_interval=5000 \
  sampling.steps=[1024] \
  sampling.num_sample_batches=1 \
  checkpointing.resume_from_ckpt=True \
  callbacks.checkpoint_every_n_steps.save_last=True \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=20000 \
  wandb.project=lm1b_full \
  wandb.name="$RUN_NAME" \
  "$@"
