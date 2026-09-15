#!/bin/bash
# Shared paths for every run script. Source this, don't execute it.
# Override on the grid (or in your shell) with: export FLM_ROOT=/scratch/$USER/semi-autoregress-flm

export FLM_ROOT="${FLM_ROOT:-/home/rares/rares_disk_data/semi-autoregress-flm}"
export FLM_DATA_DIR="${FLM_DATA_DIR:-$FLM_ROOT/datasets}"
export FLM_OUTPUT_DIR="${FLM_OUTPUT_DIR:-$FLM_ROOT/outputs}"

# Source this before uv, not just before training.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/inductor}"
# Bulk data stays on FLM_ROOT, where the quota is large: downloads, run outputs,
# scratch and wandb staging are all big and none of them need to be near the venv.
export TMPDIR="${TMPDIR:-$FLM_ROOT/tmp}"
export HF_HOME="${HF_HOME:-$FLM_ROOT/cache/huggingface}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$FLM_ROOT/cache/wandb}"

mkdir -p "$FLM_DATA_DIR" "$FLM_OUTPUT_DIR" "$UV_CACHE_DIR" "$TMPDIR" \
         "$HF_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$WANDB_CACHE_DIR"
