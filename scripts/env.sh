#!/bin/bash
# Shared paths for every run script. Source this, don't execute it.
# Override on the grid (or in your shell) with: export FLM_ROOT=/scratch/$USER/semi-autoregress-flm

export FLM_ROOT="${FLM_ROOT:-/home/rares/rares_disk_data/semi-autoregress-flm}"
export FLM_DATA_DIR="${FLM_DATA_DIR:-$FLM_ROOT/datasets}"
export FLM_OUTPUT_DIR="${FLM_OUTPUT_DIR:-$FLM_ROOT/outputs}"

# Everything below defaults to somewhere under $HOME, where the cluster quota is
# small enough that a single `uv sync` blows it ("Disk quota exceeded (os error
# 122)" part-way through unpacking an sdist). uv also needs its cache on the
# same filesystem as the venv to hardlink instead of copy, so keep both under
# FLM_ROOT. Source this before uv, not just before training.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$FLM_ROOT/cache/uv}"
export TMPDIR="${TMPDIR:-$FLM_ROOT/tmp}"
export HF_HOME="${HF_HOME:-$FLM_ROOT/cache/huggingface}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$FLM_ROOT/cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$FLM_ROOT/cache/inductor}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$FLM_ROOT/cache/wandb}"

mkdir -p "$FLM_DATA_DIR" "$FLM_OUTPUT_DIR" "$UV_CACHE_DIR" "$TMPDIR" \
         "$HF_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$WANDB_CACHE_DIR"
