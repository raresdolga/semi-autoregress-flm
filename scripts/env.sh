#!/bin/bash
# Shared paths for every run script. Source this, don't execute it.
# Override on the grid (or in your shell) with: export FLM_ROOT=/scratch/$USER/semi-autoregress-flm

export FLM_ROOT="${FLM_ROOT:-/home/rares/rares_disk_data/semi-autoregress-flm}"
export FLM_DATA_DIR="${FLM_DATA_DIR:-$FLM_ROOT/datasets}"
export FLM_OUTPUT_DIR="${FLM_OUTPUT_DIR:-$FLM_ROOT/outputs}"

mkdir -p "$FLM_DATA_DIR" "$FLM_OUTPUT_DIR"
