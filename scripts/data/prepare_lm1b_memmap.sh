#!/bin/bash
# One-off: tokenize LM1B into a flat uint16 stream for data.backend=memmap.
# This is the data prep for the FLM Table 10 replication (arXiv 2602.16813):
# LM1B, L=128, bert-base-uncased, |V|=30522.
#
# Run this before training with data=lm1b-memmap.
#
# Usage:
#   prepare_lm1b_memmap.sh                    prepare (skip splits already done)
#   prepare_lm1b_memmap.sh --overwrite        re-prepare from scratch
#   prepare_lm1b_memmap.sh --verify           prepare, then run tests/verify_memmap
#   prepare_lm1b_memmap.sh --limit-docs 20000 smoke test on a slice
#
# Any other flag is forwarded to the prep CLI; see its --help.
# ---------------------------------------------------------------------------
set -euo pipefail

source "$(dirname "$0")/../env.sh"

# --verify is consumed here; every other argument goes through to the prep CLI.
verify=0
prep_args=()
for arg in "$@"; do
  if [ "$arg" = "--verify" ]; then
    verify=1
  else
    prep_args+=("$arg")
  fi
done

# NSLOTS is set by SGE to the -pe smp core count; fall back to the local core
# count so this stays runnable outside the scheduler.
num_proc="${NSLOTS:-$(nproc)}"

python -u -m datamodules.backend_memmap \
  --dataset lm1b \
  --splits train,test \
  --tokenizer bert-base-uncased \
  --out-dir "$FLM_DATA_DIR/memmap" \
  --hf-cache-dir "$FLM_DATA_DIR/lm1b" \
  --num-proc "$num_proc" \
  ${prep_args[@]+"${prep_args[@]}"}

if [ "$verify" -eq 1 ]; then
  # Proves the stream matches a single-process reference tokenization, that prep
  # is byte-identical for any --num-proc, and that the memmap is fork/pickle
  # safe. Run on 'test': the determinism check re-tokenizes the split
  # single-process, which is not viable on LM1B's 30M-document train split.
  python -u -m tests.verify_memmap \
    --dataset lm1b \
    --split test \
    --tokenizer bert-base-uncased \
    --out-dir "$FLM_DATA_DIR/memmap" \
    --hf-cache-dir "$FLM_DATA_DIR/lm1b" \
    --window-size 128
fi
