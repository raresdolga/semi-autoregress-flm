#!/bin/bash
# One-off: tokenize wikitext2 into a flat uint16 stream for data.backend=memmap.
# Run this before scripts/local/train_flm_memmap.sh.
#
# The stream is model-length agnostic (rows are cut at read time), so this does
# not need re-running when model.length changes -- only when the dataset,
# tokenizer, or EOS setting does. Re-running is otherwise a no-op: prep skips a
# split that already has a meta.json.
#
# Usage:
#   prepare_wikitext2_memmap.sh              prepare (skip splits already done)
#   prepare_wikitext2_memmap.sh --overwrite  re-prepare from scratch
#   prepare_wikitext2_memmap.sh --verify     prepare, then run tests/verify_memmap
#
# Any other flag is forwarded to the prep CLI; see its --help.
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

python -u -m datamodules.backend_memmap \
  --dataset wikitext2 \
  --splits train,validation \
  --tokenizer gpt2 \
  --out-dir "$FLM_DATA_DIR/memmap" \
  --hf-cache-dir "$FLM_DATA_DIR/wikitext2" \
  --num-proc "$(nproc)" \
  ${prep_args[@]+"${prep_args[@]}"}

if [ "$verify" -eq 1 ]; then
  # Proves the stream matches a single-process reference tokenization, that prep
  # is byte-identical for any --num-proc, and that the memmap is fork/pickle
  # safe. The determinism check re-tokenizes the split, so this is not cheap --
  # which is why it is opt-in rather than part of every prepare.
  python -u -m tests.verify_memmap \
    --dataset wikitext2 \
    --split train \
    --tokenizer gpt2 \
    --out-dir "$FLM_DATA_DIR/memmap" \
    --hf-cache-dir "$FLM_DATA_DIR/wikitext2" \
    --window-size 128
fi
