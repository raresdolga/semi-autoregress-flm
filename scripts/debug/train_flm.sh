#!/bin/bash
# Local FLM training on a single GPU: tiny model, wikitext2, no wandb.
# '~wandb' drops the wandb config group so main.py logs to stdout only.
# Each run gets its own dir under $FLM_OUTPUT_DIR with config + checkpoints inside.
#
# For the flat-uint16 backend see scripts/local/train_flm_memmap.sh.
#
# Measured on an RTX 3050 Ti Laptop (4 GB): batch_size=8 is the ceiling, because
# FLM's corrupt_continuous materializes a (B, L, vocab) float tensor (823 MB at
# B=32) -- batch_size=16 OOMs even though the bare backbone fits B=64. Raising
# the batch buys little anyway (7.7 -> 9.6 ktok/s from B=4 to B=8): those
# elementwise kernels, not data loading, are what keeps the GPU busy.
#
# num_workers=0 is deliberate and faster here: the dataset is a local mmap, so
# worker IPC costs more than the read (2565 vs 644 batch/s at B=4).

source "$(dirname "$0")/../env.sh"

# torch.compile is decided at *import* time: models/dit.py reads DIT_USE_COMPILE
# and decorates DIT.forward when the module loads, which happens at `import algo`
# in main.py -- long before @hydra.main parses the config. So it cannot be a
# Hydra override; it has to be in the environment before python starts.
#   DIT_USE_COMPILE=1 scripts/local/train_flm.sh
export DIT_USE_COMPILE="${DIT_USE_COMPILE:-0}"

python -u -m main \
  '~wandb' \
  hydra.run.dir="$FLM_OUTPUT_DIR"'/${data.train}/${algo.name}/${now:%Y.%m.%d}/${now:%H%M%S}' \
  data=wikitext2 \
  data.cache_dir="$FLM_DATA_DIR/wikitext2" \
  algo=flm \
  model=tiny \
  model.length=128 \
  loader.global_batch_size=8 \
  loader.batch_size=8 \
  loader.eval_batch_size=8 \
  loader.num_workers=0 \
  optim.lr=3e-4 \
  trainer.devices=1 \
  trainer.precision=bf16 \
  trainer.max_steps=2000 \
  trainer.val_check_interval=500 \
  trainer.limit_val_batches=20 \
  trainer.log_every_n_steps=10 \
  eval.generate_samples=False \
  eval.compute_generative_perplexity=False \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=1000 \
  "$@"
