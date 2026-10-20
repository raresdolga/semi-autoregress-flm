#!/bin/bash
# Synthetic Markov-chain experiment (flm/synthetic/main.py, configs/synthetic.yaml).
# Sources env.sh so runs land under $FLM_OUTPUT_DIR/synthetic/.
# Extra Hydra overrides are passed through, e.g.:
#   scripts/debug/train_synthetic.sh mc.K=64 mc.D=8 mc.b=2 mc.offsets=null

source "$(dirname "$0")/../env.sh"

cd "$(dirname "$0")/../.." || exit 1

python -u -m flm.synthetic.main "$@"
