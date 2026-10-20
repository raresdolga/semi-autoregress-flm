# Synthetic Markov chain

Clean sequences come from a Markov chain with K tokens, length D and branching factor b (`mc.py`). Settings are in `configs/synthetic.yaml`. Run from the repo root:

```bash
source .venv/bin/activate
scripts/debug/train_synthetic.sh mc.K=64 mc.D=8 mc.b=2 mc.offsets=null
```

The script sources `scripts/env.sh` and writes runs to `$FLM_OUTPUT_DIR/synthetic/`. Set `mc.offsets=null` whenever b > 1, so the b offsets are drawn at random.
