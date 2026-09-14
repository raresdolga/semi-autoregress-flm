"""Data pipeline: tokenization, packing, caching and loading.

Import from the submodule that owns what you need -- this package deliberately
re-exports nothing:

    from datamodules.tokenization import get_tokenizer
    from datamodules.dataloaders import get_dataloaders, get_pseudo_dataloader
    from datamodules.dataloaders import RandomFaultTolerantSampler


Two token backends, selected per data config by `data.backend`:

    arrow  (default) -- HF `datasets` on disk, memory-mapped Arrow. Frames every
                        row as [BOS] + (block_size-2) tokens + [EOS].
    memmap           -- flat uint16 stream + np.memmap, nanoGPT/flash-attn style.
                        ~4x smaller, ~16x cheaper per item, model-length
                        agnostic. Requires an offline
                        `python -m datamodules.backend_memmap` run.

Modules, in dependency order (acyclic): paths (cache naming) and tokenization
(detokenizers + tokenizers) -> datasets (corpora + split dispatch) ->
backend_arrow / backend_memmap (token storage) -> dataloaders (samplers +
DataLoaders). The memmap backend doubles as the offline prep CLI.

Naming note: `datamodules/datasets.py` shares a name with the HuggingFace
`datasets` library. A bare `import datasets` is absolute and resolves to the
library; modules needing both import ours as `datamodules.datasets`. Never write
`from . import datasets` in a module that also uses the library.
"""
