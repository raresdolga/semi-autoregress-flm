"""Cache naming and on-disk layout conventions.

Kept in one place because the Arrow and memmap backends must agree on how a
(dataset, tokenizer, split, block_size, wrap, eos) tuple maps to a path. The
tokenizer is part of the key: without it, `configs/data/lm1b.yaml`
(bert-base-uncased) and `lm1b-gpt2.yaml` (gpt2) collide on one cache file and the
second run silently trains on wrong-vocab tokens.
"""

import os
import re


def slugify(name):
    """Filesystem-safe slug for a dataset or tokenizer name.

    'HuggingFaceFW/fineweb-edu' -> 'HuggingFaceFW__fineweb-edu'
    """
    name = str(name)
    name = name.replace("/", "__")
    return re.sub(r"[^0-9A-Za-z._-]+", "_", name)


def tokenizer_slug(tokenizer):
    """Slug identifying `tokenizer`, for use in a cache key."""
    name = getattr(tokenizer, "name_or_path", None) or type(tokenizer).__name__
    return slugify(name)


def arrow_cache_path(
    cache_dir, dataset_name, tokenizer, mode, block_size, wrap, insert_eos
):
    """Path for an Arrow-backed processed split."""
    eos_tag = "" if insert_eos else "_eosFalse"
    wrap_tag = "wrapped" if wrap else "unwrapped"
    tok = tokenizer_slug(tokenizer)
    return os.path.join(
        cache_dir, f"{dataset_name}_{tok}_{mode}_bs{block_size}_{wrap_tag}{eos_tag}.dat"
    )


# HF calls it 'validation'; on disk it is val.bin, as in nanoGPT.
_SPLIT_FILE = {"train": "train", "validation": "val", "test": "test"}


def memmap_tokens_path(memmap_dir, dataset_name, tokenizer_name, split):
    """Flat token file for one split: <dataset>/<tokenizer>/{train,val,test}.bin.

    Everything needed to read the file is either in this path or derivable: the
    dtype from the tokenizer's vocabulary size, the token count from the file
    size. There is no sidecar metadata.
    """
    name = _SPLIT_FILE.get(split, split)
    return os.path.join(
        memmap_dir, slugify(dataset_name), slugify(tokenizer_name), f"{name}.bin"
    )
