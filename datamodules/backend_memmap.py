"""Flat-uint16 token backend, following nanoGPT / flash-attention.

One file per split -- train.bin / val.bin / test.bin -- holding a contiguous
stream of token ids, and nothing else. There is no sidecar metadata: the dtype
comes from the tokenizer's vocabulary size, the token count from the file size,
the dataset/tokenizer/split from the path, and completeness from the fact that
prep publishes the file with an atomic rename, so it can never be seen partial.

Document boundaries are marked in the stream itself: prep appends the EOS token
to each document's raw text before tokenizing, as flash-attention does. So rows
are cut as tokens[i*L : (i+1)*L] for whatever model.length a run asks for, and
one prepared split serves every window size.
"""

import argparse
import mmap
import os
from itertools import chain

import datasets  # HuggingFace
import numpy as np
import torch

import datamodules.datasets  # this package's dataset dispatch, not the library
from . import paths
from .tokenization import build_tokenizer, detokenizer_for

# ---------------------------------------------------------------------------
# Runtime Dataset
# ---------------------------------------------------------------------------


class MemmapTokenDataset(torch.utils.data.Dataset):
    """Fixed-length windows over a flat token stream.

    Map-style (`__len__` + integer `__getitem__`), so the fault-tolerant samplers
    in `dataloaders.py` work unchanged, mid-epoch resume included, and the default
    collate stacks the batch.

    Args:
      path: the split's .bin -- one contiguous run of ids, no header.
      window_size: ids per row, i.e. `model.length`. Row i is
        `tokens[i*window_size : (i+1)*window_size]`, so the same file serves any
        window size and a trailing partial row is dropped.
      num_tokens: how many ids the file holds. The caller derives it from the
        file size rather than reading it from anywhere.
      dtype: numpy dtype the ids are stored as (`np.uint16`, or `np.uint32` for
        a vocabulary above 65535). Must match what prep wrote -- the bytes
        parse cleanly either way, so a mismatch yields plausible garbage rather
        than an error.
    """

    def __init__(self, path, window_size, num_tokens, dtype):
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        self.path = path
        self.dtype = dtype
        self.window_size = window_size
        self.num_tokens = num_tokens
        self.num_rows = num_tokens // window_size  # drop the remainder
        self._arr = None

    def __len__(self):
        return self.num_rows

    def _tokens(self):
        if self._arr is None:
            self._arr = np.memmap(self.path, dtype=self.dtype, mode="r")
        return self._arr

    def __getitem__(self, i):
        start = i * self.window_size
        window = self._tokens()[start : start + self.window_size]
        # int64 for the embedding lookup; ones() is float32, as the Arrow path
        # stores it, and must be a fresh writable tensor because the loss
        # assigns into it in place (trainer_base._loss).
        return {
            "input_ids": torch.from_numpy(window.astype(np.int64)),
            "attention_mask": torch.ones(self.window_size),
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_arr"] = None  # never pickle the mapping
        return state


def resolve_memmap_dir(config):
    """Root for prepared memmap data, from `data.memmap_dir` or $FLM_DATA_DIR.

    Must be run-invariant: `hydra.job.chdir: true` makes a relative path land
    inside the per-run output directory, which would re-prepare every run.
    """
    configured = config.data.get("memmap_dir", None)
    if configured:
        return str(configured)
    env = os.environ.get("FLM_DATA_DIR")
    if env:
        return os.path.join(env, "memmap")
    raise RuntimeError(
        "backend=memmap needs a run-invariant location for prepared data. Set "
        "data.memmap_dir in your data config, or export FLM_DATA_DIR (see "
        "scripts/env.sh). A relative path will not work because "
        "hydra.job.chdir moves the working directory per run."
    )


def get_dataset(
    dataset_name, tokenizer, mode, *, memmap_dir, tokenizer_name, window_size
):
    """Load a prepared split as a map-style dataset.

    Counterpart to `backend_arrow.get_dataset`: There is no `wrap`,
    `streaming` or `num_proc` here -- a packed stream cannot express padded
    per-document rows, is already local, and was tokenized offline. Nothing is
    prepared on the fly; a missing split raises and names the prep command.
    """
    path = paths.memmap_tokens_path(memmap_dir, dataset_name, tokenizer_name, mode)
    if not os.path.exists(path):
        raise RuntimeError(
            f"No prepared memmap data at {path}\n"
            "Prepare it offline first, e.g.:\n"
            "  python -m datamodules.backend_memmap --dataset <name> "
            "--splits train,validation --tokenizer <tok> "
            '--out-dir "$FLM_DATA_DIR/memmap"'
        )
    # Must match what process_dataset picked; both derive it from the tokenizer.
    dtype = np.uint16 if len(tokenizer) <= np.iinfo(np.uint16).max else np.uint32
    return MemmapTokenDataset(
        path,
        window_size=window_size,
        num_tokens=os.path.getsize(path) // np.dtype(dtype).itemsize,
        dtype=dtype,
    )


# ---------------------------------------------------------------------------
# Offline preparation
# ---------------------------------------------------------------------------


def _write_ids_to_disk(example, filename, dtype):
    """Write one row's ids at its global offset. Runs in a `map` worker."""
    with open(filename, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 0)
        length = len(example["input_ids"])
        start = example["len_offset"] - length
        arr = np.ndarray(
            (length,), dtype=dtype, buffer=mm, offset=np.dtype(dtype).itemsize * start
        )
        arr[:] = example["input_ids"]
        mm.flush()


def process_dataset(
    dataset_name,
    split,
    tokenizer_name,
    out_dir,
    hf_cache_dir,
    *,
    num_proc=None,
    limit_docs=None,
    overwrite=False,
):
    """Tokenize one split into a flat {train,val,test}.bin. Returns its path.
    The file is published by atomic rename, so it is never visible half-written.

    Args:
      dataset_name: as in `data.train` / `data.valid`, e.g. 'wikitext2'.
      split: HF split name ('train', 'validation', 'test'); 'validation' lands
        in val.bin.
      tokenizer_name: as in `data.tokenizer_name_or_path`. Part of the output
        path, and what training re-derives the dtype from.
      out_dir: memmap root. Must match `data.memmap_dir` at training time --
        prep is argparse, not Hydra, so it cannot read that config itself.
      hf_cache_dir: where the raw corpus is downloaded. Point it at the Arrow
        backend's `data.cache_dir` to avoid fetching the corpus twice.
      num_proc: tokenization processes (default: all cores). Affects speed only.
      limit_docs: truncate to N documents, for smoke tests.
      overwrite: re-tokenize even if the .bin already exists.
    """
    tokens_file = paths.memmap_tokens_path(out_dir, dataset_name, tokenizer_name, split)
    if os.path.exists(tokens_file) and not overwrite:
        print(f"[skip] {tokens_file} already prepared (--overwrite to redo)")
        return tokens_file

    num_proc = num_proc or len(os.sched_getaffinity(0))
    tokenizer = build_tokenizer(tokenizer_name)
    # uint16 up to 65535 ids (gpt2 needs 50258); a 128k-vocab tokenizer would
    # wrap silently, so widen instead. get_dataset repeats this to read back.
    if len(tokenizer) <= np.iinfo(np.uint16).max:
        np_dtype = np.uint16
    else:
        np_dtype = np.uint32

    raw, is_ready = datamodules.datasets.load_raw_split(
        dataset_name,
        split,
        hf_cache_dir,
        num_proc=num_proc,
        streaming=False,
    )
    if is_ready:
        raise ValueError(
            f"{dataset_name} is self-contained (synthetic/reflow) and has no "
            "raw text to tokenize. Use data.backend=arrow."
        )
    if limit_docs is not None:
        raw = raw.select(range(min(limit_docs, len(raw))))

    text_column = datamodules.datasets.text_column(dataset_name)
    detokenize = detokenizer_for(dataset_name)
    eos_token = tokenizer.eos_token

    print(
        f"[{dataset_name}/{split}] {len(raw)} docs -> {tokens_file}\n"
        f"  tokenizer={tokenizer_name} vocab={len(tokenizer)} "
        f"dtype={np.dtype(np_dtype).name}"
        f" eos={eos_token!r}"
        f" num_proc={num_proc}"
    )

    def mark(text):
        # Empty documents stay empty, so blank corpus lines do not decay into
        # pure separator noise (flash-attn: `(seq + eos) if seq else seq`).
        if not text:
            return text
        if detokenize is not None:
            text = detokenize(text)
        return text + eos_token

    def tokenize_concat(examples):
        # 'attention_mask' is not stored: it is all ones for packed data.
        encoded = tokenizer(
            [mark(t) for t in examples[text_column]],
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        ids = np.fromiter(chain(*encoded["input_ids"]), dtype=np_dtype)
        # A list, because this is a batched map returning one row.
        return {"input_ids": [ids], "len": [len(ids)]}

    tokenized = raw.map(
        tokenize_concat,
        batched=True,
        num_proc=num_proc,
        remove_columns=raw.column_names,
        desc=f"Tokenizing {dataset_name}/{split}",
    )

    tokenized = tokenized.add_column("len_offset", np.cumsum(tokenized["len"]))
    num_tokens = int(tokenized[-1]["len_offset"])

    os.makedirs(os.path.dirname(tokens_file), exist_ok=True)
    tmp_path = tokens_file + ".tmp"
    # Pre-size the file so workers can scatter-write into their own offsets.
    with open(tmp_path, "wb") as f:
        f.truncate(num_tokens * np.dtype(np_dtype).itemsize)
    tokenized.map(
        _write_ids_to_disk,
        fn_kwargs={"filename": tmp_path, "dtype": np_dtype},
        batched=False,
        num_proc=num_proc,
        desc="Concatenating examples",
    )
    # Atomic: the split only ever appears complete, so its mere existence is
    # what marks it ready. Nothing else is written.
    os.replace(tmp_path, tokens_file)

    print(f"  wrote {num_tokens} tokens ({os.path.getsize(tokens_file)} bytes)")
    return tokens_file


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Tokenize a corpus into a flat token stream for "
        "data.backend=memmap."
    )
    p.add_argument(
        "--dataset",
        required=True,
        help="dataset name, same values as data.train/data.valid",
    )
    p.add_argument(
        "--splits",
        default="train,validation",
        help="comma-separated splits (default: train,validation)",
    )
    p.add_argument(
        "--tokenizer", required=True, help="same value as data.tokenizer_name_or_path"
    )
    p.add_argument(
        "--out-dir", required=True, help="memmap root; must match data.memmap_dir"
    )
    p.add_argument(
        "--hf-cache-dir",
        default=None,
        help="HF download cache (default: <out-dir>/hf-cache)",
    )
    p.add_argument(
        "--num-proc",
        type=int,
        default=None,
        help="tokenization processes (default: all cores)",
    )
    p.add_argument(
        "--limit-docs",
        type=int,
        default=None,
        help="truncate to N documents (for smoke tests)",
    )
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    # Set here, not at module scope: this module is imported by every
    # training run, and disabling tokenizer threads globally would also slow
    # the Arrow path's tokenization. HF reads it at fork time, so setting it
    # before the map calls below is enough.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    datasets.disable_progress_bars()
    hf_cache_dir = args.hf_cache_dir or os.path.join(args.out_dir, "hf-cache")
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        process_dataset(
            args.dataset,
            split,
            args.tokenizer,
            args.out_dir,
            hf_cache_dir,
            num_proc=args.num_proc,
            limit_docs=args.limit_docs,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
