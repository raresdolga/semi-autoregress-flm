"""Correctness checks for a prepared memmap split.

    python -m tests.verify_memmap --dataset wikitext2 --split train \
        --tokenizer gpt2 --out-dir "$FLM_DATA_DIR/memmap" \
        --hf-cache-dir "$FLM_DATA_DIR/wikitext2"

Note what is deliberately *not* checked: equality with the Arrow backend. The
two pack differently on purpose -- Arrow frames every row as
`[BOS] + (block_size-2) + [EOS]`, memmap marks real document boundaries in a
length-agnostic stream -- so their token streams differ by design.
"""

import argparse
import hashlib
import os
import pickle
import shutil
import tempfile
import time
from itertools import chain

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import datasets  # HuggingFace
import numpy as np
import torch

import datamodules.datasets
from datamodules import backend_memmap, paths
from datamodules.backend_memmap import process_dataset
from datamodules.tokenization import build_tokenizer, detokenizer_for

PASS, FAIL = "  [ok]  ", "  [FAIL]"
_failures = []


def sha256(path):
    """Hash a token file in chunks -- these run to tens of GB."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(name, ok, detail=""):
    print(f"{PASS if ok else FAIL} {name}{' - ' + detail if detail else ''}")
    if not ok:
        _failures.append(name)
    return ok


def reference_stream(
    dataset_name,
    split,
    tokenizer_name,
    hf_cache_dir,
    np_dtype,
    limit_docs=None,
):
    """Tokenize the split in one process, no sharding, as ground truth."""
    tokenizer = build_tokenizer(tokenizer_name)
    raw, _ = datamodules.datasets.load_raw_split(
        dataset_name, split, hf_cache_dir, num_proc=1, streaming=False
    )
    if limit_docs is not None:
        raw = raw.select(range(min(limit_docs, len(raw))))
    column = datamodules.datasets.text_column(dataset_name)
    detok = detokenizer_for(dataset_name)
    eos = tokenizer.eos_token

    texts = []
    for text in raw[column]:
        if not text:
            texts.append(text)
            continue
        if detok is not None:
            text = detok(text)
        texts.append(text + eos)
    encoded = tokenizer(
        texts,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"]
    n_nonempty = sum(1 for t in texts if t)
    return (np.fromiter(chain(*encoded), dtype=np_dtype), n_nonempty)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--hf-cache-dir", default=None)
    p.add_argument("--window-size", type=int, default=128)
    p.add_argument("--limit-docs", type=int, default=None)
    p.add_argument(
        "--skip-determinism",
        action="store_true",
        help="skip the num_proc=1 re-prep (the slow check)",
    )
    args = p.parse_args(argv)

    datasets.disable_progress_bars()
    hf_cache = args.hf_cache_dir or os.path.join(args.out_dir, "hf-cache")
    tokens_file = paths.memmap_tokens_path(
        args.out_dir, args.dataset, args.tokenizer, args.split
    )

    tokenizer = build_tokenizer(args.tokenizer)
    # The id the EOS *string* becomes once embedded in a document -- which is
    # what prep writes. Not tokenizer.eos_token_id: encode() with special
    # tokens on reports 101 ([CLS]) for bert, where '[SEP]' alone gives 102.
    eos_id = tokenizer(tokenizer.eos_token, add_special_tokens=False)["input_ids"][0]
    # Everything the reader knows, it derives -- exactly as training does.
    np_dtype = np.uint16 if len(tokenizer) <= np.iinfo(np.uint16).max else np.uint32
    num_tokens = os.path.getsize(tokens_file) // np.dtype(np_dtype).itemsize

    print(f"\n{args.dataset}/{args.split} -> {tokens_file}")
    print(
        f"  {num_tokens} tokens, dtype={np.dtype(np_dtype).name} "
        f"({os.path.getsize(tokens_file)} bytes)"
    )

    actual = np.memmap(tokens_file, dtype=np_dtype, mode="r")

    # 1. Token identity against a single-process reference tokenization.
    ref, n_nonempty = reference_stream(
        args.dataset,
        args.split,
        args.tokenizer,
        hf_cache,
        np_dtype,
        args.limit_docs,
    )
    check(
        "token stream == single-process reference",
        len(ref) == len(actual) and bool(np.array_equal(ref, actual)),
        f"{len(actual)} vs {len(ref)} tokens",
    )

    # 2. Document markers landed where prep claims.
    n_eos = int((np.asarray(actual) == eos_id).sum())
    check(
        "one EOS per non-empty document",
        n_eos == n_nonempty,
        f"{n_eos} EOS vs {n_nonempty} non-empty docs",
    )

    # 3. A dtype mixup is the one failure nothing on disk can catch, now that
    #    the stream carries no metadata: uint16 and uint32 both parse cleanly.
    check(
        "all ids inside the tokenizer vocabulary",
        int(np.asarray(actual).max()) < len(tokenizer),
        f"max id {int(np.asarray(actual).max())} < {len(tokenizer)}",
    )

    ds = backend_memmap.MemmapTokenDataset(
        tokens_file, window_size=args.window_size, num_tokens=num_tokens, dtype=np_dtype
    )
    print(
        f"  {len(ds)} rows of {args.window_size}, "
        f"{num_tokens % args.window_size} tokens dropped"
    )

    # 4. The memmap handle is never pickled (an 18 GB OWT stream would OOM
    #    every spawn/DDP worker if it were).
    _ = ds[0]  # force the mapping open before pickling
    blob = len(pickle.dumps(ds))
    check(
        "pickled dataset stays tiny",
        blob < 2048,
        f"{blob} B (file is {os.path.getsize(tokens_file)} B)",
    )

    # 6. Identical batches for every worker configuration.
    refs = {}
    for num_workers in (0, 2):
        for persistent in (False, True):
            if num_workers == 0 and persistent:
                continue  # PyTorch rejects this combination
            kwargs = {"num_workers": num_workers}
            if num_workers:
                kwargs["persistent_workers"] = persistent
            loader = torch.utils.data.DataLoader(
                ds, batch_size=8, shuffle=False, **kwargs
            )
            got = []
            for epoch in range(2):
                for n, batch in enumerate(loader):
                    got.append(batch["input_ids"])
                    if n == 24:
                        break
            refs[(num_workers, persistent)] = torch.cat(got)
            del loader
    first = next(iter(refs.values()))
    check(
        "batches identical across num_workers/persistent_workers",
        all(torch.equal(first, v) for v in refs.values()),
        f"{len(refs)} configurations, 2 epochs each",
    )

    # 7. Prep is independent of --num-proc (its whole point). Re-prepares the
    #    whole split, so it is the slow check; --skip-determinism opts out.
    if not args.skip_determinism:
        alt = tempfile.mkdtemp(prefix="verify_memmap_")
        try:
            single = process_dataset(
                args.dataset,
                args.split,
                args.tokenizer,
                alt,
                hf_cache,
                num_proc=1,
                limit_docs=args.limit_docs,
                overwrite=True,
            )
            mine, theirs = sha256(tokens_file), sha256(single)
            check(
                "num_proc=1 and num_proc=N produce identical bytes",
                mine == theirs,
                f"{theirs[:16]} vs {mine[:16]}",
            )
        finally:
            # Never leave a second copy of the corpus behind.
            shutil.rmtree(alt, ignore_errors=True)

    # 8. Read cost, for the record.
    n = 2000
    start = time.time()
    for i in range(n):
        _ = ds[i % len(ds)]
    print(
        f"\n  read cost: {(time.time() - start) / n * 1e6:.1f} us/item "
        f'(Arrow + with_format("torch") measured 172 us/item)'
    )
    print(f"  {os.path.basename(tokens_file)}: {os.path.getsize(tokens_file)} bytes")

    print(
        f"\n{'FAILED: ' + ', '.join(_failures) if _failures else 'all checks passed'}"
    )
    return 1 if _failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
