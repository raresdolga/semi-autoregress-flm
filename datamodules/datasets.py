"""Corpora and the dataset-name dispatch.

Two sections:

  Locally generated datasets  corpora this repo builds itself (text8, the
                              synthetic debugging sets, the reflow tensors)
  Raw split loading           `load_raw_split`, the single dispatch point over
                              `data.train`/`data.valid` names, reaching the
                              above and the HF Hub alike. Shared by the Arrow
                              backend and by offline memmap prep, so both see
                              identical raw text.

This is the bottom of the data stack and imports no other module in the package,
which is what keeps the layering acyclic: `dataloaders` -> `*_backend` -> here.

Note the module name: it sits beside the HuggingFace `datasets` library, which
it imports. A bare `import datasets` anywhere in the package is absolute and
resolves to HuggingFace; refer to *this* module as `datamodules.datasets`, never
as `from . import datasets`, or the library gets shadowed.
"""

import json
import os
import shutil
import urllib
import zipfile

import datasets  # HuggingFace, not this module
import fsspec
import numpy as np
import requests
import torch
from einops import repeat

import utils

LOGGER = utils.get_logger(__name__)


# ---------------------------------------------------------------------------
# Locally generated datasets
# ---------------------------------------------------------------------------


class PseudoDataset(torch.utils.data.Dataset):
    """Prior samples drawn from the model, used as the input side of reflow."""

    def __init__(self, config, model):
        self.config = config
        self.num_samples = config.sampling.num_reflow_samples
        self.batch_size = config.loader.eval_batch_size
        self.create_data(model)

    def create_data(self, model):
        num_devices = torch.cuda.device_count()
        samples_per_gpu = self.num_samples // (self.batch_size * num_devices)
        if self.num_samples % (self.batch_size * num_devices) != 0:
            total_samples = (samples_per_gpu + 1) * self.batch_size * num_devices
        else:
            total_samples = samples_per_gpu * self.batch_size * num_devices
        with torch.no_grad():
            self.data = (
                model.prior_sample(total_samples, self.config.model.length)
                .cpu()
                .numpy()
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ReflowDataset(torch.utils.data.Dataset):
    """(x0, xT, t) triples written by `generate_reflow_data`."""

    def __init__(self, config):
        self.config = config
        self.load_data()

    def load_data(self):
        cache_dir = self.config.data.cache_dir
        self.x0 = np.load(os.path.join(cache_dir, "x0.npy"), allow_pickle=True)
        self.xT = np.load(os.path.join(cache_dir, "xT.npy"), allow_pickle=True)
        self.given_t = (
            np.load(os.path.join(cache_dir, "ts.npy"), allow_pickle=True)
            if os.path.exists(os.path.join(cache_dir, "ts.npy"))
            else None
        )

    def __len__(self):
        return len(self.x0)

    def __getitem__(self, idx):
        return {
            "input_ids": self.x0[idx],
            "attention_mask": np.ones_like(self.x0[idx]),
            "xT": self.xT[idx],
            "given_t": self.given_t[idx] if self.given_t is not None else 0.0,
        }


class SyntheticAlign(torch.utils.data.Dataset):
    """Every sequence is one token repeated: the simplest alignment target."""

    def __init__(self, config, N=100_000):
        self.config = config
        self.N = N
        self.L = self.config.model.length
        self.build_data()

    def build_data(self):
        self.x0 = np.random.randint(0, self.config.data.vocab_size, (self.N, 1))
        self.x0 = repeat(self.x0, "b 1 -> b l", l=self.L)

    def __len__(self):
        return len(self.x0)

    def __getitem__(self, idx):
        return {
            "input_ids": self.x0[idx],
            "attention_mask": np.ones_like(self.x0[idx]),
        }


def _generate_synthetic_data(dataset_size, seq_len, vocab_size):
    dataset = np.zeros((dataset_size, seq_len), dtype=int)
    # tokens representing sequence boundary
    dataset[:, 0] = vocab_size - 2  # bos
    dataset[:, -1] = vocab_size - 1  # eos

    for i in range(dataset_size):
        # sample from 0, 1, ..., vocab_size - 3
        temp = np.random.randint(vocab_size - 2)
        for j in reversed(range(1, seq_len - 1)):
            dataset[i, j] = temp
            if temp != 0:
                temp = temp // 4
            else:
                temp = np.random.randint(vocab_size - 2)

    return dataset


def generate_synthetic_dataset(
    train_dataset_size, validation_dataset_size, seq_len, vocab_size
):
    np.random.seed(42)
    train_data = torch.from_numpy(
        _generate_synthetic_data(train_dataset_size, seq_len, vocab_size)
    )
    train_dataset = datasets.Dataset.from_dict(
        {
            "input_ids": train_data,
            "attention_mask": torch.ones_like(train_data),
        }
    )
    train_dataset.set_format(type="torch")

    np.random.seed(41)
    validation_data = torch.from_numpy(
        _generate_synthetic_data(validation_dataset_size, seq_len, vocab_size)
    )
    validation_dataset = datasets.Dataset.from_dict(
        {
            "input_ids": validation_data,
            "attention_mask": torch.ones_like(validation_data),
        }
    )
    validation_dataset.set_format(type="torch")

    return {
        "train": train_dataset,
        "validation": validation_dataset,
    }


def generate_alpha8_dataset(
    train_dataset_size: int,
    validation_dataset_size: int,
    seq_len: int = 8,
    num_classes: int = 26,
):
    """Generate ultra-simple sequences where each sample is the same letter repeated.

    - input_ids shape: [N, 8]
    - values: integers in [0, num_classes-1]
    - attention_mask: ones with same shape
    """
    assert seq_len == 8, "alpha8 dataset expects seq_len=8"
    assert 1 <= num_classes <= 26, "num_classes should be within 1..26"

    def _make(N, seed):
        rng = np.random.default_rng(seed)
        letters = rng.integers(low=0, high=num_classes, size=(N, 1), dtype=np.int64)
        arr = np.repeat(letters, seq_len, axis=1)
        x = torch.from_numpy(arr.astype(np.int64))
        ds = datasets.Dataset.from_dict(
            {
                "input_ids": x,
                "attention_mask": torch.ones_like(x),
            }
        )
        ds.set_format(type="torch")
        return ds

    train_ds = _make(train_dataset_size, seed=1234)
    valid_ds = _make(validation_dataset_size, seed=1235)
    return {
        "train": train_ds,
        "validation": valid_ds,
    }


def get_lambada_test_dataset():
    url = "https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl"

    def read_jsonl_to_list(url):
        response = requests.get(url, stream=True)
        data_list = []

        # Process each line in the response content
        for line in response.iter_lines(decode_unicode=True):
            if line:
                data = json.loads(line)
                data_list.append(data)

        return data_list

    lambada_data = read_jsonl_to_list(url)
    dataset = datasets.Dataset.from_list(lambada_data)
    return dataset


def get_text8_dataset(cache_dir, max_seq_length=256, drop_last=True, crop_train=False):
    """Adapted from:
    https://github.com/google-research/google-research/blob/master/d3pm/text/datasets.py#L344

    Args:
      cache_dir: str, path to cache directory.
      max_seq_length: int, maximum length of sequences.
          (default: 256, as in D3PM codebase.)
      drop_last: bool, whether to drop the last incomplete
          batch. (default: True, as in D3PM codebase.)
      crop_train: bool, whether to subsample contiguous
          subsequences from training example. serves to
          make sure transformer models with absolute position
          embeddings do not have incorrect position-wise
          marginals. (default: False, but necessary to match D3PM AR)

    Returns:
      dataset: dataset.DatasetDict, with keys 'train',
          'valid', 'test'.
    """
    url = "http://mattmahoney.net/dc/text8.zip"
    if not crop_train:
        cache_dir = f"{cache_dir}/text8"
    else:
        cache_dir = f"{cache_dir}/text8-crop-train"
    split_names = ["train", "validation", "test"]
    if not all(
        [utils.fsspec_exists(os.path.join(cache_dir, split)) for split in split_names]
    ):
        # Check if raw data exists
        raw_cache_dir = os.path.join(cache_dir, "raw_data")
        if not all(
            [
                utils.fsspec_exists(os.path.join(raw_cache_dir, f"text8.{split}.txt"))
                for split in split_names
            ]
        ):
            if not utils.fsspec_exists(os.path.join(raw_cache_dir, "text8.zip")):
                utils.fsspec_mkdirs(raw_cache_dir, exist_ok=True)
                LOGGER.info("Downloading text8 from URL {}.".format(url))
                with (
                    urllib.request.urlopen(url) as in_stream,
                    open(os.path.join(raw_cache_dir, "text8.zip"), "wb") as out_file,
                ):
                    shutil.copyfileobj(in_stream, out_file)

            with fsspec.open(os.path.join(raw_cache_dir, "text8.zip"), "rb") as f:
                rawdata = zipfile.ZipFile(f).read("text8").decode("utf-8")

            # Splits taken from D3PM codebase
            splits = {
                "train": rawdata[:90000000],
                "validation": rawdata[90000000:95000000],
                "test": rawdata[95000000:],
            }

            for split, data in splits.items():
                _path = os.path.join(raw_cache_dir, f"text8.{split}.txt")
                with fsspec.open(_path, "w") as f:
                    f.write(data)
        else:
            splits = {}
            for split in split_names:
                _path = os.path.join(raw_cache_dir, f"text8.{split}.txt")
                with fsspec.open(_path, "r") as f:
                    splits[split] = f.read()

        # Chunk and save as datasets.DatasetDict
        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i : i + n]

        dataset_dict = {}
        for k, v in splits.items():
            if k == "train" and crop_train == True:
                chunk_size = 2 * max_seq_length
            else:
                chunk_size = max_seq_length
            text = list(chunks(v, chunk_size))
            if drop_last and len(text[-1]) < chunk_size:
                text = text[:-1]
            dataset_dict[k] = datasets.Dataset.from_dict({"text": text})
        dataset = datasets.DatasetDict(dataset_dict)
        dataset.save_to_disk(cache_dir)
    else:
        dataset = datasets.load_from_disk(cache_dir)

    return dataset


# ---------------------------------------------------------------------------
# Raw split loading
# ---------------------------------------------------------------------------

# Datasets loaded at a fixed split, so `dataset[mode]` would be a KeyError.
_PRELOADED_SPLIT = [
    "lambada",
    "openwebtext-train",
    "openwebtext-valid",
    "Skylion007/openwebtext-1k",
    "openwebtext-20",
]

# Already-finished torch datasets: no tokenization or grouping applies.
_SELF_CONTAINED = ["synthetic", "synthetic-alpha8"]


def text_column(dataset_name):
    """Name of the raw-text column for `dataset_name`."""
    if dataset_name == "ptb":
        return "sentence"
    if "scientific_papers" in dataset_name:
        return "article"
    return "text"


def columns_to_drop(dataset_name):
    """Non-token columns to remove after tokenizing."""
    if dataset_name == "ptb":
        return "sentence"
    if "scientific_papers" in dataset_name:
        return ["article", "abstract", "section_names"]
    if dataset_name == "ag_news":
        return ["text", "label"]
    return "text"


def bos_eos_ids(tokenizer):
    """BOS/EOS ids exactly as the original pipeline derived them.

    Read back through `encode()` rather than `bos_token_id`/`eos_token_id`, which
    is load-bearing: GPT-2 carries a `BertProcessing` post-processor (see
    `tokenization.build_tokenizer`) so both come back as 50256, and
    bert-base-uncased returns 101 ([CLS]) for *both*. Every existing checkpoint
    was trained against these ids, so do not "fix" this without a re-prep.
    """
    eos = tokenizer.encode(tokenizer.eos_token)[0]
    bos = tokenizer.encode(tokenizer.bos_token)[0]
    return bos, eos


def load_raw_split(
    dataset_name,
    mode,
    cache_dir,
    *,
    num_proc,
    streaming,
    revision=None,
    wrap=True,
    block_size=1024,
    config=None,
):
    """Load the raw split named by `dataset_name`.

    Returns `(data, is_ready)`. `is_ready=True` means `data` is already a
    finished dataset yielding `input_ids`, so the caller must not tokenize or
    group it.
    """
    if dataset_name == "wikitext103":
        dataset = datasets.load_dataset(
            "wikitext",
            name="wikitext-103-raw-v1",
            cache_dir=cache_dir,
            revision=revision,
        )
    elif dataset_name == "wikitext2":
        dataset = datasets.load_dataset(
            "wikitext", name="wikitext-2-raw-v1", cache_dir=cache_dir, revision=revision
        )
    elif dataset_name == "ptb":
        dataset = datasets.load_dataset(
            "ptb_text_only", cache_dir=cache_dir, revision=revision
        )
    elif dataset_name == "lambada":
        dataset = get_lambada_test_dataset()
    elif dataset_name == "synthetic-align":
        return SyntheticAlign(config), True
    elif dataset_name == "text8":
        assert wrap
        assert revision is None
        dataset = get_text8_dataset(cache_dir, max_seq_length=block_size)
    elif dataset_name == "text8-1k":
        assert wrap
        assert revision is None
        full_dataset = get_text8_dataset(cache_dir, max_seq_length=block_size)
        train_subset = full_dataset["train"].select(
            range(min(1000, len(full_dataset["train"])))
        )
        dataset = datasets.DatasetDict(
            {
                "train": train_subset,
                "validation": train_subset,
                "test": full_dataset["test"],
            }
        )
    elif dataset_name == "text8-crop":
        assert revision is None
        dataset = get_text8_dataset(
            cache_dir, max_seq_length=block_size, crop_train=True
        )
    elif dataset_name == "openwebtext-train":
        dataset = datasets.load_dataset(
            "openwebtext",
            split="train[:-100000]",
            cache_dir=cache_dir,
            revision=revision,
            streaming=False,
            num_proc=num_proc,
            trust_remote_code=True,
        )
    elif dataset_name == "openwebtext-valid":
        dataset = datasets.load_dataset(
            "openwebtext",
            split="train[-100000:]",
            cache_dir=cache_dir,
            revision=revision,
            streaming=False,
            num_proc=num_proc,
            trust_remote_code=True,
        )
    elif dataset_name == "scientific_papers_arxiv":
        dataset = datasets.load_dataset(
            "scientific_papers",
            "arxiv",
            trust_remote_code=True,
            cache_dir=cache_dir,
            streaming=streaming,
            revision=revision,
        )
    elif dataset_name == "scientific_papers_pubmed":
        dataset = datasets.load_dataset(
            "scientific_papers",
            "pubmed",
            trust_remote_code=True,
            cache_dir=cache_dir,
            streaming=streaming,
            revision=revision,
        )
    elif dataset_name == "ag_news":
        dataset = datasets.load_dataset(
            "ag_news", cache_dir=cache_dir, streaming=streaming, revision=revision
        )
    elif dataset_name == "synthetic-alpha8":
        # Extremely simple debugging dataset: sequences like aaaaaaaa, gggggggg (IDs 0..25)
        dataset = generate_alpha8_dataset(
            train_dataset_size=10000,
            validation_dataset_size=128,  # Small validation set for fast debugging
            seq_len=8,
            num_classes=26,
        )
    elif dataset_name == "synthetic":
        assert streaming
        assert wrap  # i.e., no pad tokens
        dataset = generate_synthetic_dataset(
            train_dataset_size=100000,
            validation_dataset_size=1024,
            seq_len=32,
            vocab_size=256,
        )
    elif dataset_name == "reflow-dataset":
        return ReflowDataset(config), True
    else:
        dataset = datasets.load_dataset(
            dataset_name,
            cache_dir=cache_dir,
            streaming=streaming,
            trust_remote_code=True,
            revision=revision,
        )

    if dataset_name in _PRELOADED_SPLIT:
        return dataset, False
    data = dataset[mode]
    return data, dataset_name in _SELF_CONTAINED
