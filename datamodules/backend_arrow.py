"""Arrow token backend: tokenize once, group into blocks, save_to_disk, mmap.

This is the original (default) pipeline. `datasets.load_from_disk` memory-maps the
Arrow file, so reads are lazy; the cost is that `attention_mask` is materialized
as float32 ones on disk and each row access goes through pyarrow (~172 us/item
measured). See `backend_memmap` for the flat-uint16 alternative.

NOTE on reproducibility: `_group_texts` runs under `map(batched=True,
batch_size=1000)` across `num_proc` contiguous shards, so the concatenation
remainder is dropped *per map batch*. Row counts therefore depend on `num_proc`
(wikitext2 train: 18414 at 1 proc, 18413 at 2, 18409 at 4). The memmap backend
concatenates each split once and is `num_proc`-independent.
"""

import functools
import itertools
import os

import datasets  # HuggingFace
import torch

import datamodules.datasets  # this package's dataset dispatch, not the library
import utils
from . import paths
from .tokenization import detokenizer_for

LOGGER = utils.get_logger(__name__)


def _group_texts(examples, block_size, bos, eos):
    # Concatenate all texts.
    concatenated_examples = list(itertools.chain(*examples["input_ids"]))
    total_length = len(concatenated_examples)
    # TODO(yair): look into not dropping the remainder but rather padding it.
    # We drop the small remainder, and if the total_length < block_size - 2
    # we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of
    # this drop, you can customize this part to your needs.
    new_block_size = block_size - 2  # [BOS] and [EOS] to be added
    total_length = (total_length // new_block_size) * new_block_size
    # Split by chunks of max_len.
    result = {}
    _values = []
    _attn_masks = []
    for i in range(0, total_length, new_block_size):
        _values.append([bos] + concatenated_examples[i : i + new_block_size] + [eos])
        _attn_masks.append(torch.ones(block_size))
    result["input_ids"] = _values
    result["attention_mask"] = _attn_masks
    return result


def get_dataset(
    dataset_name,
    tokenizer,
    mode,
    *,
    wrap,
    cache_dir,
    block_size,
    insert_eos,
    streaming,
    num_proc,
    revision=None,
    config=None,
):
    """Build one split, tokenizing and caching it on first use.

    Counterpart to `backend_memmap.get_dataset`: same name and same leading
    arguments, then whatever each backend actually honors. Arrow supports
    `wrap=False` (per-document padded rows) and `streaming`; the packed memmap
    format cannot express either. `config` is only threaded through for the
    `synthetic-align` / `reflow-dataset` branches of `load_raw_split`.
    """
    _path = paths.arrow_cache_path(
        cache_dir, dataset_name, tokenizer, mode, block_size, wrap, insert_eos
    )
    if utils.fsspec_exists(_path):
        LOGGER.info(f"Loading data from: {_path}")
        return datasets.load_from_disk(_path).with_format("torch")
    LOGGER.info(f"Generating new data at: {_path}")
    LOGGER.info(f"{streaming=}")

    crop_train = dataset_name == "text8-crop"
    if mode == "train" and crop_train:
        # double block size for sub-sampling
        block_size *= 2

    data, is_ready = datamodules.datasets.load_raw_split(
        dataset_name,
        mode,
        cache_dir,
        num_proc=num_proc,
        streaming=streaming,
        revision=revision,
        wrap=wrap,
        block_size=block_size,
        config=config,
    )
    if is_ready:
        return data

    bos, eos = datamodules.datasets.bos_eos_ids(tokenizer)
    detokenizer = detokenizer_for(dataset_name)
    text_col = datamodules.datasets.text_column(dataset_name)

    def _apply_detokenizer(detokenizer):
        def detok(text):
            for i, t in enumerate(text, 0):
                text[i] = detokenizer(t)
            return text

        return detok

    def preprocess_and_tokenize(example):
        text = example[text_col]

        if detokenizer is not None:
            text = _apply_detokenizer(detokenizer)(text)

        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "right"

        if wrap:
            tokens = tokenizer(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            if insert_eos:
                tokens = {"input_ids": [t + [eos] for t in tokens["input_ids"]]}
            # Still missing BOS, but will be added in group_texts
        else:
            tokens = tokenizer(
                text,
                max_length=block_size,
                padding="max_length",
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_token_type_ids=True,
            )
        return tokens

    if streaming:
        tokenized_dataset = data.map(preprocess_and_tokenize, batched=True)
    else:
        tokenized_dataset = data.map(
            preprocess_and_tokenize,
            batched=True,
            num_proc=num_proc,
            load_from_cache_file=True,
            desc="Tokenizing",
        )
    tokenized_dataset = tokenized_dataset.remove_columns(
        datamodules.datasets.columns_to_drop(dataset_name)
    )

    if not wrap:
        if not streaming:
            tokenized_dataset.save_to_disk(_path)
        return tokenized_dataset.with_format("torch")

    group_texts = functools.partial(
        _group_texts, block_size=block_size, bos=bos, eos=eos
    )
    if streaming:
        chunked_dataset = tokenized_dataset.map(group_texts, batched=True)
    else:
        chunked_dataset = tokenized_dataset.map(
            group_texts,
            batched=True,
            num_proc=num_proc,
            load_from_cache_file=True,
            desc="Grouping",
        )
        chunked_dataset.save_to_disk(_path)
    chunked_dataset = chunked_dataset.with_format("torch")
    return chunked_dataset
