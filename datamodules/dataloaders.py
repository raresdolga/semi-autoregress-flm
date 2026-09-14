"""Iteration order and batching: the top of the data stack.

Two sections:

  Samplers      which index comes next, resumable mid-epoch after a crash
  DataLoaders   `build_split` picks the token backend for a split,
                `get_dataloaders` wraps the result for Lightning

Depends on `datasets` and both backends; nothing in the package depends on this
module, so the layering stays acyclic:
`dataloaders` -> `backend_{arrow,memmap}` -> `datasets`.
"""

import math
import os
import typing

import torch

from . import backend_arrow, backend_memmap
from .datasets import PseudoDataset

# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------
# Adapted from:
# https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/fault_tolerant_sampler.py
# Both track a `counter` of yielded indices, so a mid-epoch checkpoint resumes
# at the same position in the same permutation.


class RandomFaultTolerantSampler(torch.utils.data.RandomSampler):
    def __init__(self, *args, generator=None, **kwargs):
        # TD [2022-07-17]: We don't force the seed to be zero. We generate random seed,
        # which should be reproducible if pl.seed_everything was called beforehand.
        # This means that changing the seed of the experiment will also change the
        # sampling order.
        if generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator().manual_seed(seed)
        kwargs.pop("shuffle", None)
        super().__init__(*args, generator=generator, **kwargs)
        self.counter = 0
        self.restarting = False

    def state_dict(self):
        return {"random_state": self.generator.get_state(), "counter": self.counter}

    def load_state_dict(self, state_dict):
        self.generator.set_state(state_dict.get("random_state"))
        self.counter = state_dict["counter"]
        # self.start_counter = self.counter
        self.restarting = True

    # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
    # epoch, and subsequent epoch will have very few batches.

    def __iter__(self) -> typing.Iterator[int]:
        n = len(self.data_source)

        self.state = self.generator.get_state()
        indices = torch.randperm(n, generator=self.generator).tolist()

        if not self.restarting:
            self.counter = 0
        else:
            indices = indices[self.counter :]
            self.restarting = False

        for index in indices:
            self.counter += 1
            yield index

        self.counter = 0


class FaultTolerantDistributedSampler(torch.utils.data.DistributedSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter = 0
        self.restarting = False

    def state_dict(self):
        return {"epoch": self.epoch, "counter": self.counter}

    def load_state_dict(self, state_dict):
        self.epoch = state_dict["epoch"]
        self.counter = state_dict["counter"]
        self.restarting = True

    # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
    # epoch, and subsequent epoch will have very few batches.
    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # type: ignore[arg-type]
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        if not self.restarting:
            self.counter = 0
        else:
            indices = indices[self.counter :]
            self.restarting = False

        for index in indices:
            self.counter += 1
            yield index

        self.counter = 0


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

# Datasets whose validation split is named 'test'.
_TEST_AS_VALIDATION = ["text8", "lm1b", "ag_news"]


def build_split(config, tokenizer, dataset_name, mode, insert_eos):
    """Build one split with the token backend named by `data.backend`.

    This is the only place config is unpacked into backend arguments. Both
    backends expose `get_dataset(dataset_name, tokenizer, mode, ...)`, but each
    takes only the options it honors, so the dispatch stays an explicit branch
    rather than a registry lookup.
    """
    backend = config.data.get("backend", "arrow")
    if backend == "arrow":
        return backend_arrow.get_dataset(
            dataset_name,
            tokenizer,
            mode,
            wrap=config.data.wrap,
            cache_dir=config.data.cache_dir,
            block_size=config.model.length,
            insert_eos=insert_eos,
            streaming=config.data.streaming,
            # Not loader.num_workers: that is a runtime knob (0 in
            # train_flm.sh), and using it here tokenized with 2 of 20 cores
            # *and* made row counts depend on it -- see backend_arrow's header.
            num_proc=config.data.get("prep_num_proc", len(os.sched_getaffinity(0))),
            revision=config.data.get(
                "train_revision" if mode == "train" else "valid_revision", None
            ),
            config=config,
        )
    if backend == "memmap":
        if not config.data.wrap:
            raise ValueError(
                "backend=memmap requires data.wrap=True (it stores a packed "
                "token stream, not per-document padded rows). Use "
                f"backend=arrow for data.wrap=False configs such as {dataset_name}."
            )
        if config.data.streaming:
            raise ValueError(
                "backend=memmap is incompatible with data.streaming=True; the "
                "point of prepared data is that it is already local."
            )
        return backend_memmap.get_dataset(
            dataset_name,
            tokenizer,
            mode,
            memmap_dir=backend_memmap.resolve_memmap_dir(config),
            tokenizer_name=config.data.tokenizer_name_or_path,
            window_size=config.model.length,
        )
    raise ValueError(f'Unknown data.backend={backend!r} (expected "arrow" or "memmap")')


def get_dataloaders(
    config, tokenizer, skip_train=False, skip_valid=False, valid_seed=None
):
    num_gpus = torch.cuda.device_count()
    assert config.loader.global_batch_size == (
        config.loader.batch_size
        * config.trainer.num_nodes
        * num_gpus
        * config.trainer.accumulate_grad_batches
    )
    if (
        config.loader.global_batch_size
        % (num_gpus * config.trainer.accumulate_grad_batches)
        != 0
    ):
        raise ValueError(
            f"Train Batch Size {config.loader.global_batch_size}"
            f"not divisible by {num_gpus} gpus with accumulation "
            f"{config.trainer.accumulate_grad_batches}."
        )
    if config.loader.eval_global_batch_size % num_gpus != 0:
        raise ValueError(
            f"Eval Batch Size for {config.loader.eval_batch_size} "
            f"not divisible by {num_gpus}."
        )

    train_set = (
        None
        if skip_train
        else build_split(
            config, tokenizer, config.data.train, "train", config.data.insert_train_eos
        )
    )

    validation_split = (
        "test" if config.data.valid in _TEST_AS_VALIDATION else "validation"
    )
    valid_set = (
        None
        if skip_valid
        else build_split(
            config,
            tokenizer,
            config.data.valid,
            validation_split,
            config.data.insert_valid_eos,
        )
    )

    if skip_train:
        train_loader = None
    else:
        if config.data.streaming:
            # An IterableDataset cannot be shuffled by the DataLoader, so
            # without this the train stream was consumed in corpus order.
            train_set = train_set.shuffle(
                seed=config.seed,
                buffer_size=config.data.get("shuffle_buffer_size", 10_000),
            )
            train_shuffle = False
        else:
            train_shuffle = True
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=config.loader.batch_size,
            shuffle=train_shuffle,
            num_workers=config.loader.num_workers,
            pin_memory=config.loader.pin_memory,
            # PyTorch rejects persistent_workers=True when num_workers == 0,
            # which is the fastest setting for a local mmap.
            persistent_workers=config.loader.num_workers > 0,
        )
        train_loader.tokenizer = tokenizer

    if skip_valid:
        valid_loader = None
    else:
        if valid_seed is None:
            shuffle_valid = False
            generator = None
        else:
            shuffle_valid = True
            generator = torch.Generator().manual_seed(valid_seed)
        valid_loader = torch.utils.data.DataLoader(
            valid_set,
            batch_size=config.loader.eval_batch_size,
            shuffle=shuffle_valid,
            generator=generator,
            num_workers=config.loader.num_workers,
            pin_memory=config.loader.pin_memory,
            persistent_workers=config.loader.num_workers > 0,
        )
        # Will be used in generative perplexity calculation
        valid_loader.tokenizer = tokenizer

    return train_loader, valid_loader


def get_pseudo_dataloader(config, tokenizer, model):
    dataset = PseudoDataset(config, model)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config.loader.eval_batch_size,
        shuffle=False,
        num_workers=config.loader.num_workers,
        pin_memory=config.loader.pin_memory,
        persistent_workers=config.loader.num_workers > 0,
    )
