"""Entry point for the synthetic Markov-chain experiment.

Run: scripts/debug/train_synthetic.sh mc.K=64 mc.b=2 mc.D=8
"""

import hydra
import omegaconf
import torch
from torch.utils.data import TensorDataset

from flm import utils
from flm.synthetic.mc import MC


def build_mc(config):
    """Creates the Markov chain from `config.mc`."""
    offsets = config.mc.offsets
    if offsets is not None:
        offsets = list(offsets)
    return MC(
        K=config.mc.K,
        D=config.mc.D,
        b=config.mc.b,
        offsets=offsets,
        seed=config.mc.seed,
    )


def make_datasets(mc, n_train, n_test, train_seed=0, test_seed=1):
    """Draws clean train and test sequences from the chain.

    Each item is a sequence of token indices of shape (D,), dtype long. Noise is
    not added here: x_t is sampled on the fly during training and evaluation.

    Args:
      mc: The Markov chain to sample from.
      n_train: Number of training sequences.
      n_test: Number of test sequences.
      train_seed: Seed of the generator for the training set.
      test_seed: Seed of the generator for the test set.

    Returns:
      train_ds: TensorDataset holding a (n_train, D) tensor.
      test_ds: TensorDataset holding a (n_test, D) tensor.
    """
    train = mc.sample(n_train, generator=torch.Generator().manual_seed(train_seed))
    test = mc.sample(n_test, generator=torch.Generator().manual_seed(test_seed))
    return TensorDataset(train), TensorDataset(test)


@hydra.main(version_base=None, config_path="../../configs", config_name="synthetic")
def main(config):
    """Builds the chain and its train/test datasets."""
    logger = utils.get_logger(__name__)
    logger.info("Config:\n%s", omegaconf.OmegaConf.to_yaml(config))
    torch.manual_seed(config.seed)

    mc = build_mc(config)
    logger.info("MC: K=%d, D=%d, b=%d, offsets=%s", mc.K, mc.D, mc.b, mc.offsets)

    train_ds, test_ds = make_datasets(
        mc,
        n_train=config.data.n_train,
        n_test=config.data.n_test,
        train_seed=config.data.train_seed,
        test_seed=config.data.test_seed,
    )
    logger.info(
        "train: %s, test: %s",
        tuple(train_ds.tensors[0].shape),
        tuple(test_ds.tensors[0].shape),
    )
    logger.info(
        "train[0]: %s, \n test[0]: %s",
        train_ds.tensors[0],
        test_ds.tensors[0],
    )
    # TODO: build the model and train.


if __name__ == "__main__":
    # Hydra supplies `config`.
    main()  # pylint: disable=no-value-for-parameter
