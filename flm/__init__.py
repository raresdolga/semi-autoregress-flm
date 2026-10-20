"""Semi-autoregressive flow language models.

Import from the submodule that owns what you need -- this package deliberately
re-exports nothing:

    from flm import algo, trainer_base, utils
    from flm.datamodules.dataloaders import get_dataloaders
    from flm.models.dit import DIT

The package is not installed (`pyproject.toml` declares no build backend), so
every entry point is run as a module from the repo root, e.g.
`python -m flm.main`. Configs, scripts and tests live outside the package.
"""
