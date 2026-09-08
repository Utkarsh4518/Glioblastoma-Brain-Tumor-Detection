"""
Hydra entry point for BraTS training.

Usage:
    python train.py task=segmentation
    python train.py data=h5 task=segmentation
    python train.py training.device=cpu data.num_workers=0 training.max_epochs=1
"""

import argparse

import hydra
from omegaconf import DictConfig

from training.run import run_training

# Hydra 1.3.x's --shell-completion help text is a lazily-computed object
# (LazyCompletionHelp) that Python 3.14's stricter argparse._check_help
# can't validate (it does `'%' not in help_string` without str()-ing it
# first), raising ValueError before any config is even parsed. Hydra has no
# release addressing this; skip the validation rather than the help text.
_original_check_help = argparse.ArgumentParser._check_help


def _lenient_check_help(self, action):
    try:
        _original_check_help(self, action)
    except (TypeError, ValueError):
        pass


argparse.ArgumentParser._check_help = _lenient_check_help


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_training(cfg)


if __name__ == "__main__":
    main()
