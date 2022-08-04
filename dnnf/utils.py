"""
"""
import logging
import random
import sys
from typing import Optional

import numpy as np


def set_random_seed(seed: Optional[int]) -> None:
    random.seed(seed)
    np.random.seed(seed)


def initialize_logging(
    name: str, verbose: bool = False, quiet: bool = False, debug: bool = False
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.propagate = False

    if debug + verbose + quiet > 1:
        raise ValueError(
            "At most one log level [verbose, quiet, debug] may be selected."
        )
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    elif quiet:
        level = logging.ERROR
    else:
        level = logging.WARNING
    logger.setLevel(level)

    formatter = logging.Formatter("%(levelname)-8s %(asctime)s (%(name)s) %(message)s")

    console_handler = logging.StreamHandler(stream=sys.stderr)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


__all__ = ["initialize_logging", "set_random_seed"]
