"""
"""
import argparse
import logging
import numpy as np
import os
import random
import sys

from contextlib import contextmanager
from functools import partial
from typing import Optional


def set_random_seed(seed: Optional[int]) -> None:
    random.seed(seed)
    np.random.seed(seed)


@contextmanager
def suppress(level=logging.DEBUG, filter_level=logging.WARNING):
    if level >= filter_level:
        yield
        return
    with open(os.dup(sys.stdout.fileno()), "wb") as stdout_copy:
        with open(os.dup(sys.stderr.fileno()), "wb") as stderr_copy:
            sys.stdout.flush()
            sys.stderr.flush()
            with open(os.devnull, "wb") as devnull:
                os.dup2(devnull.fileno(), sys.stdout.fileno())
                os.dup2(devnull.fileno(), sys.stderr.fileno())
            try:
                yield
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
                os.dup2(stdout_copy.fileno(), sys.stdout.fileno())
                os.dup2(stderr_copy.fileno(), sys.stderr.fileno())


def initialize_logging(
    name: str, verbose: bool = False, quiet: bool = False, debug: bool = False
) -> logging.Logger:
    global suppress
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
    suppress = partial(suppress, filter_level=level)

    formatter = logging.Formatter(f"%(levelname)-8s %(asctime)s (%(name)s) %(message)s")

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

