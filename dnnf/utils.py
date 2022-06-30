"""
"""
import logging
import onnx
import random
import sys
import torch
from typing import Optional

import numpy as np

ONNX_TO_TORCH_DTYPE = {
    onnx.TensorProto.DOUBLE: torch.float64,
    onnx.TensorProto.FLOAT16: torch.float16,
    onnx.TensorProto.FLOAT: torch.float32,
    onnx.TensorProto.INT8: torch.int8,
    onnx.TensorProto.INT16: torch.int16,
    onnx.TensorProto.INT32: torch.int32,
    onnx.TensorProto.INT64: torch.int64,
    onnx.TensorProto.UINT8: torch.uint8,
    onnx.TensorProto.BOOL: torch.bool,
    onnx.TensorProto.COMPLEX64: torch.complex64,
    onnx.TensorProto.COMPLEX128: torch.complex128,
}


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

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


__all__ = ["initialize_logging", "set_random_seed"]
