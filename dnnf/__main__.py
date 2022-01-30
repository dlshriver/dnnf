"""
"""
import numpy as np
import os
import time

from dnnv.nn import parse as parse_network
from dnnv.properties import parse as parse_property
from pathlib import Path
from typing import Dict, List, Optional

from .cli import parse_args
from .falsifier import falsify
from .utils import initialize_logging, set_random_seed

from dnnv.nn.graph import OperationGraph
from dnnv.nn.utils import TensorDetails

orig_input_details = OperationGraph.input_details


@property
def new_input_details(self):
    if self._input_details is None:
        _input_details = orig_input_details.fget(self)
        self._input_details = tuple(
            TensorDetails(tuple(i if i >= 0 else 1 for i in d.shape), d.dtype)
            for d in _input_details
        )
    return self._input_details


OperationGraph.input_details = new_input_details


def main(
    property: Path,
    networks: Dict[str, Path],
    prop_format: Optional[str] = None,
    save_violation: Optional[Path] = None,
    extra_args: Optional[List[str]] = None,
    **kwargs,
):
    os.setpgrp()

    phi = parse_property(property, format=prop_format, args=extra_args)
    print("Falsifying:", phi)
    for name, network in networks.items():
        dnn = parse_network(network, net_format="onnx")
        if kwargs["debug"]:
            print(f"Network {name}:")
            dnn.pprint()
            print()
        phi.concretize(**{name: dnn})
    print()

    start_t = time.time()
    result = falsify(phi, **kwargs)
    end_t = time.time()
    print("dnnf")
    if result["violation"] is not None:
        print("  result: sat")
        if save_violation is not None:
            np.save(save_violation, result["violation"])
    else:
        print("  result: unknown")
    falsification_time = result["time"]
    print(f"  falsification time: {falsification_time:.4f}")
    print(f"  total time: {end_t - start_t:.4f}", flush=True)


def __main__():
    args, extra_args = parse_args()
    set_random_seed(args.seed)
    logger = initialize_logging(
        __package__, verbose=args.verbose, quiet=args.quiet, debug=args.debug
    )
    main(**vars(args), extra_args=extra_args)
    if extra_args is not None and len(extra_args) > 0:
        logger.warning("Unused arguments: %r", extra_args)


if __name__ == "__main__":
    __main__()
