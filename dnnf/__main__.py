"""
"""
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from dnnv.nn import parse as parse_network
from dnnv.properties import parse as parse_property

from .cli import parse_args
from .falsifier import falsify
from .utils import initialize_logging, set_random_seed


def main(
    property_path: Path,
    networks: Dict[str, Path],
    prop_format: Optional[str] = None,
    save_violation: Optional[Path] = None,
    extra_args: Optional[List[str]] = None,
    **kwargs,
):
    logger = logging.getLogger(__name__)

    phi = parse_property(property_path, format=prop_format, args=extra_args)
    print("Falsifying:", phi)
    print()
    for name, network in networks.items():
        dnn = parse_network(network, net_format="onnx")
        print(f"Network {name}:")
        dnn.pprint()
        print()
        phi.concretize(**{name: dnn})

    if extra_args is not None and len(extra_args) > 0:
        logger.error("Unused arguments: %r", extra_args)
        sys.exit(1)

    start_t = time.time()
    try:
        result = falsify(phi, **kwargs)
    except KeyboardInterrupt:
        result = {"violation": None, "time": 0.0}
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
    initialize_logging(
        __package__, verbose=args.verbose, quiet=args.quiet, debug=args.debug
    )
    return main(**vars(args), extra_args=extra_args)


if __name__ == "__main__":
    __main__()
