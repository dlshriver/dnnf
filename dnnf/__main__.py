"""
"""
import multiprocessing as mp
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


def main(
    property: Path,
    networks: Dict[str, Path],
    save_violation: Optional[Path] = None,
    extra_args: Optional[List[str]] = None,
    **kwargs,
):
    os.setpgrp()

    phi = parse_property(property, args=extra_args)
    print("Falsifying:", phi)
    for name, network in networks.items():
        dnn = parse_network(network)
        dnn = dnn.simplify()
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

    os.killpg(os.getpgrp(), 9)


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
