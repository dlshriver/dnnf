#!/usr/bin/env python -O
import argparse
import logging
import psutil
import shlex
import signal
import subprocess as sp
import sys
import tempfile
import threading
import time

from typing import Optional, Sequence, Union


def memory_t(value: Union[int, str]) -> int:
    if isinstance(value, int):
        return value
    elif value.lower().endswith("g"):
        return int(value[:-1]) * 1_000_000_000
    elif value.lower().endswith("m"):
        return int(value[:-1]) * 1_000_000
    elif value.lower().endswith("k"):
        return int(value[:-1]) * 1000
    else:
        return int(value)


def memory_repr(value: int) -> str:
    if value > 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}G"
    elif value > 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    elif value > 1_000:
        return f"{value / 1_000:.2f}K"
    else:
        return f"{value}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="resmonitor - monitor resource usage")

    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument("--debug", action="store_true", help=argparse.SUPPRESS)
    verbosity_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="show messages with finer-grained information",
    )
    verbosity_group.add_argument(
        "-q", "--quiet", action="store_true", help="suppress non-essential messages"
    )

    parser.add_argument(
        "-T",
        "--time",
        dest="timeout",
        type=int,
        help="The max running time in seconds.",
    )
    parser.add_argument(
        "-M",
        "--memory",
        dest="max_memory",
        type=memory_t,
        help="The max allowed memory in bytes.",
    )

    parser.add_argument(
        "prog",
        type=str,
        nargs=argparse.REMAINDER,
        help="The program to run and monitor",
    )

    return parser.parse_args()


def get_memory_usage() -> int:
    try:
        p = psutil.Process()
        children = p.children(recursive=True)
        memory = 0
        for child in children:
            memory += child.memory_info().rss
    except psutil.NoSuchProcess:
        memory = 0
    return memory


def terminate(signum=None, frame=None):
    logger = logging.getLogger("resmonitor")
    if signum is not None:
        logger.warning("%s signal received, shutting down", signal.strsignal(signum))
    p = psutil.Process()
    if signum == 2:
        for child in p.children(recursive=True):
            child.send_signal(2)
        time.sleep(0.5)
    for child in p.children(recursive=True):
        child.terminate()
    time.sleep(0.5)
    for child in p.children(recursive=True):
        child.kill()


def monitor(
    shutdown_event: threading.Event,
    max_memory: Optional[int] = None,
    log_period: Optional[int] = None,
):
    logger = logging.getLogger("resmonitor")
    start_t = time.time()
    last_log_t = float("-inf")
    while not shutdown_event.is_set():
        time.sleep(0.001)
        now_t = time.time()
        duration_t = now_t - start_t
        mem_usage = get_memory_usage()
        if log_period is not None and now_t - last_log_t > log_period:
            logger.info(
                "Duration: %.3fs, MemUsage: %s", duration_t, memory_repr(mem_usage)
            )
            last_log_t = now_t
        if max_memory is not None and mem_usage >= max_memory:
            logger.info(
                "Duration: %.3fs, MemUsage: %s", duration_t, memory_repr(mem_usage)
            )
            logger.error(
                "Out of Memory: %s > %s",
                memory_repr(mem_usage),
                memory_repr(max_memory),
            )
            terminate()
            break


def dispatch(
    args: Sequence[str],
    max_memory: Optional[int] = None,
    timeout: Optional[int] = None,
    log_period: Optional[int] = None,
) -> int:
    logger = logging.getLogger("resmonitor")
    shutdown_event = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor, args=(shutdown_event, max_memory, log_period)
    )
    start_t = time.time()
    monitor_thread.start()
    return_code = 0
    try:
        proc = sp.run([shlex.quote(arg) for arg in args], timeout=timeout)
        logger.info("Process finished successfully.")
        return_code = proc.returncode
        if return_code < 0:
            return_code += 128
    except sp.TimeoutExpired:
        logger.error("Timeout: %.3f > %.3f", time.time() - start_t, timeout)
        return_code = 1
    shutdown_event.set()
    return return_code


def main(
    prog: Sequence[str],
    max_memory: Optional[int] = None,
    timeout: Optional[int] = None,
    debug=False,
    verbose=False,
    quiet=False,
) -> int:
    signal.signal(signal.SIGINT, terminate)
    signal.signal(signal.SIGTERM, terminate)

    logger = logging.getLogger("resmonitor")

    if debug:
        logger.setLevel(logging.DEBUG)
        log_period = 1  # seconds
    elif verbose:
        logger.setLevel(logging.INFO)
        log_period = 2  # seconds
    elif quiet:
        logger.setLevel(logging.ERROR)
        log_period = None
    else:
        logger.setLevel(logging.INFO)
        log_period = 5  # seconds

    formatter = logging.Formatter(f"%(levelname)-8s %(asctime)s (%(name)s) %(message)s")

    console_handler = logging.StreamHandler(stream=sys.stderr)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    with tempfile.TemporaryDirectory():
        result = dispatch(
            prog, max_memory=max_memory, timeout=timeout, log_period=log_period
        )
    return result


if __name__ == "__main__":
    exit(main(**vars(_parse_args())))
