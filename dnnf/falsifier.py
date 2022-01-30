import asyncio
import logging
import multiprocessing as mp
import numpy as np
import torch
import time

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dnnv.properties import Expression
from functools import partial
from typing import Any, Dict, List, Type, Union

from .backends import *
from .reduction import HPolyReduction
from .model import FalsificationModel


def _init_logging():
    from .cli import parse_args
    from .utils import initialize_logging, set_random_seed

    args, extra_args = parse_args()
    set_random_seed(args.seed)
    initialize_logging(
        __package__, verbose=args.verbose, quiet=args.quiet, debug=args.debug
    )


def falsify(
    phi: Expression,
    backend: Union[str, List[str]] = "pgd",
    n_proc: int = 1,
    n_starts=-1,
    **kwargs,
):
    logger = logging.getLogger(__name__)
    if isinstance(backend, str):
        backend = [backend]

    counter_example = None
    reduction = HPolyReduction()
    executor_params = {}
    if n_proc == 1:
        executor: Union[
            Type[ThreadPoolExecutor], Type[ProcessPoolExecutor]
        ] = ThreadPoolExecutor
    else:
        executor = ProcessPoolExecutor
        executor_params = {
            "mp_context": mp.get_context("spawn"),
            "initializer": _init_logging,
        }
    pool = executor(max_workers=n_proc, **executor_params)  # type: ignore
    tasks = []
    backend_parameters = kwargs.pop("parameters") or {}
    for backend_method in backend:
        method_name, *variant = backend_method.split(".", maxsplit=1)
        if method_name not in globals():
            raise RuntimeError(f"Unknown falsification method: {method_name}")
        if variant:
            kwargs["variant"] = variant[0]
        logger.info("Using %s backend.", backend_method)
        method = globals()[method_name]
        parameters: Dict[str, Any] = backend_parameters.get(backend_method, {})
        method_n_starts = parameters.pop("n_starts", n_starts)
        for i, prop in enumerate(reduction.reduce_property(phi)):
            logger.debug(f"subproblem {backend_method}_{i}")
            tasks.append(
                falsify_model(
                    method,
                    FalsificationModel(prop),
                    parameters=parameters,
                    n_starts=method_n_starts,
                    executor=pool,
                    _TASK_ID=f"{backend_method}_{i}",
                    **kwargs,
                )
            )
    logger.info("Starting Falsifier")
    start_t = time.time()
    counter_example = asyncio.run(wait_for_first(tasks, **kwargs))
    end_t = time.time()
    logger.info(f"falsification time: {end_t - start_t:.4f}")

    return {"violation": counter_example, "time": end_t - start_t}


async def wait_for_first(tasks, sequential=False, **kwargs):
    if sequential:
        for task in tasks:
            result = await task
            if result is not None:
                return result
    else:
        for task in asyncio.as_completed(tasks):
            result = await task
            if result is not None:
                return result


async def falsify_model(
    method, model: FalsificationModel, executor=None, n_starts=-1, **kwargs
):
    logger = logging.getLogger(__name__)
    _TASK_ID = kwargs.get("_TASK_ID", "")

    start_i = 0
    loop = asyncio.get_event_loop()
    while n_starts < 0 or start_i < n_starts:
        counter_example = await loop.run_in_executor(
            executor, partial(method, model, **kwargs)
        )
        if counter_example is not None:
            logger.info(f"FALSIFIED ({_TASK_ID}) at restart: {start_i}")
            for network, result in zip(
                model.prop.output_vars, model.prop.op_graph(counter_example)
            ):
                logger.debug("%s -> %s", network, result)
            return counter_example
        await asyncio.sleep(0)  # yield to other tasks
        start_i += 1
        if (start_i) % kwargs.get("restart_log_freq", 10) == 0:
            logger.info("RESTART(%s): %d", _TASK_ID, start_i)


def pgd(model: FalsificationModel, n_steps=100, **kwargs):
    logger = logging.getLogger(__name__)

    if kwargs.get("cuda", False):
        model.model.to("cuda")
    x = model.sample()
    for step_i in range(n_steps):
        x.requires_grad = True
        y = model(x)
        flat_y = y.flatten()
        if any(flat_y[0] <= flat_y[i] for i in range(1, len(flat_y))):
            counter_example = x.cpu().detach().numpy()
            if model.validate(counter_example):
                logger.info("FOUND COUNTEREXAMPLE")
                logger.info("FALSIFIED at step %d", step_i)
                return counter_example
        x = model.step(x, y)
        if x is None:
            break
        x = model.project_input(x)


def cleverhans(
    model: FalsificationModel,
    variant: str = "ProjectedGradientDescent",
    parameters: Dict[str, Any] = None,
    **kwargs,
):
    logger = logging.getLogger(__name__)
    if not cleverhans_backend:
        logger.critical("CleverHans is not installed!")
        raise ImportError("CleverHans is not installed!")

    if parameters is None:
        parameters = {}
    for key, value in parameters.items():
        if isinstance(value, np.ndarray):
            parameters[key] = torch.from_numpy(value)

    lb = model.input_lower_bound
    ranges = model.input_upper_bound - lb

    class Normalize(torch.nn.Module):
        def forward(self, x):
            return x * ranges + lb

    normalizer = Normalize()

    initial_x = torch.full_like(lb, 0.5)
    _x = normalizer(initial_x).detach().numpy()
    if model.validate(_x):
        logger.info("FOUND COUNTEREXAMPLE immediately")
        return _x

    pytorch_model = torch.nn.Sequential(normalizer, model.model).eval()

    device = torch.device("cpu")
    if kwargs.get("cuda", False):
        device = torch.device("cuda")
    pytorch_model.to(device)

    attack = cleverhans_backend[variant]
    result = attack(pytorch_model, initial_x.to(device), **parameters)
    counter_example = normalizer(result).detach().numpy()
    if model.validate(counter_example):
        logger.info("FOUND COUNTEREXAMPLE")
        return counter_example


def foolbox(
    model: FalsificationModel,
    variant: str = "LinfPGD",
    parameters: Dict[str, Any] = None,
    **kwargs,
):
    logger = logging.getLogger(__name__)
    if foolbox_backend is None:
        logger.critical("Foolbox is not installed!")
        raise ImportError("Foolbox is not installed!")

    device = torch.device("cpu")
    if kwargs.get("cuda", False):
        device = torch.device("cuda")

    lb = model.input_lower_bound
    ub = model.input_upper_bound
    ranges = ub - lb

    class Normalize(torch.nn.Module):
        def forward(self, x):
            return x * ranges + lb

    normalizer = Normalize()

    initial_input = torch.full_like(lb, 0.5)
    _x = normalizer(initial_input).detach().numpy()
    if model.validate(_x):
        logger.info("FOUND COUNTEREXAMPLE immediately")
        return _x

    pytorch_model = torch.nn.Sequential(normalizer, model.model).eval().to(device)
    finput = initial_input.to(device)
    flabel = torch.zeros(1, dtype=np.long, device=device)
    fmodel = foolbox_backend.PyTorchModel(pytorch_model, bounds=(0, 1), device=device)
    epsilons = [1.0]

    if parameters is None:
        parameters = {}
    attack = getattr(foolbox_backend.attacks, variant)(**parameters)
    _, advs, success = attack(fmodel, finput, flabel, epsilons=epsilons)

    for _, adv, _ in zip(success, advs, epsilons):
        counter_example = normalizer(adv).cpu().detach().numpy()
        if model.validate(counter_example):
            logger.info("FOUND COUNTEREXAMPLE")
            return counter_example
        logger.debug("Counter example could not be validated")


def tensorfuzz(model: FalsificationModel, n_steps=100, **kwargs):
    import select
    import shlex
    import subprocess as sp
    import tempfile

    logger = logging.getLogger(__name__)

    lb = model.input_lower_bound
    ub = model.input_upper_bound

    with tempfile.TemporaryDirectory() as tmpdir:
        logger.debug("Using temporary directory: %s", tmpdir)

        model_filename = f"{tmpdir}/model.onnx"
        lb_filename = f"{tmpdir}/lb.npy"
        ub_filename = f"{tmpdir}/ub.npy"

        dummy_x = torch.from_numpy(lb)
        torch.onnx.export(
            model.model,
            dummy_x,
            model_filename,
            input_names=["input"],
            dynamic_axes={"input": [0]},
        )
        np.save(lb_filename, lb[0])
        np.save(ub_filename, ub[0])

        cmd = f"tensorfuzz.sh --model={model_filename} --lb={lb_filename} --ub={ub_filename} --label=0"
        logger.debug("Running: %s", cmd)

        proc = sp.Popen(
            shlex.split(cmd), stdout=sp.PIPE, stderr=sp.PIPE, encoding="utf8"
        )
        assert proc.stderr is not None
        assert proc.stdout is not None
        stdout_lines: List[str] = []
        stderr_lines: List[str] = []
        while proc.poll() is None:
            for (name, stream, lines) in [
                ("STDOUT", proc.stdout, stdout_lines),
                ("STDERR", proc.stderr, stderr_lines),
            ]:
                ready, _, _ = select.select([stream], [], [], 0)
                if not ready:
                    continue
                line = stream.readline()
                if line == "":
                    continue
                lines.append(line)
                logger.debug(f"{{TENSORFUZZ ({name})}}: {line.strip()}")
        for line in proc.stdout.readlines():
            logger.debug(f"{{TENSORFUZZ (STDOUT)}}: {line.strip()}")
        stdout_lines.extend(stdout_lines)
        for line in proc.stderr.readlines():
            logger.debug(f"{{TENSORFUZZ (STDERR)}}: {line.strip()}")
        stderr_lines.extend(stderr_lines)
        if "Fuzzing succeeded" in "\n".join(stderr_lines):
            counter_example = np.load(f"{tmpdir}/cex.npy")[None].astype(
                model.input_details[0].dtype
            )
            if model.validate(counter_example):
                logger.info("FOUND COUNTEREXAMPLE")
                return counter_example
