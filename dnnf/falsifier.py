import asyncio
import logging
import multiprocessing as mp
import shlex
import subprocess as sp
import tempfile
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from typing import Any, Dict, List, Type, Union

import numpy as np
import torch
from dnnv.properties import Expression

from .backends import cleverhans_backend, foolbox_backend
from .cli import parse_args
from .model import FalsificationModel
from .reduction import HPolyReduction
from .utils import initialize_logging, set_random_seed


def _initializer():
    args, _ = parse_args()
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
    executor_params: Dict[str, Any] = {}
    if n_proc == 1:
        executor: Union[
            Type[ThreadPoolExecutor], Type[ProcessPoolExecutor]
        ] = ThreadPoolExecutor
    else:
        executor = ProcessPoolExecutor
        executor_params = {
            "mp_context": mp.get_context("spawn"),
            "initializer": _initializer,
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
        logger.debug("Reducing expression.")
        for i, prop in enumerate(reduction.reduce_property(phi)):
            logger.debug("subproblem %s_%d", backend_method, i)
            tasks.append(
                falsify_model(
                    method,
                    FalsificationModel(prop),
                    parameters=parameters,
                    n_starts=method_n_starts,
                    executor=pool,
                    _dnnf_task_id=f"{backend_method}_{i}",
                    **kwargs,
                )
            )
        logger.debug("Finished reduction")
    logger.info("Starting Falsifier")
    start_t = time.time()
    try:
        counter_example = asyncio.run(wait_for_first(tasks, **kwargs))
    except KeyboardInterrupt:
        counter_example = None
    end_t = time.time()
    logger.info("falsification time: %.4f", end_t - start_t)

    return {"violation": counter_example, "time": end_t - start_t}


async def wait_for_first(tasks, sequential=False, **_):
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
    _dnnf_task_id = kwargs.get("_dnnf_task_id", "")

    start_i = 0
    loop = asyncio.get_event_loop()
    while n_starts < 0 or start_i < n_starts:
        counter_example = await loop.run_in_executor(
            executor, partial(method, model, **kwargs)
        )
        if counter_example is not None:
            logger.info("FALSIFIED (%s) at restart: %d", _dnnf_task_id, start_i)
            for network, result in zip(
                model.prop.input_output_info["output_names"],
                model.prop.op_graph(counter_example),
            ):
                logger.debug("%s -> %s", network, result)
            return counter_example
        await asyncio.sleep(0)  # yield to other tasks
        start_i += 1
        if (start_i) % kwargs.get("restart_log_freq", 10) == 0:
            logger.info("RESTART(%s): %d", _dnnf_task_id, start_i)


def pgd(
    model: FalsificationModel,
    parameters: Dict[str, Any] = {},
    **kwargs,
):
    logger = logging.getLogger(__name__)

    if kwargs.get("cuda", False):
        model.pytorch_model.to("cuda")
    x = model.sample()
    n_steps = parameters.get("n_steps", 100)
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
        _x = model.step(x, y)
        if _x is None:
            break
        x = model.project_input(_x)
    return None


def newton(
    model: FalsificationModel,
    parameters: Dict[str, Any] = {},
    **kwargs,
):
    logger = logging.getLogger(__name__)
    if kwargs.get("cuda", False):
        model.pytorch_model.to("cuda")
    lb = model.input_lower_bound
    ub = model.input_upper_bound
    lb = torch.nextafter(lb, torch.full_like(lb, torch.inf))
    ub = torch.nextafter(ub, torch.full_like(ub, -torch.inf))
    dnn = model.pytorch_model
    x = torch.rand_like(lb) * (ub - lb) + lb
    n_steps = parameters.get("n_steps", 50)
    for step_i in range(n_steps):
        x.requires_grad = True
        output: torch.Tensor = dnn(x)
        y = output.flatten()[0]
        y.backward()
        if y <= 1e-16:
            counter_example = x.cpu().detach().numpy()
            if model.validate(counter_example):
                logger.info("FOUND COUNTEREXAMPLE")
                logger.info("FALSIFIED at step %d", step_i)
                return counter_example
        x_grad = x.grad
        assert x_grad is not None
        assert isinstance(x_grad, torch.Tensor)
        _x = x.detach()
        x = x - y / x_grad
        if torch.any(torch.isnan(x)):
            return None
        lb_violations = x < lb
        ub_violations = x > ub
        x[lb_violations] = lb[lb_violations]
        x[ub_violations] = ub[ub_violations]
        if torch.allclose(x, _x):
            return None
        x = x.detach()
    return None


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

    pytorch_model = torch.nn.Sequential(normalizer, model.pytorch_model).eval()

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

    pytorch_model = (
        torch.nn.Sequential(normalizer, model.pytorch_model).eval().to(device)
    )
    finput = initial_input.to(device)
    flabel = torch.zeros(1, dtype=torch.int64, device=device)
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


def tensorfuzz(model: FalsificationModel, **_):
    logger = logging.getLogger(__name__)

    lb = model.input_lower_bound
    ub = model.input_upper_bound

    with tempfile.TemporaryDirectory() as tmpdir:
        logger.debug("Using temporary directory: %s", tmpdir)

        model_filename = f"{tmpdir}/model.onnx"
        lb_filename = f"{tmpdir}/lb.npy"
        ub_filename = f"{tmpdir}/ub.npy"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(
                model.pytorch_model,
                lb,
                model_filename,
                input_names=["input"],
                dynamic_axes={"input": {0: "batch"}},
            )
        np.save(lb_filename, lb[0].numpy())
        np.save(ub_filename, ub[0].numpy())

        cmd = (
            f"tensorfuzz.sh --model={model_filename}"
            f" --lb={lb_filename} --ub={ub_filename} --label=0"
        )
        logger.debug("Running: %s", cmd)

        proc = sp.run(
            shlex.split(cmd),
            stdout=sp.PIPE,
            stderr=sp.STDOUT,
            encoding="utf8",
            check=True,
        )
        for line in proc.stdout.split("\n"):
            logger.debug("[TENSORFUZZ]: %s", line.strip())

        if "Fuzzing succeeded" in "\n".join(proc.stdout):
            counter_example = np.load(f"{tmpdir}/cex.npy")[None].astype(
                model.input_details[0].dtype
            )
            if model.validate(counter_example):
                logger.info("FOUND COUNTEREXAMPLE")
                return counter_example
    return None


__all__ = ["falsify"]
