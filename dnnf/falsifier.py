import asyncio
import logging
import multiprocessing as mp
import numpy as np
import time

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dnnv.properties import Expression
from functools import partial
from typing import Any, Dict, List, Type, Union

from .extractor import PropertyExtractor, HalfspacePolytope, HyperRectangle
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
    extractor = PropertyExtractor(HyperRectangle, HalfspacePolytope)
    if n_proc == 1:
        executor: Union[
            Type[ThreadPoolExecutor], Type[ProcessPoolExecutor]
        ] = ThreadPoolExecutor
        executor_params = {}
    else:
        executor = ProcessPoolExecutor
        executor_params = {
            "mp_context": mp.get_context("spawn"),
            "initializer": _init_logging,
        }
    pool = executor(max_workers=n_proc, **executor_params)
    tasks = []
    backend_parameters = kwargs.pop("parameters")
    for backend_method in backend:
        method_name, *variant = backend_method.split(".", maxsplit=1)
        if method_name not in globals():
            raise RuntimeError(f"Unknown falsification method: {method_name}")
        if variant:
            kwargs["variant"] = variant[0]
        logger.info("Using %s backend.", backend_method)
        method = globals()[method_name]
        parameters = backend_parameters[backend_method]
        method_n_starts = parameters.pop("n_starts", n_starts)
        for i, prop in enumerate(extractor.extract_from(~phi)):
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
                model.prop.networks, model.prop.op_graph(counter_example)
            ):
                logger.debug("%s -> %s", network, result)
            return counter_example
        await asyncio.sleep(0)  # yield to other tasks
        start_i += 1
        if (start_i) % kwargs.get("restart_log_freq", 10) == 0:
            logger.info("RESTART(%s): %d", _TASK_ID, start_i)


def pgd(model: FalsificationModel, n_steps=50, **kwargs):
    logger = logging.getLogger(__name__)

    if kwargs.get("cuda", False):
        model.model.to("cuda")
    x = model.sample()
    for step_i in range(n_steps):
        x = model.project_input(x)
        x.requires_grad = True
        y = model(x)
        if any(y[0, 0] <= y[0, i] for i in range(1, y.shape[1])):
            counter_example = x.cpu().detach().numpy()
            if model.validate(counter_example):
                logger.info("FOUND COUNTEREXAMPLE")
                logger.info("FALSIFIED at step %d", step_i)
                return counter_example
        x = model.step(x, y)
        if x is None:
            break


def cleverhans(
    model: FalsificationModel,
    variant: str = "ProjectedGradientDescent",
    parameters: Dict[str, Any] = None,
    **kwargs,
):
    import tensorflow.compat.v1 as tf
    import cleverhans.attacks as cleverhans_attacks
    from cleverhans.model import CallableModelWrapper

    logger = logging.getLogger(__name__)

    input_shape = tuple(model.input_lower_bound.shape)
    lb = model.input_lower_bound.cpu().float().numpy()
    ub = model.input_upper_bound.cpu().float().numpy()
    ranges = ub - lb

    def normalize(x):
        x_norm = x * ranges + lb
        assert x_norm.shape == x.shape
        return x_norm

    cleverhans_x = np.zeros(input_shape, dtype=np.float32) + 0.5
    if model.validate(normalize(cleverhans_x)):
        logger.info("FOUND COUNTEREXAMPLE immediately")
        return normalize(cleverhans_x)
    g = tf.Graph()
    sess = tf.Session(graph=g)
    with g.as_default():
        if parameters is None:
            parameters = {}
        if "clip_min" not in parameters:
            parameters["clip_min"] = 0
        if "clip_max" not in parameters:
            parameters["clip_max"] = 1
        for key, value in parameters.items():
            if isinstance(value, np.ndarray):
                parameters[key] = tf.convert_to_tensor(value)
        model_tf = model.as_tf()

        def callable_model(x):
            return model_tf(normalize(x))

        attack_method = getattr(cleverhans_attacks, variant)
        attack = attack_method(
            (CallableModelWrapper(callable_model, "logits")), sess=sess
        )
        x_placeholder = tf.placeholder(tf.float32, shape=input_shape)
        adv_x = attack.generate(x_placeholder, **parameters)
        adv_x_npy_ = sess.run(adv_x, feed_dict={x_placeholder: cleverhans_x})
        adv_x_npy = normalize(adv_x_npy_)
        if model.validate(adv_x_npy):
            logger.info("FOUND COUNTEREXAMPLE")
            return adv_x_npy


def foolbox(
    model: FalsificationModel,
    variant: str = "LinfPGD",
    parameters: Dict[str, Any] = None,
    **kwargs,
):
    import foolbox as fb
    import torch

    logger = logging.getLogger(__name__)

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

    pytorch_model = torch.nn.Sequential(normalizer, model.model).eval().to(device)
    finput = torch.zeros_like(lb) + 0.5
    flabel = torch.zeros(1, dtype=np.long, device=device)
    fmodel = fb.PyTorchModel(
        model.model.eval().to(device), bounds=(0, 1), device=device
    )
    epsilons = [0.5]

    if parameters is None:
        parameters = {}
    attack = getattr(fb.attacks, variant)(**parameters)
    _, advs, success = attack(fmodel, finput, flabel, epsilons=epsilons)

    for succ, adv, epsilon in zip(success, advs, epsilons):
        if not succ:
            continue
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
    import torch

    logger = logging.getLogger(__name__)

    shape, dtype = model.input_details[0]
    lb = model.input_constraint.lower_bound.reshape(shape).astype(dtype)
    ub = model.input_constraint.upper_bound.reshape(shape).astype(dtype)

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
            counter_example = np.load(f"{tmpdir}/cex.npy")[None].astype(dtype)
            if model.validate(counter_example):
                logger.info("FOUND COUNTEREXAMPLE")
                return counter_example
