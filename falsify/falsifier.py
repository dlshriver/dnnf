import asyncio
import multiprocessing as mp
import numpy as np
import time

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dnnv.properties import Expression
from functools import partial
from typing import Dict, Union

from .extractor import PropertyExtractor, HalfspacePolytope, HyperRectangle
from .model import FalsificationModel


def falsify(phi: Expression, n_proc: int = 1, **kwargs):
    counter_example = None
    extractor = PropertyExtractor(HyperRectangle, HalfspacePolytope)
    executor = ProcessPoolExecutor
    executor_params = {"mp_context": mp.get_context("spawn")}
    # executor = ThreadPoolExecutor
    # executor_params = {}
    with executor(max_workers=n_proc, **executor_params) as pool:
        tasks = [
            falsify_model(pgd, FalsificationModel(prop), executor=pool, **kwargs)
            for i, prop in enumerate(extractor.extract_from(~phi))
        ]
        counter_example = asyncio.run(wait_for_first(tasks, **kwargs))
    return counter_example


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
    start_i = 0
    loop = asyncio.get_event_loop()
    while n_starts < 0 or start_i < n_starts:
        counter_example = await loop.run_in_executor(
            executor, partial(method, model, **kwargs)
        )
        if counter_example is not None:
            print(f"FALSIFIED restart: {start_i}")
            print(" ", model.prop.networks)
            print(f"  {model.prop.op_graph(counter_example)}")
            lb = model.input_lower_bound.cpu().numpy()
            ub = model.input_upper_bound.cpu().numpy()
            center = (ub + lb) / 2
            print(" ", abs(counter_example - center).max())
            return counter_example
        await asyncio.sleep(0)  # yield to other tasks
        start_i += 1
        if (start_i) % 10 == 0:
            print("restart:", start_i)


def pgd(model: FalsificationModel, n_steps=100, **kwargs):
    # model.model.to("cuda")
    x = model.sample()
    for step_i in range(n_steps):
        x = model.project_input(x)
        x.requires_grad = True
        y = model(x)
        if y.argmax() != 0:
            counter_example = x.cpu().detach().numpy()
            if model.validate(counter_example):
                print("FOUND", step_i)
                return counter_example
        x = model.step(x, y)
        # x = x.detach()
        # x.requires_grad = True
        # x = model.step(x, model(x, relu_approx=True))
        if x is None:
            break


def cleverhans(model: FalsificationModel, n_steps=100, **kwargs):
    import tensorflow as tf
    from cleverhans.attacks import ProjectedGradientDescent
    from cleverhans.model import CallableModelWrapper

    lb = model.input_lower_bound.cpu().numpy().astype(np.float32)
    ub = model.input_upper_bound.cpu().numpy().astype(np.float32)
    epsilon = (ub - lb) / 2
    cleverhans_x = (ub + lb) / 2
    if model.validate(cleverhans_x):
        return cleverhans_x
    params = {
        "eps": tf.convert_to_tensor(epsilon),
        "eps_iter": tf.convert_to_tensor(epsilon / 2),
        "clip_min": tf.convert_to_tensor(lb),
        "clip_max": tf.convert_to_tensor(ub),
    }
    pgd = ProjectedGradientDescent(CallableModelWrapper(model.as_tf(), "logits"))
    x_placeholder = tf.placeholder(tf.float32, shape=tuple(lb.shape))
    adv_x = pgd.generate(x_placeholder, **params)
    with tf.Session().as_default():
        adv_x_ = adv_x.eval(feed_dict={x_placeholder: cleverhans_x})
        if model.validate(adv_x_):
            return adv_x_


def foolbox(model: FalsificationModel, n_steps=100, **kwargs):
    import foolbox
    import torch

    # Currently there is no way to ensure that the search remains
    # in the desired input region unless lower and upper bounds are
    # the same for all input dimensions
    lb = model.input_lower_bound.cpu().numpy().astype(np.float32)
    ub = model.input_upper_bound.cpu().numpy().astype(np.float32)
    foolbox_x = (ub + lb) / 2
    y = model(model.sample())
    foolbox_model = foolbox.models.PyTorchModel(
        model.model.eval(),
        bounds=(lb.min().item(), ub.max().item()),
        num_classes=int(y.size(1)),
        device="cpu",
    )
    attack = foolbox.attacks.PGD(foolbox_model, distance=foolbox.distances.Linf)
    adversarials = attack(foolbox_x, np.zeros(1, dtype=np.long), unpack=False)
    for adv in adversarials:
        if adv.perturbed is None:
            continue
        counter_example = adv.perturbed[None, :]
        if model.validate(counter_example):
            return counter_example
        print("Adv Example")
        print("  reported label:", adv.adversarial_class)
        print("  measured label:", model(torch.from_numpy(counter_example)))
        print("  reported distance:", adv.distance)
        print("  measured distance:", np.abs(counter_example - foolbox_x).max())
