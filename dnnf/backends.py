import numpy as np

from functools import partial

try:
    from cleverhans.torch.attacks import (
        carlini_wagner_l2,
        fast_gradient_method,
        hop_skip_jump_attack,
        projected_gradient_descent,
        spsa,
    )

    eps = 1
    clip_min = 0
    clip_max = 1
    pgd = partial(
        projected_gradient_descent.projected_gradient_descent,
        eps=eps,
        eps_iter=0.01,
        nb_iter=100,
        norm=np.inf,
        clip_min=clip_min,
        clip_max=clip_max,
    )
    basic_iterative_method = partial(pgd, rand_init=False)
    cw = partial(
        carlini_wagner_l2.carlini_wagner_l2,
        n_classes=2,
        clip_min=clip_min,
        clip_max=clip_max,
    )
    fgsm = partial(
        fast_gradient_method.fast_gradient_method,
        eps=eps,
        norm=np.inf,
        clip_min=clip_min,
        clip_max=clip_max,
    )
    hop_skip_jump = partial(
        hop_skip_jump_attack.hop_skip_jump_attack,
        norm=np.inf,
        clip_min=clip_min,
        clip_max=clip_max,
    )
    spsa_func = partial(
        spsa.spsa, eps=eps, nb_iter=100, clip_min=clip_min, clip_max=clip_max
    )

    cleverhans_backend = {
        "basic_iterative_method": basic_iterative_method,
        "BasicIterativeMethod": basic_iterative_method,
        "bim": basic_iterative_method,
        "BIM": basic_iterative_method,
        "carlini_wagner_l2": cw,
        "fast_gradient_method": fgsm,
        "FastGradientMethod": fgsm,
        "fgsm": fgsm,
        "FGSM": fgsm,
        "hop_skip_jump_attack": hop_skip_jump,
        "HopSkipJumpAttack": hop_skip_jump,
        "projected_gradient_descent": pgd,
        "ProjectedGradientDescent": pgd,
        "pgd": pgd,
        "PGD": pgd,
        "spsa": spsa_func,
    }
except ImportError:
    cleverhans_backend = {}

try:
    import foolbox as foolbox_backend
except ImportError:
    foolbox_backend = None

__all__ = ["cleverhans_backend", "foolbox_backend"]
