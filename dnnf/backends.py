import numpy as np
import os

from functools import partial

# check for cleverhans
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
    fgm = partial(
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
        "fast_gradient_method": fgm,
        "FastGradientMethod": fgm,
        "fgm": fgm,
        "FGM": fgm,
        "hop_skip_jump_attack": hop_skip_jump,
        "HopSkipJumpAttack": hop_skip_jump,
        "projected_gradient_descent": pgd,
        "ProjectedGradientDescent": pgd,
        "pgd": pgd,
        "PGD": pgd,
        "spsa": spsa_func,
    }
    cleverhans_backend_choices = (
        ("basic_iterative_method", "BasicIterativeMethod", "BIM"),
        ("carlini_wagner_l2",),
        ("fast_gradient_method", "FastGradientMethod", "FGM"),
        ("hop_skip_jump_attack", "HopSkipJumpAttack"),
        ("projected_gradient_descent", "ProjectedGradientDescent", "PGD"),
        ("spsa",),
    )
except ImportError:
    cleverhans_backend = {}
    cleverhans_backend_choices = ()

# check for foolbox
try:
    import foolbox as foolbox_backend

    foolbox_attacks = {}
    for name in dir(foolbox_backend.attacks):
        value = getattr(foolbox_backend.attacks, name)
        if (
            isinstance(value, type)
            and issubclass(value, foolbox_backend.Attack)
            and name != "Attack"
        ):
            if value not in foolbox_attacks:
                foolbox_attacks[value] = []
            foolbox_attacks[value].append(name)

    foolbox_backend_choices = tuple(
        tuple(sorted(v, key=lambda x: (0, x) if x == k.__name__ else (1, x)))
        for k, v in foolbox_attacks.items()
    )
except ImportError:
    foolbox_backend = None
    foolbox_backend_choices = ()

# check for tensorfuzz
tensorfuzz_available = False
for path in os.environ["PATH"].split(os.pathsep):
    fpath = os.path.join(path, "tensorfuzz.sh")
    if os.path.isfile(fpath) and os.access(fpath, os.X_OK):
        tensorfuzz_available = True


def get_backend_choices(group_equivalent=False):
    choices = ["pgd"]
    if group_equivalent:
        choices += list(
            ", ".join(f"cleverhans.{c}" for c in b) for b in cleverhans_backend_choices
        )
        choices += list(
            ", ".join(f"foolbox.{c}" for c in b) for b in foolbox_backend_choices
        )
    else:
        choices += list(
            f"cleverhans.{c}" for b in cleverhans_backend_choices for c in b
        )
        choices += list(f"foolbox.{c}" for b in foolbox_backend_choices for c in b)
    if tensorfuzz_available:
        choices += ["tensorfuzz"]
    return choices


__all__ = ["cleverhans_backend", "foolbox_backend"]
