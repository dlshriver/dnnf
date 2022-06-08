from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .pytorch import convert
from .reduction import HPolyProperty


class FalsificationModel:
    def __init__(self, prop: HPolyProperty):
        self.prop = prop
        self.op_graph = prop.suffixed_op_graph()
        self.input_details = self.op_graph.input_details
        self.input_shape = tuple(
            int(d) if d > 0 else 1 for d in self.input_details[0].shape
        )
        self.input_dtype = self.input_details[0].dtype
        self.input_torch_dtype = torch.from_numpy(
            np.ones((1,), dtype=self.input_dtype)
        ).dtype
        self.pytorch_model = self.as_pytorch()

    def __call__(self, *args, **kwargs):
        return self.pytorch_model(*args, **kwargs)

    def __reduce__(self):
        return FalsificationModel, (self.prop,)

    @property
    def input_lower_bound(self):
        lower_bounds = self.prop.input_lower_bounds
        assert len(lower_bounds) == 1
        lower_bound = lower_bounds[0]
        return torch.from_numpy(lower_bound.astype(self.input_dtype)).to(
            self.pytorch_model.device
        )

    @property
    def input_upper_bound(self):
        upper_bounds = self.prop.input_upper_bounds
        assert len(upper_bounds) == 1
        upper_bound = upper_bounds[0]
        return torch.from_numpy(upper_bound.astype(self.input_dtype)).to(
            self.pytorch_model.device
        )

    def as_pytorch(self):
        return convert(self.op_graph).eval()

    def as_tf(self):
        return self.op_graph.as_tf()

    def loss(self, y: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            y.reshape((1, -1)), torch.Tensor([0]).long().to(y.device)
        ) - F.cross_entropy(y.reshape((1, -1)), torch.Tensor([1]).long().to(y.device))

    def project_input(self, x: torch.Tensor) -> torch.Tensor:
        y = x.detach()
        lb = self.input_lower_bound
        ub = self.input_upper_bound
        lb_violations = y < lb
        ub_violations = y > ub
        _lb = lb[lb_violations]
        _ub = ub[ub_violations]
        y[lb_violations] = torch.nextafter(_lb, torch.full_like(_lb, torch.inf))
        y[ub_violations] = torch.nextafter(_ub, torch.full_like(_ub, -torch.inf))
        return y.detach()

    def sample(self) -> torch.Tensor:
        x = (
            torch.rand(
                self.input_shape,
                device=self.pytorch_model.device,
                dtype=self.input_torch_dtype,
            )
            * (self.input_upper_bound - self.input_lower_bound)
            + self.input_lower_bound
        )
        return x.detach()

    def step(self, x: torch.Tensor, y: torch.Tensor) -> Optional[torch.Tensor]:
        loss = self.loss(y)
        loss.backward()
        gradients = x.grad
        neg_grads = gradients < 0
        pos_grads = gradients > 0
        lb = self.input_lower_bound
        ub = self.input_upper_bound
        gradients[(x == lb) & neg_grads] = 0
        gradients[(x == ub) & pos_grads] = 0
        if gradients.abs().max().item() < 1e-12:
            return None
        lb = self.input_lower_bound
        ub = self.input_upper_bound
        epsilon = (ub - lb) / 2
        if len(gradients.shape) == 1:
            x = x + F.normalize(gradients.reshape(1, -1)).flatten() * epsilon
        else:
            x = x + F.normalize(gradients) * epsilon
        return x.detach()

    def validate(self, x: np.ndarray):
        return self.prop.validate_counter_example(x)


__all__ = ["FalsificationModel"]
