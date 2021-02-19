import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .extractor import Property, HyperRectangle


class FalsificationModel:
    def __init__(self, prop: Property):
        self.prop = prop
        self.input_constraint = prop.input_constraint
        self.output_constraint = prop.output_constraint
        self.op_graph = prop.as_operation_graph()
        self.input_details = self.op_graph.input_details
        self.input_shape = tuple(
            int(d) if d > 0 else 1 for d in self.input_details[0].shape
        )
        self.model = self.as_pytorch()
        if not isinstance(self.input_constraint, HyperRectangle):
            raise ValueError(
                "Only HyperRectangle input constraints are currently supported"
            )

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __reduce__(self):
        return FalsificationModel, (self.prop,)

    @property
    def input_lower_bound(self):
        return (
            torch.from_numpy(self.input_constraint.lower_bound)
            .reshape(self.input_shape)
            .float()
            .to(self.model.device)
        )

    @property
    def input_upper_bound(self):
        return (
            torch.from_numpy(self.input_constraint.upper_bound)
            .reshape(self.input_shape)
            .float()
            .to(self.model.device)
        )

    def as_pytorch(self):
        from .pytorch import convert

        return convert(self.op_graph.output_operations).eval()

    def as_tf(self):
        return self.op_graph.as_tf()

    def loss(self, y):
        # return -F.cross_entropy(y, torch.Tensor([1]).long().to(y.device))
        # return F.cross_entropy(y, torch.Tensor([0]).long().to(y.device))
        return F.cross_entropy(
            y, torch.Tensor([0]).long().to(y.device)
        ) - F.cross_entropy(y, torch.Tensor([1]).long().to(y.device))

    def project_input(self, x):
        y = x.detach()
        lb = self.input_lower_bound
        ub = self.input_upper_bound
        lb_violations = y < lb
        ub_violations = y > ub
        y[lb_violations] = lb[lb_violations]
        y[ub_violations] = ub[ub_violations]
        return y.detach()

    def sample(self):
        x = (
            torch.rand(self.input_shape, device=self.model.device, dtype=torch.float32,)
            * (self.input_upper_bound - self.input_lower_bound)
            + self.input_lower_bound
        )
        return x

    def step(self, x, y, alpha=0.1):
        loss = self.loss(y)
        loss.backward()
        if x.grad.abs().max().item() < 1e-12:
            return
        lb = self.input_lower_bound
        ub = self.input_upper_bound
        epsilon = (ub - lb) / 2
        x = x + F.normalize(x.grad) * epsilon * alpha
        return x

    def validate(self, x):
        if np.any(np.isnan(x)):
            return False
        if not self.input_constraint.validate(x):
            return False
        y = self.prop.op_graph(x)
        if not self.output_constraint.validate(*y):
            return False
        return True
