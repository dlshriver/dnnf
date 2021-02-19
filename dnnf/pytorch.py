import numpy as np
import torch
import torch.nn.functional as F

from dnnv.nn import Operation, OperationVisitor, operations
from typing import Iterable


def convert(operations):
    converter = PytorchConverter()
    for op in operations:
        result = converter.visit(op)
    return PytorchModel(converter.operations, converter.inputs, operations)


class PytorchModel(torch.nn.Module):
    def __init__(self, op_graph, inputs, outputs):
        super().__init__()
        self.op_graph = op_graph
        self.inputs = inputs
        self.outputs = outputs
        self.device = torch.device("cpu")

    def forward(self, *x, squeeze=True, **kwargs):
        if len(x) != len(self.inputs):
            raise ValueError("Incorrect number of inputs")
        if any(x_.device.type != self.device.type for x_ in x):
            raise ValueError("Input on incorrect device.")
        op_graph = OperationGraph(device=self.device)
        op_graph.update(self.op_graph)
        op_graph.update(kwargs)
        for input_op, x_ in zip(self.inputs, x):
            op_graph[input_op] = x_
        outputs = tuple(op_graph[output] for output in self.outputs)
        if squeeze and len(outputs) == 1:
            return outputs[0]
        return outputs

    def to(self, device, *args, **kwargs):
        self.device = torch.device(device)
        return super().to(device, *args, **kwargs)


class OperationGraph(dict):
    def __init__(self, device=torch.device("cpu")):
        super().__init__()
        self.device = device

    def __getitem__(self, index) -> torch.Tensor:
        if isinstance(index, np.ndarray):
            return torch.from_numpy(index).to(self.device)
        if isinstance(index, str):
            return super().get(index, None)
        if not isinstance(index, Operation):
            return index
        item = super().__getitem__(index)
        if callable(item):
            return item(self)
        if isinstance(item, np.ndarray):
            return torch.from_numpy(item).to(self.device)
        if isinstance(item, torch.Tensor):
            return item
        raise TypeError(f"Unsupported type: {type(item).__name__}")


class PytorchConverter(OperationVisitor):
    def __init__(self):
        super().__init__()
        self.operations = {}
        self.inputs = []

    def visit(self, operation):
        if operation not in self.operations:
            result = super().visit(operation)
            self.operations[operation] = result
        return self.operations[operation]

    def generic_visit(self, operation):
        if not hasattr(self, "visit_%s" % operation.__class__.__name__):
            raise ValueError(
                "Pytorch converter not implemented for operation type %s"
                % operation.__class__.__name__
            )
        return super().generic_visit(operation)

    def visit_Add(self, operation: operations.Add):
        self.generic_visit(operation)

        def add(operation_graph):
            a = operation_graph[operation.a]
            b = operation_graph[operation.b]
            return a + b

        return add

    def visit_Atan(self, operation: operations.Atan):
        self.generic_visit(operation)

        def atan(operation_graph):
            x = operation_graph[operation.x]
            return torch.atan(x)

        return atan

    def visit_AveragePool(self, operation: operations.AveragePool):
        self.generic_visit(operation)

        def average_pool(operation_graph):
            x = operation_graph[operation.x]
            kernel_shape = tuple(operation.kernel_shape)
            strides = tuple(operation.strides)
            pad_top, pad_left, pad_bottom, pad_right = operation.pads
            assert not operation.ceil_mode
            assert not operation.count_include_pad

            padded_x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
            result = F.avg_pool2d(padded_x, kernel_shape, strides)
            return result

        return average_pool

    def visit_BatchNormalization(self, operation: operations.BatchNormalization):
        self.generic_visit(operation)

        def batchnorm(operation_graph):
            x = operation_graph[operation.x]
            scale = operation_graph[operation.scale]
            bias = operation_graph[operation.bias]
            mean = operation_graph[operation.mean]
            variance = operation_graph[operation.variance]
            epsilon = operation_graph[operation.epsilon]

            result = F.batch_norm(x, mean, variance, scale, bias, eps=epsilon)
            return result

        return batchnorm

    def visit_Concat(self, operation: operations.Concat):
        self.generic_visit(operation)

        def concat(operation_graph):
            x = [operation_graph[x_] for x_ in operation.x]
            axis = operation.axis
            return torch.cat(x, dim=axis)

        return concat

    def visit_Conv(self, operation: operations.Conv):
        self.generic_visit(operation)

        def conv(operation_graph):
            x = operation_graph[operation.x]
            weights = operation_graph[operation.w]
            bias = operation_graph[operation.b]
            assert np.all(operation.dilations == 1)
            assert np.all(operation.group == 1)
            pad_top, pad_left, pad_bottom, pad_right = operation.pads

            x = (
                x.clone()
            )  # work around for https://github.com/pytorch/pytorch/issues/31734
            padded_x = F.pad(
                x, (pad_left, pad_right, pad_top, pad_bottom), "constant", 0
            )
            result = F.conv2d(padded_x, weights, bias, operation.strides)
            return result

        return conv

    def visit_Elu(self, operation: operations.Elu):
        self.generic_visit(operation)

        def elu(operation_graph):
            x = operation_graph[operation.x]
            return F.elu(x)

        return elu

    def visit_Flatten(self, operation: operations.Flatten):
        self.generic_visit(operation)

        def flatten(operation_graph):
            x = operation_graph[operation.x]
            axis = operation_graph[operation.axis]
            new_shape = (1, -1) if axis == 0 else (int(np.prod(x.shape[:axis])), -1)
            result = x.reshape(new_shape)
            return result

        return flatten

    def visit_Gather(self, operation: operations.Gather):
        self.generic_visit(operation)

        def gather(operation_graph):
            x = torch.as_tensor(operation_graph[operation.x])
            axis = int(operation.axis)
            indices = torch.as_tensor(operation_graph[operation.indices])
            result = torch.gather(x, axis, indices)
            return result

        return gather

    def visit_Gemm(self, operation: operations.Gemm):
        self.generic_visit(operation)

        def gemm(operation_graph):
            a = operation_graph[operation.a]
            b = operation_graph[operation.b]
            c = operation_graph[operation.c]
            alpha = operation_graph[operation.alpha]
            beta = operation_graph[operation.beta]
            transpose_a = operation_graph[operation.transpose_a]
            transpose_b = operation_graph[operation.transpose_b]
            if transpose_a:
                a = a.transpose(0, 1)
            if transpose_b:
                b = b.transpose(0, 1)
            return alpha * torch.matmul(a, b) + beta * c

        return gemm

    def visit_Input(self, operation: operations.Input):
        self.generic_visit(operation)

        self.inputs.append(operation)
        return lambda x: None

    def visit_MaxPool(self, operation: operations.MaxPool):
        self.generic_visit(operation)

        def maxpool(operation_graph):
            x = operation_graph[operation.x]
            kernel_shape = tuple(operation.kernel_shape)
            strides = tuple(operation.strides)
            pad_top, pad_left, pad_bottom, pad_right = operation.pads
            assert not operation.ceil_mode
            assert np.all(operation.dilations == 1)
            assert operation.storage_order == operation.ROW_MAJOR_STORAGE

            padded_x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
            result = F.max_pool2d(padded_x, kernel_shape, strides)
            return result

        return maxpool

    def visit_Mul(self, operation: operations.Mul):
        self.generic_visit(operation)

        def mul(operation_graph):
            a = operation_graph[operation.a]
            b = operation_graph[operation.b]
            return a * b

        return mul

    def visit_Relu(self, operation: operations.Relu):
        self.generic_visit(operation)

        def relu(operation_graph):
            x = operation_graph[operation.x]
            if operation_graph["relu_approx"]:
                return F.softplus(x, beta=1)
            return F.relu(x)

        return relu

    def visit_Reshape(self, operation: operations.Reshape):
        self.generic_visit(operation)

        def reshape(operation_graph):
            x = operation_graph[operation.x]
            shape = operation_graph[operation.shape]
            return x.reshape(tuple(shape))

        return reshape

    def visit_Shape(self, operation: operations.Shape):
        self.generic_visit(operation)

        def shape(operation_graph):
            x = operation_graph[operation.x]
            return tuple(x.shape)

        return shape

    def visit_Sigmoid(self, operation: operations.Sigmoid):
        self.generic_visit(operation)

        def sigmoid(operation_graph):
            x = operation_graph[operation.x]
            return torch.sigmoid(x)

        return sigmoid

    def visit_Softmax(self, operation: operations.Softmax):
        self.generic_visit(operation)

        def softmax(operation_graph):
            x = operation_graph[operation.x]
            axis = operation.axis
            return torch.softmax(x, dim=axis)

        return softmax

    def visit_Tanh(self, operation: operations.Tanh):
        self.generic_visit(operation)

        def tanh(operation_graph):
            x = operation_graph[operation.x]
            return torch.tanh(x)

        return tanh

    def visit_Transpose(self, operation: operations.Transpose):
        self.generic_visit(operation)

        def transpose(operation_graph):
            x = operation_graph[operation.x]
            permutation = operation_graph[operation.permutation]
            return x.permute(tuple(permutation))

        return transpose

    def visit_Unsqueeze(self, operation: operations.Unsqueeze):
        self.generic_visit(operation)

        def unsqueeze(operation_graph):
            x = operation_graph[operation.x]
            result = x
            for axis in operation.axes:
                result = torch.unsqueeze(result, int(axis))
            return result

        return unsqueeze
