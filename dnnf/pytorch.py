from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from dnnv.nn import Operation, OperationGraph, OperationVisitor, operations
from .utils import ONNX_TO_TORCH_DTYPE


def convert(op_graph: OperationGraph) -> PytorchModel:
    converter = PytorchConverter()
    for op in op_graph.output_operations:
        _ = converter.visit(op)
    return PytorchModel(
        converter.operations, converter.inputs, op_graph.output_operations
    )


class PytorchModel(torch.nn.Module):
    def __init__(self, op_cache, inputs, outputs):
        super().__init__()
        self.op_cache = op_cache
        self.inputs = inputs
        self.outputs = outputs
        self.device = torch.device("cpu")

    def forward(self, *x, squeeze=True, **kwargs):
        if len(x) != len(self.inputs):
            raise ValueError("Incorrect number of inputs")
        if any(x_.device.type != self.device.type for x_ in x):
            raise ValueError("Input on incorrect device.")
        op_cache = OpCache(device=self.device)
        op_cache.update(self.op_cache)
        op_cache.update(kwargs)
        for input_op, x_ in zip(self.inputs, x):
            op_cache[input_op] = x_
        outputs = tuple(op_cache[output] for output in self.outputs)
        if squeeze and len(outputs) == 1:
            return outputs[0]
        return outputs

    def to(self, device, *args, **kwargs):
        self.device = torch.device(device)
        return super().to(device, *args, **kwargs)


class OpCache(dict):
    def __init__(self, device=torch.device("cpu")):
        super().__init__()
        self.device = device
        self.cache = {}
        self.__hash__ = super().__hash__

    def __eq__(self, other):
        if not isinstance(other, OpCache):
            return False
        return super().__eq__(other) and self.device == other.device

    def __getitem__(self, index) -> torch.Tensor:
        if isinstance(index, np.ndarray):
            return torch.from_numpy(index).to(self.device)
        if isinstance(index, str):
            return super().get(index, None)
        if not isinstance(index, Operation):
            return index
        if index in self.cache:
            return self.cache[index]
        item = super().__getitem__(index)
        if callable(item):
            self.cache[index] = item(self)
            return self.cache[index]
        if isinstance(item, np.ndarray):
            self.cache[index] = torch.from_numpy(item).to(self.device)
            return self.cache[index]
        if isinstance(item, torch.Tensor):
            self.cache[index] = item
            return self.cache[index]
        raise TypeError(f"Unsupported type: {type(item).__name__}")


class PytorchConverter(OperationVisitor):
    def __init__(self):
        super().__init__()
        self.operations = {}
        self.inputs = []

    def visit(self, operation: Operation):
        if operation not in self.operations:
            result = super().visit(operation)
            self.operations[operation] = result
        return self.operations[operation]

    def generic_visit(self, operation: Operation):
        if not hasattr(self, f"visit_{operation}"):
            raise ValueError(
                f"Pytorch converter not implemented for operation type {operation}"
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
            pad_top, pad_left, pad_bottom, pad_right = operation.pads

            x = (
                x.clone()
            )  # work around for https://github.com/pytorch/pytorch/issues/31734
            padded_x = F.pad(
                x, (pad_left, pad_right, pad_top, pad_bottom), "constant", 0
            )
            result = F.conv2d(
                padded_x,
                weights,
                bias,
                tuple(operation.strides),
                groups=operation.group,
            )
            return result

        return conv

    def visit_ConvTranspose(self, operation: operations.ConvTranspose):
        self.generic_visit(operation)

        def convtranspose(operation_graph):
            x = operation_graph[operation.x]
            weights = operation_graph[operation.w]
            bias = operation_graph[operation.b]
            pad_top, pad_left, pad_bottom, pad_right = operation.pads

            padded_x = F.pad(
                x, (pad_left, pad_right, pad_top, pad_bottom), "constant", 0
            )
            result = F.conv_transpose2d(
                padded_x,
                weights,
                bias=bias,
                stride=operation.strides,
                output_padding=operation.output_padding,
                groups=operation.groups,
                dilations=operation.dilations,
            )
            return result

        return convtranspose

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
            result = x.flatten(axis)
            return result

        return flatten

    def visit_Gather(self, operation: operations.Gather):
        self.generic_visit(operation)

        def gather(operation_graph):
            x = torch.as_tensor(operation_graph[operation.x])
            indices = [slice(None)] * x.ndim
            indices[operation.axis] = operation.indices
            result = x[indices]
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

    def visit_LeakyRelu(self, operation: operations.LeakyRelu):
        self.generic_visit(operation)

        def leakyrelu(operation_graph):
            x = operation_graph[operation.x]
            return F.leaky_relu(x, negative_slope=operation.alpha)

        return leakyrelu

    def visit_MatMul(self, operation: operations.MatMul):
        self.generic_visit(operation)

        def matmul(operation_graph):
            a = operation_graph[operation.a]
            b = operation_graph[operation.b]
            return torch.matmul(a, b)

        return matmul

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

    def visit_OutputSelect(self, operation: operations.OutputSelect):
        self.generic_visit(operation)

        def outputselect(operation_graph):
            op = operation_graph[operation.operation]
            return op[operation.index]

        return outputselect

    def visit_Relu(self, operation: operations.Relu):
        self.generic_visit(operation)

        def relu(operation_graph):
            x = operation_graph[operation.x]
            return F.relu(x)

        return relu

    def visit_Reshape(self, operation: operations.Reshape):
        self.generic_visit(operation)

        def reshape(operation_graph):
            x = operation_graph[operation.x]
            shape = operation_graph[operation.shape]
            return x.reshape(tuple(shape))

        return reshape

    def visit_Resize(self, operation: operations.Resize):
        self.generic_visit(operation)

        def resize(operation_graph):
            x = operation_graph[operation.x]
            assert operation.coordinate_transformation_mode in [
                "asymmetric",
                "tf_crop_and_resize",
            ]
            assert operation.mode == "nearest"
            assert operation.nearest_mode == "floor"
            assert operation.exclude_outside == 0
            assert operation.roi.size == 0

            scales = operation.scales
            sizes = operation.sizes
            if sizes.size == 0:
                assert scales[0] == 1.0 and scales[1] == 1.0
                sizes = (scales * x.shape).astype(int)
            assert sizes.ndim == 1 and sizes.size == 4
            return F.interpolate(x, size=sizes[2:].tolist(), mode=operation.mode)

        return resize

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

    def visit_Split(self, operation: "operations.Split"):
        self.generic_visit(operation)

        def split(operation_graph):
            x = operation_graph[operation.x]
            result = torch.split(x, operation.split, operation.axis)
            return result

        return split

    def visit_Sub(self, operation: operations.Sub):
        self.generic_visit(operation)

        def sub(operation_graph):
            a = operation_graph[operation.a]
            b = operation_graph[operation.b]
            return a - b

        return sub

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

    def visit_Upsample(self, operation: operations.Upsample):
        self.generic_visit(operation)

        def upsample(operation_graph):
            x = operation_graph[operation.x]
            scales = operation.scales.tolist()
            mode = operation.mode
            result = torch.nn.Upsample(scale_factor=tuple(scales[2:]), mode=mode)(x)
            return result

        return upsample

    def visit_Div(self, operation: operations.Div):
        self.generic_visit(operation)

        def div(operation_graph):
            a = operation_graph[operation.a]
            b = operation_graph[operation.b]
            result = torch.div(a, b)
            return result

        return div

    def visit_Squeeze(self, operation: operations.Squeeze):
        self.generic_visit(operation)

        def squeeze(operation_graph):
            x = operation_graph[operation.x]
            axes = operation.axes
            if axes is None:
                result = torch.squeeze(x)
            else:
                result = torch.squeeze(x, dim=axes)
            return result

        return squeeze

    def visit_Expand(self, operation: operations.Expand):
        self.generic_visit(operation)

        def expand(operation_graph):
            x = operation_graph[operation.x]
            shape = operation_graph[operation.shape]
            result = x.expand(shape)
            return result

        return expand

    def visit_Clip(self, operation: operations.Clip):
        self.generic_visit(operation)

        def clip(operation_graph):
            x = operation_graph[operation.x]
            _min = operation.min
            _max = operation.max
            result = torch.clip(x, _min, _max)
            return result

        return clip

    def visit_ReduceL2(self, operation: operations.ReduceL2):
        self.generic_visit(operation)

        def reducel2(operation_graph):
            x = operation_graph[operation.x]
            axes = operation.axes
            keepdims = operation.keepdims
            result = torch.norm(x, p=2, dim=axes, keepdim=bool(keepdims))
            return result

        return reducel2

    def visit_Cast(self, operation: operations.Cast):
        self.generic_visit(operation)

        def cast(operation_graph):
            x = operation_graph[operation.x]
            result = x.type(ONNX_TO_TORCH_DTYPE[operation.to])
            return result

        return cast


__all__ = ["convert", "PytorchConverter"]
