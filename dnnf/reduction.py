from __future__ import annotations

import itertools
import logging
import numpy as np

from abc import abstractmethod, ABC
from typing import Dict, Iterator, List, Optional, Tuple, Type, Union

from dnnv.nn import OperationGraph, OperationTransformer
from dnnv.properties import (
    Add,
    And,
    Constant,
    Expression,
    Exists,
    Forall,
    Call,
    LessThan,
    LessThanOrEqual,
    Multiply,
    Network,
    Or,
    Subscript,
    Symbol,
)
from dnnv.properties.visitors import ExpressionVisitor


class ReductionError(Exception):
    pass


class Property:
    @abstractmethod
    def validate_counter_example(self, cex: np.ndarray) -> bool:
        raise NotImplementedError()


class Reduction(ABC):
    def __init__(self, *, reduction_error: Type[ReductionError] = ReductionError):
        self.reduction_error = reduction_error

    @abstractmethod
    def reduce_property(self, phi: Expression) -> Iterator[Property]:
        raise NotImplementedError()


class HPolyReductionError(ReductionError):
    pass


class HPoly:
    pass


class OpGraphMerger(OperationTransformer):
    # TODO : merge common layers (e.g. same normalization, reshaping of input)
    def __init__(self):
        super().__init__()
        self.output_operations = []
        self.input_operations = {}

    def merge(self, operation_graphs: List[OperationGraph]):
        for op_graph in operation_graphs:
            for op in op_graph.output_operations:
                self.output_operations.append(self.visit(op))
        return OperationGraph(self.output_operations)

    def visit_Input(self, operation):
        input_details = (operation.dtype, tuple(operation.shape))
        if input_details not in self.input_operations:
            self.input_operations[input_details] = self.generic_visit(operation)
        return self.input_operations[input_details]


class HPolyProperty(Property):
    def __init__(self, expr_details, input_vars, output_vars, hpoly, lb, ub):
        self.expr_details = expr_details
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.variables = output_vars + input_vars

        self.hpoly = hpoly
        self.lb = lb
        self.ub = ub

        self.var_i_map = {v: i for i, v in enumerate(self.variables)}
        self.var_offsets = [0]
        for v in self.variables:
            self.var_offsets.append(
                self.var_offsets[-1] + np.product(self.expr_details.shapes[v])
            )

        for v in self.output_vars:
            i = self.var_i_map[v]
            offset = self.var_offsets[i]
            shape = self.expr_details.shapes[v]
            for idx in np.ndindex(shape):
                flat_idx = offset + np.ravel_multi_index(idx, shape)
                if not np.isneginf(self.lb[flat_idx]):
                    hs = np.zeros((1, self.lb.shape[0] + 1))
                    hs[0, flat_idx] = -1
                    hs[0, -1] = -self.lb[flat_idx]
                    self.hpoly.append(hs)
                if not np.isposinf(self.ub[flat_idx]):
                    hs = np.zeros((1, self.ub.shape[0] + 1))
                    hs[0, flat_idx] = 1
                    hs[0, -1] = self.ub[flat_idx]
                    self.hpoly.append(hs)

        self.input_lower_bounds = []
        self.input_upper_bounds = []
        for v in self.input_vars:
            i = self.var_i_map[v]
            offset = self.var_offsets[i]
            next_offset = self.var_offsets[i + 1]
            shape = self.expr_details.shapes[v]
            lb = self.lb[offset : next_offset + 1].reshape(shape)
            ub = self.ub[offset : next_offset + 1].reshape(shape)
            self.input_lower_bounds.append(lb)
            self.input_upper_bounds.append(ub)

        op_graphs = (
            n.value for n in sum((list(v.networks) for v in self.output_vars), [])
        )
        merger = OpGraphMerger()
        self.op_graph = merger.merge(op_graphs)
        self.input_ops = tuple(merger.input_operations.values())

    def __repr__(self):
        strs = []
        for x in self.input_vars:
            for idx in np.ndindex(self.expr_details.shapes[x]):
                offset = self.var_offsets[self.var_i_map[x]]
                lb = self.lb[offset + np.product(idx)]
                ub = self.ub[offset + np.product(idx)]
                v = x[idx]
                strs.append(f"{lb} <= {v} <= {ub}")
        for y in self.output_vars:
            for idx in np.ndindex(self.expr_details.shapes[y]):
                offset = self.var_offsets[self.var_i_map[y]]
                lb = self.lb[offset + np.product(idx)]
                ub = self.ub[offset + np.product(idx)]
                v = y[idx]
                strs.append(f"{lb} <= {v} <= {ub}")
        for hs in self.hpoly:
            hs_str = []
            for v, i in self.var_i_map.items():
                offset = self.var_offsets[i]
                shape = self.expr_details.shapes[v]
                for idx in np.ndindex(shape):
                    flat_idx = np.ravel_multi_index(idx, shape) + offset
                    c = hs[0, flat_idx]
                    if abs(c) <= 1e-100:
                        continue
                    hs_str.append(f"{c}*{v[idx]}")
            b = hs[0, -1]
            strs.append(" + ".join(hs_str) + f" <= {b}")
        return "\n".join(strs)

    def validate_counter_example(self, cex: np.ndarray) -> bool:
        if np.any(np.isnan(cex)):
            return False
        if np.any(self.input_lower_bounds[0] > cex) or np.any(
            self.input_upper_bounds < cex
        ):
            return False
        y = self.op_graph(cex)
        if isinstance(y, tuple):
            flat_y = np.hstack([y_.flatten() for y_ in y])
        else:
            flat_y = y.flatten()
        flat_output = np.hstack([flat_y, cex.flatten()])
        for hs in self.hpoly:
            hy = hs[0, :-1] @ flat_output
            b = hs[0, -1]
            if np.any(hy > b):
                return False
        return True

    def suffixed_op_graph(self) -> OperationGraph:
        import dnnv.nn.operations as operations

        output_shape = self.op_graph.output_shape[0]
        axis = (0, 0, 1)[len(output_shape)]
        if len(self.op_graph.output_operations) == 1:
            new_output_op = self.op_graph.output_operations[0]
        else:
            if axis == 0:
                output_operations = [
                    operations.Reshape(o, (-1,))
                    for o in self.op_graph.output_operations
                ]
            else:
                output_operations = [
                    operations.Flatten(o, axis=axis)
                    for o in self.op_graph.output_operations
                ]
            new_output_op = operations.Concat(output_operations, axis=axis)
        if axis == 0:
            flat_input_ops = [operations.Reshape(o, (-1,)) for o in self.input_ops]
        else:
            flat_input_ops = [operations.Flatten(o, axis=axis) for o in self.input_ops]
        new_output_op = operations.Concat([new_output_op] + flat_input_ops, axis=axis)
        dtype = OperationGraph([new_output_op]).output_details[0].dtype

        Wb = np.vstack(self.hpoly)
        W = Wb[:, :-1].T.astype(dtype)
        b = -Wb[:, -1].astype(dtype)
        new_output_op = operations.Add(operations.MatMul(new_output_op, W), b)
        new_output_op = operations.Relu(new_output_op)

        k = len(self.hpoly)
        W_mask = np.zeros((k, 2), dtype=dtype)
        b_mask = np.zeros(2, dtype=dtype)
        for i in range(k):
            W_mask[i, 0] = 1
        new_output_op = operations.Add(operations.MatMul(new_output_op, W_mask), b_mask)
        new_op_graph = (
            OpGraphMerger().merge([OperationGraph([new_output_op])]).simplify()
        )
        return new_op_graph


class ExpressionDetailsInference(ExpressionVisitor):
    def __init__(self, reduction_error: Type[ReductionError] = ReductionError):
        super().__init__()
        self.reduction_error = reduction_error
        # TODO : make types and shapes symbolic so we don't need to order expressions
        self.shapes: Dict[
            Expression, Optional[Union[Tuple[int, ...], List[Tuple[int, ...]]]]
        ] = {}
        self.types: Dict[Expression, Optional[Union[Type, np.dtype]]] = {}

    def visit_Add(self, expression: Add):
        tmp_array: Optional[np.array] = None
        for expr in expression:
            self.visit(expr)
            if tmp_array is None:
                tmp_array = np.empty(self.shapes[expr], dtype=self.types[expr])
            else:
                tmp_array = tmp_array + np.empty(
                    self.shapes[expr], dtype=self.types[expr]
                )
        if tmp_array is not None:
            self.shapes[expression] = tuple(tmp_array.shape)
            self.types[expression] = tmp_array.dtype

    def visit_And(self, expression: And):
        for expr in sorted(expression, key=lambda e: -len(e.networks)):
            self.visit(expr)

    def visit_Call(self, expression: Call):
        if isinstance(expression.function, Network):
            input_details = expression.function.value.input_details
            if len(expression.args) != len(input_details):
                raise self.reduction_error(
                    "Invalid property:"
                    f" Not enough inputs for network '{expression.function}'"
                )
            if len(expression.kwargs) > 0:
                raise self.reduction_error(
                    "Unsupported property:"
                    f" Executing networks with keyword arguments is not currently supported"
                )
            for arg, d in zip(expression.args, input_details):
                if arg in self.shapes and self.shapes[arg] is not None:
                    arg_shape = self.shapes[arg]
                    assert arg_shape is not None
                    if any(
                        i1 != i2 and i2 > 0 for i1, i2 in zip(arg_shape, tuple(d.shape))
                    ):
                        raise self.reduction_error(
                            f"Invalid property: variable with multiple shapes: '{arg}'"
                        )
                self.shapes[arg] = tuple(i if i > 0 else 1 for i in d.shape)
                self.visit(arg)
            output_details = expression.function.value.output_details
            if len(output_details) == 1:
                self.shapes[expression] = output_details[0].shape
                self.types[expression] = output_details[0].dtype
            else:
                self.shapes[expression] = [d.shape for d in output_details]
                self.types[expression] = [d.dtype for d in output_details]
        else:
            raise self.reduction_error(
                "Unsupported property:"
                f" Function {expression.function} is not currently supported"
            )

    def visit_Constant(self, expression: Constant):
        value = expression.value
        if isinstance(value, np.ndarray):
            self.shapes[expression] = value.shape
            self.types[expression] = value.dtype
        elif isinstance(value, (int, float)):
            self.shapes[expression] = ()
            self.types[expression] = type(value)
        elif isinstance(value, (list, tuple)):
            try:
                arr = np.asarray(value)
                self.shapes[expression] = arr.shape
                self.types[expression] = arr.dtype
            except:
                self.shapes[expression] = None
                self.types[expression] = None
        else:
            self.shapes[expression] = None
            self.types[expression] = None

    def visit_Multiply(self, expression: Multiply):
        tmp_array: Optional[np.array] = None
        for expr in expression:
            self.visit(expr)
            if tmp_array is None:
                tmp_array = np.empty(self.shapes[expr], dtype=self.types[expr])
            else:
                tmp_array = tmp_array * np.empty(
                    self.shapes[expr], dtype=self.types[expr]
                )
        if tmp_array is not None:
            self.shapes[expression] = tuple(tmp_array.shape)
            self.types[expression] = tmp_array.dtype

    def visit_Or(self, expression: Or):
        for expr in sorted(expression, key=lambda e: -len(e.networks)):
            self.visit(expr)

    def visit_Subscript(self, expression: Subscript):
        self.visit(expression.expr)
        if not expression.index.is_concrete:
            return
        index = expression.index.value
        expr_shape = self.shapes[expression.expr]
        assert expr_shape is not None
        for i, d in zip(index, expr_shape):
            if not isinstance(i, slice) and i >= d:
                raise self.reduction_error(f"Index out of bounds: {expression}")
        self.shapes[expression] = tuple(np.empty(expr_shape)[index].shape)
        self.types[expression] = self.types[expression.expr]

    def visit_Symbol(self, expression: Symbol):
        if expression not in self.shapes:
            self.shapes[expression] = None
        if expression not in self.types:
            self.types[expression] = None


class HPolyPropertyBuilder:
    def __init__(
        self,
        expr_details: ExpressionDetailsInference,
        input_vars: List[Symbol],
        output_vars: List[Expression],
    ):
        self.expr_details = expr_details
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.variables = self.output_vars + self.input_vars
        self.var_i_map = {v: i for i, v in enumerate(self.variables)}
        self.var_offsets = [0]
        for v in self.variables:
            self.var_offsets.append(
                self.var_offsets[-1] + np.product(self.expr_details.shapes[v])
            )

        num_input_vars = 0
        for x in input_vars:
            num_input_vars += np.product(self.expr_details.shapes[x])
        num_output_vars = 0
        for y in output_vars:
            num_output_vars += np.product(self.expr_details.shapes[y])
        self.num_input_vars = num_input_vars
        self.num_output_vars = num_output_vars
        self.num_vars = num_input_vars + num_output_vars

        self.coefficients: Dict[Expression, np.array] = {}
        self.var_indices: Dict[Expression, Tuple[Expression, np.array]] = {}
        for v in self.variables:
            shape = self.expr_details.shapes[v]
            assert shape is not None
            assert isinstance(shape, tuple)
            var_ids = np.full(shape, self.var_i_map[v])
            indices = np.array([i for i in np.ndindex(shape)]).reshape(
                shape + (len(shape),)
            )
            self.var_indices[v] = (var_ids, indices)

        self.hpoly_constraints: List[np.ndarray] = []
        self.interval_constraints: Tuple[np.ndarray, np.ndarray] = (
            np.full(self.num_vars, -np.inf),
            np.full(self.num_vars, np.inf),
        )

    def add_constraint(self, variables, indices, coeffs, b, is_open):
        if is_open:
            b = np.nextafter(b, b - 1)
        if len(variables) > 1:
            hs = np.zeros((1, self.num_vars + 1))
            for v, i, c in zip(variables, indices, coeffs):
                flat_index = self.var_offsets[v] + np.ravel_multi_index(
                    i, self.expr_details.shapes[self.variables[variables[v]]]
                )
                hs[0, flat_index] = c
            hs[0, self.num_vars] = b
            self.hpoly_constraints.append(hs)
        else:
            flat_index = self.var_offsets[variables[0]] + np.ravel_multi_index(
                indices[0], self.expr_details.shapes[self.variables[variables[0]]]
            )
            coeff = coeffs[0]
            if coeff > 0:
                current_bound = self.interval_constraints[1][flat_index]
                self.interval_constraints[1][flat_index] = min(b / coeff, current_bound)
            elif coeff < 0:
                current_bound = self.interval_constraints[0][flat_index]
                self.interval_constraints[0][flat_index] = max(b / coeff, current_bound)

    def build(self) -> HPolyProperty:
        return HPolyProperty(
            self.expr_details,
            self.input_vars,
            self.output_vars,
            self.hpoly_constraints,
            *self.interval_constraints,
        )


class HPolyReduction(Reduction):
    def __init__(
        self,
        negate: bool = True,
        *,
        reduction_error: Type[ReductionError] = HPolyReductionError,
    ):
        super().__init__(reduction_error=reduction_error)
        self.logger = logging.getLogger(__name__)
        self.negate = negate
        self.expression_details = ExpressionDetailsInference(
            reduction_error=reduction_error
        )
        self._property_builder: Optional[HPolyPropertyBuilder] = None

    def reduce_property(self, phi: Expression) -> Iterator[HPolyProperty]:
        if isinstance(phi, Exists):
            raise NotImplementedError(
                "HPolyReduction currently supports only universally quantified properties"
            )  # TODO : add support
        expr = phi
        while isinstance(expr, Forall):
            expr = expr.expression
        if self.negate or True:
            expr = ~expr
        canonical_expr = expr.canonical()
        assert isinstance(canonical_expr, Or)

        self.expression_details.visit(canonical_expr)
        for expr, shape in self.expression_details.shapes.items():
            if shape is None:
                raise self.reduction_error(
                    f"Unable to infer shape for expression: {expr}"
                )

        for disjunct in canonical_expr:
            self.logger.debug("DISJUNCT: %s", disjunct)
            input_variables = disjunct.variables
            networks = disjunct.networks
            output_variables = [
                network(x)
                for network, x in itertools.product(networks, input_variables)
                if network(x) in self.expression_details.shapes
            ]

            self._property_builder = HPolyPropertyBuilder(
                self.expression_details, list(input_variables), output_variables
            )
            self.visit(disjunct)
            prop = self._property_builder.build()
            yield prop
            self._property_builder = None

    def visit(self, expression: Expression):
        method_name = "visit_%s" % expression.__class__.__name__
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(expression)

    def generic_visit(self, expression: Expression):
        raise NotImplementedError(
            f"No visitor for expression type: {expression.__class__.__name__}"
        )

    def visit_Add(self, expression: Add):
        coeffs = []
        var_indices = []
        assert self._property_builder is not None
        for expr in expression:
            self.visit(expr)
            coeffs.append(self._property_builder.coefficients[expr])
            var_indices.append(self._property_builder.var_indices[expr])
        self._property_builder.var_indices[expression] = tuple(zip(*var_indices))  # type: ignore
        self._property_builder.coefficients[expression] = coeffs

    def visit_Multiply(self, expression: Multiply):
        coeff = None
        variable = None
        if not len(expression.expressions) == 2:
            raise self.reduction_error("Property is not in canonical form.")
        for expr in expression:
            self.visit(expr)
            if expr.is_concrete:
                coeff = expr
            elif variable is None:
                variable = expr
            else:
                raise self.reduction_error(
                    "Non-linear properties are not currently supported"
                )
        assert coeff is not None
        assert variable is not None
        assert self._property_builder is not None
        coeff_shape = self.expression_details.shapes[coeff]
        variable_shape = self.expression_details.shapes[variable]
        coeff_value = np.full(coeff_shape, coeff.value)
        if coeff_shape != variable_shape:
            try:
                broadcast_shape = np.broadcast(
                    np.empty(coeff_shape), np.empty(variable_shape)
                ).shape
                assert broadcast_shape == variable_shape  # TODO: extend this
                coeff_value = np.broadcast_to(coeff_value, broadcast_shape)
            except ValueError:
                raise self.reduction_error(
                    f"Mismatched shapes in Multiply expression: {coeff_shape} and {variable_shape}"
                )
        var_ids, indices = self._property_builder.var_indices[variable]
        self._property_builder.var_indices[expression] = (var_ids, indices)
        self._property_builder.coefficients[expression] = coeff_value

    def visit_Subscript(self, expression: Subscript):
        self.visit(expression.expr)
        self.visit(expression.index)
        if not expression.index.is_concrete:
            raise self.reduction_error("Unsupported property: Symbolic subscript index")
        assert self._property_builder is not None
        var_ids, indices = self._property_builder.var_indices[expression.expr]
        new_var_ids = var_ids[expression.index.value]
        new_indices = indices[expression.index.value]
        self._property_builder.var_indices[expression] = (new_var_ids, new_indices)

    def visit_And(self, expression: And):
        for expr in sorted(expression, key=lambda e: -len(e.networks)):
            self.visit(expr)

    def visit_Call(self, expression: Call):
        if not expression in self.expression_details.shapes:
            raise self.reduction_error(f"Unknown shape for expression: {expression}")

    def visit_Constant(self, expression: Constant):
        pass

    def _add_constraint(self, expression: Union[LessThan, LessThanOrEqual]):
        self.visit(expression.expr1)
        self.visit(expression.expr2)

        lhs = expression.expr1
        rhs = expression.expr2
        lhs_shape = self.expression_details.shapes[lhs]
        rhs_shape = self.expression_details.shapes[rhs]

        assert self._property_builder is not None
        lhs_vars, lhs_indices = self._property_builder.var_indices[lhs]
        lhs_coeffs = self._property_builder.coefficients[lhs]

        assert len(lhs_coeffs) == len(lhs_vars)
        assert len(lhs_vars) == len(lhs_indices)
        assert np.all(v.shape == lhs_vars[0].shape for v in lhs_vars[1:])
        assert np.all(i.shape == lhs_indices[0].shape for i in lhs_indices[1:])

        rhs_value = np.full(rhs_shape, rhs.value)
        if lhs_shape != rhs_shape:
            try:
                broadcast_shape = np.broadcast(
                    np.empty(lhs_shape), np.empty(rhs_shape)
                ).shape
                assert broadcast_shape == lhs_shape  # TODO: extend this
                rhs_value = np.broadcast_to(rhs_value, broadcast_shape)
            except ValueError:
                raise self.reduction_error(
                    f"Mismatched shapes in {type(expression).__name__} expression: {lhs_shape} and {rhs_shape}"
                )

        for idx in np.ndindex(lhs_vars[0].shape):
            variables = tuple(v[idx] for v in lhs_vars)
            indices = tuple(i[idx] for i in lhs_indices)
            coeffs = tuple(c[idx] for c in lhs_coeffs)
            self._property_builder.add_constraint(
                variables,
                indices,
                coeffs,
                rhs_value[idx],
                is_open=(type(expression) is LessThan),
            )

    def visit_LessThanOrEqual(self, expression: LessThanOrEqual):
        self._add_constraint(expression)

    def visit_LessThan(self, expression: LessThan):
        self._add_constraint(expression)

    def visit_Symbol(self, expression: Symbol):
        pass
