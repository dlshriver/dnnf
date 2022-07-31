from __future__ import annotations

import itertools
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
from dnnv.nn import OperationGraph, OperationTransformer, operations
from dnnv.nn.utils import TensorDetails
from dnnv.properties import (
    Add,
    And,
    Call,
    Constant,
    Exists,
    Expression,
    Forall,
    LessThan,
    LessThanOrEqual,
    Multiply,
    Network,
    Or,
    Subscript,
    Symbol,
)
from dnnv.properties.visitors import DetailsInference
from scipy.optimize import linprog


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


class OpGraphMerger(OperationTransformer):
    # TODO : merge common layers (e.g. same normalization, reshaping of input)
    def __init__(self):
        super().__init__()
        self.output_operations = []
        self.input_operations = {}

    def merge(self, operation_graphs: Sequence[OperationGraph]):
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
    @classmethod
    def build(
        cls,
        input_vars: Sequence[Expression],
        output_vars: Sequence[Expression],
        hpoly: Sequence[np.ndarray],
        lb: np.ndarray,
        ub: np.ndarray,
    ):
        hpoly = list(hpoly)
        variables = tuple(output_vars) + tuple(input_vars)
        var_i_map = {v: i for i, v in enumerate(variables)}
        var_offsets = [0]
        for v in variables:
            var_offsets.append(var_offsets[-1] + np.product(v.ctx.shapes[v]))

        for v in output_vars:
            i = var_i_map[v]
            offset = var_offsets[i]
            shape = v.ctx.shapes[v]
            for idx in np.ndindex(*shape):
                flat_idx = offset + np.ravel_multi_index(idx, shape)
                if not np.isneginf(lb[flat_idx]):
                    hs = np.zeros((1, lb.shape[0] + 1))
                    hs[0, flat_idx] = -1
                    hs[0, -1] = -lb[flat_idx]
                    hpoly.append(hs)
                if not np.isposinf(ub[flat_idx]):
                    hs = np.zeros((1, ub.shape[0] + 1))
                    hs[0, flat_idx] = 1
                    hs[0, -1] = ub[flat_idx]
                    hpoly.append(hs)

        input_lower_bounds = []
        input_upper_bounds = []
        for v in input_vars:
            i = var_i_map[v]
            offset = var_offsets[i]
            next_offset = var_offsets[i + 1]
            shape = v.ctx.shapes[v]
            lower_bound = lb[offset : next_offset + 1].reshape(shape)
            upper_bound = ub[offset : next_offset + 1].reshape(shape)
            input_lower_bounds.append(lower_bound)
            input_upper_bounds.append(upper_bound)

        op_graphs = [n.value for n in sum((list(v.networks) for v in output_vars), [])]
        merger = OpGraphMerger()
        op_graph = merger.merge(op_graphs)
        input_ops = tuple(merger.input_operations.values())

        input_output_info = {
            "input_names": [str(expr) for expr in input_vars],
            "input_details": [
                TensorDetails(expr.ctx.shapes[expr], expr.ctx.types[expr])
                for expr in input_vars
            ],
            "output_names": [str(expr) for expr in output_vars],
            "output_details": [
                TensorDetails(expr.ctx.shapes[expr], expr.ctx.types[expr])
                for expr in output_vars
            ],
        }

        return cls(
            op_graph,
            hpoly,
            input_lower_bounds,
            input_upper_bounds,
            input_ops,
            input_output_info,
        )

    def __init__(
        self,
        op_graph: OperationGraph,
        hpoly: Sequence[np.ndarray],
        input_lower_bounds: Sequence[np.ndarray],
        input_upper_bounds: Sequence[np.ndarray],
        input_ops: Sequence[np.ndarray],
        input_output_info: Dict[str, Any],
    ):
        self.op_graph = op_graph
        self.hpoly = hpoly
        self.input_lower_bounds = input_lower_bounds
        self.input_upper_bounds = input_upper_bounds
        self.input_ops = input_ops
        self.input_output_info = input_output_info

    def __repr__(self):
        strs = []
        for i, (x, (shape, _)) in enumerate(
            zip(
                self.input_output_info["input_names"],
                self.input_output_info["input_details"],
            )
        ):
            for idx in np.ndindex(*shape):
                lb = self.input_lower_bounds[i][idx]
                ub = self.input_upper_bounds[i][idx]
                strs.append(f"{lb} <= {x}[{idx}] <= {ub}")
        for hs in self.hpoly:
            hs_str = []
            offset = 0
            for v, (shape, _) in itertools.chain(
                zip(
                    self.input_output_info["output_names"],
                    self.input_output_info["output_details"],
                ),
                zip(
                    self.input_output_info["input_names"],
                    self.input_output_info["input_details"],
                ),
            ):
                for idx in np.ndindex(shape):
                    flat_idx = np.ravel_multi_index(idx, shape) + offset
                    c = hs[0, flat_idx]
                    if abs(c) <= 1e-100:
                        continue
                    hs_str.append(f"{c}*{v}[{idx}]")
                offset = flat_idx + 1
            b = hs[0, -1]
            strs.append(" + ".join(hs_str) + f" <= {b}")
        return "\n".join(strs)

    def validate_counter_example(self, cex: np.ndarray) -> bool:
        if np.any(np.isnan(cex)):
            return False
        if np.any(self.input_lower_bounds[0] > cex) or np.any(
            self.input_upper_bounds[0] < cex
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


class HPolyPropertyBuilder:
    def __init__(
        self,
        input_vars: List[Symbol],
        output_vars: List[Expression],
    ):
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.variables: List[Expression] = self.output_vars + self.input_vars
        self.var_i_map = {v: i for i, v in enumerate(self.variables)}
        self.var_offsets = [0]
        for v in self.variables:
            self.var_offsets.append(self.var_offsets[-1] + np.product(v.ctx.shapes[v]))

        num_input_vars = 0
        for x in input_vars:
            num_input_vars += np.product(x.ctx.shapes[x])
        num_output_vars = 0
        for y in output_vars:
            num_output_vars += np.product(y.ctx.shapes[y])
        self.num_input_vars = num_input_vars
        self.num_output_vars = num_output_vars
        self.num_vars = num_input_vars + num_output_vars

        self.coefficients: Dict[
            Expression, Union[np.ndarray, Sequence[np.ndarray]]
        ] = {}
        self.var_indices: Dict[
            Expression,
            Union[
                Tuple[Expression, np.ndarray], Sequence[Tuple[Expression, np.ndarray]]
            ],
        ] = {}
        for v in self.variables:
            shape = v.ctx.shapes[v]
            assert shape is not None
            assert isinstance(shape, tuple)
            var_ids = np.full(shape, self.var_i_map[v])
            indices = np.array([i for i in np.ndindex(*shape)]).reshape(
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
                variable = self.variables[variables[v]]
                flat_index = self.var_offsets[v] + np.ravel_multi_index(
                    i, variable.ctx.shapes[variable]
                )
                hs[0, flat_index] = c
            hs[0, self.num_vars] = b
            self.hpoly_constraints.append(hs)
        else:
            variable = self.variables[variables[0]]
            flat_index = self.var_offsets[variables[0]] + np.ravel_multi_index(
                indices[0], variable.ctx.shapes[variable]
            )
            coeff = coeffs[0]
            if coeff > 0:
                current_bound = self.interval_constraints[1][flat_index]
                self.interval_constraints[1][flat_index] = min(b / coeff, current_bound)
            elif coeff < 0:
                current_bound = self.interval_constraints[0][flat_index]
                self.interval_constraints[0][flat_index] = max(b / coeff, current_bound)

    def build(self) -> HPolyProperty:
        if self.hpoly_constraints:
            Ab = np.vstack(self.hpoly_constraints)
            A: np.ndarray = Ab[..., :-1]
            b: np.ndarray = Ab[..., -1:]
            bounds = tuple(zip(*self.interval_constraints))
            for i in np.flatnonzero(abs(A).sum(0)):
                c = np.zeros(self.num_vars)
                c[i] = 1
                result = linprog(c, A, b, bounds=bounds, method="highs")
                if result.success:
                    current_bound = self.interval_constraints[0][i]
                    self.interval_constraints[0][i] = max(result.x[i], current_bound)
                c[i] = -1
                result = linprog(c, A, b, bounds=bounds, method="highs")
                if result.success:
                    current_bound = self.interval_constraints[1][i]
                    self.interval_constraints[1][i] = min(result.x[i], current_bound)
        return HPolyProperty.build(
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
        self._property_builder: Optional[HPolyPropertyBuilder] = None

    def reduce_property(self, phi: Expression) -> Iterator[HPolyProperty]:
        if isinstance(phi, Exists):
            raise NotImplementedError(
                "HPolyReduction currently supports only"
                " universally quantified specifications"
            )
        expr = phi
        while isinstance(expr, Forall):
            expr = expr.expression
        if self.negate:
            expr = ~expr
        self.logger.debug("Converting expression to canonical DNF.")
        canonical_expr = expr.canonical()
        assert isinstance(canonical_expr, Or)
        self.logger.debug("Running shape and type inference on expression.")
        DetailsInference().visit(canonical_expr)
        self.logger.debug("Reducing disjuncts.")
        for disjunct in canonical_expr:
            self.logger.debug("DISJUNCT: %s", disjunct)
            input_variables = disjunct.variables
            output_variables = list(
                set(
                    expr
                    for expr in disjunct.iter()
                    if isinstance(expr, Call)
                    and isinstance(expr.function, Network)
                    and expr in expr.ctx.shapes
                )
            )

            self._property_builder = HPolyPropertyBuilder(
                list(input_variables), output_variables
            )
            self.visit(disjunct)
            prop = self._property_builder.build()
            yield prop
            self._property_builder = None

    def visit(self, expression: Expression):
        method_name = f"visit_{type(expression).__name__}"
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
            coeff = self._property_builder.coefficients[expr]
            assert isinstance(coeff, np.ndarray)
            coeffs.append(coeff)
            var_indices.append(self._property_builder.var_indices[expr])
        self._property_builder.var_indices[expression] = tuple(zip(*var_indices))
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
        coeff_shape = expression.ctx.shapes[coeff]
        variable_shape = expression.ctx.shapes[variable]
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
                    "Mismatched shapes in Multiply expression:"
                    f" {coeff_shape} and {variable_shape}"
                )
        self._property_builder.var_indices[
            expression
        ] = self._property_builder.var_indices[variable]
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
        if expression not in expression.ctx.shapes:
            raise self.reduction_error(f"Unknown shape for expression: {expression}")

    def visit_Constant(self, expression: Constant):
        pass

    def _add_constraint(self, expression: Union[LessThan, LessThanOrEqual]):
        self.visit(expression.expr1)
        self.visit(expression.expr2)

        lhs = expression.expr1
        rhs = expression.expr2
        lhs_shape = expression.ctx.shapes[lhs]
        rhs_shape = expression.ctx.shapes[rhs]

        assert self._property_builder is not None
        lhs_vars, lhs_indices = self._property_builder.var_indices[lhs]
        lhs_coeffs = self._property_builder.coefficients[lhs]

        assert len(lhs_coeffs) == len(lhs_vars)
        assert len(lhs_vars) == len(lhs_indices)
        assert all(v.shape == lhs_vars[0].shape for v in lhs_vars[1:])
        assert all(i.shape == lhs_indices[0].shape for i in lhs_indices[1:])

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
                    f"Mismatched shapes in {type(expression).__name__} expression:"
                    f" {lhs_shape} and {rhs_shape}"
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
                is_open=isinstance(expression, LessThan),
            )

    def visit_LessThanOrEqual(self, expression: LessThanOrEqual):
        self._add_constraint(expression)

    def visit_LessThan(self, expression: LessThan):
        self._add_constraint(expression)

    def visit_Symbol(self, expression: Symbol):
        pass


__all__ = [
    "HPolyProperty",
    "HPolyReduction",
    "HPolyReductionError",
    "Property",
    "Reduction",
    "ReductionError",
]
