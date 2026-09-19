#!/usr/bin/env python3
"""Build the same matrix expression with module functions and operators.

Mirrors sdk/rs/tutorials/core/mat_mul_intro.rs — basic Matrix API syntax.
"""

import oa

engine = oa.Engine()

a = oa.Matrix.from_f32(engine, [2, 3], [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
b = oa.matrix.ones(engine, [2, 3])

# Element-wise arithmetic via the stateless module API.
operator_result = oa.matrix.clamp_min(
	oa.matrix.add_scalar(
		oa.matrix.scale(
			oa.matrix.add(a, b),
			0.5,
		),
		0.5,
	),
	0.0,
)

# Operator overloads express the same computation more concisely.
# Both paths share the same GPU kernels.
operator_alt = (a + b)
operator_alt = oa.matrix.scale(operator_alt, 0.5)
operator_alt = oa.matrix.add_scalar(operator_alt, 0.5)
operator_alt = oa.matrix.clamp_min(operator_alt, 0.0)

# reshape is a zero-cost semantic view; read_f32 is the explicit
# synchronization and device-to-host boundary.
reshaped = operator_result.reshape([3, 2])
values_a = operator_result.read_f32()
values_b = operator_alt.read_f32()

assert operator_result.shape == [2, 3]
assert reshaped.shape == [3, 2]
assert values_a == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
assert values_b == values_a

print(operator_result.shape, values_a)

# MatMul: C = A @ B^T  (FP32 [M,K] × [N,K] -> [M,N])
one = oa.matrix.ones(engine, [2, 3])
two = oa.matrix.full(engine, [2, 3], 2.0)
product = oa.matrix.mat_mul_nt(one, two)
prod_vals = product.read_f32()

assert product.shape == [2, 2]
assert prod_vals == [6.0] * 4

print(product.shape, prod_vals)
