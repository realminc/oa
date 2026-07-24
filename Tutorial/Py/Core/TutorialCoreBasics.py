#!/usr/bin/env python3
"""Build the same matrix expression with operators and OaFnMatrix functions.
"""

# pyright: reportWildcardImportFromLibrary=false
from oa import *


matrix_a = OaFnMatrix.FromFloats(
	[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
	[2, 3],
)
matrix_b = OaFnMatrix.Ones([2, 3])

# Matrix arithmetic is element-wise. Scalar operators use dedicated scalar
# kernels, and compound assignment preserves the object while using the
# corresponding in-place operation.
operator_result = (matrix_a + matrix_b) * 0.5
operator_result += 0.5
operator_result = OaFnMatrix.Relu(operator_result)

# The explicit operation family expresses the same computation when code needs
# dynamic dispatch, generated operation names, or direct C++ syntax parity.
functional_result = OaFnMatrix.Relu(
	OaFnMatrix.AddScalar(
		OaFnMatrix.Scale(
			OaFnMatrix.Add(matrix_a, matrix_b),
			0.5,
		),
		0.5,
	),
)

# Shape methods remain lightweight value/view operations. CopyToHost is the
# explicit synchronization and device-to-host boundary.
reshaped = operator_result.Reshape([3, 2])
operator_values = OaFnMatrix.CopyToHost(operator_result)
functional_values = OaFnMatrix.CopyToHost(functional_result)

assert operator_result.Shape() == [2, 3]
assert reshaped.Shape() == [3, 2]
assert operator_values == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
assert functional_values == operator_values
print(operator_result.Shape(), operator_values)
