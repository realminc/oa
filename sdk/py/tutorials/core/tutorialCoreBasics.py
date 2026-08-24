#!/usr/bin/env python3
"""Build the same matrix expression with operators and FnMatrix functions.
"""

# pyright: reportWildcardImportFromLibrary=false
from oa import *


matrixA = FnMatrix.fromFloats(
	[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
	[2, 3],
)
matrixB = FnMatrix.ones([2, 3])

# Matrix arithmetic is element-wise. Scalar operators use dedicated scalar
# kernels, and compound assignment preserves the object while using the
# corresponding in-place operation.
operatorResult = (matrixA + matrixB) * 0.5
operatorResult += 0.5
operatorResult = FnMatrix.relu(operatorResult)

# The explicit operation family expresses the same computation when code needs
# dynamic dispatch, generated operation names, or direct C++ syntax parity.
functionalResult = FnMatrix.relu(
	FnMatrix.addScalar(
		FnMatrix.scale(
			FnMatrix.add(matrixA, matrixB),
			0.5,
		),
		0.5,
	),
)

# Shape methods remain lightweight value/view operations. copyToHost is the
# explicit synchronization and device-to-host boundary.
reshaped = operatorResult.reshape([3, 2])
operatorValues = FnMatrix.copyToHost(operatorResult)
functionalValues = FnMatrix.copyToHost(functionalResult)

assert operatorResult.shape() == [2, 3]
assert reshaped.shape() == [3, 2]
assert operatorValues == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
assert functionalValues == operatorValues
print(operatorResult.shape(), operatorValues)
