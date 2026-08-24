#!/usr/bin/env python3
"""Add two device matrices and verify the readback.
"""

# OA_DOC_BEGIN: core-matrix-add
from oa import FnMatrix, initComputeEngine, shutdownComputeEngine


if not initComputeEngine():
	raise RuntimeError("OA could not create a Vulkan compute engine")

try:
	one = FnMatrix.ones([2, 3])
	two = FnMatrix.full([2, 3], 2.0)
	sumMatrix = FnMatrix.add(one, two)
	values = FnMatrix.copyToHost(sumMatrix)
	assert values == [3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
	print(values)
finally:
	shutdownComputeEngine()
# OA_DOC_END: core-matrix-add
