"""Python parity checks for OA's semantic Q4/Q8 weight surface."""

from __future__ import annotations

import pytest

import oa_python_test  # noqa: F401 - bootstraps source builds

import oa


@pytest.fixture(scope="module")
def engine():
	if not oa.initComputeEngine():
		pytest.fail("GPU profile requires an initialized OA Vulkan engine")
	yield
	oa.shutdownComputeEngine()


def test_native_q8_semantic_round_trip_retains_shape(engine):
	values = [-127.0, -63.0, 0.0, 63.0, 127.0]
	inputMatrix = oa.FnMatrix.fromFloats(values, [len(values)])
	weight = oa.FnMatrix.quantize(
		inputMatrix, oa.Quantization.Q8
	)
	output = oa.FnMatrix.dequantize(weight)

	assert isinstance(weight, oa.QuantMatrix)
	assert weight.getQuantization() == oa.Quantization.Q8
	assert weight.rank() == 1
	assert weight.size(0) == len(values)
	assert oa.FnMatrix.copyToHost(output) == values


def test_quantized_matmul_surfaces_consume_gpu_planes_without_weight_expand(engine):
	inputValues = [float((i * 5) % 13 - 6) * 0.25 for i in range(33)]
	weightValues = [float((i * 7) % 17 - 8) * 0.5 for i in range(33)]
	inputMatrix = oa.FnMatrix.fromFloats(inputValues, [1, 33])
	weightMatrix = oa.FnMatrix.fromFloats(weightValues, [1, 33])

	q4 = oa.FnMatrix.quantize(
		weightMatrix, oa.Quantization.Q4
	)
	q4Weight = oa.FnMatrix.dequantize(q4)
	q4Expected = sum(
		x * w for x, w in zip(inputValues, oa.FnMatrix.copyToHost(q4Weight), strict=True)
	)
	q4Output = oa.FnMatrix.matMulNt(inputMatrix, q4)
	assert oa.FnMatrix.copyToHost(q4Output) == pytest.approx([q4Expected], rel=2e-4)

	q8 = oa.FnMatrix.quantize(
		weightMatrix, oa.Quantization.Q8
	)
	q8Weight = oa.FnMatrix.dequantize(q8)
	q8Expected = sum(
		x * w for x, w in zip(inputValues, oa.FnMatrix.copyToHost(q8Weight), strict=True)
	)
	q8Output = oa.FnMatrix.matMulNt(inputMatrix, q8)
	assert oa.FnMatrix.copyToHost(q8Output) == pytest.approx([q8Expected], rel=2e-4)
