#!/usr/bin/env python3
"""Smoke tests for the OA Python bindings.

Run from a checkout with:
	python -m pytest test/py/core/test_smoke.py

If the native extension is not on PYTHONPATH, set OA_PYTHON_BUILD_DIR to the
directory containing the private `_oa` extension.

"""

from __future__ import annotations

import importlib
import math
import sys
import tempfile
from pathlib import Path
from types import ModuleType

import pytest

import oa_python_test  # noqa: F401 - bootstraps source builds

import oa
from oa._native import native

core = oa.FnMatrix
ml = oa
runtime = oa


@pytest.fixture(scope="session")
def engine():
	if not runtime.initComputeEngine():
		pytest.fail("GPU profile requires an initialized OA Vulkan engine")
	yield
	shutdown = getattr(runtime, "shutdownComputeEngine", None)
	if shutdown is not None:
		shutdown()


def _setGradEnabled(enabled: bool) -> None:
	oa.FnAutograd.setEnabled(enabled)


def test_import_surface():
	assert hasattr(runtime, "initComputeEngine")
	assert hasattr(oa, "MatrixShape")
	assert isinstance(oa.FnMatrix, ModuleType)
	assert isinstance(oa.FnAudio, ModuleType)
	assert native.FnMatrix.add is oa.FnMatrix.add
	assert native.FnAudio.normalize is oa.FnAudio.normalize
	assert native.FnLoss.crossEntropy is oa.FnLoss.crossEntropy
	assert native.FnImage.resize is oa.FnImage.resize
	assert not hasattr(native.core, "Add")
	assert not hasattr(native.ml, "Relu")
	assert not hasattr(native.audio, "Normalize")
	assert not hasattr(native.vision, "Resize")
	assert importlib.import_module("oa.FnAudio") is oa.FnAudio
	assert oa.Audio.__module__ == "oa"
	assert oa.plot.Figure.__module__ == "oa.plot"
	assert not any(name.startswith("Oa") for name in oa.__all__)
	assert "core" in oa.__all__
	assert "FnMatrix" in oa.__all__
	assert "Context" not in oa.__all__
	assert not hasattr(oa, "Context")
	assert not hasattr(oa, "oa.ExecutionSession")
	assert not hasattr(oa, "oa.ExecutionSessionGetDefault")
	for lowercaseModule in (
		"audio", "core", "crypto", "ml", "plot", "runtime", "ui", "vision"
	):
		assert hasattr(oa, lowercaseModule)
		assert not (Path(oa.__file__).parent / f"{lowercaseModule}.py").exists()
	assert not hasattr(oa, "OaMatrix")
	assert isinstance(oa.__version__, str)


def test_metric_figure_save(engine):
	config = oa.plot.FigureConfig()
	config.title = "Training evaluation"
	config.rows = 1
	config.cols = 2
	config.width = 480
	config.height = 240
	config.hSpacing = 12
	config.padding = 12
	assert config.title == "Training evaluation"
	figure = oa.plot.Figure(config)
	assert hasattr(figure, "saveTo")
	assert not hasattr(figure, "saveFig")
	figure.title("Validation summary")
	figure.xLabel("checkpoint")
	figure.yLabel("metric")
	figure.ax(0, 0).title("loss")
	figure.ax(0, 0).xLabel("step")
	figure.ax(0, 0).yLabel("loss")
	figure.ax(0, 0).plot([1.0, 0.7, 0.5, 0.4])
	figure.ax(0, 1).title("confusion")
	figure.ax(0, 1).heatmap([8.0, 1.0, 2.0, 7.0], 2, 2)

	rendered = figure.render()
	assert isinstance(rendered, oa.Image)
	assert rendered.width() == 480
	assert rendered.height() == 240
	assert rendered.channels() == 4
	assert rendered.layout() == oa.ImageLayout.Nchw
	assert rendered.format() == oa.ImageFormat.Rgba
	assert rendered.validate()

	with tempfile.NamedTemporaryFile(suffix=".png") as output:
		figure.saveTo(output.name)
		assert Path(output.name).stat().st_size > 512


def test_plot_artist_surface(engine):
	config = oa.plot.FigureConfig()
	config.rows, config.cols = 1, 2
	config.width, config.height = 480, 240
	config.theme = oa.plot.Theme.Light
	config.backgroundRgba = 0xF8F8F8FF
	assert config.theme is oa.plot.Theme.Light
	assert config.backgroundRgba == 0xF8F8F8FF

	line = oa.plot.LineStyle()
	line.label = "ROC"
	line.colorRgba = 0x6366F1FF
	line.width = 1.5
	line.antialiasSamples = 8
	assert line.label == "ROC"
	assert line.colorRgba == 0x6366F1FF
	assert line.width == pytest.approx(1.5)
	assert line.antialiasSamples == 8

	points = oa.plot.ScatterStyle()
	points.label = "calibration"
	points.colorRgba = 0x22D3EEFF
	points.radius = 3.5

	bars = oa.plot.BarStyle()
	bars.label = "scores"
	bars.colorRgba = 0xF59E0BFF
	bars.gap = 0.25

	figure = oa.plot.Figure(config)
	assert figure.rows == 1
	assert figure.cols == 2
	curves = figure.ax(0, 0)
	curves.title("evaluation", 0x0A0A0AFF)
	curves.limits(0.0, 1.0, 0.0, 1.0)
	curves.grid(True)
	curves.legend(True)
	curves.borderColor(0x525252FF)
	image = oa.Image(
		core.full([1, 3, 8, 8], 0.2),
		oa.ImageLayout.Nchw,
		oa.ImageFormat.Rgb,
	)
	curves.imshow(image)
	curves.plot([0.0, 0.2, 0.6, 1.0], [0.0, 0.65, 0.91, 1.0], line)
	curves.scatter([0.1, 0.4, 0.8], [0.15, 0.46, 0.77], points)

	distribution = figure.ax(0, 1)
	distribution.caption("bounded telemetry", 0x525252FF)
	distribution.bar([0.3, 0.6, 0.9, 0.7], bars)
	distribution.histogram(
		[0.1, 0.2, 0.2, 0.5, 0.7, 0.8, 0.9], 6, bars
	)
	distribution.autoLimits()

	rendered = figure.render()
	assert isinstance(rendered, oa.Image)
	assert rendered.validate()


def test_shape_creation():
	shape = oa.MatrixShape([3, 4])
	assert shape.rank == 2
	assert shape.numElements() == 12
	assert shape[0] == 3
	assert shape[1] == 4


def test_autograd_enable_alias():
	_setGradEnabled(False)


def test_gradient_tape_context_restores_state():
	_setGradEnabled(False)
	with oa.GradientTape():
		assert oa.FnAutograd.isEnabled() is True
	assert oa.FnAutograd.isEnabled() is False

	_setGradEnabled(True)
	assert oa.FnAutograd.isEnabled() is True

	_setGradEnabled(False)


def test_gradient_tape_rejects_saved_value_mutation(engine):
	x = core.full(2, 2, 2.0)
	x.setRequiresGrad(True)

	with oa.GradientTape() as tape:
		loss = oa.FnMatrix.sum(oa.FnMatrix.mul(x, x))
		x += 1.0
		with pytest.raises(RuntimeError, match="modified in place"):
			tape.backward(loss)

	assert core.copyToHost(x) == pytest.approx([3.0] * 4)


def test_matrix_factories_and_readback(engine):
	z = core.zeros(2, 3)
	o = core.ones(2, 3)
	f = core.full(2, 3, 5.0)

	assert z.rank() == 2
	assert o.numElements() == 6
	assert f.size(0) == 2
	assert f.size(1) == 3

	assert core.copyToHost2D(f, 2, 3) == [[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]]


def test_matrix_operator_cpp_parity(engine):
	a = core.fromFloats([1.0, 2.0, 3.0, 4.0], [2, 2])
	b = core.fromFloats([2.0, 4.0, 6.0, 8.0], [2, 2])

	matrixExpression = ((a + b) * b - a) / b
	scalarExpression = ((a + 1.0) * 2.0 - 2.0) / 2.0
	assert core.copyToHost(matrixExpression) == pytest.approx(
		[2.5, 5.5, 8.5, 11.5]
	)
	assert core.copyToHost(scalarExpression) == pytest.approx(
		[1.0, 2.0, 3.0, 4.0]
	)
	assert core.copyToHost(-a) == pytest.approx(
		[-1.0, -2.0, -3.0, -4.0]
	)

	inPlace = a.clone()
	identity = id(inPlace)
	inPlace += b
	inPlace -= b
	inPlace *= b
	inPlace /= b
	inPlace += 1.0
	inPlace -= 1.0
	inPlace *= 2.0
	inPlace /= 2.0
	assert id(inPlace) == identity
	assert core.copyToHost(inPlace) == pytest.approx(
		[1.0, 2.0, 3.0, 4.0]
	)


def test_from_floats_and_scale(engine):
	# Float feature upload + Scale — the MNIST input path. A UInt8 matrix would
	# silently produce garbage through Scale/matmul, so float data must come in
	# as Float32 via FromFloats.
	x = core.fromFloats([0.0, 64.0, 128.0, 255.0], 1, 4)
	assert x.dtype() == oa.ScalarType.Float32
	assert core.copyToHost(x) == [0.0, 64.0, 128.0, 255.0]

	s = core.scale(x, 1.0 / 255.0)

	got = core.copyToHost(s)
	expected = [0.0, 64.0 / 255.0, 128.0 / 255.0, 1.0]
	assert all(abs(a - b) < 1e-5 for a, b in zip(got, expected)), got


def test_softmax_family_selected_axis(engine):
	values = [float(i - 12) / 4.0 for i in range(24)]
	x = core.fromFloats(values, [2, 3, 4])

	softmax = core.softmax(x, dim=1)
	logSoftmax = core.logSoftmax(x, dim=1)
	softmaxBwd = core.softmaxBwd(softmax, core.ones([2, 3, 4]), dim=1)

	softmaxValues = core.copyToHost(softmax)
	logSoftmaxValues = core.copyToHost(logSoftmax)
	softmaxBwdValues = core.copyToHost(softmaxBwd)
	assert softmax.shape() == [2, 3, 4]
	assert logSoftmax.shape() == [2, 3, 4]
	assert softmaxBwd.shape() == [2, 3, 4]
	assert softmaxBwdValues == pytest.approx([0.0] * 24, abs=1.0e-6)

	for outer in range(2):
		for inner in range(4):
			indices = [outer * 12 + axis * 4 + inner for axis in range(3)]
			assert abs(sum(softmaxValues[index] for index in indices) - 1.0) < 1e-5
			assert abs(sum(math.exp(logSoftmaxValues[index]) for index in indices) - 1.0) < 1e-5


def test_mean_selected_axis(engine):
	values = [float(i - 12) / 4.0 for i in range(24)]
	x = core.fromFloats(values, [2, 3, 4])

	mean = core.mean(x, dim=1)

	got = core.copyToHost(mean)
	assert mean.shape() == [2, 1, 4]
	for outer in range(2):
		for inner in range(4):
			indices = [outer * 12 + axis * 4 + inner for axis in range(3)]
			expected = sum(values[index] for index in indices) / 3.0
			assert abs(got[outer * 4 + inner] - expected) < 1e-6


def test_matmulnt_and_elementwise(engine):
	a = core.full(2, 3, 1.0)
	b = core.full(2, 3, 2.0)

	c = core.matMulNt(a, b)
	addR = core.add(a, b)
	subR = core.sub(a, b)
	mulR = core.mul(a, b)

	assert core.copyToHost2D(c, 2, 2) == [[6.0, 6.0], [6.0, 6.0]]
	assert core.copyToHost2D(addR, 2, 3) == [[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]]
	assert core.copyToHost2D(subR, 2, 3) == [[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]]
	assert core.copyToHost2D(mulR, 2, 3) == [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]


def test_reduction_bindings(engine):
	x = core.full(2, 3, 2.0)

	total = core.sum(x)
	rows = core.sum(x, 1)
	maximum = core.max(x)

	assert core.copyToHost(total) == [12.0]
	assert rows.shape() == [2, 1]
	assert core.copyToHost(rows) == [6.0, 6.0]
	assert core.copyToHost(maximum) == [2.0]


def test_generated_index_plan_and_accuracy_bindings(engine):
	fn = oa.FnMatrix
	x = fn.fromFloats(
		[1.0, 4.0, 3.0, 2.0, 5.0, 0.0, 6.0, 1.0],
		[2, 4],
	)

	top = fn.topK(x, 2)
	assert isinstance(top, oa.TopKResult)
	assert fn.copyToHost(top.values) == [4.0, 3.0, 6.0, 5.0]
	assert fn.copyToHost(top.indices) == [1, 2, 2, 0]

	indices = fn.fromInt32([1, 2], [2, 1])
	assert fn.copyToHost(fn.gatherLastDim(x, indices)) == [4.0, 6.0]
	repeated = fn.repeatInterleave(
		fn.fromFloats([1.0, 2.0], [1, 2]), 2, 1
	)
	assert fn.copyToHost(repeated) == [1.0, 1.0, 2.0, 2.0]

	labels = fn.fromInt32([1, 2], [2])
	assert fn.copyToHost(fn.categoricalAccuracyCount(x, labels)) == [2]

	routes = fn.fromInt32([2, 0, 1, 2, 0, 1, 2, 1], [4, 2])
	plan = fn.moeExpertPlan(routes, 3)
	assert isinstance(plan, oa.MoeExpertPlan)
	assert fn.copyToHost(plan.counts) == [2, 3, 3]
	assert fn.copyToHost(plan.offsets) == [0, 2, 5, 8]

	values = fn.fromFloats(
		[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
		[4, 2],
	)
	mask = fn.fromFloats([1.0, 0.0, 1.0, 0.0], [4, 1])
	compact = fn.compactRows(values, mask)
	assert isinstance(compact, oa.CompactRowsResult)
	assert fn.copyToHost(compact.count) == [2]
	assert fn.copyToHost(compact.values)[:4] == [1.0, 2.0, 5.0, 6.0]
	assert fn.copyToHost(compact.rowMap)[:2] == [0, 2]


def test_variadic_concat_and_split_bindings(engine):
	fn = oa.FnMatrix
	a = fn.fromFloats([1.0, 2.0], [1, 2])
	b = fn.fromFloats([3.0, 4.0, 5.0], [1, 3])
	c = fn.fromFloats([6.0], [1, 1])

	concatenated = fn.concat([a, b, c], 1)
	assert concatenated.shape() == [1, 6]
	assert fn.copyToHost(concatenated) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

	splits = fn.split(concatenated, [2, 3, 1], 1)
	assert isinstance(splits, list)
	assert [split.shape() for split in splits] == [[1, 2], [1, 3], [1, 1]]
	assert [fn.copyToHost(split) for split in splits] == [
		[1.0, 2.0],
		[3.0, 4.0, 5.0],
		[6.0],
	]


def test_batched_attention_bindings(engine):
	fn = oa.FnMatrix
	a = fn.fromFloats([1.0, 2.0, 3.0, 4.0], [1, 2, 2])
	b = fn.fromFloats([5.0, 6.0, 7.0, 8.0], [1, 2, 2])
	product = fn.bmm(a, b)
	assert product.shape() == [1, 2, 2]
	assert fn.copyToHost(product) == [19.0, 22.0, 43.0, 50.0]
	productNt = fn.bmmNt(a, b)
	assert productNt.shape() == [1, 2, 2]
	assert fn.copyToHost(productNt) == [17.0, 23.0, 39.0, 53.0]

	tokens = fn.fromFloats([float(value) for value in range(16)], [4, 4])
	heads = fn.splitHeads(tokens, 2, 2, 2)
	merged = fn.mergeHeads(heads, 2, 2, 2)
	assert heads.shape() == [4, 2, 2]
	assert merged.shape() == [4, 4]
	assert fn.copyToHost(merged) == fn.copyToHost(tokens)

	q = fn.zeros([1, 3, 2])
	flash = fn.flashAttentionCausal(q, q, q, 0.5)
	assert flash.shape() == [1, 3, 2]
	assert fn.copyToHost(flash) == [0.0] * 6


def test_moe_bindings(engine):
	fn = oa.FnMatrix
	probabilities = fn.fromFloats([0.2, 0.8, 0.7, 0.3], [2, 2])
	expertIndices = fn.fromInt32([1, 0], [2, 1])
	routeWeights = fn.moeRouteWeights(probabilities, expertIndices)
	assert routeWeights.shape() == [2, 1]
	assert fn.copyToHost(routeWeights) == [1.0, 1.0]

	plan = fn.moeExpertPlan(expertIndices, 2)
	tokens = fn.fromFloats([1.0, 2.0, 3.0, 4.0], [2, 2])
	packedTokens = fn.moeGather(tokens, plan.packedToken, plan.inverse)
	assert fn.copyToHost(packedTokens) == [3.0, 4.0, 1.0, 2.0]

	weight = fn.fromFloats([1.0, 0.0, 0.0, 1.0], [2, 1, 2])
	bias = fn.zeros([2, 1])
	grouped = fn.groupedLinearM(packedTokens, weight, bias, plan.offsets)
	groupedGemm = fn.groupedGemmM(packedTokens, weight, plan.offsets)
	assert fn.copyToHost(grouped) == [3.0, 2.0]
	assert fn.copyToHost(groupedGemm) == [3.0, 2.0]

	combined = fn.moeCombine(
		grouped, routeWeights, plan.inverse, plan.packedSlot
	)
	assert combined.shape() == [2, 1]
	assert fn.copyToHost(combined) == [2.0, 3.0]

	groupedGrad = fn.groupedGemmMBwd(
		fn.ones([2, 1]), packedTokens, weight, plan.offsets
	)
	assert isinstance(groupedGrad, oa.GroupedGemmMBwdResult)
	assert groupedGrad.dInput.shape() == [2, 2]
	assert groupedGrad.dWeight.shape() == [2, 1, 2]

	linearGrad = fn.groupedLinearMBwd(
		fn.ones([2, 1]), packedTokens, weight, plan.offsets
	)
	assert isinstance(linearGrad, oa.GroupedLinearMBwdResult)
	assert linearGrad.dBias.shape() == [2, 1]
	assert fn.groupedLinearMBiasBwd(
		fn.ones([2, 1]), plan.offsets, 2
	).shape() == [2, 1]

	combineGrad = fn.moeCombineBwd(
		fn.ones([2, 1]), grouped, routeWeights, plan.inverse, plan.packedSlot
	)
	assert isinstance(combineGrad, oa.MoeCombineBwdResult)
	assert combineGrad.dPacked.shape() == [2, 1]
	assert combineGrad.dRouteGate.shape() == [2, 1]
	assert fn.moeGatherBwd(packedTokens, plan.inverse, 2).shape() == [2, 2]
	assert fn.copyToHost(
		fn.scatterAddRows(tokens, plan.packedToken, 2)
	) == [3.0, 4.0, 1.0, 2.0]

def test_recurrent_cell_and_scan_bindings(engine):
	fn = oa.FnMatrix
	batch, sequenceLength, hiddenSize = 1, 2, 2
	hidden = fn.zeros([batch, hiddenSize])
	gradHidden = fn.ones([batch, hiddenSize])

	gruGatesI = fn.zeros([batch * sequenceLength, 3 * hiddenSize])
	gruGatesH = fn.zeros([batch, 3 * hiddenSize])
	gruWeight = fn.zeros([3 * hiddenSize, hiddenSize])
	gruBias = fn.zeros([3 * hiddenSize])

	gruPointwise = fn.gruCellPointwise(
		gruGatesI, gruGatesH, hidden, hiddenSize
	)
	assert gruPointwise.shape() == [batch, hiddenSize]
	assert fn.copyToHost(gruPointwise) == [0.0] * (batch * hiddenSize)

	gruPointwiseGrad = fn.gruCellPointwiseBwd(
		gruGatesI, gruGatesH, hidden, gradHidden, hiddenSize
	)
	assert isinstance(gruPointwiseGrad, oa.GruCellPointwiseBwdResult)
	assert gruPointwiseGrad.dGatesI.shape() == gruGatesI.shape()
	assert gruPointwiseGrad.dGatesH.shape() == gruGatesH.shape()
	assert gruPointwiseGrad.dHidden.shape() == hidden.shape()

	gruLinear = fn.gruCellLinear(
		gruGatesI, hidden, gruWeight, hiddenSize, 0, 1, gruBias
	)
	assert gruLinear.shape() == hidden.shape()

	# Omitted BiasHh exercises the bindable empty-matrix optional-input path.
	gruScan = fn.gruScan(
		gruGatesI, gruWeight, hiddenSize, sequenceLength, batch
	)
	assert isinstance(gruScan, oa.GruScanResult)
	assert gruScan.out.shape() == [batch, sequenceLength, hiddenSize]
	assert gruScan.hPrev.shape() == [batch, sequenceLength, hiddenSize]
	gruScanGrad = fn.gruScanBwd(
		fn.ones([batch, sequenceLength, hiddenSize]),
		gruGatesI,
		gruScan.hPrev,
		gruWeight,
		hiddenSize,
		sequenceLength,
		batch,
	)
	assert isinstance(gruScanGrad, oa.GruScanBwdResult)
	assert gruScanGrad.dGatesI.shape() == gruGatesI.shape()
	assert gruScanGrad.dGatesH.shape() == gruGatesI.shape()

	rnnGatesI = fn.zeros([batch * sequenceLength, hiddenSize])
	rnnCellGatesI = fn.zeros([batch, hiddenSize])
	rnnGatesH = fn.zeros([batch, hiddenSize])
	rnnWeight = fn.zeros([hiddenSize, hiddenSize])
	rnnBias = fn.zeros([hiddenSize])

	rnnPointwise = fn.rnnCellPointwise(rnnCellGatesI, rnnGatesH)
	assert rnnPointwise.shape() == hidden.shape()
	assert fn.copyToHost(rnnPointwise) == [0.0] * (batch * hiddenSize)
	rnnPointwiseGrad = fn.rnnCellPointwiseBwd(
		rnnGatesI, rnnGatesH, gradHidden, hiddenSize
	)
	assert isinstance(rnnPointwiseGrad, oa.RnnCellPointwiseBwdResult)
	assert rnnPointwiseGrad.dGatesI.shape() == rnnGatesI.shape()
	assert rnnPointwiseGrad.dGatesH.shape() == rnnGatesH.shape()

	rnnLinear = fn.rnnCellLinear(
		rnnGatesI, hidden, rnnWeight, 0, 1, rnnBias
	)
	assert rnnLinear.shape() == hidden.shape()
	rnnScan = fn.rnnScan(
		rnnGatesI, rnnWeight, hiddenSize, sequenceLength, batch
	)
	assert isinstance(rnnScan, oa.RnnScanResult)
	assert rnnScan.out.shape() == [batch, sequenceLength, hiddenSize]
	assert rnnScan.hPrev.shape() == [batch, sequenceLength, hiddenSize]
	rnnScanGrad = fn.rnnScanBwd(
		fn.ones([batch, sequenceLength, hiddenSize]),
		rnnGatesI,
		rnnScan.hPrev,
		rnnWeight,
		hiddenSize,
		sequenceLength,
		batch,
	)
	assert isinstance(rnnScanGrad, oa.RnnScanBwdResult)
	assert rnnScanGrad.dGatesI.shape() == rnnGatesI.shape()
	assert rnnScanGrad.dGatesH.shape() == rnnGatesI.shape()


def test_convolution_and_norm_schema_bindings(engine):
	fn = oa.FnMatrix

	conv1dInput = fn.zeros([1, 1, 4])
	conv1dWeight = fn.ones([1, 1, 2])
	conv1dBias = fn.zeros([1])
	conv1dGrad = fn.ones([1, 1, 3])

	assert fn.conv1dGemm(
		conv1dInput, conv1dWeight, conv1dBias
	).shape() == [1, 1, 3]
	assert fn.conv1dReluGemm(
		conv1dInput, conv1dWeight, conv1dBias
	).shape() == [1, 1, 3]
	columns = fn.im2Col1d(conv1dInput, 2, 1, 0)
	assert columns.shape() == [3, 2]
	assert fn.col2Im1d(
		columns, 1, 1, 4, 2, 1, 0, 1, 3
	).shape() == [1, 1, 4]
	assert fn.conv1dBwdData(
		conv1dGrad,
		conv1dWeight,
		1,
		0,
		1,
		oa.MatrixShape([1, 1, 4]),
	).shape() == [1, 1, 4]
	conv1dWeightGrad = fn.conv1dBwdWeight(
		conv1dInput, conv1dGrad, conv1dWeight, 1, 0, 1
	)
	assert isinstance(conv1dWeightGrad, oa.Conv1dBwdWeightResult)
	assert conv1dWeightGrad.gradWeight.shape() == conv1dWeight.shape()
	assert conv1dWeightGrad.gradBias.shape() == conv1dBias.shape()

	conv2dInput = fn.zeros([1, 1, 3, 3])
	conv2dWeight = fn.ones([1, 1, 2, 2])
	conv2dGrad = fn.ones([1, 1, 2, 2])
	convTransposeGrad = fn.ones([1, 1, 3, 3])
	assert fn.conv2dBwdData(
		conv2dGrad,
		conv2dWeight,
		1,
		0,
		oa.MatrixShape([1, 1, 3, 3]),
	).shape() == conv2dInput.shape()
	conv2dWeightGrad = fn.conv2dBwdWeight(
		conv2dInput, conv2dGrad, conv2dWeight, 1, 0
	)
	assert isinstance(conv2dWeightGrad, oa.Conv2dBwdWeightResult)
	assert conv2dWeightGrad.gradWeight.shape() == conv2dWeight.shape()
	assert conv2dWeightGrad.gradBias.shape() == conv1dBias.shape()

	convTranspose = fn.convTranspose2d(
		conv2dGrad, conv2dWeight, conv1dBias, 1, 0
	)
	assert convTranspose.shape() == convTransposeGrad.shape()
	assert fn.convTranspose2dBwdData(
		convTransposeGrad,
		conv2dWeight,
		1,
		0,
		oa.MatrixShape([1, 1, 2, 2]),
	).shape() == conv2dGrad.shape()
	convTransposeWeightGrad = fn.convTranspose2dBwdWeight(
		conv2dGrad, convTransposeGrad, conv2dWeight, 1, 0
	)
	assert isinstance(
		convTransposeWeightGrad,
		oa.ConvTranspose2dBwdWeightResult,
	)
	assert convTransposeWeightGrad.gradWeight.shape() == conv2dWeight.shape()
	assert convTransposeWeightGrad.gradBias.shape() == conv1dBias.shape()

	normInput = fn.zeros([2, 2])
	normWeight = fn.ones([2])
	normBias = fn.zeros([2])
	normGate = fn.zeros([2, 2])
	normGrad = fn.ones([2, 2])
	assert fn.rmsNormGated(
		normInput, normWeight, normBias, normGate, 1e-5, True
	).shape() == normInput.shape()
	gatedNormGrad = fn.rmsNormGatedBwd(
		normInput, normWeight, normBias, normGate, normGrad, 1e-5
	)
	assert isinstance(gatedNormGrad, oa.RmsNormGatedBwdResult)
	assert gatedNormGrad.dX.shape() == normInput.shape()
	assert gatedNormGrad.dWeight.shape() == normWeight.shape()
	assert gatedNormGrad.dBias.shape() == normBias.shape()
	assert gatedNormGrad.dZ.shape() == normGate.shape()

	channelInput = fn.zeros([1, 2, 2])
	channelWeight = fn.ones([2])
	channelBias = fn.zeros([2])
	channelGrad = fn.ones([1, 2, 2])
	channelNorm = fn.channelNorm(
		channelInput, channelWeight, channelBias, 1, 2, 2, 1e-5
	)
	assert channelNorm.shape() == channelInput.shape()
	channelNormGrad = fn.channelNormBwd(
		channelInput, channelWeight, channelGrad, 1, 2, 2, 1e-5
	)
	assert isinstance(channelNormGrad, oa.ChannelNormBwdResult)
	assert channelNormGrad.dx.shape() == channelInput.shape()
	assert channelNormGrad.dWeight.shape() == channelWeight.shape()
	assert channelNormGrad.dBias.shape() == channelBias.shape()
	channelNormRelu = fn.channelNormRelu(
		channelInput, channelWeight, channelBias, 1, 2, 2, 1e-5
	)
	assert channelNormRelu.shape() == channelInput.shape()
	channelNormReluGrad = fn.channelNormReluBwd(
		channelInput,
		channelWeight,
		channelNormRelu,
		channelGrad,
		1,
		2,
		2,
		1e-5,
	)
	assert isinstance(channelNormReluGrad, oa.ChannelNormBwdResult)
	assert channelNormReluGrad.dx.shape() == channelInput.shape()


def test_state_space_and_vq_schema_bindings(engine):
	fn = oa.FnMatrix

	scanConfig = oa.SsmConfig()
	scanConfig.batch = 1
	scanConfig.seqLen = 2
	scanConfig.nHeads = 1
	scanConfig.headDim = 2
	scanConfig.stateSize = 2
	scanConfig.numRopeAngles = 1
	scanConfig.hasZ = 1
	scanConfig.hasD = 1

	stepConfig = oa.SsmConfig()
	stepConfig.batch = 1
	stepConfig.seqLen = 1
	stepConfig.nHeads = 1
	stepConfig.headDim = 2
	stepConfig.stateSize = 2
	stepConfig.numRopeAngles = 1
	stepConfig.hasZ = 1
	stepConfig.hasD = 1

	preprocessConfig = oa.Mamba3PreprocessConfig()
	preprocessConfig.dInner = 2
	preprocessConfig.dState = 2
	preprocessConfig.nHeads = 1
	preprocessConfig.numRopeAngles = 1
	preprocessConfig.nGroups = 1
	preprocessConfig.mimoRank = 1
	preprocessConfig.eps = 1e-5
	preprocessConfig.dtMin = 1e-3
	preprocessConfig.dtMax = 1e-1
	preprocessConfig.aFloor = 1e-4

	c = fn.zeros([1, 2, 1, 2])
	b = fn.zeros([1, 2, 1, 2])
	x = fn.ones([1, 2, 1, 2])
	z = fn.zeros([1, 2, 1, 2])
	adt = fn.full([1, 2, 1], -0.01)
	dt = fn.full([1, 2, 1], 0.01)
	trap = fn.zeros([1, 2, 1])
	angle = fn.zeros([1, 2, 1])
	cBias = fn.zeros([1, 2])
	bBias = fn.zeros([1, 2])
	d = fn.zeros([1])
	dOut = fn.ones([1, 2, 1, 2])

	stepC = fn.zeros([1, 1, 1, 2])
	stepB = fn.zeros([1, 1, 1, 2])
	stepX = fn.ones([1, 1, 1, 2])
	stepZ = fn.zeros([1, 1, 1, 2])
	stepAdt = fn.full([1, 1, 1], -0.01)
	stepDt = fn.full([1, 1, 1], 0.01)
	stepTrap = fn.zeros([1, 1, 1])
	stepAngle = fn.zeros([1, 1, 1])
	ssmState = fn.zeros([1, 1, 2, 2])
	angleState = fn.zeros([1, 1, 1])
	kState = fn.zeros([1, 1, 2])
	vState = fn.zeros([1, 1, 2])

	projected = fn.ones([2, 12])
	dtBias = fn.zeros([1])
	dz = fn.ones([2, 2])
	dx = fn.ones([2, 2])
	dbh = fn.ones([2, 2])
	dch = fn.ones([2, 2])
	ddt = fn.ones([2, 1])
	dadt = fn.ones([2, 1])
	dtrap = fn.ones([2, 1])
	dangle = fn.ones([2, 1])

	for prefix in ("mamba3", "empyrealm"):
		siso = getattr(fn, f"{prefix}Siso")(
			c, b, x, z, adt, dt, trap, angle, cBias, bBias, d,
			scanConfig,
		)
		assert siso.shape() == [1, 2, 1, 2]
		step = getattr(fn, f"{prefix}SisoStep")(
			stepC, stepB, stepX, stepZ, stepAdt, stepDt,
			stepTrap, stepAngle, cBias, bBias, d, ssmState,
			angleState, kState, vState, stepConfig,
		)
		assert step.shape() == [1, 1, 1, 2]
		sisoGrad = getattr(fn, f"{prefix}SisoBwd")(
			dOut, c, b, x, z, adt, dt, trap, angle, cBias, bBias,
			d, scanConfig,
		)
		assert isinstance(sisoGrad, oa.SsmBwdResult)
		assert sisoGrad.dC.shape() == c.shape()
		assert sisoGrad.dD.shape() == d.shape()

		if prefix == "empyrealm":
			adtValue = fn.empyrealmAdt(dadt, ddt, 1e-4)
			assert adtValue.shape() == dadt.shape()
			adtGrad = fn.empyrealmAdtBwd(ddt, dadt, ddt, 1e-4)
			assert isinstance(adtGrad, oa.EmpyrealmAdtBwdResult)
			assert adtGrad.dDdA.shape() == dadt.shape()
			assert adtGrad.dDt.shape() == ddt.shape()

			dtValue = fn.empyrealmDt(ddt, 1e-3, 1e-1)
			assert dtValue.shape() == ddt.shape()
			dtGrad = fn.empyrealmDtBwd(ddt, ddt, 1e-3, 1e-1)
			assert dtGrad.shape() == ddt.shape()

		preprocess = getattr(fn, f"{prefix}Preprocess")(
			projected, dtBias, preprocessConfig
		)
		assert isinstance(preprocess, oa.Mamba3PreprocessResult)
		assert preprocess.x.shape() == [2, 2]
		assert preprocess.angle.shape() == [2, 1]
		preprocessGrad = getattr(fn, f"{prefix}PreprocessBwd")(
			projected, dtBias, dz, dx, dbh, dch, ddt, dadt,
			dtrap, dangle, preprocessConfig,
		)
		assert isinstance(
			preprocessGrad,
			oa.Mamba3PreprocessBwdResult,
		)
		assert preprocessGrad.dProjected.shape() == projected.shape()
		assert preprocessGrad.dDtBias.shape() == dtBias.shape()

	mimoConfig = oa.SsmConfig()
	mimoConfig.batch = 1
	mimoConfig.seqLen = 2
	mimoConfig.nHeads = 1
	mimoConfig.nGroups = 1
	mimoConfig.headDim = 2
	mimoConfig.stateSize = 2
	mimoConfig.numRopeAngles = 1
	mimoConfig.mimoRank = 2
	mimoConfig.hasZ = 1
	mimoConfig.hasD = 1
	mimoConfig.hasOutNorm = 1
	mimoC = fn.zeros([1, 2, 2, 2])
	mimoB = fn.zeros([1, 2, 2, 2])
	mimoCBias = fn.zeros([1, 2, 2])
	mimoBBias = fn.zeros([1, 2, 2])
	mimoX = fn.ones([1, 2, 2])
	mimoZ = fn.ones([1, 2, 2])
	mimoO = fn.ones([1, 2, 2])
	mimoNorm = fn.ones([1, 2])
	mimo = fn.mamba3Mimo(
		mimoC, mimoB, x, z, adt, dt, trap, angle,
		mimoCBias, mimoBBias, d, mimoX, mimoZ, mimoO,
		mimoNorm, mimoConfig,
	)
	assert mimo.shape() == [1, 2, 1, 2]
	mimoGrad = fn.mamba3MimoBwd(
		dOut, mimoC, mimoB, x, z, adt, dt, trap, angle,
		mimoCBias, mimoBBias, d, mimoX, mimoZ, mimoO,
		mimoNorm, mimoConfig,
	)
	assert isinstance(mimoGrad, oa.Mamba3MimoBwdResult)
	assert mimoGrad.dC.shape() == mimoC.shape()
	assert mimoGrad.dNormWeight.shape() == mimoNorm.shape()

	mimoStepConfig = oa.SsmConfig()
	mimoStepConfig.batch = 1
	mimoStepConfig.seqLen = 1
	mimoStepConfig.nHeads = 1
	mimoStepConfig.nGroups = 1
	mimoStepConfig.headDim = 2
	mimoStepConfig.stateSize = 2
	mimoStepConfig.numRopeAngles = 1
	mimoStepConfig.mimoRank = 2
	mimoStepConfig.hasZ = 1
	mimoStepConfig.hasD = 1
	mimoStepConfig.hasOutNorm = 1
	mimoStep = fn.mamba3MimoStep(
		fn.zeros([1, 1, 2, 2]), fn.zeros([1, 1, 2, 2]),
		stepX, stepZ, stepAdt, stepDt, stepTrap, stepAngle,
		mimoCBias, mimoBBias, d, mimoX, mimoZ, mimoO,
		mimoNorm, fn.zeros([1, 1, 2, 2]), fn.zeros([1, 1, 1]),
		fn.zeros([1, 1, 2, 2]), fn.zeros([1, 1, 2, 2]),
		mimoStepConfig,
	)
	assert mimoStep.shape() == [1, 1, 1, 2]

	dtAdt = fn.empyrealmDtAdt(ddt, dadt, 1e-3, 1e-1, 1e-4)
	assert isinstance(dtAdt, oa.EmpyrealmDtAdtResult)
	assert dtAdt.dt.shape() == ddt.shape()
	assert dtAdt.adt.shape() == dadt.shape()

	heavy = fn.heavyTailActivation(fn.zeros([2, 2]))
	assert heavy.shape() == [2, 2]
	ze = fn.ones([2, 2])
	codebook = fn.zeros([2, 2])
	assigned = fn.vqAssign(ze, codebook)
	assert isinstance(assigned, oa.VqAssignResult)
	assert assigned.idx.shape() == [2]
	assert assigned.zq.shape() == ze.shape()
	embedSum = fn.zeros([2, 2])
	clusterSize = fn.zeros([2])
	assert fn.vqEmaUpdate(
		ze, assigned.idx, embedSum, clusterSize, codebook,
		0.99, 1e-5, 1.0, 17, False,
	) is None


def test_gradient_tape_attaches_grad_fn(engine):
	_setGradEnabled(False)

	x = core.full(2, 3, 1.0)
	x.setRequiresGrad(True)

	linear = ml.Linear(3, 2)
	for param in linear.parameters():
		param.data.setRequiresGrad(True)

	tape = oa.GradientTape()
	y = linear.forward(x)

	assert y.hasGradFn()

	del tape
	_setGradEnabled(False)


def test_recurrent_modules_forward(engine):
	# ByteEmbedding / Rnn / Gru: the NLP-suite building blocks. Verify they
	# construct, expose params, and produce the expected all-position logits shape.
	import array

	b, s, d, h, vocab = 2, 4, 8, 16, 256
	embed = ml.ByteEmbedding(d)
	rnn = ml.Rnn(d, h, 1)
	gru = ml.Gru(d, h, 1)
	head = ml.Linear(h, vocab)

	assert len(embed.parameters()) == 1
	assert len(rnn.allParameterPtrs()) >= 1
	assert len(gru.allParameterPtrs()) >= 1

	ids = core.fromBytes(
		array.array("B", [(i * 7) % 256 for i in range(b * s)]),
		b,
		s,
		oa.ScalarType.UInt8,
	)
	for rec in (rnn, gru):
		e = embed.forward(ids).reshape([b, s, d])
		out = rec.forward(e).reshape([b * s, h])
		logits = head.forward(out)
		assert logits.shape() == [b * s, vocab]


def test_nlp_suite_modules_construct(engine):
	# The rest of the NLP-suite building blocks: attention + SSM. Verify they
	# construct and expose parameters (training parity is covered by the tutorials).
	d, ff, s, vocab = 8, 16, 4, 256
	ln = ml.LayerNorm(d)
	block = ml.TransformerBlock(d, ff, s)
	mamba = ml.Mamba3Module(d, dState=16, expand=2, headDim=8, outprojNorm=True)
	empyrealm = ml.EmpyrealmCore(vocab, d, dState=16, expand=2, headDim=8)

	assert len(ln.allParameterPtrs()) >= 1
	assert len(block.allParameterPtrs()) >= 1
	assert len(mamba.allParameterPtrs()) >= 1
	assert len(empyrealm.allParameterPtrs()) >= 1


def test_module_save_load_roundtrip(engine, tmp_path):
	# Module.save/Load persist parameters through a file.
	lin = ml.Linear(4, 3)
	for p in lin.parameters():
		p.data.setRequiresGrad(True)
	path = str(tmp_path / "linear.oam")
	lin.save(path)

	reloaded = ml.Linear(4, 3)
	reloaded.load(path)

	x = core.fromFloats([1.0, 2.0, 3.0, 4.0], 1, 4)
	a = lin.forward(x)
	b = reloaded.forward(x)
	assert core.copyToHost(a) == core.copyToHost(b)


if __name__ == "__main__":
	raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
