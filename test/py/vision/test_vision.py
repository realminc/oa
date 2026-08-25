#!/usr/bin/env python3
"""Contract and GPU tests for OA's root and :mod:`oa.FnImage` surfaces."""

from __future__ import annotations

import sys

import pytest


from oa_python_test import REPO_ROOT

import oa
core = oa.FnMatrix
runtime = oa
vision = oa.FnImage

IMAGE_OPS_50 = (
	"resize", "normalize", "gaussianBlur", "convolve2d",
	"separableConvolve2d", "averageBlur", "sobel", "scharr", "laplacian",
	"erode", "dilate", "morphologyOpen", "morphologyClose",
	"morphologyGradient", "crop", "flip", "rotate",
	"thresholdBinary", "thresholdBinaryInv", "thresholdTruncate",
	"thresholdToZero", "thresholdToZeroInv", "inRange", "clamp", "invert",
	"brightnessContrast", "gammaContrast", "solarize", "posterize",
	"grayscale", "channelReorder", "alphaBlend", "composite", "erase",
	"colorTwist", "gaussianNoise", "saltPepperNoise", "sharpen",
	"medianBlur", "bilateralFilter", "unsharpMask", "morphologyTopHat",
	"morphologyBlackHat", "adaptiveThresholdMean",
	"adaptiveThresholdGaussian", "pad", "centerCrop", "remap",
	"warpAffine", "warpPerspective",
)


@pytest.fixture(scope="session")
def engine():
	if not runtime.initComputeEngine():
		pytest.fail("GPU profile requires an initialized OA Vulkan engine")
	yield
	runtime.shutdownComputeEngine()


def test_vision_import_surface():
	for name in (
		"Image", "ImageBatch", "ImageLayout", "ImageFormat",
		"ImageCodec",
		"BorderMode", "NormalizationParams", "VideoDemuxer",
		"VideoPlayer", "VideoRecorder", "ScreenCapture", "CameraCapture",
		"NmsConfig", "NmsResult", "DetectionMetricsResult",
		"SegmentationMetricsResult",
	):
		assert hasattr(oa, name), name
	for name in (
		"decodeFile", "decodeMemory", "encode", "saveFile", "canDecode",
		"canEncode", "segmentationOverlay", *IMAGE_OPS_50,
	):
		assert hasattr(vision, name), name
	for name in (
		"boxIou", "nms", "confusionMatrix", "binaryMaskCounts", "evaluate",
		"evaluateSegmentation",
	):
		assert hasattr(oa.FnDetection, name), name
	assert len(IMAGE_OPS_50) == len(set(IMAGE_OPS_50)) == 50


def test_image_metadata_contract(engine):
	tensor = core.full([1, 3, 4, 5], 0.25)
	image = oa.Image(tensor, oa.ImageLayout.Nchw, oa.ImageFormat.Rgb)
	assert image.validate()
	assert image.batchSize() == 1
	assert image.channels() == 3
	assert image.height() == 4
	assert image.width() == 5
	assert image.asMatrix().shape() == [1, 3, 4, 5]


def test_still_image_codec_roundtrip(engine, tmp_path):
	asset = REPO_ROOT / "sdk" / "asset" / "image" / "visionTestPattern320x180.png"
	image = vision.decodeFile(asset)
	assert image.validate()
	assert image.asMatrix().shape() == [1, 3, 180, 320]
	assert image.format() == oa.ImageFormat.Rgb

	encoded = vision.encode(image, oa.ImageCodec.Png)
	assert encoded.startswith(b"\x89PNG\r\n\x1a\n")
	decoded = vision.decodeMemory(encoded)
	assert decoded.asMatrix().shape() == image.asMatrix().shape()

	output = tmp_path / "roundtrip.webp"
	if vision.canEncode(oa.ImageCodec.Webp):
		vision.saveFile(output, image, quality=92)
		assert output.read_bytes()[8:12] == b"WEBP"
		assert vision.decodeFile(output).width() == 320


def test_semantic_image_intro_pipeline(engine):
	asset = REPO_ROOT / "sdk" / "asset" / "image" / "visionTestPattern320x180.jpg"
	image = vision.decodeFile(asset)
	small = vision.resize(image, 16, 9)
	adjusted = vision.brightnessContrast(small, 0.05, 1.1)
	assert isinstance(adjusted, oa.Image)
	assert adjusted.asMatrix().shape() == [1, 3, 9, 16]
	assert len(core.copyToHost(adjusted.asMatrix())) == 3 * 9 * 16
	gray = vision.grayscale(small)
	assert isinstance(gray, oa.Image)
	assert gray.format() == oa.ImageFormat.Gray
	assert gray.asMatrix().shape() == [1, 1, 9, 16]
	assert len(core.copyToHost(gray.asMatrix())) == 9 * 16


def test_normalization_params_require_three_channels():
	params = oa.NormalizationParams()
	params.mean = [0.485, 0.456, 0.406]
	params.std = [0.229, 0.224, 0.225]
	assert len(params.mean) == 3 and len(params.std) == 3
	with pytest.raises(RuntimeError):
		params.mean = [0.5]


def test_detection_postprocess_and_metrics(engine):
	boxes = core.fromFloats(
		[0.50, 0.50, 0.40, 0.40,
		 0.51, 0.50, 0.40, 0.40,
		 0.50, 0.50, 0.40, 0.40],
		[3, 4],
	)
	scores = core.fromFloats([0.90, 0.80, 0.85], [3])
	classes = core.fromInt32([0, 0, 1], [3])
	config = oa.NmsConfig()
	config.iouThreshold = 0.5
	config.maxDetections = 3
	iou = oa.FnDetection.boxIou(boxes, boxes)
	selected = oa.FnDetection.nms(boxes, scores, classes, config)
	confusion = oa.FnDetection.confusionMatrix(
		core.fromInt32([0, 1, 2, 1], [4]),
		core.fromInt32([0, 2, 2, 1], [4]),
		3,
	)
	maskCounts = oa.FnDetection.binaryMaskCounts(
		core.fromBytes([1, 1, 0, 0], [4], oa.ScalarType.UInt8),
		core.fromBytes([1, 0, 1, 0], [4], oa.ScalarType.UInt8),
	)
	assert selected.isValid()
	assert iou.shape() == [3, 3]
	assert core.copyToHost(selected.count) == [2]
	assert core.copyToHost(selected.indices)[:2] == [0, 2]
	assert core.copyToHost(confusion) == [1, 0, 0, 0, 1, 0, 0, 1, 1]
	assert core.copyToHost(maskCounts) == [1, 1, 1, 1]


def test_dataset_detection_map(engine):
	predictedBoxes = core.fromFloats([
		0.20, 0.20, 0.20, 0.20,
		0.20, 0.20, 0.20, 0.20,
		0.50, 0.50, 0.20, 0.20,
		0.80, 0.80, 0.20, 0.20,
	], [4, 4])
	targetBoxes = core.fromFloats([
		0.20, 0.20, 0.20, 0.20,
		0.80, 0.80, 0.20, 0.20,
		0.50, 0.50, 0.20, 0.20,
	], [3, 4])
	metrics = oa.FnDetection.evaluate(
		predictedBoxes,
		core.fromFloats([0.90, 0.80, 0.70, 0.60], [4]),
		core.fromInt32([0, 0, 1, 0], [4]),
		core.fromInt32([0, 0, 0, 1], [4]),
		targetBoxes,
		core.fromInt32([0, 0, 1], [3]),
		core.fromInt32([0, 1, 0], [3]),
		core.fromFloats([0.50, 0.75], [2]),
		2,
		0.75,
	)
	assert metrics.isValid()
	assert metrics.counts.shape() == [2, 2, 3]
	assert core.copyToHost(metrics.counts) == [
		1, 1, 1, 0, 0, 1,
		1, 1, 1, 0, 0, 1,
	]
	assert core.copyToHost(metrics.meanAveragePrecisionByThreshold) == pytest.approx(
		[0.9174917, 0.9174917], abs=1.0e-5
	)
	assert core.copyToHost(metrics.meanAveragePrecision) == pytest.approx(
		[0.9174917], abs=1.0e-5
	)


def test_segmentation_metrics(engine):
	metrics = oa.FnDetection.evaluateSegmentation(
		core.fromInt32([0, 1, 2, 1], [2, 2]),
		core.fromInt32([0, 2, 2, 1], [2, 2]),
		3,
	)
	assert metrics.isValid()
	assert core.copyToHost(metrics.confusion) == [
		1, 0, 0,
		0, 1, 0,
		0, 1, 1,
	]
	assert core.copyToHost(metrics.meanIou) == pytest.approx(
		[2.0 / 3.0], abs=1.0e-6
	)
	assert core.copyToHost(metrics.pixelAccuracy) == pytest.approx(
		[0.75], abs=1.0e-6
	)
	overlay = vision.segmentationOverlay(
		core.fromFloats([0.0] * 6, [1, 3, 1, 2]),
		core.fromInt32([0, 1], [1, 1, 1, 2]),
		core.fromFloats([1, 0, 0, 0, 1, 0], [2, 3]),
		0.5,
	)
	assert core.copyToHost(overlay) == pytest.approx(
		[0.5, 0.0, 0.0, 0.5, 0.0, 0.0], abs=1.0e-6
	)


def test_geometric_ops_shapes(engine):
	image = core.rand([1, 3, 8, 10])
	resized = vision.resize(image, 5, 4)
	nearest = vision.resize(
		image, 20, 16, oa.InterpolationMode.Nearest
	)
	cropped = vision.crop(image, 2, 1, 5, 6)
	flipped = vision.flip(image, horizontal=True)
	rotated = vision.rotate(image, 90)
	assert resized.shape() == [1, 3, 4, 5]
	assert nearest.shape() == [1, 3, 16, 20]
	assert cropped.shape() == [1, 3, 6, 5]
	assert flipped.shape() == [1, 3, 8, 10]
	assert rotated.shape() == [1, 3, 10, 8]


def test_normalize_and_blur_execute(engine):
	image = core.full([1, 3, 8, 8], 0.5)
	params = oa.NormalizationParams()
	params.mean = [0.5, 0.5, 0.5]
	params.std = [0.25, 0.25, 0.25]
	normalized = vision.normalize(image, params)
	blurred = vision.gaussianBlur(image, 1.0, 3)
	assert normalized.shape() == image.shape()
	assert blurred.shape() == image.shape()
	normalizedValues = core.copyToHost(normalized)
	blurredValues = core.copyToHost(blurred)
	assert len(normalizedValues) == 3 * 8 * 8
	assert max(abs(value) for value in normalizedValues) < 1.0e-6
	assert max(abs(value - 0.5) for value in blurredValues) < 1.0e-5


def test_filter_primitives_and_derivatives_execute(engine):
	values = [float(i) for i in range(9)]
	image = core.fromFloats(values, [1, 1, 3, 3])
	identity = core.fromFloats(
		[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [3, 3]
	)
	kernelX = core.fromFloats([-1.0, 0.0, 1.0], [3])
	kernelY = core.fromFloats([1.0], [1])
	copied = vision.convolve2d(
		image, identity, oa.BorderMode.Constant
	)
	separated = vision.separableConvolve2d(
		image, kernelX, kernelY, oa.BorderMode.Replicate
	)
	averaged = vision.averageBlur(
		core.full([1, 1, 3, 3], 0.25), 3, 3
	)
	sobel = vision.sobel(image, 1, 0)
	scharr = vision.scharr(image, 1, 0)
	laplacian = vision.laplacian(image)
	assert core.copyToHost(copied) == pytest.approx(values, abs=1.0e-6)
	assert core.copyToHost(separated)[4] == pytest.approx(2.0, abs=1.0e-6)
	assert core.copyToHost(averaged) == pytest.approx([0.25] * 9, abs=1.0e-6)
	assert core.copyToHost(sobel)[4] == pytest.approx(8.0, abs=1.0e-6)
	assert core.copyToHost(scharr)[4] == pytest.approx(32.0, abs=1.0e-6)
	assert core.copyToHost(laplacian)[4] == pytest.approx(0.0, abs=1.0e-6)


def test_morphology_family_executes(engine):
	ramp = core.fromFloats([float(i) for i in range(9)], [1, 1, 3, 3])
	impulse = core.fromFloats(
		[0.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0], [1, 1, 3, 3]
	)
	hole = core.fromFloats(
		[1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0], [1, 1, 3, 3]
	)
	eroded = vision.erode(ramp)
	dilated = vision.dilate(ramp)
	gradient = vision.morphologyGradient(ramp)
	opened = vision.morphologyOpen(impulse)
	closed = vision.morphologyClose(hole)
	assert core.copyToHost(eroded)[4] == pytest.approx(0.0, abs=1.0e-6)
	assert core.copyToHost(dilated)[4] == pytest.approx(8.0, abs=1.0e-6)
	assert core.copyToHost(gradient)[4] == pytest.approx(8.0, abs=1.0e-6)
	assert core.copyToHost(opened) == pytest.approx([0.0] * 9, abs=1.0e-6)
	assert core.copyToHost(closed) == pytest.approx([1.0] * 9, abs=1.0e-6)


def test_new_pixel_neighborhood_and_warp_bindings_execute(engine):
	image = core.fromFloats([0.0, 0.25, 0.5, 0.75], [1, 1, 2, 2])
	identityMap = core.fromFloats(
		[0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0], [1, 2, 2, 2]
	)
	affine = core.fromFloats([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [2, 3])
	perspective = core.fromFloats(
		[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], [3, 3]
	)
	threshold = vision.thresholdBinary(image, 0.4)
	median = vision.medianBlur(image, 3, oa.BorderMode.Replicate)
	padded = vision.pad(image, 1, 1, 1, 1, oa.BorderMode.Constant, -1.0)
	remapped = vision.remap(
		image, identityMap, oa.InterpolationMode.Nearest
	)
	affineOut = vision.warpAffine(
		image, affine, 2, 2, oa.InterpolationMode.Nearest
	)
	perspectiveOut = vision.warpPerspective(
		image, perspective, 2, 2, oa.InterpolationMode.Nearest
	)
	assert core.copyToHost(threshold) == pytest.approx([0, 0, 1, 1])
	assert len(core.copyToHost(median)) == 4
	assert padded.shape() == [1, 1, 4, 4]
	assert core.copyToHost(remapped) == pytest.approx([0, 0.25, 0.5, 0.75])
	assert core.copyToHost(affineOut) == pytest.approx([0, 0.25, 0.5, 0.75])
	assert core.copyToHost(perspectiveOut) == pytest.approx([0, 0.25, 0.5, 0.75])


def test_video_configuration_is_host_only():
	cfg = oa.VideoPlayerConfig()
	cfg.uri = "sample.mp4"
	cfg.loop = False
	cfg.filter = oa.Filter.Nearest
	assert cfg.uri == "sample.mp4"
	assert not cfg.loop
	assert not hasattr(cfg, "Path")
	assert hasattr(oa.VideoPlayer, "open")
	assert hasattr(oa.VideoPlayer, "next")
	assert not hasattr(oa.VideoPlayer, "stepForward")


def test_video_conversion_stays_on_session_owner():
	assert hasattr(oa.VideoPlayer, "currentFrameToMatrix")
	assert hasattr(oa.VideoPlayer, "currentFrameToImage")
	assert not hasattr(oa.VideoFrame, "toMatrix")
	assert not hasattr(oa.VideoFrame, "toImage")


def test_video_sessions_expose_status_bearing_close_only():
	for sessionType in (
		oa.VideoDemuxer,
		oa.VideoPlayer,
		oa.VideoRecorder,
		oa.ScreenCapture,
		oa.CameraCapture,
	):
		assert hasattr(sessionType, "close")
		assert not hasattr(sessionType, "destroy")


if __name__ == "__main__":
	raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
