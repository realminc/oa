#!/usr/bin/env python3
"""Contract and GPU smoke tests for :mod:`oa.vision`."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _add_dev_paths() -> None:
	candidates = []
	if build_dir := os.getenv("OA_PYTHON_BUILD_DIR"):
		candidates.append(Path(build_dir).expanduser())
	candidates.extend(
		[REPO_ROOT / "Build" / "Release", REPO_ROOT / "Build" / "Debug",
		 REPO_ROOT / "build", REPO_ROOT / "Source" / "Python"]
	)
	for path in candidates:
		if path.exists() and str(path) not in sys.path:
			sys.path.insert(0, str(path))


_add_dev_paths()

oa = pytest.importorskip("oa", reason="oa Python package is not importable")
core = oa.core
runtime = oa.runtime
vision = oa.vision

IMAGE_OPS_50 = (
	"Resize", "Normalize", "GaussianBlur", "Convolve2d",
	"SeparableConvolve2d", "AverageBlur", "Sobel", "Scharr", "Laplacian",
	"Erode", "Dilate", "MorphologyOpen", "MorphologyClose",
	"MorphologyGradient", "Crop", "Flip", "Rotate",
	"ThresholdBinary", "ThresholdBinaryInv", "ThresholdTruncate",
	"ThresholdToZero", "ThresholdToZeroInv", "InRange", "Clamp", "Invert",
	"BrightnessContrast", "GammaContrast", "Solarize", "Posterize",
	"Grayscale", "ChannelReorder", "AlphaBlend", "Composite", "Erase",
	"ColorTwist", "GaussianNoise", "SaltPepperNoise", "Sharpen",
	"MedianBlur", "BilateralFilter", "UnsharpMask", "MorphologyTopHat",
	"MorphologyBlackHat", "AdaptiveThresholdMean",
	"AdaptiveThresholdGaussian", "Pad", "CenterCrop", "Remap",
	"WarpAffine", "WarpPerspective",
)


@pytest.fixture(scope="session")
def engine():
	if not runtime.OaInitComputeEngine():
		pytest.skip("OA compute engine could not initialize, likely no Vulkan device")
	yield
	runtime.OaShutdownComputeEngine()


def test_vision_import_surface():
	for name in (
		"OaImage", "OaImageBatch", "OaImageLayout", "OaImageFormat",
		"OaImageCodec", "OaImageDecoder", "OaImageEncoder",
		"OaBorderMode", "OaNormalizationParams", "OaVideoStream",
		"OaVideo", "OaVideoRecorder", "OaScreenCapture", "OaCameraCapture",
		"OaNmsConfig", "OaNmsResult", "OaDetectionMetricsResult",
		"OaSegmentationMetricsResult",
		"BoxIou", "Nms", "ConfusionMatrix", "BinaryMaskCounts",
		"EvaluateDetections", "EvaluateSegmentation", "SegmentationOverlay",
		*IMAGE_OPS_50,
	):
		assert hasattr(vision, name), name
	assert len(IMAGE_OPS_50) == len(set(IMAGE_OPS_50)) == 50


def test_image_metadata_contract(engine):
	tensor = core.Full([1, 3, 4, 5], 0.25)
	image = vision.OaImage(tensor, vision.OaImageLayout.Nchw, vision.OaImageFormat.Rgb)
	assert image.Validate()
	assert image.BatchSize() == 1
	assert image.Channels() == 3
	assert image.Height() == 4
	assert image.Width() == 5
	assert image.AsMatrix().Shape() == [1, 3, 4, 5]


def test_still_image_codec_roundtrip(engine, tmp_path):
	asset = REPO_ROOT / "Asset" / "Image" / "VisionTestPattern320x180.png"
	image = vision.OaImageDecoder.LoadFile(asset)
	assert image.Validate()
	assert image.AsMatrix().Shape() == [1, 3, 180, 320]
	assert image.Format() == vision.OaImageFormat.Rgb

	encoded = vision.OaImageEncoder.Encode(image, vision.OaImageCodec.Png)
	assert encoded.startswith(b"\x89PNG\r\n\x1a\n")
	decoded = vision.OaImageDecoder.LoadMemory(encoded)
	assert decoded.AsMatrix().Shape() == image.AsMatrix().Shape()

	output = tmp_path / "roundtrip.webp"
	if vision.OaImageEncoder.Supports(vision.OaImageCodec.Webp):
		vision.OaImageEncoder.SaveFile(output, image, Quality=92)
		assert output.read_bytes()[8:12] == b"WEBP"
		assert vision.OaImageDecoder.LoadFile(output).Width() == 320


def test_semantic_image_intro_pipeline(engine):
	asset = REPO_ROOT / "Asset" / "Image" / "VisionTestPattern320x180.jpg"
	image = vision.OaImageDecoder.LoadFile(asset)
	small = vision.Resize(image, 16, 9)
	adjusted = vision.BrightnessContrast(small, 0.05, 1.1)
	assert isinstance(adjusted, vision.OaImage)
	assert adjusted.AsMatrix().Shape() == [1, 3, 9, 16]
	assert len(core.CopyToHost(adjusted.AsMatrix())) == 3 * 9 * 16


def test_normalization_params_require_three_channels():
	params = vision.OaNormalizationParams()
	params.Mean = [0.485, 0.456, 0.406]
	params.Std = [0.229, 0.224, 0.225]
	assert len(params.Mean) == 3 and len(params.Std) == 3
	with pytest.raises(RuntimeError):
		params.Mean = [0.5]


def test_detection_postprocess_and_metrics(engine):
	boxes = core.FromFloats(
		[0.50, 0.50, 0.40, 0.40,
		 0.51, 0.50, 0.40, 0.40,
		 0.50, 0.50, 0.40, 0.40],
		[3, 4],
	)
	scores = core.FromFloats([0.90, 0.80, 0.85], [3])
	classes = core.FromInt32([0, 0, 1], [3])
	config = vision.OaNmsConfig()
	config.IouThreshold = 0.5
	config.MaxDetections = 3
	with oa.Context():
		iou = vision.BoxIou(boxes, boxes)
		selected = vision.Nms(boxes, scores, classes, config)
		confusion = vision.ConfusionMatrix(
			core.FromInt32([0, 1, 2, 1], [4]),
			core.FromInt32([0, 2, 2, 1], [4]),
			3,
		)
		mask_counts = vision.BinaryMaskCounts(
			core.FromBytes([1, 1, 0, 0], [4], core.OaScalarType.UInt8),
			core.FromBytes([1, 0, 1, 0], [4], core.OaScalarType.UInt8),
		)
	assert selected.IsValid()
	assert iou.Shape() == [3, 3]
	assert core.CopyToHost(selected.Count) == [2]
	assert core.CopyToHost(selected.Indices)[:2] == [0, 2]
	assert core.CopyToHost(confusion) == [1, 0, 0, 0, 1, 0, 0, 1, 1]
	assert core.CopyToHost(mask_counts) == [1, 1, 1, 1]


def test_dataset_detection_map(engine):
	predicted_boxes = core.FromFloats([
		0.20, 0.20, 0.20, 0.20,
		0.20, 0.20, 0.20, 0.20,
		0.50, 0.50, 0.20, 0.20,
		0.80, 0.80, 0.20, 0.20,
	], [4, 4])
	target_boxes = core.FromFloats([
		0.20, 0.20, 0.20, 0.20,
		0.80, 0.80, 0.20, 0.20,
		0.50, 0.50, 0.20, 0.20,
	], [3, 4])
	with oa.Context():
		metrics = vision.EvaluateDetections(
			predicted_boxes,
			core.FromFloats([0.90, 0.80, 0.70, 0.60], [4]),
			core.FromInt32([0, 0, 1, 0], [4]),
			core.FromInt32([0, 0, 0, 1], [4]),
			target_boxes,
			core.FromInt32([0, 0, 1], [3]),
			core.FromInt32([0, 1, 0], [3]),
			core.FromFloats([0.50, 0.75], [2]),
			2,
			0.75,
		)
	assert metrics.IsValid()
	assert metrics.Counts.Shape() == [2, 2, 3]
	assert core.CopyToHost(metrics.Counts) == [
		1, 1, 1, 0, 0, 1,
		1, 1, 1, 0, 0, 1,
	]
	assert core.CopyToHost(metrics.MeanAveragePrecisionByThreshold) == pytest.approx(
		[0.9174917, 0.9174917], abs=1.0e-5
	)
	assert core.CopyToHost(metrics.MeanAveragePrecision) == pytest.approx(
		[0.9174917], abs=1.0e-5
	)


def test_segmentation_metrics(engine):
	with oa.Context():
		metrics = vision.EvaluateSegmentation(
			core.FromInt32([0, 1, 2, 1], [2, 2]),
			core.FromInt32([0, 2, 2, 1], [2, 2]),
			3,
		)
	assert metrics.IsValid()
	assert core.CopyToHost(metrics.Confusion) == [
		1, 0, 0,
		0, 1, 0,
		0, 1, 1,
	]
	assert core.CopyToHost(metrics.MeanIou) == pytest.approx(
		[2.0 / 3.0], abs=1.0e-6
	)
	assert core.CopyToHost(metrics.PixelAccuracy) == pytest.approx(
		[0.75], abs=1.0e-6
	)
	with oa.Context():
		overlay = vision.SegmentationOverlay(
			core.FromFloats([0.0] * 6, [1, 3, 1, 2]),
			core.FromInt32([0, 1], [1, 1, 1, 2]),
			core.FromFloats([1, 0, 0, 0, 1, 0], [2, 3]),
			0.5,
		)
	assert core.CopyToHost(overlay) == pytest.approx(
		[0.5, 0.0, 0.0, 0.5, 0.0, 0.0], abs=1.0e-6
	)


def test_geometric_ops_shapes(engine):
	image = core.Rand([1, 3, 8, 10])
	with oa.Context():
		resized = vision.Resize(image, 5, 4)
		nearest = vision.Resize(
			image, 20, 16, vision.OaInterpolationMode.Nearest
		)
		cropped = vision.Crop(image, 2, 1, 5, 6)
		flipped = vision.Flip(image, Horizontal=True)
		rotated = vision.Rotate(image, 90)
	assert resized.Shape() == [1, 3, 4, 5]
	assert nearest.Shape() == [1, 3, 16, 20]
	assert cropped.Shape() == [1, 3, 6, 5]
	assert flipped.Shape() == [1, 3, 8, 10]
	assert rotated.Shape() == [1, 3, 10, 8]


def test_normalize_and_blur_execute(engine):
	image = core.Full([1, 3, 8, 8], 0.5)
	params = vision.OaNormalizationParams()
	params.Mean = [0.5, 0.5, 0.5]
	params.Std = [0.25, 0.25, 0.25]
	with oa.Context():
		normalized = vision.Normalize(image, params)
		blurred = vision.GaussianBlur(image, 1.0, 3)
	assert normalized.Shape() == image.Shape()
	assert blurred.Shape() == image.Shape()
	normalized_values = core.CopyToHost(normalized)
	blurred_values = core.CopyToHost(blurred)
	assert len(normalized_values) == 3 * 8 * 8
	assert max(abs(value) for value in normalized_values) < 1.0e-6
	assert max(abs(value - 0.5) for value in blurred_values) < 1.0e-5


def test_filter_primitives_and_derivatives_execute(engine):
	values = [float(i) for i in range(9)]
	image = core.FromFloats(values, [1, 1, 3, 3])
	identity = core.FromFloats(
		[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [3, 3]
	)
	kernel_x = core.FromFloats([-1.0, 0.0, 1.0], [3])
	kernel_y = core.FromFloats([1.0], [1])
	with oa.Context():
		copied = vision.Convolve2d(
			image, identity, vision.OaBorderMode.Constant
		)
		separated = vision.SeparableConvolve2d(
			image, kernel_x, kernel_y, vision.OaBorderMode.Replicate
		)
		averaged = vision.AverageBlur(
			core.Full([1, 1, 3, 3], 0.25), 3, 3
		)
		sobel = vision.Sobel(image, 1, 0)
		scharr = vision.Scharr(image, 1, 0)
		laplacian = vision.Laplacian(image)
	assert core.CopyToHost(copied) == pytest.approx(values, abs=1.0e-6)
	assert core.CopyToHost(separated)[4] == pytest.approx(2.0, abs=1.0e-6)
	assert core.CopyToHost(averaged) == pytest.approx([0.25] * 9, abs=1.0e-6)
	assert core.CopyToHost(sobel)[4] == pytest.approx(8.0, abs=1.0e-6)
	assert core.CopyToHost(scharr)[4] == pytest.approx(32.0, abs=1.0e-6)
	assert core.CopyToHost(laplacian)[4] == pytest.approx(0.0, abs=1.0e-6)


def test_morphology_family_executes(engine):
	ramp = core.FromFloats([float(i) for i in range(9)], [1, 1, 3, 3])
	impulse = core.FromFloats(
		[0.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0], [1, 1, 3, 3]
	)
	hole = core.FromFloats(
		[1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0], [1, 1, 3, 3]
	)
	with oa.Context():
		eroded = vision.Erode(ramp)
		dilated = vision.Dilate(ramp)
		gradient = vision.MorphologyGradient(ramp)
		opened = vision.MorphologyOpen(impulse)
		closed = vision.MorphologyClose(hole)
	assert core.CopyToHost(eroded)[4] == pytest.approx(0.0, abs=1.0e-6)
	assert core.CopyToHost(dilated)[4] == pytest.approx(8.0, abs=1.0e-6)
	assert core.CopyToHost(gradient)[4] == pytest.approx(8.0, abs=1.0e-6)
	assert core.CopyToHost(opened) == pytest.approx([0.0] * 9, abs=1.0e-6)
	assert core.CopyToHost(closed) == pytest.approx([1.0] * 9, abs=1.0e-6)


def test_new_pixel_neighborhood_and_warp_bindings_execute(engine):
	image = core.FromFloats([0.0, 0.25, 0.5, 0.75], [1, 1, 2, 2])
	identity_map = core.FromFloats(
		[0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0], [1, 2, 2, 2]
	)
	affine = core.FromFloats([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [2, 3])
	perspective = core.FromFloats(
		[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], [3, 3]
	)
	with oa.Context():
		threshold = vision.ThresholdBinary(image, 0.4)
		median = vision.MedianBlur(image, 3, vision.OaBorderMode.Replicate)
		padded = vision.Pad(image, 1, 1, 1, 1, vision.OaBorderMode.Constant, -1.0)
		remapped = vision.Remap(
			image, identity_map, vision.OaInterpolationMode.Nearest
		)
		affine_out = vision.WarpAffine(
			image, affine, 2, 2, vision.OaInterpolationMode.Nearest
		)
		perspective_out = vision.WarpPerspective(
			image, perspective, 2, 2, vision.OaInterpolationMode.Nearest
		)
	assert core.CopyToHost(threshold) == pytest.approx([0, 0, 1, 1])
	assert len(core.CopyToHost(median)) == 4
	assert padded.Shape() == [1, 1, 4, 4]
	assert core.CopyToHost(remapped) == pytest.approx([0, 0.25, 0.5, 0.75])
	assert core.CopyToHost(affine_out) == pytest.approx([0, 0.25, 0.5, 0.75])
	assert core.CopyToHost(perspective_out) == pytest.approx([0, 0.25, 0.5, 0.75])


def test_video_configuration_is_host_only():
	cfg = vision.OaVideoConfig()
	cfg.Uri = "sample.mp4"
	cfg.Loop = False
	cfg.Filter = vision.OaFilter.Nearest
	assert cfg.Uri == "sample.mp4"
	assert not cfg.Loop


def test_video_conversion_stays_on_session_owner():
	assert hasattr(vision.OaVideo, "CurrentFrameToMatrix")
	assert hasattr(vision.OaVideo, "CurrentFrameToImage")
	assert not hasattr(vision.OaVideoFrame, "ToMatrix")
	assert not hasattr(vision.OaVideoFrame, "ToImage")


if __name__ == "__main__":
	raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
