"""Install OA's C++-parity Python surface from the private native modules.

This is the checked exposure manifest for the first Python convergence
checkpoint. Native registration remains grouped by implementation domain, while
the public package exposes values at the root and C++ namespaces as real Python
modules. Compatibility domain modules remain available as explicit attributes,
but they are not wildcard exports.
"""

from __future__ import annotations

from importlib.machinery import ModuleSpec
from types import ModuleType
from typing import Any
import sys


# Fixed required root inventory. Each pair is (public name, native name).
_ROOT_EXPORTS: dict[str, tuple[tuple[str, str], ...]] = {
	"core": (
		("OaFilesystem", "OaFilesystem"),
		("OaLinearWeightBiasBwdResult", "OaLinearWeightBiasBwdResult"),
		("OaMatMulPrecision", "OaMatMulPrecision"),
		("OaMatrix", "OaMatrix"),
		("OaMatrixShape", "OaMatrixShape"),
		("OaPath", "OaPath"),
		("OaPaths", "OaPaths"),
		("OaScalarType", "OaScalarType"),
	),
	"runtime": (
		("OaContext", "OaContext"),
		("OaContextGetDefault", "OaContextGetDefault"),
		("OaInitComputeEngine", "OaInitComputeEngine"),
		("OaShutdownComputeEngine", "OaShutdownComputeEngine"),
	),
	"ml": (
		("OaActivation", "OaActivation"),
		("OaAdam", "OaAdam"),
		("OaAdamW", "OaAdamW"),
		("OaByteEmbedding", "OaByteEmbedding"),
		("OaCallback", "OaCallback"),
		("OaCbProgressBar", "OaCbProgressBar"),
		("OaCbSummary", "OaCbSummary"),
		("OaCbTraining", "OaCbTraining"),
		("OaDqnLossConfig", "OaDqnLossConfig"),
		("OaDqnLossResult", "OaDqnLossResult"),
		("OaEmbedding", "OaEmbedding"),
		("OaEmpyrealmCore", "OaEmpyrealmCore"),
		("OaGaeConfig", "OaGaeConfig"),
		("OaGaeResult", "OaGaeResult"),
		("OaGpuTimingStats", "OaGpuTimingStats"),
		("OaGradientTape", "GradientTape"),
		("OaGru", "OaGru"),
		("OaItRlTraining", "OaItRlTraining"),
		("OaItRlTrainingConfig", "OaItRlTrainingConfig"),
		("OaItTraining", "OaItTraining"),
		("OaItTrainingConfig", "OaItTrainingConfig"),
		("OaLayerNorm", "OaLayerNorm"),
		("OaLinear", "OaLinear"),
		("OaMamba3Module", "OaMamba3Module"),
		("OaMetric", "OaMetric"),
		("OaMetricAccuracy", "OaMetricAccuracy"),
		("OaMetricLoss", "OaMetricLoss"),
		("OaModule", "OaModule"),
		("OaMuon", "OaMuon"),
		("OaMuonAdamWConfig", "OaMuonAdamWConfig"),
		("OaNlpArchitecture", "OaNlpArchitecture"),
		("OaNlpSuiteBatchSize", "OaNlpSuiteBatchSize"),
		("OaNlpSuiteContextLength", "OaNlpSuiteContextLength"),
		("OaNlpSuiteGenerationPrompt", "OaNlpSuiteGenerationPrompt"),
		("OaNlpSuiteGenerationSourceUnits", "OaNlpSuiteGenerationSourceUnits"),
		("OaNlpSuiteHiddenWidth", "OaNlpSuiteHiddenWidth"),
		("OaNlpSuiteModel", "OaNlpSuiteModel"),
		("OaNlpSuiteModelWidth", "OaNlpSuiteModelWidth"),
		("OaNlpSuiteRecipe", "OaNlpSuiteRecipe"),
		("OaNlpSuiteRngSeed", "OaNlpSuiteRngSeed"),
		("OaNlpSuiteSampler", "OaNlpSuiteSampler"),
		("OaNlpSuiteTrainingSteps", "OaNlpSuiteTrainingSteps"),
		("OaNlpTokenizerKind", "OaNlpTokenizerKind"),
		("OaOptimizer", "OaOptimizer"),
		("OaOptimizerComposite", "OaOptimizerComposite"),
		("OaOptimizerNoOp", "OaOptimizerNoOp"),
		("OaParameter", "OaParameter"),
		("OaPpoLossConfig", "OaPpoLossConfig"),
		("OaPpoLossResult", "OaPpoLossResult"),
		("OaRlContinuousPolicyResult", "OaRlContinuousPolicyResult"),
		("OaRlEnvironmentSpec", "OaRlEnvironmentSpec"),
		("OaRlFieldSpec", "OaRlFieldSpec"),
		("OaRlPolicyResult", "OaRlPolicyResult"),
		("OaRlReplayBatch", "OaRlReplayBatch"),
		("OaRlReplayBuffer", "OaRlReplayBuffer"),
		("OaRlReplayConfig", "OaRlReplayConfig"),
		("OaRlRolloutBatch", "OaRlRolloutBatch"),
		("OaRlRolloutBuffer", "OaRlRolloutBuffer"),
		("OaRlRolloutConfig", "OaRlRolloutConfig"),
		("OaRlSpaceKind", "OaRlSpaceKind"),
		("OaRlTrainingPhase", "OaRlTrainingPhase"),
		("OaRnn", "OaRnn"),
		("OaSGD", "OaSGD"),
		("OaSacCriticLossResult", "OaSacCriticLossResult"),
		("OaSacLossConfig", "OaSacLossConfig"),
		("OaTrainingCommandDisposition", "OaTrainingCommandDisposition"),
		("OaTrainingCommandResult", "OaTrainingCommandResult"),
		("OaTrainingMetricSample", "OaTrainingMetricSample"),
		("OaTrainingSession", "OaTrainingSession"),
		("OaTrainingSnapshot", "OaTrainingSnapshot"),
		("OaTrainingState", "OaTrainingState"),
		("OaTransformerBlock", "OaTransformerBlock"),
	),
	"audio": (
		("OaAudio", "OaAudio"),
		("OaAudioDecoder", "OaAudioDecoder"),
		("OaAudioEncoder", "OaAudioEncoder"),
		("OaChannelLayout", "OaChannelLayout"),
		("OaChannelsForLayout", "OaChannelsForLayout"),
		("OaLayoutForChannels", "OaLayoutForChannels"),
		("OaMelConfig", "OaMelConfig"),
		("OaMfccConfig", "OaMfccConfig"),
		("OaNormalizeAudioConfig", "OaNormalizeAudioConfig"),
		("OaResampleConfig", "OaResampleConfig"),
		("OaStftConfig", "OaStftConfig"),
	),
	"vision": (
		("OaBorderMode", "OaBorderMode"),
		("OaCameraCapture", "OaCameraCapture"),
		("OaCameraCaptureConfig", "OaCameraCaptureConfig"),
		("OaContainerInfo", "OaContainerInfo"),
		("OaContainerKind", "OaContainerKind"),
		("OaDetectionMetricsResult", "OaDetectionMetricsResult"),
		("OaFilter", "OaFilter"),
		("OaImage", "OaImage"),
		("OaImageBatch", "OaImageBatch"),
		("OaImageCodec", "OaImageCodec"),
		("OaImageDecoder", "OaImageDecoder"),
		("OaImageEncoder", "OaImageEncoder"),
		("OaImageFormat", "OaImageFormat"),
		("OaImageFormatChannels", "OaImageFormatChannels"),
		("OaImageLayout", "OaImageLayout"),
		("OaInterpolationMode", "OaInterpolationMode"),
		("OaNmsConfig", "OaNmsConfig"),
		("OaNmsResult", "OaNmsResult"),
		("OaNormalizationParams", "OaNormalizationParams"),
		("OaPixelFormat", "OaPixelFormat"),
		("OaScreenCapture", "OaScreenCapture"),
		("OaScreenCaptureConfig", "OaScreenCaptureConfig"),
		("OaScreenCaptureCursor", "OaScreenCaptureCursor"),
		("OaScreenCaptureTarget", "OaScreenCaptureTarget"),
		("OaSegmentationMetricsResult", "OaSegmentationMetricsResult"),
		("OaVideo", "OaVideo"),
		("OaVideoCodec", "OaVideoCodec"),
		("OaVideoConfig", "OaVideoConfig"),
		("OaVideoDecodeCapabilities", "OaVideoDecodeCapabilities"),
		("OaVideoEncodeCapabilities", "OaVideoEncodeCapabilities"),
		("OaVideoEncodeProfile", "OaVideoEncodeProfile"),
		("OaVideoFrame", "OaVideoFrame"),
		("OaVideoFrameResource", "OaVideoFrameResource"),
		("OaVideoPacket", "OaVideoPacket"),
		("OaVideoRateControl", "OaVideoRateControl"),
		("OaVideoRecorder", "OaVideoRecorder"),
		("OaVideoRecorderConfig", "OaVideoRecorderConfig"),
		("OaVideoStream", "OaVideoStream"),
		("OaVideoStreamOptions", "OaVideoStreamOptions"),
		("OaVideoStreamStats", "OaVideoStreamStats"),
		("OaYCbCrModel", "OaYCbCrModel"),
	),
	"ui": (
		("OaViewer", "OaViewer"),
	),
}


_NAMESPACE_EXPORTS: dict[str, dict[str, tuple[tuple[str, str], ...]]] = {
	"OaFnMatrix": {
		"core": tuple(
			(name, name)
			for name in (
				"Add", "AddScalar", "Argmax", "Cast", "CausalMask",
				"CopyToHost", "CopyToHost2D", "Detach", "Div", "DivScalar",
				"Empty", "Exp", "FromBytes", "FromFloats", "FromInt32", "Full",
				"Gather", "GatherBwd", "Gelu", "GeluBwd", "GetWeightDtype",
				"LinearDataBwd", "LinearWeightBiasBwd", "Log", "LogSoftmax",
				"MatMulNt", "Max", "Mean", "Mul", "Ones", "Pow", "Rand",
				"RandGlorotUniform", "RandKaimingUniform", "RandN",
				"RandXavier", "Relu", "ReluBwd", "Reshape", "Scalar", "Scale",
				"SetRngSeed", "SetWeightDtype", "Sigmoid", "Silu", "SiluBwd",
				"Slice", "Softmax", "SoftmaxBwd", "Sqrt", "Sub", "SubScalar",
				"Sum", "Tanh", "TanhBwd", "Transpose", "Zeros",
			)
		),
	},
	"OaFnLoss": {
		"ml": tuple(
			(name, name)
			for name in (
				"Bce", "BceBwd", "CrossEntropy", "CrossEntropyBwd", "Dqn",
				"L1", "L1Bwd", "Mse", "MseBwd", "Ppo",
				"PpoClippedPolicy", "PpoClippedPolicyBwd", "SacActor",
				"SacCritic", "SmoothL1", "SmoothL1Bwd",
			)
		),
	},
	"OaFnAutograd": {
		"ml": (("IsEnabled", "IsEnabled"), ("SetEnabled", "SetEnabled")),
	},
	"OaFnMetric": {
		"ml": (
			("Accuracy", "MetricAccuracy"),
			("ScalarLoss", "MetricScalarLoss"),
		),
	},
	"OaFnRl": {
		"ml": tuple(
			(name, name)
			for name in (
				"ClipReward", "EvaluateCategoricalPolicy",
				"EvaluateTanhNormalPolicy", "Gae", "NormalizeAdvantages",
				"NormalizeObservation", "SampleCategoricalPolicy",
				"SampleTanhNormalPolicy", "ScaleAction",
			)
		),
	},
	"OaFnAudio": {
		"audio": tuple(
			(name, name)
			for name in (
				"AmplitudeToDb", "Clip", "Fade", "Gain", "MelSpectrogram",
				"Mfcc", "Mix", "Normalize", "PreEmphasis", "Resample", "Stft",
				"ToMono",
			)
		),
	},
	"OaFnImage": {
		"vision": tuple(
			(name, name)
			for name in (
				"AdaptiveThresholdGaussian", "AdaptiveThresholdMean",
				"AlphaBlend", "AverageBlur", "BilateralFilter",
				"BrightnessContrast", "CenterCrop", "ChannelReorder", "Clamp",
				"ColorTwist", "Composite", "ConvertColor", "Convolve2d", "Crop",
				"Dilate", "Erase", "Erode", "Flip", "GammaContrast",
				"GaussianBlur", "GaussianNoise", "Grayscale", "InRange",
				"Invert", "Laplacian", "MedianBlur", "MorphologyBlackHat",
				"MorphologyClose", "MorphologyGradient", "MorphologyOpen",
				"MorphologyTopHat", "Pad", "Posterize", "Remap", "Resize",
				"ResizeNormalize", "Rotate", "SaltPepperNoise", "Scharr",
				"SegmentationOverlay", "SeparableConvolve2d", "Sharpen",
				"Sobel", "Solarize", "ThresholdBinary", "ThresholdBinaryInv",
				"ThresholdToZero", "ThresholdToZeroInv", "ThresholdTruncate",
				"UnsharpMask", "WarpAffine", "WarpPerspective",
			)
		),
	},
	"OaFnDetection": {
		"vision": (
			("BinaryMaskCounts", "BinaryMaskCounts"),
			("BoxIou", "BoxIou"),
			("ConfusionMatrix", "ConfusionMatrix"),
			("Evaluate", "EvaluateDetections"),
			("EvaluateSegmentation", "EvaluateSegmentation"),
			("Nms", "Nms"),
		),
	},
}


_OPTIONAL_ROOT_EXPORTS: dict[str, tuple[tuple[str, str], ...]] = {
	"crypto": (
		("OaHash", "OaHash"),
		("OaHasher", "OaHasher"),
		("OaKeypair", "OaKeypair"),
		("OaPublicKey", "OaPublicKey"),
		("OaSecretKey", "OaSecretKey"),
		("OaSignature", "OaSignature"),
		("OaGenerateKeypair", "GenerateKeypair"),
		("OaSign", "Sign"),
		("OaVerify", "Verify"),
	),
}


_OPTIONAL_NAMESPACE_EXPORTS: dict[
	str, dict[str, tuple[tuple[str, str], ...]]
] = {
	"OaFnHash": {
		"crypto": tuple(
			(name, name)
			for name in ("KeccakF1600", "MerkleRoot", "Shake128", "Shake256")
		),
	},
}


_NESTED_EXPORTS: dict[str, tuple[str, tuple[str, ...]]] = {
	"OaPlot": (
		"plot",
		("Axes", "Figure", "FigureConfig", "HeatmapStyle"),
	),
}


def _require(source: ModuleType, source_name: str, public_name: str) -> Any:
	try:
		return getattr(source, source_name)
	except AttributeError as exc:
		raise ImportError(
			f"OA Python exposure manifest expects {source.__name__}.{source_name} "
			f"for public symbol {public_name}"
		) from exc


def _publish(
	target: dict[str, Any], public_name: str, value: Any, *, owner: str
) -> None:
	previous = target.get(public_name)
	if previous is not None and previous is not value:
		raise ImportError(
			f"OA Python public symbol collision for {public_name}: {owner} "
			"does not match the previously registered object"
		)
	target[public_name] = value


def _module(name: str, doc: str) -> ModuleType:
	module = ModuleType(name, doc)
	module.__package__ = "oa"
	module.__spec__ = ModuleSpec(name, loader=None)
	return module


def _set_public_owner(value: Any, owner: str, public_name: str) -> None:
	if isinstance(value, type):
		for attribute, replacement in (
			("__module__", owner),
			("__name__", public_name),
			("__qualname__", public_name),
		):
			try:
				setattr(value, attribute, replacement)
			except (AttributeError, TypeError):
				pass


def _install_namespace(
	package: dict[str, Any],
	sources: dict[str, ModuleType],
	public_name: str,
	exports: dict[str, tuple[tuple[str, str], ...]],
) -> None:
	qualified_name = f"oa.{public_name}"
	namespace = _module(qualified_name, f"Python view of C++ namespace {public_name}.")
	names: list[str] = []
	for source_key, mappings in exports.items():
		source = sources[source_key]
		for name, source_name in mappings:
			value = _require(source, source_name, f"{public_name}.{name}")
			_publish(namespace.__dict__, name, value, owner=source.__name__)
			names.append(name)
	namespace.__all__ = tuple(sorted(set(names)))
	sys.modules[qualified_name] = namespace
	_publish(package, public_name, namespace, owner=qualified_name)


def install_surface(
	package: dict[str, Any], sources: dict[str, ModuleType]
) -> tuple[str, ...]:
	"""Install and return the fixed wildcard-export inventory."""

	public_names: list[str] = []

	for source_key, mappings in _ROOT_EXPORTS.items():
		source = sources[source_key]
		for public_name, source_name in mappings:
			value = _require(source, source_name, public_name)
			_set_public_owner(value, "oa", public_name)
			_publish(package, public_name, value, owner=source.__name__)
			public_names.append(public_name)

	for public_name, exports in _NAMESPACE_EXPORTS.items():
		_install_namespace(package, sources, public_name, exports)
		public_names.append(public_name)

	for public_name, (source_key, names) in _NESTED_EXPORTS.items():
		source = sources[source_key]
		qualified_name = f"oa.{public_name}"
		namespace = _module(
			qualified_name, f"Python view of C++ namespace {public_name}."
		)
		for name in names:
			value = _require(source, name, f"{public_name}.{name}")
			_set_public_owner(value, qualified_name, name)
			_publish(namespace.__dict__, name, value, owner=source.__name__)
		namespace.__all__ = tuple(names)
		sys.modules[qualified_name] = namespace
		_publish(package, public_name, namespace, owner=qualified_name)
		public_names.append(public_name)

	crypto = sources.get("crypto")
	if crypto is not None and bool(getattr(crypto, "available", True)):
		for public_name, source_name in _OPTIONAL_ROOT_EXPORTS["crypto"]:
			value = _require(crypto, source_name, public_name)
			_set_public_owner(value, "oa", public_name)
			_publish(package, public_name, value, owner=crypto.__name__)
			public_names.append(public_name)
		for public_name, exports in _OPTIONAL_NAMESPACE_EXPORTS.items():
			_install_namespace(package, sources, public_name, exports)
			public_names.append(public_name)

	inventory = tuple(sorted(set(public_names)))
	invalid = [name for name in inventory if not name.startswith("Oa")]
	if invalid:
		raise ImportError(
			"OA wildcard exports must use the Oa prefix: " + ", ".join(invalid)
		)
	return inventory


__all__ = ["install_surface"]
