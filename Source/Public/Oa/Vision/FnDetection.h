// GPU-native object-detection postprocess and evaluation primitives.

#pragma once

#include <Oa/Core/Matrix.h>
#include <Oa/Core/Types.h>

struct OaNmsConfig {
	OaF32 IouThreshold = 0.45F;
	OaF32 ScoreThreshold = 0.0F;
	OaI32 MaxDetections = 100;
	bool ClassAgnostic = false;
};

struct OaNmsResult {
	// Int32 [N]. The first Count entries are selected source-row indices;
	// remaining entries are -1. Keeping a fixed shape makes the result graphable.
	OaMatrix Indices;
	// UInt32 [1], resident on the GPU until the caller explicitly reads it.
	OaMatrix Count;

	[[nodiscard]] bool IsValid() const noexcept {
		return Indices.HasStorage() && Count.HasStorage();
	}
};

struct OaDetectionMetricsResult {
	// UInt32 [T,C,3]: true positives, false positives and false negatives at
	// ScoreThreshold for every IoU threshold and class.
	OaMatrix Counts;
	// FP32 [T,C,4]: precision, recall, F1 and 101-point interpolated AP.
	OaMatrix PerClass;
	// FP32 [T]: mean AP across classes that contain at least one target.
	OaMatrix MeanAveragePrecisionByThreshold;
	// FP32 [1]: mean of Map across all supplied IoU thresholds.
	OaMatrix MeanAveragePrecision;

	[[nodiscard]] bool IsValid() const noexcept {
		return Counts.HasStorage()
			&& PerClass.HasStorage()
			&& MeanAveragePrecisionByThreshold.HasStorage()
			&& MeanAveragePrecision.HasStorage();
	}
};

struct OaSegmentationMetricsResult {
	// UInt32 [C,C], rows are target classes and columns are predictions.
	OaMatrix Confusion;
	// FP32 [C,4]: precision, recall, F1/Dice and intersection-over-union.
	OaMatrix PerClass;
	// FP32 [1] each. Classes with zero union do not contribute to MeanIou.
	OaMatrix MeanIou;
	OaMatrix PixelAccuracy;

	[[nodiscard]] bool IsValid() const noexcept {
		return Confusion.HasStorage()
			&& PerClass.HasStorage()
			&& MeanIou.HasStorage()
			&& PixelAccuracy.HasStorage();
	}
};

namespace OaFnDetection {
	// Pairwise IoU for FP32 center-x/center-y/width/height boxes.
	// InA [N,4], InB [M,4] -> FP32 [N,M]. Coordinates may be normalized or
	// pixel-valued, but widths and heights must be non-negative.
	[[nodiscard]] OaMatrix BoxIou(const OaMatrix& InA, const OaMatrix& InB);

	// Deterministic class-aware NMS. Boxes are FP32 [N,4] cx/cy/w/h, scores
	// FP32 [N], and classes Int32 [N]. The implementation is GPU-resident and
	// records as one graph node; it never sorts or compacts on the host.
	[[nodiscard]] OaNmsResult Nms(
		const OaMatrix& InBoxes,
		const OaMatrix& InScores,
		const OaMatrix& InClasses,
		const OaNmsConfig& InConfig = {}
	);

	// Classification confusion matrix. Rows are target classes and columns are
	// predicted classes: Int32 [N], Int32 [N] -> UInt32 [C,C].
	[[nodiscard]] OaMatrix ConfusionMatrix(
		const OaMatrix& InPredicted,
		const OaMatrix& InTarget,
		OaI32 InClassCount
	);

	// Binary mask counts [true-positive, false-positive, false-negative,
	// true-negative]. Inputs are equally-shaped UInt8 masks.
	[[nodiscard]] OaMatrix BinaryMaskCounts(
		const OaMatrix& InPredicted,
		const OaMatrix& InTarget
	);

	// Dataset-level object-detection evaluation. Predictions are FP32 boxes
	// [P,4], FP32 scores [P], Int32 classes [P] and Int32 image IDs [P].
	// Targets are FP32 boxes [G,4], Int32 classes [G] and Int32 image IDs [G].
	// IoU thresholds are FP32 [T]. Matching is greedy by descending score,
	// class-aware and constrained to the same image. Classes without targets do
	// not contribute to mAP. All outputs and internal scratch stay GPU-resident.
	[[nodiscard]] OaDetectionMetricsResult Evaluate(
		const OaMatrix& InPredictedBoxes,
		const OaMatrix& InPredictedScores,
		const OaMatrix& InPredictedClasses,
		const OaMatrix& InPredictedImageIds,
		const OaMatrix& InTargetBoxes,
		const OaMatrix& InTargetClasses,
		const OaMatrix& InTargetImageIds,
		const OaMatrix& InIouThresholds,
		OaI32 InClassCount,
		OaF32 InScoreThreshold = 0.0F
	);

	// Multiclass semantic-segmentation metrics over flattened Int32 label
	// matrices. Labels outside [0,ClassCount) are ignored by the confusion
	// accumulator, which provides the ordinary ignore-index behavior.
	[[nodiscard]] OaSegmentationMetricsResult EvaluateSegmentation(
		const OaMatrix& InPredicted,
		const OaMatrix& InTarget,
		OaI32 InClassCount
	);
}
