// GPU-native object-detection postprocess and evaluation primitives.

#pragma once

#include <oa/core/matrix.h>
#include <oa/core/types.h>

namespace oa {

struct NmsConfig {
	oa::F32 iouThreshold = 0.45F;
	oa::F32 scoreThreshold = 0.0F;
	oa::I32 maxDetections = 100;
	bool classAgnostic = false;
};

struct NmsResult {
	// Int32 [N]. The first count entries are selected source-row indices;
	// remaining entries are -1. Keeping a fixed shape makes the result graphable.
	Matrix indices;
	// UInt32 [1], resident on the GPU until the caller explicitly reads it.
	Matrix count;

	[[nodiscard]] bool isValid() const noexcept {
		return indices.hasStorage() && count.hasStorage();
	}
};

struct DetectionMetricsResult {
	// UInt32 [T,C,3]: true positives, false positives and false negatives at
	// scoreThreshold for every IoU threshold and class.
	Matrix counts;
	// FP32 [T,C,4]: precision, recall, F1 and 101-point interpolated AP.
	Matrix perClass;
	// FP32 [T]: mean AP across classes that contain at least one target.
	Matrix meanAveragePrecisionByThreshold;
	// FP32 [1]: mean of mAP across all supplied IoU thresholds.
	Matrix meanAveragePrecision;

	[[nodiscard]] bool isValid() const noexcept {
		return counts.hasStorage()
			&& perClass.hasStorage()
			&& meanAveragePrecisionByThreshold.hasStorage()
			&& meanAveragePrecision.hasStorage();
	}
};

struct SegmentationMetricsResult {
	// UInt32 [C,C], rows are target classes and columns are predictions.
	Matrix confusion;
	// FP32 [C,4]: precision, recall, F1/Dice and intersection-over-union.
	Matrix perClass;
	// FP32 [1] each. classes with zero union do not contribute to meanIou.
	Matrix meanIou;
	Matrix pixelAccuracy;

	[[nodiscard]] bool isValid() const noexcept {
		return confusion.hasStorage()
			&& perClass.hasStorage()
			&& meanIou.hasStorage()
			&& pixelAccuracy.hasStorage();
	}
};

namespace FnDetection {
	// Pairwise IoU for FP32 center-x/center-y/width/height boxes.
	// inA [N,4], inB [M,4] -> FP32 [N,M]. Coordinates may be normalized or
	// pixel-valued, but widths and heights must be non-negative.
	[[nodiscard]] Matrix boxIou(const Matrix& inA, const Matrix& inB);

	// Deterministic class-aware NMS. boxes are FP32 [N,4] cx/cy/w/h, scores
	// FP32 [N], and classes Int32 [N]. The implementation is GPU-resident and
	// records as one graph node; it never sorts or compacts on the host.
	[[nodiscard]] NmsResult nms(
		const Matrix& inBoxes,
		const Matrix& inScores,
		const Matrix& inClasses,
		const NmsConfig& inConfig = {}
	);

	// Classification confusion matrix. rows are target classes and columns are
	// predicted classes: Int32 [N], Int32 [N] -> UInt32 [C,C].
	[[nodiscard]] Matrix confusionMatrix(
		const Matrix& inPredicted,
		const Matrix& inTarget,
		oa::I32 inClassCount
	);

	// Binary mask counts [true-positive, false-positive, false-negative,
	// true-negative]. Inputs are equally-shaped UInt8 masks.
	[[nodiscard]] Matrix binaryMaskCounts(
		const Matrix& inPredicted,
		const Matrix& inTarget
	);

	// Dataset-level object-detection evaluation. Predictions are FP32 boxes
	// [P,4], FP32 scores [P], Int32 classes [P] and Int32 image IDs [P].
	// targets are FP32 boxes [G,4], Int32 classes [G] and Int32 image IDs [G].
	// IoU thresholds are FP32 [T]. Matching is greedy by descending score,
	// class-aware and constrained to the same image. classes without targets do
	// not contribute to mAP. All outputs and internal scratch stay GPU-resident.
	[[nodiscard]] DetectionMetricsResult evaluate(
		const Matrix& inPredictedBoxes,
		const Matrix& inPredictedScores,
		const Matrix& inPredictedClasses,
		const Matrix& inPredictedImageIds,
		const Matrix& inTargetBoxes,
		const Matrix& inTargetClasses,
		const Matrix& inTargetImageIds,
		const Matrix& inIouThresholds,
		oa::I32 inClassCount,
		oa::F32 inScoreThreshold = 0.0F
	);

	// Multiclass semantic-segmentation metrics over flattened Int32 label
	// matrices. labels outside [0,classCount) are ignored by the confusion
	// accumulator, which provides the ordinary ignore-index behavior.
	[[nodiscard]] SegmentationMetricsResult evaluateSegmentation(
		const Matrix& inPredicted,
		const Matrix& inTarget,
		oa::I32 inClassCount
	);
} // namespace FnDetection

} // namespace oa
