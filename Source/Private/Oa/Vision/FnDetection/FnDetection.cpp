#include <Oa/Vision/FnDetection.h>

#include <Oa/Core/BufferAccess.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/Log.h>
#include <Oa/Runtime/Context.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

bool IsBoxMatrix(const OaMatrix& InBoxes) {
	return InBoxes.Rank() == 2 && InBoxes.Size(0) > 0
		&& InBoxes.Size(1) == 4
		&& InBoxes.GetDtype() == OaScalarType::Float32
		&& InBoxes.NumElements() <= std::numeric_limits<OaU32>::max();
}

bool IsVector(const OaMatrix& InMatrix, OaI64 InSize, OaScalarType InDtype) {
	return InMatrix.Rank() == 1 && InMatrix.Size(0) == InSize
		&& InMatrix.GetDtype() == InDtype;
}

} // namespace

OaMatrix OaFnDetection::BoxIou(
	const OaMatrix& InA,
	const OaMatrix& InB) {
	if (!IsBoxMatrix(InA) || !IsBoxMatrix(InB)) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"OaFnDetection::BoxIou expects FP32 cx/cy/w/h boxes [N,4] and [M,4]");
		return {};
	}
	const OaU32 rowsA = static_cast<OaU32>(InA.Size(0));
	const OaU32 rowsB = static_cast<OaU32>(InB.Size(0));
	if (static_cast<OaU64>(rowsA) * rowsB > std::numeric_limits<OaU32>::max()) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"OaFnDetection::BoxIou output exceeds the dispatch limit");
		return {};
	}
	OaMatrix out = OaFnMatrix::Empty(
		{static_cast<OaI64>(rowsA), static_cast<OaI64>(rowsB)},
		OaScalarType::Float32);
	struct Push { OaU32 RowsA, RowsB; } push{rowsA, rowsB};
	OaBufferAccess access[] = {
		OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
	const OaU32 pairs = rowsA * rowsB;
	OaContext::GetDefault().Add("DetectionBoxIou", {&InA, &InB, &out},
		access, &push, sizeof(push), (pairs + 255U) / 256U);
	return out;
}

OaNmsResult OaFnDetection::Nms(
	const OaMatrix& InBoxes,
	const OaMatrix& InScores,
	const OaMatrix& InClasses,
	const OaNmsConfig& InConfig) {
	const bool valid = IsBoxMatrix(InBoxes)
		&& InScores.GetShape() == OaMatrixShape{InBoxes.Size(0)}
		&& InScores.GetDtype() == OaScalarType::Float32
		&& InClasses.GetShape() == InScores.GetShape()
		&& InClasses.GetDtype() == OaScalarType::Int32
		&& std::isfinite(InConfig.IouThreshold)
		&& InConfig.IouThreshold >= 0.0F && InConfig.IouThreshold <= 1.0F
		&& std::isfinite(InConfig.ScoreThreshold)
		&& InConfig.MaxDetections > 0;
	if (!valid) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"OaFnDetection::Nms expects FP32 boxes [N,4], FP32 scores [N], Int32 classes [N], finite thresholds and MaxDetections > 0");
		return {};
	}
	const OaU32 count = static_cast<OaU32>(InBoxes.Size(0));
	const OaU32 maximum = static_cast<OaU32>(
		std::min<OaI64>(count, InConfig.MaxDetections));
	OaNmsResult result{
		.Indices = OaFnMatrix::Empty(
			{static_cast<OaI64>(maximum)}, OaScalarType::Int32),
		.Count = OaFnMatrix::Empty({1}, OaScalarType::UInt32),
	};
	struct Push {
		OaU32 Count, Maximum, ClassAgnostic;
		OaF32 IouThreshold, ScoreThreshold;
	} push{count, maximum, InConfig.ClassAgnostic ? 1U : 0U,
		InConfig.IouThreshold, InConfig.ScoreThreshold};
	OaBufferAccess access[] = {
		OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Write, OaBufferAccess::Write};
	OaContext::GetDefault().Add("DetectionNms",
		{&InBoxes, &InScores, &InClasses, &result.Indices, &result.Count},
		access, &push, sizeof(push), 1);
	return result;
}

OaMatrix OaFnDetection::ConfusionMatrix(
	const OaMatrix& InPredicted,
	const OaMatrix& InTarget,
	OaI32 InClassCount) {
	const bool valid = InPredicted.Rank() == 1 && InPredicted.Size(0) > 0
		&& InPredicted.GetDtype() == OaScalarType::Int32
		&& InTarget.GetShape() == InPredicted.GetShape()
		&& InTarget.GetDtype() == OaScalarType::Int32
		&& InClassCount > 0
		&& static_cast<OaU64>(InClassCount) * InClassCount
			<= std::numeric_limits<OaU32>::max();
	if (!valid) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"OaFnDetection::ConfusionMatrix expects Int32 predicted/target [N] and ClassCount > 0");
		return {};
	}
	OaMatrix out = OaFnMatrix::Zeros(
		{InClassCount, InClassCount}, OaScalarType::UInt32);
	struct Push { OaU32 Count, Classes; }
		push{static_cast<OaU32>(InPredicted.Size(0)),
			static_cast<OaU32>(InClassCount)};
	OaBufferAccess access[] = {
		OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::ReadWrite};
	OaContext::GetDefault().Add("DetectionConfusion",
		{&InPredicted, &InTarget, &out}, access, &push, sizeof(push),
		(push.Count + 255U) / 256U);
	return out;
}

OaMatrix OaFnDetection::BinaryMaskCounts(
	const OaMatrix& InPredicted,
	const OaMatrix& InTarget) {
	const bool valid = InPredicted.NumElements() > 0
		&& InPredicted.GetDtype() == OaScalarType::UInt8
		&& InTarget.GetShape() == InPredicted.GetShape()
		&& InTarget.GetDtype() == OaScalarType::UInt8
		&& InPredicted.NumElements() <= std::numeric_limits<OaU32>::max();
	if (!valid) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"OaFnDetection::BinaryMaskCounts expects equally-shaped UInt8 masks");
		return {};
	}
	OaMatrix out = OaFnMatrix::Zeros({4}, OaScalarType::UInt32);
	struct Push { OaU32 Count; }
		push{static_cast<OaU32>(InPredicted.NumElements())};
	OaBufferAccess access[] = {
		OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::ReadWrite};
	OaContext::GetDefault().Add("DetectionBinaryCounts",
		{&InPredicted, &InTarget, &out}, access, &push, sizeof(push),
		(push.Count + 255U) / 256U);
	return out;
}

OaDetectionMetricsResult OaFnDetection::Evaluate(
	const OaMatrix& InPredictedBoxes,
	const OaMatrix& InPredictedScores,
	const OaMatrix& InPredictedClasses,
	const OaMatrix& InPredictedImageIds,
	const OaMatrix& InTargetBoxes,
	const OaMatrix& InTargetClasses,
	const OaMatrix& InTargetImageIds,
	const OaMatrix& InIouThresholds,
	OaI32 InClassCount,
	OaF32 InScoreThreshold) {
	const OaI64 predicted = InPredictedBoxes.Rank() == 2
		? InPredictedBoxes.Size(0) : 0;
	const OaI64 targets = InTargetBoxes.Rank() == 2
		? InTargetBoxes.Size(0) : 0;
	const OaI64 thresholds = InIouThresholds.Rank() == 1
		? InIouThresholds.Size(0) : 0;
	const bool valid = IsBoxMatrix(InPredictedBoxes)
		&& IsVector(InPredictedScores, predicted, OaScalarType::Float32)
		&& IsVector(InPredictedClasses, predicted, OaScalarType::Int32)
		&& IsVector(InPredictedImageIds, predicted, OaScalarType::Int32)
		&& IsBoxMatrix(InTargetBoxes)
		&& IsVector(InTargetClasses, targets, OaScalarType::Int32)
		&& IsVector(InTargetImageIds, targets, OaScalarType::Int32)
		&& thresholds > 0
		&& InIouThresholds.GetDtype() == OaScalarType::Float32
		&& InClassCount > 0
		&& std::isfinite(InScoreThreshold);
	if (!valid) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"OaFnDetection::Evaluate expects prediction boxes/scores/classes/image IDs, target boxes/classes/image IDs, FP32 IoU thresholds [T], ClassCount > 0 and a finite score threshold");
		return {};
	}

	const OaU64 pairCount = static_cast<OaU64>(thresholds)
		* static_cast<OaU64>(InClassCount);
	const OaU64 stateElements = pairCount
		* static_cast<OaU64>(predicted + targets);
	const OaU64 curveElements = pairCount
		* static_cast<OaU64>(predicted) * 2ULL;
	if (pairCount > std::numeric_limits<OaU32>::max()
		|| stateElements > std::numeric_limits<OaU32>::max()
		|| curveElements > std::numeric_limits<OaU32>::max()) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"OaFnDetection::Evaluate scratch exceeds the 32-bit shader indexing limit");
		return {};
	}

	OaMatrix state = OaFnMatrix::Empty(
		{static_cast<OaI64>(pairCount), predicted + targets},
		OaScalarType::UInt32);
	OaMatrix curve = OaFnMatrix::Empty(
		{static_cast<OaI64>(pairCount), predicted, 2},
		OaScalarType::Float32);
	OaDetectionMetricsResult result{
		.Counts = OaFnMatrix::Empty(
			{thresholds, InClassCount, 3}, OaScalarType::UInt32),
		.PerClass = OaFnMatrix::Empty(
			{thresholds, InClassCount, 4}, OaScalarType::Float32),
		.MeanAveragePrecisionByThreshold = OaFnMatrix::Empty(
			{thresholds}, OaScalarType::Float32),
		.MeanAveragePrecision = OaFnMatrix::Empty({1}, OaScalarType::Float32),
	};

	struct CurvesPush {
		OaU32 Predicted, Targets, Thresholds, Classes, StateStride;
		OaF32 ScoreThreshold;
	} curvesPush{
		static_cast<OaU32>(predicted), static_cast<OaU32>(targets),
		static_cast<OaU32>(thresholds), static_cast<OaU32>(InClassCount),
		static_cast<OaU32>(predicted + targets), InScoreThreshold};
	OaBufferAccess curvesAccess[] = {
		OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write,
		OaBufferAccess::Write, OaBufferAccess::Write, OaBufferAccess::Write};
	OaContext::GetDefault().Add("DetectionMetricCurves",
		{&InPredictedBoxes, &InPredictedScores, &InPredictedClasses,
		 &InPredictedImageIds, &InTargetBoxes, &InTargetClasses,
		 &InTargetImageIds, &InIouThresholds, &state, &curve,
		 &result.Counts, &result.PerClass},
		curvesAccess, &curvesPush, sizeof(curvesPush),
		static_cast<OaU32>(pairCount));

	struct ApPush { OaU32 Predicted, Pairs; }
		apPush{static_cast<OaU32>(predicted), static_cast<OaU32>(pairCount)};
	OaBufferAccess apAccess[] = {
		OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::ReadWrite};
	OaContext::GetDefault().Add("DetectionAveragePrecision",
		{&curve, &result.Counts, &result.PerClass}, apAccess,
		&apPush, sizeof(apPush), static_cast<OaU32>(pairCount));

	struct MapPush { OaU32 Thresholds, Classes; }
		mapPush{static_cast<OaU32>(thresholds), static_cast<OaU32>(InClassCount)};
	OaBufferAccess mapAccess[] = {
		OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Write, OaBufferAccess::Write};
	OaContext::GetDefault().Add("DetectionMeanAveragePrecision",
		{&result.Counts, &result.PerClass,
		 &result.MeanAveragePrecisionByThreshold,
		 &result.MeanAveragePrecision}, mapAccess,
		&mapPush, sizeof(mapPush), 1);
	return result;
}

OaSegmentationMetricsResult OaFnDetection::EvaluateSegmentation(
	const OaMatrix& InPredicted,
	const OaMatrix& InTarget,
	OaI32 InClassCount) {
	const bool valid = InPredicted.NumElements() > 0
		&& InPredicted.GetShape() == InTarget.GetShape()
		&& InPredicted.GetDtype() == OaScalarType::Int32
		&& InTarget.GetDtype() == OaScalarType::Int32
		&& InClassCount > 0;
	if (!valid) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"OaFnDetection::EvaluateSegmentation expects equally-shaped non-empty Int32 labels and ClassCount > 0");
		return {};
	}
	OaMatrix predicted = InPredicted.Flatten();
	OaMatrix target = InTarget.Flatten();
	OaSegmentationMetricsResult result{
		.Confusion = ConfusionMatrix(predicted, target, InClassCount),
		.PerClass = OaFnMatrix::Empty(
			{InClassCount, 4}, OaScalarType::Float32),
		.MeanIou = OaFnMatrix::Empty({1}, OaScalarType::Float32),
		.PixelAccuracy = OaFnMatrix::Empty({1}, OaScalarType::Float32),
	};
	struct Push { OaU32 Classes; }
		push{static_cast<OaU32>(InClassCount)};
	OaBufferAccess access[] = {
		OaBufferAccess::Read, OaBufferAccess::Write,
		OaBufferAccess::Write, OaBufferAccess::Write};
	OaContext::GetDefault().Add("SegmentationMetrics",
		{&result.Confusion, &result.PerClass,
		 &result.MeanIou, &result.PixelAccuracy},
		access, &push, sizeof(push), 1);
	return result;
}
