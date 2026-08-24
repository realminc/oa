#include <oa/vision/fnDetection.h>

#include <oa/core/bufferAccess.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/log.h>
#include <oa/core/op.h>
#include <oa/runtime/executionSession.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

bool isBoxMatrix(const oa::Matrix& inBoxes) {
	return inBoxes.rank() == 2 && inBoxes.size(0) > 0
		&& inBoxes.size(1) == 4
		&& inBoxes.getDtype() == oa::ScalarType::Float32
		&& inBoxes.numElements() <= std::numeric_limits<oa::U32>::max();
}

bool isVector(const oa::Matrix& inMatrix, oa::I64 inSize, oa::ScalarType inDtype) {
	return inMatrix.rank() == 1 && inMatrix.size(0) == inSize && inMatrix.getDtype() == inDtype;
}

} // namespace

oa::Matrix oa::FnDetection::boxIou(const oa::Matrix& inA,	const oa::Matrix& inB) {
	if (!isBoxMatrix(inA) || !isBoxMatrix(inB)) {
		OaLogError(oa::LogComponent::Vision,	"oa::FnDetection::boxIou expects FP32 cx/cy/w/h boxes [N,4] and [M,4]");
		return {};
	}
	const oa::U32 rowsA = static_cast<oa::U32>(inA.size(0));
	const oa::U32 rowsB = static_cast<oa::U32>(inB.size(0));
	if (static_cast<oa::U64>(rowsA) * rowsB > std::numeric_limits<oa::U32>::max()) {
		OaLogError(oa::LogComponent::Vision,
			"oa::FnDetection::boxIou output exceeds the dispatch limit");
		return {};
	}
	auto& context = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(context);
	oa::Matrix out = oa::FnMatrix::empty(
		{static_cast<oa::I64>(rowsA), static_cast<oa::I64>(rowsB)},
		oa::ScalarType::Float32
	);
	struct Push { oa::U32 rowsA, rowsB; } push{rowsA, rowsB};
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write
	};
	const oa::U32 pairs = rowsA * rowsB;
	context.add( "DetectionBoxIou", {&inA, &inB, &out},
		access, &push, sizeof(push), (pairs + 255U) / 256U
	);
	if (not lowering.commit(
		oa::detail::opRegistry::FnDetection::boxIou,
		{&inA, &inB}, {&out}).isOk())
	{
		return {};
	}
	return out;
}

oa::NmsResult oa::FnDetection::nms(
	const oa::Matrix& inBoxes,
	const oa::Matrix& inScores,
	const oa::Matrix& inClasses,
	const oa::NmsConfig& inConfig
) {
	const bool valid = isBoxMatrix(inBoxes)
		&& inScores.getShape() == oa::MatrixShape{inBoxes.size(0)}
		&& inScores.getDtype() == oa::ScalarType::Float32
		&& inClasses.getShape() == inScores.getShape()
		&& inClasses.getDtype() == oa::ScalarType::Int32
		&& std::isfinite(inConfig.iouThreshold)
		&& inConfig.iouThreshold >= 0.0F && inConfig.iouThreshold <= 1.0F
		&& std::isfinite(inConfig.scoreThreshold)
		&& inConfig.maxDetections > 0;
	if (!valid) {
		OaLogError(oa::LogComponent::Vision,	"oa::FnDetection::nms expects FP32 boxes [N,4], FP32 scores [N], Int32 classes [N], finite thresholds and maxDetections > 0");
		return {};
	}
	auto& context = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(context);
	const oa::U32 count = static_cast<oa::U32>(inBoxes.size(0));
	const oa::U32 maximum = static_cast<oa::U32>(
		std::min<oa::I64>(count, inConfig.maxDetections));
	oa::NmsResult result{
		.indices = oa::FnMatrix::empty({static_cast<oa::I64>(maximum)}, oa::ScalarType::Int32),
		.count = oa::FnMatrix::empty({1}, oa::ScalarType::UInt32),
	};
	struct Push {
		oa::U32 count, maximum, classAgnostic;
		oa::F32 iouThreshold, scoreThreshold;
	} push {count, maximum, inConfig.classAgnostic ? 1U : 0U,
		inConfig.iouThreshold, inConfig.scoreThreshold};
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Write, oa::BufferAccess::Write
	};
	context.add( "DetectionNms",
		{&inBoxes, &inScores, &inClasses, &result.indices, &result.count},
		access, &push, sizeof(push), 1
	);
	if (not lowering.commit(
		oa::detail::opRegistry::FnDetection::nms,
		{&inBoxes, &inScores, &inClasses},
		{&result.indices, &result.count},
		{
			oa::OpAttribute::fromFloat(
				"iouThreshold", inConfig.iouThreshold),
			oa::OpAttribute::fromFloat(
				"scoreThreshold", inConfig.scoreThreshold),
			oa::OpAttribute::fromSignedInteger(
				"maxDetections", inConfig.maxDetections),
			oa::OpAttribute::fromBoolean(
				"classAgnostic", inConfig.classAgnostic),
		}).isOk())
	{
		return {};
	}
	return result;
}

oa::Matrix oa::FnDetection::confusionMatrix(
	const oa::Matrix& inPredicted,
	const oa::Matrix& inTarget,
	oa::I32 inClassCount) {
	const bool valid = inPredicted.rank() == 1 && inPredicted.size(0) > 0
		&& inPredicted.getDtype() == oa::ScalarType::Int32
		&& inTarget.getShape() == inPredicted.getShape()
		&& inTarget.getDtype() == oa::ScalarType::Int32
		&& inClassCount > 0
		&& static_cast<oa::U64>(inClassCount) * inClassCount
			<= std::numeric_limits<oa::U32>::max();
	if (!valid) {
		OaLogError(oa::LogComponent::Vision,
			"oa::FnDetection::confusionMatrix expects Int32 predicted/target [N] and ClassCount > 0");
		return {};
	}
	auto& context = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(context);
	oa::Matrix out = oa::FnMatrix::zeros(
		{inClassCount, inClassCount}, oa::ScalarType::UInt32);
	struct Push { oa::U32 count, classes; }
		push{static_cast<oa::U32>(inPredicted.size(0)),
			static_cast<oa::U32>(inClassCount)};
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::ReadWrite};
	context.add( "DetectionConfusion",
		{&inPredicted, &inTarget, &out}, access, &push, sizeof(push),
		(push.count + 255U) / 256U);
	if (not lowering.commit(
		oa::detail::opRegistry::FnDetection::confusionMatrix,
		{&inPredicted, &inTarget}, {&out},
		{oa::OpAttribute::fromSignedInteger(
			"classCount", inClassCount)}).isOk())
	{
		return {};
	}
	return out;
}

oa::Matrix oa::FnDetection::binaryMaskCounts(
	const oa::Matrix& inPredicted,
	const oa::Matrix& inTarget) {
	const bool valid = inPredicted.numElements() > 0
		&& inPredicted.getDtype() == oa::ScalarType::UInt8
		&& inTarget.getShape() == inPredicted.getShape()
		&& inTarget.getDtype() == oa::ScalarType::UInt8
		&& inPredicted.numElements() <= std::numeric_limits<oa::U32>::max();
	if (!valid) {
		OaLogError(oa::LogComponent::Vision,
			"oa::FnDetection::binaryMaskCounts expects equally-shaped UInt8 masks");
		return {};
	}
	auto& context = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(context);
	oa::Matrix out = oa::FnMatrix::zeros({4}, oa::ScalarType::UInt32);
	struct Push { oa::U32 count; }
		push{static_cast<oa::U32>(inPredicted.numElements())};
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::ReadWrite};
	context.add( "DetectionBinaryCounts",
		{&inPredicted, &inTarget, &out}, access, &push, sizeof(push),
		(push.count + 255U) / 256U);
	if (not lowering.commit(
		oa::detail::opRegistry::FnDetection::binaryMaskCounts,
		{&inPredicted, &inTarget}, {&out}).isOk())
	{
		return {};
	}
	return out;
}

oa::DetectionMetricsResult oa::FnDetection::evaluate(
	const oa::Matrix& inPredictedBoxes,
	const oa::Matrix& inPredictedScores,
	const oa::Matrix& inPredictedClasses,
	const oa::Matrix& inPredictedImageIds,
	const oa::Matrix& inTargetBoxes,
	const oa::Matrix& inTargetClasses,
	const oa::Matrix& inTargetImageIds,
	const oa::Matrix& inIouThresholds,
	oa::I32 inClassCount,
	oa::F32 inScoreThreshold) {
	const oa::I64 predicted = inPredictedBoxes.rank() == 2
		? inPredictedBoxes.size(0) : 0;
	const oa::I64 targets = inTargetBoxes.rank() == 2
		? inTargetBoxes.size(0) : 0;
	const oa::I64 thresholds = inIouThresholds.rank() == 1
		? inIouThresholds.size(0) : 0;
	const bool valid = isBoxMatrix(inPredictedBoxes)
		&& isVector(inPredictedScores, predicted, oa::ScalarType::Float32)
		&& isVector(inPredictedClasses, predicted, oa::ScalarType::Int32)
		&& isVector(inPredictedImageIds, predicted, oa::ScalarType::Int32)
		&& isBoxMatrix(inTargetBoxes)
		&& isVector(inTargetClasses, targets, oa::ScalarType::Int32)
		&& isVector(inTargetImageIds, targets, oa::ScalarType::Int32)
		&& thresholds > 0
		&& inIouThresholds.getDtype() == oa::ScalarType::Float32
		&& inClassCount > 0
		&& std::isfinite(inScoreThreshold);
	if (!valid) {
		OaLogError(oa::LogComponent::Vision,
			"oa::FnDetection::evaluate expects prediction boxes/scores/classes/image IDs, target boxes/classes/image IDs, FP32 IoU thresholds [T], ClassCount > 0 and a finite score threshold");
		return {};
	}

	auto& context = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(context);
	const oa::U64 pairCount = static_cast<oa::U64>(thresholds)
		* static_cast<oa::U64>(inClassCount);
	const oa::U64 stateElements = pairCount
		* static_cast<oa::U64>(predicted + targets);
	const oa::U64 curveElements = pairCount
		* static_cast<oa::U64>(predicted) * 2ULL;
	if (pairCount > std::numeric_limits<oa::U32>::max()
		|| stateElements > std::numeric_limits<oa::U32>::max()
		|| curveElements > std::numeric_limits<oa::U32>::max()) {
		OaLogError(oa::LogComponent::Vision,
			"oa::FnDetection::evaluate scratch exceeds the 32-bit shader indexing limit");
		return {};
	}

	oa::Matrix state = oa::FnMatrix::empty(
		{static_cast<oa::I64>(pairCount), predicted + targets},
		oa::ScalarType::UInt32);
	oa::Matrix curve = oa::FnMatrix::empty(
		{static_cast<oa::I64>(pairCount), predicted, 2},
		oa::ScalarType::Float32);
	oa::DetectionMetricsResult result{
		.counts = oa::FnMatrix::empty(
			{thresholds, inClassCount, 3}, oa::ScalarType::UInt32),
		.perClass = oa::FnMatrix::empty(
			{thresholds, inClassCount, 4}, oa::ScalarType::Float32),
		.meanAveragePrecisionByThreshold = oa::FnMatrix::empty(
			{thresholds}, oa::ScalarType::Float32),
		.meanAveragePrecision = oa::FnMatrix::empty({1}, oa::ScalarType::Float32),
	};

	struct CurvesPush {
		oa::U32 predicted, targets, thresholds, classes, stateStride;
		oa::F32 scoreThreshold;
	} curvesPush{
		static_cast<oa::U32>(predicted), static_cast<oa::U32>(targets),
		static_cast<oa::U32>(thresholds), static_cast<oa::U32>(inClassCount),
		static_cast<oa::U32>(predicted + targets), inScoreThreshold};
	oa::BufferAccess curvesAccess[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write,
		oa::BufferAccess::Write, oa::BufferAccess::Write, oa::BufferAccess::Write};
	context.add( "DetectionMetricCurves",
		{&inPredictedBoxes, &inPredictedScores, &inPredictedClasses,
		 &inPredictedImageIds, &inTargetBoxes, &inTargetClasses,
		 &inTargetImageIds, &inIouThresholds, &state, &curve,
		 &result.counts, &result.perClass},
		curvesAccess, &curvesPush, sizeof(curvesPush),
		static_cast<oa::U32>(pairCount));

	struct ApPush { oa::U32 predicted, pairs; }
		apPush{static_cast<oa::U32>(predicted), static_cast<oa::U32>(pairCount)};
	oa::BufferAccess apAccess[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::ReadWrite};
	context.add( "DetectionAveragePrecision",
		{&curve, &result.counts, &result.perClass}, apAccess,
		&apPush, sizeof(apPush), static_cast<oa::U32>(pairCount));

	struct MapPush { oa::U32 thresholds, classes; }
		mapPush{static_cast<oa::U32>(thresholds), static_cast<oa::U32>(inClassCount)};
	oa::BufferAccess mapAccess[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Write, oa::BufferAccess::Write};
	context.add( "DetectionMeanAveragePrecision",
		{&result.counts, &result.perClass,
		 &result.meanAveragePrecisionByThreshold,
		 &result.meanAveragePrecision}, mapAccess,
		&mapPush, sizeof(mapPush), 1);
	if (not lowering.commit(
		oa::detail::opRegistry::FnDetection::evaluate,
		{
			&inPredictedBoxes,
			&inPredictedScores,
			&inPredictedClasses,
			&inPredictedImageIds,
			&inTargetBoxes,
			&inTargetClasses,
			&inTargetImageIds,
			&inIouThresholds,
		},
		{
			&result.counts,
			&result.perClass,
			&result.meanAveragePrecisionByThreshold,
			&result.meanAveragePrecision,
		},
		{
			oa::OpAttribute::fromSignedInteger(
				"classCount", inClassCount),
			oa::OpAttribute::fromFloat(
				"scoreThreshold", inScoreThreshold),
		}).isOk())
	{
		return {};
	}
	return result;
}

oa::SegmentationMetricsResult oa::FnDetection::evaluateSegmentation(
	const oa::Matrix& inPredicted,
	const oa::Matrix& inTarget,
	oa::I32 inClassCount) {
	const bool valid = inPredicted.numElements() > 0
		&& inPredicted.getShape() == inTarget.getShape()
		&& inPredicted.getDtype() == oa::ScalarType::Int32
		&& inTarget.getDtype() == oa::ScalarType::Int32
		&& inClassCount > 0;
	if (!valid) {
		OaLogError(oa::LogComponent::Vision,
			"oa::FnDetection::evaluateSegmentation expects equally-shaped non-empty Int32 labels and ClassCount > 0");
		return {};
	}
	auto& context = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(context);
	oa::Matrix predicted = inPredicted.flatten();
	oa::Matrix target = inTarget.flatten();
	oa::SegmentationMetricsResult result{
		.confusion = confusionMatrix(predicted, target, inClassCount),
		.perClass = oa::FnMatrix::empty(
			{inClassCount, 4}, oa::ScalarType::Float32),
		.meanIou = oa::FnMatrix::empty({1}, oa::ScalarType::Float32),
		.pixelAccuracy = oa::FnMatrix::empty({1}, oa::ScalarType::Float32),
	};
	struct Push { oa::U32 classes; }
		push{static_cast<oa::U32>(inClassCount)};
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Write,
		oa::BufferAccess::Write, oa::BufferAccess::Write};
	context.add( "SegmentationMetrics",
		{&result.confusion, &result.perClass,
		 &result.meanIou, &result.pixelAccuracy},
		access, &push, sizeof(push), 1);
	if (not lowering.commit(
		oa::detail::opRegistry::FnDetection::evaluateSegmentation,
		{&inPredicted, &inTarget},
		{
			&result.confusion,
			&result.perClass,
			&result.meanIou,
			&result.pixelAccuracy,
		},
		{oa::OpAttribute::fromSignedInteger(
			"classCount", inClassCount)}).isOk())
	{
		return {};
	}
	return result;
}
