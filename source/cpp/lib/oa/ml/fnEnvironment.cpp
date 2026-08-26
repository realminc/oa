#include <oa/ml/fnEnvironment.h>

#include <oa/core/fnMatrix.h>
#include <oa/core/log.h>
#include <oa/core/op.h>
#include <oa/ml/autograd.h>
#include <oa/runtime/executionSession.h>

#include <oa/core/std/scalarMath.h>

namespace {

bool isFp32(const oa::Matrix& inValue) {
	return !inValue.isEmpty()
		&& inValue.getDtype() == oa::ScalarType::Float32;
}

oa::Matrix clampMatrix(
	const oa::Matrix& inValue,
	oa::F32 inMinimum,
	oa::F32 inMaximum) {
	return oa::FnMatrix::clampMax(
		oa::FnMatrix::clampMin(inValue, inMinimum), inMaximum);
}

oa::Matrix attachSemanticResult(
	oa::Matrix inResult,
	oa::OpLoweringScope& inLowering,
	const oa::OpContract& inContract,
	oa::MatrixArgs inInputs,
	oa::OpAttributeArgs inAttributes) {
	auto semantic = inLowering.commitWithId(
		inContract, inInputs, {&inResult}, inAttributes);
	if (not semantic.isOk()) return {};
	if (auto grad = inResult.getGradFn()) {
		if (not oa::FnAutograd::attachSemantic(
			grad, semantic.getValue()).isOk())
		{
			return {};
		}
	}
	return inResult;
}

} // namespace

oa::Matrix oa::FnEnvironment::normalizeObservation(
	const oa::Matrix& inObservation,
	const oa::Matrix& inMean,
	const oa::Matrix& inStddev,
	oa::F32 inEpsilon,
	oa::F32 inClip) {
	if (!isFp32(inObservation) || !isFp32(inMean) || !isFp32(inStddev)
		|| !oa::isFinite(inEpsilon) || inEpsilon <= 0.0F
		|| !oa::isFinite(inClip) || inClip <= 0.0F) {
		OaLogError(oa::LogComponent::Ml,
			"NormalizeObservation expects FP32 matrices and positive finite epsilon/clip");
		return {};
	}
	auto& context = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(context);
	const oa::Matrix normalized = oa::FnMatrix::div(
		oa::FnMatrix::sub(inObservation, inMean),
		oa::FnMatrix::addScalar(inStddev, inEpsilon));
	return attachSemanticResult(
		clampMatrix(normalized, -inClip, inClip),
		lowering,
		oa::detail::opRegistry::FnEnvironment::normalizeObservation,
		{&inObservation, &inMean, &inStddev},
		{
			oa::OpAttribute::fromFloat("epsilon", inEpsilon),
			oa::OpAttribute::fromFloat("clip", inClip),
		});
}

oa::Matrix oa::FnEnvironment::scaleAction(
	const oa::Matrix& inAction,
	oa::F32 inSourceMinimum,
	oa::F32 inSourceMaximum,
	oa::F32 inTargetMinimum,
	oa::F32 inTargetMaximum,
	bool inClamp) {
	if (!isFp32(inAction)
		|| !oa::isFinite(inSourceMinimum)
		|| !oa::isFinite(inSourceMaximum)
		|| !oa::isFinite(inTargetMinimum)
		|| !oa::isFinite(inTargetMaximum)
		|| inSourceMinimum >= inSourceMaximum
		|| inTargetMinimum >= inTargetMaximum) {
		OaLogError(oa::LogComponent::Ml,
			"ScaleAction expects FP32 actions and ordered finite bounds");
		return {};
	}
	auto& context = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(context);
	const oa::F32 sourceScale = 1.0F / (inSourceMaximum - inSourceMinimum);
	const oa::F32 targetScale = inTargetMaximum - inTargetMinimum;
	oa::Matrix source = inClamp
		? clampMatrix(inAction, inSourceMinimum, inSourceMaximum)
		: inAction;
	return attachSemanticResult(
		oa::FnMatrix::addScalar(oa::FnMatrix::scale(
			oa::FnMatrix::subScalar(source, inSourceMinimum),
			sourceScale * targetScale), inTargetMinimum),
		lowering,
		oa::detail::opRegistry::FnEnvironment::scaleAction,
		{&inAction},
		{
			oa::OpAttribute::fromFloat(
				"sourceMinimum", inSourceMinimum),
			oa::OpAttribute::fromFloat(
				"sourceMaximum", inSourceMaximum),
			oa::OpAttribute::fromFloat(
				"targetMinimum", inTargetMinimum),
			oa::OpAttribute::fromFloat(
				"targetMaximum", inTargetMaximum),
			oa::OpAttribute::fromBoolean("clamp", inClamp),
		});
}

oa::Matrix oa::FnEnvironment::clipReward(
	const oa::Matrix& inReward,
	oa::F32 inMinimum,
	oa::F32 inMaximum) {
	if (!isFp32(inReward) || !oa::isFinite(inMinimum)
		|| !oa::isFinite(inMaximum) || inMinimum > inMaximum) {
		OaLogError(oa::LogComponent::Ml,
			"ClipReward expects FP32 rewards and ordered finite bounds");
		return {};
	}
	auto& context = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(context);
	return attachSemanticResult(
		clampMatrix(inReward, inMinimum, inMaximum),
		lowering,
		oa::detail::opRegistry::FnEnvironment::clipReward,
		{&inReward},
		{
			oa::OpAttribute::fromFloat("minimum", inMinimum),
			oa::OpAttribute::fromFloat("maximum", inMaximum),
		});
}
