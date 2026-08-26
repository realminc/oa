#include <oa/ml/policy.h>

#include <oa/core/fnMatrix.h>
#include <oa/core/log.h>
#include <oa/core/op.h>
#include <oa/ml/autograd.h>
#include <oa/ml/fnMatrix.h>
#include <oa/runtime/executionSession.h>
#include <oa/core/std/limits.h>
#include <oa/core/std/scalarMath.h>

namespace {

bool validatePolicyShape(
	const oa::Matrix& inLogits,
	const oa::Matrix& inValue) {
	return inLogits.rank() == 2
		&& inLogits.size(0) > 0 && inLogits.size(1) > 1
		&& inLogits.size(0)
			<= static_cast<oa::I64>(oa::Limits<oa::U32>::max())
		&& inLogits.getDtype() == oa::ScalarType::Float32
		&& inValue.getShape() == oa::MatrixShape{inLogits.size(0)}
		&& inValue.getDtype() == oa::ScalarType::Float32;
}

bool validateContinuousPolicyShape(
	const oa::Matrix& inMean,
	const oa::Matrix& inLogStddev,
	const oa::Matrix& inValue,
	oa::F32 inMinimum,
	oa::F32 inMaximum,
	oa::F32 inEpsilon) {
	return inMean.rank() == 2 && inMean.size(0) > 0 && inMean.size(1) > 0
		&& inMean.getDtype() == oa::ScalarType::Float32
		&& inLogStddev.getShape() == inMean.getShape()
		&& inLogStddev.getDtype() == oa::ScalarType::Float32
		&& inValue.getShape() == oa::MatrixShape{inMean.size(0)}
		&& inValue.getDtype() == oa::ScalarType::Float32
		&& oa::isFinite(inMinimum) && oa::isFinite(inMaximum)
		&& inMinimum < inMaximum && oa::isFinite(inEpsilon)
		&& inEpsilon > 0.0F && inEpsilon < 1.0F;
}

bool attachSemanticOutput(
	const oa::Matrix& inOutput,
	oa::U32 inSemanticOp,
	oa::U32 inOutputIndex) {
	auto grad = inOutput.getGradFn();
	return not grad || oa::FnAutograd::attachSemantic(
		grad, inSemanticOp, inOutputIndex).isOk();
}

} // namespace

oa::PolicyResult oa::FnPolicy::evaluateCategorical(
	const oa::Matrix& inLogits,
	const oa::Matrix& inAction,
	const oa::Matrix& inValue) {
	if (!validatePolicyShape(inLogits, inValue)
		|| inAction.getShape() != oa::MatrixShape{inLogits.size(0)}
		|| inAction.getDtype() != oa::ScalarType::Int32) {
		OaLogError(oa::LogComponent::Ml,
			"oa::FnPolicy::evaluateCategorical expects FP32 logits [E,A], FP32 values [E], and Int32 actions [E]");
		return {};
	}

	auto& context = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(context);
	const oa::I64 environments = inLogits.size(0);
	const oa::Matrix logProbabilityAll = oa::FnMatrix::logSoftmax(inLogits, 1);
	const oa::Matrix actionColumn = oa::FnMatrix::reshape(
		inAction, oa::MatrixShape{environments, 1});
	const oa::Matrix selected = oa::FnMatrix::gatherLastDim(
		logProbabilityAll, actionColumn);
	const oa::Matrix probability = oa::FnMatrix::exp(logProbabilityAll);
	const oa::Matrix entropyColumn = oa::FnMatrix::neg(oa::FnMatrix::sum(
		oa::FnMatrix::mul(probability, logProbabilityAll), 1));
	oa::PolicyResult result{
		.action = inAction,
		.logProbability = oa::FnMatrix::reshape(
			selected, oa::MatrixShape{environments}),
		.entropy = oa::FnMatrix::reshape(
			entropyColumn, oa::MatrixShape{environments}),
		.value = inValue,
	};
	auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnPolicy::evaluateCategorical,
		{&inLogits, &inAction, &inValue},
		{
			&result.action,
			&result.logProbability,
			&result.entropy,
			&result.value,
		});
	if (not semantic.isOk()) return {};
	if (not attachSemanticOutput(
			result.logProbability, semantic.getValue(), 1U)
		|| not attachSemanticOutput(
			result.entropy, semantic.getValue(), 2U))
	{
		return {};
	}
	return result;
}

oa::PolicyResult oa::FnPolicy::sampleCategorical(
	const oa::Matrix& inLogits,
	const oa::Matrix& inValue,
	oa::U64 inSeed) {
	if (!validatePolicyShape(inLogits, inValue)) {
		OaLogError(oa::LogComponent::Ml,
			"oa::FnPolicy::sampleCategorical expects FP32 logits [E,A] and FP32 values [E]");
		return {};
	}
	auto& context = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(context);
	oa::Matrix action = oa::FnMatrix::sampleLogits(
		inLogits, 1.0F, 0, 1.0F, inSeed);
	oa::PolicyResult result =
		evaluateCategorical(inLogits, action, inValue);
	auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnPolicy::sampleCategorical,
		{&inLogits, &inValue},
		{
			&result.action,
			&result.logProbability,
			&result.entropy,
			&result.value,
		},
		{oa::OpAttribute::fromUnsignedInteger("seed", inSeed)});
	if (not semantic.isOk()) return {};
	if (not attachSemanticOutput(result.action, semantic.getValue(), 0U)
		|| not attachSemanticOutput(
			result.logProbability, semantic.getValue(), 1U)
		|| not attachSemanticOutput(
			result.entropy, semantic.getValue(), 2U))
	{
		return {};
	}
	return result;
}

oa::ContinuousPolicyResult oa::FnPolicy::evaluateTanhNormal(
	const oa::Matrix& inMean,
	const oa::Matrix& inLogStddev,
	const oa::Matrix& inRawAction,
	const oa::Matrix& inValue,
	oa::F32 inMinimum,
	oa::F32 inMaximum,
	oa::F32 inEpsilon) {
	if (!validateContinuousPolicyShape(inMean, inLogStddev, inValue,
			inMinimum, inMaximum, inEpsilon)
		|| inRawAction.getShape() != inMean.getShape()
		|| inRawAction.getDtype() != oa::ScalarType::Float32) {
		OaLogError(oa::LogComponent::Ml,
			"oa::FnPolicy::evaluateTanhNormal expects matching FP32 mean/log-std/raw-action [E,A] and FP32 values [E]");
		return {};
	}

	auto& context = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(context);
	constexpr oa::F32 logTwoPi = 1.8378770664093453F;
	const oa::F32 scale = 0.5F * (inMaximum - inMinimum);
	const oa::F32 bias = 0.5F * (inMaximum + inMinimum);
	const oa::Matrix logStddev = oa::FnMatrix::clampMax(
		oa::FnMatrix::clampMin(inLogStddev, -20.0F), 2.0F);
	const oa::Matrix stddev = oa::FnMatrix::exp(logStddev);
	const oa::Matrix normalized = oa::FnMatrix::div(
		oa::FnMatrix::sub(inRawAction, inMean), stddev);
	const oa::Matrix baseLogProbability = oa::FnMatrix::scale(
		oa::FnMatrix::add(
			oa::FnMatrix::add(oa::FnMatrix::mul(normalized, normalized),
				oa::FnMatrix::scale(logStddev, 2.0F)),
			oa::FnMatrix::full(inMean.getShape(), logTwoPi,
				oa::ScalarType::Float32)),
		-0.5F);
	const oa::Matrix squashed = oa::FnMatrix::tanh(inRawAction);
	const oa::Matrix action = oa::FnMatrix::addScalar(
		oa::FnMatrix::scale(squashed, scale), bias);
	const oa::Matrix jacobian = oa::FnMatrix::log(oa::FnMatrix::addScalar(
		oa::FnMatrix::scale(oa::FnMatrix::mul(squashed, squashed), -1.0F),
		1.0F + inEpsilon));
	const oa::Matrix corrected = oa::FnMatrix::sub(
		baseLogProbability,
		oa::FnMatrix::addScalar(jacobian, oa::log(scale)));
	const oa::Matrix entropyPerDimension = oa::FnMatrix::addScalar(
		logStddev, 0.5F * (1.0F + logTwoPi));
	oa::ContinuousPolicyResult result{
		.action = action,
		.rawAction = inRawAction,
		.logProbability = oa::FnMatrix::reshape(
			oa::FnMatrix::sum(corrected, 1), {inMean.size(0)}),
		.entropy = oa::FnMatrix::reshape(
			oa::FnMatrix::sum(entropyPerDimension, 1), {inMean.size(0)}),
		.value = inValue,
	};
	auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnPolicy::evaluateTanhNormal,
		{&inMean, &inLogStddev, &inRawAction, &inValue},
		{
			&result.action,
			&result.rawAction,
			&result.logProbability,
			&result.entropy,
			&result.value,
		},
		{
			oa::OpAttribute::fromFloat("minimum", inMinimum),
			oa::OpAttribute::fromFloat("maximum", inMaximum),
			oa::OpAttribute::fromFloat("epsilon", inEpsilon),
		});
	if (not semantic.isOk()) return {};
	if (not attachSemanticOutput(result.action, semantic.getValue(), 0U)
		|| not attachSemanticOutput(
			result.logProbability, semantic.getValue(), 2U)
		|| not attachSemanticOutput(
			result.entropy, semantic.getValue(), 3U))
	{
		return {};
	}
	return result;
}

oa::ContinuousPolicyResult oa::FnPolicy::sampleTanhNormal(
	const oa::Matrix& inMean,
	const oa::Matrix& inLogStddev,
	const oa::Matrix& inValue,
	oa::F32 inMinimum,
	oa::F32 inMaximum,
	oa::U64 inSeed,
	oa::F32 inEpsilon) {
	if (!validateContinuousPolicyShape(inMean, inLogStddev, inValue,
			inMinimum, inMaximum, inEpsilon)) {
		OaLogError(oa::LogComponent::Ml,
			"oa::FnPolicy::sampleTanhNormal received an invalid policy contract");
		return {};
	}
	auto& context = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(context);
	const oa::Matrix logStddev = oa::FnMatrix::clampMax(
		oa::FnMatrix::clampMin(inLogStddev, -20.0F), 2.0F);
	const oa::Matrix noise = oa::FnMatrix::philoxNormal(
		inMean, 0.0F, 1.0F, inSeed);
	const oa::Matrix rawAction = oa::FnMatrix::add(
		inMean, oa::FnMatrix::mul(oa::FnMatrix::exp(logStddev), noise));
	oa::ContinuousPolicyResult result = evaluateTanhNormal(
		inMean, logStddev, rawAction, inValue,
		inMinimum, inMaximum, inEpsilon);
	auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnPolicy::sampleTanhNormal,
		{&inMean, &inLogStddev, &inValue},
		{
			&result.action,
			&result.rawAction,
			&result.logProbability,
			&result.entropy,
			&result.value,
		},
		{
			oa::OpAttribute::fromFloat("minimum", inMinimum),
			oa::OpAttribute::fromFloat("maximum", inMaximum),
			oa::OpAttribute::fromUnsignedInteger("seed", inSeed),
			oa::OpAttribute::fromFloat("epsilon", inEpsilon),
		});
	if (not semantic.isOk()) return {};
	if (not attachSemanticOutput(result.action, semantic.getValue(), 0U)
		|| not attachSemanticOutput(
			result.rawAction, semantic.getValue(), 1U)
		|| not attachSemanticOutput(
			result.logProbability, semantic.getValue(), 2U)
		|| not attachSemanticOutput(
			result.entropy, semantic.getValue(), 3U))
	{
		return {};
	}
	return result;
}
