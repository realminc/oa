#include <oa/ml/fnLoss.h>

#include <oa/core/bufferAccess.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/log.h>
#include <oa/core/op.h>
#include <oa/ml/autograd.h>
#include <oa/ml/fnLoss.h>
#include <oa/runtime/executionSession.h>
#include <oa/core/std/limits.h>
#include <oa/core/std/scalarMath.h>

namespace {

bool isMatchingF32(const oa::Matrix& inMatrix, const oa::MatrixShape& inShape) {
	return !inMatrix.isEmpty()
		&& inMatrix.getDtype() == oa::ScalarType::Float32
		&& inMatrix.getShape() == inShape;
}

bool validatePolicyInputs(
	const oa::Matrix& inNewLogProbability,
	const oa::Matrix& inOldLogProbability,
	const oa::Matrix& inAdvantage,
	oa::F32 inClipEpsilon) {
	const oa::MatrixShape shape = inNewLogProbability.getShape();
	return shape.rank > 0 && inNewLogProbability.numElements() > 0
		&& inNewLogProbability.numElements()
			<= static_cast<oa::I64>(oa::Limits<oa::U32>::max())
		&& isMatchingF32(inNewLogProbability, shape)
		&& isMatchingF32(inOldLogProbability, shape)
		&& isMatchingF32(inAdvantage, shape)
		&& oa::isFinite(inClipEpsilon)
		&& inClipEpsilon > 0.0F && inClipEpsilon < 1.0F;
}

bool attachSemanticOutput(
	const oa::Matrix& inOutput,
	oa::U32 inOperation,
	oa::U32 inOutputIndex) {
	auto grad = inOutput.getGradFn();
	return not grad || oa::FnAutograd::attachSemantic(
		grad, inOperation, inOutputIndex).isOk();
}

class GradPpoClippedPolicy final : public oa::GradNode {
public:
	oa::F32 clipEpsilon = 0.2F;

	void backward(const oa::Matrix& inDOut, oa::Vector<oa::Matrix>& outDIn) override {
		if (outDIn.empty()) return;
		oa::Matrix grad = oa::FnLoss::ppoClippedPolicyBwd(
			saved(0), saved(1), saved(2), clipEpsilon);
		outDIn[0] = oa::FnMatrix::mul(grad, inDOut);
	}
};

} // namespace

oa::Matrix oa::FnLoss::ppoClippedPolicy(
	const oa::Matrix& inNewLogProbability,
	const oa::Matrix& inOldLogProbability,
	const oa::Matrix& inAdvantage,
	oa::F32 inClipEpsilon) {
	if (!validatePolicyInputs(
		inNewLogProbability, inOldLogProbability, inAdvantage, inClipEpsilon)) {
		OaLogError(oa::LogComponent::Ml,
			"oa::FnLoss::ppoClippedPolicy expects matching non-empty FP32 inputs and clip epsilon in (0,1)");
		return {};
	}
	setLastName("ppo_clipped_policy");
	auto& context = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(context);
	const oa::U32 count = static_cast<oa::U32>(inNewLogProbability.numElements());
	oa::Matrix perSample = oa::FnMatrix::empty(
		inNewLogProbability.getShape(), oa::ScalarType::Float32);
	struct Push { oa::U32 count; oa::F32 clipEpsilon; }
		push{.count = count, .clipEpsilon = inClipEpsilon};
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Write,
	};
	context.add(
		"RlPpoClip",
		{&inNewLogProbability, &inOldLogProbability, &inAdvantage, &perSample},
		access,
		&push,
		sizeof(push),
		oa::divCeil(count, 256U)
	);
	oa::Matrix loss = oa::FnMatrix::mean(perSample);
	if (oa::FnAutograd::isEnabled() && inNewLogProbability.requiresGrad()) {
		auto grad = oa::makeShared<GradPpoClippedPolicy>();
		grad->saveForBackward(
			inNewLogProbability, inOldLogProbability, inAdvantage);
		grad->setGraphInputs(oa::Vector<oa::Matrix>{inNewLogProbability, inOldLogProbability, inAdvantage});
		grad->clipEpsilon = inClipEpsilon;
		grad->sequenceNr_ = oa::FnAutograd::nextSeq();
		loss.mutAutograd().gradFn = grad;
	}
	auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnLoss::ppoClippedPolicy,
		{&inNewLogProbability, &inOldLogProbability, &inAdvantage},
		{&loss},
		{oa::OpAttribute::fromFloat(
			"clipEpsilon", inClipEpsilon)});
	if (not semantic.isOk()
		|| not attachSemanticOutput(loss, semantic.getValue(), 0U))
	{
		return {};
	}
	return loss;
}

oa::Matrix oa::FnLoss::ppoClippedPolicyBwd(
	const oa::Matrix& inNewLogProbability,
	const oa::Matrix& inOldLogProbability,
	const oa::Matrix& inAdvantage,
	oa::F32 inClipEpsilon) {
	if (!validatePolicyInputs(
		inNewLogProbability, inOldLogProbability, inAdvantage, inClipEpsilon)) {
		OaLogError(oa::LogComponent::Ml,
			"oa::FnLoss::ppoClippedPolicyBwd expects matching non-empty FP32 inputs and clip epsilon in (0,1)");
		return {};
	}
	auto& context = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(context);
	const oa::U32 count = static_cast<oa::U32>(inNewLogProbability.numElements());
	oa::Matrix grad = oa::FnMatrix::empty(inNewLogProbability.getShape(), oa::ScalarType::Float32);
	struct Push { oa::U32 count; oa::F32 clipEpsilon; }
		push{.count = count, .clipEpsilon = inClipEpsilon};
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Write,
	};
	context.add(
		"RlPpoClipBwd",
		{&inNewLogProbability, &inOldLogProbability, &inAdvantage, &grad},
		access,
		&push,
		sizeof(push),
		oa::divCeil(count, 256U));
	if (not lowering.commit(
		oa::detail::opRegistry::FnLoss::ppoClippedPolicyBwd,
		{&inNewLogProbability, &inOldLogProbability, &inAdvantage},
		{&grad},
		{oa::OpAttribute::fromFloat(
			"clipEpsilon", inClipEpsilon)}).isOk())
	{
		return {};
	}
	return grad;
}

oa::PpoLossResult oa::FnLoss::ppo(
	const oa::Matrix& inNewLogProbability,
	const oa::Matrix& inOldLogProbability,
	const oa::Matrix& inAdvantage,
	const oa::Matrix& inValue,
	const oa::Matrix& inTargetReturn,
	const oa::Matrix& inEntropy,
	const oa::PpoLossConfig& inConfig) {
	const oa::MatrixShape shape = inNewLogProbability.getShape();
	const bool configValid = oa::isFinite(inConfig.valueCoefficient)
		&& oa::isFinite(inConfig.entropyCoefficient)
		&& inConfig.valueCoefficient >= 0.0F
		&& inConfig.entropyCoefficient >= 0.0F;
	if (!configValid
		|| !validatePolicyInputs(inNewLogProbability, inOldLogProbability,
			inAdvantage, inConfig.clipEpsilon)
		|| !isMatchingF32(inValue, shape)
		|| !isMatchingF32(inTargetReturn, shape)
		|| !isMatchingF32(inEntropy, shape)) {
		OaLogError(oa::LogComponent::Ml,
			"oa::FnLoss::ppo expects matching non-empty FP32 rollout fields and non-negative finite coefficients");
		return {};
	}

	auto& context = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(context);
	oa::PpoLossResult result;
	result.policyLoss = ppoClippedPolicy(
		inNewLogProbability, inOldLogProbability,
		inAdvantage, inConfig.clipEpsilon);
	result.valueLoss = oa::FnLoss::mse(inValue, inTargetReturn);
	result.entropy = oa::FnMatrix::mean(inEntropy);
	result.totalLoss = oa::FnMatrix::sub(
		oa::FnMatrix::add(
			result.policyLoss,
			oa::FnMatrix::scale(result.valueLoss, inConfig.valueCoefficient)),
		oa::FnMatrix::scale(result.entropy, inConfig.entropyCoefficient));
	setLastName("ppo");
	auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnLoss::ppo,
		{
			&inNewLogProbability,
			&inOldLogProbability,
			&inAdvantage,
			&inValue,
			&inTargetReturn,
			&inEntropy,
		},
		{
			&result.policyLoss,
			&result.valueLoss,
			&result.entropy,
			&result.totalLoss,
		},
		{
			oa::OpAttribute::fromFloat(
				"clipEpsilon", inConfig.clipEpsilon),
			oa::OpAttribute::fromFloat(
				"valueCoefficient", inConfig.valueCoefficient),
			oa::OpAttribute::fromFloat(
				"entropyCoefficient", inConfig.entropyCoefficient),
		});
	if (not semantic.isOk()
		|| not attachSemanticOutput(
			result.policyLoss, semantic.getValue(), 0U)
		|| not attachSemanticOutput(
			result.valueLoss, semantic.getValue(), 1U)
		|| not attachSemanticOutput(
			result.entropy, semantic.getValue(), 2U)
		|| not attachSemanticOutput(
			result.totalLoss, semantic.getValue(), 3U))
	{
		return {};
	}
	return result;
}
