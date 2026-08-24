#include <oa/ml/fnLoss.h>

#include <oa/core/bufferAccess.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/log.h>
#include <oa/core/op.h>
#include <oa/ml/autograd.h>
#include <oa/ml/fnLoss.h>
#include <oa/runtime/executionSession.h>

#include <cmath>

namespace {

bool vectorF32(const oa::Matrix& inMatrix, oa::I64 inBatch) {
	return inMatrix.getShape() == oa::MatrixShape{inBatch}
		&& inMatrix.getDtype() == oa::ScalarType::Float32;
}

oa::Matrix minimum(const oa::Matrix& inA, const oa::Matrix& inB) {
	return oa::FnMatrix::scale(oa::FnMatrix::sub(
		oa::FnMatrix::add(inA, inB),
		oa::FnMatrix::abs(oa::FnMatrix::sub(inA, inB))), 0.5F);
}

bool attachSemanticOutput(
	const oa::Matrix& inOutput,
	oa::U32 inOperation,
	oa::U32 inOutputIndex) {
	auto grad = inOutput.getGradFn();
	return not grad || oa::FnAutograd::attachSemantic(
		grad, inOperation, inOutputIndex).isOk();
}

} // namespace

oa::SacCriticLossResult oa::FnLoss::sacCritic(
	const oa::Matrix& inQ1,
	const oa::Matrix& inQ2,
	const oa::Matrix& inReward,
	const oa::Matrix& inNextQ1,
	const oa::Matrix& inNextQ2,
	const oa::Matrix& inNextLogProbability,
	const oa::Matrix& inTerminated,
	const oa::Matrix& inTruncated,
	const oa::SacLossConfig& inConfig) {
	const oa::I64 batch = inQ1.rank() == 1 ? inQ1.size(0) : 0;
	const bool valid = batch > 0 && vectorF32(inQ1, batch)
		&& vectorF32(inQ2, batch) && vectorF32(inReward, batch)
		&& vectorF32(inNextQ1, batch) && vectorF32(inNextQ2, batch)
		&& vectorF32(inNextLogProbability, batch)
		&& !inNextQ1.requiresGrad() && !inNextQ2.requiresGrad()
		&& !inNextLogProbability.requiresGrad()
		&& inTerminated.getShape() == oa::MatrixShape{batch}
		&& inTerminated.getDtype() == oa::ScalarType::UInt8
		&& inTruncated.getShape() == oa::MatrixShape{batch}
		&& inTruncated.getDtype() == oa::ScalarType::UInt8
		&& std::isfinite(inConfig.discount)
		&& inConfig.discount >= 0.0F && inConfig.discount <= 1.0F
		&& std::isfinite(inConfig.entropyCoefficient)
		&& inConfig.entropyCoefficient >= 0.0F;
	if (!valid) {
		OaLogError(oa::LogComponent::Ml,	"oa::FnLoss::sacCritic expects matching FP32 vectors, detached targets, UInt8 boundaries and valid coefficients");
		return {};
	}
	auto& context = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(context);
	oa::Matrix target = oa::FnMatrix::empty({batch}, oa::ScalarType::Float32);
	struct Push { oa::U32 Batch; oa::F32 discount, entropyCoefficient; }
		push{static_cast<oa::U32>(batch), inConfig.discount,
			inConfig.entropyCoefficient};
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Write
	};
	context.add( "RlSacTarget",
		{&inReward, &inNextQ1, &inNextQ2, &inNextLogProbability,
		 &inTerminated, &inTruncated, &target},
		access, &push, sizeof(push),
		(static_cast<oa::U32>(batch) + 255U) / 256U
	);
	oa::SacCriticLossResult result;
	result.targetQ = target;
	result.q1Loss = oa::FnLoss::mse(inQ1, target);
	result.q2Loss = oa::FnLoss::mse(inQ2, target);
	result.totalLoss = oa::FnMatrix::add(result.q1Loss, result.q2Loss);
	setLastName("sac_critic");
	auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnLoss::sacCritic,
		{
			&inQ1,
			&inQ2,
			&inReward,
			&inNextQ1,
			&inNextQ2,
			&inNextLogProbability,
			&inTerminated,
			&inTruncated,
		},
		{
			&result.targetQ,
			&result.q1Loss,
			&result.q2Loss,
			&result.totalLoss,
		},
		{
			oa::OpAttribute::fromFloat(
				"discount", inConfig.discount),
			oa::OpAttribute::fromFloat(
				"entropyCoefficient", inConfig.entropyCoefficient),
		});
	if (not semantic.isOk()
		|| not attachSemanticOutput(
			result.q1Loss, semantic.getValue(), 1U)
		|| not attachSemanticOutput(
			result.q2Loss, semantic.getValue(), 2U)
		|| not attachSemanticOutput(
			result.totalLoss, semantic.getValue(), 3U))
	{
		return {};
	}
	return result;
}

oa::Matrix oa::FnLoss::sacActor(
	const oa::Matrix& inQ1,
	const oa::Matrix& inQ2,
	const oa::Matrix& inLogProbability,
	oa::F32 inEntropyCoefficient) {
	const oa::I64 batch = inQ1.rank() == 1 ? inQ1.size(0) : 0;
	if (batch <= 0 || !vectorF32(inQ1, batch) || !vectorF32(inQ2, batch)
		|| !vectorF32(inLogProbability, batch)
		|| !std::isfinite(inEntropyCoefficient)
		|| inEntropyCoefficient < 0.0F) {
		OaLogError(oa::LogComponent::Ml,
			"oa::FnLoss::sacActor expects matching FP32 vectors and non-negative entropy coefficient");
		return {};
	}
	setLastName("sac_actor");
	auto& context = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(context);
	oa::Matrix loss = oa::FnMatrix::mean(oa::FnMatrix::sub(
		oa::FnMatrix::scale(inLogProbability, inEntropyCoefficient),
		minimum(inQ1, inQ2)));
	auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnLoss::sacActor,
		{&inQ1, &inQ2, &inLogProbability}, {&loss},
		{oa::OpAttribute::fromFloat(
			"entropyCoefficient", inEntropyCoefficient)});
	if (not semantic.isOk()
		|| not attachSemanticOutput(loss, semantic.getValue(), 0U))
	{
		return {};
	}
	return loss;
}
