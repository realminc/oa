#include <oa/ml/fnLoss.h>

#include <oa/core/bufferAccess.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/log.h>
#include <oa/core/op.h>
#include <oa/ml/autograd.h>
#include <oa/ml/fnLoss.h>
#include <oa/runtime/executionSession.h>

#include <cmath>
#include <limits>

oa::DqnLossResult oa::FnLoss::dqn(
	const oa::Matrix& inQ,
	const oa::Matrix& inAction,
	const oa::Matrix& inReward,
	const oa::Matrix& inNextQ,
	const oa::Matrix& inTerminated,
	const oa::Matrix& inTruncated,
	const oa::DqnLossConfig& inConfig) {
	const bool valid = inQ.rank() == 2 && inQ.size(0) > 0 && inQ.size(1) > 1
		&& inQ.getDtype() == oa::ScalarType::Float32
		&& inNextQ.getShape() == inQ.getShape()
		&& inNextQ.getDtype() == oa::ScalarType::Float32
		&& !inNextQ.requiresGrad()
		&& inAction.getShape() == oa::MatrixShape{inQ.size(0)}
		&& inAction.getDtype() == oa::ScalarType::Int32
		&& inReward.getShape() == oa::MatrixShape{inQ.size(0)}
		&& inReward.getDtype() == oa::ScalarType::Float32
		&& inTerminated.getShape() == inReward.getShape()
		&& inTerminated.getDtype() == oa::ScalarType::UInt8
		&& inTruncated.getShape() == inReward.getShape()
		&& inTruncated.getDtype() == oa::ScalarType::UInt8
		&& std::isfinite(inConfig.discount)
		&& inConfig.discount >= 0.0F && inConfig.discount <= 1.0F
		&& inQ.numElements() <= std::numeric_limits<oa::U32>::max();
	if (!valid) {
		OaLogError(oa::LogComponent::Ml,
			"oa::FnLoss::dqn expects FP32 Q/next-Q [B,A], Int32 action [B], FP32 reward [B], UInt8 boundaries [B], detached next-Q and discount in [0,1]");
		return {};
	}
	auto& context = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(context);
	const oa::U32 batch = static_cast<oa::U32>(inQ.size(0));
	const oa::U32 actions = static_cast<oa::U32>(inQ.size(1));
	oa::Matrix target = oa::FnMatrix::empty({static_cast<oa::I64>(batch)},
		oa::ScalarType::Float32);
	struct Push { oa::U32 Batch, actions; oa::F32 discount; }
		push{batch, actions, inConfig.discount};
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Write};
	context.add( "RlDqnTarget",
		{&inReward, &inNextQ, &inTerminated, &inTruncated, &target},
		access, &push, sizeof(push), (batch + 255U) / 256U);
	const oa::Matrix actionColumn = oa::FnMatrix::reshape(
		inAction, {static_cast<oa::I64>(batch), 1});
	const oa::Matrix selected = oa::FnMatrix::reshape(
		oa::FnMatrix::gatherLastDim(inQ, actionColumn),
		{static_cast<oa::I64>(batch)});
	oa::DqnLossResult result{
		.selectedQ = selected,
		.targetQ = target,
		.loss = oa::FnLoss::smoothL1(selected, target),
	};
	setLastName("dqn");
	auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnLoss::dqn,
		{
			&inQ,
			&inAction,
			&inReward,
			&inNextQ,
			&inTerminated,
			&inTruncated,
		},
		{&result.selectedQ, &result.targetQ, &result.loss},
		{oa::OpAttribute::fromFloat(
			"discount", inConfig.discount)});
	if (not semantic.isOk()) return {};
	if (auto grad = result.selectedQ.getGradFn()) {
		if (not oa::FnAutograd::attachSemantic(
			grad, semantic.getValue(), 0U).isOk())
		{
			return {};
		}
	}
	if (auto grad = result.loss.getGradFn()) {
		if (not oa::FnAutograd::attachSemantic(
			grad, semantic.getValue(), 2U).isOk())
		{
			return {};
		}
	}
	return result;
}
