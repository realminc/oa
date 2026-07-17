#include <Oa/Ml/Rl/FnLoss.h>

#include <Oa/Core/BufferAccess.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/Log.h>
#include <Oa/Ml/FnLoss.h>
#include <Oa/Runtime/Context.h>

#include <cmath>
#include <limits>

OaDqnLossResult OaFnLoss::Dqn(
	const OaMatrix& InQ,
	const OaMatrix& InAction,
	const OaMatrix& InReward,
	const OaMatrix& InNextQ,
	const OaMatrix& InTerminated,
	const OaMatrix& InTruncated,
	const OaDqnLossConfig& InConfig) {
	const bool valid = InQ.Rank() == 2 && InQ.Size(0) > 0 && InQ.Size(1) > 1
		&& InQ.GetDtype() == OaScalarType::Float32
		&& InNextQ.GetShape() == InQ.GetShape()
		&& InNextQ.GetDtype() == OaScalarType::Float32
		&& !InNextQ.RequiresGrad()
		&& InAction.GetShape() == OaMatrixShape{InQ.Size(0)}
		&& InAction.GetDtype() == OaScalarType::Int32
		&& InReward.GetShape() == OaMatrixShape{InQ.Size(0)}
		&& InReward.GetDtype() == OaScalarType::Float32
		&& InTerminated.GetShape() == InReward.GetShape()
		&& InTerminated.GetDtype() == OaScalarType::UInt8
		&& InTruncated.GetShape() == InReward.GetShape()
		&& InTruncated.GetDtype() == OaScalarType::UInt8
		&& std::isfinite(InConfig.Discount)
		&& InConfig.Discount >= 0.0F && InConfig.Discount <= 1.0F
		&& InQ.NumElements() <= std::numeric_limits<OaU32>::max();
	if (!valid) {
		OA_LOG_ERROR(OaLogComponent::ML,
			"OaFnLoss::Dqn expects FP32 Q/next-Q [B,A], Int32 action [B], FP32 reward [B], UInt8 boundaries [B], detached next-Q and discount in [0,1]");
		return {};
	}
	const OaU32 batch = static_cast<OaU32>(InQ.Size(0));
	const OaU32 actions = static_cast<OaU32>(InQ.Size(1));
	OaMatrix target = OaFnMatrix::Empty({static_cast<OaI64>(batch)},
		OaScalarType::Float32);
	struct Push { OaU32 Batch, Actions; OaF32 Discount; }
		push{batch, actions, InConfig.Discount};
	OaBufferAccess access[] = {
		OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Read, OaBufferAccess::Write};
	OaContext::GetDefault().Add("RlDqnTarget",
		{&InReward, &InNextQ, &InTerminated, &InTruncated, &target},
		access, &push, sizeof(push), (batch + 255U) / 256U);
	const OaMatrix actionColumn = OaFnMatrix::Reshape(
		InAction, {static_cast<OaI64>(batch), 1});
	const OaMatrix selected = OaFnMatrix::Reshape(
		OaFnMatrix::GatherLastDim(InQ, actionColumn),
		{static_cast<OaI64>(batch)});
	OaDqnLossResult result{
		.SelectedQ = selected,
		.TargetQ = target,
		.Loss = OaFnLoss::SmoothL1(selected, target),
	};
	SetLastName("dqn");
	return result;
}
