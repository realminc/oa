#include <Oa/Ml/Rl/FnLoss.h>

#include <Oa/Core/BufferAccess.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/Log.h>
#include <Oa/Ml/FnLoss.h>
#include <Oa/Runtime/Context.h>

#include <cmath>

namespace {

bool VectorF32(const OaMatrix& InMatrix, OaI64 InBatch) {
	return InMatrix.GetShape() == OaMatrixShape{InBatch}
		&& InMatrix.GetDtype() == OaScalarType::Float32;
}

OaMatrix Minimum(const OaMatrix& InA, const OaMatrix& InB) {
	return OaFnMatrix::Scale(OaFnMatrix::Sub(
		OaFnMatrix::Add(InA, InB),
		OaFnMatrix::Abs(OaFnMatrix::Sub(InA, InB))), 0.5F);
}

} // namespace

OaSacCriticLossResult OaFnLoss::SacCritic(
	const OaMatrix& InQ1,
	const OaMatrix& InQ2,
	const OaMatrix& InReward,
	const OaMatrix& InNextQ1,
	const OaMatrix& InNextQ2,
	const OaMatrix& InNextLogProbability,
	const OaMatrix& InTerminated,
	const OaMatrix& InTruncated,
	const OaSacLossConfig& InConfig) {
	const OaI64 batch = InQ1.Rank() == 1 ? InQ1.Size(0) : 0;
	const bool valid = batch > 0 && VectorF32(InQ1, batch)
		&& VectorF32(InQ2, batch) && VectorF32(InReward, batch)
		&& VectorF32(InNextQ1, batch) && VectorF32(InNextQ2, batch)
		&& VectorF32(InNextLogProbability, batch)
		&& !InNextQ1.RequiresGrad() && !InNextQ2.RequiresGrad()
		&& !InNextLogProbability.RequiresGrad()
		&& InTerminated.GetShape() == OaMatrixShape{batch}
		&& InTerminated.GetDtype() == OaScalarType::UInt8
		&& InTruncated.GetShape() == OaMatrixShape{batch}
		&& InTruncated.GetDtype() == OaScalarType::UInt8
		&& std::isfinite(InConfig.Discount)
		&& InConfig.Discount >= 0.0F && InConfig.Discount <= 1.0F
		&& std::isfinite(InConfig.EntropyCoefficient)
		&& InConfig.EntropyCoefficient >= 0.0F;
	if (!valid) {
		OA_LOG_ERROR(OaLogComponent::ML,	"OaFnLoss::SacCritic expects matching FP32 vectors, detached targets, UInt8 boundaries and valid coefficients");
		return {};
	}
	OaMatrix target = OaFnMatrix::Empty({batch}, OaScalarType::Float32);
	struct Push { OaU32 Batch; OaF32 Discount, EntropyCoefficient; }
		push{static_cast<OaU32>(batch), InConfig.Discount,
			InConfig.EntropyCoefficient};
	OaBufferAccess access[] = {
		OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Write
	};
	OaContext::GetDefault().Add("RlSacTarget",
		{&InReward, &InNextQ1, &InNextQ2, &InNextLogProbability,
		 &InTerminated, &InTruncated, &target},
		access, &push, sizeof(push),
		(static_cast<OaU32>(batch) + 255U) / 256U
	);
	OaSacCriticLossResult result;
	result.TargetQ = target;
	result.Q1Loss = OaFnLoss::Mse(InQ1, target);
	result.Q2Loss = OaFnLoss::Mse(InQ2, target);
	result.TotalLoss = OaFnMatrix::Add(result.Q1Loss, result.Q2Loss);
	SetLastName("sac_critic");
	return result;
}

OaMatrix OaFnLoss::SacActor(
	const OaMatrix& InQ1,
	const OaMatrix& InQ2,
	const OaMatrix& InLogProbability,
	OaF32 InEntropyCoefficient) {
	const OaI64 batch = InQ1.Rank() == 1 ? InQ1.Size(0) : 0;
	if (batch <= 0 || !VectorF32(InQ1, batch) || !VectorF32(InQ2, batch)
		|| !VectorF32(InLogProbability, batch)
		|| !std::isfinite(InEntropyCoefficient)
		|| InEntropyCoefficient < 0.0F) {
		OA_LOG_ERROR(OaLogComponent::ML,
			"OaFnLoss::SacActor expects matching FP32 vectors and non-negative entropy coefficient");
		return {};
	}
	SetLastName("sac_actor");
	return OaFnMatrix::Mean(OaFnMatrix::Sub(
		OaFnMatrix::Scale(InLogProbability, InEntropyCoefficient),
		Minimum(InQ1, InQ2)));
}
