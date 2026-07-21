#include <Oa/Ml/FnFlow.h>

#include <Oa/Core/BufferAccess.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Ml/Autograd.h>
#include <Oa/Runtime/Context.h>

#include <stdexcept>

namespace {

struct MaskedMsePush {
	OaU32 Count = 0;
	OaU32 Rank = 0;
	OaU32 PredictionStrides[OA_MAX_TENSOR_DIMS]{};
	OaU32 MaskDims[OA_MAX_TENSOR_DIMS]{};
	OaU32 MaskStrides[OA_MAX_TENSOR_DIMS]{};
};

MaskedMsePush MakeMaskedMsePush(
	const OaMatrix& InPrediction,
	const OaMatrix& InMask) {
	MaskedMsePush push;
	push.Count = static_cast<OaU32>(InPrediction.NumElements());
	push.Rank = static_cast<OaU32>(InPrediction.Rank());

	OaU32 stride = 1;
	for (OaI32 dim = InPrediction.Rank() - 1; dim >= 0; --dim) {
		push.PredictionStrides[dim] = stride;
		stride *= static_cast<OaU32>(InPrediction.Size(dim));
	}
	const OaI32 rankOffset = InPrediction.Rank() - InMask.Rank();
	for (OaI32 dim = 0; dim < InPrediction.Rank(); ++dim) {
		push.MaskDims[dim] = dim < rankOffset
			? 1U : static_cast<OaU32>(InMask.Size(dim - rankOffset));
	}
	stride = 1;
	for (OaI32 dim = InPrediction.Rank() - 1; dim >= 0; --dim) {
		push.MaskStrides[dim] = stride;
		stride *= push.MaskDims[dim];
	}
	return push;
}

OaMatrix MaskedMseBackward(
	const OaMatrix& InPrediction,
	const OaMatrix& InTarget,
	const OaMatrix& InMask,
	const OaMatrix& InDenominator,
	const OaMatrix& InUpstream) {
	auto gradient = OaFnMatrix::Empty(
		InPrediction.GetShape(), InPrediction.GetDtype());
	auto push = MakeMaskedMsePush(InPrediction, InMask);
	OaBufferAccess access[] = {
		OaBufferAccess::Read,
		OaBufferAccess::Read,
		OaBufferAccess::Read,
		OaBufferAccess::Read,
		OaBufferAccess::Read,
		OaBufferAccess::Write,
	};
	OaContext::GetDefault().Add("MaskedMseBwd",
		{&InPrediction, &InTarget, &InMask, &InDenominator,
			&InUpstream, &gradient},
		access, &push, sizeof(push), (push.Count + 255U) / 256U);
	return gradient;
}

class OaGradFlowMaskedMse final : public OaGradNode {
public:
	void Backward(
		const OaMatrix& InUpstream,
		OaVec<OaMatrix>& OutInputGrads) override {
		if (OutInputGrads.Empty()) return;
		OutInputGrads[0] = MaskedMseBackward(
			Saved_[0], Saved_[1], Saved_[2], Saved_[3], InUpstream);
	}
};

OaMatrix BroadcastTime(const OaMatrix& InTime, const OaMatrix& InState) {
	if (InTime.IsEmpty() || InState.IsEmpty()) {
		throw std::invalid_argument(
			"Flow matching requires non-empty state and time matrices");
	}

	OaMatrix time = InTime;
	if (InTime.Rank() == 1 && InState.Rank() > 1 &&
		(InTime.NumElements() == 1 || InTime.Size(0) == InState.Size(0))) {
		OaMatrixShape shape;
		shape.Rank = InState.Rank();
		for (OaI32 dim = 0; dim < shape.Rank; ++dim) shape[dim] = 1;
		if (InTime.NumElements() != 1) shape[0] = InState.Size(0);
		time = InTime.Reshape(shape);
	}

	auto broadcast = time.GetShape().Broadcast(InState.GetShape());
	if (!broadcast.IsOk() || *broadcast != InState.GetShape()) {
		throw std::invalid_argument(
			"Flow time must be scalar, [B], or broadcastable to the state");
	}
	return time;
}

} // namespace

OaFlowMatchBatch OaFnFlow::LinearMatch(
	const OaMatrix& InClean,
	const OaMatrix& InNoise,
	const OaMatrix& InTime) {
	if (InClean.GetShape() != InNoise.GetShape() ||
		InClean.GetDtype() != InNoise.GetDtype()) {
		throw std::invalid_argument(
			"Flow clean/noise matrices require identical shape and dtype");
	}
	auto time = BroadcastTime(InTime, InClean);
	if (time.GetDtype() != InClean.GetDtype()) {
		time = OaFnMatrix::Cast(time, InClean.GetDtype());
	}
	auto velocity = InNoise - InClean;
	return OaFlowMatchBatch{
		.State = InClean + (velocity * time),
		.Velocity = velocity,
	};
}

OaMatrix OaFnFlow::EulerStep(
	const OaMatrix& InState,
	const OaMatrix& InVelocity,
	OaF32 InDeltaTime) {
	if (InState.GetShape() != InVelocity.GetShape() ||
		InState.GetDtype() != InVelocity.GetDtype()) {
		throw std::invalid_argument(
			"Flow Euler state/velocity require identical shape and dtype");
	}
	return InState + (InVelocity * InDeltaTime);
}

OaMatrix OaFnFlow::MaskedMse(
	const OaMatrix& InPrediction,
	const OaMatrix& InTarget,
	const OaMatrix& InMask) {
	if (InPrediction.IsEmpty() || InTarget.IsEmpty() || InMask.IsEmpty()) {
		throw std::invalid_argument(
			"Flow masked MSE requires non-empty matrices");
	}
	if (InPrediction.GetShape() != InTarget.GetShape()
		|| InPrediction.GetDtype() != InTarget.GetDtype()) {
		throw std::invalid_argument(
			"Flow masked MSE prediction/target require identical shape and dtype");
	}
	auto broadcast = InMask.GetShape().Broadcast(InPrediction.GetShape());
	if (!broadcast.IsOk() || *broadcast != InPrediction.GetShape()) {
		throw std::invalid_argument(
			"Flow masked MSE mask must be broadcastable to prediction");
	}

	auto mask = InMask;
	if (mask.GetDtype() != InPrediction.GetDtype()) {
		mask = OaFnMatrix::Cast(mask, InPrediction.GetDtype());
	}
	auto loss = OaFnMatrix::Empty(
		OaMatrixShape{1}, InPrediction.GetDtype());
	auto denominator = OaFnMatrix::Empty(
		OaMatrixShape{1}, InPrediction.GetDtype());
	auto push = MakeMaskedMsePush(InPrediction, mask);
	OaBufferAccess access[] = {
		OaBufferAccess::Read,
		OaBufferAccess::Read,
		OaBufferAccess::Read,
		OaBufferAccess::Write,
		OaBufferAccess::Write,
	};
	OaContext::GetDefault().Add("MaskedMse",
		{&InPrediction, &InTarget, &mask, &loss, &denominator},
		access, &push, sizeof(push), 1);

	if (OaFnAutograd::IsEnabled() && InPrediction.RequiresGrad()) {
		auto gradFn = OaMakeSharedPtr<OaGradFlowMaskedMse>();
		gradFn->Saved_ = OaVec<OaMatrix>{
			InPrediction, InTarget, mask, denominator};
		gradFn->SetGraphInputs(OaVec<OaMatrix>{InPrediction});
		gradFn->SequenceNr_ = OaFnAutograd::NextSeq();
		gradFn->OutputShape_ = loss.GetShape();
		loss.MutAutograd().GradFn = gradFn;
	}
	return loss;
}
