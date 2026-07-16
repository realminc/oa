#include <Oa/Ml/FnMatrix.h>

#include <Oa/Core/BufferAccess.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Ml/Autograd.h>
#include <Oa/Runtime/Context.h>

#include <stdexcept>

namespace {

constexpr OaI64 FlashMaxSeqLen = 1024;

void ValidateFlashInputs(
	const OaMatrix& InQ, const OaMatrix& InK, const OaMatrix& InV) {
	if (InQ.Rank() != 3 || InK.GetShape() != InQ.GetShape() ||
		InV.GetShape() != InQ.GetShape()) {
		throw std::invalid_argument(
			"FlashAttentionCausal expects equal Q/K/V [batchHeads,sequence,headDim]");
	}
	if (InQ.Size(0) <= 0 || InQ.Size(1) <= 0 || InQ.Size(2) <= 0 ||
		InQ.Size(1) > FlashMaxSeqLen) {
		throw std::invalid_argument(
			"FlashAttentionCausal requires non-empty dimensions and sequence <= 1024");
	}
	if (InK.GetDtype() != InQ.GetDtype() || InV.GetDtype() != InQ.GetDtype()) {
		throw std::invalid_argument("FlashAttentionCausal requires one Q/K/V dtype");
	}
	if (InQ.GetDtype() != OaScalarType::Float32) {
		throw std::invalid_argument(
			"FlashAttentionCausal v1 is verified for Float32 storage only");
	}
}

class OaGradFlashAttentionCausal final : public OaGradNode {
public:
	OaF32 Scale = 1.0F;

	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		if (OutDIn.Empty()) return;
		auto gradients = OaFnMatrix::FlashAttentionCausalBwd(
			Saved_[0], Saved_[1], Saved_[2], Saved_[3], Saved_[4], InDOut, Scale);
		if (OutDIn.Size() > 0) OutDIn[0] = gradients.GradQ;
		if (OutDIn.Size() > 1) OutDIn[1] = gradients.GradK;
		if (OutDIn.Size() > 2) OutDIn[2] = gradients.GradV;
	}
};

} // namespace

OaMatrix OaFnMatrix::FlashAttentionCausal(
	const OaMatrix& InQ, const OaMatrix& InK, const OaMatrix& InV, OaF32 InScale) {
	ValidateFlashInputs(InQ, InK, InV);
	auto output = OaFnMatrix::Empty(InQ.GetShape(), InQ.GetDtype());
	// Log-sum-exp is a numerical statistic, not an activation payload. Keep it
	// FP32 even when Q/K/V use BF16 storage so backward probability recomputation
	// has the same stability contract as forward.
	auto logSumExp = OaFnMatrix::Empty(
		OaMatrixShape{InQ.Size(0), InQ.Size(1)}, OaScalarType::Float32);
	struct Push {
		OaU32 BatchHeads;
		OaU32 SeqLen;
		OaU32 HeadDim;
		OaF32 Scale;
	} push{
		static_cast<OaU32>(InQ.Size(0)), static_cast<OaU32>(InQ.Size(1)),
		static_cast<OaU32>(InQ.Size(2)), InScale,
	};
	OaBufferAccess access[] = {
		OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Write, OaBufferAccess::Write,
	};
	OaContext::GetDefault().Add("FlashCausal", {&InQ, &InK, &InV, &output, &logSumExp},
		access, &push, sizeof(push), push.BatchHeads * push.SeqLen, 1, 1);

	if (OaFnAutograd::IsEnabled() &&
		(InQ.RequiresGrad() || InK.RequiresGrad() || InV.RequiresGrad())) {
		auto grad = OaMakeSharedPtr<OaGradFlashAttentionCausal>();
		grad->Saved_ = {InQ, InK, InV, output, logSumExp};
		grad->SetGraphInputs({InQ, InK, InV});
		grad->SequenceNr_ = OaFnAutograd::NextSeq();
		grad->OutputShape_ = output.GetShape();
		grad->Scale = InScale;
		output.MutAutograd().GradFn = grad;
	}
	return output;
}

OaFnMatrix::OaFlashAttentionBwdResult OaFnMatrix::FlashAttentionCausalBwd(
	const OaMatrix& InQ, const OaMatrix& InK, const OaMatrix& InV,
	const OaMatrix& InOutput, const OaMatrix& InLogSumExp,
	const OaMatrix& InGradOutput, OaF32 InScale) {
	ValidateFlashInputs(InQ, InK, InV);
	if (InOutput.GetShape() != InQ.GetShape() ||
		InGradOutput.GetShape() != InQ.GetShape() ||
		InLogSumExp.GetShape() != OaMatrixShape{InQ.Size(0), InQ.Size(1)} ||
		InLogSumExp.GetDtype() != OaScalarType::Float32) {
		throw std::invalid_argument("FlashAttentionCausalBwd received incompatible saved tensors");
	}
	auto gradQ = OaFnMatrix::Empty(InQ.GetShape(), InQ.GetDtype());
	auto gradK = OaFnMatrix::Empty(InK.GetShape(), InK.GetDtype());
	auto gradV = OaFnMatrix::Empty(InV.GetShape(), InV.GetDtype());
	struct Push {
		OaU32 BatchHeads;
		OaU32 SeqLen;
		OaU32 HeadDim;
		OaF32 Scale;
	} push{
		static_cast<OaU32>(InQ.Size(0)), static_cast<OaU32>(InQ.Size(1)),
		static_cast<OaU32>(InQ.Size(2)), InScale,
	};
	OaBufferAccess qAccess[] = {
		OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Write,
	};
	auto& ctx = OaContext::GetDefault();
	ctx.Add("FlashCausalBwdQ",
		{&InQ, &InK, &InV, &InOutput, &InGradOutput, &InLogSumExp, &gradQ},
		qAccess, &push, sizeof(push), push.BatchHeads * push.SeqLen, 1, 1);
	OaBufferAccess kvAccess[] = {
		OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Write, OaBufferAccess::Write,
	};
	ctx.Add("FlashCausalBwdKV",
		{&InQ, &InK, &InV, &InOutput, &InGradOutput, &InLogSumExp, &gradK, &gradV},
		kvAccess, &push, sizeof(push), push.BatchHeads * push.SeqLen, 1, 1);
	return {.GradQ = gradQ, .GradK = gradK, .GradV = gradV};
}
