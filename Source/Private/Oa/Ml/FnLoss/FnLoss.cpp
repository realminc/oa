// OaFnLoss — Loss function implementations.
// Extracted from OaFnMatrix namespace for better organization.

#include <Oa/Ml/FnLoss.h>
#include <Oa/Ml/Autograd.h>
#include <Oa/Core/Matrix.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>
#include <Oa/Core/BufferAccess.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Core/Validation.h>

#include <cassert>

namespace {
thread_local const char* gLastLossName = nullptr;
} // namespace

const char* OaFnLoss::LastName() {
	return gLastLossName;
}

void OaFnLoss::SetLastName(const char* InName) {
	gLastLossName = InName;
}

OaMatrix OaFnLoss::CrossEntropy(const OaMatrix& InLogits, const OaMatrix& InTargets) {
	gLastLossName = "cross_entropy";
	auto& ctx = OaContext::GetDefault();
	OaU32 batch = static_cast<OaU32>(InLogits.Size(0));
	OaU32 classes = static_cast<OaU32>(InLogits.Size(1));

	// Per-sample loss is always FP32 (dtype-invariant output): CrossEntropy computes
	// log_sum_exp in fp32 internally and the Sum/Mean reduction accumulates in fp32.
	// Storing it bf16 would round each ~ln(V) row loss to ~2-3 sig digits and feed a
	// lossy value into the reduction — the CrossEntropy.slang store uses OaStoreF32 to
	// match this dtype contract.
	OaMatrix perSample = OaFnMatrix::Empty(OaMatrixShape{batch}, OaScalarType::Float32);

	// Pass targets buffer directly - kernel handles UInt8/UInt32
	OaU32 targetDtype = (InTargets.GetDtype() == OaScalarType::UInt8) ? 0U : 1U;
	struct { OaU32 Batch; OaU32 Classes; OaU32 TargetDtype; } push{.Batch = batch, .Classes = classes, .TargetDtype = targetDtype};
	const OaVkBuffer logitsBuf = InLogits.VkBuf_ ? *InLogits.VkBuf_ : OaVkBuffer{};
	const OaVkBuffer targetsBuf = InTargets.VkBuf_ ? *InTargets.VkBuf_ : OaVkBuffer{};
	const OaVkBuffer lossBuf = perSample.VkBuf_ ? *perSample.VkBuf_ : OaVkBuffer{};
	OA_LOG_DEBUG(OaLogComponent::Core, "CrossEntropy: batch=%u classes=%u logits_buf=%u targets_buf=%u loss_buf=%u",
		batch, classes, logitsBuf.BindlessIndex, targetsBuf.BindlessIndex, lossBuf.BindlessIndex);

	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("CrossEntropy", {&InLogits, &InTargets, &perSample}, access, &push, sizeof(push), batch);

	OaMatrix loss = OaFnMatrix::Mean(perSample);

	// Tape attach — implicit autograd (OaAutograd.md §5). The schema flags
	// this op as manual_context so this is hand-written rather than autogen'd;
	// the Linear-family and elemwise attaches use the [ops.autograd] codegen.
	if (OaFnAutograd::IsEnabled() and InLogits.RequiresGrad()) {
		auto gradFn = OaMakeSharedPtr<OaGradCrossEntropy>();
		gradFn->Saved_         = OaVec<OaMatrix>{InLogits, InTargets};
		gradFn->SetGraphInputs(  OaVec<OaMatrix>{InLogits, InTargets});
		gradFn->SequenceNr_    = OaFnAutograd::NextSeq();
		loss.MutAutograd().GradFn = gradFn;
	}

	return loss;
}

OaMatrix OaFnLoss::CrossEntropyBwd(const OaMatrix& InLogits, const OaMatrix& InTargets) {
	gLastLossName = "cross_entropy";
	auto& ctx = OaContext::GetDefault();
	OaU32 batch = static_cast<OaU32>(InLogits.Size(0));
	OaU32 classes = static_cast<OaU32>(InLogits.Size(1));

	OaMatrix gradLogits = OaFnMatrix::Empty(InLogits.GetShape(), InLogits.GetDtype());

	// Pass targets buffer directly - kernel handles UInt8/UInt32
	OaU32 targetDtype = (InTargets.GetDtype() == OaScalarType::UInt8) ? 0U : 1U;
	struct { OaU32 Batch; OaU32 Classes; OaU32 TargetDtype; } push{.Batch = batch, .Classes = classes, .TargetDtype = targetDtype};

	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("CrossEntropyBwd", {&InLogits, &InTargets, &gradLogits}, access, &push, sizeof(push), batch);

	return gradLogits;
}

OaMatrix OaFnLoss::MaskedCrossEntropy(const OaMatrix& InLogits,
	const OaMatrix& InTargets, const OaMatrix& InMask, OaI32 InValidCount) {
	gLastLossName = "cross_entropy";
	OA_ASSERT(InLogits.Rank() == 2);
	OA_ASSERT(InTargets.NumElements() == InLogits.Size(0));
	OA_ASSERT(InMask.NumElements() == InLogits.Size(0));
	OA_ASSERT(InValidCount > 0 && InValidCount <= InLogits.Size(0));
	auto& ctx = OaContext::GetDefault();
	const OaU32 rows = static_cast<OaU32>(InLogits.Size(0));
	const OaU32 classes = static_cast<OaU32>(InLogits.Size(1));
	OaMatrix perSample = OaFnMatrix::Empty(OaMatrixShape{rows}, OaScalarType::Float32);
	const OaU32 targetDtype = (InTargets.GetDtype() == OaScalarType::UInt8) ? 0U : 1U;
	struct { OaU32 Rows, Classes, TargetDtype; } push{rows, classes, targetDtype};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("MaskedCrossEntropy", {&InLogits, &InTargets, &InMask, &perSample},
		access, &push, sizeof(push), rows);
	OaMatrix loss = OaFnMatrix::Scale(OaFnMatrix::Sum(perSample),
		1.0F / static_cast<OaF32>(InValidCount));
	if (OaFnAutograd::IsEnabled() and InLogits.RequiresGrad()) {
		auto gradFn = OaMakeSharedPtr<OaGradMaskedCrossEntropy>();
		gradFn->Saved_ = OaVec<OaMatrix>{InLogits, InTargets, InMask};
		gradFn->SetGraphInputs(OaVec<OaMatrix>{InLogits, InTargets, InMask});
		gradFn->ValidCount_ = InValidCount;
		gradFn->SequenceNr_ = OaFnAutograd::NextSeq();
		loss.MutAutograd().GradFn = gradFn;
	}
	return loss;
}

OaMatrix OaFnLoss::MaskedCrossEntropyBwd(const OaMatrix& InLogits,
	const OaMatrix& InTargets, const OaMatrix& InMask, OaI32 InValidCount) {
	gLastLossName = "cross_entropy";
	OA_ASSERT(InValidCount > 0 && InValidCount <= InLogits.Size(0));
	auto& ctx = OaContext::GetDefault();
	const OaU32 rows = static_cast<OaU32>(InLogits.Size(0));
	const OaU32 classes = static_cast<OaU32>(InLogits.Size(1));
	const OaU32 targetDtype = (InTargets.GetDtype() == OaScalarType::UInt8) ? 0U : 1U;
	OaMatrix gradLogits = OaFnMatrix::Empty(InLogits.GetShape(), InLogits.GetDtype());
	struct { OaU32 Rows, Classes, TargetDtype, ValidCount; }
		push{rows, classes, targetDtype, static_cast<OaU32>(InValidCount)};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("MaskedCrossEntropyBwd", {&InLogits, &InTargets, &InMask, &gradLogits},
		access, &push, sizeof(push), rows);
	return gradLogits;
}

// CrossEntropyLossGradBwd fused public wrapper removed (OaModule.md Phase 1).
// The CrossEntropyLossGradBwd kernel remains in the registry for Api3-style
// hand-wired graphs that call OaComputeGraph::Add("CrossEntropyLossGradBwd", ...).

// Huber and Nll losses were unimplemented stubs that asserted on call. They
// have been removed. SmoothL1 (Huber) is now implemented with fused kernels
// above. Nll remains pending a target-gather op.

// ─── SmoothL1 (Huber loss, beta=1.0) ───────────────────────────────────

OaMatrix OaFnLoss::SmoothL1(const OaMatrix& InA, const OaMatrix& InB) {
	gLastLossName = "smooth_l1";
	auto& ctx = OaContext::GetDefault();

	OaU32 count = static_cast<OaU32>(InA.NumElements());

	OaMatrix perElement = OaFnMatrix::Empty(InA.GetShape(), InA.GetDtype());

	struct { OaU32 Count; } push{.Count = count};

	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("SmoothL1", {&InA, &InB, &perElement}, access, &push, sizeof(push), OaDivCeil(count, 256U));

	OaMatrix loss = OaFnMatrix::Mean(perElement);

	if (OaFnAutograd::IsEnabled() and InA.RequiresGrad()) {
		auto gradFn = OaMakeSharedPtr<OaGradSmoothL1>();
		gradFn->Saved_      = OaVec<OaMatrix>{InA, InB};
		gradFn->SetGraphInputs(OaVec<OaMatrix>{InA, InB});
		gradFn->SequenceNr_  = OaFnAutograd::NextSeq();
		loss.MutAutograd().GradFn = gradFn;
	}

	return loss;
}

OaMatrix OaFnLoss::SmoothL1Bwd(const OaMatrix& InA, const OaMatrix& InB) {
	gLastLossName = "smooth_l1";
	auto& ctx = OaContext::GetDefault();

	OaU32 count = static_cast<OaU32>(InA.NumElements());

	OaMatrix gradA = OaFnMatrix::Empty(InA.GetShape(), InA.GetDtype());

	struct { OaU32 Count; } push{.Count = count};

	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("SmoothL1Bwd", {&InA, &InB, &gradA}, access, &push, sizeof(push), OaDivCeil(count, 256U));

	return gradA;
}

// ─── Mse (Mean Squared Error) ──────────────────────────────────────────

OaMatrix OaFnLoss::Mse(const OaMatrix& InA, const OaMatrix& InB) {
	gLastLossName = "mse";
	auto& ctx = OaContext::GetDefault();

	OaU32 count = static_cast<OaU32>(InA.NumElements());

	OaMatrix perElement = OaFnMatrix::Empty(InA.GetShape(), InA.GetDtype());

	struct { OaU32 Count; } push{.Count = count};

	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("Mse", {&InA, &InB, &perElement}, access, &push, sizeof(push), OaDivCeil(count, 256U));

	OaMatrix loss = OaFnMatrix::Mean(perElement);

	if (OaFnAutograd::IsEnabled() and InA.RequiresGrad()) {
		auto gradFn = OaMakeSharedPtr<OaGradMse>();
		gradFn->Saved_      = OaVec<OaMatrix>{InA, InB};
		gradFn->SetGraphInputs(OaVec<OaMatrix>{InA, InB});
		gradFn->SequenceNr_  = OaFnAutograd::NextSeq();
		loss.MutAutograd().GradFn = gradFn;
	}

	return loss;
}

OaMatrix OaFnLoss::MseBwd(const OaMatrix& InA, const OaMatrix& InB) {
	gLastLossName = "mse";
	auto& ctx = OaContext::GetDefault();

	OaU32 count = static_cast<OaU32>(InA.NumElements());

	OaMatrix gradA = OaFnMatrix::Empty(InA.GetShape(), InA.GetDtype());

	struct { OaU32 Count; } push{.Count = count};

	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("MseBwd", {&InA, &InB, &gradA}, access, &push, sizeof(push), OaDivCeil(count, 256U));

	return gradA;
}

// ─── L1 (Mean Absolute Error) ──────────────────────────────────────────

OaMatrix OaFnLoss::L1(const OaMatrix& InA, const OaMatrix& InB) {
	gLastLossName = "l1";
	auto& ctx = OaContext::GetDefault();

	OaU32 count = static_cast<OaU32>(InA.NumElements());

	OaMatrix perElement = OaFnMatrix::Empty(InA.GetShape(), InA.GetDtype());

	struct { OaU32 Count; } push{.Count = count};

	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("L1", {&InA, &InB, &perElement}, access, &push, sizeof(push), OaDivCeil(count, 256U));

	OaMatrix loss = OaFnMatrix::Mean(perElement);

	if (OaFnAutograd::IsEnabled() and InA.RequiresGrad()) {
		auto gradFn = OaMakeSharedPtr<OaGradL1>();
		gradFn->Saved_      = OaVec<OaMatrix>{InA, InB};
		gradFn->SetGraphInputs(OaVec<OaMatrix>{InA, InB});
		gradFn->SequenceNr_  = OaFnAutograd::NextSeq();
		loss.MutAutograd().GradFn = gradFn;
	}

	return loss;
}

OaMatrix OaFnLoss::L1Bwd(const OaMatrix& InA, const OaMatrix& InB) {
	gLastLossName = "l1";
	auto& ctx = OaContext::GetDefault();

	OaU32 count = static_cast<OaU32>(InA.NumElements());

	OaMatrix gradA = OaFnMatrix::Empty(InA.GetShape(), InA.GetDtype());

	struct { OaU32 Count; } push{.Count = count};

	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("L1Bwd", {&InA, &InB, &gradA}, access, &push, sizeof(push), OaDivCeil(count, 256U));

	return gradA;
}

// ─── Bce (Binary Cross-Entropy) ────────────────────────────────────────

OaMatrix OaFnLoss::Bce(const OaMatrix& InA, const OaMatrix& InB) {
	gLastLossName = "bce";
	auto& ctx = OaContext::GetDefault();

	OaU32 count = static_cast<OaU32>(InA.NumElements());

	OaMatrix perElement = OaFnMatrix::Empty(InA.GetShape(), InA.GetDtype());

	struct { OaU32 Count; } push{.Count = count};

	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("Bce", {&InA, &InB, &perElement}, access, &push, sizeof(push), OaDivCeil(count, 256U));

	OaMatrix loss = OaFnMatrix::Mean(perElement);

	if (OaFnAutograd::IsEnabled() and InA.RequiresGrad()) {
		auto gradFn = OaMakeSharedPtr<OaGradBce>();
		gradFn->Saved_      = OaVec<OaMatrix>{InA, InB};
		gradFn->SetGraphInputs(OaVec<OaMatrix>{InA, InB});
		gradFn->SequenceNr_  = OaFnAutograd::NextSeq();
		loss.MutAutograd().GradFn = gradFn;
	}

	return loss;
}

OaMatrix OaFnLoss::BceBwd(const OaMatrix& InA, const OaMatrix& InB) {
	gLastLossName = "bce";
	auto& ctx = OaContext::GetDefault();

	OaU32 count = static_cast<OaU32>(InA.NumElements());

	OaMatrix gradA = OaFnMatrix::Empty(InA.GetShape(), InA.GetDtype());

	struct { OaU32 Count; } push{.Count = count};

	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("BceBwd", {&InA, &InB, &gradA}, access, &push, sizeof(push), OaDivCeil(count, 256U));

	return gradA;
}
