// OaFnLoss — Loss function implementations.
// Extracted from OaFnMatrix namespace for better organization.

#include <Oa/Ml/FnLoss.h>
#include <Oa/Ml/Autograd/Nodes.h>
#include <Oa/Core/Matrix.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>
#include <Oa/Core/BufferAccess.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Core/Validation.h>
#include <Oa/Core/Operation.h>
#include <Oa/Runtime/DispatchDesc.h>
#include "../Autograd/AutogradAttach.gen.h"
#include "../../Core/FnMatrix/Reduce/FnMatrixReduceLowering.h"

#include <algorithm>
#include <cassert>
#include <limits>

namespace {
thread_local const char* gLastLossName = nullptr;
constexpr OaU32 kPortableDispatchTileWidthX = 65535U;

OaStatus RecordLossDispatch(
	OaContext& InContext, OaStringView InKernel,
	OaSpan<const OaMatrix* const> InMatrices,
	OaSpan<OaBufferAccess> InAccess,
	const void* InPush, OaU32 InPushSize,
	OaU32 InGroupsX, OaU32 InGroupsY,
	const OaOperationContract* InContract = nullptr,
	OaSemanticOperationId InSemanticOperation = OaInvalidSemanticOperationId
) {
	OaMatrixDispatchDesc desc;
	desc.Dispatch.Kernel = InKernel;
	desc.Dispatch.Access = InAccess;
	desc.Dispatch.PushData = InPush;
	desc.Dispatch.PushSize = InPushSize;
	desc.Dispatch.GroupsX = InGroupsX;
	desc.Dispatch.GroupsY = InGroupsY;
	desc.Matrices = InMatrices;
	if (InContract) {
		desc.Dispatch.Operation = InContract->Name;
		desc.Dispatch.OperationContractHash = InContract->Hash;
	}
	if (InSemanticOperation != OaInvalidSemanticOperationId) {
		desc.Dispatch.SemanticOperations =
			OaSpan<const OaSemanticOperationId>(&InSemanticOperation, 1U);
	}
	return InContext.Record(desc);
}

bool CheckedShaderBytes(const OaMatrix& InMatrix, OaU64 InElements, OaU64& OutBytes) {
	constexpr OaU64 kShaderByteAddressSpace =
		static_cast<OaU64>(std::numeric_limits<OaU32>::max()) + 1U;
	const OaU64 elementBytes = static_cast<OaU64>(OaScalarSize(InMatrix.GetDtype()));
	if (elementBytes == 0 or InElements > kShaderByteAddressSpace / elementBytes) {
		return false;
	}
	OutBytes = InElements * elementBytes;
	return true;
}

bool HasDirectShaderStorage(const OaMatrix& InMatrix, OaU64 InLogicalBytes, bool InNeedsWordTail) {
	if (InMatrix.ByteOffset() != 0 or not InMatrix.GetStride().MatchesRowMajor(InMatrix.GetShape()))	{
		return false;
	}
	const OaVkBuffer buffer = InMatrix.GetVkBuffer();
	if (not buffer.Buffer or buffer.BindlessIndex == UINT32_MAX
		or buffer.Size < InLogicalBytes) {
		return false;
	}
	OaU64 descriptorBytes = InLogicalBytes;
	if (InNeedsWordTail) {
		if (descriptorBytes > std::numeric_limits<OaU64>::max() - 3U) return false;
		descriptorBytes = (descriptorBytes + 3U) & ~OaU64{3U};
	}
	return buffer.DescriptorRange() >= descriptorBytes;
}

bool ValidateCrossEntropyInputs(const char* InOperation,
	const OaMatrix& InLogits, const OaMatrix& InTargets)
{
	const bool shapeAndDtypeValid =
		InLogits.Rank() == 2 and InLogits.Size(0) > 0 and InLogits.Size(1) > 0
		and InTargets.Rank() == 1 and InTargets.Size(0) == InLogits.Size(0)
		and (InLogits.GetDtype() == OaScalarType::Float32
			or InLogits.GetDtype() == OaScalarType::BFloat16)
		and (InTargets.GetDtype() == OaScalarType::UInt8
			or InTargets.GetDtype() == OaScalarType::UInt32
			or InTargets.GetDtype() == OaScalarType::Int32);
	if (shapeAndDtypeValid) {
		const OaU64 rows = static_cast<OaU64>(InLogits.Size(0));
		const OaU64 classes = static_cast<OaU64>(InLogits.Size(1));
		const OaU64 maxShaderIndex = std::numeric_limits<OaU32>::max();
		OaU64 logitsElements = 0;
		if (rows <= maxShaderIndex and classes <= maxShaderIndex
			and rows <= std::numeric_limits<OaU64>::max() / classes)
		{
			logitsElements = rows * classes;
			OaU64 logitsBytes = 0;
			OaU64 targetsBytes = 0;
			const bool logitsNeedWordTail =
				InLogits.GetDtype() == OaScalarType::BFloat16;
			if (CheckedShaderBytes(InLogits, logitsElements, logitsBytes)
				and CheckedShaderBytes(InTargets, rows, targetsBytes)
				and rows <= (maxShaderIndex + OaU64{1U}) / sizeof(OaF32)
				and HasDirectShaderStorage(
					InLogits, logitsBytes, logitsNeedWordTail)
				// The UInt8 target path compiles to an 8-bit storage-buffer OpLoad
				// (UniformAndStorageBuffer8BitAccess), not an enclosing-word load.
				and HasDirectShaderStorage(InTargets, targetsBytes, false))
			{
				return true;
			}
		}
	}
	OA_LOG_ERROR(OaLogComponent::ML,
		"%s: expected packed zero-offset, fully backed, shader-byte-addressable rank-two Float32/BFloat16 logits and one rank-one UInt8/UInt32/Int32 target per row",
		InOperation
	);
	return false;
}
} // namespace

const char* OaFnLoss::LastName() {
	return gLastLossName;
}

void OaFnLoss::SetLastName(const char* InName) {
	gLastLossName = InName;
}

OaMatrix OaFnLoss::CrossEntropy(const OaMatrix& InLogits, const OaMatrix& InTargets) {
	gLastLossName = "cross_entropy";
	if (not ValidateCrossEntropyInputs("CrossEntropy", InLogits, InTargets)) return {};
	auto& ctx = OaContext::GetDefault();
	OaU32 batch = static_cast<OaU32>(InLogits.Size(0));
	OaU32 classes = static_cast<OaU32>(InLogits.Size(1));

	// Per-sample loss is always FP32 (dtype-invariant output): CrossEntropy computes
	// log_sum_exp in fp32 internally and the Sum/Mean reduction accumulates in fp32.
	// Storing it bf16 would round each ~ln(V) row loss to ~2-3 sig digits and feed a
	// lossy value into the reduction — the CrossEntropy.slang store uses OaStoreF32 to
	// match this dtype contract.
	OaMatrix perSample = OaFnMatrix::Empty(OaMatrixShape{batch}, OaScalarType::Float32);
	OaMatrix loss = OaFnMatrix::Empty(OaMatrixShape{1}, OaScalarType::Float32);
	if (not perSample.HasStorage() or not loss.HasStorage()) {
		OA_LOG_ERROR(OaLogComponent::ML,
			"CrossEntropy: failed to allocate per-sample or scalar output storage");
		return {};
	}
	const auto semantic = ctx.RecordOperation(
		OaOperationRegistry::CrossEntropy,
		{&InLogits, &InTargets}, {&loss});
	if (not semantic.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::ML,
			"CrossEntropy semantic recording failed: %s",
			semantic.GetStatus().GetMessage().c_str());
		return {};
	}

	// Pass targets directly. Non-negative Int32 and UInt32 class indices have
	// identical storage, so the shader's 32-bit path intentionally accepts both.
	OaU32 targetDtype = (InTargets.GetDtype() == OaScalarType::UInt8) ? 0U : 1U;
	struct { OaU32 Batch; OaU32 Classes; OaU32 TargetDtype; } push{.Batch = batch, .Classes = classes, .TargetDtype = targetDtype};
	const OaVkBuffer logitsBuf = InLogits.VkBuf_ ? *InLogits.VkBuf_ : OaVkBuffer{};
	const OaVkBuffer targetsBuf = InTargets.VkBuf_ ? *InTargets.VkBuf_ : OaVkBuffer{};
	const OaVkBuffer lossBuf = perSample.VkBuf_ ? *perSample.VkBuf_ : OaVkBuffer{};
	OA_LOG_DEBUG(OaLogComponent::Core, "CrossEntropy: batch=%u classes=%u logits_buf=%u targets_buf=%u loss_buf=%u",
		batch, classes, logitsBuf.BindlessIndex, targetsBuf.BindlessIndex, lossBuf.BindlessIndex);

	OaBufferAccess access[] = {
		OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
	const OaMatrix* matrices[] = {&InLogits, &InTargets, &perSample};
	const OaU32 groupsX = std::min(batch, kPortableDispatchTileWidthX);
	const OaU32 groupsY = (batch + kPortableDispatchTileWidthX - 1U)
		/ kPortableDispatchTileWidthX;
	const auto dispatch = RecordLossDispatch(
		ctx, "CrossEntropy", OaSpan<const OaMatrix* const>(matrices, 3U),
		OaSpan<OaBufferAccess>(access, 3U), &push, sizeof(push),
		groupsX, groupsY, &OaOperationRegistry::CrossEntropy,
		semantic.GetValue());
	if (not dispatch.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::ML,
			"CrossEntropy dispatch recording failed: %s",
			dispatch.GetMessage().c_str());
		return {};
	}

	const auto lowering = OaFnMatrixPrivate::LowerFullMean(
		ctx, perSample, loss,
		OaOperationRegistry::CrossEntropy, semantic.GetValue());
	if (not lowering.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::ML,
			"CrossEntropy mean lowering failed: %s",
			lowering.GetMessage().c_str());
		return {};
	}

	const auto attached = OaGeneratedAutogradAttach::OaFnLoss::CrossEntropy(
		loss, InLogits, InTargets, semantic.GetValue());
	if (not attached.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::ML,
			"CrossEntropy semantic autograd attachment failed: %s",
			attached.GetMessage().c_str());
		return {};
	}

	return loss;
}

OaMatrix OaFnLoss::CrossEntropyBwd(const OaMatrix& InLogits, const OaMatrix& InTargets) {
	gLastLossName = "cross_entropy";
	if (not ValidateCrossEntropyInputs("CrossEntropyBwd", InLogits, InTargets)) return {};
	auto& ctx = OaContext::GetDefault();
	OaU32 batch = static_cast<OaU32>(InLogits.Size(0));
	OaU32 classes = static_cast<OaU32>(InLogits.Size(1));

	OaMatrix gradLogits = OaFnMatrix::Empty(InLogits.GetShape(), InLogits.GetDtype());
	if (not gradLogits.HasStorage()) {
		OA_LOG_ERROR(OaLogComponent::ML,
			"CrossEntropyBwd: failed to allocate gradient storage");
		return {};
	}

	// UInt32 and non-negative Int32 share the shader's 32-bit index path.
	OaU32 targetDtype = (InTargets.GetDtype() == OaScalarType::UInt8) ? 0U : 1U;
	struct { OaU32 Batch; OaU32 Classes; OaU32 TargetDtype; } push{.Batch = batch, .Classes = classes, .TargetDtype = targetDtype};

	OaBufferAccess access[] = {
		OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
	const OaMatrix* matrices[] = {&InLogits, &InTargets, &gradLogits};
	const OaU32 groupsX = std::min(batch, kPortableDispatchTileWidthX);
	const OaU32 groupsY = (batch + kPortableDispatchTileWidthX - 1U)
		/ kPortableDispatchTileWidthX;
	const auto dispatch = RecordLossDispatch(
		ctx, "CrossEntropyBwd", OaSpan<const OaMatrix* const>(matrices, 3U),
		OaSpan<OaBufferAccess>(access, 3U), &push, sizeof(push),
		groupsX, groupsY);
	if (not dispatch.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::ML,
			"CrossEntropyBwd dispatch recording failed: %s",
			dispatch.GetMessage().c_str());
		return {};
	}

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
