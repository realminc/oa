// OaFnMatrix — Reductions and softmax.
//
// Sum, Mean, Max, Argmax, Softmax, LogSoftmax.

#include <Oa/Core/Matrix.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>
#include <Oa/Core/BufferAccess.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Dispatch.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/ComputeGraph.h>
#include <Oa/Core/Validation.h>
#include <Oa/Ml/Autograd.h>

#include <cassert>
static OaU32 DivCeil(OaU32 InA, OaU32 InB) { return (InA + InB - 1) / InB; }

static void AttachSumGrad_(const OaMatrix& InA, OaMatrix& OutSum) {
	if (OaFnAutograd::IsEnabled() and InA.RequiresGrad()) {
		auto gradFn = OaMakeSharedPtr<OaGradSum>();
		gradFn->Saved_ = OaVec<OaMatrix>{InA};
		gradFn->SetGraphInputs(OaVec<OaMatrix>{InA});
		gradFn->SequenceNr_ = OaFnAutograd::NextSeq();
		gradFn->OutputShape_ = OutSum.GetShape();
		OutSum.MutAutograd().GradFn = gradFn;
	}
}

// Reductions
OaMatrix OaFnMatrix::Sum(const OaMatrix& InA, OaI32 InDim) {
	OaI64 n = InA.NumElements();
	if (n == 1) { OaMatrix out = InA.Clone(); AttachSumGrad_(InA, out); return out; }

	auto& ctx = OaContext::GetDefault();

	OaI32 resolvedDim = InDim;
	if (resolvedDim >= 0 and resolvedDim < InA.Rank()) {
		OaI64 outerSize = 1;
		for (OaI32 i = 0; i < resolvedDim; ++i) outerSize *= InA.Size(i);
		OaI64 dimSize = InA.Size(resolvedDim);
		OaI64 innerSize = 1;
		for (OaI32 i = resolvedDim + 1; i < InA.Rank(); ++i) innerSize *= InA.Size(i);

		OaMatrixShape outShape;
		outShape.Rank = InA.Rank();
		for (OaI32 i = 0; i < InA.Rank(); ++i)
			outShape.Dims[i] = (i == resolvedDim) ? 1 : InA.Size(i);

		OaMatrix out = OaFnMatrix::Zeros(outShape, InA.Dtype_);
		OaU32 totalOut = static_cast<OaU32>(outerSize * innerSize);
		struct { OaU32 OuterSize; OaU32 DimSize; OaU32 InnerSize; } push{
			static_cast<OaU32>(outerSize), static_cast<OaU32>(dimSize), static_cast<OaU32>(innerSize)};
		OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
		ctx.Add("SumDim", {&InA, &out}, access, &push, sizeof(push), DivCeil(totalOut, 256));

		AttachSumGrad_(InA, out);
		return out;
	}

	OaMatrix out = OaFnMatrix::Zeros(OaMatrixShape{1}, InA.Dtype_);
	struct { OaU32 Count; } push{static_cast<OaU32>(n)};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("Sum", {&InA, &out}, access, &push, sizeof(push), 1);

	AttachSumGrad_(InA, out);
	return out;
}

OaMatrix OaFnMatrix::Mean(const OaMatrix& InA, OaI32 InDim) {
	OaMatrix s = Sum(InA, InDim);
	OaF32 count = (InDim >= 0 and InDim < InA.Rank())
		? static_cast<OaF32>(InA.Size(InDim))
		: static_cast<OaF32>(InA.NumElements());
	return Scale(s, 1.0f / count);
}

OaMatrix OaFnMatrix::Max(const OaMatrix& InA, OaI32 InDim) {
	(void)InDim;
	auto& ctx = OaContext::GetDefault();

	// Compute max value (full reduction → scalar)
	OaMatrix out = OaFnMatrix::Empty(OaMatrixShape{1}, InA.Dtype_);
	struct { OaU32 Count; } push{static_cast<OaU32>(InA.NumElements())};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("Max", {&InA, &out}, access, &push, sizeof(push), 1);

	// Differentiable: MaxBwd routes the upstream scalar grad to the element(s)
	// equal to the max, so the node must save both the input and the max value.
	if (OaFnAutograd::IsEnabled() and InA.RequiresGrad()) {
		auto gradFn = OaMakeSharedPtr<OaGradMax>();
		gradFn->Saved_ = OaVec<OaMatrix>{InA, out};
		gradFn->SetGraphInputs(OaVec<OaMatrix>{InA});
		gradFn->SequenceNr_ = OaFnAutograd::NextSeq();
		gradFn->OutputShape_ = out.GetShape();
		out.MutAutograd().GradFn = gradFn;
	}

	return out;
}

OaI64 OaFnMatrix::Argmax(const OaMatrix& InA, OaI32 InDim) {
	(void)InDim;
	auto& ctx = OaContext::GetDefault();
	OaMatrix out = OaFnMatrix::Empty(OaMatrixShape{1}, OaScalarType::UInt32);
	struct { OaU32 Count; } push{static_cast<OaU32>(InA.NumElements())};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("Argmax", {&InA, &out}, access, &push, sizeof(push), 1);
	auto status = ctx.Execute();
	assert(status.IsOk() && "OaContext::Execute failed before Argmax readback");
	auto syncStatus = ctx.Sync();
	assert(syncStatus.IsOk() && "OaContext::Sync failed before Argmax readback");
	OaU32 index = 0;
	auto copyStatus = OaFnMatrix::CopyToHost(out, &index, sizeof(index));
	assert(copyStatus.IsOk() && "Argmax readback failed");
	return static_cast<OaI64>(index);
}

OaMatrix OaFnMatrix::CategoricalAccuracyCount(
	const OaMatrix& InLogits, const OaMatrix& InLabels) {
	if (InLogits.Rank() < 2 or InLogits.NumElements() == 0 or
		InLabels.NumElements() != InLogits.NumElements() / InLogits.Size(InLogits.Rank() - 1) or
		(InLabels.GetDtype() != OaScalarType::UInt8 and
		 InLabels.GetDtype() != OaScalarType::UInt32 and
		 InLabels.GetDtype() != OaScalarType::Int32)) {
		OA_LOG_ERROR(OaLogComponent::ML,
			"CategoricalAccuracyCount: expected logits [...,C] and UInt8/UInt32/Int32 labels [...]");
		return {};
	}
	const OaU32 classes = static_cast<OaU32>(InLogits.Size(InLogits.Rank() - 1));
	const OaU32 rows = static_cast<OaU32>(InLabels.NumElements());
	const OaU32 labelType = InLabels.GetDtype() == OaScalarType::UInt8 ? 0u
		: (InLabels.GetDtype() == OaScalarType::UInt32 ? 1u : 2u);
	OaMatrix out = OaFnMatrix::Empty(OaMatrixShape{1}, OaScalarType::UInt32);
	struct { OaU32 Rows, Classes, LabelType; } push{rows, classes, labelType};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
	auto& ctx = OaContext::GetDefault();
	ctx.Add("CategoricalAccuracyCount", {&InLogits, &InLabels, &out},
		access, &push, sizeof(push), 1);
	return out;
}

OaMatrix OaFnMatrix::MaskedCategoricalAccuracyCount(
	const OaMatrix& InLogits, const OaMatrix& InLabels, const OaMatrix& InMask) {
	if (InLogits.Rank() < 2 or InLogits.NumElements() == 0 or
		InLabels.NumElements() != InLogits.NumElements() / InLogits.Size(InLogits.Rank() - 1) or
		InMask.NumElements() != InLabels.NumElements() or
		InMask.GetDtype() != InLogits.GetDtype() or
		(InLabels.GetDtype() != OaScalarType::UInt8 and
		 InLabels.GetDtype() != OaScalarType::UInt32 and
		 InLabels.GetDtype() != OaScalarType::Int32) or
		(InMask.GetDtype() != OaScalarType::Float32 and
		 InMask.GetDtype() != OaScalarType::BFloat16)) {
		OA_LOG_ERROR(OaLogComponent::ML,
			"MaskedCategoricalAccuracyCount: expected logits [...,C], integer labels [...], and a same-dtype floating mask [...]");
		return {};
	}
	const OaU32 classes = static_cast<OaU32>(InLogits.Size(InLogits.Rank() - 1));
	const OaU32 rows = static_cast<OaU32>(InLabels.NumElements());
	const OaU32 labelType = InLabels.GetDtype() == OaScalarType::UInt8 ? 0u
		: (InLabels.GetDtype() == OaScalarType::UInt32 ? 1u : 2u);
	OaMatrix out = OaFnMatrix::Empty(OaMatrixShape{1}, OaScalarType::UInt32);
	struct { OaU32 Rows, Classes, LabelType; } push{rows, classes, labelType};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Read, OaBufferAccess::Write};
	auto& ctx = OaContext::GetDefault();
	ctx.Add("MaskedCategoricalAccuracyCount", {&InLogits, &InLabels, &InMask, &out},
		access, &push, sizeof(push), 1);
	return out;
}

OaMatrix OaFnMatrix::Softmax(const OaMatrix& InA, OaI32 InDim) {
	auto& ctx = OaContext::GetDefault();
	OaMatrix out = OaFnMatrix::Empty(InA.Shape_, InA.Dtype_);

	OaU32 rows, cols;
	if (InA.Rank() == 2) {
		// InDim = -1 means last dimension (dim 1), InDim = 0 means first dimension
		if (InDim == -1 || InDim == 1) {
			// Softmax over columns (last dim)
			rows = static_cast<OaU32>(InA.Size(0));
			cols = static_cast<OaU32>(InA.Size(1));
		} else if (InDim == 0) {
			// Softmax over rows (first dim) - transpose, compute, transpose back
			OaMatrix transposed = OaFnMatrix::Transpose(InA, 0, 1);
			OaMatrix softmax_out = OaFnMatrix::Softmax(transposed, -1);
			return OaFnMatrix::Transpose(softmax_out, 0, 1);
		} else {
			rows = static_cast<OaU32>(InA.Size(0));
			cols = static_cast<OaU32>(InA.Size(1));
		}
	} else {
		rows = 1;
		cols = static_cast<OaU32>(InA.NumElements());
	}

	struct { OaU32 Rows; OaU32 Cols; } push{rows, cols};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("Softmax", {&InA, &out}, access, &push, sizeof(push), rows, 1, 1);

	// Attach autograd. Softmax previously attached NO grad node, so gradient was
	// silently dropped to zero through any standalone softmax (router gating,
	// hand-rolled attention). OaGradSoftmax saves the OUTPUT (SoftmaxBwd needs y).
	if (OaFnAutograd::IsEnabled() and InA.RequiresGrad()) {
		auto gradFn = OaMakeSharedPtr<OaGradSoftmax>();
		gradFn->Saved_ = OaVec<OaMatrix>{out};
		gradFn->SetGraphInputs(OaVec<OaMatrix>{InA});
		gradFn->SequenceNr_ = OaFnAutograd::NextSeq();
		gradFn->OutputShape_ = out.GetShape();
		out.MutAutograd().GradFn = gradFn;
	}

	return out;
}

OaMatrix OaFnMatrix::LogSoftmax(const OaMatrix& InA, OaI32 InDim) {
	return Log(Softmax(InA, InDim));
}


// DescribeSum / DescribeMax buffer-level helpers retired. Sum/Mean/Max record
// through OaContext.


// ═══════════════════════════════════════════════════════════════════════════
// GPU-NATIVE OPERATIONS (VK_EXT path - zero CPU overhead)
// ═══════════════════════════════════════════════════════════════════════════
