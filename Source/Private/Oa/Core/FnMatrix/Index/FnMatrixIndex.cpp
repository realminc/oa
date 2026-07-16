// OaFnMatrix — Indexing, selection, reshape utilities, and loss.
//
// Gather, Slice, Split, Concat, Transpose (fn delegate), CrossEntropyLoss,
// Reshape (fn), RepeatInterleave, CausalMask (tensor), TopK,
// Equal, CompactRows, ScatterRows.

#include <Oa/Ml/FnMatrix.h>
#include <Oa/Ml/Autograd.h>
#include <Oa/Core/Matrix.h>
#include <Oa/Core/Memory.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>
#include <Oa/Core/BufferAccess.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Dispatch.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/ComputeGraph.h>

#include <Oa/Core/Validation.h>

#include <cassert>

static OaU32 DivCeil(OaU32 InA, OaU32 InB) { return (InA + InB - 1) / InB; }

#if not defined(NDEBUG) or defined(OA_ENABLE_VALIDATION)
static OaStatus OaValidateGather(
	const OaMatrix& InSelf, const OaMatrix& InIndices)
{
	OA_VALIDATE(InSelf.Rank() == 2, OaValidationSeverity::Error, OaLogComponent::ML,
		"Gather: InSelf must be 2D [vocab, embed_dim], got rank=%d", InSelf.Rank());
	OA_VALIDATE(InIndices.Rank() >= 1, OaValidationSeverity::Error, OaLogComponent::ML,
		"Gather: InIndices must have rank >= 1, got rank=%d", InIndices.Rank());
	return OaStatus::Ok();
}

static OaStatus OaValidateSlice(
	const OaMatrix& InSelf, OaI32 InDim, OaI64 InStart, OaI64 InEnd)
{
	OA_VALIDATE_BOUNDS(InDim, InSelf.Rank(), "Slice dim");
	OA_VALIDATE(InStart >= 0, OaValidationSeverity::Error, OaLogComponent::ML,
		"Slice: InStart=%lld must be >= 0", static_cast<OaI64>(InStart));
	OA_VALIDATE(InEnd > InStart, OaValidationSeverity::Error, OaLogComponent::ML,
		"Slice: InEnd=%lld must be > InStart=%lld",
		static_cast<OaI64>(InEnd), static_cast<OaI64>(InStart));
	OA_VALIDATE(InEnd <= InSelf.Size(InDim), OaValidationSeverity::Error, OaLogComponent::ML,
		"Slice: InEnd=%lld out of bounds for dim=%d size=%lld",
		static_cast<OaI64>(InEnd), InDim, InSelf.Size(InDim));
	return OaStatus::Ok();
}
#endif

// Indexing and selection
OaMatrix OaFnMatrix::Gather(const OaMatrix& InSelf, const OaMatrix& InIndices) {
	auto& ctx = OaContext::GetDefault();
#if not defined(NDEBUG) or defined(OA_ENABLE_VALIDATION)
	OA_ASSERT(OaValidateGather(InSelf, InIndices).IsOk());
#endif
	OaI64 numIdx = InIndices.NumElements();
	OaI64 rowSize = InSelf.Size(1);
	OaMatrix out = OaFnMatrix::Empty(OaMatrixShape{numIdx, rowSize}, InSelf.Dtype_);

	// Gather kernel now handles both UInt8 and UInt32 indices directly (no separate cast!)
	OaU32 indexDtype = 1;  // Default: UInt32
	if (InIndices.GetDtype() == OaScalarType::UInt8) {
		indexDtype = 0;  // UInt8 (byte embedding)
	}

	struct { OaU32 NumIndices; OaU32 RowSize; OaU32 IndexDtype; } push{
		static_cast<OaU32>(numIdx), static_cast<OaU32>(rowSize), indexDtype
	};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("Gather", {&InSelf, &InIndices, &out}, access, &push, sizeof(push),
		DivCeil(static_cast<OaU32>(numIdx * rowSize), 256));

	// Auto-attach the table-lookup gradient (embedding scatter-add) so a learned
	// embedding table is trained by simply calling Gather — same as every other
	// differentiable op here. Grad flows only into the table (InSelf); the integer
	// indices are non-differentiable. Previously each caller (OaByteEmbedding,
	// OaEmbedding) hand-wired this node; folding it in makes Gather GPU-native
	// differentiable and removes that duplication. The grad node uses GatherBwd
	// (a real GPU kernel), so there is no host fallback. Indices that don't come
	// from a grad-requiring table (e.g. the EMA VQ codebook) attach nothing.
	if (OaFnAutograd::IsEnabled() and InSelf.RequiresGrad()) {
		auto gradFn = OaMakeSharedPtr<OaGradGather>();
		gradFn->Saved_       = OaVec<OaMatrix>{InIndices, InSelf};
		gradFn->SetGraphInputs(  OaVec<OaMatrix>{InSelf, InIndices});
		gradFn->SequenceNr_  = OaFnAutograd::NextSeq();
		gradFn->OutputShape_ = out.GetShape();   // normalize viewed upstream back to gathered shape
		out.MutAutograd().GradFn = gradFn;
	}

	return out;
}

OaMatrix OaFnMatrix::GatherLastDim(const OaMatrix& InSelf, const OaMatrix& InIndices) {
	if (InSelf.Rank() != 2 or InIndices.Rank() != 2 or
		InIndices.GetDtype() != OaScalarType::Int32 or
		InSelf.Size(0) != InIndices.Size(0)) {
		OA_LOG_ERROR(OaLogComponent::ML,
			"GatherLastDim: expected Self[T,E] and Int32 Indices[T,K]");
		return {};
	}
	const OaU32 T = static_cast<OaU32>(InSelf.Size(0));
	const OaU32 E = static_cast<OaU32>(InSelf.Size(1));
	const OaU32 K = static_cast<OaU32>(InIndices.Size(1));
	auto out = OaFnMatrix::Empty(InIndices.GetShape(), InSelf.GetDtype());
	struct { OaU32 T, E, K; } push{T, E, K};
	OaBufferAccess access[] = {
		OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
	auto& ctx = OaContext::GetDefault();
	ctx.Add("GatherLastDim", {&InSelf, &InIndices, &out}, access, &push, sizeof(push),
		DivCeil(T * K, 256));
	if (OaFnAutograd::IsEnabled() and InSelf.RequiresGrad()) {
		auto gradFn = OaMakeSharedPtr<OaGradGatherLastDim>();
		gradFn->Saved_ = OaVec<OaMatrix>{InSelf, InIndices};
		gradFn->SetGraphInputs(OaVec<OaMatrix>{InSelf, InIndices});
		gradFn->SequenceNr_ = OaFnAutograd::NextSeq();
		gradFn->OutputShape_ = out.GetShape();
		out.MutAutograd().GradFn = gradFn;
	}
	return out;
}

OaMatrix OaFnMatrix::GatherLastDimBwd(const OaMatrix& InGradOut,
	const OaMatrix& InIndices, OaI32 InInputWidth) {
	if (InGradOut.Rank() != 2 or InIndices.GetShape() != InGradOut.GetShape() or
		InIndices.GetDtype() != OaScalarType::Int32 or InInputWidth <= 0) return {};
	const OaU32 T = static_cast<OaU32>(InGradOut.Size(0));
	const OaU32 K = static_cast<OaU32>(InGradOut.Size(1));
	const OaU32 E = static_cast<OaU32>(InInputWidth);
	auto out = OaFnMatrix::Empty(OaMatrixShape{T, E}, InGradOut.GetDtype());
	struct { OaU32 T, E, K; } push{T, E, K};
	OaBufferAccess access[] = {
		OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
	auto& ctx = OaContext::GetDefault();
	ctx.Add("GatherLastDimBwd", {&InGradOut, &InIndices, &out}, access, &push, sizeof(push),
		DivCeil(T * E, 256));
	return out;
}

OaMatrix OaFnMatrix::Slice(const OaMatrix& InSelf, OaI32 InDim, OaI64 InStart, OaI64 InEnd) {
	auto& ctx = OaContext::GetDefault();
	const OaI32 rank = InSelf.Rank();
	assert(rank >= 1 && rank <= 4 && "Slice supports rank 1 through 4");
#if not defined(NDEBUG) or defined(OA_ENABLE_VALIDATION)
	OA_ASSERT(OaValidateSlice(InSelf, InDim, InStart, InEnd).IsOk());
#endif
	(void)rank;
	OaI64 sliceLen = InEnd - InStart;
	OaMatrixShape outShape = InSelf.Shape_;
	outShape[InDim] = sliceLen;

	OaMatrix out = OaFnMatrix::Empty(outShape, InSelf.Dtype_);
	struct Push {
		OaU32 Count, Rank, Dim, SrcStart, DstStart;
		OaU32 SrcDims[4];
		OaU32 DstDims[4];
		OaU32 CopyDims[4];
	} push{};
	push.Count = static_cast<OaU32>(out.NumElements());
	push.Rank = static_cast<OaU32>(rank);
	push.Dim = static_cast<OaU32>(InDim);
	push.SrcStart = static_cast<OaU32>(InStart);
	for (OaI32 d = 0; d < rank; ++d) {
		push.SrcDims[d] = static_cast<OaU32>(InSelf.Size(d));
		push.DstDims[d] = static_cast<OaU32>(out.Size(d));
		push.CopyDims[d] = static_cast<OaU32>(out.Size(d));
	}
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("MatrixCopyRegion", {&InSelf, &out}, access, &push, sizeof(push),
		DivCeil(push.Count, 256));

	if (OaFnAutograd::IsEnabled() and InSelf.RequiresGrad()) {
		auto gradFn = OaMakeSharedPtr<OaGradSlice>();
		gradFn->InputShape_ = InSelf.GetShape();
		gradFn->Dim_ = InDim;
		gradFn->Start_ = InStart;
		gradFn->End_ = InEnd;
		gradFn->SetGraphInputs(OaVec<OaMatrix>{InSelf});
		gradFn->SequenceNr_ = OaFnAutograd::NextSeq();
		gradFn->OutputShape_ = out.GetShape();  // for tape dout normalization (Mamba3 in_proj splits + dd* flats + group broadcasts go through Slice + Reshape)
		out.MutAutograd().GradFn = gradFn;
	}

	return out;
}

OaMatrix OaFnMatrix::SliceBwd(
	OaMatrixShape InInputShape, OaI32 InDim, OaI64 InStart, OaI64 InEnd,
	const OaMatrix& InDOut) {
	auto& ctx = OaContext::GetDefault();
	const OaI32 rank = InInputShape.Rank;
	assert(rank >= 1 && rank <= 4 && "SliceBwd supports rank 1 through 4");
	assert(InDim >= 0 && InDim < rank && "SliceBwd dim out of range");
	assert(InStart >= 0 && InEnd > InStart && InEnd <= InInputShape[InDim] &&
		"SliceBwd start/end out of range");
	(void)rank;

	auto out = OaFnMatrix::Zeros(InInputShape, InDOut.GetDtype());
	struct Push {
		OaU32 Count, Rank, Dim, SrcStart, DstStart;
		OaU32 SrcDims[4];
		OaU32 DstDims[4];
		OaU32 CopyDims[4];
	} push{};
	push.Count = static_cast<OaU32>(InDOut.NumElements());
	push.Rank = static_cast<OaU32>(rank);
	push.Dim = static_cast<OaU32>(InDim);
	push.SrcStart = 0;
	push.DstStart = static_cast<OaU32>(InStart);
	for (OaI32 d = 0; d < rank; ++d) {
		push.SrcDims[d] = static_cast<OaU32>(InDOut.Size(d));
		push.DstDims[d] = static_cast<OaU32>(InInputShape[d]);
		push.CopyDims[d] = static_cast<OaU32>(InDOut.Size(d));
	}
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("MatrixCopyRegion", {&InDOut, &out}, access, &push, sizeof(push),
		DivCeil(push.Count, 256));
	return out;
}

OaVec<OaMatrix> OaFnMatrix::Split(
	const OaMatrix& InSelf, OaSpan<OaI64> InSizes, OaI32 InDim
) {
	assert(InSizes.Size() > 0 && "Split requires at least one size");
	assert(InDim >= 0 && InDim < InSelf.GetShape().Rank && "Invalid split dimension");

	// Verify sizes sum to dimension size
	OaI64 totalSize = 0;
	for (OaI64 size : InSizes) {
		totalSize += size;
	}
	assert(totalSize == InSelf.Size(InDim) && "Split sizes must sum to dimension size");
	(void)totalSize;

	// Create output tensors by slicing
	OaVec<OaMatrix> outputs;
	outputs.Reserve(InSizes.Size());

	OaI64 offset = 0;
	for (OaI64 size : InSizes) {
		auto slice = OaFnMatrix::Slice(InSelf, InDim, offset, offset + size);
		outputs.PushBack(slice);
		offset += size;
	}

	// calls above (one per piece, with correct Start offset and InputShape). Those Slice
	// GradFns handle backward correctly per-piece. Do NOT overwrite them with a shared
	// Split GradFn — a shared GradFn is deduplicated by the topo-sort, so only one
	// output's gradient would ever flow back to InSelf.

	return outputs;
}

OaMatrix OaFnMatrix::Concat(OaSpan<OaMatrix> InInputs, OaI32 InDim) {
	auto& ctx = OaContext::GetDefault();
	assert(InInputs.Size() > 0 && "Concat requires at least one input");
	assert(InDim >= 0 && InDim < InInputs[0].Rank() && "Concat dimension out of range");
	assert(InInputs[0].Rank() >= 1 && InInputs[0].Rank() <= 4 &&
		"Concat supports rank 1 through 4");

	// Calculate output shape
	OaMatrixShape outShape = InInputs[0].GetShape();
	OaI64 totalSize = 0;
	for (const auto& input : InInputs) {
		assert(input.GetShape().Rank == outShape.Rank && "All inputs must have same rank");
		assert(input.GetDtype() == InInputs[0].GetDtype() &&
			"Concat inputs must have the same dtype");
		for (OaI32 d = 0; d < outShape.Rank; ++d) {
			assert((d == InDim || input.Size(d) == outShape[d]) &&
				"Concat non-concatenated dimensions must match");
		}
		totalSize += input.Size(InDim);
	}
	outShape[InDim] = totalSize;

	// Allocate output
	OaMatrix out = OaFnMatrix::Empty(outShape, InInputs[0].GetDtype());

	OaU32 dstStart = 0;
	for (const auto& input : InInputs) {
		struct Push {
			OaU32 Count, Rank, Dim, SrcStart, DstStart;
			OaU32 SrcDims[4];
			OaU32 DstDims[4];
			OaU32 CopyDims[4];
		} push{};
		push.Count = static_cast<OaU32>(input.NumElements());
		push.Rank = static_cast<OaU32>(outShape.Rank);
		push.Dim = static_cast<OaU32>(InDim);
		push.DstStart = dstStart;
		for (OaI32 d = 0; d < outShape.Rank; ++d) {
			push.SrcDims[d] = static_cast<OaU32>(input.Size(d));
			push.DstDims[d] = static_cast<OaU32>(out.Size(d));
			push.CopyDims[d] = static_cast<OaU32>(input.Size(d));
		}
		OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
		ctx.Add("MatrixCopyRegion", {&input, &out}, access, &push, sizeof(push),
			DivCeil(push.Count, 256));
		dstStart += static_cast<OaU32>(input.Size(InDim));
	}

	if (OaFnAutograd::IsEnabled()) {
		bool anyGrad = false;
		for (const auto& input : InInputs) {
			if (input.RequiresGrad()) { anyGrad = true; break; }
		}
		if (anyGrad) {
			auto gradFn = OaMakeSharedPtr<OaGradConcat>();
			gradFn->Dim_ = InDim;
			for (const auto& input : InInputs) {
				gradFn->Sizes_.PushBack(input.Size(InDim));
			}
			OaVec<OaMatrix> inputs;
			inputs.Reserve(InInputs.Size());
			for (const auto& input : InInputs) inputs.PushBack(input);
			gradFn->SetGraphInputs(inputs);
			gradFn->SequenceNr_ = OaFnAutograd::NextSeq();
			gradFn->OutputShape_ = out.GetShape();
			out.MutAutograd().GradFn = gradFn;
		}
	}

	return out;
}

OaMatrix OaFnMatrix::Transpose(const OaMatrix& InA, OaI32 InDim0, OaI32 InDim1) {
	OaMatrix out = InA.Transpose(InDim0, InDim1);
	if (OaFnAutograd::IsEnabled() and InA.RequiresGrad()) {
		// Transpose is a metadata-only view and therefore initially aliases InA's
		// autograd metadata. Attaching the transpose node without detaching would
		// overwrite InA's producer and sever the graph (notably K -> K^T in
		// attention). This is the same invariant enforced by Reshape below.
		out.DetachForGradAttach(true);
		auto _gradFn = OaMakeSharedPtr<OaGradTranspose>();
		_gradFn->Dim0_ = InDim0;
		_gradFn->Dim1_ = InDim1;
		_gradFn->SetGraphInputs(OaVec<OaMatrix>{InA});
		_gradFn->SequenceNr_ = OaFnAutograd::NextSeq();
		_gradFn->OutputShape_ = out.GetShape();
		out.MutAutograd().GradFn = _gradFn;
	}
	return out;
}

// CrossEntropyLoss moved to Source/Private/Oa/Ml/Loss/FnLoss.cpp (OaFnLoss namespace)
// Backward compatibility alias in Source/Public/Oa/Ml/FnMatrix.h forwards to OaFnLoss::CrossEntropyLoss

// Reshape, RepeatInterleave, CausalMask (tensor), TopK, Equal, CompactRows/ScatterRows
OaMatrix OaFnMatrix::Reshape(const OaMatrix& InA, OaMatrixShape InShape) {
	OaI64 total = InA.NumElements();
	OaI64 product = 1;
	OaI32 inferDim = -1;
	for (OaI32 d = 0; d < InShape.Rank; ++d) {
		if (InShape[d] == -1) { inferDim = d; }
		else { product *= InShape[d]; }
	}
	OaMatrixShape resolved = InShape;
	if (inferDim >= 0) {
		resolved[inferDim] = (product > 0) ? (total / product) : 0;
	}
	OaMatrix out = InA.Reshape(resolved);
	if (OaFnAutograd::IsEnabled() and InA.RequiresGrad()) {
		// Reshape returns a VIEW that aliases InA's autograd meta (shared_ptr).
		// Attaching a gradfn to the view would clobber InA's gradfn (breaking
		// topo collection — the walk stops at the reshape's own node) and corrupt
		// InA's leaf-ness (IsLeaf() flips false → leaf grad never accumulates).
		// Detach to an independent meta before attaching the reshape gradfn.
		out.DetachForGradAttach(true);
		auto gradFn = OaMakeSharedPtr<OaGradReshape>();
		gradFn->InputShape_ = InA.GetShape();
		gradFn->SetGraphInputs(OaVec<OaMatrix>{InA});
		gradFn->SequenceNr_ = OaFnAutograd::NextSeq();
		gradFn->OutputShape_ = out.GetShape();
		out.MutAutograd().GradFn = gradFn;
	}
	return out;
}

OaMatrix OaFnMatrix::RepeatInterleave(const OaMatrix& InA, OaI32 InRepeats, OaI32 InDim) {
	auto& ctx = OaContext::GetDefault();
	const OaMatrixShape& inShape = InA.GetShape();
	OaMatrixShape outShape = inShape;
	outShape[InDim] *= InRepeats;
	OaMatrix out = OaFnMatrix::Empty(outShape, InA.Dtype_);

	const OaI64 numOut = outShape.NumElements();
	const OaI32 rank = inShape.Rank;

	// Build push constants with shape and stride info
	struct {
		OaU32 Count;
		OaU32 Repeats;
		OaU32 Dim;
		OaU32 Rank;
		OaU32 InShape[4];
		OaU32 InStride[4];
	} push{};

	push.Count = static_cast<OaU32>(numOut);
	push.Repeats = static_cast<OaU32>(InRepeats);
	push.Dim = static_cast<OaU32>(InDim);
	push.Rank = static_cast<OaU32>(rank);

	OaStride inStride = OaStride::RowMajor(inShape);
	for (OaI32 d = 0; d < rank and d < 4; ++d) {
		push.InShape[d] = static_cast<OaU32>(inShape[d]);
		push.InStride[d] = static_cast<OaU32>(inStride.StepElements(d));
	}

	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("RepeatInterleave", {&InA, &out}, access, &push, sizeof(push),
		DivCeil(static_cast<OaU32>(numOut), 256));

	if (OaFnAutograd::IsEnabled() and InA.RequiresGrad()) {
		auto gradFn = OaMakeSharedPtr<OaGradRepeatInterleave>();
		gradFn->Repeats_ = InRepeats;
		gradFn->Dim_ = InDim;
		gradFn->Saved_ = OaVec<OaMatrix>{InA};
		gradFn->SetGraphInputs(OaVec<OaMatrix>{InA});
		gradFn->SequenceNr_ = OaFnAutograd::NextSeq();
		gradFn->OutputShape_ = out.GetShape();
		out.MutAutograd().GradFn = gradFn;
	}

	return out;
}

OaMatrix OaFnMatrix::RepeatInterleaveBwd(const OaMatrix& InGradOut, OaMatrixShape InInputShape, OaI32 InRepeats, OaI32 InDim) {
	auto& ctx = OaContext::GetDefault();
	OaMatrix gradIn = OaFnMatrix::Empty(InInputShape, InGradOut.Dtype_);

	const OaI64 numIn = InInputShape.NumElements();
	const OaI32 rank = InInputShape.Rank;

	// Output shape (the shape of InGradOut)
	OaMatrixShape outShape = InInputShape;
	outShape[InDim] *= InRepeats;

	struct {
		OaU32 Count;
		OaU32 Repeats;
		OaU32 Dim;
		OaU32 Rank;
		OaU32 OutShape[4];
		OaU32 InStride[4];
	} push{};

	push.Count = static_cast<OaU32>(numIn);
	push.Repeats = static_cast<OaU32>(InRepeats);
	push.Dim = static_cast<OaU32>(InDim);
	push.Rank = static_cast<OaU32>(rank);

	OaStride inStride = OaStride::RowMajor(InInputShape);
	for (OaI32 d = 0; d < rank and d < 4; ++d) {
		push.OutShape[d] = static_cast<OaU32>(outShape[d]);
		push.InStride[d] = static_cast<OaU32>(inStride.StepElements(d));
	}

	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("RepeatInterleaveBwd", {&InGradOut, &gradIn}, access, &push, sizeof(push),
		DivCeil(static_cast<OaU32>(numIn), 256));

	return gradIn;
}

OaMatrix OaFnMatrix::CausalMask(const OaMatrix& InScores) {
	const OaI32 rank = InScores.Rank();
	if (rank < 2) {
		OA_LOG_ERROR(OaLogComponent::ML, "CausalMask(scores): rank must be >= 2");
		return {};
	}
	const OaI64 Tq = InScores.Size(rank - 2);
	const OaI64 Tk = InScores.Size(rank - 1);
	OaMatrix out = OaFnMatrix::Empty(InScores.GetShape(), InScores.GetDtype());
	const OaU32 count = static_cast<OaU32>(InScores.NumElements());
	struct { OaU32 count, Tq, Tk; } push{
		count, static_cast<OaU32>(Tq), static_cast<OaU32>(Tk)};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	auto& ctx = OaContext::GetDefault();
	ctx.Add("CausalMaskApply", {&InScores, &out}, access, &push, sizeof(push),
		DivCeil(count, 256));
	if (OaFnAutograd::IsEnabled() and InScores.RequiresGrad()) {
		auto gradFn = OaMakeSharedPtr<OaGradCausalMask>();
		gradFn->Saved_ = OaVec<OaMatrix>{InScores};
		gradFn->SetGraphInputs(OaVec<OaMatrix>{InScores});
		gradFn->SequenceNr_ = OaFnAutograd::NextSeq();
		gradFn->OutputShape_ = out.GetShape();
		out.MutAutograd().GradFn = gradFn;
	}
	return out;
}

OaMatrix OaFnMatrix::CausalMaskBwd(const OaMatrix& InGradOut) {
	const OaI32 rank = InGradOut.Rank();
	if (rank < 2) return {};
	const OaU32 Tq = static_cast<OaU32>(InGradOut.Size(rank - 2));
	const OaU32 Tk = static_cast<OaU32>(InGradOut.Size(rank - 1));
	const OaU32 count = static_cast<OaU32>(InGradOut.NumElements());
	OaMatrix out = OaFnMatrix::Empty(InGradOut.GetShape(), InGradOut.GetDtype());
	struct { OaU32 count, Tq, Tk; } push{count, Tq, Tk};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	auto& ctx = OaContext::GetDefault();
	ctx.Add("CausalMaskBwd", {&InGradOut, &out}, access, &push, sizeof(push),
		DivCeil(count, 256));
	return out;
}

// Note: OaFnMatrix::TopK() is implemented in DeviceMatrixFn.cpp

OaMatrix OaFnMatrix::Equal(const OaMatrix& InA, OaF32 InValue) {
	if (InA.GetDtype() != OaScalarType::Float32 and
		InA.GetDtype() != OaScalarType::BFloat16 and
		InA.GetDtype() != OaScalarType::Int32) {
		OA_LOG_ERROR(OaLogComponent::ML, "Equal: supports Float32, BFloat16 and Int32");
		return {};
	}
	OaMatrix out = OaFnMatrix::Empty(InA.GetShape(), OaScalarType::Float32);
	const OaU32 n = static_cast<OaU32>(InA.NumElements());
	struct { OaU32 n; OaU32 inputType; OaF32 value; } push{
		n,
		InA.GetDtype() == OaScalarType::Int32 ? 1u : 0u,
		InValue,
	};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	auto& ctx = OaContext::GetDefault();
	ctx.Add("Equal", {&InA, &out}, access, &push, sizeof(push), DivCeil(n, 256));
	return out;  // Comparison is intentionally non-differentiable.
}

OaMatrix OaFnMatrix::GreaterEqual(const OaMatrix& InA, OaF32 InValue) {
	if (InA.GetDtype() != OaScalarType::Float32 and
		InA.GetDtype() != OaScalarType::BFloat16) {
		OA_LOG_ERROR(OaLogComponent::ML, "GreaterEqual: supports Float32 and BFloat16");
		return {};
	}
	OaMatrix out = OaFnMatrix::Empty(InA.GetShape(), OaScalarType::Float32);
	const OaU32 n = static_cast<OaU32>(InA.NumElements());
	struct { OaU32 n; OaF32 value; } push{n, InValue};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	auto& ctx = OaContext::GetDefault();
	ctx.Add("GreaterEqual", {&InA, &out}, access, &push, sizeof(push), DivCeil(n, 256));
	return out;  // Comparison is intentionally non-differentiable.
}

OaFnMatrix::OaMoeExpertPlan OaFnMatrix::MoeExpertPlan(
	const OaMatrix& InExpertIndices, OaI32 InNumExperts) {
	if (InExpertIndices.Rank() != 2 or InExpertIndices.GetDtype() != OaScalarType::Int32 or
		InExpertIndices.Size(0) <= 0 or InExpertIndices.Size(1) <= 0 or
		InNumExperts <= 0 or InNumExperts > 256) {
		OA_LOG_ERROR(OaLogComponent::ML,
			"MoeExpertPlan: expected Int32 [T,K] and 1..256 experts");
		return {};
	}
	const OaU32 T = static_cast<OaU32>(InExpertIndices.Size(0));
	const OaU32 K = static_cast<OaU32>(InExpertIndices.Size(1));
	const OaU32 E = static_cast<OaU32>(InNumExperts);
	const OaU32 routes = T * K;
	OaMoeExpertPlan plan;
	plan.Counts = OaFnMatrix::Empty(OaMatrixShape{E}, OaScalarType::UInt32);
	plan.Offsets = OaFnMatrix::Empty(OaMatrixShape{E + 1}, OaScalarType::UInt32);
	plan.PackedToken = OaFnMatrix::Empty(OaMatrixShape{routes}, OaScalarType::UInt32);
	plan.PackedExpert = OaFnMatrix::Empty(OaMatrixShape{routes}, OaScalarType::UInt32);
	plan.PackedSlot = OaFnMatrix::Empty(OaMatrixShape{routes}, OaScalarType::UInt32);
	plan.Inverse = OaFnMatrix::Empty(OaMatrixShape{routes}, OaScalarType::UInt32);
	struct { OaU32 T, K, E, Routes; } push{T, K, E, routes};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write,
		OaBufferAccess::Write, OaBufferAccess::Write, OaBufferAccess::Write,
		OaBufferAccess::Write,
		OaBufferAccess::Write};
	auto& ctx = OaContext::GetDefault();
	ctx.Add("MoeExpertPlan", {&InExpertIndices, &plan.Counts, &plan.Offsets,
		&plan.PackedToken, &plan.PackedExpert, &plan.PackedSlot, &plan.Inverse},
		access, &push, sizeof(push), 1);
	return plan;
}

OaFnMatrix::OaCompactRowsResult OaFnMatrix::CompactRows(const OaMatrix& InSelf, const OaMatrix& InMask) {
	if (InSelf.Rank() != 2 or InMask.NumElements() != InSelf.Size(0) or
		(InMask.GetDtype() != OaScalarType::Float32 and InMask.GetDtype() != OaScalarType::BFloat16)) {
		OA_LOG_ERROR(OaLogComponent::ML,
			"CompactRows: expected input [T,D] and floating mask [T]");
		return {};
	}
	const OaU32 T = static_cast<OaU32>(InSelf.Size(0));
	const OaU32 D = static_cast<OaU32>(InSelf.Size(1));
	OaCompactRowsResult result;
	result.Values = OaFnMatrix::Zeros(InSelf.GetShape(), InSelf.GetDtype());
	result.RowMap = OaFnMatrix::Empty(OaMatrixShape{T}, OaScalarType::UInt32);
	result.Count = OaFnMatrix::Empty(OaMatrixShape{1}, OaScalarType::UInt32);
	result.DispatchArgs = OaFnMatrix::Empty(OaMatrixShape{3}, OaScalarType::UInt32);
	struct { OaU32 T, D; } push{T, D};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Write, OaBufferAccess::Write, OaBufferAccess::Write,
		OaBufferAccess::Write};
	auto& ctx = OaContext::GetDefault();
	ctx.Add("CompactRows", {&InSelf, &InMask, &result.Values, &result.RowMap,
		&result.Count, &result.DispatchArgs},
		access, &push, sizeof(push), 1);
	if (OaFnAutograd::IsEnabled() and InSelf.RequiresGrad()) {
		auto gradFn = OaMakeSharedPtr<OaGradCompactRows>();
		gradFn->RowMap_ = result.RowMap;
		gradFn->Count_ = result.Count;
		gradFn->DispatchArgs_ = result.DispatchArgs;
		gradFn->Saved_ = OaVec<OaMatrix>{InSelf};
		gradFn->SetGraphInputs(OaVec<OaMatrix>{InSelf});
		gradFn->SequenceNr_ = OaFnAutograd::NextSeq();
		gradFn->OutputShape_ = result.Values.GetShape();
		result.Values.MutAutograd().GradFn = gradFn;
	}
	return result;
}

OaMatrix OaFnMatrix::CompactRowsBwd(const OaMatrix& InGradOut, const OaMatrix& InRowMap,
	const OaMatrix& InCount, OaMatrixShape InInputShape) {
	const OaU32 T = static_cast<OaU32>(InInputShape.Dims[0]);
	const OaU32 D = static_cast<OaU32>(InInputShape.Dims[1]);
	OaMatrix gradIn = OaFnMatrix::Zeros(InInputShape, InGradOut.GetDtype());
	struct { OaU32 T, D; } push{T, D};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Read, OaBufferAccess::ReadWrite};
	auto& ctx = OaContext::GetDefault();
	ctx.Add("CompactRowsBwd", {&InGradOut, &InRowMap, &InCount, &gradIn},
		access, &push, sizeof(push), DivCeil(T * D, 256));
	return gradIn;
}

OaMatrix OaFnMatrix::CompactRowsBwd(const OaMatrix& InGradOut, const OaMatrix& InRowMap,
	const OaMatrix& InCount, const OaMatrix& InDispatchArgs,
	OaMatrixShape InInputShape) {
	const OaU32 T = static_cast<OaU32>(InInputShape.Dims[0]);
	const OaU32 D = static_cast<OaU32>(InInputShape.Dims[1]);
	OaMatrix gradIn = OaFnMatrix::Zeros(InInputShape, InGradOut.GetDtype());
	struct { OaU32 T, D; } push{T, D};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::ReadWrite};
	auto& ctx = OaContext::GetDefault();
	ctx.AddIndirect("CompactRowsBwdIndirect",
		{&InGradOut, &InRowMap, &InCount, &InDispatchArgs, &gradIn},
		access, &push, sizeof(push), InDispatchArgs);
	return gradIn;
}

OaMatrix OaFnMatrix::ScatterRows(const OaMatrix& InSelf, const OaMatrix& InSource,
	const OaMatrix& InRowMap, const OaMatrix& InCount) {
	if (InSelf.Rank() != 2 or InSource.GetShape() != InSelf.GetShape() or
		InSource.GetDtype() != InSelf.GetDtype() or
		InRowMap.GetDtype() != OaScalarType::UInt32 or InRowMap.NumElements() < InSelf.Size(0) or
		InCount.GetDtype() != OaScalarType::UInt32 or InCount.NumElements() != 1) {
		OA_LOG_ERROR(OaLogComponent::ML,
			"ScatterRows: expected matching [T,D] tensors, UInt32 row map [T], and UInt32 count [1]");
		return {};
	}
	const OaU32 T = static_cast<OaU32>(InSelf.Size(0));
	const OaU32 D = static_cast<OaU32>(InSelf.Size(1));
	OaMatrix out = OaFnMatrix::Empty(InSelf.GetShape(), InSelf.GetDtype());
	struct { OaU32 T, D; } push{T, D};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
	auto& ctx = OaContext::GetDefault();
	ctx.Add("ScatterRows", {&InSelf, &InSource, &InRowMap, &InCount, &out},
		access, &push, sizeof(push), DivCeil(T * D, 256));
	if (OaFnAutograd::IsEnabled() and (InSelf.RequiresGrad() or InSource.RequiresGrad())) {
		auto gradFn = OaMakeSharedPtr<OaGradScatterRows>();
		gradFn->RowMap_ = InRowMap;
		gradFn->Count_ = InCount;
		gradFn->Saved_ = OaVec<OaMatrix>{InSelf, InSource};
		gradFn->SetGraphInputs(OaVec<OaMatrix>{InSelf, InSource});
		gradFn->SequenceNr_ = OaFnAutograd::NextSeq();
		gradFn->OutputShape_ = out.GetShape();
		out.MutAutograd().GradFn = gradFn;
	}
	return out;
}

OaMatrix OaFnMatrix::ScatterRows(const OaMatrix& InSelf, const OaMatrix& InSource,
	const OaCompactRowsResult& InPlan) {
	if (InSelf.Rank() != 2 or InSource.GetShape() != InSelf.GetShape() or
		InSource.GetDtype() != InSelf.GetDtype() or
		InPlan.RowMap.GetDtype() != OaScalarType::UInt32 or
		InPlan.RowMap.NumElements() < InSelf.Size(0) or
		InPlan.Count.GetDtype() != OaScalarType::UInt32 or
		InPlan.Count.NumElements() != 1 or
		InPlan.DispatchArgs.GetDtype() != OaScalarType::UInt32 or
		InPlan.DispatchArgs.NumElements() != 3) {
		OA_LOG_ERROR(OaLogComponent::ML,
			"ScatterRows: invalid compact-row indirect plan");
		return {};
	}
	const OaU32 T = static_cast<OaU32>(InSelf.Size(0));
	const OaU32 D = static_cast<OaU32>(InSelf.Size(1));
	// Preserve every unselected row first, then touch only Count*D selected
	// values through GPU-authored indirect launch dimensions.
	OaMatrix out = OaFnMatrix::Copy(InSelf);
	struct { OaU32 T, D; } push{T, D};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::ReadWrite};
	auto& ctx = OaContext::GetDefault();
	ctx.AddIndirect("ScatterRowsIndirect",
		{&InSource, &InPlan.RowMap, &InPlan.Count, &InPlan.DispatchArgs, &out},
		access, &push, sizeof(push), InPlan.DispatchArgs);
	if (OaFnAutograd::IsEnabled() and (InSelf.RequiresGrad() or InSource.RequiresGrad())) {
		auto gradFn = OaMakeSharedPtr<OaGradScatterRows>();
		gradFn->RowMap_ = InPlan.RowMap;
		gradFn->Count_ = InPlan.Count;
		gradFn->DispatchArgs_ = InPlan.DispatchArgs;
		gradFn->Saved_ = OaVec<OaMatrix>{InSelf, InSource};
		gradFn->SetGraphInputs(OaVec<OaMatrix>{InSelf, InSource});
		gradFn->SequenceNr_ = OaFnAutograd::NextSeq();
		gradFn->OutputShape_ = out.GetShape();
		out.MutAutograd().GradFn = gradFn;
	}
	return out;
}

OaMatrix OaFnMatrix::ScatterRowsBwdSource(const OaMatrix& InGradOut,
	const OaMatrix& InRowMap, const OaMatrix& InCount) {
	const OaU32 T = static_cast<OaU32>(InGradOut.Size(0));
	const OaU32 D = static_cast<OaU32>(InGradOut.Size(1));
	OaMatrix gradSrc = OaFnMatrix::Zeros(InGradOut.GetShape(), InGradOut.GetDtype());
	struct { OaU32 T, D; } push{T, D};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Read, OaBufferAccess::Write};
	auto& ctx = OaContext::GetDefault();
	ctx.Add("ScatterRowsBwd", {&InGradOut, &InRowMap, &InCount, &gradSrc},
		access, &push, sizeof(push), DivCeil(T * D, 256));
	return gradSrc;
}

OaMatrix OaFnMatrix::ScatterRowsBwdSource(const OaMatrix& InGradOut,
	const OaMatrix& InRowMap, const OaMatrix& InCount,
	const OaMatrix& InDispatchArgs) {
	const OaU32 T = static_cast<OaU32>(InGradOut.Size(0));
	const OaU32 D = static_cast<OaU32>(InGradOut.Size(1));
	OaMatrix gradSrc = OaFnMatrix::Zeros(InGradOut.GetShape(), InGradOut.GetDtype());
	struct { OaU32 T, D; } push{T, D};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
	auto& ctx = OaContext::GetDefault();
	ctx.AddIndirect("ScatterRowsBwdIndirect",
		{&InGradOut, &InRowMap, &InCount, &InDispatchArgs, &gradSrc},
		access, &push, sizeof(push), InDispatchArgs);
	return gradSrc;
}

void OaFnMatrix::MoeRoutingBiasUpdate(const OaMatrix& InSelectionMask,
	OaMatrix& InOutBias, OaI32 InExpertsPerToken, OaF32 InGamma) {
	if (InSelectionMask.Rank() != 2 or InOutBias.NumElements() != InSelectionMask.Size(1) or
		InOutBias.GetDtype() != OaScalarType::Float32 or InExpertsPerToken <= 0 or
		InExpertsPerToken > InSelectionMask.Size(1) or InGamma <= 0.0F) {
		OA_LOG_ERROR(OaLogComponent::ML,
			"MoeRoutingBiasUpdate: expected mask [T,E], FP32 bias [E], 0<K<=E, gamma>0");
		return;
	}
	const OaU32 T = static_cast<OaU32>(InSelectionMask.Size(0));
	const OaU32 E = static_cast<OaU32>(InSelectionMask.Size(1));
	struct { OaU32 T, E, K; OaF32 Gamma; } push{
		T, E, static_cast<OaU32>(InExpertsPerToken), InGamma};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::ReadWrite};
	auto& ctx = OaContext::GetDefault();
	ctx.Add("MoeRoutingBiasUpdate", {&InSelectionMask, &InOutBias},
		access, &push, sizeof(push), 1);
}


// ═══════════════════════════════════════════════════════════════════════════
// GPU-NATIVE OPERATIONS (VK_EXT path - zero CPU overhead)
// ═══════════════════════════════════════════════════════════════════════════
