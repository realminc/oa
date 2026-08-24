// oa::FnMatrix — indexing, selection, reshape utilities, and loss.
//
// Gather, Slice, split, Concat, transpose (fn delegate), CrossEntropyLoss,
// reshape (fn), RepeatInterleave, causalMask (tensor), topK,
// Equal, CompactRows, ScatterRows.

#include <oa/core/fnMatrix.h>
#include <oa/core/log.h>
#include <oa/core/fnmatrix/fnMatrixInternal.h>
#include <oa/core/autograd/matrix/autogradIndex.h>
#include <oa/core/autograd/matrix/autogradShape.h>
#include "../../autograd/autogradAttach.gen.h"
#include <oa/core/matrix.h>
#include <oa/core/memory.h>
#include <oa/core/op.h>
#include <oa/core/status.h>
#include <oa/core/types.h>
#include <oa/core/bufferAccess.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/dispatchDesc.h>

#include <oa/core/validation.h>

#include <cassert>

static oa::U32 divCeil(oa::U32 inA, oa::U32 inB) { return (inA + inB - 1) / inB; }

oa::Matrix oa::FnMatrix::gatherBwd(
	const oa::Matrix& inIndices,
	const oa::Matrix& inGradOutput,
	oa::I32 inVocabSize,
	oa::I32 inEmbedDim)
{
	auto& context = oa::ExecutionSession::getActive();
	const oa::U32 numIndices =
		static_cast<oa::U32>(inIndices.numElements());
	const oa::U32 indexDtype =
		inIndices.getDtype() == oa::ScalarType::UInt8 ? 0U : 1U;
	oa::Matrix gradTable = oa::FnMatrix::empty(
		oa::MatrixShape{inVocabSize, inEmbedDim},
		inGradOutput.getDtype());
	const auto semantic = context.recordOp(
		oa::detail::opRegistry::FnMatrix::gatherBwd,
		{&inIndices, &inGradOutput},
		{&gradTable},
		{
			oa::OpAttribute::fromSignedInteger(
				"vocabSize", inVocabSize),
			oa::OpAttribute::fromSignedInteger(
				"embedDim", inEmbedDim),
		});
	if (not semantic.isOk()) return {};

	struct {
		oa::U32 vocabSize;
		oa::U32 embedDim;
		oa::U32 numIndices;
		oa::U32 indexDtype;
	} push{
		.vocabSize = static_cast<oa::U32>(inVocabSize),
		.embedDim = static_cast<oa::U32>(inEmbedDim),
		.numIndices = numIndices,
		.indexDtype = indexDtype,
	};
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read,
		oa::BufferAccess::Read,
		oa::BufferAccess::ReadWrite,
	};
	context.add(
		"GatherBwd",
		{&inIndices, &inGradOutput, &gradTable},
		access,
		&push,
		sizeof(push),
		static_cast<oa::U32>(inVocabSize),
		1,
		1,
		oa::detail::opRegistry::FnMatrix::gatherBwd.name,
		0,
		oa::detail::opRegistry::FnMatrix::gatherBwd.hash,
		0,
		0,
		semantic.getValue());
	return gradTable;
}

#if not defined(NDEBUG) or defined(OA_ENABLE_VALIDATION)
static oa::Status validateGather(
	const oa::Matrix& inSelf, const oa::Matrix& inIndices)
{
	OA_VALIDATE(inSelf.rank() == 2, oa::ValidationSeverity::Error, oa::LogComponent::Compute,
		"Gather: inSelf must be 2D [vocab, embed_dim], got rank=%d", inSelf.rank());
	OA_VALIDATE(inIndices.rank() >= 1, oa::ValidationSeverity::Error, oa::LogComponent::Compute,
		"Gather: inIndices must have rank >= 1, got rank=%d", inIndices.rank());
	return oa::Status::ok();
}

static oa::Status validateSlice(const oa::Matrix& inSelf, oa::I32 inDim, oa::I64 inStart, oa::I64 inEnd) {
	OA_VALIDATE_BOUNDS(inDim, inSelf.rank(), "Slice dim");
	OA_VALIDATE(inStart >= 0, oa::ValidationSeverity::Error, oa::LogComponent::Compute,
		"Slice: inStart=%lld must be >= 0", static_cast<oa::I64>(inStart));
	OA_VALIDATE(inEnd > inStart, oa::ValidationSeverity::Error, oa::LogComponent::Compute,
		"Slice: inEnd=%lld must be > inStart=%lld",
		static_cast<oa::I64>(inEnd), static_cast<oa::I64>(inStart));
	OA_VALIDATE(inEnd <= inSelf.size(inDim), oa::ValidationSeverity::Error, oa::LogComponent::Compute,
		"Slice: inEnd=%lld out of bounds for dim=%d size=%lld",
		static_cast<oa::I64>(inEnd), inDim, inSelf.size(inDim));
	return oa::Status::ok();
}
#endif

// Indexing and selection
oa::Matrix oa::FnMatrix::gather(const oa::Matrix& inSelf, const oa::Matrix& inIndices) {
	auto& ctx = oa::ExecutionSession::getActive();
#if not defined(NDEBUG) or defined(OA_ENABLE_VALIDATION)
	OA_ASSERT(validateGather(inSelf, inIndices).isOk());
#endif
	oa::I64 numIdx = inIndices.numElements();
	oa::I64 rowSize = inSelf.size(1);
	oa::Matrix out = oa::FnMatrix::empty(oa::MatrixShape{numIdx, rowSize}, inSelf.getDtype());
	const auto semantic = ctx.recordOp(
		oa::detail::opRegistry::FnMatrix::gather,
		{&inSelf, &inIndices}, {&out});
	if (not semantic.isOk()) return {};

	// Gather kernel now handles both UInt8 and UInt32 indices directly (no separate cast!)
	oa::U32 indexDtype = 1;  // Default: UInt32
	if (inIndices.getDtype() == oa::ScalarType::UInt8) {
		indexDtype = 0;  // uInt8 (byte embedding)
	}

	struct { oa::U32 numIndices; oa::U32 RowSize; oa::U32 indexDtype; } push{
		static_cast<oa::U32>(numIdx), static_cast<oa::U32>(rowSize), indexDtype
	};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "Gather", {&inSelf, &inIndices, &out}, access, &push, sizeof(push),
		divCeil(static_cast<oa::U32>(numIdx * rowSize), 256), 1, 1,
		oa::detail::opRegistry::FnMatrix::gather.name, 0,
		oa::detail::opRegistry::FnMatrix::gather.hash, 0, 0,
		semantic.getValue());

	const auto attached = oa::detail::generatedAutogradAttach::FnMatrix::gather(
		out, inSelf, inIndices, semantic.getValue());
	if (not attached.isOk()) {
		OaLogError(
			oa::LogComponent::Compute,
			"Gather semantic autograd attachment failed: %s",
			attached.getMessage().cStr());
		return {};
	}

	return out;
}

oa::Matrix oa::FnMatrix::gatherLastDim(const oa::Matrix& inSelf, const oa::Matrix& inIndices) {
	if (inSelf.rank() != 2 or inIndices.rank() != 2 or
		inIndices.getDtype() != oa::ScalarType::Int32 or
		inSelf.size(0) != inIndices.size(0)) {
		OaLogError(oa::LogComponent::Compute,
			"GatherLastDim: expected Self[T,E] and Int32 indices[T,K]");
		return {};
	}
	const oa::U32 T = static_cast<oa::U32>(inSelf.size(0));
	const oa::U32 E = static_cast<oa::U32>(inSelf.size(1));
	const oa::U32 K = static_cast<oa::U32>(inIndices.size(1));
	auto out = oa::FnMatrix::empty(inIndices.getShape(), inSelf.getDtype());
	struct { oa::U32 T, E, K; } push{T, E, K};
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write};
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	ctx.add( "GatherLastDim", {&inSelf, &inIndices, &out}, access, &push, sizeof(push),
		divCeil(T * K, 256));
	const auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnMatrix::gatherLastDim,
		{&inSelf, &inIndices}, {&out});
	if (not semantic.isOk()) return {};
	if (oa::FnAutograd::isEnabled() and inSelf.requiresGrad()) {
		auto gradFn = oa::makeShared<oa::GradGatherLastDim>();
		gradFn->saveForBackward({inSelf, inIndices});
		gradFn->setGraphInputs(oa::Vec<oa::Matrix>{inSelf, inIndices});
		gradFn->sequenceNr_ = oa::FnAutograd::nextSeq();
		gradFn->outputShape_ = out.getShape();
		if (not oa::FnAutograd::attachSemantic(
			gradFn, semantic.getValue()).isOk())
		{
			return {};
		}
		out.mutAutograd().gradFn = gradFn;
	}
	return out;
}

oa::Matrix oa::FnMatrix::gatherLastDimBwd(const oa::Matrix& inGradOut,
	const oa::Matrix& inIndices, oa::I32 inInputWidth) {
	if (inGradOut.rank() != 2 or inIndices.getShape() != inGradOut.getShape() or
		inIndices.getDtype() != oa::ScalarType::Int32 or inInputWidth <= 0) return {};
	const oa::U32 T = static_cast<oa::U32>(inGradOut.size(0));
	const oa::U32 K = static_cast<oa::U32>(inGradOut.size(1));
	const oa::U32 E = static_cast<oa::U32>(inInputWidth);
	auto out = oa::FnMatrix::empty(oa::MatrixShape{T, E}, inGradOut.getDtype());
	struct { oa::U32 T, E, K; } push{T, E, K};
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write};
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	ctx.add( "GatherLastDimBwd", {&inGradOut, &inIndices, &out}, access, &push, sizeof(push),
		divCeil(T * E, 256));
	if (not lowering.commit(
		oa::detail::opRegistry::FnMatrix::gatherLastDimBwd,
		{&inGradOut, &inIndices}, {&out},
		{oa::OpAttribute::fromSignedInteger(
			"inputWidth", inInputWidth)}).isOk())
	{
		return {};
	}
	return out;
}

oa::Matrix oa::FnMatrix::slice(const oa::Matrix& inSelf, oa::I32 inDim, oa::I64 inStart, oa::I64 inEnd) {
	auto& ctx = oa::ExecutionSession::getActive();
	const oa::I32 rank = inSelf.rank();
	assert(rank >= 1 && rank <= 4 && "Slice supports rank 1 through 4");
#if not defined(NDEBUG) or defined(OA_ENABLE_VALIDATION)
	OA_ASSERT(validateSlice(inSelf, inDim, inStart, inEnd).isOk());
#endif
	(void)rank;
	oa::I64 sliceLen = inEnd - inStart;
	oa::MatrixShape outShape = inSelf.getShape();
	outShape[inDim] = sliceLen;

	oa::Matrix out = oa::FnMatrix::empty(outShape, inSelf.getDtype());
	const auto semantic = ctx.recordOp(
		oa::detail::opRegistry::FnMatrix::slice, {&inSelf}, {&out},
		{
			oa::OpAttribute::fromSignedInteger("dim", inDim),
			oa::OpAttribute::fromSignedInteger("start", inStart),
			oa::OpAttribute::fromSignedInteger("end", inEnd),
		});
	if (not semantic.isOk()) return {};
	struct Push {
		oa::U32 count, rank, dim, srcStart, dstStart;
		oa::U32 srcDims[4];
		oa::U32 dstDims[4];
		oa::U32 copyDims[4];
	} push{};
	push.count = static_cast<oa::U32>(out.numElements());
	push.rank = static_cast<oa::U32>(rank);
	push.dim = static_cast<oa::U32>(inDim);
	push.srcStart = static_cast<oa::U32>(inStart);
	for (oa::I32 d = 0; d < rank; ++d) {
		push.srcDims[d] = static_cast<oa::U32>(inSelf.size(d));
		push.dstDims[d] = static_cast<oa::U32>(out.size(d));
		push.copyDims[d] = static_cast<oa::U32>(out.size(d));
	}
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "MatrixCopyRegion", {&inSelf, &out}, access, &push, sizeof(push),
		divCeil(push.count, 256), 1, 1,
		oa::detail::opRegistry::FnMatrix::slice.name, 0,
		oa::detail::opRegistry::FnMatrix::slice.hash, 0, 0,
		semantic.getValue());

	const auto attached = oa::detail::generatedAutogradAttach::FnMatrix::slice(
		out, inSelf, inDim, inStart, inEnd, semantic.getValue());
	if (not attached.isOk()) {
		OaLogError(oa::LogComponent::Compute,
			"Slice semantic autograd attachment failed: %s",
			attached.getMessage().cStr());
		return {};
	}

	return out;
}

oa::Matrix oa::FnMatrix::sliceBwd(
	oa::MatrixShape inInputShape, oa::I32 inDim, oa::I64 inStart, oa::I64 inEnd,
	const oa::Matrix& inDOut) {
	auto& ctx = oa::ExecutionSession::getActive();
	const oa::I32 rank = inInputShape.rank;
	assert(rank >= 1 && rank <= 4 && "SliceBwd supports rank 1 through 4");
	assert(inDim >= 0 && inDim < rank && "SliceBwd dim out of range");
	assert(inStart >= 0 && inEnd > inStart && inEnd <= inInputShape[inDim] &&
		"SliceBwd start/end out of range");
	(void)rank;

	oa::OpLoweringScope lowering(ctx);
	auto out = oa::FnMatrix::zeros(inInputShape, inDOut.getDtype());
	struct Push {
		oa::U32 count, rank, dim, srcStart, dstStart;
		oa::U32 srcDims[4];
		oa::U32 dstDims[4];
		oa::U32 copyDims[4];
	} push{};
	push.count = static_cast<oa::U32>(inDOut.numElements());
	push.rank = static_cast<oa::U32>(rank);
	push.dim = static_cast<oa::U32>(inDim);
	push.srcStart = 0;
	push.dstStart = static_cast<oa::U32>(inStart);
	for (oa::I32 d = 0; d < rank; ++d) {
		push.srcDims[d] = static_cast<oa::U32>(inDOut.size(d));
		push.dstDims[d] = static_cast<oa::U32>(inInputShape[d]);
		push.copyDims[d] = static_cast<oa::U32>(inDOut.size(d));
	}
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "MatrixCopyRegion", {&inDOut, &out}, access, &push, sizeof(push),
		divCeil(push.count, 256));
	if (not lowering.commit(
		oa::detail::opRegistry::FnMatrix::sliceBwd,
		{&inDOut}, {&out},
		{
			oa::OpAttribute::fromShape("inputShape", inInputShape),
			oa::OpAttribute::fromSignedInteger("dim", inDim),
			oa::OpAttribute::fromSignedInteger("start", inStart),
			oa::OpAttribute::fromSignedInteger("end", inEnd),
		}).isOk())
	{
		return {};
	}
	return out;
}

oa::Vec<oa::Matrix> oa::FnMatrix::split(
	const oa::Matrix& inSelf, oa::Span<oa::I64> inSizes, oa::I32 inDim
) {
	auto& ctx = oa::ExecutionSession::getActive();
	assert(inSizes.size() > 0 && "split requires at least one size");
	assert(inDim >= 0 && inDim < inSelf.getShape().rank && "Invalid split dimension");

	// verify sizes sum to dimension size
	oa::I64 totalSize = 0;
	for (oa::I64 size : inSizes) {
		totalSize += size;
	}
	assert(totalSize == inSelf.size(inDim) && "split sizes must sum to dimension size");
	(void)totalSize;

	// Create output tensors by slicing
	oa::Vec<oa::Matrix> outputs;
	outputs.reserve(inSizes.size());

	oa::OpLoweringScope lowering(ctx);
	oa::I64 offset = 0;
	for (oa::I64 size : inSizes) {
		auto slice = oa::FnMatrix::slice(inSelf, inDim, offset, offset + size);
		outputs.pushBack(slice);
		offset += size;
	}

	oa::Vec<const oa::Matrix*> outputPointers;
	outputPointers.reserve(outputs.size());
	for (const auto& output : outputs) outputPointers.pushBack(&output);
	const oa::Matrix* inputPointers[] = {&inSelf};
	const oa::OpAttribute attributes[] = {
		oa::OpAttribute::fromSignedInteger("dim", inDim),
	};
	const auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnMatrix::split,
		oa::Span<const oa::Matrix* const>(inputPointers, 1U),
		oa::Span<const oa::Matrix* const>(
			outputPointers.data(), outputPointers.size()),
		oa::Span<const oa::OpAttribute>(attributes, 1U));
	if (not semantic.isOk()) return {};

	// The slices own independent gradient nodes: sharing one node would let the
	// tape's deduplication process only one output. attach those nodes to the
	// corresponding outputs of the single variadic split semantic operation.
	for (oa::U32 index = 0; index < outputs.size(); ++index) {
		const auto gradFn = outputs[index].getGradFn();
		if (gradFn and not oa::FnAutograd::attachSemantic(
			gradFn, semantic.getValue(), index).isOk())
		{
			return {};
		}
	}

	return outputs;
}

oa::Matrix oa::FnMatrix::concat(oa::Span<oa::Matrix> inInputs, oa::I32 inDim) {
	auto& ctx = oa::ExecutionSession::getActive();
	assert(inInputs.size() > 0 && "Concat requires at least one input");
	assert(inDim >= 0 && inDim < inInputs[0].rank() && "Concat dimension out of range");
	assert(inInputs[0].rank() >= 1 && inInputs[0].rank() <= 4 &&
		"Concat supports rank 1 through 4");

	// Calculate output shape
	oa::MatrixShape outShape = inInputs[0].getShape();
	oa::I64 totalSize = 0;
	for (const auto& input : inInputs) {
		assert(input.getShape().rank == outShape.rank && "All inputs must have same rank");
		assert(input.getDtype() == inInputs[0].getDtype() &&
			"Concat inputs must have the same dtype");
		for (oa::I32 d = 0; d < outShape.rank; ++d) {
			assert((d == inDim || input.size(d) == outShape[d]) &&
				"Concat non-concatenated dimensions must match");
		}
		totalSize += input.size(inDim);
	}
	outShape[inDim] = totalSize;

	// allocate output
	oa::Matrix out = oa::FnMatrix::empty(outShape, inInputs[0].getDtype());

	oa::OpLoweringScope lowering(ctx);
	oa::Vec<const oa::Matrix*> inputPointers;
	inputPointers.reserve(inInputs.size());
	oa::U32 dstStart = 0;
	for (const auto& input : inInputs) {
		inputPointers.pushBack(&input);
		struct Push {
			oa::U32 count, rank, dim, srcStart, dstStart;
			oa::U32 srcDims[4];
			oa::U32 dstDims[4];
			oa::U32 copyDims[4];
		} push{};
		push.count = static_cast<oa::U32>(input.numElements());
		push.rank = static_cast<oa::U32>(outShape.rank);
		push.dim = static_cast<oa::U32>(inDim);
		push.dstStart = dstStart;
		for (oa::I32 d = 0; d < outShape.rank; ++d) {
			push.srcDims[d] = static_cast<oa::U32>(input.size(d));
			push.dstDims[d] = static_cast<oa::U32>(out.size(d));
			push.copyDims[d] = static_cast<oa::U32>(input.size(d));
		}
		oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
		ctx.add( "MatrixCopyRegion", {&input, &out}, access, &push, sizeof(push),
			divCeil(push.count, 256));
		dstStart += static_cast<oa::U32>(input.size(inDim));
	}

	const oa::Matrix* outputPointers[] = {&out};
	const oa::OpAttribute attributes[] = {
		oa::OpAttribute::fromSignedInteger("dim", inDim),
	};
	const auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnMatrix::concat,
		oa::Span<const oa::Matrix* const>(
			inputPointers.data(), inputPointers.size()),
		oa::Span<const oa::Matrix* const>(outputPointers, 1U),
		oa::Span<const oa::OpAttribute>(attributes, 1U));
	if (not semantic.isOk()) return {};

	if (oa::FnAutograd::isEnabled()) {
		bool anyGrad = false;
		for (const auto& input : inInputs) {
			if (input.requiresGrad()) { anyGrad = true; break; }
		}
		if (anyGrad) {
			auto gradFn = oa::makeShared<oa::GradConcat>();
			gradFn->dim_ = inDim;
			for (const auto& input : inInputs) {
				gradFn->sizes_.pushBack(input.size(inDim));
			}
			oa::Vec<oa::Matrix> inputs;
			inputs.reserve(inInputs.size());
			for (const auto& input : inInputs) inputs.pushBack(input);
			gradFn->setGraphInputs(inputs);
			gradFn->sequenceNr_ = oa::FnAutograd::nextSeq();
			gradFn->outputShape_ = out.getShape();
			if (not oa::FnAutograd::attachSemantic(
				gradFn, semantic.getValue()).isOk())
			{
				return {};
			}
			out.mutAutograd().gradFn = gradFn;
		}
	}

	return out;
}

oa::Matrix oa::FnMatrix::transpose(const oa::Matrix& inA, oa::I32 inDim0, oa::I32 inDim1) {
	const oa::I32 rank = inA.rank();
	oa::I32 dim0 = inDim0 < 0 ? inDim0 + rank : inDim0;
	oa::I32 dim1 = inDim1 < 0 ? inDim1 + rank : inDim1;
	if ((rank != 2 and rank != 3) or dim0 < 0 or dim0 >= rank or
		dim1 < 0 or dim1 >= rank or dim0 == dim1 or
		not ((dim0 == rank - 2 and dim1 == rank - 1) or
			(dim0 == rank - 1 and dim1 == rank - 2))) {
		OaLogError(oa::LogComponent::Compute,
			"Transpose: only the last two axes of rank-2/rank-3 matrices are supported "
			"(rank=%d, Dim0=%d, Dim1=%d)",
			rank, inDim0, inDim1);
		return {};
	}

	auto& ctx = oa::ExecutionSession::getActive();
	oa::MatrixShape outShape = inA.getShape();
	const oa::I64 rows = outShape[rank - 2];
	const oa::I64 cols = outShape[rank - 1];
	outShape[rank - 2] = cols;
	outShape[rank - 1] = rows;
	oa::Matrix out = oa::FnMatrix::empty(outShape, inA.getDtype());
	const auto semantic = ctx.recordOp(
		oa::detail::opRegistry::FnMatrix::transpose, {&inA}, {&out},
		{
			oa::OpAttribute::fromSignedInteger("dim0", dim0),
			oa::OpAttribute::fromSignedInteger("dim1", dim1),
		});
	if (not semantic.isOk()) return {};

	constexpr oa::U32 kTile = 32;
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
	if (rank == 3) {
		const oa::I64 batch = inA.size(0);
		struct { oa::U32 Batch; oa::U32 rows; oa::U32 cols; } push{
			static_cast<oa::U32>(batch),
			static_cast<oa::U32>(rows),
			static_cast<oa::U32>(cols),
		};
		ctx.add( "TransposeBatched", {&inA, &out}, access, &push, sizeof(push),
			divCeil(static_cast<oa::U32>(cols), kTile),
			divCeil(static_cast<oa::U32>(rows), kTile),
			static_cast<oa::U32>(batch),
			oa::detail::opRegistry::FnMatrix::transpose.name, 0,
			oa::detail::opRegistry::FnMatrix::transpose.hash, 0, 0,
			semantic.getValue());
	} else {
		struct { oa::U32 rows; oa::U32 cols; } push{
			static_cast<oa::U32>(rows),
			static_cast<oa::U32>(cols),
		};
		ctx.add( "TransposeTiled", {&inA, &out}, access, &push, sizeof(push),
			divCeil(static_cast<oa::U32>(cols), kTile),
			divCeil(static_cast<oa::U32>(rows), kTile), 1,
			oa::detail::opRegistry::FnMatrix::transpose.name, 0,
			oa::detail::opRegistry::FnMatrix::transpose.hash, 0, 0,
			semantic.getValue());
	}

	const auto attached = oa::detail::generatedAutogradAttach::FnMatrix::transpose(
		out, inA, dim0, dim1, semantic.getValue());
	if (not attached.isOk()) {
		OaLogError(oa::LogComponent::Compute,
			"Transpose semantic autograd attachment failed: %s",
			attached.getMessage().cStr());
		return {};
	}
	return out;
}

// Cross-entropy loss lives in the oa::FnLoss namespace.
// backward compatibility alias in source/cpp/include/oa/Ml/FnMatrix.h forwards to oa::FnLoss::CrossEntropyLoss

// reshape, RepeatInterleave, causalMask (tensor), topK, Equal, CompactRows/ScatterRows
oa::Matrix oa::FnMatrix::reshape(const oa::Matrix& inA, oa::MatrixShape inShape) {
	oa::I64 total = inA.numElements();
	oa::I64 product = 1;
	oa::I32 inferDim = -1;
	for (oa::I32 d = 0; d < inShape.rank; ++d) {
		if (inShape[d] == -1) { inferDim = d; }
		else { product *= inShape[d]; }
	}
	oa::MatrixShape resolved = inShape;
	if (inferDim >= 0) {
		resolved[inferDim] = (product > 0) ? (total / product) : 0;
	}
	oa::Matrix out = inA.reshape(resolved);
	if (oa::FnAutograd::isEnabled() and inA.requiresGrad()) {
		// reshape returns a VIEW that aliases inA's autograd meta (shared_ptr).
		// Attaching a gradfn to the view would clobber inA's gradfn (breaking
		// topo collection — the walk stops at the reshape's own node) and corrupt
		// inA's leaf-ness (isLeaf() flips false → leaf grad never accumulates).
		// detach to an independent meta before attaching the reshape gradfn.
		out.detachForGradAttach(true);
		auto gradFn = oa::makeShared<oa::GradReshape>();
		gradFn->inputShape_ = inA.getShape();
		gradFn->setGraphInputs(oa::Vec<oa::Matrix>{inA});
		gradFn->sequenceNr_ = oa::FnAutograd::nextSeq();
		gradFn->outputShape_ = out.getShape();
		out.mutAutograd().gradFn = gradFn;
	}
	return out;
}

oa::Matrix oa::FnMatrix::repeatInterleave(const oa::Matrix& inA, oa::I32 inRepeats, oa::I32 inDim) {
	auto& ctx = oa::ExecutionSession::getActive();
	const oa::MatrixShape& inShape = inA.getShape();
	oa::MatrixShape outShape = inShape;
	outShape[inDim] *= inRepeats;
	oa::Matrix out = oa::FnMatrix::empty(outShape, inA.getDtype());

	const oa::I64 numOut = outShape.numElements();
	const oa::I32 rank = inShape.rank;

	// Build push constants with shape and stride info
	struct {
		oa::U32 count;
		oa::U32 repeats;
		oa::U32 dim;
		oa::U32 rank;
		oa::U32 inShape[4];
		oa::U32 inStride[4];
	} push{};

	push.count = static_cast<oa::U32>(numOut);
	push.repeats = static_cast<oa::U32>(inRepeats);
	push.dim = static_cast<oa::U32>(inDim);
	push.rank = static_cast<oa::U32>(rank);

	oa::Stride inStride = oa::Stride::rowMajor(inShape);
	for (oa::I32 d = 0; d < rank and d < 4; ++d) {
		push.inShape[d] = static_cast<oa::U32>(inShape[d]);
		push.inStride[d] = static_cast<oa::U32>(inStride.stepElements(d));
	}

	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
	oa::OpLoweringScope lowering(ctx);
	ctx.add( "RepeatInterleave", {&inA, &out}, access, &push, sizeof(push),
		divCeil(static_cast<oa::U32>(numOut), 256));
	const auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnMatrix::repeatInterleave,
		{&inA}, {&out},
		{
			oa::OpAttribute::fromSignedInteger("repeats", inRepeats),
			oa::OpAttribute::fromSignedInteger("dim", inDim),
		});
	if (not semantic.isOk()) return {};

	if (oa::FnAutograd::isEnabled() and inA.requiresGrad()) {
		auto gradFn = oa::makeShared<oa::GradRepeatInterleave>();
		gradFn->repeats_ = inRepeats;
		gradFn->dim_ = inDim;
		gradFn->saveForBackward({inA});
		gradFn->setGraphInputs(oa::Vec<oa::Matrix>{inA});
		gradFn->sequenceNr_ = oa::FnAutograd::nextSeq();
		gradFn->outputShape_ = out.getShape();
		if (not oa::FnAutograd::attachSemantic(
			gradFn, semantic.getValue()).isOk())
		{
			return {};
		}
		out.mutAutograd().gradFn = gradFn;
	}

	return out;
}

oa::Matrix oa::FnMatrix::repeatInterleaveBwd(const oa::Matrix& inGradOut, oa::MatrixShape inInputShape, oa::I32 inRepeats, oa::I32 inDim) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::Matrix gradIn = oa::FnMatrix::empty(inInputShape, inGradOut.getDtype());

	const oa::I64 numIn = inInputShape.numElements();
	const oa::I32 rank = inInputShape.rank;

	// output shape (the shape of inGradOut)
	oa::MatrixShape outShape = inInputShape;
	outShape[inDim] *= inRepeats;

	struct {
		oa::U32 count;
		oa::U32 repeats;
		oa::U32 dim;
		oa::U32 rank;
		oa::U32 outShape[4];
		oa::U32 inStride[4];
	} push{};

	push.count = static_cast<oa::U32>(numIn);
	push.repeats = static_cast<oa::U32>(inRepeats);
	push.dim = static_cast<oa::U32>(inDim);
	push.rank = static_cast<oa::U32>(rank);

	oa::Stride inStride = oa::Stride::rowMajor(inInputShape);
	for (oa::I32 d = 0; d < rank and d < 4; ++d) {
		push.outShape[d] = static_cast<oa::U32>(outShape[d]);
		push.inStride[d] = static_cast<oa::U32>(inStride.stepElements(d));
	}

	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
	oa::OpLoweringScope lowering(ctx);
	ctx.add( "RepeatInterleaveBwd", {&inGradOut, &gradIn}, access, &push, sizeof(push),
		divCeil(static_cast<oa::U32>(numIn), 256));
	if (not lowering.commit(
		oa::detail::opRegistry::FnMatrix::repeatInterleaveBwd,
		{&inGradOut}, {&gradIn},
		{
			oa::OpAttribute::fromShape("inputShape", inInputShape),
			oa::OpAttribute::fromSignedInteger("repeats", inRepeats),
			oa::OpAttribute::fromSignedInteger("dim", inDim),
		}).isOk())
	{
		return {};
	}

	return gradIn;
}

oa::Matrix oa::FnMatrix::causalMask(const oa::Matrix& inScores) {
	const oa::I32 rank = inScores.rank();
	if (rank < 2) {
		OaLogError(oa::LogComponent::Compute, "causalMask(scores): rank must be >= 2");
		return {};
	}
	const oa::I64 Tq = inScores.size(rank - 2);
	const oa::I64 Tk = inScores.size(rank - 1);
	oa::Matrix out = oa::FnMatrix::empty(inScores.getShape(), inScores.getDtype());
	const oa::U32 count = static_cast<oa::U32>(inScores.numElements());
	struct { oa::U32 count, Tq, Tk; } push{
		count, static_cast<oa::U32>(Tq), static_cast<oa::U32>(Tk)};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	ctx.add( "CausalMaskApply", {&inScores, &out}, access, &push, sizeof(push),
		divCeil(count, 256));
	const auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnMatrix::causalMask,
		{&inScores}, {&out});
	if (not semantic.isOk()) return {};
	if (oa::FnAutograd::isEnabled() and inScores.requiresGrad()) {
		auto gradFn = oa::makeShared<oa::GradCausalMask>();
		gradFn->saveForBackward({inScores});
		gradFn->setGraphInputs(oa::Vec<oa::Matrix>{inScores});
		gradFn->sequenceNr_ = oa::FnAutograd::nextSeq();
		gradFn->outputShape_ = out.getShape();
		out.mutAutograd().gradFn = gradFn;
		if (not oa::FnAutograd::attachSemantic(
			gradFn, semantic.getValue()).isOk())
		{
			return {};
		}
	}
	return out;
}

oa::Matrix oa::FnMatrix::causalMaskBwd(const oa::Matrix& inGradOut) {
	const oa::I32 rank = inGradOut.rank();
	if (rank < 2) return {};
	const oa::U32 Tq = static_cast<oa::U32>(inGradOut.size(rank - 2));
	const oa::U32 Tk = static_cast<oa::U32>(inGradOut.size(rank - 1));
	const oa::U32 count = static_cast<oa::U32>(inGradOut.numElements());
	oa::Matrix out = oa::FnMatrix::empty(inGradOut.getShape(), inGradOut.getDtype());
	struct { oa::U32 count, Tq, Tk; } push{count, Tq, Tk};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	ctx.add( "CausalMaskBwd", {&inGradOut, &out}, access, &push, sizeof(push),
		divCeil(count, 256));
	if (not lowering.commit(
		oa::detail::opRegistry::FnMatrix::causalMaskBwd,
		{&inGradOut}, {&out}).isOk())
	{
		return {};
	}
	return out;
}

// Note: oa::FnMatrix::topK() is implemented in DeviceMatrixFn.cpp

oa::Matrix oa::FnMatrix::equal(const oa::Matrix& inA, oa::F32 inValue) {
	if (inA.getDtype() != oa::ScalarType::Float32 and
		inA.getDtype() != oa::ScalarType::BFloat16 and
		inA.getDtype() != oa::ScalarType::Int32) {
		OaLogError(oa::LogComponent::Compute, "Equal: supports Float32, BFloat16 and Int32");
		return {};
	}
	oa::Matrix out = oa::FnMatrix::empty(inA.getShape(), oa::ScalarType::Float32);
	const oa::U32 n = static_cast<oa::U32>(inA.numElements());
	struct { oa::U32 n; oa::U32 inputType; oa::F32 value; } push{
		n,
		inA.getDtype() == oa::ScalarType::Int32 ? 1u : 0u,
		inValue,
	};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	ctx.add( "Equal", {&inA, &out}, access, &push, sizeof(push), divCeil(n, 256));
	if (not lowering.commit(
		oa::detail::opRegistry::FnMatrix::equal,
		{&inA}, {&out},
		{oa::OpAttribute::fromFloat("value", inValue)}).isOk())
	{
		return {};
	}
	return out;  // Comparison is intentionally non-differentiable.
}

oa::Matrix oa::FnMatrix::greaterEqual(const oa::Matrix& inA, oa::F32 inValue) {
	if (inA.getDtype() != oa::ScalarType::Float32 and
		inA.getDtype() != oa::ScalarType::BFloat16) {
		OaLogError(oa::LogComponent::Compute, "GreaterEqual: supports Float32 and BFloat16");
		return {};
	}
	oa::Matrix out = oa::FnMatrix::empty(inA.getShape(), oa::ScalarType::Float32);
	const oa::U32 n = static_cast<oa::U32>(inA.numElements());
	struct { oa::U32 n; oa::F32 value; } push{n, inValue};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	ctx.add( "GreaterEqual", {&inA, &out}, access, &push, sizeof(push), divCeil(n, 256));
	if (not lowering.commit(
		oa::detail::opRegistry::FnMatrix::greaterEqual,
		{&inA}, {&out},
		{oa::OpAttribute::fromFloat("value", inValue)}).isOk())
	{
		return {};
	}
	return out;  // Comparison is intentionally non-differentiable.
}

oa::MoeExpertPlan oa::FnMatrix::moeExpertPlan(
	const oa::Matrix& inExpertIndices, oa::I32 inNumExperts) {
	if (inExpertIndices.rank() != 2 or inExpertIndices.getDtype() != oa::ScalarType::Int32 or
		inExpertIndices.size(0) <= 0 or inExpertIndices.size(1) <= 0 or
		inNumExperts <= 0 or inNumExperts > 256) {
		OaLogError(oa::LogComponent::Compute,
			"MoeExpertPlan: expected Int32 [T,K] and 1..256 experts");
		return {};
	}
	const oa::U32 T = static_cast<oa::U32>(inExpertIndices.size(0));
	const oa::U32 K = static_cast<oa::U32>(inExpertIndices.size(1));
	const oa::U32 E = static_cast<oa::U32>(inNumExperts);
	const oa::U32 routes = T * K;
	oa::MoeExpertPlan plan;
	plan.counts = oa::FnMatrix::empty(oa::MatrixShape{E}, oa::ScalarType::UInt32);
	plan.offsets = oa::FnMatrix::empty(oa::MatrixShape{E + 1}, oa::ScalarType::UInt32);
	plan.packedToken = oa::FnMatrix::empty(oa::MatrixShape{routes}, oa::ScalarType::UInt32);
	plan.packedExpert = oa::FnMatrix::empty(oa::MatrixShape{routes}, oa::ScalarType::UInt32);
	plan.packedSlot = oa::FnMatrix::empty(oa::MatrixShape{routes}, oa::ScalarType::UInt32);
	plan.inverse = oa::FnMatrix::empty(oa::MatrixShape{routes}, oa::ScalarType::UInt32);
	struct { oa::U32 T, K, E, Routes; } push{T, K, E, routes};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write,
		oa::BufferAccess::Write, oa::BufferAccess::Write, oa::BufferAccess::Write,
		oa::BufferAccess::Write,
		oa::BufferAccess::Write};
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	ctx.add( "MoeExpertPlan", {&inExpertIndices, &plan.counts, &plan.offsets,
		&plan.packedToken, &plan.packedExpert, &plan.packedSlot, &plan.inverse},
		access, &push, sizeof(push), 1);
	if (not lowering.commit(
		oa::detail::opRegistry::FnMatrix::moeExpertPlan,
		{&inExpertIndices},
		{&plan.counts, &plan.offsets, &plan.packedToken, &plan.packedExpert,
			&plan.packedSlot, &plan.inverse},
		{oa::OpAttribute::fromSignedInteger(
			"numExperts", inNumExperts)}).isOk())
	{
		return {};
	}
	return plan;
}

oa::CompactRowsResult oa::FnMatrix::compactRows(const oa::Matrix& inSelf, const oa::Matrix& inMask) {
	if (inSelf.rank() != 2 or inMask.numElements() != inSelf.size(0) or
		(inMask.getDtype() != oa::ScalarType::Float32 and inMask.getDtype() != oa::ScalarType::BFloat16)) {
		OaLogError(oa::LogComponent::Compute,
			"CompactRows: expected input [T,D] and floating mask [T]");
		return {};
	}
	const oa::U32 T = static_cast<oa::U32>(inSelf.size(0));
	const oa::U32 D = static_cast<oa::U32>(inSelf.size(1));
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	oa::CompactRowsResult result;
	result.values = oa::FnMatrix::zeros(inSelf.getShape(), inSelf.getDtype());
	result.rowMap = oa::FnMatrix::empty(oa::MatrixShape{T}, oa::ScalarType::UInt32);
	result.count = oa::FnMatrix::empty(oa::MatrixShape{1}, oa::ScalarType::UInt32);
	result.dispatchArgs = oa::FnMatrix::empty(oa::MatrixShape{3}, oa::ScalarType::UInt32);
	struct { oa::U32 T, D; } push{T, D};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Write, oa::BufferAccess::Write, oa::BufferAccess::Write,
		oa::BufferAccess::Write};
	ctx.add( "CompactRows", {&inSelf, &inMask, &result.values, &result.rowMap,
		&result.count, &result.dispatchArgs},
		access, &push, sizeof(push), 1);
	const auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnMatrix::compactRows,
		{&inSelf, &inMask},
		{&result.values, &result.rowMap, &result.count, &result.dispatchArgs});
	if (not semantic.isOk()) return {};
	if (oa::FnAutograd::isEnabled() and inSelf.requiresGrad()) {
		auto gradFn = oa::makeShared<oa::GradCompactRows>();
		gradFn->rowMap_ = result.rowMap;
		gradFn->count_ = result.count;
		gradFn->dispatchArgs_ = result.dispatchArgs;
		gradFn->saveForBackward({inSelf});
		gradFn->setGraphInputs(oa::Vec<oa::Matrix>{inSelf});
		gradFn->sequenceNr_ = oa::FnAutograd::nextSeq();
		gradFn->outputShape_ = result.values.getShape();
		if (not oa::FnAutograd::attachSemantic(
			gradFn, semantic.getValue()).isOk())
		{
			return {};
		}
		result.values.mutAutograd().gradFn = gradFn;
	}
	return result;
}

oa::Matrix oa::FnMatrix::compactRowsBwd(const oa::Matrix& inGradOut, const oa::Matrix& inRowMap,
	const oa::Matrix& inCount, oa::MatrixShape inInputShape) {
	const oa::U32 T = static_cast<oa::U32>(inInputShape.dims[0]);
	const oa::U32 D = static_cast<oa::U32>(inInputShape.dims[1]);
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	oa::Matrix gradIn = oa::FnMatrix::zeros(inInputShape, inGradOut.getDtype());
	struct { oa::U32 T, D; } push{T, D};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::ReadWrite};
	ctx.add( "CompactRowsBwd", {&inGradOut, &inRowMap, &inCount, &gradIn},
		access, &push, sizeof(push), divCeil(T * D, 256));
	if (not lowering.commit(
		oa::detail::opRegistry::FnMatrix::compactRowsBwd,
		{&inGradOut, &inRowMap, &inCount, nullptr}, {&gradIn},
		{oa::OpAttribute::fromShape(
			"inputShape", inInputShape)}).isOk())
	{
		return {};
	}
	return gradIn;
}

oa::Matrix oa::FnMatrix::compactRowsBwd(const oa::Matrix& inGradOut, const oa::Matrix& inRowMap,
	const oa::Matrix& inCount, const oa::Matrix& inDispatchArgs,
	oa::MatrixShape inInputShape) {
	const oa::U32 T = static_cast<oa::U32>(inInputShape.dims[0]);
	const oa::U32 D = static_cast<oa::U32>(inInputShape.dims[1]);
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	oa::Matrix gradIn = oa::FnMatrix::zeros(inInputShape, inGradOut.getDtype());
	struct { oa::U32 T, D; } push{T, D};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::ReadWrite};
	const oa::Matrix* matrices[] = {
		&inGradOut, &inRowMap, &inCount, &inDispatchArgs, &gradIn};
	oa::MatrixDispatchDesc desc;
	desc.dispatch.kernel = "CompactRowsBwdIndirect";
	desc.dispatch.access = access;
	desc.dispatch.pushData = &push;
	desc.dispatch.pushSize = sizeof(push);
	desc.matrices = matrices;
	desc.indirectArgs = &inDispatchArgs;
	const oa::Status status = ctx.record( desc);
	if (not status.isOk()) {
		OaLogError(oa::LogComponent::Compute,
			"CompactRowsBwd: failed to record indirect dispatch: %s",
			status.getMessage().cStr());
		return {};
	}
	if (not lowering.commit(
		oa::detail::opRegistry::FnMatrix::compactRowsBwd,
		{&inGradOut, &inRowMap, &inCount, &inDispatchArgs}, {&gradIn},
		{oa::OpAttribute::fromShape(
			"inputShape", inInputShape)}).isOk())
	{
		return {};
	}
	return gradIn;
}

oa::Matrix oa::FnMatrix::scatterRows(const oa::Matrix& inSelf, const oa::Matrix& inSource,
	const oa::Matrix& inRowMap, const oa::Matrix& inCount) {
	if (inSelf.rank() != 2 or inSource.getShape() != inSelf.getShape() or
		inSource.getDtype() != inSelf.getDtype() or
		inRowMap.getDtype() != oa::ScalarType::UInt32 or inRowMap.numElements() < inSelf.size(0) or
		inCount.getDtype() != oa::ScalarType::UInt32 or inCount.numElements() != 1) {
		OaLogError(oa::LogComponent::Compute,
			"ScatterRows: expected matching [T,D] tensors, UInt32 row map [T], and UInt32 count [1]");
		return {};
	}
	const oa::U32 T = static_cast<oa::U32>(inSelf.size(0));
	const oa::U32 D = static_cast<oa::U32>(inSelf.size(1));
	oa::Matrix out = oa::FnMatrix::empty(inSelf.getShape(), inSelf.getDtype());
	struct { oa::U32 T, D; } push{T, D};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write};
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	ctx.add( "ScatterRows", {&inSelf, &inSource, &inRowMap, &inCount, &out},
		access, &push, sizeof(push), divCeil(T * D, 256));
	const auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnMatrix::scatterRows,
		{&inSelf, &inSource, &inRowMap, &inCount, nullptr}, {&out});
	if (not semantic.isOk()) return {};
	if (oa::FnAutograd::isEnabled() and (inSelf.requiresGrad() or inSource.requiresGrad())) {
		auto gradFn = oa::makeShared<oa::GradScatterRows>();
		gradFn->rowMap_ = inRowMap;
		gradFn->count_ = inCount;
		gradFn->saveForBackward({inSelf, inSource});
		gradFn->setGraphInputs(oa::Vec<oa::Matrix>{inSelf, inSource});
		gradFn->sequenceNr_ = oa::FnAutograd::nextSeq();
		gradFn->outputShape_ = out.getShape();
		if (not oa::FnAutograd::attachSemantic(
			gradFn, semantic.getValue()).isOk())
		{
			return {};
		}
		out.mutAutograd().gradFn = gradFn;
	}
	return out;
}

oa::Matrix oa::FnMatrix::scatterRows(const oa::Matrix& inSelf, const oa::Matrix& inSource,
	const oa::CompactRowsResult& inPlan) {
	if (inSelf.rank() != 2 or inSource.getShape() != inSelf.getShape() or
		inSource.getDtype() != inSelf.getDtype() or
		inPlan.rowMap.getDtype() != oa::ScalarType::UInt32 or
		inPlan.rowMap.numElements() < inSelf.size(0) or
		inPlan.count.getDtype() != oa::ScalarType::UInt32 or
		inPlan.count.numElements() != 1 or
		inPlan.dispatchArgs.getDtype() != oa::ScalarType::UInt32 or
		inPlan.dispatchArgs.numElements() != 3) {
		OaLogError(oa::LogComponent::Compute,
			"ScatterRows: invalid compact-row indirect plan");
		return {};
	}
	const oa::U32 T = static_cast<oa::U32>(inSelf.size(0));
	const oa::U32 D = static_cast<oa::U32>(inSelf.size(1));
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	// Preserve every unselected row first, then touch only Count*D selected
	// values through GPU-authored indirect launch dimensions.
	oa::Matrix out = oa::FnMatrix::copy(inSelf);
	struct { oa::U32 T, D; } push{T, D};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::ReadWrite};
	const oa::Matrix* matrices[] = {
		&inSource, &inPlan.rowMap, &inPlan.count, &inPlan.dispatchArgs, &out};
	oa::MatrixDispatchDesc desc;
	desc.dispatch.kernel = "ScatterRowsIndirect";
	desc.dispatch.access = access;
	desc.dispatch.pushData = &push;
	desc.dispatch.pushSize = sizeof(push);
	desc.matrices = matrices;
	desc.indirectArgs = &inPlan.dispatchArgs;
	const oa::Status status = ctx.record( desc);
	if (not status.isOk()) {
		OaLogError(oa::LogComponent::Compute,
			"ScatterRows: failed to record indirect dispatch: %s",
			status.getMessage().cStr());
		return {};
	}
	const auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnMatrix::scatterRows,
		{&inSelf, &inSource, &inPlan.rowMap, &inPlan.count,
			&inPlan.dispatchArgs},
		{&out});
	if (not semantic.isOk()) return {};
	if (oa::FnAutograd::isEnabled() and (inSelf.requiresGrad() or inSource.requiresGrad())) {
		auto gradFn = oa::makeShared<oa::GradScatterRows>();
		gradFn->rowMap_ = inPlan.rowMap;
		gradFn->count_ = inPlan.count;
		gradFn->dispatchArgs_ = inPlan.dispatchArgs;
		gradFn->saveForBackward({inSelf, inSource});
		gradFn->setGraphInputs(oa::Vec<oa::Matrix>{inSelf, inSource});
		gradFn->sequenceNr_ = oa::FnAutograd::nextSeq();
		gradFn->outputShape_ = out.getShape();
		if (not oa::FnAutograd::attachSemantic(
			gradFn, semantic.getValue()).isOk())
		{
			return {};
		}
		out.mutAutograd().gradFn = gradFn;
	}
	return out;
}

oa::Matrix oa::FnMatrix::scatterRowsBwdSource(const oa::Matrix& inGradOut,
	const oa::Matrix& inRowMap, const oa::Matrix& inCount) {
	const oa::U32 T = static_cast<oa::U32>(inGradOut.size(0));
	const oa::U32 D = static_cast<oa::U32>(inGradOut.size(1));
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	oa::Matrix gradSrc = oa::FnMatrix::zeros(inGradOut.getShape(), inGradOut.getDtype());
	struct { oa::U32 T, D; } push{T, D};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "ScatterRowsBwd", {&inGradOut, &inRowMap, &inCount, &gradSrc},
		access, &push, sizeof(push), divCeil(T * D, 256));
	if (not lowering.commit(
		oa::detail::opRegistry::FnMatrix::scatterRowsBwdSource,
		{&inGradOut, &inRowMap, &inCount, nullptr}, {&gradSrc}).isOk())
	{
		return {};
	}
	return gradSrc;
}

oa::Matrix oa::FnMatrix::scatterRowsBwdSource(const oa::Matrix& inGradOut,
	const oa::Matrix& inRowMap, const oa::Matrix& inCount,
	const oa::Matrix& inDispatchArgs) {
	const oa::U32 T = static_cast<oa::U32>(inGradOut.size(0));
	const oa::U32 D = static_cast<oa::U32>(inGradOut.size(1));
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	oa::Matrix gradSrc = oa::FnMatrix::zeros(inGradOut.getShape(), inGradOut.getDtype());
	struct { oa::U32 T, D; } push{T, D};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write};
	const oa::Matrix* matrices[] = {
		&inGradOut, &inRowMap, &inCount, &inDispatchArgs, &gradSrc};
	oa::MatrixDispatchDesc desc;
	desc.dispatch.kernel = "ScatterRowsBwdIndirect";
	desc.dispatch.access = access;
	desc.dispatch.pushData = &push;
	desc.dispatch.pushSize = sizeof(push);
	desc.matrices = matrices;
	desc.indirectArgs = &inDispatchArgs;
	const oa::Status status = ctx.record( desc);
	if (not status.isOk()) {
		OaLogError(oa::LogComponent::Compute,
			"ScatterRowsBwdSource: failed to record indirect dispatch: %s",
			status.getMessage().cStr());
		return {};
	}
	if (not lowering.commit(
		oa::detail::opRegistry::FnMatrix::scatterRowsBwdSource,
		{&inGradOut, &inRowMap, &inCount, &inDispatchArgs},
		{&gradSrc}).isOk())
	{
		return {};
	}
	return gradSrc;
}

void oa::FnMatrix::moeRoutingBiasUpdate(const oa::Matrix& inSelectionMask,
	oa::Matrix& inOutBias, oa::I32 inExpertsPerToken, oa::F32 inGamma) {
	if (inSelectionMask.rank() != 2 or inOutBias.numElements() != inSelectionMask.size(1) or
		inOutBias.getDtype() != oa::ScalarType::Float32 or inExpertsPerToken <= 0 or
		inExpertsPerToken > inSelectionMask.size(1) or inGamma <= 0.0F) {
		OaLogError(oa::LogComponent::Compute,
			"MoeRoutingBiasUpdate: expected mask [T,E], FP32 bias [E], 0<K<=E, gamma>0");
		return;
	}
	const oa::U32 T = static_cast<oa::U32>(inSelectionMask.size(0));
	const oa::U32 E = static_cast<oa::U32>(inSelectionMask.size(1));
	struct { oa::U32 T, E, K; oa::F32 gamma; } push{
		T, E, static_cast<oa::U32>(inExpertsPerToken), inGamma};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::ReadWrite};
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	ctx.add( "MoeRoutingBiasUpdate", {&inSelectionMask, &inOutBias},
		access, &push, sizeof(push), 1);
	if (not lowering.commit(
		oa::detail::opRegistry::FnMatrix::moeRoutingBiasUpdate,
		{&inSelectionMask, &inOutBias}, {},
		{
			oa::OpAttribute::fromSignedInteger(
				"expertsPerToken", inExpertsPerToken),
			oa::OpAttribute::fromFloat("gamma", inGamma),
		}).isOk())
	{
		OaLogError(oa::LogComponent::Compute,
			"MoeRoutingBiasUpdate: semantic lowering failed");
	}
}


// ═══════════════════════════════════════════════════════════════════════════
// GPU-NATIVE OPERATIONS (VK_EXT path - zero CPU overhead)
// ═══════════════════════════════════════════════════════════════════════════
