// DeviceMatrixFn.cpp — Shared utilities and runtime management for oa::Matrix operations
//
// This file contains:
// - active-engine weight dtype resolution
// - Shared helper functions (DivCeil, matrixFlatToElementOffset, etc.)
// - oa::Matrix member functions (To, clone, etc.)
//
// actual matrix operations are modularized into category-specific files:
// - DeviceMatrixFnAlloc.cpp      — Empty, Zeros, Ones, Full, Rand, RandN, FromBytes, CausalMask, CopyToHost
// - DeviceMatrixFnElemwise.cpp   — Add, Sub, Mul, Div, scale, Neg, log, Sqrt, Pow
// - FnMatrixBlas.cpp             - MatMul and Linear semantic context ops
// - DeviceMatrixFnActivation.cpp — Gelu, Silu, Relu, Tanh, Sigmoid, LeakyRelu, Elu, Mish
// - DeviceMatrixFnReduce.cpp     — Sum, Mean, Max, Softmax, LogSoftmax
// - DeviceMatrixFnNorm.cpp       — LayerNorm, RmsNorm
// - DeviceMatrixFnPool.cpp       — AvgPool2d, MaxPool2d
// - FnMatrixIndex.cpp — Gather, Slice, Concat, Transpose, reshape, RepeatInterleave,
//   CausalMask, Equal, CompactRows and ScatterRows
// - DeviceMatrixFnView.cpp       — view operations and shape manipulations
// - DeviceMatrixFnModules.cpp    — BiasAdd, Conv1d, Conv2d (NN layer operations)
// - DeviceMatrixFnOptim.cpp      — Optimizer operations (AdamW, etc.)
// - FnMatrixRng.cpp              — Legacy RNG functions removed (use Rand/RandN/etc. instead)

#include <oa/core/matrix.h>
#include <oa/core/matrixAccess.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/fnmatrix/fnMatrixInternal.h>
#include <oa/core/log.h>
#include <oa/core/std/memory.h>
#include <oa/core/op.h>
#include <oa/core/status.h>
#include <oa/core/types.h>
#include <oa/core/bufferAccess.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/dispatchDesc.h>

#include <assert.h>

// Shared Helper Functions
static oa::U32 divCeil(oa::U32 inA, oa::U32 inB) { return (inA + inB - 1) / inB; }

oa::ScalarType oa::FnMatrix::weightDtype() {
	const auto* context = oa::ExecutionSession::getActivePtr();
	return context != nullptr
		? context->weightDtype()
		: oa::ScalarType::Float32;
}

oa::Status oa::FnMatrix::completeRecordedWork(oa::ExecutionSession& inContext) {
	auto submitted = inContext.submit();
	if (not submitted.isOk()) {
		const auto& status = submitted.getStatus();
		if (
			inContext.nodeCount() == 0U
			and status.getCode() == oa::StatusCode::FailedPrecondition
			and status.getMessage()
				== "oa::ExecutionSession::submit requires recorded device work"
		) {
			return oa::Status::ok();
		}
		return status;
	}
	return inContext.wait(submitted.getValue());
}

// oa::Matrix member Functions
oa::F32 oa::FnMatrix::scalar(const oa::Matrix& inSrc) {
	assert(inSrc.numElements() == 1 && "scalar() requires single-element tensor");
	auto& ctx = oa::ExecutionSession::getActive();
	auto completionStatus = completeRecordedWork(ctx);
	assert(completionStatus.isOk()
		and "context submission failed before scalar readback");
	return inSrc.at(0);
}

oa::Matrix oa::Matrix::clone() const {
	return hasStorage() ? oa::FnMatrix::copy(*this) : oa::Matrix{};
}

// in-Place operations
static oa::Result<oa::U32> recordBinaryInPlaceSemantic(
	oa::ExecutionSession& inContext,
	oa::Matrix& inSelf,
	const oa::Matrix& inOther,
	const oa::OpContract& inContract)
{
	return inContext.recordOp(
		inContract,
		{&inSelf, &inOther}, {&inSelf});
}

static void binaryInPlace(
	oa::Matrix& inSelf,
	const oa::Matrix& inOther,
	const char* inKernel,
	const char* inBroadcastKernel,
	const oa::OpContract& inContract)
{
	const oa::String operation(inContract.name);
	const auto inferredShape = oa::inferBinaryOpShape(
		inContract, inSelf, inOther);
	if (not inferredShape.isOk()) {
		OaLogError(oa::LogComponent::Compute, "%s validation failed: %s",
			operation.cStr(),
			inferredShape.getStatus().getMessage().cStr());
		return;
	}
	if (inferredShape.getValue() != inSelf.getShape()) {
		OaLogError(oa::LogComponent::Compute,
			"%s cannot expand the mutated input through broadcasting",
			operation.cStr());
		return;
	}

	auto& ctx = oa::ExecutionSession::getActive();
	if (inSelf.getShape() == inOther.getShape()) {
		const auto semantic = recordBinaryInPlaceSemantic(
			ctx, inSelf, inOther, inContract);
		if (not semantic.isOk()) {
			OaLogError(oa::LogComponent::Compute,
				"%s semantic recording failed: %s",
				operation.cStr(),
				semantic.getStatus().getMessage().cStr());
			return;
		}
		oa::U32 n = static_cast<oa::U32>(inSelf.numElements());
		struct { oa::U32 Count; } push{n};
		oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write};
		ctx.add( inKernel, {&inSelf, &inOther, &inSelf}, access,
			&push, sizeof(push), divCeil(n, 256), 1, 1, inContract.name, 0,
			inContract.hash, 0, 0, semantic.getValue());
		return;
	}
	const oa::MatrixShape outShape = inferredShape.getValue();

	auto aStrides = inSelf.getShape().broadcastStrides(outShape);
	auto bStrides = inOther.getShape().broadcastStrides(outShape);
	struct PushBcast {
		oa::U32 total;
		oa::U32 rank;
		oa::U32 outDims[OA_MAX_TENSOR_DIMS];
		oa::U32 aStrides[OA_MAX_TENSOR_DIMS];
		oa::U32 bStrides[OA_MAX_TENSOR_DIMS];
	};
	PushBcast push{};
	push.total = static_cast<oa::U32>(outShape.numElements());
	push.rank = static_cast<oa::U32>(outShape.rank);
	for (oa::I32 d = 0; d < outShape.rank; ++d) {
		const auto storageIdx = static_cast<oa::Usize>(d);
		push.outDims[storageIdx] =
			static_cast<oa::U32>(outShape.dims[storageIdx]);
		push.aStrides[storageIdx] =
			static_cast<oa::U32>(aStrides[storageIdx]);
		push.bStrides[storageIdx] =
			static_cast<oa::U32>(bStrides[storageIdx]);
	}
	const auto semantic = recordBinaryInPlaceSemantic(
		ctx, inSelf, inOther, inContract);
	if (not semantic.isOk()) {
		OaLogError(oa::LogComponent::Compute,
			"%s semantic recording failed: %s",
			operation.cStr(),
			semantic.getStatus().getMessage().cStr());
		return;
	}
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( inBroadcastKernel,
		{&inSelf, &inOther, &inSelf}, access, &push, sizeof(push),
		divCeil(push.total, 256), 1, 1, inContract.name, 0,
		inContract.hash, 0, 0, semantic.getValue());
}

void oa::FnMatrix::addInPlace(oa::Matrix& inSelf, const oa::Matrix& inOther) {
	binaryInPlace(inSelf, inOther, "Add", "AddBcast",
		oa::detail::opRegistry::FnMatrix::addInPlace);
}

void oa::FnMatrix::subInPlace(oa::Matrix& inSelf, const oa::Matrix& inOther) {
	binaryInPlace(inSelf, inOther, "Sub", "SubBcast",
		oa::detail::opRegistry::FnMatrix::subInPlace);
}

void oa::FnMatrix::mulInPlace(oa::Matrix& inSelf, const oa::Matrix& inOther) {
	binaryInPlace(inSelf, inOther, "Mul", "MulBcast",
		oa::detail::opRegistry::FnMatrix::mulInPlace);
}

void oa::FnMatrix::divInPlace(oa::Matrix& inSelf, const oa::Matrix& inOther) {
	binaryInPlace(inSelf, inOther, "Div", "DivBcast",
		oa::detail::opRegistry::FnMatrix::divInPlace);
}

void oa::FnMatrix::scaleInPlace(oa::Matrix& inSelf, oa::F32 inScalar) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	oa::U32 n = static_cast<oa::U32>(inSelf.numElements());
	struct { oa::U32 Count; oa::F32 alpha; } push{n, inScalar};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "Scale", {&inSelf, &inSelf}, access, &push, sizeof(push), divCeil(n, 256));
	const auto status = lowering.commit(
		oa::detail::opRegistry::FnMatrix::scaleInPlace,
		{&inSelf}, {&inSelf},
		{oa::OpAttribute::fromFloat("scalar", inScalar)});
	if (not status.isOk()) {
		OaLogError(oa::LogComponent::Compute,
			"ScaleInPlace semantic recording failed: %s",
			status.getMessage().cStr());
	}
}

void oa::FnMatrix::addScalarInPlace(oa::Matrix& inSelf, oa::F32 inScalar) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	oa::U32 n = static_cast<oa::U32>(inSelf.numElements());
	struct { oa::U32 Count; oa::F32 scalar; } push{n, inScalar};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "AddScalar", {&inSelf, &inSelf}, access, &push, sizeof(push), divCeil(n, 256));
	const auto status = lowering.commit(
		oa::detail::opRegistry::FnMatrix::addScalarInPlace,
		{&inSelf}, {&inSelf},
		{oa::OpAttribute::fromFloat("scalar", inScalar)});
	if (not status.isOk()) {
		OaLogError(oa::LogComponent::Compute,
			"AddScalarInPlace semantic recording failed: %s",
			status.getMessage().cStr());
	}
}

void oa::FnMatrix::subScalarInPlace(oa::Matrix& inSelf, oa::F32 inScalar) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	oa::U32 n = static_cast<oa::U32>(inSelf.numElements());
	struct { oa::U32 Count; oa::F32 scalar; } push{n, inScalar};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "SubScalar", {&inSelf, &inSelf}, access, &push, sizeof(push), divCeil(n, 256));
	const auto status = lowering.commit(
		oa::detail::opRegistry::FnMatrix::subScalarInPlace,
		{&inSelf}, {&inSelf},
		{oa::OpAttribute::fromFloat("scalar", inScalar)});
	if (not status.isOk()) {
		OaLogError(oa::LogComponent::Compute,
			"SubScalarInPlace semantic recording failed: %s",
			status.getMessage().cStr());
	}
}

void oa::FnMatrix::divScalarInPlace(oa::Matrix& inSelf, oa::F32 inScalar) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	oa::U32 n = static_cast<oa::U32>(inSelf.numElements());
	struct { oa::U32 Count; oa::F32 scalar; } push{n, inScalar};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "DivScalar", {&inSelf, &inSelf}, access, &push, sizeof(push), divCeil(n, 256));
	const auto status = lowering.commit(
		oa::detail::opRegistry::FnMatrix::divScalarInPlace,
		{&inSelf}, {&inSelf},
		{oa::OpAttribute::fromFloat("scalar", inScalar)});
	if (not status.isOk()) {
		OaLogError(oa::LogComponent::Compute,
			"DivScalarInPlace semantic recording failed: %s",
			status.getMessage().cStr());
	}
}

void oa::FnMatrix::fillInPlace(oa::Matrix& inSelf, oa::F32 inValue) {
	auto& ctx = oa::ExecutionSession::getActive();
	const auto semantic = ctx.recordOp(
		oa::detail::opRegistry::FnMatrix::fillInPlace,
		{&inSelf}, {&inSelf},
		{oa::OpAttribute::fromFloat("value", inValue)});
	if (not semantic.isOk()) return;
	oa::U32 n = static_cast<oa::U32>(inSelf.numElements());
	struct { oa::U32 Count; oa::F32 Value; } push{n, inValue};
	oa::BufferAccess access[] = {oa::BufferAccess::Write};
	ctx.add( "Fill", {&inSelf}, access, &push, sizeof(push), divCeil(n, 256),
		1, 1, oa::detail::opRegistry::FnMatrix::fillInPlace.name, 0,
		oa::detail::opRegistry::FnMatrix::fillInPlace.hash, 0, 0,
		semantic.getValue());
}

// Multi-tensor batch operations — fuse N dispatches into one dispatch.
// Pattern-matched to AdamwMany4: up to 4 tensors, dispatch size = max count.

void oa::FnMatrix::multiFill(oa::Span<oa::Matrix> inTensors, oa::F32 inValue) {
	auto& ctx = oa::ExecutionSession::getActive();
	const oa::U32 total = static_cast<oa::U32>(inTensors.size());
	if (total == 0) return;

	// The MultiMatrixFill shader hardcodes 4 bound buffers. The runtime prepends
	// one bindless index per *bound* buffer, so the push layout only matches when
	// exactly 4 buffers are bound. Use the fused dispatch for full groups of 4
	// and fall back to individual FillInPlace for the 1–3 remainder — otherwise the
	// indices/counts/Value misalign and the fill silently writes nothing (this
	// was the AdamW.zeroGrad() no-op bug for models with <4 parameters).
	oa::U32 i = 0;
	for (; i + 4 <= total; i += 4) {
		struct Push {
			oa::U32 count0;
			oa::U32 count1;
			oa::U32 count2;
			oa::U32 count3;
			oa::F32 value;
		} push{};
		oa::U32 maxCount = 0;
		oavk::Buffer bufs[4];
		for (oa::U32 j = 0; j < 4; ++j) {
			bufs[j] = oa::MatrixAccess::descriptor(inTensors[i + j]);
			oa::U32 c = static_cast<oa::U32>(inTensors[i + j].numElements());
			if (c > maxCount) maxCount = c;
			if (j == 0) push.count0 = c;
			if (j == 1) push.count1 = c;
			if (j == 2) push.count2 = c;
			if (j == 3) push.count3 = c;
		}
		push.value = inValue;
		oa::BufferAccess access[4] = {
			oa::BufferAccess::Write, oa::BufferAccess::Write,
			oa::BufferAccess::Write, oa::BufferAccess::Write};
		ctx.add( "MultiMatrixFill", oa::Span<oavk::Buffer>(bufs, 4), access, &push, sizeof(push), divCeil(maxCount, 256));
	}
	for (; i < total; ++i) fillInPlace(inTensors[i], inValue);
}

void oa::FnMatrix::multiAdd(oa::Span<oa::Matrix> inDst, oa::Span<const oa::Matrix> inSrc) {
	auto& ctx = oa::ExecutionSession::getActive();
	const oa::U32 total = static_cast<oa::U32>(inDst.size());
	if (total == 0U) return;
	if (total != static_cast<oa::U32>(inSrc.size())) {
		OaLogError(oa::LogComponent::Compute,
			"MultiAdd requires matching destination/source counts");
		return;
	}

	// The fused shader hardcodes eight bindless indices, so only complete groups
	// of four pairs have the declared push layout. Remainders use the direct
	// lowering, which also preserves broadcast behavior.
	oa::U32 index = 0U;
	while (index < total) {
		oa::Bool canFuse = index + 4U <= total;
		for (oa::U32 pair = 0U; pair < 4U and canFuse; ++pair) {
			canFuse = inDst[index + pair].getShape()
				== inSrc[index + pair].getShape();
		}
		if (not canFuse) {
			addInPlace(inDst[index], inSrc[index]);
			++index;
			continue;
		}

		struct Push {
			oa::U32 count0;
			oa::U32 count1;
			oa::U32 count2;
			oa::U32 count3;
		} push{};
		oa::U32* counts[] = {
			&push.count0, &push.count1, &push.count2, &push.count3,
		};
		oa::U32 maxCount = 0U;
		const oa::Matrix* matrices[8];
		oa::BufferAccess access[8];
		oa::U32 operations[4];
		for (oa::U32 pair = 0U; pair < 4U; ++pair) {
			auto& dst = inDst[index + pair];
			const auto& src = inSrc[index + pair];
			const auto semantic = recordBinaryInPlaceSemantic(
				ctx, dst, src, oa::detail::opRegistry::FnMatrix::addInPlace);
			if (not semantic.isOk()) {
				OaLogError(oa::LogComponent::Compute,
					"MultiAdd semantic recording failed: %s",
					semantic.getStatus().getMessage().cStr());
				return;
			}
			operations[pair] = semantic.getValue();
			matrices[pair * 2U] = &dst;
			matrices[pair * 2U + 1U] = &src;
			access[pair * 2U] = oa::BufferAccess::ReadWrite;
			access[pair * 2U + 1U] = oa::BufferAccess::Read;
			*counts[pair] = static_cast<oa::U32>(dst.numElements());
			maxCount = oa::max(maxCount, *counts[pair]);
		}

		oa::ComputeDispatchDesc dispatch;
		dispatch.operation = "MultiMatrixAdd";
		dispatch.semanticOps = operations;
		dispatch.kernel = "MultiMatrixAdd";
		dispatch.access = access;
		dispatch.pushData = &push;
		dispatch.pushSize = sizeof(push);
		dispatch.groupsX = divCeil(maxCount, 256U);
		const auto status = ctx.record( {
			.dispatch = dispatch,
			.matrices = matrices,
		});
		if (not status.isOk()) {
			OaLogError(oa::LogComponent::Compute,
				"MultiAdd fused lowering failed: %s",
				status.getMessage().cStr());
			return;
		}
		index += 4U;
	}
}

// SSM — Selective State-Space Scan

// Note: The actual SSM implementation is in source/cpp/lib/oa/Ml/FnMatrix/Ssm/FnMatrixSsm.cpp
// topK — top-k values and indices along the last dimension. Supports 1D
// ([E] → [K]) and 2D ([T,E] → [T,K]) input, matching torch.topk's rank
// behavior. GPU kernel (one workgroup per row); k beyond the groupshared
// "taken" bound is rejected rather than silently falling back to the CPU.
oa::TopKResult oa::FnMatrix::topK(const oa::Matrix& inA, oa::I32 inK, oa::I32 inDim) {
	// VQ seeding needs the highest-norm numCodes rows (currently up to 512).
	// The cooperative kernel is intentionally correctness-first for large K;
	// routing uses tiny K and remains on its fast path through the same primitive.
	constexpr oa::I32 kTopKMaxGpu = 1024;

	const oa::I32 rank = inA.rank();
	const oa::I32 dim = (inDim < 0) ? (rank + inDim) : inDim;

	// validate rank/dim/k explicitly: asserts compile out in release, and the
	// old assert-only path read size(1) on a rank-1 tensor → OOB → SIGSEGV.
	if ((rank != 1 and rank != 2) or dim != rank - 1 or inK < 0) {
		OaLogError(oa::LogComponent::Compute,
			"topK: unsupported args (rank=%d, dim=%d, k=%d); expects 1D/2D input, "
			"last-dim only, k>=0. Returning empty result.", rank, dim, inK);
		return {oa::Matrix{}, oa::Matrix{}};
	}

	const oa::I64 T = (rank == 2) ? inA.size(0) : 1;
	const oa::I64 E = (rank == 2) ? inA.size(1) : inA.size(0);
	const oa::I32 k = (static_cast<oa::I64>(inK) <= E) ? inK : static_cast<oa::I32>(E);

	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	const oa::MatrixShape outShape = (rank == 2) ? oa::MatrixShape{T, k} : oa::MatrixShape{k};
	oa::Matrix values  = oa::FnMatrix::empty(outShape, inA.getDtype());
	oa::Matrix indices = oa::FnMatrix::empty(outShape, oa::ScalarType::Int32);

	if (k == 0) {
		if (not lowering.commit(
			oa::detail::opRegistry::FnMatrix::topK,
			{&inA}, {&values, &indices},
			{
				oa::OpAttribute::fromSignedInteger("k", inK),
				oa::OpAttribute::fromSignedInteger("dim", inDim),
			}).isOk())
		{
			return {};
		}
		return {values, indices};
	}

	// GPU path: one workgroup per row, k cooperative-argmax passes per row.
	if (k <= kTopKMaxGpu) {
		struct { oa::U32 T; oa::U32 E; oa::U32 K; } push{
			static_cast<oa::U32>(T), static_cast<oa::U32>(E), static_cast<oa::U32>(k)};
		oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write, oa::BufferAccess::Write};
		ctx.add( "TopK", {&inA, &values, &indices}, access, &push, sizeof(push),
			static_cast<oa::U32>(T));
		if (not lowering.commit(
			oa::detail::opRegistry::FnMatrix::topK,
			{&inA}, {&values, &indices},
			{
				oa::OpAttribute::fromSignedInteger("k", inK),
				oa::OpAttribute::fromSignedInteger("dim", inDim),
			}).isOk())
		{
			return {};
		}
		// topK is a normal deferred GPU operation. Its consumers establish the
		// required graph dependency; only an explicit host readback may execute.
		return {values, indices};
	}

	OaLogError(oa::LogComponent::Compute,
		"topK: k=%d exceeds the GPU limit %d; refusing a hidden CPU fallback",
		k, kTopKMaxGpu);
	return {oa::Matrix{}, oa::Matrix{}};
}

oa::Matrix oa::FnMatrix::topKMask(const oa::Matrix& inIndices, oa::I32 inNumExperts) {
	if (inIndices.rank() != 2 or inIndices.getDtype() != oa::ScalarType::Int32 or
		inIndices.size(1) <= 0 or inNumExperts <= 0) {
		OaLogError(oa::LogComponent::Compute,
			"TopKMask: expects Int32 [T,K] indices, K>0, E>0");
		return {};
	}
	const oa::U32 T = static_cast<oa::U32>(inIndices.size(0));
	const oa::U32 K = static_cast<oa::U32>(inIndices.size(1));
	const oa::U32 E = static_cast<oa::U32>(inNumExperts);
	oa::Matrix out = oa::FnMatrix::empty(
		oa::MatrixShape{static_cast<oa::I64>(T), static_cast<oa::I64>(E)}, weightDtype());
	struct { oa::U32 T, E, K; } push{T, E, K};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	ctx.add( "TopKMask", {&inIndices, &out}, access, &push, sizeof(push), divCeil(T * E, 256));
	if (not lowering.commit(
		oa::detail::opRegistry::FnMatrix::topKMask,
		{&inIndices}, {&out},
		{oa::OpAttribute::fromSignedInteger(
			"numExperts", inNumExperts)}).isOk())
	{
		return {};
	}
	return out;
}
