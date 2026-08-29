// Manual implementations for element-wise matrix operations.
// Generation skips these because they have body = "manual_session" in
// CoreFnMatrixElemwise.toml. This file is the source of truth for Add/Sub/Mul/Div
// broadcast dispatch; operation generation never overwrites this file.

#include <oa/core/matrix.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/log.h>
#include <oa/core/types.h>
#include <oa/core/bufferAccess.h>
#include <oa/runtime/executionSession.h>
#include <oa/core/op.h>
#include <oa/core/autograd/matrix/autogradDtype.h>
#include "../../autograd/autogradAttach.gen.h"

static oa::U32 divCeil(oa::U32 inA, oa::U32 inB) { return (inA + inB - 1) / inB; }

// ═════════════════════════════════════════════════════════════════════════════
// Broadcast dispatch helpers
// ═════════════════════════════════════════════════════════════════════════════

struct PushBcast {
	oa::U32 total;
	oa::U32 rank;
	oa::U32 outDims[OA_MAX_TENSOR_DIMS];
	oa::U32 aStrides[OA_MAX_TENSOR_DIMS];
	oa::U32 bStrides[OA_MAX_TENSOR_DIMS];
};

static void fillPushBcast(PushBcast& outPush, const oa::MatrixShape& inOutShape,
	                      const oa::Array<oa::I64, OA_MAX_TENSOR_DIMS>& inAStrides,
	                      const oa::Array<oa::I64, OA_MAX_TENSOR_DIMS>& inBStrides) {
	outPush.total = static_cast<oa::U32>(inOutShape.numElements());
	outPush.rank = static_cast<oa::U32>(inOutShape.rank);
	for (oa::I32 d = 0; d < inOutShape.rank; ++d) {
		const auto storageIdx = static_cast<oa::Usize>(d);
		outPush.outDims[storageIdx] =
			static_cast<oa::U32>(inOutShape.dims[storageIdx]);
		outPush.aStrides[storageIdx] =
			static_cast<oa::U32>(inAStrides[storageIdx]);
		outPush.bStrides[storageIdx] =
			static_cast<oa::U32>(inBStrides[storageIdx]);
	}
}

using BinaryAutogradAttach = oa::Status (*)(oa::Matrix&, const oa::Matrix&,
	const oa::Matrix&, oa::U32);

static oa::Matrix dispatchBinary_(
	const oa::Matrix& inA,
	const oa::Matrix& inB,
	const char* inKernelName,
	const char* inBroadcastKernelName,
	const oa::OpContract& inContract,
	BinaryAutogradAttach inAttach)
{
	const auto inferredShape = oa::inferBinaryOpShape(inContract, inA, inB);
	if (not inferredShape.isOk()) return {};

	auto& ctx = oa::ExecutionSession::getActive();
	oa::Matrix out = oa::FnMatrix::empty(inferredShape.getValue(), inA.getDtype());
	const auto semantic = ctx.recordOp(
		inContract, {&inA, &inB}, {&out});
	if (not semantic.isOk()) return {};

	oa::BufferAccess access[] = {
		oa::BufferAccess::Read,
		oa::BufferAccess::Read,
		oa::BufferAccess::Write,
	};
	if (inA.getShape() == inB.getShape()) {
		const oa::U32 count = static_cast<oa::U32>(out.numElements());
		struct { oa::U32 Count; } push{count};
		ctx.add( inKernelName, {&inA, &inB, &out}, access,
			&push, sizeof(push), divCeil(count, 256), 1, 1,
			inContract.name, 0, inContract.hash, 0, 0,
			semantic.getValue());
	} else {
		const auto aStrides = inA.getShape().broadcastStrides(out.getShape());
		const auto bStrides = inB.getShape().broadcastStrides(out.getShape());
		PushBcast push{};
		fillPushBcast(push, out.getShape(), aStrides, bStrides);
		ctx.add( inBroadcastKernelName, {&inA, &inB, &out}, access,
			&push, sizeof(push), divCeil(push.total, 256), 1, 1,
			inContract.name, 0, inContract.hash, 0, 0,
			semantic.getValue());
	}

	const auto attached = inAttach(out, inA, inB, semantic.getValue());
	if (not attached.isOk()) {
		OaLogError(oa::LogComponent::Compute,
			"{} semantic autograd attachment failed: {}",
			oa::String(inContract.name).cStr(), attached.getMessage().cStr());
		return {};
	}
	return out;
}

// ═════════════════════════════════════════════════════════════════════════════
// Element-wise binary ops (broadcast-aware)
// ═════════════════════════════════════════════════════════════════════════════

oa::Matrix oa::FnMatrix::add(const oa::Matrix& inA, const oa::Matrix& inB) {
	return dispatchBinary_(inA, inB, "Add", "AddBcast",
		oa::detail::opRegistry::FnMatrix::add,
		&oa::detail::generatedAutogradAttach::FnMatrix::add);
}

oa::Matrix oa::FnMatrix::sub(const oa::Matrix& inA, const oa::Matrix& inB) {
	return dispatchBinary_(inA, inB, "Sub", "SubBcast",
		oa::detail::opRegistry::FnMatrix::sub,
		&oa::detail::generatedAutogradAttach::FnMatrix::sub);
}

oa::Matrix oa::FnMatrix::mul(const oa::Matrix& inA, const oa::Matrix& inB) {
	return dispatchBinary_(inA, inB, "Mul", "MulBcast",
		oa::detail::opRegistry::FnMatrix::mul,
		&oa::detail::generatedAutogradAttach::FnMatrix::mul);
}

oa::Matrix oa::FnMatrix::div(const oa::Matrix& inA, const oa::Matrix& inB) {
	return dispatchBinary_(inA, inB, "Div", "DivBcast",
		oa::detail::opRegistry::FnMatrix::div,
		&oa::detail::generatedAutogradAttach::FnMatrix::div);
}

// ═════════════════════════════════════════════════════════════════════════════
// Element-wise unary ops
// ═════════════════════════════════════════════════════════════════════════════
// Exp, Reciprocal, ClampMax, ClampMin are now auto-generated in FnMatrixElemwise.gen.cpp

// ═════════════════════════════════════════════════════════════════════════════
// dtype cast — bf16 ⇆ fp32 (mixed-precision boundary; e.g. optimizer masters)
// ═════════════════════════════════════════════════════════════════════════════
static bool isLowPrecDtype(oa::ScalarType inDtype) {
	return inDtype == oa::ScalarType::BFloat16;
}

static bool lowerCastInto(
	oa::ExecutionSession& inContext, const oa::Matrix& inSrc, oa::Matrix& outDst)
{
	if ((inSrc.getDtype() != oa::ScalarType::Float32
			and inSrc.getDtype() != oa::ScalarType::BFloat16)
		or (outDst.getDtype() != oa::ScalarType::Float32
			and outDst.getDtype() != oa::ScalarType::BFloat16))
	{
		OaLogError(oa::LogComponent::Compute,
			"CastInto admits only the proven Float32/BFloat16 conversion boundary");
		return false;
	}
	if (inSrc.getShape() != outDst.getShape()) {
		OaLogError(oa::LogComponent::Compute,
			"CastInto requires matching source and destination shapes");
		return false;
	}
	const oa::U32 count = static_cast<oa::U32>(outDst.numElements());
	struct PushCast { oa::U32 Count; } push{count};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};

	const bool srcLow = isLowPrecDtype(inSrc.getDtype());
	const bool dstLow = isLowPrecDtype(outDst.getDtype());

	if (srcLow && !dstLow) {
		// bf16 → fp32: one thread per element (fp32 store is race-free).
		inContext.add( "CastBf16ToF32", {&inSrc, &outDst}, access, &push, sizeof(push),
			divCeil(count, 256));
	} else if (!srcLow && dstLow) {
		// fp32 → bf16: pair-packed, one thread per two elements (race-free store).
		inContext.add( "CastF32ToBf16", {&inSrc, &outDst}, access, &push, sizeof(push),
			divCeil((count + 1) / 2, 256));
	} else {
		// Same precision class — plain elementwise copy of the raw storage.
		if (inSrc.byteSize() != outDst.byteSize()) {
			OaLogError(oa::LogComponent::Compute,
				"CastInto has no conversion kernel for the requested dtype pair");
			return false;
		}
		inContext.add( "Copy", {&inSrc, &outDst}, access, &push, sizeof(push),
			divCeil(count, 256));
	}
	return true;
}

void oa::FnMatrix::castInto(const oa::Matrix& inSrc, oa::Matrix& outDst) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	if (not lowerCastInto(ctx, inSrc, outDst)) return;
	const auto status = lowering.commit(
		oa::detail::opRegistry::FnMatrix::castInto,
		{&inSrc, &outDst}, {&outDst});
	if (not status.isOk()) {
		OaLogError(oa::LogComponent::Compute,
			"CastInto semantic recording failed: {}",
			status.getMessage().cStr());
	}
}

oa::Matrix oa::FnMatrix::cast(const oa::Matrix& inSrc, oa::ScalarType inDtype) {
	if (inSrc.getDtype() == inDtype) return inSrc;  // no-op: same dtype
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	oa::Matrix out = oa::FnMatrix::empty(inSrc.getShape(), inDtype);
	if (not lowerCastInto(ctx, inSrc, out)) return {};
	const auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnMatrix::cast,
		{&inSrc}, {&out},
		{oa::OpAttribute::fromSignedInteger(
			"dtype", static_cast<oa::I64>(inDtype))});
	if (not semantic.isOk()) return {};
	// Differentiable boundary: backward casts the grad back to the source dtype.
	// Enables fp32 compute islands (SSM scans) inside a bf16 autograd graph.
	if (oa::FnAutograd::isEnabled() and inSrc.requiresGrad()) {
		auto gradFn = oa::makeShared<oa::GradCast>();
		gradFn->srcDtype_ = inSrc.getDtype();
		gradFn->setGraphInputs(oa::Vector<oa::Matrix>{inSrc});
		gradFn->sequenceNr_ = oa::FnAutograd::nextSeq();
		out.setRequiresGrad(true);
		out.mutAutograd().gradFn = gradFn;
		if (not oa::FnAutograd::attachSemantic(
			gradFn, semantic.getValue()).isOk())
		{
			return {};
		}
	}
	return out;
}
