// Manual implementations for element-wise matrix operations.
// Autogenerator skips these because they have body = "manual_context" in
// CoreFnMatrixElemwise.toml. This file is the source of truth for Add/Sub/Mul/Div
// broadcast dispatch — it will never be overwritten by oafnautogen.py.

#include <Oa/Core/Matrix.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/Types.h>
#include <Oa/Core/BufferAccess.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Core/Operation.h>
#include <Oa/Ml/Autograd/Nodes.h>
#include "../../../Ml/Autograd/AutogradAttach.gen.h"

static OaU32 DivCeil(OaU32 InA, OaU32 InB) { return (InA + InB - 1) / InB; }

// ═════════════════════════════════════════════════════════════════════════════
// Broadcast dispatch helpers
// ═════════════════════════════════════════════════════════════════════════════

struct PushBcast {
	OaU32 Total;
	OaU32 Rank;
	OaU32 OutDims[OA_MAX_TENSOR_DIMS];
	OaU32 AStrides[OA_MAX_TENSOR_DIMS];
	OaU32 BStrides[OA_MAX_TENSOR_DIMS];
};

static void FillPushBcast(PushBcast& OutPush, const OaMatrixShape& InOutShape,
	                      const OaStdArray<OaI64, OA_MAX_TENSOR_DIMS>& InAStrides,
	                      const OaStdArray<OaI64, OA_MAX_TENSOR_DIMS>& InBStrides) {
	OutPush.Total = static_cast<OaU32>(InOutShape.NumElements());
	OutPush.Rank = static_cast<OaU32>(InOutShape.Rank);
	for (OaI32 d = 0; d < InOutShape.Rank; ++d) {
		OutPush.OutDims[d] = static_cast<OaU32>(InOutShape.Dims[d]);
		OutPush.AStrides[d] = static_cast<OaU32>(InAStrides[d]);
		OutPush.BStrides[d] = static_cast<OaU32>(InBStrides[d]);
	}
}

using BinaryAutogradAttach = OaStatus (*)(OaMatrix&, const OaMatrix&,
	const OaMatrix&, OaSemanticOperationId);

static OaMatrix DispatchBinary_(
	const OaMatrix& InA,
	const OaMatrix& InB,
	const char* InKernelName,
	const char* InBroadcastKernelName,
	const OaOperationContract& InContract,
	BinaryAutogradAttach InAttach)
{
	const auto inferredShape = OaInferBinaryOperationShape(InContract, InA, InB);
	if (not inferredShape.IsOk()) return {};

	auto& ctx = OaContext::GetDefault();
	OaMatrix out = OaFnMatrix::Empty(inferredShape.GetValue(), InA.Dtype_);
	const auto semantic = ctx.RecordOperation(
		InContract, {&InA, &InB}, {&out});
	if (not semantic.IsOk()) return {};

	OaBufferAccess access[] = {
		OaBufferAccess::Read,
		OaBufferAccess::Read,
		OaBufferAccess::Write,
	};
	if (InA.GetShape() == InB.GetShape()) {
		const OaU32 count = static_cast<OaU32>(out.NumElements());
		struct { OaU32 Count; } push{count};
		ctx.Add(InKernelName, {&InA, &InB, &out}, access,
			&push, sizeof(push), DivCeil(count, 256), 1, 1,
			InContract.Name, 0, InContract.Hash, 0, 0,
			semantic.GetValue());
	} else {
		const auto aStrides = InA.Shape_.BroadcastStrides(out.GetShape());
		const auto bStrides = InB.Shape_.BroadcastStrides(out.GetShape());
		PushBcast push{};
		FillPushBcast(push, out.GetShape(), aStrides, bStrides);
		ctx.Add(InBroadcastKernelName, {&InA, &InB, &out}, access,
			&push, sizeof(push), DivCeil(push.Total, 256), 1, 1,
			InContract.Name, 0, InContract.Hash, 0, 0,
			semantic.GetValue());
	}

	const auto attached = InAttach(out, InA, InB, semantic.GetValue());
	if (not attached.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"%s semantic autograd attachment failed: %s",
			OaString(InContract.Name).c_str(), attached.GetMessage().c_str());
		return {};
	}
	return out;
}

// ═════════════════════════════════════════════════════════════════════════════
// Element-wise binary ops (broadcast-aware)
// ═════════════════════════════════════════════════════════════════════════════

OaMatrix OaFnMatrix::Add(const OaMatrix& InA, const OaMatrix& InB) {
	return DispatchBinary_(InA, InB, "Add", "AddBcast",
		OaOperationRegistry::Add,
		&OaGeneratedAutogradAttach::OaFnMatrix::Add);
}

OaMatrix OaFnMatrix::Sub(const OaMatrix& InA, const OaMatrix& InB) {
	return DispatchBinary_(InA, InB, "Sub", "SubBcast",
		OaOperationRegistry::Sub,
		&OaGeneratedAutogradAttach::OaFnMatrix::Sub);
}

OaMatrix OaFnMatrix::Mul(const OaMatrix& InA, const OaMatrix& InB) {
	return DispatchBinary_(InA, InB, "Mul", "MulBcast",
		OaOperationRegistry::Mul,
		&OaGeneratedAutogradAttach::OaFnMatrix::Mul);
}

OaMatrix OaFnMatrix::Div(const OaMatrix& InA, const OaMatrix& InB) {
	return DispatchBinary_(InA, InB, "Div", "DivBcast",
		OaOperationRegistry::Div,
		&OaGeneratedAutogradAttach::OaFnMatrix::Div);
}

// ═════════════════════════════════════════════════════════════════════════════
// Element-wise unary ops
// ═════════════════════════════════════════════════════════════════════════════
// Exp, Reciprocal, ClampMax, ClampMin are now auto-generated in FnMatrixElemwise.gen.cpp

// ═════════════════════════════════════════════════════════════════════════════
// Dtype cast — bf16 ⇆ fp32 (mixed-precision boundary; e.g. optimizer masters)
// ═════════════════════════════════════════════════════════════════════════════
static bool IsLowPrecDtype(OaScalarType InDtype) {
	return InDtype == OaScalarType::BFloat16 || InDtype == OaScalarType::Float16;
}

void OaFnMatrix::CastInto(const OaMatrix& InSrc, OaMatrix& OutDst) {
	auto& ctx = OaContext::GetDefault();
	const OaU32 count = static_cast<OaU32>(OutDst.NumElements());
	struct PushCast { OaU32 Count; } push{count};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};

	const bool srcLow = IsLowPrecDtype(InSrc.GetDtype());
	const bool dstLow = IsLowPrecDtype(OutDst.GetDtype());

	if (srcLow && !dstLow) {
		// bf16 → fp32: one thread per element (fp32 store is race-free).
		ctx.Add("CastBf16ToF32", {&InSrc, &OutDst}, access, &push, sizeof(push),
			DivCeil(count, 256));
	} else if (!srcLow && dstLow) {
		// fp32 → bf16: pair-packed, one thread per two elements (race-free store).
		ctx.Add("CastF32ToBf16", {&InSrc, &OutDst}, access, &push, sizeof(push),
			DivCeil((count + 1) / 2, 256));
	} else {
		// Same precision class — plain elementwise copy of the raw storage.
		ctx.Add("Copy", {&InSrc, &OutDst}, access, &push, sizeof(push),
			DivCeil(count, 256));
	}
}

OaMatrix OaFnMatrix::Cast(const OaMatrix& InSrc, OaScalarType InDtype) {
	if (InSrc.GetDtype() == InDtype) return InSrc;  // no-op: same dtype
	OaMatrix out = OaFnMatrix::Empty(InSrc.GetShape(), InDtype);
	OaFnMatrix::CastInto(InSrc, out);
	// Differentiable boundary: backward casts the grad back to the source dtype.
	// Enables fp32 compute islands (SSM scans) inside a bf16 autograd graph.
	if (OaFnAutograd::IsEnabled() and InSrc.RequiresGrad()) {
		auto gradFn = OaMakeSharedPtr<OaGradCast>();
		gradFn->SrcDtype_ = InSrc.GetDtype();
		gradFn->SetGraphInputs(OaVec<OaMatrix>{InSrc});
		gradFn->SequenceNr_ = OaFnAutograd::NextSeq();
		out.SetRequiresGrad(true);
		out.MutAutograd().GradFn = gradFn;
	}
	return out;
}
