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
#include <Oa/Ml/Autograd.h>
#include <cassert>
#include <cstring>

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

static bool BcastKernelToOp_(const char* InKernelName, OaBcastBinOp& OutOp) {
	if (std::strcmp(InKernelName, "AddBcast") == 0) { OutOp = OaBcastBinOp::Add; return true; }
	if (std::strcmp(InKernelName, "SubBcast") == 0) { OutOp = OaBcastBinOp::Sub; return true; }
	if (std::strcmp(InKernelName, "MulBcast") == 0) { OutOp = OaBcastBinOp::Mul; return true; }
	if (std::strcmp(InKernelName, "DivBcast") == 0) { OutOp = OaBcastBinOp::Div; return true; }
	return false;
}

static OaMatrix DispatchBcast_(const OaMatrix& InA, const OaMatrix& InB, const char* InKernelName) {
	auto bcastResult = InA.Shape_.Broadcast(InB.Shape_);
	assert(bcastResult.IsOk() && "DispatchBcast_: shapes are not broadcast-compatible");
	OaMatrixShape outShape = bcastResult.GetValue();

	auto aStrides = InA.Shape_.BroadcastStrides(outShape);
	auto bStrides = InB.Shape_.BroadcastStrides(outShape);

	auto& ctx = OaContext::GetDefault();
	OaMatrix out = OaFnMatrix::Empty(outShape, InA.Dtype_);
	PushBcast push{};
	FillPushBcast(push, outShape, aStrides, bStrides);

	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add(InKernelName, {&InA, &InB, &out}, access, &push, sizeof(push), DivCeil(push.Total, 256));

	// Attach the broadcast-aware grad node. Without this the broadcast operand AND
	// everything upstream of it silently receive ZERO gradient (numel mismatch drops
	// the contribution in the tape). This was the root cause of frozen mixer params
	// (NormWeight broadcast-mul, dt_bias broadcast-add) in Mamba3 / Empyrealm.
	if (OaFnAutograd::IsEnabled() and (InA.RequiresGrad() or InB.RequiresGrad())) {
		OaBcastBinOp op{};
		if (BcastKernelToOp_(InKernelName, op)) {
			auto gradFn = OaMakeSharedPtr<OaGradBcastBinary>();
			gradFn->Op_ = op;
			gradFn->Saved_ = OaVec<OaMatrix>{InA, InB};
			gradFn->SetGraphInputs(OaVec<OaMatrix>{InA, InB});
			gradFn->SequenceNr_ = OaFnAutograd::NextSeq();
			gradFn->OutputShape_ = out.GetShape();
			out.MutAutograd().GradFn = gradFn;
		}
	}
	return out;
}

// ═════════════════════════════════════════════════════════════════════════════
// Element-wise binary ops (broadcast-aware)
// ═════════════════════════════════════════════════════════════════════════════

OaMatrix OaFnMatrix::Add(const OaMatrix& InA, const OaMatrix& InB) {
	assert((InA.GetShape() == InB.GetShape() || InA.Shape_.Broadcast(InB.Shape_).IsOk()) && "OaFnMatrix::Add requires matching or broadcast-compatible shapes");
	if (InA.GetShape() == InB.GetShape()) {
		auto& ctx = OaContext::GetDefault();
		OaU32 n = static_cast<OaU32>(InA.NumElements());
		OaMatrix out = OaFnMatrix::Empty(InA.Shape_, InA.Dtype_);
		struct { OaU32 Count; } push{n};
		OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
		ctx.Add("Add", {&InA, &InB, &out}, access, &push, sizeof(push), DivCeil(n, 256));
		if (OaFnAutograd::IsEnabled() and (InA.RequiresGrad() or InB.RequiresGrad())) {
			auto gradFn = OaMakeSharedPtr<OaGradAdd>();
			gradFn->Saved_ = OaVec<OaMatrix>{InA, InB};
			gradFn->SetGraphInputs(OaVec<OaMatrix>{InA, InB});
			gradFn->SequenceNr_ = OaFnAutograd::NextSeq();
			out.MutAutograd().GradFn = gradFn;
		}
		return out;
	}
	return DispatchBcast_(InA, InB, "AddBcast");
}

OaMatrix OaFnMatrix::Sub(const OaMatrix& InA, const OaMatrix& InB) {
	assert((InA.GetShape() == InB.GetShape() || InA.Shape_.Broadcast(InB.Shape_).IsOk()) && "OaFnMatrix::Sub requires matching or broadcast-compatible shapes");
	if (InA.GetShape() == InB.GetShape()) {
		auto& ctx = OaContext::GetDefault();
		OaU32 n = static_cast<OaU32>(InA.NumElements());
		OaMatrix out = OaFnMatrix::Empty(InA.Shape_, InA.Dtype_);
		struct { OaU32 Count; } push{n};
		OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
		ctx.Add("Sub", {&InA, &InB, &out}, access, &push, sizeof(push), DivCeil(n, 256));
		if (OaFnAutograd::IsEnabled() and (InA.RequiresGrad() or InB.RequiresGrad())) {
			auto gradFn = OaMakeSharedPtr<OaGradSub>();
			gradFn->Saved_ = OaVec<OaMatrix>{};
			gradFn->SetGraphInputs(OaVec<OaMatrix>{InA, InB});
			gradFn->SequenceNr_ = OaFnAutograd::NextSeq();
			out.MutAutograd().GradFn = gradFn;
		}
		return out;
	}
	return DispatchBcast_(InA, InB, "SubBcast");
}

OaMatrix OaFnMatrix::Mul(const OaMatrix& InA, const OaMatrix& InB) {
	assert((InA.GetShape() == InB.GetShape() || InA.Shape_.Broadcast(InB.Shape_).IsOk()) && "OaFnMatrix::Mul requires matching or broadcast-compatible shapes");
	if (InA.GetShape() == InB.GetShape()) {
		auto& ctx = OaContext::GetDefault();
		OaU32 n = static_cast<OaU32>(InA.NumElements());
		OaMatrix out = OaFnMatrix::Empty(InA.Shape_, InA.Dtype_);
		struct { OaU32 Count; } push{n};
		OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
		ctx.Add("Mul", {&InA, &InB, &out}, access, &push, sizeof(push), DivCeil(n, 256));
		if (OaFnAutograd::IsEnabled() and (InA.RequiresGrad() or InB.RequiresGrad())) {
			auto gradFn = OaMakeSharedPtr<OaGradMul>();
			gradFn->Saved_ = OaVec<OaMatrix>{InA, InB};
			gradFn->SetGraphInputs(OaVec<OaMatrix>{InA, InB});
			gradFn->SequenceNr_ = OaFnAutograd::NextSeq();
			gradFn->OutputShape_ = out.GetShape();  // central tape now normalizes upstream using this; protects bcast in GradMul bwd (the source of previous "bad optional access" during CE/head bwd)
			out.MutAutograd().GradFn = gradFn;
		}
		return out;
	}
	return DispatchBcast_(InA, InB, "MulBcast");
}

OaMatrix OaFnMatrix::Div(const OaMatrix& InA, const OaMatrix& InB) {
	assert((InA.GetShape() == InB.GetShape() || InA.Shape_.Broadcast(InB.Shape_).IsOk()) && "OaFnMatrix::Div requires matching or broadcast-compatible shapes");
	if (InA.GetShape() == InB.GetShape()) {
		auto& ctx = OaContext::GetDefault();
		OaU32 n = static_cast<OaU32>(InA.NumElements());
		OaMatrix out = OaFnMatrix::Empty(InA.Shape_, InA.Dtype_);
		struct { OaU32 Count; } push{n};
		OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
		ctx.Add("Div", {&InA, &InB, &out}, access, &push, sizeof(push), DivCeil(n, 256));
		if (OaFnAutograd::IsEnabled() and (InA.RequiresGrad() or InB.RequiresGrad())) {
			auto gradFn = OaMakeSharedPtr<OaGradDiv>();
			gradFn->Saved_ = OaVec<OaMatrix>{InA, InB};
			gradFn->SetGraphInputs(OaVec<OaMatrix>{InA, InB});
			gradFn->SequenceNr_ = OaFnAutograd::NextSeq();
			out.MutAutograd().GradFn = gradFn;
		}
		return out;
	}
	return DispatchBcast_(InA, InB, "DivBcast");
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
