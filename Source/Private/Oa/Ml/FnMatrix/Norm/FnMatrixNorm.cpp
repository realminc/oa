// OaFnMatrix — Normalization layers.
//
// LayerNorm, RmsNorm.

#include <Oa/Ml/FnMatrix.h>
#include <Oa/Ml/Autograd/Nodes.h>
#include <Oa/Core/Matrix.h>
#include <Oa/Core/Types.h>
#include <Oa/Core/BufferAccess.h>
#include <Oa/Core/Operation.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Core/Validation.h>
#include "../../Autograd/AutogradAttach.gen.h"

#include <cassert>

// Normalization
OaMatrix OaFnMatrix::LayerNorm(
	const OaMatrix& InSelf, const OaMatrix& InWeight, const OaMatrix& InBias, OaF32 InEps) {
	auto& ctx = OaContext::GetDefault();
	// Normalize over the LAST dim; leading dims are independent rows (rank>2 safe).
	const OaI64 cols = InSelf.Rank() >= 1 ? InSelf.Size(InSelf.Rank() - 1) : InSelf.NumElements();
	const OaI64 rows = cols > 0 ? InSelf.NumElements() / cols : 1;
	assert(InWeight.NumElements() == cols and InBias.NumElements() == cols);

	OaMatrix out = OaFnMatrix::Empty(InSelf.Shape_, InSelf.Dtype_);
	const auto semantic = ctx.RecordOperation(
		OaOperationRegistry::LayerNorm, {&InSelf, &InWeight, &InBias}, {&out},
		{OaOperationAttribute::FromFloat("Eps", InEps)});
	if (not semantic.IsOk()) return {};
	struct { OaU32 Rows; OaU32 Cols; OaF32 Eps; } push{
		static_cast<OaU32>(rows), static_cast<OaU32>(cols), InEps};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("LayerNorm", {&InSelf, &InWeight, &InBias, &out}, access, &push, sizeof(push),
		static_cast<OaU32>(rows), 1, 1,
		OaOperationRegistry::LayerNorm.Name, 0,
		OaOperationRegistry::LayerNorm.Hash, 0, 0, semantic.GetValue());

	const auto attached = OaGeneratedAutogradAttach::OaFnMatrix::LayerNorm(
		out, InSelf, InWeight, InBias, InEps, semantic.GetValue());
	if (not attached.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::ML,
			"LayerNorm semantic autograd attachment failed: %s",
			attached.GetMessage().c_str());
		return {};
	}

	return out;
}

OaMatrix OaFnMatrix::RmsNorm(
	const OaMatrix& InSelf, const OaMatrix& InWeight, OaF32 InEps) {
	auto& ctx = OaContext::GetDefault();
	// Normalize over the LAST dim; all leading dims are independent rows. (Rank>2 was
	// previously mis-handled as rows=Size(0)/cols=Size(1) — it normalized the wrong
	// axis and the backward wrote dWeight with the wrong stride → NaN. [B,T,C] now
	// flattens to rows=B*T, cols=C, matching the row-major buffer the kernel expects.)
	const OaI64 cols = InSelf.Rank() >= 1 ? InSelf.Size(InSelf.Rank() - 1) : InSelf.NumElements();
	const OaI64 rows = cols > 0 ? InSelf.NumElements() / cols : 1;
	assert(InWeight.NumElements() == cols);

	OaMatrix out = OaFnMatrix::Empty(InSelf.Shape_, InSelf.Dtype_);
	const auto semantic = ctx.RecordOperation(
		OaOperationRegistry::RmsNorm, {&InSelf, &InWeight}, {&out},
		{OaOperationAttribute::FromFloat("Eps", InEps)});
	if (not semantic.IsOk()) return {};
	struct { OaU32 Rows; OaU32 Cols; OaF32 Eps; } push{
		static_cast<OaU32>(rows), static_cast<OaU32>(cols), InEps};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("RmsNorm", {&InSelf, &InWeight, &out}, access, &push, sizeof(push),
		static_cast<OaU32>(rows), 1, 1,
		OaOperationRegistry::RmsNorm.Name, 0,
		OaOperationRegistry::RmsNorm.Hash, 0, 0, semantic.GetValue());

	const auto attached = OaGeneratedAutogradAttach::OaFnMatrix::RmsNorm(
		out, InSelf, InWeight, InEps, semantic.GetValue());
	if (not attached.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::ML,
			"RmsNorm semantic autograd attachment failed: %s",
			attached.GetMessage().c_str());
		return {};
	}

	return out;
}

OaMatrix OaFnMatrix::RmsNormGated(
	const OaMatrix& InSelf, const OaMatrix& InWeight, const OaMatrix& InBias,
	const OaMatrix& InZ, OaF32 InEps, bool InNormBeforeGate) {
	auto& ctx = OaContext::GetDefault();
	// Normalize over the LAST dim; leading dims are independent rows (rank>2 safe).
	const OaI64 cols = InSelf.Rank() >= 1 ? InSelf.Size(InSelf.Rank() - 1) : InSelf.NumElements();
	const OaI64 rows = cols > 0 ? InSelf.NumElements() / cols : 1;
	assert(InWeight.NumElements() == cols);
	assert(InBias.NumElements() == cols);
	assert(InZ.Shape_ == InSelf.Shape_);

	OaMatrix out = OaFnMatrix::Empty(InSelf.Shape_, InSelf.Dtype_);
	struct {
		OaU32 Rows;
		OaU32 Cols;
		OaF32 Eps;
		OaU32 HasBias;
		OaU32 HasZ;
		OaU32 NormBeforeGate;
	} push{
		static_cast<OaU32>(rows),
		static_cast<OaU32>(cols),
		InEps,
		InBias.NumElements() > 0 ? 1u : 0u,
		InZ.NumElements() > 0 ? 1u : 0u,
		InNormBeforeGate ? 1u : 0u
	};

	OaBufferAccess access[] = {
		OaBufferAccess::Read,   // x
		OaBufferAccess::Read,   // weight
		OaBufferAccess::Read,   // bias
		OaBufferAccess::Read,   // z
		OaBufferAccess::Write   // out
	};
	ctx.Add("RmsNormGated", {&InSelf, &InWeight, &InBias, &InZ, &out}, access, &push, sizeof(push),
		static_cast<OaU32>(rows));

	// Autograd (norm_before_gate path only — the in-place gate-before-norm mode mutates x).
	if (InNormBeforeGate and OaFnAutograd::IsEnabled() and
		(InSelf.RequiresGrad() or InWeight.RequiresGrad() or InBias.RequiresGrad() or InZ.RequiresGrad())) {
		auto gradFn = OaMakeSharedPtr<OaGradRmsNormGated>();
		gradFn->Saved_ = OaVec<OaMatrix>{InSelf, InWeight, InBias, InZ};
		gradFn->SetGraphInputs(OaVec<OaMatrix>{InSelf, InWeight, InBias, InZ});
		gradFn->SequenceNr_ = OaFnAutograd::NextSeq();
		gradFn->Eps_ = InEps;
		gradFn->OutputShape_ = out.GetShape();
		out.MutAutograd().GradFn = gradFn;
	}
	return out;
}

OaFnMatrix::OaRmsNormGatedBwdResult OaFnMatrix::RmsNormGatedBwd(
	const OaMatrix& InX, const OaMatrix& InWeight, const OaMatrix& InBias,
	const OaMatrix& InZ, const OaMatrix& InGradOutput, OaF32 InEps) {
	auto& ctx = OaContext::GetDefault();
	// Normalize over the LAST dim; leading dims are independent rows (rank>2 safe).
	const OaI64 cols = InX.Rank() >= 1 ? InX.Size(InX.Rank() - 1) : InX.NumElements();
	const OaI64 rows = cols > 0 ? InX.NumElements() / cols : 1;
	const OaU32 hasBias = (InBias.NumElements() > 0) ? 1u : 0u;

	OaMatrix dX = OaFnMatrix::Zeros(InX.GetShape(), InX.Dtype_);
	OaMatrix dZ = OaFnMatrix::Zeros(InZ.GetShape(), InZ.Dtype_);
	OaMatrix dWrow = OaFnMatrix::Zeros(OaMatrixShape{rows, cols}, InX.Dtype_);
	OaMatrix dBrow = OaFnMatrix::Zeros(OaMatrixShape{rows, cols}, InX.Dtype_);

	struct {
		OaU32 Rows, Cols; OaF32 Eps; OaU32 HasBias;
	} push{ static_cast<OaU32>(rows), static_cast<OaU32>(cols), InEps, hasBias };

	OaBufferAccess access[] = {
		OaBufferAccess::Read,   // x
		OaBufferAccess::Read,   // w
		OaBufferAccess::Read,   // bias
		OaBufferAccess::Read,   // z
		OaBufferAccess::Read,   // dy
		OaBufferAccess::Write,  // dx
		OaBufferAccess::Write,  // dz
		OaBufferAccess::Write,  // dw_row
		OaBufferAccess::Write   // dbias_row
	};
	ctx.Add("RmsNormGatedBwd",
		{&InX, &InWeight, &InBias, &InZ, &InGradOutput, &dX, &dZ, &dWrow, &dBrow},
		access, &push, sizeof(push), static_cast<OaU32>(rows));

	OaRmsNormGatedBwdResult result;
	result.DX = dX;
	result.DZ = dZ;
	result.DWeight = OaFnMatrix::Sum(dWrow, 0).Reshape(InWeight.GetShape());
	result.DBias = hasBias ? OaFnMatrix::Sum(dBrow, 0).Reshape(InBias.GetShape()) : InBias;
	return result;
}

OaMatrix OaFnMatrix::HeavyTailActivation(const OaMatrix& InSelf) {
	// Heavy-tail activation for data-dependent A
	// f(x) = 1 + x        if x >= 0
	//      = 1 / (1 - x)  if x < 0
	// This is positive, continuous, and differentiable at x = 0
	// Rewritten: f(x) = max(x,0) + 1/(1 + max(-x,0))
	auto pos = OaFnMatrix::ClampMin(InSelf, 0.0f);           // max(x, 0)
	auto neg = OaFnMatrix::Neg(InSelf);
	auto negPos = OaFnMatrix::ClampMin(neg, 0.0f);            // max(-x, 0)
	auto onePlus = OaFnMatrix::AddScalar(negPos, 1.0f);      // 1 + max(-x, 0)
	auto reciprocal = OaFnMatrix::Reciprocal(onePlus);         // 1 / (1 + max(-x, 0))
	return pos + reciprocal;
}

// ResidualRmsNorm — fused residual-add + RmsNorm in a single kernel dispatch.
// Out = RmsNorm(A + B, Weight, Eps). Also returns the pre-norm residual (A + B).
// Autograd: backward flows through the compositional Add + RmsNorm path, so no
// dedicated grad-fn is needed. When autograd is enabled, use the unfused path
// (Add + RmsNorm) so the tape records both ops. When disabled, use the fused
// kernel for one fewer dispatch.
OaFnMatrix::OaResidualRmsNormResult OaFnMatrix::ResidualRmsNorm(
	const OaMatrix& InA, const OaMatrix& InB,
	const OaMatrix& InWeight, OaF32 InEps)
{
	if (OaFnAutograd::IsEnabled() and
		(InA.RequiresGrad() or InB.RequiresGrad() or InWeight.RequiresGrad())) {
		auto residual = OaFnMatrix::Add(InA, InB);
		auto out = OaFnMatrix::RmsNorm(residual, InWeight, InEps);
		return {.Out = out, .Residual = residual};
	}

	auto& ctx = OaContext::GetDefault();
	const OaI64 cols = InA.Rank() >= 1 ? InA.Size(InA.Rank() - 1) : InA.NumElements();
	const OaI64 rows = cols > 0 ? InA.NumElements() / cols : 1;
	assert(InWeight.NumElements() == cols);
	assert(InB.NumElements() == InA.NumElements());

	OaMatrix out = OaFnMatrix::Empty(InA.Shape_, InA.Dtype_);
	OaMatrix residual = OaFnMatrix::Empty(InA.Shape_, InA.Dtype_);

	struct {
		OaU32 Rows;
		OaU32 Cols;
		OaF32 Eps;
	} push{static_cast<OaU32>(rows), static_cast<OaU32>(cols), InEps};

	OaBufferAccess access[] = {
		OaBufferAccess::Read,   // a
		OaBufferAccess::Read,   // b
		OaBufferAccess::Read,   // weight
		OaBufferAccess::Write,  // residual_out
		OaBufferAccess::Write   // out
	};
	ctx.Add("ResidualRmsNorm", {&InA, &InB, &InWeight, &residual, &out},
		access, &push, sizeof(push), static_cast<OaU32>(rows));

	return {.Out = out, .Residual = residual};
}
