// oa::FnMatrix — Normalization layers.
//
// LayerNorm, RmsNorm.

#include <oa/ml/fnMatrix.h>
#include <oa/core/autograd/matrix/autogradElemwise.h>
#include <oa/core/log.h>
#include <oa/ml/autograd/matrix/autogradNorm.h>
#include <oa/core/matrix.h>
#include <oa/core/types.h>
#include <oa/core/bufferAccess.h>
#include <oa/core/envFlag.h>
#include <oa/core/op.h>
#include <oa/runtime/executionSession.h>
#include <oa/core/validation.h>
#include "../../autograd/autogradAttach.gen.h"

#include <assert.h>

// Normalization
oa::Matrix oa::FnMatrix::layerNorm(
	const oa::Matrix& inSelf, const oa::Matrix& inWeight, const oa::Matrix& inBias, oa::F32 inEps) {
	auto& ctx = oa::ExecutionSession::getActive();
	// normalize over the LAST dim; leading dims are independent rows (rank>2 safe).
	const oa::I64 cols = inSelf.rank() >= 1 ? inSelf.size(inSelf.rank() - 1) : inSelf.numElements();
	const oa::I64 rows = cols > 0 ? inSelf.numElements() / cols : 1;
	assert(inWeight.numElements() == cols and inBias.numElements() == cols);

	oa::Matrix out = oa::FnMatrix::empty(inSelf.getShape(), inSelf.getDtype());
	const auto semantic = ctx.recordOp(
		oa::detail::opRegistry::FnMatrix::layerNorm, {&inSelf, &inWeight, &inBias}, {&out},
		{oa::OpAttribute::fromFloat("eps", inEps)});
	if (not semantic.isOk()) return {};
	struct { oa::U32 rows; oa::U32 cols; oa::F32 eps; } push{
		static_cast<oa::U32>(rows), static_cast<oa::U32>(cols), inEps};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write};
	const bool useNarrow = cols > 0 and cols <= 32
		and not oa::EnvFlag::isSet("OA_DISABLE_NARROW_ROW_KERNELS");
	ctx.add( useNarrow ? "LayerNormN32" : "LayerNorm",
		{&inSelf, &inWeight, &inBias, &out}, access, &push, sizeof(push),
		static_cast<oa::U32>(rows), 1, 1,
		oa::detail::opRegistry::FnMatrix::layerNorm.name, 0,
		oa::detail::opRegistry::FnMatrix::layerNorm.hash, 0, 0, semantic.getValue());

	const auto attached = oa::detail::generatedAutogradAttach::FnMatrix::layerNorm(
		out, inSelf, inWeight, inBias, inEps, semantic.getValue());
	if (not attached.isOk()) {
		OaLogError(oa::LogComponent::Ml,
			"LayerNorm semantic autograd attachment failed: %s",
			attached.getMessage().cStr());
		return {};
	}

	return out;
}

oa::Matrix oa::FnMatrix::rmsNorm(
	const oa::Matrix& inSelf, const oa::Matrix& inWeight, oa::F32 inEps) {
	auto& ctx = oa::ExecutionSession::getActive();
	// normalize over the LAST dim; all leading dims are independent rows. (rank>2 was
	// previously mis-handled as rows=size(0)/cols=size(1) — it normalized the wrong
	// axis and the backward wrote dWeight with the wrong stride → NaN. [B,T,C] now
	// flattens to rows=B*T, cols=C, matching the row-major buffer the kernel expects.)
	const oa::I64 cols = inSelf.rank() >= 1 ? inSelf.size(inSelf.rank() - 1) : inSelf.numElements();
	const oa::I64 rows = cols > 0 ? inSelf.numElements() / cols : 1;
	assert(inWeight.numElements() == cols);

	oa::Matrix out = oa::FnMatrix::empty(inSelf.getShape(), inSelf.getDtype());
	const auto semantic = ctx.recordOp(
		oa::detail::opRegistry::FnMatrix::rmsNorm, {&inSelf, &inWeight}, {&out},
		{oa::OpAttribute::fromFloat("eps", inEps)});
	if (not semantic.isOk()) return {};
	struct { oa::U32 rows; oa::U32 cols; oa::F32 eps; } push{
		static_cast<oa::U32>(rows), static_cast<oa::U32>(cols), inEps};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "RmsNorm", {&inSelf, &inWeight, &out}, access, &push, sizeof(push),
		static_cast<oa::U32>(rows), 1, 1,
		oa::detail::opRegistry::FnMatrix::rmsNorm.name, 0,
		oa::detail::opRegistry::FnMatrix::rmsNorm.hash, 0, 0, semantic.getValue());

	const auto attached = oa::detail::generatedAutogradAttach::FnMatrix::rmsNorm(
		out, inSelf, inWeight, inEps, semantic.getValue());
	if (not attached.isOk()) {
		OaLogError(oa::LogComponent::Ml,
			"RmsNorm semantic autograd attachment failed: %s",
			attached.getMessage().cStr());
		return {};
	}

	return out;
}

oa::Matrix oa::FnMatrix::rmsNormGated(
	const oa::Matrix& inSelf, const oa::Matrix& inWeight, const oa::Matrix& inBias,
	const oa::Matrix& inZ, oa::F32 inEps, bool inNormBeforeGate) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	// normalize over the LAST dim; leading dims are independent rows (rank>2 safe).
	const oa::I64 cols = inSelf.rank() >= 1 ? inSelf.size(inSelf.rank() - 1) : inSelf.numElements();
	const oa::I64 rows = cols > 0 ? inSelf.numElements() / cols : 1;
	const oa::I64 affineElements = inWeight.numElements();
	if (cols <= 0 or affineElements <= 0 or affineElements % cols != 0
		or rows % (affineElements / cols) != 0
		or (inBias.numElements() != 0 and inBias.numElements() != affineElements)
		or inZ.getShape() != inSelf.getShape())
	{
		OaLogError(oa::LogComponent::Ml,
			"RmsNormGated expects matching X/Z and broadcast affine shapes ending in the normalized dimension");
		return {};
	}
	const oa::I64 groups = affineElements / cols;

	oa::Matrix out = oa::FnMatrix::empty(inSelf.getShape(), inSelf.getDtype());
	struct {
		oa::U32 rows;
		oa::U32 cols;
		oa::U32 Groups;
		oa::F32 eps;
		oa::U32 hasBias;
		oa::U32 hasZ;
		oa::U32 NormBeforeGate;
	} push{
		static_cast<oa::U32>(rows),
		static_cast<oa::U32>(cols),
		static_cast<oa::U32>(groups),
		inEps,
		inBias.numElements() > 0 ? 1u : 0u,
		inZ.numElements() > 0 ? 1u : 0u,
		inNormBeforeGate ? 1u : 0u
	};

	oa::BufferAccess access[] = {
		oa::BufferAccess::Read,   // x
		oa::BufferAccess::Read,   // weight
		oa::BufferAccess::Read,   // bias
		oa::BufferAccess::Read,   // z
		oa::BufferAccess::Write   // out
	};
	const bool useNarrow = cols > 0 and cols <= 32
		and not oa::EnvFlag::isSet("OA_DISABLE_NARROW_ROW_KERNELS");
	const oa::Matrix* physicalBias = inBias.numElements() > 0 ? &inBias : &inWeight;
	ctx.add( useNarrow ? "RmsNormGatedN32" : "RmsNormGated",
		{&inSelf, &inWeight, physicalBias, &inZ, &out}, access, &push, sizeof(push),
		static_cast<oa::U32>(rows));

	const auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnMatrix::rmsNormGated,
		{&inSelf, &inWeight, &inBias, &inZ}, {&out},
		{
			oa::OpAttribute::fromFloat("eps", inEps),
			oa::OpAttribute::fromBoolean(
				"normBeforeGate", inNormBeforeGate),
		});
	if (not semantic.isOk()) return {};

	// autograd (norm_before_gate path only — the in-place gate-before-norm mode mutates x).
	if (inNormBeforeGate and oa::FnAutograd::isEnabled() and
		(inSelf.requiresGrad() or inWeight.requiresGrad() or inBias.requiresGrad() or inZ.requiresGrad())) {
		auto gradFn = oa::makeShared<oa::GradRmsNormGated>();
		gradFn->saveForBackward(inSelf, inWeight, inBias, inZ);
		gradFn->setGraphInputs(oa::Vector<oa::Matrix>{inSelf, inWeight, inBias, inZ});
		gradFn->sequenceNr_ = oa::FnAutograd::nextSeq();
		gradFn->eps_ = inEps;
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

oa::RmsNormGatedBwdResult oa::FnMatrix::rmsNormGatedBwd(
	const oa::Matrix& inX, const oa::Matrix& inWeight, const oa::Matrix& inBias,
	const oa::Matrix& inZ, const oa::Matrix& inGradOutput, oa::F32 inEps) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	// normalize over the LAST dim; leading dims are independent rows (rank>2 safe).
	const oa::I64 cols = inX.rank() >= 1 ? inX.size(inX.rank() - 1) : inX.numElements();
	const oa::I64 rows = cols > 0 ? inX.numElements() / cols : 1;
	const oa::I64 affineElements = inWeight.numElements();
	if (cols <= 0 or affineElements <= 0 or affineElements % cols != 0
		or rows % (affineElements / cols) != 0
		or (inBias.numElements() != 0 and inBias.numElements() != affineElements)
		or inZ.getShape() != inX.getShape()
		or inGradOutput.getShape() != inX.getShape())
	{
		OaLogError(oa::LogComponent::Ml,
			"RmsNormGatedBwd expects matching X/Z/dY and broadcast affine shapes ending in the normalized dimension");
		return {};
	}
	const oa::I64 groups = affineElements / cols;
	const oa::I64 outerRows = rows / groups;
	const oa::U32 hasBias = (inBias.numElements() > 0) ? 1u : 0u;

	oa::Matrix dX = oa::FnMatrix::zeros(inX.getShape(), inX.getDtype());
	oa::Matrix dZ = oa::FnMatrix::zeros(inZ.getShape(), inZ.getDtype());
	oa::Matrix dWrow = oa::FnMatrix::empty(
		oa::MatrixShape{outerRows, affineElements}, inX.getDtype());
	oa::Matrix dBrow = hasBias
		? oa::FnMatrix::empty(oa::MatrixShape{outerRows, affineElements}, inX.getDtype())
		: oa::FnMatrix::empty(oa::MatrixShape{1}, inX.getDtype());

	struct {
		oa::U32 rows, cols, Groups; oa::F32 eps; oa::U32 hasBias;
	} push{ static_cast<oa::U32>(rows), static_cast<oa::U32>(cols),
		static_cast<oa::U32>(groups), inEps, hasBias };

	oa::BufferAccess access[] = {
		oa::BufferAccess::Read,   // x
		oa::BufferAccess::Read,   // w
		oa::BufferAccess::Read,   // bias
		oa::BufferAccess::Read,   // z
		oa::BufferAccess::Read,   // dy
		oa::BufferAccess::Write,  // dx
		oa::BufferAccess::Write,  // dz
		oa::BufferAccess::Write,  // dw_row
		oa::BufferAccess::Write   // dbias_row
	};
	const bool useNarrow = cols > 0 and cols <= 32
		and not oa::EnvFlag::isSet("OA_DISABLE_NARROW_ROW_KERNELS");
	const oa::Matrix* physicalBias = hasBias ? &inBias : &inWeight;
	ctx.add( useNarrow ? "RmsNormGatedBwdN32" : "RmsNormGatedBwd",
		{&inX, &inWeight, physicalBias, &inZ, &inGradOutput, &dX, &dZ, &dWrow, &dBrow},
		access, &push, sizeof(push), static_cast<oa::U32>(rows));

	oa::RmsNormGatedBwdResult result;
	result.dX = dX;
	result.dZ = dZ;
	result.dWeight = oa::FnMatrix::sum(dWrow, 0).reshape(inWeight.getShape());
	// Semantic outputs must remain stored even when the corresponding optional
	// input is absent. The one-element placeholder is never attached to an input
	// gradient, and avoids authoring either a bias write or a bias reduction.
	result.dBias = hasBias
		? oa::FnMatrix::sum(dBrow, 0).reshape(inBias.getShape())
		: dBrow;
	if (not lowering.commit(
		oa::detail::opRegistry::FnMatrix::rmsNormGatedBwd,
		{&inX, &inWeight, &inBias, &inZ, &inGradOutput},
		{&result.dX, &result.dWeight, &result.dBias, &result.dZ},
		{oa::OpAttribute::fromFloat("eps", inEps)}).isOk())
	{
		return {};
	}
	return result;
}

oa::Matrix oa::FnMatrix::heavyTailActivation(const oa::Matrix& inSelf) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	// Heavy-tail activation for data-dependent A
	// f(x) = 1 + x        if x >= 0
	//      = 1 / (1 - x)  if x < 0
	// This is positive, continuous, and differentiable at x = 0
	// Rewritten: f(x) = max(x,0) + 1/(1 + max(-x,0))
	auto pos = oa::FnMatrix::clampMin(inSelf, 0.0f);           // max(x, 0)
	auto neg = oa::FnMatrix::neg(inSelf);
	auto negPos = oa::FnMatrix::clampMin(neg, 0.0f);            // max(-x, 0)
	auto onePlus = oa::FnMatrix::addScalar(negPos, 1.0f);      // 1 + max(-x, 0)
	auto reciprocal = oa::FnMatrix::reciprocal(onePlus);         // 1 / (1 + max(-x, 0))
	auto output = pos + reciprocal;
	const auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnMatrix::heavyTailActivation,
		{&inSelf},
		{&output});
	if (not semantic.isOk()) return {};
	if (auto grad = output.getGradFn()) {
		if (not oa::FnAutograd::attachSemantic(
			grad, semantic.getValue()).isOk())
		{
			return {};
		}
	}
	return output;
}

// ResidualRmsNorm — fused residual-add + RmsNorm in a single kernel dispatch.
// out = RmsNorm(A + B, weight, eps). Also returns the pre-norm residual (A + B).
// autograd keeps the fused forward and represents its two logical outputs:
// residual owns the Add adjoint, while out owns the RmsNorm adjoint whose input
// is residual. This preserves the same reverse graph without changing the
// semantic operation or dispatch count when the tape is enabled.
oa::ResidualRmsNormResult oa::FnMatrix::residualRmsNorm(
	const oa::Matrix& inA, const oa::Matrix& inB,
	const oa::Matrix& inWeight, oa::F32 inEps)
{
	auto& ctx = oa::ExecutionSession::getActive();
	const oa::I64 cols = inA.rank() >= 1 ? inA.size(inA.rank() - 1) : inA.numElements();
	const oa::I64 rows = cols > 0 ? inA.numElements() / cols : 1;
	assert(inWeight.numElements() == cols);
	assert(inB.numElements() == inA.numElements());

	oa::Matrix out = oa::FnMatrix::empty(inA.getShape(), inA.getDtype());
	oa::Matrix residual = oa::FnMatrix::empty(inA.getShape(), inA.getDtype());
	const auto semantic = ctx.recordOp(
		oa::detail::opRegistry::FnMatrix::residualRmsNorm,
		{&inA, &inB, &inWeight}, {&out, &residual},
		{oa::OpAttribute::fromFloat("eps", inEps)});
	if (not semantic.isOk()) return {};

	struct {
		oa::U32 rows;
		oa::U32 cols;
		oa::F32 eps;
	} push{static_cast<oa::U32>(rows), static_cast<oa::U32>(cols), inEps};

	oa::BufferAccess access[] = {
		oa::BufferAccess::Read,   // a
		oa::BufferAccess::Read,   // b
		oa::BufferAccess::Read,   // weight
		oa::BufferAccess::Write,  // residual_out
		oa::BufferAccess::Write   // out
	};
	ctx.add( "ResidualRmsNorm", {&inA, &inB, &inWeight, &residual, &out},
		access, &push, sizeof(push), static_cast<oa::U32>(rows), 1, 1,
		oa::detail::opRegistry::FnMatrix::residualRmsNorm.name, 0,
		oa::detail::opRegistry::FnMatrix::residualRmsNorm.hash, 0, 0,
		semantic.getValue());

	if (oa::FnAutograd::isEnabled() and
		(inA.requiresGrad() or inB.requiresGrad()))
	{
		auto residualGrad = oa::makeShared<oa::GradAdd>();
		residualGrad->setGraphInputs(oa::Vector<oa::Matrix>{inA, inB});
		residualGrad->sequenceNr_ = oa::FnAutograd::nextSeq();
		residualGrad->outputShape_ = residual.getShape();
		const auto attached = oa::FnAutograd::attachSemantic(
			residualGrad, semantic.getValue(), 1U);
		if (not attached.isOk()) return {};
		residual.mutAutograd().gradFn = residualGrad;
		residual.setRequiresGrad(true);
	}
	if (oa::FnAutograd::isEnabled() and
		(residual.requiresGrad() or inWeight.requiresGrad()))
	{
		auto outGrad = oa::makeShared<oa::GradRmsNorm>();
		outGrad->saveForBackward(residual, inWeight, out, out);
		outGrad->setGraphInputs(oa::Vector<oa::Matrix>{residual, inWeight});
		outGrad->sequenceNr_ = oa::FnAutograd::nextSeq();
		outGrad->eps_ = inEps;
		outGrad->outputShape_ = out.getShape();
		const auto attached = oa::FnAutograd::attachSemantic(
			outGrad, semantic.getValue(), 0U);
		if (not attached.isOk()) return {};
		out.mutAutograd().gradFn = outGrad;
		out.setRequiresGrad(true);
	}

	return {.out = out, .residual = residual};
}
