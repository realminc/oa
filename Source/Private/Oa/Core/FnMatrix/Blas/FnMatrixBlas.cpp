// OaFnMatrix — BLAS operations: MatMulNt (semantic recording into OaContext).
//
// MatMulNt is a general-purpose BLAS operation that stays in Core.
// Linear and LinearRelu are ML-specific neural network layers in Ml/.

#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/Matrix.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>
#include <Oa/Ml/Autograd.h>
#include <Oa/Runtime/Context.h>

#include <Oa/Core/Validation.h>

#include <cassert>

#if not defined(NDEBUG) or defined(OA_ENABLE_VALIDATION)
static OaStatus OaValidateMatMulNt(
	const OaMatrix& InA, const OaMatrix& InB)
{
	OA_VALIDATE(InA.Rank() >= 2, OaValidationSeverity::Error, OaLogComponent::ML,
		"MatMulNt: A must be >=2D, got rank=%d", InA.Rank());
	OA_VALIDATE(InB.Rank() == 2, OaValidationSeverity::Error, OaLogComponent::ML,
		"MatMulNt: B must be 2D [N,K], got rank=%d", InB.Rank());
	OA_VALIDATE_DTYPE(InA, InB, "MatMulNt");
	OA_VALIDATE_SHAPE_COMPAT(InA, InB, "MatMulNt");
	return OaStatus::Ok();
}
#endif

// MatMulNt
OaMatrix OaFnMatrix::MatMulNt(
	const OaMatrix& InA,
	const OaMatrix& InB,
	OaContextMatMulPrecision InPrecision)
{
#if not defined(NDEBUG) or defined(OA_ENABLE_VALIDATION)
	OA_ASSERT(OaValidateMatMulNt(InA, InB).IsOk());
#endif
	auto& ctx = OaContext::GetDefault();

	OaI64 K = InA.Size(InA.Rank() - 1);
	OaI64 N = InB.Size(0);

	OaI64 M2d = 1;
	for (OaI32 i = 0; i < InA.Rank() - 1; ++i) M2d *= InA.Size(i);

	OA_LOG_DEBUG(OaLogComponent::ML, "MatMulNt: A.rank=%d K=%lld B=[%lld,%lld] M2d=%lld N=%lld",
		InA.Rank(), K, InB.Size(0), InB.Size(1), M2d, N);

	OaMatrix out = OaFnMatrix::Empty(OaMatrixShape{M2d, N}, InA.Dtype_);

	OaMatrix reshapedA;
	const OaMatrix* src = &InA;
	if (InA.Rank() != 2) {
		reshapedA = InA.Reshape(OaMatrixShape{M2d, K});
		src = &reshapedA;
	}
	ctx.AddMatMul(
		*src,
		InB,
		out,
		static_cast<OaU32>(M2d),
		static_cast<OaU32>(N),
		static_cast<OaU32>(K),
		InPrecision);

	if (InA.Rank() > 2) {
		OaMatrixShape outShape;
		outShape.Rank = InA.Shape_.Rank;
		for (OaI32 i = 0; i < InA.Rank() - 1; ++i) outShape.Dims[i] = InA.Shape_.Dims[i];
		outShape.Dims[InA.Rank() - 1] = N;
		out = out.Reshape(outShape);
	}

	// Autograd: attach a MatMulNt node for the 2D activation×activation case
	// (enables attention Q@Kᵀ / attn@V to backprop). Higher-rank MatMulNt reshapes
	// A internally, so we only tape the plain 2D form.
	if (OaFnAutograd::IsEnabled() and InA.Rank() == 2 and InB.Rank() == 2
			and (InA.RequiresGrad() or InB.RequiresGrad())) {
		auto _gradFn = OaMakeSharedPtr<OaGradMatMulNt>();
		_gradFn->Saved_ = OaVec<OaMatrix>{InA, InB};
		_gradFn->SetGraphInputs(OaVec<OaMatrix>{InA, InB});
		_gradFn->SequenceNr_ = OaFnAutograd::NextSeq();
		_gradFn->OutputShape_ = out.GetShape();  // for tape dout normalization (handles Mamba post-siso + LM reshapes to the out_proj/in_proj MatMuls)
		out.MutAutograd().GradFn = _gradFn;
	}

	return out;
}
