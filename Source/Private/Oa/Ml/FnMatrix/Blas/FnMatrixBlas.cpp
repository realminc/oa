// OaFnMatrix — ML Linear layer operations (semantic recording into OaContext).
//
// Linear and LinearRelu are ML-specific neural network layers, moved from Core
// to Ml to separate general matrix operations from ML-specific functionality.
// MatMul remains in Core as it's a general BLAS operation.

#include <Oa/Ml/FnMatrix.h>
#include <Oa/Core/Matrix.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>
#include <Oa/Runtime/Context.h>

#include <Oa/Core/Validation.h>

#include <cassert>

#if not defined(NDEBUG) or defined(OA_ENABLE_VALIDATION)
static OaStatus OaValidateLinear(
	const OaMatrix& InX, const OaMatrix& InWeight)
{
	OA_VALIDATE(InX.Rank() >= 2, OaValidationSeverity::Error, OaLogComponent::ML,
		"Linear: input must be >=2D, got rank=%d", InX.Rank());
	OA_VALIDATE(InWeight.Rank() == 2, OaValidationSeverity::Error, OaLogComponent::ML,
		"Linear: weight must be 2D [N,K], got rank=%d", InWeight.Rank());
	OA_VALIDATE_DTYPE(InX, InWeight, "Linear");
	OA_VALIDATE_SHAPE_COMPAT(InX, InWeight, "Linear");
	return OaStatus::Ok();
}
#endif

OaMatrix OaFnMatrix::Linear(
	const OaMatrix& InX,
	const OaMatrix& InWeight,
	const OaMatrix& InBias)
{
#if not defined(NDEBUG) or defined(OA_ENABLE_VALIDATION)
	OA_ASSERT(OaValidateLinear(InX, InWeight).IsOk());
#endif
	assert(InX.Rank() >= 2 && "Linear: input must be >=2D");
	assert(InWeight.Rank() == 2 && "Linear: weight must be 2D [N, K]");
	assert(InX.Size(InX.Rank() - 1) == InWeight.Size(1)
		&& "Linear: input last dim must match weight K");
	if (not InBias.IsEmpty()) {
		assert(InBias.Rank() == 1 && "Linear: bias must be 1D [N]");
		assert(InBias.Size(0) == InWeight.Size(0)
			&& "Linear: bias size must match weight output dim");
		}
	
		auto& ctx = OaContext::GetDefault();

	OaI64 K = InX.Size(InX.Rank() - 1);
	OaI64 N = InWeight.Size(0);

	OaI64 M2d = 1;
	for (OaI32 i = 0; i < InX.Rank() - 1; ++i) M2d *= InX.Size(i);

	OaMatrix out = OaFnMatrix::Empty(OaMatrixShape{M2d, N}, InX.Dtype_);
	OaMatrix reshapedX;
	const OaMatrix* src = &InX;
	if (InX.Rank() != 2) {
		reshapedX = InX.Reshape(OaMatrixShape{M2d, K});
		src = &reshapedX;
	}

	ctx.AddLinear(
		*src,
		InWeight,
		InBias.IsEmpty() ? nullptr : &InBias,
		out,
		static_cast<OaU32>(M2d),
		static_cast<OaU32>(N),
		static_cast<OaU32>(K));

	if (InX.Rank() > 2) {
		OaMatrixShape outShape;
		outShape.Rank = InX.Shape_.Rank;
		for (OaI32 i = 0; i < InX.Rank() - 1; ++i) outShape.Dims[i] = InX.Shape_.Dims[i];
		outShape.Dims[InX.Rank() - 1] = N;
		out = out.Reshape(outShape);
	}

	return out;
}

OaMatrix OaFnMatrix::LinearRelu(
	const OaMatrix& InX,
	const OaMatrix& InWeight,
	const OaMatrix& InBias)
{
#if not defined(NDEBUG) or defined(OA_ENABLE_VALIDATION)
	OA_ASSERT(OaValidateLinear(InX, InWeight).IsOk());
#endif
	assert(InX.Rank() >= 2 && "LinearRelu: input must be >=2D");
	assert(InWeight.Rank() == 2 && "LinearRelu: weight must be 2D [N, K]");
	assert(InX.Size(InX.Rank() - 1) == InWeight.Size(1)
		&& "LinearRelu: input last dim must match weight K");
	if (InBias.IsEmpty()) {
		return OaFnMatrix::Relu(OaFnMatrix::Linear(InX, InWeight));
	}
	assert(InBias.Rank() == 1 && "LinearRelu: bias must be 1D [N]");
	assert(InBias.Size(0) == InWeight.Size(0)
		&& "LinearRelu: bias size must match weight output dim");

	auto& ctx = OaContext::GetDefault();

	OaI64 K = InX.Size(InX.Rank() - 1);
	OaI64 N = InWeight.Size(0);

	OaI64 M2d = 1;
	for (OaI32 i = 0; i < InX.Rank() - 1; ++i) M2d *= InX.Size(i);

	OaMatrix out = OaFnMatrix::Empty(OaMatrixShape{M2d, N}, InX.Dtype_);
	OaMatrix reshapedX;
	const OaMatrix* src = &InX;
	if (InX.Rank() != 2) {
		reshapedX = InX.Reshape(OaMatrixShape{M2d, K});
		src = &reshapedX;
	}

	ctx.AddLinearRelu(
		*src,
		InWeight,
		InBias,
		out,
		static_cast<OaU32>(M2d),
		static_cast<OaU32>(N),
		static_cast<OaU32>(K));

	if (InX.Rank() > 2) {
		OaMatrixShape outShape;
		outShape.Rank = InX.Shape_.Rank;
		for (OaI32 i = 0; i < InX.Rank() - 1; ++i) outShape.Dims[i] = InX.Shape_.Dims[i];
		outShape.Dims[InX.Rank() - 1] = N;
		out = out.Reshape(outShape);
	}

	return out;
}

OaMatrix OaFnMatrix::LinearGelu(
	const OaMatrix& InX,
	const OaMatrix& InWeight,
	const OaMatrix& InBias)
{
#if not defined(NDEBUG) or defined(OA_ENABLE_VALIDATION)
	OA_ASSERT(OaValidateLinear(InX, InWeight).IsOk());
#endif
	assert(InX.Rank() >= 2 && "LinearGelu: input must be >=2D");
	assert(InWeight.Rank() == 2 && "LinearGelu: weight must be 2D [N, K]");
	assert(InX.Size(InX.Rank() - 1) == InWeight.Size(1)
		&& "LinearGelu: input last dim must match weight K");
	if (InBias.IsEmpty()) {
		return OaFnMatrix::Gelu(OaFnMatrix::Linear(InX, InWeight));
	}
	assert(InBias.Rank() == 1 && "LinearGelu: bias must be 1D [N]");
	assert(InBias.Size(0) == InWeight.Size(0)
		&& "LinearGelu: bias size must match weight output dim");

	auto& ctx = OaContext::GetDefault();

	OaI64 K = InX.Size(InX.Rank() - 1);
	OaI64 N = InWeight.Size(0);

	OaI64 M2d = 1;
	for (OaI32 i = 0; i < InX.Rank() - 1; ++i) M2d *= InX.Size(i);

	OaMatrix out = OaFnMatrix::Empty(OaMatrixShape{M2d, N}, InX.Dtype_);
	OaMatrix reshapedX;
	const OaMatrix* src = &InX;
	if (InX.Rank() != 2) {
		reshapedX = InX.Reshape(OaMatrixShape{M2d, K});
		src = &reshapedX;
	}

	ctx.AddLinearGelu(
		*src,
		InWeight,
		InBias,
		out,
		static_cast<OaU32>(M2d),
		static_cast<OaU32>(N),
		static_cast<OaU32>(K));

	if (InX.Rank() > 2) {
		OaMatrixShape outShape;
		outShape.Rank = InX.Shape_.Rank;
		for (OaI32 i = 0; i < InX.Rank() - 1; ++i) outShape.Dims[i] = InX.Shape_.Dims[i];
		outShape.Dims[InX.Rank() - 1] = N;
		out = out.Reshape(outShape);
	}

	return out;
}
