// OaFnMatrix — Spatial pooling operations.
//
// AvgPool2d, MaxPool2d.

#include <Oa/Ml/FnMatrix.h>
#include <Oa/Ml/Autograd/Nodes.h>
#include <Oa/Core/Matrix.h>
#include <Oa/Core/Types.h>
#include <Oa/Core/BufferAccess.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Core/Validation.h>

#include <cassert>
static OaU32 DivCeil(OaU32 InA, OaU32 InB) { return (InA + InB - 1) / InB; }

// Pooling
OaMatrix OaFnMatrix::AvgPool2d(const OaMatrix& InX, OaI32 InKernelSize, OaI32 InStride, OaI32 InPadding) {
	auto& ctx = OaContext::GetDefault();
	assert(InX.Rank() == 4 && "AvgPool2d requires 4D input [N, C, H, W]");

	OaI64 N = InX.Size(0);
	OaI64 C = InX.Size(1);
	OaI64 H = InX.Size(2);
	OaI64 W = InX.Size(3);

	// Calculate output dimensions
	OaI64 H_out = (H + 2 * InPadding - InKernelSize) / InStride + 1;
	OaI64 W_out = (W + 2 * InPadding - InKernelSize) / InStride + 1;

	OaMatrix out = OaFnMatrix::Empty(OaMatrixShape{N, C, H_out, W_out}, InX.GetDtype());

	// Dispatch AvgPool2d kernel
	struct {
		OaU32 BatchSize;
		OaU32 Channels;
		OaU32 InHeight;
		OaU32 InWidth;
		OaU32 OutHeight;
		OaU32 OutWidth;
		OaU32 KernelSize;
		OaU32 Stride;
		OaU32 Padding;
	} push{
		static_cast<OaU32>(N), static_cast<OaU32>(C),
		static_cast<OaU32>(H), static_cast<OaU32>(W),
		static_cast<OaU32>(H_out), static_cast<OaU32>(W_out),
		static_cast<OaU32>(InKernelSize), static_cast<OaU32>(InStride), static_cast<OaU32>(InPadding)
	};

	OaU32 grid_x = DivCeil(static_cast<OaU32>(H_out), 16);
	OaU32 grid_y = DivCeil(static_cast<OaU32>(W_out), 16);
	OaU32 grid_z = static_cast<OaU32>(N * C);
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("AvgPool2d", {&InX, &out}, access, &push, sizeof(push), grid_x, grid_y, grid_z);

	if (OaFnAutograd::IsEnabled() and (InX.RequiresGrad())) {
		auto gradFn = OaMakeSharedPtr<OaGradAvgPool2d>();
		gradFn->Saved_ = OaVec<OaMatrix>{InX};
		gradFn->KernelSize_ = InKernelSize;
		gradFn->Stride_ = InStride;
		gradFn->Padding_ = InPadding;
		gradFn->SetGraphInputs(OaVec<OaMatrix>{InX});
		gradFn->SequenceNr_ = OaFnAutograd::NextSeq();
		gradFn->OutputShape_ = out.GetShape();
		out.MutAutograd().GradFn = gradFn;
	}

	return out;
}

OaFnMatrix::OaMaxPool2dResult OaFnMatrix::MaxPool2d(
	const OaMatrix& InX, OaI32 InKernelSize, OaI32 InStride, OaI32 InPadding
) {
	auto& ctx = OaContext::GetDefault();
	assert(InX.Rank() == 4 && "MaxPool2d requires 4D input [N, C, H, W]");

	OaI64 N = InX.Size(0);
	OaI64 C = InX.Size(1);
	OaI64 H = InX.Size(2);
	OaI64 W = InX.Size(3);

	// Calculate output dimensions
	OaI64 H_out = (H + 2 * InPadding - InKernelSize) / InStride + 1;
	OaI64 W_out = (W + 2 * InPadding - InKernelSize) / InStride + 1;

	OaMatrix out = OaFnMatrix::Empty(OaMatrixShape{N, C, H_out, W_out}, InX.GetDtype());
	OaMatrix indices = OaFnMatrix::Empty(OaMatrixShape{N, C, H_out, W_out}, OaScalarType::UInt32);

	// Dispatch MaxPool2d kernel
	struct {
		OaU32 BatchSize;
		OaU32 Channels;
		OaU32 InHeight;
		OaU32 InWidth;
		OaU32 OutHeight;
		OaU32 OutWidth;
		OaU32 KernelSize;
		OaU32 Stride;
		OaU32 Padding;
	} push{
		static_cast<OaU32>(N), static_cast<OaU32>(C),
		static_cast<OaU32>(H), static_cast<OaU32>(W),
		static_cast<OaU32>(H_out), static_cast<OaU32>(W_out),
		static_cast<OaU32>(InKernelSize), static_cast<OaU32>(InStride), static_cast<OaU32>(InPadding)
	};

	OaU32 grid_x = DivCeil(static_cast<OaU32>(H_out), 16);
	OaU32 grid_y = DivCeil(static_cast<OaU32>(W_out), 16);
	OaU32 grid_z = static_cast<OaU32>(N * C);
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write, OaBufferAccess::Write};
	ctx.Add("MaxPool2d", {&InX, &out, &indices}, access, &push, sizeof(push), grid_x, grid_y, grid_z);

	if (OaFnAutograd::IsEnabled() and (InX.RequiresGrad())) {
		auto gradFn = OaMakeSharedPtr<OaGradMaxPool2d>();
		gradFn->Saved_ = OaVec<OaMatrix>{InX, out, indices};
		gradFn->KernelSize_ = InKernelSize;
		gradFn->Stride_ = InStride;
		gradFn->Padding_ = InPadding;
		gradFn->SetGraphInputs(OaVec<OaMatrix>{InX});
		gradFn->SequenceNr_ = OaFnAutograd::NextSeq();
		gradFn->OutputShape_ = out.GetShape();
		out.MutAutograd().GradFn = gradFn;
	}

	return {out, indices};
}
