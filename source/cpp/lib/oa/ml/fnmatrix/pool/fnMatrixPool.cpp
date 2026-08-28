// oa::FnMatrix — Spatial pooling operations.
//
// AvgPool2d, MaxPool2d.

#include <oa/ml/fnMatrix.h>
#include <oa/ml/autograd/matrix/autogradPool.h>
#include <oa/core/matrix.h>
#include <oa/core/types.h>
#include <oa/core/bufferAccess.h>
#include <oa/core/op.h>
#include <oa/runtime/executionSession.h>
#include <oa/core/validation.h>

#include <assert.h>
static oa::U32 divCeil(oa::U32 inA, oa::U32 inB) { return (inA + inB - 1) / inB; }

// Pooling
oa::Matrix oa::FnMatrix::avgPool2d(const oa::Matrix& inX, oa::I32 inKernelSize, oa::I32 inStride, oa::I32 inPadding) {
	auto& ctx = oa::ExecutionSession::getActive();
	assert(inX.rank() == 4 && "AvgPool2d requires 4D input [N, C, H, W]");

	oa::I64 N = inX.size(0);
	oa::I64 C = inX.size(1);
	oa::I64 H = inX.size(2);
	oa::I64 W = inX.size(3);

	// Calculate output dimensions
	oa::I64 H_out = (H + 2 * inPadding - inKernelSize) / inStride + 1;
	oa::I64 W_out = (W + 2 * inPadding - inKernelSize) / inStride + 1;

	oa::Matrix out = oa::FnMatrix::empty(oa::MatrixShape{N, C, H_out, W_out}, inX.getDtype());
	const auto semantic = ctx.recordOp(
		oa::detail::opRegistry::FnMatrix::avgPool2d, {&inX}, {&out},
		{
			oa::OpAttribute::fromSignedInteger(
				"kernelSize", inKernelSize),
			oa::OpAttribute::fromSignedInteger("stride", inStride),
			oa::OpAttribute::fromSignedInteger("padding", inPadding),
		});
	if (not semantic.isOk()) return {};

	// Dispatch AvgPool2d kernel
	struct {
		oa::U32 batchSize;
		oa::U32 channels;
		oa::U32 inHeight;
		oa::U32 inWidth;
		oa::U32 outHeight;
		oa::U32 outWidth;
		oa::U32 KernelSize;
		oa::U32 Stride;
		oa::U32 Padding;
	} push{
		static_cast<oa::U32>(N), static_cast<oa::U32>(C),
		static_cast<oa::U32>(H), static_cast<oa::U32>(W),
		static_cast<oa::U32>(H_out), static_cast<oa::U32>(W_out),
		static_cast<oa::U32>(inKernelSize), static_cast<oa::U32>(inStride), static_cast<oa::U32>(inPadding)
	};

	oa::U32 grid_x = divCeil(static_cast<oa::U32>(H_out), 16);
	oa::U32 grid_y = divCeil(static_cast<oa::U32>(W_out), 16);
	oa::U32 grid_z = static_cast<oa::U32>(N * C);
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "AvgPool2d", {&inX, &out}, access, &push, sizeof(push),
		grid_x, grid_y, grid_z,
		oa::detail::opRegistry::FnMatrix::avgPool2d.name, 0,
		oa::detail::opRegistry::FnMatrix::avgPool2d.hash, 0, 0,
		semantic.getValue());

	if (oa::FnAutograd::isEnabled() and (inX.requiresGrad())) {
		auto gradFn = oa::makeShared<oa::GradAvgPool2d>();
		gradFn->saveForBackward(inX);
		gradFn->kernelSize_ = inKernelSize;
		gradFn->stride_ = inStride;
		gradFn->padding_ = inPadding;
		gradFn->setGraphInputs(oa::Vector<oa::Matrix>{inX});
		gradFn->sequenceNr_ = oa::FnAutograd::nextSeq();
		gradFn->outputShape_ = out.getShape();
		const auto semanticAttached = oa::FnAutograd::attachSemantic(
			gradFn, semantic.getValue());
		if (not semanticAttached.isOk()) return {};
		out.mutAutograd().gradFn = gradFn;
	}

	return out;
}

oa::MaxPool2dResult oa::FnMatrix::maxPool2d(
	const oa::Matrix& inX, oa::I32 inKernelSize, oa::I32 inStride, oa::I32 inPadding
) {
	auto& ctx = oa::ExecutionSession::getActive();
	assert(inX.rank() == 4 && "MaxPool2d requires 4D input [N, C, H, W]");

	oa::I64 N = inX.size(0);
	oa::I64 C = inX.size(1);
	oa::I64 H = inX.size(2);
	oa::I64 W = inX.size(3);

	// Calculate output dimensions
	oa::I64 H_out = (H + 2 * inPadding - inKernelSize) / inStride + 1;
	oa::I64 W_out = (W + 2 * inPadding - inKernelSize) / inStride + 1;

	oa::Matrix out = oa::FnMatrix::empty(oa::MatrixShape{N, C, H_out, W_out}, inX.getDtype());
	oa::Matrix indices = oa::FnMatrix::empty(oa::MatrixShape{N, C, H_out, W_out}, oa::ScalarType::UInt32);
	const auto semantic = ctx.recordOp(
		oa::detail::opRegistry::FnMatrix::maxPool2d, {&inX}, {&out, &indices},
		{
			oa::OpAttribute::fromSignedInteger(
				"kernelSize", inKernelSize),
			oa::OpAttribute::fromSignedInteger("stride", inStride),
			oa::OpAttribute::fromSignedInteger("padding", inPadding),
		});
	if (not semantic.isOk()) return {};

	// Dispatch MaxPool2d kernel
	struct {
		oa::U32 batchSize;
		oa::U32 channels;
		oa::U32 inHeight;
		oa::U32 inWidth;
		oa::U32 outHeight;
		oa::U32 outWidth;
		oa::U32 KernelSize;
		oa::U32 Stride;
		oa::U32 Padding;
	} push{
		static_cast<oa::U32>(N), static_cast<oa::U32>(C),
		static_cast<oa::U32>(H), static_cast<oa::U32>(W),
		static_cast<oa::U32>(H_out), static_cast<oa::U32>(W_out),
		static_cast<oa::U32>(inKernelSize), static_cast<oa::U32>(inStride), static_cast<oa::U32>(inPadding)
	};

	oa::U32 grid_x = divCeil(static_cast<oa::U32>(H_out), 16);
	oa::U32 grid_y = divCeil(static_cast<oa::U32>(W_out), 16);
	oa::U32 grid_z = static_cast<oa::U32>(N * C);
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write, oa::BufferAccess::Write};
	ctx.add( "MaxPool2d", {&inX, &out, &indices}, access, &push, sizeof(push),
		grid_x, grid_y, grid_z,
		oa::detail::opRegistry::FnMatrix::maxPool2d.name, 0,
		oa::detail::opRegistry::FnMatrix::maxPool2d.hash, 0, 0,
		semantic.getValue());

	if (oa::FnAutograd::isEnabled() and (inX.requiresGrad())) {
		auto gradFn = oa::makeShared<oa::GradMaxPool2d>();
		gradFn->saveForBackward(inX, out, indices);
		gradFn->kernelSize_ = inKernelSize;
		gradFn->stride_ = inStride;
		gradFn->padding_ = inPadding;
		gradFn->setGraphInputs(oa::Vector<oa::Matrix>{inX});
		gradFn->sequenceNr_ = oa::FnAutograd::nextSeq();
		gradFn->outputShape_ = out.getShape();
		const auto semanticAttached = oa::FnAutograd::attachSemantic(
			gradFn, semantic.getValue());
		if (not semanticAttached.isOk()) return {};
		out.mutAutograd().gradFn = gradFn;
	}

	return {out, indices};
}
