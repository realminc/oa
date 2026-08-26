// oa::FnMatrix — reductions and softmax.
//
// Sum, Mean, Max, Argmax, Softmax, LogSoftmax.

#include <oa/core/matrix.h>
#include <oa/core/log.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/fnmatrix/fnMatrixInternal.h>
#include <oa/core/status.h>
#include <oa/core/types.h>
#include <oa/core/bufferAccess.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/dispatchDesc.h>
#include <oa/core/validation.h>
#include <oa/core/op.h>
#include <oa/core/std/limits.h>
#include "../../autograd/autogradAttach.gen.h"
#include "../fnMatrixAxis.h"
#include "fnMatrixReduceLowering.h"

#include <assert.h>

static oa::Matrix commitCoreReductionResult(
	oa::Matrix inResult,
	oa::OpLoweringScope& inLowering,
	const oa::OpContract& inContract,
	oa::MatrixArgs inInputs,
	oa::OpAttributeArgs inAttributes = {}
) {
	const auto status = inLowering.commit(inContract, inInputs, {&inResult}, inAttributes);
	return status.isOk() ? inResult : oa::Matrix{};
}

oa::Matrix oa::FnMatrix::softmaxBwd(const oa::Matrix& inForwardOutput, const oa::Matrix& inGradOutput,oa::I32 inDim) {
	if (inForwardOutput.getShape() != inGradOutput.getShape()	or inForwardOutput.getDtype() != inGradOutput.getDtype())	{
		return {};
	}
	FnMatrixAxisShape axis;
	if (not resolveFnMatrixAxis(inForwardOutput, inDim, axis)) {
		return {};
	}
	auto& context = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(context);
	oa::Matrix gradInput = oa::FnMatrix::empty(inForwardOutput.getShape(), inForwardOutput.getDtype());
	struct {
		oa::U32 outerSize;
		oa::U32 dimSize;
		oa::U32 innerSize;
	} push {axis.outerSize, axis.dimSize, axis.innerSize};
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read,
		oa::BufferAccess::Read,
		oa::BufferAccess::Write,
	};
	context.add(
		"SoftmaxBwd",
		{&inForwardOutput, &inGradOutput, &gradInput},
		access,
		&push,
		sizeof(push),
		axis.groupCount()
	);
	return commitCoreReductionResult(
		gradInput,
		lowering,
		oa::detail::opRegistry::FnMatrix::softmaxBwd,
		{&inForwardOutput, &inGradOutput},
		{oa::OpAttribute::fromSignedInteger("dim", inDim)}
	);
}

oa::Matrix oa::FnMatrix::logSoftmaxBwd(
	const oa::Matrix& inForwardOutput,
	const oa::Matrix& inGradOutput,
	oa::I32 inDim)
{
	if (inForwardOutput.getShape() != inGradOutput.getShape()
		or inForwardOutput.getDtype() != inGradOutput.getDtype())
	{
		return {};
	}
	FnMatrixAxisShape axis;
	if (not resolveFnMatrixAxis(inForwardOutput, inDim, axis)) {
		return {};
	}
	auto& context = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(context);
	oa::Matrix gradInput = oa::FnMatrix::empty(
		inForwardOutput.getShape(), inForwardOutput.getDtype());
	struct {
		oa::U32 outerSize;
		oa::U32 dimSize;
		oa::U32 innerSize;
	} push{axis.outerSize, axis.dimSize, axis.innerSize};
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read,
		oa::BufferAccess::Read,
		oa::BufferAccess::Write,
	};
	context.add(
		"LogSoftmaxBwd",
		{&inForwardOutput, &inGradOutput, &gradInput},
		access,
		&push,
		sizeof(push),
		axis.groupCount());
	return commitCoreReductionResult(
		gradInput,
		lowering,
		oa::detail::opRegistry::FnMatrix::logSoftmaxBwd,
		{&inForwardOutput, &inGradOutput},
		{oa::OpAttribute::fromSignedInteger("dim", inDim)});
}

oa::Matrix oa::FnMatrix::maxBwd(
	const oa::Matrix& inInput,
	const oa::Matrix& inMaxValue,
	const oa::Matrix& inGradOutput
) {
	auto& context = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(context);
	const oa::U32 count = static_cast<oa::U32>(inInput.numElements());
	oa::Matrix gradInput = oa::FnMatrix::empty(
		inInput.getShape(), inInput.getDtype()
	);
	struct { oa::U32 Count; } push{count};
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read,
		oa::BufferAccess::Read,
		oa::BufferAccess::Read,
		oa::BufferAccess::Write,
	};
	context.add(
		"MaxBwd",
		{&inInput, &inMaxValue, &inGradOutput, &gradInput},
		access,
		&push,
		sizeof(push),
		divCeil(count, 256)
	);

	return commitCoreReductionResult(
		gradInput,
		lowering,
		oa::detail::opRegistry::FnMatrix::maxBwd,
		{&inInput, &inMaxValue, &inGradOutput}
	);
}

oa::Status oa::FnMatrixPrivate::lowerFullMean(
	oa::ExecutionSession& inContext,
	const oa::Matrix& inInput,
	oa::Matrix& outMean,
	const oa::OpContract& inContract,
	oa::U32 inSemanticOp
) {
	const oa::I64 elementCount = inInput.numElements();
	if (elementCount <= 0
		or static_cast<oa::U64>(elementCount) > oa::Limits<oa::U32>::max()
		or outMean.numElements() != 1
		or outMean.getDtype() != inInput.getDtype()
		or not inInput.hasStorage() or not outMean.hasStorage()
	) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,	"full mean lowering requires a non-empty 32-bit input and one same-dtype output");
	}

	const oa::U32 count = static_cast<oa::U32>(elementCount);
	oa::Matrix sum = oa::FnMatrix::empty(oa::MatrixShape{1}, inInput.getDtype());
	if (not sum.hasStorage()) {
		return oa::Status::error(oa::StatusCode::OutOfMemory,	"full mean lowering failed to allocate reduction storage");
	}
	struct { oa::U32 Count; } reductionPush{count};
	oa::BufferAccess reductionAccess[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Write};
	const oa::Matrix* reductionMatrices[] = {&inInput, &sum};
	oa::MatrixDispatchDesc reduction;
	reduction.dispatch.operation = inContract.name;
	if (inSemanticOp != oa::invalidSemanticOpId) {
		reduction.dispatch.semanticOps =	oa::Span<const oa::U32>(&inSemanticOp, 1U);
	}
	reduction.dispatch.opContractHash = inContract.hash;
	reduction.dispatch.kernel = "Sum";
	reduction.dispatch.access = oa::Span<oa::BufferAccess>(reductionAccess, 2U);
	reduction.dispatch.pushData = &reductionPush;
	reduction.dispatch.pushSize = sizeof(reductionPush);
	reduction.matrices = oa::Span<const oa::Matrix* const>(reductionMatrices, 2U);
	OA_RETURN_IF_ERROR(inContext.record( reduction));

	struct { oa::U32 Count; oa::F32 alpha; } scalePush{
		1U, 1.0F / static_cast<oa::F32>(count)};
	oa::BufferAccess scaleAccess[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Write};
	const oa::Matrix* scaleMatrices[] = {&sum, &outMean};
	oa::MatrixDispatchDesc scale;
	scale.dispatch.operation = inContract.name;
	if (inSemanticOp != oa::invalidSemanticOpId) {
		scale.dispatch.semanticOps =
			oa::Span<const oa::U32>(&inSemanticOp, 1U);
	}
	scale.dispatch.opContractHash = inContract.hash;
	scale.dispatch.kernel = "Scale";
	scale.dispatch.access = oa::Span<oa::BufferAccess>(scaleAccess, 2U);
	scale.dispatch.pushData = &scalePush;
	scale.dispatch.pushSize = sizeof(scalePush);
	scale.matrices = oa::Span<const oa::Matrix* const>(scaleMatrices, 2U);
	return inContext.record( scale);
}

// Reductions
oa::Matrix oa::FnMatrix::sum(const oa::Matrix& inA, oa::I32 inDim) {
	oa::I64 n = inA.numElements();
	auto& ctx = oa::ExecutionSession::getActive();

	oa::I32 resolvedDim = inDim;
	if (resolvedDim >= 0 and resolvedDim < inA.rank()) {
		oa::I64 outerSize = 1;
		for (oa::I32 i = 0; i < resolvedDim; ++i) outerSize *= inA.size(i);
		oa::I64 dimSize = inA.size(resolvedDim);
		oa::I64 innerSize = 1;
		for (oa::I32 i = resolvedDim + 1; i < inA.rank(); ++i) innerSize *= inA.size(i);

		oa::MatrixShape outShape;
		outShape.rank = inA.rank();
		for (oa::I32 i = 0; i < inA.rank(); ++i) {
			outShape.dims[static_cast<oa::Usize>(i)] =
				(i == resolvedDim) ? 1 : inA.size(i);
		}

		oa::Matrix out = oa::FnMatrix::zeros(outShape, inA.getDtype());
		const auto semantic = ctx.recordOp(
			oa::detail::opRegistry::FnMatrix::sum, {&inA}, {&out},
			{oa::OpAttribute::fromSignedInteger("dim", inDim)}
		);
		if (not semantic.isOk()) return {};
		oa::U32 totalOut = static_cast<oa::U32>(outerSize * innerSize);
		struct { oa::U32 outerSize; oa::U32 dimSize; oa::U32 innerSize; } push{
			static_cast<oa::U32>(outerSize), static_cast<oa::U32>(dimSize), static_cast<oa::U32>(innerSize)
		};
		oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
		ctx.add( "SumDim", {&inA, &out}, access, &push, sizeof(push),
			divCeil(totalOut, 256), 1, 1,
			oa::detail::opRegistry::FnMatrix::sum.name, 0, oa::detail::opRegistry::FnMatrix::sum.hash,
			0, 0, semantic.getValue()
		);

		const auto attached = oa::detail::generatedAutogradAttach::FnMatrix::sum(
			out, inA, semantic.getValue());
		if (not attached.isOk()) {
			OaLogError(oa::LogComponent::Compute,	"Sum semantic autograd attachment failed: %s",	attached.getMessage().cStr());
			return {};
		}
		return out;
	}

	oa::Matrix out = oa::FnMatrix::zeros(oa::MatrixShape{1}, inA.getDtype());
	const auto semantic = ctx.recordOp(
		oa::detail::opRegistry::FnMatrix::sum, {&inA}, {&out},
		{oa::OpAttribute::fromSignedInteger("dim", inDim)}
	);
	if (not semantic.isOk()) return {};
	struct { oa::U32 Count; } push{static_cast<oa::U32>(n)};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "Sum", {&inA, &out}, access, &push, sizeof(push), 1, 1, 1,
		oa::detail::opRegistry::FnMatrix::sum.name, 0, oa::detail::opRegistry::FnMatrix::sum.hash,
		0, 0, semantic.getValue());

	const auto attached = oa::detail::generatedAutogradAttach::FnMatrix::sum(out, inA, semantic.getValue());
	if (not attached.isOk()) {
		OaLogError(oa::LogComponent::Compute,
			"Sum semantic autograd attachment failed: %s",
			attached.getMessage().cStr());
		return {};
	}
	return out;
}

oa::Matrix oa::FnMatrix::mean(const oa::Matrix& inA, oa::I32 inDim) {
	const oa::I64 elementCount = inA.numElements();
	if (inA.rank() <= 0 or elementCount <= 0 or inDim < -1
		or inDim >= inA.rank()
		or static_cast<oa::U64>(elementCount) > oa::Limits<oa::U32>::max()
	)	{
		OaLogError(oa::LogComponent::Compute,
			"Mean: expected a non-empty matrix and dim=-1 or a valid axis");
		return {};
	}

	FnMatrixAxisShape axis;
	const bool reduceAxis = inDim >= 0;
	if (reduceAxis and not resolveFnMatrixAxis(inA, inDim, axis)) {
		OaLogError(oa::LogComponent::Compute,	"Mean: selected axis exceeds the 32-bit reduction address space");
		return {};
	}

	oa::MatrixShape outputShape{1};
	oa::U32 outputCount = 1U;
	oa::U32 reductionCount = static_cast<oa::U32>(elementCount);
	if (reduceAxis) {
		outputShape.rank = inA.rank();
		for (oa::I32 dim = 0; dim < inA.rank(); ++dim) {
			outputShape.dims[static_cast<oa::Usize>(dim)] =
				dim == inDim ? 1 : inA.size(dim);
		}
		outputCount = axis.groupCount();
		reductionCount = axis.dimSize;
	}

	auto& ctx = oa::ExecutionSession::getActive();
	oa::Matrix out = oa::FnMatrix::empty(outputShape, inA.getDtype());
	const auto semantic = ctx.recordOp(
		oa::detail::opRegistry::FnMatrix::mean, {&inA}, {&out},
		{oa::OpAttribute::fromSignedInteger("dim", inDim)}
	);
	if (not semantic.isOk()) return {};
	if (not reduceAxis) {
		const auto lowering = oa::FnMatrixPrivate::lowerFullMean(
			ctx, inA, out, oa::detail::opRegistry::FnMatrix::mean, semantic.getValue());
		if (not lowering.isOk()) {
			OaLogError(oa::LogComponent::Compute,
				"Mean lowering failed: %s", lowering.getMessage().cStr()
			);
			return {};
		}

		const auto attached = oa::detail::generatedAutogradAttach::FnMatrix::mean(out, inA, inDim, semantic.getValue());
		if (not attached.isOk()) {
			OaLogError(oa::LogComponent::Compute,	"Mean semantic autograd attachment failed: %s", attached.getMessage().cStr());
			return {};
		}
		return out;
	}

	oa::Matrix sum = oa::FnMatrix::empty(outputShape, inA.getDtype());
	oa::BufferAccess reductionAccess[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Write};
	struct { oa::U32 outerSize; oa::U32 dimSize; oa::U32 innerSize; } push{
		axis.outerSize, axis.dimSize, axis.innerSize};
	ctx.add( "SumDim", {&inA, &sum}, reductionAccess, &push, sizeof(push),
		divCeil(outputCount, 256U), 1, 1,
		oa::detail::opRegistry::FnMatrix::mean.name, 0, oa::detail::opRegistry::FnMatrix::mean.hash,
		0, 0, semantic.getValue());

	struct { oa::U32 Count; oa::F32 alpha; } scalePush{
		outputCount, 1.0F / static_cast<oa::F32>(reductionCount)};
	oa::BufferAccess scaleAccess[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "Scale", {&sum, &out}, scaleAccess, &scalePush, sizeof(scalePush),
		divCeil(outputCount, 256U), 1, 1,
		oa::detail::opRegistry::FnMatrix::mean.name, 0, oa::detail::opRegistry::FnMatrix::mean.hash,
		0, 0, semantic.getValue());

	const auto attached = oa::detail::generatedAutogradAttach::FnMatrix::mean(
		out, inA, inDim, semantic.getValue());
	if (not attached.isOk()) {
		OaLogError(oa::LogComponent::Compute,
			"Mean semantic autograd attachment failed: %s",
			attached.getMessage().cStr());
		return {};
	}
	return out;
}

oa::Matrix oa::FnMatrix::max(const oa::Matrix& inA, oa::I32 inDim) {
	(void)inDim;
	auto& ctx = oa::ExecutionSession::getActive();

	// Compute max value (full reduction → scalar)
	oa::Matrix out = oa::FnMatrix::empty(oa::MatrixShape{1}, inA.getDtype());
	const auto semantic = ctx.recordOp(
		oa::detail::opRegistry::FnMatrix::max, {&inA}, {&out});
	if (not semantic.isOk()) return {};
	struct { oa::U32 Count; } push{static_cast<oa::U32>(inA.numElements())};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "Max", {&inA, &out}, access, &push, sizeof(push), 1, 1, 1,
		oa::detail::opRegistry::FnMatrix::max.name, 0, oa::detail::opRegistry::FnMatrix::max.hash,
		0, 0, semantic.getValue()
	);

	const auto attached = oa::detail::generatedAutogradAttach::FnMatrix::max(
		out, inA, semantic.getValue());
	if (not attached.isOk()) {
		OaLogError(oa::LogComponent::Compute,
			"Max semantic autograd attachment failed: %s",
			attached.getMessage().cStr());
		return {};
	}

	return out;
}

oa::I64 oa::FnMatrix::argmax(const oa::Matrix& inA, oa::I32 inDim) {
	(void)inDim;
	auto& ctx = oa::ExecutionSession::getActive();
	oa::Matrix out = oa::FnMatrix::empty(oa::MatrixShape{1}, oa::ScalarType::UInt32);
	struct { oa::U32 Count; } push{static_cast<oa::U32>(inA.numElements())};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "Argmax", {&inA, &out}, access, &push, sizeof(push), 1);
	auto status = oa::FnMatrix::completeRecordedWork(ctx);
	assert(status.isOk() and "context submission failed before Argmax readback");
	oa::U32 index = 0;
	auto copyStatus = oa::FnMatrix::copyToHost(out, &index, sizeof(index));
	assert(copyStatus.isOk() && "Argmax readback failed");
	return static_cast<oa::I64>(index);
}

oa::Matrix oa::FnMatrix::categoricalAccuracyCount(
	const oa::Matrix& inLogits, const oa::Matrix& inLabels) {
	if (inLogits.rank() < 2 or inLogits.numElements() == 0 or
		inLabels.numElements() != inLogits.numElements() / inLogits.size(inLogits.rank() - 1) or
		(inLabels.getDtype() != oa::ScalarType::UInt8 and
		 inLabels.getDtype() != oa::ScalarType::UInt32 and
		 inLabels.getDtype() != oa::ScalarType::Int32)
	) {
		OaLogError(oa::LogComponent::Compute,	"CategoricalAccuracyCount: expected logits [...,C] and UInt8/UInt32/Int32 labels [...]");
		return {};
	}
	const oa::U32 classes = static_cast<oa::U32>(inLogits.size(inLogits.rank() - 1));
	const oa::U32 rows = static_cast<oa::U32>(inLabels.numElements());
	const oa::U32 labelType = inLabels.getDtype() == oa::ScalarType::UInt8 ? 0u
		: (inLabels.getDtype() == oa::ScalarType::UInt32 ? 1u : 2u);
	oa::Matrix out = oa::FnMatrix::empty(oa::MatrixShape{1}, oa::ScalarType::UInt32);
	struct { oa::U32 rows, classes, LabelType; } push{rows, classes, labelType};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write};
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	ctx.add(
		"CategoricalAccuracyCount", {&inLogits, &inLabels, &out},
		access, &push, sizeof(push), 1
	);
	if (not lowering.commit(
		oa::detail::opRegistry::FnMatrix::categoricalAccuracyCount,
		{&inLogits, &inLabels}, {&out}).isOk())
	{
		return {};
	}
	return out;
}

oa::Matrix oa::FnMatrix::maskedCategoricalAccuracyCount(
	const oa::Matrix& inLogits, const oa::Matrix& inLabels, const oa::Matrix& inMask) {
	if (inLogits.rank() < 2 or inLogits.numElements() == 0 or
		inLabels.numElements() != inLogits.numElements() / inLogits.size(inLogits.rank() - 1) or
		inMask.numElements() != inLabels.numElements() or
		inMask.getDtype() != inLogits.getDtype() or
		(inLabels.getDtype() != oa::ScalarType::UInt8 and
		 inLabels.getDtype() != oa::ScalarType::UInt32 and
		 inLabels.getDtype() != oa::ScalarType::Int32) or
		(inMask.getDtype() != oa::ScalarType::Float32 and
		 inMask.getDtype() != oa::ScalarType::BFloat16)) {
		OaLogError(oa::LogComponent::Compute,
			"MaskedCategoricalAccuracyCount: expected logits [...,C], integer labels [...], and a same-dtype floating mask [...]");
		return {};
	}
	const oa::U32 classes = static_cast<oa::U32>(inLogits.size(inLogits.rank() - 1));
	const oa::U32 rows = static_cast<oa::U32>(inLabels.numElements());
	const oa::U32 labelType = inLabels.getDtype() == oa::ScalarType::UInt8 ? 0u
		: (inLabels.getDtype() == oa::ScalarType::UInt32 ? 1u : 2u);
	oa::Matrix out = oa::FnMatrix::empty(oa::MatrixShape{1}, oa::ScalarType::UInt32);
	struct { oa::U32 rows, classes, LabelType; } push{rows, classes, labelType};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Write};
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	ctx.add( "MaskedCategoricalAccuracyCount", {&inLogits, &inLabels, &inMask, &out},
		access, &push, sizeof(push), 1);
	if (not lowering.commit(
		oa::detail::opRegistry::FnMatrix::maskedCategoricalAccuracyCount,
		{&inLogits, &inLabels, &inMask}, {&out}).isOk())
	{
		return {};
	}
	return out;
}

oa::Matrix oa::FnMatrix::softmax(const oa::Matrix& inA, oa::I32 inDim) {
	FnMatrixAxisShape axis;
	if (not resolveFnMatrixAxis(inA, inDim, axis)) {
		OaLogError(oa::LogComponent::Compute,
			"Softmax: expected a non-empty matrix and dim=-1 or a valid axis");
		return {};
	}
	auto& ctx = oa::ExecutionSession::getActive();
	oa::Matrix out = oa::FnMatrix::empty(inA.getShape(), inA.getDtype());
	const auto semantic = ctx.recordOp(
		oa::detail::opRegistry::FnMatrix::softmax, {&inA}, {&out},
		{oa::OpAttribute::fromSignedInteger("dim", inDim)});
	if (not semantic.isOk()) return {};

	struct { oa::U32 outerSize; oa::U32 dimSize; oa::U32 innerSize; } push{
		axis.outerSize, axis.dimSize, axis.innerSize};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "Softmax", {&inA, &out}, access, &push, sizeof(push),
		axis.groupCount(), 1, 1, oa::detail::opRegistry::FnMatrix::softmax.name, 0,
		oa::detail::opRegistry::FnMatrix::softmax.hash, 0, 0, semantic.getValue());

	const auto attached = oa::detail::generatedAutogradAttach::FnMatrix::softmax(
		out, inA, inDim, semantic.getValue());
	if (not attached.isOk()) {
		OaLogError(oa::LogComponent::Compute,
			"Softmax semantic autograd attachment failed: %s",
			attached.getMessage().cStr());
		return {};
	}
	return out;
}

oa::Matrix oa::FnMatrix::logSoftmax(const oa::Matrix& inA, oa::I32 inDim) {
	FnMatrixAxisShape axis;
	if (not resolveFnMatrixAxis(inA, inDim, axis)) {
		OaLogError(oa::LogComponent::Compute,
			"LogSoftmax: expected a non-empty matrix and dim=-1 or a valid axis");
		return {};
	}

	auto& ctx = oa::ExecutionSession::getActive();
	oa::Matrix output = oa::FnMatrix::empty(inA.getShape(), inA.getDtype());
	const auto semantic = ctx.recordOp(
		oa::detail::opRegistry::FnMatrix::logSoftmax, {&inA}, {&output},
		{oa::OpAttribute::fromSignedInteger("dim", inDim)});
	if (not semantic.isOk()) return {};

	struct { oa::U32 outerSize; oa::U32 dimSize; oa::U32 innerSize; } push{
		axis.outerSize, axis.dimSize, axis.innerSize};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "LogSoftmax", {&inA, &output}, access, &push, sizeof(push),
		axis.groupCount(), 1, 1, oa::detail::opRegistry::FnMatrix::logSoftmax.name, 0,
		oa::detail::opRegistry::FnMatrix::logSoftmax.hash, 0, 0, semantic.getValue());

	const auto attached = oa::detail::generatedAutogradAttach::FnMatrix::logSoftmax(
		output, inA, inDim, semantic.getValue());
	if (not attached.isOk()) {
		OaLogError(oa::LogComponent::Compute,
			"LogSoftmax semantic autograd attachment failed: %s",
			attached.getMessage().cStr());
		return {};
	}
	return output;
}


// DescribeSum / DescribeMax buffer-level helpers retired. Sum/Mean/Max record
// through oa::ExecutionSession.


// ═══════════════════════════════════════════════════════════════════════════
// GPU-NATIVE OPERATIONS (VK_EXT path - zero CPU overhead)
// ═══════════════════════════════════════════════════════════════════════════
