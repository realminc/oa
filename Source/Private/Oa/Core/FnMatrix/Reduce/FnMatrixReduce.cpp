// OaFnMatrix — Reductions and softmax.
//
// Sum, Mean, Max, Argmax, Softmax, LogSoftmax.

#include <Oa/Core/Matrix.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>
#include <Oa/Core/BufferAccess.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/DispatchDesc.h>
#include <Oa/Core/Validation.h>
#include <Oa/Core/Operation.h>
#include <Oa/Ml/Autograd/Nodes.h>
#include "../../../Ml/Autograd/AutogradAttach.gen.h"
#include "../FnMatrixAxis.h"
#include "FnMatrixReduceLowering.h"

#include <cassert>
static OaU32 DivCeil(OaU32 InA, OaU32 InB) { return (InA + InB - 1) / InB; }

OaStatus OaFnMatrixPrivate::LowerFullMean(
	OaContext& InContext,
	const OaMatrix& InInput,
	OaMatrix& OutMean,
	const OaOperationContract& InContract,
	OaSemanticOperationId InSemanticOperation)
{
	const OaI64 elementCount = InInput.NumElements();
	if (elementCount <= 0
		or static_cast<OaU64>(elementCount) > std::numeric_limits<OaU32>::max()
		or OutMean.NumElements() != 1
		or OutMean.GetDtype() != InInput.GetDtype()
		or not InInput.HasStorage() or not OutMean.HasStorage())
	{
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"full mean lowering requires a non-empty 32-bit input and one same-dtype output");
	}

	const OaU32 count = static_cast<OaU32>(elementCount);
	OaMatrix sum = OaFnMatrix::Empty(OaMatrixShape{1}, InInput.GetDtype());
	if (not sum.HasStorage()) {
		return OaStatus::Error(OaStatusCode::OutOfMemory,
			"full mean lowering failed to allocate reduction storage");
	}
	struct { OaU32 Count; } reductionPush{count};
	OaBufferAccess reductionAccess[] = {
		OaBufferAccess::Read, OaBufferAccess::Write};
	const OaMatrix* reductionMatrices[] = {&InInput, &sum};
	OaMatrixDispatchDesc reduction;
	reduction.Dispatch.Operation = InContract.Name;
	reduction.Dispatch.SemanticOperations =
		OaSpan<const OaSemanticOperationId>(&InSemanticOperation, 1U);
	reduction.Dispatch.OperationContractHash = InContract.Hash;
	reduction.Dispatch.Kernel = "Sum";
	reduction.Dispatch.Access = OaSpan<OaBufferAccess>(reductionAccess, 2U);
	reduction.Dispatch.PushData = &reductionPush;
	reduction.Dispatch.PushSize = sizeof(reductionPush);
	reduction.Matrices = OaSpan<const OaMatrix* const>(reductionMatrices, 2U);
	OA_RETURN_IF_ERROR(InContext.Record(reduction));

	struct { OaU32 Count; OaF32 Alpha; } scalePush{
		1U, 1.0F / static_cast<OaF32>(count)};
	OaBufferAccess scaleAccess[] = {
		OaBufferAccess::Read, OaBufferAccess::Write};
	const OaMatrix* scaleMatrices[] = {&sum, &OutMean};
	OaMatrixDispatchDesc scale;
	scale.Dispatch.Operation = InContract.Name;
	scale.Dispatch.SemanticOperations =
		OaSpan<const OaSemanticOperationId>(&InSemanticOperation, 1U);
	scale.Dispatch.OperationContractHash = InContract.Hash;
	scale.Dispatch.Kernel = "Scale";
	scale.Dispatch.Access = OaSpan<OaBufferAccess>(scaleAccess, 2U);
	scale.Dispatch.PushData = &scalePush;
	scale.Dispatch.PushSize = sizeof(scalePush);
	scale.Matrices = OaSpan<const OaMatrix* const>(scaleMatrices, 2U);
	return InContext.Record(scale);
}

// Reductions
OaMatrix OaFnMatrix::Sum(const OaMatrix& InA, OaI32 InDim) {
	OaI64 n = InA.NumElements();
	auto& ctx = OaContext::GetDefault();

	OaI32 resolvedDim = InDim;
	if (resolvedDim >= 0 and resolvedDim < InA.Rank()) {
		OaI64 outerSize = 1;
		for (OaI32 i = 0; i < resolvedDim; ++i) outerSize *= InA.Size(i);
		OaI64 dimSize = InA.Size(resolvedDim);
		OaI64 innerSize = 1;
		for (OaI32 i = resolvedDim + 1; i < InA.Rank(); ++i) innerSize *= InA.Size(i);

		OaMatrixShape outShape;
		outShape.Rank = InA.Rank();
		for (OaI32 i = 0; i < InA.Rank(); ++i)
			outShape.Dims[i] = (i == resolvedDim) ? 1 : InA.Size(i);

		OaMatrix out = OaFnMatrix::Zeros(outShape, InA.Dtype_);
		const auto semantic = ctx.RecordOperation(
			OaOperationRegistry::Sum, {&InA}, {&out},
			{OaOperationAttribute::FromSignedInteger("Dim", InDim)});
		if (not semantic.IsOk()) return {};
		OaU32 totalOut = static_cast<OaU32>(outerSize * innerSize);
		struct { OaU32 OuterSize; OaU32 DimSize; OaU32 InnerSize; } push{
			static_cast<OaU32>(outerSize), static_cast<OaU32>(dimSize), static_cast<OaU32>(innerSize)};
		OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
		ctx.Add("SumDim", {&InA, &out}, access, &push, sizeof(push),
			DivCeil(totalOut, 256), 1, 1,
			OaOperationRegistry::Sum.Name, 0, OaOperationRegistry::Sum.Hash,
			0, 0, semantic.GetValue());

		const auto attached = OaGeneratedAutogradAttach::OaFnMatrix::Sum(
			out, InA, semantic.GetValue());
		if (not attached.IsOk()) {
			OA_LOG_ERROR(OaLogComponent::Core,
				"Sum semantic autograd attachment failed: %s",
				attached.GetMessage().c_str());
			return {};
		}
		return out;
	}

	OaMatrix out = OaFnMatrix::Zeros(OaMatrixShape{1}, InA.Dtype_);
	const auto semantic = ctx.RecordOperation(
		OaOperationRegistry::Sum, {&InA}, {&out},
		{OaOperationAttribute::FromSignedInteger("Dim", InDim)});
	if (not semantic.IsOk()) return {};
	struct { OaU32 Count; } push{static_cast<OaU32>(n)};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("Sum", {&InA, &out}, access, &push, sizeof(push), 1, 1, 1,
		OaOperationRegistry::Sum.Name, 0, OaOperationRegistry::Sum.Hash,
		0, 0, semantic.GetValue());

	const auto attached = OaGeneratedAutogradAttach::OaFnMatrix::Sum(
		out, InA, semantic.GetValue());
	if (not attached.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"Sum semantic autograd attachment failed: %s",
			attached.GetMessage().c_str());
		return {};
	}
	return out;
}

OaMatrix OaFnMatrix::Mean(const OaMatrix& InA, OaI32 InDim) {
	const OaI64 elementCount = InA.NumElements();
	if (InA.Rank() <= 0 or elementCount <= 0 or InDim < -1
		or InDim >= InA.Rank()
		or static_cast<OaU64>(elementCount)
			> std::numeric_limits<OaU32>::max())
	{
		OA_LOG_ERROR(OaLogComponent::ML,
			"Mean: expected a non-empty matrix and Dim=-1 or a valid axis");
		return {};
	}

	OaFnMatrixAxisShape axis;
	const bool reduceAxis = InDim >= 0;
	if (reduceAxis and not OaResolveFnMatrixAxis(InA, InDim, axis)) {
		OA_LOG_ERROR(OaLogComponent::ML,
			"Mean: selected axis exceeds the 32-bit reduction address space");
		return {};
	}

	OaMatrixShape outputShape{1};
	OaU32 outputCount = 1U;
	OaU32 reductionCount = static_cast<OaU32>(elementCount);
	if (reduceAxis) {
		outputShape.Rank = InA.Rank();
		for (OaI32 dim = 0; dim < InA.Rank(); ++dim) {
			outputShape.Dims[dim] = dim == InDim ? 1 : InA.Size(dim);
		}
		outputCount = axis.GroupCount();
		reductionCount = axis.DimSize;
	}

	auto& ctx = OaContext::GetDefault();
	OaMatrix out = OaFnMatrix::Empty(outputShape, InA.GetDtype());
	const auto semantic = ctx.RecordOperation(
		OaOperationRegistry::Mean, {&InA}, {&out},
		{OaOperationAttribute::FromSignedInteger("Dim", InDim)});
	if (not semantic.IsOk()) return {};
	if (not reduceAxis) {
		const auto lowering = OaFnMatrixPrivate::LowerFullMean(
			ctx, InA, out, OaOperationRegistry::Mean, semantic.GetValue());
		if (not lowering.IsOk()) {
			OA_LOG_ERROR(OaLogComponent::ML,
				"Mean lowering failed: %s", lowering.GetMessage().c_str());
			return {};
		}

		const auto attached = OaGeneratedAutogradAttach::OaFnMatrix::Mean(
			out, InA, InDim, semantic.GetValue());
		if (not attached.IsOk()) {
			OA_LOG_ERROR(OaLogComponent::ML,
				"Mean semantic autograd attachment failed: %s",
				attached.GetMessage().c_str());
			return {};
		}
		return out;
	}

	OaMatrix sum = OaFnMatrix::Empty(outputShape, InA.GetDtype());
	OaBufferAccess reductionAccess[] = {
		OaBufferAccess::Read, OaBufferAccess::Write};
	struct { OaU32 OuterSize; OaU32 DimSize; OaU32 InnerSize; } push{
		axis.OuterSize, axis.DimSize, axis.InnerSize};
	ctx.Add("SumDim", {&InA, &sum}, reductionAccess, &push, sizeof(push),
		DivCeil(outputCount, 256U), 1, 1,
		OaOperationRegistry::Mean.Name, 0, OaOperationRegistry::Mean.Hash,
		0, 0, semantic.GetValue());

	struct { OaU32 Count; OaF32 Alpha; } scalePush{
		outputCount, 1.0F / static_cast<OaF32>(reductionCount)};
	OaBufferAccess scaleAccess[] = {
		OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("Scale", {&sum, &out}, scaleAccess, &scalePush, sizeof(scalePush),
		DivCeil(outputCount, 256U), 1, 1,
		OaOperationRegistry::Mean.Name, 0, OaOperationRegistry::Mean.Hash,
		0, 0, semantic.GetValue());

	const auto attached = OaGeneratedAutogradAttach::OaFnMatrix::Mean(
		out, InA, InDim, semantic.GetValue());
	if (not attached.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::ML,
			"Mean semantic autograd attachment failed: %s",
			attached.GetMessage().c_str());
		return {};
	}
	return out;
}

OaMatrix OaFnMatrix::Max(const OaMatrix& InA, OaI32 InDim) {
	(void)InDim;
	auto& ctx = OaContext::GetDefault();

	// Compute max value (full reduction → scalar)
	OaMatrix out = OaFnMatrix::Empty(OaMatrixShape{1}, InA.Dtype_);
	const auto semantic = ctx.RecordOperation(
		OaOperationRegistry::Max, {&InA}, {&out});
	if (not semantic.IsOk()) return {};
	struct { OaU32 Count; } push{static_cast<OaU32>(InA.NumElements())};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("Max", {&InA, &out}, access, &push, sizeof(push), 1, 1, 1,
		OaOperationRegistry::Max.Name, 0, OaOperationRegistry::Max.Hash,
		0, 0, semantic.GetValue());

	const auto attached = OaGeneratedAutogradAttach::OaFnMatrix::Max(
		out, InA, semantic.GetValue());
	if (not attached.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"Max semantic autograd attachment failed: %s",
			attached.GetMessage().c_str());
		return {};
	}

	return out;
}

OaI64 OaFnMatrix::Argmax(const OaMatrix& InA, OaI32 InDim) {
	(void)InDim;
	auto& ctx = OaContext::GetDefault();
	OaMatrix out = OaFnMatrix::Empty(OaMatrixShape{1}, OaScalarType::UInt32);
	struct { OaU32 Count; } push{static_cast<OaU32>(InA.NumElements())};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("Argmax", {&InA, &out}, access, &push, sizeof(push), 1);
	auto status = ctx.Execute();
	assert(status.IsOk() && "OaContext::Execute failed before Argmax readback");
	auto syncStatus = ctx.Sync();
	assert(syncStatus.IsOk() && "OaContext::Sync failed before Argmax readback");
	OaU32 index = 0;
	auto copyStatus = OaFnMatrix::CopyToHost(out, &index, sizeof(index));
	assert(copyStatus.IsOk() && "Argmax readback failed");
	return static_cast<OaI64>(index);
}

OaMatrix OaFnMatrix::CategoricalAccuracyCount(
	const OaMatrix& InLogits, const OaMatrix& InLabels) {
	if (InLogits.Rank() < 2 or InLogits.NumElements() == 0 or
		InLabels.NumElements() != InLogits.NumElements() / InLogits.Size(InLogits.Rank() - 1) or
		(InLabels.GetDtype() != OaScalarType::UInt8 and
		 InLabels.GetDtype() != OaScalarType::UInt32 and
		 InLabels.GetDtype() != OaScalarType::Int32)) {
		OA_LOG_ERROR(OaLogComponent::ML,
			"CategoricalAccuracyCount: expected logits [...,C] and UInt8/UInt32/Int32 labels [...]");
		return {};
	}
	const OaU32 classes = static_cast<OaU32>(InLogits.Size(InLogits.Rank() - 1));
	const OaU32 rows = static_cast<OaU32>(InLabels.NumElements());
	const OaU32 labelType = InLabels.GetDtype() == OaScalarType::UInt8 ? 0u
		: (InLabels.GetDtype() == OaScalarType::UInt32 ? 1u : 2u);
	OaMatrix out = OaFnMatrix::Empty(OaMatrixShape{1}, OaScalarType::UInt32);
	struct { OaU32 Rows, Classes, LabelType; } push{rows, classes, labelType};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
	auto& ctx = OaContext::GetDefault();
	ctx.Add("CategoricalAccuracyCount", {&InLogits, &InLabels, &out},
		access, &push, sizeof(push), 1);
	return out;
}

OaMatrix OaFnMatrix::MaskedCategoricalAccuracyCount(
	const OaMatrix& InLogits, const OaMatrix& InLabels, const OaMatrix& InMask) {
	if (InLogits.Rank() < 2 or InLogits.NumElements() == 0 or
		InLabels.NumElements() != InLogits.NumElements() / InLogits.Size(InLogits.Rank() - 1) or
		InMask.NumElements() != InLabels.NumElements() or
		InMask.GetDtype() != InLogits.GetDtype() or
		(InLabels.GetDtype() != OaScalarType::UInt8 and
		 InLabels.GetDtype() != OaScalarType::UInt32 and
		 InLabels.GetDtype() != OaScalarType::Int32) or
		(InMask.GetDtype() != OaScalarType::Float32 and
		 InMask.GetDtype() != OaScalarType::BFloat16)) {
		OA_LOG_ERROR(OaLogComponent::ML,
			"MaskedCategoricalAccuracyCount: expected logits [...,C], integer labels [...], and a same-dtype floating mask [...]");
		return {};
	}
	const OaU32 classes = static_cast<OaU32>(InLogits.Size(InLogits.Rank() - 1));
	const OaU32 rows = static_cast<OaU32>(InLabels.NumElements());
	const OaU32 labelType = InLabels.GetDtype() == OaScalarType::UInt8 ? 0u
		: (InLabels.GetDtype() == OaScalarType::UInt32 ? 1u : 2u);
	OaMatrix out = OaFnMatrix::Empty(OaMatrixShape{1}, OaScalarType::UInt32);
	struct { OaU32 Rows, Classes, LabelType; } push{rows, classes, labelType};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Read, OaBufferAccess::Write};
	auto& ctx = OaContext::GetDefault();
	ctx.Add("MaskedCategoricalAccuracyCount", {&InLogits, &InLabels, &InMask, &out},
		access, &push, sizeof(push), 1);
	return out;
}

OaMatrix OaFnMatrix::Softmax(const OaMatrix& InA, OaI32 InDim) {
	OaFnMatrixAxisShape axis;
	if (not OaResolveFnMatrixAxis(InA, InDim, axis)) {
		OA_LOG_ERROR(OaLogComponent::ML,
			"Softmax: expected a non-empty matrix and Dim=-1 or a valid axis");
		return {};
	}
	auto& ctx = OaContext::GetDefault();
	OaMatrix out = OaFnMatrix::Empty(InA.Shape_, InA.Dtype_);
	const auto semantic = ctx.RecordOperation(
		OaOperationRegistry::Softmax, {&InA}, {&out},
		{OaOperationAttribute::FromSignedInteger("Dim", InDim)});
	if (not semantic.IsOk()) return {};

	struct { OaU32 OuterSize; OaU32 DimSize; OaU32 InnerSize; } push{
		axis.OuterSize, axis.DimSize, axis.InnerSize};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("Softmax", {&InA, &out}, access, &push, sizeof(push),
		axis.GroupCount(), 1, 1, OaOperationRegistry::Softmax.Name, 0,
		OaOperationRegistry::Softmax.Hash, 0, 0, semantic.GetValue());

	const auto attached = OaGeneratedAutogradAttach::OaFnMatrix::Softmax(
		out, InA, InDim, semantic.GetValue());
	if (not attached.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::ML,
			"Softmax semantic autograd attachment failed: %s",
			attached.GetMessage().c_str());
		return {};
	}
	return out;
}

OaMatrix OaFnMatrix::LogSoftmax(const OaMatrix& InA, OaI32 InDim) {
	OaFnMatrixAxisShape axis;
	if (not OaResolveFnMatrixAxis(InA, InDim, axis)) {
		OA_LOG_ERROR(OaLogComponent::ML,
			"LogSoftmax: expected a non-empty matrix and Dim=-1 or a valid axis");
		return {};
	}

	auto& ctx = OaContext::GetDefault();
	OaMatrix output = OaFnMatrix::Empty(InA.GetShape(), InA.GetDtype());
	const auto semantic = ctx.RecordOperation(
		OaOperationRegistry::LogSoftmax, {&InA}, {&output},
		{OaOperationAttribute::FromSignedInteger("Dim", InDim)});
	if (not semantic.IsOk()) return {};

	struct { OaU32 OuterSize; OaU32 DimSize; OaU32 InnerSize; } push{
		axis.OuterSize, axis.DimSize, axis.InnerSize};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("LogSoftmax", {&InA, &output}, access, &push, sizeof(push),
		axis.GroupCount(), 1, 1, OaOperationRegistry::LogSoftmax.Name, 0,
		OaOperationRegistry::LogSoftmax.Hash, 0, 0, semantic.GetValue());

	const auto attached = OaGeneratedAutogradAttach::OaFnMatrix::LogSoftmax(
		output, InA, InDim, semantic.GetValue());
	if (not attached.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::ML,
			"LogSoftmax semantic autograd attachment failed: %s",
			attached.GetMessage().c_str());
		return {};
	}
	return output;
}


// DescribeSum / DescribeMax buffer-level helpers retired. Sum/Mean/Max record
// through OaContext.


// ═══════════════════════════════════════════════════════════════════════════
// GPU-NATIVE OPERATIONS (VK_EXT path - zero CPU overhead)
// ═══════════════════════════════════════════════════════════════════════════
