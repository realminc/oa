// oa::FnLoss — loss function implementations.
// Extracted from oa::FnMatrix for focused loss ownership.

#include <oa/ml/fnLoss.h>
#include <oa/core/log.h>
#include <oa/ml/autograd/loss/autogradLoss.h>
#include <oa/core/matrix.h>
#include <oa/core/matrixAccess.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/status.h>
#include <oa/core/types.h>
#include <oa/core/bufferAccess.h>
#include <oa/runtime/executionSession.h>
#include <oa/core/validation.h>
#include <oa/core/op.h>
#include <oa/runtime/dispatchDesc.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/engine/deviceAccess.h>
#include "../autograd/autogradAttach.gen.h"
#include <oa/core/fnmatrix/reduce/fnMatrixReduceLowering.h>
#include <oa/core/std/algo.h>
#include <oa/core/std/limits.h>
#include "../../runtime/descriptorValidation.h"

namespace {
thread_local const char* gLastLossName = nullptr;
constexpr oa::U32 kPortableDispatchTileWidthX = 65535U;

oa::Status recordLossDispatch(
	oa::ExecutionSession& inContext, oa::StringView inKernel,
	oa::Span<const oa::Matrix* const> inMatrices,
	oa::Span<oa::BufferAccess> inAccess,
	const void* inPush, oa::U32 inPushSize,
	oa::U32 inGroupsX, oa::U32 inGroupsY,
	const oa::OpContract* inContract = nullptr,
	oa::U32 inSemanticOp = oa::invalidSemanticOpId
) {
	oa::MatrixDispatchDesc desc;
	desc.dispatch.kernel = inKernel;
	desc.dispatch.access = inAccess;
	desc.dispatch.pushData = inPush;
	desc.dispatch.pushSize = inPushSize;
	desc.dispatch.groupsX = inGroupsX;
	desc.dispatch.groupsY = inGroupsY;
	desc.matrices = inMatrices;
	if (inContract) {
		desc.dispatch.operation = inContract->name;
		desc.dispatch.opContractHash = inContract->hash;
	}
	if (inSemanticOp != oa::invalidSemanticOpId) {
		desc.dispatch.semanticOps =
			oa::Span<const oa::U32>(&inSemanticOp, 1U);
	}
	return inContext.record( desc);
}

bool checkedShaderBytes(const oa::Matrix& inMatrix, oa::U64 inElements, oa::U64& outBytes) {
	constexpr oa::U64 kShaderByteAddressSpace =
		static_cast<oa::U64>(oa::Limits<oa::U32>::max()) + 1U;
	const oa::U64 elementBytes = static_cast<oa::U64>(oa::scalarSize(inMatrix.getDtype()));
	if (elementBytes == 0 or inElements > kShaderByteAddressSpace / elementBytes) {
		return false;
	}
	outBytes = inElements * elementBytes;
	return true;
}

bool hasDirectShaderStorage(
	const oavk::Device& inDevice,
	const oa::Matrix& inMatrix,
	oa::U64 inLogicalBytes,
	bool inNeedsWordTail
) {
	if (inMatrix.byteOffset() != 0 or not inMatrix.getStride().matchesRowMajor(inMatrix.getShape()))	{
		return false;
	}
	const oavk::Buffer buffer = oa::MatrixAccess::descriptor(inMatrix);
	if (not buffer.buffer or buffer.bindlessIndex == UINT32_MAX
		or buffer.size < inLogicalBytes) {
		return false;
	}
	oa::U64 descriptorBytes = inLogicalBytes;
	if (inNeedsWordTail) {
		if (descriptorBytes > oa::Limits<oa::U64>::max() - 3U) return false;
		descriptorBytes = (descriptorBytes + 3U) & ~oa::U64{3U};
	}
	const oa::U64 descriptorRange = buffer.descriptorRange();
	return descriptorRange >= descriptorBytes
		and oavk::validateStorageBufferDescriptor(
			inDevice, buffer, true).isOk();
}

bool hasCrossEntropyShaderStorage(
	const oa::ExecutionSession& inContext,
	const oa::Matrix& inMatrix,
	oa::U64 inElements
) {
	oa::U64 logicalBytes = 0;
	if (not checkedShaderBytes(inMatrix, inElements, logicalBytes)) return false;
	const oa::U64 elementBytes =
		static_cast<oa::U64>(oa::scalarSize(inMatrix.getDtype()));
	const bool needsWordTail = elementBytes < sizeof(oa::U32);
	return hasDirectShaderStorage(
		oa::EngineDeviceAccess::get(inContext.engine()),
		inMatrix, logicalBytes, needsWordTail);
}

bool validateCrossEntropyInputs(const char* inOperation,
	const oa::ExecutionSession& inContext,
	const oa::Matrix& inLogits, const oa::Matrix& inTargets,
	bool inLogFailure = true)
{
	const bool shapeAndDtypeValid =
		inLogits.rank() == 2 and inLogits.size(0) > 0 and inLogits.size(1) > 0
		and inTargets.rank() == 1 and inTargets.size(0) == inLogits.size(0)
		and (inLogits.getDtype() == oa::ScalarType::Float32
			or inLogits.getDtype() == oa::ScalarType::BFloat16)
		and (inTargets.getDtype() == oa::ScalarType::UInt8
			or inTargets.getDtype() == oa::ScalarType::UInt32
			or inTargets.getDtype() == oa::ScalarType::Int32);
	if (shapeAndDtypeValid) {
		const oa::U64 rows = static_cast<oa::U64>(inLogits.size(0));
		const oa::U64 classes = static_cast<oa::U64>(inLogits.size(1));
		const oa::U64 maxShaderIndex = oa::Limits<oa::U32>::max();
		const oa::U64 maxPortableRows =
			static_cast<oa::U64>(kPortableDispatchTileWidthX)
			* kPortableDispatchTileWidthX;
		oa::U64 logitsElements = 0;
		if (rows <= maxPortableRows and classes <= maxShaderIndex
			and rows <= oa::Limits<oa::U64>::max() / classes)
		{
			logitsElements = rows * classes;
			if (rows <= (maxShaderIndex + oa::U64{1U}) / sizeof(oa::F32)
				and hasCrossEntropyShaderStorage(
					inContext, inLogits, logitsElements)
				and hasCrossEntropyShaderStorage(
					inContext, inTargets, rows))
			{
				return true;
			}
		}
	}
	if (inLogFailure) {
		OaLogError(oa::LogComponent::Ml,
			"%s: expected packed zero-offset, fully backed, shader-byte-addressable rank-two Float32/BFloat16 logits and one rank-one UInt8/UInt32/Int32 target per row, with every padded descriptor range within the selected device maxStorageBufferRange",
			inOperation
		);
	}
	return false;
}

bool validateMaskedCrossEntropyInputs(
	const char* inOperation,
	const oa::ExecutionSession& inContext,
	const oa::Matrix& inLogits,
	const oa::Matrix& inTargets,
	const oa::Matrix& inMask,
	oa::I32 inValidCount
) {
	if (validateCrossEntropyInputs(
			inOperation, inContext, inLogits, inTargets, false)
		and inMask.rank() == 1
		and inMask.size(0) == inLogits.size(0)
		and inMask.getDtype() == inLogits.getDtype()
		and inValidCount > 0
		and inValidCount <= inLogits.size(0)
		and hasCrossEntropyShaderStorage(
			inContext, inMask, static_cast<oa::U64>(inLogits.size(0))))
	{
		return true;
	}
	OaLogError(oa::LogComponent::Ml,
		"%s: expected valid packed CrossEntropy inputs, a floating mask matching the logits dtype and row count, a valid selected-row count, and every padded descriptor range within the selected device maxStorageBufferRange",
		inOperation);
	return false;
}

bool validateCrossEntropyOutput(
	const char* inOperation,
	const oa::ExecutionSession& inContext,
	const oa::Matrix& inOutput,
	oa::U64 inElements
) {
	if (inOutput.hasStorage()
		and hasCrossEntropyShaderStorage(inContext, inOutput, inElements))
	{
		return true;
	}
	OaLogError(oa::LogComponent::Ml,
		"%s: output allocation does not provide a fully backed descriptor range within the selected device maxStorageBufferRange",
		inOperation);
	return false;
}

oa::Matrix commitLossResult(
	oa::Matrix inResult,
	oa::OpLoweringScope& inLowering,
	const oa::OpContract& inContract,
	oa::MatrixArgs inInputs,
	oa::OpAttributeArgs inAttributes = {}) {
	auto semantic = inLowering.commitWithId(
		inContract, inInputs, {&inResult}, inAttributes);
	if (not semantic.isOk()) return {};
	if (auto grad = inResult.getGradFn()) {
		if (not oa::FnAutograd::attachSemantic(
			grad, semantic.getValue()).isOk())
		{
			return {};
		}
	}
	return inResult;
}
} // namespace

const char* oa::FnLoss::lastName() {
	return gLastLossName;
}

void oa::FnLoss::setLastName(const char* inName) {
	gLastLossName = inName;
}

oa::Matrix oa::FnLoss::crossEntropy(const oa::Matrix& inLogits, const oa::Matrix& inTargets) {
	gLastLossName = "cross_entropy";
	auto& ctx = oa::ExecutionSession::getActive();
	if (not validateCrossEntropyInputs(
		"CrossEntropy", ctx, inLogits, inTargets)) return {};
	oa::U32 batch = static_cast<oa::U32>(inLogits.size(0));
	oa::U32 classes = static_cast<oa::U32>(inLogits.size(1));

	// Per-sample loss is always FP32 (dtype-invariant output): CrossEntropy computes
	// log_sum_exp in fp32 internally and the Sum/Mean reduction accumulates in fp32.
	// Storing it bf16 would round each ~ln(V) row loss to ~2-3 sig digits and feed a
	// lossy value into the reduction — the cross-entropy shader uses oaStoreF32 to
	// match this dtype contract.
	oa::Matrix perSample = oa::FnMatrix::empty(oa::MatrixShape{batch}, oa::ScalarType::Float32);
	oa::Matrix loss = oa::FnMatrix::empty(oa::MatrixShape{1}, oa::ScalarType::Float32);
	if (not validateCrossEntropyOutput(
			"CrossEntropy", ctx, perSample, batch)
		or not loss.hasStorage()) {
		OaLogError(oa::LogComponent::Ml,
			"CrossEntropy: failed to allocate per-sample or scalar output storage");
		return {};
	}
	const auto semantic = ctx.recordOp(
		oa::detail::opRegistry::FnLoss::crossEntropy,
		{&inLogits, &inTargets}, {&loss});
	if (not semantic.isOk()) {
		OaLogError(oa::LogComponent::Ml,
			"CrossEntropy semantic recording failed: %s",
			semantic.getStatus().getMessage().cStr());
		return {};
	}

	// Pass targets directly. Non-negative Int32 and UInt32 class indices have
	// identical storage, so the shader's 32-bit path intentionally accepts both.
	oa::U32 targetDtype = (inTargets.getDtype() == oa::ScalarType::UInt8) ? 0U : 1U;
	struct { oa::U32 batch; oa::U32 classes; oa::U32 targetDtype; }
		push{.batch = batch, .classes = classes, .targetDtype = targetDtype};
	[[maybe_unused]] const oavk::Buffer logitsBuf = oa::MatrixAccess::descriptor(inLogits);
	[[maybe_unused]] const oavk::Buffer targetsBuf = oa::MatrixAccess::descriptor(inTargets);
	[[maybe_unused]] const oavk::Buffer lossBuf = oa::MatrixAccess::descriptor(perSample);
	OaLogDebug(oa::LogComponent::Ml, "CrossEntropy: batch=%u classes=%u logits_buf=%u targets_buf=%u loss_buf=%u",
		batch, classes, logitsBuf.bindlessIndex, targetsBuf.bindlessIndex, lossBuf.bindlessIndex);

	oa::BufferAccess access[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write};
	const oa::Matrix* matrices[] = {&inLogits, &inTargets, &perSample};
	const oa::U32 groupsX = oa::min(batch, kPortableDispatchTileWidthX);
	const oa::U32 groupsY =
		1U + (batch - 1U) / kPortableDispatchTileWidthX;
	const auto dispatch = recordLossDispatch(
		ctx, "CrossEntropy", oa::Span<const oa::Matrix* const>(matrices, 3U),
		oa::Span<oa::BufferAccess>(access, 3U), &push, sizeof(push),
		groupsX, groupsY, &oa::detail::opRegistry::FnLoss::crossEntropy,
		semantic.getValue());
	if (not dispatch.isOk()) {
		OaLogError(oa::LogComponent::Ml,
			"CrossEntropy dispatch recording failed: %s",
			dispatch.getMessage().cStr());
		return {};
	}

	const auto lowering = oa::FnMatrixPrivate::lowerFullMean(
		ctx, perSample, loss,
		oa::detail::opRegistry::FnLoss::crossEntropy, semantic.getValue());
	if (not lowering.isOk()) {
		OaLogError(oa::LogComponent::Ml,
			"CrossEntropy mean lowering failed: %s",
			lowering.getMessage().cStr());
		return {};
	}

	const auto attached = oa::detail::generatedAutogradAttach::FnLoss::crossEntropy(
		loss, inLogits, inTargets, semantic.getValue());
	if (not attached.isOk()) {
		OaLogError(oa::LogComponent::Ml,
			"CrossEntropy semantic autograd attachment failed: %s",
			attached.getMessage().cStr());
		return {};
	}

	return loss;
}

oa::Matrix oa::FnLoss::crossEntropyBwd(const oa::Matrix& inLogits, const oa::Matrix& inTargets) {
	gLastLossName = "cross_entropy";
	auto& ctx = oa::ExecutionSession::getActive();
	if (not validateCrossEntropyInputs(
		"CrossEntropyBwd", ctx, inLogits, inTargets)) return {};
	oa::OpLoweringScope lowering(ctx);
	oa::U32 batch = static_cast<oa::U32>(inLogits.size(0));
	oa::U32 classes = static_cast<oa::U32>(inLogits.size(1));

	oa::Matrix gradLogits = oa::FnMatrix::empty(inLogits.getShape(), inLogits.getDtype());
	if (not validateCrossEntropyOutput(
			"CrossEntropyBwd", ctx, gradLogits,
			static_cast<oa::U64>(batch) * classes)) {
		OaLogError(oa::LogComponent::Ml,
			"CrossEntropyBwd: failed to allocate gradient storage");
		return {};
	}

	// UInt32 and non-negative Int32 share the shader's 32-bit index path.
	oa::U32 targetDtype = (inTargets.getDtype() == oa::ScalarType::UInt8) ? 0U : 1U;
	struct { oa::U32 batch; oa::U32 classes; oa::U32 targetDtype; }
		push{.batch = batch, .classes = classes, .targetDtype = targetDtype};

	oa::BufferAccess access[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write};
	const oa::Matrix* matrices[] = {&inLogits, &inTargets, &gradLogits};
	const oa::U32 groupsX = oa::min(batch, kPortableDispatchTileWidthX);
	const oa::U32 groupsY =
		1U + (batch - 1U) / kPortableDispatchTileWidthX;
	const auto dispatch = recordLossDispatch(
		ctx, "CrossEntropyBwd", oa::Span<const oa::Matrix* const>(matrices, 3U),
		oa::Span<oa::BufferAccess>(access, 3U), &push, sizeof(push),
		groupsX, groupsY);
	if (not dispatch.isOk()) {
		OaLogError(oa::LogComponent::Ml,
			"CrossEntropyBwd dispatch recording failed: %s",
			dispatch.getMessage().cStr());
		return {};
	}

	return commitLossResult(
		oa::move(gradLogits), lowering,
		oa::detail::opRegistry::FnLoss::crossEntropyBwd,
		{&inLogits, &inTargets});
}

oa::Matrix oa::FnLoss::maskedCrossEntropy(const oa::Matrix& inLogits,
	const oa::Matrix& inTargets, const oa::Matrix& inMask, oa::I32 inValidCount) {
	gLastLossName = "cross_entropy";
	auto& ctx = oa::ExecutionSession::getActive();
	if (not validateMaskedCrossEntropyInputs(
		"MaskedCrossEntropy", ctx, inLogits, inTargets,
		inMask, inValidCount)) return {};
	oa::OpLoweringScope lowering(ctx);
	const oa::U32 rows = static_cast<oa::U32>(inLogits.size(0));
	const oa::U32 classes = static_cast<oa::U32>(inLogits.size(1));
	oa::Matrix perSample = oa::FnMatrix::empty(oa::MatrixShape{rows}, oa::ScalarType::Float32);
	if (not validateCrossEntropyOutput(
			"MaskedCrossEntropy", ctx, perSample, rows)) return {};
	const oa::U32 targetDtype = (inTargets.getDtype() == oa::ScalarType::UInt8) ? 0U : 1U;
	struct { oa::U32 rows, classes, targetDtype; } push{rows, classes, targetDtype};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Write};
	const oa::Matrix* matrices[] = {
		&inLogits, &inTargets, &inMask, &perSample};
	const oa::U32 groupsX = oa::min(rows, kPortableDispatchTileWidthX);
	const oa::U32 groupsY =
		1U + (rows - 1U) / kPortableDispatchTileWidthX;
	const auto dispatch = recordLossDispatch(
		ctx, "MaskedCrossEntropy",
		oa::Span<const oa::Matrix* const>(matrices, 4U),
		oa::Span<oa::BufferAccess>(access, 4U), &push, sizeof(push),
		groupsX, groupsY);
	if (not dispatch.isOk()) {
		OaLogError(oa::LogComponent::Ml,
			"MaskedCrossEntropy dispatch recording failed: %s",
			dispatch.getMessage().cStr());
		return {};
	}
	oa::Matrix loss = oa::FnMatrix::scale(oa::FnMatrix::sum(perSample),
		1.0F / static_cast<oa::F32>(inValidCount));
	if (oa::FnAutograd::isEnabled() and inLogits.requiresGrad()) {
		auto gradFn = oa::makeShared<oa::GradMaskedCrossEntropy>();
		gradFn->saveForBackward(inLogits, inTargets, inMask);
		gradFn->setGraphInputs(oa::Vec<oa::Matrix>{inLogits, inTargets, inMask});
		gradFn->validCount_ = inValidCount;
		gradFn->sequenceNr_ = oa::FnAutograd::nextSeq();
		loss.mutAutograd().gradFn = gradFn;
	}
	return commitLossResult(
		oa::move(loss), lowering,
		oa::detail::opRegistry::FnLoss::maskedCrossEntropy,
		{&inLogits, &inTargets, &inMask},
		{oa::OpAttribute::fromSignedInteger(
			"validCount", inValidCount)});
}

oa::Matrix oa::FnLoss::maskedCrossEntropyBwd(const oa::Matrix& inLogits,
	const oa::Matrix& inTargets, const oa::Matrix& inMask, oa::I32 inValidCount) {
	gLastLossName = "cross_entropy";
	auto& ctx = oa::ExecutionSession::getActive();
	if (not validateMaskedCrossEntropyInputs(
		"MaskedCrossEntropyBwd", ctx, inLogits, inTargets,
		inMask, inValidCount)) return {};
	oa::OpLoweringScope lowering(ctx);
	const oa::U32 rows = static_cast<oa::U32>(inLogits.size(0));
	const oa::U32 classes = static_cast<oa::U32>(inLogits.size(1));
	const oa::U32 targetDtype = (inTargets.getDtype() == oa::ScalarType::UInt8) ? 0U : 1U;
	oa::Matrix gradLogits = oa::FnMatrix::empty(inLogits.getShape(), inLogits.getDtype());
	if (not validateCrossEntropyOutput(
			"MaskedCrossEntropyBwd", ctx, gradLogits,
			static_cast<oa::U64>(rows) * classes)) return {};
	struct { oa::U32 rows, classes, targetDtype, validCount; }
		push{rows, classes, targetDtype, static_cast<oa::U32>(inValidCount)};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Write};
	const oa::Matrix* matrices[] = {
		&inLogits, &inTargets, &inMask, &gradLogits};
	const oa::U32 groupsX = oa::min(rows, kPortableDispatchTileWidthX);
	const oa::U32 groupsY =
		1U + (rows - 1U) / kPortableDispatchTileWidthX;
	const auto dispatch = recordLossDispatch(
		ctx, "MaskedCrossEntropyBwd",
		oa::Span<const oa::Matrix* const>(matrices, 4U),
		oa::Span<oa::BufferAccess>(access, 4U), &push, sizeof(push),
		groupsX, groupsY);
	if (not dispatch.isOk()) {
		OaLogError(oa::LogComponent::Ml,
			"MaskedCrossEntropyBwd dispatch recording failed: %s",
			dispatch.getMessage().cStr());
		return {};
	}
	return commitLossResult(
		oa::move(gradLogits), lowering,
		oa::detail::opRegistry::FnLoss::maskedCrossEntropyBwd,
		{&inLogits, &inTargets, &inMask},
		{oa::OpAttribute::fromSignedInteger(
			"validCount", inValidCount)});
}

// CrossEntropyLossGradBwd fused public wrapper removed (oa::Module.md phase 1).
// The CrossEntropyLossGradBwd kernel remains in the registry for Api3-style
// hand-wired graphs that call oa::ExecutableGraph::add("CrossEntropyLossGradBwd", ...).

// Huber and Nll losses were unimplemented stubs that asserted on call. They
// have been removed. smoothL1 (Huber) is now implemented with fused kernels
// above. Nll remains pending a target-gather op.

// ─── smoothL1 (Huber loss, beta=1.0) ───────────────────────────────────

oa::Matrix oa::FnLoss::smoothL1(const oa::Matrix& inA, const oa::Matrix& inB) {
	gLastLossName = "smooth_l1";
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);

	oa::U32 count = static_cast<oa::U32>(inA.numElements());

	oa::Matrix perElement = oa::FnMatrix::empty(inA.getShape(), inA.getDtype());

	struct { oa::U32 count; } push{.count = count};

	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "SmoothL1", {&inA, &inB, &perElement}, access, &push, sizeof(push), oa::divCeil(count, 256U));

	oa::Matrix loss = oa::FnMatrix::mean(perElement);

	if (oa::FnAutograd::isEnabled() and inA.requiresGrad()) {
		auto gradFn = oa::makeShared<oa::GradSmoothL1>();
		gradFn->saveForBackward(inA, inB);
		gradFn->setGraphInputs(oa::Vec<oa::Matrix>{inA, inB});
		gradFn->sequenceNr_  = oa::FnAutograd::nextSeq();
		loss.mutAutograd().gradFn = gradFn;
	}

	return commitLossResult(
		oa::move(loss), lowering,
		oa::detail::opRegistry::FnLoss::smoothL1, {&inA, &inB});
}

oa::Matrix oa::FnLoss::smoothL1Bwd(const oa::Matrix& inA, const oa::Matrix& inB) {
	gLastLossName = "smooth_l1";
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);

	oa::U32 count = static_cast<oa::U32>(inA.numElements());

	oa::Matrix gradA = oa::FnMatrix::empty(inA.getShape(), inA.getDtype());

	struct { oa::U32 count; } push{.count = count};

	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "SmoothL1Bwd", {&inA, &inB, &gradA}, access, &push, sizeof(push), oa::divCeil(count, 256U));

	return commitLossResult(
		oa::move(gradA), lowering,
		oa::detail::opRegistry::FnLoss::smoothL1Bwd, {&inA, &inB});
}

// ─── mse (Mean Squared Error) ──────────────────────────────────────────

oa::Matrix oa::FnLoss::mse(const oa::Matrix& inA, const oa::Matrix& inB) {
	gLastLossName = "mse";
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);

	oa::U32 count = static_cast<oa::U32>(inA.numElements());

	oa::Matrix perElement = oa::FnMatrix::empty(inA.getShape(), inA.getDtype());

	struct { oa::U32 count; } push{.count = count};

	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "Mse", {&inA, &inB, &perElement}, access, &push, sizeof(push), oa::divCeil(count, 256U));

	oa::Matrix loss = oa::FnMatrix::mean(perElement);

	if (oa::FnAutograd::isEnabled() and inA.requiresGrad()) {
		auto gradFn = oa::makeShared<oa::GradMse>();
		gradFn->saveForBackward(inA, inB);
		gradFn->setGraphInputs(oa::Vec<oa::Matrix>{inA, inB});
		gradFn->sequenceNr_  = oa::FnAutograd::nextSeq();
		loss.mutAutograd().gradFn = gradFn;
	}

	return commitLossResult(
		oa::move(loss), lowering,
		oa::detail::opRegistry::FnLoss::mse, {&inA, &inB});
}

oa::Matrix oa::FnLoss::mseBwd(const oa::Matrix& inA, const oa::Matrix& inB) {
	gLastLossName = "mse";
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);

	oa::U32 count = static_cast<oa::U32>(inA.numElements());

	oa::Matrix gradA = oa::FnMatrix::empty(inA.getShape(), inA.getDtype());

	struct { oa::U32 count; } push{.count = count};

	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "MseBwd", {&inA, &inB, &gradA}, access, &push, sizeof(push), oa::divCeil(count, 256U));

	return commitLossResult(
		oa::move(gradA), lowering,
		oa::detail::opRegistry::FnLoss::mseBwd, {&inA, &inB});
}

// ─── L1 (Mean Absolute Error) ──────────────────────────────────────────

oa::Matrix oa::FnLoss::l1(const oa::Matrix& inA, const oa::Matrix& inB) {
	gLastLossName = "l1";
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);

	oa::U32 count = static_cast<oa::U32>(inA.numElements());

	oa::Matrix perElement = oa::FnMatrix::empty(inA.getShape(), inA.getDtype());

	struct { oa::U32 count; } push{.count = count};

	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "L1", {&inA, &inB, &perElement}, access, &push, sizeof(push), oa::divCeil(count, 256U));

	oa::Matrix loss = oa::FnMatrix::mean(perElement);

	if (oa::FnAutograd::isEnabled() and inA.requiresGrad()) {
		auto gradFn = oa::makeShared<oa::GradL1>();
		gradFn->saveForBackward(inA, inB);
		gradFn->setGraphInputs(oa::Vec<oa::Matrix>{inA, inB});
		gradFn->sequenceNr_  = oa::FnAutograd::nextSeq();
		loss.mutAutograd().gradFn = gradFn;
	}

	return commitLossResult(
		oa::move(loss), lowering,
		oa::detail::opRegistry::FnLoss::l1, {&inA, &inB});
}

oa::Matrix oa::FnLoss::l1Bwd(const oa::Matrix& inA, const oa::Matrix& inB) {
	gLastLossName = "l1";
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);

	oa::U32 count = static_cast<oa::U32>(inA.numElements());

	oa::Matrix gradA = oa::FnMatrix::empty(inA.getShape(), inA.getDtype());

	struct { oa::U32 count; } push{.count = count};

	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "L1Bwd", {&inA, &inB, &gradA}, access, &push, sizeof(push), oa::divCeil(count, 256U));

	return commitLossResult(
		oa::move(gradA), lowering,
		oa::detail::opRegistry::FnLoss::l1Bwd, {&inA, &inB});
}

// ─── bce (Binary Cross-entropy) ────────────────────────────────────────

oa::Matrix oa::FnLoss::bce(const oa::Matrix& inA, const oa::Matrix& inB) {
	gLastLossName = "bce";
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);

	oa::U32 count = static_cast<oa::U32>(inA.numElements());

	oa::Matrix perElement = oa::FnMatrix::empty(inA.getShape(), inA.getDtype());

	struct { oa::U32 count; } push{.count = count};

	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "Bce", {&inA, &inB, &perElement}, access, &push, sizeof(push), oa::divCeil(count, 256U));

	oa::Matrix loss = oa::FnMatrix::mean(perElement);

	if (oa::FnAutograd::isEnabled() and inA.requiresGrad()) {
		auto gradFn = oa::makeShared<oa::GradBce>();
		gradFn->saveForBackward(inA, inB);
		gradFn->setGraphInputs(oa::Vec<oa::Matrix>{inA, inB});
		gradFn->sequenceNr_  = oa::FnAutograd::nextSeq();
		loss.mutAutograd().gradFn = gradFn;
	}

	return commitLossResult(
		oa::move(loss), lowering,
		oa::detail::opRegistry::FnLoss::bce, {&inA, &inB});
}

oa::Matrix oa::FnLoss::bceBwd(const oa::Matrix& inA, const oa::Matrix& inB) {
	gLastLossName = "bce";
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);

	oa::U32 count = static_cast<oa::U32>(inA.numElements());

	oa::Matrix gradA = oa::FnMatrix::empty(inA.getShape(), inA.getDtype());

	struct { oa::U32 count; } push{.count = count};

	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "BceBwd", {&inA, &inB, &gradA}, access, &push, sizeof(push), oa::divCeil(count, 256U));

	return commitLossResult(
		oa::move(gradA), lowering,
		oa::detail::opRegistry::FnLoss::bceBwd, {&inA, &inB});
}
