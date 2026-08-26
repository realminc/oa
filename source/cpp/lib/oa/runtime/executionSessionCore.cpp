#include "executionSession.h"
#include "engine/engineAccess.h"

#include <oa/core/matrix.h>
#include <oa/core/matrixAccess.h>
#include <oa/runtime/dispatch.h>
#include <oa/runtime/engine.h>

#include <assert.h>

namespace {

thread_local oa::ExecutionSession* activeSession = nullptr;

} // namespace

void oa::ExecutionSession::setActive(oa::ExecutionSession* inSession) noexcept {
	activeSession = inSession;
}

oa::ExecutionSession* oa::ExecutionSession::getActivePtr() noexcept {
	return activeSession;
}

oa::ExecutionSession& oa::ExecutionSession::getActive() {
	assert(activeSession
		and "no active execution session; initialize an engine or select one");
	return *activeSession;
}

oa::ExecutionSession& oa::ExecutionSession::forEngine(
	oa::Engine& inEngine) noexcept
{
	if (activeSession != nullptr and &activeSession->engine() == &inEngine) {
		return *activeSession;
	}
	auto& impl = oa::EngineAccess::get(inEngine);
	assert(impl.session_ and "engine has no execution session");
	return *impl.session_;
}

const oa::ExecutionSession& oa::ExecutionSession::forEngine(
	const oa::Engine& inEngine) noexcept
{
	if (activeSession != nullptr and &activeSession->engine() == &inEngine) {
		return *activeSession;
	}
	const auto& impl = oa::EngineAccess::get(inEngine);
	assert(impl.session_ and "engine has no execution session");
	return *impl.session_;
}

oa::ExecutionSession::RecordingScope::RecordingScope(
	oa::ExecutionSession& inSession) noexcept
	: previous_(oa::ExecutionSession::getActivePtr())
{
	oa::ExecutionSession::setActive(&inSession);
}

oa::ExecutionSession::RecordingScope::~RecordingScope() {
	oa::ExecutionSession::setActive(previous_);
}

oa::Engine& oa::ExecutionSession::engine() const noexcept {
	assert(engine_ and "execution session has no engine");
	return *engine_;
}

oa::ScalarType oa::ExecutionSession::weightDtype() const noexcept {
	return oa::precisionDtype(engine().getPrecision());
}

oa::U32 oa::ExecutionSession::subgroupSize() const noexcept {
	return oa::EngineAccess::get(engine()).device_.info.hardware.subgroupSize;
}

oa::Status oa::ExecutionSession::record(
	const oa::MatrixDispatchDesc& inDesc)
{
	if (not inDesc.dispatch.buffers.empty()
		or not inDesc.dispatch.bufferOwners.empty()
		or inDesc.dispatch.indirect
		or inDesc.dispatch.indirectBuffer.buffer
		or inDesc.dispatch.indirectOffset != 0)
	{
		return rejectRecording(oa::Status::error(
			oa::StatusCode::InvalidArgument,
			"matrix dispatch record: raw buffer and indirect fields must be empty"));
	}

	oa::Vec<oavk::Buffer> buffers;
	oa::Vec<oa::SharedPtr<oavk::Buffer>> owners;
	buffers.reserve(inDesc.matrices.size());
	owners.reserve(inDesc.matrices.size());

	oa::U32 dtype = 0;
	oa::Bool retainsIndirectArgs = false;
	for (const oa::Matrix* matrix : inDesc.matrices) {
		if (matrix) {
			const oa::ScalarType scalarType = matrix->getDtype();
			if (scalarType == oa::ScalarType::Float16) {
				return rejectRecording(oa::Status::error(
					oa::StatusCode::DtypeMismatch,
					"Float16 storage has no generic OA dispatch ABI; use FP32/BF16 or an explicitly packed kernel"));
			}
			if (scalarType == oa::ScalarType::Float64) {
				return rejectRecording(oa::Status::error(
					oa::StatusCode::Unimplemented,
					"Float64 dispatch requires an admitted FP64 kernel route; FP32 substitution is forbidden"));
			}
			if (scalarType == oa::ScalarType::BFloat16) dtype = 1;
			if (matrix == inDesc.indirectArgs) retainsIndirectArgs = true;
		}
		if (matrix and oa::MatrixAccess::storageOwner(*matrix)) {
			buffers.pushBack(oa::MatrixAccess::descriptor(*matrix));
			owners.pushBack(oa::MatrixAccess::storageOwner(*matrix));
		} else {
			buffers.pushBack(oavk::Buffer{});
			owners.pushBack({});
		}
	}

	if (inDesc.indirectArgs
		and (not oa::MatrixAccess::storageOwner(*inDesc.indirectArgs)
			or not retainsIndirectArgs))
	{
		return rejectRecording(oa::Status::error(
			oa::StatusCode::InvalidArgument,
			"matrix dispatch record: indirect arguments must have storage and be retained in matrices"));
	}
	if (not inDesc.indirectArgs and inDesc.indirectOffset != 0) {
		return rejectRecording(oa::Status::error(
			oa::StatusCode::InvalidArgument,
			"matrix dispatch record: indirect offset without indirect arguments"));
	}

	oa::ComputeDispatchDesc dispatch = inDesc.dispatch;
	dispatch.buffers = buffers.span();
	dispatch.bufferOwners = owners.span();
	dispatch.dtype = dtype;
	if (inDesc.indirectArgs) {
		dispatch.indirectBuffer =
			oa::MatrixAccess::descriptor(*inDesc.indirectArgs);
		dispatch.indirectOffset = inDesc.indirectOffset;
		dispatch.indirect = true;
	}
	return record(dispatch);
}

oa::Result<oa::U32> oa::ExecutionSession::recordOp(
	const oa::OpContract& inContract,
	oa::MatrixArgs inInputs,
	oa::MatrixArgs inOutputs,
	oa::OpAttributeArgs inAttributes)
{
	return recordOp(inContract, inInputs.span(), inOutputs.span(), inAttributes.span());
}

oa::OpLoweringScope::OpLoweringScope(oa::ExecutionSession& inSession)
	: session_(&inSession)
	, firstNode_(inSession.nodeCount())
	, active_(true)
{
	session_->beginOpLowering();
}

oa::OpLoweringScope::~OpLoweringScope() {
	if (active_ and session_) session_->cancelOpLowering(firstNode_);
}

oa::Status oa::OpLoweringScope::commit(
	const oa::OpContract& inContract,
	oa::Span<const oa::Matrix* const> inInputs,
	oa::Span<const oa::Matrix* const> inOutputs,
	oa::Span<const oa::OpAttribute> inAttributes)
{
	auto op = commitWithId(inContract, inInputs, inOutputs, inAttributes);
	return op.isOk() ? oa::Status::ok() : op.getStatus();
}

oa::Status oa::OpLoweringScope::commit(
	const oa::OpContract& inContract,
	oa::MatrixArgs inInputs,
	oa::MatrixArgs inOutputs,
	oa::OpAttributeArgs inAttributes)
{
	return commit(inContract, inInputs.span(), inOutputs.span(), inAttributes.span());
}

oa::Result<oa::U32> oa::OpLoweringScope::commitWithId(
	const oa::OpContract& inContract,
	oa::Span<const oa::Matrix* const> inInputs,
	oa::Span<const oa::Matrix* const> inOutputs,
	oa::Span<const oa::OpAttribute> inAttributes)
{
	if (not active_ or not session_) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"operation lowering scope was already completed");
	}
	active_ = false;
	return session_->finishOpLowering(
		firstNode_, inContract, inInputs, inOutputs, inAttributes);
}

oa::Result<oa::U32> oa::OpLoweringScope::commitWithId(
	const oa::OpContract& inContract,
	oa::MatrixArgs inInputs,
	oa::MatrixArgs inOutputs,
	oa::OpAttributeArgs inAttributes)
{
	return commitWithId(inContract, inInputs.span(), inOutputs.span(), inAttributes.span());
}

void oa::ExecutionSession::add(
	oa::StringView inKernelName,
	oa::Span<oavk::Buffer> inBuffers,
	oa::Span<oa::SharedPtr<oavk::Buffer>> inBufferOwners,
	oa::Span<oa::BufferAccess> inAccess,
	const void* inPush,
	oa::U32 inPushSize,
	oa::U32 inGroupsX,
	oa::U32 inGroupsY,
	oa::U32 inGroupsZ,
	oa::StringView inOperation,
	oa::U64 inImplementationId,
	oa::U64 inOpContractHash,
	oa::U64 inKernelContentHash,
	oa::U64 inProblemContractHash,
	oa::U32 inSemanticOp)
{
	oa::ComputeDispatchDesc desc;
	desc.operation = inOperation;
	if (inSemanticOp != oa::invalidSemanticOpId) {
		desc.semanticOps = oa::Span<const oa::U32>(&inSemanticOp, 1U);
	}
	desc.implementationId = inImplementationId;
	desc.opContractHash = inOpContractHash;
	desc.problemContractHash = inProblemContractHash;
	desc.kernelContentHash = inKernelContentHash;
	desc.kernel = inKernelName;
	desc.buffers = inBuffers;
	desc.bufferOwners = inBufferOwners;
	desc.access = inAccess;
	desc.pushData = inPush;
	desc.pushSize = inPushSize;
	desc.groupsX = inGroupsX;
	desc.groupsY = inGroupsY;
	desc.groupsZ = inGroupsZ;
	(void)record(desc);
}

void oa::ExecutionSession::add(
	oa::StringView inKernelName,
	oa::Span<oavk::Buffer> inBuffers,
	oa::Span<oa::BufferAccess> inAccess,
	const void* inPush,
	oa::U32 inPushSize,
	oa::U32 inGroupsX,
	oa::U32 inGroupsY,
	oa::U32 inGroupsZ,
	oa::StringView inOperation,
	oa::U64 inImplementationId,
	oa::U64 inOpContractHash,
	oa::U64 inKernelContentHash,
	oa::U64 inProblemContractHash,
	oa::U32 inSemanticOp)
{
	add(
		inKernelName,
		inBuffers,
		oa::Span<oa::SharedPtr<oavk::Buffer>>{},
		inAccess,
		inPush,
		inPushSize,
		inGroupsX,
		inGroupsY,
		inGroupsZ,
		inOperation,
		inImplementationId,
		inOpContractHash,
		inKernelContentHash,
		inProblemContractHash,
		inSemanticOp);
}

void oa::ExecutionSession::add(
	oa::StringView inKernelName,
	oa::MatrixArgs inMatrices,
	oa::Span<oa::BufferAccess> inAccess,
	const void* inPush,
	oa::U32 inPushSize,
	oa::U32 inGroupsX,
	oa::U32 inGroupsY,
	oa::U32 inGroupsZ,
	oa::StringView inOperation,
	oa::U64 inImplementationId,
	oa::U64 inOpContractHash,
	oa::U64 inKernelContentHash,
	oa::U64 inProblemContractHash,
	oa::U32 inSemanticOp)
{
	oa::MatrixDispatchDesc desc;
	desc.dispatch.operation = inOperation;
	if (inSemanticOp != oa::invalidSemanticOpId) {
		desc.dispatch.semanticOps =
			oa::Span<const oa::U32>(&inSemanticOp, 1U);
	}
	desc.dispatch.implementationId = inImplementationId;
	desc.dispatch.opContractHash = inOpContractHash;
	desc.dispatch.problemContractHash = inProblemContractHash;
	desc.dispatch.kernelContentHash = inKernelContentHash;
	desc.dispatch.kernel = inKernelName;
	desc.dispatch.access = inAccess;
	desc.dispatch.pushData = inPush;
	desc.dispatch.pushSize = inPushSize;
	desc.dispatch.groupsX = inGroupsX;
	desc.dispatch.groupsY = inGroupsY;
	desc.dispatch.groupsZ = inGroupsZ;
	desc.matrices = inMatrices.span();
	(void)record(desc);
}
