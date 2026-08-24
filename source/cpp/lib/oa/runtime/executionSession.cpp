#include "executionSession.h"
#include "engine/deviceAccess.h"
#include "engine/engineAccess.h"

#include <oa/core/envFlag.h>
#include <oa/core/log.h>
#include <oa/core/matrix.h>
#include <oa/core/matrixAccess.h>
#include <oa/runtime/executableGraph.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/engine/resourceAccess.h>
#include <oa/runtime/engine/allocatorAccess.h>
#include <oa/runtime/engine/bindlessAccess.h>
#include <oa/runtime/engine/submissionAccess.h>
#include <oa/runtime/stream.h>
#include "descriptorValidation.h"
#include "dispatchValidation.h"

#include <cassert>
#include <vulkan/vulkan.h>

static bool executionAccessReads(oa::BufferAccess inAccess) {
	return inAccess == oa::BufferAccess::Read or inAccess == oa::BufferAccess::ReadWrite;
}

static bool executionAccessWrites(oa::BufferAccess inAccess) {
	return inAccess == oa::BufferAccess::Write or inAccess == oa::BufferAccess::ReadWrite;
}

oa::ExecutionSession::ExecutionSession(oa::Engine* inEngine)
	: engine_(inEngine)
	, graph_(new oa::ExecutableGraph())
	, builder_(graph_)
{
	assert(engine_ and "Engine cannot be null");
}

oa::ExecutionSession::~ExecutionSession() {
	// Destruction never submits or waits. Unsubmitted primary recording is reset;
	// submitted graphs move to engine retirement when their exact event is still
	// incomplete.
	if (const auto status = abandon(); not status.isOk()) {
		OaLogError(oa::LogComponent::Compute,
			"execution session abandonment failed: %s",
			status.getMessage().cStr());
	}
	for (auto* graph : deferredGraphs_) {
		delete graph;
	}
	for (auto* graph : reusableGraphs_) {
		delete graph;
	}
	if (graph_) {
		delete graph_;
	}
}

oa::Status oa::ExecutionSession::record(const oa::ComputeDispatchDesc& inDesc) {
	if (not recordingFailure_.isOk()) return recordingFailure_;
	if (not engine_) {
		OaLogError(oa::LogComponent::Compute,
			"oa::ExecutionSession::record '%.*s': null engine",
			static_cast<int>(inDesc.kernel.size()), inDesc.kernel.data());
		return failRecording_(oa::Status::error(oa::StatusCode::Internal,
			"execution session record: null engine"));
	}
	if (not inDesc.indirect) {
		const auto dispatchStatus = oavk::validateDirectComputeDispatch(
			oa::EngineDeviceAccess::get(*engine_), inDesc.groupsX, inDesc.groupsY, inDesc.groupsZ);
		if (not dispatchStatus.isOk()) {
			return failRecording_(dispatchStatus);
		}
	}
	const auto descriptorStatus = oavk::validateStorageBufferDescriptors(
		oa::EngineDeviceAccess::get(*engine_),
		inDesc.buffers,
		true,
		oa::EngineAllocatorAccess::get(*engine_).allocator);
	if (not descriptorStatus.isOk()) {
		return failRecording_(descriptorStatus);
	}
	if (inDesc.indirect) {
		const auto indirectStatus = oavk::validateIndirectComputeDispatch(
			inDesc.indirectBuffer,
			inDesc.indirectOffset,
			oa::EngineAllocatorAccess::get(*engine_).allocator);
		if (not indirectStatus.isOk()) {
			return failRecording_(indirectStatus);
		}
	}
	const auto status = builder_.record(inDesc);
	if (not status.isOk()) return failRecording_(status);
	const oa::Usize accessCount = inDesc.access.size() < inDesc.buffers.size()
		? inDesc.access.size() : inDesc.buffers.size();
	for (oa::Usize index = 0; index < accessCount; ++index) {
		if (inDesc.access[index] != oa::BufferAccess::Read) {
			inDesc.buffers[index].markMutation();
		}
	}
	executed_ = false;
	return oa::Status::ok();
}

oa::Result<oa::U32> oa::ExecutionSession::findOrAddSemanticValue(
	const oa::Matrix& inMatrix,
	oa::OpValueKind inKind,
	oa::Bool inExternal)
{
	const auto& storage = oa::MatrixAccess::storageOwner(inMatrix);
	// Search newest-first. in-place operations create a new logical value for
	// the same storage range, and subsequent readers must observe that version.
	for (oa::Usize index = semanticValueBindings_.size(); index > 0; --index) {
		const auto& binding = semanticValueBindings_[index - 1];
		if (binding.storage.get() != storage.get()
			or binding.byteOffset != inMatrix.byteOffset()
			or binding.kind != inKind
			or binding.shape != inMatrix.getShape()
			or binding.dtype != inMatrix.getDtype())
		{
			continue;
		}
		oa::Bool strideMatches = true;
		for (oa::I32 dimension = 0; dimension < binding.shape.rank; ++dimension) {
			if (binding.strides[static_cast<oa::Usize>(dimension)]
				!= inMatrix.getStride().stepElements(dimension))
			{
				strideMatches = false;
				break;
			}
		}
		if (strideMatches) return binding.value;
	}

	oa::SemanticValueDesc desc;
	desc.kind = inKind;
	desc.shape = inMatrix.getShape();
	desc.dtype = inMatrix.getDtype();
	desc.external = inExternal;
	for (oa::I32 dimension = 0; dimension < desc.shape.rank; ++dimension) {
		desc.strides[static_cast<oa::Usize>(dimension)] =
			inMatrix.getStride().stepElements(dimension);
	}
	auto added = semanticGraph_.addValue(desc);
	if (not added.isOk()) return added.getStatus();

	SemanticValueBinding binding;
	binding.storage = storage;
	binding.byteOffset = inMatrix.byteOffset();
	binding.kind = inKind;
	binding.shape = inMatrix.getShape();
	binding.dtype = inMatrix.getDtype();
	binding.value = added.getValue();
	for (oa::I32 dimension = 0; dimension < binding.shape.rank; ++dimension) {
		binding.strides[static_cast<oa::Usize>(dimension)] =
			inMatrix.getStride().stepElements(dimension);
	}
	semanticValueBindings_.pushBack(binding);
	return added.getValue();
}

oa::Result<oa::U32> oa::ExecutionSession::addSemanticOutputValue(
	const oa::Matrix& inMatrix,
	oa::OpValueKind inKind)
{
	oa::SemanticValueDesc desc;
	desc.kind = inKind;
	desc.shape = inMatrix.getShape();
	desc.dtype = inMatrix.getDtype();
	desc.external = false;
	for (oa::I32 dimension = 0; dimension < desc.shape.rank; ++dimension) {
		desc.strides[static_cast<oa::Usize>(dimension)] =
			inMatrix.getStride().stepElements(dimension);
	}
	auto added = semanticGraph_.addValue(desc);
	if (not added.isOk()) return added.getStatus();

	SemanticValueBinding binding;
	binding.storage = oa::MatrixAccess::storageOwner(inMatrix);
	binding.byteOffset = inMatrix.byteOffset();
	binding.kind = inKind;
	binding.shape = inMatrix.getShape();
	binding.dtype = inMatrix.getDtype();
	binding.value = added.getValue();
	for (oa::I32 dimension = 0; dimension < binding.shape.rank; ++dimension) {
		binding.strides[static_cast<oa::Usize>(dimension)] =
			inMatrix.getStride().stepElements(dimension);
	}
	semanticValueBindings_.pushBack(binding);
	return added.getValue();
}

oa::Result<oa::U32> oa::ExecutionSession::recordOp(
	const oa::OpContract& inContract,
	oa::Span<const oa::Matrix* const> inInputs,
	oa::Span<const oa::Matrix* const> inOutputs,
	oa::Span<const oa::OpAttribute> inAttributes)
{
	if (not recordingFailure_.isOk()) return recordingFailure_;
	if (opLoweringDepth_ > 0U) {
		return oa::invalidSemanticOpId;
	}
	const auto fail = [this](const oa::Status& inFailure)
		-> oa::Result<oa::U32>
	{
		return failRecording_(inFailure);
	};
	constexpr oa::U8 MaxPackedValueKinds =
		static_cast<oa::U8>(sizeof(oa::U32) * 2U);
	if (not inContract.acceptsInputCount(inInputs.size())
		or not inContract.acceptsOutputCount(inOutputs.size()))
	{
		return fail(oa::Status::error(oa::StatusCode::InvalidArgument,
			"execution session semantic operation arity mismatch"));
	}
	if (inContract.inputCount > MaxPackedValueKinds
		or inContract.outputCount > MaxPackedValueKinds)
	{
		return fail(oa::Status::error(oa::StatusCode::OutOfRange,
			"execution session semantic operation exceeds packed kind capacity"));
	}
	if (
		(inContract.hasVariadicInputs()
			!= (inContract.minimumVariadicInputCount > 0U))
		or (inContract.hasVariadicOutputs()
			!= (inContract.minimumVariadicOutputCount > 0U))
	)
	{
		return fail(oa::Status::error(oa::StatusCode::InvalidArgument,
			"execution session semantic operation has an invalid variadic descriptor"));
	}
	if (inContract.name.empty() or inContract.hash == 0U) {
		return fail(oa::Status::error(oa::StatusCode::InvalidArgument,
			"execution session semantic operation requires a named contract"));
	}
	oa::Vec<oa::U32> inputs;
	oa::Vec<oa::U32> outputs;
	inputs.reserve(inInputs.size());
	outputs.reserve(inOutputs.size());
	for (oa::U32 index = 0; index < inInputs.size(); ++index) {
		const auto* matrix = inInputs[index];
		if (not matrix or not matrix->hasStorage()) {
			if (inContract.isInputOptional(index)) {
				inputs.pushBack(oa::invalidSemanticValueId);
				continue;
			}
			OaLogError(oa::LogComponent::Compute,
				"Semantic operation '%.*s' input %u has no storage",
				static_cast<int>(inContract.name.size()), inContract.name.data(),
				index);
			return fail(oa::Status::error(oa::StatusCode::InvalidArgument,
				"execution session semantic operation has an invalid input matrix"));
		}
		const auto kind = inContract.inputKindAt(index);
		auto value = findOrAddSemanticValue(*matrix, kind, true);
		if (not value.isOk()) return fail(value.getStatus());
		inputs.pushBack(value.getValue());
	}
	for (oa::U32 index = 0; index < inOutputs.size(); ++index) {
		const auto* matrix = inOutputs[index];
		if (not matrix or not matrix->hasStorage()) {
			return fail(oa::Status::error(oa::StatusCode::InvalidArgument,
				"execution session semantic operation has an invalid output matrix"));
		}
		// Every semantic output is a new SSA version, including in-place writes
		// that intentionally reuse the same physical storage as an input.
		const auto kind = inContract.outputKindAt(index);
		auto value = addSemanticOutputValue(*matrix, kind);
		if (not value.isOk()) return fail(value.getStatus());
		outputs.pushBack(value.getValue());
	}
	auto operation = semanticGraph_.addOp(
		inContract,
		oa::Span<const oa::U32>(inputs.data(), inputs.size()),
		oa::Span<const oa::U32>(outputs.data(), outputs.size()),
		{}, inAttributes);
	if (not operation.isOk()) return fail(operation.getStatus());
	executed_ = false;
	return operation;
}

void oa::ExecutionSession::beginOpLowering() noexcept {
	++opLoweringDepth_;
}

oa::Result<oa::U32> oa::ExecutionSession::finishOpLowering(
	oa::U32 inFirstNode,
	const oa::OpContract& inContract,
	oa::Span<const oa::Matrix* const> inInputs,
	oa::Span<const oa::Matrix* const> inOutputs,
	oa::Span<const oa::OpAttribute> inAttributes)
{
	if (opLoweringDepth_ == 0U) {
		return failRecording_(oa::Status::error(oa::StatusCode::FailedPrecondition,
			"operation lowering scope commit without a matching begin"));
	}
	--opLoweringDepth_;
	if (opLoweringDepth_ > 0U) {
		return oa::invalidSemanticOpId;
	}
	if (not graph_ or inFirstNode > graph_->nodeCount()) {
		return failRecording_(oa::Status::error(oa::StatusCode::OutOfRange,
			"operation lowering scope node boundary is invalid"));
	}
	// A validation failure or an intentional identity/no-op emits no executable
	// work and therefore owns no semantic operation.
	if (inFirstNode == graph_->nodeCount()) {
		return oa::invalidSemanticOpId;
	}

	auto semantic = recordOp(
		inContract, inInputs, inOutputs, inAttributes);
	if (not semantic.isOk()) return semantic.getStatus();
	auto nodes = graph_->nodes();
	for (oa::U32 index = inFirstNode; index < nodes.size(); ++index) {
		auto& node = nodes[index];
		node.operation = oa::String(inContract.name);
		node.opContractHash = inContract.hash;
		node.semanticOps.clear();
		node.semanticOps.pushBack(semantic.getValue());
	}
	executed_ = false;
	return semantic.getValue();
}

void oa::ExecutionSession::cancelOpLowering(oa::U32 inFirstNode) noexcept {
	if (opLoweringDepth_ == 0U) return;
	--opLoweringDepth_;
	if (opLoweringDepth_ > 0U or not graph_
		or inFirstNode == graph_->nodeCount())
	{
		return;
	}
	(void)failRecording_(oa::Status::error(oa::StatusCode::FailedPrecondition,
		"operation lowering scope was abandoned after emitting executable work"));
}

oa::Status oa::ExecutionSession::recordView(
	const oa::Matrix& inSource,
	const oa::Matrix& inView)
{
	if (not recordingFailure_.isOk()) return recordingFailure_;
	const auto fail = [this](const oa::Status& inFailure) {
		return failRecording_(inFailure);
	};
	if (not inSource.hasStorage() or not inView.hasStorage()) {
		return fail(oa::Status::error(oa::StatusCode::InvalidArgument,
			"execution session semantic view requires stored matrices"));
	}
	if (oa::MatrixAccess::storageOwner(inSource).get()
		!= oa::MatrixAccess::storageOwner(inView).get()
		or inSource.getDtype() != inView.getDtype())
	{
		return fail(oa::Status::error(oa::StatusCode::InvalidArgument,
			"execution session semantic view must share storage and dtype"));
	}
	const oa::U64 sourceOffset = inSource.byteOffset();
	const oa::U64 viewOffset = inView.byteOffset();
	const oa::I64 relativeOffset = viewOffset >= sourceOffset
		? static_cast<oa::I64>(viewOffset - sourceOffset)
		: -static_cast<oa::I64>(sourceOffset - viewOffset);

	auto source = findOrAddSemanticValue(
		inSource, oa::OpValueKind::Matrix, true);
	if (not source.isOk()) return fail(source.getStatus());
	oa::Bool sameMetadata = inSource.byteOffset() == inView.byteOffset()
		and inSource.getShape() == inView.getShape();
	if (sameMetadata) {
		for (oa::I32 dimension = 0;
			dimension < inSource.getShape().rank; ++dimension)
		{
			if (inSource.getStride().stepElements(dimension)
				!= inView.getStride().stepElements(dimension))
			{
				sameMetadata = false;
				break;
			}
		}
	}
	if (sameMetadata) return oa::Status::ok();

	// A metadata-changing view is a new logical SSA value even when its target
	// shape happens to match an older value for the same storage. Reusing that
	// older id would create backward provenance for round trips such as
	// unsqueeze().squeeze() and would erase the current view lineage.
	auto view = addSemanticOutputValue(
		inView, oa::OpValueKind::Matrix);
	if (not view.isOk()) return fail(view.getStatus());
	const auto status = semanticGraph_.addView(
		source.getValue(), view.getValue(), relativeOffset);
	if (not status.isOk()) return fail(status);
	executed_ = false;
	return oa::Status::ok();
}

oa::Status oa::ExecutionSession::snapshotSemanticBindings(
	oa::Span<const oa::Matrix* const> inObservedOutputs,
	oa::Vec<oa::SemanticStorageBinding>& outBindings,
	oa::Vec<oa::CapturedResourceDesc>& outResourceDescs,
	oa::Vec<oa::SharedPtr<oavk::Buffer>>& outResources) const
{
	outBindings.clear();
	outResourceDescs.clear();
	outResources.clear();
	if (semanticValueBindings_.size() != semanticGraph_.valueCount()) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"semantic storage snapshot requires one binding per value");
	}
	if (stableExternalResourceCount_ > stableResourceSlots_.size()) {
		return oa::Status::error(oa::StatusCode::Internal,
			"semantic storage snapshot has an invalid replay-input prefix");
	}

	outBindings.reserve(semanticValueBindings_.size());
	outResourceDescs.reserve(
		semanticValueBindings_.size() + inObservedOutputs.size());
	outResources.reserve(
		semanticValueBindings_.size() + inObservedOutputs.size());
	const auto isStableReplayInput = [&](const oa::SharedPtr<oavk::Buffer>& inStorage) {
		for (oa::Usize index = 0; index < stableExternalResourceCount_; ++index) {
			if (stableResourceSlots_[index].get() == inStorage.get()) return true;
		}
		return false;
	};
	const auto isStableTransient = [&](const oa::SharedPtr<oavk::Buffer>& inStorage) {
		const oa::Usize usedStableResources = stableResourceFrameActive_
			? stableResourceCursor_ : stableResourceCount_;
		for (oa::Usize index = stableExternalResourceCount_;
			index < usedStableResources; ++index)
		{
			if (stableResourceSlots_[index].get() == inStorage.get()) return true;
		}
		return false;
	};
	const auto findOrAddResource = [&](const oa::SharedPtr<oavk::Buffer>& inStorage) {
		for (oa::U32 index = 0; index < outResources.size(); ++index) {
			if (outResources[index].get() == inStorage.get()) return index;
		}
		const auto resource = static_cast<oa::U32>(outResources.size());
		outResources.pushBack(inStorage);
		oa::CapturedResourceDesc desc;
		desc.resource = resource;
		desc.stableReplayInput = isStableReplayInput(inStorage);
		desc.stableTransient = isStableTransient(inStorage);
		desc.placement = inStorage->placement;
		desc.byteSize = inStorage->size;
		outResourceDescs.pushBack(desc);
		return resource;
	};
	const auto fail = [&](oa::StatusCode inCode, const char* inMessage) {
		outBindings.clear();
		outResourceDescs.clear();
		outResources.clear();
		return oa::Status::error(inCode, inMessage);
	};
	for (const auto& binding : semanticValueBindings_) {
		const auto* value = semanticGraph_.findValue(binding.value);
		if (value == nullptr or not binding.storage) {
			return fail(oa::StatusCode::FailedPrecondition,
				"semantic storage snapshot contains an unbound value");
		}
		if (value->shape != binding.shape or value->dtype != binding.dtype) {
			return fail(oa::StatusCode::FailedPrecondition,
				"semantic storage snapshot metadata diverges from the semantic graph");
		}

		const auto resource = findOrAddResource(binding.storage);
		outResourceDescs[resource].semanticExternal |= value->external;
		oa::SemanticStorageBinding snapshot;
		snapshot.value = binding.value;
		snapshot.resource = resource;
		snapshot.byteOffset = binding.byteOffset;
		snapshot.shape = binding.shape;
		snapshot.strides = binding.strides;
		snapshot.dtype = binding.dtype;
		snapshot.semanticExternal = value->external;
		snapshot.stableReplayInput = outResourceDescs[resource].stableReplayInput;
		outBindings.pushBack(snapshot);
	}

	for (const auto* output : inObservedOutputs) {
		if (output == nullptr or not oa::MatrixAccess::storageOwner(*output)) {
			return fail(oa::StatusCode::InvalidArgument,
				"captured observed output must own allocated storage");
		}
		oa::Bool appearsInGraph = false;
		for (const auto& node : graph_->nodes()) {
			for (const auto& buffer : node.buffers) {
				if (buffer.buffer
					== oa::MatrixAccess::storageOwner(*output)->buffer)
				{
					appearsInGraph = true;
					break;
				}
			}
			if (appearsInGraph) break;
		}
		if (not appearsInGraph) {
			return fail(oa::StatusCode::InvalidArgument,
				"captured observed output is not used by the recorded graph");
		}
		const auto resource = findOrAddResource(
			oa::MatrixAccess::storageOwner(*output));
		outResourceDescs[resource].observedOutput = true;
		for (auto& binding : outBindings) {
			if (binding.resource == resource) binding.observedOutput = true;
		}
	}

	// Schema migration is incomplete, so append every strongly owned executable
	// resource not already reached through a semantic value or observed output.
	// first graph appearance makes the IDs deterministic; stable-transient state
	// remains the positive eligibility proof for compatibility-only resources.
	for (const auto& node : graph_->nodes()) {
		for (const auto& owner : node.bufferOwners) {
			if (owner) (void)findOrAddResource(owner);
		}
	}

	const auto lifetimes = graph_->computeLifetimes();
	for (auto& resource : outResourceDescs) {
		const auto& owner = outResources[resource.resource];
		const oa::Usize usedStableResources = stableResourceFrameActive_
			? stableResourceCursor_ : stableResourceCount_;
		for (oa::Usize index = 0; index < usedStableResources; ++index) {
			if (stableResourceSlots_[index].get() == owner.get()) {
				++resource.captureRetainedOwnerCount;
			}
		}
		for (const auto& binding : semanticValueBindings_) {
			if (binding.storage.get() == owner.get()) {
				++resource.captureRetainedOwnerCount;
			}
		}
		for (const auto& node : graph_->nodes()) {
			for (const auto& nodeOwner : node.bufferOwners) {
				if (nodeOwner.get() == owner.get()) {
					++resource.captureRetainedOwnerCount;
				}
			}
		}
		for (const auto& lifetime : lifetimes) {
			if (lifetime.buffer != owner->buffer) continue;
			resource.hasLifetime = true;
			resource.byteSize = lifetime.size;
			resource.firstAccess = lifetime.firstAccess;
			resource.lastAccess = lifetime.lastAccess;
			break;
		}
		resource.aliasCandidate = resource.stableTransient
			and resource.hasLifetime and not resource.isExternallyLive();
	}
	return oa::Status::ok();
}

void oa::ExecutionSession::releaseStableTransientResources(
	oa::Span<void*> inRetiredHandles)
{
	const oa::Usize used = stableResourceFrameActive_
		? stableResourceCursor_ : stableResourceCount_;
	for (oa::Usize index = stableExternalResourceCount_; index < used; ++index) {
		auto& slot = stableResourceSlots_[index];
		if (not slot) continue;
		for (void* handle : inRetiredHandles) {
			if (slot->buffer == handle) {
				slot->flags |= OA_VK_BUFFER_FLAG_TRANSIENT;
				break;
			}
		}
	}
	if (stableResourceSlots_.size() > stableExternalResourceCount_) {
		stableResourceSlots_.resize(stableExternalResourceCount_);
	}
	stableResourceCount_ = stableExternalResourceCount_;
	if (stableResourceFrameActive_) {
		stableResourceCursor_ = stableExternalResourceCount_;
	}
}

void oa::ExecutionSession::clearSemanticRecording() noexcept {
	semanticGraph_.reset();
	semanticValueBindings_.clear();
}

void oa::ExecutionSession::markExecuted() noexcept {
	executed_ = true;
	clearSemanticRecording();
}

void oa::ExecutionSession::discardActiveRecording() {
	assert(graph_ and "graph is null");
	if (engine_) graph_->reset(*engine_);
	else graph_->reset();
	executed_ = true;
	clearSemanticRecording();
	recordingFailure_ = oa::Status::ok();
	opLoweringDepth_ = 0U;
}

oa::U32 oa::ExecutionSession::nodeCount() const noexcept {
	assert(graph_ and "graph is null");
	return graph_->nodeCount();
}

oa::Bool oa::ExecutionSession::hasUnexecutedWork() const noexcept {
	return not executed_ and (
		(graph_ and graph_->nodeCount() > 0)
		or semanticGraph_.operationCount() > 0);
}

oa::Status oa::ExecutionSession::validateLowering() const {
	if (not recordingFailure_.isOk()) return recordingFailure_;
	if (not graph_) {
		return oa::Status::error(oa::StatusCode::Internal,
			"execution session lowering validation has no executable graph");
	}
	return oa::validateSemanticLowering(semanticGraph_, *graph_);
}

oa::Status oa::ExecutionSession::rejectRecording(const oa::Status& inFailure) {
	return failRecording_(inFailure);
}

oa::Status oa::ExecutionSession::consumeRecordingFailure() {
	if (recordingFailure_.isOk()) return oa::Status::ok();
	oa::Status failure = recordingFailure_;
	recordingFailure_ = oa::Status::ok();
	return failure;
}

oa::Status oa::ExecutionSession::failRecording_(const oa::Status& inFailure) {
	if (inFailure.isOk()) return oa::Status::ok();
	if (recordingFailure_.isOk()) {
		recordingFailure_ = inFailure;
		OaLogError(oa::LogComponent::Compute,
			"execution recording rejected: %s",
			inFailure.getMessage().cStr());
	}

	// Recording is one transaction across its semantic and executable forms.
	// The first authoring error rolls both back immediately. If earlier graphs
	// were encoded into an unsubmitted batch, reset that primary command buffer
	// too so a later submit/Sync cannot execute only a prefix of the workload.
	if (graph_) {
		if (engine_) graph_->reset(*engine_);
		else graph_->reset();
	}
	executed_ = true;
	clearSemanticRecording();
	if (activeBatchStream_ != nullptr) {
		const auto cancelStatus = cancelActiveBatch_();
		if (not cancelStatus.isOk()) {
			OaLogError(oa::LogComponent::Compute,
				"execution session failed to cancel aborted recording batch: %s",
				cancelStatus.getMessage().cStr());
		}
	}
	return recordingFailure_;
}

oa::ExecutionSession::BatchBufferState* oa::ExecutionSession::findBatchBufferState(
	oa::Vec<BatchBufferState>& inStates, const oavk::Buffer& inBuffer)
{
	for (auto& state : inStates) {
		if (state.buffer.synchronizationIdentity()
			== inBuffer.synchronizationIdentity()) return &state;
	}
	return nullptr;
}

void oa::ExecutionSession::mergeBatchBufferState(
	oa::Vec<BatchBufferState>& inStates,
	const oavk::Buffer& inBuffer,
	oa::Bool inRead,
	oa::Bool inWrite,
	oa::Bool inIndirectRead)
{
	if (not inBuffer.buffer) return;
	auto* state = findBatchBufferState(inStates, inBuffer);
	if (not state) {
		BatchBufferState value;
		value.buffer = inBuffer;
		inStates.pushBack(value);
		state = &inStates.back();
	}
	state->read = state->read or inRead;
	state->write = state->write or inWrite;
	state->indirectRead = state->indirectRead or inIndirectRead;
}

oa::U32 oa::ExecutionSession::emitBatchBoundaryBarriers(
	void* inPrimaryCommandBuffer,
	const oa::ExecutableGraph& inIncoming)
{
	// Secondary command buffers do not create dependencies between themselves.
	// Carry exact buffer access state across graph boundaries so the primary
	// emits only real RAW/WAR/WAW barriers and leaves unrelated work independent.
	oa::Vec<BatchBufferState> incoming;
	for (const auto& node : inIncoming.nodes()) {
		for (oa::U32 i = 0; i < static_cast<oa::U32>(node.buffers.size()); ++i) {
			mergeBatchBufferState(
				incoming, node.buffers[i], executionAccessReads(node.access[i]),
				executionAccessWrites(node.access[i]), false);
		}
		if (node.indirect) {
			mergeBatchBufferState(incoming, node.indirectBuffer, true, false, true);
		}
	}

	oa::Vec<VkBufferMemoryBarrier2> barriers;
	oa::Vec<VkMemoryBarrier2> aliasBarriers;
	for (const auto& current : incoming) {
		auto* previous = findBatchBufferState(batchBufferStates_, current.buffer);
		if (not previous) continue;
		const bool hazard =
			(previous->write and (current.read or current.write))
			or (previous->read and current.write);
		if (not hazard) continue;

		VkPipelineStageFlags2 srcStages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
		if (previous->indirectRead) {
			srcStages |= VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
		}
		VkAccessFlags2 srcAccess = 0;
		if (previous->read) srcAccess |= VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
		if (previous->write) srcAccess |= VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
		if (previous->indirectRead) srcAccess |= VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
		VkPipelineStageFlags2 dstStages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
		if (current.indirectRead) {
			dstStages |= VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
		}
		VkAccessFlags2 dstAccess = 0;
		if (current.read) dstAccess |= VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
		if (current.write) dstAccess |= VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
		if (current.indirectRead) dstAccess |= VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
		if (previous->buffer.buffer != current.buffer.buffer) {
			VkMemoryBarrier2 barrier{};
			barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
			barrier.srcStageMask = srcStages;
			barrier.srcAccessMask = previous->write ? srcAccess : 0;
			barrier.dstStageMask = dstStages;
			barrier.dstAccessMask = previous->write ? dstAccess : 0;
			aliasBarriers.pushBack(barrier);
		} else {
			VkBufferMemoryBarrier2 barrier{};
			barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
			barrier.srcStageMask = srcStages;
			barrier.srcAccessMask = previous->write ? srcAccess : 0;
			barrier.dstStageMask = dstStages;
			barrier.dstAccessMask = previous->write ? dstAccess : 0;
			barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.buffer = static_cast<::VkBuffer>(current.buffer.buffer);
			barrier.offset = 0;
			barrier.size = VK_WHOLE_SIZE;
			barriers.pushBack(barrier);
		}
	}

	if (not barriers.empty() or not aliasBarriers.empty()) {
		VkDependencyInfo dependency{};
		dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
		dependency.memoryBarrierCount = static_cast<oa::U32>(aliasBarriers.size());
		dependency.pMemoryBarriers = aliasBarriers.data();
		dependency.bufferMemoryBarrierCount = static_cast<oa::U32>(barriers.size());
		dependency.pBufferMemoryBarriers = barriers.data();
		oa::EngineDeviceAccess::get(*engine_).deviceDispatch.vkCmdPipelineBarrier2(
			static_cast<VkCommandBuffer>(inPrimaryCommandBuffer), &dependency);
	}

	for (const auto& current : incoming) {
		auto* previous = findBatchBufferState(batchBufferStates_, current.buffer);
		if (previous) {
			*previous = current;
		} else {
			batchBufferStates_.pushBack(current);
		}
	}
	return static_cast<oa::U32>(barriers.size() + aliasBarriers.size());
}

oa::Result<oa::U32> oa::ExecutionSession::recordActiveGraphInBatch_(
	void* inPrimaryCommandBuffer)
{
	assert(graph_ and "graph is null");
	const auto previousStates = batchBufferStates_;
	const oa::U32 barrierCount = emitBatchBoundaryBarriers(
		inPrimaryCommandBuffer, *graph_);
	const auto status = graph_->recordReplay(*engine_, inPrimaryCommandBuffer);
	if (not status.isOk()) {
		batchBufferStates_ = previousStates;
		return status;
	}
	rotateAfterBatch();
	return barrierCount;
}

oa::Status oa::ExecutionSession::beginBatch_() {
	if (activeBatchStream_ != nullptr) return oa::Status::ok();
	if (pendingBatchStream_ != nullptr or pendingEvent_.isValid()) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"execution session requires wait() before beginning another batch");
	}
	if (engine_ == nullptr) {
		return oa::Status::error(oa::StatusCode::Internal,
			"execution session batch has no engine");
	}

	auto* stream = oa::EngineSubmissionAccess::acquireStream(*engine_);
	if (stream == nullptr) {
		return oa::Status::error(oa::StatusCode::VulkanError,
			"execution session failed to acquire a batch stream");
	}
	const auto beginStatus = stream->begin(oa::EngineDeviceAccess::get(*engine_));
	if (not beginStatus.isOk()) {
		oa::EngineSubmissionAccess::releaseStream(*engine_, stream);
		return beginStatus;
	}
	activeBatchStream_ = stream;
	return oa::Status::ok();
}

oa::Result<oa::Event> oa::ExecutionSession::submitBatch_() {
	if (activeBatchStream_ == nullptr) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"execution session has no recorded batch to submit");
	}

	auto* stream = activeBatchStream_;
	stream->recordHostReadbackBarrier();
	const auto submitStatus = stream->submit(*engine_);
	activeBatchStream_ = nullptr;
	clearBatchHazards();
	if (not submitStatus.isOk()) {
		const auto resetStatus = stream->resetUnsubmitted(oa::EngineDeviceAccess::get(*engine_));
		// A command buffer that failed to reset is quarantined in the engine-owned
		// pool until close; returning it to the free stack would permit unsafe reuse.
		if (resetStatus.isOk()) oa::EngineSubmissionAccess::releaseStream(*engine_, stream);
		reclaimCompletedGraphs();
		return not resetStatus.isOk() ? resetStatus : submitStatus;
	}

	const oa::Event completion = stream->completion(oa::EngineDeviceAccess::get(*engine_));
	pendingBatchStream_ = stream;
	pendingEvent_ = completion;
	if (not completion.isValid()) {
		return oa::Status::error(oa::StatusCode::Internal,
			"execution session submitted a batch without a valid completion event");
	}
	return completion;
}

oa::Status oa::ExecutionSession::completePendingBatch_() {
	if (pendingBatchStream_ == nullptr) {
		pendingEvent_ = {};
		return oa::Status::ok();
	}
	OA_RETURN_IF_ERROR(pendingBatchStream_->synchronize(oa::EngineDeviceAccess::get(*engine_)));
	// completion makes resource destruction legal, but the executable command
	// buffer still retains descriptor-set references until it is reset. detach
	// those references before graphs recycle or destroy their descriptor pools.
	OA_RETURN_IF_ERROR(
		pendingBatchStream_->resetUnsubmitted(oa::EngineDeviceAccess::get(*engine_)));
	oa::EngineSubmissionAccess::releaseStream(*engine_, pendingBatchStream_);
	pendingBatchStream_ = nullptr;
	pendingEvent_ = {};
	reclaimCompletedGraphs();
	return oa::Status::ok();
}

oa::Status oa::ExecutionSession::waitBatch_(const oa::Event& inEvent) {
	if (not inEvent.isValid()) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"execution session cannot wait on an invalid event");
	}
	if (pendingBatchStream_ == nullptr or not pendingEvent_.isValid()) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"execution session has no pending batch event");
	}
	if (not pendingEvent_.isSameCompletion(inEvent)) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"event does not belong to this execution session's pending batch");
	}
	OA_RETURN_IF_ERROR(inEvent.wait());
	return completePendingBatch_();
}

oa::Status oa::ExecutionSession::cancelActiveBatch_() {
	if (activeBatchStream_ == nullptr) return oa::Status::ok();
	auto* stream = activeBatchStream_;
	activeBatchStream_ = nullptr;
	const auto resetStatus = stream->resetUnsubmitted(oa::EngineDeviceAccess::get(*engine_));
	if (resetStatus.isOk()) oa::EngineSubmissionAccess::releaseStream(*engine_, stream);
	clearBatchHazards();
	reclaimCompletedGraphs();
	return resetStatus;
}

void oa::ExecutionSession::retirePendingBatch_() {
	if (pendingBatchStream_ == nullptr) {
		pendingEvent_ = {};
		return;
	}
	oa::Vec<oa::UniquePtr<oa::ExecutableGraph>> graphs;
	for (auto* graph : deferredGraphs_) {
		graphs.pushBack(oa::UniquePtr<oa::ExecutableGraph>(graph));
	}
	deferredGraphs_.clear();
	oa::EngineAccess(*engine_).retireSessionBatch(
		pendingBatchStream_, pendingEvent_, oa::move(graphs));
	pendingBatchStream_ = nullptr;
	pendingEvent_ = {};
	clearBatchHazards();
}

oa::Status oa::ExecutionSession::abandon() {
	oa::Status result = cancelActiveBatch_();
	if (pendingBatchStream_ != nullptr) {
		if (pendingEvent_.isValid() and pendingEvent_.isComplete()) {
			// The poll above is the completion proof. Destruction must not call a
			// wait path, even one with a fast-poll optimization.
			pendingBatchStream_->submitted = false;
			const auto resetStatus =
				pendingBatchStream_->resetUnsubmitted(oa::EngineDeviceAccess::get(*engine_));
			if (resetStatus.isOk()) {
				oa::EngineSubmissionAccess::releaseStream(*engine_, pendingBatchStream_);
				pendingBatchStream_ = nullptr;
				pendingEvent_ = {};
				reclaimCompletedGraphs();
			} else {
				if (result.isOk()) result = resetStatus;
				retirePendingBatch_();
			}
		} else {
			retirePendingBatch_();
		}
	}
	return result;
}

void oa::ExecutionSession::rotateAfterBatch() {
	assert(graph_ and "graph is null");
	deferredGraphs_.pushBack(graph_);
	if (not reusableGraphs_.empty()) {
		graph_ = reusableGraphs_.back();
		reusableGraphs_.popBack();
	} else {
		graph_ = new oa::ExecutableGraph();
	}
	builder_.attach(graph_);
	executed_ = true;
	clearSemanticRecording();
	recordingFailure_ = oa::Status::ok();
}

void oa::ExecutionSession::reclaimCompletedGraphs() {
	if (deferredGraphs_.empty()) return;

	for (auto* graph : deferredGraphs_) {
		graph->clearNodes();
		graph->releaseCompletedBufferOwners();
	}
	if (graph_->nodeCount() == 0) {
		delete graph_;
		graph_ = deferredGraphs_[0];
		builder_.attach(graph_);
		for (oa::Usize i = deferredGraphs_.size(); i > 1; --i) {
			reusableGraphs_.pushBack(deferredGraphs_[i - 1]);
		}
	} else {
		for (oa::Usize i = deferredGraphs_.size(); i > 0; --i) {
			reusableGraphs_.pushBack(deferredGraphs_[i - 1]);
		}
	}
	deferredGraphs_.clear();
	batchBufferStates_.clear();
}

void oa::ExecutionSession::clear() {
	assert(graph_ and "graph is null");
	if (engine_) {
		graph_->clearNodes();
	} else {
		graph_->reset();
	}
	executed_ = false;
	clearSemanticRecording();
	recordingFailure_ = oa::Status::ok();
	opLoweringDepth_ = 0U;
	if (not isBatchActive()) {
		clearBatchHazards();
		resetStats();
	}
}

void oa::ExecutionSession::beginStableResourceFrame() {
	assert(not stableResourceFrameActive_
		and "stable resource frames cannot be nested");
	stableResourceCursor_ = 0;
	stableResourceFrameActive_ = true;
	stableResourceInputsSealed_ = false;
}

void oa::ExecutionSession::endStableResourceFrame() noexcept {
	if (stableResourceInputsSealed_) {
		stableResourceCount_ = stableResourceCursor_;
	}
	stableResourceCursor_ = 0;
	stableResourceFrameActive_ = false;
	stableResourceInputsSealed_ = false;
}

void oa::ExecutionSession::sealStableResourceInputs() {
	assert(stableResourceFrameActive_
		and "stable inputs require an active resource frame");
	assert(not stableResourceInputsSealed_
		and "stable resource inputs may only be sealed once per frame");
	stableExternalResourceCount_ = stableResourceCursor_;
	stableResourceInputsSealed_ = true;
}

void oa::ExecutionSession::sealAllStableResourcesExternal() {
	assert(stableResourceFrameActive_
		and "stable resources require an active resource frame");
	stableExternalResourceCount_ = stableResourceCursor_;
	stableResourceInputsSealed_ = true;
}

oa::SharedPtr<oavk::Buffer> oa::ExecutionSession::allocateMatrixBuffer(
	oa::U64 inBytes, oa::MemoryPlacement inPlacement)
{
	if (not engine_ or inBytes == 0) return {};
	static const oa::Bool logStableResourceMisses =
		oa::EnvFlag::isSet("OA_LOG_STABLE_RESOURCE_MISSES");

	const auto allocate = [&]() -> oa::SharedPtr<oavk::Buffer> {
		auto result = oa::EngineResourceAccess::allocBuffer(*engine_, inBytes, inPlacement);
		if (not result) return {};
		return oa::EngineAccess(*engine_).adoptBufferLease(std::move(*result));
	};

	if (not stableResourceFrameActive_) return allocate();

	const oa::Usize slot = stableResourceCursor_++;
	if (slot < stableResourceSlots_.size()) {
		auto& existing = stableResourceSlots_[slot];
		// Stable frames deliberately reuse storage by allocation ordinal. A
		// retained matrix object does not change the slot's identity.
		const oa::MemoryPlacement resolved = inPlacement == oa::MemoryPlacement::Auto
			? oa::EngineResourceAccess::defaultMatrixPlacement(*engine_)
			: inPlacement;
		if (existing and existing->capacity >= inBytes
			and existing->placement == resolved)
		{
			const oa::U64 previousSize = existing->size;
			existing->size = inBytes;
			if (const auto status = oa::EngineBindlessAccess::updateBufferDescriptor(*engine_, *existing);
				not status.isOk())
			{
				existing->size = previousSize;
				return {};
			}
			return existing;
		}
		if (logStableResourceMisses) {
			OaLogInfo(oa::LogComponent::Compute,
				"Stable resource slot %zu replaced: %llu -> %llu bytes",
				slot,
				static_cast<unsigned long long>(existing ? existing->size : 0),
				static_cast<unsigned long long>(inBytes));
		}
		existing = allocate();
		return existing;
	}

	auto buffer = allocate();
	if (logStableResourceMisses) {
		OaLogInfo(oa::LogComponent::Compute,
			"Stable resource slot %zu created: %llu bytes", slot,
			static_cast<unsigned long long>(inBytes));
	}
	stableResourceSlots_.pushBack(buffer);
	return buffer;
}
