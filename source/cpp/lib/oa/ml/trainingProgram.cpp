#include <oa/ml/trainingProgram.h>
#include <oa/core/jsonWriter.h>

#include <oa/core/fnMatrix.h>
#include <oa/core/matrixAccess.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/dnn.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/engine/resourceAccess.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/executionPlan.h>

namespace {

oa::Bool isFrozenRngKernel(oa::StringView inName) {
	return inName == "PhiloxUniform" or inName == "PhiloxNormal";
}

oa::Bool isHostSteppedOptimizerKernel(oa::StringView inName) {
	return inName == "Adam" or inName == "Adamw" or inName == "AdamwMany4";
}

oa::Bool isFrozenScheduleOptimizerKernel(oa::StringView inName) {
	return inName == "MuonNesterov" or inName == "MuonApply"
		or inName == "MuonVector";
}

const char* compilationStageName(oa::TrainingCompilationStage inStage) {
	switch (inStage) {
		case oa::TrainingCompilationStage::SemanticValidation:
			return "semantic_validation";
		case oa::TrainingCompilationStage::ReplaySafety:
			return "replay_safety";
		case oa::TrainingCompilationStage::Decomposition:
			return "decomposition";
		case oa::TrainingCompilationStage::Fusion: return "fusion";
		case oa::TrainingCompilationStage::Placement: return "placement";
		case oa::TrainingCompilationStage::Precision: return "precision";
		case oa::TrainingCompilationStage::KernelSelection:
			return "kernel_selection";
		case oa::TrainingCompilationStage::LoweringValidation:
			return "lowering_validation";
		case oa::TrainingCompilationStage::MemoryPlanning:
			return "memory_planning";
		case oa::TrainingCompilationStage::SynchronizationPlanning:
			return "synchronization_planning";
		case oa::TrainingCompilationStage::CommandRecording:
			return "command_recording";
	}
	return "unknown";
}

const char* compilationStateName(oa::TrainingCompilationState inState) {
	switch (inState) {
		case oa::TrainingCompilationState::NotRun: return "not_run";
		case oa::TrainingCompilationState::Inherited: return "inherited";
		case oa::TrainingCompilationState::Analyzed: return "analyzed";
		case oa::TrainingCompilationState::Applied: return "applied";
		case oa::TrainingCompilationState::Failed: return "failed";
	}
	return "unknown";
}

const char* memoryPlacementName(oa::MemoryPlacement inPlacement) {
	switch (inPlacement) {
		case oa::MemoryPlacement::Auto: return "auto";
		case oa::MemoryPlacement::DeviceLocal: return "device_local";
		case oa::MemoryPlacement::HostUpload: return "host_upload";
		case oa::MemoryPlacement::HostReadback: return "host_readback";
		case oa::MemoryPlacement::Unified: return "unified";
	}
	return "unknown";
}

void writeJsonString(oa::internal::JsonWriter& out, oa::StringView inValue) {
	out.writeString(inValue);
}

struct MemoryAnalysisSummary {
	oa::U32 candidateCount = 0;
	oa::U32 groupCount = 0;
	oa::U64 potentialSavings = 0;
};

MemoryAnalysisSummary analyzeMemory(
	oa::Span<const oa::CapturedResourceDesc> inResources)
{
	struct Group {
		oa::MemoryPlacement placement = oa::MemoryPlacement::Auto;
		oa::Vector<const oa::CapturedResourceDesc*> members;
		oa::U64 requiredBytes = 0;
		oa::U64 totalBytes = 0;
	};
	oa::Vector<Group> groups;
	MemoryAnalysisSummary summary;
	for (const auto& resource : inResources) {
		if (not resource.aliasCandidate) continue;
		++summary.candidateCount;
		oa::Bool placed = false;
		for (auto& group : groups) {
			if (group.placement != resource.placement) continue;
			oa::Bool overlaps = false;
			for (const auto* member : group.members) {
				if (resource.firstAccess <= member->lastAccess
					and resource.lastAccess >= member->firstAccess)
				{
					overlaps = true;
					break;
				}
			}
			if (overlaps) continue;
			group.members.pushBack(&resource);
			group.requiredBytes = oa::max(group.requiredBytes, resource.byteSize);
			group.totalBytes += resource.byteSize;
			placed = true;
			break;
		}
		if (not placed) {
			Group group;
			group.placement = resource.placement;
			group.members.pushBack(&resource);
			group.requiredBytes = resource.byteSize;
			group.totalBytes = resource.byteSize;
			groups.pushBack(oa::move(group));
		}
	}
	for (const auto& group : groups) {
		if (group.members.size() < 2U) continue;
		++summary.groupCount;
		summary.potentialSavings += group.totalBytes - group.requiredBytes;
	}
	return summary;
}

} // namespace

oa::TrainingProgram::TrainingProgram()
	: plan_(oa::makeUnique<oa::ExecutionPlan>())
{}

// oa::ExecutionPlan destruction is non-blocking. A submitted graph transfers to
// engine retirement and remains alive through its exact completion.
oa::TrainingProgram::~TrainingProgram() = default;

oa::Status oa::TrainingProgram::validate_(const oa::ExecutableGraph& inGraph) {
	oa::U32 adamwStateAdvances = 0;
	oa::U32 replayAdamwUpdates = 0;
	oa::Bool adamwAdvanceSeen = false;
	for (const auto& node : inGraph.nodes()) {
		if (isFrozenRngKernel(node.shader)) {
			return oa::Status::error(oa::StatusCode::FailedPrecondition,
				"oa::TrainingProgram: RNG kernel '" + node.shader
				+ "' embeds a host seed; device-counter RNG is required before capture");
		}
		if (isHostSteppedOptimizerKernel(node.shader)) {
			return oa::Status::error(oa::StatusCode::FailedPrecondition,
				"oa::TrainingProgram: optimizer kernel '" + node.shader
				+ "' embeds mutable step state; use its replay-state variant");
		}
		if (isFrozenScheduleOptimizerKernel(node.shader)) {
			return oa::Status::error(oa::StatusCode::FailedPrecondition,
				"oa::TrainingProgram: optimizer kernel '" + node.shader
				+ "' embeds a mutable learning rate; device-state Muon is required before capture");
		}
		if (node.shader == "AdamwGraphAdvance") {
			++adamwStateAdvances;
			adamwAdvanceSeen = true;
		}
		if (node.shader == "AdamwGraph" or node.shader == "AdamwMany4Graph") {
			if (not adamwAdvanceSeen) {
				return oa::Status::error(oa::StatusCode::FailedPrecondition,
					"oa::TrainingProgram: AdamwGraphAdvance must precede every AdamW update");
			}
			++replayAdamwUpdates;
		}
	}
	if (replayAdamwUpdates > 0 and adamwStateAdvances != 1) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"oa::TrainingProgram: replay-safe AdamW requires exactly one "
			"AdamwGraphAdvance before all parameter updates");
	}
	return oa::Status::ok();
}

oa::Status oa::TrainingProgram::prepareReplayRng_(oa::Engine& inRuntime) {
	auto& graph = plan_->graph();
	const oa::U32 sourceNodeCount = graph.nodeCount();
	for (oa::U32 i = 0; i < sourceNodeCount; ++i) {
		auto& node = graph.nodes()[i];
		if (not isFrozenRngKernel(node.shader)) continue;

		oa::Matrix state = oa::FnMatrix::empty(
			oa::MatrixShape{1}, oa::ScalarType::UInt32, oa::MemoryPlacement::HostUpload);
		if (not state.hasStorage() or state.data() == nullptr) {
			return oa::Status::error(oa::StatusCode::OutOfMemory,
				"oa::TrainingProgram: failed to allocate replay RNG state");
		}
		const oa::U32 initialState = 0;
		if (const auto upload = oa::EngineResourceAccess::uploadBuffer(inRuntime,
			oa::MatrixAccess::descriptor(state), 0, &initialState,
			sizeof(initialState));
			not upload.isOk()) {
			return oa::Status::error(oa::StatusCode::VulkanError,
				"oa::TrainingProgram: failed to upload replay RNG state: "
				+ upload.getMessage());
		}

		if (node.shader == "PhiloxUniform") {
			node.shader = "PhiloxUniformGraph";
		} else if (node.shader == "PhiloxNormal") {
			node.shader = "PhiloxNormalGraph";
		} else {
			return oa::Status::error(oa::StatusCode::Internal,
				"oa::TrainingProgram: no replay-safe RNG kernel for '" + node.shader + "'");
		}
		node.buffers.pushBack(oa::MatrixAccess::descriptor(state));
		node.bufferOwners.pushBack(oa::MatrixAccess::storageOwner(state));
		node.access.pushBack(oa::BufferAccess::Read);
		rngStates_.pushBack(oa::move(state));
	}

	// advance every per-op counter after the complete step. Keeping this as a
	// separate tail dispatch prevents one RNG workgroup from publishing the next
	// counter while another workgroup is still reading the current value.
	for (auto& state : rngStates_) {
		oa::Vector<oavk::Buffer> buffers;
		oa::Vector<oa::SharedPtr<oavk::Buffer>> owners;
		oa::Vector<oa::BufferAccess> access;
		buffers.pushBack(oa::MatrixAccess::descriptor(state));
		owners.pushBack(oa::MatrixAccess::storageOwner(state));
		access.pushBack(oa::BufferAccess::Write);
		oa::ComputeDispatchDesc desc;
		desc.kernel = "PhiloxGraphAdvance";
		desc.buffers = buffers.span();
		desc.bufferOwners = owners.span();
		desc.access = access.span();
		desc.groupsX = 1;
		graph.add(desc);
	}
	return oa::Status::ok();
}

oa::Status oa::TrainingProgram::capture(
	oa::Engine& inEngine,
	const oa::TrainingProgramOptions& inOptions)
{
	auto& inContext = oa::ExecutionSession::forEngine(inEngine);
	auto& runtime = inEngine;
	auto* source = inContext.graph();
	auto* semanticSource = inContext.semanticGraph();
	if (source == nullptr or semanticSource == nullptr) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"oa::TrainingProgram::capture requires a vulkan compute context");
	}
	if (source->nodeCount() == 0) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"oa::TrainingProgram::capture requires a recorded training step");
	}
	auto sourceLowering = oa::analyzeSemanticLowering(*semanticSource, *source);
	if (not sourceLowering.isOk()) return sourceLowering.getStatus();
	auto dnnResult = oa::DnnPlanner::plan(*semanticSource);
	if (not dnnResult.isOk()) return dnnResult.getStatus();
	oa::Vector<oa::SemanticStorageBinding> semanticBindings;
	oa::Vector<oa::CapturedResourceDesc> capturedResources;
	oa::Vector<oa::SharedPtr<oavk::Buffer>> capturedResourceOwners;
	OA_RETURN_IF_ERROR(inContext.snapshotSemanticBindings(inOptions.observedOutputs, semanticBindings,
		capturedResources, capturedResourceOwners));
	const auto memoryAnalysis = analyzeMemory(
		oa::Span<const oa::CapturedResourceDesc>(
			capturedResources.data(), capturedResources.size()));
	OA_RETURN_IF_ERROR(reset());
	const oa::U32 semanticOpCount = semanticSource->operationCount();
	recordCompilationStage_(oa::TrainingCompilationStage::SemanticValidation,
		oa::TrainingCompilationState::Applied,
		semanticOpCount, semanticOpCount);
	auto& graph = plan_->graph();
	OA_RETURN_IF_ERROR(graph.copyNodesFrom(*source));
	auto semanticCopyStatus = semanticGraph_.copyFrom(*semanticSource);
	if (not semanticCopyStatus.isOk()) {
		(void)plan_->reset();
		return semanticCopyStatus;
	}
	plan_->dnnPlan_ = oa::makeUnique<oa::DnnPlan>(
		oa::move(dnnResult).getValue());
	auto rngStatus = prepareReplayRng_(runtime);
	if (not rngStatus.isOk()) {
		(void)plan_->reset();
		semanticGraph_.reset();
		rngStates_.clear();
		return rngStatus;
	}
	auto loweringResult = oa::analyzeSemanticLowering(semanticGraph_, graph);
	if (not loweringResult.isOk()) {
		(void)plan_->reset();
		semanticGraph_.reset();
		rngStates_.clear();
		return loweringResult.getStatus();
	}
	oa::SemanticLoweringAnalysis lowering =
		oa::move(loweringResult).getValue();
	if (inOptions.validateReplaySafety) {
		auto validationStatus = validate_(graph);
		if (not validationStatus.isOk()) {
			recordCompilationStage_(oa::TrainingCompilationStage::ReplaySafety,
				oa::TrainingCompilationState::Failed,
				graph.nodeCount(), graph.nodeCount());
			(void)plan_->reset();
			semanticGraph_.reset();
			rngStates_.clear();
			return validationStatus;
		}
		recordCompilationStage_(oa::TrainingCompilationStage::ReplaySafety,
			oa::TrainingCompilationState::Applied,
			graph.nodeCount(), graph.nodeCount());
	} else {
		recordCompilationStage_(oa::TrainingCompilationStage::ReplaySafety,
			oa::TrainingCompilationState::NotRun,
			graph.nodeCount(), graph.nodeCount());
	}
	recordCompilationStage_(oa::TrainingCompilationStage::Decomposition,
		oa::TrainingCompilationState::Analyzed,
		semanticOpCount, lowering.schemaOwnedNodeCount());
	recordCompilationStage_(oa::TrainingCompilationStage::Fusion,
		oa::TrainingCompilationState::Analyzed,
		semanticOpCount, plan_->dnnPartitionCount());
	recordCompilationStage_(oa::TrainingCompilationStage::Placement,
		oa::TrainingCompilationState::Inherited,
		semanticOpCount, graph.nodeCount());
	recordCompilationStage_(oa::TrainingCompilationStage::Precision,
		oa::TrainingCompilationState::Inherited,
		semanticOpCount, graph.nodeCount());
	recordCompilationStage_(oa::TrainingCompilationStage::KernelSelection,
		oa::TrainingCompilationState::Inherited,
		semanticOpCount, graph.nodeCount());
	recordCompilationStage_(oa::TrainingCompilationStage::LoweringValidation,
		oa::TrainingCompilationState::Applied,
		semanticOpCount, graph.nodeCount());

	// Materialize only resources whose complete owner count is explained by the
	// source session and this copied graph. Any surviving user matrix/view or
	// autograd owner makes the count larger and remains on its original storage.
	oa::Vector<oa::Matrix> aliasMatrices;
	oa::Vector<oa::U32> aliasResourceIds;
	oa::Vector<oa::U32> permittedOwners;
	oa::Vector<void*> originalHandles;
	aliasMatrices.reserve(memoryAnalysis.candidateCount);
	aliasResourceIds.reserve(memoryAnalysis.candidateCount);
	permittedOwners.reserve(memoryAnalysis.candidateCount);
	originalHandles.reserve(memoryAnalysis.candidateCount);
	for (const auto& resource : capturedResources) {
		if (not resource.aliasCandidate) continue;
		auto& owner = capturedResourceOwners[resource.resource];
		if (not owner) continue;
		oa::U32 copiedGraphOwners = 0;
		for (const auto& node : graph.nodes()) {
			for (const auto& nodeOwner : node.bufferOwners) {
				if (nodeOwner.get() == owner.get()) ++copiedGraphOwners;
			}
		}
		const long explainedOwners = static_cast<long>(
			1U + copiedGraphOwners + resource.captureRetainedOwnerCount);
		if (owner.useCount() != explainedOwners) continue;

		oa::Matrix matrix;
		oa::MatrixAccess::storageOwner(matrix) = oa::move(owner);
		oa::MatrixAccess::hostOwner(matrix).reset();
		aliasMatrices.pushBack(oa::move(matrix));
		aliasResourceIds.pushBack(resource.resource);
		permittedOwners.pushBack(resource.captureRetainedOwnerCount);
		originalHandles.pushBack(
			oa::MatrixAccess::storageOwner(aliasMatrices.back())->buffer);
	}
	oa::Vector<oa::Matrix*> eligibleMatrices;
	eligibleMatrices.reserve(aliasMatrices.size());
	for (auto& matrix : aliasMatrices) eligibleMatrices.pushBack(&matrix);
	oa::Status aliasStatus = graph.materializeAliases(
		runtime, eligibleMatrices.span(),
		oa::Span<const oa::U32>(permittedOwners.data(), permittedOwners.size()));
	// Aliasing is an optimization. allocation pressure or an ownership mismatch
	// leaves the copied graph unchanged and capture proceeds without reuse. The
	// deterministic compilation report retains the reason rather than hiding it.
	const oa::String aliasFallbackReason = aliasStatus.isOk()
		? oa::String{} : aliasStatus.getMessage();
	oa::Vector<void*> retiredHandles;
	for (oa::U32 index = 0; index < aliasMatrices.size(); ++index) {
		const auto resource = aliasResourceIds[index];
		auto& matrix = aliasMatrices[index];
		const auto& storage = oa::MatrixAccess::storageOwner(matrix);
		if (storage and storage->buffer != originalHandles[index]) {
			capturedResources[resource].aliasMaterialized = true;
			retiredHandles.pushBack(originalHandles[index]);
		}
		capturedResourceOwners[resource] =
			oa::move(oa::MatrixAccess::storageOwner(matrix));
	}
	const oa::U64 materializedAliasSavings = graph.materializedAliasSavings();
	recordCompilationStage_(oa::TrainingCompilationStage::MemoryPlanning,
		materializedAliasSavings > 0U
			? oa::TrainingCompilationState::Applied
			: oa::TrainingCompilationState::Analyzed,
		static_cast<oa::U32>(capturedResources.size()),
		materializedAliasSavings > 0U
			? static_cast<oa::U32>(retiredHandles.size())
			: memoryAnalysis.candidateCount);
	graph.setHostReadbackRequired(inOptions.hostReadbackRequired);
	graph.setReplayTimingEnabled(inOptions.enableGpuTiming);
	auto status = plan_->compile(runtime);
	if (not status.isOk()) {
		recordCompilationStage_(oa::TrainingCompilationStage::CommandRecording,
			oa::TrainingCompilationState::Failed,
			graph.nodeCount(), 0U);
		(void)plan_->reset();
		semanticGraph_.reset();
		rngStates_.clear();
		return status;
	}
	const auto graphStats = graph.getStats();
	recordCompilationStage_(
		oa::TrainingCompilationStage::SynchronizationPlanning,
		oa::TrainingCompilationState::Applied,
		graph.nodeCount(), graphStats.barrierCount + graphStats.hostBarrierCount);
	recordCompilationStage_(oa::TrainingCompilationStage::CommandRecording,
		oa::TrainingCompilationState::Applied,
		graph.nodeCount(), graph.nodeCount());
	semanticStorageBindings_ = oa::move(semanticBindings);
	loweringAnalysis_ = oa::move(lowering);
	capturedResources_ = oa::move(capturedResources);
	capturedResourceOwners_ = oa::move(capturedResourceOwners);
	aliasCandidateCount_ = memoryAnalysis.candidateCount;
	aliasMaterializationEligibleCount_ =
		static_cast<oa::U32>(eligibleMatrices.size());
	plannedAliasGroupCount_ = memoryAnalysis.groupCount;
	potentialAliasSavings_ = memoryAnalysis.potentialSavings;
	materializedAliasSavings_ = materializedAliasSavings;
	aliasMaterializationFallbackReason_ = aliasFallbackReason;
	// capture consumes one coherent recording, not only its executable nodes.
	// Clearing the private recorder resets dispatches, semantic SSA bindings, and recording
	// state together after the independently owned program compiled successfully.
	inContext.clear();
	inContext.releaseStableTransientResources(retiredHandles.span());
	return oa::Status::ok();
}

oa::Status oa::TrainingProgram::replay() {
	if (not isCaptured()) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"oa::TrainingProgram::replay called before capture");
	}
	return plan_->replay();
}

oa::Result<oa::Event> oa::TrainingProgram::replayAsync() {
	if (not isCaptured()) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"oa::TrainingProgram::replayAsync called before capture");
	}
	return plan_->replayAsync();
}

oa::Status oa::TrainingProgram::wait() {
	return plan_->wait();
}

oa::Status oa::TrainingProgram::reset() {
	OA_RETURN_IF_ERROR(plan_->reset());
	semanticGraph_.reset();
	loweringAnalysis_ = {};
	semanticStorageBindings_.clear();
	capturedResources_.clear();
	capturedResourceOwners_.clear();
	aliasCandidateCount_ = 0;
	aliasMaterializationEligibleCount_ = 0;
	plannedAliasGroupCount_ = 0;
	potentialAliasSavings_ = 0;
	materializedAliasSavings_ = 0;
	aliasMaterializationFallbackReason_.clear();
	compilationStages_.clear();
	rngStates_.clear();
	return oa::Status::ok();
}

oa::Bool oa::TrainingProgram::isCaptured() const noexcept {
	return plan_ and plan_->isCompiled();
}

oa::U32 oa::TrainingProgram::nodeCount() const noexcept {
	return plan_ ? plan_->graph().nodeCount() : 0U;
}

oa::GraphStats oa::TrainingProgram::stats() const {
	return plan_->graph().getStats();
}

oa::F64 oa::TrainingProgram::lastGpuMs() const noexcept {
	return plan_ ? plan_->graph().lastReplayGpuMs() : 0.0;
}

oa::String oa::TrainingProgram::debugReportJson(oa::StringView inName) const {
	return plan_->graph().debugReportJson(inName);
}

void oa::TrainingProgram::recordCompilationStage_(
	oa::TrainingCompilationStage inStage,
	oa::TrainingCompilationState inState,
	oa::U32 inInputCount,
	oa::U32 inOutputCount)
{
	compilationStages_.pushBack({
		.stage = inStage,
		.state = inState,
		.inputCount = inInputCount,
		.outputCount = inOutputCount,
	});
}

oa::String oa::TrainingProgram::compilationDebugReportJson(
	oa::StringView inName) const
{
	oa::internal::JsonWriter out;
	out << "{\n  \"schema\": \"oa.training_compilation.v2\",\n"
		<< "  \"name\": ";
	writeJsonString(out, inName);
	out << ",\n  \"captured\": " << (isCaptured() ? "true" : "false")
		<< ",\n  \"stages\": [";
	for (oa::U32 i = 0; i < compilationStages_.size(); ++i) {
		const auto& stage = compilationStages_[i];
		out << (i == 0U ? "\n" : ",\n")
			<< "    {\"stage\": \"" << compilationStageName(stage.stage)
			<< "\", \"state\": \"" << compilationStateName(stage.state)
			<< "\", \"input_count\": " << stage.inputCount
			<< ", \"output_count\": " << stage.outputCount << "}";
	}
	if (not compilationStages_.empty()) out << '\n';
	out << "  ],\n  \"dnn_plan\": {\n"
		<< "    \"graph_hash\": " << (plan_ ? plan_->dnnGraphHash() : 0U) << ",\n"
		<< "    \"source_operation_count\": "
		<< (plan_ ? plan_->dnnSourceOpCount() : 0U) << ",\n"
		<< "    \"captured_operation_count\": "
		<< (plan_ ? plan_->dnnCapturedOpCount() : 0U) << ",\n"
		<< "    \"partition_count\": "
		<< (plan_ ? plan_->dnnPartitionCount() : 0U) << ",\n"
		<< "    \"recognized_partition_count\": "
		<< (plan_ ? plan_->dnnRecognizedPartitionCount() : 0U) << "\n"
		<< "  },\n  \"lowering_analysis\": {\n"
		<< "    \"operation_count\": " << loweringAnalysis_.operationCount() << ",\n"
		<< "    \"schema_owned_node_count\": "
		<< loweringAnalysis_.schemaOwnedNodeCount() << ",\n"
		<< "    \"compatibility_node_count\": "
		<< loweringAnalysis_.compatibilityNodeCount() << ",\n"
		<< "    \"direct_operation_count\": "
		<< loweringAnalysis_.directOpCount() << ",\n"
		<< "    \"decomposed_operation_count\": "
		<< loweringAnalysis_.decomposedOpCount() << ",\n"
		<< "    \"fused_operation_count\": "
		<< loweringAnalysis_.fusedOpCount() << ",\n"
		<< "    \"fused_node_count\": "
		<< loweringAnalysis_.fusedNodeCount() << ",\n"
		<< "    \"maximum_nodes_per_operation\": "
		<< loweringAnalysis_.maximumNodesPerOp() << ",\n"
		<< "    \"maximum_operations_per_node\": "
		<< loweringAnalysis_.maximumOpsPerNode() << ",\n"
		<< "    \"operations\": [";
	for (oa::U32 operation = 0;
		operation < loweringAnalysis_.operationCount(); ++operation)
	{
		out << (operation == 0U ? "\n" : ",\n")
			<< "      {\"operation\": " << operation
			<< ", \"executable_node_count\": "
			<< loweringAnalysis_.executableNodeCount(operation) << "}";
	}
	if (loweringAnalysis_.operationCount() > 0U) out << '\n';
	out << "    ]\n  },\n  \"memory_analysis\": {\n"
		<< "    \"resource_count\": " << capturedResources_.size() << ",\n"
		<< "    \"candidate_count\": " << aliasCandidateCount_ << ",\n"
		<< "    \"materialization_eligible_count\": "
		<< aliasMaterializationEligibleCount_ << ",\n"
		<< "    \"alias_group_count\": " << plannedAliasGroupCount_ << ",\n"
		<< "    \"potential_savings_bytes\": " << potentialAliasSavings_ << ",\n"
		<< "    \"materialized_savings_bytes\": "
		<< materializedAliasSavings_ << ",\n"
		<< "    \"materialized\": "
		<< (materializedAliasSavings_ > 0U ? "true" : "false") << ",\n"
		<< "    \"fallback_reason\": ";
	writeJsonString(out, aliasMaterializationFallbackReason_);
	out << ",\n"
		<< "    \"resources\": [";
	for (oa::U32 i = 0; i < capturedResources_.size(); ++i) {
		const auto& resource = capturedResources_[i];
		out << (i == 0U ? "\n" : ",\n")
			<< "      {\"resource\": " << resource.resource
			<< ", \"bytes\": " << resource.byteSize
			<< ", \"placement\": \"" << memoryPlacementName(resource.placement)
			<< "\", \"has_lifetime\": " << (resource.hasLifetime ? "true" : "false")
			<< ", \"first_access\": " << resource.firstAccess
			<< ", \"last_access\": " << resource.lastAccess
			<< ", \"semantic_external\": "
			<< (resource.semanticExternal ? "true" : "false")
			<< ", \"stable_replay_input\": "
			<< (resource.stableReplayInput ? "true" : "false")
			<< ", \"stable_transient\": "
			<< (resource.stableTransient ? "true" : "false")
			<< ", \"observed_output\": "
			<< (resource.observedOutput ? "true" : "false")
			<< ", \"alias_candidate\": "
			<< (resource.aliasCandidate ? "true" : "false")
			<< ", \"alias_materialized\": "
			<< (resource.aliasMaterialized ? "true" : "false") << "}";
	}
	if (not capturedResources_.empty()) out << '\n';
	out << "    ]\n  }\n}\n";
	return out.take();
}
