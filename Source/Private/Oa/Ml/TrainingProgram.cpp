#include <Oa/Ml/TrainingProgram.h>

#include <Oa/Core/FnMatrix.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/ExecutionMemory.h>
#include "../Runtime/ExecutionPlan.h"

#include <cstring>
#include <iomanip>
#include <sstream>

namespace {

OaBool IsFrozenRngKernel(OaStringView InName) {
	return InName == "PhiloxUniform" or InName == "PhiloxNormal";
}

OaBool IsHostSteppedOptimizerKernel(OaStringView InName) {
	return InName == "Adam" or InName == "Adamw" or InName == "AdamwMany4";
}

OaBool IsFrozenScheduleOptimizerKernel(OaStringView InName) {
	return InName == "MuonNesterov" or InName == "MuonApply"
		or InName == "MuonVector";
}

const char* CompilationStageName(OaTrainingCompilationStage InStage) {
	switch (InStage) {
		case OaTrainingCompilationStage::SemanticValidation:
			return "semantic_validation";
		case OaTrainingCompilationStage::ReplaySafety:
			return "replay_safety";
		case OaTrainingCompilationStage::Decomposition:
			return "decomposition";
		case OaTrainingCompilationStage::Fusion: return "fusion";
		case OaTrainingCompilationStage::Placement: return "placement";
		case OaTrainingCompilationStage::Precision: return "precision";
		case OaTrainingCompilationStage::KernelSelection:
			return "kernel_selection";
		case OaTrainingCompilationStage::LoweringValidation:
			return "lowering_validation";
		case OaTrainingCompilationStage::MemoryPlanning:
			return "memory_planning";
		case OaTrainingCompilationStage::SynchronizationPlanning:
			return "synchronization_planning";
		case OaTrainingCompilationStage::CommandRecording:
			return "command_recording";
	}
	return "unknown";
}

const char* CompilationStateName(OaTrainingCompilationState InState) {
	switch (InState) {
		case OaTrainingCompilationState::NotRun: return "not_run";
		case OaTrainingCompilationState::Inherited: return "inherited";
		case OaTrainingCompilationState::Analyzed: return "analyzed";
		case OaTrainingCompilationState::Applied: return "applied";
		case OaTrainingCompilationState::Failed: return "failed";
	}
	return "unknown";
}

const char* MemoryPlacementName(OaMemoryPlacement InPlacement) {
	switch (InPlacement) {
		case OaMemoryPlacement::Auto: return "auto";
		case OaMemoryPlacement::DeviceLocal: return "device_local";
		case OaMemoryPlacement::HostUpload: return "host_upload";
		case OaMemoryPlacement::HostReadback: return "host_readback";
		case OaMemoryPlacement::Unified: return "unified";
	}
	return "unknown";
}

void WriteJsonString(std::ostringstream& Out, OaStringView InValue) {
	Out << '"';
	for (const char value : InValue) {
		switch (value) {
			case '"': Out << "\\\""; break;
			case '\\': Out << "\\\\"; break;
			case '\b': Out << "\\b"; break;
			case '\f': Out << "\\f"; break;
			case '\n': Out << "\\n"; break;
			case '\r': Out << "\\r"; break;
			case '\t': Out << "\\t"; break;
			default:
				if (static_cast<unsigned char>(value) < 0x20U) {
					Out << "\\u" << std::hex << std::setw(4)
						<< std::setfill('0')
						<< static_cast<unsigned>(
							static_cast<unsigned char>(value))
						<< std::dec << std::setfill(' ');
				} else {
					Out << value;
				}
		}
	}
	Out << '"';
}

struct OaMemoryAnalysisSummary {
	OaU32 CandidateCount = 0;
	OaU32 GroupCount = 0;
	OaU64 PotentialSavings = 0;
};

OaMemoryAnalysisSummary AnalyzeMemory(
	OaSpan<const OaCapturedResourceDesc> InResources)
{
	struct Group {
		OaMemoryPlacement Placement = OaMemoryPlacement::Auto;
		OaVec<const OaCapturedResourceDesc*> Members;
		OaU64 RequiredBytes = 0;
		OaU64 TotalBytes = 0;
	};
	OaVec<Group> groups;
	OaMemoryAnalysisSummary summary;
	for (const auto& resource : InResources) {
		if (not resource.AliasCandidate) continue;
		++summary.CandidateCount;
		OaBool placed = false;
		for (auto& group : groups) {
			if (group.Placement != resource.Placement) continue;
			OaBool overlaps = false;
			for (const auto* member : group.Members) {
				if (resource.FirstAccess <= member->LastAccess
					and resource.LastAccess >= member->FirstAccess)
				{
					overlaps = true;
					break;
				}
			}
			if (overlaps) continue;
			group.Members.PushBack(&resource);
			group.RequiredBytes = std::max(group.RequiredBytes, resource.ByteSize);
			group.TotalBytes += resource.ByteSize;
			placed = true;
			break;
		}
		if (not placed) {
			Group group;
			group.Placement = resource.Placement;
			group.Members.PushBack(&resource);
			group.RequiredBytes = resource.ByteSize;
			group.TotalBytes = resource.ByteSize;
			groups.PushBack(std::move(group));
		}
	}
	for (const auto& group : groups) {
		if (group.Members.Size() < 2U) continue;
		++summary.GroupCount;
		summary.PotentialSavings += group.TotalBytes - group.RequiredBytes;
	}
	return summary;
}

} // namespace

OaTrainingProgram::OaTrainingProgram()
	: Plan_(OaMakeUniquePtr<OaExecutionPlan>())
{}

// OaExecutionPlan destruction is non-blocking. A submitted graph transfers to
// engine retirement and remains alive through its exact completion.
OaTrainingProgram::~OaTrainingProgram() = default;

OaStatus OaTrainingProgram::Validate_(const OaComputeGraph& InGraph) {
	OaU32 adamwStateAdvances = 0;
	OaU32 replayAdamwUpdates = 0;
	OaBool adamwAdvanceSeen = false;
	for (const auto& node : InGraph.Nodes()) {
		if (IsFrozenRngKernel(node.Shader)) {
			return OaStatus::Error(OaStatusCode::FailedPrecondition,
				"OaTrainingProgram: RNG kernel '" + node.Shader
				+ "' embeds a host seed; device-counter RNG is required before capture");
		}
		if (IsHostSteppedOptimizerKernel(node.Shader)) {
			return OaStatus::Error(OaStatusCode::FailedPrecondition,
				"OaTrainingProgram: optimizer kernel '" + node.Shader
				+ "' embeds mutable step state; use its replay-state variant");
		}
		if (IsFrozenScheduleOptimizerKernel(node.Shader)) {
			return OaStatus::Error(OaStatusCode::FailedPrecondition,
				"OaTrainingProgram: optimizer kernel '" + node.Shader
				+ "' embeds a mutable learning rate; device-state Muon is required before capture");
		}
		if (node.Shader == "AdamwGraphAdvance") {
			++adamwStateAdvances;
			adamwAdvanceSeen = true;
		}
		if (node.Shader == "AdamwGraph" or node.Shader == "AdamwMany4Graph") {
			if (not adamwAdvanceSeen) {
				return OaStatus::Error(OaStatusCode::FailedPrecondition,
					"OaTrainingProgram: AdamwGraphAdvance must precede every AdamW update");
			}
			++replayAdamwUpdates;
		}
	}
	if (replayAdamwUpdates > 0 and adamwStateAdvances != 1) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"OaTrainingProgram: replay-safe AdamW requires exactly one "
			"AdamwGraphAdvance before all parameter updates");
	}
	return OaStatus::Ok();
}

OaStatus OaTrainingProgram::PrepareReplayRng_(OaEngine& InRuntime) {
	auto& graph = Plan_->Graph();
	const OaU32 sourceNodeCount = graph.NodeCount();
	for (OaU32 i = 0; i < sourceNodeCount; ++i) {
		auto& node = graph.Nodes()[i];
		if (not IsFrozenRngKernel(node.Shader)) continue;

		OaMatrix state = OaFnMatrix::Empty(
			OaMatrixShape{1}, OaScalarType::UInt32, OaMemoryPlacement::HostUpload);
		if (not state.HasStorage() or state.Data() == nullptr) {
			return OaStatus::Error(OaStatusCode::OutOfMemory,
				"OaTrainingProgram: failed to allocate replay RNG state");
		}
		*state.DataAs<OaU32>() = 0;
		if (not InRuntime.Allocator.FlushHostBuffer(
			state.GetVkBuffer(), 0, sizeof(OaU32))) {
			return OaStatus::Error(OaStatusCode::VulkanError,
				"OaTrainingProgram: failed to flush replay RNG state");
		}

		node.Shader += "Graph";
		node.Buffers.PushBack(state.GetVkBuffer());
		node.BufferOwners.PushBack(state.VkBuf_);
		node.Access.PushBack(OaBufferAccess::Read);
		RngStates_.PushBack(std::move(state));
	}

	// Advance every per-op counter after the complete step. Keeping this as a
	// separate tail dispatch prevents one RNG workgroup from publishing the next
	// counter while another workgroup is still reading the current value.
	for (auto& state : RngStates_) {
		OaVec<OaVkBuffer> buffers;
		OaVec<OaSharedPtr<OaVkBuffer>> owners;
		OaVec<OaBufferAccess> access;
		buffers.PushBack(state.GetVkBuffer());
		owners.PushBack(state.VkBuf_);
		access.PushBack(OaBufferAccess::Write);
		OaComputeDispatchDesc desc;
		desc.Kernel = "PhiloxGraphAdvance";
		desc.Buffers = buffers.Span();
		desc.BufferOwners = owners.Span();
		desc.Access = access.Span();
		desc.GroupsX = 1;
		graph.Add(desc);
	}
	return OaStatus::Ok();
}

OaStatus OaTrainingProgram::Capture(
	OaContext& InContext,
	const OaTrainingProgramOptions& InOptions)
{
	auto* runtime = InContext.GetEngine();
	auto* source = InContext.Graph();
	auto* semanticSource = InContext.SemanticGraph();
	if (runtime == nullptr or source == nullptr or semanticSource == nullptr) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"OaTrainingProgram::Capture requires a Vulkan compute context");
	}
	if (source->NodeCount() == 0) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"OaTrainingProgram::Capture requires a recorded training step");
	}
	auto sourceLowering = OaAnalyzeSemanticLowering(*semanticSource, *source);
	if (not sourceLowering.IsOk()) return sourceLowering.GetStatus();
	OaVec<OaSemanticStorageBinding> semanticBindings;
	OaVec<OaCapturedResourceDesc> capturedResources;
	OaVec<OaSharedPtr<OaVkBuffer>> capturedResourceOwners;
	OA_RETURN_IF_ERROR(OaExecutionMemory::SnapshotSemanticBindings(
		InContext, InOptions.ObservedOutputs, semanticBindings,
		capturedResources, capturedResourceOwners));
	const auto memoryAnalysis = AnalyzeMemory(
		OaSpan<const OaCapturedResourceDesc>(
			capturedResources.Data(), capturedResources.Size()));
	OA_RETURN_IF_ERROR(Reset());
	const OaU32 semanticOperationCount = semanticSource->OperationCount();
	RecordCompilationStage_(OaTrainingCompilationStage::SemanticValidation,
		OaTrainingCompilationState::Applied,
		semanticOperationCount, semanticOperationCount);
	auto& graph = Plan_->Graph();
	OA_RETURN_IF_ERROR(graph.CopyNodesFrom(*source));
	auto semanticCopyStatus = SemanticGraph_.CopyFrom(*semanticSource);
	if (not semanticCopyStatus.IsOk()) {
		(void)Plan_->Reset();
		return semanticCopyStatus;
	}
	auto rngStatus = PrepareReplayRng_(*runtime);
	if (not rngStatus.IsOk()) {
		(void)Plan_->Reset();
		SemanticGraph_.Reset();
		RngStates_.Clear();
		return rngStatus;
	}
	auto loweringResult = OaAnalyzeSemanticLowering(SemanticGraph_, graph);
	if (not loweringResult.IsOk()) {
		(void)Plan_->Reset();
		SemanticGraph_.Reset();
		RngStates_.Clear();
		return loweringResult.GetStatus();
	}
	OaSemanticLoweringAnalysis lowering =
		OaStdMove(loweringResult).GetValue();
	if (InOptions.ValidateReplaySafety) {
		auto validationStatus = Validate_(graph);
		if (not validationStatus.IsOk()) {
			RecordCompilationStage_(OaTrainingCompilationStage::ReplaySafety,
				OaTrainingCompilationState::Failed,
				graph.NodeCount(), graph.NodeCount());
			(void)Plan_->Reset();
			SemanticGraph_.Reset();
			RngStates_.Clear();
			return validationStatus;
		}
		RecordCompilationStage_(OaTrainingCompilationStage::ReplaySafety,
			OaTrainingCompilationState::Applied,
			graph.NodeCount(), graph.NodeCount());
	} else {
		RecordCompilationStage_(OaTrainingCompilationStage::ReplaySafety,
			OaTrainingCompilationState::NotRun,
			graph.NodeCount(), graph.NodeCount());
	}
	RecordCompilationStage_(OaTrainingCompilationStage::Decomposition,
		OaTrainingCompilationState::Analyzed,
		semanticOperationCount, lowering.SchemaOwnedNodeCount());
	RecordCompilationStage_(OaTrainingCompilationStage::Fusion,
		OaTrainingCompilationState::Analyzed,
		semanticOperationCount, lowering.SchemaOwnedNodeCount());
	RecordCompilationStage_(OaTrainingCompilationStage::Placement,
		OaTrainingCompilationState::Inherited,
		semanticOperationCount, graph.NodeCount());
	RecordCompilationStage_(OaTrainingCompilationStage::Precision,
		OaTrainingCompilationState::Inherited,
		semanticOperationCount, graph.NodeCount());
	RecordCompilationStage_(OaTrainingCompilationStage::KernelSelection,
		OaTrainingCompilationState::Inherited,
		semanticOperationCount, graph.NodeCount());
	RecordCompilationStage_(OaTrainingCompilationStage::LoweringValidation,
		OaTrainingCompilationState::Applied,
		semanticOperationCount, graph.NodeCount());

	// Materialize only resources whose complete owner count is explained by the
	// source session and this copied graph. Any surviving user matrix/view or
	// autograd owner makes the count larger and remains on its original storage.
	OaVec<OaMatrix> aliasMatrices;
	OaVec<OaSemanticResourceId> aliasResourceIds;
	OaVec<OaU32> permittedOwners;
	OaVec<void*> originalHandles;
	aliasMatrices.Reserve(memoryAnalysis.CandidateCount);
	aliasResourceIds.Reserve(memoryAnalysis.CandidateCount);
	permittedOwners.Reserve(memoryAnalysis.CandidateCount);
	originalHandles.Reserve(memoryAnalysis.CandidateCount);
	for (const auto& resource : capturedResources) {
		if (not resource.AliasCandidate) continue;
		auto& owner = capturedResourceOwners[resource.Resource];
		if (not owner) continue;
		OaU32 copiedGraphOwners = 0;
		for (const auto& node : graph.Nodes()) {
			for (const auto& nodeOwner : node.BufferOwners) {
				if (nodeOwner.Get() == owner.Get()) ++copiedGraphOwners;
			}
		}
		const long explainedOwners = static_cast<long>(
			1U + copiedGraphOwners + resource.CaptureRetainedOwnerCount);
		if (owner.UseCount() != explainedOwners) continue;

		OaMatrix matrix;
		matrix.VkBuf_ = std::move(owner);
		matrix.Data_.Reset();
		aliasMatrices.PushBack(std::move(matrix));
		aliasResourceIds.PushBack(resource.Resource);
		permittedOwners.PushBack(resource.CaptureRetainedOwnerCount);
		originalHandles.PushBack(aliasMatrices.Back().VkBuf_->Buffer);
	}
	OaVec<OaMatrix*> eligibleMatrices;
	eligibleMatrices.Reserve(aliasMatrices.Size());
	for (auto& matrix : aliasMatrices) eligibleMatrices.PushBack(&matrix);
	OaStatus aliasStatus = graph.MaterializeAliases(
		*runtime, eligibleMatrices.Span(),
		OaSpan<const OaU32>(permittedOwners.Data(), permittedOwners.Size()));
	// Aliasing is an optimization. Allocation pressure or an ownership mismatch
	// leaves the copied graph unchanged and capture proceeds without reuse. The
	// deterministic compilation report retains the reason rather than hiding it.
	const OaString aliasFallbackReason = aliasStatus.IsOk()
		? OaString{} : aliasStatus.GetMessage();
	OaVec<void*> retiredHandles;
	for (OaU32 index = 0; index < aliasMatrices.Size(); ++index) {
		const auto resource = aliasResourceIds[index];
		auto& matrix = aliasMatrices[index];
		if (matrix.VkBuf_ and matrix.VkBuf_->Buffer != originalHandles[index]) {
			capturedResources[resource].AliasMaterialized = true;
			retiredHandles.PushBack(originalHandles[index]);
		}
		capturedResourceOwners[resource] = std::move(matrix.VkBuf_);
	}
	const OaU64 materializedAliasSavings = graph.MaterializedAliasSavings();
	RecordCompilationStage_(OaTrainingCompilationStage::MemoryPlanning,
		materializedAliasSavings > 0U
			? OaTrainingCompilationState::Applied
			: OaTrainingCompilationState::Analyzed,
		static_cast<OaU32>(capturedResources.Size()),
		materializedAliasSavings > 0U
			? static_cast<OaU32>(retiredHandles.Size())
			: memoryAnalysis.CandidateCount);
	graph.SetHostReadbackRequired(InOptions.HostReadbackRequired);
	graph.SetReplayTimingEnabled(InOptions.EnableGpuTiming);
	auto status = Plan_->Compile(*runtime);
	if (not status.IsOk()) {
		RecordCompilationStage_(OaTrainingCompilationStage::CommandRecording,
			OaTrainingCompilationState::Failed,
			graph.NodeCount(), 0U);
		(void)Plan_->Reset();
		SemanticGraph_.Reset();
		RngStates_.Clear();
		return status;
	}
	const auto graphStats = graph.GetStats();
	RecordCompilationStage_(
		OaTrainingCompilationStage::SynchronizationPlanning,
		OaTrainingCompilationState::Applied,
		graph.NodeCount(), graphStats.BarrierCount + graphStats.HostBarrierCount);
	RecordCompilationStage_(OaTrainingCompilationStage::CommandRecording,
		OaTrainingCompilationState::Applied,
		graph.NodeCount(), graph.NodeCount());
	SemanticStorageBindings_ = OaStdMove(semanticBindings);
	LoweringAnalysis_ = OaStdMove(lowering);
	CapturedResources_ = OaStdMove(capturedResources);
	CapturedResourceOwners_ = OaStdMove(capturedResourceOwners);
	AliasCandidateCount_ = memoryAnalysis.CandidateCount;
	AliasMaterializationEligibleCount_ =
		static_cast<OaU32>(eligibleMatrices.Size());
	PlannedAliasGroupCount_ = memoryAnalysis.GroupCount;
	PotentialAliasSavings_ = memoryAnalysis.PotentialSavings;
	MaterializedAliasSavings_ = materializedAliasSavings;
	AliasMaterializationFallbackReason_ = aliasFallbackReason;
	// Capture consumes one coherent recording, not only its executable nodes.
	// OaContext::Clear resets dispatches, semantic SSA bindings, and recording
	// state together after the independently owned program compiled successfully.
	InContext.Clear();
	OaExecutionMemory::ReleaseStableTransients(InContext, retiredHandles.Span());
	return OaStatus::Ok();
}

OaStatus OaTrainingProgram::Replay() {
	if (not IsCaptured()) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"OaTrainingProgram::Replay called before Capture");
	}
	return Plan_->Replay();
}

OaResult<OaCompletionToken> OaTrainingProgram::ReplayAsync() {
	if (not IsCaptured()) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"OaTrainingProgram::ReplayAsync called before Capture");
	}
	return Plan_->ReplayAsync();
}

OaStatus OaTrainingProgram::Wait() {
	return Plan_->Wait();
}

OaStatus OaTrainingProgram::Reset() {
	OA_RETURN_IF_ERROR(Plan_->Reset());
	SemanticGraph_.Reset();
	LoweringAnalysis_ = {};
	SemanticStorageBindings_.Clear();
	CapturedResources_.Clear();
	CapturedResourceOwners_.Clear();
	AliasCandidateCount_ = 0;
	AliasMaterializationEligibleCount_ = 0;
	PlannedAliasGroupCount_ = 0;
	PotentialAliasSavings_ = 0;
	MaterializedAliasSavings_ = 0;
	AliasMaterializationFallbackReason_.clear();
	CompilationStages_.Clear();
	RngStates_.Clear();
	return OaStatus::Ok();
}

OaBool OaTrainingProgram::IsCaptured() const noexcept {
	return Plan_ and Plan_->IsCompiled();
}

OaU32 OaTrainingProgram::NodeCount() const noexcept {
	return Plan_ ? Plan_->Graph().NodeCount() : 0U;
}

OaGraphStats OaTrainingProgram::Stats() const {
	return Plan_->Graph().GetStats();
}

OaF64 OaTrainingProgram::LastGpuMs() const noexcept {
	return Plan_ ? Plan_->Graph().LastReplayGpuMs() : 0.0;
}

OaString OaTrainingProgram::DebugReportJson(OaStringView InName) const {
	return Plan_->Graph().DebugReportJson(InName);
}

void OaTrainingProgram::RecordCompilationStage_(
	OaTrainingCompilationStage InStage,
	OaTrainingCompilationState InState,
	OaU32 InInputCount,
	OaU32 InOutputCount)
{
	CompilationStages_.PushBack({
		.Stage = InStage,
		.State = InState,
		.InputCount = InInputCount,
		.OutputCount = InOutputCount,
	});
}

OaString OaTrainingProgram::CompilationDebugReportJson(
	OaStringView InName) const
{
	std::ostringstream out;
	out << "{\n  \"schema\": \"oa.training_compilation.v2\",\n"
		<< "  \"name\": ";
	WriteJsonString(out, InName);
	out << ",\n  \"captured\": " << (IsCaptured() ? "true" : "false")
		<< ",\n  \"stages\": [";
	for (OaU32 i = 0; i < CompilationStages_.Size(); ++i) {
		const auto& stage = CompilationStages_[i];
		out << (i == 0U ? "\n" : ",\n")
			<< "    {\"stage\": \"" << CompilationStageName(stage.Stage)
			<< "\", \"state\": \"" << CompilationStateName(stage.State)
			<< "\", \"input_count\": " << stage.InputCount
			<< ", \"output_count\": " << stage.OutputCount << "}";
	}
	if (not CompilationStages_.Empty()) out << '\n';
	out << "  ],\n  \"lowering_analysis\": {\n"
		<< "    \"operation_count\": " << LoweringAnalysis_.OperationCount() << ",\n"
		<< "    \"schema_owned_node_count\": "
		<< LoweringAnalysis_.SchemaOwnedNodeCount() << ",\n"
		<< "    \"compatibility_node_count\": "
		<< LoweringAnalysis_.CompatibilityNodeCount() << ",\n"
		<< "    \"direct_operation_count\": "
		<< LoweringAnalysis_.DirectOperationCount() << ",\n"
		<< "    \"decomposed_operation_count\": "
		<< LoweringAnalysis_.DecomposedOperationCount() << ",\n"
		<< "    \"fused_operation_count\": "
		<< LoweringAnalysis_.FusedOperationCount() << ",\n"
		<< "    \"fused_node_count\": "
		<< LoweringAnalysis_.FusedNodeCount() << ",\n"
		<< "    \"maximum_nodes_per_operation\": "
		<< LoweringAnalysis_.MaximumNodesPerOperation() << ",\n"
		<< "    \"maximum_operations_per_node\": "
		<< LoweringAnalysis_.MaximumOperationsPerNode() << ",\n"
		<< "    \"operations\": [";
	for (OaU32 operation = 0;
		operation < LoweringAnalysis_.OperationCount(); ++operation)
	{
		out << (operation == 0U ? "\n" : ",\n")
			<< "      {\"operation\": " << operation
			<< ", \"executable_node_count\": "
			<< LoweringAnalysis_.ExecutableNodeCount(operation) << "}";
	}
	if (LoweringAnalysis_.OperationCount() > 0U) out << '\n';
	out << "    ]\n  },\n  \"memory_analysis\": {\n"
		<< "    \"resource_count\": " << CapturedResources_.Size() << ",\n"
		<< "    \"candidate_count\": " << AliasCandidateCount_ << ",\n"
		<< "    \"materialization_eligible_count\": "
		<< AliasMaterializationEligibleCount_ << ",\n"
		<< "    \"alias_group_count\": " << PlannedAliasGroupCount_ << ",\n"
		<< "    \"potential_savings_bytes\": " << PotentialAliasSavings_ << ",\n"
		<< "    \"materialized_savings_bytes\": "
		<< MaterializedAliasSavings_ << ",\n"
		<< "    \"materialized\": "
		<< (MaterializedAliasSavings_ > 0U ? "true" : "false") << ",\n"
		<< "    \"fallback_reason\": ";
	WriteJsonString(out, AliasMaterializationFallbackReason_);
	out << ",\n"
		<< "    \"resources\": [";
	for (OaU32 i = 0; i < CapturedResources_.Size(); ++i) {
		const auto& resource = CapturedResources_[i];
		out << (i == 0U ? "\n" : ",\n")
			<< "      {\"resource\": " << resource.Resource
			<< ", \"bytes\": " << resource.ByteSize
			<< ", \"placement\": \"" << MemoryPlacementName(resource.Placement)
			<< "\", \"has_lifetime\": " << (resource.HasLifetime ? "true" : "false")
			<< ", \"first_access\": " << resource.FirstAccess
			<< ", \"last_access\": " << resource.LastAccess
			<< ", \"semantic_external\": "
			<< (resource.SemanticExternal ? "true" : "false")
			<< ", \"stable_replay_input\": "
			<< (resource.StableReplayInput ? "true" : "false")
			<< ", \"stable_transient\": "
			<< (resource.StableTransient ? "true" : "false")
			<< ", \"observed_output\": "
			<< (resource.ObservedOutput ? "true" : "false")
			<< ", \"alias_candidate\": "
			<< (resource.AliasCandidate ? "true" : "false")
			<< ", \"alias_materialized\": "
			<< (resource.AliasMaterialized ? "true" : "false") << "}";
	}
	if (not CapturedResources_.Empty()) out << '\n';
	out << "    ]\n  }\n}\n";
	return OaString(out.str());
}
