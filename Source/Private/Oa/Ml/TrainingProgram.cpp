#include <Oa/Ml/TrainingProgram.h>

#include <Oa/Core/FnMatrix.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Engine.h>

#include <cstring>

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

} // namespace

OaTrainingProgram::~OaTrainingProgram() {
	(void)Reset();
}

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

OaStatus OaTrainingProgram::PrepareReplayRng_(OaComputeEngine& InRuntime) {
	const OaU32 sourceNodeCount = Graph_.NodeCount();
	for (OaU32 i = 0; i < sourceNodeCount; ++i) {
		auto& node = Graph_.Nodes()[i];
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
		Graph_.Add(desc);
	}
	return OaStatus::Ok();
}

OaStatus OaTrainingProgram::Capture(
	OaContext& InContext,
	const OaTrainingProgramOptions& InOptions)
{
	auto* runtime = InContext.GetEngine();
	auto* source = InContext.Graph();
	if (runtime == nullptr or source == nullptr) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"OaTrainingProgram::Capture requires a Vulkan compute context");
	}
	if (source->NodeCount() == 0) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"OaTrainingProgram::Capture requires a recorded training step");
	}
	OA_RETURN_IF_ERROR(Reset());
	OA_RETURN_IF_ERROR(Graph_.CopyNodesFrom(*source));
	auto rngStatus = PrepareReplayRng_(*runtime);
	if (not rngStatus.IsOk()) {
		Graph_.Destroy(runtime->Device);
		RngStates_.Clear();
		return rngStatus;
	}
	if (InOptions.ValidateReplaySafety) {
		auto validationStatus = Validate_(Graph_);
		if (not validationStatus.IsOk()) {
			Graph_.Destroy(runtime->Device);
			RngStates_.Clear();
			return validationStatus;
		}
	}
	Graph_.SetHostReadbackRequired(InOptions.HostReadbackRequired);
	Graph_.SetReplayTimingEnabled(InOptions.EnableGpuTiming);
	auto status = Graph_.Compile(*runtime);
	if (not status.IsOk()) {
		Graph_.Destroy(runtime->Device);
		RngStates_.Clear();
		return status;
	}
	Runtime_ = runtime;
	Captured_ = true;
	source->ClearNodes();
	return OaStatus::Ok();
}

OaStatus OaTrainingProgram::Replay() {
	if (not Captured_ or Runtime_ == nullptr) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"OaTrainingProgram::Replay called before Capture");
	}
	return Graph_.Replay(*Runtime_);
}

OaResult<OaCompletionToken> OaTrainingProgram::ReplayAsync() {
	if (not Captured_ or Runtime_ == nullptr) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"OaTrainingProgram::ReplayAsync called before Capture");
	}
	return Graph_.ReplayAsync(*Runtime_);
}

OaStatus OaTrainingProgram::Wait() {
	if (not Captured_ or Runtime_ == nullptr) return OaStatus::Ok();
	return Graph_.WaitForPendingReplay(Runtime_->Device);
}

OaStatus OaTrainingProgram::Reset() {
	if (Runtime_ != nullptr) {
		auto waitStatus = Graph_.WaitForPendingReplay(Runtime_->Device);
		if (not waitStatus.IsOk()) return waitStatus;
		Graph_.Destroy(Runtime_->Device);
	} else {
		Graph_.Reset();
	}
	Runtime_ = nullptr;
	Captured_ = false;
	RngStates_.Clear();
	return OaStatus::Ok();
}
