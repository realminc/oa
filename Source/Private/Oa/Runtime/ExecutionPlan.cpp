#include "ExecutionPlan.h"

#include <Oa/Runtime/ComputeGraph.h>
#include <Oa/Runtime/Engine.h>

OaExecutionPlan::OaExecutionPlan()
	: Graph_(OaMakeUniquePtr<OaComputeGraph>())
{}

OaExecutionPlan::~OaExecutionPlan() {
	if (not Graph_) return;

	if (Engine_ != nullptr) {
		const auto completion = Graph_->LastCompletion(Engine_->Device);
		if (completion.IsValid()) {
			// The engine owns the Vulkan device and is the only safe non-blocking
			// retirement owner after this plan disappears.
			Engine_->RetireExecutionPlan(OaStdMove(Graph_));
			Engine_ = nullptr;
			return;
		}
		Graph_->Destroy(Engine_->Device);
	} else {
		Graph_->Reset();
	}
	Graph_.Reset();
	Engine_ = nullptr;
}

OaComputeGraph& OaExecutionPlan::Graph() noexcept {
	return *Graph_;
}

const OaComputeGraph& OaExecutionPlan::Graph() const noexcept {
	return *Graph_;
}

OaStatus OaExecutionPlan::Compile(OaEngine& InEngine) {
	if (Compiled_) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"execution plan is already compiled");
	}
	Engine_ = &InEngine;
	const auto status = Graph_->Compile(InEngine);
	if (status.IsOk()) Compiled_ = true;
	return status;
}

OaStatus OaExecutionPlan::Replay() {
	if (not Compiled_ or Engine_ == nullptr) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"execution plan replay called before compile");
	}
	return Graph_->Replay(*Engine_);
}

OaResult<OaCompletionToken> OaExecutionPlan::ReplayAsync() {
	if (not Compiled_ or Engine_ == nullptr) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"execution plan asynchronous replay called before compile");
	}
	return Graph_->ReplayAsync(*Engine_);
}

OaStatus OaExecutionPlan::Wait() {
	if (not Compiled_ or Engine_ == nullptr) return OaStatus::Ok();
	return Graph_->WaitForPendingReplay(Engine_->Device);
}

OaStatus OaExecutionPlan::Reset() {
	if (not Graph_) return OaStatus::Ok();
	if (Engine_ != nullptr) {
		OA_RETURN_IF_ERROR(Graph_->WaitForPendingReplay(Engine_->Device));
		Graph_->Destroy(Engine_->Device);
	} else {
		Graph_->Reset();
	}
	Engine_ = nullptr;
	Compiled_ = false;
	return OaStatus::Ok();
}
