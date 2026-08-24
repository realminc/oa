#include <oa/runtime/executionPlan.h>

#include <oa/runtime/executableGraph.h>
#include <oa/runtime/dnn.h>
#include <oa/runtime/engine.h>
#include "engine/engineAccess.h"

oa::ExecutionPlan::ExecutionPlan()
	: graph_(oa::makeUnique<oa::ExecutableGraph>())
{}

oa::ExecutionPlan::~ExecutionPlan() {
	release_();
}

oa::ExecutionPlan::ExecutionPlan(oa::ExecutionPlan&& inOther) noexcept
	: graph_(oa::move(inOther.graph_))
	, dnnPlan_(oa::move(inOther.dnnPlan_))
	, engine_(inOther.engine_)
	, compiled_(inOther.compiled_)
{
	inOther.engine_ = nullptr;
	inOther.compiled_ = false;
}

oa::ExecutionPlan& oa::ExecutionPlan::operator=(
	oa::ExecutionPlan&& inOther) noexcept
{
	if (this == &inOther) {
		return *this;
	}
	release_();
	graph_ = oa::move(inOther.graph_);
	dnnPlan_ = oa::move(inOther.dnnPlan_);
	engine_ = inOther.engine_;
	compiled_ = inOther.compiled_;
	inOther.engine_ = nullptr;
	inOther.compiled_ = false;
	return *this;
}

void oa::ExecutionPlan::release_() noexcept {
	if (not graph_) {
		dnnPlan_.reset();
		return;
	}

	if (engine_ != nullptr) {
		const auto completion = graph_->lastCompletion(*engine_);
		if (completion.isValid()) {
			// The engine owns the vulkan device and is the only safe non-blocking
			// retirement owner after this plan disappears.
			oa::EngineAccess(*engine_).retireExecutionPlan(oa::move(graph_));
			dnnPlan_.reset();
			engine_ = nullptr;
			return;
		}
	} else {
		graph_->reset();
	}

	graph_.reset();
	dnnPlan_.reset();
	engine_ = nullptr;
	compiled_ = false;
}

oa::ExecutableGraph& oa::ExecutionPlan::graph() noexcept {
	return *graph_;
}

const oa::ExecutableGraph& oa::ExecutionPlan::graph() const noexcept {
	return *graph_;
}

oa::U32 oa::ExecutionPlan::nodeCount() const noexcept {
	return graph_ ? graph_->nodeCount() : 0U;
}

oa::U64 oa::ExecutionPlan::dnnGraphHash() const noexcept {
	return dnnPlan_ ? dnnPlan_->graphHash : 0U;
}

oa::U32 oa::ExecutionPlan::dnnSourceOpCount() const noexcept {
	return dnnPlan_ ? dnnPlan_->sourceOpCount : 0U;
}

oa::U32 oa::ExecutionPlan::dnnCapturedOpCount() const noexcept {
	return dnnPlan_ ? dnnPlan_->capturedOpCount : 0U;
}

oa::U32 oa::ExecutionPlan::dnnPartitionCount() const noexcept {
	return dnnPlan_ ? static_cast<oa::U32>(dnnPlan_->partitions.size()) : 0U;
}

oa::U32 oa::ExecutionPlan::dnnRecognizedPartitionCount() const noexcept {
	return dnnPlan_ ? dnnPlan_->recognizedPartitionCount : 0U;
}

oa::Status oa::ExecutionPlan::compile(oa::Engine& inEngine) {
	if (compiled_) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition, "execution plan is already compiled");
	}

	engine_ = &inEngine;
	const auto status = graph_->compile(inEngine);

	if (status.isOk()) {
		compiled_ = true;
	}

	return status;
}

oa::Status oa::ExecutionPlan::replay() {
	if (not compiled_ or engine_ == nullptr) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition, "execution plan replay called before compile");
	}
	return graph_->replay(*engine_);
}

oa::Result<oa::Event> oa::ExecutionPlan::replayAsync() {
	if (not compiled_ or engine_ == nullptr) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition, "execution plan asynchronous replay called before compile");
	}
	return graph_->replayAsync(*engine_);
}

oa::Status oa::ExecutionPlan::wait() {
	if (not compiled_ or engine_ == nullptr) {
		return oa::Status::ok();
	}
	return graph_->waitForPendingReplay(*engine_);
}

oa::Status oa::ExecutionPlan::reset() {
	if (not graph_) {
		dnnPlan_.reset();
		return oa::Status::ok();
	}

	if (engine_ != nullptr) {
		OA_RETURN_IF_ERROR(graph_->waitForPendingReplay(*engine_));
		graph_->reset(*engine_);
	} else {
		graph_->reset();
	}

	engine_ = nullptr;
	compiled_ = false;
	dnnPlan_.reset();

	return oa::Status::ok();
}
