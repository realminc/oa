#pragma once

#include <oa/core/status.h>
#include <oa/core/types.h>
#include <oa/runtime/event.h>

namespace oa {

class ExecutableGraph;
struct DnnPlan;
class Engine;
class TrainingProgram;

// move-only compiled executable work captured by oa::Engine. The plan borrows
// its engine and owns the graph and retained resources through exact completion.
// Destruction never waits: submitted work transfers to engine retirement.
class ExecutionPlan {
public:
	ExecutionPlan();
	~ExecutionPlan();

	ExecutionPlan(const ExecutionPlan&) = delete;
	ExecutionPlan& operator=(const ExecutionPlan&) = delete;

	ExecutionPlan(ExecutionPlan&& inOther) noexcept;
	ExecutionPlan& operator=(ExecutionPlan&& inOther) noexcept;

	[[nodiscard]] oa::Bool isCompiled() const noexcept { return compiled_; }
	[[nodiscard]] oa::U32 nodeCount() const noexcept;
	// Read-only graph-compiler evidence. These queries do not expose or control
	// kernel/vendor selection and remain stable across plan replay.
	[[nodiscard]] oa::U64 dnnGraphHash() const noexcept;
	[[nodiscard]] oa::U32 dnnSourceOpCount() const noexcept;
	[[nodiscard]] oa::U32 dnnCapturedOpCount() const noexcept;
	[[nodiscard]] oa::U32 dnnPartitionCount() const noexcept;
	[[nodiscard]] oa::U32 dnnRecognizedPartitionCount() const noexcept;

private:
	friend class Engine;
	friend class TrainingProgram;

	[[nodiscard]] oa::ExecutableGraph& graph() noexcept;
	[[nodiscard]] const oa::ExecutableGraph& graph() const noexcept;
	[[nodiscard]] oa::Status compile(oa::Engine& inEngine);
	[[nodiscard]] oa::Status replay();
	[[nodiscard]] oa::Result<oa::Event> replayAsync();
	[[nodiscard]] oa::Status wait();

	// Explicit reset is a host completion boundary. Destruction never waits: a
	// submitted graph is transferred to engine retirement instead.
	[[nodiscard]] oa::Status reset();
	void release_() noexcept;

	oa::UniquePtr<oa::ExecutableGraph> graph_;
	UniquePtr<DnnPlan> dnnPlan_;
	oa::Engine* engine_ = nullptr;
	oa::Bool compiled_ = false;
};

} // namespace oa
