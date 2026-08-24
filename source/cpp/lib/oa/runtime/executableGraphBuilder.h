#pragma once

#include <oa/core/status.h>
#include <oa/runtime/dispatchDesc.h>

namespace oa {

class ExecutableGraph;

// Internal recording boundary. It validates and copies executable dispatch
// descriptors into the currently attached graph, but owns no engine,
// submission state, or graph lifetime. oa::ExecutionSession remains a compatibility
// facade while recording and execution are progressively separated.
class ExecutableGraphBuilder {
public:
	explicit ExecutableGraphBuilder(ExecutableGraph* inGraph = nullptr) noexcept
		: graph_(inGraph) {}

	void attach(oa::ExecutableGraph* inGraph) noexcept { graph_ = inGraph; }
	[[nodiscard]] oa::ExecutableGraph* graph() const noexcept { return graph_; }
	[[nodiscard]] oa::Status record(const oa::ComputeDispatchDesc& inDesc);

private:
	oa::ExecutableGraph* graph_ = nullptr;
};

} // namespace oa
