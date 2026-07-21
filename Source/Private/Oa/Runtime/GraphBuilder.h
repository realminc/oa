#pragma once

#include <Oa/Core/Status.h>
#include <Oa/Runtime/DispatchDesc.h>

class OaComputeGraph;

// Internal recording boundary. It validates and copies executable dispatch
// descriptors into the currently attached graph, but owns no engine,
// submission state, or graph lifetime. OaContext remains a compatibility
// facade while recording and execution are progressively separated.
class OaGraphBuilder {
public:
	explicit OaGraphBuilder(OaComputeGraph* InGraph = nullptr) noexcept
		: Graph_(InGraph) {}

	void Attach(OaComputeGraph* InGraph) noexcept { Graph_ = InGraph; }
	[[nodiscard]] OaComputeGraph* Graph() const noexcept { return Graph_; }
	[[nodiscard]] OaStatus Record(const OaComputeDispatchDesc& InDesc);

private:
	OaComputeGraph* Graph_ = nullptr;
};
