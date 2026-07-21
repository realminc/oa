#pragma once

#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>
#include <Oa/Runtime/Sync.h>

class OaComputeGraph;
class OaEngine;

// Internal owner for one immutable compiled executable graph. Authoring and
// semantic compilation remain outside this object; after Compile succeeds the
// plan owns replay, exact completion, and Vulkan graph lifetime.
class OaExecutionPlan {
public:
	OaExecutionPlan();
	~OaExecutionPlan();

	OaExecutionPlan(const OaExecutionPlan&) = delete;
	OaExecutionPlan& operator=(const OaExecutionPlan&) = delete;
	OaExecutionPlan(OaExecutionPlan&&) = delete;
	OaExecutionPlan& operator=(OaExecutionPlan&&) = delete;

	[[nodiscard]] OaComputeGraph& Graph() noexcept;
	[[nodiscard]] const OaComputeGraph& Graph() const noexcept;
	[[nodiscard]] OaStatus Compile(OaEngine& InEngine);
	[[nodiscard]] OaStatus Replay();
	[[nodiscard]] OaResult<OaCompletionToken> ReplayAsync();
	[[nodiscard]] OaStatus Wait();

	// Explicit reset is a host completion boundary. Destruction never waits: a
	// submitted graph is transferred to engine retirement instead.
	[[nodiscard]] OaStatus Reset();
	[[nodiscard]] OaBool IsCompiled() const noexcept { return Compiled_; }

private:
	OaUniquePtr<OaComputeGraph> Graph_;
	OaEngine* Engine_ = nullptr;
	OaBool Compiled_ = false;
};
