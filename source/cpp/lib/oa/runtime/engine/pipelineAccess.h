#pragma once

#include "engineAccess.h"

namespace oa {

// Private bridge for executable lowering and focused white-box tests. pipeline
// lookup is internal engine policy; public callers request kernels through the
// engine execution surface instead of mutating the registry.
class EnginePipelineAccess {
public:
	[[nodiscard]] static oa::Status ensure(
		oa::Engine& inEngine,
		oa::StringView inName,
		oa::Span<const oa::U8> inSpirv,
		const oa::PipelineSpec& inSpec)
	{
		return oa::EngineAccess(inEngine).ensurePipeline(inName, inSpirv, inSpec);
	}

	[[nodiscard]] static oa::PipelineRegistry& get(oa::Engine& inEngine) noexcept {
		return oa::EngineAccess::get(inEngine).pipelines_;
	}

	[[nodiscard]] static const oa::PipelineRegistry& get(
		const oa::Engine& inEngine) noexcept
	{
		return oa::EngineAccess::get(inEngine).pipelines_;
	}
};

} // namespace oa
