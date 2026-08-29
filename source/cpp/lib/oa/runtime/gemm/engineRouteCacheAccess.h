#pragma once

#include "../engine/engineAccess.h"
#include "routeCache.h"

#include <oa/core/std/assert.h>

// Private bridge for GEMM lowering and its focused tests. The learned route
// cache is engine-owned implementation policy, not part of oa::Engine's public
// execution-owner contract.
class oa::Engine::Impl::GemmState {
public:
	oa::UniquePtr<oa::GemmRouteCache> routeCache;
};

namespace oa {

class GemmRouteCacheAccess {
public:
	[[nodiscard]] static oa::GemmRouteCache* get(oa::Engine& inEngine) noexcept {
		auto& impl = oa::EngineAccess::get(inEngine);
		return impl.gemmState_
			? impl.gemmState_->routeCache.get()
			: nullptr;
	}

	[[nodiscard]] static const oa::GemmRouteCache* get(
		const oa::Engine& inEngine) noexcept
	{
		const auto& impl = oa::EngineAccess::get(inEngine);
		return impl.gemmState_
			? impl.gemmState_->routeCache.get()
			: nullptr;
	}

	// Focused router tests replace the cache with isolated owned state and
	// restore the previous owner at scope exit. Production code does not call it.
	[[nodiscard]] static oa::UniquePtr<oa::GemmRouteCache> replaceForTesting(
		oa::Engine& inEngine,
		oa::UniquePtr<oa::GemmRouteCache>&& inReplacement) noexcept
	{
		auto& impl = oa::EngineAccess::get(inEngine);
		OA_REQUIRE(impl.gemmState_);
		auto previous = oa::move(impl.gemmState_->routeCache);
		impl.gemmState_->routeCache = oa::move(inReplacement);
		return previous;
	}
};

} // namespace oa
