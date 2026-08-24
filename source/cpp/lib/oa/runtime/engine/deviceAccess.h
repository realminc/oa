#pragma once

#include "engineAccess.h"

namespace oa {

// Private bridge for vulkan lowering and focused white-box tests that require
// exact physical/logical device or queue identity. Public callers use semantic
// oa::Engine capability queries and never borrow the raw device aggregate.
class EngineDeviceAccess {
public:
	[[nodiscard]] static oavk::Device& get(oa::Engine& inEngine) noexcept {
		return oa::EngineAccess::get(inEngine).device_;
	}

	[[nodiscard]] static const oavk::Device& get(
		const oa::Engine& inEngine) noexcept
	{
		return oa::EngineAccess::get(inEngine).device_;
	}
};

} // namespace oa
