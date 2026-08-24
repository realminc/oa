#pragma once

#include "engineAccess.h"

namespace oa {

// Private bridge for vulkan lowering that must allocate or bind descriptor
// heap slots. high-level callers never select, replace, or inspect this state.
class EngineBindlessAccess {
public:
	[[nodiscard]] static oa::U32 registerBuffer(
		oa::Engine& inEngine,
		oavk::Buffer& inOutBuffer)
	{
		return oa::EngineAccess(inEngine).registerBuffer(inOutBuffer);
	}

	[[nodiscard]] static oa::Status updateBufferDescriptor(
		oa::Engine& inEngine,
		const oavk::Buffer& inBuffer)
	{
		return oa::EngineAccess(inEngine).updateBufferDescriptor(inBuffer);
	}

	static void deregisterBuffer(
		oa::Engine& inEngine,
		oavk::Buffer& inOutBuffer)
	{
		oa::EngineAccess(inEngine).deregisterBuffer(inOutBuffer);
	}

	[[nodiscard]] static oavk::BindlessHeap& get(oa::Engine& inEngine) noexcept {
		return oa::EngineAccess::get(inEngine).bindless_;
	}

	[[nodiscard]] static const oavk::BindlessHeap& get(
		const oa::Engine& inEngine) noexcept {
		return oa::EngineAccess::get(inEngine).bindless_;
	}
};

} // namespace oa
