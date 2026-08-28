#pragma once

#include "engineAccess.h"

namespace oa {

// Private bridge for vulkan lowering and allocator conformance tests that need
// raw VMA identity, alias allocation, or explicit mapped-memory coherence.
// high-level callers allocate, upload, read back, and free through oa::Engine.
class EngineAllocatorAccess {
public:
	[[nodiscard]] static RuntimeAllocator& get(oa::Engine& inEngine) noexcept {
		return oa::EngineAccess::get(inEngine).allocator_;
	}

	[[nodiscard]] static const RuntimeAllocator& get(
		const oa::Engine& inEngine) noexcept {
		return oa::EngineAccess::get(inEngine).allocator_;
	}
};

} // namespace oa
