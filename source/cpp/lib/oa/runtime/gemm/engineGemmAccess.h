#pragma once

#include "../engine/engineAccess.h"

namespace oa {

// Private bridge for GEMM lowering and its focused conformance tests. Public
// callers express numeric intent; they never inspect route capability bits.
class EngineGemmAccess {
public:
	[[nodiscard]] static oa::U64 capsMask(const oa::Engine& inEngine) {
		return oa::EngineAccess::gemmCapsMask(inEngine);
	}
};

} // namespace oa
