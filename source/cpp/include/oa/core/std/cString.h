#pragma once

// OA standard library C-string scanning without a <cstring> dependency. Compiler
// builtins preserve the platform C-string contract and select its qualified
// implementation; bulk byte copy/zero remains in oa::memcpy / oa::memzero.

#define OA_TYPES_H_SKIP_REST
#include <oa/core/types.h>
#undef OA_TYPES_H_SKIP_REST

#include <oa/core/assert.h>

namespace oa {

[[nodiscard]] inline oa::Usize strlen(const char* inStr) {
	if (inStr == nullptr) {
		return 0;
	}
	return static_cast<oa::Usize>(__builtin_strlen(inStr));
}

[[nodiscard]] inline oa::I32 strcmp(const char* inA, const char* inB) {
	OA_REQUIRE_MSG(inA != nullptr, "strcmp requires a left C string");
	OA_REQUIRE_MSG(inB != nullptr, "strcmp requires a right C string");
	return static_cast<oa::I32>(__builtin_strcmp(inA, inB));
}

[[nodiscard]] inline oa::I32 strncmp(const char* inA, const char* inB, oa::Usize inN) {
	if (inN == 0) return 0;
	OA_REQUIRE_MSG(inA != nullptr, "strncmp requires a left byte range");
	OA_REQUIRE_MSG(inB != nullptr, "strncmp requires a right byte range");
	return static_cast<oa::I32>(__builtin_strncmp(inA, inB, inN));
}

[[nodiscard]] inline const char* strchr(const char* inStr, char inCh) {
	OA_REQUIRE_MSG(inStr != nullptr, "strchr requires a C string");
	return __builtin_strchr(inStr, static_cast<unsigned char>(inCh));
}

} // namespace oa
