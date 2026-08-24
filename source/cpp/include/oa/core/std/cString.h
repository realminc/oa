#pragma once

// OA standard library C-string scanning — clean-room, auditable, no <cstring> dependency.
// These are genuinely native (simple loops the compiler vectorizes); bulk byte
// copy/zero lives in oa::memcpy / oa::memzero (Memory.h), not here.

#define OA_TYPES_H_SKIP_REST
#include <oa/core/types.h>
#undef OA_TYPES_H_SKIP_REST

namespace oa {

[[nodiscard]] inline oa::Usize strlen(const char* inStr) {
	if (inStr == nullptr) {
		return 0;
	}
	const char* p = inStr;
	while (*p != '\0') {
		++p;
	}
	return static_cast<oa::Usize>(p - inStr);
}

[[nodiscard]] inline oa::I32 strcmp(const char* inA, const char* inB) {
	while (*inA != '\0' && *inA == *inB) {
		++inA;
		++inB;
	}
	return static_cast<oa::I32>(static_cast<unsigned char>(*inA))
	     - static_cast<oa::I32>(static_cast<unsigned char>(*inB));
}

[[nodiscard]] inline oa::I32 strncmp(const char* inA, const char* inB, oa::Usize inN) {
	for (oa::Usize i = 0; i < inN; ++i) {
		const unsigned char a = static_cast<unsigned char>(inA[i]);
		const unsigned char b = static_cast<unsigned char>(inB[i]);
		if (a != b) {
			return static_cast<oa::I32>(a) - static_cast<oa::I32>(b);
		}
		if (a == 0) {
			return 0;   // both hit the terminator together
		}
	}
	return 0;
}

[[nodiscard]] inline const char* strchr(const char* inStr, char inCh) {
	for (; *inStr != '\0'; ++inStr) {
		if (*inStr == inCh) {
			return inStr;
		}
	}
	return (inCh == '\0') ? inStr : nullptr;   // matches strchr: finds the '\0'
}

} // namespace oa
