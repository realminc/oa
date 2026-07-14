#pragma once

// OaStd C-string scanning — clean-room, auditable, no <cstring> dependency.
// These are genuinely native (simple loops the compiler vectorizes); bulk byte
// copy/zero lives in OaMemcpy / OaMemzero (Memory.h), not here.

#define OA_TYPES_H_SKIP_REST
#include <Oa/Core/Types.h>
#undef OA_TYPES_H_SKIP_REST

[[nodiscard]] inline OaUsize OaStdStrlen(const char* InStr) {
	if (InStr == nullptr) {
		return 0;
	}
	const char* p = InStr;
	while (*p != '\0') {
		++p;
	}
	return static_cast<OaUsize>(p - InStr);
}

[[nodiscard]] inline OaI32 OaStdStrcmp(const char* InA, const char* InB) {
	while (*InA != '\0' && *InA == *InB) {
		++InA;
		++InB;
	}
	return static_cast<OaI32>(static_cast<unsigned char>(*InA))
	     - static_cast<OaI32>(static_cast<unsigned char>(*InB));
}

[[nodiscard]] inline OaI32 OaStdStrncmp(const char* InA, const char* InB, OaUsize InN) {
	for (OaUsize i = 0; i < InN; ++i) {
		const unsigned char a = static_cast<unsigned char>(InA[i]);
		const unsigned char b = static_cast<unsigned char>(InB[i]);
		if (a != b) {
			return static_cast<OaI32>(a) - static_cast<OaI32>(b);
		}
		if (a == 0) {
			return 0;   // both hit the terminator together
		}
	}
	return 0;
}

[[nodiscard]] inline const char* OaStdStrchr(const char* InStr, char InCh) {
	for (; *InStr != '\0'; ++InStr) {
		if (*InStr == InCh) {
			return InStr;
		}
	}
	return (InCh == '\0') ? InStr : nullptr;   // matches strchr: finds the '\0'
}

[[nodiscard]] inline OaI32 OaStdMemcmp(const void* InA, const void* InB, OaUsize InN) {
	const unsigned char* a = static_cast<const unsigned char*>(InA);
	const unsigned char* b = static_cast<const unsigned char*>(InB);
	for (OaUsize i = 0; i < InN; ++i) {
		if (a[i] != b[i]) {
			return static_cast<OaI32>(a[i]) - static_cast<OaI32>(b[i]);
		}
	}
	return 0;
}
