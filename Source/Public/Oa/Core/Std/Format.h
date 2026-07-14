#pragma once

// Number → OaStdString and printf-style OaFormat, without std::to_string /
// std::ostringstream / std::format (no exceptions, no heavy includes — just
// snprintf/vsnprintf). This is the ergonomic string-building surface: prefer
// OaFormat(...) over hand-rolled snprintf-into-char-buffer at call sites.

#define OA_TYPES_H_SKIP_REST
#include <Oa/Core/Types.h>
#undef OA_TYPES_H_SKIP_REST

#include <Oa/Core/Std/String.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>

[[nodiscard]] inline OaStdString OaToString(OaU32 InV) {
	char buf[16];
	if (std::snprintf(buf, sizeof(buf), "%u", static_cast<unsigned>(InV)) <= 0) {
		return {};
	}
	return OaStdString(buf);
}

[[nodiscard]] inline OaStdString OaToString(OaI64 InV) {
	char buf[24];
	if (std::snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(InV)) <= 0) {
		return {};
	}
	return OaStdString(buf);
}

// Floating point → OaStdString. "%g" gives a compact form (no trailing zeros,
// switches to exponent for very large/small). A 32-byte buffer holds any double
// "%g" output, so there is no truncation.
[[nodiscard]] inline OaStdString OaToString(double InV) {
	char buf[32];
	if (std::snprintf(buf, sizeof(buf), "%g", InV) <= 0) {
		return {};
	}
	return OaStdString(buf);
}

[[nodiscard]] inline OaStdString OaToString(float InV) {
	return OaToString(static_cast<double>(InV));
}

// printf-style formatting → OaStdString. Replaces the snprintf-into-char-buffer
// idiom: `OaString s = OaFormat("%s: %d (%.2f)", name, id, ratio);`. Measures
// the exact length, formats on the stack for the common small case, and falls
// back to a heap scratch buffer (freed before return) for long results. The
// format attribute lets the compiler type-check the varargs against the string.
[[nodiscard]]
#if defined(__GNUC__) || defined(__clang__)
__attribute__((format(printf, 1, 2)))
#endif
inline OaStdString OaFormat(const char* InFmt, ...) {
	va_list args;
	va_start(args, InFmt);
	va_list argsCopy;
	va_copy(argsCopy, args);

	const int needed = std::vsnprintf(nullptr, 0, InFmt, args);
	va_end(args);
	if (needed <= 0) {
		va_end(argsCopy);
		return {};
	}
	const OaStdString::size_type len = static_cast<OaStdString::size_type>(needed);

	char stack[256];
	if (len < sizeof(stack)) {
		std::vsnprintf(stack, sizeof(stack), InFmt, argsCopy);
		va_end(argsCopy);
		return OaStdString(stack, len);
	}

	char* heap = static_cast<char*>(std::malloc(len + 1));
	if (heap == nullptr) {
		va_end(argsCopy);
		return {};
	}
	std::vsnprintf(heap, len + 1, InFmt, argsCopy);
	va_end(argsCopy);
	OaStdString out(heap, len);
	std::free(heap);
	return out;
}
