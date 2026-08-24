#pragma once

// Number → oa::String and printf-style format, without std::to_string /
// std::ostringstream / std::format (no exceptions, no heavy includes — just
// snprintf/vsnprintf). This is the ergonomic string-building surface: prefer
// format(...) over hand-rolled snprintf-into-char-buffer at call sites.

#define OA_TYPES_H_SKIP_REST
#include <oa/core/types.h>
#undef OA_TYPES_H_SKIP_REST

#include <oa/core/std/string.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>

namespace oa {

[[nodiscard]] inline String toString(U32 inV) {
	char buf[16];
	if (std::snprintf(buf, sizeof(buf), "%u", static_cast<unsigned>(inV)) <= 0) {
		return {};
	}
	return oa::String(buf);
}

[[nodiscard]] inline String toString(I64 inV) {
	char buf[24];
	if (std::snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(inV)) <= 0) {
		return {};
	}
	return oa::String(buf);
}

// Floating point → oa::String. "%g" gives a compact form (no trailing zeros,
// switches to exponent for very large/small). A 32-byte buffer holds any double
// "%g" output, so there is no truncation.
[[nodiscard]] inline String toString(double inV) {
	char buf[32];
	if (std::snprintf(buf, sizeof(buf), "%g", inV) <= 0) {
		return {};
	}
	return oa::String(buf);
}

[[nodiscard]] inline String toString(float inV) {
	return toString(static_cast<double>(inV));
}

// printf-style formatting → oa::String. Replaces the snprintf-into-char-buffer
// idiom: `oa::String s = format("%s: %d (%.2f)", name, id, ratio);`. Measures
// the exact length, formats on the stack for the common small case, and falls
// back to a heap scratch buffer (freed before return) for long results. The
// format attribute lets the compiler type-check the varargs against the string.
[[nodiscard]]
#if defined(__GNUC__) || defined(__clang__)
__attribute__((format(printf, 1, 2)))
#endif
inline String format(const char* inFmt, ...) {
	va_list args;
	va_start(args, inFmt);
	va_list argsCopy;
	va_copy(argsCopy, args);

	const int needed = std::vsnprintf(nullptr, 0, inFmt, args);
	va_end(args);
	if (needed <= 0) {
		va_end(argsCopy);
		return {};
	}
	const oa::String::size_type len = static_cast<oa::String::size_type>(needed);

	char stack[256];
	if (len < sizeof(stack)) {
		std::vsnprintf(stack, sizeof(stack), inFmt, argsCopy);
		va_end(argsCopy);
		return oa::String(stack, len);
	}

	char* heap = static_cast<char*>(std::malloc(len + 1));
	if (heap == nullptr) {
		va_end(argsCopy);
		return {};
	}
	std::vsnprintf(heap, len + 1, inFmt, argsCopy);
	va_end(argsCopy);
	oa::String out(heap, len);
	std::free(heap);
	return out;
}

} // namespace oa
