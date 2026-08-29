#include <oa/core/std/assert.h>

#include <stdio.h>
#include <stdlib.h>

[[noreturn]] void oa::detail::assertionFailed(
	const char* inExpression,
	const char* inFile,
	int inLine
) noexcept {
	::fprintf(stderr, "OA assertion failed: %s (%s:%d)\n",
		inExpression, inFile, inLine);
	::fflush(stderr);
	::abort();
}

[[noreturn]] void oa::detail::contractFailed(
	const char* inExpression,
	const char* inMessage,
	const char* inFile,
	int inLine
) noexcept {
	if (inMessage != nullptr && inMessage[0] != '\0') {
		::fprintf(stderr, "OA contract failed: %s: %s (%s:%d)\n",
			inExpression, inMessage, inFile, inLine);
	} else {
		::fprintf(stderr, "OA contract failed: %s (%s:%d)\n",
			inExpression, inFile, inLine);
	}
	::fflush(stderr);
	::abort();
}
