#pragma once

// OA foundation contracts. This header deliberately has no dependency on the
// rest of OA so scalar and container headers can use it during bootstrap.

namespace oa::detail {

[[noreturn]] void assertionFailed(
	const char* inExpression,
	const char* inFile,
	int inLine
) noexcept;

[[noreturn]] void contractFailed(
	const char* inExpression,
	const char* inMessage,
	const char* inFile,
	int inLine
) noexcept;

} // namespace oa::detail

// OA owns its prefixed contract macros. Accepting an earlier definition would
// let include order silently replace or disable the foundation failure policy.
#if defined(OA_ASSERT) || defined(OA_REQUIRE) || defined(OA_REQUIRE_MSG)
	#error "OA foundation contract macro was defined before <oa/core/std/assert.h>"
#endif

// Debug-only internal invariant. The expression is not evaluated in Release.
#ifdef NDEBUG
	#define OA_ASSERT(expression) ((void)0)
#else
	#define OA_ASSERT(expression) \
		((expression) ? (void)0 : ::oa::detail::assertionFailed( \
			#expression, __FILE__, __LINE__))
#endif

// Always-on precondition for operations whose failure cannot be represented by
// the return type. The expression is evaluated exactly once in every build.
#define OA_REQUIRE(expression) \
	((expression) ? (void)0 : ::oa::detail::contractFailed( \
		#expression, nullptr, __FILE__, __LINE__))

#define OA_REQUIRE_MSG(expression, message) \
	((expression) ? (void)0 : ::oa::detail::contractFailed( \
		#expression, message, __FILE__, __LINE__))
