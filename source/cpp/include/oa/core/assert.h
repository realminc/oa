#pragma once

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

#ifndef OA_ASSERT
	#ifdef NDEBUG
		#define OA_ASSERT(expression) ((void)0)
	#else
		#define OA_ASSERT(expression) \
			((expression) ? (void)0 : ::oa::detail::assertionFailed( \
				#expression, __FILE__, __LINE__))
	#endif
#endif

// Always-on precondition for public operations whose failure cannot be
// represented by the return type. Unlike OA_ASSERT, this remains active in
// Release builds and evaluates its expression exactly once.
#ifndef OA_REQUIRE
	#define OA_REQUIRE(expression) \
		((expression) ? (void)0 : ::oa::detail::contractFailed( \
			#expression, nullptr, __FILE__, __LINE__))
#endif

#ifndef OA_REQUIRE_MSG
	#define OA_REQUIRE_MSG(expression, message) \
		((expression) ? (void)0 : ::oa::detail::contractFailed( \
			#expression, message, __FILE__, __LINE__))
#endif
