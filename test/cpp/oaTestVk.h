// oaTestVk.h — TEST_VK alias for OA's engine-initializing test suites.
// Lives in Test/ — does NOT ship with the library.
//
// TEST_VK is a thin, purely-cosmetic alias for GoogleTest's fixture macro
// TEST_F. It marks a test suite that initializes the OA vulkan device (the
// oa_add_ml_test targets), distinguishing those from pure-CPU gtest fixtures
// which keep TEST_F. It expands verbatim to TEST_F, so every GoogleTest
// fixture semantic (SetUp/TearDown, the fixture class, etc.) applies
// unchanged.
//
// Visible everywhere via two paths:
//   • oaTest.h #includes this header (for suites that include oaTest.h)
//   • oa_add_ml_test force-includes it (-include) so gtest-direct and
//     generated .gen.cpp sources see it without per-file include churn.

#pragma once

#include <gtest/gtest.h>
#include <oa/core/std/string.h>

#include <ostream>

// GoogleTest formats assertion diagnostics through hosted iostreams. Keep
// that adapter in the test harness so oa::String itself remains independent
// of the hosted standard string and stream surface.
#ifndef OA_HOSTED_TEXT_STREAM_OPERATORS
#define OA_HOSTED_TEXT_STREAM_OPERATORS

namespace oa {

inline std::ostream& operator<<(std::ostream& inOut, oa::StringView inValue) {
	if (not inValue.empty()) {
		inOut.write(inValue.data(), static_cast<std::streamsize>(inValue.size()));
	}
	return inOut;
}

inline std::ostream& operator<<(std::ostream& inOut, const oa::String& inValue) {
	return inOut << inValue.view();
}

} // namespace oa

#endif // OA_HOSTED_TEXT_STREAM_OPERATORS

#define TEST_VK(Fixture, Name) TEST_F(Fixture, Name)
