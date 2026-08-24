#pragma once

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

[[nodiscard]] inline bool testForbidSkips() noexcept {
	const char* value = std::getenv("OA_TEST_FORBID_SKIPS");
	return value != nullptr
		and value[0] != '\0'
		and std::strcmp(value, "0") != 0;
}

[[nodiscard]] inline int runAllTests() {
	const int result = RUN_ALL_TESTS();
	const int skipped = testing::UnitTest::GetInstance()->skipped_test_count();
	if (testForbidSkips() and skipped != 0) {
		std::fprintf(stderr,
			"OA native profile forbids skips: %d test case(s) skipped\n",
			skipped);
		return 1;
	}
	return result;
}
