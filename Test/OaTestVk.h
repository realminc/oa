// OaTestVk.h — TEST_VK alias for OA's engine-initializing test suites.
// Lives in Test/ — does NOT ship with the library.
//
// TEST_VK is a thin, purely-cosmetic alias for GoogleTest's fixture macro
// TEST_F. It marks a test suite that initializes the OA Vulkan device (the
// oa_add_ml_test targets), distinguishing those from pure-CPU gtest fixtures
// which keep TEST_F. It expands verbatim to TEST_F, so every GoogleTest
// fixture semantic (SetUp/TearDown, the fixture class, etc.) applies
// unchanged.
//
// Visible everywhere via two paths:
//   • OaTest.h #includes this header (for suites that include OaTest.h)
//   • oa_add_ml_test force-includes it (-include) so gtest-direct and
//     generated .gen.cpp sources see it without per-file include churn.

#pragma once

#include <gtest/gtest.h>

#define TEST_VK(Fixture, Name) TEST_F(Fixture, Name)
