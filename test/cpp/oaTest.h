// Shared test helpers built on Google Test.
// Lives in Test/ — does NOT ship with the library.
//
// usage:
//   #include "oaTest.h"              (from Test/Core/, Test/Ml/, …)
//   #include "../oaTest.h"           (one level deeper)
//   #include "oaStdTest.h"          (from Test/Core/std/Test*.cpp — oaTest.h + <oa/core/std.h>)

#pragma once

#include <gtest/gtest.h>
#include "oaTestVk.h"  // TEST_VK alias for engine-initializing suites
#include <oa/oa.h>
#include <oa/runtime/dispatch.h>
#include <oa/runtime/pipeline.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/engine/deviceAccess.h>
#include <oa/runtime/engine/resourceAccess.h>
#include <oa/core/filesystem.h>
#include <oa/core/paths.h>
#include <chrono>
#include <cassert>
#include <functional>
#include <ostream>
#include <string>

[[nodiscard]] inline std::string testStdString(oa::StringView inText) {
	return std::string(inText.data(), inText.size());
}
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>

// Test-only diagnostic interop. Production OA text types deliberately do not
// include or expose hosted iostream machinery.
#ifndef OA_HOSTED_TEXT_STREAM_OPERATORS
#define OA_HOSTED_TEXT_STREAM_OPERATORS

namespace oa {

inline std::ostream& operator<<(std::ostream& inOut, oa::StringView inValue) {
	if (!inValue.empty()) {
		inOut.write(inValue.data(), static_cast<std::streamsize>(inValue.size()));
	}
	return inOut;
}

inline std::ostream& operator<<(std::ostream& inOut, const oa::String& inValue) {
	return inOut << inValue.view();
}

} // namespace oa

#endif // OA_HOSTED_TEXT_STREAM_OPERATORS

// OA_DEVICE — optional vulkan device selection for tests/benches (same semantics as oa::EngineConfig).
//   integrated | igpu  → oa::DevicePreference::Integrated
//   discrete | dgpu   → oa::DevicePreference::Discrete
//   cpu                 → oa::DevicePreference::Cpu
//   0, 1, 2, …          → oa::DevicePreference::ByIndex + deviceIndex
//
// Use testMergeDeviceEnv(engineConfig_) from oa::ComputeApp::Setup to align app runs with gtest device pick.

static inline void testMergeDeviceEnv(oa::EngineConfig& inOut) {
	const char* dev = std::getenv("OA_DEVICE");
	if (!dev || !*dev) return;
	if (std::strcmp(dev, "integrated") == 0 || std::strcmp(dev, "igpu") == 0) {
		inOut.devicePref = oa::DevicePreference::Integrated;
		return;
	}
	if (std::strcmp(dev, "discrete") == 0 || std::strcmp(dev, "dgpu") == 0) {
		inOut.devicePref = oa::DevicePreference::Discrete;
		return;
	}
	if (std::strcmp(dev, "cpu") == 0) {
		inOut.devicePref = oa::DevicePreference::Cpu;
		return;
	}
	char* end = nullptr;
	unsigned long idx = std::strtoul(dev, &end, 10);
	if (end != dev && *end == '\0' && idx <= 0xFFFFu) {
		inOut.devicePref = oa::DevicePreference::ByIndex;
		inOut.deviceIndex = static_cast<oa::U32>(idx);
	}
}

static inline oa::EngineConfig testEngineConfig(oa::Precision inPrecision) {
	oa::EngineConfig cfg;
	cfg.precision = inPrecision;
	cfg.appName = "OaTest";
	if (const char* v = std::getenv("OA_VK_VALIDATION");
		v != nullptr && v[0] == '1')
	{
		cfg.enableValidation = true;
	}
	// Validation evidence should instrument the workload under test, not spend
	// most of the run eagerly compiling every unrelated embedded shader. The
	// runtime's lazy path preserves the same pipeline ABI and is independently
	// exercised whenever the selected operation is first recorded.
	if (const char* mode = std::getenv("OA_VK_VALIDATION_MODE");
		mode != nullptr && mode[0] != '\0')
	{
		cfg.preloadEmbeddedPipelines = false;
	}
	testMergeDeviceEnv(cfg);
	return cfg;
}

// Overload: explicit numericMode for accuracy/determinism tests.
static inline oa::EngineConfig testEngineConfig(oa::Precision inPrecision, oa::NumericMode inMode) {
	oa::EngineConfig cfg = testEngineConfig(inPrecision);
	cfg.numericMode = inMode;
	return cfg;
}

// Complete exactly the work recorded by a test context. submit first so a
// recording failure or missing lowering remains visible; only the canonical
// "no recorded device work" result is normalized to an empty-test no-op.
[[nodiscard]] static inline oa::Status testSubmitAndWait(oa::ExecutionSession& inContext) {
	auto submitted = inContext.submit();
	if (not submitted.isOk()) {
		const auto& status = submitted.getStatus();
		if (
			inContext.nodeCount() == 0U
			and status.getCode() == oa::StatusCode::FailedPrecondition
			and status.getMessage()
				== "oa::ExecutionSession::submit requires recorded device work"
		)
		{
			return oa::Status::ok();
		}
		return status;
	}
	return inContext.wait(submitted.getValue());
}

// ─── Device-aware tolerance helpers (oaNumericStability.md §9.1) ─────────────
//
// Tier A (elementwise) and Tier B (reductions) bounds scale with iGPU vs dGPU
// and with subgroup size for BF16 reductions. Tier C (GEMM) BF16 bound also
// scales with the discovered CoopMat-K and the GEMM's K_total.
//
// Use these instead of EXPECT_NEAR with a flat constant in any cross-vendor
// kernel test. On RTX 5090 they produce identical bounds to the original
// constants; on Ada/RDNA3/Xe2/iGPU they widen automatically.

// Returns 1.5 for integrated GPUs, 1.0 for discrete and unknown.
[[nodiscard]] static inline float testToleranceDeviceScale(const oavk::Device& inDevice) {
	return inDevice.info.hardware.deviceType == oa::DeviceType::VkIntegrated ? 1.5F : 1.0F;
}

// Subgroup-size scale for BF16 reductions per §4.3.
[[nodiscard]] static inline float testToleranceSubgroupScale(const oavk::Device& inDevice) {
	const oa::U32 sg = inDevice.info.hardware.subgroupSize;
	if (sg >= 64) return 1.5F;
	if (sg <= 16) return 0.75F;
	return 1.0F;
}

// Tier A — elementwise BF16. base 5e-4.
[[nodiscard]] static inline float testToleranceElemwiseBf16(const oavk::Device& inDevice) {
	return 5.0e-4F * testToleranceDeviceScale(inDevice);
}

// Tier B — reductions BF16. base 2e-3, scales with subgroup size.
[[nodiscard]] static inline float testToleranceReduceBf16(const oavk::Device& inDevice) {
	return 2.0e-3F * testToleranceSubgroupScale(inDevice) * testToleranceDeviceScale(inDevice);
}

// Tier C — GEMM BF16. base depends on per-tile CoopMat-K, scales by sqrt(K_total/K_tile).
[[nodiscard]] static inline float testToleranceGemmCmSgBf16(const oavk::Device& inDevice, oa::U32 inKTotal) {
	const auto& bf16Shape = inDevice.info.software.coopMatShapes.bf16AccFp32;
	const oa::U32 kTile = bf16Shape.available ? bf16Shape.k : 16U;
	const float base = kTile == 16 ? 2.0e-2F : 3.0e-2F;  // K=32 (RDNA3) bound is wider
	const float kScale = static_cast<float>(std::sqrt(
		static_cast<double>(inKTotal) / static_cast<double>(kTile)));
	return base * kScale * testToleranceDeviceScale(inDevice);
}

#define EXPECT_NEAR_ELEMWISE_BF16(actual, expected, rt) \
	EXPECT_NEAR((actual), (expected), testToleranceElemwiseBf16( \
		oa::EngineDeviceAccess::get((rt))))

#define EXPECT_NEAR_REDUCE_BF16(actual, expected, rt) \
	EXPECT_NEAR((actual), (expected), testToleranceReduceBf16( \
		oa::EngineDeviceAccess::get((rt))))

#define EXPECT_NEAR_GEMM_BF16(actual, expected, rt, K_total) \
	EXPECT_NEAR((actual), (expected), testToleranceGemmCmSgBf16( \
		oa::EngineDeviceAccess::get((rt)), (K_total)))

// Tier D — vision color conversion (uint8 output): max abs error <= 2/255 ≈ 2 lsb.
#define EXPECT_NEAR_VISION_U8(actual, expected) \
	EXPECT_LE(std::abs(static_cast<int>(actual) - static_cast<int>(expected)), 2)

static inline oa::Path testAssetPath(oa::StringView inRelativePath) {
	return oa::Paths::asset(inRelativePath);
}

// Test-only suite ownership. This inline variable is shared across translation
// units in one test executable and deliberately does not leak into liboa.
inline oa::Engine* testVkEngine = nullptr;

[[nodiscard]] static inline oa::Engine* testEnginePtr() noexcept {
	return testVkEngine;
}

[[nodiscard]] static inline oa::Engine& testEngine() {
	assert(testVkEngine != nullptr
		&& "VkTestEnvironment did not create the suite engine");
	return *testVkEngine;
}

// True after VkTestEnvironment::SetUp when oa::Engine::Create succeeded.
static inline bool vkTestEngineOk() {
	return testVkEngine != nullptr
		&& oa::EngineDeviceAccess::get(*testVkEngine).device != nullptr;
}

// Matrix Assertions — use at() for BF16 safety

static inline void expectMatrixNear(const oa::Matrix& inA, const oa::Matrix& inB, oa::F32 inEps = 1e-5f) {
	ASSERT_EQ(inA.getShape(), inB.getShape()) << "Shape mismatch";
	oa::I64 n = inA.numElements();
	for (oa::I64 i = 0; i < n; ++i) {
		EXPECT_NEAR(inA.at(i), inB.at(i), inEps) << "Mismatch at index " << i;
	}
}

static inline void expectShape(const oa::Matrix& inMatrix, std::initializer_list<oa::I64> inExpected) {
	oa::MatrixShape expected;
	expected.rank = static_cast<oa::I32>(inExpected.size());
	oa::Usize index = 0;
	for (const oa::I64 dimension : inExpected) {
		expected.dims[index++] = dimension;
	}
	EXPECT_EQ(inMatrix.getShape(), expected) << "Shape mismatch";
}

static inline void expectZero(const oa::Matrix& inMatrix) {
	for (oa::I64 i = 0; i < inMatrix.numElements(); ++i) {
		EXPECT_FLOAT_EQ(inMatrix.at(i), 0.0f) << "Non-zero at index " << i;
	}
}

static inline void expectFinite(const oa::Matrix& inTensor) {
	for (oa::I64 i = 0; i < inTensor.numElements(); ++i) {
		EXPECT_TRUE(std::isfinite(inTensor.at(i))) << "NaN/Inf at index " << i;
	}
}

static inline void expectValidProbability(const oa::Matrix& inTensor, oa::I32 inDim = -1) {
	(void)inDim;
	oa::I64 lastDim = inTensor.getShape()[inTensor.rank() - 1];
	oa::I64 batches = inTensor.numElements() / lastDim;
	for (oa::I64 b = 0; b < batches; ++b) {
		oa::F32 sum = 0.0f;
		for (oa::I64 i = 0; i < lastDim; ++i) {
			oa::F32 v = inTensor.at(b * lastDim + i);
			EXPECT_GE(v, 0.0f) << "Negative probability at batch " << b << " index " << i;
			sum += v;
		}
		EXPECT_NEAR(sum, 1.0f, 1e-4f) << "Probabilities don't sum to 1 at batch " << b;
	}
}

// Benchmarking

static inline double benchmark(const char* inName, oa::I32 inIterations, std::function<void()> inFunc) {
	for (oa::I32 i = 0; i < 3; ++i) inFunc();

	auto start = std::chrono::high_resolution_clock::now();
	for (oa::I32 i = 0; i < inIterations; ++i) inFunc();
	auto end = std::chrono::high_resolution_clock::now();

	double ms = std::chrono::duration<double, std::milli>(end - start).count();
	double avg = ms / inIterations;
	printf("  %-36s %8.3f ms  (%d iters, %.1f ms total)\n", inName, avg, inIterations, ms);
	return avg;
}

// vulkan Engine Fixture — init oa::Engine for all ML tests (lavapipe for CI)
// set OA_TEST_BF16=1 to run with BF16 precision.

class VkTestEnvironment : public ::testing::Environment {
public:
	void SetUp() override {
		testVkEngine = nullptr;
		oa::Precision prec = oa::Precision::FP32;
		const char* bf16Env = std::getenv("OA_TEST_BF16");
		if (bf16Env && oa::String(bf16Env) == "1") {
			prec = oa::Precision::BF16;
		}
		oa::EngineConfig ecfg = testEngineConfig(prec);
		auto result = oa::Engine::create(ecfg);
		if (!result) {
			fprintf(stderr,
				"VkTestEnvironment: Failed to create oa::Engine: %s\n",
				result.getStatus().toString().cStr());
			return;
		}
		engine_ = std::move(*result);   // take ownership of the pinned engine
		testVkEngine = engine_.get();
		if (const char* profile = std::getenv("OA_REQUIRE_DEVICE_PROFILE");
			profile != nullptr and profile[0] != '\0')
		{
			const auto& device = oa::EngineDeviceAccess::get(*engine_);
			if (std::strcmp(profile, "iris-xe-tgl") == 0) {
				EXPECT_EQ(device.info.hardware.vendorId, 0x8086U)
					<< "iris-xe-tgl profile requires an Intel device";
				EXPECT_EQ(device.info.hardware.deviceId, 0x9A49U)
					<< "iris-xe-tgl profile requires the validated TGL GT2 device";
				EXPECT_EQ(engine_->deviceType(), oa::DeviceType::VkIntegrated)
					<< "iris-xe-tgl profile requires the integrated-GPU route";
			} else {
				ADD_FAILURE() << "unknown OA_REQUIRE_DEVICE_PROFILE: " << profile;
			}
		}
		// create() preloaded the build-generated embedded shader registry and
		// selected the context owned by the suite engine.
	}

	void TearDown() override {
		// flush the engine-owned context before engine teardown. Close clears the
		// thread default before releasing the context, so no state survives the suite.
		if (oa::ExecutionSession::getActivePtr()) {
			auto& ctx = oa::ExecutionSession::getActive();
			(void)testSubmitAndWait(ctx);
			ctx.clear();
		}
		oa::ExecutionSession::setActive(nullptr);
		if (engine_) {
			const auto closeStatus = engine_->close();
			EXPECT_TRUE(closeStatus.isOk()) << closeStatus.toString();
			testVkEngine = nullptr;
			engine_.reset();
		}
	}

private:
	oa::UniquePtr<oa::Engine> engine_;
};

// Optional TEST_F base: asserts gtest global vulkan environment created the engine.
class VkEngineTestFixture : public ::testing::Test {
protected:
	void SetUp() override {
		ASSERT_TRUE(vkTestEngineOk()) << "VkTestEnvironment did not create oa::Engine (see stderr)";
	}

	static oa::Engine& rt() { return testEngine(); }
};

// Test Helpers

static inline oa::Matrix makeByteIndices(std::initializer_list<oa::U8> inValues) {
	auto t = oa::FnMatrix::empty(oa::MatrixShape{static_cast<oa::I64>(inValues.size())}, oa::ScalarType::UInt8);
	oa::U8* data = t.dataAs<oa::U8>();
	oa::I64 i = 0;
	for (auto v : inValues) data[i++] = v;
	return t;
}

static inline oa::Matrix makeTestTensor(oa::MatrixShape inShape, oa::F32 inStart = 0.0f, oa::F32 inStep = 1.0f) {
	auto t = oa::FnMatrix::empty(inShape);
	for (oa::I64 i = 0; i < t.numElements(); ++i) {
		t.set(i, inStart + static_cast<oa::F32>(i) * inStep);
	}
	return t;
}
