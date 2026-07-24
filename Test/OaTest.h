// Shared test helpers built on Google Test.
// Lives in Test/ — does NOT ship with the library.
//
// Usage:
//   #include "OaTest.h"              (from Test/Core/, Test/Ml/, …)
//   #include "../OaTest.h"           (one level deeper)
//   #include "OaStdTest.h"          (from Test/Core/Std/Test*.cpp — OaTest.h + <Oa/Core/Std.h>)

#pragma once

#include <gtest/gtest.h>
#include "OaTestVk.h"  // TEST_VK alias for engine-initializing suites
#include <Oa/Oa.h>
#include <Oa/Runtime/Pipeline.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Core/Filesystem.h>
#include <Oa/Core/Paths.h>
#include <chrono>
#include <functional>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>

// OA_DEVICE — optional Vulkan device selection for tests/benches (same semantics as OaEngineConfig).
//   integrated | igpu  → OaDevicePreference::Integrated
//   discrete | dgpu   → OaDevicePreference::Discrete
//   cpu                 → OaDevicePreference::Cpu
//   0, 1, 2, …          → OaDevicePreference::ByIndex + DeviceIndex
//
// Use OaTestMergeDeviceEnv(EngineConfig_) from OaComputeApp::Setup to align app runs with gtest device pick.

static inline void OaTestMergeDeviceEnv(OaEngineConfig& InOut) {
	const char* dev = std::getenv("OA_DEVICE");
	if (!dev || !*dev) return;
	if (std::strcmp(dev, "integrated") == 0 || std::strcmp(dev, "igpu") == 0) {
		InOut.DevicePref = OaDevicePreference::Integrated;
		return;
	}
	if (std::strcmp(dev, "discrete") == 0 || std::strcmp(dev, "dgpu") == 0) {
		InOut.DevicePref = OaDevicePreference::Discrete;
		return;
	}
	if (std::strcmp(dev, "cpu") == 0) {
		InOut.DevicePref = OaDevicePreference::Cpu;
		return;
	}
	char* end = nullptr;
	unsigned long idx = std::strtoul(dev, &end, 10);
	if (end != dev && *end == '\0' && idx <= 0xFFFFu) {
		InOut.DevicePref = OaDevicePreference::ByIndex;
		InOut.DeviceIndex = static_cast<OaU32>(idx);
	}
}

static inline OaEngineConfig OaTestEngineConfig(OaPrecision InPrecision) {
	OaEngineConfig cfg;
	cfg.Precision = InPrecision;
	cfg.AppName = "OaTest";
	cfg.MeshVulkanIndices = {};
	if (const char* v = std::getenv("OA_VK_VALIDATION");
		v != nullptr && v[0] == '1')
	{
		cfg.EnableValidation = true;
	}
	// Validation evidence should instrument the workload under test, not spend
	// most of the run eagerly compiling every unrelated embedded shader. The
	// runtime's lazy path preserves the same pipeline ABI and is independently
	// exercised whenever the selected operation is first recorded.
	if (const char* mode = std::getenv("OA_VK_VALIDATION_MODE");
		mode != nullptr && mode[0] != '\0')
	{
		cfg.PreloadEmbeddedPipelines = false;
	}
	OaTestMergeDeviceEnv(cfg);
	return cfg;
}

// Overload: explicit NumericMode for accuracy/determinism tests.
static inline OaEngineConfig OaTestEngineConfig(OaPrecision InPrecision, OaNumericMode InMode) {
	OaEngineConfig cfg = OaTestEngineConfig(InPrecision);
	cfg.NumericMode = InMode;
	return cfg;
}

// ─── Device-aware tolerance helpers (OaNumericStability.md §9.1) ─────────────
//
// Tier A (elementwise) and Tier B (reductions) bounds scale with iGPU vs dGPU
// and with subgroup size for BF16 reductions. Tier C (GEMM) BF16 bound also
// scales with the discovered CoopMat-K and the GEMM's K_total.
//
// Use these instead of EXPECT_NEAR with a flat constant in any cross-vendor
// kernel test. On RTX 5090 they produce identical bounds to the original
// constants; on Ada/RDNA3/Xe2/iGPU they widen automatically.

// Returns 1.5 for integrated GPUs, 1.0 for discrete and unknown.
[[nodiscard]] static inline float OaTestToleranceDeviceScale(const OaVkDevice& InDevice) {
	return InDevice.Info.Hardware.DeviceType == OaDeviceType::VkIntegrated ? 1.5F : 1.0F;
}

// Subgroup-size scale for BF16 reductions per §4.3.
[[nodiscard]] static inline float OaTestToleranceSubgroupScale(const OaVkDevice& InDevice) {
	const OaU32 sg = InDevice.Info.Hardware.SubgroupSize;
	if (sg >= 64) return 1.5F;
	if (sg <= 16) return 0.75F;
	return 1.0F;
}

// Tier A — elementwise BF16. Base 5e-4.
[[nodiscard]] static inline float OaTestToleranceElemwiseBf16(const OaVkDevice& InDevice) {
	return 5.0e-4F * OaTestToleranceDeviceScale(InDevice);
}

// Tier B — reductions BF16. Base 2e-3, scales with subgroup size.
[[nodiscard]] static inline float OaTestToleranceReduceBf16(const OaVkDevice& InDevice) {
	return 2.0e-3F * OaTestToleranceSubgroupScale(InDevice) * OaTestToleranceDeviceScale(InDevice);
}

// Tier C — GEMM BF16. Base depends on per-tile CoopMat-K, scales by sqrt(K_total/K_tile).
[[nodiscard]] static inline float OaTestToleranceGemmCmSgBf16(const OaVkDevice& InDevice, OaU32 InKTotal) {
	const auto& bf16Shape = InDevice.Info.Software.CoopMatShapes.Bf16AccFp32;
	const OaU32 kTile = bf16Shape.Available ? bf16Shape.K : 16U;
	const float base = kTile == 16 ? 2.0e-2F : 3.0e-2F;  // K=32 (RDNA3) bound is wider
	const float kScale = static_cast<float>(std::sqrt(
		static_cast<double>(InKTotal) / static_cast<double>(kTile)));
	return base * kScale * OaTestToleranceDeviceScale(InDevice);
}

#define EXPECT_NEAR_ELEMWISE_BF16(actual, expected, rt) \
	EXPECT_NEAR((actual), (expected), OaTestToleranceElemwiseBf16((rt).Device))

#define EXPECT_NEAR_REDUCE_BF16(actual, expected, rt) \
	EXPECT_NEAR((actual), (expected), OaTestToleranceReduceBf16((rt).Device))

#define EXPECT_NEAR_GEMM_BF16(actual, expected, rt, K_total) \
	EXPECT_NEAR((actual), (expected), OaTestToleranceGemmCmSgBf16((rt).Device, (K_total)))

// Tier D — vision color conversion (uint8 output): max abs error <= 2/255 ≈ 2 lsb.
#define EXPECT_NEAR_VISION_U8(actual, expected) \
	EXPECT_LE(std::abs(static_cast<int>(actual) - static_cast<int>(expected)), 2)

static inline OaPath OaTestAssetPath(OaStringView InRelativePath) {
	return OaPaths::Asset(InRelativePath);
}

// True after OaVkTestEnvironment::SetUp when OaEngine::Create succeeded (global device valid).
static inline bool OaVkTestEngineOk() {
	OaEngine* g = OaEngine::GetGlobal();
	return g != nullptr && g->Device.Device != nullptr;
}

// Matrix Assertions — use At() for BF16 safety

static inline void OaExpectMatrixNear(const OaMatrix& InA, const OaMatrix& InB, OaF32 InEps = 1e-5f) {
	ASSERT_EQ(InA.GetShape(), InB.GetShape()) << "Shape mismatch";
	OaI64 n = InA.NumElements();
	for (OaI64 i = 0; i < n; ++i) {
		EXPECT_NEAR(InA.At(i), InB.At(i), InEps) << "Mismatch at index " << i;
	}
}

// Legacy alias for compatibility
static inline void OaExpectTensorNear(const OaMatrix& InA, const OaMatrix& InB, OaF32 InEps = 1e-5f) {
	OaExpectMatrixNear(InA, InB, InEps);
}

static inline void OaExpectShape(const OaMatrix& InMatrix, std::initializer_list<OaI64> InExpected) {
	OaMatrixShape expected(InExpected);
	EXPECT_EQ(InMatrix.GetShape(), expected) << "Shape mismatch";
}

static inline void OaExpectZero(const OaMatrix& InMatrix) {
	for (OaI64 i = 0; i < InMatrix.NumElements(); ++i) {
		EXPECT_FLOAT_EQ(InMatrix.At(i), 0.0f) << "Non-zero at index " << i;
	}
}

static inline void OaExpectFinite(const OaMatrix& InTensor) {
	for (OaI64 i = 0; i < InTensor.NumElements(); ++i) {
		EXPECT_TRUE(std::isfinite(InTensor.At(i))) << "NaN/Inf at index " << i;
	}
}

static inline void OaExpectValidProbability(const OaMatrix& InTensor, OaI32 InDim = -1) {
	(void)InDim;
	OaI64 lastDim = InTensor.GetShape()[InTensor.Rank() - 1];
	OaI64 batches = InTensor.NumElements() / lastDim;
	for (OaI64 b = 0; b < batches; ++b) {
		OaF32 sum = 0.0f;
		for (OaI64 i = 0; i < lastDim; ++i) {
			OaF32 v = InTensor.At(b * lastDim + i);
			EXPECT_GE(v, 0.0f) << "Negative probability at batch " << b << " index " << i;
			sum += v;
		}
		EXPECT_NEAR(sum, 1.0f, 1e-4f) << "Probabilities don't sum to 1 at batch " << b;
	}
}

// DEPRECATED: OaEngine::Create() now automatically calls EnsureAllEmbeddedLiboaPipelines().
// This function is kept for backward compatibility with tests that create engines with RegisterAsGlobal=false.
// Load every *.spv from OA_SPIRV_DIR into InRt (dtype spec constant matches engine precision).
static inline void OaTestLoadShaders(OaEngine& InRt) {
	// Skip if shaders are embedded (OA_EMBED_SHADERS=ON) - they're already loaded by Create()
	// This function is now a no-op since embedded shaders are the default
	(void)InRt; // Suppress unused parameter warning
}

// Benchmarking

static inline double OaBenchmark(const char* InName, OaI32 InIterations, std::function<void()> InFunc) {
	for (OaI32 i = 0; i < 3; ++i) InFunc();

	auto start = std::chrono::high_resolution_clock::now();
	for (OaI32 i = 0; i < InIterations; ++i) InFunc();
	auto end = std::chrono::high_resolution_clock::now();

	double ms = std::chrono::duration<double, std::milli>(end - start).count();
	double avg = ms / InIterations;
	printf("  %-36s %8.3f ms  (%d iters, %.1f ms total)\n", InName, avg, InIterations, ms);
	return avg;
}

// Vulkan Engine Fixture — Init OaEngine for all ML tests (lavapipe for CI)
// Set OA_TEST_BF16=1 to run with BF16 precision.

class OaVkTestEnvironment : public ::testing::Environment {
public:
	void SetUp() override {
		OaPrecision prec = OaPrecision::FP32;
		const char* bf16Env = std::getenv("OA_TEST_BF16");
		if (bf16Env && OaString(bf16Env) == "1") {
			prec = OaPrecision::BF16;
		}
		OaEngineConfig ecfg = OaTestEngineConfig(prec);
		auto result = OaEngine::Create(ecfg);
		if (!result) {
			fprintf(stderr,
				"OaVkTestEnvironment: Failed to create OaEngine: %s\n",
				result.GetStatus().ToString().c_str());
			return;
		}
		Engine_ = std::move(*result);   // take ownership of the pinned engine
		// Note: EnsureAllEmbeddedLiboaPipelines() is now called automatically during Create()
		// Create() already selected the context owned by the global engine.
		
		// Load shaders from spirv/ directory for debug builds (OA_EMBED_SHADERS=OFF)
		OaTestLoadShaders(*Engine_);
	}

	void TearDown() override {
		// Flush the engine-owned context before engine teardown. Destroy clears the
		// thread default before releasing the context, so no state survives the suite.
		if (OaContext::GetDefaultPtr()) {
			auto& ctx = OaContext::GetDefault();
			(void)ctx.Execute();
			(void)ctx.Sync();
			ctx.Clear();
		}
		OaContext::SetDefault(nullptr);
		if (Engine_) {
			Engine_->Destroy();
			Engine_.reset();
		}
	}

private:
	OaUniquePtr<OaEngine> Engine_;
};

// Optional TEST_F base: asserts gtest global Vulkan environment created the engine.
class OaVkEngineTestFixture : public ::testing::Test {
protected:
	void SetUp() override {
		ASSERT_TRUE(OaVkTestEngineOk()) << "OaVkTestEnvironment did not create OaEngine (see stderr)";
	}

	static OaEngine& Rt() { return *OaEngine::GetGlobal(); }
};

// Test Helpers

static inline OaMatrix OaMakeByteIndices(std::initializer_list<OaU8> InValues) {
	auto t = OaFnMatrix::Empty(OaMatrixShape{static_cast<OaI64>(InValues.size())}, OaScalarType::UInt8);
	OaU8* data = t.DataAs<OaU8>();
	OaI64 i = 0;
	for (auto v : InValues) data[i++] = v;
	return t;
}

static inline OaMatrix OaMakeTestTensor(OaMatrixShape InShape, OaF32 InStart = 0.0f, OaF32 InStep = 1.0f) {
	auto t = OaFnMatrix::Empty(InShape);
	for (OaI64 i = 0; i < t.NumElements(); ++i) {
		t.Set(i, InStart + static_cast<OaF32>(i) * InStep);
	}
	return t;
}
