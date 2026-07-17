#include "../../OaTest.h"

#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/GemmRouteCache.h>
#include <Oa/Runtime/MatmulTypes.h>
#include <Oa/Runtime/Gemm/Router.h>
#include <Oa/Runtime/Gemm/Dispatch.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Ml/FnMatrix.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <unordered_set>
#include <vector>

namespace {

OaU64 HashString(const char* InText) {
	OaU64 h = 0xcbf29ce484222325ULL;
	for (const char* p = InText; p != nullptr and *p != '\0'; ++p) {
		h ^= static_cast<OaU8>(*p);
		h *= 0x100000001b3ULL;
	}
	return h;
}

OaRouteCacheKey MakeRawKey(
	const OaComputeEngine& InRt,
	OaU32 InM,
	OaU32 InN,
	OaU32 InK,
	OaGemmPrecision InPrecision)
{
	OaRouteCacheKey key{};
	key.VendorId = InRt.Device.Info.Hardware.VendorId;
	key.DeviceId = InRt.Device.Info.Hardware.DeviceId;
	key.DriverId = InRt.Device.Info.Software.DriverId;
	key.DriverVersionHash = HashString(InRt.Device.Info.Software.DriverVersion.c_str());
	key.ShaderBuildId = OaMatmulRegistry::ShaderBuildId();
	key.M = InM; key.N = InN; key.K = InK;
	key.BatchCount = 1U;
	key.ARowStride = InK; key.AColStride = 1U; key.ABatchStride = InM * InK;
	key.BRowStride = InK; key.BColStride = 1U; key.BBatchStride = InN * InK;
	key.CRowStride = InN; key.CColStride = 1U; key.CBatchStride = InM * InN;
	key.APrecision = OaGemmPrecision::Fp32;
	key.BPrecision = OaGemmPrecision::Fp32;
	key.OutputPrecision = OaGemmPrecision::Fp32;
	key.RequestedPrecision = InPrecision;
	key.Epilogue = OaGemmEpilogue::None;
	key.AContiguous = true;
	key.BContiguous = true;
	key.BTransposed = true;
	key.RequiresPreActivation = false;
	key.Training = false;
	return key;
}

class ScopedRouteCache {
public:
	explicit ScopedRouteCache(OaComputeEngine& InRt)
		: Rt_(InRt), Previous_(InRt.GemmRouteCache) {
		Rt_.GemmRouteCache = &Local_;
	}
	~ScopedRouteCache() { Rt_.GemmRouteCache = Previous_; }
	OaGemmRouteCache& Get() { return Local_; }

private:
	OaComputeEngine& Rt_;
	OaGemmRouteCache* Previous_;
	OaGemmRouteCache Local_;
};

} // namespace

TEST(GemmRegistry, StableVariantIdsAreUniqueAndResolvable) {
	std::unordered_set<OaMatmulVariantId> ids;
	for (const auto& variant : OaMatmulRegistry::All()) {
		EXPECT_NE(variant.Id, OaInvalidMatmulVariantId);
		EXPECT_TRUE(ids.insert(variant.Id).second) << variant.KernelName;
		const auto* resolved = OaMatmulRegistry::Find(variant.Id);
		ASSERT_NE(resolved, nullptr);
		EXPECT_STREQ(resolved->KernelName, variant.KernelName);
		EXPECT_EQ(resolved->Epilogue, variant.Epilogue);
	}
}

TEST(GemmRouteCache, ExactShapesDoNotAlias) {
	OaGemmRouteCache cache;
	OaRouteCacheKey aligned{};
	aligned.M = 64; aligned.N = 64; aligned.K = 64;
	aligned.Epilogue = OaGemmEpilogue::None;
	OaRouteCacheKey unaligned = aligned;
	unaligned.M = 127;

	const auto sg = OaMatmulVariantIdFromName("GemmCmSgBf16");
	cache.Update(aligned, sg, 0.25F, 1);
	OaMatmulVariantId winner = OaInvalidMatmulVariantId;
	EXPECT_TRUE(cache.Query(aligned, winner));
	EXPECT_EQ(winner, sg);
	EXPECT_FALSE(cache.Query(unaligned, winner));
}

TEST(GemmRouteCache, LayoutAndDualOutputContractsDoNotAlias) {
	OaGemmRouteCache cache;
	OaRouteCacheKey base{};
	base.M = 64; base.N = 64; base.K = 64;
	base.APrecision = OaGemmPrecision::Fp32;
	base.BPrecision = OaGemmPrecision::Fp32;
	base.OutputPrecision = OaGemmPrecision::Fp32;
	base.RequestedPrecision = OaGemmPrecision::Auto;
	base.Epilogue = OaGemmEpilogue::BiasGelu;
	base.AContiguous = true;
	base.BContiguous = true;
	base.BTransposed = true;
	const auto winnerId = OaMatmulVariantIdFromName("GemmBiasGeluTiled");
	cache.Update(base, winnerId, 0.25F, 1);

	OaMatmulVariantId winner = OaInvalidMatmulVariantId;
	EXPECT_TRUE(cache.Query(base, winner));
	EXPECT_EQ(winner, winnerId);
	auto wrongLayout = base;
	wrongLayout.BTransposed = false;
	EXPECT_FALSE(cache.Query(wrongLayout, winner));
	auto wrongOutputContract = base;
	wrongOutputContract.RequiresPreActivation = true;
	EXPECT_FALSE(cache.Query(wrongOutputContract, winner));
}

TEST(GemmRouteCache, RequestedPrecisionPoliciesDoNotAlias) {
	OaGemmRouteCache cache;
	OaRouteCacheKey autoKey{};
	autoKey.M = 64; autoKey.N = 64; autoKey.K = 64;
	autoKey.APrecision = OaGemmPrecision::Fp32;
	autoKey.BPrecision = OaGemmPrecision::Fp32;
	autoKey.OutputPrecision = OaGemmPrecision::Fp32;
	autoKey.RequestedPrecision = OaGemmPrecision::Auto;
	const auto tiled = OaMatmulVariantIdFromName("GemmTiled");
	cache.Update(autoKey, tiled, 0.25F, 1);

	OaMatmulVariantId winner = OaInvalidMatmulVariantId;
	EXPECT_TRUE(cache.Query(autoKey, winner));
	auto fp32Key = autoKey;
	fp32Key.RequestedPrecision = OaGemmPrecision::Fp32;
	EXPECT_FALSE(cache.Query(fp32Key, winner));
	auto bf16Key = autoKey;
	bf16Key.RequestedPrecision = OaGemmPrecision::Bf16;
	EXPECT_FALSE(cache.Query(bf16Key, winner));
}

TEST(GemmRouteCache, VersionedRoundTripRejectsLegacyBytes) {
	const char* path = "/tmp/oa_test_gemm_route_cache.bin";
	const char* legacyPath = "/tmp/oa_test_gemm_route_cache_legacy.bin";
	std::remove(path);
	std::remove(legacyPath);

	OaGemmRouteCache source;
	OaRouteCacheKey key{};
	key.VendorId = 0x8086U;
	key.DeviceId = 0x9a49U;
	key.DriverId = 3U;
	key.DriverVersionHash = 17U;
	key.ShaderBuildId = 23U;
	key.M = 64; key.N = 96; key.K = 32;
	key.APrecision = OaGemmPrecision::Fp32;
	key.BPrecision = OaGemmPrecision::Fp32;
	key.OutputPrecision = OaGemmPrecision::Fp32;
	key.RequestedPrecision = OaGemmPrecision::Auto;
	key.Epilogue = OaGemmEpilogue::BiasGelu;
	key.AContiguous = true;
	key.BContiguous = true;
	key.BTransposed = true;
	key.RequiresPreActivation = false;
	key.Training = true;
	const auto tiled = OaMatmulVariantIdFromName("GemmTiled");
	source.Update(key, tiled, 0.5F, 9);
	ASSERT_TRUE(source.Save(path));

	OaGemmRouteCache loaded;
	ASSERT_TRUE(loaded.Load(path));
	OaMatmulVariantId winner = OaInvalidMatmulVariantId;
	EXPECT_TRUE(loaded.Query(key, winner));
	EXPECT_EQ(winner, tiled);
	{
		std::FILE* f = std::fopen(path, "ab");
		ASSERT_NE(f, nullptr);
		const OaU8 trailingGarbage = 0xffU;
		ASSERT_EQ(std::fwrite(&trailingGarbage, sizeof(trailingGarbage), 1, f), 1U);
		std::fclose(f);
	}
	OaGemmRouteCache trailingRejected;
	trailingRejected.Update(key, OaMatmulVariantIdFromName("GemmNaive"), 1.0F, 1);
	EXPECT_FALSE(trailingRejected.Load(path));
	EXPECT_FALSE(trailingRejected.Query(key, winner));

	{
		std::FILE* f = std::fopen(legacyPath, "wb");
		ASSERT_NE(f, nullptr);
		const OaU64 oldEmptyCount = 0;
		ASSERT_EQ(std::fwrite(&oldEmptyCount, sizeof(oldEmptyCount), 1, f), 1U);
		std::fclose(f);
	}
	OaGemmRouteCache rejected;
	EXPECT_FALSE(rejected.Load(legacyPath));
	std::remove(path);
	std::remove(legacyPath);
}

TEST(GemmRouteCache, AggregatedTimingMetadataRoundTrips) {
	OaGemmRouteCache source;
	OaRouteCacheKey key{};
	key.M = 23U; key.N = 17U; key.K = 11U;
	key.BatchCount = 1U;
	key.ARowStride = key.K; key.AColStride = 1U; key.ABatchStride = key.M * key.K;
	key.BRowStride = key.K; key.BColStride = 1U; key.BBatchStride = key.N * key.K;
	key.CRowStride = key.N; key.CColStride = 1U; key.CBatchStride = key.M * key.N;
	key.APrecision = OaGemmPrecision::Fp32;
	key.BPrecision = OaGemmPrecision::Fp32;
	key.OutputPrecision = OaGemmPrecision::Fp32;
	key.RequestedPrecision = OaGemmPrecision::Auto;
	key.BTransposed = true;
	const auto winner = OaMatmulVariantIdFromName("GemmTiled");
	source.Update(key, winner, 0.125F, 0.175F, 19U, 42U);

	const char* path = "/tmp/oa_test_gemm_route_stats.bin";
	std::remove(path);
	ASSERT_TRUE(source.Save(path));
	OaGemmRouteCache loaded;
	ASSERT_TRUE(loaded.Load(path));
	ASSERT_EQ(loaded.Map.size(), 1U);
	const auto& value = loaded.Map.begin()->second;
	EXPECT_EQ(value.WinnerVariant, winner);
	EXPECT_FLOAT_EQ(value.MedianGpuTimeMs, 0.125F);
	EXPECT_FLOAT_EQ(value.P95GpuTimeMs, 0.175F);
	EXPECT_EQ(value.SampleCount, 19U);
	EXPECT_EQ(value.LastUpdatedStep, 42U);
	std::remove(path);
}

TEST(GemmRouter, Fp32RejectsIllegalCachedBf16Winner) {
	if (not OaVkTestEngineOk()) { GTEST_SKIP(); }
	auto& rt = *OaComputeEngine::GetGlobal();
	ScopedRouteCache scoped(rt);
	auto key = MakeRawKey(rt, 64, 64, 64, OaGemmPrecision::Fp32);
	scoped.Get().Update(key, OaMatmulVariantIdFromName("GemmCmSgBf16"), 0.01F, 1);

	const auto route = OaGemmRouter::Select(rt, 64, 64, 64, OaGemmPrecision::Fp32);
	EXPECT_EQ(route.Kernel, OaGemmKernel::TiledFp32);
	EXPECT_EQ(route.ActualPrec, OaGemmPrecision::Fp32);
}

TEST(GemmRouter, Fp32ReplaysLegalCachedWinner) {
	if (not OaVkTestEngineOk()) { GTEST_SKIP(); }
	auto& rt = *OaComputeEngine::GetGlobal();
	ScopedRouteCache scoped(rt);
	auto key = MakeRawKey(rt, 64, 64, 64, OaGemmPrecision::Fp32);
	scoped.Get().Update(key, OaMatmulVariantIdFromName("GemmNaive"), 0.01F, 1);

	// Naive is legal but is not the heuristic choice for this shape. Returning
	// it proves that the explicit-FP32 cache path was actually exercised.
	const auto route = OaGemmRouter::Select(rt, 64, 64, 64, OaGemmPrecision::Fp32);
	EXPECT_EQ(route.Kernel, OaGemmKernel::Naive);
	EXPECT_EQ(route.ActualPrec, OaGemmPrecision::Fp32);
}

TEST(GemmRouter, Fp32ShapeRoutesAreLegal) {
	if (not OaVkTestEngineOk()) { GTEST_SKIP(); }
	auto& rt = *OaComputeEngine::GetGlobal();
	ScopedRouteCache scoped(rt);

	const auto tiny = OaGemmRouter::Select(rt, 1, 1, 1, OaGemmPrecision::Fp32);
	EXPECT_EQ(tiny.Kernel, OaGemmKernel::Naive);
	const auto tiled = OaGemmRouter::Select(rt, 63, 65, 17, OaGemmPrecision::Fp32);
	EXPECT_EQ(tiled.Kernel, OaGemmKernel::TiledFp32);
	EXPECT_EQ(tiled.Gx, 1U);
	EXPECT_EQ(tiled.Gy, 2U);
	const auto* variant = OaMatmulRegistry::Find(tiled.Variant);
	ASSERT_NE(variant, nullptr);
	EXPECT_STREQ(variant->KernelName, tiled.KernelName);
}

TEST(GemmRouter, ImmutablePlanRejectsContractAndDeviceDrift) {
	if (not OaVkTestEngineOk()) { GTEST_SKIP(); }
	auto& rt = *OaComputeEngine::GetGlobal();
	ScopedRouteCache scoped(rt);
	auto problem = OaGemmRouter::ProblemForRaw(
		130, 193, 71,
		OaStoragePrecision::Fp32, OaStoragePrecision::Fp32, true);
	problem.PrecisionHint = OaGemmPrecision::Fp32;
	problem.Training = true;

	const auto plan = OaGemmRouter::Plan(rt, problem);
	ASSERT_TRUE(static_cast<bool>(plan));
	EXPECT_TRUE(OaGemmRouter::ValidatePlan(rt, plan, problem));
	EXPECT_NE(plan.ShaderContentHash, 0U);
	EXPECT_EQ(plan.Grid.X, 3U);
	EXPECT_EQ(plan.Grid.Y, 4U);

	auto wrongProblem = problem;
	wrongProblem.K += 1U;
	EXPECT_FALSE(OaGemmRouter::ValidatePlan(rt, plan, wrongProblem));

	auto stalePlan = plan;
	stalePlan.RegistryBuildId ^= 1U;
	EXPECT_FALSE(OaGemmRouter::ValidatePlan(rt, stalePlan, problem));

	auto wrongGrid = plan;
	wrongGrid.Grid.X += 1U;
	EXPECT_FALSE(OaGemmRouter::ValidatePlan(rt, wrongGrid, problem));

	auto wrongShader = plan;
	wrongShader.ShaderContentHash ^= 1U;
	EXPECT_FALSE(OaGemmRouter::ValidatePlan(rt, wrongShader, problem));
}

TEST(GemmRouter, RawPlanPreservesLegacySelectionContract) {
	if (not OaVkTestEngineOk()) { GTEST_SKIP(); }
	auto& rt = *OaComputeEngine::GetGlobal();
	ScopedRouteCache scoped(rt);
	const auto legacy = OaGemmRouter::Select(
		rt, 64, 96, 32, OaGemmPrecision::Fp32);
	auto problem = OaGemmRouter::ProblemForRaw(
		64, 96, 32,
		OaStoragePrecision::Fp32, OaStoragePrecision::Fp32, true);
	problem.Training = false;
	problem.PrecisionHint = OaGemmPrecision::Fp32;
	const auto plan = OaGemmRouter::Plan(rt, problem);

	ASSERT_TRUE(static_cast<bool>(plan));
	EXPECT_EQ(plan.Variant, legacy.Variant);
	EXPECT_EQ(plan.Kernel, legacy.Kernel);
	EXPECT_EQ(plan.Grid.X, legacy.Gx);
	EXPECT_EQ(plan.Grid.Y, legacy.Gy);
	EXPECT_EQ(plan.Grid.Z, legacy.Gz);
	EXPECT_TRUE(OaGemmRouter::ValidatePlan(rt, plan, problem));
}

TEST(GemmRouter, PlanPreferenceCanBypassMeasuredCache) {
	if (not OaVkTestEngineOk()) { GTEST_SKIP(); }
	auto& rt = *OaComputeEngine::GetGlobal();
	ScopedRouteCache scoped(rt);
	auto problem = OaGemmRouter::ProblemForRaw(
		64, 64, 64,
		OaStoragePrecision::Fp32, OaStoragePrecision::Fp32, true);
	problem.PrecisionHint = OaGemmPrecision::Fp32;
	problem.Training = false;
	scoped.Get().Update(
		OaGemmRouter::CacheKey(rt, problem),
		OaMatmulVariantIdFromName("GemmNaive"), 0.01F, 1);

	const auto cached = OaGemmRouter::Plan(rt, problem);
	EXPECT_EQ(cached.Kernel, OaGemmKernel::Naive);

	OaMatmulPreference heuristicOnly{};
	heuristicOnly.UseMeasuredCache = false;
	const auto heuristic = OaGemmRouter::Plan(rt, problem, heuristicOnly);
	EXPECT_EQ(heuristic.Kernel, OaGemmKernel::TiledFp32);
}

TEST(GemmRouter, GeneratedVariantHonorsDeviceLaunchLimits) {
	if (not OaVkTestEngineOk()) { GTEST_SKIP(); }
	auto& rt = *OaComputeEngine::GetGlobal();
	auto& hw = rt.Device.Info.Hardware;
	const OaU32 savedShared = hw.MaxComputeSharedMemoryBytes;
	const OaU32 savedInvocations = hw.MaxComputeWorkGroupInvocations;
	const OaU32 savedSize = hw.MaxComputeWorkGroupSize;

	auto problem = OaGemmRouter::ProblemForRaw(
		4096, 384, 1536,
		OaStoragePrecision::Fp32, OaStoragePrecision::Fp32, true);
	problem.Training = true;
	problem.PrecisionHint = OaGemmPrecision::Fp32;
	const auto* control = OaMatmulRegistry::Find(
		OaMatmulVariantIdFromName("GemmTiled"));
	const auto* k32 = OaMatmulRegistry::Find(
		OaMatmulVariantIdFromName("GemmTiledK32"));
	ASSERT_NE(control, nullptr);
	ASSERT_NE(k32, nullptr);

	// K=32 needs 16,896 bytes while the K=16 control needs 8,704. A device at
	// Vulkan's 16 KiB minimum must retain the control and reject K=32.
	hw.MaxComputeSharedMemoryBytes = 16U * 1024U;
	EXPECT_TRUE(OaGemmRouter::IsVariantLegal(rt, *control, problem));
	EXPECT_FALSE(OaGemmRouter::IsVariantLegal(rt, *k32, problem));

	// The same legality boundary owns both x-size and total-invocation limits.
	hw.MaxComputeSharedMemoryBytes = savedShared;
	hw.MaxComputeWorkGroupInvocations = 128U;
	hw.MaxComputeWorkGroupSize = 128U;
	EXPECT_FALSE(OaGemmRouter::IsVariantLegal(rt, *control, problem));

	hw.MaxComputeSharedMemoryBytes = savedShared;
	hw.MaxComputeWorkGroupInvocations = savedInvocations;
	hw.MaxComputeWorkGroupSize = savedSize;
}

TEST(GemmRouter, FusedFp32RoutesPreserveExactEpilogue) {
	if (not OaVkTestEngineOk()) { GTEST_SKIP(); }
	auto& rt = *OaComputeEngine::GetGlobal();
	ScopedRouteCache scoped(rt);
	auto problem = OaGemmRouter::ProblemForRaw(
		64, 64, 64,
		OaStoragePrecision::Fp32, OaStoragePrecision::Fp32, true);
	problem.PrecisionHint = OaGemmPrecision::Fp32;
	problem.Training = true;

	problem.Epilogue = OaGemmEpilogue::Bias;
	const auto bias = OaGemmRouter::Select(rt, problem);
	EXPECT_STREQ(bias.KernelName, "GemmBiasTiled");
	problem.Epilogue = OaGemmEpilogue::BiasRelu;
	const auto relu = OaGemmRouter::Select(rt, problem);
	EXPECT_STREQ(relu.KernelName, "GemmBiasReluTiled");
	problem.Epilogue = OaGemmEpilogue::BiasGelu;
	const auto gelu = OaGemmRouter::Select(rt, problem);
	EXPECT_STREQ(gelu.KernelName, "GemmBiasGeluTiled");

	EXPECT_NE(bias.Variant, relu.Variant);
	EXPECT_NE(relu.Variant, gelu.Variant);
	EXPECT_EQ(OaMatmulRegistry::Find(bias.Variant)->Epilogue, OaGemmEpilogue::Bias);
	EXPECT_EQ(OaMatmulRegistry::Find(relu.Variant)->Epilogue, OaGemmEpilogue::BiasRelu);
	EXPECT_EQ(OaMatmulRegistry::Find(gelu.Variant)->Epilogue, OaGemmEpilogue::BiasGelu);
}

TEST(GemmRouter, RejectsCachedWinnerFromDifferentEpilogue) {
	if (not OaVkTestEngineOk()) { GTEST_SKIP(); }
	auto& rt = *OaComputeEngine::GetGlobal();
	ScopedRouteCache scoped(rt);
	auto key = MakeRawKey(rt, 64, 64, 64, OaGemmPrecision::Fp32);
	key.Epilogue = OaGemmEpilogue::BiasRelu;
	key.Training = true;
	scoped.Get().Update(
		key, OaMatmulVariantIdFromName("GemmBiasGeluTiled"), 0.01F, 1);

	auto problem = OaGemmRouter::ProblemForRaw(
		64, 64, 64,
		OaStoragePrecision::Fp32, OaStoragePrecision::Fp32, true);
	problem.PrecisionHint = OaGemmPrecision::Fp32;
	problem.Epilogue = OaGemmEpilogue::BiasRelu;
	problem.Training = true;
	const auto route = OaGemmRouter::Select(rt, problem);
	EXPECT_STREQ(route.KernelName, "GemmBiasReluTiled");
}

TEST(GemmRouter, GeneratedTiledVariantsMatchCpuOnIrregularTails) {
	if (not OaVkTestEngineOk()) { GTEST_SKIP(); }

	// Deliberately crosses several 64x64 tiles in both dimensions and leaves
	// partial M/N/K tails. Every generated raw FP32 tiled variant is forced, so
	// adding schema geometry automatically expands the correctness gate.
	constexpr OaU32 M = 130U;
	constexpr OaU32 N = 193U;
	constexpr OaU32 K = 71U;
	std::vector<OaF32> aData(M * K);
	std::vector<OaF32> bData(N * K);
	for (OaU32 i = 0; i < M * K; ++i) {
		aData[i] = static_cast<OaF32>(static_cast<OaI32>(i % 29U) - 14) * 0.03125F;
	}
	for (OaU32 i = 0; i < N * K; ++i) {
		bData[i] = static_cast<OaF32>(static_cast<OaI32>(i % 23U) - 11) * 0.025F;
	}

	auto a = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(aData.data()), aData.size() * sizeof(OaF32)),
		OaMatrixShape{M, K});
	auto b = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(bData.data()), bData.size() * sizeof(OaF32)),
		OaMatrixShape{N, K});

	for (const auto& variant : OaMatmulRegistry::All()) {
		if (variant.Kernel != OaGemmKernel::TiledFp32
			or variant.Epilogue != OaGemmEpilogue::None)
		{
			continue;
		}
		OaGemmRouter::ForceVariant(M, N, K, variant.Id);
		auto c = OaFnMatrix::MatMulNt(a, b, OaContextMatMulPrecision::Fp32);
		OaGemmRouter::ClearForced();

		std::vector<OaF32> got(M * N);
		ASSERT_TRUE(OaFnMatrix::CopyToHost(c, got.data(), got.size() * sizeof(OaF32)).IsOk());
		for (OaU32 row = 0; row < M; ++row) {
			for (OaU32 col = 0; col < N; ++col) {
				OaF32 expected = 0.0F;
				for (OaU32 k = 0; k < K; ++k) {
					expected += aData[row * K + k] * bData[col * K + k];
				}
				EXPECT_NEAR(got[row * N + col], expected, 2.0e-4F)
					<< variant.KernelName << " row=" << row << " col=" << col;
			}
		}
	}
}

TEST(GemmRouter, GeneratedTiledEpiloguesMatchCpuOnIrregularTails) {
	if (not OaVkTestEngineOk()) { GTEST_SKIP(); }

	constexpr OaU32 M = 67U;
	constexpr OaU32 N = 73U;
	constexpr OaU32 K = 19U;
	std::vector<OaF32> xData(M * K);
	std::vector<OaF32> wData(N * K);
	std::vector<OaF32> biasData(N);
	for (OaU32 i = 0; i < M * K; ++i) {
		xData[i] = static_cast<OaF32>(static_cast<OaI32>(i % 17U) - 8) * 0.03125F;
	}
	for (OaU32 i = 0; i < N * K; ++i) {
		wData[i] = static_cast<OaF32>(static_cast<OaI32>(i % 13U) - 6) * 0.025F;
	}
	for (OaU32 i = 0; i < N; ++i) {
		biasData[i] = static_cast<OaF32>(static_cast<OaI32>(i % 11U) - 5) * 0.02F;
	}

	auto x = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(xData.data()), xData.size() * sizeof(OaF32)),
		OaMatrixShape{M, K});
	auto w = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(wData.data()), wData.size() * sizeof(OaF32)),
		OaMatrixShape{N, K});
	auto bias = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(biasData.data()), biasData.size() * sizeof(OaF32)),
		OaMatrixShape{N});

	for (const auto& variant : OaMatmulRegistry::All()) {
		if (variant.Kernel != OaGemmKernel::TiledFp32
			or variant.Epilogue == OaGemmEpilogue::None) {
			continue;
		}
		OaGemmRouter::ForceVariant(M, N, K, variant.Id);
		OaMatrix output;
		switch (variant.Epilogue) {
			case OaGemmEpilogue::Bias:
				output = OaFnMatrix::Linear(x, w, bias);
				break;
			case OaGemmEpilogue::BiasRelu:
				output = OaFnMatrix::LinearRelu(x, w, bias);
				break;
			case OaGemmEpilogue::BiasGelu:
				output = OaFnMatrix::LinearGelu(x, w, bias);
				break;
			default:
				OaGemmRouter::ClearForced();
				FAIL() << "unexpected generated FP32 epilogue";
		}
		OaGemmRouter::ClearForced();

		std::vector<OaF32> got(M * N);
		ASSERT_TRUE(OaFnMatrix::CopyToHost(
			output, got.data(), got.size() * sizeof(OaF32)).IsOk());
		for (OaU32 row = 0; row < M; ++row) {
			for (OaU32 col = 0; col < N; ++col) {
				OaF32 expected = biasData[col];
				for (OaU32 k = 0; k < K; ++k) {
					expected += xData[row * K + k] * wData[col * K + k];
				}
				if (variant.Epilogue == OaGemmEpilogue::BiasRelu) {
					expected = std::max(0.0F, expected);
				} else if (variant.Epilogue == OaGemmEpilogue::BiasGelu) {
					const OaF32 x3 = expected * expected * expected;
					expected = 0.5F * expected * (1.0F + std::tanh(
						0.7978845608F * (expected + 0.044715F * x3)));
				}
				EXPECT_NEAR(got[row * N + col], expected, 3.0e-4F)
					<< variant.KernelName << " row=" << row << " col=" << col;
			}
		}
	}
}

TEST(GemmRouter, ArbitraryRowAndColumnStridesMatchCpu) {
	if (not OaVkTestEngineOk()) { GTEST_SKIP(); }
	const std::vector<OaF32> aStorage = {
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
	};
	const std::vector<OaF32> bStorage = {
		1, 2,
		3, 4,
		5, 6,
	};
	auto aBase = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(aStorage.data()),
			aStorage.size() * sizeof(OaF32)), OaMatrixShape{3, 4});
	auto bBase = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(bStorage.data()),
			bStorage.size() * sizeof(OaF32)), OaMatrixShape{3, 2});
	const OaI32 permutation[] = {1, 0};
	auto a = aBase.Permute(OaSpan<const OaI32>(permutation)); // [4,3], strides [1,4]
	auto b = bBase.Permute(OaSpan<const OaI32>(permutation)); // [2,3], strides [1,2]
	auto c = OaFnMatrix::MatMulNt(a, b, OaContextMatMulPrecision::Fp32);

	std::vector<OaF32> got(8);
	ASSERT_TRUE(OaFnMatrix::CopyToHost(c, got.data(), got.size() * sizeof(OaF32)).IsOk());
	const std::vector<OaF32> expected = {61, 76, 70, 88, 79, 100, 88, 112};
	EXPECT_EQ(got, expected);
}

TEST(GemmRouter, StridedBatchPlanMatchesCpu) {
	if (not OaVkTestEngineOk()) { GTEST_SKIP(); }
	auto& rt = *OaComputeEngine::GetGlobal();
	const std::vector<OaF32> aData = {
		1, 2, 3, 4, 5, 6,
		2, 0, 1, 1, 3, 2,
	};
	const std::vector<OaF32> bData = {
		1, 0, 1, 0, 1, 1,
		1, 2, 0, 0, 1, 2,
	};
	auto a = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(aData.data()),
			aData.size() * sizeof(OaF32)), OaMatrixShape{2, 2, 3});
	auto b = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(bData.data()),
			bData.size() * sizeof(OaF32)), OaMatrixShape{2, 2, 3});
	auto c = OaFnMatrix::Empty(OaMatrixShape{2, 2, 2});
	auto problem = OaGemmRouter::ProblemForRaw(
		2, 2, 3, OaStoragePrecision::Fp32, OaStoragePrecision::Fp32, true);
	problem.BatchCount = 2;
	problem.A.BatchStride = 6;
	problem.B.BatchStride = 6;
	problem.C.BatchStride = 4;
	const auto plan = OaGemmRouter::Plan(rt, problem);
	ASSERT_TRUE(static_cast<bool>(plan));
	EXPECT_EQ(plan.Kernel, OaGemmKernel::StridedFp32);
	EXPECT_EQ(plan.Grid.Z, 2U);
	OaVkBuffer buffers[] = {a.GetVkBuffer(), b.GetVkBuffer(), c.GetVkBuffer()};
	ASSERT_TRUE(OaGemmDispatch::ExecutePlan(rt, plan, problem, buffers).IsOk());

	std::vector<OaF32> got(8);
	ASSERT_TRUE(OaFnMatrix::CopyToHost(c, got.data(), got.size() * sizeof(OaF32)).IsOk());
	const std::vector<OaF32> expected = {4, 5, 10, 11, 2, 2, 7, 7};
	EXPECT_EQ(got, expected);
}
