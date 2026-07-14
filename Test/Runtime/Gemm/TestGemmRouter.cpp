#include "../../OaTest.h"

#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/GemmRouteCache.h>
#include <Oa/Runtime/MatmulTypes.h>
#include <Oa/Runtime/Gemm/Router.h>

#include <cstdio>

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
	key.ShaderBuildId = OaMatmulRegistry::BuildId();
	key.Variant = OaGemmKernel::Auto;
	key.M = InM; key.N = InN; key.K = InK;
	key.APrecision = InPrecision;
	key.BPrecision = InPrecision;
	key.Epilogue = OaGemmEpilogue::None;
	key.Training = false;
	key.UseTMA = InRt.IsBlackwell();
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

TEST(GemmRouteCache, ExactShapesDoNotAlias) {
	OaGemmRouteCache cache;
	OaRouteCacheKey aligned{};
	aligned.M = 64; aligned.N = 64; aligned.K = 64;
	aligned.Epilogue = OaGemmEpilogue::None;
	OaRouteCacheKey unaligned = aligned;
	unaligned.M = 127;

	cache.Update(aligned, OaGemmKernel::GemmCmSgBf16, 0.25F, 1);
	OaGemmKernel winner = OaGemmKernel::Auto;
	EXPECT_TRUE(cache.Query(aligned, winner));
	EXPECT_EQ(winner, OaGemmKernel::GemmCmSgBf16);
	EXPECT_FALSE(cache.Query(unaligned, winner));
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
	key.Variant = OaGemmKernel::Auto;
	key.M = 64; key.N = 96; key.K = 32;
	key.APrecision = OaGemmPrecision::Fp32;
	key.BPrecision = OaGemmPrecision::Fp32;
	key.Epilogue = OaGemmEpilogue::BiasGelu;
	key.Training = true;
	source.Update(key, OaGemmKernel::TiledFp32, 0.5F, 9);
	ASSERT_TRUE(source.Save(path));

	OaGemmRouteCache loaded;
	ASSERT_TRUE(loaded.Load(path));
	OaGemmKernel winner = OaGemmKernel::Auto;
	EXPECT_TRUE(loaded.Query(key, winner));
	EXPECT_EQ(winner, OaGemmKernel::TiledFp32);
	{
		std::FILE* f = std::fopen(path, "ab");
		ASSERT_NE(f, nullptr);
		const OaU8 trailingGarbage = 0xffU;
		ASSERT_EQ(std::fwrite(&trailingGarbage, sizeof(trailingGarbage), 1, f), 1U);
		std::fclose(f);
	}
	OaGemmRouteCache trailingRejected;
	trailingRejected.Update(key, OaGemmKernel::Naive, 1.0F, 1);
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

TEST(GemmRouter, Fp32RejectsIllegalCachedBf16Winner) {
	if (not OaVkTestEngineOk()) { GTEST_SKIP(); }
	auto& rt = *OaComputeEngine::GetGlobal();
	ScopedRouteCache scoped(rt);
	auto key = MakeRawKey(rt, 64, 64, 64, OaGemmPrecision::Fp32);
	scoped.Get().Update(key, OaGemmKernel::GemmCmSgBf16, 0.01F, 1);

	const auto route = OaGemmRouter::Select(rt, 64, 64, 64, OaGemmPrecision::Fp32);
	EXPECT_EQ(route.Kernel, OaGemmKernel::TiledFp32);
	EXPECT_EQ(route.ActualPrec, OaGemmPrecision::Fp32);
}

TEST(GemmRouter, Fp32ReplaysLegalCachedWinner) {
	if (not OaVkTestEngineOk()) { GTEST_SKIP(); }
	auto& rt = *OaComputeEngine::GetGlobal();
	ScopedRouteCache scoped(rt);
	auto key = MakeRawKey(rt, 64, 64, 64, OaGemmPrecision::Fp32);
	scoped.Get().Update(key, OaGemmKernel::Naive, 0.01F, 1);

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
}
