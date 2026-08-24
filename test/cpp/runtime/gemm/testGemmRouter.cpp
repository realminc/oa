#include "../../oaTest.h"

#include <oa/runtime/engine.h>
#include <oa/runtime/gemm/engineRouteCacheAccess.h>
#include <oa/runtime/matmulTypes.h>
#include <oa/runtime/type.h>
#include <oa/runtime/gemm/router.h>
#include <oa/runtime/gemm/dispatch.h>
#include <oa/runtime/gemm/graphLowering.h>
#include <oa/runtime/gemm/tuner.h>
#include <oa/runtime/executionSession.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/matrixAccess.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_set>
#include <vector>

static_assert(static_cast<oa::U8>(oa::StoragePrecision::Fp32) == 0);
static_assert(static_cast<oa::U8>(oa::StoragePrecision::Bf16) == 2);
static_assert(static_cast<oa::U8>(oa::GemmKernel::Auto) == 0);
static_assert(static_cast<oa::U8>(oa::GemmKernel::TiledFp32) == 4);
static_assert(static_cast<oa::U8>(oa::GemmKernel::Naive) == 5);
static_assert(static_cast<oa::U8>(oa::GemmKernel::CoopVec) == 6);
static_assert(static_cast<oa::U8>(oa::GemmKernel::GemmCmSgBf16) == 11);
static_assert(static_cast<oa::U8>(oa::GemmKernel::GemmCmWgBf16) == 12);
static_assert(static_cast<oa::U8>(oa::GemmKernel::StridedFp32) == 13);
static_assert(static_cast<oa::U8>(oa::GemmKernel::SmallMFp32) == 14);
static_assert(static_cast<oa::U8>(oa::GemmKernel::StridedTiledFp32) == 15);
static_assert(static_cast<oa::U8>(oa::GemmPath::Standard) == 0);
static_assert(static_cast<oa::U8>(oa::GemmPath::CoopVec) == 1);
static_assert(static_cast<oa::U8>(oa::GemmPrecision::Auto) == 0);
static_assert(static_cast<oa::U8>(oa::GemmPrecision::Fp32) == 1);
static_assert(static_cast<oa::U8>(oa::GemmPrecision::Bf16) == 2);

namespace {

oa::U64 hashString(const char* inText) {
	oa::U64 h = 0xcbf29ce484222325ULL;
	for (const char* p = inText; p != nullptr and *p != '\0'; ++p) {
		h ^= static_cast<oa::U8>(*p);
		h *= 0x100000001b3ULL;
	}
	return h;
}

oa::RouteCacheKey makeRawKey(
	const oa::Engine& inRt,
	oa::U32 inM,
	oa::U32 inN,
	oa::U32 inK,
	oa::GemmPrecision inPrecision)
{
	oa::RouteCacheKey key{};
	key.vendorId = oa::EngineDeviceAccess::get(inRt).info.hardware.vendorId;
	key.deviceId = oa::EngineDeviceAccess::get(inRt).info.hardware.deviceId;
	key.driverId = oa::EngineDeviceAccess::get(inRt).info.software.driverId;
	key.driverVersionHash = hashString(oa::EngineDeviceAccess::get(inRt).info.software.driverVersion.cStr());
	key.shaderBuildId = oa::matmulRegistry::shaderBuildId();
	key.m = inM; key.n = inN; key.k = inK;
	key.batchCount = 1U;
	key.aRowStride = inK; key.aColStride = 1U; key.aBatchStride = inM * inK;
	key.bRowStride = inK; key.bColStride = 1U; key.bBatchStride = inN * inK;
	key.cRowStride = inN; key.cColStride = 1U; key.cBatchStride = inM * inN;
	key.aPrecision = oa::GemmPrecision::Fp32;
	key.bPrecision = oa::GemmPrecision::Fp32;
	key.outputPrecision = oa::GemmPrecision::Fp32;
	key.requestedPrecision = inPrecision;
	key.epilogue = oa::GemmEpilogue::None;
	key.aContiguous = true;
	key.bContiguous = true;
	key.bTransposed = true;
	key.requiresPreActivation = false;
	key.training = false;
	return key;
}

class ScopedRouteCache {
public:
	explicit ScopedRouteCache(oa::Engine& inRt)
		: rt_(inRt)
		, previous_(oa::GemmRouteCacheAccess::replaceForTesting(
			inRt, oa::makeUnique<oa::GemmRouteCache>()))
	{}
	~ScopedRouteCache() {
		(void)oa::GemmRouteCacheAccess::replaceForTesting(
			rt_, oa::move(previous_));
	}
	oa::GemmRouteCache& get() {
		return *oa::GemmRouteCacheAccess::get(rt_);
	}

private:
	oa::Engine& rt_;
	oa::UniquePtr<oa::GemmRouteCache> previous_;
};

class ScopedEnvironmentValue {
public:
	ScopedEnvironmentValue(const char* inName, const char* inValue)
		: name_(inName) {
		if (const char* previous = std::getenv(inName); previous != nullptr) {
			hadPrevious_ = true;
			previous_ = previous;
		}
		set(inValue);
	}

	~ScopedEnvironmentValue() {
		if (hadPrevious_) set(previous_.c_str());
		else clear();
	}

private:
	void set(const char* inValue) const {
#ifdef _WIN32
		_putenv_s(name_, inValue);
#else
		setenv(name_, inValue, 1);
#endif
	}

	void clear() const {
#ifdef _WIN32
		_putenv_s(name_, "");
#else
		unsetenv(name_);
#endif
	}

	const char* name_;
	bool hadPrevious_ = false;
	std::string previous_;
};

oa::Matrix executeRequiredVariant(
	const oa::Matrix& inA,
	const oa::Matrix& inB,
	const oa::Matrix* inBias,
	oa::U32 inM,
	oa::U32 inN,
	oa::U32 inK,
	oa::GemmEpilogue inEpilogue,
	oa::U64 inVariant)
{
	auto output = oa::FnMatrix::empty(
		oa::MatrixShape{inM, inN}, oa::ScalarType::Float32);
	if (output.isEmpty()) return {};
	auto& context = oa::ExecutionSession::getActive();
	const oa::Status recorded = oa::GemmGraphLowering::record(context, {
		.a = &inA,
		.b = &inB,
		.bias = inBias,
		.c = &output,
		.m = inM,
		.n = inN,
		.k = inK,
		.precision = oa::MatMulPrecision::Fp32,
		.epilogue = inEpilogue,
		.preference = {.requiredVariant = inVariant},
		.operation = "GemmRouterTest::requiredVariant",
	});
	if (not recorded.isOk()) {
		ADD_FAILURE() << recorded.getMessage().data();
		return {};
	}
	auto submitted = context.submit();
	if (not submitted.isOk()) {
		ADD_FAILURE() << submitted.getStatus().getMessage().data();
		return {};
	}
	const oa::Status waited = context.wait(submitted.getValue());
	if (not waited.isOk()) {
		ADD_FAILURE() << waited.getMessage().data();
		return {};
	}
	return output;
}

} // namespace

TEST(GemmRouter, StructuredSelectionClassificationMatchesFallbackContract) {
	ScopedEnvironmentValue permitRequestedPrecision("OA_GEMM_FORCE_FP32", "0");
	auto problem = oa::GemmRouter::problemForRaw(
		32U, 32U, 32U,
		oa::StoragePrecision::Fp32, oa::StoragePrecision::Fp32, true);
	problem.precisionHint = oa::GemmPrecision::Fp32;

	oa::MatmulPlan plan{};
	plan.kernel = oa::GemmKernel::TiledFp32;
	plan.actualPrecision = oa::GemmPrecision::Fp32;
	EXPECT_EQ(oa::GemmRouter::classifySelection(plan, problem),
		oa::KernelSelectionKind::Direct);

	plan.kernel = oa::GemmKernel::StridedTiledFp32;
	EXPECT_EQ(oa::GemmRouter::classifySelection(plan, problem),
		oa::KernelSelectionKind::LayoutFallback);

	plan.kernel = oa::GemmKernel::Naive;
	EXPECT_EQ(oa::GemmRouter::classifySelection(plan, problem),
		oa::KernelSelectionKind::NaiveFallback);

	problem.precisionHint = oa::GemmPrecision::Bf16;
	EXPECT_EQ(oa::GemmRouter::classifySelection(plan, problem),
		oa::KernelSelectionKind::PrecisionFallback);
}

TEST(GemmRegistry, StableVariantIdsAreUniqueAndResolvable) {
	std::unordered_set<oa::U64> ids;
	for (const auto& variant : oa::matmulRegistry::all()) {
		EXPECT_NE(variant.id, oa::invalidMatmulVariantId);
		EXPECT_TRUE(ids.insert(variant.id).second) << variant.kernelName;
		const auto* resolved = oa::matmulRegistry::find(variant.id);
		ASSERT_NE(resolved, nullptr);
		EXPECT_STREQ(resolved->kernelName, variant.kernelName);
		EXPECT_EQ(resolved->epilogue, variant.epilogue);
	}
}

TEST(GemmRouteCache, ExactShapesDoNotAlias) {
	oa::GemmRouteCache cache;
	oa::RouteCacheKey aligned{};
	aligned.m = 64; aligned.n = 64; aligned.k = 64;
	aligned.epilogue = oa::GemmEpilogue::None;
	oa::RouteCacheKey unaligned = aligned;
	unaligned.m = 127;

	const auto sg = oa::matmulVariantIdFromName("GemmCmSgBf16");
	cache.update(aligned, sg, 0.25F, 1);
	oa::U64 winner = oa::invalidMatmulVariantId;
	EXPECT_TRUE(cache.query(aligned, winner));
	EXPECT_EQ(winner, sg);
	EXPECT_FALSE(cache.query(unaligned, winner));
}

TEST(GemmRouteCache, LayoutAndDualOutputContractsDoNotAlias) {
	oa::GemmRouteCache cache;
	oa::RouteCacheKey base{};
	base.m = 64; base.n = 64; base.k = 64;
	base.aPrecision = oa::GemmPrecision::Fp32;
	base.bPrecision = oa::GemmPrecision::Fp32;
	base.outputPrecision = oa::GemmPrecision::Fp32;
	base.requestedPrecision = oa::GemmPrecision::Auto;
	base.epilogue = oa::GemmEpilogue::BiasGelu;
	base.aContiguous = true;
	base.bContiguous = true;
	base.bTransposed = true;
	const auto winnerId = oa::matmulVariantIdFromName("GemmBiasGeluTiled");
	cache.update(base, winnerId, 0.25F, 1);

	oa::U64 winner = oa::invalidMatmulVariantId;
	EXPECT_TRUE(cache.query(base, winner));
	EXPECT_EQ(winner, winnerId);
	auto wrongLayout = base;
	wrongLayout.bTransposed = false;
	EXPECT_FALSE(cache.query(wrongLayout, winner));
	auto wrongOutputContract = base;
	wrongOutputContract.requiresPreActivation = true;
	EXPECT_FALSE(cache.query(wrongOutputContract, winner));
}

TEST(GemmRouteCache, RequestedPrecisionPoliciesDoNotAlias) {
	oa::GemmRouteCache cache;
	oa::RouteCacheKey autoKey{};
	autoKey.m = 64; autoKey.n = 64; autoKey.k = 64;
	autoKey.aPrecision = oa::GemmPrecision::Fp32;
	autoKey.bPrecision = oa::GemmPrecision::Fp32;
	autoKey.outputPrecision = oa::GemmPrecision::Fp32;
	autoKey.requestedPrecision = oa::GemmPrecision::Auto;
	const auto tiled = oa::matmulVariantIdFromName("GemmTiled");
	cache.update(autoKey, tiled, 0.25F, 1);

	oa::U64 winner = oa::invalidMatmulVariantId;
	EXPECT_TRUE(cache.query(autoKey, winner));
	auto fp32Key = autoKey;
	fp32Key.requestedPrecision = oa::GemmPrecision::Fp32;
	EXPECT_FALSE(cache.query(fp32Key, winner));
	auto bf16Key = autoKey;
	bf16Key.requestedPrecision = oa::GemmPrecision::Bf16;
	EXPECT_FALSE(cache.query(bf16Key, winner));
}

TEST(GemmRouteCache, VersionedRoundTripRejectsLegacyBytes) {
	const char* path = "/tmp/oa_test_gemm_route_cache.bin";
	const char* legacyPath = "/tmp/oa_test_gemm_route_cache_legacy.bin";
	std::remove(path);
	std::remove(legacyPath);

	oa::GemmRouteCache source;
	oa::RouteCacheKey key{};
	key.vendorId = 0x8086U;
	key.deviceId = 0x9a49U;
	key.driverId = 3U;
	key.driverVersionHash = 17U;
	key.shaderBuildId = 23U;
	key.m = 64; key.n = 96; key.k = 32;
	key.aPrecision = oa::GemmPrecision::Fp32;
	key.bPrecision = oa::GemmPrecision::Fp32;
	key.outputPrecision = oa::GemmPrecision::Fp32;
	key.requestedPrecision = oa::GemmPrecision::Auto;
	key.epilogue = oa::GemmEpilogue::BiasGelu;
	key.aContiguous = true;
	key.bContiguous = true;
	key.bTransposed = true;
	key.requiresPreActivation = false;
	key.training = true;
	const auto tiled = oa::matmulVariantIdFromName("GemmTiled");
	source.update(key, tiled, 0.5F, 9);
	ASSERT_TRUE(source.save(path));

	oa::GemmRouteCache loaded;
	ASSERT_TRUE(loaded.load(path));
	oa::U64 winner = oa::invalidMatmulVariantId;
	EXPECT_TRUE(loaded.query(key, winner));
	EXPECT_EQ(winner, tiled);
	{
		std::FILE* f = std::fopen(path, "ab");
		ASSERT_NE(f, nullptr);
		const oa::U8 trailingGarbage = 0xffU;
		ASSERT_EQ(std::fwrite(&trailingGarbage, sizeof(trailingGarbage), 1, f), 1U);
		std::fclose(f);
	}
	oa::GemmRouteCache trailingRejected;
	trailingRejected.update(key, oa::matmulVariantIdFromName("GemmNaive"), 1.0F, 1);
	EXPECT_FALSE(trailingRejected.load(path));
	EXPECT_FALSE(trailingRejected.query(key, winner));

	{
		std::FILE* f = std::fopen(legacyPath, "wb");
		ASSERT_NE(f, nullptr);
		const oa::U64 oldEmptyCount = 0;
		ASSERT_EQ(std::fwrite(&oldEmptyCount, sizeof(oldEmptyCount), 1, f), 1U);
		std::fclose(f);
	}
	oa::GemmRouteCache rejected;
	EXPECT_FALSE(rejected.load(legacyPath));
	std::remove(path);
	std::remove(legacyPath);
}

TEST(GemmRouteCache, AggregatedTimingMetadataRoundTrips) {
	oa::GemmRouteCache source;
	oa::RouteCacheKey key{};
	key.m = 23U; key.n = 17U; key.k = 11U;
	key.batchCount = 1U;
	key.aRowStride = key.k; key.aColStride = 1U; key.aBatchStride = key.m * key.k;
	key.bRowStride = key.k; key.bColStride = 1U; key.bBatchStride = key.n * key.k;
	key.cRowStride = key.n; key.cColStride = 1U; key.cBatchStride = key.m * key.n;
	key.aPrecision = oa::GemmPrecision::Fp32;
	key.bPrecision = oa::GemmPrecision::Fp32;
	key.outputPrecision = oa::GemmPrecision::Fp32;
	key.requestedPrecision = oa::GemmPrecision::Auto;
	key.bTransposed = true;
	const auto winner = oa::matmulVariantIdFromName("GemmTiled");
	source.update(key, winner, 0.125F, 0.175F, 19U, 42U);

	const char* path = "/tmp/oa_test_gemm_route_stats.bin";
	std::remove(path);
	ASSERT_TRUE(source.save(path));
	oa::GemmRouteCache loaded;
	ASSERT_TRUE(loaded.load(path));
	ASSERT_EQ(loaded.map.size(), 1U);
	const auto& value = loaded.map.begin()->second;
	EXPECT_EQ(value.winnerVariant, winner);
	EXPECT_FLOAT_EQ(value.medianGpuTimeMs, 0.125F);
	EXPECT_FLOAT_EQ(value.p95GpuTimeMs, 0.175F);
	EXPECT_EQ(value.sampleCount, 19U);
	EXPECT_EQ(value.lastUpdatedStep, 42U);
	std::remove(path);
}

TEST(GemmRouteCache, PublicationSequenceBelongsToEachCache) {
	oa::RouteCacheKey first{};
	first.m = 8U; first.n = 8U; first.k = 8U;
	first.aRowStride = 8U; first.aBatchStride = 64U;
	first.bRowStride = 8U; first.bBatchStride = 64U;
	first.cRowStride = 8U; first.cBatchStride = 64U;
	first.bTransposed = true;
	auto second = first;
	second.m = 9U;
	second.aBatchStride = 72U;
	second.cBatchStride = 72U;
	const auto tiled = oa::matmulVariantIdFromName("GemmTiled");

	oa::GemmRouteCache cacheA;
	cacheA.publish(first, tiled, 0.10F, 0.12F, 7U);
	cacheA.publish(second, tiled, 0.11F, 0.13F, 7U);
	ASSERT_EQ(cacheA.map.size(), 2U);
	EXPECT_EQ(cacheA.map.at(first).lastUpdatedStep, 1U);
	EXPECT_EQ(cacheA.map.at(second).lastUpdatedStep, 2U);

	oa::GemmRouteCache cacheB;
	cacheB.publish(first, tiled, 0.10F, 0.12F, 7U);
	ASSERT_EQ(cacheB.map.size(), 1U);
	EXPECT_EQ(cacheB.map.at(first).lastUpdatedStep, 1U);
}

TEST(GemmRouter, Fp32RejectsIllegalCachedBf16Winner) {
	if (not vkTestEngineOk()) { GTEST_SKIP(); }
	auto& rt = testEngine();
	ScopedRouteCache scoped(rt);
	auto key = makeRawKey(rt, 64, 64, 64, oa::GemmPrecision::Fp32);
	scoped.get().update(key, oa::matmulVariantIdFromName("GemmCmSgBf16"), 0.01F, 1);

	const auto route = oa::GemmRouter::select(rt, 64, 64, 64, oa::GemmPrecision::Fp32);
	EXPECT_EQ(route.kernel, oa::GemmKernel::TiledFp32);
	EXPECT_EQ(route.actualPrec, oa::GemmPrecision::Fp32);
}

TEST(GemmRouter, Fp32ReplaysLegalCachedWinner) {
	if (not vkTestEngineOk()) { GTEST_SKIP(); }
	auto& rt = testEngine();
	ScopedRouteCache scoped(rt);
	auto key = makeRawKey(rt, 64, 64, 64, oa::GemmPrecision::Fp32);
	scoped.get().update(key, oa::matmulVariantIdFromName("GemmNaive"), 0.01F, 1);

	// Naive is legal but is not the heuristic choice for this shape. Returning
	// it proves that the explicit-FP32 cache path was actually exercised.
	const auto route = oa::GemmRouter::select(rt, 64, 64, 64, oa::GemmPrecision::Fp32);
	EXPECT_EQ(route.kernel, oa::GemmKernel::Naive);
	EXPECT_EQ(route.actualPrec, oa::GemmPrecision::Fp32);
}

TEST(GemmRouter, Fp32ShapeRoutesAreLegal) {
	if (not vkTestEngineOk()) { GTEST_SKIP(); }
	auto& rt = testEngine();
	ScopedRouteCache scoped(rt);

	const auto tiny = oa::GemmRouter::select(rt, 1, 1, 1, oa::GemmPrecision::Fp32);
	EXPECT_EQ(tiny.kernel, oa::GemmKernel::Naive);
	const auto tiled = oa::GemmRouter::select(rt, 63, 65, 17, oa::GemmPrecision::Fp32);
	EXPECT_EQ(tiled.kernel, oa::GemmKernel::TiledFp32);
	EXPECT_EQ(tiled.gx, 1U);
	EXPECT_EQ(tiled.gy, 2U);
	const auto* variant = oa::matmulRegistry::find(tiled.variant);
	ASSERT_NE(variant, nullptr);
	EXPECT_STREQ(variant->kernelName, tiled.kernelName);
}

TEST(GemmRouter, ImmutablePlanRejectsContractAndDeviceDrift) {
	if (not vkTestEngineOk()) { GTEST_SKIP(); }
	auto& rt = testEngine();
	ScopedRouteCache scoped(rt);
	auto problem = oa::GemmRouter::problemForRaw(
		130, 193, 71,
		oa::StoragePrecision::Fp32, oa::StoragePrecision::Fp32, true);
	problem.precisionHint = oa::GemmPrecision::Fp32;
	problem.training = true;

	const auto plan = oa::GemmRouter::plan(rt, problem);
	ASSERT_TRUE(static_cast<bool>(plan));
	EXPECT_TRUE(oa::GemmRouter::validatePlan(rt, plan, problem));
	EXPECT_NE(plan.shaderContentHash, 0U);
	EXPECT_EQ(plan.grid.x, 3U);
	EXPECT_EQ(plan.grid.y, 4U);

	auto wrongProblem = problem;
	wrongProblem.k += 1U;
	EXPECT_FALSE(oa::GemmRouter::validatePlan(rt, plan, wrongProblem));

	auto stalePlan = plan;
	stalePlan.registryBuildId ^= 1U;
	EXPECT_FALSE(oa::GemmRouter::validatePlan(rt, stalePlan, problem));

	auto wrongGrid = plan;
	wrongGrid.grid.x += 1U;
	EXPECT_FALSE(oa::GemmRouter::validatePlan(rt, wrongGrid, problem));

	auto wrongShader = plan;
	wrongShader.shaderContentHash ^= 1U;
	EXPECT_FALSE(oa::GemmRouter::validatePlan(rt, wrongShader, problem));
}

TEST(GemmRouter, CanonicalLaunchDescribesCoopVecAbi) {
	auto problem = oa::GemmRouter::problemForRaw(
		1, 37, 19,
		oa::StoragePrecision::Fp32, oa::StoragePrecision::Fp32, true);
	oa::MatmulPlan validatedPlan{
		.variant = 1U,
		.kernelName = "GemmCoopVec",
		.kernel = oa::GemmKernel::CoopVec,
		.path = oa::GemmPath::CoopVec,
		.grid = {.x = 2U, .y = 1U, .z = 1U},
	};

	auto described = oa::GemmDispatch::describeValidatedPlan(validatedPlan, problem);
	ASSERT_TRUE(described.isOk());
	const auto& launch = described.getValue();
	EXPECT_STREQ(launch.kernelName, validatedPlan.kernelName);
	EXPECT_EQ(launch.bufferCount, 3U);
	EXPECT_EQ(launch.bufferOrder[0], 1U);
	EXPECT_EQ(launch.bufferOrder[1], 0U);
	EXPECT_EQ(launch.bufferOrder[2], 2U);
	EXPECT_EQ(launch.grid.x, 2U);
	ASSERT_EQ(launch.pushSize, 2U * sizeof(oa::U32));
	oa::U32 push[2]{};
	std::memcpy(push, launch.pushData, launch.pushSize);
	EXPECT_EQ(push[0], problem.n);
	EXPECT_EQ(push[1], problem.k);
}

TEST(GemmRouter, RawPlanPreservesLegacySelectionContract) {
	if (not vkTestEngineOk()) { GTEST_SKIP(); }
	auto& rt = testEngine();
	ScopedRouteCache scoped(rt);
	const auto legacy = oa::GemmRouter::select(
		rt, 64, 96, 32, oa::GemmPrecision::Fp32);
	auto problem = oa::GemmRouter::problemForRaw(
		64, 96, 32,
		oa::StoragePrecision::Fp32, oa::StoragePrecision::Fp32, true);
	problem.training = false;
	problem.precisionHint = oa::GemmPrecision::Fp32;
	const auto plan = oa::GemmRouter::plan(rt, problem);

	ASSERT_TRUE(static_cast<bool>(plan));
	EXPECT_EQ(plan.variant, legacy.variant);
	EXPECT_EQ(plan.kernel, legacy.kernel);
	EXPECT_EQ(plan.grid.x, legacy.gx);
	EXPECT_EQ(plan.grid.y, legacy.gy);
	EXPECT_EQ(plan.grid.z, legacy.gz);
	EXPECT_TRUE(oa::GemmRouter::validatePlan(rt, plan, problem));
}

TEST(GemmRouter, PlanPreferenceCanBypassMeasuredCache) {
	if (not vkTestEngineOk()) { GTEST_SKIP(); }
	auto& rt = testEngine();
	ScopedRouteCache scoped(rt);
	auto problem = oa::GemmRouter::problemForRaw(
		64, 64, 64,
		oa::StoragePrecision::Fp32, oa::StoragePrecision::Fp32, true);
	problem.precisionHint = oa::GemmPrecision::Fp32;
	problem.training = false;
	scoped.get().update(
		oa::GemmRouter::cacheKey(rt, problem),
		oa::matmulVariantIdFromName("GemmNaive"), 0.01F, 1);

	const auto cached = oa::GemmRouter::plan(rt, problem);
	EXPECT_EQ(cached.kernel, oa::GemmKernel::Naive);

	oa::MatmulPreference heuristicOnly{};
	heuristicOnly.useMeasuredCache = false;
	const auto heuristic = oa::GemmRouter::plan(rt, problem, heuristicOnly);
	EXPECT_EQ(heuristic.kernel, oa::GemmKernel::TiledFp32);

	{
		ScopedEnvironmentValue disableCache("OA_DISABLE_GEMM_ROUTE_CACHE", "1");
		const auto environmentHeuristic = oa::GemmRouter::plan(rt, problem);
		EXPECT_EQ(environmentHeuristic.kernel, oa::GemmKernel::TiledFp32);
	}
	const auto restored = oa::GemmRouter::plan(rt, problem);
	EXPECT_EQ(restored.kernel, oa::GemmKernel::Naive);
}

TEST(GemmRouter, RequiredVariantIsRequestLocalAndFailsClosed) {
	if (not vkTestEngineOk()) { GTEST_SKIP(); }
	auto& rt = testEngine();
	ScopedRouteCache scoped(rt);
	auto problem = oa::GemmRouter::problemForRaw(
		64U, 64U, 64U,
		oa::StoragePrecision::Fp32, oa::StoragePrecision::Fp32, true);
	problem.precisionHint = oa::GemmPrecision::Fp32;

	const auto naiveId = oa::matmulVariantIdFromName("GemmNaive");
	const auto required = oa::GemmRouter::plan(
		rt, problem, {.requiredVariant = naiveId});
	ASSERT_TRUE(static_cast<bool>(required));
	EXPECT_EQ(required.variant, naiveId);

	const auto ordinary = oa::GemmRouter::plan(rt, problem);
	ASSERT_TRUE(static_cast<bool>(ordinary));
	EXPECT_EQ(ordinary.kernel, oa::GemmKernel::TiledFp32);

	const auto incompatible = oa::GemmRouter::plan(rt, problem, {
		.requiredVariant = oa::matmulVariantIdFromName("GemmCmSgBf16"),
	});
	EXPECT_FALSE(static_cast<bool>(incompatible));
}

TEST(GemmRouter, GeneratedVariantHonorsDeviceLaunchLimits) {
	if (not vkTestEngineOk()) { GTEST_SKIP(); }
	auto& rt = testEngine();
	auto& hw = oa::EngineDeviceAccess::get(rt).info.hardware;
	const oa::U32 savedShared = hw.maxComputeSharedMemoryBytes;
	const oa::U32 savedInvocations = hw.maxComputeWorkGroupInvocations;
	const oa::U32 savedSize = hw.maxComputeWorkGroupSize;

	auto problem = oa::GemmRouter::problemForRaw(
		4096, 384, 1536,
		oa::StoragePrecision::Fp32, oa::StoragePrecision::Fp32, true);
	problem.training = true;
	problem.precisionHint = oa::GemmPrecision::Fp32;
	const auto* control = oa::matmulRegistry::find(
		oa::matmulVariantIdFromName("GemmTiled"));
	const auto* k32 = oa::matmulRegistry::find(
		oa::matmulVariantIdFromName("GemmTiledK32"));
	const auto* narrowN = oa::matmulRegistry::find(
		oa::matmulVariantIdFromName("GemmTiledN32"));
	ASSERT_NE(control, nullptr);
	ASSERT_NE(k32, nullptr);
	ASSERT_NE(narrowN, nullptr);

	// K=32 needs 16,896 bytes while the K=16 control needs 8,704. A device at
	// vulkan's 16 KiB minimum must retain the control and reject K=32.
	hw.maxComputeSharedMemoryBytes = 16U * 1024U;
	EXPECT_TRUE(oa::GemmRouter::isVariantLegal(rt, *control, problem));
	EXPECT_FALSE(oa::GemmRouter::isVariantLegal(rt, *k32, problem));

	// The same legality boundary owns both x-size and total-invocation limits.
	hw.maxComputeSharedMemoryBytes = savedShared;
	hw.maxComputeWorkGroupInvocations = 128U;
	hw.maxComputeWorkGroupSize = 128U;
	EXPECT_FALSE(oa::GemmRouter::isVariantLegal(rt, *control, problem));
	EXPECT_TRUE(oa::GemmRouter::isVariantLegal(rt, *narrowN, problem));

	hw.maxComputeSharedMemoryBytes = savedShared;
	hw.maxComputeWorkGroupInvocations = savedInvocations;
	hw.maxComputeWorkGroupSize = savedSize;
}

TEST(GemmRouter, FusedFp32RoutesPreserveExactEpilogue) {
	if (not vkTestEngineOk()) { GTEST_SKIP(); }
	auto& rt = testEngine();
	ScopedRouteCache scoped(rt);
	auto problem = oa::GemmRouter::problemForRaw(
		64, 64, 64,
		oa::StoragePrecision::Fp32, oa::StoragePrecision::Fp32, true);
	problem.precisionHint = oa::GemmPrecision::Fp32;
	problem.training = true;

	problem.epilogue = oa::GemmEpilogue::Bias;
	const auto bias = oa::GemmRouter::select(rt, problem);
	EXPECT_STREQ(bias.kernelName, "GemmBiasTiled");
	problem.epilogue = oa::GemmEpilogue::BiasRelu;
	const auto relu = oa::GemmRouter::select(rt, problem);
	EXPECT_STREQ(relu.kernelName, "GemmBiasReluTiled");
	problem.epilogue = oa::GemmEpilogue::BiasGelu;
	const auto gelu = oa::GemmRouter::select(rt, problem);
	EXPECT_STREQ(gelu.kernelName, "GemmBiasGeluTiled");
	problem.epilogue = oa::GemmEpilogue::BiasSilu;
	const auto silu = oa::GemmRouter::select(rt, problem);
	EXPECT_STREQ(silu.kernelName, "GemmBiasSiluTiled");

	EXPECT_NE(bias.variant, relu.variant);
	EXPECT_NE(relu.variant, gelu.variant);
	EXPECT_NE(gelu.variant, silu.variant);
	EXPECT_EQ(oa::matmulRegistry::find(bias.variant)->epilogue, oa::GemmEpilogue::Bias);
	EXPECT_EQ(oa::matmulRegistry::find(relu.variant)->epilogue, oa::GemmEpilogue::BiasRelu);
	EXPECT_EQ(oa::matmulRegistry::find(gelu.variant)->epilogue, oa::GemmEpilogue::BiasGelu);
	EXPECT_EQ(oa::matmulRegistry::find(silu.variant)->epilogue, oa::GemmEpilogue::BiasSilu);
}

TEST(GemmRouter, RejectsCachedWinnerFromDifferentEpilogue) {
	if (not vkTestEngineOk()) { GTEST_SKIP(); }
	auto& rt = testEngine();
	ScopedRouteCache scoped(rt);
	auto key = makeRawKey(rt, 64, 64, 64, oa::GemmPrecision::Fp32);
	key.epilogue = oa::GemmEpilogue::BiasRelu;
	key.training = true;
	scoped.get().update(
		key, oa::matmulVariantIdFromName("GemmBiasGeluTiled"), 0.01F, 1);

	auto problem = oa::GemmRouter::problemForRaw(
		64, 64, 64,
		oa::StoragePrecision::Fp32, oa::StoragePrecision::Fp32, true);
	problem.precisionHint = oa::GemmPrecision::Fp32;
	problem.epilogue = oa::GemmEpilogue::BiasRelu;
	problem.training = true;
	const auto route = oa::GemmRouter::select(rt, problem);
	EXPECT_STREQ(route.kernelName, "GemmBiasReluTiled");
}

TEST(GemmTuner, RuntimeLoweringBenchmarksEverySupportedEpilogue) {
	if (not vkTestEngineOk()) { GTEST_SKIP(); }
	auto& rt = testEngine();
	ScopedRouteCache scoped(rt);
	const oa::GemmEpilogue epilogues[] = {
		oa::GemmEpilogue::None,
		oa::GemmEpilogue::Bias,
		oa::GemmEpilogue::BiasRelu,
		oa::GemmEpilogue::BiasGelu,
		oa::GemmEpilogue::BiasSilu,
	};
	for (const auto epilogue : epilogues) {
		oa::GemmTunerResult result{};
		const oa::GemmTunerShape shape{
			.m = 17U,
			.n = 19U,
			.k = 13U,
			.name = "runtime_lowering_epilogue",
			.epilogue = epilogue,
		};
		const oa::Status status =
			oa::GemmTuner::benchmarkShape(rt, shape, 1U, 1U, result);
		ASSERT_TRUE(status.isOk()) << status.getMessage().data();
		ASSERT_FALSE(result.rankedCandidates.empty());
		for (const auto& candidate : result.rankedCandidates) {
			const auto* variant = oa::matmulRegistry::find(candidate.variant);
			ASSERT_NE(variant, nullptr);
			EXPECT_EQ(variant->epilogue, epilogue);
			EXPECT_NE(variant->kernel, oa::GemmKernel::StridedFp32)
				<< "canonical tuning must not benchmark the layout fallback";
			EXPECT_NE(variant->kernel, oa::GemmKernel::StridedTiledFp32)
				<< "canonical tuning must not benchmark a strided specialization";
		}
	}
}

TEST(GemmTuner, DefaultShapesCoverNlpFusedContracts) {
	bool qkvBias = false;
	bool ffn1BiasGelu = false;
	bool ffn2Bias = false;
	for (const auto& shape : oa::GemmTuner::getDefaultShapes()) {
		qkvBias |= shape.m == 1024U and shape.n == 32U and shape.k == 32U
			and shape.epilogue == oa::GemmEpilogue::Bias;
		ffn1BiasGelu |= shape.m == 1024U and shape.n == 64U and shape.k == 32U
			and shape.epilogue == oa::GemmEpilogue::BiasGelu;
		ffn2Bias |= shape.m == 1024U and shape.n == 32U and shape.k == 64U
			and shape.epilogue == oa::GemmEpilogue::Bias;
	}
	EXPECT_TRUE(qkvBias);
	EXPECT_TRUE(ffn1BiasGelu);
	EXPECT_TRUE(ffn2Bias);
}

TEST(GemmTuner, NumericalOracleRejectsCorruptOutput) {
	const oa::GemmTunerShape shape{
		.m = 2U,
		.n = 3U,
		.k = 4U,
		.name = "numerical_publish_oracle",
		.epilogue = oa::GemmEpilogue::BiasGelu,
	};
	const std::vector<oa::F32> a = {
		0.25F, -0.50F, 0.75F, 1.00F,
		-1.00F, 0.50F, 0.25F, -0.75F,
	};
	const std::vector<oa::F32> b = {
		0.50F, 0.25F, -0.50F, 1.00F,
		-0.25F, 0.75F, 0.50F, -0.50F,
		1.00F, -0.50F, 0.25F, 0.75F,
	};
	const std::vector<oa::F32> bias = {0.10F, -0.20F, 0.30F};
	std::vector<oa::F32> output(shape.m * shape.n);
	for (oa::U32 row = 0U; row < shape.m; ++row) {
		for (oa::U32 col = 0U; col < shape.n; ++col) {
			oa::F32 value = bias[col];
			for (oa::U32 k = 0U; k < shape.k; ++k) {
				value += a[row * shape.k + k] * b[col * shape.k + k];
			}
			const oa::F32 x3 = value * value * value;
			output[row * shape.n + col] = 0.5F * value * (1.0F + std::tanh(
				0.7978845608F * (value + 0.044715F * x3)));
		}
	}
	auto validate = [&](const std::vector<oa::F32>& candidate) {
		return oa::GemmTuner::validateNumericalOutput(
			shape,
			oa::Span<const oa::F32>(a.data(), a.size()),
			oa::Span<const oa::F32>(b.data(), b.size()),
			oa::Span<const oa::F32>(bias.data(), bias.size()),
			oa::Span<const oa::F32>(candidate.data(), candidate.size()));
	};
	EXPECT_TRUE(validate(output).isOk());
	output[0] += 0.25F;
	const oa::Status corrupt = validate(output);
	EXPECT_EQ(corrupt.getCode(), oa::StatusCode::DataLoss);
	EXPECT_NE(corrupt.getMessage().find("row=0 col=0"), oa::String::Npos);

	// Cross the exhaustive-output threshold and corrupt the bottom-right tail,
	// which is part of the bounded product-shape sample set.
	const oa::GemmTunerShape productShape{
		.m = 65U,
		.n = 65U,
		.k = 1U,
		.name = "numerical_publish_sampled_oracle",
		.epilogue = oa::GemmEpilogue::None,
	};
	const std::vector<oa::F32> productA(productShape.m, 0.5F);
	const std::vector<oa::F32> productB(productShape.n, -0.25F);
	std::vector<oa::F32> productOutput(productShape.m * productShape.n, -0.125F);
	productOutput.back() = 1.0F;
	const oa::Status tailCorrupt = oa::GemmTuner::validateNumericalOutput(
		productShape,
		oa::Span<const oa::F32>(productA.data(), productA.size()),
		oa::Span<const oa::F32>(productB.data(), productB.size()),
		{},
		oa::Span<const oa::F32>(productOutput.data(), productOutput.size()));
	EXPECT_EQ(tailCorrupt.getCode(), oa::StatusCode::DataLoss);
	EXPECT_NE(tailCorrupt.getMessage().find("row=64 col=64"), oa::String::Npos);
}

TEST(GemmRouter, GeneratedTiledVariantsMatchCpuOnIrregularTails) {
	if (not vkTestEngineOk()) { GTEST_SKIP(); }

	// Deliberately crosses several 64x64 tiles in both dimensions and leaves
	// partial M/N/K tails. Every generated raw FP32 tiled variant is forced, so
	// adding schema geometry automatically expands the correctness gate.
	constexpr oa::U32 M = 130U;
	constexpr oa::U32 N = 193U;
	constexpr oa::U32 K = 71U;
	std::vector<oa::F32> aData(M * K);
	std::vector<oa::F32> bData(N * K);
	for (oa::U32 i = 0; i < M * K; ++i) {
		aData[i] = static_cast<oa::F32>(static_cast<oa::I32>(i % 29U) - 14) * 0.03125F;
	}
	for (oa::U32 i = 0; i < N * K; ++i) {
		bData[i] = static_cast<oa::F32>(static_cast<oa::I32>(i % 23U) - 11) * 0.025F;
	}

	auto a = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(aData.data()), aData.size() * sizeof(oa::F32)),
		oa::MatrixShape{M, K});
	auto b = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(bData.data()), bData.size() * sizeof(oa::F32)),
		oa::MatrixShape{N, K});

	for (const auto& variant : oa::matmulRegistry::all()) {
		if (variant.kernel != oa::GemmKernel::TiledFp32
			or variant.epilogue != oa::GemmEpilogue::None)
		{
			continue;
		}
		auto c = executeRequiredVariant(
			a, b, nullptr, M, N, K, oa::GemmEpilogue::None, variant.id);

		std::vector<oa::F32> got(M * N);
		ASSERT_TRUE(oa::FnMatrix::copyToHost(c, got.data(), got.size() * sizeof(oa::F32)).isOk());
		for (oa::U32 row = 0; row < M; ++row) {
			for (oa::U32 col = 0; col < N; ++col) {
				oa::F32 expected = 0.0F;
				for (oa::U32 k = 0; k < K; ++k) {
					expected += aData[row * K + k] * bData[col * K + k];
				}
				EXPECT_NEAR(got[row * N + col], expected, 2.0e-4F)
					<< variant.kernelName << " row=" << row << " col=" << col;
			}
		}
	}
}

TEST(GemmRouter, GeneratedTiledEpiloguesMatchCpuOnIrregularTails) {
	if (not vkTestEngineOk()) { GTEST_SKIP(); }

	constexpr oa::U32 M = 67U;
	constexpr oa::U32 N = 73U;
	constexpr oa::U32 K = 19U;
	std::vector<oa::F32> xData(M * K);
	std::vector<oa::F32> wData(N * K);
	std::vector<oa::F32> biasData(N);
	for (oa::U32 i = 0; i < M * K; ++i) {
		xData[i] = static_cast<oa::F32>(static_cast<oa::I32>(i % 17U) - 8) * 0.03125F;
	}
	for (oa::U32 i = 0; i < N * K; ++i) {
		wData[i] = static_cast<oa::F32>(static_cast<oa::I32>(i % 13U) - 6) * 0.025F;
	}
	for (oa::U32 i = 0; i < N; ++i) {
		biasData[i] = static_cast<oa::F32>(static_cast<oa::I32>(i % 11U) - 5) * 0.02F;
	}

	auto x = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(xData.data()), xData.size() * sizeof(oa::F32)),
		oa::MatrixShape{M, K});
	auto w = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(wData.data()), wData.size() * sizeof(oa::F32)),
		oa::MatrixShape{N, K});
	auto bias = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(biasData.data()), biasData.size() * sizeof(oa::F32)),
		oa::MatrixShape{N});

	for (const auto& variant : oa::matmulRegistry::all()) {
		if (variant.kernel != oa::GemmKernel::TiledFp32
			or variant.epilogue == oa::GemmEpilogue::None) {
			continue;
		}
		auto output = executeRequiredVariant(
			x, w, &bias, M, N, K, variant.epilogue, variant.id);

		std::vector<oa::F32> got(M * N);
		ASSERT_TRUE(oa::FnMatrix::copyToHost(
			output, got.data(), got.size() * sizeof(oa::F32)).isOk());
		for (oa::U32 row = 0; row < M; ++row) {
			for (oa::U32 col = 0; col < N; ++col) {
				oa::F32 expected = biasData[col];
				for (oa::U32 k = 0; k < K; ++k) {
					expected += xData[row * K + k] * wData[col * K + k];
				}
				if (variant.epilogue == oa::GemmEpilogue::BiasRelu) {
					expected = std::max(0.0F, expected);
				} else if (variant.epilogue == oa::GemmEpilogue::BiasGelu) {
					const oa::F32 x3 = expected * expected * expected;
					expected = 0.5F * expected * (1.0F + std::tanh(
						0.7978845608F * (expected + 0.044715F * x3)));
				} else if (variant.epilogue == oa::GemmEpilogue::BiasSilu) {
					expected /= 1.0F + std::exp(-expected);
				}
				EXPECT_NEAR(got[row * N + col], expected, 3.0e-4F)
					<< variant.kernelName << " row=" << row << " col=" << col;
			}
		}
	}
}

TEST(GemmRouter, GeneratedSmallMVariantsMatchCpuOnIrregularTails) {
	if (not vkTestEngineOk()) { GTEST_SKIP(); }

	// Stay inside the generated maxM=4 legality boundary while retaining odd
	// M/N/K tails. A larger M would make the forced variant illegal and could
	// accidentally validate the ordinary tiled fallback instead.
	constexpr oa::U32 M = 3U;
	constexpr oa::U32 N = 37U;
	constexpr oa::U32 K = 71U;
	std::vector<oa::F32> xData(M * K);
	std::vector<oa::F32> wData(N * K);
	std::vector<oa::F32> biasData(N);
	for (oa::U32 i = 0; i < M * K; ++i) {
		xData[i] = static_cast<oa::F32>(static_cast<oa::I32>(i % 19U) - 9) * 0.025F;
	}
	for (oa::U32 i = 0; i < N * K; ++i) {
		wData[i] = static_cast<oa::F32>(static_cast<oa::I32>(i % 17U) - 8) * 0.03125F;
	}
	for (oa::U32 i = 0; i < N; ++i) {
		biasData[i] = static_cast<oa::F32>(static_cast<oa::I32>(i % 13U) - 6) * 0.02F;
	}

	auto x = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(xData.data()),
			xData.size() * sizeof(oa::F32)), oa::MatrixShape{M, K});
	auto w = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(wData.data()),
			wData.size() * sizeof(oa::F32)), oa::MatrixShape{N, K});
	auto bias = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(biasData.data()),
			biasData.size() * sizeof(oa::F32)), oa::MatrixShape{N});

	oa::U32 tested = 0U;
	for (const auto& variant : oa::matmulRegistry::all()) {
		if (variant.kernel != oa::GemmKernel::SmallMFp32) continue;
		++tested;
		auto forcedProblem = oa::GemmRouter::problemForRaw(
			M, N, K,
			oa::StoragePrecision::Fp32, oa::StoragePrecision::Fp32, true);
		forcedProblem.epilogue = variant.epilogue;
		const auto forcedPlan = oa::GemmRouter::plan(
			testEngine(), forcedProblem,
			{.requiredVariant = variant.id});
		EXPECT_EQ(forcedPlan.variant, variant.id)
			<< "forced parity test did not select " << variant.kernelName;
		const oa::Matrix* variantBias = variant.epilogue == oa::GemmEpilogue::None
			? nullptr : &bias;
		auto output = executeRequiredVariant(
			x, w, variantBias, M, N, K, variant.epilogue, variant.id);

		std::vector<oa::F32> got(M * N);
		ASSERT_TRUE(oa::FnMatrix::copyToHost(
			output, got.data(), got.size() * sizeof(oa::F32)).isOk());
		for (oa::U32 row = 0; row < M; ++row) {
			for (oa::U32 col = 0; col < N; ++col) {
				oa::F32 expected = variant.epilogue == oa::GemmEpilogue::None
					? 0.0F : biasData[col];
				for (oa::U32 k = 0; k < K; ++k) {
					expected += xData[row * K + k] * wData[col * K + k];
				}
				if (variant.epilogue == oa::GemmEpilogue::BiasRelu) {
					expected = std::max(0.0F, expected);
				} else if (variant.epilogue == oa::GemmEpilogue::BiasGelu) {
					const oa::F32 x3 = expected * expected * expected;
					expected = 0.5F * expected * (1.0F + std::tanh(
						0.7978845608F * (expected + 0.044715F * x3)));
				} else if (variant.epilogue == oa::GemmEpilogue::BiasSilu) {
					expected /= 1.0F + std::exp(-expected);
				}
				EXPECT_NEAR(got[row * N + col], expected, 5.0e-4F)
					<< variant.kernelName << " row=" << row << " col=" << col;
			}
		}
	}
	EXPECT_EQ(tested, 5U);
}

TEST(GemmRouter, SmallMMaximumIsAnExactLegalityBoundary) {
	if (not vkTestEngineOk()) { GTEST_SKIP(); }
	auto small = oa::GemmRouter::problemForRaw(
		4U, 256U, 256U,
		oa::StoragePrecision::Fp32, oa::StoragePrecision::Fp32, true);
	oa::MatmulPreference heuristicOnly{};
	heuristicOnly.useMeasuredCache = false;
	const auto smallPlan = oa::GemmRouter::plan(
		testEngine(), small, heuristicOnly);
	ASSERT_TRUE(static_cast<bool>(smallPlan));
	EXPECT_EQ(smallPlan.kernel, oa::GemmKernel::SmallMFp32);

	auto ordinary = small;
	ordinary.m = 5U;
	ordinary.a.batchStride = ordinary.m * ordinary.k;
	ordinary.c.batchStride = ordinary.m * ordinary.n;
	const auto ordinaryPlan = oa::GemmRouter::plan(
		testEngine(), ordinary, heuristicOnly);
	ASSERT_TRUE(static_cast<bool>(ordinaryPlan));
	EXPECT_NE(ordinaryPlan.kernel, oa::GemmKernel::SmallMFp32);
}

TEST(GemmRouter, ArbitraryRowAndColumnStridesMatchCpu) {
	if (not vkTestEngineOk()) { GTEST_SKIP(); }
	const std::vector<oa::F32> aStorage = {
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
	};
	const std::vector<oa::F32> bStorage = {
		1, 2,
		3, 4,
		5, 6,
	};
	auto aBase = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(aStorage.data()),
			aStorage.size() * sizeof(oa::F32)), oa::MatrixShape{3, 4});
	auto bBase = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(bStorage.data()),
			bStorage.size() * sizeof(oa::F32)), oa::MatrixShape{3, 2});
	const oa::I32 permutation[] = {1, 0};
	auto a = aBase.permute(oa::Span<const oa::I32>(permutation)); // [4,3], strides [1,4]
	auto b = bBase.permute(oa::Span<const oa::I32>(permutation)); // [2,3], strides [1,2]
	auto c = oa::FnMatrix::matMulNt(a, b, oa::MatMulPrecision::Fp32);

	std::vector<oa::F32> got(8);
	ASSERT_TRUE(oa::FnMatrix::copyToHost(c, got.data(), got.size() * sizeof(oa::F32)).isOk());
	const std::vector<oa::F32> expected = {61, 76, 70, 88, 79, 100, 88, 112};
	EXPECT_EQ(got, expected);
}

TEST(GemmRouter, StridedTiledHeuristicKeepsSmallAndSkinnyWorkScalar) {
	if (not vkTestEngineOk()) { GTEST_SKIP(); }
	oa::MatmulPreference heuristicOnly{};
	heuristicOnly.useMeasuredCache = false;
	auto planBatch = [&](oa::U32 M, oa::U32 N, oa::U32 K) {
		auto problem = oa::GemmRouter::problemForRaw(
			M, N, K,
			oa::StoragePrecision::Fp32, oa::StoragePrecision::Fp32, true);
		problem.batchCount = 8U;
		problem.a.batchStride = M * K;
		problem.b.batchStride = N * K;
		problem.c.batchStride = M * N;
		problem.precisionHint = oa::GemmPrecision::Fp32;
		return oa::GemmRouter::plan(testEngine(), problem, heuristicOnly);
	};

	const auto medium = planBatch(32U, 32U, 32U);
	ASSERT_TRUE(static_cast<bool>(medium));
	EXPECT_EQ(medium.kernel, oa::GemmKernel::StridedTiledFp32);

	const auto small = planBatch(16U, 16U, 64U);
	ASSERT_TRUE(static_cast<bool>(small));
	EXPECT_EQ(small.kernel, oa::GemmKernel::StridedFp32);

	// Equal arithmetic work to the tiled threshold, but too little reuse in
	// either output dimension to amortize a 64x64 cooperative workgroup.
	const auto skinny = planBatch(8U, 8U, 512U);
	ASSERT_TRUE(static_cast<bool>(skinny));
	EXPECT_EQ(skinny.kernel, oa::GemmKernel::StridedFp32);
}

TEST(GemmRouter, StridedBatchPlanMatchesCpu) {
	if (not vkTestEngineOk()) { GTEST_SKIP(); }
	auto& rt = testEngine();
	const std::vector<oa::F32> aData = {
		1, 2, 3, 4, 5, 6,
		2, 0, 1, 1, 3, 2,
	};
	const std::vector<oa::F32> bData = {
		1, 0, 1, 0, 1, 1,
		1, 2, 0, 0, 1, 2,
	};
	auto a = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(aData.data()),
			aData.size() * sizeof(oa::F32)), oa::MatrixShape{2, 2, 3});
	auto b = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(bData.data()),
			bData.size() * sizeof(oa::F32)), oa::MatrixShape{2, 2, 3});
	auto c = oa::FnMatrix::empty(oa::MatrixShape{2, 2, 2});
	auto problem = oa::GemmRouter::problemForRaw(
		2, 2, 3, oa::StoragePrecision::Fp32, oa::StoragePrecision::Fp32, true);
	problem.batchCount = 2;
	problem.a.batchStride = 6;
	problem.b.batchStride = 6;
	problem.c.batchStride = 4;
	const auto plan = oa::GemmRouter::plan(rt, problem);
	ASSERT_TRUE(static_cast<bool>(plan));
	EXPECT_EQ(plan.kernel, oa::GemmKernel::StridedFp32);
	EXPECT_EQ(plan.grid.z, 2U);
	oavk::Buffer buffers[] = {oa::MatrixAccess::descriptor(a),
		oa::MatrixAccess::descriptor(b), oa::MatrixAccess::descriptor(c)};
	ASSERT_TRUE(oa::GemmDispatch::executePlan(rt, plan, problem, buffers).isOk());

	std::vector<oa::F32> got(8);
	ASSERT_TRUE(oa::FnMatrix::copyToHost(c, got.data(), got.size() * sizeof(oa::F32)).isOk());
	const std::vector<oa::F32> expected = {4, 5, 10, 11, 2, 2, 7, 7};
	EXPECT_EQ(got, expected);
}

TEST(GemmRouter, TiledStridedBatchMatchesCpuOnOffsetsAndOddTails) {
	if (not vkTestEngineOk()) { GTEST_SKIP(); }
	constexpr oa::U32 batchCount = 2U;
	constexpr oa::U32 M = 65U;
	constexpr oa::U32 N = 67U;
	constexpr oa::U32 K = 19U;
	const oa::MatmulLayout aLayout{
		.offset = 3U,
		.rowStride = 39U,
		.colStride = 2U,
		.batchStride = 2542U,
	};
	const oa::MatmulLayout bLayout{
		.offset = 5U,
		.rowStride = 59U,
		.colStride = 3U,
		.batchStride = 3964U,
	};
	const oa::MatmulLayout cLayout{
		.offset = 7U,
		.rowStride = 137U,
		.colStride = 2U,
		.batchStride = 8918U,
	};
	const oa::U32 aElements = aLayout.offset
		+ (batchCount - 1U) * aLayout.batchStride
		+ (M - 1U) * aLayout.rowStride + (K - 1U) * aLayout.colStride + 1U;
	const oa::U32 bElements = bLayout.offset
		+ (batchCount - 1U) * bLayout.batchStride
		+ (N - 1U) * bLayout.rowStride + (K - 1U) * bLayout.colStride + 1U;
	const oa::U32 cElements = cLayout.offset
		+ (batchCount - 1U) * cLayout.batchStride
		+ (M - 1U) * cLayout.rowStride + (N - 1U) * cLayout.colStride + 1U;
	std::vector<oa::F32> aData(aElements, -17.0F);
	std::vector<oa::F32> bData(bElements, -19.0F);
	std::vector<oa::F32> cData(cElements, -23.0F);
	for (oa::U32 batch = 0U; batch < batchCount; ++batch) {
		for (oa::U32 row = 0U; row < M; ++row) {
			for (oa::U32 k = 0U; k < K; ++k) {
				const oa::U32 index = aLayout.offset + batch * aLayout.batchStride
					+ row * aLayout.rowStride + k * aLayout.colStride;
				aData[index] = static_cast<oa::F32>(
					static_cast<oa::I32>((batch * 7U + row * 3U + k) % 23U) - 11)
					* 0.03125F;
			}
		}
		for (oa::U32 col = 0U; col < N; ++col) {
			for (oa::U32 k = 0U; k < K; ++k) {
				const oa::U32 index = bLayout.offset + batch * bLayout.batchStride
					+ col * bLayout.rowStride + k * bLayout.colStride;
				bData[index] = static_cast<oa::F32>(
					static_cast<oa::I32>((batch * 5U + col * 2U + k * 3U) % 19U) - 9)
					* 0.025F;
			}
		}
	}

	auto a = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(aData.data()),
			aData.size() * sizeof(oa::F32)), oa::MatrixShape{aElements});
	auto b = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(bData.data()),
			bData.size() * sizeof(oa::F32)), oa::MatrixShape{bElements});
	auto c = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(cData.data()),
			cData.size() * sizeof(oa::F32)), oa::MatrixShape{cElements});
	auto problem = oa::GemmRouter::problemForRaw(
		M, N, K, oa::StoragePrecision::Fp32, oa::StoragePrecision::Fp32, true);
	problem.batchCount = batchCount;
	problem.a = aLayout;
	problem.b = bLayout;
	problem.c = cLayout;
	problem.aContiguous = false;
	problem.bContiguous = false;
	problem.precisionHint = oa::GemmPrecision::Fp32;
	oa::MatmulPreference heuristicOnly{};
	heuristicOnly.useMeasuredCache = false;
	const auto plan = oa::GemmRouter::plan(
		testEngine(), problem, heuristicOnly);
	ASSERT_TRUE(static_cast<bool>(plan));
	EXPECT_EQ(plan.kernel, oa::GemmKernel::StridedTiledFp32);
	EXPECT_EQ(plan.grid.x, 2U);
	EXPECT_EQ(plan.grid.y, 2U);
	EXPECT_EQ(plan.grid.z, batchCount);
	oavk::Buffer buffers[] = {oa::MatrixAccess::descriptor(a),
		oa::MatrixAccess::descriptor(b), oa::MatrixAccess::descriptor(c)};
	ASSERT_TRUE(oa::GemmDispatch::executePlan(
		testEngine(), plan, problem, buffers).isOk());

	std::vector<oa::F32> got(cElements);
	ASSERT_TRUE(oa::FnMatrix::copyToHost(
		c, got.data(), got.size() * sizeof(oa::F32)).isOk());
	for (oa::U32 batch = 0U; batch < batchCount; ++batch) {
		for (oa::U32 row = 0U; row < M; ++row) {
			for (oa::U32 col = 0U; col < N; ++col) {
				oa::F32 expected = 0.0F;
				for (oa::U32 k = 0U; k < K; ++k) {
					const oa::U32 aIndex = aLayout.offset
						+ batch * aLayout.batchStride
						+ row * aLayout.rowStride + k * aLayout.colStride;
					const oa::U32 bIndex = bLayout.offset
						+ batch * bLayout.batchStride
						+ col * bLayout.rowStride + k * bLayout.colStride;
					expected += aData[aIndex] * bData[bIndex];
				}
				const oa::U32 cIndex = cLayout.offset
					+ batch * cLayout.batchStride
					+ row * cLayout.rowStride + col * cLayout.colStride;
				EXPECT_NEAR(got[cIndex], expected, 2.0e-4F)
					<< "batch=" << batch << " row=" << row << " col=" << col;
			}
		}
	}
}
