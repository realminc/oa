#include <oa/runtime/gemm/router.h>
#include <oa/runtime/matmulTypes.h>
#include <oa/core/envFlag.h>
#include <oa/core/log.h>
#include <oa/core/validation.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/device.h>
#include <oa/runtime/pipeline.h>
#include <oa/runtime/spirv.h>
#include <oa/runtime/gemmTypes.h>
#include "routeCache.h"
#include "engineRouteCacheAccess.h"
#include "engineGemmAccess.h"
#include "../engine/engineAccess.h"
#include "../engine/deviceAccess.h"

#include <atomic>
#include <cstring>

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

namespace {

// tile constants now live in oa::matmulRegistry::kVariants (see registry.cpp).
// The CoopVec workgroup-N denominator is read from the registry too.

inline oa::U32 divCeil(oa::U32 inA, oa::U32 inB) { return (inA + inB - 1U) / inB; }

inline void hashMix(oa::U64& inOutHash, oa::U64 inValue) {
	inOutHash ^= inValue;
	inOutHash *= 0x100000001b3ULL;
}

oa::U64 hashString(const char* inText) {
	oa::U64 h = 0xcbf29ce484222325ULL;
	for (const char* p = inText; p != nullptr and *p != '\0'; ++p) {
		hashMix(h, static_cast<oa::U8>(*p));
	}
	return h;
}

oa::U64 problemContractHash(const oa::MatmulProblem& inProblem) {
	oa::U64 h = 0xcbf29ce484222325ULL;
	hashMix(h, inProblem.m); hashMix(h, inProblem.n); hashMix(h, inProblem.k);
	hashMix(h, inProblem.batchCount);
	auto hashLayout = [&](const oa::MatmulLayout& layout) {
		hashMix(h, layout.offset); hashMix(h, layout.rowStride);
		hashMix(h, layout.colStride); hashMix(h, layout.batchStride);
	};
	hashLayout(inProblem.a); hashLayout(inProblem.b); hashLayout(inProblem.c);
	hashMix(h, static_cast<oa::U8>(inProblem.aMaster));
	hashMix(h, static_cast<oa::U8>(inProblem.bMaster));
	hashMix(h, static_cast<oa::U8>(inProblem.requestedOutput));
	hashMix(h, inProblem.aContiguous); hashMix(h, inProblem.bContiguous);
	hashMix(h, inProblem.bTransposed);
	hashMix(h, static_cast<oa::U8>(inProblem.epilogue));
	hashMix(h, inProblem.requiresPreActivation); hashMix(h, inProblem.training);
	hashMix(h, static_cast<oa::U8>(inProblem.precisionHint));
	return h;
}

oa::U64 deviceContractHash(const oa::Engine& inRt) {
	const auto& hw = oa::EngineDeviceAccess::get(inRt).info.hardware;
	const auto& sw = oa::EngineDeviceAccess::get(inRt).info.software;
	oa::U64 h = 0xcbf29ce484222325ULL;
	hashMix(h, hw.vendorId); hashMix(h, hw.deviceId); hashMix(h, sw.driverId);
	hashMix(h, hashString(sw.driverVersion.cStr()));
	hashMix(h, oa::EngineGemmAccess::capsMask(inRt));
	hashMix(h, hw.maxComputeWorkGroupInvocations);
	hashMix(h, hw.maxComputeWorkGroupSize);
	hashMix(h, hw.maxComputeSharedMemoryBytes);
	return h;
}

oa::GemmPrecision toGemmPrecision(oa::StoragePrecision inPrecision) {
	return inPrecision == oa::StoragePrecision::Bf16
		? oa::GemmPrecision::Bf16
		: oa::GemmPrecision::Fp32;
}

// Build one cache key from the complete operation contract. layout and
// dual-output fields are part of the key because replaying a winner across
// either boundary can select a shader with a different buffer contract.
static oa::RouteCacheKey buildRouteCacheKeyLocal(
	const oa::Engine& inRt,
	const oa::MatmulProblem& inProblem)
{
	oa::RouteCacheKey key{};
	key.vendorId = oa::EngineDeviceAccess::get(inRt).info.hardware.vendorId;
	key.deviceId = oa::EngineDeviceAccess::get(inRt).info.hardware.deviceId;
	key.driverId = oa::EngineDeviceAccess::get(inRt).info.software.driverId;
	key.driverVersionHash = hashString(oa::EngineDeviceAccess::get(inRt).info.software.driverVersion.cStr());
	key.shaderBuildId = oa::matmulRegistry::shaderBuildId();
	key.m = inProblem.m;
	key.n = inProblem.n;
	key.k = inProblem.k;
	key.batchCount = inProblem.batchCount;
	key.aOffset = inProblem.a.offset; key.aRowStride = inProblem.a.rowStride;
	key.aColStride = inProblem.a.colStride; key.aBatchStride = inProblem.a.batchStride;
	key.bOffset = inProblem.b.offset; key.bRowStride = inProblem.b.rowStride;
	key.bColStride = inProblem.b.colStride; key.bBatchStride = inProblem.b.batchStride;
	key.cOffset = inProblem.c.offset; key.cRowStride = inProblem.c.rowStride;
	key.cColStride = inProblem.c.colStride; key.cBatchStride = inProblem.c.batchStride;
	key.aPrecision = toGemmPrecision(inProblem.aMaster);
	key.bPrecision = toGemmPrecision(inProblem.bMaster);
	key.outputPrecision = toGemmPrecision(inProblem.requestedOutput);
	key.requestedPrecision = inProblem.precisionHint;
	key.epilogue = inProblem.epilogue;
	key.aContiguous = inProblem.aContiguous;
	key.bContiguous = inProblem.bContiguous;
	key.bTransposed = inProblem.bTransposed;
	key.requiresPreActivation = inProblem.requiresPreActivation;
	key.training = inProblem.training;
	return key;
}

static oa::GemmPrecision resolvePrecision(const oa::MatmulProblem& inProblem) {
	if (oa::EnvFlag::isSet("OA_GEMM_FORCE_FP32")) {
		return oa::GemmPrecision::Fp32;
	}
	if (inProblem.precisionHint != oa::GemmPrecision::Auto) {
		return inProblem.precisionHint;
	}
	return (inProblem.aMaster == oa::StoragePrecision::Bf16
		or inProblem.bMaster == oa::StoragePrecision::Bf16)
		? oa::GemmPrecision::Bf16
		: oa::GemmPrecision::Auto;
}

static bool variantLegalResolved(
	const oa::Engine& inRt,
	const oa::MatmulVariant& inVariant,
	const oa::MatmulProblem& inProblem,
	oa::GemmPrecision inPrecision)
{
	if (inProblem.m == 0U or inProblem.n == 0U or inProblem.k == 0U
		or inProblem.batchCount == 0U) {
		return false;
	}
	if (inVariant.epilogue != inProblem.epilogue
		or inVariant.dualOutput != inProblem.requiresPreActivation
		or inVariant.outputPrecision != inProblem.requestedOutput) {
		return false;
	}
	const bool canonicalA = inProblem.a.offset == 0U
		and inProblem.a.rowStride == inProblem.k and inProblem.a.colStride == 1U
		and (inProblem.batchCount == 1U or inProblem.a.batchStride == inProblem.m * inProblem.k);
	const bool canonicalB = inProblem.b.offset == 0U
		and inProblem.b.rowStride == inProblem.k and inProblem.b.colStride == 1U
		and (inProblem.batchCount == 1U or inProblem.b.batchStride == inProblem.n * inProblem.k);
	const bool canonicalC = inProblem.c.offset == 0U
		and inProblem.c.rowStride == inProblem.n and inProblem.c.colStride == 1U
		and (inProblem.batchCount == 1U or inProblem.c.batchStride == inProblem.m * inProblem.n);
	const bool canonical = inProblem.aContiguous and inProblem.bContiguous
		and canonicalA and canonicalB and canonicalC;
	if (not canonical and not inVariant.supportsArbitraryLayout) {
		return false;
	}
	if (inProblem.batchCount > 1U and not inVariant.supportsBatch) return false;
	if (inVariant.maxM != 0U and inProblem.m > inVariant.maxM) return false;
	if (inVariant.requiresTransposedB and not inProblem.bTransposed) {
		return false;
	}
	if (inPrecision == oa::GemmPrecision::Fp32
		and (inVariant.aPrecision != oa::StoragePrecision::Fp32
			or inVariant.bPrecision != oa::StoragePrecision::Fp32)) {
		return false;
	}
	if (inPrecision == oa::GemmPrecision::Bf16
		and oa::GemmRouter::precisionAvailable(inRt, oa::GemmPrecision::Bf16)
		and (inVariant.aPrecision != oa::StoragePrecision::Bf16
			or inVariant.bPrecision != oa::StoragePrecision::Bf16)) {
		return false;
	}
	if (not oa::matmulRegistry::capsSatisfy(
		oa::EngineGemmAccess::capsMask(inRt), inVariant.requiredCapsMask))
	{
		return false;
	}
	const auto& hw = oa::EngineDeviceAccess::get(inRt).info.hardware;
	if ((hw.maxComputeWorkGroupInvocations != 0U
			and inVariant.workgroupInvocations > hw.maxComputeWorkGroupInvocations)
		or (hw.maxComputeWorkGroupSize != 0U
			and inVariant.workgroupInvocations > hw.maxComputeWorkGroupSize)
		or (hw.maxComputeSharedMemoryBytes != 0U
			and inVariant.sharedMemoryBytes > hw.maxComputeSharedMemoryBytes)) {
		return false;
	}
	if (inVariant.requiresAligned
		and ((inProblem.m % inVariant.tileM) != 0U
			or (inProblem.n % inVariant.tileN) != 0U
			or (inProblem.k % inVariant.tileK) != 0U)) {
		return false;
	}
	switch (inVariant.kernel) {
		case oa::GemmKernel::TiledFp32:
		case oa::GemmKernel::Naive:
		case oa::GemmKernel::SmallMFp32:
			return inProblem.batchCount == 1U;
		case oa::GemmKernel::StridedFp32:
		case oa::GemmKernel::StridedTiledFp32:
			return inPrecision != oa::GemmPrecision::Bf16
				and inProblem.epilogue == oa::GemmEpilogue::None;
		case oa::GemmKernel::GemmCmSgBf16:
			return inPrecision != oa::GemmPrecision::Fp32
				and oa::GemmRouter::isGemmCmSgBf16Suitable(
					inRt, inProblem.m, inProblem.n, inProblem.k);
		case oa::GemmKernel::GemmCmWgBf16:
			return inPrecision != oa::GemmPrecision::Fp32
				and oa::GemmRouter::isGemmCmWgBf16Suitable(
					inRt, inProblem.m, inProblem.n, inProblem.k);
		case oa::GemmKernel::CoopVec: {
			constexpr oa::U32 kNvidia = 0x10DEU;
			const bool trustedVendor = oa::EngineDeviceAccess::get(inRt).info.hardware.vendorId == kNvidia
				or oa::EnvFlag::isSet("OA_FORCE_COOPVEC");
			return inPrecision != oa::GemmPrecision::Fp32
				and inProblem.epilogue == oa::GemmEpilogue::None
				and inProblem.m == 1U and trustedVendor
				and oa::matmulRegistry::capsSatisfy(
					oa::EngineGemmAccess::capsMask(inRt), oa::kCapCoopVec);
		}
		default:
			return false;
	}
}

oa::GemmRouteResult resultForVariant(
	const oa::MatmulVariant& inVariant, const oa::MatmulProblem& inProblem) {
	oa::U32 gx = 1U;
	oa::U32 gy = 1U;
	oa::U32 gz = 1U;
	oa::GemmPrecision prec = oa::GemmPrecision::Fp32;
	switch (inVariant.aPrecision) {
		case oa::StoragePrecision::Bf16: prec = oa::GemmPrecision::Bf16; break;
		case oa::StoragePrecision::Fp32: prec = oa::GemmPrecision::Fp32; break;
	}
	switch (inVariant.path) {
		case oa::GemmPath::Standard:
			if (inVariant.kernel == oa::GemmKernel::Naive
				or inVariant.kernel == oa::GemmKernel::StridedFp32) {
				gx = divCeil(inProblem.m * inProblem.n, 256U);
				gz = inProblem.batchCount;
			} else {
				gx = divCeil(inProblem.m, inVariant.tileM);
				gy = divCeil(inProblem.n, inVariant.tileN);
				if (inVariant.kernel == oa::GemmKernel::StridedTiledFp32) {
					gz = inProblem.batchCount;
				}
			}
			break;
		case oa::GemmPath::CoopVec:
			gx = divCeil(inProblem.n, inVariant.tileN);
			gy = 1U;
			break;
	}
	return {.variant = inVariant.id, .kernelName = inVariant.kernelName,
		         .kernel = inVariant.kernel, .path = inVariant.path,
		         .actualPrec = prec, .gx = gx, .gy = gy, .gz = gz};
}

oa::MatmulPlan planForVariant(
	const oa::Engine& inRt,
	const oa::MatmulVariant& inVariant,
	const oa::MatmulProblem& inProblem)
{
	const auto route = resultForVariant(inVariant, inProblem);
	return {
		.variant = route.variant,
		.kernelName = route.kernelName,
		.kernel = route.kernel,
		.path = route.path,
		.actualPrecision = route.actualPrec,
		.grid = {.x = route.gx, .y = route.gy, .z = route.gz},
		.workspaceBytes = 0,
		.problemContractHash = problemContractHash(inProblem),
		.deviceContractHash = deviceContractHash(inRt),
		.registryBuildId = oa::matmulRegistry::buildId(),
		.shaderContentHash = oa::matmulRegistry::shaderContentHash(route.variant),
	};
}

oa::GemmRouteResult routeForPlan(const oa::MatmulPlan& inPlan) {
	if (not inPlan) {
		return {};
	}
	return {
		.variant = inPlan.variant,
		.kernelName = inPlan.kernelName,
		.kernel = inPlan.kernel,
		.path = inPlan.path,
		.actualPrec = inPlan.actualPrecision,
		.gx = inPlan.grid.x,
		.gy = inPlan.grid.y,
		.gz = inPlan.grid.z,
	};
}

// ── Emit debug log + counters ─────────────────────────────────────────────────
// NOTE: Per-call OaLogDebug was removed from the hot path — it caused a 2.6x
// wall-time slowdown in Debug builds. counters (compiled out in release) stay
// in. The per-call INFO log below is gated on the OA_LOG_GEMM_ROUTER env knob:
// the env lookup is cached in a thread-safe atomic the first time it's read,
// so hot-path cost is one acquire-load of a bool.

bool gemmRouterLogEnabled() {
	static std::atomic<int> sCached{-1};  // -1=unread, 0=off, 1=on
	int c = sCached.load(std::memory_order_acquire);
	if (c < 0) {
		c = oa::EnvFlag::isSet("OA_LOG_GEMM_ROUTER") ? 1 : 0;
		sCached.store(c, std::memory_order_release);
	}
	return c == 1;
}

const char* pathName(oa::GemmPath inPath) {
	switch (inPath) {
		case oa::GemmPath::Standard: return "Standard";
		case oa::GemmPath::CoopVec:  return "CoopVec";
	}
	return "?";
}

const char* precName(oa::GemmPrecision inPrec) {
	switch (inPrec) {
		case oa::GemmPrecision::Bf16: return "BF16";
		case oa::GemmPrecision::Fp32: return "FP32";
		case oa::GemmPrecision::Auto: return "Auto";
	}
	return "?";
}

void logAndCount(
	const oa::GemmRouteResult& inR,
	oa::KernelSelectionKind inSelection,
	oa::GemmPrecision inRequestedPrecision,
	oa::U32 inM,
	oa::U32 inN,
	oa::U32 inK)
{
	// Only increment counters (compiled out in release)
	OA_DEBUG_COUNTER_INC_NAMED(inR.kernelName);

	// Opt-in per-call decision log (OA_LOG_GEMM_ROUTER=1).
	// This line is a stable machine-readable evidence contract consumed by
	// Tools/Diagnostics/oaevidence.py. Keep it one record per plan.
	if (gemmRouterLogEnabled()) {
		const char* fallback = "none";
		switch (inSelection) {
			case oa::KernelSelectionKind::PrecisionFallback: fallback = "precision"; break;
			case oa::KernelSelectionKind::LayoutFallback: fallback = "layout"; break;
			case oa::KernelSelectionKind::NaiveFallback: fallback = "naive"; break;
			case oa::KernelSelectionKind::Unspecified:
			case oa::KernelSelectionKind::Direct: break;
		}
		OaLogInfo(oa::LogComponent::Compute,
			"GemmRouter: M=%u N=%u K=%u requested=%s actual=%s kernel=%s "
			"path=%s fallback=%s grid=%u,%u,%u",
			inM, inN, inK, precName(inRequestedPrecision), precName(inR.actualPrec),
			inR.kernelName, pathName(inR.path), fallback, inR.gx, inR.gy, inR.gz);
	}

	// Only warn on Naive fallback for large gEMMs (performance issue)
	OA_WARN_PERF(
		inR.kernel == oa::GemmKernel::Naive and inM * inN * inK > 1024U,
		"GemmRouter: Naive path for M=%u N=%u K=%u — no tensor cores available",
		inM, inN, inK);
}

} // namespace

oa::KernelSelectionKind oa::GemmRouter::classifySelection(
	const oa::MatmulPlan& inPlan,
	const oa::MatmulProblem& inProblem)
{
	const oa::GemmPrecision requested = resolvePrecision(inProblem);
	if (requested == oa::GemmPrecision::Bf16
		and inPlan.actualPrecision != oa::GemmPrecision::Bf16)
	{
		return oa::KernelSelectionKind::PrecisionFallback;
	}
	if (inPlan.kernel == oa::GemmKernel::StridedFp32
		or inPlan.kernel == oa::GemmKernel::StridedTiledFp32)
	{
		return oa::KernelSelectionKind::LayoutFallback;
	}
	if (inPlan.kernel == oa::GemmKernel::Naive) {
		return oa::KernelSelectionKind::NaiveFallback;
	}
	return oa::KernelSelectionKind::Direct;
}

oa::RouteCacheKey oa::GemmRouter::cacheKey(
	const oa::Engine& inRt,
	const oa::MatmulProblem& inProblem) {
	return buildRouteCacheKeyLocal(inRt, inProblem);
}

bool oa::GemmRouter::isVariantLegal(
	const oa::Engine& inRt,
	const oa::MatmulVariant& inVariant,
	const oa::MatmulProblem& inProblem) {
	return variantLegalResolved(inRt, inVariant, inProblem, resolvePrecision(inProblem));
}

// ─────────────────────────────────────────────────────────────────────────────
// oa::GemmRouter::Select
// ─────────────────────────────────────────────────────────────────────────────

oa::GemmRouteResult oa::GemmRouter::select(
	const oa::Engine& inRt,
	oa::U32                    inM,
	oa::U32                    inN,
	oa::U32                    inK,
	oa::GemmPrecision          inPrec)
{
	auto problem = problemForRaw(
		inM, inN, inK,
		oa::StoragePrecision::Fp32, oa::StoragePrecision::Fp32, true);
	problem.training = false;
	problem.precisionHint = inPrec;
	return select(inRt, problem);
}

// ─────────────────────────────────────────────────────────────────────────────
// oa::EngineAccess::GemmCapsMask — lazy-init cache for ComputeCapsMask.
// The mask depends only on software/hardware info populated at device init,
// so we compute it once and stash it on the engine. Two threads racing the
// first read will both compute the same value and CAS the result; the
// std::memory_order_relaxed loads/stores are fine because the cap mask is
// idempotent — there is no payload the reader depends on through the atomic.
// ─────────────────────────────────────────────────────────────────────────────

oa::U64 oa::EngineAccess::gemmCapsMask(const oa::Engine& inEngine) {
	auto& impl = get(inEngine);
	oa::U64 cached = impl.gemmCapsMask_.load(std::memory_order_relaxed);
	if (cached != 0U) {
		return cached;
	}
	const oa::U64 computed = oa::matmulRegistry::computeCapsMask(inEngine);
	impl.gemmCapsMask_.store(computed, std::memory_order_relaxed);
	return computed;
}

// ─────────────────────────────────────────────────────────────────────────────
// plan(problem) — the only planner. Legacy Select callers adapt the immutable
// plan back to the old route result while execution paths migrate.
// ─────────────────────────────────────────────────────────────────────────────

oa::MatmulPlan oa::GemmRouter::plan(
	const oa::Engine& inRt,
	const oa::MatmulProblem& inProblem,
	oa::MatmulPreference inPreference)
{
	oa::MatmulProblem problem = inProblem;
	const bool forceFp32 = oa::EnvFlag::isSet("OA_GEMM_FORCE_FP32");
	const bool disableMeasuredCache =
		oa::EnvFlag::isSet("OA_DISABLE_GEMM_ROUTE_CACHE");
	const oa::GemmPrecision precision = resolvePrecision(problem);

	auto finish = [&](const oa::MatmulVariant& variant) {
		auto plan = planForVariant(inRt, variant, problem);
		auto route = routeForPlan(plan);
		logAndCount(route, classifySelection(plan, problem), precision,
			problem.m, problem.n, problem.k);
		return plan;
	};
	auto findLegal = [&](oa::U64 id) -> const oa::MatmulVariant* {
		const auto* variant = oa::matmulRegistry::find(id);
		return variant != nullptr and variantLegalResolved(inRt, *variant, problem, precision)
			? variant
			: nullptr;
	};

	// Explicit diagnostic/tuning isolation. A required variant is a property
	// of this one plan request, never mutable router state shared by engines or
	// concurrent recordings. An illegal requirement fails closed.
	if (inPreference.requiredVariant != oa::invalidMatmulVariantId) {
		if (const auto* required = findLegal(inPreference.requiredVariant)) {
			return finish(*required);
		}
		return {};
	}

	// The route cache stores a stable variant identity and is always revalidated
	// against the live registry, full problem contract, device caps and shape.
	const auto* routeCache = oa::GemmRouteCacheAccess::get(inRt);
	if (inPreference.useMeasuredCache and routeCache != nullptr
		and not forceFp32 and not disableMeasuredCache) {
		oa::U64 cached = oa::invalidMatmulVariantId;
		if (routeCache->query(cacheKey(inRt, problem), cached)) {
			if (const auto* winner = findLegal(cached)) {
				return finish(*winner);
			}
		}
	}

	// M=1 decode is the only distinct path. The legality predicate keeps this
	// NVIDIA-specific extension vendor-gated unless explicitly overridden.
	if (problem.epilogue == oa::GemmEpilogue::None and problem.m == 1U
		and precision != oa::GemmPrecision::Fp32) {
		if (const auto* coopVec = findLegal(
			oa::matmulVariantIdFromName("GemmCoopVec"))) {
			return finish(*coopVec);
		}
	}

	// Tensor-core families are ordered by the current measured policy. The
	// registry supplies exact epilogue rows, so the same code covers raw, Bias,
	// Bias+ReLU, Bias+GELU and dual-output SiLU without name construction.
	if (precision != oa::GemmPrecision::Fp32) {
		for (const oa::GemmKernel family : {
			oa::GemmKernel::GemmCmWgBf16,
			oa::GemmKernel::GemmCmSgBf16}) {
			for (const auto& variant : oa::matmulRegistry::all()) {
				if (variant.kernel == family
					and variantLegalResolved(inRt, variant, problem, precision)) {
					return finish(variant);
				}
			}
		}
	}

	// Non-canonical views and strided batches have two generated implementations
	// under one exact descriptor. Medium/large work uses the tiled shared-memory
	// family; tiny or skinny work retains the scalar route to avoid a mostly
	// empty 64x64 workgroup. contiguous kernels never see explicit offsets/strides.
	const bool canonical = problem.batchCount == 1U
		and problem.aContiguous and problem.bContiguous
		and problem.a.offset == 0U and problem.b.offset == 0U
		and problem.c.offset == 0U
		and problem.a.rowStride == problem.k and problem.a.colStride == 1U
		and problem.b.rowStride == problem.k and problem.b.colStride == 1U
		and problem.c.rowStride == problem.n and problem.c.colStride == 1U;
	if (not canonical) {
		const oa::U64 work = static_cast<oa::U64>(problem.m)
			* problem.n * problem.k;
		const bool reuseTile = problem.m >= 16U and problem.n >= 16U
			and work >= 32768U;
		const oa::GemmKernel first = reuseTile
			? oa::GemmKernel::StridedTiledFp32
			: oa::GemmKernel::StridedFp32;
		const oa::GemmKernel second = first == oa::GemmKernel::StridedTiledFp32
			? oa::GemmKernel::StridedFp32
			: oa::GemmKernel::StridedTiledFp32;
		for (const oa::GemmKernel family : {first, second}) {
			for (const auto& variant : oa::matmulRegistry::all()) {
				if (variant.kernel == family and variantLegalResolved(
					inRt, variant, problem, oa::GemmPrecision::Fp32)) {
					return finish(variant);
				}
			}
		}
	}

	// Portable FP32 fallback. Tiny raw GEMMs retain the scalar path; every
	// fused contract uses a generated epilogue variant when one exists. The
	// small-M registry rows carry their exact maximum-M legality bound.
	const bool preferNaive = problem.epilogue == oa::GemmEpilogue::None
		and problem.m * problem.n < 64U;
	if (not preferNaive) {
		for (const auto& variant : oa::matmulRegistry::all()) {
			if (variant.kernel == oa::GemmKernel::SmallMFp32
				and variantLegalResolved(
					inRt, variant, problem, oa::GemmPrecision::Fp32)) {
				return finish(variant);
			}
		}
	}
	for (const oa::GemmKernel family : preferNaive
		? std::initializer_list<oa::GemmKernel>{oa::GemmKernel::Naive, oa::GemmKernel::TiledFp32}
		: std::initializer_list<oa::GemmKernel>{oa::GemmKernel::TiledFp32, oa::GemmKernel::Naive}) {
		for (const auto& variant : oa::matmulRegistry::all()) {
			if (variant.kernel == family
				and variant.aPrecision == oa::StoragePrecision::Fp32
				and variantLegalResolved(inRt, variant, problem, oa::GemmPrecision::Fp32)) {
				return finish(variant);
			}
		}
	}

	OaLogError(oa::LogComponent::Compute,
		"GemmRouter: no legal variant for M=%u N=%u K=%u epilogue=%u",
		problem.m, problem.n, problem.k, static_cast<oa::U32>(problem.epilogue));
	return {};
}

bool oa::GemmRouter::validatePlan(
	const oa::Engine& inRt,
	const oa::MatmulPlan& inPlan,
	const oa::MatmulProblem& inProblem)
{
	if (not inPlan
		or inPlan.registryBuildId != oa::matmulRegistry::buildId()
		or inPlan.problemContractHash != problemContractHash(inProblem)
		or inPlan.deviceContractHash != deviceContractHash(inRt)) {
		return false;
	}
	const auto* variant = oa::matmulRegistry::find(inPlan.variant);
	if (variant == nullptr
		or std::strcmp(variant->kernelName, inPlan.kernelName) != 0
		or variant->kernel != inPlan.kernel
		or variant->path != inPlan.path
		or inPlan.shaderContentHash == 0U
		or inPlan.shaderContentHash != oa::matmulRegistry::shaderContentHash(variant->id)
		or not isVariantLegal(inRt, *variant, inProblem)) {
		return false;
	}
	const auto expected = resultForVariant(*variant, inProblem);
	return inPlan.actualPrecision == expected.actualPrec
		and inPlan.grid.x == expected.gx
		and inPlan.grid.y == expected.gy
		and inPlan.grid.z == expected.gz
		and inPlan.workspaceBytes == 0U;
}

oa::GemmRouteResult oa::GemmRouter::select(
	const oa::Engine& inRt,
	const oa::MatmulProblem& inProblem)
{
	return routeForPlan(plan(inRt, inProblem));
}

oa::MatmulProblem oa::GemmRouter::problemForRaw(
	oa::U32 inM, oa::U32 inN, oa::U32 inK,
	oa::StoragePrecision inAMaster,
	oa::StoragePrecision inBMaster,
	bool  inBTransposed)
{
	oa::MatmulProblem p;
	p.m                       = inM;
	p.n                       = inN;
	p.k                       = inK;
	p.batchCount              = 1U;
	p.a                       = {.offset = 0U, .rowStride = inK,
		.colStride = 1U, .batchStride = inM * inK};
	p.b                       = {.offset = 0U, .rowStride = inK,
		.colStride = 1U, .batchStride = inN * inK};
	p.c                       = {.offset = 0U, .rowStride = inN,
		.colStride = 1U, .batchStride = inM * inN};
	p.aMaster                 = inAMaster;
	p.bMaster                 = inBMaster;
	p.requestedOutput         = oa::StoragePrecision::Fp32;
	p.aContiguous             = true;
	p.bContiguous             = true;
	p.bTransposed             = inBTransposed;
	p.epilogue                = oa::GemmEpilogue::None;
	p.requiresPreActivation   = false;
	p.training                = true;
	p.precisionHint           = oa::GemmPrecision::Auto;
	return p;
}

// ─────────────────────────────────────────────────────────────────────────────
// PrecisionAvailable
// ─────────────────────────────────────────────────────────────────────────────

bool oa::GemmRouter::precisionAvailable(
	const oa::Engine& inRt,
	oa::GemmPrecision          inPrec)
{
	const auto& sw = oa::EngineDeviceAccess::get(inRt).info.software;
	switch (inPrec) {
		case oa::GemmPrecision::Fp32: return true;
		case oa::GemmPrecision::Bf16:
			return sw.shaderBfloat16CooperativeMatrixEnabled and sw.shaderBfloat16TypeEnabled;
		case oa::GemmPrecision::Auto:
			return true;
		default:
			return false;
	}
}

bool oa::GemmRouter::isGemmCmSgBf16Suitable(
	const oa::Engine& inRt,
	oa::U32                    inM,
	oa::U32                    inN,
	oa::U32                    inK) {
	// Portable CoopMat1 triplet — GemmCmSgBf16 uses only VK_KHR_cooperative_matrix
	// at the universal 16×16×16 bf16-in/fp32-acc shape, so it qualifies on
	// AMD RDNA3.5/Strix as well as NVIDIA. No NV CoopMat2 dependency.
	const oa::U64 caps = oa::EngineGemmAccess::capsMask(inRt);
	const oa::U64 need = oa::kCapCoopMat1Khr | oa::kCapCoopMat1Bf16Input | oa::kCapCoopMat1Fp32Acc;
	if (not oa::matmulRegistry::capsSatisfy(caps, need)) {
		return false;
	}
	// GemmCmSgBf16 uses 16×16 fragments and a direct-to-global store (raw) or
	// SMEM-staged element-wise copy (fused). M and N must be multiples of 16.
	if ((inM % 16U) != 0U or (inN % 16U) != 0U) {
		return false;
	}
	const oa::U32 minDim = 64U;
	return inM >= minDim and inN >= minDim and inK >= minDim;
}

bool oa::GemmRouter::isGemmCmWgBf16Suitable(
	const oa::Engine& inRt,
	oa::U32                    inM,
	oa::U32                    inN,
	oa::U32                    inK
) {
	// workgroup-scope 32x32x16 KHR CoopMat. NVIDIA-favored; requires the device
	// to report a workgroup-scope BF16 input shape.
	const oa::U64 caps = oa::EngineGemmAccess::capsMask(inRt);
	const oa::U64 need = oa::kCapCoopMat1Khr | oa::kCapCoopMat1WorkgroupBf16 | oa::kCapCoopMat1Fp32Acc;
	if (not oa::matmulRegistry::capsSatisfy(caps, need)) {
		return false;
	}
	if (oa::EnvFlag::isSet("OA_DISABLE_COOPMAT2")) {
		return false;
	}
	// 32x32 output blocks require M,N multiples of 32 for the fast path.
	if ((inM % 32U) != 0U or (inN % 32U) != 0U) {
		return false;
	}
	const oa::U32 minDim = 64U;
	return inM >= minDim and inN >= minDim and inK >= minDim;
}
