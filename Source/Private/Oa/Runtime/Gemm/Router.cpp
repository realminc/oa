#include <Oa/Runtime/Gemm/Router.h>
#include <Oa/Runtime/MatmulTypes.h>
#include <Oa/Core/EnvFlag.h>
#include <Oa/Core/Log.h>
#include <Oa/Core/Validation.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Device.h>
#include <Oa/Runtime/Pipeline.h>
#include <Oa/Runtime/Spirv.h>
#include <Oa/Runtime/GemmTypes.h>
#include <Oa/Runtime/GemmRouteCache.h>

#include <atomic>
#include <cstring>
#include <mutex>
#include <unordered_map>

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

namespace {

// Tile constants now live in OaMatmulRegistry::kVariants (see Registry.cpp).
// The CoopVec workgroup-N denominator is read from the registry too.

inline OaU32 DivCeil(OaU32 InA, OaU32 InB) { return (InA + InB - 1U) / InB; }

inline void HashMix(OaU64& InOutHash, OaU64 InValue) {
	InOutHash ^= InValue;
	InOutHash *= 0x100000001b3ULL;
}

OaU64 HashString(const char* InText) {
	OaU64 h = 0xcbf29ce484222325ULL;
	for (const char* p = InText; p != nullptr and *p != '\0'; ++p) {
		HashMix(h, static_cast<OaU8>(*p));
	}
	return h;
}

OaU64 ProblemContractHash(const OaMatmulProblem& InProblem) {
	OaU64 h = 0xcbf29ce484222325ULL;
	HashMix(h, InProblem.M); HashMix(h, InProblem.N); HashMix(h, InProblem.K);
	HashMix(h, InProblem.BatchCount);
	auto hashLayout = [&](const OaMatmulLayout& layout) {
		HashMix(h, layout.Offset); HashMix(h, layout.RowStride);
		HashMix(h, layout.ColStride); HashMix(h, layout.BatchStride);
	};
	hashLayout(InProblem.A); hashLayout(InProblem.B); hashLayout(InProblem.C);
	HashMix(h, static_cast<OaU8>(InProblem.AMaster));
	HashMix(h, static_cast<OaU8>(InProblem.BMaster));
	HashMix(h, static_cast<OaU8>(InProblem.RequestedOutput));
	HashMix(h, InProblem.AContiguous); HashMix(h, InProblem.BContiguous);
	HashMix(h, InProblem.BTransposed);
	HashMix(h, static_cast<OaU8>(InProblem.Epilogue));
	HashMix(h, InProblem.RequiresPreActivation); HashMix(h, InProblem.Training);
	HashMix(h, static_cast<OaU8>(InProblem.PrecisionHint));
	return h;
}

OaU64 DeviceContractHash(const OaEngine& InRt) {
	const auto& hw = InRt.Device.Info.Hardware;
	const auto& sw = InRt.Device.Info.Software;
	OaU64 h = 0xcbf29ce484222325ULL;
	HashMix(h, hw.VendorId); HashMix(h, hw.DeviceId); HashMix(h, sw.DriverId);
	HashMix(h, HashString(sw.DriverVersion.c_str()));
	HashMix(h, InRt.GemmCapsMask());
	HashMix(h, hw.MaxComputeWorkGroupInvocations);
	HashMix(h, hw.MaxComputeWorkGroupSize);
	HashMix(h, hw.MaxComputeSharedMemoryBytes);
	return h;
}

OaGemmPrecision ToGemmPrecision(OaStoragePrecision InPrecision) {
	return InPrecision == OaStoragePrecision::Bf16
		? OaGemmPrecision::Bf16
		: OaGemmPrecision::Fp32;
}

// Build one cache key from the complete operation contract. Layout and
// dual-output fields are part of the key because replaying a winner across
// either boundary can select a shader with a different buffer contract.
static OaRouteCacheKey BuildRouteCacheKeyLocal(
	const OaEngine& InRt,
	const OaMatmulProblem& InProblem)
{
	OaRouteCacheKey key{};
	key.VendorId = InRt.Device.Info.Hardware.VendorId;
	key.DeviceId = InRt.Device.Info.Hardware.DeviceId;
	key.DriverId = InRt.Device.Info.Software.DriverId;
	key.DriverVersionHash = HashString(InRt.Device.Info.Software.DriverVersion.c_str());
	key.ShaderBuildId = OaMatmulRegistry::ShaderBuildId();
	key.M = InProblem.M;
	key.N = InProblem.N;
	key.K = InProblem.K;
	key.BatchCount = InProblem.BatchCount;
	key.AOffset = InProblem.A.Offset; key.ARowStride = InProblem.A.RowStride;
	key.AColStride = InProblem.A.ColStride; key.ABatchStride = InProblem.A.BatchStride;
	key.BOffset = InProblem.B.Offset; key.BRowStride = InProblem.B.RowStride;
	key.BColStride = InProblem.B.ColStride; key.BBatchStride = InProblem.B.BatchStride;
	key.COffset = InProblem.C.Offset; key.CRowStride = InProblem.C.RowStride;
	key.CColStride = InProblem.C.ColStride; key.CBatchStride = InProblem.C.BatchStride;
	key.APrecision = ToGemmPrecision(InProblem.AMaster);
	key.BPrecision = ToGemmPrecision(InProblem.BMaster);
	key.OutputPrecision = ToGemmPrecision(InProblem.RequestedOutput);
	key.RequestedPrecision = InProblem.PrecisionHint;
	key.Epilogue = InProblem.Epilogue;
	key.AContiguous = InProblem.AContiguous;
	key.BContiguous = InProblem.BContiguous;
	key.BTransposed = InProblem.BTransposed;
	key.RequiresPreActivation = InProblem.RequiresPreActivation;
	key.Training = InProblem.Training;
	return key;
}

static OaGemmPrecision ResolvePrecision(const OaMatmulProblem& InProblem) {
	if (OaEnvFlag::IsSet("OA_GEMM_FORCE_FP32")) {
		return OaGemmPrecision::Fp32;
	}
	if (InProblem.PrecisionHint != OaGemmPrecision::Auto) {
		return InProblem.PrecisionHint;
	}
	return (InProblem.AMaster == OaStoragePrecision::Bf16
		or InProblem.BMaster == OaStoragePrecision::Bf16)
		? OaGemmPrecision::Bf16
		: OaGemmPrecision::Auto;
}

static bool VariantLegalResolved(
	const OaEngine& InRt,
	const OaMatmulVariant& InVariant,
	const OaMatmulProblem& InProblem,
	OaGemmPrecision InPrecision)
{
	if (InProblem.M == 0U or InProblem.N == 0U or InProblem.K == 0U
		or InProblem.BatchCount == 0U) {
		return false;
	}
	if (InVariant.Epilogue != InProblem.Epilogue
		or InVariant.DualOutput != InProblem.RequiresPreActivation
		or InVariant.OutputPrecision != InProblem.RequestedOutput) {
		return false;
	}
	const bool canonicalA = InProblem.A.Offset == 0U
		and InProblem.A.RowStride == InProblem.K and InProblem.A.ColStride == 1U
		and (InProblem.BatchCount == 1U or InProblem.A.BatchStride == InProblem.M * InProblem.K);
	const bool canonicalB = InProblem.B.Offset == 0U
		and InProblem.B.RowStride == InProblem.K and InProblem.B.ColStride == 1U
		and (InProblem.BatchCount == 1U or InProblem.B.BatchStride == InProblem.N * InProblem.K);
	const bool canonicalC = InProblem.C.Offset == 0U
		and InProblem.C.RowStride == InProblem.N and InProblem.C.ColStride == 1U
		and (InProblem.BatchCount == 1U or InProblem.C.BatchStride == InProblem.M * InProblem.N);
	const bool canonical = InProblem.AContiguous and InProblem.BContiguous
		and canonicalA and canonicalB and canonicalC;
	if (not canonical and not InVariant.SupportsArbitraryLayout) {
		return false;
	}
	if (InProblem.BatchCount > 1U and not InVariant.SupportsBatch) return false;
	if (InVariant.RequiresTransposedB and not InProblem.BTransposed) {
		return false;
	}
	if (InPrecision == OaGemmPrecision::Fp32
		and (InVariant.APrecision != OaStoragePrecision::Fp32
			or InVariant.BPrecision != OaStoragePrecision::Fp32)) {
		return false;
	}
	if (InPrecision == OaGemmPrecision::Bf16
		and OaGemmRouter::PrecisionAvailable(InRt, OaGemmPrecision::Bf16)
		and (InVariant.APrecision != OaStoragePrecision::Bf16
			or InVariant.BPrecision != OaStoragePrecision::Bf16)) {
		return false;
	}
	if (not OaMatmulRegistry::CapsSatisfy(InRt.GemmCapsMask(), InVariant.RequiredCapsMask)) {
		return false;
	}
	const auto& hw = InRt.Device.Info.Hardware;
	if ((hw.MaxComputeWorkGroupInvocations != 0U
			and InVariant.WorkgroupInvocations > hw.MaxComputeWorkGroupInvocations)
		or (hw.MaxComputeWorkGroupSize != 0U
			and InVariant.WorkgroupInvocations > hw.MaxComputeWorkGroupSize)
		or (hw.MaxComputeSharedMemoryBytes != 0U
			and InVariant.SharedMemoryBytes > hw.MaxComputeSharedMemoryBytes)) {
		return false;
	}
	if (InVariant.RequiresAligned
		and ((InProblem.M % InVariant.TileM) != 0U
			or (InProblem.N % InVariant.TileN) != 0U
			or (InProblem.K % InVariant.TileK) != 0U)) {
		return false;
	}
	switch (InVariant.Kernel) {
		case OaGemmKernel::TiledFp32:
		case OaGemmKernel::Naive:
			return InProblem.BatchCount == 1U;
		case OaGemmKernel::StridedFp32:
			return InPrecision != OaGemmPrecision::Bf16
				and InProblem.Epilogue == OaGemmEpilogue::None;
		case OaGemmKernel::GemmCmSgBf16:
			return InPrecision != OaGemmPrecision::Fp32
				and OaGemmRouter::IsGemmCmSgBf16Suitable(
					InRt, InProblem.M, InProblem.N, InProblem.K);
		case OaGemmKernel::GemmCmWgBf16:
			return InPrecision != OaGemmPrecision::Fp32
				and OaGemmRouter::IsGemmCmWgBf16Suitable(
					InRt, InProblem.M, InProblem.N, InProblem.K);
		case OaGemmKernel::CoopVec: {
			constexpr OaU32 kNvidia = 0x10DEU;
			const bool trustedVendor = InRt.Device.Info.Hardware.VendorId == kNvidia
				or OaEnvFlag::IsSet("OA_FORCE_COOPVEC");
			return InPrecision != OaGemmPrecision::Fp32
				and InProblem.Epilogue == OaGemmEpilogue::None
				and InProblem.M == 1U and trustedVendor
				and OaMatmulRegistry::CapsSatisfy(InRt.GemmCapsMask(), kCapCoopVec);
		}
		default:
			return false;
	}
}

// Find the raw-GEMM registry entry for an OaGemmKernel enum (no bias, no
// activation). For the routes that need it, returns the tile dims to compute
// the workgroup grid without hardcoding the constants in this file.
const OaMatmulVariant* FindRawGemmVariant(OaGemmKernel InKernel, OaGemmPath InPath) {
	for (const auto& v : OaMatmulRegistry::All()) {
		if (v.Kernel != InKernel) {
			continue;
		}
		if (v.Path != InPath) {
			continue;
		}
		if (v.Epilogue != OaGemmEpilogue::None) {
			continue;
		}
		return &v;
	}
	return nullptr;
}

OaGemmRouteResult ResultForVariant(
	const OaMatmulVariant& InVariant, const OaMatmulProblem& InProblem) {
	OaU32 gx = 1U;
	OaU32 gy = 1U;
	OaU32 gz = 1U;
	OaGemmPrecision prec = OaGemmPrecision::Fp32;
	switch (InVariant.APrecision) {
		case OaStoragePrecision::Bf16: prec = OaGemmPrecision::Bf16; break;
		case OaStoragePrecision::Fp32: prec = OaGemmPrecision::Fp32; break;
	}
	switch (InVariant.Path) {
		case OaGemmPath::Standard:
			if (InVariant.Kernel == OaGemmKernel::Naive
				or InVariant.Kernel == OaGemmKernel::StridedFp32) {
				gx = DivCeil(InProblem.M * InProblem.N, 256U);
				gz = InProblem.BatchCount;
			} else {
				gx = DivCeil(InProblem.M, InVariant.TileM);
				gy = DivCeil(InProblem.N, InVariant.TileN);
			}
			break;
		case OaGemmPath::CoopVec:
			gx = DivCeil(InProblem.N, InVariant.TileN);
			gy = 1U;
			break;
	}
	return {.Variant = InVariant.Id, .KernelName = InVariant.KernelName,
		         .Kernel = InVariant.Kernel, .Path = InVariant.Path,
		         .ActualPrec = prec, .Gx = gx, .Gy = gy, .Gz = gz};
}

OaMatmulPlan PlanForVariant(
	const OaEngine& InRt,
	const OaMatmulVariant& InVariant,
	const OaMatmulProblem& InProblem)
{
	const auto route = ResultForVariant(InVariant, InProblem);
	return {
		.Variant = route.Variant,
		.KernelName = route.KernelName,
		.Kernel = route.Kernel,
		.Path = route.Path,
		.ActualPrecision = route.ActualPrec,
		.Grid = {.X = route.Gx, .Y = route.Gy, .Z = route.Gz},
		.WorkspaceBytes = 0,
		.ProblemContractHash = ProblemContractHash(InProblem),
		.DeviceContractHash = DeviceContractHash(InRt),
		.RegistryBuildId = OaMatmulRegistry::BuildId(),
		.ShaderContentHash = OaMatmulRegistry::ShaderContentHash(route.Variant),
	};
}

OaGemmRouteResult RouteForPlan(const OaMatmulPlan& InPlan) {
	if (not InPlan) {
		return {};
	}
	return {
		.Variant = InPlan.Variant,
		.KernelName = InPlan.KernelName,
		.Kernel = InPlan.Kernel,
		.Path = InPlan.Path,
		.ActualPrec = InPlan.ActualPrecision,
		.Gx = InPlan.Grid.X,
		.Gy = InPlan.Grid.Y,
		.Gz = InPlan.Grid.Z,
	};
}

// ── ForceKernel override map ──────────────────────────────────────────────────

struct ForceKey {
	OaU32 M, N, K;
	bool operator==(const ForceKey& o) const noexcept {
		return M == o.M and N == o.N and K == o.K;
	}
};
struct ForceKeyHash {
	OaU64 operator()(const ForceKey& k) const noexcept {
		OaU64 h = 0xcbf29ce484222325ULL;
		auto mix = [&](OaU32 v) { h ^= v; h *= 0x100000001b3ULL; };
		mix(k.M); mix(k.N); mix(k.K);
		return h;
	}
};

struct ForcedMap {
	std::mutex                                                    Mtx;
	std::unordered_map<ForceKey, OaMatmulVariantId, ForceKeyHash> Map;
};

ForcedMap& GetForcedMap() {
	static ForcedMap s;
	return s;
}

// ── Emit debug log + counters ─────────────────────────────────────────────────
// NOTE: Per-call OA_LOG_DEBUG was removed from the hot path — it caused a 2.6x
// wall-time slowdown in Debug builds. Counters (compiled out in Release) stay
// in. The per-call INFO log below is gated on the OA_LOG_GEMM_ROUTER env knob:
// the env lookup is cached in a thread-safe atomic the first time it's read,
// so hot-path cost is one acquire-load of a bool.

bool GemmRouterLogEnabled() {
	static std::atomic<int> sCached{-1};  // -1=unread, 0=off, 1=on
	int c = sCached.load(std::memory_order_acquire);
	if (c < 0) {
		c = OaEnvFlag::IsSet("OA_LOG_GEMM_ROUTER") ? 1 : 0;
		sCached.store(c, std::memory_order_release);
	}
	return c == 1;
}

const char* PathName(OaGemmPath InPath) {
	switch (InPath) {
		case OaGemmPath::Standard: return "Standard";
		case OaGemmPath::CoopVec:  return "CoopVec";
	}
	return "?";
}

const char* PrecName(OaGemmPrecision InPrec) {
	switch (InPrec) {
		case OaGemmPrecision::Bf16: return "BF16";
		case OaGemmPrecision::Fp32: return "FP32";
		case OaGemmPrecision::Auto: return "Auto";
	}
	return "?";
}

void LogAndCount(
	const OaGemmRouteResult& InR,
	OaGemmPrecision InRequestedPrecision,
	OaU32 InM,
	OaU32 InN,
	OaU32 InK)
{
	// Only increment counters (compiled out in Release)
	OA_DEBUG_COUNTER_INC_NAMED(InR.KernelName);

	// Opt-in per-call decision log (OA_LOG_GEMM_ROUTER=1).
	// This line is a stable machine-readable evidence contract consumed by
	// Tools/Diagnostics/oaevidence.py. Keep it one record per plan.
	if (GemmRouterLogEnabled()) {
		const char* fallback = "none";
		if (InRequestedPrecision == OaGemmPrecision::Bf16
			and InR.ActualPrec != OaGemmPrecision::Bf16) {
			fallback = "precision";
		} else if (InR.Kernel == OaGemmKernel::StridedFp32) {
			fallback = "layout";
		} else if (InR.Kernel == OaGemmKernel::Naive) {
			fallback = "naive";
		}
		OA_LOG_INFO(OaLogComponent::Core,
			"GemmRouter: M=%u N=%u K=%u requested=%s actual=%s kernel=%s "
			"path=%s fallback=%s grid=%u,%u,%u",
			InM, InN, InK, PrecName(InRequestedPrecision), PrecName(InR.ActualPrec),
			InR.KernelName, PathName(InR.Path), fallback, InR.Gx, InR.Gy, InR.Gz);
	}

	// Only warn on Naive fallback for large GEMMs (performance issue)
	OA_WARN_PERF(
		InR.Kernel == OaGemmKernel::Naive and InM * InN * InK > 1024U,
		"GemmRouter: Naive path for M=%u N=%u K=%u — no tensor cores available",
		InM, InN, InK);
}

} // namespace

OaRouteCacheKey OaGemmRouter::CacheKey(
	const OaEngine& InRt,
	const OaMatmulProblem& InProblem) {
	return BuildRouteCacheKeyLocal(InRt, InProblem);
}

bool OaGemmRouter::IsVariantLegal(
	const OaEngine& InRt,
	const OaMatmulVariant& InVariant,
	const OaMatmulProblem& InProblem) {
	return VariantLegalResolved(InRt, InVariant, InProblem, ResolvePrecision(InProblem));
}

// ─────────────────────────────────────────────────────────────────────────────
// OaGemmRouter::Select
// ─────────────────────────────────────────────────────────────────────────────

OaGemmRouteResult OaGemmRouter::Select(
	const OaEngine& InRt,
	OaU32                    InM,
	OaU32                    InN,
	OaU32                    InK,
	OaGemmPrecision          InPrec)
{
	auto problem = ProblemForRaw(
		InM, InN, InK,
		OaStoragePrecision::Fp32, OaStoragePrecision::Fp32, true);
	problem.Training = false;
	problem.PrecisionHint = InPrec;
	return Select(InRt, problem);
}

// ─────────────────────────────────────────────────────────────────────────────
// OaEngine::GemmCapsMask — lazy-init cache for ComputeCapsMask.
// The mask depends only on Software/Hardware info populated at device init,
// so we compute it once and stash it on the engine. Two threads racing the
// first read will both compute the same value and CAS the result; the
// std::memory_order_relaxed loads/stores are fine because the cap mask is
// idempotent — there is no payload the reader depends on through the atomic.
// ─────────────────────────────────────────────────────────────────────────────

OaU64 OaEngine::GemmCapsMask() const {
	OaU64 cached = GemmCapsMask_.load(std::memory_order_relaxed);
	if (cached != 0U) {
		return cached;
	}
	const OaU64 computed = OaMatmulRegistry::ComputeCapsMask(*this);
	GemmCapsMask_.store(computed, std::memory_order_relaxed);
	return computed;
}

// ─────────────────────────────────────────────────────────────────────────────
// Plan(problem) — the only planner. Legacy Select callers adapt the immutable
// plan back to the old route result while execution paths migrate.
// ─────────────────────────────────────────────────────────────────────────────

OaMatmulPlan OaGemmRouter::Plan(
	const OaEngine& InRt,
	const OaMatmulProblem& InProblem,
	OaMatmulPreference InPreference)
{
	OaMatmulProblem problem = InProblem;
	const bool forceFp32 = OaEnvFlag::IsSet("OA_GEMM_FORCE_FP32");
	const OaGemmPrecision precision = ResolvePrecision(problem);

	auto finish = [&](const OaMatmulVariant& variant) {
		auto plan = PlanForVariant(InRt, variant, problem);
		auto route = RouteForPlan(plan);
		LogAndCount(route, precision, problem.M, problem.N, problem.K);
		return plan;
	};
	auto findLegal = [&](OaMatmulVariantId id) -> const OaMatmulVariant* {
		const auto* variant = OaMatmulRegistry::Find(id);
		return variant != nullptr and VariantLegalResolved(InRt, *variant, problem, precision)
			? variant
			: nullptr;
	};

	// Explicit benchmark isolation. An incompatible forced variant is ignored;
	// executing a shader with a different epilogue or output contract is never
	// a valid benchmark.
	{
		auto& forcedMap = GetForcedMap();
		std::lock_guard<std::mutex> lock(forcedMap.Mtx);
		const auto it = forcedMap.Map.find(
			{.M = problem.M, .N = problem.N, .K = problem.K});
		if (it != forcedMap.Map.end()) {
			if (const auto* forced = findLegal(it->second)) {
				return finish(*forced);
			}
		}
	}

	// The route cache stores a stable variant identity and is always revalidated
	// against the live registry, full problem contract, device caps and shape.
	if (InPreference.UseMeasuredCache and InRt.GemmRouteCache != nullptr and not forceFp32) {
		OaMatmulVariantId cached = OaInvalidMatmulVariantId;
		if (InRt.GemmRouteCache->Query(CacheKey(InRt, problem), cached)) {
			if (const auto* winner = findLegal(cached)) {
				return finish(*winner);
			}
		}
	}

	// M=1 decode is the only distinct path. The legality predicate keeps this
	// NVIDIA-specific extension vendor-gated unless explicitly overridden.
	if (problem.Epilogue == OaGemmEpilogue::None and problem.M == 1U
		and precision != OaGemmPrecision::Fp32) {
		if (const auto* coopVec = findLegal(
			OaMatmulVariantIdFromName("GemmCoopVec"))) {
			return finish(*coopVec);
		}
	}

	// Tensor-core families are ordered by the current measured policy. The
	// registry supplies exact epilogue rows, so the same code covers raw, Bias,
	// Bias+ReLU, Bias+GELU and dual-output SiLU without name construction.
	if (precision != OaGemmPrecision::Fp32) {
		for (const OaGemmKernel family : {
			OaGemmKernel::GemmCmWgBf16,
			OaGemmKernel::GemmCmSgBf16}) {
			for (const auto& variant : OaMatmulRegistry::All()) {
				if (variant.Kernel == family
					and VariantLegalResolved(InRt, variant, problem, precision)) {
					return finish(variant);
				}
			}
		}
	}

	// Non-canonical views and strided batches use the universal fallback. It is
	// deliberately selected before contiguous FP32 heuristics; those kernels do
	// not understand explicit offsets/strides and must never be misrouted here.
	for (const auto& variant : OaMatmulRegistry::All()) {
		if (variant.Kernel == OaGemmKernel::StridedFp32
			and VariantLegalResolved(InRt, variant, problem, OaGemmPrecision::Fp32)) {
			const bool canonical = problem.BatchCount == 1U
				and problem.AContiguous and problem.BContiguous
				and problem.A.Offset == 0U and problem.B.Offset == 0U and problem.C.Offset == 0U
				and problem.A.RowStride == problem.K and problem.A.ColStride == 1U
				and problem.B.RowStride == problem.K and problem.B.ColStride == 1U
				and problem.C.RowStride == problem.N and problem.C.ColStride == 1U;
			if (not canonical) return finish(variant);
		}
	}

	// Portable FP32 fallback. Tiny raw GEMMs retain the scalar path; every
	// fused contract uses its tiled epilogue variant when one exists.
	const bool preferNaive = problem.Epilogue == OaGemmEpilogue::None
		and problem.M * problem.N < 64U;
	for (const OaGemmKernel family : preferNaive
		? std::initializer_list<OaGemmKernel>{OaGemmKernel::Naive, OaGemmKernel::TiledFp32}
		: std::initializer_list<OaGemmKernel>{OaGemmKernel::TiledFp32, OaGemmKernel::Naive}) {
		for (const auto& variant : OaMatmulRegistry::All()) {
			if (variant.Kernel == family
				and variant.APrecision == OaStoragePrecision::Fp32
				and VariantLegalResolved(InRt, variant, problem, OaGemmPrecision::Fp32)) {
				return finish(variant);
			}
		}
	}

	OA_LOG_ERROR(OaLogComponent::Core,
		"GemmRouter: no legal variant for M=%u N=%u K=%u epilogue=%u",
		problem.M, problem.N, problem.K, static_cast<OaU32>(problem.Epilogue));
	return {};
}

bool OaGemmRouter::ValidatePlan(
	const OaEngine& InRt,
	const OaMatmulPlan& InPlan,
	const OaMatmulProblem& InProblem)
{
	if (not InPlan
		or InPlan.RegistryBuildId != OaMatmulRegistry::BuildId()
		or InPlan.ProblemContractHash != ProblemContractHash(InProblem)
		or InPlan.DeviceContractHash != DeviceContractHash(InRt)) {
		return false;
	}
	const auto* variant = OaMatmulRegistry::Find(InPlan.Variant);
	if (variant == nullptr
		or std::strcmp(variant->KernelName, InPlan.KernelName) != 0
		or variant->Kernel != InPlan.Kernel
		or variant->Path != InPlan.Path
		or InPlan.ShaderContentHash == 0U
		or InPlan.ShaderContentHash != OaMatmulRegistry::ShaderContentHash(variant->Id)
		or not IsVariantLegal(InRt, *variant, InProblem)) {
		return false;
	}
	const auto expected = ResultForVariant(*variant, InProblem);
	return InPlan.ActualPrecision == expected.ActualPrec
		and InPlan.Grid.X == expected.Gx
		and InPlan.Grid.Y == expected.Gy
		and InPlan.Grid.Z == expected.Gz
		and InPlan.WorkspaceBytes == 0U;
}

OaGemmRouteResult OaGemmRouter::Select(
	const OaEngine& InRt,
	const OaMatmulProblem& InProblem)
{
	return RouteForPlan(Plan(InRt, InProblem));
}

OaMatmulProblem OaGemmRouter::ProblemForRaw(
	OaU32 InM, OaU32 InN, OaU32 InK,
	OaStoragePrecision InAMaster,
	OaStoragePrecision InBMaster,
	bool  InBTransposed)
{
	OaMatmulProblem p;
	p.M                       = InM;
	p.N                       = InN;
	p.K                       = InK;
	p.BatchCount              = 1U;
	p.A                       = {.Offset = 0U, .RowStride = InK,
		.ColStride = 1U, .BatchStride = InM * InK};
	p.B                       = {.Offset = 0U, .RowStride = InK,
		.ColStride = 1U, .BatchStride = InN * InK};
	p.C                       = {.Offset = 0U, .RowStride = InN,
		.ColStride = 1U, .BatchStride = InM * InN};
	p.AMaster                 = InAMaster;
	p.BMaster                 = InBMaster;
	p.RequestedOutput         = OaStoragePrecision::Fp32;
	p.AContiguous             = true;
	p.BContiguous             = true;
	p.BTransposed             = InBTransposed;
	p.Epilogue                = OaGemmEpilogue::None;
	p.RequiresPreActivation   = false;
	p.Training                = true;
	p.PrecisionHint           = OaGemmPrecision::Auto;
	return p;
}

// ─────────────────────────────────────────────────────────────────────────────
// ForceKernel / ClearForced
// ─────────────────────────────────────────────────────────────────────────────

void OaGemmRouter::ForceKernel(OaU32 InM, OaU32 InN, OaU32 InK, OaGemmKernel InKernel) {
	const OaMatmulVariant* variant = FindRawGemmVariant(
		InKernel,
		InKernel == OaGemmKernel::CoopVec ? OaGemmPath::CoopVec : OaGemmPath::Standard);
	ForceVariant(InM, InN, InK,
		variant != nullptr ? variant->Id : OaInvalidMatmulVariantId);
}

void OaGemmRouter::ForceVariant(
	OaU32 InM, OaU32 InN, OaU32 InK, OaMatmulVariantId InVariant) {
	auto& fm = GetForcedMap();
	std::lock_guard<std::mutex> lock(fm.Mtx);
	fm.Map[{.M = InM, .N = InN, .K = InK}] = InVariant;
}

void OaGemmRouter::ClearForced() {
	auto& fm = GetForcedMap();
	std::lock_guard<std::mutex> lock(fm.Mtx);
	fm.Map.clear();
}

// ─────────────────────────────────────────────────────────────────────────────
// PrecisionAvailable
// ─────────────────────────────────────────────────────────────────────────────

bool OaGemmRouter::PrecisionAvailable(
	const OaEngine& InRt,
	OaGemmPrecision          InPrec)
{
	const auto& sw = InRt.Device.Info.Software;
	switch (InPrec) {
		case OaGemmPrecision::Fp32: return true;
		case OaGemmPrecision::Bf16:
			return sw.ShaderBfloat16CooperativeMatrixEnabled and sw.ShaderBfloat16TypeEnabled;
		case OaGemmPrecision::Auto:
			return true;
		default:
			return false;
	}
}

bool OaGemmRouter::IsGemmCmSgBf16Suitable(
	const OaEngine& InRt,
	OaU32                    InM,
	OaU32                    InN,
	OaU32                    InK) {
	// Portable CoopMat1 triplet — GemmCmSgBf16 uses only VK_KHR_cooperative_matrix
	// at the universal 16×16×16 bf16-in/fp32-acc shape, so it qualifies on
	// AMD RDNA3.5/Strix as well as NVIDIA. No NV CoopMat2 dependency.
	const OaU64 caps = InRt.GemmCapsMask();
	const OaU64 need = kCapCoopMat1Khr | kCapCoopMat1Bf16Input | kCapCoopMat1Fp32Acc;
	if (not OaMatmulRegistry::CapsSatisfy(caps, need)) {
		return false;
	}
	// GemmCmSgBf16 uses 16×16 fragments and a direct-to-global store (raw) or
	// SMEM-staged element-wise copy (fused). M and N must be multiples of 16.
	if ((InM % 16U) != 0U or (InN % 16U) != 0U) {
		return false;
	}
	const OaU32 minDim = 64U;
	return InM >= minDim and InN >= minDim and InK >= minDim;
}

bool OaGemmRouter::IsGemmCmWgBf16Suitable(
	const OaEngine& InRt,
	OaU32                    InM,
	OaU32                    InN,
	OaU32                    InK
) {
	// Workgroup-scope 32x32x16 KHR CoopMat. NVIDIA-favored; requires the device
	// to report a workgroup-scope BF16 input shape.
	const OaU64 caps = InRt.GemmCapsMask();
	const OaU64 need = kCapCoopMat1Khr | kCapCoopMat1WorkgroupBf16 | kCapCoopMat1Fp32Acc;
	if (not OaMatmulRegistry::CapsSatisfy(caps, need)) {
		return false;
	}
	if (OaEnvFlag::IsSet("OA_DISABLE_COOPMAT2")) {
		return false;
	}
	// 32x32 output blocks require M,N multiples of 32 for the fast path.
	if ((InM % 32U) != 0U or (InN % 32U) != 0U) {
		return false;
	}
	const OaU32 minDim = 64U;
	return InM >= minDim and InN >= minDim and InK >= minDim;
}
