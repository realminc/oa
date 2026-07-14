#include <Oa/Runtime/Gemm/Router.h>
#include <Oa/Runtime/Gemm/Cache.h>
#include <Oa/Runtime/MatmulTypes.h>
#include <Oa/Core/EnvFlag.h>
#include <Oa/Core/Log.h>
#include <Oa/Core/Validation.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Device.h>
#include <Oa/Runtime/Pipeline.h>
#include <Oa/Runtime/GemmTypes.h>
#include <Oa/Runtime/GemmRouteCache.h>

#include <atomic>
#include <mutex>
#include <unordered_map>

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

namespace {

// Tile constants now live in OaMatmulRegistry::kVariants (see Registry.cpp).
// The CoopVec workgroup-N denominator is read from the registry too.

inline OaU32 DivCeil(OaU32 InA, OaU32 InB) { return (InA + InB - 1U) / InB; }

// Build cache key from routing parameters
static OaRouteCacheKey BuildRouteCacheKey(
	const OaComputeEngine& InRt,
	OaGemmKernel             InVariant,
	OaU32                    InM,
	OaU32                    InN,
	OaU32                    InK,
	OaGemmPrecision          InAPrec,
	OaGemmPrecision          InBPrec,
	OaGemmEpilogue           InEpilogue,
	bool                     InTraining)
{
	auto hashString = [](const char* text) {
		OaU64 h = 0xcbf29ce484222325ULL;
		for (const char* p = text; p != nullptr and *p != '\0'; ++p) {
			h ^= static_cast<OaU8>(*p);
			h *= 0x100000001b3ULL;
		}
		return h;
	};
	OaRouteCacheKey key{};
	key.VendorId = InRt.Device.Info.Hardware.VendorId;
	key.DeviceId = InRt.Device.Info.Hardware.DeviceId;
	key.DriverId = InRt.Device.Info.Software.DriverId;
	key.DriverVersionHash = hashString(InRt.Device.Info.Software.DriverVersion.c_str());
	key.ShaderBuildId = OaMatmulRegistry::BuildId();
	key.Variant = InVariant;
	key.M = InM;
	key.N = InN;
	key.K = InK;
	key.APrecision = InAPrec;
	key.BPrecision = InBPrec;
	key.Epilogue = InEpilogue;
	key.Training = InTraining;
	key.UseTMA = InRt.IsBlackwell();  // Enable TMA for Blackwell GPUs
	return key;
}

static bool CachedWinnerLegal(
	const OaComputeEngine& InRt,
	OaGemmKernel InWinner,
	OaU32 InM,
	OaU32 InN,
	OaU32 InK,
	OaGemmPrecision InPrecision)
{
	switch (InWinner) {
		case OaGemmKernel::TiledFp32:
		case OaGemmKernel::Naive:
			return true;
		case OaGemmKernel::GemmCmSgBf16:
			return InPrecision != OaGemmPrecision::Fp32
				and OaGemmRouter::IsGemmCmSgBf16Suitable(InRt, InM, InN, InK);
		case OaGemmKernel::GemmCmWgBf16:
			return InPrecision != OaGemmPrecision::Fp32
				and OaGemmRouter::IsGemmCmWgBf16Suitable(InRt, InM, InN, InK);
		case OaGemmKernel::CoopVec: {
			constexpr OaU32 kNvidia = 0x10DEU;
			const bool trustedVendor = InRt.Device.Info.Hardware.VendorId == kNvidia
				or OaEnvFlag::IsSet("OA_FORCE_COOPVEC");
			return InPrecision != OaGemmPrecision::Fp32 and InM == 1U and trustedVendor
				and OaMatmulRegistry::CapsSatisfy(InRt.GemmCapsMask(), kCapCoopVec);
		}
		default:
			return false;
	}
}

// ScoreVariant — occupancy heuristic ported from ggml's
// ggml_vk_guess_matmul_pipeline. Reward filling the device once, penalize
// gross overspill. Higher score = better fit for this shape. Variants below
// minimum legality must be filtered before scoring; this function does not
// know about caps or activation requirements.
//
// fill   = min(1, totalTiles / cores)        plateaus at one wave per SM
// spill  = max(0, totalTiles - 4 * cores) / (4 * cores)  overspill > 4 waves
// score  = fill - 0.25 * spill                tolerates moderate spill
inline OaF32 ScoreVariant(const OaMatmulVariant& InVariant,
                          OaU32 InM, OaU32 InN, OaU32 InCores)
{
	if (InVariant.TileM == 0U or InVariant.TileN == 0U or InCores == 0U) {
		return 0.0F;
	}
	const OaU32 tilesM = DivCeil(InM, InVariant.TileM);
	const OaU32 tilesN = DivCeil(InN, InVariant.TileN);
	const OaU32 total  = tilesM * tilesN;
	const OaF32 cores  = static_cast<OaF32>(InCores);
	const OaF32 fill   = std::min(1.0F, static_cast<OaF32>(total) / cores);
	const OaF32 spill  = std::max(0.0F,
		(static_cast<OaF32>(total) - 4.0F * cores) / (4.0F * cores));
	return fill - 0.25F * spill;
}

// PickBestRawVariant — among raw-GEMM rows in the registry whose Kernel
// enum is in InCandidateKernels and whose required caps are live, return
// the variant with the highest ScoreVariant for (M, N) on the device.
// Returns nullptr when no candidate matches. InCandidateKernels is a
// stack-allocated initializer list; the helper is templated to avoid
// pulling OaSpan into the closure of every Select branch.
template <OaUsize InCount>
const OaMatmulVariant* PickBestRawVariant(
	const OaGemmKernel (&InCandidateKernels)[InCount],
	OaGemmPath  InPath,
	OaU32       InM,
	OaU32       InN,
	OaU32       InCores,
	OaU64       InCapsAvailable)
{
	const OaMatmulVariant* best = nullptr;
	OaF32 bestScore = -1.0F;
	for (const auto& v : OaMatmulRegistry::All()) {
		if (v.Path != InPath) {
			continue;
		}
		if (v.SupportsBias or v.SupportsActivation) {
			continue;
		}
		bool match = false;
		for (OaUsize i = 0; i < InCount; ++i) {
			if (v.Kernel == InCandidateKernels[i]) {
				match = true;
				break;
			}
		}
		if (not match) {
			continue;
		}
		if (not OaMatmulRegistry::CapsSatisfy(InCapsAvailable, v.RequiredCapsMask)) {
			continue;
		}
		const OaF32 score = ScoreVariant(v, InM, InN, InCores);
		if (score > bestScore) {
			bestScore = score;
			best = &v;
		}
	}
	return best;
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
		if (v.SupportsBias or v.SupportsActivation) {
			continue;
		}
		return &v;
	}
	return nullptr;
}

// Build a route result for a Standard-path kernel using the registry's tile.
// gx = DivCeil(M, TileM), gy = DivCeil(N, TileN). Returns Naive fallback if
// the kernel isn't in the registry (shouldn't happen — the registry is the
// source of truth for shipping kernels).
OaGemmRouteResult ResultForKernel(OaGemmKernel InKernel, OaGemmPath InPath,
                                   OaU32 InM, OaU32 InN, OaU32 InNumSMs) {
	const OaMatmulVariant* v = FindRawGemmVariant(InKernel, InPath);
	if (v == nullptr) {
		return {.KernelName = "GemmNaive", .Kernel = OaGemmKernel::Naive,
		         .Path = OaGemmPath::Standard, .ActualPrec = OaGemmPrecision::Fp32,
		         .Gx = ((InM * InN) + 255U) / 256U, .Gy = 1U};
	}
	OaU32 gx = 1U;
	OaU32 gy = 1U;
	OaGemmPrecision prec = OaGemmPrecision::Fp32;
	switch (v->APrecision) {
		case OaStoragePrecision::Bf16: prec = OaGemmPrecision::Bf16; break;
		case OaStoragePrecision::Fp32: prec = OaGemmPrecision::Fp32; break;
	}
	switch (InPath) {
		case OaGemmPath::Standard:
			// Standard uses the per-tile grid. Select fills these in directly,
			// but ResultForKernel is also a fallback for ForceKernel paths and
			// route-cache promotions. Persistent variants override: one workgroup
			// per SM, walking tiles in a strided loop inside the shader.
			if (v->Persistent) {
				gx = InNumSMs;
				gy = 1U;
			} else {
				gx = DivCeil(InM, v->TileM);
				gy = DivCeil(InN, v->TileN);
			}
			break;
		case OaGemmPath::StreamK:
			gx = InNumSMs;
			gy = 1U;
			break;
		case OaGemmPath::CoopVec:
			// CoopVec GEMV: workgroup covers v->TileN rows of output.
			gx = DivCeil(InN, v->TileN);
			gy = 1U;
			break;
	}
	return {.KernelName = v->KernelName, .Kernel = InKernel, .Path = InPath,
	         .ActualPrec = prec, .Gx = gx, .Gy = gy};
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
	std::unordered_map<ForceKey, OaGemmKernel, ForceKeyHash>    Map;
};

ForcedMap& GetForcedMap() {
	static ForcedMap s;
	return s;
}

// ── Kernel name / precision lookup tables ────────────────────────────────────

const char* KernelName(OaGemmKernel InKernel) {
	switch (InKernel) {
		case OaGemmKernel::TiledFp32:         return "GemmTiled";
		case OaGemmKernel::CoopVec:           return "GemmCoopVec";
		default:                              return "GemmNaive";
	}
}

OaGemmPrecision KernelPrec(OaGemmKernel InKernel) {
	switch (InKernel) {
		case OaGemmKernel::CoopVec:     return OaGemmPrecision::Bf16;
		default:                        return OaGemmPrecision::Fp32;
	}
}

OaGemmRouteResult MakeResult(OaGemmKernel InKernel, OaGemmPath InPath, OaU32 InGx, OaU32 InGy) {
	return {
		.KernelName  = KernelName(InKernel),
		.Kernel      = InKernel,
		.Path        = InPath,
		.ActualPrec  = KernelPrec(InKernel),
		.Gx          = InGx,
		.Gy          = InGy,
	};
}

OaGemmRouteResult Naive(OaU32 InM, OaU32 InN) {
	return MakeResult(OaGemmKernel::Naive, OaGemmPath::Standard,
		((InM * InN) + 255U) / 256U, 1U);
}

OaGemmRouteResult Tiled(OaU32 InM, OaU32 InN) {
	return ResultForKernel(OaGemmKernel::TiledFp32, OaGemmPath::Standard,
		InM, InN, /*InNumSMs=*/0U);
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
		case OaGemmPath::StreamK:  return "StreamK";
		case OaGemmPath::CoopVec:  return "CoopVec";
	}
	return "?";
}

const char* PrecName(OaGemmPrecision InPrec) {
	switch (InPrec) {
		case OaGemmPrecision::Bf16: return "BF16";
		case OaGemmPrecision::Fp32: return "FP32";
		default:                    return "?";
	}
}

void LogAndCount(const OaGemmRouteResult& InR, OaU32 InM, OaU32 InN, OaU32 InK) {
	// Only increment counters (compiled out in Release)
	OA_DEBUG_COUNTER_INC_NAMED(InR.KernelName);

	// Opt-in per-call decision log (OA_LOG_GEMM_ROUTER=1).
	// Documents which kernel was selected for each (M, N, K) shape.
	if (GemmRouterLogEnabled()) {
		OA_LOG_INFO(OaLogComponent::Core,
			"GemmRouter: M=%u N=%u K=%u prec=%s -> %s (path=%s, gx=%u, gy=%u)",
			InM, InN, InK, PrecName(InR.ActualPrec),
			InR.KernelName, PathName(InR.Path), InR.Gx, InR.Gy);
	}

	// Only warn on Naive fallback for large GEMMs (performance issue)
	OA_WARN_PERF(
		InR.Kernel == OaGemmKernel::Naive and InM * InN * InK > 1024U,
		"GemmRouter: Naive path for M=%u N=%u K=%u — no tensor cores available",
		InM, InN, InK);
}

} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// OaGemmRouter::Select
// ─────────────────────────────────────────────────────────────────────────────

OaGemmRouteResult OaGemmRouter::Select(
	const OaComputeEngine& InRt,
	OaU32                    InM,
	OaU32                    InN,
	OaU32                    InK,
	OaGemmPrecision          InPrec)
{
	// ── Step 0a: Deterministic fp32 override ─────────────────────────────────
	// OA_GEMM_FORCE_FP32 forces every GEMM to the exact fp32 Tiled/Naive path,
	// bypassing the bf16 Auto routing AND the learned route cache (which can
	// return a bf16 winner even for an Fp32 request). This is required for
	// finite-difference gradient checking: bf16's ~3-digit mantissa swallows the
	// small (≈1e-4) weight perturbations, collapsing numerical gradients to ~0
	// and producing false backward-pass failures. Also useful for bit-stable
	// debugging / regression triage.
	const bool forceFp32 = OaEnvFlag::IsSet("OA_GEMM_FORCE_FP32");
	if (forceFp32) {
		InPrec = OaGemmPrecision::Fp32;
	}

	// ── Step 0: ForceKernel override (benchmarking / isolation) ──────────────
	{
		auto& fm = GetForcedMap();
		std::lock_guard<std::mutex> lock(fm.Mtx);
		auto it = fm.Map.find({.M = InM, .N = InN, .K = InK});
		if (it != fm.Map.end()) {
			OaGemmKernel forced = it->second;
			OaGemmPath path = OaGemmPath::Standard;
			switch (forced) {
				case OaGemmKernel::CoopVec:
					path = OaGemmPath::CoopVec; break;
				default: break;
			}
			auto r = ResultForKernel(forced, path, InM, InN,
				InRt.Device.Info.Hardware.NumSMs);
			OA_LOG_DEBUG(OaLogComponent::Core,
				"GemmRouter M=%u N=%u K=%u → '%s' (FORCED)",
				InM, InN, InK, r.KernelName);
			return r;
		}
	}

	// ── Step 1: Route cache lookup (learned profitability — PRIMARY) ────────
	// The autotuner benchmarks the real production path and stores winners here.
	// This takes precedence over heuristics, including the Fp32 fast-path, but
	// CachedWinnerLegal revalidates precision/caps/shape before replay. Explicit
	// Fp32 can reuse a measured Fp32 winner; it can never replay BF16/CoopMat.
	// OA_GEMM_FORCE_FP32 remains a deterministic cache-bypass override.
	if (InRt.GemmRouteCache && not forceFp32) {
		auto key = BuildRouteCacheKey(
			InRt, OaGemmKernel::Auto, InM, InN, InK, InPrec, InPrec,
			OaGemmEpilogue::None, false);
		OaGemmKernel cachedWinner;
		if (InRt.GemmRouteCache->Query(key, cachedWinner)
			and CachedWinnerLegal(InRt, cachedWinner, InM, InN, InK, InPrec)) {
			OaGemmPath path = OaGemmPath::Standard;
			switch (cachedWinner) {
				case OaGemmKernel::CoopVec:
					path = OaGemmPath::CoopVec; break;
				default: break;
			}
			auto r = ResultForKernel(cachedWinner, path, InM, InN,
				InRt.Device.Info.Hardware.NumSMs);
			OA_LOG_DEBUG(OaLogComponent::Core,
				"GemmRouter M=%u N=%u K=%u → '%s' (ROUTE CACHE)",
				InM, InN, InK, r.KernelName);
			LogAndCount(r, InM, InN, InK);
			return r;
		}
	}

	// ── Step 2: Fp32 fast-path (no CoopMat) ─────────────────────────────────
	// Only used when there is NO route-cache entry for this shape+precision.
	// If the autotuner hasn't seen this shape yet, we fall back to the safest
	// known-good kernel. For Fp32 that's always Tiled (or Naive for trivial).
	if (InPrec == OaGemmPrecision::Fp32) {
		auto r = (InM * InN >= 64U and InK >= 1U) ? Tiled(InM, InN) : Naive(InM, InN);
		LogAndCount(r, InM, InN, InK);
		return r;
	}

	// ── Step 3: Pipeline cache lookup (legacy, to be deprecated) ────────────
	{
		OaGemmKernel cached = InRt.Pipelines.LookupGemmKernel(InM, InN, InK);
		if (cached != OaGemmKernel::Auto) {
			OaGemmPath path = OaGemmPath::Standard;
			switch (cached) {
				case OaGemmKernel::CoopVec:
					path = OaGemmPath::CoopVec; break;
				default: break;
			}
			auto r = ResultForKernel(cached, path, InM, InN,
				InRt.Device.Info.Hardware.NumSMs);
			LogAndCount(r, InM, InN, InK);
			return r;
		}
	}

	const OaU32 nSMs = InRt.Device.Info.Hardware.NumSMs;

	// Device cap mask — lazy-cached on the engine via GemmCapsMask(). The
	// branch guards below read the relevant CoopMat1/CoopVec bits, which
	// keeps the gate aligned with each variant's RequiredCapsMask.
	const OaU64 caps    = InRt.GemmCapsMask();
	const bool  hasBf16 = OaMatmulRegistry::CapsSatisfy(caps, kCapCoopMat1Bf16Input);
	const bool  hasCv   = OaMatmulRegistry::CapsSatisfy(caps, kCapCoopVec);

	// ── Step 3: InPrec == Bf16 requested explicitly ───────────────────────────
	if (InPrec == OaGemmPrecision::Bf16) {
		if (not hasBf16) {
			OA_LOG_DEBUG(OaLogComponent::Core,
				"GemmRouter: Bf16 requested but device lacks BF16 CoopMat — using Naive M=%u N=%u K=%u",
				InM, InN, InK);
			auto r = Naive(InM, InN);
			LogAndCount(r, InM, InN, InK);
			return r;
		}
		// Fall through to BF16 path below
	}

	// ── Step 4: M==1 decode path — CooperativeVector GEMV (NVIDIA Blackwell+ only) ───
	// NOTE: VK_NV_cooperative_vector is NVIDIA-specific (no VK_EXT equivalent exists).
	// This is a Blackwell+ optimization for decode inference (M=1 GEMV).
	// Vendor-gated at routing time (NOT at pipeline-load time per OaLlamaCppVulkanLessons.md
	// PR-1 item E): the pipeline can be loaded on any device that compiles the shader, but
	// we only route to it on NVIDIA. Opt-in via OA_FORCE_COOPVEC=1 bypasses the vendor gate
	// for testing on AMD/Intel where the extension is exposed but untrusted.
	constexpr OaU32 kVulkanVendorIdNvidia = 0x10DEU;
	const bool nvidia = InRt.Device.Info.Hardware.VendorId == kVulkanVendorIdNvidia;
	const bool coopvec_ok = hasCv && (nvidia || OaEnvFlag::IsSet("OA_FORCE_COOPVEC"));
	if (InM == 1U and coopvec_ok) {
		auto r = ResultForKernel(OaGemmKernel::CoopVec, OaGemmPath::CoopVec,
			InM, InN, nSMs);
		LogAndCount(r, InM, InN, InK);
		return r;
	}

	// ── Step 4b: tuned KHR CoopMat GEMM route (NVIDIA + AMD RDNA3.5/Strix) ─
	// Prefer the workgroup-scope 32x32x16 variant when the device advertises it
	// and the shape is 32-aligned. Otherwise fall back to the subgroup-scope
	// 16x16x16 variant which is universally supported on KHR CoopMat BF16.
	if (InPrec != OaGemmPrecision::Fp32) {
		if (IsGemmCmWgBf16Suitable(InRt, InM, InN, InK)) {
			OaGemmRouteResult r{
				.KernelName = "GemmCmWgBf16",
				.Kernel     = OaGemmKernel::GemmCmWgBf16,
				.Path       = OaGemmPath::Standard,
				.ActualPrec = OaGemmPrecision::Bf16,
				.Gx         = DivCeil(InM, 64U),
				.Gy         = DivCeil(InN, 64U),
			};
			OA_LOG_DEBUG(OaLogComponent::Core,
				"GemmRouter M=%u N=%u K=%u → 'GemmCmWgBf16' (64x64 KHR workgroup)",
				InM, InN, InK);
			LogAndCount(r, InM, InN, InK);
			return r;
		}
		if (IsGemmCmSgBf16Suitable(InRt, InM, InN, InK)) {
			OaGemmRouteResult r{
				.KernelName = "GemmCmSgBf16",
				.Kernel     = OaGemmKernel::GemmCmSgBf16,
				.Path       = OaGemmPath::Standard,
				.ActualPrec = OaGemmPrecision::Bf16,
				.Gx         = DivCeil(InM, 128U),
				.Gy         = DivCeil(InN, 128U),
			};
			OA_LOG_DEBUG(OaLogComponent::Core,
				"GemmRouter M=%u N=%u K=%u → 'GemmCmSgBf16' (128x128 KHR subgroup)",
				InM, InN, InK);
			LogAndCount(r, InM, InN, InK);
			return r;
		}
	}

	// ── Step 6: Software-tiled FP32 ──────────────────────────────────────────
	// FIX: Lowered threshold from 16 to 0 for N dimension — Tiled works fine with
	// N=1 and is always faster than Naive for M*N >= 64.
	if (InM * InN >= 64U and InK >= 1U) {
		auto r = Tiled(InM, InN);
		LogAndCount(r, InM, InN, InK);
		return r;
	}

	// ── Step 8: Naive fallback ────────────────────────────────────────────────
	auto r = Naive(InM, InN);
	LogAndCount(r, InM, InN, InK);
	return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// OaComputeEngine::GemmCapsMask — lazy-init cache for ComputeCapsMask.
// The mask depends only on Software/Hardware info populated at device init,
// so we compute it once and stash it on the engine. Two threads racing the
// first read will both compute the same value and CAS the result; the
// std::memory_order_relaxed loads/stores are fine because the cap mask is
// idempotent — there is no payload the reader depends on through the atomic.
// ─────────────────────────────────────────────────────────────────────────────

OaU64 OaComputeEngine::GemmCapsMask() const {
	OaU64 cached = GemmCapsMask_.load(std::memory_order_relaxed);
	if (cached != 0U) {
		return cached;
	}
	const OaU64 computed = OaMatmulRegistry::ComputeCapsMask(*this);
	GemmCapsMask_.store(computed, std::memory_order_relaxed);
	return computed;
}

// ─────────────────────────────────────────────────────────────────────────────
// Select(problem) — canonical R5 entry. Today this forwards to the (M,N,K,prec)
// overload; the problem struct carries the additional fields (mirror state,
// bias/activation flags) that future selection layers will consume directly.
// Pre-R5 callers using the (M,N,K,prec) form are unaffected.
// ─────────────────────────────────────────────────────────────────────────────

OaGemmRouteResult OaGemmRouter::Select(
	const OaComputeEngine& InRt,
	const OaMatmulProblem&   InProblem)
{
	OaGemmPrecision prec = InProblem.PrecisionHint;
	if (prec == OaGemmPrecision::Auto) {
		switch (InProblem.AMaster) {
			case OaStoragePrecision::Bf16: prec = OaGemmPrecision::Bf16; break;
			case OaStoragePrecision::Fp32: prec = OaGemmPrecision::Auto; break;
		}
	}
	return Select(InRt, InProblem.M, InProblem.N, InProblem.K, prec);
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
	auto& fm = GetForcedMap();
	std::lock_guard<std::mutex> lock(fm.Mtx);
	fm.Map[{.M = InM, .N = InN, .K = InK}] = InKernel;
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
	const OaComputeEngine& InRt,
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
	const OaComputeEngine& InRt,
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
	const OaComputeEngine& InRt,
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
