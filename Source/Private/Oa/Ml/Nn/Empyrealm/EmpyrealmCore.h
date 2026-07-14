#pragma once

// Oa includes first.
#include <Oa/Ml/Nn.h>
#include <Oa/Ml/Module.h>

// Forward declaration for the mixer (defined in EmpyrealmModule.h).
class OaEmpyrealmModule;

// Tag type selecting the embedded-only OaEmpyrealmCore constructor (no embed
// module is created or registered; the caller drives the mixer via
// ForwardEmbedded with its own [B, S, D] features). Disambiguates from the
// byte ctor, whose two leading ints would otherwise collide.
struct OaEmpyrealmCoreEmbeddedOnly {};

// OaEmpyrealmCore — Empyrealm-style sequential modeling core.
//
// High-utilization reusable backbone:
//   input projection (byte embed or custom) → mixer + flat per-token residual
//   → mixed features [B*Seq, DModel].
//
// The mixer is OaEmpyrealmModule, which dispatches Empyrealm* kernels
// (EmpyrealmDt, EmpyrealmAdt, EmpyrealmSiso) — renamed copies of the Mamba3*
// kernels with identical SPIR-V today, ready for future architecture-specific
// divergence.
//
// Shader layout (for fusion / branding):
//   Ssm/Mamba3/   — original Mamba3Siso* (untouched reference, used by OaMamba3Module)
//   Ssm/Empyrealm/ — ported/copied starting point; will host fused Empyrealm* variants
//                    (e.g. custom one-node mixers) while Mamba3 stays pristine.
//
// The reconstruction tutorial demonstrates the intended usage. Brand Mamba3 tech
// as Empyrealm for custom evolution.
class OaEmpyrealmCore : public OaModule {
public:
	// Byte/token constructor. Forward expects [B, Seq] uint8/index matrix.
	// Creates internal OaEmbedding.
	OaEmpyrealmCore(
		OaI32 InVocabSize, OaI32 InDModel,
	  OaI32 InDState = 32,
	  OaI32 InExpand = 2,
	  OaI32 InHeadDim = 16,
	  OaI32 InNGroups = 1,
	  OaF32 InRopeFraction = 0.5f,
	  bool InIsMimo = false,
	  OaI32 InMimoRank = 1,
	  OaF32 InDtMin = 0.001f,
	  OaF32 InDtMax = 0.1f,
	  OaF32 InDtInitFloor = 1e-4f,
	  OaF32 InAFloor = 1e-4f,
		bool InIsOutprojNorm = true
	);
	// UseFused parameter REMOVED 2026-06-18 — fused experiment deleted.
	// ForwardEmbedded always uses the verified Mamba-3 path.

	// General constructor for specialization. Pass custom projection
	// (e.g. OaLinear(poseDim, dModel) for motion features). The module
	// must output [B, Seq, DModel] (or compatible with ForwardEmbedded).
	OaEmpyrealmCore(
		OaSharedPtr<OaModule> InEmbedModule, OaI32 InDModel,
		OaI32 InDState = 32,
		OaI32 InExpand = 2,
		OaI32 InHeadDim = 16,
		OaI32 InNGroups = 1,
		OaF32 InRopeFraction = 0.5f,
		bool InIsMimo = false,
		OaI32 InMimoRank = 1,
		OaF32 InDtMin = 0.001f,
		OaF32 InDtMax = 0.1f,
		OaF32 InDtInitFloor = 1e-4f,
		OaF32 InAFloor = 1e-4f,
		bool InIsOutprojNorm = true
	);
	// UseFused parameter REMOVED 2026-06-18 — fused experiment deleted.

	// Embedded-only constructor. No embed module is created or registered, so
	// no dead/zero-gradient embed params leak into AllParameterPtrs(), the
	// optimizer, or checkpoints. The caller must feed pre-embedded [B, S, D]
	// features via ForwardEmbedded(); Forward() is unavailable in this mode and
	// asserts. Used by specializations that own their own input projection.
	OaEmpyrealmCore(
		OaEmpyrealmCoreEmbeddedOnly, OaI32 InDModel,
	  OaI32 InDState = 32,
	  OaI32 InExpand = 2,
	  OaI32 InHeadDim = 16,
	  OaI32 InNGroups = 1,
	  OaF32 InRopeFraction = 0.5f,
	  bool InIsMimo = false,
	  OaI32 InMimoRank = 1,
	  OaF32 InDtMin = 0.001f,
	  OaF32 InDtMax = 0.1f,
	  OaF32 InDtInitFloor = 1e-4f,
	  OaF32 InAFloor = 1e-4f,
	  bool InIsOutprojNorm = true
	);

	// Destructors.
	~OaEmpyrealmCore() = default;

	// Methods.
	OaMatrix Forward(const OaMatrix& InInput) override;

	// Mixer + flat residual on pre-embedded [B, S, D] or [B*S, D] features.
	// Returns flat [B*Seq, DModel].
	OaMatrix ForwardEmbedded(const OaMatrix& InEmbedded);

	[[nodiscard]] OaSharedPtr<OaEmpyrealmModule> Mixer() const noexcept { return Mixer_; }

	void ResetState(OaI32 InBatch);

private:
	OaSharedPtr<OaModule>       Embed_;
	OaSharedPtr<OaEmpyrealmModule> Mixer_;
	OaI32                       DModel_;
	bool                        IsByteMode_ = false;
	// UseFused_ removed 2026-06-18 — fused experiment deleted.
};
