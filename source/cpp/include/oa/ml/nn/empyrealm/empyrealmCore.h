#pragma once

// OA includes first.
#include <oa/ml/nn.h>
#include <oa/ml/module.h>

namespace oa {

// forward declaration for the mixer (defined in EmpyrealmModule.h).
class EmpyrealmModule;

// tag type selecting the embedded-only EmpyrealmCore constructor (no embed
// module is created or registered; the caller drives the mixer via
// ForwardEmbedded with its own [B, S, D] features). Disambiguates from the
// byte ctor, whose two leading ints would otherwise collide.
struct EmpyrealmCoreEmbeddedOnly {};

// EmpyrealmCore — empyrealm-style sequential modeling core.
//
// high-utilization reusable backbone:
//   input projection (byte embed or custom) → mixer + flat per-token residual
//   → mixed features [B*seq, dModel].
//
// The mixer is EmpyrealmModule, which dispatches empyrealm* kernels
// (EmpyrealmDt, EmpyrealmAdt, EmpyrealmSiso) — renamed copies of the Mamba3*
// kernels with identical SPIR-V today, ready for future architecture-specific
// divergence.
//
// shader layout (for fusion / branding):
//   Ssm/Mamba3/   — original Mamba3Siso* (untouched reference, used by Mamba3Module)
//   Ssm/empyrealm/ — ported/copied starting point; will host fused empyrealm* variants
//                    (e.g. custom one-node mixers) while Mamba3 stays pristine.
//
// The reconstruction tutorial demonstrates the intended usage. Brand Mamba3 tech
// as empyrealm for custom evolution.
class EmpyrealmCore : public oa::Module {
public:
	// Byte/token constructor. forward expects [B, seq] uint8/index matrix.
	// Creates internal oa::Embedding.
	EmpyrealmCore(
		oa::I32 inVocabSize, oa::I32 inDModel,
	  oa::I32 inDState = 32,
	  oa::I32 inExpand = 2,
	  oa::I32 inHeadDim = 16,
	  oa::I32 inNGroups = 1,
	  oa::F32 inRopeFraction = 0.5f,
	  bool inIsMimo = false,
	  oa::I32 inMimoRank = 1,
	  oa::F32 inDtMin = 0.001f,
	  oa::F32 inDtMax = 0.1f,
	  oa::F32 inDtInitFloor = 1e-4f,
	  oa::F32 inAFloor = 1e-4f,
		bool inIsOutprojNorm = true
	);
	// UseFused parameter REMOVED 2026-06-18 — fused experiment deleted.
	// ForwardEmbedded always uses the verified Mamba-3 path.

	// General constructor for specialization. Pass custom projection
	// (e.g. oa::Linear(poseDim, dModel) for motion features). The module
	// must output [B, seq, dModel] (or compatible with ForwardEmbedded).
	EmpyrealmCore(
		oa::SharedPtr<oa::Module> inEmbedModule, oa::I32 inDModel,
		oa::I32 inDState = 32,
		oa::I32 inExpand = 2,
		oa::I32 inHeadDim = 16,
		oa::I32 inNGroups = 1,
		oa::F32 inRopeFraction = 0.5f,
		bool inIsMimo = false,
		oa::I32 inMimoRank = 1,
		oa::F32 inDtMin = 0.001f,
		oa::F32 inDtMax = 0.1f,
		oa::F32 inDtInitFloor = 1e-4f,
		oa::F32 inAFloor = 1e-4f,
		bool inIsOutprojNorm = true
	);
	// UseFused parameter REMOVED 2026-06-18 — fused experiment deleted.

	// Embedded-only constructor. No embed module is created or registered, so
	// no dead/zero-gradient embed params leak into allParameterPtrs(), the
	// optimizer, or checkpoints. The caller must feed pre-embedded [B, S, D]
	// features via forwardEmbedded(); forward() is unavailable in this mode and
	// asserts. Used by specializations that own their own input projection.
	EmpyrealmCore(
		EmpyrealmCoreEmbeddedOnly, oa::I32 inDModel,
	  oa::I32 inDState = 32,
	  oa::I32 inExpand = 2,
	  oa::I32 inHeadDim = 16,
	  oa::I32 inNGroups = 1,
	  oa::F32 inRopeFraction = 0.5f,
	  bool inIsMimo = false,
	  oa::I32 inMimoRank = 1,
	  oa::F32 inDtMin = 0.001f,
	  oa::F32 inDtMax = 0.1f,
	  oa::F32 inDtInitFloor = 1e-4f,
	  oa::F32 inAFloor = 1e-4f,
	  bool inIsOutprojNorm = true
	);

	// Destructors.
	~EmpyrealmCore() = default;

	// Methods.
	oa::Matrix forward(const oa::Matrix& inInput) override;

	// Mixer + flat residual on pre-embedded [B, S, D] or [B*S, D] features.
	// Returns flat [B*seq, dModel].
	oa::Matrix forwardEmbedded(const oa::Matrix& inEmbedded);

	[[nodiscard]] oa::SharedPtr<EmpyrealmModule> mixer() const noexcept { return mixer_; }

	void resetState(oa::I32 inBatch);

private:
	oa::SharedPtr<oa::Module>       embed_;
	oa::SharedPtr<EmpyrealmModule> mixer_;
	oa::I32                       dModel_;
	bool                        isByteMode_ = false;
	// useFused_ removed 2026-06-18 — fused experiment deleted.
};

} // namespace oa
