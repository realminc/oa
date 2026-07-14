// OaEmpyrealmCore implementation.

#include "EmpyrealmModule.h"

#include <Oa/Core/FnMatrix.h>
#include <Oa/Ml/Nn.h>
#include <Oa/Ml/Autograd.h>

#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define NVTX_RANGE_PUSH(name) nvtxRangePushA(name)
#define NVTX_RANGE_POP() nvtxRangePop()
#else
#define NVTX_RANGE_PUSH(name) ((void)0)
#define NVTX_RANGE_POP() ((void)0)
#endif

OaEmpyrealmCore::OaEmpyrealmCore(
	OaI32 InVocabSize, OaI32 InDModel,
	OaI32 InDState, OaI32 InExpand, OaI32 InHeadDim,
	OaI32 InNGroups, OaF32 InRopeFraction, bool InIsMimo, OaI32 InMimoRank,
	OaF32 InDtMin, OaF32 InDtMax, OaF32 InDtInitFloor, OaF32 InAFloor, bool InIsOutprojNorm)
	: DModel_(InDModel)
	, IsByteMode_(true)
{
	Embed_ = OaMakeSharedPtr<OaEmbedding>(InVocabSize, InDModel);

	// Full parametric (SISO or MIMO, all dt/rope/out-norm knobs exposed — no more hardcodes).
	Mixer_ = OaMakeSharedPtr<OaEmpyrealmModule>(
		InDModel, InDState, InExpand, InHeadDim,
		InNGroups, InRopeFraction, InIsMimo, InMimoRank,
		InDtMin, InDtMax, InDtInitFloor, InAFloor, InIsOutprojNorm
	);

	RegisterModule("embed", Embed_);
	RegisterModule("mixer", Mixer_);
}

OaEmpyrealmCore::OaEmpyrealmCore(
	OaSharedPtr<OaModule> InEmbedModule, OaI32 InDModel,
	OaI32 InDState, OaI32 InExpand, OaI32 InHeadDim,
	OaI32 InNGroups, OaF32 InRopeFraction, bool InIsMimo, OaI32 InMimoRank,
	OaF32 InDtMin, OaF32 InDtMax, OaF32 InDtInitFloor, OaF32 InAFloor, bool InIsOutprojNorm)
	: Embed_(std::move(InEmbedModule))
	, DModel_(InDModel)
	, IsByteMode_(false)
{
	Mixer_ = OaMakeSharedPtr<OaEmpyrealmModule>(
		InDModel, InDState, InExpand, InHeadDim,
		InNGroups, InRopeFraction, InIsMimo, InMimoRank,
		InDtMin, InDtMax, InDtInitFloor, InAFloor, InIsOutprojNorm
	);

	RegisterModule("embed", Embed_);
	RegisterModule("mixer", Mixer_);
}

OaEmpyrealmCore::OaEmpyrealmCore(
	OaEmpyrealmCoreEmbeddedOnly, OaI32 InDModel,
	OaI32 InDState, OaI32 InExpand, OaI32 InHeadDim,
	OaI32 InNGroups, OaF32 InRopeFraction, bool InIsMimo, OaI32 InMimoRank,
	OaF32 InDtMin, OaF32 InDtMax, OaF32 InDtInitFloor, OaF32 InAFloor, bool InIsOutprojNorm)
	: DModel_(InDModel)
	, IsByteMode_(false)
{
	// No Embed_ — caller drives the mixer via ForwardEmbedded with its own
	// [B, S, D] features. Only the mixer is registered, so no dead embed params
	// leak into the optimizer/checkpoints.
	Mixer_ = OaMakeSharedPtr<OaEmpyrealmModule>(
		InDModel, InDState, InExpand, InHeadDim,
		InNGroups, InRopeFraction, InIsMimo, InMimoRank,
		InDtMin, InDtMax, InDtInitFloor, InAFloor, InIsOutprojNorm
	);

	RegisterModule("mixer", Mixer_);
}

OaMatrix OaEmpyrealmCore::Forward(const OaMatrix& InInput) {
	NVTX_RANGE_PUSH("EmpyrealmCore::Forward");
	OA_ASSERT(Embed_ and "OaEmpyrealmCore::Forward called in embedded-only mode; "
	                     "use ForwardEmbedded with pre-embedded [B, S, D] features");
	OaMatrix emb = Embed_->Forward(InInput);
	// Embedding/Gather always return a flat [B*S, D]. Recover the real
	// [batch, seqLen, D] structure from the token grid so the SSM scan runs B
	// independent length-S sequences (parallel) instead of one length-(B*S)
	// scan — the latter both serializes the scan (≈7x slower here) and bleeds
	// state across the B independent windows. Mirrors the Mamba3 reference
	// wiring. Only rank-2 token grids [B, S] carry this structure; anything
	// else is passed through to ForwardEmbedded's existing handling.
	if (InInput.Rank() == 2) {
		emb = emb.Reshape(OaMatrixShape{InInput.Size(0), InInput.Size(1), DModel_});
	}
	auto ret = ForwardEmbedded(emb);
	NVTX_RANGE_POP();
	return ret;
}

OaMatrix OaEmpyrealmCore::ForwardEmbedded(const OaMatrix& InEmbedded) {
	NVTX_RANGE_PUSH("EmpyrealmCore::ForwardEmbedded");
	OaI32 batch = 0;
	OaI32 seqLen = 0;
	OaMatrix emb3d;

	if (InEmbedded.Rank() == 3) {
		batch = static_cast<OaI32>(InEmbedded.Size(0));
		seqLen = static_cast<OaI32>(InEmbedded.Size(1));
		emb3d = InEmbedded;
	} else {
		// Flat [B*S, D] case — return flat after residual.
		OaI64 total = InEmbedded.Size(0);
		batch = 1;
		seqLen = static_cast<OaI32>(total);
		emb3d = InEmbedded.Reshape(OaMatrixShape{1, seqLen, DModel_});
	}

	auto m = Mixer_;
	auto ef = emb3d.Reshape(OaMatrixShape{static_cast<OaI64>(batch) * seqLen, DModel_});

	// Empyrealm path: the mixer dispatches Empyrealm* kernels (EmpyrealmDt,
	// EmpyrealmAdt, EmpyrealmSiso) — renamed copies of the Mamba3* kernels with
	// identical SPIR-V today, ready for future architecture-specific divergence.
	NVTX_RANGE_PUSH("EmpyrealmMixer");
	auto out3d = m->Forward(emb3d);  // [B, S, D]
	NVTX_RANGE_POP();
	auto yf = out3d.Reshape(OaMatrixShape{static_cast<OaI64>(batch) * seqLen, DModel_});
	auto mixed = yf + ef;
	NVTX_RANGE_POP();  // EmpyrealmCore::ForwardEmbedded
	return mixed;
}

void OaEmpyrealmCore::ResetState(OaI32 InBatch) {
	if (Mixer_) {
		Mixer_->ResetState(InBatch);
	}
}
