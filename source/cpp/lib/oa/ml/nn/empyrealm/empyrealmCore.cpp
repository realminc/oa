// oa::EmpyrealmCore implementation.

#include "empyrealmModule.h"

#include <oa/core/fnMatrix.h>
#include <oa/ml/nn.h>

#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define NVTX_RANGE_PUSH(name) nvtxRangePushA(name)
#define NVTX_RANGE_POP() nvtxRangePop()
#else
#define NVTX_RANGE_PUSH(name) ((void)0)
#define NVTX_RANGE_POP() ((void)0)
#endif

oa::EmpyrealmCore::EmpyrealmCore(
	oa::I32 inVocabSize, oa::I32 inDModel,
	oa::I32 inDState, oa::I32 inExpand, oa::I32 inHeadDim,
	oa::I32 inNGroups, oa::F32 inRopeFraction, bool inIsMimo, oa::I32 inMimoRank,
	oa::F32 inDtMin, oa::F32 inDtMax, oa::F32 inDtInitFloor, oa::F32 inAFloor, bool inIsOutprojNorm)
	: dModel_(inDModel)
	, isByteMode_(true)
{
	embed_ = oa::makeShared<oa::Embedding>(inVocabSize, inDModel);

	// Full parametric (SISO or MIMO, all dt/rope/out-norm knobs exposed — no more hardcodes).
	mixer_ = oa::makeShared<oa::EmpyrealmModule>(
		inDModel, inDState, inExpand, inHeadDim,
		inNGroups, inRopeFraction, inIsMimo, inMimoRank,
		inDtMin, inDtMax, inDtInitFloor, inAFloor, inIsOutprojNorm
	);

	registerModule("embed", embed_);
	registerModule("mixer", mixer_);
}

oa::EmpyrealmCore::EmpyrealmCore(
	oa::SharedPtr<oa::Module> inEmbedModule, oa::I32 inDModel,
	oa::I32 inDState, oa::I32 inExpand, oa::I32 inHeadDim,
	oa::I32 inNGroups, oa::F32 inRopeFraction, bool inIsMimo, oa::I32 inMimoRank,
	oa::F32 inDtMin, oa::F32 inDtMax, oa::F32 inDtInitFloor, oa::F32 inAFloor, bool inIsOutprojNorm)
	: embed_(oa::move(inEmbedModule))
	, dModel_(inDModel)
	, isByteMode_(false)
{
	mixer_ = oa::makeShared<oa::EmpyrealmModule>(
		inDModel, inDState, inExpand, inHeadDim,
		inNGroups, inRopeFraction, inIsMimo, inMimoRank,
		inDtMin, inDtMax, inDtInitFloor, inAFloor, inIsOutprojNorm
	);

	registerModule("embed", embed_);
	registerModule("mixer", mixer_);
}

oa::EmpyrealmCore::EmpyrealmCore(
	oa::EmpyrealmCoreEmbeddedOnly, oa::I32 inDModel,
	oa::I32 inDState, oa::I32 inExpand, oa::I32 inHeadDim,
	oa::I32 inNGroups, oa::F32 inRopeFraction, bool inIsMimo, oa::I32 inMimoRank,
	oa::F32 inDtMin, oa::F32 inDtMax, oa::F32 inDtInitFloor, oa::F32 inAFloor, bool inIsOutprojNorm)
	: dModel_(inDModel)
	, isByteMode_(false)
{
	// No embed_ — caller drives the mixer via ForwardEmbedded with its own
	// [B, S, D] features. Only the mixer is registered, so no dead embed params
	// leak into the optimizer/checkpoints.
	mixer_ = oa::makeShared<oa::EmpyrealmModule>(
		inDModel, inDState, inExpand, inHeadDim,
		inNGroups, inRopeFraction, inIsMimo, inMimoRank,
		inDtMin, inDtMax, inDtInitFloor, inAFloor, inIsOutprojNorm
	);

	registerModule("mixer", mixer_);
}

oa::Matrix oa::EmpyrealmCore::forward(const oa::Matrix& inInput) {
	NVTX_RANGE_PUSH("EmpyrealmCore::forward");
	OA_ASSERT(embed_ and "oa::EmpyrealmCore::forward called in embedded-only mode; "
	                     "use forwardEmbedded with pre-embedded [B, S, D] features");
	oa::Matrix emb = embed_->forward(inInput);
	// Embedding/Gather always return a flat [B*S, D]. Recover the real
	// [batch, seqLen, D] structure from the token grid so the SSM scan runs B
	// independent length-S sequences (parallel) instead of one length-(B*S)
	// scan — the latter both serializes the scan (≈7x slower here) and bleeds
	// state across the B independent windows. Mirrors the Mamba3 reference
	// wiring. Only rank-2 token grids [B, S] carry this structure; anything
	// else is passed through to ForwardEmbedded's existing handling.
	if (inInput.rank() == 2) {
		emb = emb.reshape(oa::MatrixShape{inInput.size(0), inInput.size(1), dModel_});
	}
	auto ret = forwardEmbedded(emb);
	NVTX_RANGE_POP();
	return ret;
}

oa::Matrix oa::EmpyrealmCore::forwardEmbedded(const oa::Matrix& inEmbedded) {
	NVTX_RANGE_PUSH("EmpyrealmCore::forwardEmbedded");
	oa::I32 batch = 0;
	oa::I32 seqLen = 0;
	oa::Matrix emb3d;

	if (inEmbedded.rank() == 3) {
		batch = static_cast<oa::I32>(inEmbedded.size(0));
		seqLen = static_cast<oa::I32>(inEmbedded.size(1));
		emb3d = inEmbedded;
	} else {
		// Flat [B*S, D] case — return flat after residual.
		oa::I64 total = inEmbedded.size(0);
		batch = 1;
		seqLen = static_cast<oa::I32>(total);
		emb3d = inEmbedded.reshape(oa::MatrixShape{1, seqLen, dModel_});
	}

	auto m = mixer_;
	auto ef = emb3d.reshape(oa::MatrixShape{static_cast<oa::I64>(batch) * seqLen, dModel_});

	// empyrealm path: the mixer dispatches empyrealm* kernels (EmpyrealmDt,
	// EmpyrealmAdt, EmpyrealmSiso) — renamed copies of the Mamba3* kernels with
	// identical SPIR-V today, ready for future architecture-specific divergence.
	NVTX_RANGE_PUSH("EmpyrealmMixer");
	auto out3d = m->forward(emb3d);  // [B, S, D] — calls oa::EmpyrealmModule::forward
	NVTX_RANGE_POP();
	auto yf = out3d.reshape(oa::MatrixShape{static_cast<oa::I64>(batch) * seqLen, dModel_});
	auto mixed = yf + ef;
	NVTX_RANGE_POP();  // EmpyrealmCore::ForwardEmbedded
	return mixed;
}

void oa::EmpyrealmCore::resetState(oa::I32 inBatch) {
	if (mixer_) {
		mixer_->resetState(inBatch);
	}
}
