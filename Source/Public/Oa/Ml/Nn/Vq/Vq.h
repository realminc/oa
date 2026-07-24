#pragma once

// OaVectorQuantizer / OaResidualVectorQuantizer — VQ-VAE discrete bottleneck.
//
// General-purpose vector quantization (van den Oord 2017), not tied to any
// modality: a sequence/feature latent z_e is snapped to its nearest codebook
// entry to produce a discrete token, round-tripped through a decoder. This is the
// front-end for discrete-token generation (MoMask / MotionDreamer / T2M-GPT):
// tokenize a signal, then learn a generative prior over the tokens.
//
// Wraps the on-GPU primitives (no host round-trip, no new autograd surface):
//   OaFnMatrix::VqAssign    — per-row nearest-code argmin + gather
//   OaFnMatrix::Detach      — straight-through estimator (forward z_q, grad → z_e)
//   OaFnMatrix::VqEmaUpdate — EMA codebook update + dead-code reinit
//   OaFnMatrix::Gather      — token id → code lookup (decode path)
//
// The codebook is NOT a gradient parameter: it tracks the encoder-latent
// distribution by EMA, which keeps it stable and stops it collapsing to a handful
// of live codes. Pair it with an RMS-normalized encoder latent (unit-RMS z_e) so
// the latent/codebook magnitudes can't run away. The EMA state is registered as
// persistent module Buffers() so a trained codebook checkpoints/round-trips.
//
// Surfaced via <Oa/Ml/Nn.h>.

#include <Oa/Ml/Module.h>
#include <Oa/Ml/FnMatrix.h>

struct OaVectorQuantizerConfig {
	OaI32 NumCodes   = 256;     // K — codebook entries
	OaI32 CodeDim    = 64;      // D — latent dimension
	OaF32 CommitBeta = 0.25f;   // commitment-loss weight (pulls encoder → its code)
	OaF32 EmaDecay   = 0.99f;   // EMA γ for the codebook
	OaF32 EmaEps     = 1e-5f;   // division floor
	OaF32 DeadThresh = 1.0f;    // revive codes whose EMA count falls below this
	bool  NormCode   = false;   // rescale each codebook row to unit RMS (cosine VQ). Pair
	                            // with a unit-RMS z_e: turns L2 assignment into cosine and
	                            // stops EMA-shrunk codes collapsing onto the centroid.
};

struct OaVqResult {
	OaMatrix Quantized;    // [N, D] straight-through: z_e + Detach(z_q - z_e)
	OaMatrix Idx;          // [N] int32 nearest-code indices
	OaMatrix CommitLoss;   // scalar β·MSE(z_e, sg z_q)
};

class OaVectorQuantizer : public OaModule {
public:
	explicit OaVectorQuantizer(const OaVectorQuantizerConfig& InConfig);
	~OaVectorQuantizer() = default;

	// z_e [N, D] (RMS-normalized latents recommended) → quantized (STE) + code
	// indices + commitment loss. Records entirely on-GPU.
	[[nodiscard]] OaVqResult Quantize(const OaMatrix& InZe);

	// Module Forward returns just the straight-through quantized tensor.
	OaMatrix Forward(const OaMatrix& InZe) override { return Quantize(InZe).Quantized; }

	// EMA codebook update + dead-code reinit. Call ONCE per step AFTER the optimizer
	// step, with this step's z_e and the Idx returned by Quantize(). In-place on the
	// codebook buffer (so it stays the same checkpointed Buffers() entry).
	void EmaUpdate(const OaMatrix& InZe, const OaMatrix& InIdx);

	// Token → latent: gather the code vectors for the given indices ([N] Int32, the
	// dtype VqAssign emits, or any generated ids). The inference-time inverse of the
	// nearest-code assignment — feed generated token ids straight back through it to
	// reconstruct z_q [N, D] for the decoder. Pure lookup, no STE.
	[[nodiscard]] OaMatrix Lookup(const OaMatrix& InIdx) const;

	// Data-dependent init: seed the K codes from the K HIGHEST-NORM rows of InLatents
	// ([>= K, D] encoder outputs) — NOT the first K, which is degenerate for residual
	// VQ (rows a shallow level used → ~zero residual → a deeper codebook seeds all
	// zeros and dies). Highest-norm rows are never zero, so every level gets live,
	// distinct codes. Writes the codebook IN-PLACE (CopyFrom) so the registered
	// Buffers() entry stays valid. Realizes InLatents (Execute/Sync) before reading.
	void Seed(const OaMatrix& InLatents);

	[[nodiscard]] OaMatrix&       Codebook()       noexcept { return Codebook_; }
	[[nodiscard]] const OaMatrix& Codebook() const noexcept { return Codebook_; }
	[[nodiscard]] const OaVectorQuantizerConfig& Config() const noexcept { return Config_; }

private:
	OaVectorQuantizerConfig Config_;
	// Persistent EMA state — registered as Buffers() (checkpointed, never gradient
	// params: the codebook moves by EMA, not by backprop).
	OaMatrix Codebook_;      // [K, D]
	OaMatrix EmbedSum_;      // [K, D] EMA cluster sums  m_k
	OaMatrix ClusterSize_;   // [K]    EMA cluster counts N_k
	OaU32    EmaStep_ = 0;   // increments per EmaUpdate → per-step dead-code revival seed
};

// ─── Residual Vector Quantizer (RVQ) ────────────────────────────────────────
// Stacks Q OaVectorQuantizer levels: level 0 quantizes z_e, level q quantizes the
// residual left by levels 0..q-1. The quantized output is the SUM of all levels'
// codes, so Q tokens per frame give far finer reconstruction than one (K^Q effective
// codes) — the basis for MoMask / MotionDreamer residual-token generation, where a
// masked model later predicts these per-level tokens. Each level keeps its own EMA
// codebook; the straight-through estimate and commitment loss apply once on the total.
struct OaResidualVqResult {
	OaMatrix        Quantized;   // [N, D] straight-through total: z_e + Detach(Σzq - z_e)
	OaVec<OaMatrix> Idx;         // Q × [N] int32 — per-level code indices (the tokens)
	OaVec<OaMatrix> Residuals;   // Q × [N, D] — per-level input residual (for the EMA update)
	OaMatrix        CommitLoss;  // scalar β·MSE(z_e, sg Σzq)
};

class OaResidualVectorQuantizer : public OaModule {
public:
	OaResidualVectorQuantizer(const OaVectorQuantizerConfig& InConfig, OaI32 InNumLevels);
	~OaResidualVectorQuantizer() = default;

	// z_e [N, D] → straight-through total quantization + per-level tokens + per-level
	// residuals (kept for EmaUpdate) + commitment loss. Records entirely on-GPU.
	[[nodiscard]] OaResidualVqResult Quantize(const OaMatrix& InZe);
	OaMatrix Forward(const OaMatrix& InZe) override { return Quantize(InZe).Quantized; }

	// Per-level EMA codebook update; call once per step AFTER the optimizer step with
	// the Result returned by this step's Quantize().
	void EmaUpdate(const OaResidualVqResult& InResult);

	// Token → latent: sum each level's gathered code vectors. The inference-time
	// inverse of Quantize for the SUMMED RVQ output — pass per-level generated token
	// ids (one [N] Int32 per level, shallow→deep) → z_q [N, D] for the decoder. InIdx
	// may carry fewer than NumLevels() levels (e.g. a model that only generates level
	// 0); only the supplied levels are summed.
	[[nodiscard]] OaMatrix Lookup(const OaVec<OaMatrix>& InIdx) const;

	// Greedy data-dependent seed: seed level 0 from the latents, then each deeper level
	// from the running residual under the already-seeded shallower levels.
	void Seed(const OaMatrix& InLatents);

	[[nodiscard]] OaI32 NumLevels() const noexcept { return static_cast<OaI32>(Levels_.Size()); }
	[[nodiscard]] OaVectorQuantizer& Level(OaI32 q) { return *Levels_[static_cast<OaUsize>(q)]; }

private:
	OaVectorQuantizerConfig               Config_;
	OaVec<OaSharedPtr<OaVectorQuantizer>> Levels_;
};
