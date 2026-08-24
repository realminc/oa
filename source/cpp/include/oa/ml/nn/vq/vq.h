#pragma once

// VectorQuantizer / ResidualVectorQuantizer — VQ-VAE discrete bottleneck.
//
// General-purpose vector quantization (van den Oord 2017), not tied to any
// modality: a sequence/feature latent z_e is snapped to its nearest codebook
// entry to produce a discrete token, round-tripped through a decoder. This is the
// front-end for discrete-token generation (MoMask / MotionDreamer / T2M-GPT):
// tokenize a signal, then learn a generative prior over the tokens.
//
// Wraps the on-GPU primitives (no host round-trip, no new autograd surface):
//   oa::FnMatrix::VqAssign    — per-row nearest-code argmin + gather
//   oa::FnMatrix::detach      — straight-through estimator (forward z_q, grad → z_e)
//   oa::FnMatrix::VqEmaUpdate — EMA codebook update + dead-code reinit
//   oa::FnMatrix::Gather      — token id → code lookup (decode path)
//
// The codebook is NOT a gradient parameter: it tracks the encoder-latent
// distribution by EMA, which keeps it stable and stops it collapsing to a handful
// of live codes. Pair it with an RMS-normalized encoder latent (unit-RMS z_e) so
// the latent/codebook magnitudes can't run away. The EMA state is registered as
// persistent module buffers() so a trained codebook checkpoints/round-trips.
//
// Surfaced through <oa/ml/nn.h>.

#include <oa/ml/module.h>
#include <oa/ml/fnMatrix.h>

namespace oa {

struct VectorQuantizerConfig {
	oa::I32 numCodes   = 256;     // K — codebook entries
	oa::I32 codeDim    = 64;      // D — latent dimension
	oa::F32 commitBeta = 0.25f;   // commitment-loss weight (pulls encoder → its code)
	oa::F32 emaDecay   = 0.99f;   // EMA γ for the codebook
	oa::F32 emaEps     = 1e-5f;   // division floor
	oa::F32 deadThresh = 1.0f;    // revive codes whose EMA count falls below this
	bool  normCode   = false;   // rescale each codebook row to unit RMS (cosine VQ). Pair
	                            // with a unit-RMS z_e: turns L2 assignment into cosine and
	                            // stops EMA-shrunk codes collapsing onto the centroid.
};

struct VqResult {
	oa::Matrix quantized;    // [N, D] straight-through: z_e + detach(z_q - z_e)
	oa::Matrix idx;          // [N] int32 nearest-code indices
	oa::Matrix commitLoss;   // scalar β·MSE(z_e, sg z_q)
};

class VectorQuantizer : public oa::Module {
public:
	explicit VectorQuantizer(const VectorQuantizerConfig& inConfig);
	~VectorQuantizer() = default;

	// z_e [N, D] (RMS-normalized latents recommended) → quantized (STE) + code
	// indices + commitment loss. Records entirely on-GPU.
	[[nodiscard]] VqResult quantize(const oa::Matrix& inZe);

	// Module forward returns just the straight-through quantized tensor.
	oa::Matrix forward(const oa::Matrix& inZe) override { return quantize(inZe).quantized; }

	// EMA codebook update + dead-code reinit. call ONCE per step AFTER the optimizer
	// step, with this step's z_e and the idx returned by quantize(). in-place on the
	// codebook buffer (so it stays the same checkpointed buffers() entry).
	void emaUpdate(const oa::Matrix& inZe, const oa::Matrix& inIdx);

	// Token → latent: gather the code vectors for the given indices ([N] Int32, the
	// dtype VqAssign emits, or any generated ids). The inference-time inverse of the
	// nearest-code assignment — feed generated token ids straight back through it to
	// reconstruct z_q [N, D] for the decoder. Pure lookup, no STE.
	[[nodiscard]] oa::Matrix lookup(const oa::Matrix& inIdx) const;

	// Data-dependent init: seed the K codes from the K HIGHEST-NORM rows of inLatents
	// ([>= K, D] encoder outputs) — NOT the first K, which is degenerate for residual
	// VQ (rows a shallow level used → ~zero residual → a deeper codebook seeds all
	// zeros and dies). Highest-norm rows are never zero, so every level gets live,
	// distinct codes. writes the codebook IN-PLACE (copyFrom) so the registered
	// buffers() entry stays valid. Completes inLatents before reading.
	void seed(const oa::Matrix& inLatents);

	[[nodiscard]] oa::Matrix&       codebook()       noexcept { return codebook_; }
	[[nodiscard]] const oa::Matrix& codebook() const noexcept { return codebook_; }
	[[nodiscard]] const VectorQuantizerConfig& config() const noexcept { return config_; }

private:
	VectorQuantizerConfig config_;
	// persistent EMA state — registered as buffers() (checkpointed, never gradient
	// params: the codebook moves by EMA, not by backprop).
	oa::Matrix codebook_;      // [K, D]
	oa::Matrix embedSum_;      // [K, D] EMA cluster sums  m_k
	oa::Matrix clusterSize_;   // [K]    EMA cluster counts N_k
	oa::U32    emaStep_ = 0;   // increments per emaUpdate → per-step dead-code revival seed
};

// ─── residual Vector quantizer (RVQ) ────────────────────────────────────────
// Stacks Q VectorQuantizer levels: level 0 quantizes z_e, level q quantizes the
// residual left by levels 0..q-1. The quantized output is the SUM of all levels'
// codes, so Q tokens per frame give far finer reconstruction than one (K^Q effective
// codes) — the basis for MoMask / MotionDreamer residual-token generation, where a
// masked model later predicts these per-level tokens. Each level keeps its own EMA
// codebook; the straight-through estimate and commitment loss apply once on the total.
struct ResidualVqResult {
	oa::Matrix        quantized;   // [N, D] straight-through total: z_e + detach(Σzq - z_e)
	oa::Vec<oa::Matrix> idx;         // Q × [N] int32 — per-level code indices (the tokens)
	oa::Vec<oa::Matrix> residuals;   // Q × [N, D] — per-level input residual (for the EMA update)
	oa::Matrix        commitLoss;  // scalar β·MSE(z_e, sg Σzq)
};

class ResidualVectorQuantizer : public oa::Module {
public:
	ResidualVectorQuantizer(const VectorQuantizerConfig& inConfig, oa::I32 inNumLevels);
	~ResidualVectorQuantizer() = default;

	// z_e [N, D] → straight-through total quantization + per-level tokens + per-level
	// residuals (kept for emaUpdate) + commitment loss. Records entirely on-GPU.
	[[nodiscard]] ResidualVqResult quantize(const oa::Matrix& inZe);
	oa::Matrix forward(const oa::Matrix& inZe) override { return quantize(inZe).quantized; }

	// Per-level EMA codebook update; call once per step AFTER the optimizer step with
	// the result returned by this step's quantize().
	void emaUpdate(const ResidualVqResult& inResult);

	// Token → latent: sum each level's gathered code vectors. The inference-time
	// inverse of quantize for the SUMMED RVQ output — pass per-level generated token
	// ids (one [N] Int32 per level, shallow→deep) → z_q [N, D] for the decoder. inIdx
	// may carry fewer than numLevels() levels (e.g. a model that only generates level
	// 0); only the supplied levels are summed.
	[[nodiscard]] oa::Matrix lookup(const oa::Vec<oa::Matrix>& inIdx) const;

	// Greedy data-dependent seed: seed level 0 from the latents, then each deeper level
	// from the running residual under the already-seeded shallower levels.
	void seed(const oa::Matrix& inLatents);

	[[nodiscard]] oa::I32 numLevels() const noexcept { return static_cast<oa::I32>(levels_.size()); }
	[[nodiscard]] VectorQuantizer& level(oa::I32 q) { return *levels_[static_cast<oa::Usize>(q)]; }

private:
	VectorQuantizerConfig               config_;
	oa::Vec<oa::SharedPtr<VectorQuantizer>> levels_;
};

} // namespace oa
