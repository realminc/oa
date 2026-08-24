#pragma once

// AlmTokenizerAg — stage 1 VQ-VAE (faithful T2M-GPT / MotionGPT tokenizer).
//
// Verified against pytorch-comparison/MotionGPT/mGPT/archs/mgpt_vq.py. The defining
// MotionGPT property is TEMPORAL DOWNSAMPLING: a strided 1-D conv stack maps T motion
// frames to T / 2^downT discrete tokens (downT=3 -> 8x), so each token spans a short
// motion phrase, not one frame. That is what makes the downstream token LM tractable
// (an 8x shorter sequence) and the tokens semantic.
//
//   x [B, T, inputDim]                       (body channels; root carried separately)
//     -> transpose -> [B, inputDim, T]       (channels-first for Conv1d)
//     -> EncIn  Conv1d(inputDim->w, 3,1,1) ReLU
//     -> downT x { Conv1d(W->w, 4, stride 2, pad 1) ; resnet1D(W, depth) }  2x each
//     -> EncOut Conv1d(W->codeDim, 3,1,1)
//     -> transpose -> [B, T/8, codeDim] -> flatten [N, codeDim] -> rMSNorm (unit-RMS z_e)
//     -> oa::ResidualVectorQuantizer(NumLevels=1, EMA codebook)   the discrete bottleneck
//     -> DecIn  Conv1d(codeDim->w, 3,1,1) ReLU
//     -> DecT  x { resnet1D(W, depth) ; ConvTranspose1d(W->w, 4, stride 2, pad 1) }  2x each
//     -> DecMid Conv1d(W->w, 3,1,1) ReLU
//     -> DecOut Conv1d(W->inputDim, 3,1,1)
//     -> transpose -> [B, T, inputDim]
//
// Reuses hardened OA primitives: oa::Conv1d (strided, im2col+GEMM w/ composed autograd),
// oa::ConvTranspose1d (learnable 2x upsample = adjoint of Conv1d), and
// oa::ResidualVectorQuantizer (EMA codebook, checkpointed buffers, dead-code reset).
//
// Deviations from the reference, documented on purpose (v1):
//   - Single-level VQ (NumLevels=1) — matches mgpt_vq's single QuantizeEMAReset.
//   - dilation=3 on 3-tap residual convs (matches Python baseline Resnet1D).
//   - Decoder upsamples with learnable oa::ConvTranspose1d (stride 2) rather than the
//     reference's fixed nearest Upsample + Conv — strictly more expressive, same 2x.
//   - RMS-normalized latent before the codebook (MoMask trick) bounds the
//     latent/codebook feedback; the reference relies on plain magnitude.

#include <ml/nn/alm/almConfig.h>
#include <oa/ml.h>
#include <oa/ml/nn.h>            // oa::ResidualVectorQuantizer, oa::ResidualVqResult

namespace oa {
class Conv1d;
class ConvTranspose1d;
class LayerNorm;

// VQ-VAE tokenizer module
class AlmTokenizerAg : public oa::Module {
public:
	explicit AlmTokenizerAg(const AlmTokenizerConfig& inConfig);
	~AlmTokenizerAg() = default;

	oa::Matrix forward(const oa::Matrix&) override { return {}; }   // driven via Encode/Decode

	// [B,T,inputDim] → z_e [B·(T/Factor), codeDim], unit-RMS per row. T must be a
	// multiple of downsampleFactor().
	[[nodiscard]] oa::Matrix encode(const oa::Matrix& inX, oa::I32 inBatch, oa::I32 inSeqLen);
	// z_e [N, codeDim] → straight-through quantization + level tokens + commit loss.
	[[nodiscard]] oa::ResidualVqResult quantize(const oa::Matrix& inZe) { return rvq_->quantize(inZe); }
	// z_q [N, codeDim] → reconstruction [B·(inTokLen·Factor), inputDim]. inTokLen is the
	// per-batch token length (T/Factor); output frame length is inTokLen·Factor.
	[[nodiscard]] oa::Matrix decode(const oa::Matrix& inZq, oa::I32 inBatch, oa::I32 inTokLen);

	// Per-step EMA codebook update (after opt.step, with this step's quantize result).
	void emaUpdate(const oa::ResidualVqResult& inResult) { rvq_->emaUpdate(inResult); }
	// Data-dependent codebook seed from a warm batch of latents (needs ≥ numCodes rows).
	void seed(const oa::Matrix& inLatents) { rvq_->seed(inLatents); }

	// Inference helpers (no STE): sequence → per-level token id streams, and back.
	[[nodiscard]] oa::Vec<oa::Matrix> tokenize(const oa::Matrix& inX, oa::I32 inBatch, oa::I32 inSeqLen);
	[[nodiscard]] oa::Matrix detokenize(const oa::Vec<oa::Matrix>& inIdx, oa::I32 inBatch, oa::I32 inTokLen);

	[[nodiscard]] oa::I32 downsampleFactor() const noexcept { return factor_; }
	[[nodiscard]] oa::ResidualVectorQuantizer& rvq() noexcept { return *rvq_; }
	[[nodiscard]] const AlmTokenizerConfig& config() const noexcept { return config_; }

private:
	// Pre-norm residual block: x + convB(reLU(convA(reLU(LN(x))))). Each LN is a per-site
	// learnable-affine channel norm (the reference's norm="GN" path). Parameter-FREE norm
	// here makes pre-norm scale a degenerate flat direction → the output layer drifts and
	// recon explodes; the learnable scale removes that degeneracy. Consumes depth LNs from
	// inLn starting at inLnCursor (advanced in place).
	[[nodiscard]] oa::Matrix resStack(const oa::Vec<oa::SharedPtr<oa::Conv1d>>& inConvs,
		const oa::Vec<oa::SharedPtr<oa::LayerNorm>>& inLn, oa::Usize& inLnCursor, const oa::Matrix& inH) const;
	// Learnable-affine channel norm on a [B,C,T] activation (LayerNorm over C=width):
	// transpose C→last, LN(weight+bias), transpose back.
	[[nodiscard]] oa::Matrix normC(const oa::SharedPtr<oa::LayerNorm>& inLn, const oa::Matrix& inH) const;
	// Fused ChannelNorm + ReLU. 1 dispatch instead of 2.
	[[nodiscard]] oa::Matrix normCRelu(const oa::SharedPtr<oa::LayerNorm>& inLn, const oa::Matrix& inH) const;
	// Fused Conv1d + ReLU via oa::FnMatrix::conv1dReluGemm (im2col + matmul + relu).
	[[nodiscard]] oa::Matrix convRelu(const oa::SharedPtr<oa::Conv1d>& inConv, const oa::Matrix& inH) const;
	// Bare Conv1d forward (no activation) via oa::FnMatrix::conv1dGemm (im2col + matmul).
	[[nodiscard]] oa::Matrix convFwd(const oa::SharedPtr<oa::Conv1d>& inConv, const oa::Matrix& inH) const;

	AlmTokenizerConfig config_;
	oa::I32 factor_ = 8;   // 2^downT
	oa::SharedPtr<oa::Conv1d> encIn_, encOut_, decIn_, decMid_, decOut_;
	oa::Vec<oa::SharedPtr<oa::Conv1d>>          encDown_;  // downT strided downsample convs
	oa::Vec<oa::SharedPtr<oa::ConvTranspose1d>> decUp_;    // downT learnable 2× upsample convs
	oa::Vec<oa::SharedPtr<oa::Conv1d>> encRes_;    // downT stages × 2·depth convs (flat)
	oa::Vec<oa::SharedPtr<oa::Conv1d>> decRes_;
	oa::Vec<oa::SharedPtr<oa::LayerNorm>> encLn_;  // learnable-affine norms, encoder call order
	oa::Vec<oa::SharedPtr<oa::LayerNorm>> decLn_;  // learnable-affine norms, decoder call order
	oa::SharedPtr<oa::ResidualVectorQuantizer> rvq_;
};

} // namespace oa
