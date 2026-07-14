#pragma once

// OaAlmTokenizerAg — Stage 1 VQ-VAE (faithful T2M-GPT / MotionGPT tokenizer).
//
// Verified against pytorch-comparison/MotionGPT/mGPT/archs/mgpt_vq.py. The defining
// MotionGPT property is TEMPORAL DOWNSAMPLING: a strided 1-D conv stack maps T motion
// frames to T / 2^DownT discrete tokens (DownT=3 -> 8x), so each token spans a short
// motion phrase, not one frame. That is what makes the downstream token LM tractable
// (an 8x shorter sequence) and the tokens semantic.
//
//   x [B, T, InputDim]                       (body channels; root carried separately)
//     -> transpose -> [B, InputDim, T]       (channels-first for Conv1d)
//     -> EncIn  Conv1d(InputDim->W, 3,1,1) ReLU
//     -> DownT x { Conv1d(W->W, 4, stride 2, pad 1) ; Resnet1D(W, Depth) }  2x each
//     -> EncOut Conv1d(W->CodeDim, 3,1,1)
//     -> transpose -> [B, T/8, CodeDim] -> flatten [N, CodeDim] -> RMSNorm (unit-RMS z_e)
//     -> OaResidualVectorQuantizer(NumLevels=1, EMA codebook)   the discrete bottleneck
//     -> DecIn  Conv1d(CodeDim->W, 3,1,1) ReLU
//     -> DecT  x { Resnet1D(W, Depth) ; ConvTranspose1d(W->W, 4, stride 2, pad 1) }  2x each
//     -> DecMid Conv1d(W->W, 3,1,1) ReLU
//     -> DecOut Conv1d(W->InputDim, 3,1,1)
//     -> transpose -> [B, T, InputDim]
//
// Reuses hardened OA primitives: OaConv1d (strided, im2col+GEMM w/ composed autograd),
// OaConvTranspose1d (learnable 2x upsample = adjoint of Conv1d), and
// OaResidualVectorQuantizer (EMA codebook, checkpointed Buffers, dead-code reset).
//
// Deviations from the reference, documented on purpose (v1):
//   - Single-level VQ (NumLevels=1) — matches mgpt_vq's single QuantizeEMAReset.
//   - Dilation=3 on 3-tap residual convs (matches Python baseline Resnet1D).
//   - Decoder upsamples with learnable OaConvTranspose1d (stride 2) rather than the
//     reference's fixed nearest Upsample + Conv — strictly more expressive, same 2x.
//   - RMS-normalized latent before the codebook (MoMask trick) bounds the
//     latent/codebook feedback; the reference relies on plain magnitude.

#include <Ml/Nn/Alm/AlmConfig.h>
#include <Oa/Ml.h>
#include <Oa/Ml/Nn.h>            // OaResidualVectorQuantizer, OaResidualVqResult

class OaConv1d;
class OaConvTranspose1d;
class OaLayerNorm;

// VQ-VAE tokenizer module
class OaAlmTokenizerAg : public OaModule {
public:
	explicit OaAlmTokenizerAg(const OaAlmTokenizerConfig& InConfig);
	~OaAlmTokenizerAg() = default;

	OaMatrix Forward(const OaMatrix&) override { return {}; }   // driven via Encode/Decode

	// [B,T,InputDim] → z_e [B·(T/Factor), CodeDim], unit-RMS per row. T must be a
	// multiple of DownsampleFactor().
	[[nodiscard]] OaMatrix Encode(const OaMatrix& InX, OaI32 InBatch, OaI32 InSeqLen);
	// z_e [N, CodeDim] → straight-through quantization + level tokens + commit loss.
	[[nodiscard]] OaResidualVqResult Quantize(const OaMatrix& InZe) { return Rvq_->Quantize(InZe); }
	// z_q [N, CodeDim] → reconstruction [B·(InTokLen·Factor), InputDim]. InTokLen is the
	// per-batch token length (T/Factor); output frame length is InTokLen·Factor.
	[[nodiscard]] OaMatrix Decode(const OaMatrix& InZq, OaI32 InBatch, OaI32 InTokLen);

	// Per-step EMA codebook update (after opt.Step, with this step's Quantize result).
	void EmaUpdate(const OaResidualVqResult& InResult) { Rvq_->EmaUpdate(InResult); }
	// Data-dependent codebook seed from a warm batch of latents (needs ≥ NumCodes rows).
	void Seed(const OaMatrix& InLatents) { Rvq_->Seed(InLatents); }

	// Inference helpers (no STE): sequence → per-level token id streams, and back.
	[[nodiscard]] OaVec<OaMatrix> Tokenize(const OaMatrix& InX, OaI32 InBatch, OaI32 InSeqLen);
	[[nodiscard]] OaMatrix Detokenize(const OaVec<OaMatrix>& InIdx, OaI32 InBatch, OaI32 InTokLen);

	[[nodiscard]] OaI32 DownsampleFactor() const noexcept { return Factor_; }
	[[nodiscard]] OaResidualVectorQuantizer& Rvq() noexcept { return *Rvq_; }
	[[nodiscard]] const OaAlmTokenizerConfig& Config() const noexcept { return Config_; }

private:
	// Pre-norm residual block: x + ConvB(ReLU(ConvA(ReLU(LN(x))))). Each LN is a per-site
	// learnable-affine channel norm (the reference's norm="GN" path). Parameter-FREE norm
	// here makes pre-norm scale a degenerate flat direction → the output layer drifts and
	// recon explodes; the learnable scale removes that degeneracy. Consumes Depth LNs from
	// InLn starting at InLnCursor (advanced in place).
	[[nodiscard]] OaMatrix ResStack(const OaVec<OaSharedPtr<OaConv1d>>& InConvs,
		const OaVec<OaSharedPtr<OaLayerNorm>>& InLn, OaUsize& InLnCursor, const OaMatrix& InH) const;
	// Learnable-affine channel norm on a [B,C,T] activation (LayerNorm over C=Width):
	// transpose C→last, LN(weight+bias), transpose back.
	[[nodiscard]] OaMatrix NormC(const OaSharedPtr<OaLayerNorm>& InLn, const OaMatrix& InH) const;
	// Fused ChannelNorm + ReLU. 1 dispatch instead of 2.
	[[nodiscard]] OaMatrix NormCRelu(const OaSharedPtr<OaLayerNorm>& InLn, const OaMatrix& InH) const;
	// Fused Conv1d + ReLU via OaFnMatrix::Conv1dReluGemm (im2col + matmul + relu).
	[[nodiscard]] OaMatrix ConvRelu(const OaSharedPtr<OaConv1d>& InConv, const OaMatrix& InH) const;
	// Bare Conv1d forward (no activation) via OaFnMatrix::Conv1dGemm (im2col + matmul).
	[[nodiscard]] OaMatrix ConvFwd(const OaSharedPtr<OaConv1d>& InConv, const OaMatrix& InH) const;

	OaAlmTokenizerConfig Config_;
	OaI32 Factor_ = 8;   // 2^DownT
	OaSharedPtr<OaConv1d> EncIn_, EncOut_, DecIn_, DecMid_, DecOut_;
	OaVec<OaSharedPtr<OaConv1d>>          EncDown_;  // DownT strided downsample convs
	OaVec<OaSharedPtr<OaConvTranspose1d>> DecUp_;    // DownT learnable 2× upsample convs
	OaVec<OaSharedPtr<OaConv1d>> EncRes_;    // DownT stages × 2·Depth convs (flat)
	OaVec<OaSharedPtr<OaConv1d>> DecRes_;
	OaVec<OaSharedPtr<OaLayerNorm>> EncLn_;  // learnable-affine norms, encoder call order
	OaVec<OaSharedPtr<OaLayerNorm>> DecLn_;  // learnable-affine norms, decoder call order
	OaSharedPtr<OaResidualVectorQuantizer> Rvq_;
};
