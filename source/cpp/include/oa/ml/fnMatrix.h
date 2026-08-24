#pragma once

#include <oa/core/matrix.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/matrixShape.h>
#include <oa/core/status.h>
#include <oa/ml/fnLoss.h>
#include <oa/ml/quantMatrix.h>

// oa::FnMatrix ML extension — neural network operations.
// Extends oa::FnMatrix (core/fnMatrix.h) with ML-specific static methods.
// include both <oa/core/fnMatrix.h> and <oa/ml/fnMatrix.h> to access all methods.

namespace oa {

// --- ML result structs (domain types, not bound to FnMatrix) ---

struct LinearWeightBiasBwdResult {
	Matrix gradWeight;
	Matrix gradBias;
};

struct LayerNormBwdResult {
	Matrix dX;
	Matrix dWeight;
	Matrix dBias;
};

struct RmsNormBwdResult {
	Matrix dX;
	Matrix dWeight;
};

struct RmsNormGatedBwdResult {
	Matrix dX;
	Matrix dWeight;
	Matrix dBias;
	Matrix dZ;
};

struct ChannelNormBwdResult {
	Matrix dx;
	Matrix dWeight;
	Matrix dBias;
};

struct SwigluBwdResult {
	Matrix dGate;
	Matrix dUp;
};

struct GruCellPointwiseBwdResult {
	Matrix dGatesI;   // [B, 3H] grad w.r.t. the input projection
	Matrix dGatesH;   // [B, 3H] grad w.r.t. the hidden projection
	Matrix dHidden;   // [B, H]  grad w.r.t. the previous hidden state
};

struct GruScanResult {
	Matrix out;    // [B, S, H] hidden states h^{(1..s)}
	Matrix hPrev;  // [B, S, H] previous hidden states h^{(0..s-1)} (saved for bwd)
};

struct GruScanBwdResult {
	Matrix dGatesI;  // [B*S, 3H]
	Matrix dGatesH;  // [B*S, 3H]
};

struct RnnCellPointwiseBwdResult {
	Matrix dGatesI;   // [B, H] grad w.r.t. the input projection
	Matrix dGatesH;   // [B, H] grad w.r.t. the hidden projection
};

struct RnnScanResult {
	Matrix out;    // [B, S, H] hidden states h^{(1..s)}
	Matrix hPrev;  // [B, S, H] previous hidden states h^{(0..s-1)} (saved for bwd)
};

struct RnnScanBwdResult {
	Matrix dGatesI;  // [B*S, H]
	Matrix dGatesH;  // [B*S, H]
};

struct VqAssignResult {
	Matrix idx;   // [N]    int32 — nearest codebook index per row
	Matrix zq;    // [N, D] float — the gathered winning code per row
};

struct FlashAttentionBwdResult {
	Matrix dq;
	Matrix dk;
	Matrix dv;
};

struct Conv1dBwdWeightResult {
	Matrix gradWeight;
	Matrix gradBias;
};

struct Conv2dBwdWeightResult {
	Matrix gradWeight;
	Matrix gradBias;
};

struct ConvTranspose2dBwdWeightResult {
	Matrix gradWeight;
	Matrix gradBias;
};

struct MaxPool2dResult {
	Matrix out;      // [N, C, H_out, W_out] pooled output
	Matrix indices;  // [N, C, H_out, W_out] uint32 argmax indices (for backward pass)
};

struct BatchNorm2dBwdResult {
	Matrix dx;
	Matrix dGamma;
	Matrix dBias;
};

} // namespace oa

// ML operations reopen the same stateless oa::FnMatrix namespace as Core.
// Namespace extension is ordinary C++; it needs no class inheritance, alias,
// or forwarding layer.
namespace oa {

namespace FnMatrix {

// --- GRU ---

/// gruCellPointwise: fused GRU pointwise forward.
///   r = sigmoid(gatesI[r] + gatesH[r])
///   z = sigmoid(gatesI[z] + gatesH[z])
///   n = tanh(gatesI[n] + r * gatesH[n])
///   h_new = (1 - z) * n + z * hPrev
/// @param inGatesI  [B, 3H] input projection (reset|update|candidate along dim 1)
///                  or [B*T, 3H] with inTimeOffset = t*B to index row t without Slice
/// @param inGatesH  [B, 3H] hidden projection
/// @param inHidden  [B, H]  previous hidden state
/// @param inHiddenSize H
/// @param inTimeOffset row offset into inGatesI (in units of 3H), default 0
[[nodiscard]] Matrix gruCellPointwise(
	const Matrix& inGatesI, const Matrix& inGatesH, const Matrix& inHidden,
	oa::I32 inHiddenSize, oa::U32 inTimeOffset = 0, oa::U32 inBatchStride = 1);

/// gruCellPointwiseBwd: fused GRU pointwise backward.
/// Returns gradients w.r.t. gatesI, gatesH and the previous hidden state.
/// @param inTimeOffset row offset into inGatesI / dGatesI (in rows of 3H), default 0
/// @param inBatchStride row stride between batches in inGatesI (T for batch-major, 1 for contiguous), default 1
[[nodiscard]] GruCellPointwiseBwdResult gruCellPointwiseBwd(
	const Matrix& inGatesI, const Matrix& inGatesH, const Matrix& inHidden,
	const Matrix& inGradOutput, oa::I32 inHiddenSize, oa::U32 inTimeOffset = 0, oa::U32 inBatchStride = 1);

/// gruCellLinear: fused GRU recurrent step — Linear(h, W_hh) + gruCellPointwise.
/// Replaces the per-timestep pair of dispatches with one kernel. The hidden
/// projection required by reverse mode is retained internally rather than
/// exposed as an output parameter.
[[nodiscard]] Matrix gruCellLinear(
	const Matrix& inGatesI,
	const Matrix& inHidden,
	const Matrix& inWeightHh,
	oa::I32 inHiddenSize,
	oa::U32 inTimeOffset,
	oa::U32 inBatchStride,
	const Matrix& inBiasHh = Matrix{});

/// gruScan: whole-sequence GRU recurrent scan in ONE dispatch (one workgroup per
/// batch, looping all timesteps). Mathematically identical to running gruCellLinear
/// for each timestep, but collapses S dispatches into 1. The recurrent weight/bias
/// gradient is computed separately via linearWeightBiasBwd on the saved hPrev.
/// @param inGatesI [B*S, 3H] precomputed input projection (row b*S+t = timestep t)
/// @param inSeqLen S
[[nodiscard]] GruScanResult gruScan(
	const Matrix& inGatesI,
	const Matrix& inWeightHh,
	oa::I32 inHiddenSize,
	oa::I32 inSeqLen,
	oa::I32 inBatch,
	const Matrix& inBiasHh = Matrix{}
);

/// gruScanBwd: BPTT recurrence scan (backward of gruScan) in ONE dispatch.
/// Produces gradients w.r.t. the input projection gatesI and the hidden projection
/// gatesH (the latter drives the separate linearWeightBiasBwd weight-grad call).
[[nodiscard]] GruScanBwdResult gruScanBwd(
	const Matrix& inDOut,
	const Matrix& inGatesI,
	const Matrix& inHPrev,
	const Matrix& inWeightHh,
	oa::I32 inHiddenSize,
	oa::I32 inSeqLen,
	oa::I32 inBatch,
	const Matrix& inBiasHh = Matrix{});

// --- RNN ---

/// rnnCellPointwise: fused vanilla-RNN pointwise forward, h_new = tanh(gatesI + gatesH).
/// @param inGatesI  [B, H] input projection  (W_ih x + b_ih)
/// @param inGatesH  [B, H] hidden projection (W_hh h_prev + b_hh)
[[nodiscard]] Matrix rnnCellPointwise(const Matrix& inGatesI, const Matrix& inGatesH);

/// rnnCellPointwiseBwd: fused vanilla-RNN pointwise backward.
/// Returns gradients w.r.t. gatesI and gatesH (both equal to dL/da). inGatesI is
/// the whole-sequence [B*T, H] projection; inTimeOffset/inBatchStride select this
/// timestep's rows so dGatesI is scattered into the full buffer (zeros elsewhere).
[[nodiscard]] RnnCellPointwiseBwdResult rnnCellPointwiseBwd(
	const Matrix& inGatesI, const Matrix& inGatesH, const Matrix& inGradOutput,
	oa::I32 inHiddenSize, oa::U32 inTimeOffset = 0, oa::U32 inBatchStride = 1);

/// rnnCellLinear: fused vanilla-RNN recurrent step — Linear(h, W_hh) + rnnCellPointwise.
/// Replaces the per-timestep pair of dispatches with one kernel. inGatesI is the whole
/// input projection [B*T, H]; inTimeOffset/inBatchStride index the current timestep's
/// row directly, so oa::Rnn needs no per-step Slice. The hidden projection required
/// by reverse mode is retained internally rather than exposed as an output parameter.
[[nodiscard]] Matrix rnnCellLinear(
	const Matrix& inGatesI,
	const Matrix& inHidden,
	const Matrix& inWeightHh,
	oa::U32 inTimeOffset = 0,
	oa::U32 inBatchStride = 1,
	const Matrix& inBiasHh = Matrix{});

/// rnnScan: whole-sequence vanilla-RNN recurrent scan in ONE dispatch.
/// Mathematically identical to running rnnCellLinear for each timestep, but collapses
/// S dispatches into 1. The recurrent weight/bias gradient is computed separately via
/// linearWeightBiasBwd on the saved hPrev.
/// @param inGatesI [B*S, H] precomputed input projection (row b*S+t = timestep t)
[[nodiscard]] RnnScanResult rnnScan(
	const Matrix& inGatesI,
	const Matrix& inWeightHh,
	oa::I32 inHiddenSize,
	oa::I32 inSeqLen,
	oa::I32 inBatch,
	const Matrix& inBiasHh = Matrix{});

/// rnnScanBwd: BPTT recurrence scan (backward of rnnScan) in ONE dispatch.
[[nodiscard]] RnnScanBwdResult rnnScanBwd(
	const Matrix& inDOut,
	const Matrix& inGatesI,
	const Matrix& inHPrev,
	const Matrix& inWeightHh,
	oa::I32 inHiddenSize,
	oa::I32 inSeqLen,
	oa::I32 inBatch,
	const Matrix& inBiasHh = Matrix{});

// --- VQ ---

/// detach: stop-gradient. Returns a view that SHARES inSelf's device buffer but
/// carries no autograd linkage (leaf, requiresGrad=false), so backward terminates
/// here. Metadata-only: no kernel, no copy. This is the primitive the straight-
/// through estimator needs.
[[nodiscard]] Matrix detach(const Matrix& inSelf);

/// vqAssign: vector-quantization nearest-code assignment (VQ-VAE codebook lookup).
/// inZe: [N, D] latents. inCodebook: [K, D] codes. Returns the per-row argmin index
/// (int32 [N]) and the gathered winning code (float [N, D]) by squared L2 distance.
[[nodiscard]] VqAssignResult vqAssign(const Matrix& inZe, const Matrix& inCodebook);

/// vqEmaUpdate: EMA codebook update + dead-code reinit (van den Oord 2017).
/// The codebook is NOT gradient-trained; each entry tracks the running mean of encoder
/// outputs assigned to it, and dead codes are revived from live encoder rows.
///   inZe [N,D], inIdx [N] int32 (from vqAssign), ioEmbedSum [K,D], ioClusterSize [K],
///   outCodebook [K,D]. inDecay = EMA γ; inEps = division floor; inDeadThreshold =
///   revive codes whose EMA count falls below it; inSeed = per-step seed; inNormalize
///   rescales each codebook row to unit RMS (cosine VQ).
void vqEmaUpdate(const Matrix& inZe, const Matrix& inIdx,
	Matrix& ioEmbedSum, Matrix& ioClusterSize, Matrix& outCodebook,
	oa::F32 inDecay, oa::F32 inEps, oa::F32 inDeadThreshold, oa::U32 inSeed,
	bool inNormalize);

// --- generated ML operations ---
// Regenerate via: python3 tools/gen/fn/generate.py --live
#include <oa/ml/fnmatrix/fnMatrix.gen.h>

// --- backward pass activations ---

/// reluBwd: backward pass for ReLU activation.
/// Computes: dInput = dOutput * (forwardOutput > 0)
[[nodiscard]] Matrix reluBwd(const Matrix& inForwardOutput, const Matrix& inGradOutput);

/// tanhBwd: backward pass for Tanh activation.
/// Computes: dInput = dOutput * (1 - tanh(x)^2), using saved tanh(x) output.
[[nodiscard]] Matrix tanhBwd(const Matrix& inForwardOutput, const Matrix& inGradOutput);

/// geluBwd: backward pass for GELU activation.
/// Computes: dInput = dOutput * gelu'(x)
/// @param inInput: forward INPUT x (gelu'(x) is a function of input, not output)
[[nodiscard]] Matrix geluBwd(const Matrix& inInput, const Matrix& inGradOutput);

/// siluBwd: backward pass for SiLU activation.
/// @param inInput: forward INPUT x (silu'(x) is a function of input, not output)
[[nodiscard]] Matrix siluBwd(const Matrix& inInput, const Matrix& inGradOutput);

/// sigmoidBwd: backward pass for Sigmoid activation.
/// Computes: dInput = dOutput * sigmoid(x) * (1 - sigmoid(x))
[[nodiscard]] Matrix sigmoidBwd(const Matrix& inForwardOutput, const Matrix& inGradOutput);

/// leakyReluBwd: backward pass for LeakyReLU activation.
/// Computes: dInput = dOutput * (x > 0 ? 1 : alpha)
[[nodiscard]] Matrix leakyReluBwd(const Matrix& inForwardOutput, const Matrix& inGradOutput,
	oa::F32 inAlpha = 0.01f);

/// eluBwd: backward pass for ELU activation.
/// Computes: dInput = dOutput * (x > 0 ? 1 : alpha * exp(x))
[[nodiscard]] Matrix eluBwd(const Matrix& inForwardOutput, const Matrix& inGradOutput,
	oa::F32 inAlpha = 1.0f);

/// mishBwd: backward pass for Mish activation.
/// @param inInput: forward INPUT x (mish'(x) depends on x directly)
[[nodiscard]] Matrix mishBwd(const Matrix& inInput, const Matrix& inGradOutput);

/// softplusBwd: backward pass for Softplus activation.
/// @param inForwardOutput: output y = softplus(a) from the forward pass
/// @return dOut * sigmoid(a) = dOut * (1 - e^-y)
[[nodiscard]] Matrix softplusBwd(const Matrix& inForwardOutput, const Matrix& inGradOutput);

/// siluMulBwd: backward pass for SiluMul activation.
/// @param inInput: forward INPUT (gate||up); the forward output is not invertible
[[nodiscard]] Matrix siluMulBwd(const Matrix& inInput, const Matrix& inGradOutput);

/// gegluBwd: backward pass for GEGLU activation.
/// @param inInput: forward INPUT (up||gate); up*GELU(gate) is not invertible
[[nodiscard]] Matrix gegluBwd(const Matrix& inInput, const Matrix& inGradOutput);

/// swigluBwd: backward pass for SwiGLU activation.
/// @return SwigluBwdResult with dGate, dUp
[[nodiscard]] SwigluBwdResult swigluBwd(
	const Matrix& inGate, const Matrix& inUp, const Matrix& inOut,
	const Matrix& inGradOutput);

/// softmaxScaledMasked: fused transformer attention score normalisation.
/// Computes: out = softmax(scores * scale + mask) over the last dimension.
[[nodiscard]] Matrix softmaxScaledMasked(
	const Matrix& inScores, const Matrix& inMask, oa::F32 inScale);

/// softmaxScaledMaskedBwd: backward for the fused attention score op.
/// Returns dScores = softmaxOut * (dOut - sum(dOut * softmaxOut)) * scale.
[[nodiscard]] Matrix softmaxScaledMaskedBwd(
	const Matrix& inForwardOutput, const Matrix& inGradOutput, oa::F32 inScale);

// --- Linear ---

/// linear: linear layer (fully connected). output = input @ weight^T + bias
/// @param inX       [batch, inFeatures]
/// @param inWeight  [outFeatures, inFeatures]
/// @param inBias    [outFeatures] (optional)
[[nodiscard]] Matrix linear(
	const Matrix& inX,
	const Matrix& inWeight,
	const Matrix& inBias = Matrix{});

/// linearRelu: fused linear + ReLU. output = reLU(input @ weight^T + bias)
[[nodiscard]] Matrix linearRelu(
	const Matrix& inX,
	const Matrix& inWeight,
	const Matrix& inBias);

/// linearGelu: fused linear + GELU. output = GELU(input @ weight^T + bias)
/// The fused forward discards the pre-activation; the autograd node recomputes
/// it (one GEMM) for geluBwd in the backward pass.
[[nodiscard]] Matrix linearGelu(
	const Matrix& inX,
	const Matrix& inWeight,
	const Matrix& inBias);

/// linearSilu: fused linear + SiLU. output = siLU(input @ weight^T + bias)
[[nodiscard]] Matrix linearSilu(
	const Matrix& inX,
	const Matrix& inWeight,
	const Matrix& inBias);

/// linearDataBwd: backward for linear layer (input gradient).
/// Computes: dInput = dOutput @ weight^T
[[nodiscard]] Matrix linearDataBwd(const Matrix& inGradOutput, const Matrix& inWeight);

/// linearWeightBwd: backward for linear layer (weight gradient).
/// Computes: dWeight = input^T @ dOutput
[[nodiscard]] Matrix linearWeightBwd(const Matrix& inInput, const Matrix& inGradOutput);

/// linearWeightBiasBwd: fused weight and bias gradient for linear layer.
/// Computes: dWeight = input^T @ dOutput, dBias = sum(dOutput, dim=0)
[[nodiscard]] LinearWeightBiasBwdResult linearWeightBiasBwd(const Matrix& inInput, const Matrix& inGradOutput);

/// linearDataReluBwd: fused linear data gradient followed by ReLU backward.
/// Computes: dInput = (dOutput @ weight) * (activation > 0)
[[nodiscard]] Matrix linearDataReluBwd(
	const Matrix& inGradOutput,
	const Matrix& inWeight,
	const Matrix& inActivation);

/// linearReluBwdData: fused in-layer linearRelu(x,W,b) backward, data path.
/// For y = reLU(x @ W^T + b), computes dx = (dy * (act > 0)) @ W in a
/// single dispatch. gate is applied INSIDE the inner sum (no materialization
/// of dz), the opposite fusion direction of linearDataReluBwd.
[[nodiscard]] Matrix linearReluBwdData(
	const Matrix& inGradOutput,
	const Matrix& inWeight,
	const Matrix& inActivation);

// --- Normalization ---

// LayerNorm and RmsNorm forward declarations are generated from MlFnMatrixNorm.toml.

[[nodiscard]] Matrix rmsNormGated(
	const Matrix& inSelf, const Matrix& inWeight, const Matrix& inBias,
	const Matrix& inZ, oa::F32 inEps, bool inNormBeforeGate = true);

[[nodiscard]] Matrix heavyTailActivation(const Matrix& inSelf);

/// layerNormBwd: backward pass for LayerNorm.
[[nodiscard]] LayerNormBwdResult layerNormBwd(
	const Matrix& inX, const Matrix& inWeight, const Matrix& inBias,
	const Matrix& inOut, const Matrix& inMean, const Matrix& inRstd,
	const Matrix& inGradOutput, oa::F32 inEps = 1e-5F);

/// rmsNormBwd: backward pass for RmsNorm.
[[nodiscard]] RmsNormBwdResult rmsNormBwd(
	const Matrix& inX, const Matrix& inWeight,
	const Matrix& inGradOutput, oa::F32 inEps = 1e-5F);

/// rmsNormGatedBwd: backward for rmsNormGated (normBeforeGate = true).
/// Returns grads w.r.t. x, weight, bias, z.
[[nodiscard]] RmsNormGatedBwdResult rmsNormGatedBwd(
	const Matrix& inX, const Matrix& inWeight, const Matrix& inBias,
	const Matrix& inZ, const Matrix& inGradOutput, oa::F32 inEps);

/// channelNorm: fused LayerNorm over the channel axis of [B,C,T] without
/// transposing. Replaces Transpose+LayerNorm+transpose (3 dispatches) with 1.
[[nodiscard]] Matrix channelNorm(
	const Matrix& inX, const Matrix& inWeight, const Matrix& inBias,
	oa::I32 inBatch, oa::I32 inChannels, oa::I32 inSeqLen, oa::F32 inEps);

[[nodiscard]] ChannelNormBwdResult channelNormBwd(
	const Matrix& inX, const Matrix& inWeight,
	const Matrix& inGradOutput,
	oa::I32 inBatch, oa::I32 inChannels, oa::I32 inSeqLen, oa::F32 inEps);

/// channelNormRelu: fused channelNorm + ReLU on [B,C,T].
[[nodiscard]] Matrix channelNormRelu(
	const Matrix& inX, const Matrix& inWeight, const Matrix& inBias,
	oa::I32 inBatch, oa::I32 inChannels, oa::I32 inSeqLen, oa::F32 inEps);

[[nodiscard]] ChannelNormBwdResult channelNormReluBwd(
	const Matrix& inX, const Matrix& inWeight, const Matrix& inFwdOut,
	const Matrix& inGradOutput,
	oa::I32 inBatch, oa::I32 inChannels, oa::I32 inSeqLen, oa::F32 inEps);

// --- Conv1d ---

/// conv1dGemm: 1-D convolution executed as im2col + a single matmul.
/// inX [N, inC, L], inWeight [outC, inC, K], inBias [outC] -> [N, outC, outL]
[[nodiscard]] Matrix conv1dGemm(
	const Matrix& inX, const Matrix& inWeight, const Matrix& inBias,
	oa::I32 inStride = 1, oa::I32 inPadding = 0, oa::I32 inDilation = 1);

/// conv1dReluGemm: conv1dGemm with the ReLU folded into the GEMM bias epilogue.
[[nodiscard]] Matrix conv1dReluGemm(
	const Matrix& inX, const Matrix& inWeight, const Matrix& inBias,
	oa::I32 inStride = 1, oa::I32 inPadding = 0, oa::I32 inDilation = 1);

/// im2Col1d: unfold a 1-D conv input [N, inC, L] into the GEMM-ready column
/// matrix [N*outL, inC*K]. The building block of conv1dGemm.
[[nodiscard]] Matrix im2Col1d(
	const Matrix& inX, oa::I32 inK, oa::I32 inStride, oa::I32 inPadding, oa::I32 inDilation = 1);

/// col2Im1d: backward of im2Col1d — fold a column-matrix gradient [N*outL, inC*K]
/// back into input shape [N, inC, L], accumulating over overlapping windows.
[[nodiscard]] Matrix col2Im1d(
	const Matrix& inDCols, oa::I32 inN, oa::I32 inC, oa::I32 inL, oa::I32 inK,
	oa::I32 inStride, oa::I32 inPadding, oa::I32 inDilation, oa::I32 inOutL);

/// conv1dBwdData: backward for 1D convolution (input gradient).
[[nodiscard]] Matrix conv1dBwdData(
	const Matrix& inDOut,
	const Matrix& inWeight,
	oa::U32 inStride,
	oa::U32 inPadding,
	oa::U32 inDilation,
	const MatrixShape& inInputShape);

/// conv1dBwdWeight: fused weight and bias gradient for 1D convolution.
[[nodiscard]] Conv1dBwdWeightResult conv1dBwdWeight(
	const Matrix& inInput,
	const Matrix& inDOut,
	const Matrix& inWeight,
	oa::U32 inStride,
	oa::U32 inPadding,
	oa::U32 inDilation);

// --- Conv2d ---

[[nodiscard]] Matrix conv2d(
	const Matrix& inX, const Matrix& inWeight, const Matrix& inBias,
	oa::U32 inStride, oa::U32 inPadding, oa::U32 inGroups = 1);

/// conv2dBwdData: backward for 2D convolution (input gradient).
[[nodiscard]] Matrix conv2dBwdData(
	const Matrix& inDOut,
	const Matrix& inWeight,
	oa::U32 inStride,
	oa::U32 inPadding,
	const MatrixShape& inInputShape,
	oa::U32 inGroups = 1);

/// conv2dBwdWeight: fused weight and bias gradient for 2D convolution.
[[nodiscard]] Conv2dBwdWeightResult conv2dBwdWeight(
	const Matrix& inInput,
	const Matrix& inDOut,
	const Matrix& inWeight,
	oa::U32 inStride,
	oa::U32 inPadding,
	oa::U32 inGroups = 1);

/// convTranspose2d: 2D transposed convolution (learnable upsampling).
/// input: [N, inC, H, W], weight: [inC, outC, K, K], Bias: [outC]
/// output: [N, outC, H_out, W_out] where H_out = (H - 1) * S - 2P + K.
[[nodiscard]] Matrix convTranspose2d(
	const Matrix& inX, const Matrix& inWeight, const Matrix& inBias,
	oa::U32 inStride, oa::U32 inPadding);

/// convTranspose2dBwdData: backward for 2D transposed convolution (input gradient).
[[nodiscard]] Matrix convTranspose2dBwdData(
	const Matrix& inDOut,
	const Matrix& inWeight,
	oa::U32 inStride,
	oa::U32 inPadding,
	const MatrixShape& inInputShape);

/// convTranspose2dBwdWeight: fused weight and bias gradient for 2D transposed convolution.
[[nodiscard]] ConvTranspose2dBwdWeightResult convTranspose2dBwdWeight(
	const Matrix& inInput,
	const Matrix& inDOut,
	const Matrix& inWeight,
	oa::U32 inStride,
	oa::U32 inPadding);

// --- Pooling ---

[[nodiscard]] Matrix avgPool2d(const Matrix& inX, oa::I32 inKernelSize, oa::I32 inStride, oa::I32 inPadding);

[[nodiscard]] MaxPool2dResult maxPool2d(const Matrix& inX, oa::I32 inKernelSize, oa::I32 inStride, oa::I32 inPadding);

/// maxPool2dBwd: backward for 2D max pooling.
[[nodiscard]] Matrix maxPool2dBwd(
	const Matrix& inX, const Matrix& inIndices, const Matrix& inGradOutput,
	oa::I32 inKernelSize, oa::I32 inStride, oa::I32 inPadding
);

/// avgPool2dBwd: backward for 2D average pooling.
[[nodiscard]] Matrix avgPool2dBwd(
	const Matrix& inX, const Matrix& inGradOutput,
	oa::I32 inKernelSize, oa::I32 inStride, oa::I32 inPadding
);

// --- Batched linear algebra ---

/// bmm: per-batch matrix multiply, A[N,M,K] @ B[N,K,P] = out[N,M,P].
[[nodiscard]] Matrix bmm(const Matrix& inA, const Matrix& inB);

/// bmmNt: per-batch matrix multiply with transposed right operand storage,
/// A[N,M,K] @ B[N,P,K]^T = out[N,M,P].
[[nodiscard]] Matrix bmmNt(const Matrix& inA, const Matrix& inB);

/// splitHeads: [B*S,D] -> [B*H,S,D/H]. Multi-head permutation.
[[nodiscard]] Matrix splitHeads(const Matrix& inX, oa::I32 inBatch, oa::I32 inSeqLen, oa::I32 inNumHeads);

/// mergeHeads: [B*H,S,D/H] -> [B*S,D]. exact inverse of splitHeads.
[[nodiscard]] Matrix mergeHeads(const Matrix& inX, oa::I32 inBatch, oa::I32 inSeqLen, oa::I32 inNumHeads);

/// flashAttentionCausal: IO-aware causal scaled dot-product attention.
/// Q/K/V and output use contiguous [batchHeads, sequence, headDim] storage.
[[nodiscard]] Matrix flashAttentionCausal(const Matrix& inQ, const Matrix& inK, const Matrix& inV, oa::F32 inScale);

/// flashAttentionCausalBwd: explicit adjoint for the FlashAttention autograd node.
[[nodiscard]] FlashAttentionBwdResult flashAttentionCausalBwd(
	const Matrix& inQ, const Matrix& inK, const Matrix& inV,
	const Matrix& inOutput, const Matrix& inLogSumExp,
	const Matrix& inGradOutput, oa::F32 inScale);

// --- BatchNorm ---

/// upsampleBwd: backward pass for Upsample (Nearest or Bilinear).
[[nodiscard]] Matrix upsampleBwd(const Matrix& inInput, const Matrix& inDOut, oa::I32 inScaleFactor, bool inIsBilinear);

/// batchNorm2dBwd: backward pass for BatchNorm2d.
[[nodiscard]] BatchNorm2dBwdResult batchNorm2dBwd(
	const Matrix& inX, const Matrix& inGamma, const Matrix& inBeta,
	const Matrix& inMean, const Matrix& inVar, const Matrix& inOut,
	const Matrix& inDOut, oa::F32 inEps, bool inIsTraining
);

// Optimizer operations are in ml/fnOptim.h (use oa::FnOptim::adamWStep, etc.)

} // namespace FnMatrix

} // namespace oa
