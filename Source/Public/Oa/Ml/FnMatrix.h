#pragma once

#include <Oa/Core/Matrix.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/MatrixShape.h>
#include <Oa/Core/Status.h>
#include <Oa/Ml/FnLoss.h>

class OaComputeEngine;
class OaVkBatch;

// OaFnMatrix — ML module extension for neural network operations.
// This header extends the OaFnMatrix namespace from Core with ML-specific functions.
// Include <Oa/Core/FnMatrix.h> and <Oa/Ml/FnMatrix.h> to access all functions.
namespace OaFnMatrix {

struct OaLinearWeightBiasBwdResult {
	OaMatrix GradWeight;
	OaMatrix GradBias;
};

struct OaLayerNormBwdResult {
	OaMatrix DX;
	OaMatrix DWeight;
	OaMatrix DBias;
};

struct OaRmsNormBwdResult {
	OaMatrix DX;
	OaMatrix DWeight;
};

struct OaRmsNormGatedBwdResult {
	OaMatrix DX;
	OaMatrix DWeight;
	OaMatrix DBias;
	OaMatrix DZ;
};

struct OaChannelNormBwdResult {
	OaMatrix DX;
	OaMatrix DWeight;
	OaMatrix DBias;
};

struct OaSwigluBwdResult {
	OaMatrix GateGrad;
	OaMatrix UpGrad;
};

// GRU recurrent cell — fused pointwise gate combine.
// Forward: given the input/hidden gate projections gates_i, gates_h (each [B, 3H],
// row-major reset|update|candidate) and the previous hidden h_prev [B, H], produces
// the new hidden state [B, H] in a single kernel. Replaces ~10 elementwise dispatches.
struct OaGruCellPointwiseBwdResult {
	OaMatrix DGatesI;   // [B, 3H] grad w.r.t. the input projection
	OaMatrix DGatesH;   // [B, 3H] grad w.r.t. the hidden projection
	OaMatrix DHidden;   // [B, H]  grad w.r.t. the previous hidden state
};

/// GruCellPointwise: fused GRU pointwise forward.
///   r = sigmoid(gates_i[r] + gates_h[r])
///   z = sigmoid(gates_i[z] + gates_h[z])
///   n = tanh(gates_i[n] + r * gates_h[n])
///   h_new = (1 - z) * n + z * h_prev
/// @param InGatesI  [B, 3H] input projection (reset|update|candidate along dim 1)
///                  or [B*T, 3H] with InTimeOffset = t*B to index row t without Slice
/// @param InGatesH  [B, 3H] hidden projection
/// @param InHidden  [B, H]  previous hidden state
/// @param InHiddenSize H
/// @param InTimeOffset row offset into InGatesI (in units of 3H), default 0
[[nodiscard]] OaMatrix GruCellPointwise(
	const OaMatrix& InGatesI, const OaMatrix& InGatesH, const OaMatrix& InHidden,
	OaI32 InHiddenSize, OaU32 InTimeOffset = 0, OaU32 InBatchStride = 1);

/// GruCellPointwiseBwd: fused GRU pointwise backward.
/// Returns gradients w.r.t. gates_i, gates_h and the previous hidden state.
/// @param InTimeOffset row offset into InGatesI / dGatesI (in rows of 3H), default 0
/// @param InBatchStride row stride between batches in InGatesI (T for batch-major, 1 for contiguous), default 1
[[nodiscard]] OaGruCellPointwiseBwdResult GruCellPointwiseBwd(
	const OaMatrix& InGatesI, const OaMatrix& InGatesH, const OaMatrix& InHidden,
	const OaMatrix& InGradOutput, OaI32 InHiddenSize, OaU32 InTimeOffset = 0, OaU32 InBatchStride = 1);

/// GruCellLinear: fused GRU recurrent step — Linear(h, W_hh) + GruCellPointwise.
/// Replaces the per-timestep pair of dispatches with one kernel. When
/// OutGatesH is non-null, the hidden projection is also written so the existing
/// backward path (GruCellPointwiseBwd + LinearBwd) can be reused.
[[nodiscard]] OaMatrix GruCellLinear(
	const OaMatrix& InGatesI,
	const OaMatrix& InHidden,
	const OaMatrix& InWeightHh,
	const OaMatrix* InBiasHh,
	OaI32 InHiddenSize,
	OaU32 InTimeOffset,
	OaU32 InBatchStride,
	OaMatrix* OutGatesH = nullptr);

/// GruScan: whole-sequence GRU recurrent scan in ONE dispatch (one workgroup per
/// batch, looping all timesteps). Mathematically identical to running GruCellLinear
/// for each timestep, but collapses S dispatches into 1. The recurrent weight/bias
/// gradient is computed separately via LinearWeightBiasBwd on the saved Hprev.
struct OaGruScanResult {
	OaMatrix Out;    // [B, S, H] hidden states h^{(1..S)}
	OaMatrix Hprev;  // [B, S, H] previous hidden states h^{(0..S-1)} (saved for bwd)
};

/// @param InGatesI [B*S, 3H] precomputed input projection (row b*S+t = timestep t)
/// @param InSeqLen S
[[nodiscard]] OaGruScanResult GruScan(
	const OaMatrix& InGatesI,
	const OaMatrix& InWeightHh,
	const OaMatrix* InBiasHh,
	OaI32 InHiddenSize,
	OaI32 InSeqLen,
	OaI32 InBatch
);

/// GruScanBwd: BPTT recurrence scan (backward of GruScan) in ONE dispatch.
/// Produces gradients w.r.t. the input projection gates_i and the hidden projection
/// gates_h (the latter drives the separate LinearWeightBiasBwd weight-grad call).
struct OaGruScanBwdResult {
	OaMatrix DGatesI;  // [B*S, 3H]
	OaMatrix DGatesH;  // [B*S, 3H]
};

[[nodiscard]] OaGruScanBwdResult GruScanBwd(
	const OaMatrix& InDOut,        // [B, S, H] grad w.r.t. Out
	const OaMatrix& InGatesI,      // [B*S, 3H]
	const OaMatrix& InHprev,       // [B, S, H]
	const OaMatrix& InWeightHh,    // [3H, H]
	const OaMatrix* InBiasHh,
	OaI32 InHiddenSize,
	OaI32 InSeqLen,
	OaI32 InBatch
);

// Vanilla RNN cell — fused pointwise combine. Forward: given the input/hidden
// gate projections gates_i, gates_h (each [B, H]), produces h_new = tanh(gates_i
// + gates_h) in a single kernel, replacing the Add + Tanh dispatch pair.
struct OaRnnCellPointwiseBwdResult {
	OaMatrix DGatesI;   // [B, H] grad w.r.t. the input projection
	OaMatrix DGatesH;   // [B, H] grad w.r.t. the hidden projection
};

/// RnnCellPointwise: fused vanilla-RNN pointwise forward, h_new = tanh(gates_i + gates_h).
/// @param InGatesI  [B, H] input projection  (W_ih x + b_ih)
/// @param InGatesH  [B, H] hidden projection (W_hh h_prev + b_hh)
[[nodiscard]] OaMatrix RnnCellPointwise(const OaMatrix& InGatesI, const OaMatrix& InGatesH);

/// RnnCellPointwiseBwd: fused vanilla-RNN pointwise backward.
/// Returns gradients w.r.t. gates_i and gates_h (both equal to dL/da). InGatesI is
/// the whole-sequence [B*T, H] projection; InTimeOffset/InBatchStride select this
/// timestep's rows so DGatesI is scattered into the full buffer (zeros elsewhere).
[[nodiscard]] OaRnnCellPointwiseBwdResult RnnCellPointwiseBwd(
	const OaMatrix& InGatesI, const OaMatrix& InGatesH, const OaMatrix& InGradOutput,
	OaI32 InHiddenSize, OaU32 InTimeOffset = 0, OaU32 InBatchStride = 1
);

/// RnnCellLinear: fused vanilla-RNN recurrent step — Linear(h, W_hh) + RnnCellPointwise.
/// Replaces the per-timestep pair of dispatches with one kernel. InGi is the whole
/// input projection [B*T, H]; InTimeOffset/InBatchStride index the current timestep's
/// row directly, so OaRnn needs no per-step Slice. When OutGatesH is non-null, the
/// hidden projection is also written so the existing backward path
/// (RnnCellPointwiseBwd + LinearBwd) can be reused.
[[nodiscard]] OaMatrix RnnCellLinear(
	const OaMatrix& InGi,
	const OaMatrix& InHidden,
	const OaMatrix& InWeightHh,
	const OaMatrix* InBiasHh,
	OaU32 InTimeOffset = 0,
	OaU32 InBatchStride = 1,
	OaMatrix* OutGatesH = nullptr
);

/// RnnScan: whole-sequence vanilla-RNN recurrent scan in ONE dispatch.
/// Mathematically identical to running RnnCellLinear for each timestep, but collapses
/// S dispatches into 1. The recurrent weight/bias gradient is computed separately via
/// LinearWeightBiasBwd on the saved Hprev.
struct OaRnnScanResult {
	OaMatrix Out;    // [B, S, H] hidden states h^{(1..S)}
	OaMatrix Hprev;  // [B, S, H] previous hidden states h^{(0..S-1)} (saved for bwd)
};

/// @param InGatesI [B*S, H] precomputed input projection (row b*S+t = timestep t)
[[nodiscard]] OaRnnScanResult RnnScan(
	const OaMatrix& InGatesI,
	const OaMatrix& InWeightHh,
	const OaMatrix* InBiasHh,
	OaI32 InHiddenSize,
	OaI32 InSeqLen,
	OaI32 InBatch
);

/// RnnScanBwd: BPTT recurrence scan (backward of RnnScan) in ONE dispatch.
struct OaRnnScanBwdResult {
	OaMatrix DGatesI;  // [B*S, H]
	OaMatrix DGatesH;  // [B*S, H]
};

[[nodiscard]] OaRnnScanBwdResult RnnScanBwd(
	const OaMatrix& InDOut,        // [B, S, H] grad w.r.t. Out
	const OaMatrix& InGatesI,      // [B*S, H]
	const OaMatrix& InHprev,       // [B, S, H]
	const OaMatrix& InWeightHh,    // [H, H]
	const OaMatrix* InBiasHh,
	OaI32 InHiddenSize,
	OaI32 InSeqLen,
	OaI32 InBatch
);


struct OaVqAssignResult {
	OaMatrix Idx;   // [N]    int32 — nearest codebook index per row
	OaMatrix Zq;    // [N, D] float — the gathered winning code per row
};

// Detach — stop-gradient. Returns a view that SHARES InSelf's device buffer but
// carries no autograd linkage (leaf, RequiresGrad=false), so Backward terminates
// here. Metadata-only: no kernel, no copy. This is the primitive the straight-
// through estimator needs — z_q_st = z_e + Detach(z_q - z_e) gives forward value
// z_q while routing the gradient straight to z_e.
[[nodiscard]] OaMatrix Detach(const OaMatrix& InSelf);

// VqAssign — vector-quantization nearest-code assignment (VQ-VAE codebook lookup).
// InZe: [N, D] latents. InCodebook: [K, D] codes. Returns the per-row argmin index
// (int32 [N]) and the gathered winning code (float [N, D]) by squared L2 distance.
// Records one on-GPU dispatch; non-differentiable (no autograd node). See
// Source/Private/Oa/Ml/Shader/Compute/Ops/VqAssign.slang.
[[nodiscard]] OaVqAssignResult VqAssign(const OaMatrix& InZe, const OaMatrix& InCodebook);

// VqEmaUpdate — EMA codebook update + dead-code reinit (van den Oord 2017). The
// codebook is NOT gradient-trained; each entry tracks the running mean of encoder
// outputs assigned to it, and dead codes are revived from live encoder rows. Records
// one on-GPU dispatch that mutates IoEmbedSum / IoClusterSize / OutCodebook in place.
//   InZe [N,D], InIdx [N] int32 (from VqAssign), IoEmbedSum [K,D], IoClusterSize [K],
//   OutCodebook [K,D]. InDecay = EMA γ; InEps = division floor; InDeadThreshold =
//   revive codes whose EMA count falls below it; InSeed = per-step seed mixed into the
//   dead-code revival row so revived codes scatter across steps; InNormalize rescales each
//   codebook row to unit RMS (cosine VQ — pair with unit-RMS z_e). See Ops/VqEmaUpdate.slang.
void VqEmaUpdate(const OaMatrix& InZe, const OaMatrix& InIdx,
                 OaMatrix& IoEmbedSum, OaMatrix& IoClusterSize, OaMatrix& OutCodebook,
                 OaF32 InDecay, OaF32 InEps, OaF32 InDeadThreshold, OaU32 InSeed,
                 bool InNormalize);

// ─── Generated ML Operations ──────────────────────────────────────
// Regenerate via: python3 Tools/OaFnAutogen/oafnautogen.py
#include "../../../Private/Oa/Ml/FnMatrix/FnMatrix.gen.h"

// ─── Backward Pass Operations ─────────────────────────────────────

/// ReluBwd: Backward pass for ReLU activation.
/// Computes: d_input = d_output * (forward_output > 0)
/// @param InForwardOutput: Output from forward ReLU pass
/// @param InGradOutput: Gradient flowing back from next layer
/// @return Gradient w.r.t. input
[[nodiscard]] OaMatrix ReluBwd(const OaMatrix& InForwardOutput, const OaMatrix& InGradOutput);

/// TanhBwd: Backward pass for Tanh activation.
/// Computes: d_input = d_output * (1 - tanh(x)^2), using saved tanh(x) output.
/// @param InForwardOutput: Output from forward Tanh pass (tanh(x))
/// @param InGradOutput: Gradient flowing back from next layer
/// @return Gradient w.r.t. input
[[nodiscard]] OaMatrix TanhBwd(const OaMatrix& InForwardOutput, const OaMatrix& InGradOutput);

/// GeluBwd: Backward pass for GELU activation.
/// Computes: d_input = d_output * gelu'(x)
/// @param InInput: Forward INPUT x. Required — gelu'(x) is a function of the
///        input, not the output y=GELU(x) (which is not invertible to x).
/// @param InGradOutput: Gradient flowing back from next layer
/// @return Gradient w.r.t. input
[[nodiscard]] OaMatrix GeluBwd(const OaMatrix& InInput, const OaMatrix& InGradOutput);

/// SiluBwd: Backward pass for SiLU activation.
/// Computes: d_input = d_output * silu'(x)
/// @param InInput: Forward INPUT x. Required — silu'(x) is a function of the
///        input, not the output y=SiLU(x).
/// @param InGradOutput: Gradient flowing back from next layer
/// @return Gradient w.r.t. input
[[nodiscard]] OaMatrix SiluBwd(const OaMatrix& InInput, const OaMatrix& InGradOutput);

/// SoftmaxScaledMasked: fused transformer attention score normalisation.
/// Computes: out = softmax(scores * scale + mask) over the last dimension.
/// Replaces the Scale → Add(mask) → Softmax chain with one dispatch.
[[nodiscard]] OaMatrix SoftmaxScaledMasked(
	const OaMatrix& InScores, const OaMatrix& InMask, OaF32 InScale);

/// SoftmaxScaledMaskedBwd: backward pass for the fused attention score op.
/// Returns d_scores = softmax_out * (d_out - sum(d_out * softmax_out)) * scale.
[[nodiscard]] OaMatrix SoftmaxScaledMaskedBwd(
	const OaMatrix& InForwardOutput, const OaMatrix& InGradOutput, OaF32 InScale);

/// SoftmaxBwd: Backward pass for Softmax.
/// Computes: d_input = d_output * softmax(x) - softmax(x) * sum(d_output * softmax(x))
/// @param InForwardOutput: Output from forward Softmax pass
/// @param InGradOutput: Gradient flowing back from next layer
/// @return Gradient w.r.t. input
[[nodiscard]] OaMatrix SoftmaxBwd(const OaMatrix& InForwardOutput, const OaMatrix& InGradOutput);

/// TanhBwd: Backward pass for Tanh activation.
/// Computes: d_input = d_output * (1 - tanh(x)^2)
/// @param InForwardOutput: Output from forward Tanh pass
/// @param InGradOutput: Gradient flowing back from next layer
/// @return Gradient w.r.t. input
[[nodiscard]] OaMatrix TanhBwd(const OaMatrix& InForwardOutput, const OaMatrix& InGradOutput);

/// SigmoidBwd: Backward pass for Sigmoid activation.
/// Computes: d_input = d_output * sigmoid(x) * (1 - sigmoid(x))
/// @param InForwardOutput: Output from forward Sigmoid pass
/// @param InGradOutput: Gradient flowing back from next layer
/// @return Gradient w.r.t. input
[[nodiscard]] OaMatrix SigmoidBwd(const OaMatrix& InForwardOutput, const OaMatrix& InGradOutput);

/// LeakyReluBwd: Backward pass for LeakyReLU activation.
/// Computes: d_input = d_output * (x > 0 ? 1 : alpha)
/// @param InForwardOutput: Output from forward LeakyReLU pass
/// @param InGradOutput: Gradient flowing back from next layer
/// @param InAlpha: Negative-slope used in the forward pass (default 0.01)
/// @return Gradient w.r.t. input
[[nodiscard]] OaMatrix LeakyReluBwd(const OaMatrix& InForwardOutput, const OaMatrix& InGradOutput,
	OaF32 InAlpha = 0.01f);

/// EluBwd: Backward pass for ELU activation.
/// Computes: d_input = d_output * (x > 0 ? 1 : alpha * exp(x))
/// @param InForwardOutput: Output from forward ELU pass
/// @param InGradOutput: Gradient flowing back from next layer
/// @param InAlpha: Alpha used in the forward pass (default 1.0)
/// @return Gradient w.r.t. input
[[nodiscard]] OaMatrix EluBwd(const OaMatrix& InForwardOutput, const OaMatrix& InGradOutput,
	OaF32 InAlpha = 1.0f);

/// MishBwd: Backward pass for Mish activation.
/// @param InInput: Forward INPUT x. Required — mish'(x) depends on x directly
///        (tanh(softplus(x)) + x·sigmoid(x)·sech²(softplus(x))), not on y=Mish(x).
/// @param InGradOutput: Gradient flowing back from next layer
/// @return Gradient w.r.t. input
[[nodiscard]] OaMatrix MishBwd(const OaMatrix& InInput, const OaMatrix& InGradOutput);

/// SoftplusBwd: Backward pass for Softplus activation.
/// @param InForwardOutput: Output y = softplus(a) from the forward pass
/// @param InGradOutput: Gradient flowing back from next layer
/// @return Gradient w.r.t. input, dOut * sigmoid(a) = dOut * (1 - e^-y)
[[nodiscard]] OaMatrix SoftplusBwd(const OaMatrix& InForwardOutput, const OaMatrix& InGradOutput);

/// SiluMulBwd: Backward pass for SiluMul activation.
/// @param InInput: Forward INPUT (gate||up). Required — the forward output
///        SiLU(gate)*up is not invertible, so the input must be saved/replayed.
/// @param InGradOutput: Gradient flowing back from next layer (first half used)
/// @return Gradient w.r.t. input (same shape as InInput)
[[nodiscard]] OaMatrix SiluMulBwd(const OaMatrix& InInput, const OaMatrix& InGradOutput);

/// GegluBwd: Backward pass for GEGLU activation.
/// @param InInput: Forward INPUT (up||gate). Required — up*GELU(gate) is not
///        invertible, so the input must be saved/replayed.
/// @param InGradOutput: Gradient flowing back from next layer (first half used)
/// @return Gradient w.r.t. input (same shape as InInput)
[[nodiscard]] OaMatrix GegluBwd(const OaMatrix& InInput, const OaMatrix& InGradOutput);

/// MaxBwd: Backward pass for the full-reduction Max (X → scalar max).
/// Routes the upstream scalar gradient to the element(s) equal to the max:
/// grad_in[i] = (X[i] == max) ? grad_out : 0.
/// @param InInput: Original input X to the forward Max (full shape)
/// @param InMaxValue: Forward Max output (scalar [1])
/// @param InGradOutput: Upstream gradient (scalar [1])
/// @return Gradient w.r.t. input (full shape)
[[nodiscard]] OaMatrix MaxBwd(const OaMatrix& InInput, const OaMatrix& InMaxValue, const OaMatrix& InGradOutput);

// CrossEntropyBwd: REMOVED (broken duplicate, 4 buffers for 3-index shader + wrong push).
// Use OaFnLoss::CrossEntropyBwd instead (correct: 3 buffers, proper push struct).
// The live autograd path already uses OaFnLoss::CrossEntropyBwd.

/// SwigluBwd: Backward pass for SwiGLU activation.
/// @param InGate: Forward gate input
/// @param InUp: Forward up input
/// @param InOut: Forward output
/// @param InGradOutput: Gradient flowing back
/// @return Gradient result struct with GateGrad, UpGrad
[[nodiscard]] OaSwigluBwdResult SwigluBwd(
	const OaMatrix& InGate, const OaMatrix& InUp, const OaMatrix& InOut,
	const OaMatrix& InGradOutput
);

/// GatherBwd: Backward pass for table lookup (embedding scatter-add).
/// Accumulates gradients from each gathered row back into the embedding table.
/// @param InIndices:    [num_indices] u8 or u32 lookup indices used in the forward pass
/// @param InGradOutput: [num_indices, embed_dim] gradients flowing back from next layer
/// @param InVocabSize:  number of rows in the embedding table
/// @param InEmbedDim:   columns per embedding row
/// @return [vocab_size, embed_dim] gradient w.r.t. the embedding table
[[nodiscard]] OaMatrix GatherBwd(
	const OaMatrix& InIndices,
	const OaMatrix& InGradOutput,
	OaI32 InVocabSize,
	OaI32 InEmbedDim
);

// ─── Forward Pass Operations ──────────────────────────────────────

/// Linear: Linear layer (fully connected layer).
/// Computes: output = input @ weight^T + bias
/// @param InX: Input tensor [batch, in_features]
/// @param InWeight: Weight matrix [out_features, in_features]
/// @param InBias: Optional bias vector [out_features]
/// @return Output tensor [batch, out_features]
[[nodiscard]] OaMatrix Linear(
	const OaMatrix& InX,
	const OaMatrix& InWeight,
	const OaMatrix& InBias = OaMatrix{});

/// LinearRelu: Fused linear layer with ReLU activation.
/// Computes: output = ReLU(input @ weight^T + bias)
/// @param InX: Input tensor [batch, in_features]
/// @param InWeight: Weight matrix [out_features, in_features]
/// @param InBias: Bias vector [out_features]
/// @return Output tensor [batch, out_features]
[[nodiscard]] OaMatrix LinearRelu(
	const OaMatrix& InX,
	const OaMatrix& InWeight,
	const OaMatrix& InBias);

/// LinearGelu: Fused linear layer with GELU activation.
/// Computes: output = GELU(input @ weight^T + bias) in a single dispatch.
/// The fused forward discards the pre-activation, so the matching
/// OaGradLinearGelu recomputes it (one GEMM) for GeluBwd in the backward pass.
/// @param InX: Input tensor [batch, in_features]
/// @param InWeight: Weight matrix [out_features, in_features]
/// @param InBias: Bias vector [out_features]
/// @return Output tensor [batch, out_features]
[[nodiscard]] OaMatrix LinearGelu(
	const OaMatrix& InX,
	const OaMatrix& InWeight,
	const OaMatrix& InBias);

/// LinearDataBwd: Backward pass for linear layer (input gradient).
/// Computes: d_input = d_output @ weight^T
/// @param InGradOutput: Gradient w.r.t. layer output [batch, out_features]
/// @param InWeight: Layer weights [out_features, in_features]
/// @return Gradient w.r.t. input [batch, in_features]
[[nodiscard]] OaMatrix LinearDataBwd(const OaMatrix& InGradOutput, const OaMatrix& InWeight);

/// LinearWeightBwd: Backward pass for linear layer (weight gradient).
/// Computes: d_weight = input^T @ d_output
/// @param InInput: Forward pass input [batch, in_features]
/// @param InGradOutput: Gradient w.r.t. layer output [batch, out_features]
/// @return Gradient w.r.t. weights [out_features, in_features]
[[nodiscard]] OaMatrix LinearWeightBwd(const OaMatrix& InInput, const OaMatrix& InGradOutput);

/// LinearWeightBiasBwd: Fused weight and bias gradient for linear layer.
/// Computes: d_weight = input^T @ d_output, d_bias = sum(d_output, dim=0)
/// @param InInput: Forward pass input [batch, in_features]
/// @param InGradOutput: Gradient w.r.t. layer output [batch, out_features]
/// @return GradWeight [out_features, in_features] and GradBias [out_features]
[[nodiscard]] OaLinearWeightBiasBwdResult LinearWeightBiasBwd(const OaMatrix& InInput, const OaMatrix& InGradOutput);

/// LinearDataReluBwd: Fused linear data gradient followed by ReLU backward.
/// Computes: d_input = (d_output @ weight) * (activation > 0)
/// @param InGradOutput: Gradient w.r.t. layer output [batch, out_features]
/// @param InWeight: Forward layer weights [out_features, in_features]
/// @param InActivation: Forward ReLU activation [batch, in_features]
/// @return Gradient w.r.t. ReLU input [batch, in_features]
[[nodiscard]] OaMatrix LinearDataReluBwd(
	const OaMatrix& InGradOutput,
	const OaMatrix& InWeight,
	const OaMatrix& InActivation
);

/// LinearReluBwdData: Fused in-layer LinearRelu(x,W,b) backward, data path.
/// For y = ReLU(x @ W^T + b), computes d_x = (d_y * (act > 0)) @ W in a
/// single dispatch. Gate is applied INSIDE the inner sum (no materialization
/// of d_z), which is the opposite fusion direction of LinearDataReluBwd.
/// @param InGradOutput: Gradient w.r.t. layer output [batch, out_features]
/// @param InWeight: Forward layer weights [out_features, in_features]
/// @param InActivation: Post-ReLU forward output of this layer [batch, out_features]
/// @return Gradient w.r.t. layer input [batch, in_features]
[[nodiscard]] OaMatrix LinearReluBwdData(
	const OaMatrix& InGradOutput,
	const OaMatrix& InWeight,
	const OaMatrix& InActivation);

// ─── Normalization Layers ─────────────────────────────────────────

[[nodiscard]] OaMatrix LayerNorm(
	const OaMatrix& InSelf, const OaMatrix& InWeight, const OaMatrix& InBias,
	OaF32 InEps
);
[[nodiscard]] OaMatrix RmsNorm(
	const OaMatrix& InSelf, const OaMatrix& InWeight, OaF32 InEps
);
[[nodiscard]] OaMatrix RmsNormGated(
	const OaMatrix& InSelf, const OaMatrix& InWeight, const OaMatrix& InBias,
	const OaMatrix& InZ, OaF32 InEps, bool InNormBeforeGate = true
);
[[nodiscard]] OaMatrix HeavyTailActivation(const OaMatrix& InSelf);

/// LayerNormBwd: Backward pass for LayerNorm.
/// @param InX: Forward input
/// @param InWeight: Forward weight
/// @param InBias: Forward bias
/// @param InOut: Forward output
/// @param InMean: Forward mean
/// @param InRstd: Forward reciprocal std
/// @param InGradOutput: Gradient flowing back
/// @return Gradient result struct with DX, DWeight, DBias
[[nodiscard]] OaLayerNormBwdResult LayerNormBwd(
	const OaMatrix& InX, const OaMatrix& InWeight, const OaMatrix& InBias,
	const OaMatrix& InOut, const OaMatrix& InMean, const OaMatrix& InRstd,
	const OaMatrix& InGradOutput
);

/// RmsNormBwd: Backward pass for RmsNorm.
/// @param InX: Forward input
/// @param InWeight: Forward weight
/// @param InOut: Forward output
/// @param InRstd: Forward reciprocal std
/// @param InGradOutput: Gradient flowing back
/// @return Gradient result struct with DX, DWeight
[[nodiscard]] OaRmsNormBwdResult RmsNormBwd(
	const OaMatrix& InX, const OaMatrix& InWeight,
	const OaMatrix& InOut, const OaMatrix& InRstd,
	const OaMatrix& InGradOutput
);

/// RmsNormGatedBwd: backward for RmsNormGated (norm_before_gate = true).
/// Returns grads w.r.t. x, weight, bias, z.
[[nodiscard]] OaRmsNormGatedBwdResult RmsNormGatedBwd(
	const OaMatrix& InX, const OaMatrix& InWeight, const OaMatrix& InBias,
	const OaMatrix& InZ, const OaMatrix& InGradOutput, OaF32 InEps
);

// ChannelNorm: fused LayerNorm over the channel axis of [B,C,T] without
// transposing. Replaces Transpose+LayerNorm+Transpose (3 dispatches) with 1.
// Implementation in Extensions/Private/Ml/FnMatrix/FnMatrixChannelNorm.cpp.
[[nodiscard]] OaMatrix ChannelNorm(
	const OaMatrix& InX, const OaMatrix& InWeight, const OaMatrix& InBias,
	OaI32 InBatch, OaI32 InChannels, OaI32 InSeqLen, OaF32 InEps
);

[[nodiscard]] OaChannelNormBwdResult ChannelNormBwd(
	const OaMatrix& InX, const OaMatrix& InWeight,
	const OaMatrix& InGradOutput,
	OaI32 InBatch, OaI32 InChannels, OaI32 InSeqLen, OaF32 InEps
);

// ChannelNormRelu: fused ChannelNorm + ReLU on [B,C,T]. Saves 1 dispatch per
// site vs ChannelNorm + Relu. Implementation in Extensions.
[[nodiscard]] OaMatrix ChannelNormRelu(
	const OaMatrix& InX, const OaMatrix& InWeight, const OaMatrix& InBias,
	OaI32 InBatch, OaI32 InChannels, OaI32 InSeqLen, OaF32 InEps
);

[[nodiscard]] OaChannelNormBwdResult ChannelNormReluBwd(
	const OaMatrix& InX, const OaMatrix& InWeight, const OaMatrix& InFwdOut,
	const OaMatrix& InGradOutput,
	OaI32 InBatch, OaI32 InChannels, OaI32 InSeqLen, OaF32 InEps
);

// Conv1dGemm: 1-D convolution executed as im2col + a single matmul, so the heavy
// work runs on the tensor-core GEMM stack (bf16 CmSg/CmWg or fp32 GemmTiled by
// engine precision). This is THE 1-D conv path: OaConv1d::Forward and every
// tokenizer route through it; the scalar direct-conv kernel was retired.
// Implementation in Extensions/.../FnMatrixConv1dGemm.cpp.
//   InX [N, InC, L], InWeight [OutC, InC, K], InBias [OutC] -> [N, OutC, OutL]
[[nodiscard]] OaMatrix Conv1dGemm(
	const OaMatrix& InX, const OaMatrix& InWeight, const OaMatrix& InBias,
	OaI32 InStride = 1, OaI32 InPadding = 0, OaI32 InDilation = 1
);

// Conv1dReluGemm: Conv1dGemm with the ReLU folded into the GEMM bias epilogue.
[[nodiscard]] OaMatrix Conv1dReluGemm(
	const OaMatrix& InX, const OaMatrix& InWeight, const OaMatrix& InBias,
	OaI32 InStride = 1, OaI32 InPadding = 0, OaI32 InDilation = 1
);

// Im2Col1d: unfold a 1-D conv input [N, InC, L] into the GEMM-ready column
// matrix [N*OutL, InC*K] (row = n*OutL+ol, col = ic*K+k). Differentiable
// (backward = Col2Im1d). The building block of Conv1dGemm.
[[nodiscard]] OaMatrix Im2Col1d(
	const OaMatrix& InX, OaI32 InK, OaI32 InStride, OaI32 InPadding, OaI32 InDilation = 1
);

// Col2Im1d: backward of Im2Col1d — fold a column-matrix gradient [N*OutL, InC*K]
// back into input shape [N, InC, L], accumulating over overlapping windows.
[[nodiscard]] OaMatrix Col2Im1d(
	const OaMatrix& InDCols, OaI32 InN, OaI32 InC, OaI32 InL, OaI32 InK,
	OaI32 InStride, OaI32 InPadding, OaI32 InDilation, OaI32 InOutL
);

/// MaxPool2dBwd: Backward pass for 2D max pooling.
/// @param InX: Forward input (provides [N,C,H,W] and the gradInput shape)
/// @param InIndices: Argmax flat-input-indices from the forward pass
/// @param InGradOutput: Gradient flowing back (provides out [H,W])
/// @param InKernelSize, InStride, InPadding: pooling params (needed to walk the window)
/// @return Gradient w.r.t. input
[[nodiscard]] OaMatrix MaxPool2dBwd(
	const OaMatrix& InX, const OaMatrix& InIndices, const OaMatrix& InGradOutput,
	OaI32 InKernelSize, OaI32 InStride, OaI32 InPadding
);

/// AvgPool2dBwd: Backward pass for 2D average pooling.
/// @param InX: Forward input
/// @param InGradOutput: Gradient flowing back
/// @param InKernelSize: Pooling kernel size
/// @param InStride: Pooling stride
/// @param InPadding: Pooling padding
/// @return Gradient w.r.t. input
[[nodiscard]] OaMatrix AvgPool2dBwd(const OaMatrix& InX, const OaMatrix& InGradOutput, OaI32 InKernelSize, OaI32 InStride, OaI32 InPadding);

// Softmax/LogSoftmax are in Core/FnMatrix.h (reduction operations)

// ─── Batched linear algebra ───────────────────────────────────────
// Bmm: per-batch matrix multiply, A[N,M,K] @ B[N,K,P] = out[N,M,P] (out[n]=A[n]@B[n]).
// Differentiable (OaGradBmm: dA=dOut@Bᵀ, dB=Aᵀ@dOut). The enabler for differentiable
// forward kinematics (per-frame 3x3 rotation composition) and any per-row small-matrix
// algebra. F32.
[[nodiscard]] OaMatrix Bmm(const OaMatrix& InA, const OaMatrix& InB);

/// Attention head layout transforms, both fully materialized on GPU.
/// SplitHeads: [B*S,D] -> [B*H,S,D/H]. MergeHeads is the exact inverse.
[[nodiscard]] OaMatrix SplitHeads(
	const OaMatrix& InX, OaI32 InBatch, OaI32 InSeqLen, OaI32 InNumHeads);
[[nodiscard]] OaMatrix MergeHeads(
	const OaMatrix& InX, OaI32 InBatch, OaI32 InSeqLen, OaI32 InNumHeads);

// ─── Neural Network Layers ────────────────────────────────────────

// BiasAdd is now generated - see FnMatrixEmbed.gen.h
// (Scalar Conv1d forward retired — use Conv1dGemm above. Conv1dBwdData/Conv1dBwdWeight
//  below survive as the adjoints backing OaConvTranspose1d.)

/// Conv1dBwdData: Backward pass for 1D convolution (input gradient).
/// Computes: d_input = d_output conv_transpose(weight)
[[nodiscard]] OaMatrix Conv1dBwdData(
	const OaMatrix& InDOut,
	const OaMatrix& InWeight,
	OaU32 InStride,
	OaU32 InPadding,
	OaU32 InDilation,
	const OaMatrixShape& InInputShape
);

struct OaConv1dBwdWeightResult {
	OaMatrix GradWeight;
	OaMatrix GradBias;
};

/// Conv1dBwdWeight: Fused weight and bias gradient for 1D convolution.
/// Computes: d_weight = input conv_transpose(d_output), d_bias = sum(d_output)
[[nodiscard]] OaConv1dBwdWeightResult Conv1dBwdWeight(
	const OaMatrix& InInput,
	const OaMatrix& InDOut,
	const OaMatrix& InWeight,
	OaU32 InStride,
	OaU32 InPadding,
	OaU32 InDilation
);

[[nodiscard]] OaMatrix Conv2d(
	const OaMatrix& InX, const OaMatrix& InWeight, const OaMatrix& InBias,
	OaU32 InStride, OaU32 InPadding, OaU32 InGroups = 1
);

/// Conv2dBwdData: Backward pass for 2D convolution (input gradient).
/// Computes: d_input = d_output conv_transpose(weight)
[[nodiscard]] OaMatrix Conv2dBwdData(
	const OaMatrix& InDOut,
	const OaMatrix& InWeight,
	OaU32 InStride,
	OaU32 InPadding,
	const OaMatrixShape& InInputShape,
	OaU32 InGroups = 1
);

struct OaConv2dBwdWeightResult {
	OaMatrix GradWeight;
	OaMatrix GradBias;
};

/// Conv2dBwdWeight: Fused weight and bias gradient for 2D convolution.
/// Computes: d_weight = input conv_transpose(d_output), d_bias = sum(d_output)
[[nodiscard]] OaConv2dBwdWeightResult Conv2dBwdWeight(
	const OaMatrix& InInput,
	const OaMatrix& InDOut,
	const OaMatrix& InWeight,
	OaU32 InStride,
	OaU32 InPadding,
	OaU32 InGroups = 1
);

/// ConvTranspose2d: 2D transposed convolution (learnable upsampling).
/// Input: [N, InC, H, W], Weight: [InC, OutC, K, K], Bias: [OutC]
/// Output: [N, OutC, H_out, W_out] where H_out = (H - 1) * S - 2P + K.
[[nodiscard]] OaMatrix ConvTranspose2d(
	const OaMatrix& InX, const OaMatrix& InWeight, const OaMatrix& InBias,
	OaU32 InStride, OaU32 InPadding
);

/// ConvTranspose2dBwdData: Backward pass for 2D transposed convolution (input gradient).
/// Computes: d_input = conv2d(d_output, weight).
[[nodiscard]] OaMatrix ConvTranspose2dBwdData(
	const OaMatrix& InDOut,
	const OaMatrix& InWeight,
	OaU32 InStride,
	OaU32 InPadding,
	const OaMatrixShape& InInputShape
);

struct OaConvTranspose2dBwdWeightResult {
	OaMatrix GradWeight;
	OaMatrix GradBias;
};

/// ConvTranspose2dBwdWeight: Fused weight and bias gradient for 2D transposed convolution.
[[nodiscard]] OaConvTranspose2dBwdWeightResult ConvTranspose2dBwdWeight(
	const OaMatrix& InInput,
	const OaMatrix& InDOut,
	const OaMatrix& InWeight,
	OaU32 InStride,
	OaU32 InPadding
);

// ─── Pooling Operations ───────────────────────────────────────────

[[nodiscard]] OaMatrix AvgPool2d(
	const OaMatrix& InX, OaI32 InKernelSize, OaI32 InStride, OaI32 InPadding
);

struct OaMaxPool2dResult {
	OaMatrix Out;      // [N, C, H_out, W_out] pooled output
	OaMatrix Indices;  // [N, C, H_out, W_out] uint32 argmax indices (for backward pass)
};

[[nodiscard]] OaMaxPool2dResult MaxPool2d(
	const OaMatrix& InX, OaI32 InKernelSize, OaI32 InStride, OaI32 InPadding
);

// ─── Quantization Operations (llama.cpp compatibility) ─────────────

/// QuantizeQ4_0: Quantize FP32/BF16 to Q4_0 (llama.cpp format).
/// @param InInput: Input tensor [N] of FP32/BF16
/// @param InScale: Per-block scale tensor [num_blocks] of FP32
/// @return Packed Q4_0 tensor [num_blocks * 16] of UInt8
[[nodiscard]] OaMatrix QuantizeQ4_0(const OaMatrix& InInput, const OaMatrix& InScale);

/// DequantizeQ4_0: Dequantize Q4_0 to FP32/BF16 (llama.cpp format).
/// @param InInput: Packed Q4_0 tensor [num_blocks * 16] of UInt8
/// @param InScale: Per-block scale tensor [num_blocks] of FP32
/// @param InCount: Number of elements to dequantize
/// @return FP32 tensor [InCount]
[[nodiscard]] OaMatrix DequantizeQ4_0(const OaMatrix& InInput, const OaMatrix& InScale, OaI64 InCount);

/// ComputeScaleQ4_0: Compute per-block scale for Q4_0 quantization.
/// @param InInput: Input tensor [N] of FP32/BF16
/// @return Per-block scale tensor [num_blocks] of FP32
[[nodiscard]] OaMatrix ComputeScaleQ4_0(const OaMatrix& InInput);

struct OaBatchNorm2dBwdResult {
	OaMatrix DX;
	OaMatrix DGamma;
	OaMatrix DBias;
};

/// SliceBwd: Backward pass for Slice.
/// Creates a zero tensor of InInputShape and copies InDOut into [InStart, InEnd).
[[nodiscard]] OaMatrix SliceBwd(OaMatrixShape InInputShape, OaI32 InDim, OaI64 InStart, OaI64 InEnd, const OaMatrix& InDOut);

/// UpsampleBwd: Backward pass for Upsample (Nearest or Bilinear).
[[nodiscard]] OaMatrix UpsampleBwd(const OaMatrix& InInput, const OaMatrix& InDOut, OaI32 InScaleFactor, bool InIsBilinear);

/// BatchNorm2dBwd: Backward pass for BatchNorm2d.
[[nodiscard]] OaBatchNorm2dBwdResult BatchNorm2dBwd(
	const OaMatrix& InX, const OaMatrix& InGamma, const OaMatrix& InBeta,
	const OaMatrix& InMean, const OaMatrix& InVar, const OaMatrix& InOut,
	const OaMatrix& InDOut, OaF32 InEps, bool InIsTraining);

// Optimizer operations moved to Ml/FnOptim.h (use OaFnOptim::AdamWStep, etc.)
// Stochastic rounding operations removed (legacy buffer-level API)

} // namespace OaFnMatrix
