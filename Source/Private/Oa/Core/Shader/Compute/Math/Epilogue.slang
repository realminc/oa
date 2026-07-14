// Epilogue — Reusable epilogue operations for fused GEMM kernels
//
// Provides common post-GEMM operations that can be fused into the final
// write-back stage to eliminate separate kernel launches.
//
// Import: `import Math;` or `import Epilogue;`
// Requires: Storage module for OaLoad/OaStore

module Epilogue;

import Storage;

// ============================================================================
// Basic Epilogues
// ============================================================================

/// Store result with no transformation
/// C[idx] = acc
[ForceInline]
public void EpilogueStore(
	RWByteAddressBuffer C,
	uint idx,
	float acc)
{
	OaStore(C, idx, acc);
}

/// Store result with accumulation (for gradient accumulation)
/// C[idx] += acc
[ForceInline]
public void EpilogueAccumulate(
	RWByteAddressBuffer C,
	uint idx,
	float acc)
{
	float existing = OaLoad(C, idx);
	OaStore(C, idx, existing + acc);
}

/// Store result with scaling
/// C[idx] = alpha * acc
[ForceInline]
public void EpilogueScale(
	RWByteAddressBuffer C,
	uint idx,
	float acc,
	float alpha)
{
	OaStore(C, idx, alpha * acc);
}

// ============================================================================
// Bias Epilogues
// ============================================================================

/// Fused GEMM + Bias
/// C[idx] = acc + bias[col]
[ForceInline]
public void EpilogueBias(
	RWByteAddressBuffer C,
	RWByteAddressBuffer bias,
	uint idx,
	uint col,
	float acc)
{
	float b = OaLoad(bias, col);
	OaStore(C, idx, acc + b);
}

/// Fused GEMM + Bias + ReLU
/// C[idx] = max(0, acc + bias[col])
[ForceInline]
public void EpilogueBiasRelu(
	RWByteAddressBuffer C,
	RWByteAddressBuffer bias,
	uint idx,
	uint col,
	float acc)
{
	float b = OaLoad(bias, col);
	float result = max(0.0f, acc + b);
	OaStore(C, idx, result);
}

/// Fused GEMM + Bias + GELU (approximate)
/// C[idx] = GELU(acc + bias[col])
/// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
[ForceInline]
public void EpilogueBiasGelu(
	RWByteAddressBuffer C,
	RWByteAddressBuffer bias,
	uint idx,
	uint col,
	float acc)
{
	float b = OaLoad(bias, col);
	float x = acc + b;

	// GELU approximation
	float x3 = x * x * x;
	float inner = 0.7978845608f * (x + 0.044715f * x3);  // sqrt(2/π) ≈ 0.7978845608
	float result = 0.5f * x * (1.0f + tanh(inner));

	OaStore(C, idx, result);
}

/// Fused GEMM + Bias + SiLU (Swish)
/// C[idx] = SiLU(acc + bias[col])
/// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
[ForceInline]
public void EpilogueBiasSilu(
	RWByteAddressBuffer C,
	RWByteAddressBuffer bias,
	uint idx,
	uint col,
	float acc)
{
	float b = OaLoad(bias, col);
	float x = acc + b;

	// SiLU = x * sigmoid(x)
	float sigmoid = 1.0f / (1.0f + exp(-x));
	float result = x * sigmoid;

	OaStore(C, idx, result);
}

// ============================================================================
// Residual Epilogues
// ============================================================================

/// Fused GEMM + Residual
/// C[idx] = acc + residual[idx]
[ForceInline]
public void EpilogueResidual(
	RWByteAddressBuffer C,
	RWByteAddressBuffer residual,
	uint idx,
	float acc)
{
	float res = OaLoad(residual, idx);
	OaStore(C, idx, acc + res);
}

/// Fused GEMM + Bias + Residual
/// C[idx] = acc + bias[col] + residual[idx]
[ForceInline]
public void EpilogueBiasResidual(
	RWByteAddressBuffer C,
	RWByteAddressBuffer bias,
	RWByteAddressBuffer residual,
	uint idx,
	uint col,
	float acc)
{
	float b = OaLoad(bias, col);
	float res = OaLoad(residual, idx);
	OaStore(C, idx, acc + b + res);
}

/// Fused GEMM + Bias + Residual + ReLU
/// C[idx] = max(0, acc + bias[col] + residual[idx])
[ForceInline]
public void EpilogueBiasResidualRelu(
	RWByteAddressBuffer C,
	RWByteAddressBuffer bias,
	RWByteAddressBuffer residual,
	uint idx,
	uint col,
	float acc)
{
	float b = OaLoad(bias, col);
	float res = OaLoad(residual, idx);
	float result = max(0.0f, acc + b + res);
	OaStore(C, idx, result);
}

// ============================================================================
// Normalization Epilogues
// ============================================================================

/// Fused GEMM + LayerNorm (simplified, per-row)
/// Requires pre-computed mean and variance for the row
/// C[idx] = (acc - mean) / sqrt(var + eps)
[ForceInline]
public void EpilogueLayerNorm(
	RWByteAddressBuffer C,
	uint idx,
	float acc,
	float mean,
	float var,
	float eps)
{
	float normalized = (acc - mean) / sqrt(var + eps);
	OaStore(C, idx, normalized);
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Fast tanh approximation for GELU
/// tanh(x) ≈ x * (27 + x²) / (27 + 9x²) for |x| < 3
public float FastTanh(float x) {
	// Clamp to avoid overflow
	x = clamp(x, -3.0f, 3.0f);
	float x2 = x * x;
	return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

/// Fast sigmoid approximation
/// sigmoid(x) ≈ 0.5 + 0.5 * tanh(x/2)
public float FastSigmoid(float x) {
	return 0.5f + 0.5f * FastTanh(0.5f * x);
}