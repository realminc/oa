#pragma once

// Quantization Block Structures
//
// Block-wise quantization formats from llama.cpp/GGML.
// Each format packs weights into fixed-size blocks with scales/zero-points.
//
// Reference: llama.cpp/ggml/src/ggml-common.h
//
// Naming convention: block_qX_Y where:
//   - X = bits per weight (1, 2, 4, 5, 6, 8)
//   - Y = variant (0, 1, K for K-quants)
//
// K-quants use super-blocks of 256 elements with hierarchical scales.

#include <Oa/Core/Types.h>

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

// QK_K = super-block size for K-quants (256 elements)
constexpr OaU32 QK_K = 256;
constexpr OaU32 K_SCALE_SIZE = 12;

// Basic quant block sizes
constexpr OaU32 QK4_0 = 32;
constexpr OaU32 QK4_1 = 32;
constexpr OaU32 QK5_0 = 32;
constexpr OaU32 QK5_1 = 32;
constexpr OaU32 QK8_0 = 32;
constexpr OaU32 QK8_1 = 32;

// ─────────────────────────────────────────────────────────────────────────────
// Basic Quantization Blocks (32 elements per block)
// ─────────────────────────────────────────────────────────────────────────────

// Q4_0: 4-bit quantization, 32 elements per block
// weight = d * q (symmetric, no zero-point)
// 4.5 bits per weight
struct OaBlockQ4_0 {
	OaU16 d;              // delta (FP16)
	OaU8 qs[QK4_0 / 2];   // nibbles (4-bit quants, 2 per byte)
};
static_assert(sizeof(OaBlockQ4_0) == sizeof(OaU16) + QK4_0 / 2, "wrong q4_0 block size");

// Q4_1: 4-bit quantization with min, 32 elements per block
// weight = d * q + m (asymmetric)
// 5.0 bits per weight
struct OaBlockQ4_1 {
	OaU16 d;              // delta (FP16)
	OaU16 m;              // min (FP16)
	OaU8 qs[QK4_1 / 2];   // nibbles
};
static_assert(sizeof(OaBlockQ4_1) == 2 * sizeof(OaU16) + QK4_1 / 2, "wrong q4_1 block size");

// Q5_0: 5-bit quantization, 32 elements per block
// weight = d * q (symmetric)
// 5.5 bits per weight
struct OaBlockQ5_0 {
	OaU16 d;              // delta (FP16)
	OaU8 qh[4];           // high bits (5th bit of each quant)
	OaU8 qs[QK5_0 / 2];   // low 4 bits (nibbles)
};
static_assert(sizeof(OaBlockQ5_0) == sizeof(OaU16) + sizeof(OaU32) + QK5_0 / 2, "wrong q5_0 block size");

// Q5_1: 5-bit quantization with min, 32 elements per block
// weight = d * q + m (asymmetric)
// 6.0 bits per weight
struct OaBlockQ5_1 {
	OaU16 d;              // delta (FP16)
	OaU16 m;              // min (FP16)
	OaU8 qh[4];           // high bits
	OaU8 qs[QK5_1 / 2];   // low 4 bits
};
static_assert(sizeof(OaBlockQ5_1) == 2 * sizeof(OaU16) + sizeof(OaU32) + QK5_1 / 2, "wrong q5_1 block size");

// Q8_0: 8-bit quantization, 32 elements per block
// weight = d * q (symmetric)
// 8.5 bits per weight
struct OaBlockQ8_0 {
	OaU16 d;              // delta (FP16)
	OaI8 qs[QK8_0];       // quants (signed 8-bit)
};
static_assert(sizeof(OaBlockQ8_0) == sizeof(OaU16) + QK8_0, "wrong q8_0 block size");

// Q8_1: 8-bit quantization with sum, 32 elements per block
// weight = d * q, with precomputed sum for dot products
// 9.0 bits per weight
struct OaBlockQ8_1 {
	OaU16 d;              // delta (FP16)
	OaU16 s;              // d * sum(qs[i]) (FP16)
	OaI8 qs[QK8_1];       // quants (signed 8-bit)
};
static_assert(sizeof(OaBlockQ8_1) == 2 * sizeof(OaU16) + QK8_1, "wrong q8_1 block size");

// ─────────────────────────────────────────────────────────────────────────────
// K-Quants (256-element super-blocks with hierarchical scales)
// ─────────────────────────────────────────────────────────────────────────────

// Q2_K: 2-bit quantization
// 16 blocks of 16 elements each
// weight = a * q + b
// Effectively 2.625 bits per weight
struct OaBlockQ2_K {
	OaU8 scales[QK_K / 16];  // scales and mins, quantized with 4 bits
	OaU8 qs[QK_K / 4];       // quants (2 bits per element)
	OaU16 d;                 // super-block scale for quantized scales (FP16)
	OaU16 dmin;              // super-block scale for quantized mins (FP16)
};
static_assert(sizeof(OaBlockQ2_K) == 2 * sizeof(OaU16) + QK_K / 16 + QK_K / 4, "wrong q2_K block size");

// Q3_K: 3-bit quantization
// 16 blocks of 16 elements each
// weight = a * q
// Effectively 3.4375 bits per weight
struct OaBlockQ3_K {
	OaU8 hmask[QK_K / 8];    // quants - high bit
	OaU8 qs[QK_K / 4];       // quants - low 2 bits
	OaU8 scales[12];         // scales, quantized with 6 bits
	OaU16 d;                 // super-block scale (FP16)
};
static_assert(sizeof(OaBlockQ3_K) == sizeof(OaU16) + QK_K / 4 + QK_K / 8 + 12, "wrong q3_K block size");

// Q4_K: 4-bit quantization
// 8 blocks of 32 elements each
// weight = a * q + b
// Effectively 4.5 bits per weight
struct OaBlockQ4_K {
	OaU16 d;                      // super-block scale for quantized scales (FP16)
	OaU16 dmin;                   // super-block scale for quantized mins (FP16)
	OaU8 scales[K_SCALE_SIZE];    // scales and mins, quantized with 6 bits
	OaU8 qs[QK_K / 2];            // 4-bit quants (2 per byte)
};
static_assert(sizeof(OaBlockQ4_K) == 2 * sizeof(OaU16) + K_SCALE_SIZE + QK_K / 2, "wrong q4_K block size");

// Q5_K: 5-bit quantization
// 8 blocks of 32 elements each
// weight = a * q + b
// Effectively 5.5 bits per weight
struct OaBlockQ5_K {
	OaU16 d;                      // super-block scale for quantized scales (FP16)
	OaU16 dmin;                   // super-block scale for quantized mins (FP16)
	OaU8 scales[K_SCALE_SIZE];    // scales and mins, quantized with 6 bits
	OaU8 qh[QK_K / 8];            // quants, high bit
	OaU8 qs[QK_K / 2];            // quants, low 4 bits
};
static_assert(sizeof(OaBlockQ5_K) == 2 * sizeof(OaU16) + K_SCALE_SIZE + QK_K / 2 + QK_K / 8, "wrong q5_K block size");

// Q6_K: 6-bit quantization
// 16 blocks of 16 elements each
// weight = a * q
// Effectively 6.5625 bits per weight
struct OaBlockQ6_K {
	OaU8 ql[QK_K / 2];       // quants, lower 4 bits
	OaU8 qh[QK_K / 4];       // quants, upper 2 bits
	OaI8 scales[QK_K / 16];  // scales, quantized with 8 bits
	OaU16 d;                 // super-block scale (FP16)
};
static_assert(sizeof(OaBlockQ6_K) == sizeof(OaU16) + QK_K / 16 + 3 * QK_K / 4, "wrong q6_K block size");

// Q8_K: 8-bit quantization (K-quant variant)
// Used for intermediate quantization and dot products
// 256 elements per block
struct OaBlockQ8_K {
	OaU16 d;                 // delta (FP16)
	OaI8 qs[QK_K];           // quants (signed 8-bit)
	OaI16 bsums[QK_K / 16];  // sum of quants in blocks of 16
};
static_assert(sizeof(OaBlockQ8_K) == sizeof(OaU16) + QK_K + QK_K / 16 * sizeof(OaI16), "wrong q8_K block size");

// ─────────────────────────────────────────────────────────────────────────────
// Helper Functions
// ─────────────────────────────────────────────────────────────────────────────

// Get block size in elements for each quantization type
[[nodiscard]] constexpr OaU32 OaQuantBlockSize(OaScalarType InType) {
	switch (InType) {
		case OaScalarType::Q4_0:  return QK4_0;
		case OaScalarType::Q4_1:  return QK4_1;
		case OaScalarType::Q5_0:  return QK5_0;
		case OaScalarType::Q5_1:  return QK5_1;
		case OaScalarType::Q8_0:  return QK8_0;
		case OaScalarType::Q8_1:  return QK8_1;
		case OaScalarType::Q2_K:  return QK_K;
		case OaScalarType::Q3_K:  return QK_K;
		case OaScalarType::Q4_K:  return QK_K;
		case OaScalarType::Q5_K:  return QK_K;
		case OaScalarType::Q6_K:  return QK_K;
		case OaScalarType::Q8_K:  return QK_K;
		default: return 1;
	}
}

// Get block size in bytes for each quantization type
[[nodiscard]] constexpr OaU32 OaQuantBlockBytes(OaScalarType InType) {
	switch (InType) {
		case OaScalarType::Q4_0:  return sizeof(OaBlockQ4_0);
		case OaScalarType::Q4_1:  return sizeof(OaBlockQ4_1);
		case OaScalarType::Q5_0:  return sizeof(OaBlockQ5_0);
		case OaScalarType::Q5_1:  return sizeof(OaBlockQ5_1);
		case OaScalarType::Q8_0:  return sizeof(OaBlockQ8_0);
		case OaScalarType::Q8_1:  return sizeof(OaBlockQ8_1);
		case OaScalarType::Q2_K:  return sizeof(OaBlockQ2_K);
		case OaScalarType::Q3_K:  return sizeof(OaBlockQ3_K);
		case OaScalarType::Q4_K:  return sizeof(OaBlockQ4_K);
		case OaScalarType::Q5_K:  return sizeof(OaBlockQ5_K);
		case OaScalarType::Q6_K:  return sizeof(OaBlockQ6_K);
		case OaScalarType::Q8_K:  return sizeof(OaBlockQ8_K);
		default: return 0;
	}
}

// Check if a scalar type is a quantized format
[[nodiscard]] constexpr bool OaIsQuantized(OaScalarType InType) {
	return OaQuantBlockBytes(InType) > 0;
}

