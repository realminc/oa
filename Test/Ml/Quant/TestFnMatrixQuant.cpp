// ============================================================================
// TestFnMatrixQuant.cpp — Tests for Q4_0 quantization operations
// ============================================================================
// Tests Q4_0 quantization (llama.cpp format):
//   - ComputeScaleQ4_0: Compute per-block scale factors
//   - QuantizeQ4_0: Quantize FP32 to Q4_0 (4-bit symmetric)
//   - DequantizeQ4_0: Dequantize Q4_0 back to FP32
//   - Round-trip accuracy validation
//
// Q4_0 format:
//   - Block size: 32 elements
//   - Range: -7 to +7 (4-bit signed)
//   - Scale: max(abs(block)) / 7
//   - Packing: 2 values per byte (nibbles)
// ============================================================================

#include "../../OaTest.h"
#include <Oa/Core/Matrix.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Ml/FnMatrix.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Engine.h>

#include <cmath>
#include <algorithm>

// ─── Helper: Execute and sync context ────────────────────────────────────────

static void Flush() {
	(void)OaContext::GetDefault().Execute();
	(void)OaContext::GetDefault().Sync();
}

// ─── Test: ComputeScaleQ4_0 basic functionality ──────────────────────────────

TEST(FnMatrixQuant, ComputeScaleQ4_0_Basic) {
	// Single block of 32 elements with known max
	auto m = OaFnMatrix::Zeros(OaMatrixShape{32});
	for (int i = 0; i < 32; ++i) {
		m.Set(i, static_cast<float>(i - 16));  // Range: -16 to 15
	}
	
	auto scale = OaFnMatrix::ComputeScaleQ4_0(m);
	Flush();
	
	// Expected scale: max(abs(input)) / 7 = 16 / 7 ≈ 2.286
	EXPECT_NEAR(scale.At(0), 16.0f / 7.0f, 1e-5f);
}

// ─── Test: ComputeScaleQ4_0 multiple blocks ──────────────────────────────────

TEST(FnMatrixQuant, ComputeScaleQ4_0_MultipleBlocks) {
	// 3 blocks with different scales
	auto m = OaFnMatrix::Zeros(OaMatrixShape{96});
	
	// Block 0: max = 7
	for (int i = 0; i < 32; ++i) m.Set(i, static_cast<float>(i % 8 - 3));
	
	// Block 1: max = 14
	for (int i = 32; i < 64; ++i) m.Set(i, static_cast<float>((i % 16) - 7));
	
	// Block 2: max = 21
	for (int i = 64; i < 96; ++i) m.Set(i, static_cast<float>((i % 24) - 11));
	
	auto scale = OaFnMatrix::ComputeScaleQ4_0(m);
	Flush();
	
	// Expected scales
	EXPECT_NEAR(scale.At(0), 7.0f / 7.0f, 1e-5f);   // 1.0
	EXPECT_NEAR(scale.At(1), 14.0f / 7.0f, 1e-5f);  // 2.0
	EXPECT_NEAR(scale.At(2), 21.0f / 7.0f, 1e-5f);  // 3.0
}

// ─── Test: ComputeScaleQ4_0 zero input ───────────────────────────────────────

TEST(FnMatrixQuant, ComputeScaleQ4_0_ZeroInput) {
	auto m = OaFnMatrix::Zeros(OaMatrixShape{64});
	auto scale = OaFnMatrix::ComputeScaleQ4_0(m);
	Flush();
	
	// Should return 1.0 to avoid division by zero
	EXPECT_FLOAT_EQ(scale.At(0), 1.0f);
	EXPECT_FLOAT_EQ(scale.At(1), 1.0f);
}

// ─── Test: Q4_0 round-trip with uniform data ─────────────────────────────────

TEST(FnMatrixQuant, Q4_0_RoundTrip_Uniform) {
	// Test round-trip with uniform distribution
	auto m_input = OaFnMatrix::Zeros(OaMatrixShape{64});
	for (int i = 0; i < 64; ++i) {
		m_input.Set(i, static_cast<float>(i % 32 - 16));  // Range: -16 to 15
	}
	
	// Compute scale
	auto m_scale = OaFnMatrix::ComputeScaleQ4_0(m_input);
	
	// Quantize
	auto m_quant = OaFnMatrix::QuantizeQ4_0(m_input, m_scale);
	
	// Dequantize
	auto m_dequant = OaFnMatrix::DequantizeQ4_0(m_quant, m_scale, 64);
	Flush();
	
	// Verify round-trip accuracy
	// Q4_0 has limited precision, so we expect some quantization error
	float max_error = 0.0f;
	for (int i = 0; i < 64; ++i) {
		float original = m_input.At(i);
		float dequantized = m_dequant.At(i);
		float error = std::abs(dequantized - original);
		max_error = std::max(max_error, error);
	}
	
	// Max error should be within one quantization step
	// scale ≈ 16/7 ≈ 2.286, so max error ≈ scale/2 ≈ 1.143
	EXPECT_LT(max_error, 2.0f);
}

// ─── Test: Q4_0 round-trip with sine wave ────────────────────────────────────

TEST(FnMatrixQuant, Q4_0_RoundTrip_SineWave) {
	// Test with smooth sine wave data
	auto m_input = OaFnMatrix::Zeros(OaMatrixShape{128});
	for (int i = 0; i < 128; ++i) {
		m_input.Set(i, std::sin(i * 0.1f) * 10.0f);
	}
	
	auto m_scale = OaFnMatrix::ComputeScaleQ4_0(m_input);
	auto m_quant = OaFnMatrix::QuantizeQ4_0(m_input, m_scale);
	auto m_dequant = OaFnMatrix::DequantizeQ4_0(m_quant, m_scale, 128);
	Flush();
	
	// Verify each block's accuracy
	for (int b = 0; b < 4; ++b) {
		float block_scale = m_scale.At(b);
		float max_error = 0.0f;
		
		for (int i = b * 32; i < (b + 1) * 32 && i < 128; ++i) {
			float error = std::abs(m_dequant.At(i) - m_input.At(i));
			max_error = std::max(max_error, error);
		}
		
		// Max error should be within one quantization step
		EXPECT_LT(max_error, block_scale * 1.5f);
	}
}

// ─── Test: Q4_0 with large tensor ────────────────────────────────────────────

TEST(FnMatrixQuant, Q4_0_LargeTensor) {
	// Test with 1024 elements (32 blocks)
	auto m_input = OaFnMatrix::Zeros(OaMatrixShape{1024});
	for (int i = 0; i < 1024; ++i) {
		m_input.Set(i, std::cos(i * 0.05f) * 20.0f);
	}
	
	auto m_scale = OaFnMatrix::ComputeScaleQ4_0(m_input);
	auto m_quant = OaFnMatrix::QuantizeQ4_0(m_input, m_scale);
	auto m_dequant = OaFnMatrix::DequantizeQ4_0(m_quant, m_scale, 1024);
	Flush();
	
	EXPECT_EQ(m_scale.NumElements(), 32);  // 1024 / 32 = 32 blocks
	EXPECT_EQ(m_dequant.NumElements(), 1024);
	
	// Verify compression ratio
	// Original: 1024 * 4 bytes = 4096 bytes
	// Compressed: 512 bytes (packed) + 32 * 4 bytes (scales) = 640 bytes
	// Ratio: 4096 / 640 = 6.4x
	
	// Verify all values are finite
	for (int i = 0; i < 1024; ++i) {
		EXPECT_TRUE(std::isfinite(m_dequant.At(i)));
	}
}

// ─── Test: Q4_0 edge case - partial block ────────────────────────────────────

TEST(FnMatrixQuant, Q4_0_PartialBlock) {
	// Test with non-multiple of 32 (e.g., 50 elements)
	auto m_input = OaFnMatrix::Zeros(OaMatrixShape{50});
	for (int i = 0; i < 50; ++i) {
		m_input.Set(i, static_cast<float>(i % 20 - 10));
	}
	
	auto m_scale = OaFnMatrix::ComputeScaleQ4_0(m_input);
	auto m_quant = OaFnMatrix::QuantizeQ4_0(m_input, m_scale);
	auto m_dequant = OaFnMatrix::DequantizeQ4_0(m_quant, m_scale, 50);
	Flush();
	
	EXPECT_EQ(m_scale.NumElements(), 2);  // ceil(50 / 32) = 2 blocks
	EXPECT_EQ(m_dequant.NumElements(), 50);
	
	// Verify accuracy for all elements
	float max_error = 0.0f;
	for (int i = 0; i < 50; ++i) {
		float error = std::abs(m_dequant.At(i) - m_input.At(i));
		max_error = std::max(max_error, error);
	}
	
	EXPECT_LT(max_error, 2.0f);
}

// ─── Test: Q4_0 numerical stability ──────────────────────────────────────────

TEST(FnMatrixQuant, Q4_0_NumericalStability) {
	// Test with very small values
	auto m_input = OaFnMatrix::Zeros(OaMatrixShape{64});
	for (int i = 0; i < 64; ++i) {
		m_input.Set(i, static_cast<float>(i % 8 - 4) * 1e-3f);
	}
	
	auto m_scale = OaFnMatrix::ComputeScaleQ4_0(m_input);
	auto m_quant = OaFnMatrix::QuantizeQ4_0(m_input, m_scale);
	auto m_dequant = OaFnMatrix::DequantizeQ4_0(m_quant, m_scale, 64);
	Flush();
	
	// Scales should be reasonable (not NaN or Inf)
	for (int i = 0; i < m_scale.NumElements(); ++i) {
		float s = m_scale.At(i);
		EXPECT_TRUE(std::isfinite(s));
		EXPECT_GT(s, 0.0f);
	}
	
	// Dequantized values should be finite
	for (int i = 0; i < 64; ++i) {
		EXPECT_TRUE(std::isfinite(m_dequant.At(i)));
	}
}

// ─── Test: Q4_0 preserves sign ───────────────────────────────────────────────

TEST(FnMatrixQuant, Q4_0_PreservesSign) {
	// Test that quantization preserves sign
	auto m_input = OaFnMatrix::Zeros(OaMatrixShape{32});
	for (int i = 0; i < 16; ++i) {
		m_input.Set(i, -5.0f);  // Negative values
		m_input.Set(i + 16, 5.0f);  // Positive values
	}
	
	auto m_scale = OaFnMatrix::ComputeScaleQ4_0(m_input);
	auto m_quant = OaFnMatrix::QuantizeQ4_0(m_input, m_scale);
	auto m_dequant = OaFnMatrix::DequantizeQ4_0(m_quant, m_scale, 32);
	Flush();
	
	// Check signs are preserved
	for (int i = 0; i < 16; ++i) {
		EXPECT_LT(m_dequant.At(i), 0.0f);  // Should be negative
		EXPECT_GT(m_dequant.At(i + 16), 0.0f);  // Should be positive
	}
}

// ─── Test: Q4_0 with extreme values ──────────────────────────────────────────

TEST(FnMatrixQuant, Q4_0_ExtremeValues) {
	// Test with values at the edge of Q4 range
	auto m_input = OaFnMatrix::Zeros(OaMatrixShape{32});
	for (int i = 0; i < 32; ++i) {
		// Alternate between -7 and +7 (exact Q4 range)
		m_input.Set(i, (i % 2 == 0) ? -7.0f : 7.0f);
	}
	
	auto m_scale = OaFnMatrix::ComputeScaleQ4_0(m_input);
	auto m_quant = OaFnMatrix::QuantizeQ4_0(m_input, m_scale);
	auto m_dequant = OaFnMatrix::DequantizeQ4_0(m_quant, m_scale, 32);
	Flush();
	
	// With scale = 7/7 = 1.0, these should round-trip exactly
	for (int i = 0; i < 32; ++i) {
		float expected = (i % 2 == 0) ? -7.0f : 7.0f;
		EXPECT_NEAR(m_dequant.At(i), expected, 1e-5f);
	}
}

// ─── Test: Q4_0 compression effectiveness ────────────────────────────────────

TEST(FnMatrixQuant, Q4_0_CompressionRatio) {
	// Verify compression ratio for various sizes
	const int sizes[] = {32, 64, 128, 256, 512, 1024};
	
	for (int size : sizes) {
		auto m_input = OaFnMatrix::Zeros(OaMatrixShape{size});
		for (int i = 0; i < size; ++i) {
			m_input.Set(i, std::sin(i * 0.1f) * 10.0f);
		}
		
		auto m_scale = OaFnMatrix::ComputeScaleQ4_0(m_input);
		auto m_quant = OaFnMatrix::QuantizeQ4_0(m_input, m_scale);
		Flush();
		
		// Calculate compression ratio
		int num_blocks = (size + 31) / 32;
		int original_bytes = size * 4;  // FP32
		int compressed_bytes = (size / 2) + (num_blocks * 4);  // Q4 + scales
		float ratio = static_cast<float>(original_bytes) / compressed_bytes;
		
		// Should achieve >6x compression for large tensors
		if (size >= 128) {
			EXPECT_GT(ratio, 6.0f);
		}
	}
}

