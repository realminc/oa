// Tests for quantization operations
// Quantize, Dequantize for various quantization formats

#include <gtest/gtest.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/RuntimeGlobal.h>
#include <vector>
#include <cmath>

static OaComputeEngine* GRt = nullptr;

class TestFnMatrixQuant : public ::testing::Test {
protected:
	static void SetUpTestSuite() {
		OaEngineConfig cfg{};
		cfg.AppName = "TestFnMatrixQuant";
		auto r = OaComputeEngine::Create(cfg);
		ASSERT_TRUE(r.IsOk()) << r.GetStatus().GetMessage();
		static OaUniquePtr<OaComputeEngine> rt = std::move(*r);
		GRt = rt.get();
		OaRuntimeGlobal::SetRuntime(GRt);
	}
};

// Helper to copy matrix to host
static std::vector<float> CopyToHost(const OaMatrix& m) {
	std::vector<float> result(static_cast<size_t>(m.GetShape().NumElements()));
	[[maybe_unused]] auto status = OaFnMatrix::CopyToHost(m, result.data(), result.size() * sizeof(float));
	return result;
}

// Helper to create matrix from host data
static OaMatrix CreateFromHost(const std::vector<float>& data, OaMatrixShape shape) {
	return OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(data.data()), data.size() * sizeof(float)),
		shape
	);
}

// ============================================================================
// Q8_0 Tests (8-bit quantization, block size 32)
// ============================================================================

TEST_F(TestFnMatrixQuant, Quantize_Q8_0_Basic) {
	// Test basic Q8_0 quantization
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	
	// Create input with known values
	std::vector<float> input_data(64);
	for (size_t i = 0; i < input_data.size(); ++i) {
		input_data[i] = static_cast<float>(i) - 32.0f;  // Range: -32 to 31
	}
	
	auto input = CreateFromHost(input_data, OaMatrixShape{64});
	
	// Quantize to Q8_0
	auto quantized = OaFnMatrix::Quantize(input, OaScalarType::Q8_0);
	
	// Check that quantized tensor has correct type
	EXPECT_EQ(quantized.GetDtype(), OaScalarType::Q8_0);
	
	// Dequantize back to FP32
	auto dequantized = OaFnMatrix::Dequantize(quantized);
	
	EXPECT_EQ(dequantized.GetDtype(), OaScalarType::Float32);
	EXPECT_EQ(dequantized.GetShape().NumElements(), 64);
	
	// Check that round-trip preserves approximate values
	auto result = CopyToHost(dequantized);
	for (size_t i = 0; i < input_data.size(); ++i) {
		// Q8_0 has ~1% quantization error
		EXPECT_NEAR(result[i], input_data[i], std::abs(input_data[i]) * 0.02f + 0.5f)
			<< "Mismatch at index " << i;
	}
}

TEST_F(TestFnMatrixQuant, Quantize_Q8_0_AllZeros) {
	// Test quantization of all-zero tensor
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	
	auto input = OaFnMatrix::Zeros(OaMatrixShape{128});
	auto quantized = OaFnMatrix::Quantize(input, OaScalarType::Q8_0);
	auto dequantized = OaFnMatrix::Dequantize(quantized);
	
	auto result = CopyToHost(dequantized);
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_FLOAT_EQ(result[i], 0.0f) << "Index " << i;
	}
}

TEST_F(TestFnMatrixQuant, Quantize_Q8_0_LargeRange) {
	// Test quantization with large value range
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	
	std::vector<float> input_data(64);
	for (size_t i = 0; i < input_data.size(); ++i) {
		input_data[i] = (static_cast<float>(i) - 32.0f) * 10.0f;  // Range: -320 to 310
	}
	
	auto input = CreateFromHost(input_data, OaMatrixShape{64});
	auto quantized = OaFnMatrix::Quantize(input, OaScalarType::Q8_0);
	auto dequantized = OaFnMatrix::Dequantize(quantized);
	
	auto result = CopyToHost(dequantized);
	for (size_t i = 0; i < input_data.size(); ++i) {
		EXPECT_NEAR(result[i], input_data[i], std::abs(input_data[i]) * 0.02f + 1.0f)
			<< "Mismatch at index " << i;
	}
}

// ============================================================================
// Q4_0 Tests (4-bit quantization, block size 32)
// ============================================================================

TEST_F(TestFnMatrixQuant, Quantize_Q4_0_Basic) {
	// Test basic Q4_0 quantization (4-bit, lower precision)
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	
	std::vector<float> input_data(64);
	for (size_t i = 0; i < input_data.size(); ++i) {
		input_data[i] = static_cast<float>(i % 16) - 8.0f;  // Range: -8 to 7
	}
	
	auto input = CreateFromHost(input_data, OaMatrixShape{64});
	auto quantized = OaFnMatrix::Quantize(input, OaScalarType::Q4_0);
	
	EXPECT_EQ(quantized.GetDtype(), OaScalarType::Q4_0);
	
	auto dequantized = OaFnMatrix::Dequantize(quantized);
	auto result = CopyToHost(dequantized);
	
	// Q4_0 has lower precision (~5% error)
	for (size_t i = 0; i < input_data.size(); ++i) {
		EXPECT_NEAR(result[i], input_data[i], std::abs(input_data[i]) * 0.1f + 1.0f)
			<< "Mismatch at index " << i;
	}
}

TEST_F(TestFnMatrixQuant, Quantize_Q4_0_SmallValues) {
	// Test Q4_0 with small values near zero
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	
	std::vector<float> input_data(64);
	for (size_t i = 0; i < input_data.size(); ++i) {
		input_data[i] = (static_cast<float>(i % 8) - 4.0f) * 0.1f;  // Range: -0.4 to 0.3
	}
	
	auto input = CreateFromHost(input_data, OaMatrixShape{64});
	auto quantized = OaFnMatrix::Quantize(input, OaScalarType::Q4_0);
	auto dequantized = OaFnMatrix::Dequantize(quantized);
	
	auto result = CopyToHost(dequantized);
	for (size_t i = 0; i < input_data.size(); ++i) {
		EXPECT_NEAR(result[i], input_data[i], 0.2f)
			<< "Mismatch at index " << i;
	}
}

// ============================================================================
// Q4_K Tests (4-bit K-quantization with better quality)
// ============================================================================

TEST_F(TestFnMatrixQuant, Quantize_Q4_K_Basic) {
	// Test Q4_K quantization (improved 4-bit format)
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	
	std::vector<float> input_data(256);  // Q4_K uses larger blocks
	for (size_t i = 0; i < input_data.size(); ++i) {
		input_data[i] = std::sin(static_cast<float>(i) * 0.1f) * 10.0f;
	}
	
	auto input = CreateFromHost(input_data, OaMatrixShape{256});
	auto quantized = OaFnMatrix::Quantize(input, OaScalarType::Q4_K);
	
	EXPECT_EQ(quantized.GetDtype(), OaScalarType::Q4_K);
	
	auto dequantized = OaFnMatrix::Dequantize(quantized);
	auto result = CopyToHost(dequantized);
	
	// Q4_K should have better quality than Q4_0
	for (size_t i = 0; i < input_data.size(); ++i) {
		EXPECT_NEAR(result[i], input_data[i], std::abs(input_data[i]) * 0.05f + 0.5f)
			<< "Mismatch at index " << i;
	}
}

// ============================================================================
// Q6_K Tests (6-bit K-quantization, higher quality)
// ============================================================================

TEST_F(TestFnMatrixQuant, Quantize_Q6_K_Basic) {
	// Test Q6_K quantization (6-bit, higher quality)
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	
	std::vector<float> input_data(256);
	for (size_t i = 0; i < input_data.size(); ++i) {
		input_data[i] = static_cast<float>(i) - 128.0f;
	}
	
	auto input = CreateFromHost(input_data, OaMatrixShape{256});
	auto quantized = OaFnMatrix::Quantize(input, OaScalarType::Q6_K);
	
	EXPECT_EQ(quantized.GetDtype(), OaScalarType::Q6_K);
	
	auto dequantized = OaFnMatrix::Dequantize(quantized);
	auto result = CopyToHost(dequantized);
	
	// Q6_K should have very good quality (~1% error)
	for (size_t i = 0; i < input_data.size(); ++i) {
		EXPECT_NEAR(result[i], input_data[i], std::abs(input_data[i]) * 0.015f + 0.3f)
			<< "Mismatch at index " << i;
	}
}

// ============================================================================
// Multi-dimensional Quantization Tests
// ============================================================================

TEST_F(TestFnMatrixQuant, Quantize_2D_Matrix) {
	// Test quantization of 2D matrix
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	
	std::vector<float> input_data(8 * 16);  // 8x16 matrix
	for (size_t i = 0; i < input_data.size(); ++i) {
		input_data[i] = static_cast<float>(i % 32) - 16.0f;
	}
	
	auto input = CreateFromHost(input_data, OaMatrixShape{8, 16});
	auto quantized = OaFnMatrix::Quantize(input, OaScalarType::Q8_0);
	auto dequantized = OaFnMatrix::Dequantize(quantized);
	
	EXPECT_EQ(dequantized.GetShape().Rank, 2);
	EXPECT_EQ(dequantized.GetShape()[0], 8);
	EXPECT_EQ(dequantized.GetShape()[1], 16);
	
	auto result = CopyToHost(dequantized);
	for (size_t i = 0; i < input_data.size(); ++i) {
		EXPECT_NEAR(result[i], input_data[i], std::abs(input_data[i]) * 0.02f + 0.5f)
			<< "Mismatch at index " << i;
	}
}

TEST_F(TestFnMatrixQuant, Quantize_3D_Tensor) {
	// Test quantization of 3D tensor
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	
	std::vector<float> input_data(4 * 8 * 8);  // 4x8x8 tensor
	for (size_t i = 0; i < input_data.size(); ++i) {
		input_data[i] = std::cos(static_cast<float>(i) * 0.05f) * 5.0f;
	}
	
	auto input = CreateFromHost(input_data, OaMatrixShape{4, 8, 8});
	auto quantized = OaFnMatrix::Quantize(input, OaScalarType::Q4_K);
	auto dequantized = OaFnMatrix::Dequantize(quantized);
	
	EXPECT_EQ(dequantized.GetShape().Rank, 3);
	EXPECT_EQ(dequantized.GetShape()[0], 4);
	EXPECT_EQ(dequantized.GetShape()[1], 8);
	EXPECT_EQ(dequantized.GetShape()[2], 8);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(TestFnMatrixQuant, Quantize_MinimumSize) {
	// Test quantization with minimum valid size (32 elements for Q8_0)
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	
	std::vector<float> input_data(32);
	for (size_t i = 0; i < input_data.size(); ++i) {
		input_data[i] = static_cast<float>(i);
	}
	
	auto input = CreateFromHost(input_data, OaMatrixShape{32});
	auto quantized = OaFnMatrix::Quantize(input, OaScalarType::Q8_0);
	auto dequantized = OaFnMatrix::Dequantize(quantized);
	
	EXPECT_EQ(dequantized.GetShape().NumElements(), 32);
}

TEST_F(TestFnMatrixQuant, Quantize_NegativeValues) {
	// Test quantization with all negative values
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	
	std::vector<float> input_data(64);
	for (size_t i = 0; i < input_data.size(); ++i) {
		input_data[i] = -static_cast<float>(i + 1);  // -1 to -64
	}
	
	auto input = CreateFromHost(input_data, OaMatrixShape{64});
	auto quantized = OaFnMatrix::Quantize(input, OaScalarType::Q8_0);
	auto dequantized = OaFnMatrix::Dequantize(quantized);
	
	auto result = CopyToHost(dequantized);
	for (size_t i = 0; i < input_data.size(); ++i) {
		EXPECT_NEAR(result[i], input_data[i], std::abs(input_data[i]) * 0.02f + 0.5f)
			<< "Mismatch at index " << i;
	}
}

