// Manual tests for Core/FnMatrix convolution operations
// Conv2d, Im2Col, Col2Im

#include <gtest/gtest.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Context.h>
#include <vector>

static OaEngine* GRt = nullptr;

class TestFnMatrixConv : public ::testing::Test {
protected:
	static void SetUpTestSuite() {
		OaEngineConfig cfg{};
		cfg.AppName = "TestFnMatrixConv";
		auto r = OaEngine::Create(cfg);
		ASSERT_TRUE(r.IsOk()) << r.GetStatus().GetMessage();
		static OaUniquePtr<OaEngine> rt = std::move(*r);
		GRt = rt.get();
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
// Conv2d Tests
// ============================================================================

TEST_F(TestFnMatrixConv, Conv2d_Simple3x3) {
	// Test simple 3x3 convolution with stride=1, padding=0
	// Input: 1x1x5x5 (batch=1, channels=1, height=5, width=5)
	// Kernel: 1x1x3x3 (out_channels=1, in_channels=1, kh=3, kw=3)
	// Output: 1x1x3x3
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	
	// Create simple input (all ones)
	auto input = OaFnMatrix::Ones(OaMatrixShape{1, 1, 5, 5});
	
	// Create simple kernel (all ones) - should sum 9 values
	auto kernel = OaFnMatrix::Ones(OaMatrixShape{1, 1, 3, 3});
	
	auto output = OaFnMatrix::Conv2d(input, kernel, 1, 0);
	
	// Output shape should be [1, 1, 3, 3]
	EXPECT_EQ(output.GetShape().Rank, 4);
	EXPECT_EQ(output.GetShape()[0], 1);  // batch
	EXPECT_EQ(output.GetShape()[1], 1);  // out_channels
	EXPECT_EQ(output.GetShape()[2], 3);  // out_height = (5 - 3) / 1 + 1 = 3
	EXPECT_EQ(output.GetShape()[3], 3);  // out_width = (5 - 3) / 1 + 1 = 3
	
	// Each output value should be 9.0 (sum of 3x3 kernel over 3x3 input region)
	auto result = CopyToHost(output);
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_FLOAT_EQ(result[i], 9.0f) << "Mismatch at index " << i;
	}
}

TEST_F(TestFnMatrixConv, Conv2d_WithPadding) {
	// Test convolution with padding=1
	// Input: 1x1x3x3, Kernel: 1x1x3x3, Padding: 1
	// Output: 1x1x3x3 (same size due to padding)
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	
	auto input = OaFnMatrix::Ones(OaMatrixShape{1, 1, 3, 3});
	auto kernel = OaFnMatrix::Ones(OaMatrixShape{1, 1, 3, 3});
	
	auto output = OaFnMatrix::Conv2d(input, kernel, 1, 1);
	
	// Output shape should be [1, 1, 3, 3] due to padding
	EXPECT_EQ(output.GetShape()[0], 1);
	EXPECT_EQ(output.GetShape()[1], 1);
	EXPECT_EQ(output.GetShape()[2], 3);
	EXPECT_EQ(output.GetShape()[3], 3);
	
	// Corner pixels see 4 input values, edge pixels see 6, center sees 9
	auto result = CopyToHost(output);
	EXPECT_EQ(result.size(), 9);
	
	// Corners (4 values each)
	EXPECT_FLOAT_EQ(result[0], 4.0f);  // top-left
	EXPECT_FLOAT_EQ(result[2], 4.0f);  // top-right
	EXPECT_FLOAT_EQ(result[6], 4.0f);  // bottom-left
	EXPECT_FLOAT_EQ(result[8], 4.0f);  // bottom-right
	
	// Edges (6 values each)
	EXPECT_FLOAT_EQ(result[1], 6.0f);  // top-middle
	EXPECT_FLOAT_EQ(result[3], 6.0f);  // left-middle
	EXPECT_FLOAT_EQ(result[5], 6.0f);  // right-middle
	EXPECT_FLOAT_EQ(result[7], 6.0f);  // bottom-middle
	
	// Center (9 values)
	EXPECT_FLOAT_EQ(result[4], 9.0f);  // center
}

TEST_F(TestFnMatrixConv, Conv2d_WithStride2) {
	// Test convolution with stride=2
	// Input: 1x1x6x6, Kernel: 1x1x3x3, Stride: 2
	// Output: 1x1x2x2
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	
	auto input = OaFnMatrix::Ones(OaMatrixShape{1, 1, 6, 6});
	auto kernel = OaFnMatrix::Ones(OaMatrixShape{1, 1, 3, 3});
	
	auto output = OaFnMatrix::Conv2d(input, kernel, 2, 0);
	
	// Output shape: (6 - 3) / 2 + 1 = 2
	EXPECT_EQ(output.GetShape()[2], 2);
	EXPECT_EQ(output.GetShape()[3], 2);
	
	auto result = CopyToHost(output);
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_FLOAT_EQ(result[i], 9.0f) << "Mismatch at index " << i;
	}
}

TEST_F(TestFnMatrixConv, Conv2d_MultiChannel) {
	// Test convolution with multiple input and output channels
	// Input: 1x2x4x4 (batch=1, in_channels=2)
	// Kernel: 3x2x3x3 (out_channels=3, in_channels=2)
	// Output: 1x3x2x2
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	
	auto input = OaFnMatrix::Ones(OaMatrixShape{1, 2, 4, 4});
	auto kernel = OaFnMatrix::Ones(OaMatrixShape{3, 2, 3, 3});
	
	auto output = OaFnMatrix::Conv2d(input, kernel, 1, 0);
	
	EXPECT_EQ(output.GetShape()[0], 1);  // batch
	EXPECT_EQ(output.GetShape()[1], 3);  // out_channels
	EXPECT_EQ(output.GetShape()[2], 2);  // out_height
	EXPECT_EQ(output.GetShape()[3], 2);  // out_width
	
	// Each output value should be 18.0 (2 input channels * 9 kernel values)
	auto result = CopyToHost(output);
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_FLOAT_EQ(result[i], 18.0f) << "Mismatch at index " << i;
	}
}

TEST_F(TestFnMatrixConv, Conv2d_BatchSize2) {
	// Test convolution with batch size > 1
	// Input: 2x1x4x4 (batch=2)
	// Kernel: 1x1x3x3
	// Output: 2x1x2x2
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	
	auto input = OaFnMatrix::Ones(OaMatrixShape{2, 1, 4, 4});
	auto kernel = OaFnMatrix::Ones(OaMatrixShape{1, 1, 3, 3});
	
	auto output = OaFnMatrix::Conv2d(input, kernel, 1, 0);
	
	EXPECT_EQ(output.GetShape()[0], 2);  // batch
	EXPECT_EQ(output.GetShape()[1], 1);  // out_channels
	EXPECT_EQ(output.GetShape()[2], 2);  // out_height
	EXPECT_EQ(output.GetShape()[3], 2);  // out_width
	
	auto result = CopyToHost(output);
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_FLOAT_EQ(result[i], 9.0f) << "Mismatch at index " << i;
	}
}

// ============================================================================
// Im2Col Tests
// ============================================================================

TEST_F(TestFnMatrixConv, Im2Col_Simple) {
	// Test im2col transformation
	// Input: 1x1x4x4, KernelSize: 3x3, Stride: 1, Padding: 0
	// Output: 1x9x4 (batch=1, kernel_elements=9, num_patches=4)
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	
	// Create input with sequential values
	std::vector<float> input_data(16);
	for (size_t i = 0; i < 16; ++i) {
		input_data[i] = static_cast<float>(i);
	}
	auto input = CreateFromHost(input_data, OaMatrixShape{1, 1, 4, 4});
	
	auto output = OaFnMatrix::Im2Col(input, 3, 1, 0);
	
	// Output shape: [batch, kernel_h * kernel_w * in_channels, num_patches]
	// num_patches = ((4 - 3) / 1 + 1) * ((4 - 3) / 1 + 1) = 2 * 2 = 4
	EXPECT_EQ(output.GetShape().Rank, 3);
	EXPECT_EQ(output.GetShape()[0], 1);   // batch
	EXPECT_EQ(output.GetShape()[1], 9);   // 3*3*1 kernel elements
	EXPECT_EQ(output.GetShape()[2], 4);   // 4 patches
	
	// Verify structure: each column should contain a 3x3 patch
	auto result = CopyToHost(output);
	
	// First patch (top-left 3x3)
	std::vector<float> expected_patch0 = {0, 1, 2, 4, 5, 6, 8, 9, 10};
	for (size_t i = 0; i < 9; ++i) {
		EXPECT_FLOAT_EQ(result[i * 4 + 0], expected_patch0[i]) 
			<< "Patch 0, element " << i;
	}
}

TEST_F(TestFnMatrixConv, Im2Col_WithPadding) {
	// Test im2col with padding
	// Input: 1x1x3x3, KernelSize: 3x3, Stride: 1, Padding: 1
	// Output: 1x9x9 (3x3 output with padding)
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	
	auto input = OaFnMatrix::Ones(OaMatrixShape{1, 1, 3, 3});
	auto output = OaFnMatrix::Im2Col(input, 3, 1, 1);
	
	// With padding=1, output size = ((3 + 2*1 - 3) / 1 + 1)^2 = 3^2 = 9
	EXPECT_EQ(output.GetShape()[0], 1);
	EXPECT_EQ(output.GetShape()[1], 9);
	EXPECT_EQ(output.GetShape()[2], 9);
}

TEST_F(TestFnMatrixConv, Im2Col_WithStride2) {
	// Test im2col with stride=2
	// Input: 1x1x6x6, KernelSize: 3x3, Stride: 2, Padding: 0
	// Output: 1x9x4 (2x2 patches)
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	
	auto input = OaFnMatrix::Ones(OaMatrixShape{1, 1, 6, 6});
	auto output = OaFnMatrix::Im2Col(input, 3, 2, 0);
	
	// num_patches = ((6 - 3) / 2 + 1)^2 = 2^2 = 4
	EXPECT_EQ(output.GetShape()[0], 1);
	EXPECT_EQ(output.GetShape()[1], 9);
	EXPECT_EQ(output.GetShape()[2], 4);
}

// ============================================================================
// Col2Im Tests
// ============================================================================

TEST_F(TestFnMatrixConv, Col2Im_Simple) {
	// Test col2im transformation (inverse of im2col)
	// Input: 1x9x4 (columns), Output: 1x1x4x4 (image)
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	
	// Create column matrix (all ones)
	auto columns = OaFnMatrix::Ones(OaMatrixShape{1, 9, 4});
	
	// Convert back to image
	auto output = OaFnMatrix::Col2Im(columns, 4, 4, 3, 1, 0);
	
	EXPECT_EQ(output.GetShape().Rank, 4);
	EXPECT_EQ(output.GetShape()[0], 1);  // batch
	EXPECT_EQ(output.GetShape()[1], 1);  // channels
	EXPECT_EQ(output.GetShape()[2], 4);  // height
	EXPECT_EQ(output.GetShape()[3], 4);  // width
}

TEST_F(TestFnMatrixConv, Col2Im_RoundTrip) {
	// Test that im2col followed by col2im preserves structure
	// (Note: values may accumulate due to overlapping patches)
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	
	auto input = OaFnMatrix::Ones(OaMatrixShape{1, 1, 4, 4});
	
	// im2col
	auto columns = OaFnMatrix::Im2Col(input, 3, 1, 0);
	
	// col2im
	auto reconstructed = OaFnMatrix::Col2Im(columns, 4, 4, 3, 1, 0);
	
	// Shape should match
	EXPECT_EQ(reconstructed.GetShape()[0], input.GetShape()[0]);
	EXPECT_EQ(reconstructed.GetShape()[1], input.GetShape()[1]);
	EXPECT_EQ(reconstructed.GetShape()[2], input.GetShape()[2]);
	EXPECT_EQ(reconstructed.GetShape()[3], input.GetShape()[3]);
	
	// Note: Values will be different due to overlapping patches accumulating
	// This is expected behavior for col2im (used in backward pass)
}

TEST_F(TestFnMatrixConv, Col2Im_WithPadding) {
	// Test col2im with padding
	
	OaContext::RecordingScope ctx_scope(OaContext::GetDefault());
	
	auto columns = OaFnMatrix::Ones(OaMatrixShape{1, 9, 9});
	auto output = OaFnMatrix::Col2Im(columns, 3, 3, 3, 1, 1);
	
	EXPECT_EQ(output.GetShape()[0], 1);
	EXPECT_EQ(output.GetShape()[1], 1);
	EXPECT_EQ(output.GetShape()[2], 3);
	EXPECT_EQ(output.GetShape()[3], 3);
}

