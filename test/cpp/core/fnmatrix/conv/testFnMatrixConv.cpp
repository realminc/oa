// Manual tests for Core/FnMatrix convolution operations
// Conv2d, Im2Col, Col2Im

#include <gtest/gtest.h>
#include <oa/core/fnMatrix.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/executionSession.h>
#include <vector>

static oa::Engine* GRt = nullptr;

class TestFnMatrixConv : public ::testing::Test {
protected:
	static void setUpTestSuite() {
		oa::EngineConfig cfg{};
		cfg.appName = "TestFnMatrixConv";
		auto r = oa::Engine::create(cfg);
		ASSERT_TRUE(r.isOk()) << r.getStatus().getMessage();
		static oa::UniquePtr<oa::Engine> rt = std::move(*r);
		GRt = rt.get();
	}
};

// Helper to copy matrix to host
static std::vector<float> copyToHost(const oa::Matrix& m) {
	std::vector<float> result(static_cast<size_t>(m.getShape().numElements()));
	[[maybe_unused]] auto status = oa::FnMatrix::copyToHost(m, result.data(), result.size() * sizeof(float));
	return result;
}

// Helper to create matrix from host data
static oa::Matrix createFromHost(const std::vector<float>& data, oa::MatrixShape shape) {
	return oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(data.data()), data.size() * sizeof(float)),
		shape
	);
}

// ============================================================================
// Conv2d Tests
// ============================================================================

TEST_F(TestFnMatrixConv, Conv2d_Simple3x3) {
	// Test simple 3x3 convolution with stride=1, padding=0
	// input: 1x1x5x5 (batch=1, channels=1, height=5, width=5)
	// kernel: 1x1x3x3 (out_channels=1, in_channels=1, kh=3, kw=3)
	// output: 1x1x3x3
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	
	// Create simple input (all ones)
	auto input = oa::FnMatrix::ones(oa::MatrixShape{1, 1, 5, 5});
	
	// Create simple kernel (all ones) - should sum 9 values
	auto kernel = oa::FnMatrix::ones(oa::MatrixShape{1, 1, 3, 3});
	
	auto output = oa::FnMatrix::conv2d(input, kernel, 1, 0);
	
	// output shape should be [1, 1, 3, 3]
	EXPECT_EQ(output.getShape().rank, 4);
	EXPECT_EQ(output.getShape()[0], 1);  // batch
	EXPECT_EQ(output.getShape()[1], 1);  // out_channels
	EXPECT_EQ(output.getShape()[2], 3);  // out_height = (5 - 3) / 1 + 1 = 3
	EXPECT_EQ(output.getShape()[3], 3);  // out_width = (5 - 3) / 1 + 1 = 3
	
	// Each output value should be 9.0 (sum of 3x3 kernel over 3x3 input region)
	auto result = copyToHost(output);
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_FLOAT_EQ(result[i], 9.0f) << "Mismatch at index " << i;
	}
}

TEST_F(TestFnMatrixConv, Conv2d_WithPadding) {
	// Test convolution with padding=1
	// input: 1x1x3x3, kernel: 1x1x3x3, Padding: 1
	// output: 1x1x3x3 (same size due to padding)
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	
	auto input = oa::FnMatrix::ones(oa::MatrixShape{1, 1, 3, 3});
	auto kernel = oa::FnMatrix::ones(oa::MatrixShape{1, 1, 3, 3});
	
	auto output = oa::FnMatrix::conv2d(input, kernel, 1, 1);
	
	// output shape should be [1, 1, 3, 3] due to padding
	EXPECT_EQ(output.getShape()[0], 1);
	EXPECT_EQ(output.getShape()[1], 1);
	EXPECT_EQ(output.getShape()[2], 3);
	EXPECT_EQ(output.getShape()[3], 3);
	
	// Corner pixels see 4 input values, edge pixels see 6, center sees 9
	auto result = copyToHost(output);
	EXPECT_EQ(result.size(), 9);
	
	// corners (4 values each)
	EXPECT_FLOAT_EQ(result[0], 4.0f);  // top-left
	EXPECT_FLOAT_EQ(result[2], 4.0f);  // top-right
	EXPECT_FLOAT_EQ(result[6], 4.0f);  // bottom-left
	EXPECT_FLOAT_EQ(result[8], 4.0f);  // bottom-right
	
	// edges (6 values each)
	EXPECT_FLOAT_EQ(result[1], 6.0f);  // top-middle
	EXPECT_FLOAT_EQ(result[3], 6.0f);  // left-middle
	EXPECT_FLOAT_EQ(result[5], 6.0f);  // right-middle
	EXPECT_FLOAT_EQ(result[7], 6.0f);  // bottom-middle
	
	// center (9 values)
	EXPECT_FLOAT_EQ(result[4], 9.0f);  // center
}

TEST_F(TestFnMatrixConv, Conv2d_WithStride2) {
	// Test convolution with stride=2
	// input: 1x1x6x6, kernel: 1x1x3x3, Stride: 2
	// output: 1x1x2x2
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	
	auto input = oa::FnMatrix::ones(oa::MatrixShape{1, 1, 6, 6});
	auto kernel = oa::FnMatrix::ones(oa::MatrixShape{1, 1, 3, 3});
	
	auto output = oa::FnMatrix::conv2d(input, kernel, 2, 0);
	
	// output shape: (6 - 3) / 2 + 1 = 2
	EXPECT_EQ(output.getShape()[2], 2);
	EXPECT_EQ(output.getShape()[3], 2);
	
	auto result = copyToHost(output);
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_FLOAT_EQ(result[i], 9.0f) << "Mismatch at index " << i;
	}
}

TEST_F(TestFnMatrixConv, Conv2d_MultiChannel) {
	// Test convolution with multiple input and output channels
	// input: 1x2x4x4 (batch=1, in_channels=2)
	// kernel: 3x2x3x3 (out_channels=3, in_channels=2)
	// output: 1x3x2x2
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	
	auto input = oa::FnMatrix::ones(oa::MatrixShape{1, 2, 4, 4});
	auto kernel = oa::FnMatrix::ones(oa::MatrixShape{3, 2, 3, 3});
	
	auto output = oa::FnMatrix::conv2d(input, kernel, 1, 0);
	
	EXPECT_EQ(output.getShape()[0], 1);  // batch
	EXPECT_EQ(output.getShape()[1], 3);  // out_channels
	EXPECT_EQ(output.getShape()[2], 2);  // out_height
	EXPECT_EQ(output.getShape()[3], 2);  // out_width
	
	// Each output value should be 18.0 (2 input channels * 9 kernel values)
	auto result = copyToHost(output);
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_FLOAT_EQ(result[i], 18.0f) << "Mismatch at index " << i;
	}
}

TEST_F(TestFnMatrixConv, Conv2d_BatchSize2) {
	// Test convolution with batch size > 1
	// input: 2x1x4x4 (batch=2)
	// kernel: 1x1x3x3
	// output: 2x1x2x2
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	
	auto input = oa::FnMatrix::ones(oa::MatrixShape{2, 1, 4, 4});
	auto kernel = oa::FnMatrix::ones(oa::MatrixShape{1, 1, 3, 3});
	
	auto output = oa::FnMatrix::conv2d(input, kernel, 1, 0);
	
	EXPECT_EQ(output.getShape()[0], 2);  // batch
	EXPECT_EQ(output.getShape()[1], 1);  // out_channels
	EXPECT_EQ(output.getShape()[2], 2);  // out_height
	EXPECT_EQ(output.getShape()[3], 2);  // out_width
	
	auto result = copyToHost(output);
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_FLOAT_EQ(result[i], 9.0f) << "Mismatch at index " << i;
	}
}

// ============================================================================
// Im2Col Tests
// ============================================================================

TEST_F(TestFnMatrixConv, Im2Col_Simple) {
	// Test im2col transformation
	// input: 1x1x4x4, KernelSize: 3x3, Stride: 1, Padding: 0
	// output: 1x9x4 (batch=1, kernel_elements=9, num_patches=4)
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	
	// Create input with sequential values
	std::vector<float> input_data(16);
	for (size_t i = 0; i < 16; ++i) {
		input_data[i] = static_cast<float>(i);
	}
	auto input = createFromHost(input_data, oa::MatrixShape{1, 1, 4, 4});
	
	auto output = oa::FnMatrix::im2Col(input, 3, 1, 0);
	
	// output shape: [batch, kernel_h * kernel_w * in_channels, num_patches]
	// num_patches = ((4 - 3) / 1 + 1) * ((4 - 3) / 1 + 1) = 2 * 2 = 4
	EXPECT_EQ(output.getShape().rank, 3);
	EXPECT_EQ(output.getShape()[0], 1);   // batch
	EXPECT_EQ(output.getShape()[1], 9);   // 3*3*1 kernel elements
	EXPECT_EQ(output.getShape()[2], 4);   // 4 patches
	
	// verify structure: each column should contain a 3x3 patch
	auto result = copyToHost(output);
	
	// first patch (top-left 3x3)
	std::vector<float> expected_patch0 = {0, 1, 2, 4, 5, 6, 8, 9, 10};
	for (size_t i = 0; i < 9; ++i) {
		EXPECT_FLOAT_EQ(result[i * 4 + 0], expected_patch0[i]) 
			<< "patch 0, element " << i;
	}
}

TEST_F(TestFnMatrixConv, Im2Col_WithPadding) {
	// Test im2col with padding
	// input: 1x1x3x3, KernelSize: 3x3, Stride: 1, Padding: 1
	// output: 1x9x9 (3x3 output with padding)
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	
	auto input = oa::FnMatrix::ones(oa::MatrixShape{1, 1, 3, 3});
	auto output = oa::FnMatrix::im2Col(input, 3, 1, 1);
	
	// With padding=1, output size = ((3 + 2*1 - 3) / 1 + 1)^2 = 3^2 = 9
	EXPECT_EQ(output.getShape()[0], 1);
	EXPECT_EQ(output.getShape()[1], 9);
	EXPECT_EQ(output.getShape()[2], 9);
}

TEST_F(TestFnMatrixConv, Im2Col_WithStride2) {
	// Test im2col with stride=2
	// input: 1x1x6x6, KernelSize: 3x3, Stride: 2, Padding: 0
	// output: 1x9x4 (2x2 patches)
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	
	auto input = oa::FnMatrix::ones(oa::MatrixShape{1, 1, 6, 6});
	auto output = oa::FnMatrix::im2Col(input, 3, 2, 0);
	
	// num_patches = ((6 - 3) / 2 + 1)^2 = 2^2 = 4
	EXPECT_EQ(output.getShape()[0], 1);
	EXPECT_EQ(output.getShape()[1], 9);
	EXPECT_EQ(output.getShape()[2], 4);
}

// ============================================================================
// Col2Im Tests
// ============================================================================

TEST_F(TestFnMatrixConv, Col2Im_Simple) {
	// Test col2im transformation (inverse of im2col)
	// input: 1x9x4 (columns), output: 1x1x4x4 (image)
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	
	// Create column matrix (all ones)
	auto columns = oa::FnMatrix::ones(oa::MatrixShape{1, 9, 4});
	
	// convert back to image
	auto output = oa::FnMatrix::col2Im(columns, 4, 4, 3, 1, 0);
	
	EXPECT_EQ(output.getShape().rank, 4);
	EXPECT_EQ(output.getShape()[0], 1);  // batch
	EXPECT_EQ(output.getShape()[1], 1);  // channels
	EXPECT_EQ(output.getShape()[2], 4);  // height
	EXPECT_EQ(output.getShape()[3], 4);  // width
}

TEST_F(TestFnMatrixConv, Col2Im_RoundTrip) {
	// Test that im2col followed by col2im preserves structure
	// (Note: values may accumulate due to overlapping patches)
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	
	auto input = oa::FnMatrix::ones(oa::MatrixShape{1, 1, 4, 4});
	
	// im2col
	auto columns = oa::FnMatrix::im2Col(input, 3, 1, 0);
	
	// col2im
	auto reconstructed = oa::FnMatrix::col2Im(columns, 4, 4, 3, 1, 0);
	
	// Shape should match
	EXPECT_EQ(reconstructed.getShape()[0], input.getShape()[0]);
	EXPECT_EQ(reconstructed.getShape()[1], input.getShape()[1]);
	EXPECT_EQ(reconstructed.getShape()[2], input.getShape()[2]);
	EXPECT_EQ(reconstructed.getShape()[3], input.getShape()[3]);
	
	// Note: values will be different due to overlapping patches accumulating
	// This is expected behavior for col2im (used in backward pass)
}

TEST_F(TestFnMatrixConv, Col2Im_WithPadding) {
	// Test col2im with padding
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	
	auto columns = oa::FnMatrix::ones(oa::MatrixShape{1, 9, 9});
	auto output = oa::FnMatrix::col2Im(columns, 3, 3, 3, 1, 1);
	
	EXPECT_EQ(output.getShape()[0], 1);
	EXPECT_EQ(output.getShape()[1], 1);
	EXPECT_EQ(output.getShape()[2], 3);
	EXPECT_EQ(output.getShape()[3], 3);
}
