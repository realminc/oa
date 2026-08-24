// Tests for additional NN modules
// AdaptiveAvgPool2d, Flatten, SwiGLU, Softmax

#include <oa/ml/nn.h>
#include <oaTest.h>

// ============================================================================
// ADAPTIVE AVERAGE POOLING TESTS
// ============================================================================

TEST(AdaptiveAvgPool2d, DownsampleToSinglePixel) {
	// Test downsampling to 1x1 (global average pooling)
	oa::AdaptiveAvgPool2d pool(1, 1);
	
	// input: [1, 3, 8, 8] -> output: [1, 3, 1, 1]
	auto input = oa::FnMatrix::rand(oa::MatrixShape{1, 3, 8, 8});
	auto output = pool.forward(input);
	
	ASSERT_EQ(output.getShape().rank, 4);
	EXPECT_EQ(output.getShape()[0], 1);  // batch
	EXPECT_EQ(output.getShape()[1], 3);  // channels
	EXPECT_EQ(output.getShape()[2], 1);  // height
	EXPECT_EQ(output.getShape()[3], 1);  // width
	
	expectFinite(output);
}

TEST(AdaptiveAvgPool2d, DownsampleTo2x2) {
	// Test downsampling to 2x2
	oa::AdaptiveAvgPool2d pool(2, 2);
	
	// input: [2, 4, 16, 16] -> output: [2, 4, 2, 2]
	auto input = oa::FnMatrix::rand(oa::MatrixShape{2, 4, 16, 16});
	auto output = pool.forward(input);
	
	ASSERT_EQ(output.getShape().rank, 4);
	EXPECT_EQ(output.getShape()[0], 2);
	EXPECT_EQ(output.getShape()[1], 4);
	EXPECT_EQ(output.getShape()[2], 2);
	EXPECT_EQ(output.getShape()[3], 2);
	
	expectFinite(output);
}

TEST(AdaptiveAvgPool2d, DownsampleTo7x7) {
	// Test downsampling to 7x7 (common in ResNet)
	oa::AdaptiveAvgPool2d pool(7, 7);
	
	// input: [1, 64, 14, 14] -> output: [1, 64, 7, 7]
	auto input = oa::FnMatrix::rand(oa::MatrixShape{1, 64, 14, 14});
	auto output = pool.forward(input);
	
	ASSERT_EQ(output.getShape().rank, 4);
	EXPECT_EQ(output.getShape()[0], 1);
	EXPECT_EQ(output.getShape()[1], 64);
	EXPECT_EQ(output.getShape()[2], 7);
	EXPECT_EQ(output.getShape()[3], 7);
	
	expectFinite(output);
}

TEST(AdaptiveAvgPool2d, NonSquareOutput) {
	// Test non-square output size
	oa::AdaptiveAvgPool2d pool(4, 8);
	
	// input: [1, 16, 32, 64] -> output: [1, 16, 4, 8]
	auto input = oa::FnMatrix::rand(oa::MatrixShape{1, 16, 32, 64});
	auto output = pool.forward(input);
	
	ASSERT_EQ(output.getShape().rank, 4);
	EXPECT_EQ(output.getShape()[0], 1);
	EXPECT_EQ(output.getShape()[1], 16);
	EXPECT_EQ(output.getShape()[2], 4);
	EXPECT_EQ(output.getShape()[3], 8);
	
	expectFinite(output);
}

TEST(AdaptiveAvgPool2d, BatchProcessing) {
	// Test with batch size > 1
	oa::AdaptiveAvgPool2d pool(3, 3);
	
	// input: [8, 32, 12, 12] -> output: [8, 32, 3, 3]
	auto input = oa::FnMatrix::rand(oa::MatrixShape{8, 32, 12, 12});
	auto output = pool.forward(input);
	
	ASSERT_EQ(output.getShape().rank, 4);
	EXPECT_EQ(output.getShape()[0], 8);
	EXPECT_EQ(output.getShape()[1], 32);
	EXPECT_EQ(output.getShape()[2], 3);
	EXPECT_EQ(output.getShape()[3], 3);
	
	expectFinite(output);
}

TEST(AdaptiveAvgPool2d, IdentityMapping) {
	// Test when output size equals input size (should be identity-like)
	oa::AdaptiveAvgPool2d pool(8, 8);
	
	// input: [1, 3, 8, 8] -> output: [1, 3, 8, 8]
	auto input = oa::FnMatrix::rand(oa::MatrixShape{1, 3, 8, 8});
	auto output = pool.forward(input);
	
	ASSERT_EQ(output.getShape().rank, 4);
	EXPECT_EQ(output.getShape()[0], 1);
	EXPECT_EQ(output.getShape()[1], 3);
	EXPECT_EQ(output.getShape()[2], 8);
	EXPECT_EQ(output.getShape()[3], 8);
	
	expectFinite(output);
}

// ============================================================================
// FLATTEN TESTS
// ============================================================================

TEST(Flatten, Flatten2DTo1D) {
	// Test flattening 2D to 1D
	oa::Flatten flatten(0, -1);
	
	// input: [4, 8] -> output: [32]
	auto input = oa::FnMatrix::rand(oa::MatrixShape{4, 8});
	auto output = flatten.forward(input);
	
	ASSERT_EQ(output.getShape().rank, 1);
	EXPECT_EQ(output.getShape()[0], 32);
	
	expectFinite(output);
}

TEST(Flatten, Flatten3DKeepBatch) {
	// Test flattening 3D keeping batch dimension
	oa::Flatten flatten(1, -1);
	
	// input: [2, 4, 8] -> output: [2, 32]
	auto input = oa::FnMatrix::rand(oa::MatrixShape{2, 4, 8});
	auto output = flatten.forward(input);
	
	ASSERT_EQ(output.getShape().rank, 2);
	EXPECT_EQ(output.getShape()[0], 2);
	EXPECT_EQ(output.getShape()[1], 32);
	
	expectFinite(output);
}

TEST(Flatten, Flatten4DKeepBatch) {
	// Test flattening 4D (typical CNN output) keeping batch
	oa::Flatten flatten(1, -1);
	
	// input: [8, 64, 7, 7] -> output: [8, 3136]
	auto input = oa::FnMatrix::rand(oa::MatrixShape{8, 64, 7, 7});
	auto output = flatten.forward(input);
	
	ASSERT_EQ(output.getShape().rank, 2);
	EXPECT_EQ(output.getShape()[0], 8);
	EXPECT_EQ(output.getShape()[1], 64 * 7 * 7);
	
	expectFinite(output);
}

TEST(Flatten, FlattenPartialRange) {
	// Test flattening only middle dimensions
	oa::Flatten flatten(1, 2);
	
	// input: [2, 3, 4, 5] -> output: [2, 12, 5]
	auto input = oa::FnMatrix::rand(oa::MatrixShape{2, 3, 4, 5});
	auto output = flatten.forward(input);
	
	ASSERT_EQ(output.getShape().rank, 3);
	EXPECT_EQ(output.getShape()[0], 2);
	EXPECT_EQ(output.getShape()[1], 12);  // 3 * 4
	EXPECT_EQ(output.getShape()[2], 5);
	
	expectFinite(output);
}

TEST(Flatten, FlattenSingleDimension) {
	// Test flattening a single dimension (no-op)
	oa::Flatten flatten(1, 1);
	
	// input: [2, 8, 4] -> output: [2, 8, 4] (unchanged)
	auto input = oa::FnMatrix::rand(oa::MatrixShape{2, 8, 4});
	auto output = flatten.forward(input);
	
	ASSERT_EQ(output.getShape().rank, 3);
	EXPECT_EQ(output.getShape()[0], 2);
	EXPECT_EQ(output.getShape()[1], 8);
	EXPECT_EQ(output.getShape()[2], 4);
	
	expectFinite(output);
}

// ============================================================================
// SWIGLU TESTS
// ============================================================================

TEST(Swiglu, BasicForward) {
	// Test basic SwiGLU forward pass
	oa::Swiglu swiglu(128, 512);
	
	// input: [4, 16, 128] -> output: [4, 16, 128]
	auto input = oa::FnMatrix::rand(oa::MatrixShape{4, 16, 128});
	auto output = swiglu.forward(input);
	
	ASSERT_EQ(output.getShape().rank, 3);
	EXPECT_EQ(output.getShape()[0], 4);
	EXPECT_EQ(output.getShape()[1], 16);
	EXPECT_EQ(output.getShape()[2], 128);
	
	expectFinite(output);
}

TEST(Swiglu, LargeHiddenDim) {
	// Test with large hidden dimension
	oa::Swiglu swiglu(512, 2048);
	
	// input: [2, 32, 512] -> output: [2, 32, 512]
	auto input = oa::FnMatrix::rand(oa::MatrixShape{2, 32, 512});
	auto output = swiglu.forward(input);
	
	ASSERT_EQ(output.getShape().rank, 3);
	EXPECT_EQ(output.getShape()[0], 2);
	EXPECT_EQ(output.getShape()[1], 32);
	EXPECT_EQ(output.getShape()[2], 512);
	
	expectFinite(output);
}

TEST(Swiglu, SmallBatch) {
	// Test with small batch size
	oa::Swiglu swiglu(64, 256);
	
	// input: [1, 8, 64] -> output: [1, 8, 64]
	auto input = oa::FnMatrix::rand(oa::MatrixShape{1, 8, 64});
	auto output = swiglu.forward(input);
	
	ASSERT_EQ(output.getShape().rank, 3);
	EXPECT_EQ(output.getShape()[0], 1);
	EXPECT_EQ(output.getShape()[1], 8);
	EXPECT_EQ(output.getShape()[2], 64);
	
	expectFinite(output);
}

// ============================================================================
// SOFTMAX TESTS
// ============================================================================

TEST(Softmax, BasicSoftmax) {
	// Test basic softmax on last dimension
	oa::Softmax softmax(-1);
	
	// input: [2, 4, 8] -> output: [2, 4, 8]
	auto input = oa::FnMatrix::rand(oa::MatrixShape{2, 4, 8});
	auto output = softmax.forward(input);
	
	ASSERT_EQ(output.getShape().rank, 3);
	EXPECT_EQ(output.getShape()[0], 2);
	EXPECT_EQ(output.getShape()[1], 4);
	EXPECT_EQ(output.getShape()[2], 8);
	
	expectFinite(output);
}

TEST(Softmax, SoftmaxDim0) {
	// Test softmax on dimension 0
	oa::Softmax softmax(0);
	
	// input: [4, 8] -> output: [4, 8]
	auto input = oa::FnMatrix::rand(oa::MatrixShape{4, 8});
	auto output = softmax.forward(input);
	
	ASSERT_EQ(output.getShape().rank, 2);
	EXPECT_EQ(output.getShape()[0], 4);
	EXPECT_EQ(output.getShape()[1], 8);
	
	expectFinite(output);
}

TEST(Softmax, SoftmaxDim1) {
	// Test softmax on dimension 1
	oa::Softmax softmax(1);
	
	// input: [8, 16, 32] -> output: [8, 16, 32]
	auto input = oa::FnMatrix::rand(oa::MatrixShape{8, 16, 32});
	auto output = softmax.forward(input);
	
	ASSERT_EQ(output.getShape().rank, 3);
	EXPECT_EQ(output.getShape()[0], 8);
	EXPECT_EQ(output.getShape()[1], 16);
	EXPECT_EQ(output.getShape()[2], 32);
	
	expectFinite(output);
}

TEST(Softmax, LargeVocab) {
	// Test softmax with large vocabulary (typical in language models)
	oa::Softmax softmax(-1);
	
	// input: [4, 128, 50000] -> output: [4, 128, 50000]
	auto input = oa::FnMatrix::rand(oa::MatrixShape{4, 128, 50000});
	auto output = softmax.forward(input);
	
	ASSERT_EQ(output.getShape().rank, 3);
	EXPECT_EQ(output.getShape()[0], 4);
	EXPECT_EQ(output.getShape()[1], 128);
	EXPECT_EQ(output.getShape()[2], 50000);
	
	expectFinite(output);
}

// ============================================================================
// IDENTITY TESTS
// ============================================================================

TEST(Identity, PassthroughShape) {
	// Test that Identity preserves shape
	oa::Identity identity;
	
	auto input = oa::FnMatrix::rand(oa::MatrixShape{2, 3, 4, 5});
	auto output = identity.forward(input);
	
	ASSERT_EQ(output.getShape().rank, 4);
	EXPECT_EQ(output.getShape()[0], 2);
	EXPECT_EQ(output.getShape()[1], 3);
	EXPECT_EQ(output.getShape()[2], 4);
	EXPECT_EQ(output.getShape()[3], 5);
	
	expectFinite(output);
}

TEST(Identity, PassthroughValues) {
	// Test that Identity preserves values (approximately)
	oa::Identity identity;
	
	std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
	auto input = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(input_data.data()), input_data.size() * sizeof(float)),
		oa::MatrixShape{4}
	);
	
	auto output = identity.forward(input);
	
	std::vector<float> output_data(4);
	[[maybe_unused]] auto status = oa::FnMatrix::copyToHost(output, output_data.data(), 4 * sizeof(float));
	
	for (size_t i = 0; i < input_data.size(); ++i) {
		EXPECT_FLOAT_EQ(output_data[i], input_data[i]) << "index " << i;
	}
}

