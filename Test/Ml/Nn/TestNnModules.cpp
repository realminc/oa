// Tests for additional NN modules
// AdaptiveAvgPool2d, Flatten, SwiGLU, Softmax

#include <Oa/Ml/Nn.h>
#include <OaTest.h>

// ============================================================================
// ADAPTIVE AVERAGE POOLING TESTS
// ============================================================================

TEST(OaAdaptiveAvgPool2d, DownsampleToSinglePixel) {
	// Test downsampling to 1x1 (global average pooling)
	OaAdaptiveAvgPool2d pool(1, 1);
	
	// Input: [1, 3, 8, 8] -> Output: [1, 3, 1, 1]
	auto input = OaFnMatrix::Rand(OaMatrixShape{1, 3, 8, 8});
	auto output = pool.Forward(input);
	
	ASSERT_EQ(output.GetShape().Rank, 4);
	EXPECT_EQ(output.GetShape()[0], 1);  // batch
	EXPECT_EQ(output.GetShape()[1], 3);  // channels
	EXPECT_EQ(output.GetShape()[2], 1);  // height
	EXPECT_EQ(output.GetShape()[3], 1);  // width
	
	OaExpectFinite(output);
}

TEST(OaAdaptiveAvgPool2d, DownsampleTo2x2) {
	// Test downsampling to 2x2
	OaAdaptiveAvgPool2d pool(2, 2);
	
	// Input: [2, 4, 16, 16] -> Output: [2, 4, 2, 2]
	auto input = OaFnMatrix::Rand(OaMatrixShape{2, 4, 16, 16});
	auto output = pool.Forward(input);
	
	ASSERT_EQ(output.GetShape().Rank, 4);
	EXPECT_EQ(output.GetShape()[0], 2);
	EXPECT_EQ(output.GetShape()[1], 4);
	EXPECT_EQ(output.GetShape()[2], 2);
	EXPECT_EQ(output.GetShape()[3], 2);
	
	OaExpectFinite(output);
}

TEST(OaAdaptiveAvgPool2d, DownsampleTo7x7) {
	// Test downsampling to 7x7 (common in ResNet)
	OaAdaptiveAvgPool2d pool(7, 7);
	
	// Input: [1, 64, 14, 14] -> Output: [1, 64, 7, 7]
	auto input = OaFnMatrix::Rand(OaMatrixShape{1, 64, 14, 14});
	auto output = pool.Forward(input);
	
	ASSERT_EQ(output.GetShape().Rank, 4);
	EXPECT_EQ(output.GetShape()[0], 1);
	EXPECT_EQ(output.GetShape()[1], 64);
	EXPECT_EQ(output.GetShape()[2], 7);
	EXPECT_EQ(output.GetShape()[3], 7);
	
	OaExpectFinite(output);
}

TEST(OaAdaptiveAvgPool2d, NonSquareOutput) {
	// Test non-square output size
	OaAdaptiveAvgPool2d pool(4, 8);
	
	// Input: [1, 16, 32, 64] -> Output: [1, 16, 4, 8]
	auto input = OaFnMatrix::Rand(OaMatrixShape{1, 16, 32, 64});
	auto output = pool.Forward(input);
	
	ASSERT_EQ(output.GetShape().Rank, 4);
	EXPECT_EQ(output.GetShape()[0], 1);
	EXPECT_EQ(output.GetShape()[1], 16);
	EXPECT_EQ(output.GetShape()[2], 4);
	EXPECT_EQ(output.GetShape()[3], 8);
	
	OaExpectFinite(output);
}

TEST(OaAdaptiveAvgPool2d, BatchProcessing) {
	// Test with batch size > 1
	OaAdaptiveAvgPool2d pool(3, 3);
	
	// Input: [8, 32, 12, 12] -> Output: [8, 32, 3, 3]
	auto input = OaFnMatrix::Rand(OaMatrixShape{8, 32, 12, 12});
	auto output = pool.Forward(input);
	
	ASSERT_EQ(output.GetShape().Rank, 4);
	EXPECT_EQ(output.GetShape()[0], 8);
	EXPECT_EQ(output.GetShape()[1], 32);
	EXPECT_EQ(output.GetShape()[2], 3);
	EXPECT_EQ(output.GetShape()[3], 3);
	
	OaExpectFinite(output);
}

TEST(OaAdaptiveAvgPool2d, IdentityMapping) {
	// Test when output size equals input size (should be identity-like)
	OaAdaptiveAvgPool2d pool(8, 8);
	
	// Input: [1, 3, 8, 8] -> Output: [1, 3, 8, 8]
	auto input = OaFnMatrix::Rand(OaMatrixShape{1, 3, 8, 8});
	auto output = pool.Forward(input);
	
	ASSERT_EQ(output.GetShape().Rank, 4);
	EXPECT_EQ(output.GetShape()[0], 1);
	EXPECT_EQ(output.GetShape()[1], 3);
	EXPECT_EQ(output.GetShape()[2], 8);
	EXPECT_EQ(output.GetShape()[3], 8);
	
	OaExpectFinite(output);
}

// ============================================================================
// FLATTEN TESTS
// ============================================================================

TEST(OaFlatten, Flatten2DTo1D) {
	// Test flattening 2D to 1D
	OaFlatten flatten(0, -1);
	
	// Input: [4, 8] -> Output: [32]
	auto input = OaFnMatrix::Rand(OaMatrixShape{4, 8});
	auto output = flatten.Forward(input);
	
	ASSERT_EQ(output.GetShape().Rank, 1);
	EXPECT_EQ(output.GetShape()[0], 32);
	
	OaExpectFinite(output);
}

TEST(OaFlatten, Flatten3DKeepBatch) {
	// Test flattening 3D keeping batch dimension
	OaFlatten flatten(1, -1);
	
	// Input: [2, 4, 8] -> Output: [2, 32]
	auto input = OaFnMatrix::Rand(OaMatrixShape{2, 4, 8});
	auto output = flatten.Forward(input);
	
	ASSERT_EQ(output.GetShape().Rank, 2);
	EXPECT_EQ(output.GetShape()[0], 2);
	EXPECT_EQ(output.GetShape()[1], 32);
	
	OaExpectFinite(output);
}

TEST(OaFlatten, Flatten4DKeepBatch) {
	// Test flattening 4D (typical CNN output) keeping batch
	OaFlatten flatten(1, -1);
	
	// Input: [8, 64, 7, 7] -> Output: [8, 3136]
	auto input = OaFnMatrix::Rand(OaMatrixShape{8, 64, 7, 7});
	auto output = flatten.Forward(input);
	
	ASSERT_EQ(output.GetShape().Rank, 2);
	EXPECT_EQ(output.GetShape()[0], 8);
	EXPECT_EQ(output.GetShape()[1], 64 * 7 * 7);
	
	OaExpectFinite(output);
}

TEST(OaFlatten, FlattenPartialRange) {
	// Test flattening only middle dimensions
	OaFlatten flatten(1, 2);
	
	// Input: [2, 3, 4, 5] -> Output: [2, 12, 5]
	auto input = OaFnMatrix::Rand(OaMatrixShape{2, 3, 4, 5});
	auto output = flatten.Forward(input);
	
	ASSERT_EQ(output.GetShape().Rank, 3);
	EXPECT_EQ(output.GetShape()[0], 2);
	EXPECT_EQ(output.GetShape()[1], 12);  // 3 * 4
	EXPECT_EQ(output.GetShape()[2], 5);
	
	OaExpectFinite(output);
}

TEST(OaFlatten, FlattenSingleDimension) {
	// Test flattening a single dimension (no-op)
	OaFlatten flatten(1, 1);
	
	// Input: [2, 8, 4] -> Output: [2, 8, 4] (unchanged)
	auto input = OaFnMatrix::Rand(OaMatrixShape{2, 8, 4});
	auto output = flatten.Forward(input);
	
	ASSERT_EQ(output.GetShape().Rank, 3);
	EXPECT_EQ(output.GetShape()[0], 2);
	EXPECT_EQ(output.GetShape()[1], 8);
	EXPECT_EQ(output.GetShape()[2], 4);
	
	OaExpectFinite(output);
}

// ============================================================================
// SWIGLU TESTS
// ============================================================================

TEST(OaSwiGLU, BasicForward) {
	// Test basic SwiGLU forward pass
	OaSwiGLU swiglu(128, 512);
	
	// Input: [4, 16, 128] -> Output: [4, 16, 128]
	auto input = OaFnMatrix::Rand(OaMatrixShape{4, 16, 128});
	auto output = swiglu.Forward(input);
	
	ASSERT_EQ(output.GetShape().Rank, 3);
	EXPECT_EQ(output.GetShape()[0], 4);
	EXPECT_EQ(output.GetShape()[1], 16);
	EXPECT_EQ(output.GetShape()[2], 128);
	
	OaExpectFinite(output);
}

TEST(OaSwiGLU, LargeHiddenDim) {
	// Test with large hidden dimension
	OaSwiGLU swiglu(512, 2048);
	
	// Input: [2, 32, 512] -> Output: [2, 32, 512]
	auto input = OaFnMatrix::Rand(OaMatrixShape{2, 32, 512});
	auto output = swiglu.Forward(input);
	
	ASSERT_EQ(output.GetShape().Rank, 3);
	EXPECT_EQ(output.GetShape()[0], 2);
	EXPECT_EQ(output.GetShape()[1], 32);
	EXPECT_EQ(output.GetShape()[2], 512);
	
	OaExpectFinite(output);
}

TEST(OaSwiGLU, SmallBatch) {
	// Test with small batch size
	OaSwiGLU swiglu(64, 256);
	
	// Input: [1, 8, 64] -> Output: [1, 8, 64]
	auto input = OaFnMatrix::Rand(OaMatrixShape{1, 8, 64});
	auto output = swiglu.Forward(input);
	
	ASSERT_EQ(output.GetShape().Rank, 3);
	EXPECT_EQ(output.GetShape()[0], 1);
	EXPECT_EQ(output.GetShape()[1], 8);
	EXPECT_EQ(output.GetShape()[2], 64);
	
	OaExpectFinite(output);
}

// ============================================================================
// SOFTMAX TESTS
// ============================================================================

TEST(OaSoftmax, BasicSoftmax) {
	// Test basic softmax on last dimension
	OaSoftmax softmax(-1);
	
	// Input: [2, 4, 8] -> Output: [2, 4, 8]
	auto input = OaFnMatrix::Rand(OaMatrixShape{2, 4, 8});
	auto output = softmax.Forward(input);
	
	ASSERT_EQ(output.GetShape().Rank, 3);
	EXPECT_EQ(output.GetShape()[0], 2);
	EXPECT_EQ(output.GetShape()[1], 4);
	EXPECT_EQ(output.GetShape()[2], 8);
	
	OaExpectFinite(output);
}

TEST(OaSoftmax, SoftmaxDim0) {
	// Test softmax on dimension 0
	OaSoftmax softmax(0);
	
	// Input: [4, 8] -> Output: [4, 8]
	auto input = OaFnMatrix::Rand(OaMatrixShape{4, 8});
	auto output = softmax.Forward(input);
	
	ASSERT_EQ(output.GetShape().Rank, 2);
	EXPECT_EQ(output.GetShape()[0], 4);
	EXPECT_EQ(output.GetShape()[1], 8);
	
	OaExpectFinite(output);
}

TEST(OaSoftmax, SoftmaxDim1) {
	// Test softmax on dimension 1
	OaSoftmax softmax(1);
	
	// Input: [8, 16, 32] -> Output: [8, 16, 32]
	auto input = OaFnMatrix::Rand(OaMatrixShape{8, 16, 32});
	auto output = softmax.Forward(input);
	
	ASSERT_EQ(output.GetShape().Rank, 3);
	EXPECT_EQ(output.GetShape()[0], 8);
	EXPECT_EQ(output.GetShape()[1], 16);
	EXPECT_EQ(output.GetShape()[2], 32);
	
	OaExpectFinite(output);
}

TEST(OaSoftmax, LargeVocab) {
	// Test softmax with large vocabulary (typical in language models)
	OaSoftmax softmax(-1);
	
	// Input: [4, 128, 50000] -> Output: [4, 128, 50000]
	auto input = OaFnMatrix::Rand(OaMatrixShape{4, 128, 50000});
	auto output = softmax.Forward(input);
	
	ASSERT_EQ(output.GetShape().Rank, 3);
	EXPECT_EQ(output.GetShape()[0], 4);
	EXPECT_EQ(output.GetShape()[1], 128);
	EXPECT_EQ(output.GetShape()[2], 50000);
	
	OaExpectFinite(output);
}

// ============================================================================
// IDENTITY TESTS
// ============================================================================

TEST(OaIdentity, PassthroughShape) {
	// Test that Identity preserves shape
	OaIdentity identity;
	
	auto input = OaFnMatrix::Rand(OaMatrixShape{2, 3, 4, 5});
	auto output = identity.Forward(input);
	
	ASSERT_EQ(output.GetShape().Rank, 4);
	EXPECT_EQ(output.GetShape()[0], 2);
	EXPECT_EQ(output.GetShape()[1], 3);
	EXPECT_EQ(output.GetShape()[2], 4);
	EXPECT_EQ(output.GetShape()[3], 5);
	
	OaExpectFinite(output);
}

TEST(OaIdentity, PassthroughValues) {
	// Test that Identity preserves values (approximately)
	OaIdentity identity;
	
	std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
	auto input = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(input_data.data()), input_data.size() * sizeof(float)),
		OaMatrixShape{4}
	);
	
	auto output = identity.Forward(input);
	
	std::vector<float> output_data(4);
	[[maybe_unused]] auto status = OaFnMatrix::CopyToHost(output, output_data.data(), 4 * sizeof(float));
	
	for (size_t i = 0; i < input_data.size(); ++i) {
		EXPECT_FLOAT_EQ(output_data[i], input_data[i]) << "Index " << i;
	}
}

