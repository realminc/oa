// Test Nn Layer Operations
// Tests for complex neural network layers: Embedding, Normalization, Convolution

#include <Oa/Ml/Nn.h>
#include <OaTest.h>

// ============================================================================
// EMBEDDING TESTS
// ============================================================================

TEST(OaEmbedding, BasicLookup) {
	OaEmbedding embed(10, 4);
	auto out = embed.Forward(OaMakeByteIndices({1, 5, 9}));
	
	// Expected output shape: [3, 4]
	ASSERT_EQ(out.GetShape().Rank, 2);
	ASSERT_EQ(out.GetShape()[0], 3);
	ASSERT_EQ(out.GetShape()[1], 4);
	
	// Verify output is finite
	OaExpectFinite(out);
}

TEST(OaEmbedding, BatchLookup) {
	OaEmbedding embed(256, 64);
	
	// Create sequence of indices: [10] - single sequence of 10 tokens
	OaVec<OaU8> indices_data(10);
	for (OaI32 i = 0; i < 10; ++i) {
		indices_data[i] = static_cast<OaU8>(i % 256);
	}
	
	auto indices = OaFnMatrix::Empty(OaMatrixShape{10}, OaScalarType::UInt8);
	std::memcpy(indices.DataAs<OaU8>(), indices_data.data(), 10);
	
	auto out = embed.Forward(indices);
	
	// Expected output shape: [10, 64]
	ASSERT_EQ(out.GetShape().Rank, 2);
	ASSERT_EQ(out.GetShape()[0], 10);
	ASSERT_EQ(out.GetShape()[1], 64);
	
	OaExpectFinite(out);
}

TEST(OaEmbedding, LargeVocab) {
	// Test with realistic vocabulary size
	OaEmbedding embed(50000, 512);
	auto out = embed.Forward(OaMakeByteIndices({0, 100, 200}));
	
	ASSERT_EQ(out.GetShape().Rank, 2);
	ASSERT_EQ(out.GetShape()[0], 3);
	ASSERT_EQ(out.GetShape()[1], 512);
	
	OaExpectFinite(out);
}

// ============================================================================
// LAYER NORMALIZATION TESTS
// ============================================================================

TEST(OaLayerNorm, BasicNormalization) {
	OaLayerNorm ln(4);
	auto out = ln.Forward(OaFnMatrix::Rand(OaMatrixShape{2, 4}));
	
	// Expected output shape: [2, 4]
	ASSERT_EQ(out.GetShape().Rank, 2);
	ASSERT_EQ(out.GetShape()[0], 2);
	ASSERT_EQ(out.GetShape()[1], 4);
	
	OaExpectFinite(out);
}

TEST(OaLayerNorm, BatchNormalization) {
	OaLayerNorm ln(128);
	auto out = ln.Forward(OaFnMatrix::Rand(OaMatrixShape{4, 16, 128}));
	
	// Expected output shape: [4, 16, 128]
	ASSERT_EQ(out.GetShape().Rank, 3);
	ASSERT_EQ(out.GetShape()[0], 4);
	ASSERT_EQ(out.GetShape()[1], 16);
	ASSERT_EQ(out.GetShape()[2], 128);
	
	OaExpectFinite(out);
}

TEST(OaLayerNorm, LargeFeatures) {
	OaLayerNorm ln(2048);
	auto out = ln.Forward(OaFnMatrix::Rand(OaMatrixShape{8, 2048}));
	
	ASSERT_EQ(out.GetShape().Rank, 2);
	ASSERT_EQ(out.GetShape()[0], 8);
	ASSERT_EQ(out.GetShape()[1], 2048);
	
	OaExpectFinite(out);
}

// ============================================================================
// RMS NORMALIZATION TESTS
// ============================================================================

TEST(OaRmsNorm, BasicNormalization) {
	OaRmsNorm rms(4);
	auto out = rms.Forward(OaFnMatrix::Rand(OaMatrixShape{2, 4}));
	
	// Expected output shape: [2, 4]
	ASSERT_EQ(out.GetShape().Rank, 2);
	ASSERT_EQ(out.GetShape()[0], 2);
	ASSERT_EQ(out.GetShape()[1], 4);
	
	OaExpectFinite(out);
}

TEST(OaRmsNorm, BatchNormalization) {
	OaRmsNorm rms(256);
	auto out = rms.Forward(OaFnMatrix::Rand(OaMatrixShape{2, 32, 256}));
	
	// Expected output shape: [2, 32, 256]
	ASSERT_EQ(out.GetShape().Rank, 3);
	ASSERT_EQ(out.GetShape()[0], 2);
	ASSERT_EQ(out.GetShape()[1], 32);
	ASSERT_EQ(out.GetShape()[2], 256);
	
	OaExpectFinite(out);
}

TEST(OaRmsNorm, LargeFeatures) {
	OaRmsNorm rms(4096);
	auto out = rms.Forward(OaFnMatrix::Rand(OaMatrixShape{4, 4096}));
	
	ASSERT_EQ(out.GetShape().Rank, 2);
	ASSERT_EQ(out.GetShape()[0], 4);
	ASSERT_EQ(out.GetShape()[1], 4096);
	
	OaExpectFinite(out);
}

// ============================================================================
// CONVOLUTION TESTS
// ============================================================================

TEST(OaConv1d, BasicConvolution) {
	OaConv1d conv(1, 1, 3);
	auto out = conv.Forward(OaFnMatrix::Rand(OaMatrixShape{1, 1, 10}));
	
	// Expected output shape: [1, 1, 8] (with default padding=0, stride=1)
	ASSERT_EQ(out.GetShape().Rank, 3);
	ASSERT_EQ(out.GetShape()[0], 1);
	ASSERT_EQ(out.GetShape()[1], 1);
	ASSERT_EQ(out.GetShape()[2], 8);
	
	OaExpectFinite(out);
}

TEST(OaConv1d, MultiChannel) {
	OaConv1d conv(3, 16, 5);
	auto out = conv.Forward(OaFnMatrix::Rand(OaMatrixShape{2, 3, 32}));
	
	// Expected output shape: [2, 16, 28]
	ASSERT_EQ(out.GetShape().Rank, 3);
	ASSERT_EQ(out.GetShape()[0], 2);
	ASSERT_EQ(out.GetShape()[1], 16);
	ASSERT_EQ(out.GetShape()[2], 28);
	
	OaExpectFinite(out);
}

TEST(OaConv2d, BasicConvolution) {
	OaConv2d conv(1, 1, 3);
	auto out = conv.Forward(OaFnMatrix::Rand(OaMatrixShape{1, 1, 8, 8}));
	
	// Expected output shape: [1, 1, 6, 6] (with default padding=0, stride=1)
	ASSERT_EQ(out.GetShape().Rank, 4);
	ASSERT_EQ(out.GetShape()[0], 1);
	ASSERT_EQ(out.GetShape()[1], 1);
	ASSERT_EQ(out.GetShape()[2], 6);
	ASSERT_EQ(out.GetShape()[3], 6);
	
	OaExpectFinite(out);
}

TEST(OaConv2d, MultiChannel) {
	OaConv2d conv(3, 32, 3);
	auto out = conv.Forward(OaFnMatrix::Rand(OaMatrixShape{4, 3, 32, 32}));
	
	// Expected output shape: [4, 32, 30, 30]
	ASSERT_EQ(out.GetShape().Rank, 4);
	ASSERT_EQ(out.GetShape()[0], 4);
	ASSERT_EQ(out.GetShape()[1], 32);
	ASSERT_EQ(out.GetShape()[2], 30);
	ASSERT_EQ(out.GetShape()[3], 30);
	
	OaExpectFinite(out);
}

TEST(OaConv2d, LargeKernel) {
	OaConv2d conv(16, 64, 7);
	auto out = conv.Forward(OaFnMatrix::Rand(OaMatrixShape{2, 16, 64, 64}));
	
	// Expected output shape: [2, 64, 58, 58]
	ASSERT_EQ(out.GetShape().Rank, 4);
	ASSERT_EQ(out.GetShape()[0], 2);
	ASSERT_EQ(out.GetShape()[1], 64);
	ASSERT_EQ(out.GetShape()[2], 58);
	ASSERT_EQ(out.GetShape()[3], 58);
	
	OaExpectFinite(out);
}

// ============================================================================
// LINEAR LAYER TESTS
// ============================================================================

TEST(OaLinear, BasicForward) {
	OaLinear linear(4, 2);
	auto out = linear.Forward(OaFnMatrix::Rand(OaMatrixShape{1, 4}));
	
	// Expected output shape: [1, 2]
	ASSERT_EQ(out.GetShape().Rank, 2);
	ASSERT_EQ(out.GetShape()[0], 1);
	ASSERT_EQ(out.GetShape()[1], 2);
	
	OaExpectFinite(out);
}

TEST(OaLinear, BatchForward) {
	OaLinear linear(128, 64);
	auto out = linear.Forward(OaFnMatrix::Rand(OaMatrixShape{8, 128}));
	
	// Expected output shape: [8, 64]
	ASSERT_EQ(out.GetShape().Rank, 2);
	ASSERT_EQ(out.GetShape()[0], 8);
	ASSERT_EQ(out.GetShape()[1], 64);
	
	OaExpectFinite(out);
}

TEST(OaLinear, SequenceForward) {
	OaLinear linear(512, 256);
	auto out = linear.Forward(OaFnMatrix::Rand(OaMatrixShape{4, 32, 512}));
	
	// Expected output shape: [4, 32, 256]
	ASSERT_EQ(out.GetShape().Rank, 3);
	ASSERT_EQ(out.GetShape()[0], 4);
	ASSERT_EQ(out.GetShape()[1], 32);
	ASSERT_EQ(out.GetShape()[2], 256);
	
	OaExpectFinite(out);
}

TEST(OaLinear, LargeLayer) {
	OaLinear linear(2048, 4096);
	auto out = linear.Forward(OaFnMatrix::Rand(OaMatrixShape{2, 2048}));
	
	ASSERT_EQ(out.GetShape().Rank, 2);
	ASSERT_EQ(out.GetShape()[0], 2);
	ASSERT_EQ(out.GetShape()[1], 4096);
	
	OaExpectFinite(out);
}

// ============================================================================
// ACTIVATION LAYER TESTS
// ============================================================================

TEST(OaGelu, BasicActivation) {
	OaGelu gelu;
	auto out = gelu.Forward(OaFnMatrix::Rand(OaMatrixShape{4, 8}));
	
	ASSERT_EQ(out.GetShape().Rank, 2);
	ASSERT_EQ(out.GetShape()[0], 4);
	ASSERT_EQ(out.GetShape()[1], 8);
	
	OaExpectFinite(out);
}

TEST(OaRelu, BasicActivation) {
	OaRelu relu;
	auto out = relu.Forward(OaFnMatrix::Rand(OaMatrixShape{4, 8}));
	
	ASSERT_EQ(out.GetShape().Rank, 2);
	ASSERT_EQ(out.GetShape()[0], 4);
	ASSERT_EQ(out.GetShape()[1], 8);
	
	OaExpectFinite(out);
}

TEST(OaSilu, BasicActivation) {
	OaSilu silu;
	auto out = silu.Forward(OaFnMatrix::Rand(OaMatrixShape{4, 8}));
	
	ASSERT_EQ(out.GetShape().Rank, 2);
	ASSERT_EQ(out.GetShape()[0], 4);
	ASSERT_EQ(out.GetShape()[1], 8);
	
	OaExpectFinite(out);
}

TEST(OaSoftmax, BasicActivation) {
	OaSoftmax softmax(-1);
	auto input = OaFnMatrix::Rand(OaMatrixShape{4, 10});
	auto out = softmax.Forward(input);
	
	ASSERT_EQ(out.GetShape().Rank, 2);
	ASSERT_EQ(out.GetShape()[0], 4);
	ASSERT_EQ(out.GetShape()[1], 10);
	
	// Execute graph and copy to host for validation
	(void)OaContext::GetDefault().Execute();
	(void)OaContext::GetDefault().Sync();
	
	OaExpectFinite(out);
	OaExpectValidProbability(out);
}

// ============================================================================
// POOLING LAYER TESTS
// ============================================================================

TEST(OaMaxPool2d, BasicPooling) {
	OaMaxPool2d pool(2);
	auto out = pool.Forward(OaFnMatrix::Rand(OaMatrixShape{1, 1, 8, 8}));
	
	// Expected output shape: [1, 1, 4, 4]
	ASSERT_EQ(out.GetShape().Rank, 4);
	ASSERT_EQ(out.GetShape()[0], 1);
	ASSERT_EQ(out.GetShape()[1], 1);
	ASSERT_EQ(out.GetShape()[2], 4);
	ASSERT_EQ(out.GetShape()[3], 4);
	
	OaExpectFinite(out);
}

TEST(OaAvgPool2d, BasicPooling) {
	OaAvgPool2d pool(2);
	auto out = pool.Forward(OaFnMatrix::Rand(OaMatrixShape{2, 3, 16, 16}));
	
	// Expected output shape: [2, 3, 8, 8]
	ASSERT_EQ(out.GetShape().Rank, 4);
	ASSERT_EQ(out.GetShape()[0], 2);
	ASSERT_EQ(out.GetShape()[1], 3);
	ASSERT_EQ(out.GetShape()[2], 8);
	ASSERT_EQ(out.GetShape()[3], 8);
	
	OaExpectFinite(out);
}

