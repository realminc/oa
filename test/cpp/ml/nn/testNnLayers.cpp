// Test nn layer operations
// Tests for complex neural network layers: Embedding, Normalization, Convolution

#include <oa/ml/nn.h>
#include <oaTest.h>

// ============================================================================
// EMBEDDING TESTS
// ============================================================================

TEST(Embedding, BasicLookup) {
	oa::Embedding embed(10, 4);
	auto out = embed.forward(makeByteIndices({1, 5, 9}));
	
	// expected output shape: [3, 4]
	ASSERT_EQ(out.getShape().rank, 2);
	ASSERT_EQ(out.getShape()[0], 3);
	ASSERT_EQ(out.getShape()[1], 4);
	
	// verify output is finite
	expectFinite(out);
}

TEST(Embedding, BatchLookup) {
	oa::Embedding embed(256, 64);
	
	// Create sequence of indices: [10] - single sequence of 10 tokens
	oa::Vec<oa::U8> indices_data(10);
	for (oa::I32 i = 0; i < 10; ++i) {
		indices_data[i] = static_cast<oa::U8>(i % 256);
	}
	
	auto indices = oa::FnMatrix::empty(oa::MatrixShape{10}, oa::ScalarType::UInt8);
	std::memcpy(indices.dataAs<oa::U8>(), indices_data.data(), 10);
	
	auto out = embed.forward(indices);
	
	// expected output shape: [10, 64]
	ASSERT_EQ(out.getShape().rank, 2);
	ASSERT_EQ(out.getShape()[0], 10);
	ASSERT_EQ(out.getShape()[1], 64);
	
	expectFinite(out);
}

TEST(Embedding, LargeVocab) {
	// Test with realistic vocabulary size
	oa::Embedding embed(50000, 512);
	auto out = embed.forward(makeByteIndices({0, 100, 200}));
	
	ASSERT_EQ(out.getShape().rank, 2);
	ASSERT_EQ(out.getShape()[0], 3);
	ASSERT_EQ(out.getShape()[1], 512);
	
	expectFinite(out);
}

// ============================================================================
// LAYER NORMALIZATION TESTS
// ============================================================================

TEST(LayerNorm, BasicNormalization) {
	oa::LayerNorm ln(4);
	auto out = ln.forward(oa::FnMatrix::rand(oa::MatrixShape{2, 4}));
	
	// expected output shape: [2, 4]
	ASSERT_EQ(out.getShape().rank, 2);
	ASSERT_EQ(out.getShape()[0], 2);
	ASSERT_EQ(out.getShape()[1], 4);
	
	expectFinite(out);
}

TEST(LayerNorm, BatchNormalization) {
	oa::LayerNorm ln(128);
	auto out = ln.forward(oa::FnMatrix::rand(oa::MatrixShape{4, 16, 128}));
	
	// expected output shape: [4, 16, 128]
	ASSERT_EQ(out.getShape().rank, 3);
	ASSERT_EQ(out.getShape()[0], 4);
	ASSERT_EQ(out.getShape()[1], 16);
	ASSERT_EQ(out.getShape()[2], 128);
	
	expectFinite(out);
}

TEST(LayerNorm, LargeFeatures) {
	oa::LayerNorm ln(2048);
	auto out = ln.forward(oa::FnMatrix::rand(oa::MatrixShape{8, 2048}));
	
	ASSERT_EQ(out.getShape().rank, 2);
	ASSERT_EQ(out.getShape()[0], 8);
	ASSERT_EQ(out.getShape()[1], 2048);
	
	expectFinite(out);
}

// ============================================================================
// RMS NORMALIZATION TESTS
// ============================================================================

TEST(RmsNorm, BasicNormalization) {
	oa::RmsNorm rms(4);
	auto out = rms.forward(oa::FnMatrix::rand(oa::MatrixShape{2, 4}));
	
	// expected output shape: [2, 4]
	ASSERT_EQ(out.getShape().rank, 2);
	ASSERT_EQ(out.getShape()[0], 2);
	ASSERT_EQ(out.getShape()[1], 4);
	
	expectFinite(out);
}

TEST(RmsNorm, BatchNormalization) {
	oa::RmsNorm rms(256);
	auto out = rms.forward(oa::FnMatrix::rand(oa::MatrixShape{2, 32, 256}));
	
	// expected output shape: [2, 32, 256]
	ASSERT_EQ(out.getShape().rank, 3);
	ASSERT_EQ(out.getShape()[0], 2);
	ASSERT_EQ(out.getShape()[1], 32);
	ASSERT_EQ(out.getShape()[2], 256);
	
	expectFinite(out);
}

TEST(RmsNorm, LargeFeatures) {
	oa::RmsNorm rms(4096);
	auto out = rms.forward(oa::FnMatrix::rand(oa::MatrixShape{4, 4096}));
	
	ASSERT_EQ(out.getShape().rank, 2);
	ASSERT_EQ(out.getShape()[0], 4);
	ASSERT_EQ(out.getShape()[1], 4096);
	
	expectFinite(out);
}

// ============================================================================
// CONVOLUTION TESTS
// ============================================================================

TEST(Conv1d, BasicConvolution) {
	oa::Conv1d conv(1, 1, 3);
	auto out = conv.forward(oa::FnMatrix::rand(oa::MatrixShape{1, 1, 10}));
	
	// expected output shape: [1, 1, 8] (with default padding=0, stride=1)
	ASSERT_EQ(out.getShape().rank, 3);
	ASSERT_EQ(out.getShape()[0], 1);
	ASSERT_EQ(out.getShape()[1], 1);
	ASSERT_EQ(out.getShape()[2], 8);
	
	expectFinite(out);
}

TEST(Conv1d, MultiChannel) {
	oa::Conv1d conv(3, 16, 5);
	auto out = conv.forward(oa::FnMatrix::rand(oa::MatrixShape{2, 3, 32}));
	
	// expected output shape: [2, 16, 28]
	ASSERT_EQ(out.getShape().rank, 3);
	ASSERT_EQ(out.getShape()[0], 2);
	ASSERT_EQ(out.getShape()[1], 16);
	ASSERT_EQ(out.getShape()[2], 28);
	
	expectFinite(out);
}

TEST(Conv2d, BasicConvolution) {
	oa::Conv2d conv(1, 1, 3);
	auto out = conv.forward(oa::FnMatrix::rand(oa::MatrixShape{1, 1, 8, 8}));
	
	// expected output shape: [1, 1, 6, 6] (with default padding=0, stride=1)
	ASSERT_EQ(out.getShape().rank, 4);
	ASSERT_EQ(out.getShape()[0], 1);
	ASSERT_EQ(out.getShape()[1], 1);
	ASSERT_EQ(out.getShape()[2], 6);
	ASSERT_EQ(out.getShape()[3], 6);
	
	expectFinite(out);
}

TEST(Conv2d, MultiChannel) {
	oa::Conv2d conv(3, 32, 3);
	auto out = conv.forward(oa::FnMatrix::rand(oa::MatrixShape{4, 3, 32, 32}));
	
	// expected output shape: [4, 32, 30, 30]
	ASSERT_EQ(out.getShape().rank, 4);
	ASSERT_EQ(out.getShape()[0], 4);
	ASSERT_EQ(out.getShape()[1], 32);
	ASSERT_EQ(out.getShape()[2], 30);
	ASSERT_EQ(out.getShape()[3], 30);
	
	expectFinite(out);
}

TEST(Conv2d, LargeKernel) {
	oa::Conv2d conv(16, 64, 7);
	auto out = conv.forward(oa::FnMatrix::rand(oa::MatrixShape{2, 16, 64, 64}));
	
	// expected output shape: [2, 64, 58, 58]
	ASSERT_EQ(out.getShape().rank, 4);
	ASSERT_EQ(out.getShape()[0], 2);
	ASSERT_EQ(out.getShape()[1], 64);
	ASSERT_EQ(out.getShape()[2], 58);
	ASSERT_EQ(out.getShape()[3], 58);
	
	expectFinite(out);
}

// ============================================================================
// LINEAR LAYER TESTS
// ============================================================================

TEST(Linear, BasicForward) {
	oa::Linear linear(4, 2);
	auto out = linear.forward(oa::FnMatrix::rand(oa::MatrixShape{1, 4}));
	
	// expected output shape: [1, 2]
	ASSERT_EQ(out.getShape().rank, 2);
	ASSERT_EQ(out.getShape()[0], 1);
	ASSERT_EQ(out.getShape()[1], 2);
	
	expectFinite(out);
}

TEST(Linear, BatchForward) {
	oa::Linear linear(128, 64);
	auto out = linear.forward(oa::FnMatrix::rand(oa::MatrixShape{8, 128}));
	
	// expected output shape: [8, 64]
	ASSERT_EQ(out.getShape().rank, 2);
	ASSERT_EQ(out.getShape()[0], 8);
	ASSERT_EQ(out.getShape()[1], 64);
	
	expectFinite(out);
}

TEST(Linear, SequenceForward) {
	oa::Linear linear(512, 256);
	auto out = linear.forward(oa::FnMatrix::rand(oa::MatrixShape{4, 32, 512}));
	
	// expected output shape: [4, 32, 256]
	ASSERT_EQ(out.getShape().rank, 3);
	ASSERT_EQ(out.getShape()[0], 4);
	ASSERT_EQ(out.getShape()[1], 32);
	ASSERT_EQ(out.getShape()[2], 256);
	
	expectFinite(out);
}

TEST(Linear, LargeLayer) {
	oa::Linear linear(2048, 4096);
	auto out = linear.forward(oa::FnMatrix::rand(oa::MatrixShape{2, 2048}));
	
	ASSERT_EQ(out.getShape().rank, 2);
	ASSERT_EQ(out.getShape()[0], 2);
	ASSERT_EQ(out.getShape()[1], 4096);
	
	expectFinite(out);
}

// ============================================================================
// ACTIVATION LAYER TESTS
// ============================================================================

TEST(Gelu, BasicActivation) {
	oa::Gelu gelu;
	auto out = gelu.forward(oa::FnMatrix::rand(oa::MatrixShape{4, 8}));
	
	ASSERT_EQ(out.getShape().rank, 2);
	ASSERT_EQ(out.getShape()[0], 4);
	ASSERT_EQ(out.getShape()[1], 8);
	
	expectFinite(out);
}

TEST(Relu, BasicActivation) {
	oa::Relu relu;
	auto out = relu.forward(oa::FnMatrix::rand(oa::MatrixShape{4, 8}));
	
	ASSERT_EQ(out.getShape().rank, 2);
	ASSERT_EQ(out.getShape()[0], 4);
	ASSERT_EQ(out.getShape()[1], 8);
	
	expectFinite(out);
}

TEST(Silu, BasicActivation) {
	oa::Silu silu;
	auto out = silu.forward(oa::FnMatrix::rand(oa::MatrixShape{4, 8}));
	
	ASSERT_EQ(out.getShape().rank, 2);
	ASSERT_EQ(out.getShape()[0], 4);
	ASSERT_EQ(out.getShape()[1], 8);
	
	expectFinite(out);
}

TEST(Softmax, BasicActivation) {
	oa::Softmax softmax(-1);
	auto input = oa::FnMatrix::rand(oa::MatrixShape{4, 10});
	auto out = softmax.forward(input);
	
	ASSERT_EQ(out.getShape().rank, 2);
	ASSERT_EQ(out.getShape()[0], 4);
	ASSERT_EQ(out.getShape()[1], 10);
	
	// execute graph and copy to host for validation
	(void)testSubmitAndWait(oa::ExecutionSession::getActive());
	
	expectFinite(out);
	expectValidProbability(out);
}

// ============================================================================
// POOLING LAYER TESTS
// ============================================================================

TEST(MaxPool2d, BasicPooling) {
	oa::MaxPool2d pool(2);
	auto out = pool.forward(oa::FnMatrix::rand(oa::MatrixShape{1, 1, 8, 8}));
	
	// expected output shape: [1, 1, 4, 4]
	ASSERT_EQ(out.getShape().rank, 4);
	ASSERT_EQ(out.getShape()[0], 1);
	ASSERT_EQ(out.getShape()[1], 1);
	ASSERT_EQ(out.getShape()[2], 4);
	ASSERT_EQ(out.getShape()[3], 4);
	
	expectFinite(out);
}

TEST(AvgPool2d, BasicPooling) {
	oa::AvgPool2d pool(2);
	auto out = pool.forward(oa::FnMatrix::rand(oa::MatrixShape{2, 3, 16, 16}));
	
	// expected output shape: [2, 3, 8, 8]
	ASSERT_EQ(out.getShape().rank, 4);
	ASSERT_EQ(out.getShape()[0], 2);
	ASSERT_EQ(out.getShape()[1], 3);
	ASSERT_EQ(out.getShape()[2], 8);
	ASSERT_EQ(out.getShape()[3], 8);
	
	expectFinite(out);
}
