// Test/Ml/Byte/TestByteEmbedding.cpp
// Tests for oa::ByteEmbedding module

#include <gtest/gtest.h>
#include <oa/core.h>
#include <oa/ml.h>
#include <oa/runtime/executionSession.h>
#include <vector>

// Helper to copy matrix to host
static std::vector<float> copyMatrixToHost(const oa::Matrix& mat) {
	std::vector<float> result(mat.numElements());
	[[maybe_unused]] auto copy_result = oa::FnMatrix::copyToHost(mat, result.data(), result.size() * sizeof(float));
	return result;
}

// Helper to check if all values are finite
static void expectFinite(const std::vector<float>& data, const char* name) {
	for (size_t i = 0; i < data.size(); ++i) {
		EXPECT_TRUE(std::isfinite(data[i])) << name << " contains non-finite value at index " << i;
	}
}

class ByteEmbedding : public ::testing::Test {
protected:
	void SetUp() override {
		// initialize runtime if needed
	}
};

// ============================================================================
// oa::ByteEmbedding Tests
// ============================================================================

TEST_F(ByteEmbedding, ConstructionAndDModel) {
	// Test basic construction
	const oa::I32 d_model = 64;
	oa::ByteEmbedding embed(d_model);
	
	EXPECT_EQ(embed.dModel(), d_model);
	
	// Check that weight parameter exists and has correct shape
	auto params = embed.parameters();
	ASSERT_EQ(params.size(), 1);
	EXPECT_EQ(params[0].first, "weight");
	
	auto weight = params[0].second;
	EXPECT_EQ(weight.getShape()[0], 256);  // oa::ByteVocabSize
	EXPECT_EQ(weight.getShape()[1], d_model);
}

TEST_F(ByteEmbedding, ForwardSingleToken) {
	// Test forward pass with single token
	const oa::I32 d_model = 32;
	oa::ByteEmbedding embed(d_model);
	
	// Create input: [1, 1] with byte value 65 ('A')
	std::vector<oa::U8> input_data = {65};
	auto input = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(input_data.data(), input_data.size()),
		oa::MatrixShape{1, 1}
	);
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto output = embed.forward(input);
	
	// output should be [1, 1, d_model]
	EXPECT_EQ(output.getShape()[0], 1);  // batch
	EXPECT_EQ(output.getShape()[1], 1);  // seq
	EXPECT_EQ(output.getShape()[2], d_model);
	
	auto result = copyMatrixToHost(output);
	ASSERT_EQ(result.size(), d_model);
	expectFinite(result, "ByteEmbedding output");
}

TEST_F(ByteEmbedding, ForwardSequence) {
	// Test forward pass with sequence
	const oa::I32 d_model = 64;
	const oa::I32 seq_len = 8;
	oa::ByteEmbedding embed(d_model);
	
	// Create input: [1, seq_len] with byte values "Hello!"
	std::vector<oa::U8> input_data = {72, 101, 108, 108, 111, 33, 0, 0};  // "Hello!  "
	auto input = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(input_data.data(), input_data.size()),
		oa::MatrixShape{1, seq_len}
	);
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto output = embed.forward(input);
	
	// output should be [1, seq_len, d_model]
	EXPECT_EQ(output.getShape()[0], 1);
	EXPECT_EQ(output.getShape()[1], seq_len);
	EXPECT_EQ(output.getShape()[2], d_model);
	
	auto result = copyMatrixToHost(output);
	ASSERT_EQ(result.size(), seq_len * d_model);
	expectFinite(result, "ByteEmbedding sequence output");
}

TEST_F(ByteEmbedding, ForwardBatch) {
	// Test forward pass with batch
	const oa::I32 d_model = 48;
	const oa::I32 batch = 4;
	const oa::I32 seq_len = 6;
	oa::ByteEmbedding embed(d_model);
	
	// Create input: [batch, seq_len]
	std::vector<oa::U8> input_data(batch * seq_len);
	for (oa::I32 i = 0; i < batch * seq_len; ++i) {
		input_data[i] = static_cast<oa::U8>((i * 17 + 65) % 256);  // Pseudo-random bytes
	}
	
	auto input = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(input_data.data(), input_data.size()),
		oa::MatrixShape{batch, seq_len}
	);
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto output = embed.forward(input);
	
	// output should be [batch, seq_len, d_model]
	EXPECT_EQ(output.getShape()[0], batch);
	EXPECT_EQ(output.getShape()[1], seq_len);
	EXPECT_EQ(output.getShape()[2], d_model);
	
	auto result = copyMatrixToHost(output);
	ASSERT_EQ(result.size(), batch * seq_len * d_model);
	expectFinite(result, "ByteEmbedding batch output");
}

TEST_F(ByteEmbedding, ForwardAllByteValues) {
	// Test that all 256 byte values produce valid embeddings
	const oa::I32 d_model = 32;
	oa::ByteEmbedding embed(d_model);
	
	// Create input with all byte values 0-255
	std::vector<oa::U8> input_data(256);
	for (oa::I32 i = 0; i < 256; ++i) {
		input_data[i] = static_cast<oa::U8>(i);
	}
	
	auto input = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(input_data.data(), input_data.size()),
		oa::MatrixShape{1, 256}
	);
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto output = embed.forward(input);
	
	// output should be [1, 256, d_model]
	EXPECT_EQ(output.getShape()[0], 1);
	EXPECT_EQ(output.getShape()[1], 256);
	EXPECT_EQ(output.getShape()[2], d_model);
	
	auto result = copyMatrixToHost(output);
	expectFinite(result, "ByteEmbedding all bytes output");
	
	// Check that different bytes produce different embeddings
	bool all_same = true;
	for (oa::I32 i = 1; i < 256; ++i) {
		for (oa::I32 j = 0; j < d_model; ++j) {
			if (std::abs(result[i * d_model + j] - result[j]) > 1e-6f) {
				all_same = false;
				break;
			}
		}
		if (!all_same) break;
	}
	EXPECT_FALSE(all_same) << "All byte embeddings should not be identical";
}

TEST_F(ByteEmbedding, ForwardLargeModel) {
	// Test with larger d_model
	const oa::I32 d_model = 512;
	const oa::I32 seq_len = 16;
	oa::ByteEmbedding embed(d_model);
	
	std::vector<oa::U8> input_data(seq_len);
	for (oa::I32 i = 0; i < seq_len; ++i) {
		input_data[i] = static_cast<oa::U8>((i * 13 + 32) % 128);  // ASCII range
	}
	
	auto input = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(input_data.data(), input_data.size()),
		oa::MatrixShape{1, seq_len}
	);
	
	oa::ExecutionSession::RecordingScope ctx_scope(oa::ExecutionSession::getActive());
	auto output = embed.forward(input);
	
	EXPECT_EQ(output.getShape()[0], 1);
	EXPECT_EQ(output.getShape()[1], seq_len);
	EXPECT_EQ(output.getShape()[2], d_model);
	
	auto result = copyMatrixToHost(output);
	expectFinite(result, "ByteEmbedding large model output");
}

TEST_F(ByteEmbedding, ParameterCount) {
	// verify parameter count is correct
	const oa::I32 d_model = 128;
	oa::ByteEmbedding embed(d_model);
	
	auto params = embed.parameters();
	ASSERT_EQ(params.size(), 1);
	
	auto weight = params[0].second;
	oa::I32 param_count = weight.numElements();
	EXPECT_EQ(param_count, 256 * d_model);  // 256 bytes * d_model
}
