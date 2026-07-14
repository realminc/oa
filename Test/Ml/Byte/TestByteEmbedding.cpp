// Test/Ml/Byte/TestByteEmbedding.cpp
// Tests for OaByteEmbedding module

#include <gtest/gtest.h>
#include <Oa/Core.h>
#include <Oa/Ml.h>
#include <vector>

// Helper to copy matrix to host
static std::vector<float> CopyMatrixToHost(const OaMatrix& mat) {
	std::vector<float> result(mat.NumElements());
	[[maybe_unused]] auto copy_result = OaFnMatrix::CopyToHost(mat, result.data(), result.size() * sizeof(float));
	return result;
}

// Helper to check if all values are finite
static void OaExpectFinite(const std::vector<float>& data, const char* name) {
	for (size_t i = 0; i < data.size(); ++i) {
		EXPECT_TRUE(std::isfinite(data[i])) << name << " contains non-finite value at index " << i;
	}
}

class ByteEmbedding : public ::testing::Test {
protected:
	void SetUp() override {
		// Initialize runtime if needed
	}
};

// ============================================================================
// OaByteEmbedding Tests
// ============================================================================

TEST_F(ByteEmbedding, ConstructionAndDModel) {
	// Test basic construction
	const OaI32 d_model = 64;
	OaByteEmbedding embed(d_model);
	
	EXPECT_EQ(embed.DModel(), d_model);
	
	// Check that weight parameter exists and has correct shape
	auto params = embed.Parameters();
	ASSERT_EQ(params.size(), 1);
	EXPECT_EQ(params[0].first, "weight");
	
	auto weight = params[0].second;
	EXPECT_EQ(weight.GetShape()[0], 256);  // OA_BYTE_VOCAB_SIZE
	EXPECT_EQ(weight.GetShape()[1], d_model);
}

TEST_F(ByteEmbedding, ForwardSingleToken) {
	// Test forward pass with single token
	const OaI32 d_model = 32;
	OaByteEmbedding embed(d_model);
	
	// Create input: [1, 1] with byte value 65 ('A')
	std::vector<OaU8> input_data = {65};
	auto input = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(input_data.data(), input_data.size()),
		OaMatrixShape{1, 1}
	);
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto output = embed.Forward(input);
	
	// Output should be [1, 1, d_model]
	EXPECT_EQ(output.GetShape()[0], 1);  // batch
	EXPECT_EQ(output.GetShape()[1], 1);  // seq
	EXPECT_EQ(output.GetShape()[2], d_model);
	
	auto result = CopyMatrixToHost(output);
	ASSERT_EQ(result.size(), d_model);
	OaExpectFinite(result, "ByteEmbedding output");
}

TEST_F(ByteEmbedding, ForwardSequence) {
	// Test forward pass with sequence
	const OaI32 d_model = 64;
	const OaI32 seq_len = 8;
	OaByteEmbedding embed(d_model);
	
	// Create input: [1, seq_len] with byte values "Hello!"
	std::vector<OaU8> input_data = {72, 101, 108, 108, 111, 33, 0, 0};  // "Hello!  "
	auto input = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(input_data.data(), input_data.size()),
		OaMatrixShape{1, seq_len}
	);
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto output = embed.Forward(input);
	
	// Output should be [1, seq_len, d_model]
	EXPECT_EQ(output.GetShape()[0], 1);
	EXPECT_EQ(output.GetShape()[1], seq_len);
	EXPECT_EQ(output.GetShape()[2], d_model);
	
	auto result = CopyMatrixToHost(output);
	ASSERT_EQ(result.size(), seq_len * d_model);
	OaExpectFinite(result, "ByteEmbedding sequence output");
}

TEST_F(ByteEmbedding, ForwardBatch) {
	// Test forward pass with batch
	const OaI32 d_model = 48;
	const OaI32 batch = 4;
	const OaI32 seq_len = 6;
	OaByteEmbedding embed(d_model);
	
	// Create input: [batch, seq_len]
	std::vector<OaU8> input_data(batch * seq_len);
	for (OaI32 i = 0; i < batch * seq_len; ++i) {
		input_data[i] = static_cast<OaU8>((i * 17 + 65) % 256);  // Pseudo-random bytes
	}
	
	auto input = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(input_data.data(), input_data.size()),
		OaMatrixShape{batch, seq_len}
	);
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto output = embed.Forward(input);
	
	// Output should be [batch, seq_len, d_model]
	EXPECT_EQ(output.GetShape()[0], batch);
	EXPECT_EQ(output.GetShape()[1], seq_len);
	EXPECT_EQ(output.GetShape()[2], d_model);
	
	auto result = CopyMatrixToHost(output);
	ASSERT_EQ(result.size(), batch * seq_len * d_model);
	OaExpectFinite(result, "ByteEmbedding batch output");
}

TEST_F(ByteEmbedding, ForwardAllByteValues) {
	// Test that all 256 byte values produce valid embeddings
	const OaI32 d_model = 32;
	OaByteEmbedding embed(d_model);
	
	// Create input with all byte values 0-255
	std::vector<OaU8> input_data(256);
	for (OaI32 i = 0; i < 256; ++i) {
		input_data[i] = static_cast<OaU8>(i);
	}
	
	auto input = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(input_data.data(), input_data.size()),
		OaMatrixShape{1, 256}
	);
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto output = embed.Forward(input);
	
	// Output should be [1, 256, d_model]
	EXPECT_EQ(output.GetShape()[0], 1);
	EXPECT_EQ(output.GetShape()[1], 256);
	EXPECT_EQ(output.GetShape()[2], d_model);
	
	auto result = CopyMatrixToHost(output);
	OaExpectFinite(result, "ByteEmbedding all bytes output");
	
	// Check that different bytes produce different embeddings
	bool all_same = true;
	for (OaI32 i = 1; i < 256; ++i) {
		for (OaI32 j = 0; j < d_model; ++j) {
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
	const OaI32 d_model = 512;
	const OaI32 seq_len = 16;
	OaByteEmbedding embed(d_model);
	
	std::vector<OaU8> input_data(seq_len);
	for (OaI32 i = 0; i < seq_len; ++i) {
		input_data[i] = static_cast<OaU8>((i * 13 + 32) % 128);  // ASCII range
	}
	
	auto input = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(input_data.data(), input_data.size()),
		OaMatrixShape{1, seq_len}
	);
	
	OaContext::Scope ctx_scope(OaContext::GetDefault());
	auto output = embed.Forward(input);
	
	EXPECT_EQ(output.GetShape()[0], 1);
	EXPECT_EQ(output.GetShape()[1], seq_len);
	EXPECT_EQ(output.GetShape()[2], d_model);
	
	auto result = CopyMatrixToHost(output);
	OaExpectFinite(result, "ByteEmbedding large model output");
}

TEST_F(ByteEmbedding, ParameterCount) {
	// Verify parameter count is correct
	const OaI32 d_model = 128;
	OaByteEmbedding embed(d_model);
	
	auto params = embed.Parameters();
	ASSERT_EQ(params.size(), 1);
	
	auto weight = params[0].second;
	OaI32 param_count = weight.NumElements();
	EXPECT_EQ(param_count, 256 * d_model);  // 256 bytes * d_model
}

