// Byte-Level ML Tests (Vulkan dispatch via OaVkTestEnvironment)

#include "../../OaTest.h"

TEST(Byte, EncodeText) {
	auto t = OaByteEncoder::EncodeText("Hello");
	EXPECT_EQ(t.NumElements(), 5);
	EXPECT_EQ(t.DataAs<const OaU8>()[0], 'H');
	EXPECT_EQ(t.DataAs<const OaU8>()[4], 'o');
}

TEST(Byte, DecodeArgmax) {
	auto logits = OaFnMatrix::Zeros(OaMatrixShape{3, 256});
	logits.Set(0 * 256 + 'H', 10.0f);
	logits.Set(1 * 256 + 'i', 10.0f);
	logits.Set(2 * 256 + '!', 10.0f);
	auto text = OaByteEncoder::DecodeText(logits);
	EXPECT_EQ(text, "Hi!");
}

TEST(Byte, SampleLowTemp) {
	auto logits = OaFnMatrix::Zeros(OaMatrixShape{1, 256});
	logits.Set(65, 10.0f);  // 'A'
	auto sampled = OaByteEncoder::Sample(logits, 0.1f, 0.9f);
	EXPECT_EQ(sampled.Size(), 1u);
	EXPECT_EQ(sampled[0], 65);
}

TEST(Byte, ByteEmbeddingForward) {
	OaByteEmbedding emb(32);
	auto bytes = OaFnMatrix::Empty(OaMatrixShape{2, 5}, OaScalarType::UInt8);
	OaU8* b = bytes.DataAs<OaU8>();
	for (int i = 0; i < 10; ++i) b[i] = static_cast<OaU8>(i * 25);
	auto out = emb.Forward(bytes);
	OaExpectShape(out, {10, 32});
}

TEST(Byte, ByteEmbeddingConsistency) {
	OaByteEmbedding emb(16);
	auto bytes = OaMakeByteIndices({42, 99, 42});
	auto out = emb.Forward(bytes);
	for (int d = 0; d < 16; ++d) {
		EXPECT_FLOAT_EQ(out.At(0 * 16 + d), out.At(2 * 16 + d));
	}
}

TEST(Byte, ByteEmbeddingUInt32IndicesMatchUInt8) {
	OaByteEmbedding emb(8);
	auto u8 = OaMakeByteIndices({10, 200, 5});
	auto u32 = OaFnMatrix::Empty(OaMatrixShape{3}, OaScalarType::UInt32);
	u32.DataAs<OaU32>()[0] = 10;
	u32.DataAs<OaU32>()[1] = 200;
	u32.DataAs<OaU32>()[2] = 5;
	auto a = emb.Forward(u8);
	auto b = emb.Forward(u32);
	for (OaI64 i = 0; i < a.NumElements(); ++i) {
		EXPECT_FLOAT_EQ(a.At(i), b.At(i));
	}
}

TEST(Byte, ByteHeadForward) {
	OaByteHead head(32);
	auto out = head.Forward(OaFnMatrix::Rand(OaMatrixShape{10, 32}));
	OaExpectShape(out, {10, 256});
	OaExpectFinite(out);
}

TEST(Byte, Constants) {
	EXPECT_EQ(OA_BYTE_VOCAB_SIZE, 256);
	EXPECT_EQ(OA_BYTE_PAD, 0x00);
	EXPECT_EQ(OA_BYTE_BOS, 0x01);
	EXPECT_EQ(OA_BYTE_EOS, 0x02);
}
