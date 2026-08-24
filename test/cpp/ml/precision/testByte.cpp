// Byte-level ML tests (vulkan dispatch via VkTestEnvironment)

#include "../../oaTest.h"

TEST(Byte, EncodeText) {
	auto t = oa::ByteEncoder::encodeText("Hello");
	EXPECT_EQ(t.numElements(), 5);
	EXPECT_EQ(t.dataAs<const oa::U8>()[0], 'H');
	EXPECT_EQ(t.dataAs<const oa::U8>()[4], 'o');
}

TEST(Byte, DecodeArgmax) {
	auto logits = oa::FnMatrix::zeros(oa::MatrixShape{3, 256});
	logits.set(0 * 256 + 'H', 10.0f);
	logits.set(1 * 256 + 'i', 10.0f);
	logits.set(2 * 256 + '!', 10.0f);
	auto text = oa::ByteEncoder::decodeText(logits);
	EXPECT_EQ(text, "Hi!");
}

TEST(Byte, SampleLowTemp) {
	auto logits = oa::FnMatrix::zeros(oa::MatrixShape{1, 256});
	logits.set(65, 10.0f);  // 'A'
	auto sampled = oa::ByteEncoder::sample(logits, 0.1f, 0.9f);
	EXPECT_EQ(sampled.size(), 1u);
	EXPECT_EQ(sampled[0], 65);
}

TEST(Byte, ByteEmbeddingForward) {
	oa::ByteEmbedding emb(32);
	auto bytes = oa::FnMatrix::empty(oa::MatrixShape{2, 5}, oa::ScalarType::UInt8);
	oa::U8* b = bytes.dataAs<oa::U8>();
	for (int i = 0; i < 10; ++i) b[i] = static_cast<oa::U8>(i * 25);
	auto out = emb.forward(bytes);
	expectShape(out, {10, 32});
}

TEST(Byte, ByteEmbeddingConsistency) {
	oa::ByteEmbedding emb(16);
	auto bytes = makeByteIndices({42, 99, 42});
	auto out = emb.forward(bytes);
	for (int d = 0; d < 16; ++d) {
		EXPECT_FLOAT_EQ(out.at(0 * 16 + d), out.at(2 * 16 + d));
	}
}

TEST(Byte, ByteEmbeddingUInt32IndicesMatchUInt8) {
	oa::ByteEmbedding emb(8);
	auto u8 = makeByteIndices({10, 200, 5});
	auto u32 = oa::FnMatrix::empty(oa::MatrixShape{3}, oa::ScalarType::UInt32);
	u32.dataAs<oa::U32>()[0] = 10;
	u32.dataAs<oa::U32>()[1] = 200;
	u32.dataAs<oa::U32>()[2] = 5;
	auto a = emb.forward(u8);
	auto b = emb.forward(u32);
	for (oa::I64 i = 0; i < a.numElements(); ++i) {
		EXPECT_FLOAT_EQ(a.at(i), b.at(i));
	}
}

TEST(Byte, ByteHeadForward) {
	oa::ByteHead head(32);
	auto out = head.forward(oa::FnMatrix::rand(oa::MatrixShape{10, 32}));
	expectShape(out, {10, 256});
	expectFinite(out);
}

TEST(Byte, Constants) {
	EXPECT_EQ(oa::ByteVocabSize, 256);
	EXPECT_EQ(oa::BytePad, 0x00);
	EXPECT_EQ(oa::ByteBos, 0x01);
	EXPECT_EQ(oa::ByteEos, 0x02);
}
