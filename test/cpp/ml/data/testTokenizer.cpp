#include <oa/ml/tokenizer.h>

#include <gtest/gtest.h>

#include <cstdio>
#include <cstring>

namespace {

void expectSame(const oa::Vector<oa::I32>& inA, const oa::Vector<oa::I32>& inB) {
	ASSERT_EQ(inA.size(), inB.size());
	for (oa::Usize i = 0; i < inA.size(); ++i) EXPECT_EQ(inA[i], inB[i]) << "token " << i;
}

} // namespace

TEST(BpeTokenizer, LearnsFullWidthNestedTokensAndRoundTrips) {
	constexpr const char* corpus =
		"forward strike forward strike backward guard forward strike";
	oa::BpeTokenizer tokenizer(320);
	tokenizer.train(corpus, 64);

	EXPECT_GT(tokenizer.numMerges(), 0);
	EXPECT_EQ(tokenizer.vocabSize(), 256 + tokenizer.numMerges());
	const auto encoded = tokenizer.encode(corpus);
	EXPECT_LT(encoded.size(), std::strlen(corpus));
	bool hasLearnedToken = false;
	for (oa::I32 token : encoded) hasLearnedToken |= token >= 256;
	EXPECT_TRUE(hasLearnedToken);
	EXPECT_EQ(tokenizer.decode(encoded), corpus);
}

TEST(BpeTokenizer, TrainingIsDeterministic) {
	constexpr const char* corpus = "left right left right fast slow fast slow";
	oa::BpeTokenizer a(300), b(300);
	a.train(corpus, 44);
	b.train(corpus, 44);
	EXPECT_EQ(a.numMerges(), b.numMerges());
	expectSame(a.encode(corpus), b.encode(corpus));
}

TEST(BpeTokenizer, PersistsExactVocabulary) {
	constexpr const char* corpus = "a fighter steps forward and swings the sword";
	const oa::String path = "/tmp/oa_bpe_tokenizer_test.txt";
	oa::BpeTokenizer trained(320);
	trained.train(corpus, 64);
	ASSERT_TRUE(trained.save(path).isOk());

	oa::BpeTokenizer loaded;
	ASSERT_TRUE(loaded.load(path).isOk());
	EXPECT_EQ(loaded.vocabSize(), trained.vocabSize());
	expectSame(loaded.encode(corpus), trained.encode(corpus));
	EXPECT_EQ(loaded.decode(loaded.encode(corpus)), corpus);
	std::remove(path.cStr());
}
