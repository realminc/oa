#include <Oa/Ml/Tokenizer.h>

#include <gtest/gtest.h>

#include <cstdio>
#include <cstring>

namespace {

void ExpectSame(const OaVec<OaI32>& InA, const OaVec<OaI32>& InB) {
	ASSERT_EQ(InA.Size(), InB.Size());
	for (OaUsize i = 0; i < InA.Size(); ++i) EXPECT_EQ(InA[i], InB[i]) << "token " << i;
}

} // namespace

TEST(BpeTokenizer, LearnsFullWidthNestedTokensAndRoundTrips) {
	constexpr const char* corpus =
		"forward strike forward strike backward guard forward strike";
	OaBpeTokenizer tokenizer(320);
	tokenizer.Train(corpus, 64);

	EXPECT_GT(tokenizer.NumMerges(), 0);
	EXPECT_EQ(tokenizer.VocabSize(), 256 + tokenizer.NumMerges());
	const auto encoded = tokenizer.Encode(corpus);
	EXPECT_LT(encoded.Size(), std::strlen(corpus));
	bool hasLearnedToken = false;
	for (OaI32 token : encoded) hasLearnedToken |= token >= 256;
	EXPECT_TRUE(hasLearnedToken);
	EXPECT_EQ(tokenizer.Decode(encoded), corpus);
}

TEST(BpeTokenizer, TrainingIsDeterministic) {
	constexpr const char* corpus = "left right left right fast slow fast slow";
	OaBpeTokenizer a(300), b(300);
	a.Train(corpus, 44);
	b.Train(corpus, 44);
	EXPECT_EQ(a.NumMerges(), b.NumMerges());
	ExpectSame(a.Encode(corpus), b.Encode(corpus));
}

TEST(BpeTokenizer, PersistsExactVocabulary) {
	constexpr const char* corpus = "a fighter steps forward and swings the sword";
	const OaString path = "/tmp/oa_bpe_tokenizer_test.txt";
	OaBpeTokenizer trained(320);
	trained.Train(corpus, 64);
	ASSERT_TRUE(trained.Save(path).IsOk());

	OaBpeTokenizer loaded;
	ASSERT_TRUE(loaded.Load(path).IsOk());
	EXPECT_EQ(loaded.VocabSize(), trained.VocabSize());
	ExpectSame(loaded.Encode(corpus), trained.Encode(corpus));
	EXPECT_EQ(loaded.Decode(loaded.Encode(corpus)), corpus);
	std::remove(path.CStr());
}
