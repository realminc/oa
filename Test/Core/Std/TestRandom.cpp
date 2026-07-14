#include "OaStdTest.h"

#include <Oa/Core/Std/Random.h>

#include <algorithm>
#include <cmath>
#include <vector>

// Golden PCG32 stream for seed=42, seq=1, computed independently (Python port of
// the exact algorithm). If these ever change, the seeded RNG stream shifted and
// every reproducible run (shuffles, weight init, dropout) moved with it.
TEST(OaStdRandom, GoldenU32Stream) {
	OaStdRandom rng(42u, 1u);
	const OaU32 expected[8] = {
		0x4DF1CCF9u, 0xE5838752u, 0x58ED9E10u, 0xF3E37B51u,
		0xE7664374u, 0x6AFDE4A8u, 0x8712391Eu, 0x738FC318u,
	};
	for (int i = 0; i < 8; ++i) {
		EXPECT_EQ(rng.NextU32(), expected[i]) << "at index " << i;
	}
}

TEST(OaStdRandom, GoldenU64Pairs) {
	OaStdRandom rng(42u, 1u);
	const OaU64 expected[4] = {
		0x4DF1CCF9E5838752ULL, 0x58ED9E10F3E37B51ULL,
		0xE76643746AFDE4A8ULL, 0x8712391E738FC318ULL,
	};
	for (int i = 0; i < 4; ++i) {
		EXPECT_EQ(rng.NextU64(), expected[i]) << "at index " << i;
	}
}

TEST(OaStdRandom, SameSeedSameStream) {
	OaStdRandom a(123u);
	OaStdRandom b(123u);
	for (int i = 0; i < 1000; ++i) {
		EXPECT_EQ(a.NextU32(), b.NextU32());
	}
}

TEST(OaStdRandom, DifferentSequenceDecorrelates) {
	OaStdRandom a(7u, 1u);
	OaStdRandom b(7u, 2u);
	int same = 0;
	for (int i = 0; i < 1000; ++i) {
		if (a.NextU32() == b.NextU32()) {
			++same;
		}
	}
	EXPECT_LT(same, 10);  // independent streams should rarely coincide
}

TEST(OaStdRandom, FloatDoubleInUnitRange) {
	OaStdRandom rng(99u);
	for (int i = 0; i < 100000; ++i) {
		const OaF32 f = rng.NextFloat();
		EXPECT_GE(f, 0.0F);
		EXPECT_LT(f, 1.0F);
		const OaF64 d = rng.NextDouble();
		EXPECT_GE(d, 0.0);
		EXPECT_LT(d, 1.0);
	}
}

TEST(OaStdRandom, RangeInclusiveAndCoversEndpoints) {
	OaStdRandom rng(5u);
	bool sawLo = false;
	bool sawHi = false;
	for (int i = 0; i < 100000; ++i) {
		const OaI64 v = rng.NextRange(3, 7);
		ASSERT_GE(v, 3);
		ASSERT_LE(v, 7);
		sawLo = sawLo || (v == 3);
		sawHi = sawHi || (v == 7);
	}
	EXPECT_TRUE(sawLo);
	EXPECT_TRUE(sawHi);
}

TEST(OaStdRandom, RangeDegenerate) {
	OaStdRandom rng(1u);
	EXPECT_EQ(rng.NextRange(5, 5), 5);
	EXPECT_EQ(rng.NextRange(9, 2), 9);  // max <= min returns min
}

TEST(OaStdRandom, GaussianMeanAndStdDev) {
	OaStdRandom rng(2024u);
	const int N = 200000;
	double sum   = 0.0;
	double sumSq = 0.0;
	for (int i = 0; i < N; ++i) {
		const double g = rng.NextGaussian(1.0, 2.0);
		sum   += g;
		sumSq += g * g;
	}
	const double mean = sum / N;
	const double var  = sumSq / N - mean * mean;
	EXPECT_NEAR(mean, 1.0, 0.05);
	EXPECT_NEAR(std::sqrt(var), 2.0, 0.05);
}

TEST(OaStdRandom, ShuffleIsPermutation) {
	OaStdRandom rng(77u);
	std::vector<int> v(100);
	for (int i = 0; i < 100; ++i) {
		v[i] = i;
	}
	rng.Shuffle(v.data(), v.size());
	std::sort(v.begin(), v.end());
	for (int i = 0; i < 100; ++i) {
		EXPECT_EQ(v[i], i);  // same multiset → a real permutation
	}
}

TEST(OaStdRandom, RawStateCheckpoint) {
	OaStdRandom rng(555u);
	for (int i = 0; i < 50; ++i) {
		rng.NextU32();
	}
	const OaU64 state = rng.RawState();
	const OaU64 inc   = rng.RawInc();
	const OaU32 a = rng.NextU32();
	const OaU32 b = rng.NextU32();

	OaStdRandom restored;
	restored.SetRawState(state, inc);
	EXPECT_EQ(restored.NextU32(), a);
	EXPECT_EQ(restored.NextU32(), b);
}
