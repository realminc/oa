#include "oaStdTest.h"

#include <oa/core/std/random.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

// Golden PCG32 stream for seed=42, seq=1, computed independently (Python port of
// the exact algorithm). If these ever change, the seeded RNG stream shifted and
// every reproducible run (shuffles, weight init, dropout) moved with it.
TEST(Random, GoldenU32Stream) {
	oa::Random rng(42u, 1u);
	const oa::U32 expected[8] = {
		0x4DF1CCF9u, 0xE5838752u, 0x58ED9E10u, 0xF3E37B51u,
		0xE7664374u, 0x6AFDE4A8u, 0x8712391Eu, 0x738FC318u,
	};
	for (int i = 0; i < 8; ++i) {
		EXPECT_EQ(rng.nextU32(), expected[i]) << "at index " << i;
	}
}

TEST(Random, GoldenU64Pairs) {
	oa::Random rng(42u, 1u);
	const oa::U64 expected[4] = {
		0x4DF1CCF9E5838752ULL, 0x58ED9E10F3E37B51ULL,
		0xE76643746AFDE4A8ULL, 0x8712391E738FC318ULL,
	};
	for (int i = 0; i < 4; ++i) {
		EXPECT_EQ(rng.nextU64(), expected[i]) << "at index " << i;
	}
}

TEST(Random, SameSeedSameStream) {
	oa::Random a(123u);
	oa::Random b(123u);
	for (int i = 0; i < 1000; ++i) {
		EXPECT_EQ(a.nextU32(), b.nextU32());
	}
}

TEST(Random, DifferentSequenceDecorrelates) {
	oa::Random a(7u, 1u);
	oa::Random b(7u, 2u);
	int same = 0;
	for (int i = 0; i < 1000; ++i) {
		if (a.nextU32() == b.nextU32()) {
			++same;
		}
	}
	EXPECT_LT(same, 10);  // independent streams should rarely coincide
}

TEST(Random, FloatDoubleInUnitRange) {
	oa::Random rng(99u);
	for (int i = 0; i < 100000; ++i) {
		const oa::F32 f = rng.nextFloat();
		EXPECT_GE(f, 0.0F);
		EXPECT_LT(f, 1.0F);
		const oa::F64 d = rng.nextDouble();
		EXPECT_GE(d, 0.0);
		EXPECT_LT(d, 1.0);
	}
}

TEST(Random, RangeInclusiveAndCoversEndpoints) {
	oa::Random rng(5u);
	bool sawLo = false;
	bool sawHi = false;
	for (int i = 0; i < 100000; ++i) {
		const oa::I64 v = rng.nextRange(3, 7);
		ASSERT_GE(v, 3);
		ASSERT_LE(v, 7);
		sawLo = sawLo || (v == 3);
		sawHi = sawHi || (v == 7);
	}
	EXPECT_TRUE(sawLo);
	EXPECT_TRUE(sawHi);
}

TEST(Random, RangeDegenerate) {
	oa::Random rng(1u);
	EXPECT_EQ(rng.nextRange(5, 5), 5);
	EXPECT_EQ(rng.nextRange(9, 2), 9);  // max <= min returns min
}

TEST(Random, SignedRangeHandlesFullDomainAndExtremeIntervals) {
	constexpr oa::I64 minimum = std::numeric_limits<oa::I64>::min();
	constexpr oa::I64 maximum = std::numeric_limits<oa::I64>::max();
	oa::Random rng(0xD15EA5EULL);

	for (int iteration = 0; iteration < 10'000; ++iteration) {
		const oa::I64 full = rng.nextRange(minimum, maximum);
		EXPECT_GE(full, minimum);
		EXPECT_LE(full, maximum);

		const oa::I64 low = rng.nextRange(minimum, minimum + 4'096);
		EXPECT_GE(low, minimum);
		EXPECT_LE(low, minimum + 4'096);

		const oa::I64 high = rng.nextRange(maximum - 4'096, maximum);
		EXPECT_GE(high, maximum - 4'096);
		EXPECT_LE(high, maximum);
	}
}

TEST(Random, GaussianMeanAndStdDev) {
	oa::Random rng(2024u);
	const int N = 200000;
	double sum   = 0.0;
	double sumSq = 0.0;
	for (int i = 0; i < N; ++i) {
		const double g = rng.nextGaussian(1.0, 2.0);
		sum   += g;
		sumSq += g * g;
	}
	const double mean = sum / N;
	const double var  = sumSq / N - mean * mean;
	EXPECT_NEAR(mean, 1.0, 0.05);
	EXPECT_NEAR(std::sqrt(var), 2.0, 0.05);
}

TEST(Random, ShuffleIsPermutation) {
	oa::Random rng(77u);
	std::vector<int> v(100);
	for (int i = 0; i < 100; ++i) {
		v[static_cast<std::size_t>(i)] = i;
	}
	rng.shuffle(v.data(), v.size());
	std::sort(v.begin(), v.end());
	for (int i = 0; i < 100; ++i) {
		EXPECT_EQ(v[static_cast<std::size_t>(i)], i);  // same multiset → a real permutation
	}
}

TEST(Random, RawStateCheckpoint) {
	oa::Random rng(555u);
	for (int i = 0; i < 50; ++i) {
		rng.nextU32();
	}
	const oa::U64 state = rng.rawState();
	const oa::U64 inc   = rng.rawInc();
	const oa::U32 a = rng.nextU32();
	const oa::U32 b = rng.nextU32();

	oa::Random restored;
	restored.setRawState(state, inc);
	EXPECT_EQ(restored.nextU32(), a);
	EXPECT_EQ(restored.nextU32(), b);
}

TEST(Random, CompleteCheckpointRestoresCachedGaussianExactly) {
	oa::Random rng(777u, 9u);
	(void)rng.nextGaussian(); // Creates a cached Box-Muller partner.
	const oa::Random::Checkpoint checkpoint = rng.checkpoint();
	const oa::F64 expectedGaussian = rng.nextGaussian(3.0, 2.0);
	const oa::U32 expectedInteger = rng.nextU32();
	(void)rng.nextGaussian();
	(void)rng.nextU64();

	rng.restoreCheckpoint(checkpoint);

	EXPECT_DOUBLE_EQ(rng.nextGaussian(3.0, 2.0), expectedGaussian);
	EXPECT_EQ(rng.nextU32(), expectedInteger);
}
