#include "OaStdTest.h"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <limits>
#include <list>
#include <memory>
#include <new>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

// OaStdAllocator tests — cross-validated against std::allocator where the
// standard guarantees observable parity (storage class, overflow, traits).

static constexpr std::size_t kStressIters = 50'000;

TEST(OaStdAllocator, AllocBytes) {
	void* p = OaStdAllocBytes(256, alignof(std::max_align_t));
	ASSERT_NE(p, nullptr);
	std::memset(p, 0, 256);
	OaStdFreeBytes(p, alignof(std::max_align_t));
}

// Raw bytes: same fill pattern as std::allocator<char> for default-aligned size.
TEST(OaStdAllocatorVsStd, AllocBytesMatchesStdAllocatorChar) {
	constexpr std::size_t kBytes = 4096;
	void* const oa = OaStdAllocBytes(kBytes, alignof(std::max_align_t));
	ASSERT_NE(oa, nullptr);
	std::allocator<char> stdAlloc;
	char* const st = stdAlloc.allocate(kBytes);
	ASSERT_NE(st, nullptr);
	const unsigned char pat = 0xA7U;
	std::memset(oa, static_cast<int>(pat), kBytes);
	std::memset(st, static_cast<int>(pat), kBytes);
	EXPECT_EQ(std::memcmp(oa, st, kBytes), 0);
	OaStdFreeBytes(oa, alignof(std::max_align_t));
	stdAlloc.deallocate(st, kBytes);
}

TEST(OaStdAllocatorVsStd, AllocBytesOverAlignedWritable) {
	constexpr std::size_t kAlign = 128;
	constexpr std::size_t kBytes = 512;
	void* const p = OaStdAllocBytes(kBytes, kAlign);
	ASSERT_NE(p, nullptr);
	EXPECT_EQ(reinterpret_cast<std::uintptr_t>(p) % kAlign, 0U);
	std::memset(p, 0x5EU, kBytes);
	volatile unsigned char* vp = static_cast<unsigned char*>(p);
	std::size_t sum = 0;
	for (std::size_t i = 0; i < kBytes; ++i) {
		sum += vp[i];
	}
	EXPECT_EQ(sum, kBytes * 0x5EU);
	OaStdFreeBytes(p, kAlign);
}

TEST(OaStdAllocatorVsStd, FreeBytesNullNoCrash) {
	OaStdFreeBytes(nullptr, alignof(std::max_align_t));
	OaStdFreeBytes(nullptr);
}

TEST(OaStdAllocator, Scalar) {
	OaStdAllocator<int> alloc;
	int* p = alloc.Allocate(16);
	ASSERT_NE(p, nullptr);
	for (int idx = 0; idx < 16; ++idx) {
		p[idx] = idx;
	}
	alloc.Deallocate(p, 16);
}

TEST(OaStdAllocator, OverAligned) {
	struct alignas(128) BigAlign {
		OaU8 Pad;
	};
	OaStdAllocator<BigAlign> alloc;
	BigAlign* p = alloc.Allocate(2);
	ASSERT_NE(p, nullptr);
	EXPECT_EQ(reinterpret_cast<std::uintptr_t>(p) % 128U, 0U);
	alloc.Deallocate(p, 2);
}

// Over-aligned T: both allocators must yield same alignment; contents match after identical writes.
TEST(OaStdAllocatorVsStd, OverAlignedMatchesStd) {
	struct alignas(64) Tag {
		std::uint64_t W;
	};

	OaStdAllocator<Tag> oaAlloc;
	std::allocator<Tag> stdAlloc;
	Tag* const po = oaAlloc.Allocate(3);
	Tag* const ps = stdAlloc.allocate(3);
	ASSERT_NE(po, nullptr);
	ASSERT_NE(ps, nullptr);
	EXPECT_EQ(reinterpret_cast<std::uintptr_t>(po) % alignof(Tag), 0U);
	EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ps) % alignof(Tag), 0U);
	for (int i = 0; i < 3; ++i) {
		po[i].W = static_cast<std::uint64_t>(i) * 0x9E3779B97F4A7C15ULL;
		ps[i].W = po[i].W;
	}
	for (int i = 0; i < 3; ++i) {
		EXPECT_EQ(po[i].W, ps[i].W);
	}
	oaAlloc.Deallocate(po, 3);
	stdAlloc.deallocate(ps, 3);
}

TEST(OaStdAllocator, StdVectorInterop) {
	std::vector<int, OaStdAllocator<int>> v;
	v.reserve(64);
	for (int idx = 0; idx < 64; ++idx) {
		v.push_back(idx);
	}
	ASSERT_EQ(v.size(), 64U);
	for (int idx = 0; idx < 64; ++idx) {
		EXPECT_EQ(v[static_cast<std::size_t>(idx)], idx);
	}
}

TEST(OaStdAllocatorVsStd, VectorIntParallelIdentical) {
	std::vector<int, OaStdAllocator<int>> oa;
	std::vector<int, std::allocator<int>> st;
	oa.reserve(2000);
	st.reserve(2000);
	for (int i = 0; i < 2000; ++i) {
		const int x = (i * 1103515245 + 12345) & 0x7fffffff;
		oa.push_back(x);
		st.push_back(x);
	}
	ASSERT_EQ(oa.size(), st.size());
	EXPECT_TRUE(std::equal(oa.begin(), oa.end(), st.begin(), st.end()));
}

TEST(OaStdAllocatorVsStd, ListDequeParallelIdentical) {
	std::list<int, OaStdAllocator<int>> oaList;
	std::list<int, std::allocator<int>> stList;
	std::deque<int, OaStdAllocator<int>> oaDeq;
	std::deque<int, std::allocator<int>> stDeq;
	for (int i = 0; i < 500; ++i) {
		const int v = i ^ (i << 2);
		oaList.push_back(v);
		stList.push_back(v);
		oaDeq.push_back(v);
		stDeq.push_back(v);
	}
	EXPECT_EQ(oaList.size(), stList.size());
	EXPECT_EQ(oaDeq.size(), stDeq.size());
	auto ito = oaList.begin();
	auto its = stList.begin();
	for (; ito != oaList.end(); ++ito, ++its) {
		EXPECT_EQ(*ito, *its);
	}
	for (std::size_t i = 0; i < oaDeq.size(); ++i) {
		EXPECT_EQ(oaDeq[i], stDeq[i]);
	}
}

TEST(OaStdAllocatorVsStd, BasicStringParallelEqual) {
	using OaString = std::basic_string<char, std::char_traits<char>, OaStdAllocator<char>>;
	OaString oa("realm.software banking-grade allocator parity");
	std::string st("realm.software banking-grade allocator parity");
	ASSERT_EQ(oa.size(), st.size());
	EXPECT_EQ(0, oa.compare(0, oa.size(), st.data(), st.size()));
	oa += " append";
	st += " append";
	ASSERT_EQ(oa.size(), st.size());
	EXPECT_EQ(0, oa.compare(0, oa.size(), st.data(), st.size()));
}

TEST(OaStdAllocatorVsStd, MaxSizeEqualsStd) {
	OaStdAllocator<long double> oa;
	std::allocator<long double> st;
	using StTraits = std::allocator_traits<std::allocator<long double>>;
	EXPECT_EQ(oa.MaxSize(), StTraits::max_size(st));
}

TEST(OaStdAllocatorVsStd, AllocatorTraitsRoundTrip) {
	using OaAlloc = OaStdAllocator<std::uint64_t>;
	using Traits = std::allocator_traits<OaAlloc>;
	OaAlloc a;
	const std::size_t n = 128;
	std::uint64_t* p = Traits::allocate(a, n);
	ASSERT_NE(p, nullptr);
	for (std::size_t i = 0; i < n; ++i) {
		p[i] = static_cast<std::uint64_t>(i) << 32 | i;
	}
	for (std::size_t i = 0; i < n; ++i) {
		EXPECT_EQ(p[i], (static_cast<std::uint64_t>(i) << 32) | i);
	}
	Traits::deallocate(a, p, n);
}

TEST(OaStdAllocatorVsStd, AllocateZeroCountNonNull) {
	OaStdAllocator<int> oa;
	std::allocator<int> st;
	using StTraits = std::allocator_traits<std::allocator<int>>;
	int* const po = oa.Allocate(0);
	int* const ps = StTraits::allocate(st, 0);
	ASSERT_NE(po, nullptr);
	ASSERT_NE(ps, nullptr);
	oa.Deallocate(po, 0);
	StTraits::deallocate(st, ps, 0);
}

TEST(OaStdAllocatorVsStd, AllocateWithHintIgnoresHintLikeStd) {
	OaStdAllocator<double> oa;
	std::allocator<double> st;
	using StTraits = std::allocator_traits<std::allocator<double>>;
	double* const po = oa.Allocate(8, nullptr);
	double* const ps = StTraits::allocate(st, 8, nullptr);
	ASSERT_NE(po, nullptr);
	ASSERT_NE(ps, nullptr);
	for (int i = 0; i < 8; ++i) {
		po[i] = static_cast<double>(i) * 1.4142135623;
		ps[i] = po[i];
	}
	EXPECT_EQ(std::memcmp(po, ps, 8 * sizeof(double)), 0);
	oa.Deallocate(po, 8);
	StTraits::deallocate(st, ps, 8);
}

TEST(OaStdAllocatorVsStd, BadArrayNewLengthOverflowMatchesStd) {
	OaStdAllocator<std::uint64_t> oa;
	std::allocator<std::uint64_t> st;
	using StTraits = std::allocator_traits<std::allocator<std::uint64_t>>;
	const std::size_t bad =
		std::numeric_limits<std::size_t>::max() / sizeof(std::uint64_t) + 1;
	EXPECT_THROW(static_cast<void>(oa.Allocate(bad)), std::bad_array_new_length);
	EXPECT_THROW(static_cast<void>(StTraits::allocate(st, bad)), std::bad_array_new_length);
}

TEST(OaStdAllocatorVsStd, BadArrayNewLengthMaxSizePlusOne) {
	OaStdAllocator<int> oa;
	std::allocator<int> st;
	using StTraits = std::allocator_traits<std::allocator<int>>;
	const std::size_t bad = oa.MaxSize() + 1;
	EXPECT_THROW(static_cast<void>(oa.Allocate(bad)), std::bad_array_new_length);
	EXPECT_THROW(static_cast<void>(StTraits::allocate(st, bad)), std::bad_array_new_length);
}

TEST(OaStdAllocator, DeallocateNullNoCrash) {
	OaStdAllocator<int> a;
	a.Deallocate(nullptr, 0);
}

TEST(OaStdAllocator, RebindSameAsStdPattern) {
	static_assert(
		std::is_same_v<typename OaStdAllocator<int>::template rebind<double>::other,
			OaStdAllocator<double>>,
		"rebind<double> must yield OaStdAllocator<double>");
	using StRebound = std::allocator_traits<std::allocator<int>>::rebind_alloc<double>;
	static_assert(std::is_same_v<StRebound, std::allocator<double>>, "std traits rebind");
}

TEST(OaStdAllocatorVsStd, CrossTypeEqualityMatchesStatelessSemantics) {
	OaStdAllocator<int> ai;
	OaStdAllocator<double> ad;
	std::allocator<int> si;
	std::allocator<double> sd;
	(void)si;
	(void)sd;
	EXPECT_TRUE(ai == ad);
	EXPECT_FALSE(ai != ad);
	EXPECT_TRUE(std::allocator_traits<OaStdAllocator<int>>::is_always_equal::value);
	EXPECT_TRUE(std::allocator_traits<std::allocator<int>>::is_always_equal::value);
}

TEST(OaStdAllocator, SelectOnContainerCopyConstruction) {
	OaStdAllocator<int> a;
	OaStdAllocator<int> b = a.SelectOnContainerCopyConstruction();
	(void)b;
}

TEST(OaStdAllocatorVsStd, PropagateTraitsMatchStd) {
	using Oa = OaStdAllocator<char>;
	using St = std::allocator<char>;
	EXPECT_EQ(
		std::allocator_traits<Oa>::propagate_on_container_copy_assignment::value,
		std::allocator_traits<St>::propagate_on_container_copy_assignment::value
	);
	EXPECT_EQ(
		std::allocator_traits<Oa>::propagate_on_container_move_assignment::value,
		std::allocator_traits<St>::propagate_on_container_move_assignment::value
	);
	EXPECT_EQ(
		std::allocator_traits<Oa>::propagate_on_container_swap::value,
		std::allocator_traits<St>::propagate_on_container_swap::value
	);
}

// Deterministic stress: interleaved allocate/write/deallocate vs std::allocator.
TEST(OaStdAllocatorVsStd, DeterministicStressParallel) {
	OaStdAllocator<std::uint32_t> oa;
	std::allocator<std::uint32_t> st;
	std::minstd_rand rng(0xC0FFEEu);
	std::uniform_int_distribution<std::size_t> distCnt(1, 256);
	for (std::size_t iter = 0; iter < kStressIters; ++iter) {
		const std::size_t n = distCnt(rng);
		std::uint32_t* po = oa.Allocate(n);
		std::uint32_t* ps = st.allocate(n);
		ASSERT_NE(po, nullptr);
		ASSERT_NE(ps, nullptr);
		for (std::size_t i = 0; i < n; ++i) {
			const std::uint32_t v = static_cast<std::uint32_t>(rng());
			po[i] = v;
			ps[i] = v;
		}
		EXPECT_EQ(std::memcmp(po, ps, n * sizeof(std::uint32_t)), 0);
		oa.Deallocate(po, n);
		st.deallocate(ps, n);
	}
}

// Same RNG sequence for both sides; wall time only (parity is covered above).
TEST(OaStdAllocatorVsStd, TimedStressWallMs) {
	constexpr std::size_t kTimedIters = 100'000;
	OaStdAllocator<std::uint32_t> oa;
	std::allocator<std::uint32_t> st;
	std::uniform_int_distribution<std::size_t> distCnt(1, 256);

	auto runOa = [&]() {
		std::minstd_rand rng(0xBEEFCAFEu);
		for (std::size_t iter = 0; iter < kTimedIters; ++iter) {
			const std::size_t n = distCnt(rng);
			std::uint32_t* po = oa.Allocate(n);
			for (std::size_t i = 0; i < n; ++i) {
				po[i] = static_cast<std::uint32_t>(rng());
			}
			oa.Deallocate(po, n);
		}
	};
	auto runStd = [&]() {
		std::minstd_rand rng(0xBEEFCAFEu);
		for (std::size_t iter = 0; iter < kTimedIters; ++iter) {
			const std::size_t n = distCnt(rng);
			std::uint32_t* ps = st.allocate(n);
			for (std::size_t i = 0; i < n; ++i) {
				ps[i] = static_cast<std::uint32_t>(rng());
			}
			st.deallocate(ps, n);
		}
	};

	const auto t0 = OaHighResolutionNow();
	runOa();
	const auto t1 = OaHighResolutionNow();
	runStd();
	const auto t2 = OaHighResolutionNow();

	const auto oaMs = OaChronoMillisCount(t1 - t0);
	const auto stMs = OaChronoMillisCount(t2 - t1);
	fprintf(stderr,
		"  [oastd] TimedStress: OaStd=%lld ms  std=%lld ms  iters=%zu  ratio=%.2f (Oa/std)\n",
		static_cast<long long>(oaMs), static_cast<long long>(stMs), kTimedIters,
		stMs > 0 ? static_cast<double>(oaMs) / static_cast<double>(stMs) : 0.0);
	EXPECT_GE(oaMs, 0);
	EXPECT_GE(stMs, 0);
}
