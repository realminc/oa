#include "oaStdTest.h"

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

// oa::Allocator tests — cross-validated against std::allocator where the
// standard guarantees observable parity (storage class, overflow, traits).

static constexpr std::size_t kStressIters = 50'000;

TEST(Allocator, AllocBytes) {
	void* p = oa::allocBytes(256, alignof(std::max_align_t));
	ASSERT_NE(p, nullptr);
	std::memset(p, 0, 256);
	oa::freeBytes(p, alignof(std::max_align_t));
}

TEST(Allocator, TryAllocReportsInvalidAlignment) {
	const oa::AllocationResult result = oa::tryAllocBytes(256, 3);
	EXPECT_TRUE(result.isError());
	EXPECT_EQ(result.error, oa::AllocationError::InvalidAlignment);
	EXPECT_EQ(result.data, nullptr);
}

TEST(Allocator, TryAllocReportsArraySizeOverflow) {
	const oa::AllocationResult result = oa::tryAllocArray(
		std::numeric_limits<oa::Usize>::max(), 2, alignof(oa::U64));
	EXPECT_TRUE(result.isError());
	EXPECT_EQ(result.error, oa::AllocationError::SizeOverflow);
	EXPECT_EQ(result.data, nullptr);
}

TEST(Allocator, TryAllocZeroSizeIsSuccessfulAndNull) {
	const oa::AllocationResult result = oa::tryAllocBytes(0, 64);
	EXPECT_TRUE(result.isOk());
	EXPECT_EQ(result.data, nullptr);
}

TEST(Allocator, ReallocPreservesDefaultAlignedBytes) {
	constexpr oa::Usize kInitialBytes = 64;
	constexpr oa::Usize kExpandedBytes = 4096;
	auto* bytes = static_cast<oa::U8*>(oa::allocBytes(kInitialBytes));
	ASSERT_NE(bytes, nullptr);
	for (oa::Usize index = 0; index < kInitialBytes; ++index) {
		bytes[index] = static_cast<oa::U8>(index ^ 0xA5U);
	}

	bytes = static_cast<oa::U8*>(oa::reallocBytes(bytes, kExpandedBytes));
	ASSERT_NE(bytes, nullptr);
	for (oa::Usize index = 0; index < kInitialBytes; ++index) {
		EXPECT_EQ(bytes[index], static_cast<oa::U8>(index ^ 0xA5U));
	}
	oa::freeBytes(bytes);
}

TEST(Allocator, TryReallocRejectsOverAlignment) {
	const oa::AllocationResult result = oa::tryReallocBytes(nullptr, 64, 128);
	EXPECT_TRUE(result.isError());
	EXPECT_EQ(result.error, oa::AllocationError::InvalidAlignment);
	EXPECT_EQ(result.data, nullptr);
}

TEST(Allocator, ReallocZeroSizeReleasesAndReturnsNull) {
	void* bytes = oa::allocBytes(64);
	ASSERT_NE(bytes, nullptr);
	const oa::AllocationResult result = oa::tryReallocBytes(bytes, 0);
	EXPECT_TRUE(result.isOk());
	EXPECT_EQ(result.data, nullptr);
}

// Raw bytes: same fill pattern as std::allocator<char> for default-aligned size.
TEST(StdAllocatorVsStd, AllocBytesMatchesStdAllocatorChar) {
	constexpr std::size_t kBytes = 4096;
	void* const oa = oa::allocBytes(kBytes, alignof(std::max_align_t));
	ASSERT_NE(oa, nullptr);
	std::allocator<char> stdAlloc;
	char* const st = stdAlloc.allocate(kBytes);
	ASSERT_NE(st, nullptr);
	const unsigned char pat = 0xA7U;
	std::memset(oa, static_cast<int>(pat), kBytes);
	std::memset(st, static_cast<int>(pat), kBytes);
	EXPECT_EQ(std::memcmp(oa, st, kBytes), 0);
	oa::freeBytes(oa, alignof(std::max_align_t));
	stdAlloc.deallocate(st, kBytes);
}

TEST(StdAllocatorVsStd, AllocBytesOverAlignedWritable) {
	constexpr std::size_t kAlign = 128;
	constexpr std::size_t kBytes = 512;
	void* const p = oa::allocBytes(kBytes, kAlign);
	ASSERT_NE(p, nullptr);
	EXPECT_EQ(reinterpret_cast<std::uintptr_t>(p) % kAlign, 0U);
	std::memset(p, 0x5EU, kBytes);
	volatile unsigned char* vp = static_cast<unsigned char*>(p);
	std::size_t sum = 0;
	for (std::size_t i = 0; i < kBytes; ++i) {
		sum += vp[i];
	}
	EXPECT_EQ(sum, kBytes * 0x5EU);
	oa::freeBytes(p, kAlign);
}

TEST(StdAllocatorVsStd, FreeBytesNullNoCrash) {
	oa::freeBytes(nullptr, alignof(std::max_align_t));
	oa::freeBytes(nullptr);
}

TEST(Allocator, Scalar) {
	oa::Allocator<int> alloc;
	int* p = alloc.allocate(16);
	ASSERT_NE(p, nullptr);
	for (int idx = 0; idx < 16; ++idx) {
		p[idx] = idx;
	}
	alloc.deallocate(p, 16);
}

TEST(Allocator, OverAligned) {
	struct alignas(128) BigAlign {
		oa::U8 Pad;
	};
	oa::Allocator<BigAlign> alloc;
	BigAlign* p = alloc.allocate(2);
	ASSERT_NE(p, nullptr);
	EXPECT_EQ(reinterpret_cast<std::uintptr_t>(p) % 128U, 0U);
	alloc.deallocate(p, 2);
}

// Over-aligned T: both allocators must yield same alignment; contents match after identical writes.
TEST(StdAllocatorVsStd, OverAlignedMatchesStd) {
	struct alignas(64) Tag {
		std::uint64_t w;
	};

	oa::Allocator<Tag> oaAlloc;
	std::allocator<Tag> stdAlloc;
	Tag* const po = oaAlloc.allocate(3);
	Tag* const ps = stdAlloc.allocate(3);
	ASSERT_NE(po, nullptr);
	ASSERT_NE(ps, nullptr);
	EXPECT_EQ(reinterpret_cast<std::uintptr_t>(po) % alignof(Tag), 0U);
	EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ps) % alignof(Tag), 0U);
	for (int i = 0; i < 3; ++i) {
		po[i].w = static_cast<std::uint64_t>(i) * 0x9E3779B97F4A7C15ULL;
		ps[i].w = po[i].w;
	}
	for (int i = 0; i < 3; ++i) {
		EXPECT_EQ(po[i].w, ps[i].w);
	}
	oaAlloc.deallocate(po, 3);
	stdAlloc.deallocate(ps, 3);
}

TEST(Allocator, StdVectorInterop) {
	std::vector<int, oa::Allocator<int>> v;
	v.reserve(64);
	for (int idx = 0; idx < 64; ++idx) {
		v.push_back(idx);
	}
	ASSERT_EQ(v.size(), 64U);
	for (int idx = 0; idx < 64; ++idx) {
		EXPECT_EQ(v[static_cast<std::size_t>(idx)], idx);
	}
}

TEST(StdAllocatorVsStd, VectorIntParallelIdentical) {
	std::vector<int, oa::Allocator<int>> oa;
	std::vector<int, std::allocator<int>> st;
	oa.reserve(2000);
	st.reserve(2000);
	for (int i = 0; i < 2000; ++i) {
		const auto state = static_cast<std::uint32_t>(i) * 1103515245U + 12345U;
		const int x = static_cast<int>(state & 0x7fffffffU);
		oa.push_back(x);
		st.push_back(x);
	}
	ASSERT_EQ(oa.size(), st.size());
	EXPECT_TRUE(std::equal(oa.begin(), oa.end(), st.begin(), st.end()));
}

TEST(StdAllocatorVsStd, ListDequeParallelIdentical) {
	std::list<int, oa::Allocator<int>> oaList;
	std::list<int, std::allocator<int>> stList;
	std::deque<int, oa::Allocator<int>> oaDeq;
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

TEST(StdAllocatorVsStd, BasicStringParallelEqual) {
	using AllocatorString =
		std::basic_string<char, std::char_traits<char>, oa::Allocator<char>>;
	AllocatorString oaString("realm.software banking-grade allocator parity");
	std::string st("realm.software banking-grade allocator parity");
	ASSERT_EQ(oaString.size(), st.size());
	EXPECT_EQ(0, oaString.compare(0, oaString.size(), st.data(), st.size()));
	oaString += " append";
	st += " append";
	ASSERT_EQ(oaString.size(), st.size());
	EXPECT_EQ(0, oaString.compare(0, oaString.size(), st.data(), st.size()));
}

TEST(StdAllocatorVsStd, MaxSizeEqualsStd) {
	oa::Allocator<long double> oa;
	std::allocator<long double> st;
	using StTraits = std::allocator_traits<std::allocator<long double>>;
	EXPECT_EQ(oa.maxSize(), StTraits::max_size(st));
}

TEST(StdAllocatorVsStd, AllocatorTraitsRoundTrip) {
	using AllocatorType = oa::Allocator<std::uint64_t>;
	using Traits = std::allocator_traits<AllocatorType>;
	AllocatorType a;
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

TEST(Allocator, AllocateZeroCountIsSuccessfulAndNull) {
	oa::Allocator<int> oa;
	int* const po = oa.allocate(0);
	EXPECT_EQ(po, nullptr);
	oa.deallocate(po, 0);
}

TEST(StdAllocatorVsStd, AllocateWithHintIgnoresHintLikeStd) {
	oa::Allocator<double> oa;
	std::allocator<double> st;
	using StTraits = std::allocator_traits<std::allocator<double>>;
	double* const po = oa.allocate(8, nullptr);
	double* const ps = StTraits::allocate(st, 8, nullptr);
	ASSERT_NE(po, nullptr);
	ASSERT_NE(ps, nullptr);
	for (int i = 0; i < 8; ++i) {
		po[i] = static_cast<double>(i) * 1.4142135623;
		ps[i] = po[i];
	}
	EXPECT_EQ(std::memcmp(po, ps, 8 * sizeof(double)), 0);
	oa.deallocate(po, 8);
	StTraits::deallocate(st, ps, 8);
}

TEST(Allocator, OverflowTerminatesThroughFatalPath) {
	oa::Allocator<std::uint64_t> oa;
	const std::size_t bad =
		std::numeric_limits<std::size_t>::max() / sizeof(std::uint64_t) + 1;
	EXPECT_DEATH(static_cast<void>(oa.allocate(bad)), "OA allocation failure: size overflow");
}

TEST(Allocator, MaxSizePlusOneTerminatesThroughFatalPath) {
	oa::Allocator<int> oa;
	const std::size_t bad = oa.maxSize() + 1;
	EXPECT_DEATH(static_cast<void>(oa.allocate(bad)), "OA allocation failure: size overflow");
}

TEST(Allocator, DeallocateNullNoCrash) {
	oa::Allocator<int> a;
	a.deallocate(nullptr, 0);
}

TEST(Allocator, RebindSameAsStdPattern) {
	static_assert(
		std::is_same_v<typename oa::Allocator<int>::template rebind<double>::other,
			oa::Allocator<double>>,
		"rebind<double> must yield oa::Allocator<double>");
	using StRebound = std::allocator_traits<std::allocator<int>>::rebind_alloc<double>;
	static_assert(std::is_same_v<StRebound, std::allocator<double>>, "std traits rebind");
}

TEST(StdAllocatorVsStd, CrossTypeEqualityMatchesStatelessSemantics) {
	oa::Allocator<int> ai;
	oa::Allocator<double> ad;
	std::allocator<int> si;
	std::allocator<double> sd;
	(void)si;
	(void)sd;
	EXPECT_TRUE(ai == ad);
	EXPECT_FALSE(ai != ad);
	EXPECT_TRUE(std::allocator_traits<oa::Allocator<int>>::is_always_equal::value);
	EXPECT_TRUE(std::allocator_traits<std::allocator<int>>::is_always_equal::value);
}

TEST(Allocator, SelectOnContainerCopyConstruction) {
	oa::Allocator<int> a;
	oa::Allocator<int> b = a.selectOnContainerCopyConstruction();
	(void)b;
}

TEST(StdAllocatorVsStd, PropagateTraitsMatchStd) {
	using Oa = oa::Allocator<char>;
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
TEST(StdAllocatorVsStd, DeterministicStressParallel) {
	oa::Allocator<std::uint32_t> oa;
	std::allocator<std::uint32_t> st;
	std::minstd_rand rng(0xC0FFEEu);
	std::uniform_int_distribution<std::size_t> distCnt(1, 256);
	for (std::size_t iter = 0; iter < kStressIters; ++iter) {
		const std::size_t n = distCnt(rng);
		std::uint32_t* po = oa.allocate(n);
		std::uint32_t* ps = st.allocate(n);
		ASSERT_NE(po, nullptr);
		ASSERT_NE(ps, nullptr);
		for (std::size_t i = 0; i < n; ++i) {
			const std::uint32_t v = static_cast<std::uint32_t>(rng());
			po[i] = v;
			ps[i] = v;
		}
		EXPECT_EQ(std::memcmp(po, ps, n * sizeof(std::uint32_t)), 0);
		oa.deallocate(po, n);
		st.deallocate(ps, n);
	}
}

// Same RNG sequence for both sides; wall time only (parity is covered above).
TEST(StdAllocatorVsStd, TimedStressWallMs) {
	constexpr std::size_t kTimedIters = 100'000;
	oa::Allocator<std::uint32_t> oa;
	std::allocator<std::uint32_t> st;
	std::uniform_int_distribution<std::size_t> distCnt(1, 256);

	auto runOa = [&]() {
		std::minstd_rand rng(0xBEEFCAFEu);
		for (std::size_t iter = 0; iter < kTimedIters; ++iter) {
			const std::size_t n = distCnt(rng);
			std::uint32_t* po = oa.allocate(n);
			for (std::size_t i = 0; i < n; ++i) {
				po[i] = static_cast<std::uint32_t>(rng());
			}
			oa.deallocate(po, n);
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

	const auto t0 = oa::highResolutionNow();
	runOa();
	const auto t1 = oa::highResolutionNow();
	runStd();
	const auto t2 = oa::highResolutionNow();

	const auto oaMs = (t1 - t0).milliseconds();
	const auto stMs = (t2 - t1).milliseconds();
	fprintf(stderr,
		"  [oastd] TimedStress: OaStd=%lld ms  std=%lld ms  iters=%zu  ratio=%.2f (Oa/std)\n",
		static_cast<long long>(oaMs), static_cast<long long>(stMs), kTimedIters,
		stMs > 0 ? static_cast<double>(oaMs) / static_cast<double>(stMs) : 0.0);
	EXPECT_GE(oaMs, 0);
	EXPECT_GE(stMs, 0);
}
