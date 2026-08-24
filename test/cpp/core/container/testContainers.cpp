// oa::Vec vs std::vector — parity, growth behavior, and printed speed comparison.

#include "../../oaTest.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

namespace {

template<typename T>
static bool vecsEqual(const oa::Vec<T>& inOa, const std::vector<T>& inStd) {
	if (inOa.size() != inStd.size()) return false;
	for (oa::Usize i = 0; i < inOa.size(); ++i) {
		if (!(inOa[i] == inStd[i])) return false;
	}
	return true;
}

template<typename F>
static double timeAvgMsPerIter(oa::I32 inIters, F&& inFunc) {
	for (oa::I32 w = 0; w < 3; ++w) inFunc();
	const auto t0 = std::chrono::steady_clock::now();
	for (oa::I32 i = 0; i < inIters; ++i) inFunc();
	const auto t1 = std::chrono::steady_clock::now();
	const double totalMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
	return totalMs / static_cast<double>(inIters);
}

static void printSpeedTableHeader() {
	printf("\n");
	printf("=== oa::Vec vs std::vector — ms per full loop (lower = faster) ===\n");
	printf("    std/Oa:  >1.0 => oa::Vec faster   |   <1.0 => std::vector faster\n");
	printf("------------------------------------------------------------------------\n");
	printf("  %-46s %10s %10s %10s\n", "scenario", "oa::Vec ms", "std ms", "std/Oa");
	printf("------------------------------------------------------------------------\n");
}

static void printSpeedRow(const char* inLabel, double inOaMs, double inStdMs) {
	const double ratio = inOaMs > 0.0 ? inStdMs / inOaMs : 0.0;
	printf("  %-46s %10.3f %10.3f %10.2f\n", inLabel, inOaMs, inStdMs, ratio);
}

static double percentile(std::vector<double>& inSorted, double inP) {
	if (inSorted.empty()) return 0.0;
	const size_t n = inSorted.size();
	const double idx = inP * static_cast<double>(n - 1);
	const size_t lo = static_cast<size_t>(idx);
	const size_t hi = std::min(lo + 1, n - 1);
	const double f = idx - static_cast<double>(lo);
	return (inSorted[lo] * (1.0 - f)) + (inSorted[hi] * f);
}

template<typename Fo, typename Fs>
static void benchTrials(
	const char* inLabel,
	oa::I32 inTrials,
	oa::I32 inInnerIters,
	Fo&& inOaBench,
	Fs&& inStdBench) {
	std::vector<double> oaMs;
	std::vector<double> stMs;
	oaMs.reserve(static_cast<size_t>(inTrials));
	stMs.reserve(static_cast<size_t>(inTrials));
	for (oa::I32 t = 0; t < inTrials; ++t) {
		oaMs.push_back(timeAvgMsPerIter(inInnerIters, inOaBench));
		stMs.push_back(timeAvgMsPerIter(inInnerIters, inStdBench));
	}
	std::sort(oaMs.begin(), oaMs.end());
	std::sort(stMs.begin(), stMs.end());
	const double oaMed = percentile(oaMs, 0.5);
	const double stMed = percentile(stMs, 0.5);
	const double ratioMed = oaMed > 0.0 ? stMed / oaMed : 0.0;
	printf("  %-46s\n", inLabel);
	printf("    oa::Vec  min/med/max: %8.3f / %8.3f / %8.3f ms\n",
		oaMs.front(), oaMed, oaMs.back());
	printf("    std    min/med/max: %8.3f / %8.3f / %8.3f ms\n",
		stMs.front(), stMed, stMs.back());
	printf("    std/Oa (median): %6.2f\n", ratioMed);
	printf("\n");
}

} // namespace

TEST(CoreContainers, VecPushBackMatchesStdInt) {
	oa::Vec<oa::I32> oa;
	std::vector<oa::I32> st;
	for (oa::I32 i = 0; i < 1000; ++i) {
		oa.pushBack(i);
		st.push_back(i);
	}
	EXPECT_TRUE(vecsEqual(oa, st));
	EXPECT_EQ(oa.size(), 1000u);
}

TEST(CoreContainers, VecReserveResizeErase) {
	oa::Vec<oa::I32> oa;
	std::vector<oa::I32> st;
	oa.reserve(500);
	st.reserve(500);
	for (oa::I32 i = 0; i < 200; ++i) {
		oa.pushBack(i);
		st.push_back(i);
	}
	oa.resize(50);
	st.resize(50);
	EXPECT_TRUE(vecsEqual(oa, st));
	oa.resize(80, -1);
	st.resize(80, -1);
	EXPECT_TRUE(vecsEqual(oa, st));
	oa.erase(oa.begin() + 10, oa.begin() + 20);
	st.erase(st.begin() + 10, st.begin() + 20);
	EXPECT_TRUE(vecsEqual(oa, st));
}

TEST(CoreContainers, VecStringNonTrivial) {
	oa::Vec<std::string> oa;
	std::vector<std::string> st;
	oa.pushBack("hello");
	st.push_back("hello");
	oa.pushBack("realm");
	st.push_back("realm");
	EXPECT_TRUE(vecsEqual(oa, st));
	oa.insert(oa.begin() + 1, "middle");
	st.insert(st.begin() + 1, "middle");
	EXPECT_TRUE(vecsEqual(oa, st));
}

TEST(CoreContainers, VecAppendTrivialMatchesMemcpy) {
	oa::Vec<oa::I32> oa;
	std::vector<oa::I32> st;
	oa::I32 chunk[256];
	for (oa::I32 i = 0; i < 256; ++i) chunk[i] = i * 3;
	for (int r = 0; r < 4; ++r) {
		oa.append(chunk, 256);
		st.insert(st.end(), chunk, chunk + 256);
	}
	EXPECT_TRUE(vecsEqual(oa, st));
	EXPECT_EQ(oa.size(), 1024u);
}

TEST(CoreContainers, VecEqualityOperator) {
	oa::Vec<oa::I32> a;
	oa::Vec<oa::I32> b;
	a.pushBack(1);
	a.pushBack(2);
	b.pushBack(1);
	b.pushBack(2);
	EXPECT_TRUE(a == b);
	b.pushBack(3);
	EXPECT_FALSE(a == b);
}

// run:  ./test_containers --gtest_filter=CoreContainers.benchSpeedVsStd
TEST(CoreContainers, benchSpeedVsStd) {
	oa::Vec<oa::I32> oaI;
	std::vector<oa::I32> stI;
	oa::Vec<oa::F64> oaD;
	std::vector<oa::F64> stD;

	printSpeedTableHeader();

	// 1) Repeated growth: clear keeps capacity (steady-state appends, no realloc on inner loop).
	{
		const oa::I32 n = 500000;
		const oa::I32 iters = 80;
		const double oaMs = timeAvgMsPerIter(iters, [&]() {
			oaI.clear();
			for (oa::I32 i = 0; i < n; ++i) oaI.pushBack(i);
		});
		const double stMs = timeAvgMsPerIter(iters, [&]() {
			stI.clear();
			for (oa::I32 i = 0; i < n; ++i) stI.push_back(i);
		});
		printSpeedRow("push_back i32, N=500k, clear() only", oaMs, stMs);
		EXPECT_EQ(oaI.size(), static_cast<oa::Usize>(n));
		EXPECT_EQ(stI.size(), static_cast<size_t>(n));
	}

	// 2) Cold growth every iteration: release capacity then grow from empty.
	// oa::Vec uses realloc for trivial T — expect faster than typical std::vector (alloc+copy+free).
	{
		const oa::I32 n = 500000;
		const oa::I32 iters = 20;
		const double oaMs = timeAvgMsPerIter(iters, [&]() {
			oaI.clear();
			oaI.shrinkToFit();
			for (oa::I32 i = 0; i < n; ++i) oaI.pushBack(i);
		});
		const double stMs = timeAvgMsPerIter(iters, [&]() {
			stI.clear();
			stI.shrink_to_fit();
			for (oa::I32 i = 0; i < n; ++i) stI.push_back(i);
		});
		printSpeedRow("push_back i32, N=500k, clear+shrink_to_fit", oaMs, stMs);
		const double stdPerOa = oaMs > 0.0 ? stMs / oaMs : 0.0;
		EXPECT_GE(stdPerOa, 1.0) << "oa::Vec realloc path should be at least parity vs std::vector here";
	}

	// 3) No realloc: single reserve then fill.
	{
		const oa::I32 n = 500000;
		const oa::I32 iters = 100;
		const double oaMs = timeAvgMsPerIter(iters, [&]() {
			oaI.clear();
			oaI.reserve(static_cast<oa::Usize>(n));
			for (oa::I32 i = 0; i < n; ++i) oaI.pushBack(i);
		});
		const double stMs = timeAvgMsPerIter(iters, [&]() {
			stI.clear();
			stI.reserve(static_cast<size_t>(n));
			for (oa::I32 i = 0; i < n; ++i) stI.push_back(i);
		});
		printSpeedRow("reserve(N)+push_back i32, N=500k", oaMs, stMs);
	}

	// 4) Wider element: double.
	{
		const oa::I32 n = 250000;
		const oa::I32 iters = 30;
		const double oaMs = timeAvgMsPerIter(iters, [&]() {
			oaD.clear();
			for (oa::I32 i = 0; i < n; ++i) oaD.pushBack(static_cast<oa::F64>(i));
		});
		const double stMs = timeAvgMsPerIter(iters, [&]() {
			stD.clear();
			for (oa::I32 i = 0; i < n; ++i) stD.push_back(static_cast<oa::F64>(i));
		});
		printSpeedRow("push_back f64, N=250k, clear() only", oaMs, stMs);
	}

	printf("------------------------------------------------------------------------\n");
	printf("(Build with: cmake --preset release && ninja -C build/release)\n\n");
	fflush(stdout);
}

// Full statistical report: pin CPU (e.g. taskset -c 0) for stable numbers.
// run: ./test_containers --gtest_filter=CoreContainers.benchSpeedVsStdFullReport
TEST(CoreContainers, benchSpeedVsStdFullReport) {
	oa::Vec<oa::I32> oaI;
	std::vector<oa::I32> stI;
	oa::Vec<oa::U8> oaU;
	std::vector<oa::U8> stU;

	printf("\n");
	printf("================================================================================\n");
	printf(" oa::Vec vs std::vector — FULL REPORT (median over independent trials)\n");
	printf("================================================================================\n");
#if defined(__clang__)
	printf(" Compiler: Clang %s\n", __clang_version__);
#elif defined(__GNUC__)
	printf(" Compiler: GCC %d.%d.%d\n", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#else
	printf(" Compiler: (other)\n");
#endif
	printf(" C++ __cplusplus = %ld\n", static_cast<long>(__cplusplus));
	printf(" std/Oa median > 1.0 => std slower => oa::Vec faster for that scenario.\n");
	printf("--------------------------------------------------------------------------------\n\n");

	const oa::I32 kTrials = 21;
	const oa::I32 kInnerPush = 40;

	benchTrials(
		"push_back i32, N=500k, clear() only (capacity retained)",
		kTrials,
		kInnerPush,
		[&]() {
			oaI.clear();
			for (oa::I32 i = 0; i < 500000; ++i) oaI.pushBack(i);
		},
		[&]() {
			stI.clear();
			for (oa::I32 i = 0; i < 500000; ++i) stI.push_back(i);
		});

	benchTrials(
		"push_back i32, N=500k, clear+shrink_to_fit (cold growth / realloc path)",
		11,
		12,
		[&]() {
			oaI.clear();
			oaI.shrinkToFit();
			for (oa::I32 i = 0; i < 500000; ++i) oaI.pushBack(i);
		},
		[&]() {
			stI.clear();
			stI.shrink_to_fit();
			for (oa::I32 i = 0; i < 500000; ++i) stI.push_back(i);
		});

	benchTrials(
		"reserve(500k)+push_back i32 (no realloc in inner loop)",
		kTrials,
		80,
		[&]() {
			oaI.clear();
			oaI.reserve(500000);
			for (oa::I32 i = 0; i < 500000; ++i) oaI.pushBack(i);
		},
		[&]() {
			stI.clear();
			stI.reserve(500000);
			for (oa::I32 i = 0; i < 500000; ++i) stI.push_back(i);
		});

	static oa::U8 chunk[4096];
	for (oa::Usize i = 0; i < 4096; ++i) chunk[i] = static_cast<oa::U8>(i & 0xFF);
	constexpr oa::Usize kChunk = 4096;
	constexpr oa::Usize kBulkTotal = static_cast<oa::Usize>(10ull * 1000ull * 1000ull);
	const oa::I32 kFullChunks = static_cast<oa::I32>(kBulkTotal / kChunk);
	const oa::Usize kTail = kBulkTotal - (static_cast<oa::Usize>(kFullChunks) * kChunk);
	benchTrials(
		"append/insert 4Ki chunks -> 10M u8 (oa::memcpy vs insert iterator loop)",
		15,
		8,
		[&]() {
			oaU.clear();
			oaU.reserve(kBulkTotal);
			for (oa::I32 c = 0; c < kFullChunks; ++c) oaU.append(chunk, kChunk);
			if (kTail != 0) oaU.append(chunk, kTail);
		},
		[&]() {
			stU.clear();
			stU.reserve(kBulkTotal);
			for (oa::I32 c = 0; c < kFullChunks; ++c) {
				stU.insert(stU.end(), chunk, chunk + kChunk);
			}
			if (kTail != 0) stU.insert(stU.end(), chunk, chunk + kTail);
		});

	EXPECT_EQ(oaI.size(), 500000u);
	EXPECT_EQ(stI.size(), 500000u);
	EXPECT_EQ(oaU.size(), kBulkTotal);
	EXPECT_EQ(stU.size(), kBulkTotal);

	printf("--------------------------------------------------------------------------------\n");
	printf(" Note: steady push_back is often allocator + micro-arch noise; bulk append shows\n");
	printf("       oa::memcpy vs libstdc++/libc++ insert loop. Prefer reserve+append for hot IO.\n");
	printf("       Fixed arrays (no growth) remain fastest when max size is known — OaStd.md.\n");
	printf("================================================================================\n\n");
	fflush(stdout);
}
