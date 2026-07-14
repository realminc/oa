// OaVec vs std::vector — parity, growth behavior, and printed speed comparison.

#include "../../OaTest.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

namespace {

template<typename T>
static bool VecsEqual(const OaVec<T>& InOa, const std::vector<T>& InStd) {
	if (InOa.Size() != InStd.size()) return false;
	for (OaUsize i = 0; i < InOa.Size(); ++i) {
		if (!(InOa[i] == InStd[i])) return false;
	}
	return true;
}

template<typename F>
static double TimeAvgMsPerIter(OaI32 InIters, F&& InFunc) {
	for (OaI32 w = 0; w < 3; ++w) InFunc();
	const auto t0 = std::chrono::steady_clock::now();
	for (OaI32 i = 0; i < InIters; ++i) InFunc();
	const auto t1 = std::chrono::steady_clock::now();
	const double totalMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
	return totalMs / static_cast<double>(InIters);
}

static void PrintSpeedTableHeader() {
	printf("\n");
	printf("=== OaVec vs std::vector — ms per full loop (lower = faster) ===\n");
	printf("    std/Oa:  >1.0 => OaVec faster   |   <1.0 => std::vector faster\n");
	printf("------------------------------------------------------------------------\n");
	printf("  %-46s %10s %10s %10s\n", "scenario", "OaVec ms", "std ms", "std/Oa");
	printf("------------------------------------------------------------------------\n");
}

static void PrintSpeedRow(const char* InLabel, double InOaMs, double InStdMs) {
	const double ratio = InOaMs > 0.0 ? InStdMs / InOaMs : 0.0;
	printf("  %-46s %10.3f %10.3f %10.2f\n", InLabel, InOaMs, InStdMs, ratio);
}

static double Percentile(std::vector<double>& InSorted, double InP) {
	if (InSorted.empty()) return 0.0;
	const size_t n = InSorted.size();
	const double idx = InP * static_cast<double>(n - 1);
	const size_t lo = static_cast<size_t>(idx);
	const size_t hi = std::min(lo + 1, n - 1);
	const double f = idx - static_cast<double>(lo);
	return (InSorted[lo] * (1.0 - f)) + (InSorted[hi] * f);
}

template<typename Fo, typename Fs>
static void BenchTrials(
	const char* InLabel,
	OaI32 InTrials,
	OaI32 InInnerIters,
	Fo&& InOaBench,
	Fs&& InStdBench) {
	std::vector<double> oaMs;
	std::vector<double> stMs;
	oaMs.reserve(static_cast<size_t>(InTrials));
	stMs.reserve(static_cast<size_t>(InTrials));
	for (OaI32 t = 0; t < InTrials; ++t) {
		oaMs.push_back(TimeAvgMsPerIter(InInnerIters, InOaBench));
		stMs.push_back(TimeAvgMsPerIter(InInnerIters, InStdBench));
	}
	std::sort(oaMs.begin(), oaMs.end());
	std::sort(stMs.begin(), stMs.end());
	const double oaMed = Percentile(oaMs, 0.5);
	const double stMed = Percentile(stMs, 0.5);
	const double ratioMed = oaMed > 0.0 ? stMed / oaMed : 0.0;
	printf("  %-46s\n", InLabel);
	printf("    OaVec  min/med/max: %8.3f / %8.3f / %8.3f ms\n",
		oaMs.front(), oaMed, oaMs.back());
	printf("    std    min/med/max: %8.3f / %8.3f / %8.3f ms\n",
		stMs.front(), stMed, stMs.back());
	printf("    std/Oa (median): %6.2f\n", ratioMed);
	printf("\n");
}

} // namespace

TEST(CoreContainers, OaVecPushBackMatchesStdInt) {
	OaVec<OaI32> oa;
	std::vector<OaI32> st;
	for (OaI32 i = 0; i < 1000; ++i) {
		oa.PushBack(i);
		st.push_back(i);
	}
	EXPECT_TRUE(VecsEqual(oa, st));
	EXPECT_EQ(oa.Size(), 1000u);
}

TEST(CoreContainers, OaVecReserveResizeErase) {
	OaVec<OaI32> oa;
	std::vector<OaI32> st;
	oa.Reserve(500);
	st.reserve(500);
	for (OaI32 i = 0; i < 200; ++i) {
		oa.PushBack(i);
		st.push_back(i);
	}
	oa.Resize(50);
	st.resize(50);
	EXPECT_TRUE(VecsEqual(oa, st));
	oa.Resize(80, -1);
	st.resize(80, -1);
	EXPECT_TRUE(VecsEqual(oa, st));
	oa.Erase(oa.Begin() + 10, oa.Begin() + 20);
	st.erase(st.begin() + 10, st.begin() + 20);
	EXPECT_TRUE(VecsEqual(oa, st));
}

TEST(CoreContainers, OaVecStringNonTrivial) {
	OaVec<std::string> oa;
	std::vector<std::string> st;
	oa.PushBack("hello");
	st.push_back("hello");
	oa.PushBack("realm");
	st.push_back("realm");
	EXPECT_TRUE(VecsEqual(oa, st));
	oa.Insert(oa.Begin() + 1, "middle");
	st.insert(st.begin() + 1, "middle");
	EXPECT_TRUE(VecsEqual(oa, st));
}

TEST(CoreContainers, OaVecAppendTrivialMatchesMemcpy) {
	OaVec<OaI32> oa;
	std::vector<OaI32> st;
	OaI32 chunk[256];
	for (OaI32 i = 0; i < 256; ++i) chunk[i] = i * 3;
	for (int r = 0; r < 4; ++r) {
		oa.Append(chunk, 256);
		st.insert(st.end(), chunk, chunk + 256);
	}
	EXPECT_TRUE(VecsEqual(oa, st));
	EXPECT_EQ(oa.Size(), 1024u);
}

TEST(CoreContainers, OaVecEqualityOperator) {
	OaVec<OaI32> a;
	OaVec<OaI32> b;
	a.PushBack(1);
	a.PushBack(2);
	b.PushBack(1);
	b.PushBack(2);
	EXPECT_TRUE(a == b);
	b.PushBack(3);
	EXPECT_FALSE(a == b);
}

// Run:  ./test_containers --gtest_filter=CoreContainers.BenchSpeedVsStd
TEST(CoreContainers, BenchSpeedVsStd) {
	OaVec<OaI32> oaI;
	std::vector<OaI32> stI;
	OaVec<OaF64> oaD;
	std::vector<OaF64> stD;

	PrintSpeedTableHeader();

	// 1) Repeated growth: clear keeps capacity (steady-state appends, no realloc on inner loop).
	{
		const OaI32 n = 500000;
		const OaI32 iters = 80;
		const double oaMs = TimeAvgMsPerIter(iters, [&]() {
			oaI.Clear();
			for (OaI32 i = 0; i < n; ++i) oaI.PushBack(i);
		});
		const double stMs = TimeAvgMsPerIter(iters, [&]() {
			stI.clear();
			for (OaI32 i = 0; i < n; ++i) stI.push_back(i);
		});
		PrintSpeedRow("push_back i32, N=500k, clear() only", oaMs, stMs);
		EXPECT_EQ(oaI.Size(), static_cast<OaUsize>(n));
		EXPECT_EQ(stI.size(), static_cast<size_t>(n));
	}

	// 2) Cold growth every iteration: release capacity then grow from empty.
	// OaVec uses realloc for trivial T — expect faster than typical std::vector (alloc+copy+free).
	{
		const OaI32 n = 500000;
		const OaI32 iters = 20;
		const double oaMs = TimeAvgMsPerIter(iters, [&]() {
			oaI.Clear();
			oaI.ShrinkToFit();
			for (OaI32 i = 0; i < n; ++i) oaI.PushBack(i);
		});
		const double stMs = TimeAvgMsPerIter(iters, [&]() {
			stI.clear();
			stI.shrink_to_fit();
			for (OaI32 i = 0; i < n; ++i) stI.push_back(i);
		});
		PrintSpeedRow("push_back i32, N=500k, clear+shrink_to_fit", oaMs, stMs);
		const double stdPerOa = oaMs > 0.0 ? stMs / oaMs : 0.0;
		EXPECT_GE(stdPerOa, 1.0) << "OaVec realloc path should be at least parity vs std::vector here";
	}

	// 3) No realloc: single reserve then fill.
	{
		const OaI32 n = 500000;
		const OaI32 iters = 100;
		const double oaMs = TimeAvgMsPerIter(iters, [&]() {
			oaI.Clear();
			oaI.Reserve(static_cast<OaUsize>(n));
			for (OaI32 i = 0; i < n; ++i) oaI.PushBack(i);
		});
		const double stMs = TimeAvgMsPerIter(iters, [&]() {
			stI.clear();
			stI.reserve(static_cast<size_t>(n));
			for (OaI32 i = 0; i < n; ++i) stI.push_back(i);
		});
		PrintSpeedRow("reserve(N)+push_back i32, N=500k", oaMs, stMs);
	}

	// 4) Wider element: double.
	{
		const OaI32 n = 250000;
		const OaI32 iters = 30;
		const double oaMs = TimeAvgMsPerIter(iters, [&]() {
			oaD.Clear();
			for (OaI32 i = 0; i < n; ++i) oaD.PushBack(static_cast<OaF64>(i));
		});
		const double stMs = TimeAvgMsPerIter(iters, [&]() {
			stD.clear();
			for (OaI32 i = 0; i < n; ++i) stD.push_back(static_cast<OaF64>(i));
		});
		PrintSpeedRow("push_back f64, N=250k, clear() only", oaMs, stMs);
	}

	printf("------------------------------------------------------------------------\n");
	printf("(Build with: cmake --preset release && ninja -C build/release)\n\n");
	fflush(stdout);
}

// Full statistical report: pin CPU (e.g. taskset -c 0) for stable numbers.
// Run: ./test_containers --gtest_filter=CoreContainers.BenchSpeedVsStdFullReport
TEST(CoreContainers, BenchSpeedVsStdFullReport) {
	OaVec<OaI32> oaI;
	std::vector<OaI32> stI;
	OaVec<OaU8> oaU;
	std::vector<OaU8> stU;

	printf("\n");
	printf("================================================================================\n");
	printf(" OaVec vs std::vector — FULL REPORT (median over independent trials)\n");
	printf("================================================================================\n");
#if defined(__clang__)
	printf(" Compiler: Clang %s\n", __clang_version__);
#elif defined(__GNUC__)
	printf(" Compiler: GCC %d.%d.%d\n", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#else
	printf(" Compiler: (other)\n");
#endif
	printf(" C++ __cplusplus = %ld\n", static_cast<long>(__cplusplus));
	printf(" std/Oa median > 1.0 => std slower => OaVec faster for that scenario.\n");
	printf("--------------------------------------------------------------------------------\n\n");

	const OaI32 kTrials = 21;
	const OaI32 kInnerPush = 40;

	BenchTrials(
		"push_back i32, N=500k, clear() only (capacity retained)",
		kTrials,
		kInnerPush,
		[&]() {
			oaI.Clear();
			for (OaI32 i = 0; i < 500000; ++i) oaI.PushBack(i);
		},
		[&]() {
			stI.clear();
			for (OaI32 i = 0; i < 500000; ++i) stI.push_back(i);
		});

	BenchTrials(
		"push_back i32, N=500k, clear+shrink_to_fit (cold growth / realloc path)",
		11,
		12,
		[&]() {
			oaI.Clear();
			oaI.ShrinkToFit();
			for (OaI32 i = 0; i < 500000; ++i) oaI.PushBack(i);
		},
		[&]() {
			stI.clear();
			stI.shrink_to_fit();
			for (OaI32 i = 0; i < 500000; ++i) stI.push_back(i);
		});

	BenchTrials(
		"reserve(500k)+push_back i32 (no realloc in inner loop)",
		kTrials,
		80,
		[&]() {
			oaI.Clear();
			oaI.Reserve(500000);
			for (OaI32 i = 0; i < 500000; ++i) oaI.PushBack(i);
		},
		[&]() {
			stI.clear();
			stI.reserve(500000);
			for (OaI32 i = 0; i < 500000; ++i) stI.push_back(i);
		});

	static OaU8 chunk[4096];
	for (OaUsize i = 0; i < 4096; ++i) chunk[i] = static_cast<OaU8>(i & 0xFF);
	constexpr OaUsize kChunk = 4096;
	constexpr OaUsize kBulkTotal = static_cast<OaUsize>(10ull * 1000ull * 1000ull);
	const OaI32 kFullChunks = static_cast<OaI32>(kBulkTotal / kChunk);
	const OaUsize kTail = kBulkTotal - (static_cast<OaUsize>(kFullChunks) * kChunk);
	BenchTrials(
		"append/insert 4Ki chunks -> 10M u8 (OaMemcpy vs insert iterator loop)",
		15,
		8,
		[&]() {
			oaU.Clear();
			oaU.Reserve(kBulkTotal);
			for (OaI32 c = 0; c < kFullChunks; ++c) oaU.Append(chunk, kChunk);
			if (kTail != 0) oaU.Append(chunk, kTail);
		},
		[&]() {
			stU.clear();
			stU.reserve(kBulkTotal);
			for (OaI32 c = 0; c < kFullChunks; ++c) {
				stU.insert(stU.end(), chunk, chunk + kChunk);
			}
			if (kTail != 0) stU.insert(stU.end(), chunk, chunk + kTail);
		});

	EXPECT_EQ(oaI.Size(), 500000u);
	EXPECT_EQ(stI.size(), 500000u);
	EXPECT_EQ(oaU.Size(), kBulkTotal);
	EXPECT_EQ(stU.size(), kBulkTotal);

	printf("--------------------------------------------------------------------------------\n");
	printf(" Note: steady push_back is often allocator + micro-arch noise; bulk append shows\n");
	printf("       OaMemcpy vs libstdc++/libc++ insert loop. Prefer reserve+append for hot IO.\n");
	printf("       Fixed arrays (no growth) remain fastest when max size is known — OaStd.md.\n");
	printf("================================================================================\n\n");
	fflush(stdout);
}
