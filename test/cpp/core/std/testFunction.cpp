#include "oaStdTest.h"

#include <functional>

namespace {

struct LargeCallable {
	int padding[16]{};
	int bias{0};

	int operator()(int inValue) { return inValue + bias; }
};

struct ReentrantFnCallable {
	oa::Fn<void()>* owner = nullptr;
	bool* armed = nullptr;
	int* destructions = nullptr;

	ReentrantFnCallable() = default;
	ReentrantFnCallable(
		oa::Fn<void()>* inOwner,
		bool* inArmed,
		int* inDestructions
	) : owner(inOwner), armed(inArmed), destructions(inDestructions) {}

	ReentrantFnCallable(const ReentrantFnCallable&) = default;
	ReentrantFnCallable(ReentrantFnCallable&& inOther) noexcept
		: owner(inOther.owner), armed(inOther.armed), destructions(inOther.destructions) {
		inOther.owner = nullptr;
		inOther.armed = nullptr;
		inOther.destructions = nullptr;
	}

	void operator()() const noexcept {}

	~ReentrantFnCallable() {
		if (destructions != nullptr) ++*destructions;
		if (owner != nullptr && armed != nullptr && *armed) {
			*owner = oa::Fn<void()>{};
		}
	}
};

} // namespace

TEST(Fn, Call) {
	oa::Fn<int(int)> fn = [](int x) { return x * 2; };
	EXPECT_EQ(fn(21), 42);
}

TEST(Fn, EmptyAndSwap) {
	oa::Fn<int()> a;
	oa::Fn<int()> b = [] { return 1; };
	EXPECT_TRUE(a.empty());
	EXPECT_FALSE(b.empty());
	EXPECT_DEATH((void)a(), "OA contract failed: vtable_ != nullptr");
	a.swap(b);
	EXPECT_FALSE(a.empty());
	EXPECT_TRUE(b.empty());
	EXPECT_DEATH((void)b(), "OA contract failed: vtable_ != nullptr");
}

TEST(Fn, NullptrConstructionIsEmpty) {
	oa::Fn<int()> fn = nullptr;
	EXPECT_TRUE(fn.empty());
}

TEST(Fn, TypedNullFunctionPointerConstructionIsEmpty) {
	int (*function)(int) = nullptr;
	oa::Fn<int(int)> fn(function);
	EXPECT_TRUE(fn.empty());
}

TEST(Fn, ClearPublishesEmptyBeforeReentrantCallableDestruction) {
	oa::Fn<void()> fn;
	bool armed = false;
	int destructions = 0;
	fn = oa::Fn<void()>(ReentrantFnCallable{&fn, &armed, &destructions});
	armed = true;

	fn = oa::Fn<void()>{};

	EXPECT_TRUE(fn.empty());
	EXPECT_EQ(destructions, 1);
}

TEST(Fn, HeapFallbackCopiesAndMoves) {
	oa::Fn<int(int)> original = LargeCallable{.bias = 7};
	oa::Fn<int(int)> copy = original;
	oa::Fn<int(int)> moved = oa::move(original);

	EXPECT_EQ(copy(5), 12);
	EXPECT_EQ(moved(9), 16);
	EXPECT_TRUE(original.empty());
}

TEST(StdFnVsStd, SameLambdaResultAsStdFunction) {
	const auto lam = [](int x) { return x * x + 1; };
	oa::Fn<int(int)> oa(lam);
	std::function<int(int)> st(lam);
	stdEchoCurrentTest();
	stdExpectGotInt("function(7)", static_cast<long long>(st(7)), static_cast<long long>(oa(7)));
	EXPECT_EQ(oa(7), st(7));
}

TEST(StdFnVsStd, TimedInvokeWallUs) {
	constexpr int kCalls = 300'000;
	oa::Fn<int(int)> oa = [](int x) { return x ^ (x >> 2); };
	std::function<int(int)> st = [](int x) { return x ^ (x >> 2); };
	volatile long long sinkOa = 0;
	const auto t0 = oa::highResolutionNow();
	for (int i = 0; i < kCalls; ++i) {
		sinkOa += oa(i);
	}
	const auto t1 = oa::highResolutionNow();
	volatile long long sinkSt = 0;
	for (int i = 0; i < kCalls; ++i) {
		sinkSt += st(i);
	}
	const auto t2 = oa::highResolutionNow();
	stdReportCompareSequentialRuns(
		"oa::Fn::operator() x300k", t0, t1, "std::function::operator() x300k", t2);
	stdExpectGotInt("function invoke sum", static_cast<long long>(sinkSt), static_cast<long long>(sinkOa));
	EXPECT_EQ(sinkOa, sinkSt);
}
