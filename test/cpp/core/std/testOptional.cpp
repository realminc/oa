#include "oaStdTest.h"

#include <optional>

namespace {

struct LifetimeProbe {
	int* destructions = nullptr;
	int value = 0;

	LifetimeProbe(int* inDestructions, int inValue)
		: destructions(inDestructions), value(inValue) {}

	~LifetimeProbe() { ++*destructions; }
};

struct ReentrantOptionalProbe;
using ReentrantOptional = oa::Optional<ReentrantOptionalProbe>;

struct ReentrantOptionalProbe {
	ReentrantOptional* owner = nullptr;
	bool* observedEmpty = nullptr;
	int* destructions = nullptr;

	ReentrantOptionalProbe(
		ReentrantOptional* inOwner,
		bool* inObservedEmpty,
		int* inDestructions
	) : owner(inOwner)
	  , observedEmpty(inObservedEmpty)
	  , destructions(inDestructions) {}

	~ReentrantOptionalProbe() {
		++*destructions;
		*observedEmpty = !owner->hasValue();
		owner->reset();
	}
};

} // namespace

TEST(Lifetime, ConstructAndDestroyCallerOwnedStorage) {
	alignas(LifetimeProbe) unsigned char storage[sizeof(LifetimeProbe)]{};
	int destructions = 0;
	auto* probe = oa::constructAt(
		reinterpret_cast<LifetimeProbe*>(storage), &destructions, 42);
	EXPECT_EQ(probe->value, 42);
	EXPECT_EQ(oa::launder(reinterpret_cast<LifetimeProbe*>(storage)), probe);
	oa::destroyAt(probe);
	EXPECT_EQ(destructions, 1);
}

TEST(Optional, ValueOr) {
	oa::Optional<int> empty;
	oa::Optional<int> filled(7);
	EXPECT_FALSE(empty.hasValue());
	EXPECT_TRUE(filled.hasValue());
	EXPECT_EQ(filled.value(), 7);
	EXPECT_EQ(empty.valueOr(99), 99);

	constexpr int kLoops = 200'000;
	oa::Optional<int> oa;
	std::optional<int> st;
	const auto t0 = oa::highResolutionNow();
	volatile long long sinkOa = 0;
	for (int i = 0; i < kLoops; ++i) {
		sinkOa += oa.valueOr(i);
	}
	const auto t1 = oa::highResolutionNow();
	volatile long long sinkSt = 0;
	for (int i = 0; i < kLoops; ++i) {
		sinkSt += st.value_or(i);
	}
	const auto t2 = oa::highResolutionNow();
	stdReportCompareSequentialRuns(
		"oa::Optional::valueOr (empty) x200k", t0, t1,
		"std::optional::value_or (empty) x200k", t2);
	stdExpectGotInt("empty ValueOr sum tail", static_cast<long long>(sinkSt % 1000),
		static_cast<long long>(sinkOa % 1000));
	EXPECT_EQ(sinkOa, sinkSt);
}

TEST(Optional, ValueRejectsEmpty) {
	oa::Optional<int> empty;
	EXPECT_DEATH(static_cast<void>(empty.value()), "OA contract failed: engaged_");
}

TEST(Optional, EmplaceAndReset) {
	oa::Optional<int> o;
	EXPECT_EQ(o.get(), nullptr);
	o.emplace(5);
	ASSERT_TRUE(o.hasValue());
	ASSERT_NE(o.get(), nullptr);
	EXPECT_EQ(*o.get(), 5);
	EXPECT_EQ(o.value(), 5);
	const oa::Optional<int>& readOnly = o;
	ASSERT_NE(readOnly.get(), nullptr);
	EXPECT_EQ(*readOnly.get(), 5);
	o.reset();
	EXPECT_FALSE(o.hasValue());
	EXPECT_EQ(o.get(), nullptr);
}

TEST(Optional, ResetPublishesEmptyBeforeValueDestruction) {
	ReentrantOptional value;
	bool observedEmpty = false;
	int destructions = 0;
	value.emplace(&value, &observedEmpty, &destructions);

	value.reset();

	EXPECT_FALSE(value.hasValue());
	EXPECT_TRUE(observedEmpty);
	EXPECT_EQ(destructions, 1);
}

TEST(Optional, CopyAndMovePreserveOwnership) {
	oa::Optional<int> original(11);
	oa::Optional<int> copy(original);
	ASSERT_TRUE(copy.hasValue());
	EXPECT_EQ(copy.value(), 11);

	oa::Optional<int> moved(oa::move(original));
	ASSERT_TRUE(moved.hasValue());
	EXPECT_EQ(moved.value(), 11);
	EXPECT_FALSE(original.hasValue());
}

TEST(StdOptionalVsStd, ParallelEmplaceResetSequence) {
	oa::Optional<int> oa;
	std::optional<int> st;
	EXPECT_EQ(oa.hasValue(), st.has_value());
	oa.emplace(42);
	st.emplace(42);
	EXPECT_EQ(oa.value(), *st);
	oa.reset();
	st.reset();
	EXPECT_FALSE(oa.hasValue());
	EXPECT_FALSE(st.has_value());

	const auto t0 = oa::highResolutionNow();
	for (int i = 0; i < 150'000; ++i) {
		oa::Optional<int> x;
		x.emplace(i);
		x.reset();
	}
	const auto t1 = oa::highResolutionNow();
	for (int i = 0; i < 150'000; ++i) {
		std::optional<int> x;
		x.emplace(i);
		x.reset();
	}
	const auto t2 = oa::highResolutionNow();
	stdEchoCurrentTest();
	stdExpectGotInt("optional has_value after reset", static_cast<long long>(st.has_value()),
		static_cast<long long>(oa.hasValue()));
	stdReportCompareMsLines(
		"oa::Optional emplace+reset x150k", stdWallMs(t1 - t0),
		"std::optional emplace+reset x150k", stdWallMs(t2 - t1));
}

TEST(StdOptionalVsStd, ValueOrMatchesStd) {
	oa::Optional<int> oa;
	std::optional<int> st;
	EXPECT_EQ(oa.valueOr(-1), st.value_or(-1));
	oa.emplace(7);
	st = 7;
	EXPECT_EQ(oa.valueOr(-1), st.value_or(-1));
	stdEchoCurrentTest();
	stdExpectGotInt("ValueOr with value", static_cast<long long>(st.value_or(-1)),
		static_cast<long long>(oa.valueOr(-1)));
}

TEST(StdOptionalVsStd, TimedValueOrWallUs) {
	constexpr int kLoops = 500'000;
	oa::Optional<int> oa;
	oa.emplace(123);
	std::optional<int> st(123);
	volatile long long sinkOa = 0;
	const auto t0 = oa::highResolutionNow();
	for (int i = 0; i < kLoops; ++i) {
		sinkOa += oa.valueOr(i);
	}
	const auto t1 = oa::highResolutionNow();
	volatile long long sinkSt = 0;
	for (int i = 0; i < kLoops; ++i) {
		sinkSt += st.value_or(i);
	}
	const auto t2 = oa::highResolutionNow();
	stdReportCompareSequentialRuns(
		"oa::Optional::valueOr (filled) x500k", t0, t1,
		"std::optional::value_or (filled) x500k", t2);
	stdExpectGotInt("optional ValueOr sum tail", static_cast<long long>(sinkSt % 1000),
		static_cast<long long>(sinkOa % 1000));
	EXPECT_EQ(sinkOa, sinkSt);
}
