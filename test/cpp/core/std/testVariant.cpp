#include "oaStdTest.h"

#include <stdexcept>
#include <type_traits>
#include <variant>

namespace {

struct alignas(64) OverAlignedVariantValue {
	int value{0};
};

struct TrackedVariantValue {
	static inline int alive = 0;
	int value{0};

	TrackedVariantValue() { ++alive; }
	explicit TrackedVariantValue(int inValue) : value(inValue) { ++alive; }
	TrackedVariantValue(const TrackedVariantValue& inOther) : value(inOther.value) {
		++alive;
	}
	TrackedVariantValue(TrackedVariantValue&& inOther) noexcept : value(inOther.value) {
		++alive;
		inOther.value = 0;
	}
	TrackedVariantValue& operator=(TrackedVariantValue&&) noexcept = default;
	~TrackedVariantValue() { --alive; }
};

struct ThrowingMoveVariantValue {
	static inline int alive = 0;
	static inline bool throwOnMove = false;

	explicit ThrowingMoveVariantValue(int inValue = 0) : value(inValue) { ++alive; }
	ThrowingMoveVariantValue(const ThrowingMoveVariantValue&) = delete;
	ThrowingMoveVariantValue(ThrowingMoveVariantValue&& inOther) : value(inOther.value) {
		if (throwOnMove) {
			throw std::runtime_error("injected Variant move failure");
		}
		++alive;
		inOther.value = -1;
	}
	ThrowingMoveVariantValue& operator=(ThrowingMoveVariantValue&&) = delete;
	~ThrowingMoveVariantValue() { --alive; }

	int value = 0;
};

struct ReentrantVariantProbe;
using ReentrantVariant = oa::Variant<int, ReentrantVariantProbe>;

struct ReentrantVariantProbe {
	ReentrantVariant* owner = nullptr;
	bool* observedEmpty = nullptr;
	int* destructions = nullptr;

	ReentrantVariantProbe(
		ReentrantVariant* inOwner,
		bool* inObservedEmpty,
		int* inDestructions
	) : owner(inOwner)
	  , observedEmpty(inObservedEmpty)
	  , destructions(inDestructions) {}

	ReentrantVariantProbe(const ReentrantVariantProbe&) = default;
	ReentrantVariantProbe(ReentrantVariantProbe&& inOther) noexcept = default;
	ReentrantVariantProbe& operator=(ReentrantVariantProbe&&) noexcept = default;

	~ReentrantVariantProbe() {
		++*destructions;
		*observedEmpty = owner->empty();
	}
};

} // namespace

TEST(Variant, IndexAndGet) {
	oa::Variant<int, float> v{3.14f};
	EXPECT_EQ(v.index(), 1U);
	EXPECT_FLOAT_EQ(v.get<float>(), 3.14f);

	oa::Variant<int, float> oa{1.0f};
	std::variant<int, float> st{1.0f};
	constexpr int kLoops = 300'000;
	const auto t0 = oa::highResolutionNow();
	volatile float sinkOa = 0;
	for (int i = 0; i < kLoops; ++i) {
		(void)i;
		sinkOa += oa.get<float>();
	}
	const auto t1 = oa::highResolutionNow();
	volatile float sinkSt = 0;
	for (int i = 0; i < kLoops; ++i) {
		(void)i;
		sinkSt += std::get<float>(st);
	}
	const auto t2 = oa::highResolutionNow();
	stdReportCompareSequentialRuns(
		"oa::Variant::get<float> x300k", t0, t1, "std::get<float> x300k", t2);
	stdExpectGotFloat("variant float sum (scaled)", sinkSt / 1000.f, sinkOa / 1000.f);
	EXPECT_FLOAT_EQ(sinkOa, sinkSt);
}

TEST(Variant, WrongAlternativeRejectsContract) {
	oa::Variant<int, float> value{3};
	EXPECT_DEATH((void)value.get<float>(), "OA contract failed: index_ == Index");
}

TEST(Variant, HoldsAlternativeAndVisit) {
	oa::Variant<int, const char*> v{7};
	ASSERT_TRUE(v.holdsAlternative<int>());
	int sum = 0;
	v.visit([&sum](auto&& x) {
		using U = std::decay_t<decltype(x)>;
		if constexpr (std::is_same_v<U, int>) {
			sum += x;
		}
	});
	EXPECT_EQ(sum, 7);

	oa::Variant<int, float> oa{42};
	std::variant<int, float> st{42};
	constexpr int kVisits = 200'000;
	const auto t0 = oa::highResolutionNow();
	volatile int sinkOa = 0;
	for (int i = 0; i < kVisits; ++i) {
		oa.visit([&sinkOa](auto&& x) {
			using U = std::decay_t<decltype(x)>;
			if constexpr (std::is_same_v<U, int>) {
				sinkOa += x;
			}
		});
	}
	const auto t1 = oa::highResolutionNow();
	volatile int sinkSt = 0;
	for (int i = 0; i < kVisits; ++i) {
		sinkSt += std::visit([](auto&& x) { return static_cast<int>(x); }, st);
	}
	const auto t2 = oa::highResolutionNow();
	stdReportCompareSequentialRuns(
		"oa::Variant::visit x200k", t0, t1, "std::visit x200k", t2);
	const long long expectSum = static_cast<long long>(42) * kVisits;
	stdExpectGotInt("visit accumulated sum", expectSum, static_cast<long long>(sinkOa));
	EXPECT_EQ(sinkOa, sinkSt);
	EXPECT_EQ(sinkOa, expectSum);
}

TEST(Variant, EmplaceAndSwap) {
	oa::Variant<int, float> a{1};
	oa::Variant<int, float> b{2.5f};
	EXPECT_TRUE(a.holdsAlternative<int>());
	a.emplace<float>(1.5f);
	EXPECT_TRUE(a.holdsAlternative<float>());
	EXPECT_TRUE(b.holdsAlternative<float>());
	a.swap(b);
	EXPECT_FLOAT_EQ(a.get<float>(), 2.5f);
	EXPECT_FLOAT_EQ(b.get<float>(), 1.5f);

	constexpr int kSwaps = 500'000;
	volatile oa::Usize sinkOa = 0;
	const auto t0 = oa::highResolutionNow();
	for (int i = 0; i < kSwaps; ++i) {
		oa::Variant<int, float> x{1};
		oa::Variant<int, float> y{2.0f};
		x.swap(y);
		sinkOa += x.index();
	}
	const auto t1 = oa::highResolutionNow();
	volatile std::size_t sinkStd = 0;
	for (int i = 0; i < kSwaps; ++i) {
		std::variant<int, float> x{1};
		std::variant<int, float> y{2.0f};
		std::swap(x, y);
		sinkStd += x.index();
	}
	const auto t2 = oa::highResolutionNow();
	stdReportCompareSequentialRuns(
		"oa::Variant::swap x500k", t0, t1, "std::swap(variant) x500k", t2);
	EXPECT_EQ(sinkOa, sinkStd);
}

TEST(Variant, CopyMoveAndEmptyState) {
	oa::Variant<int, oa::String> original{oa::String("realm")};
	oa::Variant<int, oa::String> copy{original};
	oa::Variant<int, oa::String> moved{oa::move(original)};

	EXPECT_EQ(copy.get<oa::String>(), "realm");
	EXPECT_EQ(moved.get<oa::String>(), "realm");
	EXPECT_TRUE(original.empty());
	EXPECT_DEATH(original.visit([](auto&&) {}),
		"OA contract failed: index_ != Npos");
}

TEST(Variant, EmplacePublishesEmptyBeforeAlternativeDestruction) {
	ReentrantVariant value{0};
	bool observedEmpty = false;
	int destructions = 0;
	value.emplace<ReentrantVariantProbe>(&value, &observedEmpty, &destructions);

	value.emplace<int>(23);

	EXPECT_TRUE(value.holdsAlternative<int>());
	EXPECT_EQ(value.get<int>(), 23);
	EXPECT_TRUE(observedEmpty);
	EXPECT_EQ(destructions, 1);
}

TEST(Variant, PreservesOverAlignment) {
	using Value = oa::Variant<int, OverAlignedVariantValue>;
	static_assert(alignof(Value) >= alignof(OverAlignedVariantValue));

	Value value{OverAlignedVariantValue{.value = 19}};
	EXPECT_EQ(value.get<OverAlignedVariantValue>().value, 19);
}

TEST(Variant, OwnsEachAlternativeLifetimeOnce) {
	TrackedVariantValue::alive = 0;
	{
		oa::Variant<TrackedVariantValue, int> first{TrackedVariantValue{7}};
		EXPECT_EQ(TrackedVariantValue::alive, 1);
		{
			auto copy = first;
			auto moved = oa::move(first);
			EXPECT_EQ(TrackedVariantValue::alive, 2);
			EXPECT_TRUE(first.empty());
			EXPECT_EQ(copy.get<TrackedVariantValue>().value, 7);
			EXPECT_EQ(moved.get<TrackedVariantValue>().value, 7);
			moved.emplace<int>(11);
			EXPECT_EQ(TrackedVariantValue::alive, 1);
		}
		EXPECT_EQ(TrackedVariantValue::alive, 0);
	}
	EXPECT_EQ(TrackedVariantValue::alive, 0);
}

TEST(Variant, ThrowingMoveAssignmentLeavesDestinationEmpty) {
	ThrowingMoveVariantValue::alive = 0;
	ThrowingMoveVariantValue::throwOnMove = false;
	{
		oa::Variant<int, ThrowingMoveVariantValue> source{0};
		source.emplace<ThrowingMoveVariantValue>(31);
		oa::Variant<int, ThrowingMoveVariantValue> destination{7};

		ThrowingMoveVariantValue::throwOnMove = true;
		EXPECT_THROW(destination = oa::move(source), std::runtime_error);
		EXPECT_TRUE(destination.empty());
		EXPECT_EQ(source.get<ThrowingMoveVariantValue>().value, 31);
		EXPECT_EQ(ThrowingMoveVariantValue::alive, 1);
		ThrowingMoveVariantValue::throwOnMove = false;
	}
	EXPECT_EQ(ThrowingMoveVariantValue::alive, 0);
}

TEST(StdVariantVsStd, VisitSumMatchesStdVisit) {
	oa::Variant<int, float> oa{2.5f};
	std::variant<int, float> st{2.5f};
	int sumOa = 0;
	oa.visit([&sumOa](auto&& x) {
		using U = std::decay_t<decltype(x)>;
		if constexpr (std::is_same_v<U, int>) {
			sumOa += x;
		} else {
			sumOa += static_cast<int>(x);
		}
	});
	int sumSt = std::visit([](auto&& x) { return static_cast<int>(x); }, st);
	stdEchoCurrentTest();
	stdExpectGotInt("variant visit sum", static_cast<long long>(sumSt), static_cast<long long>(sumOa));
	EXPECT_EQ(sumOa, sumSt);
}
