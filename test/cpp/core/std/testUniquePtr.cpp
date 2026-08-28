#include "oaStdTest.h"

#include <memory>
#include <type_traits>

namespace {

struct StdCountingUniqueDeleter {
	int* deletes = nullptr;

	void operator()(int* inPtr) const noexcept {
		++*deletes;
		delete inPtr;
	}
};

struct StdStatefulUniqueDeleter {
	int identity = 0;
	int* observedIdentity = nullptr;

	void operator()(int* inPtr) const noexcept {
		*observedIdentity = identity;
		delete inPtr;
	}
};

struct StdReentrantUniqueDeleter {
	void* owner = nullptr;
	int* deletes = nullptr;

	void operator()(int* inPtr) const noexcept;
};

using StdReentrantUniquePtr = oa::UniquePtr<int, StdReentrantUniqueDeleter>;

void StdReentrantUniqueDeleter::operator()(int* inPtr) const noexcept {
	++*deletes;
	static_cast<StdReentrantUniquePtr*>(owner)->reset();
	delete inPtr;
}

struct StdThrowingMoveUniqueDeleter {
	StdThrowingMoveUniqueDeleter() = default;
	StdThrowingMoveUniqueDeleter(const StdThrowingMoveUniqueDeleter&) = delete;
	StdThrowingMoveUniqueDeleter(StdThrowingMoveUniqueDeleter&&) noexcept(false) {}
	StdThrowingMoveUniqueDeleter& operator=(StdThrowingMoveUniqueDeleter&&) noexcept(false) {
		return *this;
	}
	void operator()(int* inPtr) const noexcept { delete inPtr; }
};

struct StdIncompleteUniqueValue;
struct StdIncompleteUniqueDeleter {
	int* deletes = nullptr;
	void operator()(StdIncompleteUniqueValue* inPtr) const noexcept;
};
using StdIncompleteUniquePtr =
	oa::UniquePtr<StdIncompleteUniqueValue, StdIncompleteUniqueDeleter>;

void consumeIncompleteUniquePtr(StdIncompleteUniquePtr inOwner) {
	(void)inOwner;
}

struct StdIncompleteUniqueValue {
	int value = 0;
};

void StdIncompleteUniqueDeleter::operator()(StdIncompleteUniqueValue* inPtr) const noexcept {
	++*deletes;
	delete inPtr;
}

static_assert(!std::is_move_constructible_v<
	oa::UniquePtr<int, StdThrowingMoveUniqueDeleter>>);
static_assert(!std::is_constructible_v<
	oa::UniquePtr<int, StdThrowingMoveUniqueDeleter>,
	int*,
	StdThrowingMoveUniqueDeleter&&>);

} // namespace

TEST(UniquePtr, MakeUnique) {
	auto p = oa::makeUnique<int>(33);
	ASSERT_TRUE(static_cast<bool>(p));
	EXPECT_EQ(*p, 33);
}

TEST(UniquePtr, MoveAndReset) {
	auto a = oa::makeUnique<int>(7);
	oa::UniquePtr<int> b = oa::move(a);
	EXPECT_FALSE(static_cast<bool>(a));
	ASSERT_TRUE(static_cast<bool>(b));
	EXPECT_EQ(*b, 7);
	b.reset();
	EXPECT_FALSE(static_cast<bool>(b));
}

TEST(UniquePtr, ReleaseTransfersOwnership) {
	auto p = oa::makeUnique<int>(99);
	int* released = p.release();
	ASSERT_NE(released, nullptr);
	EXPECT_FALSE(static_cast<bool>(p));
	EXPECT_EQ(*released, 99);
	delete released;
}

TEST(UniquePtr, SelfResetDoesNotDeleteOwnedObject) {
	int deletes = 0;
	oa::UniquePtr<int, StdCountingUniqueDeleter> owner(
		new int(73), StdCountingUniqueDeleter{&deletes});
	int* same = owner.get();

	owner.reset(same);

	ASSERT_TRUE(owner);
	EXPECT_EQ(owner.get(), same);
	EXPECT_EQ(*owner, 73);
	EXPECT_EQ(deletes, 0);
	owner.reset();
	EXPECT_EQ(deletes, 1);
}

TEST(UniquePtr, ResetPublishesNullBeforeReentrantDeleter) {
	int deletes = 0;
	StdReentrantUniquePtr owner;
	owner = StdReentrantUniquePtr(
		new int(19), StdReentrantUniqueDeleter{&owner, &deletes});

	owner.reset();

	EXPECT_FALSE(owner);
	EXPECT_EQ(deletes, 1);
}

TEST(UniquePtr, CustomDeleterSupportsIncompleteOwnershipBoundary) {
	int deletes = 0;
	consumeIncompleteUniquePtr(StdIncompleteUniquePtr(
		new StdIncompleteUniqueValue{41}, StdIncompleteUniqueDeleter{&deletes}));
	EXPECT_EQ(deletes, 1);
}

TEST(UniquePtr, MoveTransfersMatchingDeleterState) {
	int sourceDeletionIdentity = 0;
	oa::UniquePtr<int, StdStatefulUniqueDeleter> source(
		new int(7), StdStatefulUniqueDeleter{41, &sourceDeletionIdentity});
	oa::UniquePtr<int, StdStatefulUniqueDeleter> moved(oa::move(source));
	EXPECT_FALSE(source);
	moved.reset();
	EXPECT_EQ(sourceDeletionIdentity, 41);

	int oldDestinationIdentity = 0;
	int incomingIdentity = 0;
	oa::UniquePtr<int, StdStatefulUniqueDeleter> destination(
		new int(1), StdStatefulUniqueDeleter{11, &oldDestinationIdentity});
	oa::UniquePtr<int, StdStatefulUniqueDeleter> incoming(
		new int(2), StdStatefulUniqueDeleter{29, &incomingIdentity});
	destination = oa::move(incoming);
	EXPECT_EQ(oldDestinationIdentity, 11);
	EXPECT_FALSE(incoming);
	destination.reset();
	EXPECT_EQ(incomingIdentity, 29);
}

TEST(StdUniquePtrVsStd, DerefMatchesParallelStdUniquePtr) {
	auto oa = oa::makeUnique<int>(1001);
	std::unique_ptr<int> st = std::make_unique<int>(1001);
	stdEchoCurrentTest();
	stdExpectGotInt("unique_ptr *", static_cast<long long>(*st), static_cast<long long>(*oa));
	EXPECT_EQ(*oa, *st);
}

TEST(StdUniquePtrVsStd, TimedMakeResetWallUs) {
	constexpr int kIters = 80'000;
	const auto t0 = oa::highResolutionNow();
	for (int i = 0; i < kIters; ++i) {
		auto p = oa::makeUnique<int>(i);
		(void)*p;
		p.reset();
	}
	const auto t1 = oa::highResolutionNow();
	for (int i = 0; i < kIters; ++i) {
		auto p = std::make_unique<int>(i);
		(void)*p;
		p.reset();
	}
	const auto t2 = oa::highResolutionNow();
	stdReportCompareSequentialRuns(
		"oa::makeUnique+Reset x80k", t0, t1, "std::make_unique+reset x80k", t2);
}
