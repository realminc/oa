#include "oaStdTest.h"

#include <atomic>
#include <memory>
#include <stdexcept>
#include <thread>

namespace {

struct StdControlProbe final : oa::SharedControl {
	int releases = 0;
	int destroys = 0;

	void releaseObject() noexcept override {
		++releases;
	}

	void destroyControl() noexcept override {
		++destroys;
	}
};

struct StdLifetimeProbe {
	std::atomic<int>* destructions = nullptr;

	~StdLifetimeProbe() {
		destructions->fetch_add(1, std::memory_order_relaxed);
	}
};

struct StdThrowingCopyDeleter {
	int* copies = nullptr;
	int* deletes = nullptr;
	int throwOnCopy = 0;

	StdThrowingCopyDeleter(int& inCopies, int& inDeletes, int inThrowOnCopy)
		: copies(&inCopies), deletes(&inDeletes), throwOnCopy(inThrowOnCopy) {}

	StdThrowingCopyDeleter(const StdThrowingCopyDeleter& inOther)
		: copies(inOther.copies)
		, deletes(inOther.deletes)
		, throwOnCopy(inOther.throwOnCopy)
	{
		++*copies;
		if (*copies == throwOnCopy) {
			throw std::runtime_error("injected deleter copy failure");
		}
	}

	StdThrowingCopyDeleter(StdThrowingCopyDeleter&&) noexcept = default;

	void operator()(StdLifetimeProbe* inPtr) noexcept {
		++*deletes;
		delete inPtr;
	}
};

struct StdReentrantSharedValue {
	oa::SharedPtr<StdReentrantSharedValue>* owner = nullptr;
	int* destructions = nullptr;

	~StdReentrantSharedValue() {
		++*destructions;
		if (owner != nullptr) owner->reset();
	}
};

struct StdThrowingSharedConstruction {
	int* destructions = nullptr;

	explicit StdThrowingSharedConstruction(int* inDestructions)
		: destructions(inDestructions) {
		throw std::runtime_error("injected makeShared construction failure");
	}

	~StdThrowingSharedConstruction() { ++*destructions; }
};

struct StdReentrantWeakControlDeleter {
	oa::WeakPtr<int>* owner = nullptr;
	bool* armed = nullptr;
	int* destructorCallbacks = nullptr;

	StdReentrantWeakControlDeleter() = default;
	StdReentrantWeakControlDeleter(
		oa::WeakPtr<int>* inOwner,
		bool* inArmed,
		int* inDestructorCallbacks
	) : owner(inOwner), armed(inArmed), destructorCallbacks(inDestructorCallbacks) {}
	StdReentrantWeakControlDeleter(const StdReentrantWeakControlDeleter&) = delete;
	StdReentrantWeakControlDeleter& operator=(const StdReentrantWeakControlDeleter&) = delete;
	StdReentrantWeakControlDeleter(StdReentrantWeakControlDeleter&& inOther) noexcept
		: owner(inOther.owner)
		, armed(inOther.armed)
		, destructorCallbacks(inOther.destructorCallbacks) {
		inOther.owner = nullptr;
		inOther.armed = nullptr;
		inOther.destructorCallbacks = nullptr;
	}

	void operator()(int* inPtr) const noexcept { delete inPtr; }

	~StdReentrantWeakControlDeleter() {
		if (owner != nullptr && armed != nullptr && *armed) {
			++*destructorCallbacks;
			owner->reset();
		}
	}
};

}  // namespace

TEST(SharedPtr, MakeShared) {
	auto p = oa::makeShared<int>(44);
	ASSERT_TRUE(static_cast<bool>(p));
	EXPECT_EQ(*p, 44);
	EXPECT_EQ(p.useCount(), 1);
}

TEST(SharedPtr, CopySharesUseCount) {
	auto a = oa::makeShared<int>(7);
	oa::SharedPtr<int> b = a;
	EXPECT_EQ(a.useCount(), 2);
	EXPECT_EQ(b.useCount(), 2);
	EXPECT_EQ(*a, 7);
	EXPECT_EQ(*b, 7);
}

TEST(SharedPtr, MoveLeavesEmpty) {
	auto a = oa::makeShared<int>(3);
	oa::SharedPtr<int> b = std::move(a);
	EXPECT_FALSE(static_cast<bool>(a));
	EXPECT_TRUE(static_cast<bool>(b));
	EXPECT_EQ(*b, 3);
	EXPECT_EQ(b.useCount(), 1);
}

TEST(SharedPtr, Reset) {
	auto p = oa::makeShared<int>(99);
	p.reset();
	EXPECT_FALSE(p);
	EXPECT_EQ(p.useCount(), 0);
}

TEST(SharedPtr, ResetPublishesEmptyBeforeReentrantDestruction) {
	int destructions = 0;
	oa::SharedPtr<StdReentrantSharedValue> owner;
	owner = oa::makeShared<StdReentrantSharedValue>(&owner, &destructions);

	owner.reset();

	EXPECT_FALSE(owner);
	EXPECT_EQ(destructions, 1);
}

TEST(SharedPtr, AssignmentPublishesIncomingBeforeRetiringOldOwner) {
	int oldDestructions = 0;
	int incomingDestructions = 0;
	oa::SharedPtr<StdReentrantSharedValue> destination;
	destination = oa::makeShared<StdReentrantSharedValue>(
		&destination, &oldDestructions);
	auto incoming = oa::makeShared<StdReentrantSharedValue>(
		nullptr, &incomingDestructions);

	destination = incoming;

	// The old value's destructor re-entered destination.reset(). It observed and
	// safely released the already-published incoming state.
	EXPECT_FALSE(destination);
	EXPECT_EQ(oldDestructions, 1);
	EXPECT_EQ(incoming.useCount(), 1);
	EXPECT_EQ(incomingDestructions, 0);
}

TEST(SharedPtr, ThrowingMakeSharedConstructionReleasesControlStorage) {
	int destructions = 0;
	EXPECT_THROW(
		(void)oa::makeShared<StdThrowingSharedConstruction>(&destructions),
		std::runtime_error);
	EXPECT_EQ(destructions, 0);
}

TEST(SharedPtr, CustomDeleter) {
	static int calls = 0;
	struct D {
		void operator()(int* inP) {
			++calls;
			delete inP;
		}
	};
	calls = 0;
	{
		oa::SharedPtr<int> p(new int(5), D{});
		EXPECT_EQ(*p, 5);
	}
	EXPECT_EQ(calls, 1);
}

TEST(SharedPtr, ControlBlockKeepsImplicitWeakOwner) {
	StdControlProbe control;
	EXPECT_EQ(control.strong.load(oa::MemoryOrder::Relaxed), 1);
	EXPECT_EQ(control.weak.load(oa::MemoryOrder::Relaxed), 1);

	control.incWeak();
	EXPECT_EQ(control.weak.load(oa::MemoryOrder::Relaxed), 2);

	control.decStrong();
	EXPECT_EQ(control.strong.load(oa::MemoryOrder::Relaxed), 0);
	EXPECT_EQ(control.weak.load(oa::MemoryOrder::Relaxed), 1);
	EXPECT_EQ(control.releases, 1);
	EXPECT_EQ(control.destroys, 0);

	control.decWeak();
	EXPECT_EQ(control.weak.load(oa::MemoryOrder::Relaxed), 0);
	EXPECT_EQ(control.destroys, 1);
}

TEST(SharedPtr, FinalOwnerWithoutWeakObserverReleasesAndDestroysOnce) {
	StdControlProbe control;

	control.decStrong();

	EXPECT_EQ(control.strong.load(oa::MemoryOrder::Relaxed), 0);
	EXPECT_EQ(control.weak.load(oa::MemoryOrder::Relaxed), 1);
	EXPECT_EQ(control.releases, 1);
	EXPECT_EQ(control.destroys, 1);
}

TEST(SharedPtr, ReferenceCountOverflowFailsClosed) {
	StdControlProbe control;
	control.strong.store(oa::SharedControl::MaxReferenceCount,
		oa::MemoryOrder::Relaxed);
	EXPECT_DEATH(control.incStrong(), "SharedPtr strong reference count overflow");

	control.strong.store(1, oa::MemoryOrder::Relaxed);
	control.weak.store(oa::SharedControl::MaxReferenceCount,
		oa::MemoryOrder::Relaxed);
	EXPECT_DEATH(control.incWeak(), "WeakPtr reference count overflow");
}

TEST(SharedPtr, ConcurrentLastStrongAndWeakRelease) {
	constexpr int iterations = 256;
	std::atomic<int> destructions{0};

	for (int iteration = 0; iteration < iterations; ++iteration) {
		auto strong = oa::makeShared<StdLifetimeProbe>(&destructions);
		oa::WeakPtr<StdLifetimeProbe> weak(strong);
		std::atomic<bool> start{false};

		std::thread strongThread(
			[owner = std::move(strong), &start]() mutable {
				while (not start.load(std::memory_order_acquire)) {
					std::this_thread::yield();
				}
				owner.reset();
			});
		std::thread weakThread(
			[owner = std::move(weak), &start]() mutable {
				while (not start.load(std::memory_order_acquire)) {
					std::this_thread::yield();
				}
				owner.reset();
			});

		start.store(true, std::memory_order_release);
		strongThread.join();
		weakThread.join();
	}

	EXPECT_EQ(destructions.load(std::memory_order_relaxed), iterations);
}

static_assert(!oa::isSharedDeleterV<StdThrowingCopyDeleter>,
	"throwing deleters must be rejected at the SharedPtr contract boundary");

TEST(SharedPtr, VoidCustomDeleterReleasesOnce) {
	int deletes = 0;
	auto deleter = [&deletes](void* inPtr) noexcept {
		++deletes;
		delete static_cast<int*>(inPtr);
	};

	{
		oa::SharedPtr<void> owner(new int(41), deleter);
		ASSERT_TRUE(owner);
		EXPECT_EQ(*static_cast<int*>(owner.get()), 41);
	}
	EXPECT_EQ(deletes, 1);
}

TEST(WeakPtr, FromSharedTracksUseCount) {
	auto s = oa::makeShared<int>(11);
	oa::WeakPtr<int> w(s);
	EXPECT_FALSE(w.expired());
	EXPECT_EQ(w.useCount(), 1);
	EXPECT_EQ(s.useCount(), 1);
}

TEST(WeakPtr, LockWhileAlive) {
	auto s = oa::makeShared<int>(7);
	oa::WeakPtr<int> w(s);
	auto l = w.lock();
	ASSERT_TRUE(static_cast<bool>(l));
	EXPECT_EQ(*l, 7);
	EXPECT_EQ(s.useCount(), 2);
}

TEST(WeakPtr, ExpiredAfterLastSharedDestroyed) {
	oa::WeakPtr<int> w;
	{
		auto s = oa::makeShared<int>(3);
		w = oa::WeakPtr<int>(s);
		EXPECT_FALSE(w.expired());
	}
	EXPECT_TRUE(w.expired());
	EXPECT_FALSE(static_cast<bool>(w.lock()));
}

TEST(WeakPtr, CopySharesControlBlock) {
	auto s = oa::makeShared<int>(9);
	oa::WeakPtr<int> a(s);
	oa::WeakPtr<int> b = a;
	EXPECT_FALSE(a.expired());
	EXPECT_FALSE(b.expired());
	s.reset();
	EXPECT_TRUE(a.expired());
	EXPECT_TRUE(b.expired());
}

TEST(WeakPtr, AssignmentPublishesIncomingBeforeOldControlDestruction) {
	oa::WeakPtr<int> destination;
	bool armed = false;
	int destructorCallbacks = 0;
	oa::SharedPtr<int> oldOwner(
		new int(1),
		StdReentrantWeakControlDeleter{
			&destination, &armed, &destructorCallbacks});
	destination = oa::WeakPtr<int>(oldOwner);
	oldOwner.reset();

	auto incomingOwner = oa::makeShared<int>(2);
	oa::WeakPtr<int> incoming(incomingOwner);
	armed = true;
	destination = incoming;

	EXPECT_TRUE(destination.expired());
	EXPECT_EQ(destination.useCount(), 0);
	EXPECT_FALSE(incoming.expired());
	EXPECT_EQ(destructorCallbacks, 1);
}

TEST(StdSharedPtrVsStd, UseCountAfterCopyMatchesPattern) {
	auto oa = oa::makeShared<int>(55);
	std::shared_ptr<int> st = std::make_shared<int>(55);
	oa::SharedPtr<int> oa2 = oa;
	std::shared_ptr<int> st2 = st;
	stdEchoCurrentTest();
	stdExpectGotInt("shared use_count Oa", 2, static_cast<long long>(oa.useCount()));
	stdExpectGotInt("shared use_count std", 2, static_cast<long long>(st.use_count()));
	EXPECT_EQ(oa.useCount(), 2u);
	EXPECT_EQ(st.use_count(), 2u);
	EXPECT_EQ(*oa, *st);
}

TEST(StdSharedPtrVsStd, TimedMakeSharedWallUs) {
	constexpr int kIters = 50'000;
	const auto t0 = oa::highResolutionNow();
	for (int i = 0; i < kIters; ++i) {
		auto p = oa::makeShared<int>(i);
		(void)p.useCount();
		p.reset();
	}
	const auto t1 = oa::highResolutionNow();
	for (int i = 0; i < kIters; ++i) {
		auto p = std::make_shared<int>(i);
		(void)p.use_count();
		p.reset();
	}
	const auto t2 = oa::highResolutionNow();
	stdReportCompareSequentialRuns(
		"oa::makeShared+Reset x50k", t0, t1, "std::make_shared+reset x50k", t2);
}
