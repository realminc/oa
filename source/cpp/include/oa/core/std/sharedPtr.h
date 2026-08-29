#pragma once

// Native SharedPtr / WeakPtr — atomic strong + weak counts and a type-erased
// control block allocated through OA's host-allocation boundary.
//
// `makeShared<T>(...)` allocates control block + `T` in one slab (inline storage).
// Thread-safe refcounting; `lock()` from weak promotes only if object still alive.

#include <oa/core/std/allocator.h>
#include <oa/core/std/atomic.h>
#include <oa/core/std/assert.h>
#include <oa/core/std/lifetime.h>
#include <oa/core/std/typeTraits.h>
#include <oa/core/std/uniquePtr.h>
#include <oa/core/std/utility.h>

namespace oa {

template<typename T>
class WeakPtr;

struct SharedControl {
	static constexpr oa::U32 MaxReferenceCount = 0x7fffffffU;

	// 32-bit counters keep the common inline control block in the smaller
	// allocator size class while still admitting more owners than OA can use.
	oa::Atomic<oa::U32> strong{1};
	// One implicit weak reference keeps the control block alive while any
	// strong owner exists. Explicit WeakPtr instances add to this count.
	oa::Atomic<oa::U32> weak{1};

	void incStrong() noexcept {
		const oa::U32 previous = strong.fetchAdd(1, oa::MemoryOrder::Relaxed);
		OA_REQUIRE_MSG(previous != 0, "SharedPtr increment after final release");
		OA_REQUIRE_MSG(previous < MaxReferenceCount,
			"SharedPtr strong reference count overflow");
	}

	bool incStrongIfNonzero() noexcept {
		oa::U32 n = strong.load(oa::MemoryOrder::Relaxed);
		for (;;) {
			if (n == 0) {
				return false;
			}
			OA_REQUIRE_MSG(n < MaxReferenceCount,
				"SharedPtr strong reference count overflow");
			if (strong.compareExchangeWeak(
				n, n + 1, oa::MemoryOrder::AcquireRelease)) {
				return true;
			}
		}
	}

	void incWeak() noexcept {
		const oa::U32 previous = weak.fetchAdd(1, oa::MemoryOrder::Relaxed);
		OA_REQUIRE_MSG(previous != 0, "WeakPtr increment after final release");
		OA_REQUIRE_MSG(previous < MaxReferenceCount,
			"WeakPtr reference count overflow");
	}

	void decStrong() noexcept {
		// A sole owner with no weak observer cannot race with another legal
		// reference-count operation: making another owner requires an existing
		// owner or weak observer. Acquire the prior release that left this owner
		// last, then retire without a locked read-modify-write.
		if (strong.load(oa::MemoryOrder::Acquire) == 1
			&& weak.load(oa::MemoryOrder::Acquire) == 1) {
			strong.store(0, oa::MemoryOrder::Relaxed);
			releaseObjectAndDestroyControl();
			return;
		}
		const oa::U32 previous = strong.fetchSub(1, oa::MemoryOrder::AcquireRelease);
		OA_REQUIRE_MSG(previous != 0, "SharedPtr reference count underflow");
		if (previous != 1) {
			return;
		}
		// The implicit weak owner is the only remaining weak reference in the
		// common make/release path. Once the final strong owner is gone, no new
		// WeakPtr can be created unless one already exists. Fuse object and
		// control-block destruction in that path so release needs one dynamic
		// dispatch and no second atomic decrement.
		if (weak.load(oa::MemoryOrder::Acquire) == 1) {
			releaseObjectAndDestroyControl();
			return;
		}
		releaseObject();
		decWeak();
	}

	void decWeak() noexcept {
		const oa::U32 previous = weak.fetchSub(1, oa::MemoryOrder::AcquireRelease);
		OA_REQUIRE_MSG(previous != 0, "WeakPtr reference count underflow");
		if (previous != 1) {
			return;
		}
		destroyControl();
	}

	virtual void releaseObject() noexcept = 0;
	virtual void destroyControl() noexcept = 0;
	virtual void releaseObjectAndDestroyControl() noexcept {
		releaseObject();
		destroyControl();
	}

protected:
	virtual ~SharedControl() = default;
};

template<typename T, typename Deleter>
struct SharedControlDeleter final : SharedControl {
	T* ptr{};
	Deleter deleter;

	SharedControlDeleter(T* inP, const Deleter& inD) : ptr(inP), deleter(inD) {}

	SharedControlDeleter(T* inP, Deleter&& inD) noexcept(
		oa::IsNothrowMoveConstructibleV<Deleter>)
		: ptr(inP), deleter(oa::move(inD)) {}

	void releaseObject() noexcept override {
		if (ptr) {
			// Publish the retired state before invoking user code. A deleter may
			// release another owner which reaches this control block recursively.
			T* retired = ptr;
			ptr = nullptr;
			deleter(retired);
		}
	}

	void destroyControl() noexcept override {
		auto* self = this;
		oa::destroyAt(self);
		oa::freeBytes(self, alignof(SharedControlDeleter));
	}

	void releaseObjectAndDestroyControl() noexcept override {
		if (ptr) {
			T* retired = ptr;
			ptr = nullptr;
			deleter(retired);
		}
		destroyControl();
	}
};

template<typename Control, typename... Args>
[[nodiscard]] Control* allocateSharedControl_(Args&&... inArgs) {
	class StorageGuard final {
	public:
		explicit StorageGuard(void* inStorage) noexcept : storage_(inStorage) {}
		~StorageGuard() {
			if (storage_) {
				oa::freeBytes(storage_, alignof(Control));
			}
		}

		StorageGuard(const StorageGuard&) = delete;
		StorageGuard& operator=(const StorageGuard&) = delete;

		void release() noexcept { storage_ = nullptr; }

	private:
		void* storage_;
	};

	void* storage = oa::allocBytes(sizeof(Control), alignof(Control));
	StorageGuard guard(storage);
	Control* control = oa::constructAt(
		static_cast<Control*>(storage), oa::forward<Args>(inArgs)...);
	guard.release();
	return control;
}

template<typename T, typename Deleter>
[[nodiscard]] SharedControl* createSharedControl_(T* inP, Deleter& inD) {
	using StoredDeleter = oa::DecayT<Deleter>;
	if constexpr (oa::IsCopyConstructibleV<StoredDeleter>) {
		static_assert(oa::IsNothrowCopyConstructibleV<StoredDeleter>,
			"A SharedPtr deleter must be nothrow copy constructible");
		return allocateSharedControl_<SharedControlDeleter<T, StoredDeleter>>(
			inP, static_cast<const StoredDeleter&>(inD));
	} else {
		static_assert(oa::IsNothrowMoveConstructibleV<StoredDeleter>,
			"A move-only SharedPtr deleter must be nothrow move constructible");
		return allocateSharedControl_<SharedControlDeleter<T, StoredDeleter>>(
			inP, oa::move(inD));
	}
}

template<typename Deleter>
inline constexpr bool IsSharedDeleterV =
	(oa::IsCopyConstructibleV<oa::DecayT<Deleter>>
		&& oa::IsNothrowCopyConstructibleV<oa::DecayT<Deleter>>)
	|| (!oa::IsCopyConstructibleV<oa::DecayT<Deleter>>
		&& oa::IsNothrowMoveConstructibleV<oa::DecayT<Deleter>>);

template<typename T>
[[nodiscard]] SharedControl* createDefaultSharedControl_(T* inP) {
	oa::DefaultDelete<T> deleter;
	return createSharedControl_(inP, deleter);
}

template<typename T>
struct SharedControlInline final : SharedControl {
	alignas(T) unsigned char buffer[sizeof(T)];

	template<typename... Args>
	explicit SharedControlInline(Args&&... inArgs) {
		oa::constructAt(reinterpret_cast<T*>(buffer), oa::forward<Args>(inArgs)...);
	}

	T* objectPtr_() noexcept {
		return oa::launder(reinterpret_cast<T*>(buffer));
	}

	void releaseObject() noexcept override {
		oa::destroyAt(objectPtr_());
	}

	void destroyControl() noexcept override {
		auto* self = this;
		oa::destroyAt(self);
		oa::freeBytes(self, alignof(SharedControlInline));
	}

	void releaseObjectAndDestroyControl() noexcept override {
		oa::destroyAt(objectPtr_());
		destroyControl();
	}
};

template<typename T>
class SharedPtr {
public:
	using element_type = T;
	using weak_type = WeakPtr<T>;

	template<typename U, typename... Args>
	requires (!oa::IsVoidV<U>)
	friend SharedPtr<U> makeShared(Args&&... inArgs);

	friend class WeakPtr<T>;

	template<typename U>
	friend class SharedPtr;

	SharedPtr() noexcept = default;

	SharedPtr(decltype(nullptr)) noexcept {}

	explicit SharedPtr(T* inP) requires (!oa::IsVoidV<T>)
		: control_(inP ? createDefaultSharedControl_(inP) : nullptr)
		, ptr_(inP) {}

	template<typename Deleter>
	requires oa::IsSharedDeleterV<Deleter>
	SharedPtr(T* inP, Deleter inD)
		: control_(inP ? createSharedControl_(inP, inD) : nullptr)
		, ptr_(inP) {}

	SharedPtr(const SharedPtr& inO) noexcept : control_(inO.control_), ptr_(inO.ptr_) {
		if (control_) {
			control_->incStrong();
		}
	}

	SharedPtr& operator=(const SharedPtr& inO) noexcept {
		if (control_ == inO.control_) {
			return *this;
		}
		// Acquire the incoming owner before retiring the old one. Destruction of
		// the old object may re-enter this SharedPtr; it must observe a complete
		// new state rather than the control block currently being released.
		SharedPtr replacement(inO);
		swap(replacement);
		return *this;
	}

	SharedPtr(SharedPtr&& inO) noexcept : control_(inO.control_), ptr_(inO.ptr_) {
		inO.control_ = nullptr;
		inO.ptr_ = nullptr;
	}

	template<typename U>
	requires (!oa::IsVoidV<T> && !oa::IsSameV<T, U>
		&& oa::IsConvertibleV<U*, T*>)
	SharedPtr(const SharedPtr<U>& inO) noexcept
		: control_(inO.control_), ptr_(static_cast<T*>(inO.ptr_)) {
		if (control_) {
			control_->incStrong();
		}
	}

	template<typename U>
	requires (!oa::IsVoidV<T> && !oa::IsSameV<T, U>
		&& oa::IsConvertibleV<U*, T*>)
	SharedPtr(SharedPtr<U>&& inO) noexcept
		: control_(inO.control_), ptr_(static_cast<T*>(inO.ptr_)) {
		inO.control_ = nullptr;
		inO.ptr_ = nullptr;
	}

	SharedPtr& operator=(SharedPtr&& inO) noexcept {
		if (this != &inO) {
			SharedPtr replacement(oa::move(inO));
			swap(replacement);
		}
		return *this;
	}

	SharedPtr& operator=(decltype(nullptr)) noexcept {
		reset();
		return *this;
	}

	~SharedPtr() { reset(); }

	[[nodiscard]] T* get() const noexcept { return ptr_; }
	[[nodiscard]] long useCount() const noexcept {
		return control_ ? control_->strong.load(oa::MemoryOrder::Relaxed) : 0;
	}

	explicit operator bool() const noexcept { return ptr_ != nullptr; }

	[[nodiscard]] T& operator*() const requires (!oa::IsVoidV<T>) { return *ptr_; }
	[[nodiscard]] T* operator->() const noexcept requires (!oa::IsVoidV<T>) { return ptr_; }

	void reset() noexcept {
		SharedControl* retired = control_;
		control_ = nullptr;
		ptr_ = nullptr;
		if (retired) retired->decStrong();
	}

	void reset(T* inP) requires (!oa::IsVoidV<T>) {
		SharedPtr replacement(inP);
		swap(replacement);
	}

	template<typename Deleter>
	requires oa::IsSharedDeleterV<Deleter>
	void reset(T* inP, Deleter inD) {
		SharedPtr replacement(inP, oa::move(inD));
		swap(replacement);
	}

	void swap(SharedPtr& inO) noexcept {
		SharedControl* tc = control_;
		T* tp = ptr_;
		control_ = inO.control_;
		ptr_ = inO.ptr_;
		inO.control_ = tc;
		inO.ptr_ = tp;
	}

private:
	SharedControl* control_{nullptr};
	T* ptr_{nullptr};

	SharedPtr(SharedControl* inCb, T* inPtr) noexcept : control_(inCb), ptr_(inPtr) {}
};

template<typename T>
[[nodiscard]] inline bool operator==(
	const SharedPtr<T>& inP,
	decltype(nullptr)
) noexcept {
	return inP.get() == nullptr;
}
template<typename T>
[[nodiscard]] inline bool operator==(
	decltype(nullptr),
	const SharedPtr<T>& inP
) noexcept {
	return inP.get() == nullptr;
}
template<typename T>
[[nodiscard]] inline bool operator!=(
	const SharedPtr<T>& inP,
	decltype(nullptr)
) noexcept {
	return inP.get() != nullptr;
}
template<typename T>
[[nodiscard]] inline bool operator!=(
	decltype(nullptr),
	const SharedPtr<T>& inP
) noexcept {
	return inP.get() != nullptr;
}

template<typename T>
class WeakPtr {
public:
	using element_type = T;

	WeakPtr() noexcept = default;

	WeakPtr(const WeakPtr& inO) noexcept : control_(inO.control_), ptr_(inO.ptr_) {
		if (control_) {
			control_->incWeak();
		}
	}

	WeakPtr& operator=(const WeakPtr& inO) noexcept {
		if (control_ == inO.control_) {
			return *this;
		}
		WeakPtr replacement(inO);
		swap(replacement);
		return *this;
	}

	WeakPtr(WeakPtr&& inO) noexcept : control_(inO.control_), ptr_(inO.ptr_) {
		inO.control_ = nullptr;
		inO.ptr_ = nullptr;
	}

	WeakPtr& operator=(WeakPtr&& inO) noexcept {
		if (this != &inO) {
			WeakPtr replacement(oa::move(inO));
			swap(replacement);
		}
		return *this;
	}

	explicit WeakPtr(const SharedPtr<T>& inS) noexcept : control_(inS.control_), ptr_(inS.ptr_) {
		if (control_) {
			control_->incWeak();
		}
	}

	~WeakPtr() { reset(); }

	[[nodiscard]] long useCount() const noexcept {
		return control_ ? control_->strong.load(oa::MemoryOrder::Relaxed) : 0;
	}

	[[nodiscard]] bool expired() const noexcept {
		return !control_
			|| control_->strong.load(oa::MemoryOrder::Acquire) == 0;
	}

	[[nodiscard]] SharedPtr<T> lock() const noexcept {
		if (!control_ || !control_->incStrongIfNonzero()) {
			return {};
		}
		return SharedPtr<T>(control_, ptr_);
	}

	void reset() noexcept {
		SharedControl* retired = control_;
		control_ = nullptr;
		ptr_ = nullptr;
		if (retired) retired->decWeak();
	}

	void swap(WeakPtr& inO) noexcept {
		SharedControl* tc = control_;
		T* tp = ptr_;
		control_ = inO.control_;
		ptr_ = inO.ptr_;
		inO.control_ = tc;
		inO.ptr_ = tp;
	}

private:
	SharedControl* control_{nullptr};
	T* ptr_{nullptr};
};

template<>
class SharedPtr<void> {
public:
	using element_type = void;
	using weak_type = WeakPtr<void>;

	friend class WeakPtr<void>;

	SharedPtr() noexcept = default;

	SharedPtr(decltype(nullptr)) noexcept {}

	template<typename Deleter>
	requires oa::IsSharedDeleterV<Deleter>
	SharedPtr(void* inP, Deleter inD)
		: control_(inP ? createSharedControl_(inP, inD) : nullptr)
		, ptr_(inP) {}

	SharedPtr(const SharedPtr& inO) noexcept : control_(inO.control_), ptr_(inO.ptr_) {
		if (control_) {
			control_->incStrong();
		}
	}

	SharedPtr& operator=(const SharedPtr& inO) noexcept {
		if (control_ == inO.control_) {
			return *this;
		}
		SharedPtr replacement(inO);
		swap(replacement);
		return *this;
	}

	SharedPtr(SharedPtr&& inO) noexcept : control_(inO.control_), ptr_(inO.ptr_) {
		inO.control_ = nullptr;
		inO.ptr_ = nullptr;
	}

	SharedPtr& operator=(SharedPtr&& inO) noexcept {
		if (this != &inO) {
			SharedPtr replacement(oa::move(inO));
			swap(replacement);
		}
		return *this;
	}

	SharedPtr& operator=(decltype(nullptr)) noexcept {
		reset();
		return *this;
	}

	~SharedPtr() { reset(); }

	[[nodiscard]] void* get() const noexcept { return ptr_; }
	[[nodiscard]] long useCount() const noexcept {
		return control_ ? control_->strong.load(oa::MemoryOrder::Relaxed) : 0;
	}

	explicit operator bool() const noexcept { return ptr_ != nullptr; }

	void reset() noexcept {
		SharedControl* retired = control_;
		control_ = nullptr;
		ptr_ = nullptr;
		if (retired) retired->decStrong();
	}

	template<typename Deleter>
	requires oa::IsSharedDeleterV<Deleter>
	void reset(void* inP, Deleter inD) {
		SharedPtr replacement(inP, oa::move(inD));
		swap(replacement);
	}

	void swap(SharedPtr& inO) noexcept {
		SharedControl* tc = control_;
		void* tp = ptr_;
		control_ = inO.control_;
		ptr_ = inO.ptr_;
		inO.control_ = tc;
		inO.ptr_ = tp;
	}

private:
	SharedControl* control_{nullptr};
	void* ptr_{nullptr};

	SharedPtr(SharedControl* inCb, void* inPtr) noexcept : control_(inCb), ptr_(inPtr) {}
};

template<>
class WeakPtr<void> {
public:
	using element_type = void;

	WeakPtr() noexcept = default;

	WeakPtr(const WeakPtr& inO) noexcept : control_(inO.control_), ptr_(inO.ptr_) {
		if (control_) {
			control_->incWeak();
		}
	}

	WeakPtr& operator=(const WeakPtr& inO) noexcept {
		if (control_ == inO.control_) {
			return *this;
		}
		WeakPtr replacement(inO);
		swap(replacement);
		return *this;
	}

	WeakPtr(WeakPtr&& inO) noexcept : control_(inO.control_), ptr_(inO.ptr_) {
		inO.control_ = nullptr;
		inO.ptr_ = nullptr;
	}

	WeakPtr& operator=(WeakPtr&& inO) noexcept {
		if (this != &inO) {
			WeakPtr replacement(oa::move(inO));
			swap(replacement);
		}
		return *this;
	}

	explicit WeakPtr(const SharedPtr<void>& inS) noexcept : control_(inS.control_), ptr_(inS.ptr_) {
		if (control_) {
			control_->incWeak();
		}
	}

	~WeakPtr() { reset(); }

	[[nodiscard]] long useCount() const noexcept {
		return control_ ? control_->strong.load(oa::MemoryOrder::Relaxed) : 0;
	}

	[[nodiscard]] bool expired() const noexcept {
		return !control_
			|| control_->strong.load(oa::MemoryOrder::Acquire) == 0;
	}

	[[nodiscard]] SharedPtr<void> lock() const noexcept {
		if (!control_ || !control_->incStrongIfNonzero()) {
			return {};
		}
		return SharedPtr<void>(control_, ptr_);
	}

	void reset() noexcept {
		SharedControl* retired = control_;
		control_ = nullptr;
		ptr_ = nullptr;
		if (retired) retired->decWeak();
	}

	void swap(WeakPtr& inO) noexcept {
		SharedControl* tc = control_;
		void* tp = ptr_;
		control_ = inO.control_;
		ptr_ = inO.ptr_;
		inO.control_ = tc;
		inO.ptr_ = tp;
	}

private:
	SharedControl* control_{nullptr};
	void* ptr_{nullptr};
};

template<typename T, typename... Args>
requires (!oa::IsVoidV<T>)
[[nodiscard]] SharedPtr<T> makeShared(Args&&... inArgs) {
	auto* cb = allocateSharedControl_<SharedControlInline<T>>(
		oa::forward<Args>(inArgs)...);
	return SharedPtr<T>(cb, cb->objectPtr_());
}

} // namespace oa
