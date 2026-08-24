#pragma once

// Native SharedPtr / WeakPtr — atomic strong + weak counts, type-erased control block.
//
// `makeShared<T>(...)` allocates control block + `T` in one slab (inline storage).
// Thread-safe refcounting; `lock()` from weak promotes only if object still alive.

#include <oa/core/std/typeTraits.h>
#include <oa/core/std/utility.h>

#include <atomic>
#include <cstddef>
#include <memory>
#include <new>
#include <type_traits>

namespace oa {

template<typename T>
class WeakPtr;

struct SharedControl {
	std::atomic<long> strong{1};
	// One implicit weak reference keeps the control block alive while any
	// strong owner exists. Explicit WeakPtr instances add to this count.
	std::atomic<long> weak{1};

	void incStrong() noexcept {
		strong.fetch_add(1, std::memory_order_relaxed);
	}

	bool incStrongIfNonzero() noexcept {
		long n = strong.load(std::memory_order_relaxed);
		for (;;) {
			if (n == 0) {
				return false;
			}
			if (strong.compare_exchange_weak(n, n + 1, std::memory_order_acq_rel, std::memory_order_relaxed)) {
				return true;
			}
		}
	}

	void incWeak() noexcept {
		weak.fetch_add(1, std::memory_order_relaxed);
	}

	void decStrong() noexcept {
		if (strong.fetch_sub(1, std::memory_order_acq_rel) != 1) {
			return;
		}
		releaseObject();
		decWeak();
	}

	void decWeak() noexcept {
		if (weak.fetch_sub(1, std::memory_order_acq_rel) != 1) {
			return;
		}
		destroyControl();
	}

	virtual void releaseObject() noexcept = 0;
	virtual void destroyControl() noexcept = 0;

protected:
	virtual ~SharedControl() = default;
};

template<typename T, typename Deleter>
struct SharedControlDeleter final : SharedControl {
	T* ptr{};
	Deleter deleter;

	SharedControlDeleter(T* inP, const Deleter& inD) : ptr(inP), deleter(inD) {}

	SharedControlDeleter(T* inP, Deleter&& inD) noexcept(std::is_nothrow_move_constructible_v<Deleter>)
		: ptr(inP), deleter(oa::move(inD)) {}

	void releaseObject() noexcept override {
		if (ptr) {
			deleter(ptr);
			ptr = nullptr;
		}
	}

	void destroyControl() noexcept override {
		delete this;
	}
};

template<typename T, typename Deleter>
[[nodiscard]] SharedControl* createSharedControl_(T* inP, Deleter& inD) {
	using StoredDeleter = oa::DecayT<Deleter>;
	try {
		if constexpr (std::is_copy_constructible_v<StoredDeleter>) {
			return new SharedControlDeleter<T, StoredDeleter>(
				inP, static_cast<const StoredDeleter&>(inD));
		} else {
			static_assert(std::is_nothrow_move_constructible_v<StoredDeleter>,
				"A move-only SharedPtr deleter must be nothrow move constructible");
			return new SharedControlDeleter<T, StoredDeleter>(
				inP, oa::move(inD));
		}
	} catch (...) {
		inD(inP);
		throw;
	}
}

template<typename T>
[[nodiscard]] SharedControl* createDefaultSharedControl_(T* inP) {
	std::default_delete<T> deleter;
	return createSharedControl_(inP, deleter);
}

template<typename T>
struct SharedControlInline final : SharedControl {
	alignas(T) unsigned char buffer[sizeof(T)];

	template<typename... Args>
	explicit SharedControlInline(Args&&... inArgs) {
		new (buffer) T(oa::forward<Args>(inArgs)...);
	}

	T* objectPtr_() noexcept {
		return std::launder(reinterpret_cast<T*>(buffer));
	}

	void releaseObject() noexcept override {
		objectPtr_()->~T();
	}

	void destroyControl() noexcept override {
		delete this;
	}
};

template<typename T>
class SharedPtr {
public:
	using element_type = T;
	using weak_type = WeakPtr<T>;

	template<typename U, typename... Args>
	requires (!std::is_void_v<U>)
	friend SharedPtr<U> makeShared(Args&&... inArgs);

	friend class WeakPtr<T>;

	template<typename U>
	friend class SharedPtr;

	SharedPtr() noexcept = default;

	SharedPtr(std::nullptr_t) noexcept {}

	explicit SharedPtr(T* inP) requires (!std::is_void_v<T>)
		: control_(inP ? createDefaultSharedControl_(inP) : nullptr)
		, ptr_(inP) {}

	template<typename Deleter>
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
		if (control_) {
			control_->decStrong();
		}
		control_ = inO.control_;
		ptr_ = inO.ptr_;
		if (control_) {
			control_->incStrong();
		}
		return *this;
	}

	SharedPtr(SharedPtr&& inO) noexcept : control_(inO.control_), ptr_(inO.ptr_) {
		inO.control_ = nullptr;
		inO.ptr_ = nullptr;
	}

	template<typename U>
	requires (!std::is_void_v<T> && !std::is_same_v<T, U> && std::is_convertible_v<U*, T*>)
	SharedPtr(const SharedPtr<U>& inO) noexcept
		: control_(inO.control_), ptr_(static_cast<T*>(inO.ptr_)) {
		if (control_) {
			control_->incStrong();
		}
	}

	template<typename U>
	requires (!std::is_void_v<T> && !std::is_same_v<T, U> && std::is_convertible_v<U*, T*>)
	SharedPtr(SharedPtr<U>&& inO) noexcept
		: control_(inO.control_), ptr_(static_cast<T*>(inO.ptr_)) {
		inO.control_ = nullptr;
		inO.ptr_ = nullptr;
	}

	SharedPtr& operator=(SharedPtr&& inO) noexcept {
		if (this != &inO) {
			if (control_) {
				control_->decStrong();
			}
			control_ = inO.control_;
			ptr_ = inO.ptr_;
			inO.control_ = nullptr;
			inO.ptr_ = nullptr;
		}
		return *this;
	}

	SharedPtr& operator=(std::nullptr_t) noexcept {
		reset();
		return *this;
	}

	~SharedPtr() {
		if (control_) {
			control_->decStrong();
		}
	}

	[[nodiscard]] T* get() const noexcept { return ptr_; }
	[[nodiscard]] long useCount() const noexcept {
		return control_ ? control_->strong.load(std::memory_order_relaxed) : 0;
	}

	explicit operator bool() const noexcept { return ptr_ != nullptr; }

	[[nodiscard]] T& operator*() const requires (!std::is_void_v<T>) { return *ptr_; }
	[[nodiscard]] T* operator->() const noexcept requires (!std::is_void_v<T>) { return ptr_; }

	void reset() noexcept {
		if (control_) {
			control_->decStrong();
			control_ = nullptr;
			ptr_ = nullptr;
		}
	}

	void reset(T* inP) requires (!std::is_void_v<T>) {
		SharedPtr replacement(inP);
		swap(replacement);
	}

	template<typename Deleter>
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
[[nodiscard]] inline bool operator==(const SharedPtr<T>& inP, std::nullptr_t) noexcept {
	return inP.get() == nullptr;
}
template<typename T>
[[nodiscard]] inline bool operator==(std::nullptr_t, const SharedPtr<T>& inP) noexcept {
	return inP.get() == nullptr;
}
template<typename T>
[[nodiscard]] inline bool operator!=(const SharedPtr<T>& inP, std::nullptr_t) noexcept {
	return inP.get() != nullptr;
}
template<typename T>
[[nodiscard]] inline bool operator!=(std::nullptr_t, const SharedPtr<T>& inP) noexcept {
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
		if (control_) {
			control_->decWeak();
		}
		control_ = inO.control_;
		ptr_ = inO.ptr_;
		if (control_) {
			control_->incWeak();
		}
		return *this;
	}

	WeakPtr(WeakPtr&& inO) noexcept : control_(inO.control_), ptr_(inO.ptr_) {
		inO.control_ = nullptr;
		inO.ptr_ = nullptr;
	}

	WeakPtr& operator=(WeakPtr&& inO) noexcept {
		if (this != &inO) {
			if (control_) {
				control_->decWeak();
			}
			control_ = inO.control_;
			ptr_ = inO.ptr_;
			inO.control_ = nullptr;
			inO.ptr_ = nullptr;
		}
		return *this;
	}

	explicit WeakPtr(const SharedPtr<T>& inS) noexcept : control_(inS.control_), ptr_(inS.ptr_) {
		if (control_) {
			control_->incWeak();
		}
	}

	~WeakPtr() {
		if (control_) {
			control_->decWeak();
		}
	}

	[[nodiscard]] long useCount() const noexcept {
		return control_ ? control_->strong.load(std::memory_order_relaxed) : 0;
	}

	[[nodiscard]] bool expired() const noexcept {
		return !control_ || control_->strong.load(std::memory_order_acquire) == 0;
	}

	[[nodiscard]] SharedPtr<T> lock() const noexcept {
		if (!control_ || !control_->incStrongIfNonzero()) {
			return {};
		}
		return SharedPtr<T>(control_, ptr_);
	}

	void reset() noexcept {
		if (control_) {
			control_->decWeak();
			control_ = nullptr;
			ptr_ = nullptr;
		}
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

	SharedPtr(std::nullptr_t) noexcept {}

	template<typename Deleter>
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
		if (control_) {
			control_->decStrong();
		}
		control_ = inO.control_;
		ptr_ = inO.ptr_;
		if (control_) {
			control_->incStrong();
		}
		return *this;
	}

	SharedPtr(SharedPtr&& inO) noexcept : control_(inO.control_), ptr_(inO.ptr_) {
		inO.control_ = nullptr;
		inO.ptr_ = nullptr;
	}

	SharedPtr& operator=(SharedPtr&& inO) noexcept {
		if (this != &inO) {
			if (control_) {
				control_->decStrong();
			}
			control_ = inO.control_;
			ptr_ = inO.ptr_;
			inO.control_ = nullptr;
			inO.ptr_ = nullptr;
		}
		return *this;
	}

	SharedPtr& operator=(std::nullptr_t) noexcept {
		reset();
		return *this;
	}

	~SharedPtr() {
		if (control_) {
			control_->decStrong();
		}
	}

	[[nodiscard]] void* get() const noexcept { return ptr_; }
	[[nodiscard]] long useCount() const noexcept {
		return control_ ? control_->strong.load(std::memory_order_relaxed) : 0;
	}

	explicit operator bool() const noexcept { return ptr_ != nullptr; }

	void reset() noexcept {
		if (control_) {
			control_->decStrong();
			control_ = nullptr;
			ptr_ = nullptr;
		}
	}

	template<typename Deleter>
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
		if (control_) {
			control_->decWeak();
		}
		control_ = inO.control_;
		ptr_ = inO.ptr_;
		if (control_) {
			control_->incWeak();
		}
		return *this;
	}

	WeakPtr(WeakPtr&& inO) noexcept : control_(inO.control_), ptr_(inO.ptr_) {
		inO.control_ = nullptr;
		inO.ptr_ = nullptr;
	}

	WeakPtr& operator=(WeakPtr&& inO) noexcept {
		if (this != &inO) {
			if (control_) {
				control_->decWeak();
			}
			control_ = inO.control_;
			ptr_ = inO.ptr_;
			inO.control_ = nullptr;
			inO.ptr_ = nullptr;
		}
		return *this;
	}

	explicit WeakPtr(const SharedPtr<void>& inS) noexcept : control_(inS.control_), ptr_(inS.ptr_) {
		if (control_) {
			control_->incWeak();
		}
	}

	~WeakPtr() {
		if (control_) {
			control_->decWeak();
		}
	}

	[[nodiscard]] long useCount() const noexcept {
		return control_ ? control_->strong.load(std::memory_order_relaxed) : 0;
	}

	[[nodiscard]] bool expired() const noexcept {
		return !control_ || control_->strong.load(std::memory_order_acquire) == 0;
	}

	[[nodiscard]] SharedPtr<void> lock() const noexcept {
		if (!control_ || !control_->incStrongIfNonzero()) {
			return {};
		}
		return SharedPtr<void>(control_, ptr_);
	}

	void reset() noexcept {
		if (control_) {
			control_->decWeak();
			control_ = nullptr;
			ptr_ = nullptr;
		}
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
requires (!std::is_void_v<T>)
[[nodiscard]] SharedPtr<T> makeShared(Args&&... inArgs) {
	auto* cb = new SharedControlInline<T>(oa::forward<Args>(inArgs)...);
	return SharedPtr<T>(cb, cb->objectPtr_());
}

} // namespace oa
