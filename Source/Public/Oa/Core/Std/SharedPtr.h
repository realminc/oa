#pragma once

// Native OaStdSharedPtr / OaStdWeakPtr — atomic strong + weak counts, type-erased control block.
//
// `OaStdMakeShared<T>(...)` allocates control block + `T` in one slab (inline storage).
// Thread-safe refcounting; `Lock()` from weak promotes only if object still alive.

#include <Oa/Core/Std/TypeTraits.h>
#include <Oa/Core/Std/Utility.h>

#include <atomic>
#include <cstddef>
#include <new>
#include <type_traits>

template<typename T>
class OaStdWeakPtr;

struct OaStdSharedControl {
	std::atomic<long> Strong{1};
	std::atomic<long> Weak{0};

	void IncStrong() noexcept {
		Strong.fetch_add(1, std::memory_order_relaxed);
	}

	bool IncStrongIfNonzero() noexcept {
		long n = Strong.load(std::memory_order_relaxed);
		for (;;) {
			if (n == 0) {
				return false;
			}
			if (Strong.compare_exchange_weak(n, n + 1, std::memory_order_acq_rel, std::memory_order_relaxed)) {
				return true;
			}
		}
	}

	void IncWeak() noexcept {
		Weak.fetch_add(1, std::memory_order_relaxed);
	}

	void DecStrong() noexcept {
		if (Strong.fetch_sub(1, std::memory_order_acq_rel) != 1) {
			return;
		}
		ReleaseObject();
		if (Weak.load(std::memory_order_acquire) == 0) {
			DestroyControl();
		}
	}

	void DecWeak() noexcept {
		if (Weak.fetch_sub(1, std::memory_order_acq_rel) != 1) {
			return;
		}
		if (Strong.load(std::memory_order_acquire) == 0) {
			DestroyControl();
		}
	}

	virtual void ReleaseObject() noexcept = 0;
	virtual void DestroyControl() noexcept = 0;

protected:
	virtual ~OaStdSharedControl() = default;
};

template<typename T, typename Deleter>
struct OaStdSharedControlDeleter final : OaStdSharedControl {
	T* Ptr{};
	Deleter D{};

	OaStdSharedControlDeleter(T* InP, Deleter InD) : Ptr(InP), D(OaStdMove(InD)) {}

	void ReleaseObject() noexcept override {
		if (Ptr) {
			D(Ptr);
			Ptr = nullptr;
		}
	}

	void DestroyControl() noexcept override {
		delete this;
	}
};

template<typename T>
struct OaStdSharedControlInline final : OaStdSharedControl {
	alignas(T) unsigned char Buf[sizeof(T)];

	template<typename... Args>
	explicit OaStdSharedControlInline(Args&&... InArgs) {
		new (Buf) T(OaStdForward<Args>(InArgs)...);
	}

	T* ObjectPtr() noexcept {
		return std::launder(reinterpret_cast<T*>(Buf));
	}

	void ReleaseObject() noexcept override {
		ObjectPtr()->~T();
	}

	void DestroyControl() noexcept override {
		delete this;
	}
};

template<typename T>
class OaStdSharedPtr {
public:
	using element_type = T;
	using weak_type = OaStdWeakPtr<T>;

	template<typename U, typename... Args>
	requires (!std::is_void_v<U>)
	friend OaStdSharedPtr<U> OaStdMakeShared(Args&&... InArgs);

	friend class OaStdWeakPtr<T>;

	template<typename U>
	friend class OaStdSharedPtr;

	OaStdSharedPtr() noexcept = default;

	OaStdSharedPtr(std::nullptr_t) noexcept {}

	explicit OaStdSharedPtr(T* InP) requires (!std::is_void_v<T>)
		: Ptr_(InP)
		, Cb_(InP ? static_cast<OaStdSharedControl*>(new OaStdSharedControlDeleter<T, std::default_delete<T>>(
					  InP, std::default_delete<T>{}))
				: nullptr) {}

	template<typename Deleter>
	OaStdSharedPtr(T* InP, Deleter InD)
		: Ptr_(InP)
		, Cb_(InP ? static_cast<OaStdSharedControl*>(new OaStdSharedControlDeleter<T, OaStdDecayT<Deleter>>(
					  InP, OaStdForward<Deleter>(InD)))
				: nullptr) {}

	OaStdSharedPtr(const OaStdSharedPtr& InO) noexcept : Cb_(InO.Cb_), Ptr_(InO.Ptr_) {
		if (Cb_) {
			Cb_->IncStrong();
		}
	}

	OaStdSharedPtr& operator=(const OaStdSharedPtr& InO) noexcept {
		if (Cb_ == InO.Cb_) {
			return *this;
		}
		if (Cb_) {
			Cb_->DecStrong();
		}
		Cb_ = InO.Cb_;
		Ptr_ = InO.Ptr_;
		if (Cb_) {
			Cb_->IncStrong();
		}
		return *this;
	}

	OaStdSharedPtr(OaStdSharedPtr&& InO) noexcept : Cb_(InO.Cb_), Ptr_(InO.Ptr_) {
		InO.Cb_ = nullptr;
		InO.Ptr_ = nullptr;
	}

	template<typename U>
	requires (!std::is_void_v<T> && !std::is_same_v<T, U> && std::is_convertible_v<U*, T*>)
	OaStdSharedPtr(const OaStdSharedPtr<U>& InO) noexcept
		: Cb_(InO.Cb_), Ptr_(static_cast<T*>(InO.Ptr_)) {
		if (Cb_) {
			Cb_->IncStrong();
		}
	}

	template<typename U>
	requires (!std::is_void_v<T> && !std::is_same_v<T, U> && std::is_convertible_v<U*, T*>)
	OaStdSharedPtr(OaStdSharedPtr<U>&& InO) noexcept
		: Cb_(InO.Cb_), Ptr_(static_cast<T*>(InO.Ptr_)) {
		InO.Cb_ = nullptr;
		InO.Ptr_ = nullptr;
	}

	OaStdSharedPtr& operator=(OaStdSharedPtr&& InO) noexcept {
		if (this != &InO) {
			if (Cb_) {
				Cb_->DecStrong();
			}
			Cb_ = InO.Cb_;
			Ptr_ = InO.Ptr_;
			InO.Cb_ = nullptr;
			InO.Ptr_ = nullptr;
		}
		return *this;
	}

	OaStdSharedPtr& operator=(std::nullptr_t) noexcept {
		Reset();
		return *this;
	}

	~OaStdSharedPtr() {
		if (Cb_) {
			Cb_->DecStrong();
		}
	}

	[[nodiscard]] T* Get() const noexcept { return Ptr_; }
	[[nodiscard]] T* get() const noexcept { return Ptr_; }
	[[nodiscard]] long UseCount() const noexcept {
		return Cb_ ? Cb_->Strong.load(std::memory_order_relaxed) : 0;
	}

	explicit operator bool() const noexcept { return Ptr_ != nullptr; }

	[[nodiscard]] T& operator*() const requires (!std::is_void_v<T>) { return *Ptr_; }
	[[nodiscard]] T* operator->() const noexcept requires (!std::is_void_v<T>) { return Ptr_; }

	void Reset() noexcept {
		if (Cb_) {
			Cb_->DecStrong();
			Cb_ = nullptr;
			Ptr_ = nullptr;
		}
	}

	void Reset(T* InP) requires (!std::is_void_v<T>) {
		Reset();
		if (!InP) {
			return;
		}
		Cb_ = static_cast<OaStdSharedControl*>(
			new OaStdSharedControlDeleter<T, std::default_delete<T>>(InP, std::default_delete<T>{}));
		Ptr_ = InP;
	}

	template<typename Deleter>
	void Reset(T* InP, Deleter InD) {
		Reset();
		if (!InP) {
			return;
		}
		Cb_ = static_cast<OaStdSharedControl*>(
			new OaStdSharedControlDeleter<T, OaStdDecayT<Deleter>>(InP, OaStdForward<Deleter>(InD)));
		Ptr_ = InP;
	}

	void Swap(OaStdSharedPtr& InO) noexcept {
		OaStdSharedControl* tc = Cb_;
		T* tp = Ptr_;
		Cb_ = InO.Cb_;
		Ptr_ = InO.Ptr_;
		InO.Cb_ = tc;
		InO.Ptr_ = tp;
	}

private:
	OaStdSharedControl* Cb_{nullptr};
	T* Ptr_{nullptr};

	OaStdSharedPtr(OaStdSharedControl* InCb, T* InPtr) noexcept : Cb_(InCb), Ptr_(InPtr) {}
};

template<typename T>
[[nodiscard]] inline bool operator==(const OaStdSharedPtr<T>& InP, std::nullptr_t) noexcept {
	return InP.Get() == nullptr;
}
template<typename T>
[[nodiscard]] inline bool operator==(std::nullptr_t, const OaStdSharedPtr<T>& InP) noexcept {
	return InP.Get() == nullptr;
}
template<typename T>
[[nodiscard]] inline bool operator!=(const OaStdSharedPtr<T>& InP, std::nullptr_t) noexcept {
	return InP.Get() != nullptr;
}
template<typename T>
[[nodiscard]] inline bool operator!=(std::nullptr_t, const OaStdSharedPtr<T>& InP) noexcept {
	return InP.Get() != nullptr;
}

template<typename T>
class OaStdWeakPtr {
public:
	using element_type = T;

	OaStdWeakPtr() noexcept = default;

	OaStdWeakPtr(const OaStdWeakPtr& InO) noexcept : Cb_(InO.Cb_), Ptr_(InO.Ptr_) {
		if (Cb_) {
			Cb_->IncWeak();
		}
	}

	OaStdWeakPtr& operator=(const OaStdWeakPtr& InO) noexcept {
		if (Cb_ == InO.Cb_) {
			return *this;
		}
		if (Cb_) {
			Cb_->DecWeak();
		}
		Cb_ = InO.Cb_;
		Ptr_ = InO.Ptr_;
		if (Cb_) {
			Cb_->IncWeak();
		}
		return *this;
	}

	OaStdWeakPtr(OaStdWeakPtr&& InO) noexcept : Cb_(InO.Cb_), Ptr_(InO.Ptr_) {
		InO.Cb_ = nullptr;
		InO.Ptr_ = nullptr;
	}

	OaStdWeakPtr& operator=(OaStdWeakPtr&& InO) noexcept {
		if (this != &InO) {
			if (Cb_) {
				Cb_->DecWeak();
			}
			Cb_ = InO.Cb_;
			Ptr_ = InO.Ptr_;
			InO.Cb_ = nullptr;
			InO.Ptr_ = nullptr;
		}
		return *this;
	}

	explicit OaStdWeakPtr(const OaStdSharedPtr<T>& InS) noexcept : Cb_(InS.Cb_), Ptr_(InS.Ptr_) {
		if (Cb_) {
			Cb_->IncWeak();
		}
	}

	~OaStdWeakPtr() {
		if (Cb_) {
			Cb_->DecWeak();
		}
	}

	[[nodiscard]] long UseCount() const noexcept {
		return Cb_ ? Cb_->Strong.load(std::memory_order_relaxed) : 0;
	}

	[[nodiscard]] bool Expired() const noexcept {
		return !Cb_ || Cb_->Strong.load(std::memory_order_acquire) == 0;
	}

	[[nodiscard]] OaStdSharedPtr<T> Lock() const noexcept {
		if (!Cb_ || !Cb_->IncStrongIfNonzero()) {
			return {};
		}
		return OaStdSharedPtr<T>(Cb_, Ptr_);
	}

	void Reset() noexcept {
		if (Cb_) {
			Cb_->DecWeak();
			Cb_ = nullptr;
			Ptr_ = nullptr;
		}
	}

	void Swap(OaStdWeakPtr& InO) noexcept {
		OaStdSharedControl* tc = Cb_;
		T* tp = Ptr_;
		Cb_ = InO.Cb_;
		Ptr_ = InO.Ptr_;
		InO.Cb_ = tc;
		InO.Ptr_ = tp;
	}

private:
	OaStdSharedControl* Cb_{nullptr};
	T* Ptr_{nullptr};
};

template<>
class OaStdSharedPtr<void> {
public:
	using element_type = void;
	using weak_type = OaStdWeakPtr<void>;

	friend class OaStdWeakPtr<void>;

	OaStdSharedPtr() noexcept = default;

	OaStdSharedPtr(std::nullptr_t) noexcept {}

	template<typename Deleter>
	OaStdSharedPtr(void* InP, Deleter InD)
		: Cb_(InP ? static_cast<OaStdSharedControl*>(new OaStdSharedControlDeleter<void, OaStdDecayT<Deleter>>(
					  InP, OaStdForward<Deleter>(InD)))
				: nullptr)
		, Ptr_(InP) {}

	OaStdSharedPtr(const OaStdSharedPtr& InO) noexcept : Cb_(InO.Cb_), Ptr_(InO.Ptr_) {
		if (Cb_) {
			Cb_->IncStrong();
		}
	}

	OaStdSharedPtr& operator=(const OaStdSharedPtr& InO) noexcept {
		if (Cb_ == InO.Cb_) {
			return *this;
		}
		if (Cb_) {
			Cb_->DecStrong();
		}
		Cb_ = InO.Cb_;
		Ptr_ = InO.Ptr_;
		if (Cb_) {
			Cb_->IncStrong();
		}
		return *this;
	}

	OaStdSharedPtr(OaStdSharedPtr&& InO) noexcept : Cb_(InO.Cb_), Ptr_(InO.Ptr_) {
		InO.Cb_ = nullptr;
		InO.Ptr_ = nullptr;
	}

	OaStdSharedPtr& operator=(OaStdSharedPtr&& InO) noexcept {
		if (this != &InO) {
			if (Cb_) {
				Cb_->DecStrong();
			}
			Cb_ = InO.Cb_;
			Ptr_ = InO.Ptr_;
			InO.Cb_ = nullptr;
			InO.Ptr_ = nullptr;
		}
		return *this;
	}

	OaStdSharedPtr& operator=(std::nullptr_t) noexcept {
		Reset();
		return *this;
	}

	~OaStdSharedPtr() {
		if (Cb_) {
			Cb_->DecStrong();
		}
	}

	[[nodiscard]] void* Get() const noexcept { return Ptr_; }
	[[nodiscard]] void* get() const noexcept { return Ptr_; }
	[[nodiscard]] long UseCount() const noexcept {
		return Cb_ ? Cb_->Strong.load(std::memory_order_relaxed) : 0;
	}

	explicit operator bool() const noexcept { return Ptr_ != nullptr; }

	void Reset() noexcept {
		if (Cb_) {
			Cb_->DecStrong();
			Cb_ = nullptr;
			Ptr_ = nullptr;
		}
	}

	template<typename Deleter>
	void Reset(void* InP, Deleter InD) {
		Reset();
		if (!InP) {
			return;
		}
		Cb_ = static_cast<OaStdSharedControl*>(
			new OaStdSharedControlDeleter<void, OaStdDecayT<Deleter>>(InP, OaStdForward<Deleter>(InD)));
		Ptr_ = InP;
	}

	void Swap(OaStdSharedPtr& InO) noexcept {
		OaStdSharedControl* tc = Cb_;
		void* tp = Ptr_;
		Cb_ = InO.Cb_;
		Ptr_ = InO.Ptr_;
		InO.Cb_ = tc;
		InO.Ptr_ = tp;
	}

private:
	OaStdSharedControl* Cb_{nullptr};
	void* Ptr_{nullptr};

	OaStdSharedPtr(OaStdSharedControl* InCb, void* InPtr) noexcept : Cb_(InCb), Ptr_(InPtr) {}
};

template<>
class OaStdWeakPtr<void> {
public:
	using element_type = void;

	OaStdWeakPtr() noexcept = default;

	OaStdWeakPtr(const OaStdWeakPtr& InO) noexcept : Cb_(InO.Cb_), Ptr_(InO.Ptr_) {
		if (Cb_) {
			Cb_->IncWeak();
		}
	}

	OaStdWeakPtr& operator=(const OaStdWeakPtr& InO) noexcept {
		if (Cb_ == InO.Cb_) {
			return *this;
		}
		if (Cb_) {
			Cb_->DecWeak();
		}
		Cb_ = InO.Cb_;
		Ptr_ = InO.Ptr_;
		if (Cb_) {
			Cb_->IncWeak();
		}
		return *this;
	}

	OaStdWeakPtr(OaStdWeakPtr&& InO) noexcept : Cb_(InO.Cb_), Ptr_(InO.Ptr_) {
		InO.Cb_ = nullptr;
		InO.Ptr_ = nullptr;
	}

	OaStdWeakPtr& operator=(OaStdWeakPtr&& InO) noexcept {
		if (this != &InO) {
			if (Cb_) {
				Cb_->DecWeak();
			}
			Cb_ = InO.Cb_;
			Ptr_ = InO.Ptr_;
			InO.Cb_ = nullptr;
			InO.Ptr_ = nullptr;
		}
		return *this;
	}

	explicit OaStdWeakPtr(const OaStdSharedPtr<void>& InS) noexcept : Cb_(InS.Cb_), Ptr_(InS.Ptr_) {
		if (Cb_) {
			Cb_->IncWeak();
		}
	}

	~OaStdWeakPtr() {
		if (Cb_) {
			Cb_->DecWeak();
		}
	}

	[[nodiscard]] long UseCount() const noexcept {
		return Cb_ ? Cb_->Strong.load(std::memory_order_relaxed) : 0;
	}

	[[nodiscard]] bool Expired() const noexcept {
		return !Cb_ || Cb_->Strong.load(std::memory_order_acquire) == 0;
	}

	[[nodiscard]] OaStdSharedPtr<void> Lock() const noexcept {
		if (!Cb_ || !Cb_->IncStrongIfNonzero()) {
			return {};
		}
		return OaStdSharedPtr<void>(Cb_, Ptr_);
	}

	void Reset() noexcept {
		if (Cb_) {
			Cb_->DecWeak();
			Cb_ = nullptr;
			Ptr_ = nullptr;
		}
	}

	void Swap(OaStdWeakPtr& InO) noexcept {
		OaStdSharedControl* tc = Cb_;
		void* tp = Ptr_;
		Cb_ = InO.Cb_;
		Ptr_ = InO.Ptr_;
		InO.Cb_ = tc;
		InO.Ptr_ = tp;
	}

private:
	OaStdSharedControl* Cb_{nullptr};
	void* Ptr_{nullptr};
};

template<typename T, typename... Args>
requires (!std::is_void_v<T>)
[[nodiscard]] OaStdSharedPtr<T> OaStdMakeShared(Args&&... InArgs) {
	auto* cb = new OaStdSharedControlInline<T>(OaStdForward<Args>(InArgs)...);
	return OaStdSharedPtr<T>(cb, cb->ObjectPtr());
}
