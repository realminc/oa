#pragma once

// OA unique ownership — raw pointer + deleter; `makeUnique` uses `new T`
// (including `alignas` / over-aligned T).
//
// `reset` applies the deleter then nulls; the type is move-only and does not
// export a hosted standard-library ownership conversion.
//
// Incomplete-type support (PImpl): DefaultDelete requires T to be complete at
// the destruction point, as enforced by its static assertion. A custom deleter
// may explicitly support incomplete T and is always invoked; ownership is never
// silently discarded.

#include <oa/core/std/utility.h>
#include <oa/core/std/typeTraits.h>

namespace oa {

// True when T is a complete type at the point of instantiation.
template<typename T, typename = void>
inline constexpr bool isCompleteTypeV = false;
template<typename T>
inline constexpr bool isCompleteTypeV<T, oa::VoidT<decltype(sizeof(T))>> = true;

template<typename T>
struct DefaultDelete {
	constexpr DefaultDelete() noexcept = default;

	template<typename U, typename = oa::EnableIfT<oa::isConvertibleV<U*, T*>>>
	DefaultDelete(const DefaultDelete<U>&) noexcept {}

	void operator()(T* inP) const noexcept {
		static_assert(sizeof(T) > 0, "DefaultDelete incomplete type");
		delete inP;
	}
};

template<typename T, typename Deleter = DefaultDelete<T>>
class UniquePtr {
public:
	using pointer = T*;
	using element_type = T;
	using deleter_type = Deleter;

	UniquePtr() noexcept = default;

	explicit UniquePtr(pointer inP) noexcept requires (!oa::isVoidV<T>)
		: ptr_(inP)
	{}

	UniquePtr(pointer inP, const Deleter& inD) noexcept
		requires(oa::isNothrowCopyConstructibleV<Deleter>)
		: ptr_(inP), deleter_(inD)
	{}

	UniquePtr(pointer, const Deleter&)
		requires(!oa::isNothrowCopyConstructibleV<Deleter>) = delete;

	UniquePtr(pointer inP, Deleter&& inD) noexcept
		requires(oa::isNothrowMoveConstructibleV<Deleter>)
		: ptr_(inP), deleter_(oa::move(inD))
	{}

	UniquePtr(pointer, Deleter&&)
		requires(!oa::isNothrowMoveConstructibleV<Deleter>) = delete;

	UniquePtr(const UniquePtr&) = delete;
	UniquePtr& operator=(const UniquePtr&) = delete;

	UniquePtr(UniquePtr&& inO) noexcept
		requires(oa::isNothrowMoveConstructibleV<Deleter>)
		: ptr_(nullptr), deleter_(oa::move(inO.deleter_)) {
		// Do not relinquish ownership until the destination deleter exists.
		ptr_ = inO.release();
	}

	UniquePtr(UniquePtr&&) requires(!oa::isNothrowMoveConstructibleV<Deleter>) = delete;

	template<typename U, typename E,
		typename = oa::EnableIfT<!oa::isSameV<U, T> || !oa::isSameV<E, Deleter>>,
		typename = oa::EnableIfT<oa::isConvertibleV<U*, T*>>,
		typename = oa::EnableIfT<oa::isNothrowConstructibleV<Deleter, E&&>>>
	UniquePtr(UniquePtr<U, E>&& inO) noexcept
		: ptr_(nullptr), deleter_(oa::move(inO.getDeleter())) {
		ptr_ = inO.release();
	}

	UniquePtr& operator=(UniquePtr&& inO) noexcept
		requires(oa::isNothrowMoveAssignableV<Deleter>) {
		if (this != &inO) {
			reset();
			deleter_ = oa::move(inO.deleter_);
			ptr_ = inO.release();
		}
		return *this;
	}

	UniquePtr& operator=(UniquePtr&&)
		requires(!oa::isNothrowMoveAssignableV<Deleter>) = delete;

	~UniquePtr() noexcept { reset(); }

	[[nodiscard]] pointer get() const noexcept { return ptr_; }

	[[nodiscard]] Deleter& getDeleter() noexcept { return deleter_; }

	[[nodiscard]] const Deleter& getDeleter() const noexcept { return deleter_; }

	explicit operator bool() const noexcept { return ptr_ != nullptr; }

	[[nodiscard]] T& operator*() const requires (!oa::isVoidV<T>) { return *ptr_; }

	[[nodiscard]] pointer operator->() const noexcept requires (!oa::isVoidV<T>) { return ptr_; }

	[[nodiscard]] pointer release() noexcept {
		pointer p = ptr_;
		ptr_ = nullptr;
		return p;
	}

	void reset(pointer inP = pointer{}) noexcept {
		if (inP == ptr_) {
			return;
		}
		pointer old = ptr_;
		ptr_ = inP;
		if (old) {
			// DefaultDelete deliberately fails compilation when instantiated at an
			// incomplete destruction point. A custom deleter may legally own an
			// incomplete type and must never be silently skipped.
			getDeleter()(old);
		}
	}

	void swap(UniquePtr& inO) noexcept
		requires(oa::isNothrowSwappableV<Deleter>) {
		oa::swapValues(deleter_, inO.deleter_);
		oa::swapValues(ptr_, inO.ptr_);
	}

private:
	pointer ptr_{nullptr};
	Deleter deleter_{};
};

template<typename T, typename D>
[[nodiscard]] inline bool operator==(const UniquePtr<T, D>& inP, decltype(nullptr)) noexcept {
	return inP.get() == nullptr;
}
template<typename T, typename D>
[[nodiscard]] inline bool operator==(decltype(nullptr), const UniquePtr<T, D>& inP) noexcept {
	return inP.get() == nullptr;
}
template<typename T, typename D>
[[nodiscard]] inline bool operator!=(const UniquePtr<T, D>& inP, decltype(nullptr)) noexcept {
	return inP.get() != nullptr;
}
template<typename T, typename D>
[[nodiscard]] inline bool operator!=(decltype(nullptr), const UniquePtr<T, D>& inP) noexcept {
	return inP.get() != nullptr;
}

template<typename T, typename... Args>
[[nodiscard]] UniquePtr<T> makeUnique(Args&&... inArgs) {
	return UniquePtr<T>(new T(oa::forward<Args>(inArgs)...));
}

} // namespace oa
