#pragma once

// OA unique ownership — raw pointer + deleter; `makeUnique` uses `new T`
// (including `alignas` / over-aligned T).
//
// `reset` applies the deleter then nulls; the type is move-only and does not
// export a hosted standard-library ownership conversion.
//
// Incomplete-type support (PImpl): reset guards the deleter call with a
// compile-time completeness check so `~UniquePtr<T>` never instantiates the
// deleter in TUs where T is forward-declared. The out-of-line destructor of the
// owning class (defined where T is complete) handles deletion.

#include <oa/core/std/utility.h>
#include <oa/core/std/typeTraits.h>

namespace oa {

// True when T is a complete type at the point of instantiation.
template<typename T, typename = void>
inline constexpr bool IsCompleteType = false;
template<typename T>
inline constexpr bool IsCompleteType<T, oa::VoidT<decltype(sizeof(T))>> = true;

template<typename T>
struct DefaultDelete {
	constexpr DefaultDelete() noexcept = default;

	template<typename U, typename = oa::EnableIfT<oa::IsConvertibleV<U*, T*>>>
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

	explicit UniquePtr(pointer inP) noexcept requires (!oa::IsVoidV<T>) : ptr_(inP) {}

	UniquePtr(pointer inP, Deleter inD) noexcept : ptr_(inP), deleter_(oa::move(inD)) {}

	UniquePtr(const UniquePtr&) = delete;
	UniquePtr& operator=(const UniquePtr&) = delete;

	UniquePtr(UniquePtr&& inO) noexcept
		: ptr_(inO.ptr_), deleter_(oa::move(inO.deleter_)) {
		inO.ptr_ = nullptr;
	}

	template<typename U, typename E,
		typename = oa::EnableIfT<!oa::IsSameV<U, T> || !oa::IsSameV<E, Deleter>>,
		typename = oa::EnableIfT<oa::IsConvertibleV<U*, T*>>>
	UniquePtr(UniquePtr<U, E>&& inO) noexcept(oa::IsNothrowConstructibleV<Deleter, E&&>)
		: ptr_(inO.release()), deleter_(oa::move(inO.getDeleter())) {}

	UniquePtr& operator=(UniquePtr&& inO) noexcept(
		oa::IsNothrowMoveAssignableV<Deleter>) {
		if (this != &inO) {
			reset();
			ptr_ = inO.ptr_;
			deleter_ = oa::move(inO.deleter_);
			inO.ptr_ = nullptr;
		}
		return *this;
	}

	~UniquePtr() { reset(); }

	[[nodiscard]] pointer get() const noexcept { return ptr_; }

	[[nodiscard]] Deleter& getDeleter() noexcept { return deleter_; }

	[[nodiscard]] const Deleter& getDeleter() const noexcept { return deleter_; }

	explicit operator bool() const noexcept { return ptr_ != nullptr; }

	[[nodiscard]] T& operator*() const requires (!oa::IsVoidV<T>) { return *ptr_; }

	[[nodiscard]] pointer operator->() const noexcept requires (!oa::IsVoidV<T>) { return ptr_; }

	[[nodiscard]] pointer release() noexcept {
		pointer p = ptr_;
		ptr_ = nullptr;
		return p;
	}

	void reset(pointer inP = pointer{}) noexcept {
		pointer old = ptr_;
		ptr_ = inP;
		if (old) {
			// Guard with a compile-time completeness check: in TUs that only
			// forward-declare T (pimpl pattern), this branch is suppressed so
			// the deleter is never instantiated with an incomplete type.
			// The owning class's out-of-line dtor runs reset in a completing TU.
			if constexpr (IsCompleteType<T>) {
				getDeleter()(old);
			}
		}
	}

	void swap(UniquePtr& inO) noexcept {
		oa::swapValues(ptr_, inO.ptr_);
		oa::swapValues(deleter_, inO.deleter_);
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
