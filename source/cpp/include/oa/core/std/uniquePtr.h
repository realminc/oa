#pragma once

// phase 2b OA standard library — raw pointer + deleter; `makeUnique` uses `new T` (including `alignas` / over-aligned T).
//
// `reset` applies deleter then nulls; move-only; `stdPtr() &&` transfers ownership to `std::unique_ptr`.
//
// Incomplete-type support (pimpl): reset guards the deleter call with a compile-time completeness
// check so `~UniquePtr<T>` never instantiates the deleter in TUs where T is forward-declared.
// The out-of-line destructor of the owning class (defined where T is complete) handles deletion.

#include <oa/core/std/utility.h>

#include <cstddef>
#include <memory>
#include <type_traits>

namespace oa {

// True when T is a complete type at the point of instantiation.
template<typename T, typename = void>
inline constexpr bool IsCompleteType = false;
template<typename T>
inline constexpr bool IsCompleteType<T, std::void_t<decltype(sizeof(T))>> = true;

template<typename T>
struct DefaultDelete {
	constexpr DefaultDelete() noexcept = default;

	template<typename U, typename = std::enable_if_t<std::is_convertible_v<U*, T*>>>
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

	explicit UniquePtr(pointer inP) noexcept requires (!std::is_void_v<T>) : ptr_(inP) {}

	UniquePtr(pointer inP, Deleter inD) noexcept : ptr_(inP), deleter_(oa::move(inD)) {}

	UniquePtr(std::unique_ptr<T, Deleter>&& inP) noexcept
		: deleter_(oa::move(inP.get_deleter())), ptr_(inP.release()) {}

	UniquePtr(const UniquePtr&) = delete;
	UniquePtr& operator=(const UniquePtr&) = delete;

	UniquePtr(UniquePtr&& inO) noexcept : ptr_(inO.ptr_), deleter_(oa::move(inO.deleter_)) { inO.ptr_ = nullptr; }

	template<typename U, typename E,
		typename = std::enable_if_t<!std::is_same_v<U, T> || !std::is_same_v<E, Deleter>>,
		typename = std::enable_if_t<std::is_convertible_v<U*, T*>>>
	UniquePtr(UniquePtr<U, E>&& inO) noexcept(std::is_nothrow_constructible_v<Deleter, E&&>)
		: ptr_(inO.release()), deleter_(oa::move(inO.getDeleter())) {}

	UniquePtr& operator=(UniquePtr&& inO) noexcept(
		std::is_nothrow_move_assignable_v<Deleter>) {
		if (this != &inO) {
			reset();
			ptr_ = inO.ptr_;
			deleter_ = oa::move(inO.deleter_);
			inO.ptr_ = nullptr;
		}
		return *this;
	}

	~UniquePtr() { reset(); }

	[[nodiscard]] std::unique_ptr<T, Deleter> stdPtr() && noexcept {
		return std::unique_ptr<T, Deleter>(release(), oa::move(getDeleter()));
	}

	[[nodiscard]] pointer get() const noexcept { return ptr_; }

	[[nodiscard]] Deleter& getDeleter() noexcept { return deleter_; }

	[[nodiscard]] const Deleter& getDeleter() const noexcept { return deleter_; }

	explicit operator bool() const noexcept { return ptr_ != nullptr; }

	[[nodiscard]] T& operator*() const requires (!std::is_void_v<T>) { return *ptr_; }

	[[nodiscard]] pointer operator->() const noexcept requires (!std::is_void_v<T>) { return ptr_; }

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
[[nodiscard]] inline bool operator==(const UniquePtr<T, D>& inP, std::nullptr_t) noexcept {
	return inP.get() == nullptr;
}
template<typename T, typename D>
[[nodiscard]] inline bool operator==(std::nullptr_t, const UniquePtr<T, D>& inP) noexcept {
	return inP.get() == nullptr;
}
template<typename T, typename D>
[[nodiscard]] inline bool operator!=(const UniquePtr<T, D>& inP, std::nullptr_t) noexcept {
	return inP.get() != nullptr;
}
template<typename T, typename D>
[[nodiscard]] inline bool operator!=(std::nullptr_t, const UniquePtr<T, D>& inP) noexcept {
	return inP.get() != nullptr;
}

template<typename T, typename... Args>
[[nodiscard]] UniquePtr<T> makeUnique(Args&&... inArgs) {
	return UniquePtr<T>(new T(oa::forward<Args>(inArgs)...));
}

} // namespace oa
