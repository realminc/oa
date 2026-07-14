#pragma once

// Phase 2b OaStd — raw pointer + deleter; `OaStdMakeUnique` uses `new T` (including `alignas` / over-aligned T).
//
// `Reset` applies deleter then nulls; move-only; `StdPtr() &&` transfers ownership to `std::unique_ptr`.
//
// Incomplete-type support (pimpl): Reset guards the deleter call with a compile-time completeness
// check so `~OaStdUniquePtr<T>` never instantiates the deleter in TUs where T is forward-declared.
// The out-of-line destructor of the owning class (defined where T is complete) handles deletion.

#include <Oa/Core/Std/Utility.h>

#include <cstddef>
#include <memory>
#include <type_traits>

// True when T is a complete type at the point of instantiation.
template<typename T, typename = void>
inline constexpr bool OaIsCompleteType = false;
template<typename T>
inline constexpr bool OaIsCompleteType<T, std::void_t<decltype(sizeof(T))>> = true;

template<typename T>
struct OaStdDefaultDelete {
	constexpr OaStdDefaultDelete() noexcept = default;

	template<typename U, typename = std::enable_if_t<std::is_convertible_v<U*, T*>>>
	OaStdDefaultDelete(const OaStdDefaultDelete<U>&) noexcept {}

	void operator()(T* InP) const noexcept {
		static_assert(sizeof(T) > 0, "OaStdDefaultDelete incomplete type");
		delete InP;
	}
};

template<typename T, typename Deleter = OaStdDefaultDelete<T>>
class OaStdUniquePtr {
public:
	using pointer = T*;
	using element_type = T;
	using deleter_type = Deleter;

	OaStdUniquePtr() noexcept = default;

	explicit OaStdUniquePtr(pointer InP) noexcept requires (!std::is_void_v<T>) : Ptr_(InP) {}

	OaStdUniquePtr(pointer InP, Deleter InD) noexcept : Ptr_(InP), Del_(OaStdMove(InD)) {}

	OaStdUniquePtr(std::unique_ptr<T, Deleter>&& InP) noexcept
		: Del_(OaStdMove(InP.get_deleter())), Ptr_(InP.release()) {}

	OaStdUniquePtr(const OaStdUniquePtr&) = delete;
	OaStdUniquePtr& operator=(const OaStdUniquePtr&) = delete;

	OaStdUniquePtr(OaStdUniquePtr&& InO) noexcept : Ptr_(InO.Ptr_), Del_(OaStdMove(InO.Del_)) { InO.Ptr_ = nullptr; }

	template<typename U, typename E,
		typename = std::enable_if_t<!std::is_same_v<U, T> || !std::is_same_v<E, Deleter>>,
		typename = std::enable_if_t<std::is_convertible_v<U*, T*>>>
	OaStdUniquePtr(OaStdUniquePtr<U, E>&& InO) noexcept(std::is_nothrow_constructible_v<Deleter, E&&>)
		: Ptr_(InO.Release()), Del_(OaStdMove(InO.GetDeleter())) {}

	OaStdUniquePtr& operator=(OaStdUniquePtr&& InO) noexcept(
		std::is_nothrow_move_assignable_v<Deleter>) {
		if (this != &InO) {
			Reset();
			Ptr_ = InO.Ptr_;
			Del_ = OaStdMove(InO.Del_);
			InO.Ptr_ = nullptr;
		}
		return *this;
	}

	~OaStdUniquePtr() { Reset(); }

	[[nodiscard]] std::unique_ptr<T, Deleter> StdPtr() && noexcept {
		return std::unique_ptr<T, Deleter>(Release(), OaStdMove(GetDeleter()));
	}

	[[nodiscard]] pointer Get() const noexcept { return Ptr_; }
	[[nodiscard]] pointer get() const noexcept { return Ptr_; }

	[[nodiscard]] Deleter& GetDeleter() noexcept { return Del_; }

	[[nodiscard]] const Deleter& GetDeleter() const noexcept { return Del_; }

	explicit operator bool() const noexcept { return Ptr_ != nullptr; }

	[[nodiscard]] T& operator*() const requires (!std::is_void_v<T>) { return *Ptr_; }

	[[nodiscard]] pointer operator->() const noexcept requires (!std::is_void_v<T>) { return Ptr_; }

	[[nodiscard]] pointer Release() noexcept {
		pointer p = Ptr_;
		Ptr_ = nullptr;
		return p;
	}

	void Reset(pointer InP = pointer{}) noexcept {
		pointer old = Ptr_;
		Ptr_ = InP;
		if (old) {
			// Guard with a compile-time completeness check: in TUs that only
			// forward-declare T (pimpl pattern), this branch is suppressed so
			// the deleter is never instantiated with an incomplete type.
			// The owning class's out-of-line dtor runs Reset in a completing TU.
			if constexpr (OaIsCompleteType<T>) {
				GetDeleter()(old);
			}
		}
	}

	void reset(pointer InP = pointer{}) noexcept { Reset(InP); }

	void Swap(OaStdUniquePtr& InO) noexcept {
		OaStdSwap(Ptr_, InO.Ptr_);
		OaStdSwap(Del_, InO.Del_);
	}

private:
	pointer Ptr_{nullptr};
	Deleter Del_{};
};

template<typename T, typename D>
[[nodiscard]] inline bool operator==(const OaStdUniquePtr<T, D>& InP, std::nullptr_t) noexcept {
	return InP.Get() == nullptr;
}
template<typename T, typename D>
[[nodiscard]] inline bool operator==(std::nullptr_t, const OaStdUniquePtr<T, D>& InP) noexcept {
	return InP.Get() == nullptr;
}
template<typename T, typename D>
[[nodiscard]] inline bool operator!=(const OaStdUniquePtr<T, D>& InP, std::nullptr_t) noexcept {
	return InP.Get() != nullptr;
}
template<typename T, typename D>
[[nodiscard]] inline bool operator!=(std::nullptr_t, const OaStdUniquePtr<T, D>& InP) noexcept {
	return InP.Get() != nullptr;
}

template<typename T, typename... Args>
[[nodiscard]] OaStdUniquePtr<T> OaStdMakeUnique(Args&&... InArgs) {
	return OaStdUniquePtr<T>(new T(OaStdForward<Args>(InArgs)...));
}
