#pragma once

// Inline optional value with explicit lifetime and no standard-library optional
// interop. Empty checked access terminates through the always-on OA contract.

#include <oa/core/std/assert.h>
#include <oa/core/std/lifetime.h>
#include <oa/core/std/typeTraits.h>

namespace oa {

template<typename T>
class Optional {
public:
	Optional() noexcept = default;

	Optional(const T& inValue) : engaged_(true) { oa::constructAt(ptr_(), inValue); }

	Optional(T&& inValue) : engaged_(true) { oa::constructAt(ptr_(), oa::move(inValue)); }

	Optional(const Optional& inOther) : engaged_(inOther.engaged_) {
		if (engaged_) {
			oa::constructAt(ptr_(), *inOther.ptr_());
		}
	}

	Optional(Optional&& inOther) noexcept(oa::IsNothrowMoveConstructibleV<T>)
		: engaged_(inOther.engaged_) {
		if (engaged_) {
			oa::constructAt(ptr_(), oa::move(*inOther.ptr_()));
			inOther.engaged_ = false;
			oa::destroyAt(inOther.ptr_());
		}
	}

	Optional& operator=(const Optional& inOther) {
		if (this == &inOther) {
			return *this;
		}
		if (inOther.engaged_) {
			if (engaged_) {
				*ptr_() = *inOther.ptr_();
			} else {
				oa::constructAt(ptr_(), *inOther.ptr_());
				engaged_ = true;
			}
		} else {
			reset();
		}
		return *this;
	}

	Optional& operator=(Optional&& inOther) noexcept(
		oa::IsNothrowMoveAssignableV<T> && oa::IsNothrowMoveConstructibleV<T>) {
		if (this == &inOther) {
			return *this;
		}
		if (inOther.engaged_) {
			if (engaged_) {
				*ptr_() = oa::move(*inOther.ptr_());
			} else {
				oa::constructAt(ptr_(), oa::move(*inOther.ptr_()));
				engaged_ = true;
			}
			inOther.engaged_ = false;
			oa::destroyAt(inOther.ptr_());
		} else {
			reset();
		}
		return *this;
	}

	~Optional() { reset(); }

	[[nodiscard]] bool hasValue() const noexcept { return engaged_; }

	explicit operator bool() const noexcept { return engaged_; }

	[[nodiscard]] T* get() noexcept { return engaged_ ? ptr_() : nullptr; }

	[[nodiscard]] const T* get() const noexcept { return engaged_ ? ptr_() : nullptr; }

	[[nodiscard]] T& value() noexcept {
		OA_REQUIRE(engaged_);
		return *ptr_();
	}

	[[nodiscard]] const T& value() const noexcept {
		OA_REQUIRE(engaged_);
		return *ptr_();
	}

	[[nodiscard]] T& operator*() noexcept { return value(); }
	[[nodiscard]] const T& operator*() const noexcept { return value(); }
	[[nodiscard]] T* operator->() noexcept {
		OA_REQUIRE(engaged_);
		return ptr_();
	}
	[[nodiscard]] const T* operator->() const noexcept {
		OA_REQUIRE(engaged_);
		return ptr_();
	}

	template<typename U>
	[[nodiscard]] T valueOr(U&& inDefault) const& {
		return engaged_ ? *ptr_() : static_cast<T>(oa::forward<U>(inDefault));
	}

	template<typename U>
	[[nodiscard]] T valueOr(U&& inDefault) && {
		return engaged_ ? oa::move(*ptr_()) : static_cast<T>(oa::forward<U>(inDefault));
	}

	template<typename... Args>
	T& emplace(Args&&... inArgs) {
		reset();
		oa::constructAt(ptr_(), oa::forward<Args>(inArgs)...);
		engaged_ = true;
		return *ptr_();
	}

	void reset() noexcept {
		if (engaged_) {
			T* retired = ptr_();
			engaged_ = false;
			oa::destroyAt(retired);
		}
	}

private:
	alignas(T) unsigned char storage_[sizeof(T)]{};
	bool engaged_{false};

	T* ptr_() noexcept { return reinterpret_cast<T*>(storage_); }

	const T* ptr_() const noexcept { return reinterpret_cast<const T*>(storage_); }
};

} // namespace oa
