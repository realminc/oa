#pragma once

// phase 2b OA standard library — `alignas(T)` untyped storage + engaged flag; `stdOptional()` copies/moves to `std::optional`.
//
// Lifecycle: placement new on construct/assign; explicit destroy when clearing or replacing.
// Interop: `std::nullopt`, `std::optional<T>` in/out; no `std::optional` in the hot storage path.

#include <oa/core/std/typeTraits.h>

#include <new>
#include <optional>

namespace oa {

template<typename T>
class Optional {
public:
	Optional() noexcept = default;

	Optional([[maybe_unused]] std::nullopt_t inNullopt) noexcept {}

	Optional(const T& inValue) : engaged_(true) { new (storage_) T(inValue); }

	Optional(T&& inValue) : engaged_(true) { new (storage_) T(oa::move(inValue)); }

	explicit Optional(const std::optional<T>& inOther) {
		if (inOther.has_value()) {
			new (storage_) T(*inOther);
			engaged_ = true;
		}
	}

	Optional(const Optional& inOther) : engaged_(inOther.engaged_) {
		if (engaged_) {
			new (storage_) T(*inOther.ptr_());
		}
	}

	Optional(Optional&& inOther) noexcept(oa::IsNothrowMoveConstructibleV<T>)
		: engaged_(inOther.engaged_) {
		if (engaged_) {
			new (storage_) T(oa::move(*inOther.ptr_()));
			inOther.ptr_()->~T();
			inOther.engaged_ = false;
		}
	}

	Optional& operator=([[maybe_unused]] std::nullopt_t inNullopt) noexcept {
		reset();
		return *this;
	}

	Optional& operator=(const Optional& inOther) {
		if (this == &inOther) {
			return *this;
		}
		if (inOther.engaged_) {
			if (engaged_) {
				*ptr_() = *inOther.ptr_();
			} else {
				new (storage_) T(*inOther.ptr_());
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
				new (storage_) T(oa::move(*inOther.ptr_()));
				engaged_ = true;
			}
			inOther.ptr_()->~T();
			inOther.engaged_ = false;
		} else {
			reset();
		}
		return *this;
	}

	~Optional() { reset(); }

	[[nodiscard]] bool hasValue() const noexcept { return engaged_; }

	explicit operator bool() const noexcept { return engaged_; }

	[[nodiscard]] T& value() {
		if (!engaged_) {
			throw std::bad_optional_access();
		}
		return *ptr_();
	}

	[[nodiscard]] const T& value() const {
		if (!engaged_) {
			throw std::bad_optional_access();
		}
		return *ptr_();
	}

	[[nodiscard]] T& operator*() { return value(); }
	[[nodiscard]] const T& operator*() const { return value(); }
	[[nodiscard]] T* operator->() { return ptr_(); }
	[[nodiscard]] const T* operator->() const { return ptr_(); }

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
		new (storage_) T(oa::forward<Args>(inArgs)...);
		engaged_ = true;
		return *ptr_();
	}

	void reset() noexcept {
		if (engaged_) {
			ptr_()->~T();
			engaged_ = false;
		}
	}

	[[nodiscard]] std::optional<T> stdOptional() const& {
		if (!engaged_) {
			return std::nullopt;
		}
		return *ptr_();
	}

	[[nodiscard]] std::optional<T> stdOptional() && {
		if (!engaged_) {
			return std::nullopt;
		}
		std::optional<T> out(oa::move(*ptr_()));
		ptr_()->~T();
		engaged_ = false;
		return out;
	}

private:
	alignas(T) unsigned char storage_[sizeof(T)]{};
	bool engaged_{false};

	T* ptr_() noexcept { return reinterpret_cast<T*>(storage_); }

	const T* ptr_() const noexcept { return reinterpret_cast<const T*>(storage_); }
};

} // namespace oa
