#pragma once

// Phase 2b OaStd — `alignas(T)` untyped storage + engaged flag; `StdOptional()` copies/moves to `std::optional`.
//
// Lifecycle: placement new on construct/assign; explicit destroy when clearing or replacing.
// Interop: `std::nullopt`, `std::optional<T>` in/out; no `std::optional` in the hot storage path.

#include <Oa/Core/Std/TypeTraits.h>

#include <new>
#include <optional>

template<typename T>
class OaStdOptional {
public:
	OaStdOptional() noexcept = default;

	OaStdOptional([[maybe_unused]] std::nullopt_t InNullopt) noexcept {}

	OaStdOptional(const T& InVal) : Engaged_(true) { new (Storage_) T(InVal); }

	OaStdOptional(T&& InVal) : Engaged_(true) { new (Storage_) T(OaStdMove(InVal)); }

	explicit OaStdOptional(const std::optional<T>& InO) {
		if (InO.has_value()) {
			new (Storage_) T(*InO);
			Engaged_ = true;
		}
	}

	OaStdOptional(const OaStdOptional& InO) : Engaged_(InO.Engaged_) {
		if (Engaged_) {
			new (Storage_) T(*InO.Ptr());
		}
	}

	OaStdOptional(OaStdOptional&& InO) noexcept(OaStdIsNothrowMoveConstructibleV<T>)
		: Engaged_(InO.Engaged_) {
		if (Engaged_) {
			new (Storage_) T(OaStdMove(*InO.Ptr()));
			InO.Ptr()->~T();
			InO.Engaged_ = false;
		}
	}

	OaStdOptional& operator=([[maybe_unused]] std::nullopt_t InNullopt) noexcept {
		Reset();
		return *this;
	}

	OaStdOptional& operator=(const OaStdOptional& InO) {
		if (this == &InO) {
			return *this;
		}
		if (InO.Engaged_) {
			if (Engaged_) {
				*Ptr() = *InO.Ptr();
			} else {
				new (Storage_) T(*InO.Ptr());
				Engaged_ = true;
			}
		} else {
			Reset();
		}
		return *this;
	}

	OaStdOptional& operator=(OaStdOptional&& InO) noexcept(
		OaStdIsNothrowMoveAssignableV<T> && OaStdIsNothrowMoveConstructibleV<T>) {
		if (this == &InO) {
			return *this;
		}
		if (InO.Engaged_) {
			if (Engaged_) {
				*Ptr() = OaStdMove(*InO.Ptr());
			} else {
				new (Storage_) T(OaStdMove(*InO.Ptr()));
				Engaged_ = true;
			}
			InO.Ptr()->~T();
			InO.Engaged_ = false;
		} else {
			Reset();
		}
		return *this;
	}

	~OaStdOptional() { Reset(); }

	[[nodiscard]] bool HasValue() const noexcept { return Engaged_; }
	[[nodiscard]] bool has_value() const noexcept { return Engaged_; }

	explicit operator bool() const noexcept { return Engaged_; }

	[[nodiscard]] T& Value() {
		if (!Engaged_) {
			throw std::bad_optional_access();
		}
		return *Ptr();
	}

	[[nodiscard]] const T& Value() const {
		if (!Engaged_) {
			throw std::bad_optional_access();
		}
		return *Ptr();
	}

	[[nodiscard]] T& value() { return Value(); }
	[[nodiscard]] const T& value() const { return Value(); }

	[[nodiscard]] T& operator*() { return Value(); }
	[[nodiscard]] const T& operator*() const { return Value(); }
	[[nodiscard]] T* operator->() { return Ptr(); }
	[[nodiscard]] const T* operator->() const { return Ptr(); }

	template<typename U>
	[[nodiscard]] T ValueOr(U&& InDefault) const& {
		return Engaged_ ? *Ptr() : static_cast<T>(OaStdForward<U>(InDefault));
	}

	template<typename U>
	[[nodiscard]] T ValueOr(U&& InDefault) && {
		return Engaged_ ? OaStdMove(*Ptr()) : static_cast<T>(OaStdForward<U>(InDefault));
	}

	template<typename... Args>
	T& Emplace(Args&&... InArgs) {
		Reset();
		new (Storage_) T(OaStdForward<Args>(InArgs)...);
		Engaged_ = true;
		return *Ptr();
	}

	void Reset() noexcept {
		if (Engaged_) {
			Ptr()->~T();
			Engaged_ = false;
		}
	}

	[[nodiscard]] std::optional<T> StdOptional() const& {
		if (!Engaged_) {
			return std::nullopt;
		}
		return *Ptr();
	}

	[[nodiscard]] std::optional<T> StdOptional() && {
		if (!Engaged_) {
			return std::nullopt;
		}
		std::optional<T> out(OaStdMove(*Ptr()));
		Ptr()->~T();
		Engaged_ = false;
		return out;
	}

private:
	alignas(T) unsigned char Storage_[sizeof(T)]{};
	bool Engaged_{false};

	T* Ptr() noexcept { return reinterpret_cast<T*>(Storage_); }

	const T* Ptr() const noexcept { return reinterpret_cast<const T*>(Storage_); }
};
