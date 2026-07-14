#pragma once

// OaStdLimits<T> — the std::numeric_limits replacement, PascalCase.
//
// HONEST NOTE: numeric limits are compile-time constants baked into the type
// system; there is nothing to clean-room. This wraps <limits> purely for naming
// consistency (like OaStdLimits<float>::Max() instead of
// std::numeric_limits<float>::max()).

#include <limits>

template<typename T>
struct OaStdLimits {
	[[nodiscard]] static constexpr T Min()      noexcept { return std::numeric_limits<T>::min(); }
	[[nodiscard]] static constexpr T Max()      noexcept { return std::numeric_limits<T>::max(); }
	[[nodiscard]] static constexpr T Lowest()   noexcept { return std::numeric_limits<T>::lowest(); }
	[[nodiscard]] static constexpr T Epsilon()  noexcept { return std::numeric_limits<T>::epsilon(); }
	[[nodiscard]] static constexpr T Infinity() noexcept { return std::numeric_limits<T>::infinity(); }
	[[nodiscard]] static constexpr T QuietNaN() noexcept { return std::numeric_limits<T>::quiet_NaN(); }

	static constexpr bool IsSigned  = std::numeric_limits<T>::is_signed;
	static constexpr bool IsInteger = std::numeric_limits<T>::is_integer;
	static constexpr bool HasNaN    = std::numeric_limits<T>::has_quiet_NaN;
	static constexpr int  Digits    = std::numeric_limits<T>::digits;
	static constexpr int  Digits10  = std::numeric_limits<T>::digits10;
};

// Short public alias (canonical name stays OaStdLimits; see OaStd.md naming).
template<typename T>
using OaLimits = OaStdLimits<T>;
