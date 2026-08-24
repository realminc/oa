#pragma once

#include <limits>

namespace oa {

template<typename T>
struct Limits {
	[[nodiscard]] static constexpr T min()      noexcept { return std::numeric_limits<T>::min(); }
	[[nodiscard]] static constexpr T max()      noexcept { return std::numeric_limits<T>::max(); }
	[[nodiscard]] static constexpr T lowest()   noexcept { return std::numeric_limits<T>::lowest(); }
	[[nodiscard]] static constexpr T epsilon()  noexcept { return std::numeric_limits<T>::epsilon(); }
	[[nodiscard]] static constexpr T infinity() noexcept { return std::numeric_limits<T>::infinity(); }
	[[nodiscard]] static constexpr T quietNaN() noexcept { return std::numeric_limits<T>::quiet_NaN(); }

	static constexpr bool isSigned  = std::numeric_limits<T>::is_signed;
	static constexpr bool isInteger = std::numeric_limits<T>::is_integer;
	static constexpr bool hasNaN    = std::numeric_limits<T>::has_quiet_NaN;
	static constexpr int digits     = std::numeric_limits<T>::digits;
	static constexpr int digits10   = std::numeric_limits<T>::digits10;
};

} // namespace oa
