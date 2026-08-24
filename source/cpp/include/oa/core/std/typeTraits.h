#pragma once

// Subset of `<type_traits>` for OA standard library. Add traits here as call sites migrate off libstd.

#include <oa/core/std/utility.h>

#include <cstddef>
#include <type_traits>

namespace oa {

template<typename T>
struct RemoveCv {
	using Type = T;
};

template<typename T>
struct RemoveCv<const T> {
	using Type = typename RemoveCv<T>::Type;
};

template<typename T>
struct RemoveCv<volatile T> {
	using Type = typename RemoveCv<T>::Type;
};

template<typename T>
using RemoveCvT = typename RemoveCv<T>::Type;

template<typename U>
struct DecayNoRefImpl {
	using Type = RemoveCvT<U>;
};

template<typename T, std::size_t N>
struct DecayNoRefImpl<T[N]> {
	using Type = T*;
};

template<typename T>
struct DecayNoRefImpl<T[]> {
	using Type = T*;
};

template<typename R, typename... Args>
struct DecayNoRefImpl<R(Args...)> {
	using Type = R (*)(Args...);
};

template<typename R, typename... Args>
struct DecayNoRefImpl<R(Args...) noexcept> {
	using Type = R (*)(Args...) noexcept;
};

template<typename T>
struct Decay {
	using Type = typename DecayNoRefImpl<RemoveReferenceT<T>>::Type;
};

template<typename T>
using DecayT = typename Decay<T>::Type;

// Nothrow traits — thin `std` aliases kept at this boundary.

template<typename T>
inline constexpr bool IsNothrowMoveConstructibleV =
	std::is_nothrow_move_constructible_v<T>;

template<typename T>
inline constexpr bool IsNothrowMoveAssignableV =
	std::is_nothrow_move_assignable_v<T>;

template<typename T>
inline constexpr bool IsNothrowSwappableV = std::is_nothrow_swappable_v<T>;

template<typename T>
using RemoveCvrefT = std::remove_cvref_t<T>;

template<typename T>
using RemoveConstT = std::remove_const_t<T>;

template<bool Cond, typename T = void>
using EnableIfT = std::enable_if_t<Cond, T>;

template<typename T>
inline constexpr bool IsTriviallyCopyableV = std::is_trivially_copyable_v<T>;

template<typename T>
inline constexpr bool IsIntegralV = std::is_integral_v<T>;

template<typename T, typename U>
inline constexpr bool IsSameV = std::is_same_v<T, U>;

template<typename T>
inline constexpr bool IsCopyConstructibleV = std::is_copy_constructible_v<T>;

template<typename T>
inline constexpr bool IsTriviallyDestructibleV = std::is_trivially_destructible_v<T>;

template<typename T>
inline constexpr bool IsArithmeticV = std::is_arithmetic_v<T>;

template<typename T>
inline constexpr bool IsEnumV = std::is_enum_v<T>;

template<typename T>
inline constexpr bool IsPointerV = std::is_pointer_v<T>;

} // namespace oa
