#pragma once

// Subset of `<type_traits>` for OaStd. Add traits here as call sites migrate off libstd.

#include <Oa/Core/Std/Utility.h>

#include <cstddef>

template<typename T>
struct OaStdRemoveCv {
	using Type = T;
};

template<typename T>
struct OaStdRemoveCv<const T> {
	using Type = typename OaStdRemoveCv<T>::Type;
};

template<typename T>
struct OaStdRemoveCv<volatile T> {
	using Type = typename OaStdRemoveCv<T>::Type;
};

template<typename T>
using OaStdRemoveCvT = typename OaStdRemoveCv<T>::Type;

template<typename U>
struct OaStdDecayNoRefImpl {
	using Type = OaStdRemoveCvT<U>;
};

template<typename T, std::size_t N>
struct OaStdDecayNoRefImpl<T[N]> {
	using Type = T*;
};

template<typename T>
struct OaStdDecayNoRefImpl<T[]> {
	using Type = T*;
};

template<typename R, typename... Args>
struct OaStdDecayNoRefImpl<R(Args...)> {
	using Type = R (*)(Args...);
};

template<typename R, typename... Args>
struct OaStdDecayNoRefImpl<R(Args...) noexcept> {
	using Type = R (*)(Args...) noexcept;
};

template<typename T>
struct OaStdDecay {
	using Type = typename OaStdDecayNoRefImpl<OaStdRemoveReferenceT<T>>::Type;
};

template<typename T>
using OaStdDecayT = typename OaStdDecay<T>::Type;

// Nothrow traits — thin `std` aliases; keeps `<type_traits>` at this boundary for OaStd headers.
#include <type_traits>

template<typename T>
inline constexpr bool OaStdIsNothrowMoveConstructibleV =
	std::is_nothrow_move_constructible_v<T>;

template<typename T>
inline constexpr bool OaStdIsNothrowMoveAssignableV =
	std::is_nothrow_move_assignable_v<T>;

template<typename T>
inline constexpr bool OaStdIsNothrowSwappableV = std::is_nothrow_swappable_v<T>;

template<typename T>
using OaStdRemoveCvrefT = std::remove_cvref_t<T>;

template<typename T>
using OaStdRemoveConstT = std::remove_const_t<T>;

template<bool Cond, typename T = void>
using OaStdEnableIfT = std::enable_if_t<Cond, T>;

template<typename T>
inline constexpr bool OaStdIsTriviallyCopyableV = std::is_trivially_copyable_v<T>;

template<typename T>
inline constexpr bool OaStdIsIntegralV = std::is_integral_v<T>;

template<typename T, typename U>
inline constexpr bool OaStdIsSameV = std::is_same_v<T, U>;

template<typename T>
inline constexpr bool OaStdIsCopyConstructibleV = std::is_copy_constructible_v<T>;

template<typename T>
inline constexpr bool OaStdIsTriviallyDestructibleV = std::is_trivially_destructible_v<T>;

template<typename T>
inline constexpr bool OaStdIsArithmeticV = std::is_arithmetic_v<T>;

template<typename T>
inline constexpr bool OaStdIsEnumV = std::is_enum_v<T>;

template<typename T>
inline constexpr bool OaStdIsPointerV = std::is_pointer_v<T>;
