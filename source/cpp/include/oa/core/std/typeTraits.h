#pragma once

// Bounded OA type traits. Compiler predicates provide the properties that the
// language cannot express directly; no hosted C++ standard-library header is
// required.

#include <oa/core/std/utility.h>

namespace oa {

template<bool Value>
struct BoolConstant {
	static constexpr bool value = Value;
	using Type = BoolConstant;
	constexpr operator bool() const noexcept { return Value; }
};

using TrueType = BoolConstant<true>;
using FalseType = BoolConstant<false>;

template<typename...>
using VoidT = void;

template<typename T>
struct RemoveConst { using Type = T; };

template<typename T>
struct RemoveConst<const T> { using Type = T; };

template<typename T>
using RemoveConstT = typename RemoveConst<T>::Type;

template<typename T>
struct RemoveVolatile { using Type = T; };

template<typename T>
struct RemoveVolatile<volatile T> { using Type = T; };

template<typename T>
struct RemoveCv {
	using Type = typename RemoveVolatile<typename RemoveConst<T>::Type>::Type;
};

template<typename T>
using RemoveCvT = typename RemoveCv<T>::Type;

template<typename T>
using RemoveCvrefT = RemoveCvT<RemoveReferenceT<T>>;

template<bool Condition, typename T = void>
struct EnableIf {};

template<typename T>
struct EnableIf<true, T> { using Type = T; };

template<bool Condition, typename T = void>
using EnableIfT = typename EnableIf<Condition, T>::Type;

template<typename A, typename B>
inline constexpr bool IsSameV = false;

template<typename T>
inline constexpr bool IsSameV<T, T> = true;

template<typename T>
inline constexpr bool IsConstV = false;

template<typename T>
inline constexpr bool IsConstV<const T> = true;

template<typename T>
inline constexpr bool IsVoidV = IsSameV<RemoveCvT<T>, void>;

template<typename T>
struct IsPointerImpl { static constexpr bool Value = false; };

template<typename T>
struct IsPointerImpl<T*> { static constexpr bool Value = true; };

template<typename T>
inline constexpr bool IsPointerV = IsPointerImpl<RemoveCvT<T>>::Value;

template<typename T>
inline constexpr bool IsIntegralBaseV = false;

template<> inline constexpr bool IsIntegralBaseV<bool> = true;
template<> inline constexpr bool IsIntegralBaseV<char> = true;
template<> inline constexpr bool IsIntegralBaseV<signed char> = true;
template<> inline constexpr bool IsIntegralBaseV<unsigned char> = true;
template<> inline constexpr bool IsIntegralBaseV<wchar_t> = true;
template<> inline constexpr bool IsIntegralBaseV<char8_t> = true;
template<> inline constexpr bool IsIntegralBaseV<char16_t> = true;
template<> inline constexpr bool IsIntegralBaseV<char32_t> = true;
template<> inline constexpr bool IsIntegralBaseV<short> = true;
template<> inline constexpr bool IsIntegralBaseV<unsigned short> = true;
template<> inline constexpr bool IsIntegralBaseV<int> = true;
template<> inline constexpr bool IsIntegralBaseV<unsigned int> = true;
template<> inline constexpr bool IsIntegralBaseV<long> = true;
template<> inline constexpr bool IsIntegralBaseV<unsigned long> = true;
template<> inline constexpr bool IsIntegralBaseV<long long> = true;
template<> inline constexpr bool IsIntegralBaseV<unsigned long long> = true;

template<typename T>
inline constexpr bool IsIntegralV = IsIntegralBaseV<RemoveCvT<T>>;

template<typename T, bool = IsIntegralV<T>>
struct IsSignedImpl { static constexpr bool Value = false; };

template<typename T>
struct IsSignedImpl<T, true> {
	using ValueType = RemoveCvT<T>;
	static constexpr bool Value = ValueType(-1) < ValueType(0);
};

template<typename T>
inline constexpr bool IsSignedV = IsSignedImpl<T>::Value;

template<typename T>
inline constexpr bool IsFloatingPointBaseV = false;

template<> inline constexpr bool IsFloatingPointBaseV<float> = true;
template<> inline constexpr bool IsFloatingPointBaseV<double> = true;
template<> inline constexpr bool IsFloatingPointBaseV<long double> = true;

template<typename T>
inline constexpr bool IsFloatingPointV = IsFloatingPointBaseV<RemoveCvT<T>>;

template<typename T>
inline constexpr bool IsArithmeticV = IsIntegralV<T> or IsFloatingPointV<T>;

template<typename T>
inline constexpr bool IsEnumV = __is_enum(T);

template<typename From, typename To>
inline constexpr bool IsConvertibleV = __is_convertible(From, To);

template<typename T, typename... Args>
inline constexpr bool IsConstructibleV = __is_constructible(T, Args...);

template<typename T, typename... Args>
inline constexpr bool IsNothrowConstructibleV = __is_nothrow_constructible(T, Args...);

template<typename T>
inline constexpr bool IsCopyConstructibleV = __is_constructible(T, const T&);

template<typename T>
inline constexpr bool IsNothrowCopyConstructibleV =
	__is_nothrow_constructible(T, const T&);

template<typename T>
inline constexpr bool IsNothrowMoveConstructibleV = __is_nothrow_constructible(T, T&&);

template<typename T>
inline constexpr bool IsNothrowMoveAssignableV = __is_nothrow_assignable(T&, T&&);

template<typename T>
inline constexpr bool IsNothrowSwappableV =
	IsNothrowMoveConstructibleV<T> and IsNothrowMoveAssignableV<T>;

template<typename T>
inline constexpr bool IsTriviallyCopyableV = __is_trivially_copyable(T);

template<typename T>
inline constexpr bool IsTriviallyDestructibleV = __is_trivially_destructible(T);

template<typename U>
struct DecayNoRefImpl { using Type = RemoveCvT<U>; };

template<typename T, decltype(sizeof(0)) Count>
struct DecayNoRefImpl<T[Count]> { using Type = T*; };

template<typename T>
struct DecayNoRefImpl<T[]> { using Type = T*; };

template<typename R, typename... Args>
struct DecayNoRefImpl<R(Args...)> { using Type = R (*)(Args...); };

template<typename R, typename... Args>
struct DecayNoRefImpl<R(Args...) noexcept> { using Type = R (*)(Args...) noexcept; };

template<typename T>
struct Decay { using Type = typename DecayNoRefImpl<RemoveReferenceT<T>>::Type; };

template<typename T>
using DecayT = typename Decay<T>::Type;

} // namespace oa
