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
inline constexpr bool isSameV = false;

template<typename T>
inline constexpr bool isSameV<T, T> = true;

template<typename T>
inline constexpr bool isConstV = false;

template<typename T>
inline constexpr bool isConstV<const T> = true;

template<typename T>
inline constexpr bool isVoidV = isSameV<RemoveCvT<T>, void>;

template<typename T>
struct IsPointerImpl { static constexpr bool value = false; };

template<typename T>
struct IsPointerImpl<T*> { static constexpr bool value = true; };

template<typename T>
inline constexpr bool isPointerV = IsPointerImpl<RemoveCvT<T>>::value;

template<typename T>
inline constexpr bool isIntegralBaseV = false;

template<> inline constexpr bool isIntegralBaseV<bool> = true;
template<> inline constexpr bool isIntegralBaseV<char> = true;
template<> inline constexpr bool isIntegralBaseV<signed char> = true;
template<> inline constexpr bool isIntegralBaseV<unsigned char> = true;
template<> inline constexpr bool isIntegralBaseV<wchar_t> = true;
template<> inline constexpr bool isIntegralBaseV<char8_t> = true;
template<> inline constexpr bool isIntegralBaseV<char16_t> = true;
template<> inline constexpr bool isIntegralBaseV<char32_t> = true;
template<> inline constexpr bool isIntegralBaseV<short> = true;
template<> inline constexpr bool isIntegralBaseV<unsigned short> = true;
template<> inline constexpr bool isIntegralBaseV<int> = true;
template<> inline constexpr bool isIntegralBaseV<unsigned int> = true;
template<> inline constexpr bool isIntegralBaseV<long> = true;
template<> inline constexpr bool isIntegralBaseV<unsigned long> = true;
template<> inline constexpr bool isIntegralBaseV<long long> = true;
template<> inline constexpr bool isIntegralBaseV<unsigned long long> = true;

template<typename T>
inline constexpr bool isIntegralV = isIntegralBaseV<RemoveCvT<T>>;

template<typename T, bool = isIntegralV<T>>
struct IsSignedImpl { static constexpr bool value = false; };

template<typename T>
struct IsSignedImpl<T, true> {
	using ValueType = RemoveCvT<T>;
	static constexpr bool value = ValueType(-1) < ValueType(0);
};

template<typename T>
inline constexpr bool isSignedV = IsSignedImpl<T>::value;

template<typename T>
inline constexpr bool isFloatingPointBaseV = false;

template<> inline constexpr bool isFloatingPointBaseV<float> = true;
template<> inline constexpr bool isFloatingPointBaseV<double> = true;
template<> inline constexpr bool isFloatingPointBaseV<long double> = true;

template<typename T>
inline constexpr bool isFloatingPointV = isFloatingPointBaseV<RemoveCvT<T>>;

template<typename T>
inline constexpr bool isArithmeticV = isIntegralV<T> or isFloatingPointV<T>;

template<typename T>
inline constexpr bool isEnumV = __is_enum(T);

template<typename From, typename To>
inline constexpr bool isConvertibleV = __is_convertible(From, To);

template<typename T, typename... Args>
inline constexpr bool isConstructibleV = __is_constructible(T, Args...);

template<typename T, typename... Args>
inline constexpr bool isNothrowConstructibleV = __is_nothrow_constructible(T, Args...);

template<typename T>
inline constexpr bool isCopyConstructibleV = __is_constructible(T, const T&);

template<typename T>
inline constexpr bool isNothrowCopyConstructibleV =
	__is_nothrow_constructible(T, const T&);

template<typename T>
inline constexpr bool isNothrowMoveConstructibleV = __is_nothrow_constructible(T, T&&);

template<typename T>
inline constexpr bool isNothrowMoveAssignableV = __is_nothrow_assignable(T&, T&&);

template<typename T>
inline constexpr bool isNothrowSwappableV =
	isNothrowMoveConstructibleV<T> and isNothrowMoveAssignableV<T>;

template<typename T>
inline constexpr bool isTriviallyCopyableV = __is_trivially_copyable(T);

template<typename T>
inline constexpr bool isTriviallyDestructibleV = __is_trivially_destructible(T);

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
