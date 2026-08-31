#pragma once

// move / forward / swapValues — no `<utility>`, no `<type_traits>`.

namespace oa {

template<typename To, typename From>
[[nodiscard]] constexpr To bitCast(const From& inValue) noexcept {
	static_assert(sizeof(To) == sizeof(From),
		"oa::bitCast requires source and destination types of equal size");
	static_assert(__is_trivially_copyable(To) and __is_trivially_copyable(From),
		"oa::bitCast requires trivially copyable types");
	return __builtin_bit_cast(To, inValue);
}

template<typename T>
struct RemoveReference {
	using Type = T;
};

template<typename T>
struct RemoveReference<T&> {
	using Type = T;
};

template<typename T>
struct RemoveReference<T&&> {
	using Type = T;
};

template<typename T>
using RemoveReferenceT = typename RemoveReference<T>::Type;

template<typename T>
inline constexpr bool isLvalueReferenceV = false;

template<typename T>
inline constexpr bool isLvalueReferenceV<T&> = true;

template<typename T>
[[nodiscard]] constexpr RemoveReferenceT<T>&& move(T&& inArg) noexcept {
	return static_cast<RemoveReferenceT<T>&&>(inArg);
}

template<typename T>
[[nodiscard]] constexpr T&& forward(RemoveReferenceT<T>& inArg) noexcept {
	return static_cast<T&&>(inArg);
}

template<typename T>
[[nodiscard]] constexpr T&& forward(RemoveReferenceT<T>&& inArg) noexcept {
	static_assert(!isLvalueReferenceV<T>, "forward: invalid rvalue overload");
	return static_cast<T&&>(inArg);
}

template<typename T>
[[nodiscard]] RemoveReferenceT<T>&& declval() noexcept;

template<typename T, __SIZE_TYPE__ N>
[[nodiscard]] constexpr __SIZE_TYPE__ arraySize(const T (&)[N]) noexcept {
	return N;
}

// Deliberately not named `swap`: an unconstrained `oa::swap(T&, T&)` enters
// argument-dependent lookup for every OA type and collides with the Standard
// Library's own generic swap inside algorithms such as std::sort.
template<typename T>
constexpr void swapValues(T& inA, T& inB) noexcept(
	noexcept(T(move(inA))) && noexcept(inA = move(inB)) && noexcept(inB = move(inA))) {
	T tmp(move(inA));
	inA = move(inB);
	inB = move(tmp);
}

} // namespace oa
