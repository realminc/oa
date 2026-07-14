#pragma once

// OaStdMove / OaStdForward / OaStdSwap — no `<utility>`, no `<type_traits>`.

template<typename T>
struct OaStdRemoveReference {
	using Type = T;
};

template<typename T>
struct OaStdRemoveReference<T&> {
	using Type = T;
};

template<typename T>
struct OaStdRemoveReference<T&&> {
	using Type = T;
};

template<typename T>
using OaStdRemoveReferenceT = typename OaStdRemoveReference<T>::Type;

template<typename T>
inline constexpr bool OaStdIsLvalueReferenceV = false;

template<typename T>
inline constexpr bool OaStdIsLvalueReferenceV<T&> = true;

template<typename T>
[[nodiscard]] constexpr OaStdRemoveReferenceT<T>&& OaStdMove(T&& InArg) noexcept {
	return static_cast<OaStdRemoveReferenceT<T>&&>(InArg);
}

template<typename T>
[[nodiscard]] constexpr T&& OaStdForward(OaStdRemoveReferenceT<T>& InArg) noexcept {
	return static_cast<T&&>(InArg);
}

template<typename T>
[[nodiscard]] constexpr T&& OaStdForward(OaStdRemoveReferenceT<T>&& InArg) noexcept {
	static_assert(!OaStdIsLvalueReferenceV<T>, "OaStdForward: invalid rvalue overload");
	return static_cast<T&&>(InArg);
}

template<typename T>
constexpr void OaStdSwap(T& InA, T& InB) noexcept(
	noexcept(T(OaStdMove(InA))) && noexcept(InA = OaStdMove(InB)) && noexcept(InB = OaStdMove(InA))) {
	T tmp(OaStdMove(InA));
	InA = OaStdMove(InB);
	InB = OaStdMove(tmp);
}
