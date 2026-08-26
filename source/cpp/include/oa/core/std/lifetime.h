#pragma once

// OA object-lifetime primitives for caller-owned storage. The custom placement
// tag keeps construction in the C++ language layer without importing the
// hosted <new> header or replacing the ordinary allocation operators.

#include <oa/core/std/utility.h>

namespace oa {

struct PlacementTag final {};
inline constexpr PlacementTag Placement{};

} // namespace oa

[[nodiscard]] inline void* operator new(
	decltype(sizeof(0)),
	void* inStorage,
	oa::PlacementTag
) noexcept {
	return inStorage;
}

inline void operator delete(
	void*,
	void*,
	oa::PlacementTag
) noexcept {}

namespace oa {

template<typename T, typename... Args>
T* constructAt(T* inStorage, Args&&... inArgs) noexcept(
	noexcept(T(oa::forward<Args>(inArgs)...))
) {
	return ::new (static_cast<void*>(inStorage), oa::Placement)
		T(oa::forward<Args>(inArgs)...);
}

template<typename T>
void destroyAt(T* inObject) noexcept(noexcept(inObject->~T())) {
	inObject->~T();
}

template<typename T>
[[nodiscard]] T* launder(T* inObject) noexcept {
	return __builtin_launder(inObject);
}

} // namespace oa
