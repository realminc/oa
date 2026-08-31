#pragma once

// Bounded hash and equality primitives for OA's admitted scalar, pointer, and
// text keys. Hash values are internal container routing values, not serialized
// identities or cryptographic digests.

#include <oa/core/std/string.h>
#include <oa/core/std/typeTraits.h>

namespace oa {

[[nodiscard]] constexpr oa::Usize mixKeyHash(oa::Usize inValue) noexcept {
	if constexpr (sizeof(oa::Usize) == 8) {
		inValue ^= inValue >> 30U;
		inValue *= static_cast<oa::Usize>(0xbf58476d1ce4e5b9ULL);
		inValue ^= inValue >> 27U;
		inValue *= static_cast<oa::Usize>(0x94d049bb133111ebULL);
		inValue ^= inValue >> 31U;
	} else {
		inValue ^= inValue >> 16U;
		inValue *= static_cast<oa::Usize>(0x7feb352dU);
		inValue ^= inValue >> 15U;
		inValue *= static_cast<oa::Usize>(0x846ca68bU);
		inValue ^= inValue >> 16U;
	}
	return inValue;
}

[[nodiscard]] inline oa::Usize hashTextKey(oa::StringView inText) noexcept {
	const oa::Usize offset = sizeof(oa::Usize) == 8
		? static_cast<oa::Usize>(14695981039346656037ULL)
		: static_cast<oa::Usize>(2166136261U);
	const oa::Usize prime = sizeof(oa::Usize) == 8
		? static_cast<oa::Usize>(1099511628211ULL)
		: static_cast<oa::Usize>(16777619U);
	oa::Usize value = offset;
	for (oa::Usize index = 0; index < inText.size(); ++index) {
		value ^= static_cast<oa::Usize>(static_cast<unsigned char>(inText[index]));
		value *= prime;
	}
	return value;
}

template<typename T>
struct KeyHash {
	[[nodiscard]] oa::Usize operator()(T inValue) const noexcept {
		static_assert(
			oa::isIntegralV<T> or oa::isEnumV<T> or oa::isPointerV<T>,
			"oa::KeyHash supports only admitted scalar, enum, pointer, and text keys"
		);
		if constexpr (oa::isPointerV<T>) {
			return oa::mixKeyHash(reinterpret_cast<oa::Usize>(inValue));
		} else {
			// Open-addressed tables preserve locality for sequential scalar keys
			// when their native representation is retained. This is a routing
			// value, not a collision-resistant or serialized digest.
			return static_cast<oa::Usize>(inValue);
		}
	}
};

template<>
struct KeyHash<oa::StringView> {
	[[nodiscard]] oa::Usize operator()(oa::StringView inValue) const noexcept {
		return oa::hashTextKey(inValue);
	}
};

template<>
struct KeyHash<oa::String> {
	[[nodiscard]] oa::Usize operator()(const oa::String& inValue) const noexcept {
		return oa::hashTextKey(inValue.view());
	}
};

template<typename T>
struct KeyEqual {
	[[nodiscard]] constexpr bool operator()(const T& inA, const T& inB) const
		noexcept(noexcept(inA == inB)) {
		return inA == inB;
	}
};

} // namespace oa
