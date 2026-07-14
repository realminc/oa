#pragma once

// OaStdIterTraits, OaStdDistance, OaStdAdvance, OaStdNext — no `<iterator>` for those.
// Random-access iterators use `last - first` and `it + n`; otherwise forward stepping.
// Primary **`OaStdIterTraits<It>`** (like `std::iterator_traits`) expects nested `difference_type` / `value_type` /
// `pointer` / `reference`;
// raw pointers use partial specializations.
//
// **`#include <iterator>`** (below) only for `OaStdReverseIterator`, `OaStdIterValueT` (via traits),
// and C++20 **`OaStdIsContiguousIteratorV`** (`std::contiguous_iterator`). Prefer **`OaStdDistance`** over
// `std::distance` at call sites.

#include <Oa/Core/Std/TypeTraits.h>

#include <cstddef>

template<typename It>
struct OaStdIterTraits {
	using difference_type = typename It::difference_type;
	using value_type = typename It::value_type;
	using pointer = typename It::pointer;
	using reference = typename It::reference;
};

template<typename T>
struct OaStdIterTraits<T*> {
	using difference_type = std::ptrdiff_t;
	using value_type = OaStdRemoveCvT<T>;
	using pointer = T*;
	using reference = T&;
};

template<typename T>
struct OaStdIterTraits<const T*> {
	using difference_type = std::ptrdiff_t;
	using value_type = OaStdRemoveCvT<T>;
	using pointer = const T*;
	using reference = const T&;
};

template<typename It>
inline constexpr bool OaStdIsRandomAccessIteratorV = requires(const It& InA, const It& InB,
	typename OaStdIterTraits<It>::difference_type InN) {
	InB - InA;
	InA + InN;
};

template<typename It>
inline constexpr bool OaStdIsBidirectionalIteratorV = requires(It InIt) {
	--InIt;
	++InIt;
	*InIt;
};

template<typename It>
[[nodiscard]] constexpr typename OaStdIterTraits<It>::difference_type OaStdDistance(
	It InFirst, It InLast) {
	using Diff = typename OaStdIterTraits<It>::difference_type;
	if constexpr (OaStdIsRandomAccessIteratorV<It>) {
		return InLast - InFirst;
	}
	Diff len = 0;
	for (; InFirst != InLast; ++InFirst) {
		++len;
	}
	return len;
}

template<typename It, typename Diff>
constexpr void OaStdAdvance(It& InIt, Diff InN) {
	if constexpr (OaStdIsRandomAccessIteratorV<It>) {
		InIt += InN;
	} else if constexpr (OaStdIsBidirectionalIteratorV<It>) {
		if (InN >= 0) {
			for (Diff k = 0; k < InN; ++k) {
				++InIt;
			}
		} else {
			for (Diff k = 0; k < -InN; ++k) {
				--InIt;
			}
		}
	} else {
		for (Diff k = 0; k < InN; ++k) {
			++InIt;
		}
	}
}

template<typename It>
[[nodiscard]] constexpr It OaStdNext(It InIt, typename OaStdIterTraits<It>::difference_type InN = 1) {
	OaStdAdvance(InIt, InN);
	return InIt;
}

#include <iterator>

template<typename It>
using OaStdReverseIterator = std::reverse_iterator<It>;

template<typename It>
using OaStdIterValueT = typename OaStdIterTraits<It>::value_type;

#if __cplusplus >= 202002L
template<typename It>
inline constexpr bool OaStdIsContiguousIteratorV = std::contiguous_iterator<It>;
#else
template<typename It>
inline constexpr bool OaStdIsContiguousIteratorV = false;
#endif
