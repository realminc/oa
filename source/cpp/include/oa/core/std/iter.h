#pragma once

// IterTraits, distance, advance, next — no `<iterator>` for those.
// Random-access iterators use `last - first` and `it + n`; otherwise forward stepping.
// Primary **`IterTraits<It>`** (like `std::iterator_traits`) expects nested `difference_type` / `value_type` /
// `pointer` / `reference`;
// raw pointers use partial specializations.
//
// **`#include <iterator>`** (below) only for `ReverseIterator`, `IterValueT` (via traits),
// and C++20 **`IsContiguousIteratorV`** (`std::contiguous_iterator`). Prefer **`distance`** over
// `std::distance` at call sites.

#include <oa/core/std/typeTraits.h>

#include <cstddef>
#include <iterator>

namespace oa {

template<typename It>
struct IterTraits {
	using difference_type = typename It::difference_type;
	using value_type = typename It::value_type;
	using pointer = typename It::pointer;
	using reference = typename It::reference;
};

template<typename T>
struct IterTraits<T*> {
	using difference_type = std::ptrdiff_t;
	using value_type = oa::RemoveCvT<T>;
	using pointer = T*;
	using reference = T&;
};

template<typename T>
struct IterTraits<const T*> {
	using difference_type = std::ptrdiff_t;
	using value_type = oa::RemoveCvT<T>;
	using pointer = const T*;
	using reference = const T&;
};

template<typename It>
inline constexpr bool IsRandomAccessIteratorV = requires(const It& inA, const It& inB,
	typename IterTraits<It>::difference_type inN) {
	inB - inA;
	inA + inN;
};

template<typename It>
inline constexpr bool IsBidirectionalIteratorV = requires(It inIt) {
	--inIt;
	++inIt;
	*inIt;
};

template<typename It>
[[nodiscard]] constexpr typename IterTraits<It>::difference_type distance(
	It inFirst, It inLast) {
	using Diff = typename IterTraits<It>::difference_type;
	if constexpr (IsRandomAccessIteratorV<It>) {
		return inLast - inFirst;
	}
	Diff len = 0;
	for (; inFirst != inLast; ++inFirst) {
		++len;
	}
	return len;
}

template<typename It, typename Diff>
constexpr void advance(It& inIt, Diff inN) {
	if constexpr (IsRandomAccessIteratorV<It>) {
		inIt += inN;
	} else if constexpr (IsBidirectionalIteratorV<It>) {
		if (inN >= 0) {
			for (Diff k = 0; k < inN; ++k) {
				++inIt;
			}
		} else {
			for (Diff k = 0; k < -inN; ++k) {
				--inIt;
			}
		}
	} else {
		for (Diff k = 0; k < inN; ++k) {
			++inIt;
		}
	}
}

template<typename It>
[[nodiscard]] constexpr It next(It inIt, typename IterTraits<It>::difference_type inN = 1) {
	advance(inIt, inN);
	return inIt;
}

template<typename It>
using ReverseIterator = std::reverse_iterator<It>;

template<typename It>
using IterValueT = typename IterTraits<It>::value_type;

#if __cplusplus >= 202002L
template<typename It>
inline constexpr bool IsContiguousIteratorV = std::contiguous_iterator<It>;
#else
template<typename It>
inline constexpr bool IsContiguousIteratorV = false;
#endif

} // namespace oa
