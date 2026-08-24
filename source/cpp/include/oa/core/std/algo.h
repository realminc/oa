#pragma once

// OA iterator algorithms with `oa::Span` overloads.
//
// sort: in-place heapsort, **O(n log n)**, unstable. Bounds APIs match `std` for the given comparator.
// `lower_bound` / `upper_bound` / `binary_search` use **`distance` / `advance` / `next`** (`Iter.h`).

#include <oa/core/std/iter.h>
#include <oa/core/std/span.h>
#include <oa/core/std/utility.h>

namespace oa {

struct DefaultLess {
	template<typename A, typename B>
	constexpr bool operator()(const A& inA, const B& inB) const noexcept(noexcept(inA < inB)) {
		return inA < inB;
	}
};

struct AlgoHeapSort {
	template<typename RandIt, typename Cmp>
	static void siftDown(RandIt inFirst,
		typename IterTraits<RandIt>::difference_type inRoot,
		typename IterTraits<RandIt>::difference_type inEnd,
		Cmp inCmp) {
		using Diff = typename IterTraits<RandIt>::difference_type;
		for (;;) {
			Diff child = inRoot * 2 + 1;
			if (child >= inEnd) {
				break;
			}
			const Diff right = child + 1;
			if (right < inEnd && inCmp(*(inFirst + child), *(inFirst + right))) {
				child = right;
			}
			if (!inCmp(*(inFirst + inRoot), *(inFirst + child))) {
				break;
			}
			oa::swapValues(*(inFirst + inRoot), *(inFirst + child));
			inRoot = child;
		}
	}

	template<typename RandIt, typename Cmp>
	static void sort(RandIt inFirst, RandIt inLast, Cmp inCmp) {
		using Diff = typename IterTraits<RandIt>::difference_type;
		const Diff n = inLast - inFirst;
		if (n < 2) {
			return;
		}
		for (Diff idx = n / 2 - 1; idx >= 0; --idx) {
			siftDown(inFirst, idx, n, inCmp);
		}
		for (Diff heapEnd = n; heapEnd > 1; --heapEnd) {
			oa::swapValues(*inFirst, *(inFirst + (heapEnd - 1)));
			siftDown(inFirst, 0, heapEnd - 1, inCmp);
		}
	}
};

template<typename It, typename T, typename Cmp>
[[nodiscard]] It lowerBound(It inFirst, It inLast, const T& inVal, Cmp inCmp) {
	using Diff = typename IterTraits<It>::difference_type;
	Diff len = distance(inFirst, inLast);
	while (len > 0) {
		const Diff half = len / 2;
		It mid = inFirst;
		advance(mid, half);
		if (inCmp(*mid, inVal)) {
			inFirst = next(mid);
			len -= half + 1;
		} else {
			len = half;
		}
	}
	return inFirst;
}

template<typename It, typename T>
[[nodiscard]] It lowerBound(It inFirst, It inLast, const T& inVal) {
	return lowerBound(inFirst, inLast, inVal, DefaultLess{});
}

template<typename It, typename T, typename Cmp>
[[nodiscard]] It upperBound(It inFirst, It inLast, const T& inVal, Cmp inCmp) {
	using Diff = typename IterTraits<It>::difference_type;
	Diff len = distance(inFirst, inLast);
	while (len > 0) {
		const Diff half = len / 2;
		It mid = inFirst;
		advance(mid, half);
		if (inCmp(inVal, *mid)) {
			len = half;
		} else {
			inFirst = next(mid);
			len -= half + 1;
		}
	}
	return inFirst;
}

template<typename It, typename T>
[[nodiscard]] It upperBound(It inFirst, It inLast, const T& inVal) {
	return upperBound(inFirst, inLast, inVal, DefaultLess{});
}

template<typename It, typename T, typename Cmp>
[[nodiscard]] bool binarySearch(It inFirst, It inLast, const T& inVal, Cmp inCmp) {
	inFirst = lowerBound(inFirst, inLast, inVal, inCmp);
	return inFirst != inLast && !inCmp(inVal, *inFirst);
}

template<typename It, typename T>
[[nodiscard]] bool binarySearch(It inFirst, It inLast, const T& inVal) {
	return binarySearch(inFirst, inLast, inVal, DefaultLess{});
}

template<typename It, typename T>
[[nodiscard]] It find(It inFirst, It inLast, const T& inVal) {
	for (; inFirst != inLast; ++inFirst) {
		if (*inFirst == inVal) {
			return inFirst;
		}
	}
	return inLast;
}

template<typename T, typename U>
[[nodiscard]] typename oa::Span<T>::iterator find(oa::Span<T> inSpan, const U& inVal) {
	return find(inSpan.data(), inSpan.data() + inSpan.size(), inVal);
}

template<typename It, typename Pred>
[[nodiscard]] It findIf(It inFirst, It inLast, Pred inPred) {
	for (; inFirst != inLast; ++inFirst) {
		if (inPred(*inFirst)) {
			return inFirst;
		}
	}
	return inLast;
}

template<typename T, typename Pred>
[[nodiscard]] typename oa::Span<T>::iterator findIf(oa::Span<T> inSpan, Pred inPred) {
	return findIf(inSpan.data(), inSpan.data() + inSpan.size(), inPred);
}

template<typename It, typename Cmp>
void sort(It inFirst, It inLast, Cmp inCmp) {
	AlgoHeapSort::sort(inFirst, inLast, inCmp);
}

template<typename It>
void sort(It inFirst, It inLast) {
	oa::sort(inFirst, inLast, DefaultLess{});
}

template<typename T, typename Cmp>
void sort(oa::Span<T> inSpan, Cmp inCmp) {
	sort(inSpan.data(), inSpan.data() + inSpan.size(), inCmp);
}

template<typename T>
void sort(oa::Span<T> inSpan) {
	sort(inSpan.data(), inSpan.data() + inSpan.size());
}

template<typename It, typename T>
[[nodiscard]] auto count(It inFirst, It inLast, const T& inVal) {
	typename IterTraits<It>::difference_type n = 0;
	for (; inFirst != inLast; ++inFirst) {
		if (*inFirst == inVal) {
			++n;
		}
	}
	return n;
}

template<typename T, typename U>
[[nodiscard]] auto count(oa::Span<T> inSpan, const U& inVal) {
	return count(inSpan.data(), inSpan.data() + inSpan.size(), inVal);
}

template<typename It1, typename It2>
[[nodiscard]] bool equal(It1 inFirst1, It1 inLast1, It2 inFirst2, It2 inLast2) {
	while (inFirst1 != inLast1 && inFirst2 != inLast2) {
		if (!(*inFirst1 == *inFirst2)) {
			return false;
		}
		++inFirst1;
		++inFirst2;
	}
	return inFirst1 == inLast1 && inFirst2 == inLast2;
}

template<typename T, typename U>
[[nodiscard]] bool equal(oa::Span<T> inA, oa::Span<U> inB) {
	if (inA.size() != inB.size()) {
		return false;
	}
	return equal(inA.data(), inA.data() + inA.size(), inB.data(), inB.data() + inB.size());
}

template<typename It, typename T>
void fill(It inFirst, It inLast, const T& inVal) {
	for (; inFirst != inLast; ++inFirst) {
		*inFirst = inVal;
	}
}

template<typename T, typename U>
void fill(oa::Span<T> inSpan, const U& inVal) {
	fill(inSpan.data(), inSpan.data() + inSpan.size(), inVal);
}

template<typename It, typename outIt>
outIt copy(It inFirst, It inLast, outIt inDest) {
	while (inFirst != inLast) {
		*inDest = *inFirst;
		++inDest;
		++inFirst;
	}
	return inDest;
}

template<typename T, typename outIt>
outIt copy(oa::Span<T> inSpan, outIt inDest) {
	return copy(inSpan.data(), inSpan.data() + inSpan.size(), inDest);
}

template<typename It, typename Pred>
[[nodiscard]] bool allOf(It inFirst, It inLast, Pred inPred) {
	for (; inFirst != inLast; ++inFirst) {
		if (!inPred(*inFirst)) {
			return false;
		}
	}
	return true;
}

template<typename T, typename Pred>
[[nodiscard]] bool allOf(oa::Span<T> inSpan, Pred inPred) {
	return allOf(inSpan.data(), inSpan.data() + inSpan.size(), inPred);
}

template<typename It, typename Pred>
[[nodiscard]] bool anyOf(It inFirst, It inLast, Pred inPred) {
	for (; inFirst != inLast; ++inFirst) {
		if (inPred(*inFirst)) {
			return true;
		}
	}
	return false;
}

template<typename T, typename Pred>
[[nodiscard]] bool anyOf(oa::Span<T> inSpan, Pred inPred) {
	return anyOf(inSpan.data(), inSpan.data() + inSpan.size(), inPred);
}

template<typename It, typename Pred>
[[nodiscard]] bool noneOf(It inFirst, It inLast, Pred inPred) {
	for (; inFirst != inLast; ++inFirst) {
		if (inPred(*inFirst)) {
			return false;
		}
	}
	return true;
}

template<typename T, typename Pred>
[[nodiscard]] bool noneOf(oa::Span<T> inSpan, Pred inPred) {
	return noneOf(inSpan.data(), inSpan.data() + inSpan.size(), inPred);
}

// Scalar two-argument min/max (the std::min / std::max replacement — distinct
// from minElement/maxElement which scan a range). Returns by const&
// like std, so ties return the first argument.
template<typename T>
[[nodiscard]] constexpr const T& min(const T& inA, const T& inB) {
	return (inB < inA) ? inB : inA;
}
template<typename T, typename Cmp>
[[nodiscard]] constexpr const T& min(const T& inA, const T& inB, Cmp inCmp) {
	return inCmp(inB, inA) ? inB : inA;
}
template<typename T>
[[nodiscard]] constexpr const T& max(const T& inA, const T& inB) {
	return (inA < inB) ? inB : inA;
}
template<typename T, typename Cmp>
[[nodiscard]] constexpr const T& max(const T& inA, const T& inB, Cmp inCmp) {
	return inCmp(inA, inB) ? inB : inA;
}

template<typename T>
[[nodiscard]] const T& clamp(const T& inVal, const T& inLo, const T& inHi) {
	if (inVal < inLo) {
		return inLo;
	}
	if (inHi < inVal) {
		return inHi;
	}
	return inVal;
}

template<typename It>
void reverse(It inFirst, It inLast) {
	while (inFirst != inLast && inFirst != --inLast) {
		oa::swapValues(*inFirst++, *inLast);
	}
}

template<typename T>
void reverse(oa::Span<T> inSpan) {
	reverse(inSpan.data(), inSpan.data() + inSpan.size());
}

template<typename It>
[[nodiscard]] It unique(It inFirst, It inLast) {
	if (inFirst == inLast) {
		return inLast;
	}
	It dest = inFirst;
	while (++inFirst != inLast) {
		if (!(*dest == *inFirst)) {
			*++dest = oa::move(*inFirst);
		}
	}
	return ++dest;
}

template<typename T>
[[nodiscard]] typename oa::Span<T>::iterator unique(oa::Span<T> inSpan) {
	return unique(inSpan.data(), inSpan.data() + inSpan.size());
}

template<typename It>
[[nodiscard]] It minElement(It inFirst, It inLast) {
	if (inFirst == inLast) {
		return inLast;
	}
	It best = inFirst;
	for (++inFirst; inFirst != inLast; ++inFirst) {
		if (*inFirst < *best) {
			best = inFirst;
		}
	}
	return best;
}

template<typename It, typename Cmp>
[[nodiscard]] It minElement(It inFirst, It inLast, Cmp inCmp) {
	if (inFirst == inLast) {
		return inLast;
	}
	It best = inFirst;
	for (++inFirst; inFirst != inLast; ++inFirst) {
		if (inCmp(*inFirst, *best)) {
			best = inFirst;
		}
	}
	return best;
}

template<typename It>
[[nodiscard]] It maxElement(It inFirst, It inLast) {
	if (inFirst == inLast) {
		return inLast;
	}
	It best = inFirst;
	for (++inFirst; inFirst != inLast; ++inFirst) {
		if (*best < *inFirst) {
			best = inFirst;
		}
	}
	return best;
}

template<typename It, typename Cmp>
[[nodiscard]] It maxElement(It inFirst, It inLast, Cmp inCmp) {
	if (inFirst == inLast) {
		return inLast;
	}
	It best = inFirst;
	for (++inFirst; inFirst != inLast; ++inFirst) {
		if (inCmp(*best, *inFirst)) {
			best = inFirst;
		}
	}
	return best;
}

template<typename T>
[[nodiscard]] typename oa::Span<T>::iterator minElement(oa::Span<T> inSpan) {
	return minElement(inSpan.data(), inSpan.data() + inSpan.size());
}

template<typename T, typename Cmp>
[[nodiscard]] typename oa::Span<T>::iterator minElement(oa::Span<T> inSpan, Cmp inCmp) {
	return minElement(inSpan.data(), inSpan.data() + inSpan.size(), inCmp);
}

template<typename T>
[[nodiscard]] typename oa::Span<T>::iterator maxElement(oa::Span<T> inSpan) {
	return maxElement(inSpan.data(), inSpan.data() + inSpan.size());
}

template<typename T, typename Cmp>
[[nodiscard]] typename oa::Span<T>::iterator maxElement(oa::Span<T> inSpan, Cmp inCmp) {
	return maxElement(inSpan.data(), inSpan.data() + inSpan.size(), inCmp);
}

} // namespace oa
