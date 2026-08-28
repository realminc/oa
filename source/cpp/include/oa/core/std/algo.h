#pragma once

// OA iterator algorithms with `oa::Span` overloads.
//
// sort: in-place introsort, **O(n log n)** worst case, unstable. Median-of-three
// partitioning handles the common path, insertion sort closes small partitions,
// and heapsort is the bounded-depth fallback.
// `lower_bound` / `upper_bound` / `binary_search` use **`distance` / `advance` / `next`** (`Iter.h`).

#include <oa/core/std/iter.h>
#include <oa/core/std/span.h>
#include <oa/core/std/utility.h>

namespace oa {

template<typename T>
[[nodiscard]] constexpr T gcd(T inA, T inB) noexcept {
	static_assert(oa::IsIntegralV<T>, "oa::gcd requires an integral type");
	static_assert(sizeof(T) <= sizeof(oa::U64),
		"oa::gcd currently admits integral types up to 64 bits");
	constexpr bool isSigned = static_cast<T>(-1) < static_cast<T>(0);
	const auto magnitude = [](T inValue) constexpr -> oa::U64 {
		const oa::U64 raw = static_cast<oa::U64>(inValue);
		if constexpr (isSigned) {
			return inValue < 0 ? oa::U64{0} - raw : raw;
		}
		return raw;
	};

	oa::U64 a = magnitude(inA);
	oa::U64 b = magnitude(inB);
	while (b != 0) {
		const oa::U64 remainder = a % b;
		a = b;
		b = remainder;
	}
	if constexpr (isSigned) {
		constexpr unsigned bits = static_cast<unsigned>(sizeof(T) * __CHAR_BIT__);
		constexpr oa::U64 maxValue = bits == 64U
			? (~oa::U64{0} >> 1U)
			: ((oa::U64{1} << (bits - 1U)) - 1U);
		OA_REQUIRE_MSG(a <= maxValue, "oa::gcd result is not representable");
	}
	return static_cast<T>(a);
}

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

struct AlgoIntroSort {
	static constexpr __PTRDIFF_TYPE__ insertionThreshold = 24;

	template<typename RandIt, typename Cmp>
	static void insertionSort(RandIt inFirst, RandIt inLast, Cmp inCmp) {
		if (inFirst == inLast) return;
		for (RandIt current = inFirst + 1; current != inLast; ++current) {
			typename IterTraits<RandIt>::value_type value = oa::move(*current);
			RandIt insertion = current;
			while (insertion != inFirst and inCmp(value, *(insertion - 1))) {
				*insertion = oa::move(*(insertion - 1));
				--insertion;
			}
			*insertion = oa::move(value);
		}
	}

	template<typename RandIt, typename Cmp>
	[[nodiscard]] static RandIt partition(RandIt inFirst, RandIt inLast, Cmp inCmp) {
		RandIt middle = inFirst + (inLast - inFirst) / 2;
		RandIt lastValue = inLast - 1;
		if (inCmp(*middle, *inFirst)) oa::swapValues(*middle, *inFirst);
		if (inCmp(*lastValue, *middle)) oa::swapValues(*lastValue, *middle);
		if (inCmp(*middle, *inFirst)) oa::swapValues(*middle, *inFirst);
		oa::swapValues(*inFirst, *middle);

		RandIt left = inFirst + 1;
		RandIt right = lastValue;
		for (;;) {
			while (left <= right and inCmp(*left, *inFirst)) ++left;
			while (left <= right and inCmp(*inFirst, *right)) --right;
			if (left > right) break;
			oa::swapValues(*left, *right);
			++left;
			--right;
		}
		oa::swapValues(*inFirst, *right);
		return right;
	}

	template<typename RandIt, typename Cmp>
	static void sortLoop(
		RandIt inFirst,
		RandIt inLast,
		typename IterTraits<RandIt>::difference_type inDepth,
		Cmp inCmp
	) {
		using Diff = typename IterTraits<RandIt>::difference_type;
		while (inLast - inFirst > static_cast<Diff>(insertionThreshold)) {
			if (inDepth == 0) {
				AlgoHeapSort::sort(inFirst, inLast, inCmp);
				return;
			}
			--inDepth;
			RandIt pivot = partition(inFirst, inLast, inCmp);
			RandIt rightFirst = pivot + 1;
			if (pivot - inFirst < inLast - rightFirst) {
				sortLoop(inFirst, pivot, inDepth, inCmp);
				inFirst = rightFirst;
			} else {
				sortLoop(rightFirst, inLast, inDepth, inCmp);
				inLast = pivot;
			}
		}
		insertionSort(inFirst, inLast, inCmp);
	}

	template<typename RandIt, typename Cmp>
	static void sort(RandIt inFirst, RandIt inLast, Cmp inCmp) {
		using Diff = typename IterTraits<RandIt>::difference_type;
		const Diff count = inLast - inFirst;
		if (count < 2) return;
		Diff depth = 0;
		for (Diff value = count; value > 1; value /= 2) ++depth;
		sortLoop(inFirst, inLast, depth * 2, inCmp);
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
	return find(inSpan.begin(), inSpan.end(), inVal);
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
	return findIf(inSpan.begin(), inSpan.end(), inPred);
}

template<typename It, typename Cmp>
void sort(It inFirst, It inLast, Cmp inCmp) {
	AlgoIntroSort::sort(inFirst, inLast, inCmp);
}

template<typename It>
void sort(It inFirst, It inLast) {
	oa::sort(inFirst, inLast, DefaultLess{});
}

// Stable insertion sort is intentionally used for the small control-plane
// collections that require stable ordering. It performs no allocation and
// keeps equal elements in their original order.
template<typename RandIt, typename Cmp>
void stableSort(RandIt inFirst, RandIt inLast, Cmp inCmp) {
	if (inFirst == inLast) return;
	for (RandIt current = inFirst + 1; current != inLast; ++current) {
		auto value = oa::move(*current);
		RandIt insertion = current;
		while (insertion != inFirst && inCmp(value, *(insertion - 1))) {
			*insertion = oa::move(*(insertion - 1));
			--insertion;
		}
		*insertion = oa::move(value);
	}
}

template<typename RandIt>
void stableSort(RandIt inFirst, RandIt inLast) {
	oa::stableSort(inFirst, inLast, DefaultLess{});
}

template<typename T, typename Cmp>
void sort(oa::Span<T> inSpan, Cmp inCmp) {
	if (inSpan.empty()) return;
	sort(inSpan.begin(), inSpan.end(), inCmp);
}

template<typename T>
void sort(oa::Span<T> inSpan) {
	if (inSpan.empty()) return;
	sort(inSpan.begin(), inSpan.end());
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
	return count(inSpan.begin(), inSpan.end(), inVal);
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
	return equal(inA.begin(), inA.end(), inB.begin(), inB.end());
}

template<typename It, typename T>
void fill(It inFirst, It inLast, const T& inVal) {
	for (; inFirst != inLast; ++inFirst) {
		*inFirst = inVal;
	}
}

template<typename T, typename U>
void fill(oa::Span<T> inSpan, const U& inVal) {
	fill(inSpan.begin(), inSpan.end(), inVal);
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
	return copy(inSpan.begin(), inSpan.end(), inDest);
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
	return allOf(inSpan.begin(), inSpan.end(), inPred);
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
	return anyOf(inSpan.begin(), inSpan.end(), inPred);
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
	return noneOf(inSpan.begin(), inSpan.end(), inPred);
}

// Scalar two-argument min/max (the oa::min / oa::max replacement — distinct
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
	reverse(inSpan.begin(), inSpan.end());
}

template<typename It, typename Pred>
[[nodiscard]] It removeIf(It inFirst, It inLast, Pred inPred) {
	inFirst = oa::findIf(inFirst, inLast, inPred);
	if (inFirst == inLast) return inLast;
	for (It current = inFirst; ++current != inLast;) {
		if (not inPred(*current)) {
			*inFirst = oa::move(*current);
			++inFirst;
		}
	}
	return inFirst;
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
	return unique(inSpan.begin(), inSpan.end());
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
	return minElement(inSpan.begin(), inSpan.end());
}

template<typename T, typename Cmp>
	[[nodiscard]] typename oa::Span<T>::iterator minElement(oa::Span<T> inSpan, Cmp inCmp) {
	return minElement(inSpan.begin(), inSpan.end(), inCmp);
}

template<typename T>
	[[nodiscard]] typename oa::Span<T>::iterator maxElement(oa::Span<T> inSpan) {
	return maxElement(inSpan.begin(), inSpan.end());
}

template<typename T, typename Cmp>
	[[nodiscard]] typename oa::Span<T>::iterator maxElement(oa::Span<T> inSpan, Cmp inCmp) {
	return maxElement(inSpan.begin(), inSpan.end(), inCmp);
}

} // namespace oa
