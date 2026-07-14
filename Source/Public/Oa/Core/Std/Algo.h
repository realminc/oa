#pragma once

// OaStdAlgo — iterator helpers + **`OaStdSpan` overloads** (no `<algorithm>`, no `<iterator>` / `<utility>`).
//
// Sort: in-place heapsort, **O(n log n)**, unstable. Bounds APIs match `std` for the given comparator.
// `lower_bound` / `upper_bound` / `binary_search` use **`OaStdDistance` / `OaStdAdvance` / `OaStdNext`** (`Iter.h`).

#include <Oa/Core/Std/Iter.h>
#include <Oa/Core/Std/Span.h>
#include <Oa/Core/Std/Utility.h>

struct OaStdDefaultLess {
	template<typename A, typename B>
	constexpr bool operator()(const A& InA, const B& InB) const noexcept(noexcept(InA < InB)) {
		return InA < InB;
	}
};

struct OaStdAlgoHeapSort {
	template<typename RandIt, typename Cmp>
	static void SiftDown(RandIt InFirst,
		typename OaStdIterTraits<RandIt>::difference_type InRoot,
		typename OaStdIterTraits<RandIt>::difference_type InEnd,
		Cmp InCmp) {
		using Diff = typename OaStdIterTraits<RandIt>::difference_type;
		for (;;) {
			Diff child = InRoot * 2 + 1;
			if (child >= InEnd) {
				break;
			}
			const Diff right = child + 1;
			if (right < InEnd && InCmp(*(InFirst + child), *(InFirst + right))) {
				child = right;
			}
			if (!InCmp(*(InFirst + InRoot), *(InFirst + child))) {
				break;
			}
			OaStdSwap(*(InFirst + InRoot), *(InFirst + child));
			InRoot = child;
		}
	}

	template<typename RandIt, typename Cmp>
	static void Sort(RandIt InFirst, RandIt InLast, Cmp InCmp) {
		using Diff = typename OaStdIterTraits<RandIt>::difference_type;
		const Diff n = InLast - InFirst;
		if (n < 2) {
			return;
		}
		for (Diff idx = n / 2 - 1; idx >= 0; --idx) {
			SiftDown(InFirst, idx, n, InCmp);
		}
		for (Diff heapEnd = n; heapEnd > 1; --heapEnd) {
			OaStdSwap(*InFirst, *(InFirst + (heapEnd - 1)));
			SiftDown(InFirst, 0, heapEnd - 1, InCmp);
		}
	}
};

template<typename It, typename T, typename Cmp>
[[nodiscard]] It OaStdLowerBound(It InFirst, It InLast, const T& InVal, Cmp InCmp) {
	using Diff = typename OaStdIterTraits<It>::difference_type;
	Diff len = OaStdDistance(InFirst, InLast);
	while (len > 0) {
		const Diff half = len / 2;
		It mid = InFirst;
		OaStdAdvance(mid, half);
		if (InCmp(*mid, InVal)) {
			InFirst = OaStdNext(mid);
			len -= half + 1;
		} else {
			len = half;
		}
	}
	return InFirst;
}

template<typename It, typename T>
[[nodiscard]] It OaStdLowerBound(It InFirst, It InLast, const T& InVal) {
	return OaStdLowerBound(InFirst, InLast, InVal, OaStdDefaultLess{});
}

template<typename It, typename T, typename Cmp>
[[nodiscard]] It OaStdUpperBound(It InFirst, It InLast, const T& InVal, Cmp InCmp) {
	using Diff = typename OaStdIterTraits<It>::difference_type;
	Diff len = OaStdDistance(InFirst, InLast);
	while (len > 0) {
		const Diff half = len / 2;
		It mid = InFirst;
		OaStdAdvance(mid, half);
		if (InCmp(InVal, *mid)) {
			len = half;
		} else {
			InFirst = OaStdNext(mid);
			len -= half + 1;
		}
	}
	return InFirst;
}

template<typename It, typename T>
[[nodiscard]] It OaStdUpperBound(It InFirst, It InLast, const T& InVal) {
	return OaStdUpperBound(InFirst, InLast, InVal, OaStdDefaultLess{});
}

template<typename It, typename T, typename Cmp>
[[nodiscard]] bool OaStdBinarySearch(It InFirst, It InLast, const T& InVal, Cmp InCmp) {
	InFirst = OaStdLowerBound(InFirst, InLast, InVal, InCmp);
	return InFirst != InLast && !InCmp(InVal, *InFirst);
}

template<typename It, typename T>
[[nodiscard]] bool OaStdBinarySearch(It InFirst, It InLast, const T& InVal) {
	return OaStdBinarySearch(InFirst, InLast, InVal, OaStdDefaultLess{});
}

template<typename It, typename T>
[[nodiscard]] It OaStdFind(It InFirst, It InLast, const T& InVal) {
	for (; InFirst != InLast; ++InFirst) {
		if (*InFirst == InVal) {
			return InFirst;
		}
	}
	return InLast;
}

template<typename T, typename U>
[[nodiscard]] typename OaStdSpan<T>::iterator OaStdFind(OaStdSpan<T> InSpan, const U& InVal) {
	return OaStdFind(InSpan.Data(), InSpan.Data() + InSpan.Size(), InVal);
}

template<typename It, typename Pred>
[[nodiscard]] It OaStdFindIf(It InFirst, It InLast, Pred InPred) {
	for (; InFirst != InLast; ++InFirst) {
		if (InPred(*InFirst)) {
			return InFirst;
		}
	}
	return InLast;
}

template<typename T, typename Pred>
[[nodiscard]] typename OaStdSpan<T>::iterator OaStdFindIf(OaStdSpan<T> InSpan, Pred InPred) {
	return OaStdFindIf(InSpan.Data(), InSpan.Data() + InSpan.Size(), InPred);
}

template<typename It, typename Cmp>
void OaStdSort(It InFirst, It InLast, Cmp InCmp) {
	OaStdAlgoHeapSort::Sort(InFirst, InLast, InCmp);
}

template<typename It>
void OaStdSort(It InFirst, It InLast) {
	OaStdSort(InFirst, InLast, OaStdDefaultLess{});
}

template<typename T, typename Cmp>
void OaStdSort(OaStdSpan<T> InSpan, Cmp InCmp) {
	OaStdSort(InSpan.Data(), InSpan.Data() + InSpan.Size(), InCmp);
}

template<typename T>
void OaStdSort(OaStdSpan<T> InSpan) {
	OaStdSort(InSpan.Data(), InSpan.Data() + InSpan.Size());
}

template<typename It, typename T>
[[nodiscard]] auto OaStdCount(It InFirst, It InLast, const T& InVal) {
	typename OaStdIterTraits<It>::difference_type n = 0;
	for (; InFirst != InLast; ++InFirst) {
		if (*InFirst == InVal) {
			++n;
		}
	}
	return n;
}

template<typename T, typename U>
[[nodiscard]] auto OaStdCount(OaStdSpan<T> InSpan, const U& InVal) {
	return OaStdCount(InSpan.Data(), InSpan.Data() + InSpan.Size(), InVal);
}

template<typename It1, typename It2>
[[nodiscard]] bool OaStdEqual(It1 InFirst1, It1 InLast1, It2 InFirst2, It2 InLast2) {
	while (InFirst1 != InLast1 && InFirst2 != InLast2) {
		if (!(*InFirst1 == *InFirst2)) {
			return false;
		}
		++InFirst1;
		++InFirst2;
	}
	return InFirst1 == InLast1 && InFirst2 == InLast2;
}

template<typename T, typename U>
[[nodiscard]] bool OaStdEqual(OaStdSpan<T> InA, OaStdSpan<U> InB) {
	if (InA.Size() != InB.Size()) {
		return false;
	}
	return OaStdEqual(InA.Data(), InA.Data() + InA.Size(), InB.Data(), InB.Data() + InB.Size());
}

template<typename It, typename T>
void OaStdFill(It InFirst, It InLast, const T& InVal) {
	for (; InFirst != InLast; ++InFirst) {
		*InFirst = InVal;
	}
}

template<typename T, typename U>
void OaStdFill(OaStdSpan<T> InSpan, const U& InVal) {
	OaStdFill(InSpan.Data(), InSpan.Data() + InSpan.Size(), InVal);
}

template<typename It, typename OutIt>
OutIt OaStdCopy(It InFirst, It InLast, OutIt InDest) {
	while (InFirst != InLast) {
		*InDest = *InFirst;
		++InDest;
		++InFirst;
	}
	return InDest;
}

template<typename T, typename OutIt>
OutIt OaStdCopy(OaStdSpan<T> InSpan, OutIt InDest) {
	return OaStdCopy(InSpan.Data(), InSpan.Data() + InSpan.Size(), InDest);
}

template<typename It, typename Pred>
[[nodiscard]] bool OaStdAllOf(It InFirst, It InLast, Pred InPred) {
	for (; InFirst != InLast; ++InFirst) {
		if (!InPred(*InFirst)) {
			return false;
		}
	}
	return true;
}

template<typename T, typename Pred>
[[nodiscard]] bool OaStdAllOf(OaStdSpan<T> InSpan, Pred InPred) {
	return OaStdAllOf(InSpan.Data(), InSpan.Data() + InSpan.Size(), InPred);
}

template<typename It, typename Pred>
[[nodiscard]] bool OaStdAnyOf(It InFirst, It InLast, Pred InPred) {
	for (; InFirst != InLast; ++InFirst) {
		if (InPred(*InFirst)) {
			return true;
		}
	}
	return false;
}

template<typename T, typename Pred>
[[nodiscard]] bool OaStdAnyOf(OaStdSpan<T> InSpan, Pred InPred) {
	return OaStdAnyOf(InSpan.Data(), InSpan.Data() + InSpan.Size(), InPred);
}

template<typename It, typename Pred>
[[nodiscard]] bool OaStdNoneOf(It InFirst, It InLast, Pred InPred) {
	for (; InFirst != InLast; ++InFirst) {
		if (InPred(*InFirst)) {
			return false;
		}
	}
	return true;
}

template<typename T, typename Pred>
[[nodiscard]] bool OaStdNoneOf(OaStdSpan<T> InSpan, Pred InPred) {
	return OaStdNoneOf(InSpan.Data(), InSpan.Data() + InSpan.Size(), InPred);
}

// Scalar two-argument min/max (the std::min / std::max replacement — distinct
// from OaStdMinElement/OaStdMaxElement which scan a range). Returns by const&
// like std, so ties return the first argument.
template<typename T>
[[nodiscard]] constexpr const T& OaStdMin(const T& InA, const T& InB) {
	return (InB < InA) ? InB : InA;
}
template<typename T, typename Cmp>
[[nodiscard]] constexpr const T& OaStdMin(const T& InA, const T& InB, Cmp InCmp) {
	return InCmp(InB, InA) ? InB : InA;
}
template<typename T>
[[nodiscard]] constexpr const T& OaStdMax(const T& InA, const T& InB) {
	return (InA < InB) ? InB : InA;
}
template<typename T, typename Cmp>
[[nodiscard]] constexpr const T& OaStdMax(const T& InA, const T& InB, Cmp InCmp) {
	return InCmp(InA, InB) ? InB : InA;
}

template<typename T>
[[nodiscard]] const T& OaStdClamp(const T& InVal, const T& InLo, const T& InHi) {
	if (InVal < InLo) {
		return InLo;
	}
	if (InHi < InVal) {
		return InHi;
	}
	return InVal;
}

template<typename It>
void OaStdReverse(It InFirst, It InLast) {
	while (InFirst != InLast && InFirst != --InLast) {
		OaStdSwap(*InFirst++, *InLast);
	}
}

template<typename T>
void OaStdReverse(OaStdSpan<T> InSpan) {
	OaStdReverse(InSpan.Data(), InSpan.Data() + InSpan.Size());
}

template<typename It>
[[nodiscard]] It OaStdUnique(It InFirst, It InLast) {
	if (InFirst == InLast) {
		return InLast;
	}
	It dest = InFirst;
	while (++InFirst != InLast) {
		if (!(*dest == *InFirst)) {
			*++dest = OaStdMove(*InFirst);
		}
	}
	return ++dest;
}

template<typename T>
[[nodiscard]] typename OaStdSpan<T>::iterator OaStdUnique(OaStdSpan<T> InSpan) {
	return OaStdUnique(InSpan.Data(), InSpan.Data() + InSpan.Size());
}

template<typename It>
[[nodiscard]] It OaStdMinElement(It InFirst, It InLast) {
	if (InFirst == InLast) {
		return InLast;
	}
	It best = InFirst;
	for (++InFirst; InFirst != InLast; ++InFirst) {
		if (*InFirst < *best) {
			best = InFirst;
		}
	}
	return best;
}

template<typename It, typename Cmp>
[[nodiscard]] It OaStdMinElement(It InFirst, It InLast, Cmp InCmp) {
	if (InFirst == InLast) {
		return InLast;
	}
	It best = InFirst;
	for (++InFirst; InFirst != InLast; ++InFirst) {
		if (InCmp(*InFirst, *best)) {
			best = InFirst;
		}
	}
	return best;
}

template<typename It>
[[nodiscard]] It OaStdMaxElement(It InFirst, It InLast) {
	if (InFirst == InLast) {
		return InLast;
	}
	It best = InFirst;
	for (++InFirst; InFirst != InLast; ++InFirst) {
		if (*best < *InFirst) {
			best = InFirst;
		}
	}
	return best;
}

template<typename It, typename Cmp>
[[nodiscard]] It OaStdMaxElement(It InFirst, It InLast, Cmp InCmp) {
	if (InFirst == InLast) {
		return InLast;
	}
	It best = InFirst;
	for (++InFirst; InFirst != InLast; ++InFirst) {
		if (InCmp(*best, *InFirst)) {
			best = InFirst;
		}
	}
	return best;
}

template<typename T>
[[nodiscard]] typename OaStdSpan<T>::iterator OaStdMinElement(OaStdSpan<T> InSpan) {
	return OaStdMinElement(InSpan.Data(), InSpan.Data() + InSpan.Size());
}

template<typename T, typename Cmp>
[[nodiscard]] typename OaStdSpan<T>::iterator OaStdMinElement(OaStdSpan<T> InSpan, Cmp InCmp) {
	return OaStdMinElement(InSpan.Data(), InSpan.Data() + InSpan.Size(), InCmp);
}

template<typename T>
[[nodiscard]] typename OaStdSpan<T>::iterator OaStdMaxElement(OaStdSpan<T> InSpan) {
	return OaStdMaxElement(InSpan.Data(), InSpan.Data() + InSpan.Size());
}

template<typename T, typename Cmp>
[[nodiscard]] typename OaStdSpan<T>::iterator OaStdMaxElement(OaStdSpan<T> InSpan, Cmp InCmp) {
	return OaStdMaxElement(InSpan.Data(), InSpan.Data() + InSpan.Size(), InCmp);
}
