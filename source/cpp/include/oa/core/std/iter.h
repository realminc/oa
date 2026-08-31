#pragma once

// OA iterator primitives. This header is freestanding: it does not import the
// C++ standard iterator machinery into public OA headers.

#include <oa/core/std/typeTraits.h>
#include <oa/core/std/assert.h>

namespace oa {

struct ForwardIteratorTag {};

template<typename It>
struct IterTraits {
	using difference_type = typename It::difference_type;
	using value_type = typename It::value_type;
	using pointer = typename It::pointer;
	using reference = typename It::reference;
};

template<typename T>
struct IterTraits<T*> {
	using difference_type = __PTRDIFF_TYPE__;
	using value_type = oa::RemoveCvT<T>;
	using pointer = T*;
	using reference = T&;
};

template<typename T>
struct IterTraits<const T*> {
	using difference_type = __PTRDIFF_TYPE__;
	using value_type = oa::RemoveCvT<T>;
	using pointer = const T*;
	using reference = const T&;
};

template<typename It>
inline constexpr bool isRandomAccessIteratorV = requires(
	const It& inA,
	const It& inB,
	typename IterTraits<It>::difference_type inN
) {
	inB - inA;
	inA + inN;
};

template<typename It>
inline constexpr bool isBidirectionalIteratorV = requires(It inIt) {
	--inIt;
	++inIt;
	*inIt;
};

template<typename It>
[[nodiscard]] constexpr typename IterTraits<It>::difference_type distance(
	It inFirst,
	It inLast
) {
	using Diff = typename IterTraits<It>::difference_type;
	if constexpr (isRandomAccessIteratorV<It>) {
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
	if constexpr (isRandomAccessIteratorV<It>) {
		inIt += inN;
	} else if constexpr (isBidirectionalIteratorV<It>) {
		// Count toward zero instead of negating a potentially minimum signed
		// difference value.
		while (inN > 0) {
			++inIt;
			--inN;
		}
		while (inN < 0) {
			--inIt;
			++inN;
		}
	} else {
		OA_REQUIRE_MSG(inN >= 0,
			"oa::advance cannot move a forward iterator backwards");
		for (Diff k = 0; k < inN; ++k) {
			++inIt;
		}
	}
}

template<typename It>
[[nodiscard]] constexpr It next(
	It inIt,
	typename IterTraits<It>::difference_type inN = 1
) {
	advance(inIt, inN);
	return inIt;
}

template<typename It>
class ReverseIterator {
public:
	using iterator_type = It;
	using difference_type = typename IterTraits<It>::difference_type;
	using value_type = typename IterTraits<It>::value_type;
	using pointer = typename IterTraits<It>::pointer;
	using reference = typename IterTraits<It>::reference;

	constexpr ReverseIterator() = default;
	explicit constexpr ReverseIterator(It inCurrent) : current_(inCurrent) {}

	template<typename Other>
	requires oa::isConvertibleV<const Other&, It>
	constexpr ReverseIterator(const ReverseIterator<Other>& inOther)
		: current_(inOther.base()) {}

	[[nodiscard]] constexpr It base() const { return current_; }

	[[nodiscard]] constexpr reference operator*() const {
		It value = current_;
		--value;
		return *value;
	}

	[[nodiscard]] constexpr pointer operator->() const { return &operator*(); }

	constexpr ReverseIterator& operator++() {
		--current_;
		return *this;
	}

	constexpr ReverseIterator operator++(int) {
		ReverseIterator value(*this);
		--current_;
		return value;
	}

	constexpr ReverseIterator& operator--() {
		++current_;
		return *this;
	}

	constexpr ReverseIterator operator--(int) {
		ReverseIterator value(*this);
		++current_;
		return value;
	}

	constexpr ReverseIterator& operator+=(difference_type inOffset) {
		current_ -= inOffset;
		return *this;
	}

	constexpr ReverseIterator& operator-=(difference_type inOffset) {
		current_ += inOffset;
		return *this;
	}

	[[nodiscard]] constexpr ReverseIterator operator+(difference_type inOffset) const {
		return ReverseIterator(current_ - inOffset);
	}

	[[nodiscard]] constexpr ReverseIterator operator-(difference_type inOffset) const {
		return ReverseIterator(current_ + inOffset);
	}

	[[nodiscard]] constexpr reference operator[](difference_type inOffset) const {
		return *(*this + inOffset);
	}

private:
	It current_{};
};

template<typename Left, typename Right>
[[nodiscard]] constexpr bool operator==(
	const ReverseIterator<Left>& inLeft,
	const ReverseIterator<Right>& inRight
) {
	return inLeft.base() == inRight.base();
}

template<typename Left, typename Right>
[[nodiscard]] constexpr bool operator!=(
	const ReverseIterator<Left>& inLeft,
	const ReverseIterator<Right>& inRight
) {
	return !(inLeft == inRight);
}

template<typename Left, typename Right>
[[nodiscard]] constexpr bool operator<(
	const ReverseIterator<Left>& inLeft,
	const ReverseIterator<Right>& inRight
) {
	return inRight.base() < inLeft.base();
}

template<typename Left, typename Right>
[[nodiscard]] constexpr bool operator>(
	const ReverseIterator<Left>& inLeft,
	const ReverseIterator<Right>& inRight
) {
	return inRight < inLeft;
}

template<typename Left, typename Right>
[[nodiscard]] constexpr bool operator<=(
	const ReverseIterator<Left>& inLeft,
	const ReverseIterator<Right>& inRight
) {
	return !(inRight < inLeft);
}

template<typename Left, typename Right>
[[nodiscard]] constexpr bool operator>=(
	const ReverseIterator<Left>& inLeft,
	const ReverseIterator<Right>& inRight
) {
	return !(inLeft < inRight);
}

template<typename Left, typename Right>
[[nodiscard]] constexpr auto operator-(
	const ReverseIterator<Left>& inLeft,
	const ReverseIterator<Right>& inRight
) -> decltype(inRight.base() - inLeft.base()) {
	return inRight.base() - inLeft.base();
}

template<typename It>
[[nodiscard]] constexpr ReverseIterator<It> operator+(
	typename ReverseIterator<It>::difference_type inOffset,
	const ReverseIterator<It>& inIterator
) {
	return inIterator + inOffset;
}

template<typename It>
using IterValueT = typename IterTraits<It>::value_type;

template<typename T>
[[nodiscard]] constexpr T* toAddress(T* inPointer) noexcept {
	return inPointer;
}

template<typename Pointer>
[[nodiscard]] constexpr auto toAddress(const Pointer& inPointer) noexcept {
	return oa::toAddress(inPointer.operator->());
}

// Pointers are the only iterators for which OA currently promises contiguous
// storage. Other iterator families remain correct through element-wise copy.
template<typename It>
inline constexpr bool isContiguousIteratorV = oa::isPointerV<It>;

} // namespace oa
