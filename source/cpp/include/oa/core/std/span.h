#pragma once

// Native Span — non-owning `T*` + size (dynamic extent).
//
// Iterators are raw pointers. Construction and slicing use the always-on OA
// contract. A null pointer is valid only for an empty span.

#include <oa/core/std/array.h>
#include <oa/core/std/assert.h>
#include <oa/core/std/typeTraits.h>

namespace oa {

template<typename T>
class Vector;

template<typename T>
class Span {
public:
	static constexpr oa::Usize DynamicExtent = static_cast<oa::Usize>(-1);

	using element_type = T;
	using value_type = oa::RemoveCvT<T>;
	using size_type = oa::Usize;
	using pointer = T*;
	using const_pointer = const T*;
	using reference = T&;
	using const_reference = const T&;
	using iterator = pointer;
	using const_iterator = const_pointer;

	Span() noexcept = default;

	Span(pointer inPtr, size_type inCount) noexcept : ptr_(inPtr), size_(inCount) {
		OA_REQUIRE(inPtr != nullptr || inCount == 0);
		OA_REQUIRE(inCount <= static_cast<size_type>(-1) / sizeof(T));
	}

	template<typename U>
	Span(oa::Vector<U>& inVec) noexcept
		requires(oa::IsConvertibleV<U*, pointer>)
		: Span(inVec.data(), inVec.size()) {}

	template<typename U>
	Span(const oa::Vector<U>& inVec) noexcept
		requires(oa::IsConvertibleV<const U*, pointer>)
		: Span(inVec.data(), inVec.size()) {}

	template<oa::Usize N>
	Span(oa::Array<value_type, N>& inArr) noexcept
		requires(!oa::IsConstV<T>)
		: ptr_(inArr.data()), size_(N) {}

	template<oa::Usize N>
	Span(const oa::Array<value_type, N>& inArr) noexcept
		requires(oa::IsConstV<T>)
		: ptr_(inArr.data()), size_(N) {}

	template<oa::Usize N>
	Span(T (&inArr)[N]) noexcept : ptr_(inArr), size_(N) {}

	[[nodiscard]] size_type size() const noexcept { return size_; }
	[[nodiscard]] size_type sizeBytes() const noexcept { return size_ * sizeof(T); }
	[[nodiscard]] bool empty() const noexcept { return size_ == 0; }

	[[nodiscard]] pointer data() const noexcept { return ptr_; }

	[[nodiscard]] reference front() const noexcept {
		OA_REQUIRE(!empty());
		return *ptr_;
	}
	[[nodiscard]] reference back() const noexcept {
		OA_REQUIRE(!empty());
		return ptr_[size_ - 1U];
	}
	[[nodiscard]] reference operator[](size_type inIdx) const { return ptr_[inIdx]; }

	[[nodiscard]] Span<T> first(size_type inCount) const noexcept {
		OA_REQUIRE(inCount <= size_);
		return Span<T>(ptr_, inCount);
	}

	[[nodiscard]] Span<T> subSpan(
		size_type inOffset,
		size_type inCount = DynamicExtent
	) const noexcept {
		OA_REQUIRE(inOffset <= size_);
		size_type const rem = size_ - inOffset;
		size_type const ext = inCount == DynamicExtent ? rem : inCount;
		OA_REQUIRE(inCount == DynamicExtent || ext <= rem);
		return Span<T>(ptr_ == nullptr ? nullptr : ptr_ + inOffset, ext);
	}

	[[nodiscard]] iterator begin() const noexcept { return ptr_; }
	[[nodiscard]] iterator end() const noexcept {
		return ptr_ == nullptr ? nullptr : ptr_ + size_;
	}

private:
	pointer ptr_ = nullptr;
	size_type size_ = 0;
};

template<typename T>
inline typename Span<T>::iterator begin(Span<T> inS) noexcept {
	return inS.begin();
}
template<typename T>
inline typename Span<T>::iterator end(Span<T> inS) noexcept {
	return inS.end();
}

} // namespace oa
