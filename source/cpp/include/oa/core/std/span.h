#pragma once

// Native Span — non-owning `T*` + size (dynamic extent).
//
// Iterators: raw pointers; `first`/`Last`/`subSpan` use `assert` on bounds (debug).
// Interop: `stdSpan()` → `std::span`; template ctors from `std::array` / `oa::Array`; explicit ctor from `std::span`.

#include <oa/core/std/array.h>

#include <array>
#include <cassert>
#include <cstddef>
#include <span>
#include <type_traits>

namespace oa {

template<typename T>
class Span {
public:
	static constexpr std::size_t DynamicExtent = static_cast<std::size_t>(-1);

	using element_type = T;
	using value_type = std::remove_cv_t<T>;
	using size_type = std::size_t;
	using pointer = T*;
	using const_pointer = const T*;
	using reference = T&;
	using const_reference = const T&;
	using iterator = pointer;
	using const_iterator = const_pointer;

	Span() noexcept = default;

	Span(pointer inPtr, size_type inCount) noexcept : ptr_(inPtr), size_(inCount) {}

	template<std::size_t N>
	Span(std::array<value_type, N>& inArr) noexcept
		requires(!std::is_const_v<T>)
		: ptr_(inArr.data()), size_(N) {}

	template<std::size_t N>
	Span(const std::array<value_type, N>& inArr) noexcept
		requires(std::is_const_v<T>)
		: ptr_(inArr.data()), size_(N) {}

	template<std::size_t N>
	Span(oa::Array<value_type, N>& inArr) noexcept
		requires(!std::is_const_v<T>)
		: ptr_(inArr.data()), size_(N) {}

	template<std::size_t N>
	Span(const oa::Array<value_type, N>& inArr) noexcept
		requires(std::is_const_v<T>)
		: ptr_(inArr.data()), size_(N) {}

	template<std::size_t N>
	Span(T (&inArr)[N]) noexcept : ptr_(inArr), size_(N) {}

	explicit Span(std::span<T> inS) noexcept : ptr_(inS.data()), size_(inS.size()) {}

	[[nodiscard]] std::span<T> stdSpan() const noexcept { return std::span<T>(ptr_, size_); }

	[[nodiscard]] size_type size() const noexcept { return size_; }
	[[nodiscard]] size_type sizeBytes() const noexcept { return size_ * sizeof(T); }
	[[nodiscard]] bool empty() const noexcept { return size_ == 0; }

	[[nodiscard]] pointer data() const noexcept { return ptr_; }

	[[nodiscard]] reference front() const { return *ptr_; }
	[[nodiscard]] reference back() const { return ptr_[size_ - 1U]; }
	[[nodiscard]] reference operator[](size_type inIdx) const { return ptr_[inIdx]; }

	[[nodiscard]] Span<T> first(size_type inCount) const {
		assert(inCount <= size_);
		return Span<T>(ptr_, inCount);
	}

	[[nodiscard]] Span<T> subSpan(size_type inOffset, size_type inCount = DynamicExtent) const {
		assert(inOffset <= size_);
		size_type const rem = size_ - inOffset;
		size_type const ext = inCount == DynamicExtent ? rem : inCount;
		assert(inCount == DynamicExtent || ext <= rem);
		return Span<T>(ptr_ + inOffset, ext);
	}

	[[nodiscard]] iterator begin() const noexcept { return ptr_; }
	[[nodiscard]] iterator end() const noexcept { return ptr_ + size_; }

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
