#pragma once

// Native OaStdSpan — non-owning `T*` + size (dynamic extent).
//
// Iterators: raw pointers; `First`/`Last`/`SubSpan` use `assert` on bounds (debug).
// Interop: `StdSpan()` → `std::span`; template ctors from `std::array` / `OaStdArray`; explicit ctor from `std::span`.

#include <Oa/Core/Std/Array.h>

#include <array>
#include <cassert>
#include <cstddef>
#include <span>
#include <type_traits>

template<typename T>
class OaStdSpan {
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

	OaStdSpan() noexcept = default;

	OaStdSpan(pointer InPtr, size_type InCount) noexcept : Ptr_(InPtr), Size_(InCount) {}

	template<std::size_t N>
	OaStdSpan(std::array<value_type, N>& InArr) noexcept
		requires(!std::is_const_v<T>)
		: Ptr_(InArr.data()), Size_(N) {}

	template<std::size_t N>
	OaStdSpan(const std::array<value_type, N>& InArr) noexcept
		requires(std::is_const_v<T>)
		: Ptr_(InArr.data()), Size_(N) {}

	template<std::size_t N>
	OaStdSpan(OaStdArray<value_type, N>& InArr) noexcept
		requires(!std::is_const_v<T>)
		: Ptr_(InArr.data()), Size_(N) {}

	template<std::size_t N>
	OaStdSpan(const OaStdArray<value_type, N>& InArr) noexcept
		requires(std::is_const_v<T>)
		: Ptr_(InArr.data()), Size_(N) {}

	template<std::size_t N>
	OaStdSpan(T (&InArr)[N]) noexcept : Ptr_(InArr), Size_(N) {}

	explicit OaStdSpan(std::span<T> InS) noexcept : Ptr_(InS.data()), Size_(InS.size()) {}

	[[nodiscard]] std::span<T> StdSpan() const noexcept { return std::span<T>(Ptr_, Size_); }

	[[nodiscard]] size_type Size() const noexcept { return Size_; }
	[[nodiscard]] size_type size() const noexcept { return Size_; }
	[[nodiscard]] size_type SizeBytes() const noexcept { return Size_ * sizeof(T); }
	[[nodiscard]] bool Empty() const noexcept { return Size_ == 0; }
	[[nodiscard]] bool empty() const noexcept { return Size_ == 0; }

	[[nodiscard]] pointer Data() const noexcept { return Ptr_; }
	[[nodiscard]] pointer data() const noexcept { return Ptr_; }

	[[nodiscard]] reference Front() const { return *Ptr_; }
	[[nodiscard]] reference Back() const { return Ptr_[Size_ - 1U]; }
	[[nodiscard]] reference operator[](size_type InIdx) const { return Ptr_[InIdx]; }

	[[nodiscard]] OaStdSpan<T> First(size_type InCount) const {
		assert(InCount <= Size_);
		return OaStdSpan<T>(Ptr_, InCount);
	}

	[[nodiscard]] OaStdSpan<T> SubSpan(size_type InOffset, size_type InCount = DynamicExtent) const {
		assert(InOffset <= Size_);
		size_type const rem = Size_ - InOffset;
		size_type const ext = InCount == DynamicExtent ? rem : InCount;
		assert(InCount == DynamicExtent || ext <= rem);
		return OaStdSpan<T>(Ptr_ + InOffset, ext);
	}

	[[nodiscard]] iterator Begin() const noexcept { return Ptr_; }
	[[nodiscard]] iterator End() const noexcept { return Ptr_ + Size_; }

	[[nodiscard]] iterator begin() const noexcept { return Begin(); }
	[[nodiscard]] iterator end() const noexcept { return End(); }

private:
	pointer Ptr_ = nullptr;
	size_type Size_ = 0;
};

template<typename T>
inline typename OaStdSpan<T>::iterator begin(OaStdSpan<T> InS) noexcept {
	return InS.Begin();
}
template<typename T>
inline typename OaStdSpan<T>::iterator end(OaStdSpan<T> InS) noexcept {
	return InS.End();
}
