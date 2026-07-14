#pragma once

// Phase 2b OaStd — stack `T[N]`; `StdArray()` copies to `std::array` at boundaries.
//
// Iterators: raw `T*` / `const T*`; `At` / `operator[]`; `At` throws `std::out_of_range` on bad index.

#include <Oa/Core/Std/TypeTraits.h>

#include <array>
#include <cstddef>
#include <stdexcept>

template<typename T, std::size_t N>
class OaStdArray {
public:
	using value_type = T;
	using size_type = std::size_t;
	using reference = T&;
	using const_reference = const T&;
	using iterator = T*;
	using const_iterator = const T*;

	[[nodiscard]] reference At(size_type InIdx) {
		if (InIdx >= N) {
			throw std::out_of_range("OaStdArray::At");
		}
		return Elems_[InIdx];
	}

	[[nodiscard]] const_reference At(size_type InIdx) const {
		if (InIdx >= N) {
			throw std::out_of_range("OaStdArray::At");
		}
		return Elems_[InIdx];
	}

	[[nodiscard]] reference operator[](size_type InIdx) noexcept { return Elems_[InIdx]; }

	[[nodiscard]] const_reference operator[](size_type InIdx) const noexcept { return Elems_[InIdx]; }

	[[nodiscard]] constexpr size_type Size() const noexcept { return N; }

	[[nodiscard]] constexpr bool Empty() const noexcept { return N == 0; }

	[[nodiscard]] T* Data() noexcept { return Elems_; }

	[[nodiscard]] const T* Data() const noexcept { return Elems_; }

	[[nodiscard]] reference Front() noexcept { return Elems_[0]; }

	[[nodiscard]] const_reference Front() const noexcept { return Elems_[0]; }

	[[nodiscard]] reference Back() noexcept { return Elems_[N - 1U]; }

	[[nodiscard]] const_reference Back() const noexcept { return Elems_[N - 1U]; }

	void Fill(const T& InVal) {
		for (size_type i = 0; i < N; ++i) {
			Elems_[i] = InVal;
		}
	}

	void Swap(OaStdArray& InO) noexcept(OaStdIsNothrowSwappableV<T>) {
		for (size_type i = 0; i < N; ++i) {
			OaStdSwap(Elems_[i], InO.Elems_[i]);
		}
	}

	[[nodiscard]] iterator Begin() noexcept { return Elems_; }

	[[nodiscard]] const_iterator Begin() const noexcept { return Elems_; }

	[[nodiscard]] iterator End() noexcept { return Elems_ + N; }

	[[nodiscard]] const_iterator End() const noexcept { return Elems_ + N; }

	[[nodiscard]] std::array<T, N> StdArray() const {
		std::array<T, N> out{};
		for (size_type i = 0; i < N; ++i) {
			out[i] = Elems_[i];
		}
		return out;
	}

	[[nodiscard]] friend bool operator==(const OaStdArray& InL, const OaStdArray& InR) noexcept {
		for (size_type idx = 0; idx < N; ++idx) {
			if (InL.Elems_[idx] != InR.Elems_[idx]) {
				return false;
			}
		}
		return true;
	}

	[[nodiscard]] friend bool operator!=(const OaStdArray& InL, const OaStdArray& InR) noexcept {
		return not (InL == InR);
	}

	[[nodiscard]] friend bool operator<(const OaStdArray& InL, const OaStdArray& InR) noexcept {
		for (size_type idx = 0; idx < N; ++idx) {
			if (InL.Elems_[idx] < InR.Elems_[idx]) {
				return true;
			}
			if (InR.Elems_[idx] < InL.Elems_[idx]) {
				return false;
			}
		}
		return false;
	}

	// std::array-shaped aliases (snake_case) for OaArray / drop-in use.
	void fill(const T& InVal) { Fill(InVal); }

	[[nodiscard]] T* data() noexcept { return Data(); }

	[[nodiscard]] const T* data() const noexcept { return Data(); }

	[[nodiscard]] constexpr size_type size() const noexcept { return N; }

	[[nodiscard]] constexpr size_type max_size() const noexcept { return N; }

private:
	T Elems_[N]{};
};

template<typename T>
class OaStdArray<T, 0> {
public:
	using value_type = T;
	using size_type = std::size_t;
	using reference = T&;
	using const_reference = const T&;
	using iterator = T*;
	using const_iterator = const T*;

	[[nodiscard]] reference At(size_type /*InIdx*/) {
		throw std::out_of_range("OaStdArray::At");
	}

	[[nodiscard]] const_reference At(size_type /*InIdx*/) const {
		throw std::out_of_range("OaStdArray::At");
	}

	[[nodiscard]] constexpr size_type Size() const noexcept { return 0; }

	[[nodiscard]] constexpr bool Empty() const noexcept { return true; }

	[[nodiscard]] T* Data() noexcept { return nullptr; }

	[[nodiscard]] const T* Data() const noexcept { return nullptr; }

	void Fill(const T& /*InVal*/) noexcept {}

	void fill(const T& /*InVal*/) noexcept {}

	void Swap(OaStdArray& /*InO*/) noexcept {}

	[[nodiscard]] T* data() noexcept { return nullptr; }

	[[nodiscard]] const T* data() const noexcept { return nullptr; }

	[[nodiscard]] constexpr size_type size() const noexcept { return 0; }

	[[nodiscard]] constexpr size_type max_size() const noexcept { return 0; }

	[[nodiscard]] iterator Begin() noexcept { return nullptr; }

	[[nodiscard]] const_iterator Begin() const noexcept { return nullptr; }

	[[nodiscard]] iterator End() noexcept { return nullptr; }

	[[nodiscard]] const_iterator End() const noexcept { return nullptr; }

	[[nodiscard]] std::array<T, 0> StdArray() const { return {}; }

	[[nodiscard]] friend bool operator==(const OaStdArray& InL, const OaStdArray& InR) noexcept {
		(void)InL;
		(void)InR;
		return true;
	}

	[[nodiscard]] friend bool operator!=(const OaStdArray& InL, const OaStdArray& InR) noexcept {
		(void)InL;
		(void)InR;
		return false;
	}

	[[nodiscard]] friend bool operator<(const OaStdArray& InL, const OaStdArray& InR) noexcept {
		(void)InL;
		(void)InR;
		return false;
	}
};

template<typename T, std::size_t N>
inline typename OaStdArray<T, N>::iterator begin(OaStdArray<T, N>& InA) noexcept {
	return InA.Begin();
}

template<typename T, std::size_t N>
inline typename OaStdArray<T, N>::const_iterator begin(const OaStdArray<T, N>& InA) noexcept {
	return InA.Begin();
}

template<typename T, std::size_t N>
inline typename OaStdArray<T, N>::iterator end(OaStdArray<T, N>& InA) noexcept {
	return InA.End();
}

template<typename T, std::size_t N>
inline typename OaStdArray<T, N>::const_iterator end(const OaStdArray<T, N>& InA) noexcept {
	return InA.End();
}
