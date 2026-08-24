#pragma once

// phase 2b OA standard library — stack `T[N]`; `stdArray()` copies to `std::array` at boundaries.
//
// Iterators: raw `T*` / `const T*`; `at` / `operator[]`; `at` throws `std::out_of_range` on bad index.

#include <oa/core/std/typeTraits.h>

#include <array>
#include <cstddef>
#include <stdexcept>

namespace oa {

template<typename T, std::size_t N>
class Array {
public:
	using value_type = T;
	using size_type = std::size_t;
	using reference = T&;
	using const_reference = const T&;
	using iterator = T*;
	using const_iterator = const T*;

	[[nodiscard]] reference at(size_type inIdx) {
		if (inIdx >= N) {
			throw std::out_of_range("Array::at");
		}
		return elems_[inIdx];
	}

	[[nodiscard]] const_reference at(size_type inIdx) const {
		if (inIdx >= N) {
			throw std::out_of_range("Array::at");
		}
		return elems_[inIdx];
	}

	[[nodiscard]] reference operator[](size_type inIdx) noexcept { return elems_[inIdx]; }

	[[nodiscard]] const_reference operator[](size_type inIdx) const noexcept { return elems_[inIdx]; }

	[[nodiscard]] constexpr size_type size() const noexcept { return N; }

	[[nodiscard]] constexpr bool empty() const noexcept { return N == 0; }

	[[nodiscard]] T* data() noexcept { return elems_; }

	[[nodiscard]] const T* data() const noexcept { return elems_; }

	[[nodiscard]] reference front() noexcept { return elems_[0]; }

	[[nodiscard]] const_reference front() const noexcept { return elems_[0]; }

	[[nodiscard]] reference back() noexcept { return elems_[N - 1U]; }

	[[nodiscard]] const_reference back() const noexcept { return elems_[N - 1U]; }

	void fill(const T& inVal) {
		for (size_type i = 0; i < N; ++i) {
			elems_[i] = inVal;
		}
	}

	void swap(Array& inO) noexcept(oa::IsNothrowSwappableV<T>) {
		for (size_type i = 0; i < N; ++i) {
			oa::swapValues(elems_[i], inO.elems_[i]);
		}
	}

	[[nodiscard]] iterator begin() noexcept { return elems_; }

	[[nodiscard]] const_iterator begin() const noexcept { return elems_; }

	[[nodiscard]] iterator end() noexcept { return elems_ + N; }

	[[nodiscard]] const_iterator end() const noexcept { return elems_ + N; }

	[[nodiscard]] std::array<T, N> stdArray() const {
		std::array<T, N> out{};
		for (size_type i = 0; i < N; ++i) {
			out[i] = elems_[i];
		}
		return out;
	}

	[[nodiscard]] friend bool operator==(const Array& inL, const Array& inR) noexcept {
		for (size_type idx = 0; idx < N; ++idx) {
			if (inL.elems_[idx] != inR.elems_[idx]) {
				return false;
			}
		}
		return true;
	}

	[[nodiscard]] friend bool operator!=(const Array& inL, const Array& inR) noexcept {
		return not (inL == inR);
	}

	[[nodiscard]] friend bool operator<(const Array& inL, const Array& inR) noexcept {
		for (size_type idx = 0; idx < N; ++idx) {
			if (inL.elems_[idx] < inR.elems_[idx]) {
				return true;
			}
			if (inR.elems_[idx] < inL.elems_[idx]) {
				return false;
			}
		}
		return false;
	}

private:
	T elems_[N]{};
};

template<typename T>
class Array<T, 0> {
public:
	using value_type = T;
	using size_type = std::size_t;
	using reference = T&;
	using const_reference = const T&;
	using iterator = T*;
	using const_iterator = const T*;

	[[nodiscard]] reference at(size_type /*inIdx*/) {
		throw std::out_of_range("Array::at");
	}

	[[nodiscard]] const_reference at(size_type /*inIdx*/) const {
		throw std::out_of_range("Array::at");
	}

	[[nodiscard]] constexpr size_type size() const noexcept { return 0; }

	[[nodiscard]] constexpr bool empty() const noexcept { return true; }

	[[nodiscard]] T* data() noexcept { return nullptr; }

	[[nodiscard]] const T* data() const noexcept { return nullptr; }

	void fill(const T& /*inVal*/) noexcept {}

	void swap(Array& /*inO*/) noexcept {}

	[[nodiscard]] iterator begin() noexcept { return nullptr; }

	[[nodiscard]] const_iterator begin() const noexcept { return nullptr; }

	[[nodiscard]] iterator end() noexcept { return nullptr; }

	[[nodiscard]] const_iterator end() const noexcept { return nullptr; }

	[[nodiscard]] std::array<T, 0> stdArray() const { return {}; }

	[[nodiscard]] friend bool operator==(const Array& inL, const Array& inR) noexcept {
		(void)inL;
		(void)inR;
		return true;
	}

	[[nodiscard]] friend bool operator!=(const Array& inL, const Array& inR) noexcept {
		(void)inL;
		(void)inR;
		return false;
	}

	[[nodiscard]] friend bool operator<(const Array& inL, const Array& inR) noexcept {
		(void)inL;
		(void)inR;
		return false;
	}
};

template<typename T, std::size_t N>
inline typename Array<T, N>::iterator begin(Array<T, N>& inA) noexcept {
	return inA.begin();
}

template<typename T, std::size_t N>
inline typename Array<T, N>::const_iterator begin(const Array<T, N>& inA) noexcept {
	return inA.begin();
}

template<typename T, std::size_t N>
inline typename Array<T, N>::iterator end(Array<T, N>& inA) noexcept {
	return inA.end();
}

template<typename T, std::size_t N>
inline typename Array<T, N>::const_iterator end(const Array<T, N>& inA) noexcept {
	return inA.end();
}

} // namespace oa
