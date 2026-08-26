#pragma once

// Fixed-capacity OA storage. The object owns exactly N inline elements and
// performs no allocation. Checked access uses the always-on OA contract path;
// no exception or hosted C++ standard-library type crosses this header.

#define OA_TYPES_H_SKIP_REST
#include <oa/core/types.h>
#undef OA_TYPES_H_SKIP_REST

#include <oa/core/assert.h>
#include <oa/core/std/typeTraits.h>

namespace oa {

template<typename T, oa::Usize N>
class Array {
public:
	using value_type = T;
	using size_type = oa::Usize;
	using reference = T&;
	using const_reference = const T&;
	using iterator = T*;
	using const_iterator = const T*;

	constexpr Array() = default;

	template<typename First, typename... Rest>
		requires (sizeof...(Rest) + 1U == N)
	constexpr Array(First inFirst, Rest... inRest)
		: elems_{static_cast<T>(inFirst), static_cast<T>(inRest)...} {}

	[[nodiscard]] reference at(size_type inIndex) noexcept {
		OA_REQUIRE(inIndex < N);
		return elems_[inIndex];
	}

	[[nodiscard]] const_reference at(size_type inIndex) const noexcept {
		OA_REQUIRE(inIndex < N);
		return elems_[inIndex];
	}

	[[nodiscard]] reference operator[](size_type inIndex) noexcept {
		OA_ASSERT(inIndex < N);
		return elems_[inIndex];
	}

	[[nodiscard]] const_reference operator[](size_type inIndex) const noexcept {
		OA_ASSERT(inIndex < N);
		return elems_[inIndex];
	}

	[[nodiscard]] static constexpr size_type size() noexcept { return N; }

	[[nodiscard]] static constexpr bool empty() noexcept { return false; }

	[[nodiscard]] T* data() noexcept { return elems_; }

	[[nodiscard]] const T* data() const noexcept { return elems_; }

	[[nodiscard]] reference front() noexcept {
		OA_ASSERT(N > 0);
		return elems_[0];
	}

	[[nodiscard]] const_reference front() const noexcept {
		OA_ASSERT(N > 0);
		return elems_[0];
	}

	[[nodiscard]] reference back() noexcept {
		OA_ASSERT(N > 0);
		return elems_[N - 1U];
	}

	[[nodiscard]] const_reference back() const noexcept {
		OA_ASSERT(N > 0);
		return elems_[N - 1U];
	}

	void fill(const T& inValue) {
		for (size_type index = 0; index < N; ++index) {
			elems_[index] = inValue;
		}
	}

	void swap(Array& inOther) noexcept(oa::IsNothrowSwappableV<T>) {
		for (size_type index = 0; index < N; ++index) {
			oa::swapValues(elems_[index], inOther.elems_[index]);
		}
	}

	[[nodiscard]] iterator begin() noexcept { return elems_; }

	[[nodiscard]] const_iterator begin() const noexcept { return elems_; }

	[[nodiscard]] iterator end() noexcept { return elems_ + N; }

	[[nodiscard]] const_iterator end() const noexcept { return elems_ + N; }

	[[nodiscard]] friend bool operator==(const Array& inLeft, const Array& inRight) noexcept {
		for (size_type index = 0; index < N; ++index) {
			if (inLeft.elems_[index] != inRight.elems_[index]) {
				return false;
			}
		}
		return true;
	}

	[[nodiscard]] friend bool operator!=(const Array& inLeft, const Array& inRight) noexcept {
		return not (inLeft == inRight);
	}

	[[nodiscard]] friend bool operator<(const Array& inLeft, const Array& inRight) noexcept {
		for (size_type index = 0; index < N; ++index) {
			if (inLeft.elems_[index] < inRight.elems_[index]) {
				return true;
			}
			if (inRight.elems_[index] < inLeft.elems_[index]) {
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
	using size_type = oa::Usize;
	using reference = T&;
	using const_reference = const T&;
	using iterator = T*;
	using const_iterator = const T*;

	[[nodiscard]] reference at(size_type /*inIndex*/) noexcept {
		OA_REQUIRE(false);
	}

	[[nodiscard]] const_reference at(size_type /*inIndex*/) const noexcept {
		OA_REQUIRE(false);
	}

	[[nodiscard]] static constexpr size_type size() noexcept { return 0; }

	[[nodiscard]] static constexpr bool empty() noexcept { return true; }

	[[nodiscard]] T* data() noexcept { return nullptr; }

	[[nodiscard]] const T* data() const noexcept { return nullptr; }

	void fill(const T& /*inValue*/) noexcept {}

	void swap(Array& /*inOther*/) noexcept {}

	[[nodiscard]] iterator begin() noexcept { return nullptr; }

	[[nodiscard]] const_iterator begin() const noexcept { return nullptr; }

	[[nodiscard]] iterator end() noexcept { return nullptr; }

	[[nodiscard]] const_iterator end() const noexcept { return nullptr; }

	[[nodiscard]] friend bool operator==(const Array&, const Array&) noexcept { return true; }

	[[nodiscard]] friend bool operator!=(const Array&, const Array&) noexcept { return false; }

	[[nodiscard]] friend bool operator<(const Array&, const Array&) noexcept { return false; }
};

template<typename T, oa::Usize N>
inline typename Array<T, N>::iterator begin(Array<T, N>& inArray) noexcept {
	return inArray.begin();
}

template<typename T, oa::Usize N>
inline typename Array<T, N>::const_iterator begin(const Array<T, N>& inArray) noexcept {
	return inArray.begin();
}

template<typename T, oa::Usize N>
inline typename Array<T, N>::iterator end(Array<T, N>& inArray) noexcept {
	return inArray.end();
}

template<typename T, oa::Usize N>
inline typename Array<T, N>::const_iterator end(const Array<T, N>& inArray) noexcept {
	return inArray.end();
}

} // namespace oa
