#pragma once

#include <oa/core/vlm/quaternion.h>

#include <oa/core/std/assert.h>
#include <oa/core/std/typeTraits.h>

namespace oa {

namespace vlm {

namespace detail {
template <typename T> struct Mat3;
template <typename T> struct Mat4;
} // namespace detail

template <typename T>
[[nodiscard]] constexpr detail::Mat3<T> add(
	const detail::Mat3<T>& inA,
	const detail::Mat3<T>& inB) noexcept;
template <typename T>
[[nodiscard]] constexpr detail::Mat3<T> sub(
	const detail::Mat3<T>& inA,
	const detail::Mat3<T>& inB) noexcept;
template <typename T>
[[nodiscard]] constexpr detail::Mat3<T> scale(
	const detail::Mat3<T>& inValue,
	T inScale) noexcept;
template <typename T>
[[nodiscard]] constexpr detail::Mat3<T> divide(
	const detail::Mat3<T>& inValue,
	T inScale) noexcept;
template <typename T>
[[nodiscard]] constexpr detail::Mat3<T> matrixMul(
	const detail::Mat3<T>& inA,
	const detail::Mat3<T>& inB) noexcept;
template <typename T>
[[nodiscard]] constexpr detail::Vec3<T> transform(
	const detail::Vec3<T>& inValue,
	const detail::Mat3<T>& inMatrix) noexcept;
template <typename T>
[[nodiscard]] constexpr detail::Mat4<T> add(
	const detail::Mat4<T>& inA,
	const detail::Mat4<T>& inB) noexcept;
template <typename T>
[[nodiscard]] constexpr detail::Mat4<T> sub(
	const detail::Mat4<T>& inA,
	const detail::Mat4<T>& inB) noexcept;
template <typename T>
[[nodiscard]] constexpr detail::Mat4<T> scale(
	const detail::Mat4<T>& inValue,
	T inScale) noexcept;
template <typename T>
[[nodiscard]] constexpr detail::Mat4<T> divide(
	const detail::Mat4<T>& inValue,
	T inScale) noexcept;
template <typename T>
[[nodiscard]] constexpr detail::Mat4<T> matrixMul(
	const detail::Mat4<T>& inA,
	const detail::Mat4<T>& inB) noexcept;
template <typename T>
[[nodiscard]] constexpr detail::Vec4<T> transform(
	const detail::Vec4<T>& inValue,
	const detail::Mat4<T>& inMatrix) noexcept;

namespace detail {

template <typename T>
struct Mat3 {
	static_assert(oa::IsFloatingPointV<T>);

	// Row-major storage and row-vector multiplication: transformed = value * M.
	T m[3][3] = {};

	[[nodiscard]] constexpr T& at(Usize inRow, Usize inColumn) noexcept {
		OA_REQUIRE(inRow < 3U and inColumn < 3U);
		return m[inRow][inColumn];
	}
	[[nodiscard]] constexpr const T& at(
		Usize inRow,
		Usize inColumn) const noexcept {
		OA_REQUIRE(inRow < 3U and inColumn < 3U);
		return m[inRow][inColumn];
	}
	[[nodiscard]] bool isFinite() const noexcept {
		for (Usize row = 0U; row < 3U; ++row) {
			for (Usize column = 0U; column < 3U; ++column) {
				if (not oa::isFinite(m[row][column])) return false;
			}
		}
		return true;
	}

	[[nodiscard]] static constexpr Mat3 identity() noexcept {
		Mat3 result{};
		result.m[0][0] = T(1);
		result.m[1][1] = T(1);
		result.m[2][2] = T(1);
		return result;
	}

	[[nodiscard]] constexpr Mat3 operator+(
		const Mat3& inOther) const noexcept {
		return oa::vlm::add(*this, inOther);
	}
	[[nodiscard]] constexpr Mat3 operator-() const noexcept {
		return oa::vlm::scale(*this, T(-1));
	}
	[[nodiscard]] constexpr Mat3 operator-(
		const Mat3& inOther) const noexcept {
		return oa::vlm::sub(*this, inOther);
	}
	[[nodiscard]] constexpr Mat3 operator*(T inScale) const noexcept {
		return oa::vlm::scale(*this, inScale);
	}
	[[nodiscard]] constexpr Mat3 operator/(T inScale) const noexcept {
		return oa::vlm::divide(*this, inScale);
	}
	[[nodiscard]] constexpr Mat3 operator*(
		const Mat3& inOther) const noexcept {
		return oa::vlm::matrixMul(*this, inOther);
	}
	constexpr Mat3& operator+=(const Mat3& inOther) noexcept {
		*this = oa::vlm::add(*this, inOther);
		return *this;
	}
	constexpr Mat3& operator-=(const Mat3& inOther) noexcept {
		*this = oa::vlm::sub(*this, inOther);
		return *this;
	}
	constexpr Mat3& operator*=(T inScale) noexcept {
		*this = oa::vlm::scale(*this, inScale);
		return *this;
	}
	constexpr Mat3& operator/=(T inScale) noexcept {
		*this = oa::vlm::divide(*this, inScale);
		return *this;
	}
	constexpr Mat3& operator*=(const Mat3& inOther) noexcept {
		*this = oa::vlm::matrixMul(*this, inOther);
		return *this;
	}

	[[nodiscard]] constexpr bool operator==(
		const Mat3& inOther) const noexcept = default;
};

template <typename T>
[[nodiscard]] constexpr Mat3<T> operator*(
	T inScale,
	const Mat3<T>& inMatrix) noexcept {
	return oa::vlm::scale(inMatrix, inScale);
}

template <typename T>
[[nodiscard]] constexpr Vec3<T> operator*(
	const Vec3<T>& inValue,
	const Mat3<T>& inMatrix) noexcept {
	return oa::vlm::transform(inValue, inMatrix);
}

template <typename T>
struct Mat4 {
	static_assert(oa::IsFloatingPointV<T>);

	// Row-major storage and row-vector multiplication: transformed = value * M.
	T m[4][4] = {};

	[[nodiscard]] constexpr T& at(Usize inRow, Usize inColumn) noexcept {
		OA_REQUIRE(inRow < 4U and inColumn < 4U);
		return m[inRow][inColumn];
	}
	[[nodiscard]] constexpr const T& at(
		Usize inRow,
		Usize inColumn) const noexcept {
		OA_REQUIRE(inRow < 4U and inColumn < 4U);
		return m[inRow][inColumn];
	}
	[[nodiscard]] bool isFinite() const noexcept {
		for (Usize row = 0U; row < 4U; ++row) {
			for (Usize column = 0U; column < 4U; ++column) {
				if (not oa::isFinite(m[row][column])) return false;
			}
		}
		return true;
	}

	[[nodiscard]] static constexpr Mat4 identity() noexcept {
		Mat4 result{};
		result.m[0][0] = T(1);
		result.m[1][1] = T(1);
		result.m[2][2] = T(1);
		result.m[3][3] = T(1);
		return result;
	}

	[[nodiscard]] constexpr Mat4 operator+(
		const Mat4& inOther) const noexcept {
		return oa::vlm::add(*this, inOther);
	}
	[[nodiscard]] constexpr Mat4 operator-() const noexcept {
		return oa::vlm::scale(*this, T(-1));
	}
	[[nodiscard]] constexpr Mat4 operator-(
		const Mat4& inOther) const noexcept {
		return oa::vlm::sub(*this, inOther);
	}
	[[nodiscard]] constexpr Mat4 operator*(T inScale) const noexcept {
		return oa::vlm::scale(*this, inScale);
	}
	[[nodiscard]] constexpr Mat4 operator/(T inScale) const noexcept {
		return oa::vlm::divide(*this, inScale);
	}
	[[nodiscard]] constexpr Mat4 operator*(
		const Mat4& inOther) const noexcept {
		return oa::vlm::matrixMul(*this, inOther);
	}
	constexpr Mat4& operator+=(const Mat4& inOther) noexcept {
		*this = oa::vlm::add(*this, inOther);
		return *this;
	}
	constexpr Mat4& operator-=(const Mat4& inOther) noexcept {
		*this = oa::vlm::sub(*this, inOther);
		return *this;
	}
	constexpr Mat4& operator*=(T inScale) noexcept {
		*this = oa::vlm::scale(*this, inScale);
		return *this;
	}
	constexpr Mat4& operator/=(T inScale) noexcept {
		*this = oa::vlm::divide(*this, inScale);
		return *this;
	}
	constexpr Mat4& operator*=(const Mat4& inOther) noexcept {
		*this = oa::vlm::matrixMul(*this, inOther);
		return *this;
	}

	[[nodiscard]] constexpr bool operator==(const Mat4& inOther) const noexcept = default;
};

template <typename T>
[[nodiscard]] constexpr Mat4<T> operator*(
	T inScale,
	const Mat4<T>& inMatrix) noexcept {
	return oa::vlm::scale(inMatrix, inScale);
}

template <typename T>
[[nodiscard]] constexpr Vec4<T> operator*(
	const Vec4<T>& inValue,
	const Mat4<T>& inMatrix) noexcept {
	return oa::vlm::transform(inValue, inMatrix);
}

} // namespace detail

template <typename T>
[[nodiscard]] constexpr detail::Mat3<T> add(
	const detail::Mat3<T>& inA,
	const detail::Mat3<T>& inB) noexcept {
	detail::Mat3<T> result{};
	for (I32 row = 0; row < 3; ++row) {
		result.m[row][0] = inA.m[row][0] + inB.m[row][0];
		result.m[row][1] = inA.m[row][1] + inB.m[row][1];
		result.m[row][2] = inA.m[row][2] + inB.m[row][2];
	}
	return result;
}

template <typename T>
[[nodiscard]] constexpr detail::Mat3<T> sub(
	const detail::Mat3<T>& inA,
	const detail::Mat3<T>& inB) noexcept {
	detail::Mat3<T> result{};
	for (I32 row = 0; row < 3; ++row) {
		for (I32 column = 0; column < 3; ++column) {
			result.m[row][column] = inA.m[row][column] - inB.m[row][column];
		}
	}
	return result;
}

template <typename T>
[[nodiscard]] constexpr detail::Mat3<T> scale(
	const detail::Mat3<T>& inValue,
	T inScale) noexcept {
	detail::Mat3<T> result{};
	for (I32 row = 0; row < 3; ++row) {
		for (I32 column = 0; column < 3; ++column) {
			result.m[row][column] = inValue.m[row][column] * inScale;
		}
	}
	return result;
}

template <typename T>
[[nodiscard]] constexpr detail::Mat3<T> divide(
	const detail::Mat3<T>& inValue,
	T inScale) noexcept {
	detail::Mat3<T> result{};
	for (I32 row = 0; row < 3; ++row) {
		for (I32 column = 0; column < 3; ++column) {
			result.m[row][column] = inValue.m[row][column] / inScale;
		}
	}
	return result;
}

template <typename T>
[[nodiscard]] constexpr detail::Mat3<T> matrixMul(
	const detail::Mat3<T>& inA,
	const detail::Mat3<T>& inB) noexcept {
	detail::Mat3<T> result{};
	for (I32 row = 0; row < 3; ++row) {
		for (I32 column = 0; column < 3; ++column) {
			for (I32 inner = 0; inner < 3; ++inner) {
				result.m[row][column] += inA.m[row][inner] * inB.m[inner][column];
			}
		}
	}
	return result;
}

template <typename T>
[[nodiscard]] constexpr detail::Vec3<T> transform(
	const detail::Vec3<T>& inValue,
	const detail::Mat3<T>& inMatrix) noexcept {
	return {
		inValue.x * inMatrix.m[0][0] + inValue.y * inMatrix.m[1][0]
			+ inValue.z * inMatrix.m[2][0],
		inValue.x * inMatrix.m[0][1] + inValue.y * inMatrix.m[1][1]
			+ inValue.z * inMatrix.m[2][1],
		inValue.x * inMatrix.m[0][2] + inValue.y * inMatrix.m[1][2]
			+ inValue.z * inMatrix.m[2][2],
	};
}

template <typename T>
[[nodiscard]] constexpr detail::Mat4<T> add(
	const detail::Mat4<T>& inA,
	const detail::Mat4<T>& inB) noexcept {
	detail::Mat4<T> result{};
	for (I32 row = 0; row < 4; ++row) {
		for (I32 column = 0; column < 4; ++column) {
			result.m[row][column] = inA.m[row][column] + inB.m[row][column];
		}
	}
	return result;
}

template <typename T>
[[nodiscard]] constexpr detail::Mat4<T> sub(
	const detail::Mat4<T>& inA,
	const detail::Mat4<T>& inB) noexcept {
	detail::Mat4<T> result{};
	for (I32 row = 0; row < 4; ++row) {
		for (I32 column = 0; column < 4; ++column) {
			result.m[row][column] = inA.m[row][column] - inB.m[row][column];
		}
	}
	return result;
}

template <typename T>
[[nodiscard]] constexpr detail::Mat4<T> scale(
	const detail::Mat4<T>& inValue,
	T inScale) noexcept {
	detail::Mat4<T> result{};
	for (I32 row = 0; row < 4; ++row) {
		for (I32 column = 0; column < 4; ++column) {
			result.m[row][column] = inValue.m[row][column] * inScale;
		}
	}
	return result;
}

template <typename T>
[[nodiscard]] constexpr detail::Mat4<T> divide(
	const detail::Mat4<T>& inValue,
	T inScale) noexcept {
	detail::Mat4<T> result{};
	for (I32 row = 0; row < 4; ++row) {
		for (I32 column = 0; column < 4; ++column) {
			result.m[row][column] = inValue.m[row][column] / inScale;
		}
	}
	return result;
}

template <typename T>
[[nodiscard]] constexpr detail::Mat4<T> matrixMul(
	const detail::Mat4<T>& inA,
	const detail::Mat4<T>& inB) noexcept {
	detail::Mat4<T> result{};
	for (I32 row = 0; row < 4; ++row) {
		result.m[row][0] = inA.m[row][0] * inB.m[0][0]
			+ inA.m[row][1] * inB.m[1][0]
			+ inA.m[row][2] * inB.m[2][0]
			+ inA.m[row][3] * inB.m[3][0];
		result.m[row][1] = inA.m[row][0] * inB.m[0][1]
			+ inA.m[row][1] * inB.m[1][1]
			+ inA.m[row][2] * inB.m[2][1]
			+ inA.m[row][3] * inB.m[3][1];
		result.m[row][2] = inA.m[row][0] * inB.m[0][2]
			+ inA.m[row][1] * inB.m[1][2]
			+ inA.m[row][2] * inB.m[2][2]
			+ inA.m[row][3] * inB.m[3][2];
		result.m[row][3] = inA.m[row][0] * inB.m[0][3]
			+ inA.m[row][1] * inB.m[1][3]
			+ inA.m[row][2] * inB.m[2][3]
			+ inA.m[row][3] * inB.m[3][3];
	}
	return result;
}

template <typename T>
[[nodiscard]] constexpr detail::Vec4<T> transform(
	const detail::Vec4<T>& inValue,
	const detail::Mat4<T>& inMatrix) noexcept {
	return {
		inValue.x * inMatrix.m[0][0] + inValue.y * inMatrix.m[1][0]
			+ inValue.z * inMatrix.m[2][0] + inValue.w * inMatrix.m[3][0],
		inValue.x * inMatrix.m[0][1] + inValue.y * inMatrix.m[1][1]
			+ inValue.z * inMatrix.m[2][1] + inValue.w * inMatrix.m[3][1],
		inValue.x * inMatrix.m[0][2] + inValue.y * inMatrix.m[1][2]
			+ inValue.z * inMatrix.m[2][2] + inValue.w * inMatrix.m[3][2],
		inValue.x * inMatrix.m[0][3] + inValue.y * inMatrix.m[1][3]
			+ inValue.z * inMatrix.m[2][3] + inValue.w * inMatrix.m[3][3],
	};
}

using Mat3 = detail::Mat3<F32>;
using Mat4 = detail::Mat4<F32>;
using DMat3 = detail::Mat3<F64>;
using DMat4 = detail::Mat4<F64>;

} // namespace vlm

} // namespace oa
