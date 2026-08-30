#pragma once

// OA Vulkan linear-math vectors.
//
// These are scalar value types. They do not encode a coordinate space by
// themselves; OA's spatial contract is documented by <oa/core/vlm.h> and
// conversions belong at external-format boundaries.

#include <oa/core/math.h>

#include <oa/core/std/assert.h>
#include <oa/core/std/limits.h>
#include <oa/core/std/scalarMath.h>
#include <oa/core/std/typeTraits.h>

namespace oa {

namespace vlm {

namespace detail {
template <typename T> struct Vec2;
template <typename T> struct Vec3;
template <typename T> struct Vec4;
template <typename T>
[[nodiscard]] OA_NOINLINE Vec2<T> normalizeScaled(
	const Vec2<T>& inValue) noexcept;
template <typename T>
[[nodiscard]] OA_NOINLINE Vec3<T> normalizeScaled(
	const Vec3<T>& inValue) noexcept;
template <typename T>
[[nodiscard]] OA_NOINLINE Vec4<T> normalizeScaled(
	const Vec4<T>& inValue) noexcept;
} // namespace detail

template <typename T>
[[nodiscard]] constexpr detail::Vec2<T> add(const detail::Vec2<T>& inA,	const detail::Vec2<T>& inB) noexcept;

template <typename T>
[[nodiscard]] constexpr detail::Vec2<T> sub(const detail::Vec2<T>& inA,	const detail::Vec2<T>& inB) noexcept;

template <typename T>
[[nodiscard]] constexpr detail::Vec2<T> scale(const detail::Vec2<T>& inValue,	T inScale) noexcept;

template <typename T>
[[nodiscard]] constexpr detail::Vec2<T> divide(const detail::Vec2<T>& inValue, T inScale) noexcept;

template <typename T>
[[nodiscard]] constexpr detail::Vec3<T> add(const detail::Vec3<T>& inA,	const detail::Vec3<T>& inB) noexcept;

template <typename T>
[[nodiscard]] constexpr detail::Vec3<T> sub(const detail::Vec3<T>& inA,	const detail::Vec3<T>& inB) noexcept;

template <typename T>
[[nodiscard]] constexpr detail::Vec3<T> scale(const detail::Vec3<T>& inValue,	T inScale) noexcept;

template <typename T>
[[nodiscard]] constexpr detail::Vec3<T> divide(const detail::Vec3<T>& inValue, T inScale) noexcept;

template <typename T>
[[nodiscard]] constexpr detail::Vec4<T> add(const detail::Vec4<T>& inA,	const detail::Vec4<T>& inB) noexcept;

template <typename T>
[[nodiscard]] constexpr detail::Vec4<T> sub(const detail::Vec4<T>& inA,	const detail::Vec4<T>& inB) noexcept;

template <typename T>
[[nodiscard]] constexpr detail::Vec4<T> scale(const detail::Vec4<T>& inValue,	T inScale) noexcept;

template <typename T>
[[nodiscard]] constexpr detail::Vec4<T> divide(const detail::Vec4<T>& inValue, T inScale) noexcept;

namespace detail {

template <typename T>
struct Vec2 {
	static_assert(oa::IsFloatingPointV<T>);

	T x = T(0);
	T y = T(0);

	[[nodiscard]] constexpr T& at(Usize inIndex) noexcept {
		OA_REQUIRE(inIndex < 2U);
		return inIndex == 0U ? x : y;
	}
	[[nodiscard]] constexpr const T& at(Usize inIndex) const noexcept {
		OA_REQUIRE(inIndex < 2U);
		return inIndex == 0U ? x : y;
	}

	[[nodiscard]] bool isFinite() const noexcept {
		return oa::isFinite(x) and oa::isFinite(y);
	}
	[[nodiscard]] constexpr T lengthSquared() const noexcept {
		return x * x + y * y;
	}
	[[nodiscard]] T length() const noexcept {
		const T squaredLength = lengthSquared();
		if (oa::isFinite(squaredLength) and squaredLength > T(0)) {
			return oa::sqrt(squaredLength);
		}
		if (not isFinite()) return oa::sqrt(squaredLength);
		const T magnitude = oa::max(oa::abs(x), oa::abs(y));
		if (magnitude == T(0)) return T(0);
		const T sx = x / magnitude;
		const T sy = y / magnitude;
		return magnitude * oa::sqrt(sx * sx + sy * sy);
	}
	[[nodiscard]] Vec2 normalized() const noexcept {
		const T squaredLength = lengthSquared();
		if (OA_LIKELY(
			oa::isFinite(squaredLength) and squaredLength > T(0))) {
			return *this * (T(1) / oa::sqrt(squaredLength));
		}
		return normalizeScaled(*this);
	}

	[[nodiscard]] constexpr Vec2 operator-() const noexcept {
		return oa::vlm::scale(*this, T(-1));
	}
	[[nodiscard]] constexpr Vec2 operator+(const Vec2& inOther) const noexcept {
		return oa::vlm::add(*this, inOther);
	}
	[[nodiscard]] constexpr Vec2 operator-(const Vec2& inOther) const noexcept {
		return oa::vlm::sub(*this, inOther);
	}
	[[nodiscard]] constexpr Vec2 operator*(T inScale) const noexcept {
		return oa::vlm::scale(*this, inScale);
	}
	[[nodiscard]] constexpr Vec2 operator/(T inScale) const noexcept {
		return oa::vlm::divide(*this, inScale);
	}
	constexpr Vec2& operator+=(const Vec2& inOther) noexcept {
		*this = oa::vlm::add(*this, inOther);
		return *this;
	}
	constexpr Vec2& operator-=(const Vec2& inOther) noexcept {
		*this = oa::vlm::sub(*this, inOther);
		return *this;
	}
	constexpr Vec2& operator*=(T inScale) noexcept {
		*this = oa::vlm::scale(*this, inScale);
		return *this;
	}
	constexpr Vec2& operator/=(T inScale) noexcept {
		*this = oa::vlm::divide(*this, inScale);
		return *this;
	}

	[[nodiscard]] constexpr bool operator==(const Vec2& inOther) const noexcept = default;
};

template <typename T>
[[nodiscard]] OA_NOINLINE Vec2<T> normalizeScaled(
	const Vec2<T>& inValue) noexcept {
	const T magnitude = oa::max(oa::abs(inValue.x), oa::abs(inValue.y));
	OA_REQUIRE_MSG(
		inValue.isFinite() and magnitude > T(0),
		"VLM cannot normalize a zero-length or non-finite vector");
	const Vec2<T> scaled{inValue.x / magnitude, inValue.y / magnitude};
	const T scaledLength = scaled.length();
	OA_REQUIRE(oa::isFinite(scaledLength) and scaledLength > T(0));
	return scaled / scaledLength;
}

template <typename T>
[[nodiscard]] constexpr Vec2<T> operator*(
	T inScale,
	const Vec2<T>& inVector) noexcept {
	return inVector * inScale;
}

template <typename T>
struct Vec3 {
	static_assert(oa::IsFloatingPointV<T>);

	T x = T(0);
	T y = T(0);
	T z = T(0);

	[[nodiscard]] constexpr T& at(Usize inIndex) noexcept {
		OA_REQUIRE(inIndex < 3U);
		if (inIndex == 0U) return x;
		return inIndex == 1U ? y : z;
	}
	[[nodiscard]] constexpr const T& at(Usize inIndex) const noexcept {
		OA_REQUIRE(inIndex < 3U);
		if (inIndex == 0U) return x;
		return inIndex == 1U ? y : z;
	}

	[[nodiscard]] bool isFinite() const noexcept {
		return oa::isFinite(x) and oa::isFinite(y) and oa::isFinite(z);
	}
	[[nodiscard]] constexpr T lengthSquared() const noexcept {
		return x * x + y * y + z * z;
	}
	[[nodiscard]] T length() const noexcept {
		const T squaredLength = lengthSquared();
		if (oa::isFinite(squaredLength) and squaredLength > T(0)) {
			return oa::sqrt(squaredLength);
		}
		if (not isFinite()) return oa::sqrt(squaredLength);
		const T magnitude = oa::max(
			oa::abs(x), oa::max(oa::abs(y), oa::abs(z)));
		if (magnitude == T(0)) return T(0);
		const T sx = x / magnitude;
		const T sy = y / magnitude;
		const T sz = z / magnitude;
		return magnitude * oa::sqrt(sx * sx + sy * sy + sz * sz);
	}
	[[nodiscard]] Vec3 normalized() const noexcept {
		const T squaredLength = lengthSquared();
		if (OA_LIKELY(
			oa::isFinite(squaredLength) and squaredLength > T(0))) {
			return *this * (T(1) / oa::sqrt(squaredLength));
		}
		return normalizeScaled(*this);
	}

	[[nodiscard]] constexpr Vec3 operator-() const noexcept {
		return oa::vlm::scale(*this, T(-1));
	}
	[[nodiscard]] constexpr Vec3 operator+(const Vec3& inOther) const noexcept {
		return oa::vlm::add(*this, inOther);
	}
	[[nodiscard]] constexpr Vec3 operator-(const Vec3& inOther) const noexcept {
		return oa::vlm::sub(*this, inOther);
	}
	[[nodiscard]] constexpr Vec3 operator*(T inScale) const noexcept {
		return oa::vlm::scale(*this, inScale);
	}
	[[nodiscard]] constexpr Vec3 operator/(T inScale) const noexcept {
		return oa::vlm::divide(*this, inScale);
	}
	constexpr Vec3& operator+=(const Vec3& inOther) noexcept {
		*this = oa::vlm::add(*this, inOther);
		return *this;
	}
	constexpr Vec3& operator-=(const Vec3& inOther) noexcept {
		*this = oa::vlm::sub(*this, inOther);
		return *this;
	}
	constexpr Vec3& operator*=(T inScale) noexcept {
		*this = oa::vlm::scale(*this, inScale);
		return *this;
	}
	constexpr Vec3& operator/=(T inScale) noexcept {
		*this = oa::vlm::divide(*this, inScale);
		return *this;
	}

	[[nodiscard]] constexpr bool operator==(const Vec3& inOther) const noexcept = default;
};

template <typename T>
[[nodiscard]] OA_NOINLINE Vec3<T> normalizeScaled(
	const Vec3<T>& inValue) noexcept {
	const T magnitude = oa::max(
		oa::abs(inValue.x),
		oa::max(oa::abs(inValue.y), oa::abs(inValue.z)));
	OA_REQUIRE_MSG(
		inValue.isFinite() and magnitude > T(0),
		"VLM cannot normalize a zero-length or non-finite vector");
	const Vec3<T> scaled{
		inValue.x / magnitude, inValue.y / magnitude, inValue.z / magnitude};
	const T scaledLength = scaled.length();
	OA_REQUIRE(oa::isFinite(scaledLength) and scaledLength > T(0));
	return scaled / scaledLength;
}

template <typename T>
[[nodiscard]] constexpr Vec3<T> operator*(
	T inScale,
	const Vec3<T>& inVector) noexcept {
	return inVector * inScale;
}

template <typename T>
struct Vec4 {
	static_assert(oa::IsFloatingPointV<T>);

	T x = T(0);
	T y = T(0);
	T z = T(0);
	T w = T(0);

	[[nodiscard]] constexpr T& at(Usize inIndex) noexcept {
		OA_REQUIRE(inIndex < 4U);
		if (inIndex == 0U) return x;
		if (inIndex == 1U) return y;
		return inIndex == 2U ? z : w;
	}
	[[nodiscard]] constexpr const T& at(Usize inIndex) const noexcept {
		OA_REQUIRE(inIndex < 4U);
		if (inIndex == 0U) return x;
		if (inIndex == 1U) return y;
		return inIndex == 2U ? z : w;
	}

	[[nodiscard]] bool isFinite() const noexcept {
		return oa::isFinite(x) and oa::isFinite(y) and oa::isFinite(z) and oa::isFinite(w);
	}
	[[nodiscard]] constexpr T lengthSquared() const noexcept {
		return (x * x + y * y) + (z * z + w * w);
	}
	[[nodiscard]] T length() const noexcept {
		const T squaredLength = lengthSquared();
		if (oa::isFinite(squaredLength) and squaredLength > T(0)) {
			return oa::sqrt(squaredLength);
		}
		if (not isFinite()) return oa::sqrt(squaredLength);
		const T magnitude = oa::max(
			oa::max(oa::abs(x), oa::abs(y)),
			oa::max(oa::abs(z), oa::abs(w)));
		if (magnitude == T(0)) return T(0);
		const T sx = x / magnitude;
		const T sy = y / magnitude;
		const T sz = z / magnitude;
		const T sw = w / magnitude;
		return magnitude * oa::sqrt(
			sx * sx + sy * sy + sz * sz + sw * sw);
	}
	[[nodiscard]] Vec4 normalized() const noexcept {
		const T squaredLength = lengthSquared();
		if (OA_LIKELY(
			oa::isFinite(squaredLength) and squaredLength > T(0))) {
			return *this * (T(1) / oa::sqrt(squaredLength));
		}
		return normalizeScaled(*this);
	}

	[[nodiscard]] constexpr Vec4 operator-() const noexcept {
		return oa::vlm::scale(*this, T(-1));
	}
	[[nodiscard]] constexpr Vec4 operator+(const Vec4& inOther) const noexcept {
		return oa::vlm::add(*this, inOther);
	}
	[[nodiscard]] constexpr Vec4 operator-(const Vec4& inOther) const noexcept {
		return oa::vlm::sub(*this, inOther);
	}
	[[nodiscard]] constexpr Vec4 operator*(T inScale) const noexcept {
		return oa::vlm::scale(*this, inScale);
	}
	[[nodiscard]] constexpr Vec4 operator/(T inScale) const noexcept {
		return oa::vlm::divide(*this, inScale);
	}
	constexpr Vec4& operator+=(const Vec4& inOther) noexcept {
		*this = oa::vlm::add(*this, inOther);
		return *this;
	}
	constexpr Vec4& operator-=(const Vec4& inOther) noexcept {
		*this = oa::vlm::sub(*this, inOther);
		return *this;
	}
	constexpr Vec4& operator*=(T inScale) noexcept {
		*this = oa::vlm::scale(*this, inScale);
		return *this;
	}
	constexpr Vec4& operator/=(T inScale) noexcept {
		*this = oa::vlm::divide(*this, inScale);
		return *this;
	}

	[[nodiscard]] constexpr bool operator==(const Vec4& inOther) const noexcept = default;
};

template <typename T>
[[nodiscard]] OA_NOINLINE Vec4<T> normalizeScaled(
	const Vec4<T>& inValue) noexcept {
	const T magnitude = oa::max(
		oa::max(oa::abs(inValue.x), oa::abs(inValue.y)),
		oa::max(oa::abs(inValue.z), oa::abs(inValue.w)));
	OA_REQUIRE_MSG(
		inValue.isFinite() and magnitude > T(0),
		"VLM cannot normalize a zero-length or non-finite vector");
	const Vec4<T> scaled{
		inValue.x / magnitude, inValue.y / magnitude,
		inValue.z / magnitude, inValue.w / magnitude};
	const T scaledLength = scaled.length();
	OA_REQUIRE(oa::isFinite(scaledLength) and scaledLength > T(0));
	return scaled / scaledLength;
}

template <typename T>
[[nodiscard]] constexpr Vec4<T> operator*(
	T inScale,
	const Vec4<T>& inVector) noexcept {
	return oa::vlm::scale(inVector, inScale);
}

} // namespace detail

template <typename T>
[[nodiscard]] constexpr detail::Vec2<T> add(const detail::Vec2<T>& inA,	const detail::Vec2<T>& inB) noexcept {
	return {inA.x + inB.x, inA.y + inB.y};
}

template <typename T>
[[nodiscard]] constexpr detail::Vec2<T> sub(const detail::Vec2<T>& inA,	const detail::Vec2<T>& inB) noexcept {
	return {inA.x - inB.x, inA.y - inB.y};
}

template <typename T>
[[nodiscard]] constexpr detail::Vec2<T> scale(const detail::Vec2<T>& inValue,	T inScale) noexcept {
	return {inValue.x * inScale, inValue.y * inScale};
}

template <typename T>
[[nodiscard]] constexpr detail::Vec2<T> divide(const detail::Vec2<T>& inValue,
	T inScale) noexcept {
	return {inValue.x / inScale, inValue.y / inScale};
}

template <typename T>
[[nodiscard]] constexpr detail::Vec3<T> add(const detail::Vec3<T>& inA,	const detail::Vec3<T>& inB) noexcept {
	return {inA.x + inB.x, inA.y + inB.y, inA.z + inB.z};
}

template <typename T>
[[nodiscard]] constexpr detail::Vec3<T> sub(const detail::Vec3<T>& inA,	const detail::Vec3<T>& inB) noexcept {
	return {inA.x - inB.x, inA.y - inB.y, inA.z - inB.z};
}

template <typename T>
[[nodiscard]] constexpr detail::Vec3<T> scale(const detail::Vec3<T>& inValue, T inScale) noexcept {
	return {inValue.x * inScale, inValue.y * inScale, inValue.z * inScale};
}

template <typename T>
[[nodiscard]] constexpr detail::Vec3<T> divide(const detail::Vec3<T>& inValue, T inScale) noexcept {
	return {inValue.x / inScale, inValue.y / inScale, inValue.z / inScale};
}

template <typename T>
[[nodiscard]] constexpr detail::Vec4<T> add(const detail::Vec4<T>& inA,	const detail::Vec4<T>& inB) noexcept {
	return {
		inA.x + inB.x, inA.y + inB.y,
		inA.z + inB.z, inA.w + inB.w
	};
}

template <typename T>
[[nodiscard]] constexpr detail::Vec4<T> sub(const detail::Vec4<T>& inA,	const detail::Vec4<T>& inB) noexcept {
	return {
		inA.x - inB.x, inA.y - inB.y,
		inA.z - inB.z, inA.w - inB.w
	};
}

template <typename T>
[[nodiscard]] constexpr detail::Vec4<T> scale(const detail::Vec4<T>& inValue,	T inScale) noexcept {
	return {
		inValue.x * inScale, inValue.y * inScale,
		inValue.z * inScale, inValue.w * inScale
	};
}

template <typename T>
[[nodiscard]] constexpr detail::Vec4<T> divide(const detail::Vec4<T>& inValue, T inScale) noexcept {
	return {
		inValue.x / inScale, inValue.y / inScale,
		inValue.z / inScale, inValue.w / inScale
	};
}

using Vec2 = detail::Vec2<F32>;
using Vec3 = detail::Vec3<F32>;
using Vec4 = detail::Vec4<F32>;
using DVec2 = detail::Vec2<F64>;
using DVec3 = detail::Vec3<F64>;
using DVec4 = detail::Vec4<F64>;

} // namespace vlm

} // namespace oa
