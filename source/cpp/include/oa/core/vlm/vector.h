#pragma once

// OA Vulkan linear-math vectors.
//
// These are scalar value types. They do not encode a coordinate space by
// themselves; OA's spatial contract is documented by <oa/core/vlm.h> and
// conversions belong at external-format boundaries.

#include <oa/core/math.h>

#include <oa/core/std/limits.h>
#include <oa/core/std/scalarMath.h>
#include <oa/core/std/typeTraits.h>

namespace oa {

namespace vlm {

namespace detail {
template <typename T> struct Vec2;
template <typename T> struct Vec3;
template <typename T> struct Vec4;
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

	[[nodiscard]] bool isFinite() const noexcept {
		return oa::isFinite(x) and oa::isFinite(y);
	}
	[[nodiscard]] constexpr T lengthSquared() const noexcept {
		return x * x + y * y;
	}
	[[nodiscard]] T length() const noexcept {
		return oa::sqrt(lengthSquared());
	}
	[[nodiscard]] Vec2 normalized() const noexcept {
		const T valueLength = length();
		if (not oa::isFinite(valueLength) or valueLength <= oa::Limits<T>::epsilon()) {
			return {};
		}
		return *this / valueLength;
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

	[[nodiscard]] bool isFinite() const noexcept {
		return oa::isFinite(x) and oa::isFinite(y) and oa::isFinite(z);
	}
	[[nodiscard]] constexpr T lengthSquared() const noexcept {
		return x * x + y * y + z * z;
	}
	[[nodiscard]] T length() const noexcept {
		return oa::sqrt(lengthSquared());
	}
	[[nodiscard]] Vec3 normalized() const noexcept {
		const T valueLength = length();
		if (not oa::isFinite(valueLength) or valueLength <= oa::Limits<T>::epsilon()) {
			return {};
		}
		return *this / valueLength;
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

	[[nodiscard]] bool isFinite() const noexcept {
		return oa::isFinite(x) and oa::isFinite(y) and oa::isFinite(z) and oa::isFinite(w);
	}
	[[nodiscard]] constexpr T lengthSquared() const noexcept {
		return x * x + y * y + z * z + w * w;
	}
	[[nodiscard]] T length() const noexcept {
		return oa::sqrt(lengthSquared());
	}
	[[nodiscard]] Vec4 normalized() const noexcept {
		const T valueLength = length();
		if (not oa::isFinite(valueLength) or valueLength <= oa::Limits<T>::epsilon()) {
			return {};
		}
		return *this / valueLength;
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
