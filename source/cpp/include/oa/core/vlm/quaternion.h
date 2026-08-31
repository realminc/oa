#pragma once

#include <oa/core/vlm/vector.h>

#include <oa/core/std/assert.h>
#include <oa/core/std/limits.h>
#include <oa/core/std/scalarMath.h>
#include <oa/core/std/typeTraits.h>

namespace oa {

namespace vlm {

namespace detail {
template <typename T> struct Quat;
template <typename T>
[[nodiscard]] OA_NOINLINE Quat<T> normalizeScaled(
	const Quat<T>& inValue) noexcept;
} // namespace detail

template <typename T>
[[nodiscard]] constexpr detail::Quat<T> add(
	const detail::Quat<T>& inA,
	const detail::Quat<T>& inB) noexcept;
template <typename T>
[[nodiscard]] constexpr detail::Quat<T> sub(
	const detail::Quat<T>& inA,
	const detail::Quat<T>& inB) noexcept;
template <typename T>
[[nodiscard]] constexpr detail::Quat<T> scale(
	const detail::Quat<T>& inValue,
	T inScale) noexcept;
template <typename T>
[[nodiscard]] constexpr detail::Quat<T> divide(
	const detail::Quat<T>& inValue,
	T inScale) noexcept;
template <typename T>
[[nodiscard]] constexpr detail::Quat<T> quaternionMul(
	const detail::Quat<T>& inA,
	const detail::Quat<T>& inB) noexcept;

namespace detail {

template <typename T>
struct Quat {
	static_assert(oa::isFloatingPointV<T>);

	// Vector-first storage matches shader and animation payloads: (x, y, z, w).
	T x = T(0);
	T y = T(0);
	T z = T(0);
	T w = T(1);

	[[nodiscard]] static constexpr Quat identity() noexcept { return {}; }
	[[nodiscard]] static Quat fromAxisAngle(
		const detail::Vec3<T>& inAxis,
		T inRadians) noexcept {
		const T axisMagnitude = oa::max(
			oa::abs(inAxis.x),
			oa::max(oa::abs(inAxis.y), oa::abs(inAxis.z)));
		OA_REQUIRE_MSG(
			inAxis.isFinite() and axisMagnitude > T(0)
				and oa::isFinite(inRadians),
			"VLM axis-angle requires a finite non-zero axis and finite angle");
		const detail::Vec3<T> axis = inAxis.normalized();
		const T halfAngle = inRadians * T(0.5);
		const T sine = oa::sin(halfAngle);
		return {axis.x * sine, axis.y * sine, axis.z * sine, oa::cos(halfAngle)};
	}

	[[nodiscard]] bool isFinite() const noexcept {
		return oa::isFinite(x) and oa::isFinite(y) and oa::isFinite(z) and oa::isFinite(w);
	}
	[[nodiscard]] constexpr T normSquared() const noexcept {
		return (x * x + y * y) + (z * z + w * w);
	}
	[[nodiscard]] T norm() const noexcept {
		const T squaredNorm = normSquared();
		if (oa::isFinite(squaredNorm) and squaredNorm > T(0)) {
			return oa::sqrt(squaredNorm);
		}
		if (not isFinite()) return oa::sqrt(squaredNorm);
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
	[[nodiscard]] Quat normalized() const noexcept {
		const T squaredNorm = normSquared();
		if (OA_LIKELY(oa::isFinite(squaredNorm) and squaredNorm > T(0))) {
			return *this * (T(1) / oa::sqrt(squaredNorm));
		}
		return normalizeScaled(*this);
	}
	[[nodiscard]] constexpr Quat conjugate() const noexcept {
		return {-x, -y, -z, w};
	}
	[[nodiscard]] detail::Vec3<T> rotate(
		const detail::Vec3<T>& inVector) const noexcept {
		const T squaredNorm = normSquared();
		const T unitTolerance = oa::Limits<T>::epsilon() * T(8);
		const Quat unit = oa::isFinite(squaredNorm)
			and oa::abs(squaredNorm - T(1)) <= unitTolerance
			? *this
			: normalized();
		const detail::Vec3<T> imaginary{unit.x, unit.y, unit.z};
		const detail::Vec3<T> firstCross{
			imaginary.y * inVector.z - imaginary.z * inVector.y,
			imaginary.z * inVector.x - imaginary.x * inVector.z,
			imaginary.x * inVector.y - imaginary.y * inVector.x,
		};
		const detail::Vec3<T> twiceCross = T(2) * firstCross;
		const detail::Vec3<T> secondCross{
			imaginary.y * twiceCross.z - imaginary.z * twiceCross.y,
			imaginary.z * twiceCross.x - imaginary.x * twiceCross.z,
			imaginary.x * twiceCross.y - imaginary.y * twiceCross.x,
		};
		return inVector + unit.w * twiceCross + secondCross;
	}
	[[nodiscard]] detail::Vec3<T> inverseRotate(const detail::Vec3<T>& inVector) const noexcept {
		return conjugate().normalized().rotate(inVector);
	}

	[[nodiscard]] constexpr Quat operator-() const noexcept {
		return oa::vlm::scale(*this, T(-1));
	}
	[[nodiscard]] constexpr Quat operator+(const Quat& inOther) const noexcept {
		return oa::vlm::add(*this, inOther);
	}
	[[nodiscard]] constexpr Quat operator-(const Quat& inOther) const noexcept {
		return oa::vlm::sub(*this, inOther);
	}
	[[nodiscard]] constexpr Quat operator*(T inScale) const noexcept {
		return oa::vlm::scale(*this, inScale);
	}
	[[nodiscard]] constexpr Quat operator/(T inScale) const noexcept {
		return oa::vlm::divide(*this, inScale);
	}
	[[nodiscard]] constexpr Quat operator*(const Quat& inOther) const noexcept {
		return oa::vlm::quaternionMul(*this, inOther);
	}
	constexpr Quat& operator+=(const Quat& inOther) noexcept {
		*this = oa::vlm::add(*this, inOther);
		return *this;
	}
	constexpr Quat& operator-=(const Quat& inOther) noexcept {
		*this = oa::vlm::sub(*this, inOther);
		return *this;
	}
	constexpr Quat& operator*=(T inScale) noexcept {
		*this = oa::vlm::scale(*this, inScale);
		return *this;
	}
	constexpr Quat& operator/=(T inScale) noexcept {
		*this = oa::vlm::divide(*this, inScale);
		return *this;
	}
	constexpr Quat& operator*=(const Quat& inOther) noexcept {
		*this = oa::vlm::quaternionMul(*this, inOther);
		return *this;
	}

	[[nodiscard]] constexpr bool operator==(
		const Quat& inOther) const noexcept = default;
};

template <typename T>
[[nodiscard]] OA_NOINLINE Quat<T> normalizeScaled(
	const Quat<T>& inValue) noexcept {
	const T magnitude = oa::max(
		oa::max(oa::abs(inValue.x), oa::abs(inValue.y)),
		oa::max(oa::abs(inValue.z), oa::abs(inValue.w)));
	OA_REQUIRE_MSG(
		inValue.isFinite() and magnitude > T(0),
		"VLM cannot normalize a zero-length or non-finite quaternion");
	const Quat<T> scaled{
		inValue.x / magnitude, inValue.y / magnitude,
		inValue.z / magnitude, inValue.w / magnitude};
	const T scaledNorm = scaled.norm();
	OA_REQUIRE(oa::isFinite(scaledNorm) and scaledNorm > T(0));
	return scaled / scaledNorm;
}

template <typename T>
[[nodiscard]] constexpr Quat<T> operator*(
	T inScale,
	const Quat<T>& inQuaternion) noexcept {
	return oa::vlm::scale(inQuaternion, inScale);
}

} // namespace detail

template <typename T>
[[nodiscard]] constexpr detail::Quat<T> add(
	const detail::Quat<T>& inA,
	const detail::Quat<T>& inB) noexcept {
	return {
		inA.x + inB.x, inA.y + inB.y,
		inA.z + inB.z, inA.w + inB.w};
}

template <typename T>
[[nodiscard]] constexpr detail::Quat<T> sub(
	const detail::Quat<T>& inA,
	const detail::Quat<T>& inB) noexcept {
	return {
		inA.x - inB.x, inA.y - inB.y,
		inA.z - inB.z, inA.w - inB.w};
}

template <typename T>
[[nodiscard]] constexpr detail::Quat<T> scale(
	const detail::Quat<T>& inValue,
	T inScale) noexcept {
	return {
		inValue.x * inScale, inValue.y * inScale,
		inValue.z * inScale, inValue.w * inScale};
}

template <typename T>
[[nodiscard]] constexpr detail::Quat<T> divide(
	const detail::Quat<T>& inValue,
	T inScale) noexcept {
	return {
		inValue.x / inScale, inValue.y / inScale,
		inValue.z / inScale, inValue.w / inScale};
}

template <typename T>
[[nodiscard]] constexpr detail::Quat<T> quaternionMul(
	const detail::Quat<T>& inA,
	const detail::Quat<T>& inB) noexcept {
	return {
		inA.w * inB.x + inA.x * inB.w + inA.y * inB.z - inA.z * inB.y,
		inA.w * inB.y - inA.x * inB.z + inA.y * inB.w + inA.z * inB.x,
		inA.w * inB.z + inA.x * inB.y - inA.y * inB.x + inA.z * inB.w,
		inA.w * inB.w - inA.x * inB.x - inA.y * inB.y - inA.z * inB.z,
	};
}

using Quat = detail::Quat<F32>;
using DQuat = detail::Quat<F64>;

} // namespace vlm

} // namespace oa
