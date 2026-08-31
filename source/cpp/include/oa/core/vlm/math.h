#pragma once

#include <oa/core/vlm/matrix.h>

#include <oa/core/std/algo.h>
#include <oa/core/std/limits.h>
#include <oa/core/std/scalarMath.h>
#include <oa/core/std/typeTraits.h>
#include <oa/core/std/utility.h>

namespace oa {

namespace vlm {

template <typename T>
inline constexpr T Pi = T(3.141592653589793238462643383279502884L);

template <typename T>
inline constexpr T Tolerance = oa::isSameV<T, F32> ? T(1.0e-5F) : T(1.0e-12);

template <typename T>
inline constexpr T InverseTolerance = oa::Limits<T>::epsilon() * T(32);

template <typename T>
[[nodiscard]] constexpr bool isValidTolerance(T inTolerance) noexcept {
	return oa::isFinite(inTolerance) and inTolerance >= T(0);
}

template <typename T>
[[nodiscard]] constexpr T radians(T inDegrees) noexcept {
	return inDegrees * Pi<T> / T(180);
}

template <typename T>
[[nodiscard]] constexpr T degrees(T inRadians) noexcept {
	return inRadians * T(180) / Pi<T>;
}

template <typename T>
[[nodiscard]] constexpr T dot(
	const detail::Vec2<T>& inA,
	const detail::Vec2<T>& inB
) noexcept {
	return inA.x * inB.x + inA.y * inB.y;
}

template <typename T>
[[nodiscard]] constexpr T dot(
	const detail::Vec3<T>& inA,
	const detail::Vec3<T>& inB
) noexcept {
	return inA.x * inB.x + inA.y * inB.y + inA.z * inB.z;
}

template <typename T>
[[nodiscard]] constexpr T dot(
	const detail::Vec4<T>& inA,
	const detail::Vec4<T>& inB
) noexcept {
	return (inA.x * inB.x + inA.y * inB.y)
		+ (inA.z * inB.z + inA.w * inB.w);
}

template <typename T>
[[nodiscard]] constexpr detail::Vec3<T> cross(
	const detail::Vec3<T>& inA,
	const detail::Vec3<T>& inB
) noexcept {
	return {
		inA.y * inB.z - inA.z * inB.y,
		inA.z * inB.x - inA.x * inB.z,
		inA.x * inB.y - inA.y * inB.x,
	};
}

template <typename T>
[[nodiscard]] constexpr T lengthSquared(const detail::Vec2<T>& inValue) noexcept {
	return inValue.lengthSquared();
}

template <typename T>
[[nodiscard]] constexpr T lengthSquared(const detail::Vec3<T>& inValue) noexcept {
	return inValue.lengthSquared();
}

template <typename T>
[[nodiscard]] constexpr T lengthSquared(const detail::Vec4<T>& inValue) noexcept {
	return inValue.lengthSquared();
}

template <typename T>
[[nodiscard]] T length(const detail::Vec2<T>& inValue) noexcept {
	return inValue.length();
}

template <typename T>
[[nodiscard]] T length(const detail::Vec3<T>& inValue) noexcept {
	return inValue.length();
}

template <typename T>
[[nodiscard]] T length(const detail::Vec4<T>& inValue) noexcept {
	return inValue.length();
}

template <typename T>
[[nodiscard]] detail::Vec2<T> normalize(const detail::Vec2<T>& inValue) noexcept {
	return inValue.normalized();
}

template <typename T>
[[nodiscard]] detail::Vec3<T> normalize(const detail::Vec3<T>& inValue) noexcept {
	return inValue.normalized();
}

template <typename T>
[[nodiscard]] detail::Vec4<T> normalize(const detail::Vec4<T>& inValue) noexcept {
	return inValue.normalized();
}

template <typename T>
[[nodiscard]] bool tryNormalize(
	const detail::Vec2<T>& inValue,
	detail::Vec2<T>& outNormalized) noexcept {
	const T squaredLength = inValue.lengthSquared();
	if (oa::isFinite(squaredLength) and squaredLength > T(0)) {
		const detail::Vec2<T> result =
			inValue * (T(1) / oa::sqrt(squaredLength));
		outNormalized = result;
		return true;
	}
	const T magnitude = oa::max(oa::abs(inValue.x), oa::abs(inValue.y));
	if (not inValue.isFinite() or magnitude <= T(0)) return false;
	const detail::Vec2<T> scaled = inValue / magnitude;
	const T scaledLength = length(scaled);
	if (not oa::isFinite(scaledLength) or scaledLength <= T(0)) return false;
	const detail::Vec2<T> result = scaled / scaledLength;
	if (not result.isFinite()) return false;
	outNormalized = result;
	return true;
}

template <typename T>
[[nodiscard]] bool tryNormalize(
	const detail::Vec3<T>& inValue,
	detail::Vec3<T>& outNormalized) noexcept {
	const T squaredLength = inValue.lengthSquared();
	if (oa::isFinite(squaredLength) and squaredLength > T(0)) {
		const detail::Vec3<T> result =
			inValue * (T(1) / oa::sqrt(squaredLength));
		outNormalized = result;
		return true;
	}
	const T magnitude = oa::max(
		oa::abs(inValue.x), oa::max(oa::abs(inValue.y), oa::abs(inValue.z)));
	if (not inValue.isFinite() or magnitude <= T(0)) return false;
	const detail::Vec3<T> scaled = inValue / magnitude;
	const T scaledLength = length(scaled);
	if (not oa::isFinite(scaledLength) or scaledLength <= T(0)) return false;
	const detail::Vec3<T> result = scaled / scaledLength;
	if (not result.isFinite()) return false;
	outNormalized = result;
	return true;
}

template <typename T>
[[nodiscard]] bool tryNormalize(
	const detail::Vec4<T>& inValue,
	detail::Vec4<T>& outNormalized) noexcept {
	const T squaredLength = inValue.lengthSquared();
	if (oa::isFinite(squaredLength) and squaredLength > T(0)) {
		const detail::Vec4<T> result =
			inValue * (T(1) / oa::sqrt(squaredLength));
		outNormalized = result;
		return true;
	}
	const T magnitude = oa::max(
		oa::max(oa::abs(inValue.x), oa::abs(inValue.y)),
		oa::max(oa::abs(inValue.z), oa::abs(inValue.w)));
	if (not inValue.isFinite() or magnitude <= T(0)) return false;
	const detail::Vec4<T> scaled = inValue / magnitude;
	const T scaledLength = length(scaled);
	if (not oa::isFinite(scaledLength) or scaledLength <= T(0)) return false;
	const detail::Vec4<T> result = scaled / scaledLength;
	if (not result.isFinite()) return false;
	outNormalized = result;
	return true;
}

namespace detail {

template <typename T>
[[nodiscard]] bool hasLengthAboveTolerance(
	const Vec3<T>& inValue,
	T inTolerance) noexcept {
	const T squaredLength = inValue.lengthSquared();
	const T squaredTolerance = inTolerance * inTolerance;
	if (oa::isFinite(squaredLength) and oa::isFinite(squaredTolerance)) {
		return squaredLength > squaredTolerance;
	}
	const T magnitude = oa::max(
		oa::abs(inValue.x),
		oa::max(oa::abs(inValue.y), oa::abs(inValue.z)));
	if (not inValue.isFinite() or magnitude <= T(0)) return false;
	const Vec3<T> scaled = inValue / magnitude;
	const T scaledLength = scaled.length();
	return oa::isFinite(scaledLength) and scaledLength > T(0)
		and magnitude > inTolerance / scaledLength;
}

template <typename T>
[[nodiscard]] bool tryNormalizeAboveTolerance(
	const Vec3<T>& inValue,
	Vec3<T>& outNormalized,
	T inTolerance) noexcept {
	const T squaredLength = inValue.lengthSquared();
	const T squaredTolerance = inTolerance * inTolerance;
	if (oa::isFinite(squaredLength) and oa::isFinite(squaredTolerance)) {
		if (squaredLength <= squaredTolerance) return false;
		outNormalized = inValue * (T(1) / oa::sqrt(squaredLength));
		return true;
	}
	if (not hasLengthAboveTolerance(inValue, inTolerance)) return false;
	return tryNormalize(inValue, outNormalized);
}

} // namespace detail

template <typename T>
[[nodiscard]] constexpr detail::Vec2<T> lerp(
	const detail::Vec2<T>& inA,
	const detail::Vec2<T>& inB,
	T inT
) noexcept {
	const T inverseT = T(1) - inT;
	return {
		inA.x * inverseT + inB.x * inT,
		inA.y * inverseT + inB.y * inT,
	};
}

template <typename T>
[[nodiscard]] constexpr detail::Vec3<T> lerp(
	const detail::Vec3<T>& inA,
	const detail::Vec3<T>& inB,
	T inT
) noexcept {
	const T inverseT = T(1) - inT;
	return {
		inA.x * inverseT + inB.x * inT,
		inA.y * inverseT + inB.y * inT,
		inA.z * inverseT + inB.z * inT,
	};
}

template <typename T>
[[nodiscard]] constexpr detail::Vec4<T> lerp(
	const detail::Vec4<T>& inA,
	const detail::Vec4<T>& inB,
	T inT
) noexcept {
	const T inverseT = T(1) - inT;
	return {
		inA.x * inverseT + inB.x * inT,
		inA.y * inverseT + inB.y * inT,
		inA.z * inverseT + inB.z * inT,
		inA.w * inverseT + inB.w * inT,
	};
}

template <typename T>
[[nodiscard]] constexpr T distanceSquared(
	const detail::Vec2<T>& inA,
	const detail::Vec2<T>& inB
) noexcept {
	return lengthSquared(sub(inB, inA));
}

template <typename T>
[[nodiscard]] constexpr T distanceSquared(
	const detail::Vec3<T>& inA,
	const detail::Vec3<T>& inB
) noexcept {
	return lengthSquared(sub(inB, inA));
}

template <typename T>
[[nodiscard]] constexpr T distanceSquared(
	const detail::Vec4<T>& inA,
	const detail::Vec4<T>& inB
) noexcept {
	return lengthSquared(sub(inB, inA));
}

template <typename T>
[[nodiscard]] T distance(
	const detail::Vec2<T>& inA,
	const detail::Vec2<T>& inB
) noexcept {
	return oa::sqrt(distanceSquared(inA, inB));
}

template <typename T>
[[nodiscard]] T distance(
	const detail::Vec3<T>& inA,
	const detail::Vec3<T>& inB
) noexcept {
	return oa::sqrt(distanceSquared(inA, inB));
}

template <typename T>
[[nodiscard]] T distance(
	const detail::Vec4<T>& inA,
	const detail::Vec4<T>& inB
) noexcept {
	return oa::sqrt(distanceSquared(inA, inB));
}

template <typename T>
[[nodiscard]] constexpr detail::Vec3<T> reflect(
	const detail::Vec3<T>& inIncident,
	const detail::Vec3<T>& inNormal
) noexcept {
	return sub(inIncident, scale(inNormal, T(2) * dot(inNormal, inIncident)));
}

template <typename T>
[[nodiscard]] detail::Vec3<T> refract(
	const detail::Vec3<T>& inIncident,
	const detail::Vec3<T>& inNormal,
	T inEta
) noexcept {
	const T normalDotIncident = dot(inNormal, inIncident);
	const T discriminant = T(1) - inEta * inEta
		* (T(1) - normalDotIncident * normalDotIncident);
	if (discriminant < T(0)) return {};
	return sub(
		scale(inIncident, inEta),
		scale(inNormal, inEta * normalDotIncident + oa::sqrt(discriminant)));
}

template <typename T>
[[nodiscard]] constexpr detail::Vec3<T> faceForward(
	const detail::Vec3<T>& inNormal,
	const detail::Vec3<T>& inIncident,
	const detail::Vec3<T>& inReferenceNormal
) noexcept {
	return dot(inReferenceNormal, inIncident) < T(0) ? inNormal : -inNormal;
}

template <typename T>
[[nodiscard]] constexpr detail::Vec2<T> abs(
	const detail::Vec2<T>& inValue
) noexcept {
	return {
		inValue.x < T(0) ? -inValue.x : inValue.x,
		inValue.y < T(0) ? -inValue.y : inValue.y,
	};
}

template <typename T>
[[nodiscard]] constexpr detail::Vec3<T> abs(
	const detail::Vec3<T>& inValue
) noexcept {
	return {
		inValue.x < T(0) ? -inValue.x : inValue.x,
		inValue.y < T(0) ? -inValue.y : inValue.y,
		inValue.z < T(0) ? -inValue.z : inValue.z,
	};
}

template <typename T>
[[nodiscard]] constexpr detail::Vec4<T> abs(
	const detail::Vec4<T>& inValue
) noexcept {
	return {
		inValue.x < T(0) ? -inValue.x : inValue.x,
		inValue.y < T(0) ? -inValue.y : inValue.y,
		inValue.z < T(0) ? -inValue.z : inValue.z,
		inValue.w < T(0) ? -inValue.w : inValue.w,
	};
}

template <typename T>
[[nodiscard]] constexpr detail::Vec2<T> min(
	const detail::Vec2<T>& inA,
	const detail::Vec2<T>& inB) noexcept {
	return {
		inA.x < inB.x ? inA.x : inB.x,
		inA.y < inB.y ? inA.y : inB.y,
	};
}

template <typename T>
[[nodiscard]] constexpr detail::Vec3<T> min(
	const detail::Vec3<T>& inA,
	const detail::Vec3<T>& inB
) noexcept {
	return {
		inA.x < inB.x ? inA.x : inB.x,
		inA.y < inB.y ? inA.y : inB.y,
		inA.z < inB.z ? inA.z : inB.z,
	};
}

template <typename T>
[[nodiscard]] constexpr detail::Vec4<T> min(
	const detail::Vec4<T>& inA,
	const detail::Vec4<T>& inB) noexcept {
	return {
		inA.x < inB.x ? inA.x : inB.x,
		inA.y < inB.y ? inA.y : inB.y,
		inA.z < inB.z ? inA.z : inB.z,
		inA.w < inB.w ? inA.w : inB.w,
	};
}

template <typename T>
[[nodiscard]] constexpr detail::Vec2<T> max(
	const detail::Vec2<T>& inA,
	const detail::Vec2<T>& inB) noexcept {
	return {
		inA.x > inB.x ? inA.x : inB.x,
		inA.y > inB.y ? inA.y : inB.y,
	};
}

template <typename T>
[[nodiscard]] constexpr detail::Vec3<T> max(
	const detail::Vec3<T>& inA,
	const detail::Vec3<T>& inB
) noexcept {
	return {
		inA.x > inB.x ? inA.x : inB.x,
		inA.y > inB.y ? inA.y : inB.y,
		inA.z > inB.z ? inA.z : inB.z,
	};
}

template <typename T>
[[nodiscard]] constexpr detail::Vec4<T> max(
	const detail::Vec4<T>& inA,
	const detail::Vec4<T>& inB) noexcept {
	return {
		inA.x > inB.x ? inA.x : inB.x,
		inA.y > inB.y ? inA.y : inB.y,
		inA.z > inB.z ? inA.z : inB.z,
		inA.w > inB.w ? inA.w : inB.w,
	};
}

template <typename T>
[[nodiscard]] constexpr detail::Vec2<T> clamp(
	const detail::Vec2<T>& inValue,
	const detail::Vec2<T>& inMinimum,
	const detail::Vec2<T>& inMaximum) noexcept {
	return min(max(inValue, inMinimum), inMaximum);
}

template <typename T>
[[nodiscard]] constexpr detail::Vec3<T> clamp(
	const detail::Vec3<T>& inValue,
	const detail::Vec3<T>& inMinimum,
	const detail::Vec3<T>& inMaximum
) noexcept {
	return min(max(inValue, inMinimum), inMaximum);
}

template <typename T>
[[nodiscard]] constexpr detail::Vec4<T> clamp(
	const detail::Vec4<T>& inValue,
	const detail::Vec4<T>& inMinimum,
	const detail::Vec4<T>& inMaximum) noexcept {
	return min(max(inValue, inMinimum), inMaximum);
}

template <typename T>
[[nodiscard]] constexpr detail::Vec2<T> componentMul(
	const detail::Vec2<T>& inA,
	const detail::Vec2<T>& inB) noexcept {
	return {inA.x * inB.x, inA.y * inB.y};
}

template <typename T>
[[nodiscard]] constexpr detail::Vec3<T> componentMul(
	const detail::Vec3<T>& inA,
	const detail::Vec3<T>& inB) noexcept {
	return {inA.x * inB.x, inA.y * inB.y, inA.z * inB.z};
}

template <typename T>
[[nodiscard]] constexpr detail::Vec4<T> componentMul(
	const detail::Vec4<T>& inA,
	const detail::Vec4<T>& inB) noexcept {
	return {
		inA.x * inB.x, inA.y * inB.y,
		inA.z * inB.z, inA.w * inB.w};
}

template <typename T>
[[nodiscard]] constexpr detail::Vec2<T> componentDivide(
	const detail::Vec2<T>& inA,
	const detail::Vec2<T>& inB) noexcept {
	return {inA.x / inB.x, inA.y / inB.y};
}

template <typename T>
[[nodiscard]] constexpr detail::Vec3<T> componentDivide(
	const detail::Vec3<T>& inA,
	const detail::Vec3<T>& inB) noexcept {
	return {inA.x / inB.x, inA.y / inB.y, inA.z / inB.z};
}

template <typename T>
[[nodiscard]] constexpr detail::Vec4<T> componentDivide(
	const detail::Vec4<T>& inA,
	const detail::Vec4<T>& inB) noexcept {
	return {
		inA.x / inB.x, inA.y / inB.y,
		inA.z / inB.z, inA.w / inB.w};
}

template <typename T>
[[nodiscard]] constexpr T componentMin(const detail::Vec2<T>& inValue) noexcept {
	return oa::min(inValue.x, inValue.y);
}

template <typename T>
[[nodiscard]] constexpr T componentMin(const detail::Vec3<T>& inValue) noexcept {
	return oa::min(inValue.x, oa::min(inValue.y, inValue.z));
}

template <typename T>
[[nodiscard]] constexpr T componentMin(const detail::Vec4<T>& inValue) noexcept {
	return oa::min(oa::min(inValue.x, inValue.y), oa::min(inValue.z, inValue.w));
}

template <typename T>
[[nodiscard]] constexpr T componentMax(const detail::Vec2<T>& inValue) noexcept {
	return oa::max(inValue.x, inValue.y);
}

template <typename T>
[[nodiscard]] constexpr T componentMax(const detail::Vec3<T>& inValue) noexcept {
	return oa::max(inValue.x, oa::max(inValue.y, inValue.z));
}

template <typename T>
[[nodiscard]] constexpr T componentMax(const detail::Vec4<T>& inValue) noexcept {
	return oa::max(oa::max(inValue.x, inValue.y), oa::max(inValue.z, inValue.w));
}

template <typename T>
[[nodiscard]] bool tryProjectVector(
	const detail::Vec3<T>& inValue,
	const detail::Vec3<T>& inOnto,
	detail::Vec3<T>& outProjection,
	T inTolerance = Tolerance<T>) noexcept {
	if (not isValidTolerance(inTolerance)) return false;
	const T denominator = dot(inOnto, inOnto);
	const T squaredTolerance = inTolerance * inTolerance;
	if (oa::isFinite(denominator) and denominator > squaredTolerance
		and oa::isFinite(squaredTolerance)) {
		const T factor = dot(inValue, inOnto) / denominator;
		const detail::Vec3<T> projection = inOnto * factor;
		if (oa::isFinite(factor) and projection.isFinite()) {
			outProjection = projection;
			return true;
		}
	}
	if (not inValue.isFinite()) return false;
	if (not detail::hasLengthAboveTolerance(inOnto, inTolerance)) return false;
	detail::Vec3<T> onto{};
	if (not tryNormalize(inOnto, onto)) return false;
	const T valueMagnitude = oa::max(
		oa::abs(inValue.x),
		oa::max(oa::abs(inValue.y), oa::abs(inValue.z)));
	detail::Vec3<T> projection{};
	if (valueMagnitude > T(0)) {
		const detail::Vec3<T> scaledValue = inValue / valueMagnitude;
		projection = (onto * dot(scaledValue, onto)) * valueMagnitude;
	}
	if (not projection.isFinite()) return false;
	outProjection = projection;
	return true;
}

template <typename T>
[[nodiscard]] bool tryRejectVector(
	const detail::Vec3<T>& inValue,
	const detail::Vec3<T>& inFrom,
	detail::Vec3<T>& outRejection,
	T inTolerance = Tolerance<T>) noexcept {
	detail::Vec3<T> projection{};
	if (not tryProjectVector(inValue, inFrom, projection, inTolerance)) {
		return false;
	}
	const detail::Vec3<T> rejection = inValue - projection;
	if (not rejection.isFinite()) return false;
	outRejection = rejection;
	return true;
}

template <typename T>
[[nodiscard]] bool tryAngleBetween(
	const detail::Vec3<T>& inA,
	const detail::Vec3<T>& inB,
	T& outRadians,
	T inTolerance = Tolerance<T>) noexcept {
	if (not isValidTolerance(inTolerance)) return false;
	const T squaredA = inA.lengthSquared();
	const T squaredB = inB.lengthSquared();
	const T squaredTolerance = inTolerance * inTolerance;
	const T squaredProduct = squaredA * squaredB;
	if (oa::isFinite(squaredA) and oa::isFinite(squaredB)
		and oa::isFinite(squaredTolerance) and squaredA > squaredTolerance
		and squaredB > squaredTolerance and oa::isFinite(squaredProduct)
		and squaredProduct > T(0)) {
		const T cosine = dot(inA, inB) / oa::sqrt(squaredProduct);
		const T angle = oa::acos(oa::clamp(cosine, T(-1), T(1)));
		if (oa::isFinite(cosine) and oa::isFinite(angle)) {
			outRadians = angle;
			return true;
		}
	}
	if (not detail::hasLengthAboveTolerance(inA, inTolerance)
		or not detail::hasLengthAboveTolerance(inB, inTolerance)) return false;
	detail::Vec3<T> a{};
	detail::Vec3<T> b{};
	if (not tryNormalize(inA, a) or not tryNormalize(inB, b)) return false;
	const T angle = oa::acos(oa::clamp(
		dot(a, b), T(-1), T(1)));
	if (not oa::isFinite(angle)) return false;
	outRadians = angle;
	return true;
}

template <typename T>
[[nodiscard]] bool tryPerpendicular(
	const detail::Vec3<T>& inDirection,
	detail::Vec3<T>& outPerpendicular,
	T inTolerance = Tolerance<T>) noexcept {
	if (not isValidTolerance(inTolerance)
		or not detail::hasLengthAboveTolerance(inDirection, inTolerance)) {
		return false;
	}
	detail::Vec3<T> direction{};
	if (not tryNormalize(inDirection, direction)) return false;
	const detail::Vec3<T> candidate =
		oa::abs(direction.x) <= oa::abs(direction.y)
		and oa::abs(direction.x) <= oa::abs(direction.z)
		? detail::Vec3<T>{T(1), T(0), T(0)}
		: (oa::abs(direction.y) <= oa::abs(direction.z)
			? detail::Vec3<T>{T(0), T(1), T(0)}
			: detail::Vec3<T>{T(0), T(0), T(1)});
	const detail::Vec3<T> perpendicularValue = cross(direction, candidate);
	return tryNormalize(perpendicularValue, outPerpendicular);
}

template <typename T>
[[nodiscard]] bool trySignedAngleBetween(
	const detail::Vec3<T>& inA,
	const detail::Vec3<T>& inB,
	const detail::Vec3<T>& inAxis,
	T& outRadians,
	T inTolerance = Tolerance<T>) noexcept {
	if (not isValidTolerance(inTolerance)
		or not detail::hasLengthAboveTolerance(inA, inTolerance)
		or not detail::hasLengthAboveTolerance(inB, inTolerance)
		or not detail::hasLengthAboveTolerance(inAxis, inTolerance)) {
		return false;
	}
	detail::Vec3<T> a{};
	detail::Vec3<T> b{};
	detail::Vec3<T> axis{};
	if (not tryNormalize(inA, a) or not tryNormalize(inB, b)
		or not tryNormalize(inAxis, axis)) {
		return false;
	}
	const T sine = dot(axis, cross(a, b));
	const T cosine = oa::clamp(dot(a, b), T(-1), T(1));
	const T angle = oa::atan2(sine, cosine);
	if (not oa::isFinite(angle)) return false;
	outRadians = angle;
	return true;
}

template <typename T>
[[nodiscard]] constexpr bool approximatelyEqual(
	T inA,
	T inB,
	T inAbsoluteTolerance = Tolerance<T>,
	T inRelativeTolerance = Tolerance<T>
) noexcept {
	const T difference = inA >= inB ? inA - inB : inB - inA;
	const T magnitudeA = inA >= T(0) ? inA : -inA;
	const T magnitudeB = inB >= T(0) ? inB : -inB;
	const T scaleValue = magnitudeA > magnitudeB ? magnitudeA : magnitudeB;
	return difference <= inAbsoluteTolerance
		or difference <= inRelativeTolerance * scaleValue;
}

template <typename T>
[[nodiscard]] constexpr bool approximatelyEqual(
	const detail::Vec2<T>& inA,
	const detail::Vec2<T>& inB,
	T inAbsoluteTolerance = Tolerance<T>,
	T inRelativeTolerance = Tolerance<T>
) noexcept {
	return approximatelyEqual(inA.x, inB.x, inAbsoluteTolerance, inRelativeTolerance)
		and approximatelyEqual(inA.y, inB.y, inAbsoluteTolerance, inRelativeTolerance);
}

template <typename T>
[[nodiscard]] constexpr bool approximatelyEqual(
	const detail::Vec3<T>& inA,
	const detail::Vec3<T>& inB,
	T inAbsoluteTolerance = Tolerance<T>,
	T inRelativeTolerance = Tolerance<T>
) noexcept {
	return approximatelyEqual(inA.x, inB.x, inAbsoluteTolerance, inRelativeTolerance)
		and approximatelyEqual(inA.y, inB.y, inAbsoluteTolerance, inRelativeTolerance)
		and approximatelyEqual(inA.z, inB.z, inAbsoluteTolerance, inRelativeTolerance);
}

template <typename T>
[[nodiscard]] constexpr bool approximatelyEqual(
	const detail::Vec4<T>& inA,
	const detail::Vec4<T>& inB,
	T inAbsoluteTolerance = Tolerance<T>,
	T inRelativeTolerance = Tolerance<T>
) noexcept {
	return approximatelyEqual(inA.x, inB.x, inAbsoluteTolerance, inRelativeTolerance)
		and approximatelyEqual(inA.y, inB.y, inAbsoluteTolerance, inRelativeTolerance)
		and approximatelyEqual(inA.z, inB.z, inAbsoluteTolerance, inRelativeTolerance)
		and approximatelyEqual(inA.w, inB.w, inAbsoluteTolerance, inRelativeTolerance);
}

template <typename T>
[[nodiscard]] constexpr bool approximatelyEqual(
	const detail::Quat<T>& inA,
	const detail::Quat<T>& inB,
	T inAbsoluteTolerance = Tolerance<T>,
	T inRelativeTolerance = Tolerance<T>
) noexcept {
	return approximatelyEqual(inA.x, inB.x, inAbsoluteTolerance, inRelativeTolerance)
		and approximatelyEqual(inA.y, inB.y, inAbsoluteTolerance, inRelativeTolerance)
		and approximatelyEqual(inA.z, inB.z, inAbsoluteTolerance, inRelativeTolerance)
		and approximatelyEqual(inA.w, inB.w, inAbsoluteTolerance, inRelativeTolerance);
}

template <typename T>
[[nodiscard]] constexpr bool approximatelyEqual(
	const detail::Mat4<T>& inA,
	const detail::Mat4<T>& inB,
	T inAbsoluteTolerance = Tolerance<T>,
	T inRelativeTolerance = Tolerance<T>
) noexcept {
	for (I32 row = 0; row < 4; ++row) {
		for (I32 column = 0; column < 4; ++column) {
			if (not approximatelyEqual(
				inA.m[row][column], inB.m[row][column],
				inAbsoluteTolerance, inRelativeTolerance)) {
				return false;
			}
		}
	}
	return true;
}

[[nodiscard]] inline constexpr Quat quaternionIdentity() noexcept {
	return Quat::identity();
}

template <typename T>
[[nodiscard]] bool tryNormalize(
	const detail::Quat<T>& inQuaternion,
	detail::Quat<T>& outNormalized) noexcept {
	const T squaredNorm = inQuaternion.normSquared();
	if (oa::isFinite(squaredNorm) and squaredNorm > T(0)) {
		const detail::Quat<T> result =
			inQuaternion * (T(1) / oa::sqrt(squaredNorm));
		outNormalized = result;
		return true;
	}
	const T magnitude = oa::max(
		oa::max(oa::abs(inQuaternion.x), oa::abs(inQuaternion.y)),
		oa::max(oa::abs(inQuaternion.z), oa::abs(inQuaternion.w))
	);
	if (not inQuaternion.isFinite() or magnitude <= T(0)) return false;
	const detail::Quat<T> scaled = inQuaternion / magnitude;
	const T scaledNorm = scaled.norm();
	if (not oa::isFinite(scaledNorm) or scaledNorm <= T(0)) return false;
	const detail::Quat<T> result = scaled / scaledNorm;
	if (not result.isFinite()) return false;
	outNormalized = result;
	return true;
}

template <typename T>
[[nodiscard]] bool tryQuaternionFromAxisAngle(
	const detail::Vec3<T>& inAxis,
	T inRadians,
	detail::Quat<T>& outRotation) noexcept {
	detail::Vec3<T> axis{};
	if (not oa::isFinite(inRadians) or not tryNormalize(inAxis, axis)) {
		return false;
	}
	const T halfAngle = inRadians * T(0.5);
	const T sine = oa::sin(halfAngle);
	const detail::Quat<T> rotation{
		axis.x * sine, axis.y * sine, axis.z * sine, oa::cos(halfAngle)};
	if (not rotation.isFinite()) return false;
	outRotation = rotation;
	return true;
}

template <typename T>
[[nodiscard]] detail::Quat<T> quaternionFromAxisAngle(
	const detail::Vec3<T>& inAxis,
	T inRadians) noexcept {
	detail::Quat<T> result{};
	const bool valid = tryQuaternionFromAxisAngle(inAxis, inRadians, result);
	OA_REQUIRE_MSG(
		valid,
		"VLM axis-angle requires a finite non-zero axis and finite angle");
	return result;
}

enum class RotationOrder : oa::U8 {
	Xyz,
	Xzy,
	Yxz,
	Yzx,
	Zxy,
	Zyx,
};

[[nodiscard]] constexpr bool isValidRotationOrder(RotationOrder inOrder) noexcept {
	switch (inOrder) {
		case RotationOrder::Xyz:
		case RotationOrder::Xzy:
		case RotationOrder::Yxz:
		case RotationOrder::Yzx:
		case RotationOrder::Zxy:
		case RotationOrder::Zyx: return true;
	}
	return false;
}

template <typename T>
[[nodiscard]] detail::Quat<T> quaternionFromEulerRadians(
	const detail::Vec3<T>& inAngles,
	RotationOrder inOrder) noexcept {
	OA_REQUIRE_MSG(inAngles.isFinite(), "VLM Euler angles must be finite");
	OA_REQUIRE_MSG(isValidRotationOrder(inOrder), "VLM rotation order is invalid");
	const detail::Quat<T> x = quaternionFromAxisAngle(
		{T(1), T(0), T(0)}, inAngles.x);
	const detail::Quat<T> y = quaternionFromAxisAngle(
		{T(0), T(1), T(0)}, inAngles.y);
	const detail::Quat<T> z = quaternionFromAxisAngle(
		{T(0), T(0), T(1)}, inAngles.z);
	switch (inOrder) {
		case RotationOrder::Xyz: return z * (y * x);
		case RotationOrder::Xzy: return y * (z * x);
		case RotationOrder::Yxz: return z * (x * y);
		case RotationOrder::Yzx: return x * (z * y);
		case RotationOrder::Zxy: return y * (x * z);
		case RotationOrder::Zyx: return x * (y * z);
	}
	OA_REQUIRE_MSG(false, "VLM rotation order is invalid");
	return detail::Quat<T>::identity();
}

template <typename T>
[[nodiscard]] detail::Quat<T> quaternionFromEulerDegrees(
	const detail::Vec3<T>& inAngles,
	RotationOrder inOrder) noexcept {
	return quaternionFromEulerRadians(
		detail::Vec3<T>{
			radians(inAngles.x),
			radians(inAngles.y),
			radians(inAngles.z)
		},
		inOrder);
}

template <typename T>
[[nodiscard]] detail::Vec3<T> quaternionToEulerRadians(
	const detail::Quat<T>& inQuaternion,
	RotationOrder inOrder) noexcept {
	OA_REQUIRE_MSG(isValidRotationOrder(inOrder), "VLM rotation order is invalid");
	const detail::Quat<T> q = inQuaternion.normalized();
	const T xx = q.x * q.x;
	const T yy = q.y * q.y;
	const T zz = q.z * q.z;
	const T xy = q.x * q.y;
	const T xz = q.x * q.z;
	const T yz = q.y * q.z;
	const T xw = q.x * q.w;
	const T yw = q.y * q.w;
	const T zw = q.z * q.w;
	// Conventional column-vector rotation elements. OA's row-vector matrix is
	// their transpose; calculating them directly avoids a convention switch.
	const T m11 = T(1) - T(2) * (yy + zz);
	const T m12 = T(2) * (xy - zw);
	const T m13 = T(2) * (xz + yw);
	const T m21 = T(2) * (xy + zw);
	const T m22 = T(1) - T(2) * (xx + zz);
	const T m23 = T(2) * (yz - xw);
	const T m31 = T(2) * (xz - yw);
	const T m32 = T(2) * (yz + xw);
	const T m33 = T(1) - T(2) * (xx + yy);
	const T lock = T(1) - oa::Limits<T>::epsilon() * T(64);
	detail::Vec3<T> result{};
	switch (inOrder) {
		case RotationOrder::Zyx:
			result.y = oa::asin(oa::clamp(m13, T(-1), T(1)));
			if (oa::abs(m13) < lock) {
				result.x = oa::atan2(-m23, m33);
				result.z = oa::atan2(-m12, m11);
			} else {
				result.x = oa::atan2(m32, m22);
			}
			break;
		case RotationOrder::Zxy:
			result.x = oa::asin(-oa::clamp(m23, T(-1), T(1)));
			if (oa::abs(m23) < lock) {
				result.y = oa::atan2(m13, m33);
				result.z = oa::atan2(m21, m22);
			} else {
				result.y = oa::atan2(-m31, m11);
			}
			break;
		case RotationOrder::Yxz:
			result.x = oa::asin(oa::clamp(m32, T(-1), T(1)));
			if (oa::abs(m32) < lock) {
				result.y = oa::atan2(-m31, m33);
				result.z = oa::atan2(-m12, m22);
			} else {
				result.z = oa::atan2(m21, m11);
			}
			break;
		case RotationOrder::Xyz:
			result.y = oa::asin(-oa::clamp(m31, T(-1), T(1)));
			if (oa::abs(m31) < lock) {
				result.x = oa::atan2(m32, m33);
				result.z = oa::atan2(m21, m11);
			} else {
				result.z = oa::atan2(-m12, m22);
			}
			break;
		case RotationOrder::Xzy:
			result.z = oa::asin(oa::clamp(m21, T(-1), T(1)));
			if (oa::abs(m21) < lock) {
				result.x = oa::atan2(-m23, m22);
				result.y = oa::atan2(-m31, m11);
			} else {
				result.y = oa::atan2(m13, m33);
			}
			break;
		case RotationOrder::Yzx:
			result.z = oa::asin(-oa::clamp(m12, T(-1), T(1)));
			if (oa::abs(m12) < lock) {
				result.x = oa::atan2(m32, m22);
				result.y = oa::atan2(m13, m11);
			} else {
				result.x = oa::atan2(-m23, m33);
			}
			break;
	}
	return result;
}

template <typename T>
[[nodiscard]] detail::Vec3<T> quaternionToEulerDegrees(
	const detail::Quat<T>& inQuaternion,
	RotationOrder inOrder) noexcept {
	const detail::Vec3<T> value = quaternionToEulerRadians(
		inQuaternion, inOrder);
	return {degrees(value.x), degrees(value.y), degrees(value.z)};
}

// Compatibility camera spelling: yaw Z, pitch Y, roll X, applied X/Y/Z.
template <typename T>
[[nodiscard]] detail::Quat<T> quaternionFromEuler(
	T inYawDeg,
	T inPitchDeg,
	T inRollDeg) noexcept {
	return quaternionFromEulerDegrees(
		detail::Vec3<T>{inRollDeg, inPitchDeg, inYawDeg},
		RotationOrder::Xyz);
}

template <typename T>
[[nodiscard]] detail::Vec3<T> quaternionToEuler(
	const detail::Quat<T>& inQuaternion) noexcept {
	const detail::Vec3<T> xyz = quaternionToEulerDegrees(
		inQuaternion, RotationOrder::Xyz);
	return {xyz.z, xyz.y, xyz.x};
}

template <typename T>
[[nodiscard]] constexpr T quaternionDot(
	const detail::Quat<T>& inA,
	const detail::Quat<T>& inB
) noexcept {
	return inA.x * inB.x + inA.y * inB.y
		+ inA.z * inB.z + inA.w * inB.w;
}

template <typename T>
[[nodiscard]] bool tryInverse(
	const detail::Quat<T>& inQuaternion,
	detail::Quat<T>& outInverse,
	T inTolerance = InverseTolerance<T>
) noexcept {
	if (not isValidTolerance(inTolerance) or not inQuaternion.isFinite()) {
		return false;
	}
	const T magnitude = oa::max(
		oa::max(oa::abs(inQuaternion.x), oa::abs(inQuaternion.y)),
		oa::max(oa::abs(inQuaternion.z), oa::abs(inQuaternion.w)));
	if (magnitude <= inTolerance) return false;
	const detail::Quat<T> scaled = inQuaternion / magnitude;
	const T scaledNormSquared = scaled.normSquared();
	if (not oa::isFinite(scaledNormSquared)
		or scaledNormSquared <= T(0)) {
		return false;
	}
	const detail::Quat<T> inverse =
		(scaled.conjugate() / scaledNormSquared) / magnitude;
	if (not inverse.isFinite()) return false;
	outInverse = inverse;
	return true;
}

template <typename T>
[[nodiscard]] detail::Quat<T> nlerp(
	const detail::Quat<T>& inA,
	const detail::Quat<T>& inB,
	T inT
) noexcept {
	const T unitTolerance = oa::Limits<T>::epsilon() * T(8);
	const T squaredA = inA.normSquared();
	const T squaredB = inB.normSquared();
	const detail::Quat<T> a = oa::isFinite(squaredA)
		and oa::abs(squaredA - T(1)) <= unitTolerance
		? inA
		: inA.normalized();
	detail::Quat<T> b = oa::isFinite(squaredB)
		and oa::abs(squaredB - T(1)) <= unitTolerance
		? inB
		: inB.normalized();
	if (quaternionDot(a, b) < T(0)) b = -b;
	return (a + (b - a) * inT).normalized();
}

template <typename T>
[[nodiscard]] detail::Quat<T> slerp(
	const detail::Quat<T>& inA,
	const detail::Quat<T>& inB,
	T inT
) noexcept {
	const T unitTolerance = oa::Limits<T>::epsilon() * T(8);
	const T squaredA = inA.normSquared();
	const T squaredB = inB.normSquared();
	const detail::Quat<T> a = oa::isFinite(squaredA)
		and oa::abs(squaredA - T(1)) <= unitTolerance
		? inA
		: inA.normalized();
	detail::Quat<T> b = oa::isFinite(squaredB)
		and oa::abs(squaredB - T(1)) <= unitTolerance
		? inB
		: inB.normalized();
	T cosine = quaternionDot(a, b);
	if (cosine < T(0)) {
		b = -b;
		cosine = -cosine;
	}
	cosine = oa::clamp(cosine, T(-1), T(1));
	if (cosine > T(1) - Tolerance<T>) return nlerp(a, b, inT);
	const T angle = oa::acos(cosine);
	const T sine = oa::sin(angle);
	const T weightA = oa::sin((T(1) - inT) * angle) / sine;
	const T weightB = oa::sin(inT * angle) / sine;
	return (a * weightA + b * weightB).normalized();
}

template <typename T>
[[nodiscard]] detail::Mat4<T> quaternionToMatrix(const detail::Quat<T>& inQuaternion) noexcept {
	const T squaredNorm = inQuaternion.normSquared();
	const T unitTolerance = oa::Limits<T>::epsilon() * T(8);
	const detail::Quat<T> q = oa::isFinite(squaredNorm)
		and oa::abs(squaredNorm - T(1)) <= unitTolerance
		? inQuaternion
		: inQuaternion.normalized();
	const T xx = q.x * q.x;
	const T yy = q.y * q.y;
	const T zz = q.z * q.z;
	const T xy = q.x * q.y;
	const T xz = q.x * q.z;
	const T yz = q.y * q.z;
	const T xw = q.x * q.w;
	const T yw = q.y * q.w;
	const T zw = q.z * q.w;
	detail::Mat4<T> result = detail::Mat4<T>::identity();
	// Transposed from the conventional column-vector form so v * M and
	// quaternion.rotate(v) produce the same direction.
	result.m[0][0] = T(1) - T(2) * (yy + zz);
	result.m[0][1] = T(2) * (xy + zw);
	result.m[0][2] = T(2) * (xz - yw);
	result.m[1][0] = T(2) * (xy - zw);
	result.m[1][1] = T(1) - T(2) * (xx + zz);
	result.m[1][2] = T(2) * (yz + xw);
	result.m[2][0] = T(2) * (xz + yw);
	result.m[2][1] = T(2) * (yz - xw);
	result.m[2][2] = T(1) - T(2) * (xx + yy);
	return result;
}

namespace detail {

template <typename T>
[[nodiscard]] detail::Quat<T> quaternionFromRotationMatrixUnchecked(
	const detail::Mat4<T>& inMatrix) noexcept {
	// The standard extraction is expressed for column-vector matrices. OA's
	// row-vector rotation is its transpose, so read through that transpose here.
	const auto at = [&inMatrix](I32 inRow, I32 inColumn) noexcept -> T {
		return inMatrix.m[inColumn][inRow];
	};
	const T m00 = at(0, 0);
	const T m11 = at(1, 1);
	const T m22 = at(2, 2);
	const T trace = m00 + m11 + m22;
	detail::Quat<T> result;
	if (trace > T(0)) {
		const T scale = oa::sqrt(trace + T(1)) * T(2);
		result.w = T(0.25) * scale;
		result.x = (at(2, 1) - at(1, 2)) / scale;
		result.y = (at(0, 2) - at(2, 0)) / scale;
		result.z = (at(1, 0) - at(0, 1)) / scale;
	} else if (m00 > m11 and m00 > m22) {
		const T scale = oa::sqrt(T(1) + m00 - m11 - m22) * T(2);
		result.w = (at(2, 1) - at(1, 2)) / scale;
		result.x = T(0.25) * scale;
		result.y = (at(0, 1) + at(1, 0)) / scale;
		result.z = (at(0, 2) + at(2, 0)) / scale;
	} else if (m11 > m22) {
		const T scale = oa::sqrt(T(1) + m11 - m00 - m22) * T(2);
		result.w = (at(0, 2) - at(2, 0)) / scale;
		result.x = (at(0, 1) + at(1, 0)) / scale;
		result.y = T(0.25) * scale;
		result.z = (at(1, 2) + at(2, 1)) / scale;
	} else {
		const T scale = oa::sqrt(T(1) + m22 - m00 - m11) * T(2);
		result.w = (at(1, 0) - at(0, 1)) / scale;
		result.x = (at(0, 2) + at(2, 0)) / scale;
		result.y = (at(1, 2) + at(2, 1)) / scale;
		result.z = T(0.25) * scale;
	}
	return result.normalized();
}

} // namespace detail

template <typename T>
[[nodiscard]] detail::Vec3<T> rotateVector(
	const detail::Quat<T>& inQuaternion,
	const detail::Vec3<T>& inVector
) noexcept {
	return inQuaternion.rotate(inVector);
}

[[nodiscard]] inline constexpr Mat4 matrixIdentity() noexcept {
	return Mat4::identity();
}

template <typename T>
[[nodiscard]] constexpr detail::Mat4<T> transpose(const detail::Mat4<T>& inMatrix) noexcept {
	detail::Mat4<T> result{};
	for (I32 row = 0; row < 4; ++row) {
		for (I32 col = 0; col < 4; ++col) {
			result.m[row][col] = inMatrix.m[col][row];
		}
	}
	return result;
}

namespace detail {

template <typename T>
[[nodiscard]] bool tryInverseMat3(
	const Mat3<T>& inMatrix,
	Mat3<T>& outInverse,
	T inTolerance) noexcept {
	if (not isValidTolerance(inTolerance)) return false;
	T magnitude = T(0);
	for (I32 row = 0; row < 3; ++row) {
		for (I32 column = 0; column < 3; ++column) {
			const T value = inMatrix.m[row][column];
			if (not oa::isFinite(value)) return false;
			magnitude = oa::max(magnitude, oa::abs(value));
		}
	}
	if (magnitude <= T(0)) return false;
	const T directDeterminant =
		inMatrix.m[0][0]
			* (inMatrix.m[1][1] * inMatrix.m[2][2]
				- inMatrix.m[1][2] * inMatrix.m[2][1])
		- inMatrix.m[0][1]
			* (inMatrix.m[1][0] * inMatrix.m[2][2]
				- inMatrix.m[1][2] * inMatrix.m[2][0])
		+ inMatrix.m[0][2]
			* (inMatrix.m[1][0] * inMatrix.m[2][1]
				- inMatrix.m[1][1] * inMatrix.m[2][0]);
	const T scaledDirectDeterminant =
		directDeterminant / magnitude / magnitude / magnitude;
	if (oa::isFinite(directDeterminant)
		and oa::isFinite(scaledDirectDeterminant)
		and directDeterminant != T(0)
		and scaledDirectDeterminant != T(0)) {
		if (oa::abs(scaledDirectDeterminant) <= inTolerance) return false;
		const T inverseDeterminant = T(1) / directDeterminant;
		Mat3<T> directInverse{};
		directInverse.m[0][0] = (inMatrix.m[1][1] * inMatrix.m[2][2]
			- inMatrix.m[1][2] * inMatrix.m[2][1]) * inverseDeterminant;
		directInverse.m[0][1] = (inMatrix.m[0][2] * inMatrix.m[2][1]
			- inMatrix.m[0][1] * inMatrix.m[2][2]) * inverseDeterminant;
		directInverse.m[0][2] = (inMatrix.m[0][1] * inMatrix.m[1][2]
			- inMatrix.m[0][2] * inMatrix.m[1][1]) * inverseDeterminant;
		directInverse.m[1][0] = (inMatrix.m[1][2] * inMatrix.m[2][0]
			- inMatrix.m[1][0] * inMatrix.m[2][2]) * inverseDeterminant;
		directInverse.m[1][1] = (inMatrix.m[0][0] * inMatrix.m[2][2]
			- inMatrix.m[0][2] * inMatrix.m[2][0]) * inverseDeterminant;
		directInverse.m[1][2] = (inMatrix.m[0][2] * inMatrix.m[1][0]
			- inMatrix.m[0][0] * inMatrix.m[1][2]) * inverseDeterminant;
		directInverse.m[2][0] = (inMatrix.m[1][0] * inMatrix.m[2][1]
			- inMatrix.m[1][1] * inMatrix.m[2][0]) * inverseDeterminant;
		directInverse.m[2][1] = (inMatrix.m[0][1] * inMatrix.m[2][0]
			- inMatrix.m[0][0] * inMatrix.m[2][1]) * inverseDeterminant;
		directInverse.m[2][2] = (inMatrix.m[0][0] * inMatrix.m[1][1]
			- inMatrix.m[0][1] * inMatrix.m[1][0]) * inverseDeterminant;

		T matrixNorm = T(0);
		T inverseNorm = T(0);
		bool directFinite = true;
		for (I32 row = 0; row < 3; ++row) {
			T matrixRowSum = T(0);
			T inverseRowSum = T(0);
			for (I32 column = 0; column < 3; ++column) {
				directFinite = directFinite
					and oa::isFinite(directInverse.m[row][column]);
				matrixRowSum += oa::abs(inMatrix.m[row][column]);
				inverseRowSum += oa::abs(directInverse.m[row][column]);
			}
			matrixNorm = oa::max(matrixNorm, matrixRowSum);
			inverseNorm = oa::max(inverseNorm, inverseRowSum);
		}
		const T condition = matrixNorm * inverseNorm;
		if (directFinite and oa::isFinite(condition) and condition > T(0)) {
			if (T(1) / condition <= inTolerance) return false;
			outInverse = directInverse;
			return true;
		}
	}
	Mat3<T> scaled{};
	for (I32 row = 0; row < 3; ++row) {
		for (I32 column = 0; column < 3; ++column) {
			scaled.m[row][column] = inMatrix.m[row][column] / magnitude;
		}
	}
	const T valueDeterminant =
		scaled.m[0][0]
			* (scaled.m[1][1] * scaled.m[2][2]
				- scaled.m[1][2] * scaled.m[2][1])
		- scaled.m[0][1]
			* (scaled.m[1][0] * scaled.m[2][2]
				- scaled.m[1][2] * scaled.m[2][0])
		+ scaled.m[0][2]
			* (scaled.m[1][0] * scaled.m[2][1]
				- scaled.m[1][1] * scaled.m[2][0]);
	if (not oa::isFinite(valueDeterminant)
		or oa::abs(valueDeterminant) <= inTolerance) {
		return false;
	}
	const T inverseDeterminant = T(1) / valueDeterminant;
	Mat3<T> inverse{};
	inverse.m[0][0] = (scaled.m[1][1] * scaled.m[2][2]
		- scaled.m[1][2] * scaled.m[2][1]) * inverseDeterminant / magnitude;
	inverse.m[0][1] = (scaled.m[0][2] * scaled.m[2][1]
		- scaled.m[0][1] * scaled.m[2][2]) * inverseDeterminant / magnitude;
	inverse.m[0][2] = (scaled.m[0][1] * scaled.m[1][2]
		- scaled.m[0][2] * scaled.m[1][1]) * inverseDeterminant / magnitude;
	inverse.m[1][0] = (scaled.m[1][2] * scaled.m[2][0]
		- scaled.m[1][0] * scaled.m[2][2]) * inverseDeterminant / magnitude;
	inverse.m[1][1] = (scaled.m[0][0] * scaled.m[2][2]
		- scaled.m[0][2] * scaled.m[2][0]) * inverseDeterminant / magnitude;
	inverse.m[1][2] = (scaled.m[0][2] * scaled.m[1][0]
		- scaled.m[0][0] * scaled.m[1][2]) * inverseDeterminant / magnitude;
	inverse.m[2][0] = (scaled.m[1][0] * scaled.m[2][1]
		- scaled.m[1][1] * scaled.m[2][0]) * inverseDeterminant / magnitude;
	inverse.m[2][1] = (scaled.m[0][1] * scaled.m[2][0]
		- scaled.m[0][0] * scaled.m[2][1]) * inverseDeterminant / magnitude;
	inverse.m[2][2] = (scaled.m[0][0] * scaled.m[1][1]
		- scaled.m[0][1] * scaled.m[1][0]) * inverseDeterminant / magnitude;

	T matrixNorm = T(0);
	T inverseNorm = T(0);
	for (I32 row = 0; row < 3; ++row) {
		T matrixRowSum = T(0);
		T inverseRowSum = T(0);
		for (I32 column = 0; column < 3; ++column) {
			if (not oa::isFinite(inverse.m[row][column])) return false;
			matrixRowSum += oa::abs(inMatrix.m[row][column]);
			inverseRowSum += oa::abs(inverse.m[row][column]);
		}
		matrixNorm = oa::max(matrixNorm, matrixRowSum);
		inverseNorm = oa::max(inverseNorm, inverseRowSum);
	}
	const T condition = matrixNorm * inverseNorm;
	if (not oa::isFinite(condition) or condition <= T(0)
		or T(1) / condition <= inTolerance) {
		return false;
	}
	outInverse = inverse;
	return true;
}

} // namespace detail

namespace detail {

template <typename T>
[[nodiscard]] OA_NOINLINE T determinantScaledFallback(
	const Mat4<T>& inMatrix) noexcept {
	Mat4<T> reduced = inMatrix;
	T result = T(1);
	I32 sign = 1;
	for (I32 pivotColumn = 0; pivotColumn < 4; ++pivotColumn) {
		I32 pivotRow = pivotColumn;
		T pivotMagnitude = oa::abs(reduced.m[pivotRow][pivotColumn]);
		for (I32 row = pivotColumn + 1; row < 4; ++row) {
			const T magnitude = oa::abs(reduced.m[row][pivotColumn]);
			if (magnitude > pivotMagnitude) {
				pivotMagnitude = magnitude;
				pivotRow = row;
			}
		}
		if (not oa::isFinite(pivotMagnitude)) {
			return oa::Limits<T>::quietNaN();
		}
		if (pivotMagnitude == T(0)) return T(0);
		if (pivotRow != pivotColumn) {
			for (I32 column = 0; column < 4; ++column) {
				oa::swapValues(
					reduced.m[pivotRow][column],
					reduced.m[pivotColumn][column]);
			}
			sign = -sign;
		}
		const T pivot = reduced.m[pivotColumn][pivotColumn];
		result *= pivot;
		for (I32 row = pivotColumn + 1; row < 4; ++row) {
			const T factor = reduced.m[row][pivotColumn] / pivot;
			for (I32 column = pivotColumn + 1; column < 4; ++column) {
				reduced.m[row][column] -=
					factor * reduced.m[pivotColumn][column];
			}
		}
	}
	return sign < 0 ? -result : result;
}

} // namespace detail

template <typename T>
[[nodiscard]] T determinant(const detail::Mat4<T>& inMatrix) noexcept {
	const T subFactor00 = inMatrix.m[2][2] * inMatrix.m[3][3]
		- inMatrix.m[3][2] * inMatrix.m[2][3];
	const T subFactor01 = inMatrix.m[2][1] * inMatrix.m[3][3]
		- inMatrix.m[3][1] * inMatrix.m[2][3];
	const T subFactor02 = inMatrix.m[2][1] * inMatrix.m[3][2]
		- inMatrix.m[3][1] * inMatrix.m[2][2];
	const T subFactor03 = inMatrix.m[2][0] * inMatrix.m[3][3]
		- inMatrix.m[3][0] * inMatrix.m[2][3];
	const T subFactor04 = inMatrix.m[2][0] * inMatrix.m[3][2]
		- inMatrix.m[3][0] * inMatrix.m[2][2];
	const T subFactor05 = inMatrix.m[2][0] * inMatrix.m[3][1]
		- inMatrix.m[3][0] * inMatrix.m[2][1];
	const T cofactor0 = inMatrix.m[1][1] * subFactor00
		- inMatrix.m[1][2] * subFactor01
		+ inMatrix.m[1][3] * subFactor02;
	const T cofactor1 = -(inMatrix.m[1][0] * subFactor00
		- inMatrix.m[1][2] * subFactor03
		+ inMatrix.m[1][3] * subFactor04);
	const T cofactor2 = inMatrix.m[1][0] * subFactor01
		- inMatrix.m[1][1] * subFactor03
		+ inMatrix.m[1][3] * subFactor05;
	const T cofactor3 = -(inMatrix.m[1][0] * subFactor02
		- inMatrix.m[1][1] * subFactor04
		+ inMatrix.m[1][2] * subFactor05);
	const T direct = inMatrix.m[0][0] * cofactor0
		+ inMatrix.m[0][1] * cofactor1
		+ inMatrix.m[0][2] * cofactor2
		+ inMatrix.m[0][3] * cofactor3;
	if (oa::isFinite(direct)) return direct;
	return detail::determinantScaledFallback(inMatrix);
}

template <typename T>
[[nodiscard]] bool tryInverse(
	const detail::Mat4<T>& inMatrix,
	detail::Mat4<T>& outInverse,
	T inTolerance = InverseTolerance<T>
) noexcept {
	if (not isValidTolerance(inTolerance)) return false;
	const bool affine = inMatrix.m[0][3] == T(0)
		and inMatrix.m[1][3] == T(0)
		and inMatrix.m[2][3] == T(0)
		and inMatrix.m[3][3] == T(1);
	if (affine) {
		detail::Mat3<T> linear{};
		for (I32 row = 0; row < 3; ++row) {
			for (I32 column = 0; column < 3; ++column) {
				linear.m[row][column] = inMatrix.m[row][column];
			}
		}
		detail::Mat3<T> inverseLinear{};
		if (not detail::tryInverseMat3(
			linear, inverseLinear, inTolerance)) {
			return false;
		}
		const detail::Vec3<T> translationValue{
			inMatrix.m[3][0], inMatrix.m[3][1], inMatrix.m[3][2]};
		const detail::Vec3<T> inverseTranslation = transform(
			-translationValue, inverseLinear);
		if (not inverseTranslation.isFinite()) return false;
		detail::Mat4<T> inverse = detail::Mat4<T>::identity();
		for (I32 row = 0; row < 3; ++row) {
			for (I32 column = 0; column < 3; ++column) {
				inverse.m[row][column] = inverseLinear.m[row][column];
			}
		}
		inverse.m[3][0] = inverseTranslation.x;
		inverse.m[3][1] = inverseTranslation.y;
		inverse.m[3][2] = inverseTranslation.z;
		outInverse = inverse;
		return true;
	}
	T augmented[4][8] = {};
	T rowScale[4] = {};
	for (I32 row = 0; row < 4; ++row) {
		for (I32 column = 0; column < 4; ++column) {
			const T value = inMatrix.m[row][column];
			if (not oa::isFinite(value)) return false;
			augmented[row][column] = value;
			rowScale[row] = oa::max(rowScale[row], oa::abs(value));
		}
		if (rowScale[row] <= T(0)) return false;
		augmented[row][row + 4] = T(1);
	}
	for (I32 pivotColumn = 0; pivotColumn < 4; ++pivotColumn) {
		I32 pivotRow = pivotColumn;
		T pivotRatio =
			oa::abs(augmented[pivotRow][pivotColumn]) / rowScale[pivotRow];
		for (I32 row = pivotColumn + 1; row < 4; ++row) {
			const T ratio =
				oa::abs(augmented[row][pivotColumn]) / rowScale[row];
			if (ratio > pivotRatio) {
				pivotRatio = ratio;
				pivotRow = row;
			}
		}
		if (not oa::isFinite(pivotRatio) or pivotRatio <= inTolerance) {
			return false;
		}
		if (pivotRow != pivotColumn) {
			for (I32 column = 0; column < 8; ++column) {
				oa::swapValues(augmented[pivotRow][column], augmented[pivotColumn][column]);
			}
			oa::swapValues(rowScale[pivotRow], rowScale[pivotColumn]);
		}
		const T inversePivot = T(1) / augmented[pivotColumn][pivotColumn];
		for (I32 column = 0; column < 8; ++column) {
			augmented[pivotColumn][column] *= inversePivot;
		}
		for (I32 row = 0; row < 4; ++row) {
			if (row == pivotColumn) continue;
			const T factor = augmented[row][pivotColumn];
			for (I32 column = 0; column < 8; ++column) {
				augmented[row][column] -= factor * augmented[pivotColumn][column];
			}
		}
	}
	detail::Mat4<T> inverse{};
	for (I32 row = 0; row < 4; ++row) {
		for (I32 column = 0; column < 4; ++column) {
			inverse.m[row][column] = augmented[row][column + 4];
			if (not oa::isFinite(inverse.m[row][column])) return false;
		}
	}
	const detail::Mat4<T> residual = matrixMul(inMatrix, inverse);
	const T residualFactor = oa::isSameV<T, F32> ? T(256) : T(65536);
	const T residualTolerance = oa::max(
		Tolerance<T> * residualFactor, inTolerance * residualFactor);
	for (I32 row = 0; row < 4; ++row) {
		for (I32 column = 0; column < 4; ++column) {
			const T expected = row == column ? T(1) : T(0);
			if (not oa::isFinite(residual.m[row][column])
				or oa::abs(residual.m[row][column] - expected)
					> residualTolerance) {
				return false;
			}
		}
	}
	outInverse = inverse;
	return true;
}

template <typename T>
[[nodiscard]] constexpr detail::Vec3<T> transformPoint(
	const detail::Vec3<T>& inPoint,
	const detail::Mat4<T>& inMatrix
) noexcept {
	const detail::Vec4<T> value = transform(
		{inPoint.x, inPoint.y, inPoint.z, T(1)}, inMatrix);
	return {value.x, value.y, value.z};
}

template <typename T>
[[nodiscard]] constexpr detail::Vec3<T> transformDirection(
	const detail::Vec3<T>& inDirection,
	const detail::Mat4<T>& inMatrix
) noexcept {
	const detail::Vec4<T> value = transform(
		{inDirection.x, inDirection.y, inDirection.z, T(0)}, inMatrix);
	return {value.x, value.y, value.z};
}

template <typename T>
[[nodiscard]] bool tryProjectPoint(
	const detail::Vec3<T>& inPoint,
	const detail::Mat4<T>& inMatrix,
	detail::Vec3<T>& outPoint,
	T inTolerance = Tolerance<T>
) noexcept {
	if (not isValidTolerance(inTolerance)) return false;
	const detail::Vec4<T> homogeneous = transform(
		{inPoint.x, inPoint.y, inPoint.z, T(1)}, inMatrix);
	if (not homogeneous.isFinite() or oa::abs(homogeneous.w) <= inTolerance) return false;
	const detail::Vec3<T> projected{
		homogeneous.x / homogeneous.w,
		homogeneous.y / homogeneous.w,
		homogeneous.z / homogeneous.w,
	};
	if (not projected.isFinite()) return false;
	outPoint = projected;
	return true;
}

template <typename T>
[[nodiscard]] bool tryTransformNormal(
	const detail::Vec3<T>& inNormal,
	const detail::Mat4<T>& inMatrix,
	detail::Vec3<T>& outNormal,
	T inTolerance = InverseTolerance<T>
) noexcept {
	if (not isValidTolerance(inTolerance)) return false;
	detail::Mat4<T> inverse{};
	if (not tryInverse(inMatrix, inverse, inTolerance)) return false;
	const detail::Vec3<T> transformed = transformDirection(inNormal, transpose(inverse));
	const T transformedLength = length(transformed);
	if (not oa::isFinite(transformedLength) or transformedLength <= inTolerance) return false;
	outNormal = transformed / transformedLength;
	return true;
}

template <typename T>
[[nodiscard]] constexpr detail::Mat4<T> translation(const detail::Vec3<T>& inTranslation) noexcept {
	detail::Mat4<T> result = detail::Mat4<T>::identity();
	result.m[3][0] = inTranslation.x;
	result.m[3][1] = inTranslation.y;
	result.m[3][2] = inTranslation.z;
	return result;
}

template <typename T>
[[nodiscard]] constexpr detail::Mat4<T> scaleMatrix(const detail::Vec3<T>& inScale) noexcept {
	detail::Mat4<T> result = detail::Mat4<T>::identity();
	result.m[0][0] = inScale.x;
	result.m[1][1] = inScale.y;
	result.m[2][2] = inScale.z;
	return result;
}

template <typename T>
[[nodiscard]] detail::Mat4<T> composeTrs(
	const detail::Vec3<T>& inTranslation,
	const detail::Quat<T>& inRotation,
	const detail::Vec3<T>& inScale
) noexcept {
	detail::Mat4<T> result = quaternionToMatrix(inRotation);
	for (I32 column = 0; column < 3; ++column) {
		result.m[0][column] *= inScale.x;
		result.m[1][column] *= inScale.y;
		result.m[2][column] *= inScale.z;
	}
	result.m[3][0] = inTranslation.x;
	result.m[3][1] = inTranslation.y;
	result.m[3][2] = inTranslation.z;
	return result;
}

template <typename T>
[[nodiscard]] detail::Vec3<T> sphericalToCartesian(
	T inYawRad,
	T inPitchRad,
	T inRadius
) noexcept {
	const T cosineYaw = oa::cos(inYawRad);
	const T sineYaw = oa::sin(inYawRad);
	const T cosinePitch = oa::cos(inPitchRad);
	const T sinePitch = oa::sin(inPitchRad);
	return {
		(inRadius * sineYaw) * cosinePitch,
		inRadius * sinePitch,
		(inRadius * cosineYaw) * cosinePitch,
	};
}

template <typename T>
[[nodiscard]] detail::Vec3<T> cartesianToSpherical(const detail::Vec3<T>& inValue) noexcept {
	const T radius = length(inValue);
	if (radius < Tolerance<T>) {
		return {};
	}
	return {
		oa::atan2(inValue.x, inValue.z),
		oa::asin(inValue.y / radius),
		radius,
	};
}

} // namespace vlm

} // namespace oa
