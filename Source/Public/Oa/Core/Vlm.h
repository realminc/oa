// Vlm — Vulkan Linear Math
//
// Realm-syntax 3D linear algebra for render, animation, and spatial compute.
// Initially backed by the vendored GLM in Source/ThirdParty/glm; native
// implementations will replace GLM calls incrementally.

#pragma once

#include <Oa/Core/Math.h>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cmath>

struct VlmVec2 {
	OaF32 X = 0.0F;
	OaF32 Y = 0.0F;

	constexpr VlmVec2() = default;
	constexpr VlmVec2(OaF32 InX, OaF32 InY) : X(InX), Y(InY) {}

	[[nodiscard]] constexpr VlmVec2 operator+(const VlmVec2& InB) const noexcept { return {X + InB.X, Y + InB.Y}; }
	[[nodiscard]] constexpr VlmVec2 operator-(const VlmVec2& InB) const noexcept { return {X - InB.X, Y - InB.Y}; }
	[[nodiscard]] constexpr VlmVec2 operator*(OaF32 InS) const noexcept { return {X * InS, Y * InS}; }
	[[nodiscard]] constexpr VlmVec2 operator/(OaF32 InS) const noexcept { return {X / InS, Y / InS}; }
	constexpr VlmVec2& operator+=(const VlmVec2& InB) noexcept { X += InB.X; Y += InB.Y; return *this; }
	constexpr VlmVec2& operator-=(const VlmVec2& InB) noexcept { X -= InB.X; Y -= InB.Y; return *this; }
};

struct VlmVec3 {
	OaF32 X = 0.0F;
	OaF32 Y = 0.0F;
	OaF32 Z = 0.0F;
};

struct VlmVec4 {
	OaF32 X = 0.0F;
	OaF32 Y = 0.0F;
	OaF32 Z = 0.0F;
	OaF32 W = 0.0F;
};

struct VlmQuat {
	OaF32 X = 0.0F;
	OaF32 Y = 0.0F;
	OaF32 Z = 0.0F;
	OaF32 W = 1.0F;
};

struct VlmMat4 {
	OaF32 M[4][4] = {};

	[[nodiscard]] static VlmMat4 Identity() {
		VlmMat4 r{};
		r.M[0][0] = 1.0F; r.M[1][1] = 1.0F;
		r.M[2][2] = 1.0F; r.M[3][3] = 1.0F;
		return r;
	}
};

namespace Vlm {

// ─── GLM interop (VLM stores row-major, GLM is column-major) ──────────────

namespace detail {

[[nodiscard]] inline glm::vec2 ToGLM(const VlmVec2& InV) noexcept { return glm::vec2(InV.X, InV.Y); }
[[nodiscard]] inline VlmVec2 FromGLM(const glm::vec2& InV) noexcept { return VlmVec2(InV.x, InV.y); }

[[nodiscard]] inline glm::vec3 ToGLM(const VlmVec3& InV) noexcept { return glm::vec3(InV.X, InV.Y, InV.Z); }
[[nodiscard]] inline VlmVec3 FromGLM(const glm::vec3& InV) noexcept { return VlmVec3(InV.x, InV.y, InV.z); }

[[nodiscard]] inline glm::vec4 ToGLM(const VlmVec4& InV) noexcept { return glm::vec4(InV.X, InV.Y, InV.Z, InV.W); }
[[nodiscard]] inline VlmVec4 FromGLM(const glm::vec4& InV) noexcept { return VlmVec4(InV.x, InV.y, InV.z, InV.w); }

[[nodiscard]] inline glm::quat ToGLM(const VlmQuat& InQ) noexcept { return glm::quat(InQ.W, InQ.X, InQ.Y, InQ.Z); }
[[nodiscard]] inline VlmQuat FromGLM(const glm::quat& InQ) noexcept { return VlmQuat(InQ.x, InQ.y, InQ.z, InQ.w); }

// Row-vector convention: VLM stores M[row][col] = P_VLM[row][col]. GLM stores
// column-major P_GLM = P_VLM^T. The two storages share the same memory layout,
// so these conversions are direct copies.
[[nodiscard]] inline glm::mat4 ToGLM(const VlmMat4& InM) noexcept {
	glm::mat4 r;
	for (OaI32 col = 0; col < 4; ++col) {
		for (OaI32 row = 0; row < 4; ++row) {
			r[col][row] = InM.M[col][row];
		}
	}
	return r;
}

[[nodiscard]] inline VlmMat4 FromGLM(const glm::mat4& InM) noexcept {
	VlmMat4 r;
	for (OaI32 row = 0; row < 4; ++row) {
		for (OaI32 col = 0; col < 4; ++col) {
			r.M[row][col] = InM[row][col];
		}
	}
	return r;
}

// Column-vector convention: used only by QuaternionToMatrix, which returns the
// rotation matrix R intended for column-vector multiplication (R · v). VLM
// storage is still row-major, so we transpose the GLM column-major storage.
[[nodiscard]] inline glm::mat4 ToGLMColumnVector(const VlmMat4& InM) noexcept {
	glm::mat4 r;
	for (OaI32 col = 0; col < 4; ++col) {
		for (OaI32 row = 0; row < 4; ++row) {
			r[col][row] = InM.M[row][col];
		}
	}
	return r;
}

[[nodiscard]] inline VlmMat4 FromGLMColumnVector(const glm::mat4& InM) noexcept {
	VlmMat4 r;
	for (OaI32 col = 0; col < 4; ++col) {
		for (OaI32 row = 0; row < 4; ++row) {
			r.M[row][col] = InM[col][row];
		}
	}
	return r;
}

} // namespace detail

// ─── Constants ───────────────────────────────────────────────────────────

inline constexpr OaF32 PI      = 3.14159265358979323846F;
inline constexpr OaF32 EPSILON = 0.0001F;

// ─── Vector Operations ─────────────────────────────────────────────────

[[nodiscard]] inline VlmVec3 Add(const VlmVec3& InA, const VlmVec3& InB) noexcept {
	return detail::FromGLM(detail::ToGLM(InA) + detail::ToGLM(InB));
}

[[nodiscard]] inline VlmVec3 Sub(const VlmVec3& InA, const VlmVec3& InB) noexcept {
	return detail::FromGLM(detail::ToGLM(InA) - detail::ToGLM(InB));
}

[[nodiscard]] inline VlmVec3 Scale(const VlmVec3& InV, OaF32 InS) noexcept {
	return detail::FromGLM(detail::ToGLM(InV) * InS);
}

[[nodiscard]] inline OaF32 Dot(const VlmVec3& InA, const VlmVec3& InB) noexcept {
	return glm::dot(detail::ToGLM(InA), detail::ToGLM(InB));
}

[[nodiscard]] inline VlmVec3 Cross(const VlmVec3& InA, const VlmVec3& InB) noexcept {
	return detail::FromGLM(glm::cross(detail::ToGLM(InA), detail::ToGLM(InB)));
}

[[nodiscard]] inline OaF32 LengthSquared(const VlmVec3& InV) noexcept {
	const auto v = detail::ToGLM(InV);
	return glm::dot(v, v);
}

[[nodiscard]] inline OaF32 Length(const VlmVec3& InV) noexcept {
	return glm::length(detail::ToGLM(InV));
}

[[nodiscard]] inline VlmVec3 Normalize(const VlmVec3& InV) noexcept {
	const auto v = detail::ToGLM(InV);
	const auto len = glm::length(v);
	if (len < EPSILON) {
		return VlmVec3{.X = 0.0F, .Y = 0.0F, .Z = 0.0F};
	}
	return detail::FromGLM(glm::normalize(v));
}

[[nodiscard]] inline VlmVec3 Lerp(const VlmVec3& InA, const VlmVec3& InB, OaF32 InT) noexcept {
	return detail::FromGLM(glm::mix(detail::ToGLM(InA), detail::ToGLM(InB), InT));
}

// ─── Quaternion Operations ───────────────────────────────────────────────

[[nodiscard]] inline VlmQuat QuaternionIdentity() noexcept {
	return detail::FromGLM(glm::quat(1.0F, 0.0F, 0.0F, 0.0F));
}

// Construct quaternion from yaw/pitch/roll (ZYX order, degrees)
[[nodiscard]] inline VlmQuat QuaternionFromEuler(OaF32 InYawDeg, OaF32 InPitchDeg, OaF32 InRollDeg) noexcept {
	OaF32 yaw = InYawDeg * PI / 180.0F;
	OaF32 pitch = InPitchDeg * PI / 180.0F;
	OaF32 roll = InRollDeg * PI / 180.0F;

	OaF32 cy = std::cos(yaw * 0.5F);
	OaF32 sy = std::sin(yaw * 0.5F);
	OaF32 cp = std::cos(pitch * 0.5F);
	OaF32 sp = std::sin(pitch * 0.5F);
	OaF32 cr = std::cos(roll * 0.5F);
	OaF32 sr = std::sin(roll * 0.5F);

	return {
		.X = (cy * cp) * sr - (sy * sp) * cr,
		.Y = (sy * cp) * sr + (cy * sp) * cr,
		.Z = (sy * cp) * cr - (cy * sp) * sr,
		.W = (cy * cp) * cr + (sy * sp) * sr
	};
}

// Extract euler angles from quaternion (ZYX order, returns degrees)
[[nodiscard]] inline VlmVec3 QuaternionToEuler(const VlmQuat& InQ) noexcept {
	OaF32 sinr_cosp = 2.0F * (InQ.W * InQ.X + InQ.Y * InQ.Z);
	OaF32 cosr_cosp = 1.0F - 2.0F * (InQ.X * InQ.X + InQ.Y * InQ.Y);
	OaF32 roll = std::atan2(sinr_cosp, cosr_cosp);

	OaF32 sinp = 2.0F * (InQ.W * InQ.Y - InQ.Z * InQ.X);
	OaF32 pitch;
	if (std::abs(sinp) >= 1.0F) {
		pitch = std::copysign(PI / 2.0F, sinp);
	} else {
		pitch = std::asin(sinp);
	}

	OaF32 siny_cosp = 2.0F * (InQ.W * InQ.Z + InQ.X * InQ.Y);
	OaF32 cosy_cosp = 1.0F - 2.0F * (InQ.Y * InQ.Y + InQ.Z * InQ.Z);
	OaF32 yaw = std::atan2(siny_cosp, cosy_cosp);

	return {
		.X = yaw * 180.0F / PI,
		.Y = pitch * 180.0F / PI,
		.Z = roll * 180.0F / PI
	};
}

// Convert unit quaternion to 3x3 rotation matrix (stored in VlmMat4).
// Returns the column-vector rotation R (R · v), consistent with historical Vlm.
[[nodiscard]] inline VlmMat4 QuaternionToMatrix(const VlmQuat& InQ) noexcept {
	return detail::FromGLMColumnVector(glm::mat4_cast(detail::ToGLM(InQ)));
}

// Rotate a vector by a quaternion
[[nodiscard]] inline VlmVec3 RotateVector(const VlmQuat& InQ, const VlmVec3& InV) noexcept {
	return detail::FromGLM(detail::ToGLM(InQ) * detail::ToGLM(InV));
}

// ─── Matrix Operations ─────────────────────────────────────────────────

[[nodiscard]] inline VlmMat4 MatrixIdentity() noexcept {
	return VlmMat4::Identity();
}

[[nodiscard]] inline VlmMat4 MatrixMul(const VlmMat4& InA, const VlmMat4& InB) noexcept {
	return detail::FromGLM(detail::ToGLM(InB) * detail::ToGLM(InA));
}

// Perspective projection matrix for row-vector shaders.
// Vulkan clip depth is [0, 1]. Raster-space Y is handled by the viewport,
// not by changing the world/camera coordinate system here.
[[nodiscard]] inline VlmMat4 Perspective(OaF32 InFovYDeg, OaF32 InAspect, OaF32 InNear, OaF32 InFar) noexcept {
	return detail::FromGLM(glm::perspectiveRH_ZO(InFovYDeg * PI / 180.0F, InAspect, InNear, InFar));
}

// Orthographic projection matrix for Vulkan clip depth [0, 1].
[[nodiscard]] inline VlmMat4 Orthographic(OaF32 InWidth, OaF32 InHeight, OaF32 InNear, OaF32 InFar, OaF32 InZoom = 1.0F) noexcept {
	OaF32 halfW = InWidth * 0.5F / InZoom;
	OaF32 halfH = InHeight * 0.5F / InZoom;
	return detail::FromGLM(glm::orthoRH_ZO(-halfW, halfW, -halfH, halfH, InNear, InFar));
}

// Look-at view matrix (right-handed, Y-up)
[[nodiscard]] inline VlmMat4 LookAt(const VlmVec3& InEye, const VlmVec3& InTarget, const VlmVec3& InWorldUp) noexcept {
	return detail::FromGLM(glm::lookAtRH(detail::ToGLM(InEye), detail::ToGLM(InTarget), detail::ToGLM(InWorldUp)));
}

// Translation matrix
[[nodiscard]] inline VlmMat4 Translation(const VlmVec3& InT) noexcept {
	return detail::FromGLM(glm::translate(glm::mat4(1.0F), detail::ToGLM(InT)));
}

// Scale matrix
[[nodiscard]] inline VlmMat4 ScaleMatrix(const VlmVec3& InS) noexcept {
	return detail::FromGLM(glm::scale(glm::mat4(1.0F), detail::ToGLM(InS)));
}

// ─── Spherical / Orbit helpers ─────────────────────────────────────────

// Spherical coordinates to cartesian (Y-up)
[[nodiscard]] inline VlmVec3 SphericalToCartesian(OaF32 InYawRad, OaF32 InPitchRad, OaF32 InRadius) noexcept {
	OaF32 cy = std::cos(InYawRad);
	OaF32 sy = std::sin(InYawRad);
	OaF32 cp = std::cos(InPitchRad);
	OaF32 sp = std::sin(InPitchRad);
	return {
		.X = (InRadius * sy) * cp,
		.Y = InRadius * sp,
		.Z = (InRadius * cy) * cp
	};
}

// Cartesion to spherical (Y-up)
[[nodiscard]] inline VlmVec3 CartesianToSpherical(const VlmVec3& InV) noexcept {
	OaF32 radius = Length(InV);
	if (radius < EPSILON) {
		return VlmVec3{.X = 0.0F, .Y = 0.0F, .Z = 0.0F};
	}
	OaF32 yaw = std::atan2(InV.X, InV.Z);
	OaF32 pitch = std::asin(InV.Y / radius);
	return {
		.X = yaw,
		.Y = pitch,
		.Z = radius
	};
}

} // namespace Vlm
