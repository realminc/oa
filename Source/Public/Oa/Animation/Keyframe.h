// OaKeyframe — Keyframe data structures for animation.
//
// Defines keyframe types for different animation data:
// - Position, rotation, scale (3D transforms)
// - Color (RGBA)
// - Scalar values (float parameters)
// - Custom user data
//
// Supports interpolation between keyframes (see Curve.h for interpolation).
//
// Usage:
//   OaKeyframeVec3 kf1(0.0f, {0.0f, 0.0f, 0.0f});
//   OaKeyframeVec3 kf2(1.0f, {1.0f, 1.0f, 1.0f});
//   // Interpolate at t=0.5 using Curve.h

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Vlm.h>

// Keyframe interpolation modes (used by Curve.h)
enum class OaInterpolationMode : OaU8 {
	Constant,     // Step function
	Linear,       // Linear interpolation
	Bezier,       // Bezier curve
	Cubic,        // Cubic spline
	Hermite,      // Hermite spline
};

// Tangent modes for smooth interpolation
enum class OaTangentMode : OaU8 {
	Auto,         // Automatic tangent calculation
	Linear,       // Linear tangent
	Stepped,      // Stepped (constant)
	Flat,         // Flat tangent
	Custom,       // User-specified tangent
};

// Base keyframe structure
struct OaKeyframeBase {
	OaF32                Time      = 0.0f;
	OaInterpolationMode Interp    = OaInterpolationMode::Linear;
	OaTangentMode       InTangent  = OaTangentMode::Auto;
	OaTangentMode       OutTangent = OaTangentMode::Auto;
};

// Vector3 keyframe (position, scale)
struct OaKeyframeVec3 : public OaKeyframeBase {
	VlmVec3 Value = {0.0f, 0.0f, 0.0f};
	VlmVec3 InTangentValue  = {0.0f, 0.0f, 0.0f};
	VlmVec3 OutTangentValue = {0.0f, 0.0f, 0.0f};

	OaKeyframeVec3() = default;
	OaKeyframeVec3(OaF32 InTime, const VlmVec3& InValue)
		: OaKeyframeBase{InTime}, Value(InValue) {}
};

// Quaternion keyframe (rotation)
struct OaKeyframeQuat : public OaKeyframeBase {
	VlmQuat Value = {0.0f, 0.0f, 0.0f, 1.0f};  // Identity quaternion
	VlmQuat InTangentValue  = {0.0f, 0.0f, 0.0f, 0.0f};
	VlmQuat OutTangentValue = {0.0f, 0.0f, 0.0f, 0.0f};

	OaKeyframeQuat() = default;
	OaKeyframeQuat(OaF32 InTime, const VlmQuat& InValue)
		: OaKeyframeBase{InTime}, Value(InValue) {}
};

// Color keyframe (RGBA)
struct OaKeyframeColor : public OaKeyframeBase {
	VlmVec4 Value = {1.0f, 1.0f, 1.0f, 1.0f};  // White
	VlmVec4 InTangentValue  = {0.0f, 0.0f, 0.0f, 0.0f};
	VlmVec4 OutTangentValue = {0.0f, 0.0f, 0.0f, 0.0f};

	OaKeyframeColor() = default;
	OaKeyframeColor(OaF32 InTime, const VlmVec4& InValue)
		: OaKeyframeBase{InTime}, Value(InValue) {}
};

// Scalar keyframe (float parameters)
struct OaKeyframeFloat : public OaKeyframeBase {
	OaF32 Value           = 0.0f;
	OaF32 InTangentValue  = 0.0f;
	OaF32 OutTangentValue = 0.0f;

	OaKeyframeFloat() = default;
	OaKeyframeFloat(OaF32 InTime, OaF32 InValue)
		: OaKeyframeBase{InTime}, Value(InValue) {}
};

// Keyframe track (sorted list of keyframes for one property)
template<typename T>
class OaKeyframeTrack {
public:
	void AddKeyframe(const T& InKeyframe);
	void RemoveKeyframe(OaU32 InIndex);
	void Clear();

	[[nodiscard]] OaU32 GetKeyframeCount() const noexcept;
	[[nodiscard]] const T& GetKeyframe(OaU32 InIndex) const;
	[[nodiscard]] T& GetKeyframe(OaU32 InIndex);

	// Find keyframe at or before given time
	[[nodiscard]] OaI32 FindKeyframeAt(OaF32 InTime) const;

private:
	OaVec<T> Keyframes_;
};
