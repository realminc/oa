// OaCurve — Animation curves and interpolation.
//
// Provides interpolation between keyframes for smooth animation.
// Supports various interpolation modes: linear, bezier, cubic spline, etc.
//
// Usage:
//   OaCurveVec3 curve;
//   curve.AddKeyframe({0.0f, {0.0f, 0.0f, 0.0f}});
//   curve.AddKeyframe({1.0f, {1.0f, 1.0f, 1.0f}});
//   VlmVec3 value = curve.Evaluate(0.5f);  // Interpolate at t=0.5

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Vlm.h>
#include <Oa/Animation/Keyframe.h>

// Vector3 curve (position, scale)
class OaCurveVec3 {
public:
	void AddKeyframe(const OaKeyframeVec3& InKeyframe);
	void Clear();

	// Evaluate at given time
	[[nodiscard]] VlmVec3 Evaluate(OaF32 InTime) const;

	// Get derivative (velocity) at given time
	[[nodiscard]] VlmVec3 Derivative(OaF32 InTime) const;

	[[nodiscard]] OaU32 GetKeyframeCount() const noexcept;

private:
	OaVec<OaKeyframeVec3> Keyframes_;
};

// Quaternion curve (rotation, uses spherical linear interpolation)
class OaCurveQuat {
public:
	void AddKeyframe(const OaKeyframeQuat& InKeyframe);
	void Clear();

	// Evaluate at given time (uses SLERP)
	[[nodiscard]] VlmQuat Evaluate(OaF32 InTime) const;

	[[nodiscard]] OaU32 GetKeyframeCount() const noexcept;

private:
	OaVec<OaKeyframeQuat> Keyframes_;
};

// Color curve (RGBA)
class OaCurveColor {
public:
	void AddKeyframe(const OaKeyframeColor& InKeyframe);
	void Clear();

	// Evaluate at given time
	[[nodiscard]] VlmVec4 Evaluate(OaF32 InTime) const;

	[[nodiscard]] OaU32 GetKeyframeCount() const noexcept;

private:
	OaVec<OaKeyframeColor> Keyframes_;
};

// Scalar curve (float parameters)
class OaCurveFloat {
public:
	void AddKeyframe(const OaKeyframeFloat& InKeyframe);
	void Clear();

	// Evaluate at given time
	[[nodiscard]] OaF32 Evaluate(OaF32 InTime) const;

	// Get derivative at given time
	[[nodiscard]] OaF32 Derivative(OaF32 InTime) const;

	[[nodiscard]] OaU32 GetKeyframeCount() const noexcept;

private:
	OaVec<OaKeyframeFloat> Keyframes_;
};

// Easing functions (for animation without keyframes)
namespace OaEasing {
	// Linear
	OaF32 Linear(OaF32 InT);

	// Quadratic
	OaF32 EaseInQuad(OaF32 InT);
	OaF32 EaseOutQuad(OaF32 InT);
	OaF32 EaseInOutQuad(OaF32 InT);

	// Cubic
	OaF32 EaseInCubic(OaF32 InT);
	OaF32 EaseOutCubic(OaF32 InT);
	OaF32 EaseInOutCubic(OaF32 InT);

	// Quartic
	OaF32 EaseInQuart(OaF32 InT);
	OaF32 EaseOutQuart(OaF32 InT);
	OaF32 EaseInOutQuart(OaF32 InT);

	// Exponential
	OaF32 EaseInExpo(OaF32 InT);
	OaF32 EaseOutExpo(OaF32 InT);
	OaF32 EaseInOutExpo(OaF32 InT);

	// Elastic
	OaF32 EaseOutElastic(OaF32 InT);

	// Bounce
	OaF32 EaseOutBounce(OaF32 InT);
}
