// oa::AnimCurve — immutable sparse animation curve.

#pragma once

#include <oa/animation/animClip.h>
#include <oa/animation/keyframe.h>
#include <oa/core/status.h>

namespace oa {

class FnAnim;

template<typename T>
class AnimCurve {
	static_assert(
		oa::isAnimCurveValueV<T>,
		"AnimCurve supports F32, vlm::Vec3, and vlm::Quat values");

public:
	[[nodiscard]] oa::AnimInterpolation getInterpolation() const noexcept {
		return interpolation_;
	}
	[[nodiscard]] oa::U32 getKeyframeCount() const noexcept {
		return static_cast<oa::U32>(keyframes_.size());
	}
	[[nodiscard]] oa::Span<const oa::Keyframe<T>> getKeyframes() const noexcept {
		return {keyframes_.data(), keyframes_.size()};
	}
	[[nodiscard]] oa::Duration getStartTime() const noexcept {
		return keyframes_.front().time;
	}
	[[nodiscard]] oa::Duration getEndTime() const noexcept {
		return keyframes_.back().time;
	}
	[[nodiscard]] oa::Duration getDuration() const noexcept {
		return oa::Duration::fromNanoseconds(
			keyframes_.back().time.nanoseconds()
				- keyframes_.front().time.nanoseconds());
	}
	[[nodiscard]] oa::Result<T> sample(
		oa::Duration inTime,
		oa::AnimWrapMode inWrapMode = oa::AnimWrapMode::Clamp
	) const;

private:
	friend class FnAnim;
	AnimCurve() = default;

	oa::AnimInterpolation interpolation_ = oa::AnimInterpolation::Linear;
	oa::Vector<oa::Keyframe<T>> keyframes_;
};

} // namespace oa
