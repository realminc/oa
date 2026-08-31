// oa::AnimClip — uniform sampled skeletal-animation clip.

#pragma once

#include <oa/animation/skelPose.h>
#include <oa/core/status.h>

namespace oa {

class FnAnim;

enum class AnimWrapMode : oa::U8 {
	Clamp,
	Loop,
};

[[nodiscard]] constexpr bool isValidAnimWrapMode(
	oa::AnimWrapMode inMode) noexcept {
	switch (inMode) {
		case oa::AnimWrapMode::Clamp:
		case oa::AnimWrapMode::Loop: return true;
	}
	return false;
}

class AnimClip {
public:
	[[nodiscard]] oa::U32 getSkeletonId() const noexcept;
	[[nodiscard]] oa::U32 getJointCount() const noexcept;
	[[nodiscard]] oa::U32 getFrameCount() const noexcept;
	[[nodiscard]] oa::F32 getFramesPerSecond() const noexcept;
	[[nodiscard]] oa::F64 getDurationSeconds() const noexcept;
	[[nodiscard]] oa::Span<const oa::Transform> getSamples() const noexcept;
	[[nodiscard]] oa::Result<oa::SkelPose> sample(
		oa::F64 inTimeSeconds,
		oa::AnimWrapMode inWrapMode = oa::AnimWrapMode::Clamp
	) const;

private:
	friend class FnAnim;
	AnimClip() = default;

	oa::U32 skeletonId_ = 0U;
	oa::U32 jointCount_ = 0U;
	oa::U32 frameCount_ = 0U;
	oa::F32 framesPerSecond_ = 0.0F;
	oa::Vector<oa::Transform> samples_;
};

} // namespace oa
