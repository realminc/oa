#include <oa/animation/fnAnim.h>

namespace oa {

oa::U32 AnimClip::getSkeletonId() const noexcept {
	return skeletonId_;
}

oa::U32 AnimClip::getJointCount() const noexcept {
	return jointCount_;
}

oa::U32 AnimClip::getFrameCount() const noexcept {
	return frameCount_;
}

oa::F32 AnimClip::getFramesPerSecond() const noexcept {
	return framesPerSecond_;
}

oa::F64 AnimClip::getDurationSeconds() const noexcept {
	if (frameCount_ <= 1U or framesPerSecond_ <= 0.0F) return 0.0;
	return static_cast<oa::F64>(frameCount_ - 1U)
		/ static_cast<oa::F64>(framesPerSecond_);
}

oa::Span<const oa::Transform> AnimClip::getSamples() const noexcept {
	return {samples_.data(), samples_.size()};
}

oa::Result<oa::SkelPose> AnimClip::sample(
	oa::F64 inTimeSeconds,
	oa::AnimWrapMode inWrapMode) const {
	return oa::FnAnim::sample(*this, inTimeSeconds, inWrapMode);
}

} // namespace oa
