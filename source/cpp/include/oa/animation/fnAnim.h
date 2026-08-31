// oa::FnAnim — stateless pose and uniform-clip operations.

#pragma once

#include <oa/animation/animClip.h>
#include <oa/animation/animCurve.h>
#include <oa/core/status.h>

namespace oa {

class FnAnim final {
public:
	FnAnim() = delete;

	[[nodiscard]] static oa::Result<oa::SkelPose> createPose(
		oa::U32 inSkeletonId,
		oa::Span<const oa::Transform> inLocalTransforms
	);
	[[nodiscard]] static oa::Status validate(const oa::SkelPose& inPose);

	[[nodiscard]] static oa::Result<oa::AnimClip> createClip(
		oa::U32 inSkeletonId,
		oa::U32 inJointCount,
		oa::F32 inFramesPerSecond,
		oa::Span<const oa::Transform> inSamples
	);
	[[nodiscard]] static oa::Status validate(const oa::AnimClip& inClip);
	[[nodiscard]] static oa::Result<oa::SkelPose> getFrame(
		const oa::AnimClip& inClip,
		oa::U32 inFrame
	);
	[[nodiscard]] static oa::Result<oa::SkelPose> sample(
		const oa::AnimClip& inClip,
		oa::F64 inTimeSeconds,
		oa::AnimWrapMode inWrapMode = oa::AnimWrapMode::Clamp
	);

	[[nodiscard]] static oa::Result<oa::AnimCurve<oa::F32>> createCurve(
		oa::AnimInterpolation inInterpolation,
		oa::Span<const oa::Keyframe<oa::F32>> inKeyframes
	);
	[[nodiscard]] static oa::Result<oa::AnimCurve<oa::vlm::Vec3>> createCurve(
		oa::AnimInterpolation inInterpolation,
		oa::Span<const oa::Keyframe<oa::vlm::Vec3>> inKeyframes
	);
	[[nodiscard]] static oa::Result<oa::AnimCurve<oa::vlm::Quat>> createCurve(
		oa::AnimInterpolation inInterpolation,
		oa::Span<const oa::Keyframe<oa::vlm::Quat>> inKeyframes
	);
	template<typename T, oa::Usize Count>
	requires oa::isAnimCurveValueV<T>
	[[nodiscard]] static oa::Result<oa::AnimCurve<T>> createCurve(
		oa::AnimInterpolation inInterpolation,
		const oa::Keyframe<T> (&inKeyframes)[Count]
	) {
		return createCurve(
			inInterpolation,
			oa::Span<const oa::Keyframe<T>>(inKeyframes, Count));
	}

	[[nodiscard]] static oa::Status validate(
		const oa::AnimCurve<oa::F32>& inCurve);
	[[nodiscard]] static oa::Status validate(
		const oa::AnimCurve<oa::vlm::Vec3>& inCurve);
	[[nodiscard]] static oa::Status validate(
		const oa::AnimCurve<oa::vlm::Quat>& inCurve);

	[[nodiscard]] static oa::Result<oa::F32> sample(
		const oa::AnimCurve<oa::F32>& inCurve,
		oa::Duration inTime,
		oa::AnimWrapMode inWrapMode = oa::AnimWrapMode::Clamp
	);
	[[nodiscard]] static oa::Result<oa::vlm::Vec3> sample(
		const oa::AnimCurve<oa::vlm::Vec3>& inCurve,
		oa::Duration inTime,
		oa::AnimWrapMode inWrapMode = oa::AnimWrapMode::Clamp
	);
	[[nodiscard]] static oa::Result<oa::vlm::Quat> sample(
		const oa::AnimCurve<oa::vlm::Quat>& inCurve,
		oa::Duration inTime,
		oa::AnimWrapMode inWrapMode = oa::AnimWrapMode::Clamp
	);

	[[nodiscard]] static oa::Result<oa::Vector<oa::F32>> bake(
		const oa::AnimCurve<oa::F32>& inCurve,
		oa::Duration inStartTime,
		oa::F32 inFramesPerSecond,
		oa::U32 inFrameCount,
		oa::AnimWrapMode inWrapMode = oa::AnimWrapMode::Clamp
	);
	[[nodiscard]] static oa::Result<oa::Vector<oa::vlm::Vec3>> bake(
		const oa::AnimCurve<oa::vlm::Vec3>& inCurve,
		oa::Duration inStartTime,
		oa::F32 inFramesPerSecond,
		oa::U32 inFrameCount,
		oa::AnimWrapMode inWrapMode = oa::AnimWrapMode::Clamp
	);
	[[nodiscard]] static oa::Result<oa::Vector<oa::vlm::Quat>> bake(
		const oa::AnimCurve<oa::vlm::Quat>& inCurve,
		oa::Duration inStartTime,
		oa::F32 inFramesPerSecond,
		oa::U32 inFrameCount,
		oa::AnimWrapMode inWrapMode = oa::AnimWrapMode::Clamp
	);
};

} // namespace oa
