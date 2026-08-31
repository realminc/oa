#include <oa/animation/fnAnim.h>

#include <oa/core/fnTransform.h>
#include <oa/core/std/limits.h>
#include <oa/core/std/scalarMath.h>

namespace {

[[nodiscard]] bool checkedSampleCount(
	oa::U32 inFrameCount,
	oa::U32 inJointCount,
	oa::Usize& outCount) noexcept {
	if (inFrameCount == 0U or inJointCount == 0U) return false;
	const oa::Usize frames = static_cast<oa::Usize>(inFrameCount);
	const oa::Usize joints = static_cast<oa::Usize>(inJointCount);
	if (frames > oa::Limits<oa::Usize>::max() / joints) return false;
	outCount = frames * joints;
	return true;
}

[[nodiscard]] oa::Status validateTransforms(
	oa::Span<const oa::Transform> inTransforms) {
	if (inTransforms.empty()) {
		return oa::Status::invalidArgument("Animation transform array is empty");
	}
	for (const oa::Transform& transform : inTransforms) {
		OA_RETURN_IF_ERROR(oa::FnTransform::validate(transform));
	}
	return oa::Status::ok();
}

[[nodiscard]] oa::Status validateClipMetadata(const oa::AnimClip& inClip) {
	if (inClip.getJointCount() == 0U
		or inClip.getFrameCount() == 0U
		or not oa::isFinite(inClip.getFramesPerSecond())
		or inClip.getFramesPerSecond() <= 0.0F) {
		return oa::Status::invalidArgument("AnimClip metadata is invalid");
	}
	oa::Usize expected = 0U;
	if (not checkedSampleCount(
		inClip.getFrameCount(), inClip.getJointCount(), expected)
		or expected != inClip.getSamples().size()) {
		return oa::Status::invalidArgument("AnimClip sample count is invalid");
	}
	return oa::Status::ok();
}

[[nodiscard]] oa::Transform interpolateTransform(
	const oa::Transform& inA,
	const oa::Transform& inB,
	oa::F32 inWeight) {
	oa::Transform result;
	result.setPosition(oa::vlm::lerp(
		inA.getPosition(), inB.getPosition(), inWeight));
	result.setRotation(oa::vlm::slerp(
		inA.getRotation(), inB.getRotation(), inWeight));
	result.setScale(oa::vlm::lerp(
		inA.getScale(), inB.getScale(), inWeight));
	result.setShear(oa::vlm::lerp(
		inA.getShear(), inB.getShear(), inWeight));
	return result;
}

template<typename T>
[[nodiscard]] bool curveValueIsFinite(const T& inValue) noexcept {
	if constexpr (oa::isSameV<T, oa::F32>) {
		return oa::isFinite(inValue);
	} else {
		return inValue.isFinite();
	}
}

template<typename T>
[[nodiscard]] bool curveValueIsValid(const T& inValue) noexcept {
	if (not curveValueIsFinite(inValue)) return false;
	if constexpr (oa::isSameV<T, oa::vlm::Quat>) {
		const oa::F32 magnitude = oa::max(
			oa::max(oa::abs(inValue.x), oa::abs(inValue.y)),
			oa::max(oa::abs(inValue.z), oa::abs(inValue.w)));
		return magnitude > 0.0F;
	}
	return true;
}

[[nodiscard]] bool curveSpanIsRepresentable(
	oa::I64 inFirst,
	oa::I64 inLast) noexcept {
	return inFirst >= 0
		or inLast <= oa::Limits<oa::I64>::max() + inFirst;
}

template<typename T>
[[nodiscard]] oa::Status validateCurveKeyframes(
	oa::AnimInterpolation inInterpolation,
	oa::Span<const oa::Keyframe<T>> inKeyframes) {
	if (not oa::isValidAnimInterpolation(inInterpolation)) {
		return oa::Status::invalidArgument(
			"AnimCurve interpolation mode is invalid");
	}
	if (inKeyframes.empty()) {
		return oa::Status::invalidArgument(
			"AnimCurve requires at least one keyframe");
	}
	if (inKeyframes.size() > oa::Limits<oa::U32>::max()) {
		return oa::Status::error(
			oa::StatusCode::OutOfRange,
			"AnimCurve keyframe count exceeds the supported range");
	}
	for (oa::Usize index = 0U; index < inKeyframes.size(); ++index) {
		const oa::Keyframe<T>& keyframe = inKeyframes[index];
		if (not curveValueIsValid(keyframe.value)
			or not curveValueIsFinite(keyframe.inTangent)
			or not curveValueIsFinite(keyframe.outTangent)) {
			return oa::Status::invalidArgument(
				"AnimCurve values and tangents must be finite; quaternion values must be non-zero");
		}
		if (index > 0U
			and keyframe.time <= inKeyframes[index - 1U].time) {
			return oa::Status::invalidArgument(
				"AnimCurve keyframe times must be strictly increasing");
		}
	}
	if (not curveSpanIsRepresentable(
		inKeyframes.front().time.nanoseconds(),
		inKeyframes.back().time.nanoseconds())) {
		return oa::Status::error(
			oa::StatusCode::OutOfRange,
			"AnimCurve time span exceeds the supported duration range");
	}
	return oa::Status::ok();
}

template<typename T>
[[nodiscard]] oa::Status validateCurveMetadata(
	const oa::AnimCurve<T>& inCurve) {
	const oa::Span<const oa::Keyframe<T>> keyframes = inCurve.getKeyframes();
	if (not oa::isValidAnimInterpolation(inCurve.getInterpolation())
		or keyframes.empty()
		or keyframes.size() > oa::Limits<oa::U32>::max()
		or keyframes.front().time > keyframes.back().time
		or not curveSpanIsRepresentable(
			keyframes.front().time.nanoseconds(),
			keyframes.back().time.nanoseconds())) {
		return oa::Status::invalidArgument("AnimCurve metadata is invalid");
	}
	return oa::Status::ok();
}

template<typename T>
[[nodiscard]] T normalizeCurveValue(const T& inValue) noexcept {
	if constexpr (oa::isSameV<T, oa::vlm::Quat>) {
		return inValue.normalized();
	}
	return inValue;
}

template<typename T>
[[nodiscard]] T interpolateCurveLinear(
	const T& inA,
	const T& inB,
	oa::F32 inWeight) noexcept {
	if constexpr (oa::isSameV<T, oa::F32>) {
		return inA * (1.0F - inWeight) + inB * inWeight;
	} else if constexpr (oa::isSameV<T, oa::vlm::Vec3>) {
		return oa::vlm::lerp(inA, inB, inWeight);
	} else {
		return oa::vlm::slerp(inA, inB, inWeight);
	}
}

template<typename T>
[[nodiscard]] oa::Result<T> interpolateCurveCubic(
	const oa::Keyframe<T>& inA,
	const oa::Keyframe<T>& inB,
	oa::F32 inWeight,
	oa::F32 inSegmentSeconds) {
	const oa::F32 squared = inWeight * inWeight;
	const oa::F32 cubed = squared * inWeight;
	const oa::F32 valueAWeight = 2.0F * cubed - 3.0F * squared + 1.0F;
	const oa::F32 tangentAWeight = cubed - 2.0F * squared + inWeight;
	const oa::F32 valueBWeight = -2.0F * cubed + 3.0F * squared;
	const oa::F32 tangentBWeight = cubed - squared;
	const T value =
		inA.value * valueAWeight
		+ inA.outTangent * (inSegmentSeconds * tangentAWeight)
		+ inB.value * valueBWeight
		+ inB.inTangent * (inSegmentSeconds * tangentBWeight);
	if (not curveValueIsValid(value)) {
		return oa::Status::error(
			oa::StatusCode::OutOfRange,
			"AnimCurve cubic interpolation produced a non-finite or zero quaternion value");
	}
	return normalizeCurveValue(value);
}

[[nodiscard]] oa::I64 wrapCurveTime(
	oa::I64 inTime,
	oa::I64 inFirst,
	oa::I64 inLast) noexcept {
	const oa::U64 span = static_cast<oa::U64>(inLast - inFirst);
	if (inTime >= inFirst) {
		const oa::U64 distance =
			static_cast<oa::U64>(inTime) - static_cast<oa::U64>(inFirst);
		return inFirst + static_cast<oa::I64>(distance % span);
	}
	const oa::U64 distance =
		static_cast<oa::U64>(inFirst) - static_cast<oa::U64>(inTime);
	const oa::U64 remainder = distance % span;
	const oa::U64 offset = remainder == 0U ? 0U : span - remainder;
	return inFirst + static_cast<oa::I64>(offset);
}

template<typename T>
[[nodiscard]] oa::Result<T> sampleCurveValidated(
	const oa::AnimCurve<T>& inCurve,
	oa::Duration inTime,
	oa::AnimWrapMode inWrapMode) {
	const oa::Span<const oa::Keyframe<T>> keyframes = inCurve.getKeyframes();
	if (keyframes.size() == 1U) return keyframes.front().value;

	const oa::I64 first = keyframes.front().time.nanoseconds();
	const oa::I64 last = keyframes.back().time.nanoseconds();
	oa::I64 time = inTime.nanoseconds();
	if (inWrapMode == oa::AnimWrapMode::Loop) {
		time = wrapCurveTime(time, first, last);
	} else if (time <= first) {
		return keyframes.front().value;
	} else if (time >= last) {
		return keyframes.back().value;
	}

	oa::Usize lower = 1U;
	oa::Usize upper = keyframes.size();
	while (lower < upper) {
		const oa::Usize middle = lower + (upper - lower) / 2U;
		if (keyframes[middle].time.nanoseconds() <= time) {
			lower = middle + 1U;
		} else {
			upper = middle;
		}
	}
	const oa::Usize indexB = lower;
	const oa::Usize indexA = indexB - 1U;
	const oa::Keyframe<T>& keyA = keyframes[indexA];
	const oa::Keyframe<T>& keyB = keyframes[indexB];
	if (inCurve.getInterpolation() == oa::AnimInterpolation::Step) {
		return keyA.value;
	}
	const oa::I64 segmentNanoseconds =
		keyB.time.nanoseconds() - keyA.time.nanoseconds();
	const oa::F32 weight = static_cast<oa::F32>(
		static_cast<oa::F64>(time - keyA.time.nanoseconds())
			/ static_cast<oa::F64>(segmentNanoseconds));
	if (inCurve.getInterpolation() == oa::AnimInterpolation::Linear) {
		const T value = interpolateCurveLinear(keyA.value, keyB.value, weight);
		if (not curveValueIsValid(value)) {
			return oa::Status::error(
				oa::StatusCode::OutOfRange,
				"AnimCurve linear interpolation produced an invalid value");
		}
		return normalizeCurveValue(value);
	}
	return interpolateCurveCubic(
		keyA,
		keyB,
		weight,
		static_cast<oa::F32>(segmentNanoseconds) / 1'000'000'000.0F);
}

[[nodiscard]] bool checkedBakeTime(
	oa::I64 inStartNanoseconds,
	oa::U32 inFrame,
	oa::F32 inFramesPerSecond,
	oa::I64& outNanoseconds) noexcept {
	const oa::F64 seconds = static_cast<oa::F64>(inFrame)
		/ static_cast<oa::F64>(inFramesPerSecond);
	const oa::F64 nanoseconds = seconds * 1'000'000'000.0;
	if (not oa::isFinite(nanoseconds)
		or nanoseconds < 0.0
		or nanoseconds >= 0x1.0p63) {
		return false;
	}
	const oa::I64 offset = static_cast<oa::I64>(nanoseconds);
	if (inStartNanoseconds > 0
		and offset > oa::Limits<oa::I64>::max() - inStartNanoseconds) {
		return false;
	}
	outNanoseconds = inStartNanoseconds + offset;
	return true;
}

template<typename T>
[[nodiscard]] oa::Result<T> sampleCurve(
	const oa::AnimCurve<T>& inCurve,
	oa::Duration inTime,
	oa::AnimWrapMode inWrapMode) {
	OA_RETURN_IF_ERROR(validateCurveMetadata(inCurve));
	if (not oa::isValidAnimWrapMode(inWrapMode)) {
		return oa::Status::invalidArgument("AnimCurve wrap mode is invalid");
	}
	return sampleCurveValidated(inCurve, inTime, inWrapMode);
}

template<typename T>
[[nodiscard]] oa::Result<oa::Vector<T>> bakeCurve(
	const oa::AnimCurve<T>& inCurve,
	oa::Duration inStartTime,
	oa::F32 inFramesPerSecond,
	oa::U32 inFrameCount,
	oa::AnimWrapMode inWrapMode) {
	OA_RETURN_IF_ERROR(validateCurveMetadata(inCurve));
	if (not oa::isValidAnimWrapMode(inWrapMode)) {
		return oa::Status::invalidArgument("AnimCurve wrap mode is invalid");
	}
	if (not oa::isFinite(inFramesPerSecond)
		or inFramesPerSecond <= 0.0F
		or inFrameCount == 0U) {
		return oa::Status::invalidArgument(
			"AnimCurve baking requires finite positive FPS and frame count");
	}

	oa::Vector<T> values;
	values.reserve(inFrameCount);
	for (oa::U32 frame = 0U; frame < inFrameCount; ++frame) {
		oa::I64 nanoseconds = 0;
		if (not checkedBakeTime(
			inStartTime.nanoseconds(),
			frame,
			inFramesPerSecond,
			nanoseconds)) {
			return oa::Status::error(
				oa::StatusCode::OutOfRange,
				"AnimCurve bake time exceeds the supported duration range");
		}
		oa::Result<T> value = sampleCurveValidated(
			inCurve,
			oa::Duration::fromNanoseconds(nanoseconds),
			inWrapMode);
		if (value.isError()) return value.getStatus();
		values.pushBack(value.getValue());
	}
	return values;
}

} // namespace

namespace oa {

oa::Result<oa::SkelPose> FnAnim::createPose(
	oa::U32 inSkeletonId,
	oa::Span<const oa::Transform> inLocalTransforms) {
	OA_RETURN_IF_ERROR(validateTransforms(inLocalTransforms));
	if (inLocalTransforms.size() > oa::Limits<oa::U32>::max()) {
		return oa::Status::error(
			oa::StatusCode::OutOfRange,
			"SkelPose joint count exceeds the supported range");
	}
	oa::SkelPose pose;
	pose.skeletonId_ = inSkeletonId;
	pose.localTransforms_.assign(
		inLocalTransforms.begin(), inLocalTransforms.end());
	return pose;
}

oa::Status FnAnim::validate(const oa::SkelPose& inPose) {
	if (inPose.localTransforms_.size() > oa::Limits<oa::U32>::max()) {
		return oa::Status::error(
			oa::StatusCode::OutOfRange,
			"SkelPose joint count exceeds the supported range");
	}
	return validateTransforms(inPose.getLocalTransforms());
}

oa::Result<oa::AnimClip> FnAnim::createClip(
	oa::U32 inSkeletonId,
	oa::U32 inJointCount,
	oa::F32 inFramesPerSecond,
	oa::Span<const oa::Transform> inSamples) {
	if (inJointCount == 0U
		or not oa::isFinite(inFramesPerSecond)
		or inFramesPerSecond <= 0.0F
		or inSamples.empty()
		or inSamples.size() % inJointCount != 0U) {
		return oa::Status::invalidArgument(
			"AnimClip requires positive joint/frame counts, finite positive FPS, and complete frames");
	}
	const oa::Usize frameCount = inSamples.size() / inJointCount;
	if (frameCount > oa::Limits<oa::U32>::max()) {
		return oa::Status::error(
			oa::StatusCode::OutOfRange,
			"AnimClip frame count exceeds the supported range");
	}
	OA_RETURN_IF_ERROR(validateTransforms(inSamples));

	oa::AnimClip clip;
	clip.skeletonId_ = inSkeletonId;
	clip.jointCount_ = inJointCount;
	clip.frameCount_ = static_cast<oa::U32>(frameCount);
	clip.framesPerSecond_ = inFramesPerSecond;
	clip.samples_.assign(inSamples.begin(), inSamples.end());
	return clip;
}

oa::Status FnAnim::validate(const oa::AnimClip& inClip) {
	OA_RETURN_IF_ERROR(validateClipMetadata(inClip));
	return validateTransforms(inClip.getSamples());
}

oa::Result<oa::SkelPose> FnAnim::getFrame(
	const oa::AnimClip& inClip,
	oa::U32 inFrame) {
	OA_RETURN_IF_ERROR(validateClipMetadata(inClip));
	if (inFrame >= inClip.frameCount_) {
		return oa::Status::error(
			oa::StatusCode::OutOfRange,
			"AnimClip frame is out of range");
	}
	const oa::Usize offset =
		static_cast<oa::Usize>(inFrame) * inClip.jointCount_;
	return createPose(
		inClip.skeletonId_,
		{inClip.samples_.data() + offset, inClip.jointCount_});
}

oa::Result<oa::SkelPose> FnAnim::sample(
	const oa::AnimClip& inClip,
	oa::F64 inTimeSeconds,
	oa::AnimWrapMode inWrapMode) {
	OA_RETURN_IF_ERROR(validateClipMetadata(inClip));
	if (not oa::isFinite(inTimeSeconds)) {
		return oa::Status::invalidArgument("AnimClip sample time must be finite");
	}
	if (not oa::isValidAnimWrapMode(inWrapMode)) {
		return oa::Status::invalidArgument("AnimClip wrap mode is invalid");
	}
	if (inClip.frameCount_ == 1U) return getFrame(inClip, 0U);

	const oa::F64 fps = static_cast<oa::F64>(inClip.framesPerSecond_);
	oa::F64 framePosition = 0.0;
	oa::U32 frameA = 0U;
	oa::U32 frameB = 0U;
	if (inWrapMode == oa::AnimWrapMode::Loop) {
		const oa::F64 period = static_cast<oa::F64>(inClip.frameCount_) / fps;
		oa::F64 wrapped = oa::fmod(inTimeSeconds, period);
		if (wrapped < 0.0) wrapped += period;
		framePosition = wrapped * fps;
		frameA = static_cast<oa::U32>(oa::floor(framePosition));
		if (frameA >= inClip.frameCount_) frameA = inClip.frameCount_ - 1U;
		frameB = (frameA + 1U) % inClip.frameCount_;
	} else {
		const oa::F64 maximumFrame =
			static_cast<oa::F64>(inClip.frameCount_ - 1U);
		const oa::F64 duration = maximumFrame / fps;
		if (inTimeSeconds <= 0.0) {
			framePosition = 0.0;
		} else if (inTimeSeconds >= duration) {
			framePosition = maximumFrame;
		} else {
			framePosition = inTimeSeconds * fps;
		}
		frameA = static_cast<oa::U32>(oa::floor(framePosition));
		frameB = oa::min(frameA + 1U, inClip.frameCount_ - 1U);
	}
	const oa::F32 weight = static_cast<oa::F32>(
		framePosition - oa::floor(framePosition));

	oa::SkelPose pose;
	pose.skeletonId_ = inClip.skeletonId_;
	pose.localTransforms_.reserve(inClip.jointCount_);
	const oa::Usize offsetA =
		static_cast<oa::Usize>(frameA) * inClip.jointCount_;
	const oa::Usize offsetB =
		static_cast<oa::Usize>(frameB) * inClip.jointCount_;
	for (oa::U32 joint = 0U; joint < inClip.jointCount_; ++joint) {
		pose.localTransforms_.pushBack(interpolateTransform(
			inClip.samples_[offsetA + joint],
			inClip.samples_[offsetB + joint],
			weight));
	}
	return pose;
}

oa::Result<oa::AnimCurve<oa::F32>> FnAnim::createCurve(
	oa::AnimInterpolation inInterpolation,
	oa::Span<const oa::Keyframe<oa::F32>> inKeyframes) {
	OA_RETURN_IF_ERROR(validateCurveKeyframes(inInterpolation, inKeyframes));
	oa::AnimCurve<oa::F32> curve;
	curve.interpolation_ = inInterpolation;
	curve.keyframes_.assign(inKeyframes.begin(), inKeyframes.end());
	return curve;
}

oa::Result<oa::AnimCurve<oa::vlm::Vec3>> FnAnim::createCurve(
	oa::AnimInterpolation inInterpolation,
	oa::Span<const oa::Keyframe<oa::vlm::Vec3>> inKeyframes) {
	OA_RETURN_IF_ERROR(validateCurveKeyframes(inInterpolation, inKeyframes));
	oa::AnimCurve<oa::vlm::Vec3> curve;
	curve.interpolation_ = inInterpolation;
	curve.keyframes_.assign(inKeyframes.begin(), inKeyframes.end());
	return curve;
}

oa::Result<oa::AnimCurve<oa::vlm::Quat>> FnAnim::createCurve(
	oa::AnimInterpolation inInterpolation,
	oa::Span<const oa::Keyframe<oa::vlm::Quat>> inKeyframes) {
	OA_RETURN_IF_ERROR(validateCurveKeyframes(inInterpolation, inKeyframes));
	oa::AnimCurve<oa::vlm::Quat> curve;
	curve.interpolation_ = inInterpolation;
	curve.keyframes_.assign(inKeyframes.begin(), inKeyframes.end());
	for (oa::Keyframe<oa::vlm::Quat>& keyframe : curve.keyframes_) {
		keyframe.value = keyframe.value.normalized();
	}
	return curve;
}

oa::Status FnAnim::validate(const oa::AnimCurve<oa::F32>& inCurve) {
	return validateCurveKeyframes(
		inCurve.getInterpolation(), inCurve.getKeyframes());
}

oa::Status FnAnim::validate(const oa::AnimCurve<oa::vlm::Vec3>& inCurve) {
	return validateCurveKeyframes(
		inCurve.getInterpolation(), inCurve.getKeyframes());
}

oa::Status FnAnim::validate(const oa::AnimCurve<oa::vlm::Quat>& inCurve) {
	return validateCurveKeyframes(
		inCurve.getInterpolation(), inCurve.getKeyframes());
}

oa::Result<oa::F32> FnAnim::sample(
	const oa::AnimCurve<oa::F32>& inCurve,
	oa::Duration inTime,
	oa::AnimWrapMode inWrapMode) {
	return sampleCurve(inCurve, inTime, inWrapMode);
}

oa::Result<oa::vlm::Vec3> FnAnim::sample(
	const oa::AnimCurve<oa::vlm::Vec3>& inCurve,
	oa::Duration inTime,
	oa::AnimWrapMode inWrapMode) {
	return sampleCurve(inCurve, inTime, inWrapMode);
}

oa::Result<oa::vlm::Quat> FnAnim::sample(
	const oa::AnimCurve<oa::vlm::Quat>& inCurve,
	oa::Duration inTime,
	oa::AnimWrapMode inWrapMode) {
	return sampleCurve(inCurve, inTime, inWrapMode);
}

oa::Result<oa::Vector<oa::F32>> FnAnim::bake(
	const oa::AnimCurve<oa::F32>& inCurve,
	oa::Duration inStartTime,
	oa::F32 inFramesPerSecond,
	oa::U32 inFrameCount,
	oa::AnimWrapMode inWrapMode) {
	return bakeCurve(
		inCurve, inStartTime, inFramesPerSecond, inFrameCount, inWrapMode);
}

oa::Result<oa::Vector<oa::vlm::Vec3>> FnAnim::bake(
	const oa::AnimCurve<oa::vlm::Vec3>& inCurve,
	oa::Duration inStartTime,
	oa::F32 inFramesPerSecond,
	oa::U32 inFrameCount,
	oa::AnimWrapMode inWrapMode) {
	return bakeCurve(
		inCurve, inStartTime, inFramesPerSecond, inFrameCount, inWrapMode);
}

oa::Result<oa::Vector<oa::vlm::Quat>> FnAnim::bake(
	const oa::AnimCurve<oa::vlm::Quat>& inCurve,
	oa::Duration inStartTime,
	oa::F32 inFramesPerSecond,
	oa::U32 inFrameCount,
	oa::AnimWrapMode inWrapMode) {
	return bakeCurve(
		inCurve, inStartTime, inFramesPerSecond, inFrameCount, inWrapMode);
}

template<typename T>
oa::Result<T> AnimCurve<T>::sample(
	oa::Duration inTime,
	oa::AnimWrapMode inWrapMode) const {
	return oa::FnAnim::sample(*this, inTime, inWrapMode);
}

template class AnimCurve<oa::F32>;
template class AnimCurve<oa::vlm::Vec3>;
template class AnimCurve<oa::vlm::Quat>;

} // namespace oa
