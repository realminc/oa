#include <oa/animation/skelPose.h>

#include <oa/core/std/assert.h>

namespace oa {

oa::U32 SkelPose::getSkeletonId() const noexcept {
	return skeletonId_;
}

oa::U32 SkelPose::getJointCount() const noexcept {
	return static_cast<oa::U32>(localTransforms_.size());
}

oa::Span<const oa::Transform> SkelPose::getLocalTransforms() const noexcept {
	return {localTransforms_.data(), localTransforms_.size()};
}

const oa::Transform& SkelPose::getLocalTransform(oa::U32 inJoint) const {
	OA_REQUIRE_MSG(inJoint < localTransforms_.size(), "SkelPose joint is out of range");
	return localTransforms_[inJoint];
}

} // namespace oa
