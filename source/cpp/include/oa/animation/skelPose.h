// oa::SkelPose — one validated local-transform sample for a skeleton.

#pragma once

#include <oa/core/std/span.h>
#include <oa/core/transform.h>
#include <oa/core/types.h>

namespace oa {

class FnAnim;

class SkelPose {
public:
	[[nodiscard]] oa::U32 getSkeletonId() const noexcept;
	[[nodiscard]] oa::U32 getJointCount() const noexcept;
	[[nodiscard]] oa::Span<const oa::Transform> getLocalTransforms() const noexcept;
	[[nodiscard]] const oa::Transform& getLocalTransform(oa::U32 inJoint) const;

private:
	friend class FnAnim;
	SkelPose() = default;

	oa::U32 skeletonId_ = 0U;
	oa::Vector<oa::Transform> localTransforms_;
};

} // namespace oa
