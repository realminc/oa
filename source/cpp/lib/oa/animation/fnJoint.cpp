#include <oa/animation/fnJoint.h>

#include <oa/core/fnTransform.h>
#include <oa/core/std/assert.h>
#include <oa/core/std/scalarMath.h>

namespace {

[[nodiscard]] bool validQuaternion(const oa::vlm::Quat& inValue) noexcept {
	return inValue.isFinite()
		and oa::isFinite(inValue.normSquared())
		and inValue.normSquared() > 0.0F;
}

[[nodiscard]] bool validStiffness(const oa::vlm::Vec3& inValue) noexcept {
	return inValue.isFinite()
		and inValue.x >= 0.0F
		and inValue.y >= 0.0F
		and inValue.z >= 0.0F;
}

[[nodiscard]] bool validLimits(
	const oa::JointRotationLimits& inLimits) noexcept {
	if (not inLimits.minimumDegrees.isFinite()
		or not inLimits.maximumDegrees.isFinite()) {
		return false;
	}
	const bool xValid = not (
		inLimits.minimumEnabled.x and inLimits.maximumEnabled.x)
		or inLimits.minimumDegrees.x <= inLimits.maximumDegrees.x;
	const bool yValid = not (
		inLimits.minimumEnabled.y and inLimits.maximumEnabled.y)
		or inLimits.minimumDegrees.y <= inLimits.maximumDegrees.y;
	const bool zValid = not (
		inLimits.minimumEnabled.z and inLimits.maximumEnabled.z)
		or inLimits.minimumDegrees.z <= inLimits.maximumDegrees.z;
	return xValid and yValid and zValid;
}

[[nodiscard]] oa::F32 clampAxis(
	oa::F32 inValue,
	oa::F32 inMinimum,
	oa::F32 inMaximum,
	bool inMinimumEnabled,
	bool inMaximumEnabled) noexcept {
	if (inMinimumEnabled and inValue < inMinimum) return inMinimum;
	if (inMaximumEnabled and inValue > inMaximum) return inMaximum;
	return inValue;
}

} // namespace

namespace oa {

oa::Status FnJoint::validate(const oa::Joint& inJoint) {
	OA_RETURN_IF_ERROR(oa::FnTransform::validate(inJoint.transform_));
	if (not validQuaternion(inJoint.orientation_)
		or not validQuaternion(inJoint.scaleOrientation_)) {
		return oa::Status::invalidArgument(
			"Joint orientations must be finite non-zero quaternions");
	}
	if (not oa::vlm::isValidRotationOrder(inJoint.rotationOrder_)) {
		return oa::Status::invalidArgument("Joint rotation order is invalid");
	}
	if (not validLimits(inJoint.rotationLimits_)) {
		return oa::Status::invalidArgument("Joint rotation limits are invalid");
	}
	if (not inJoint.preferredAngles_.isFinite()) {
		return oa::Status::invalidArgument(
			"Joint preferred angles must be finite");
	}
	if (not validStiffness(inJoint.stiffness_)) {
		return oa::Status::invalidArgument(
			"Joint stiffness must be finite and non-negative");
	}
	return oa::Status::ok();
}

oa::Transform& FnJoint::getTransform(oa::Joint& inJoint) noexcept {
	return inJoint.transform_;
}

const oa::Transform& FnJoint::getTransform(
	const oa::Joint& inJoint) noexcept {
	return inJoint.transform_;
}

void FnJoint::setOrientation(
	oa::Joint& inJoint,
	const oa::vlm::Quat& inOrientation) {
	OA_REQUIRE_MSG(
		validQuaternion(inOrientation),
		"Joint orientation must be a finite non-zero quaternion");
	inJoint.orientation_ = inOrientation.normalized();
}

const oa::vlm::Quat& FnJoint::getOrientation(
	const oa::Joint& inJoint) noexcept {
	return inJoint.orientation_;
}

void FnJoint::setScaleOrientation(
	oa::Joint& inJoint,
	const oa::vlm::Quat& inOrientation) {
	OA_REQUIRE_MSG(
		validQuaternion(inOrientation),
		"Joint scale orientation must be a finite non-zero quaternion");
	inJoint.scaleOrientation_ = inOrientation.normalized();
}

const oa::vlm::Quat& FnJoint::getScaleOrientation(
	const oa::Joint& inJoint) noexcept {
	return inJoint.scaleOrientation_;
}

void FnJoint::setSegmentScaleCompensate(
	oa::Joint& inJoint,
	bool inEnabled) noexcept {
	inJoint.segmentScaleCompensate_ = inEnabled;
}

bool FnJoint::getSegmentScaleCompensate(
	const oa::Joint& inJoint) noexcept {
	return inJoint.segmentScaleCompensate_;
}

void FnJoint::setRotationOrder(
	oa::Joint& inJoint,
	oa::vlm::RotationOrder inOrder) {
	OA_REQUIRE_MSG(
		oa::vlm::isValidRotationOrder(inOrder),
		"Joint rotation order is invalid");
	inJoint.rotationOrder_ = inOrder;
}

oa::vlm::RotationOrder FnJoint::getRotationOrder(
	const oa::Joint& inJoint) noexcept {
	return inJoint.rotationOrder_;
}

void FnJoint::setDegreesOfFreedom(
	oa::Joint& inJoint,
	const oa::JointDegreesOfFreedom& inDegreesOfFreedom) noexcept {
	inJoint.degreesOfFreedom_ = inDegreesOfFreedom;
}

const oa::JointDegreesOfFreedom& FnJoint::getDegreesOfFreedom(
	const oa::Joint& inJoint) noexcept {
	return inJoint.degreesOfFreedom_;
}

void FnJoint::setRotationLimits(
	oa::Joint& inJoint,
	const oa::JointRotationLimits& inLimits) {
	OA_REQUIRE_MSG(validLimits(inLimits), "Joint rotation limits are invalid");
	inJoint.rotationLimits_ = inLimits;
}

const oa::JointRotationLimits& FnJoint::getRotationLimits(
	const oa::Joint& inJoint) noexcept {
	return inJoint.rotationLimits_;
}

void FnJoint::setPreferredAngles(
	oa::Joint& inJoint,
	const oa::vlm::Vec3& inDegrees) {
	OA_REQUIRE_MSG(inDegrees.isFinite(), "Joint preferred angles must be finite");
	inJoint.preferredAngles_ = inDegrees;
}

const oa::vlm::Vec3& FnJoint::getPreferredAngles(
	const oa::Joint& inJoint) noexcept {
	return inJoint.preferredAngles_;
}

void FnJoint::setStiffness(
	oa::Joint& inJoint,
	const oa::vlm::Vec3& inStiffness) {
	OA_REQUIRE_MSG(
		validStiffness(inStiffness),
		"Joint stiffness must be finite and non-negative");
	inJoint.stiffness_ = inStiffness;
}

const oa::vlm::Vec3& FnJoint::getStiffness(
	const oa::Joint& inJoint) noexcept {
	return inJoint.stiffness_;
}

oa::vlm::Quat FnJoint::getOrientedRotation(
	const oa::Joint& inJoint) noexcept {
	return (inJoint.orientation_ * inJoint.transform_.getRotation()).normalized();
}

oa::vlm::Vec3 FnJoint::clampRotationDegrees(
	const oa::Joint& inJoint,
	const oa::vlm::Vec3& inDegrees) noexcept {
	OA_REQUIRE_MSG(inDegrees.isFinite(), "Joint rotation angles must be finite");
	const oa::JointRotationLimits& limits = inJoint.rotationLimits_;
	return {
		clampAxis(
			inDegrees.x, limits.minimumDegrees.x, limits.maximumDegrees.x,
			limits.minimumEnabled.x, limits.maximumEnabled.x),
		clampAxis(
			inDegrees.y, limits.minimumDegrees.y, limits.maximumDegrees.y,
			limits.minimumEnabled.y, limits.maximumEnabled.y),
		clampAxis(
			inDegrees.z, limits.minimumDegrees.z, limits.maximumDegrees.z,
			limits.minimumEnabled.z, limits.maximumEnabled.z),
	};
}

oa::vlm::Mat4 FnJoint::getMatrix(
	const oa::Joint& inJoint,
	const oa::vlm::Vec3& inParentScale) noexcept {
	OA_REQUIRE_MSG(validate(inJoint).isOk(), "Joint matrix requires valid state");
	OA_REQUIRE_MSG(inParentScale.isFinite(), "Joint parent scale must be finite");

	oa::vlm::Vec3 inverseParentScale{1.0F, 1.0F, 1.0F};
	if (inJoint.segmentScaleCompensate_) {
		OA_REQUIRE_MSG(
			inParentScale.x != 0.0F
				and inParentScale.y != 0.0F
				and inParentScale.z != 0.0F,
			"Joint segment-scale compensation requires non-zero parent scale");
		inverseParentScale = {
			1.0F / inParentScale.x,
			1.0F / inParentScale.y,
			1.0F / inParentScale.z,
		};
	}

	oa::vlm::AffineDecomposition scaleAndShear{};
	scaleAndShear.scale = inJoint.transform_.getScale();
	scaleAndShear.shear = inJoint.transform_.getShear();
	return oa::vlm::composeAffine(scaleAndShear)
		* oa::vlm::quaternionToMatrix(inJoint.scaleOrientation_)
		* oa::vlm::quaternionToMatrix(inJoint.transform_.getRotation())
		* oa::vlm::quaternionToMatrix(inJoint.orientation_)
		* oa::vlm::scaleMatrix(inverseParentScale)
		* oa::vlm::translation(inJoint.transform_.getPosition());
}

} // namespace oa
