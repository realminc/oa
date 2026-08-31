#include <oa/core/fnTransform.h>

#include <oa/core/std/assert.h>

namespace oa {

bool FnTransform::isFinite(const oa::Transform& inTransform) noexcept {
	return inTransform.translation_.isFinite()
		and inTransform.rotation_.isFinite()
		and inTransform.scale_.isFinite()
		and inTransform.shear_.isFinite()
		and oa::isFinite(inTransform.rotation_.normSquared())
		and inTransform.rotation_.normSquared() > 0.0F;
}

oa::Status FnTransform::validate(const oa::Transform& inTransform) {
	if (not isFinite(inTransform)) {
		return oa::Status::invalidArgument("Transform requires finite translation, rotation, scale, and shear with a non-zero quaternion");
	}
	return oa::Status::ok();
}

void FnTransform::setPosition(
	oa::Transform& inTransform,
	const oa::vlm::Vec3& inPosition) {
	OA_REQUIRE_MSG(inPosition.isFinite(), "Transform position must be finite");
	inTransform.translation_ = inPosition;
}

void FnTransform::setRotation(
	oa::Transform& inTransform,
	const oa::vlm::Quat& inRotation) {
	OA_REQUIRE_MSG(
		inRotation.isFinite() and inRotation.normSquared() > 0.0F,
		"Transform rotation must be a finite non-zero quaternion");
	inTransform.rotation_ = inRotation.normalized();
}

void FnTransform::setScale(
	oa::Transform& inTransform,
	const oa::vlm::Vec3& inScale) {
	OA_REQUIRE_MSG(inScale.isFinite(), "Transform scale must be finite");
	inTransform.scale_ = inScale;
}

void FnTransform::setShear(
	oa::Transform& inTransform,
	const oa::vlm::Vec3& inShear) {
	OA_REQUIRE_MSG(inShear.isFinite(), "Transform shear must be finite");
	inTransform.shear_ = inShear;
}

const oa::vlm::Vec3& FnTransform::getPosition(
	const oa::Transform& inTransform) noexcept {
	return inTransform.translation_;
}

const oa::vlm::Quat& FnTransform::getRotation(
	const oa::Transform& inTransform) noexcept {
	return inTransform.rotation_;
}

const oa::vlm::Vec3& FnTransform::getScale(
	const oa::Transform& inTransform) noexcept {
	return inTransform.scale_;
}

const oa::vlm::Vec3& FnTransform::getShear(
	const oa::Transform& inTransform) noexcept {
	return inTransform.shear_;
}

oa::vlm::Vec3 FnTransform::getForward(
	const oa::Transform& inTransform) noexcept {
	return oa::vlm::rotateVector(
		inTransform.rotation_.normalized(), {0.0F, 0.0F, -1.0F});
}

oa::vlm::Vec3 FnTransform::getRight(
	const oa::Transform& inTransform) noexcept {
	return oa::vlm::rotateVector(
		inTransform.rotation_.normalized(), {1.0F, 0.0F, 0.0F});
}

oa::vlm::Vec3 FnTransform::getUp(
	const oa::Transform& inTransform) noexcept {
	return oa::vlm::rotateVector(
		inTransform.rotation_.normalized(), {0.0F, 1.0F, 0.0F});
}

void FnTransform::lookAt(
	oa::Transform& inTransform,
	const oa::vlm::Vec3& inTarget,
	const oa::vlm::Vec3& inUp) {
	OA_REQUIRE_MSG(inTarget.isFinite(), "Transform look-at target must be finite");
	setRotation(
		inTransform,
		oa::vlm::lookRotation(inTarget - inTransform.translation_, inUp));
}

void FnTransform::setRotationDegrees(
	oa::Transform& inTransform,
	const oa::vlm::Vec3& inDegrees,
	oa::vlm::RotationOrder inOrder) {
	setRotation(
		inTransform,
		oa::vlm::quaternionFromEulerDegrees(inDegrees, inOrder));
}

oa::vlm::Vec3 FnTransform::getRotationDegrees(
	const oa::Transform& inTransform,
	oa::vlm::RotationOrder inOrder) noexcept {
	return oa::vlm::quaternionToEulerDegrees(inTransform.rotation_, inOrder);
}

void FnTransform::panLocal(
	oa::Transform& inTransform,
	oa::F32 inRight,
	oa::F32 inUp,
	oa::F32 inForward) {
	OA_REQUIRE_MSG(
		oa::isFinite(inRight) and oa::isFinite(inUp) and oa::isFinite(inForward),
		"Transform local pan distances must be finite");
	inTransform.translation_ += getRight(inTransform) * inRight;
	inTransform.translation_ += getUp(inTransform) * inUp;
	inTransform.translation_ += getForward(inTransform) * inForward;
}

oa::vlm::Mat4 FnTransform::getMatrix(
	const oa::Transform& inTransform) noexcept {
	OA_REQUIRE_MSG(isFinite(inTransform), "Transform matrix requires valid finite state");
	oa::vlm::AffineDecomposition decomposition{};
	decomposition.translation = inTransform.translation_;
	decomposition.rotation = inTransform.rotation_.normalized();
	decomposition.scale = inTransform.scale_;
	decomposition.shear = inTransform.shear_;
	return oa::vlm::composeAffine(decomposition);
}

oa::Status FnTransform::setMatrix(
	oa::Transform& outTransform,
	const oa::vlm::Mat4& inMatrix) noexcept {
	oa::vlm::AffineDecomposition decomposition{};
	if (not oa::vlm::tryDecomposeAffine(inMatrix, decomposition)) {
		return oa::Status::invalidArgument(
			"Transform matrix must be finite, affine, and non-singular");
	}
	oa::Transform result{};
	result.translation_ = decomposition.translation;
	result.rotation_ = decomposition.rotation;
	result.scale_ = decomposition.scale;
	result.shear_ = decomposition.shear;
	outTransform = result;
	return oa::Status::ok();
}

void FnTransform::quaternionToSixD(
	const oa::vlm::Quat& inQuaternion,
	oa::F32 outRotation[6]) noexcept {
	OA_REQUIRE_MSG(outRotation != nullptr, "Transform 6D rotation output is null");
	const oa::vlm::Mat4 matrix =
		oa::vlm::quaternionToMatrix(inQuaternion.normalized());
	outRotation[0] = matrix.m[0][0];
	outRotation[1] = matrix.m[0][1];
	outRotation[2] = matrix.m[0][2];
	outRotation[3] = matrix.m[1][0];
	outRotation[4] = matrix.m[1][1];
	outRotation[5] = matrix.m[1][2];
}

oa::vlm::Quat FnTransform::quaternionFromSixD(
	const oa::F32 inRotation[6]) noexcept {
	OA_REQUIRE_MSG(inRotation != nullptr, "Transform 6D rotation input is null");
	oa::vlm::Mat3 basis{};
	if (not oa::vlm::tryOrthonormalBasis(
		{inRotation[0], inRotation[1], inRotation[2]},
		{inRotation[3], inRotation[4], inRotation[5]}, basis)) {
		return oa::vlm::Quat::identity();
	}
	oa::vlm::Quat rotation{};
	if (not oa::vlm::tryQuaternionFromRotationMatrix(basis, rotation)) {
		return oa::vlm::Quat::identity();
	}
	return rotation;
}

} // namespace oa
