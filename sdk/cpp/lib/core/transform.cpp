#include <core/transform.h>

oa::vlm::Quat oa::eulerXyzDegToQuat(const oa::vlm::Vec3& inDegXyz) noexcept {
	return oa::vlm::quaternionFromEulerDegrees(
		inDegXyz, oa::vlm::EulerOrder::Xyz);
}

void oa::quaternionToSixD(
	const oa::vlm::Quat& inQuaternion,
	oa::F32 outRotation[6]) noexcept {
	const oa::vlm::Mat4 matrix = oa::vlm::quaternionToMatrix(inQuaternion);
	outRotation[0] = matrix.m[0][0];
	outRotation[1] = matrix.m[0][1];
	outRotation[2] = matrix.m[0][2];
	outRotation[3] = matrix.m[1][0];
	outRotation[4] = matrix.m[1][1];
	outRotation[5] = matrix.m[1][2];
}

oa::vlm::Quat oa::quaternionFromSixD(const oa::F32 inRotation[6]) noexcept {
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

oa::vlm::Mat4 oa::trsMatrix(
	const oa::vlm::Vec3& inScale,
	const oa::vlm::Quat& inRot,
	const oa::vlm::Vec3& inTrans) noexcept {
	return oa::vlm::composeTrs(inTrans, inRot, inScale);
}

oa::vlm::Vec3 oa::transformPoint(
	const oa::vlm::Mat4& inM,
	const oa::vlm::Vec3& inP
) noexcept {
	return oa::vlm::transformPoint(inP, inM);
}
