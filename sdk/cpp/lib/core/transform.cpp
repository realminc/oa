#include <core/transform.h>

#include <cmath>

oa::vlm::Quat oa::eulerXyzDegToQuat(const oa::vlm::Vec3& inDegXyz) noexcept {
	const oa::F32 deg2rad = oa::vlm::Pi<oa::F32> / 180.0f;
	const oa::F32 hx = inDegXyz.x * deg2rad * 0.5f;
	const oa::F32 hy = inDegXyz.y * deg2rad * 0.5f;
	const oa::F32 hz = inDegXyz.z * deg2rad * 0.5f;
	const oa::vlm::Quat qx = { std::sin(hx), 0.0f, 0.0f, std::cos(hx) };
	const oa::vlm::Quat qy = { 0.0f, std::sin(hy), 0.0f, std::cos(hy) };
	const oa::vlm::Quat qz = { 0.0f, 0.0f, std::sin(hz), std::cos(hz) };
	// apply X first, then Y, then Z  ⇒  qz ⊗ qy ⊗ qx.
	return qz * (qy * qx);
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
	const oa::vlm::Vec3 first = oa::vlm::normalize(oa::vlm::Vec3{
		inRotation[0], inRotation[1], inRotation[2]});
	const oa::vlm::Vec3 candidate{
		inRotation[3], inRotation[4], inRotation[5]};
	const oa::vlm::Vec3 second = oa::vlm::normalize(candidate - first
		* oa::vlm::dot(first, candidate));
	const oa::vlm::Vec3 third = oa::vlm::cross(first, second);
	oa::vlm::Mat4 matrix = oa::vlm::Mat4::identity();
	matrix.m[0][0] = first.x;
	matrix.m[0][1] = first.y;
	matrix.m[0][2] = first.z;
	matrix.m[1][0] = second.x;
	matrix.m[1][1] = second.y;
	matrix.m[1][2] = second.z;
	matrix.m[2][0] = third.x;
	matrix.m[2][1] = third.y;
	matrix.m[2][2] = third.z;
	return oa::vlm::quaternionFromMatrix(matrix);
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
