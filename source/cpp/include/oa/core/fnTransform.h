// oa::FnTransform — stateless operations on oa::Transform values.

#pragma once

#include <oa/core/transform.h>

namespace oa {

class FnTransform final {
public:
	FnTransform() = delete;

	[[nodiscard]] static bool isFinite(const oa::Transform& inTransform) noexcept;
	[[nodiscard]] static oa::Status validate(const oa::Transform& inTransform);

	static void setPosition(oa::Transform& inTransform,	const oa::vlm::Vec3& inPosition);
	static void setRotation(oa::Transform& inTransform, const oa::vlm::Quat& inRotation);
	static void setScale(oa::Transform& inTransform, const oa::vlm::Vec3& inScale);
	static void setShear(oa::Transform& inTransform, const oa::vlm::Vec3& inShear);

	[[nodiscard]] static const oa::vlm::Vec3& getPosition(const oa::Transform& inTransform) noexcept;
	[[nodiscard]] static const oa::vlm::Quat& getRotation(const oa::Transform& inTransform) noexcept;
	[[nodiscard]] static const oa::vlm::Vec3& getScale(const oa::Transform& inTransform) noexcept;
	[[nodiscard]] static const oa::vlm::Vec3& getShear(const oa::Transform& inTransform) noexcept;

	[[nodiscard]] static oa::vlm::Vec3 getForward(const oa::Transform& inTransform) noexcept;
	[[nodiscard]] static oa::vlm::Vec3 getRight(const oa::Transform& inTransform) noexcept;
	[[nodiscard]] static oa::vlm::Vec3 getUp(const oa::Transform& inTransform) noexcept;

	static void lookAt(
		oa::Transform& inTransform,
		const oa::vlm::Vec3& inTarget,
		const oa::vlm::Vec3& inUp = {0.0F, 1.0F, 0.0F}
	);
	static void setRotationDegrees(
		oa::Transform& inTransform,
		const oa::vlm::Vec3& inDegrees,
		oa::vlm::RotationOrder inOrder = oa::vlm::RotationOrder::Xyz
	);
	[[nodiscard]] static oa::vlm::Vec3 getRotationDegrees(
		const oa::Transform& inTransform,
		oa::vlm::RotationOrder inOrder = oa::vlm::RotationOrder::Xyz
	) noexcept;
	static void panLocal(
		oa::Transform& inTransform,
		oa::F32 inRight,
		oa::F32 inUp,
		oa::F32 inForward
	);

	[[nodiscard]] static oa::vlm::Mat4 getMatrix(
		const oa::Transform& inTransform
	) noexcept;
	// Decomposes a finite, non-singular affine matrix. Failure leaves the output
	// unchanged.
	[[nodiscard]] static oa::Status setMatrix(
		oa::Transform& outTransform,
		const oa::vlm::Mat4& inMatrix
	) noexcept;

	static void quaternionToSixD(
		const oa::vlm::Quat& inQuaternion,
		oa::F32 outRotation[6]
	) noexcept;
	[[nodiscard]] static oa::vlm::Quat quaternionFromSixD(
		const oa::F32 inRotation[6]
	) noexcept;
};

} // namespace oa
