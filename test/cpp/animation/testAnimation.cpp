#include "../oaTest.h"

#include <oa/animation.h>
#include <oa/core/std/limits.h>

static_assert(not __is_constructible(oa::FnJoint));
static_assert(not __is_constructible(oa::FnAnim));
static_assert(not __is_constructible(oa::SkelPose));
static_assert(not __is_constructible(oa::AnimClip));
static_assert(not __is_constructible(oa::AnimCurve<oa::F32>));
static_assert(not __is_constructible(oa::AnimCurve<oa::vlm::Vec3>));
static_assert(not __is_constructible(oa::AnimCurve<oa::vlm::Quat>));

namespace {

void expectVecNear(
	const oa::vlm::Vec3& inActual,
	const oa::vlm::Vec3& inExpected,
	oa::F32 inTolerance = 1.0e-4F) {
	EXPECT_NEAR(inActual.x, inExpected.x, inTolerance);
	EXPECT_NEAR(inActual.y, inExpected.y, inTolerance);
	EXPECT_NEAR(inActual.z, inExpected.z, inTolerance);
}

void expectMatrixNear(
	const oa::vlm::Mat4& inActual,
	const oa::vlm::Mat4& inExpected,
	oa::F32 inTolerance = 1.0e-4F) {
	for (oa::U32 row = 0U; row < 4U; ++row) {
		for (oa::U32 column = 0U; column < 4U; ++column) {
			EXPECT_NEAR(
				inActual.m[row][column],
				inExpected.m[row][column],
				inTolerance);
		}
	}
}

oa::Transform transformAt(
	const oa::vlm::Vec3& inPosition,
	const oa::vlm::Quat& inRotation = oa::vlm::Quat::identity(),
	const oa::vlm::Vec3& inScale = {1.0F, 1.0F, 1.0F}) {
	return {inPosition, inRotation, inScale};
}

} // namespace

TEST(Joint, MemberAndFunctionSetOperateOnOneLockedValue) {
	oa::Joint joint;
	joint.getTransform().setPosition({2.0F, 3.0F, 4.0F});
	joint.setOrientation(oa::vlm::Quat::fromAxisAngle(
		{0.0F, 1.0F, 0.0F}, oa::vlm::radians(25.0F)));
	oa::FnJoint::setScaleOrientation(
		joint,
		oa::vlm::Quat::fromAxisAngle(
			{1.0F, 0.0F, 0.0F}, oa::vlm::radians(15.0F)));
	joint.setRotationOrder(oa::vlm::RotationOrder::Zyx);
	joint.setPreferredAngles({10.0F, 20.0F, 30.0F});
	joint.setStiffness({0.1F, 0.2F, 0.3F});

	EXPECT_TRUE(oa::FnJoint::validate(joint).isOk());
	EXPECT_EQ(joint.getRotationOrder(), oa::vlm::RotationOrder::Zyx);
	expectVecNear(
		oa::FnJoint::getPreferredAngles(joint), {10.0F, 20.0F, 30.0F});
	expectVecNear(joint.getStiffness(), {0.1F, 0.2F, 0.3F});
}

TEST(Joint, MatrixMatchesMayaJointFactorOrder) {
	oa::Joint joint;
	joint.getTransform().setPosition({7.0F, -3.0F, 2.0F});
	joint.getTransform().setScale({2.0F, 3.0F, 4.0F});
	joint.getTransform().setShear({0.1F, -0.2F, 0.3F});
	joint.setScaleOrientation(oa::vlm::Quat::fromAxisAngle(
		{1.0F, 0.0F, 0.0F}, oa::vlm::radians(13.0F)));
	joint.getTransform().setRotation(oa::vlm::Quat::fromAxisAngle(
		{0.0F, 1.0F, 0.0F}, oa::vlm::radians(29.0F)));
	joint.setOrientation(oa::vlm::Quat::fromAxisAngle(
		{0.0F, 0.0F, 1.0F}, oa::vlm::radians(-37.0F)));

	oa::vlm::AffineDecomposition scaleAndShear{};
	scaleAndShear.scale = joint.getTransform().getScale();
	scaleAndShear.shear = joint.getTransform().getShear();
	const oa::vlm::Vec3 parentScale{5.0F, 2.0F, 0.5F};
	const oa::vlm::Mat4 expected =
		oa::vlm::composeAffine(scaleAndShear)
		* oa::vlm::quaternionToMatrix(joint.getScaleOrientation())
		* oa::vlm::quaternionToMatrix(joint.getTransform().getRotation())
		* oa::vlm::quaternionToMatrix(joint.getOrientation())
		* oa::vlm::scaleMatrix(oa::vlm::Vec3{0.2F, 0.5F, 2.0F})
		* oa::vlm::translation(joint.getTransform().getPosition());
	expectMatrixNear(joint.getMatrix(parentScale), expected, 2.0e-5F);
	expectMatrixNear(
		oa::vlm::quaternionToMatrix(joint.getOrientedRotation()),
		oa::vlm::quaternionToMatrix(joint.getTransform().getRotation())
			* oa::vlm::quaternionToMatrix(joint.getOrientation()),
		2.0e-5F);
}

TEST(Joint, SegmentScaleCompensationCanBeDisabled) {
	oa::Joint joint;
	joint.getTransform().setScale({2.0F, 3.0F, 4.0F});
	const oa::vlm::Vec3 parentScale{5.0F, 6.0F, 7.0F};
	const oa::vlm::Mat4 compensated = joint.getMatrix(parentScale);
	joint.setSegmentScaleCompensate(false);
	const oa::vlm::Mat4 uncompensated = joint.getMatrix(parentScale);

	EXPECT_NE(compensated, uncompensated);
	expectMatrixNear(
		uncompensated,
		oa::vlm::scaleMatrix(oa::vlm::Vec3{2.0F, 3.0F, 4.0F}));
}

TEST(Joint, OrientationAndLimitsMatchAuthoringSemantics) {
	oa::Joint joint;
	joint.setOrientation(oa::vlm::quaternionFromEulerDegrees(
		oa::vlm::Vec3{0.0F, 0.0F, 90.0F},
		oa::vlm::RotationOrder::Xyz));
	joint.getTransform().setRotation(oa::vlm::quaternionFromEulerDegrees(
		oa::vlm::Vec3{0.0F, 0.0F, 90.0F},
		oa::vlm::RotationOrder::Xyz));
	expectVecNear(
		oa::vlm::rotateVector(
			joint.getOrientedRotation(), {1.0F, 0.0F, 0.0F}),
		{-1.0F, 0.0F, 0.0F});

	oa::JointRotationLimits limits;
	limits.minimumDegrees = {-10.0F, -20.0F, -30.0F};
	limits.maximumDegrees = {10.0F, 20.0F, 30.0F};
	limits.minimumEnabled = {true, false, true};
	limits.maximumEnabled = {true, true, false};
	joint.setRotationLimits(limits);
	expectVecNear(
		joint.clampRotationDegrees({-50.0F, 50.0F, 50.0F}),
		{-10.0F, 20.0F, 50.0F});
}

TEST(Joint, RejectsInvalidAuthoringState) {
	oa::Joint joint;
	EXPECT_DEATH(
		joint.setOrientation(oa::vlm::Quat{0.0F, 0.0F, 0.0F, 0.0F}), "");
	EXPECT_DEATH(
		joint.setRotationOrder(static_cast<oa::vlm::RotationOrder>(255U)), "");
	EXPECT_DEATH(joint.setStiffness({-1.0F, 0.0F, 0.0F}), "");

	oa::JointRotationLimits limits;
	limits.minimumDegrees.x = 1.0F;
	limits.maximumDegrees.x = -1.0F;
	limits.minimumEnabled.x = true;
	limits.maximumEnabled.x = true;
	EXPECT_DEATH(joint.setRotationLimits(limits), "");
	EXPECT_DEATH(
		static_cast<void>(joint.getMatrix({0.0F, 1.0F, 1.0F})), "");
}

TEST(Anim, CreatesPoseAndUniformClip) {
	const oa::Transform frames[] = {
		transformAt({0.0F, 0.0F, 0.0F}),
		transformAt({1.0F, 0.0F, 0.0F}),
		transformAt({10.0F, 0.0F, 0.0F}),
		transformAt({11.0F, 0.0F, 0.0F}),
	};
	auto clipResult = oa::FnAnim::createClip(42U, 2U, 1.0F, frames);
	ASSERT_TRUE(clipResult.isOk())
		<< clipResult.getStatus().toString().cStr();
	const oa::AnimClip& clip = *clipResult;
	EXPECT_EQ(clip.getSkeletonId(), 42U);
	EXPECT_EQ(clip.getJointCount(), 2U);
	EXPECT_EQ(clip.getFrameCount(), 2U);
	EXPECT_DOUBLE_EQ(clip.getDurationSeconds(), 1.0);

	auto frame = oa::FnAnim::getFrame(clip, 1U);
	ASSERT_TRUE(frame.isOk()) << frame.getStatus().toString().cStr();
	EXPECT_EQ(frame->getSkeletonId(), 42U);
	expectVecNear(
		frame->getLocalTransform(1U).getPosition(), {11.0F, 0.0F, 0.0F});
	EXPECT_DEATH(
		static_cast<void>(frame->getLocalTransform(frame->getJointCount())), "");
}

TEST(Anim, SamplesTranslationRotationScaleAndWrapModes) {
	const oa::vlm::Quat quarterTurn = oa::vlm::Quat::fromAxisAngle(
		{0.0F, 0.0F, 1.0F}, oa::vlm::radians(90.0F));
	oa::Transform frames[] = {
		transformAt({0.0F, 0.0F, 0.0F}),
		transformAt({10.0F, 0.0F, 0.0F}, quarterTurn, {3.0F, 3.0F, 3.0F}),
	};
	frames[1].setShear({0.2F, 0.4F, 0.6F});
	auto clipResult = oa::FnAnim::createClip(7U, 1U, 1.0F, frames);
	ASSERT_TRUE(clipResult.isOk());
	const oa::AnimClip& clip = *clipResult;

	auto middle = clip.sample(0.5);
	ASSERT_TRUE(middle.isOk());
	const oa::Transform& transform = middle->getLocalTransform(0U);
	expectVecNear(transform.getPosition(), {5.0F, 0.0F, 0.0F});
	expectVecNear(transform.getScale(), {2.0F, 2.0F, 2.0F});
	expectVecNear(transform.getShear(), {0.1F, 0.2F, 0.3F});
	expectVecNear(
		transform.getRotation().rotate({1.0F, 0.0F, 0.0F}),
		{0.70710678F, 0.70710678F, 0.0F});

	auto before = clip.sample(-100.0);
	auto after = clip.sample(100.0);
	auto looped = clip.sample(2.5, oa::AnimWrapMode::Loop);
	auto negativeLoop = clip.sample(-0.5, oa::AnimWrapMode::Loop);
	ASSERT_TRUE(
		before.isOk() and after.isOk()
		and looped.isOk() and negativeLoop.isOk());
	expectVecNear(
		before->getLocalTransform(0U).getPosition(), {0.0F, 0.0F, 0.0F});
	expectVecNear(
		after->getLocalTransform(0U).getPosition(), {10.0F, 0.0F, 0.0F});
	expectVecNear(
		looped->getLocalTransform(0U).getPosition(), {5.0F, 0.0F, 0.0F});
	expectVecNear(
		negativeLoop->getLocalTransform(0U).getPosition(),
		{5.0F, 0.0F, 0.0F});
}

TEST(Anim, RejectsMalformedValues) {
	const oa::Transform transform;
	EXPECT_TRUE(oa::FnAnim::createPose(0U, {}).isError());
	EXPECT_TRUE(oa::FnAnim::createClip(0U, 0U, 30.0F, {}).isError());
	EXPECT_TRUE(oa::FnAnim::createClip(0U, 2U, 30.0F, {&transform, 1U}).isError());

	const oa::Transform frames[] = {transform};
	auto clip = oa::FnAnim::createClip(0U, 1U, 30.0F, frames);
	ASSERT_TRUE(clip.isOk());
	EXPECT_TRUE(clip->sample(oa::Limits<oa::F64>::infinity()).isError());
	EXPECT_TRUE(clip->sample(
		0.0,
		static_cast<oa::AnimWrapMode>(oa::Limits<oa::U8>::max())).isError());
	auto extreme = clip->sample(oa::Limits<oa::F64>::max());
	ASSERT_TRUE(extreme.isOk());
	EXPECT_EQ(extreme->getJointCount(), 1U);
}

TEST(AnimCurve, LinearScalarUsesSignedClipRelativeTimeAndWrap) {
	const oa::Keyframe<oa::F32> keyframes[] = {
		{.time = oa::Duration::fromSeconds(-1), .value = -10.0F},
		{.time = oa::Duration::fromSeconds(0), .value = 0.0F},
		{.time = oa::Duration::fromSeconds(1), .value = 10.0F},
	};
	auto curveResult = oa::FnAnim::createCurve(
		oa::AnimInterpolation::Linear, keyframes);
	ASSERT_TRUE(curveResult.isOk())
		<< curveResult.getStatus().toString().cStr();
	const oa::AnimCurve<oa::F32>& curve = *curveResult;
	EXPECT_EQ(curve.getKeyframeCount(), 3U);
	EXPECT_EQ(curve.getStartTime(), oa::Duration::fromSeconds(-1));
	EXPECT_EQ(curve.getEndTime(), oa::Duration::fromSeconds(1));
	EXPECT_EQ(curve.getDuration(), oa::Duration::fromSeconds(2));
	EXPECT_TRUE(oa::FnAnim::validate(curve).isOk());

	auto middle = curve.sample(oa::Duration::fromMilliseconds(500));
	auto before = curve.sample(oa::Duration::fromSeconds(-100));
	auto after = curve.sample(oa::Duration::fromSeconds(100));
	auto looped = curve.sample(
		oa::Duration::fromMilliseconds(1'500), oa::AnimWrapMode::Loop);
	auto negativeLoop = curve.sample(
		oa::Duration::fromMilliseconds(-1'500), oa::AnimWrapMode::Loop);
	ASSERT_TRUE(
		middle.isOk() and before.isOk() and after.isOk()
		and looped.isOk() and negativeLoop.isOk());
	EXPECT_FLOAT_EQ(*middle, 5.0F);
	EXPECT_FLOAT_EQ(*before, -10.0F);
	EXPECT_FLOAT_EQ(*after, 10.0F);
	EXPECT_FLOAT_EQ(*looped, -5.0F);
	EXPECT_FLOAT_EQ(*negativeLoop, 5.0F);
}

TEST(AnimCurve, StepUsesTheLeftKeyAndExactKeysAdvance) {
	const oa::Keyframe<oa::F32> keyframes[] = {
		{.time = oa::Duration::fromSeconds(0), .value = 2.0F},
		{.time = oa::Duration::fromSeconds(1), .value = 7.0F},
		{.time = oa::Duration::fromSeconds(2), .value = 9.0F},
	};
	auto curve = oa::FnAnim::createCurve(
		oa::AnimInterpolation::Step, keyframes);
	ASSERT_TRUE(curve.isOk());
	auto beforeSecond = curve->sample(oa::Duration::fromMilliseconds(999));
	auto atSecond = curve->sample(oa::Duration::fromSeconds(1));
	ASSERT_TRUE(beforeSecond.isOk() and atSecond.isOk());
	EXPECT_FLOAT_EQ(*beforeSecond, 2.0F);
	EXPECT_FLOAT_EQ(*atSecond, 7.0F);
}

TEST(AnimCurve, CubicSplineUsesPerSecondTangents) {
	const oa::Keyframe<oa::F32> keyframes[] = {
		{
			.time = oa::Duration::fromSeconds(0),
			.value = 0.0F,
			.outTangent = 2.0F,
		},
		{
			.time = oa::Duration::fromSeconds(2),
			.value = 2.0F,
		},
	};
	auto curve = oa::FnAnim::createCurve(
		oa::AnimInterpolation::CubicSpline, keyframes);
	ASSERT_TRUE(curve.isOk());
	auto value = curve->sample(oa::Duration::fromSeconds(1));
	ASSERT_TRUE(value.isOk());
	EXPECT_FLOAT_EQ(*value, 1.5F);
}

TEST(AnimCurve, SupportsVec3AndNormalizedQuaternionCurves) {
	const oa::Keyframe<oa::vlm::Quat> defaultQuaternionKey;
	EXPECT_FLOAT_EQ(defaultQuaternionKey.value.normSquared(), 1.0F);
	EXPECT_FLOAT_EQ(defaultQuaternionKey.inTangent.normSquared(), 0.0F);
	EXPECT_FLOAT_EQ(defaultQuaternionKey.outTangent.normSquared(), 0.0F);

	const oa::Keyframe<oa::vlm::Vec3> positionKeys[] = {
		{
			.time = oa::Duration::fromSeconds(0),
			.value = {0.0F, 2.0F, 4.0F},
		},
		{
			.time = oa::Duration::fromSeconds(1),
			.value = {10.0F, 4.0F, 0.0F},
		},
	};
	auto positions = oa::FnAnim::createCurve(
		oa::AnimInterpolation::Linear, positionKeys);
	ASSERT_TRUE(positions.isOk());
	auto position = positions->sample(oa::Duration::fromMilliseconds(250));
	ASSERT_TRUE(position.isOk());
	expectVecNear(*position, {2.5F, 2.5F, 3.0F});

	const oa::vlm::Quat quarterTurn = oa::vlm::Quat::fromAxisAngle(
		{0.0F, 0.0F, 1.0F}, oa::vlm::radians(90.0F));
	const oa::Keyframe<oa::vlm::Quat> rotationKeys[] = {
		{
			.time = oa::Duration::fromSeconds(0),
			.value = {0.0F, 0.0F, 0.0F, 2.0F},
			.inTangent = {0.0F, 0.0F, 0.0F, 0.0F},
			.outTangent = {0.0F, 0.0F, 0.0F, 0.0F},
		},
		{
			.time = oa::Duration::fromSeconds(1),
			.value = quarterTurn,
			.inTangent = {0.0F, 0.0F, 0.0F, 0.0F},
			.outTangent = {0.0F, 0.0F, 0.0F, 0.0F},
		},
	};
	auto rotations = oa::FnAnim::createCurve(
		oa::AnimInterpolation::CubicSpline, rotationKeys);
	ASSERT_TRUE(rotations.isOk());
	EXPECT_NEAR(rotations->getKeyframes().front().value.norm(), 1.0F, 1.0e-6F);
	auto rotation = rotations->sample(oa::Duration::fromMilliseconds(500));
	ASSERT_TRUE(rotation.isOk());
	EXPECT_NEAR(rotation->norm(), 1.0F, 1.0e-6F);
	expectVecNear(
		rotation->rotate({1.0F, 0.0F, 0.0F}),
		{0.70710678F, 0.70710678F, 0.0F});
}

TEST(AnimCurve, BakeUsesExplicitUniformSampleContract) {
	const oa::Keyframe<oa::F32> keyframes[] = {
		{.time = oa::Duration::fromSeconds(0), .value = 0.0F},
		{.time = oa::Duration::fromSeconds(1), .value = 10.0F},
	};
	auto curve = oa::FnAnim::createCurve(
		oa::AnimInterpolation::Linear, keyframes);
	ASSERT_TRUE(curve.isOk());
	auto baked = oa::FnAnim::bake(
		*curve, oa::Duration{}, 2.0F, 3U);
	ASSERT_TRUE(baked.isOk());
	ASSERT_EQ(baked->size(), 3U);
	EXPECT_FLOAT_EQ((*baked)[0], 0.0F);
	EXPECT_FLOAT_EQ((*baked)[1], 5.0F);
	EXPECT_FLOAT_EQ((*baked)[2], 10.0F);

	EXPECT_TRUE(oa::FnAnim::bake(
		*curve, oa::Duration{}, 0.0F, 3U).isError());
	EXPECT_TRUE(oa::FnAnim::bake(
		*curve, oa::Duration{}, 2.0F, 0U).isError());
	EXPECT_TRUE(oa::FnAnim::bake(
		*curve,
		oa::Duration::fromNanoseconds(oa::Limits<oa::I64>::max()),
		1.0F,
		2U).isError());
}

TEST(AnimCurve, RejectsMalformedKeysAndModes) {
	const oa::Keyframe<oa::F32> duplicateTimes[] = {
		{.time = oa::Duration::fromSeconds(1), .value = 1.0F},
		{.time = oa::Duration::fromSeconds(1), .value = 2.0F},
	};
	const oa::Keyframe<oa::F32> nonFinite[] = {
		{
			.time = oa::Duration{},
			.value = oa::Limits<oa::F32>::quietNaN(),
		},
	};
	const oa::Keyframe<oa::F32> excessiveSpan[] = {
		{
			.time = oa::Duration::fromNanoseconds(
				oa::Limits<oa::I64>::min()),
			.value = 1.0F,
		},
		{
			.time = oa::Duration::fromNanoseconds(
				oa::Limits<oa::I64>::max()),
			.value = 2.0F,
		},
	};
	const oa::Keyframe<oa::vlm::Quat> zeroQuaternion[] = {
		{
			.time = oa::Duration{},
			.value = {0.0F, 0.0F, 0.0F, 0.0F},
			.inTangent = {0.0F, 0.0F, 0.0F, 0.0F},
			.outTangent = {0.0F, 0.0F, 0.0F, 0.0F},
		},
	};
	EXPECT_TRUE(oa::FnAnim::createCurve(
		oa::AnimInterpolation::Linear,
		oa::Span<const oa::Keyframe<oa::F32>>{}).isError());
	EXPECT_TRUE(oa::FnAnim::createCurve(
		oa::AnimInterpolation::Linear, duplicateTimes).isError());
	EXPECT_TRUE(oa::FnAnim::createCurve(
		oa::AnimInterpolation::Linear, nonFinite).isError());
	EXPECT_TRUE(oa::FnAnim::createCurve(
		oa::AnimInterpolation::Linear, excessiveSpan).isError());
	EXPECT_TRUE(oa::FnAnim::createCurve(
		oa::AnimInterpolation::Linear, zeroQuaternion).isError());
	EXPECT_TRUE(oa::FnAnim::createCurve(
		static_cast<oa::AnimInterpolation>(255U), duplicateTimes).isError());

	const oa::Keyframe<oa::F32> valid[] = {
		{.time = oa::Duration{}, .value = 1.0F},
	};
	auto curve = oa::FnAnim::createCurve(
		oa::AnimInterpolation::Linear, valid);
	ASSERT_TRUE(curve.isOk());
	EXPECT_TRUE(curve->sample(
		oa::Duration{}, static_cast<oa::AnimWrapMode>(255U)).isError());
}
