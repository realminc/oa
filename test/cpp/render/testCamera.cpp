#include <oa/render/camera.h>
#include <oa/render/fnCamera.h>
#include <oa/core/fnTransform.h>

#include <gtest/gtest.h>

#include <cmath>

static_assert(not __is_constructible(oa::FnCamera));

namespace {

constexpr oa::F32 tolerance = 1.0e-5F;

void expectMatrixNear(const oa::vlm::Mat4& inActual, const oa::vlm::Mat4& inExpected) {
	for (oa::U32 column = 0; column < 4; ++column) {
		for (oa::U32 row = 0; row < 4; ++row) {
			EXPECT_NEAR(inActual.m[column][row], inExpected.m[column][row], tolerance)
				<< "matrix element [" << column << "][" << row << "]";
		}
	}
}

} // namespace

TEST(Camera, PerspectiveStateAndMatricesUseRenderContract) {
	oa::Camera camera(
		{0.0F, 0.0F, 5.0F},
		{0.0F, 0.0F, 0.0F},
		60.0F,
		2.0F,
		0.25F,
		200.0F);

	EXPECT_EQ(camera.getProjectionType(), oa::CameraProjection::Perspective);
	EXPECT_FLOAT_EQ(camera.getAspectRatio(), 2.0F);
	EXPECT_FLOAT_EQ(camera.getNear(), 0.25F);
	EXPECT_FLOAT_EQ(camera.getFar(), 200.0F);

	const oa::vlm::Vec3 forward = camera.getTransform().getForward();
	EXPECT_NEAR(forward.x, 0.0F, tolerance);
	EXPECT_NEAR(forward.y, 0.0F, tolerance);
	EXPECT_NEAR(forward.z, -1.0F, tolerance);

	expectMatrixNear(
		camera.getViewProjectionMatrix(),
		oa::vlm::matrixMul(camera.getViewMatrix(), camera.getProjectionMatrix()));
}

TEST(Camera, LookAtReconstructsCanonicalOffAxisViewMatrix) {
	const oa::vlm::Vec3 position{16.0F, 12.0F, 16.0F};
	const oa::vlm::Vec3 target{0.0F, 2.0F, 0.0F};
	const oa::vlm::Vec3 up{0.0F, 1.0F, 0.0F};
	oa::Camera camera;
	oa::FnCamera::initPerspective(
		camera, position, target, 48.0F, 4.0F / 3.0F, 0.1F, 100.0F);

	expectMatrixNear(
		oa::FnCamera::getViewMatrix(camera),
		oa::vlm::lookAt(position, target, up));
}

TEST(Camera, FunctionalAndMemberSurfacesOperateOnOneValue) {
	oa::Camera camera;
	oa::FnTransform::setPosition(
		camera.getTransform(), {2.0F, 3.0F, 4.0F});
	EXPECT_EQ(
		camera.getTransform().getPosition(),
		(oa::vlm::Vec3{2.0F, 3.0F, 4.0F}));

	camera.setAspectRatio(2.5F);
	EXPECT_FLOAT_EQ(oa::FnCamera::getAspectRatio(camera), 2.5F);

	oa::FnCamera::setOrthographic(camera, 640.0F, 360.0F, 0.0F, 10.0F);
	EXPECT_EQ(camera.getProjectionType(), oa::CameraProjection::Orthographic);
	EXPECT_EQ(camera.getOrthographicSize(), (oa::vlm::Vec2{640.0F, 360.0F}));
}

TEST(Camera, OrthographicFitPreservesContentAspect) {
	oa::Camera camera(640.0F, 480.0F, 0.0F, 10.0F);
	camera.fitToWindow(1920.0F, 1080.0F);
	camera.setZoom(2.0F);
	camera.setOffset(0.25F, -0.5F);

	EXPECT_EQ(camera.getProjectionType(), oa::CameraProjection::Orthographic);
	EXPECT_NEAR(
		camera.getOrthographicSize().x,
		480.0F * (1920.0F / 1080.0F), tolerance);
	EXPECT_FLOAT_EQ(camera.getOrthographicSize().y, 480.0F);
	EXPECT_FLOAT_EQ(camera.getZoom(), 2.0F);
	EXPECT_FLOAT_EQ(camera.getOffset().x, 0.25F);
	EXPECT_FLOAT_EQ(camera.getOffset().y, -0.5F);
	expectMatrixNear(
		camera.getProjectionMatrix(),
		oa::vlm::orthographicShifted(
			camera.getOrthographicSize().x,
			camera.getOrthographicSize().y,
			camera.getNear(), camera.getFar(), camera.getZoom(),
			camera.getOffset()));
}

TEST(Camera, LensShiftUsesVlmProjectionAuthority) {
	oa::Camera camera(
		{0.0F, 0.0F, 5.0F}, {0.0F, 0.0F, 0.0F},
		55.0F, 16.0F / 9.0F, 0.2F, 500.0F);
	camera.setOffset(0.125F, -0.25F);
	expectMatrixNear(
		camera.getProjectionMatrix(),
		oa::vlm::perspectiveShifted(
			camera.getEffectiveFovY(), camera.getAspectRatio(),
			camera.getNear(), camera.getFar(), camera.getOffset()));
}

TEST(Camera, FocalLengthOverridesFieldOfView) {
	oa::Camera camera;
	camera.setFovY(70.0F);
	camera.setSensorHeight(36.0F);
	camera.setFocalLength(50.0F);

	constexpr oa::F32 Pi = 3.14159265358979323846F;
	const oa::F32 expected = 2.0F * std::atan(36.0F / 100.0F) * 180.0F / Pi;
	EXPECT_NEAR(camera.getEffectiveFovY(), expected, tolerance);

	camera.setFocalLength(0.0F);
	EXPECT_FLOAT_EQ(camera.getEffectiveFovY(), 70.0F);
}
