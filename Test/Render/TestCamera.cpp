#include <Oa/Render/Camera.h>
#include <Oa/Ui/Camera.h>

#include <gtest/gtest.h>

#include <cmath>

namespace {

constexpr OaF32 Tolerance = 1.0e-5F;

void ExpectMatrixNear(const VlmMat4& InActual, const VlmMat4& InExpected) {
	for (OaU32 column = 0; column < 4; ++column) {
		for (OaU32 row = 0; row < 4; ++row) {
			EXPECT_NEAR(InActual.M[column][row], InExpected.M[column][row], Tolerance)
				<< "matrix element [" << column << "][" << row << "]";
		}
	}
}

} // namespace

TEST(OaCamera, PerspectiveStateAndMatricesUseRenderContract) {
	OaCamera camera(
		{0.0F, 0.0F, 5.0F},
		{0.0F, 0.0F, 0.0F},
		60.0F,
		2.0F,
		0.25F,
		200.0F);

	EXPECT_EQ(camera.GetProjectionType(), OaCameraProjection::Perspective);
	EXPECT_FLOAT_EQ(camera.GetState().Aspect, 2.0F);
	EXPECT_FLOAT_EQ(camera.GetNear(), 0.25F);
	EXPECT_FLOAT_EQ(camera.GetFar(), 200.0F);

	const VlmVec3 forward = camera.GetForward();
	EXPECT_NEAR(forward.X, 0.0F, Tolerance);
	EXPECT_NEAR(forward.Y, 0.0F, Tolerance);
	EXPECT_NEAR(forward.Z, -1.0F, Tolerance);

	ExpectMatrixNear(
		camera.GetViewProjectionMatrix(),
		Vlm::MatrixMul(camera.GetViewMatrix(), camera.GetProjectionMatrix()));
}

TEST(OaCamera, LookAtReconstructsCanonicalOffAxisViewMatrix) {
	const VlmVec3 position{16.0F, 12.0F, 16.0F};
	const VlmVec3 target{0.0F, 2.0F, 0.0F};
	const VlmVec3 up{0.0F, 1.0F, 0.0F};
	OaCameraState state;
	OaFnCamera::InitPerspective(
		state, position, target, 48.0F, 4.0F / 3.0F, 0.1F, 100.0F);

	ExpectMatrixNear(
		OaFnCamera::GetViewMatrix(state),
		Vlm::LookAt(position, target, up));
}

TEST(OaCamera, OrthographicFitPreservesContentAspect) {
	OaCamera camera(640.0F, 480.0F, 0.0F, 10.0F);
	camera.FitToWindow(1920.0F, 1080.0F);
	camera.SetZoom(2.0F);
	camera.SetOffset(0.25F, -0.5F);

	EXPECT_EQ(camera.GetProjectionType(), OaCameraProjection::Orthographic);
	EXPECT_NEAR(camera.GetState().OrthoWidth, 480.0F * (1920.0F / 1080.0F), Tolerance);
	EXPECT_FLOAT_EQ(camera.GetState().OrthoHeight, 480.0F);
	EXPECT_FLOAT_EQ(camera.GetZoom(), 2.0F);
	EXPECT_FLOAT_EQ(camera.GetOffset().X, 0.25F);
	EXPECT_FLOAT_EQ(camera.GetOffset().Y, -0.5F);
}

TEST(OaCamera, FocalLengthOverridesFieldOfView) {
	OaCamera camera;
	camera.SetFovY(70.0F);
	camera.SetSensorHeight(36.0F);
	camera.SetFocalLength(50.0F);

	constexpr OaF32 Pi = 3.14159265358979323846F;
	const OaF32 expected = 2.0F * std::atan(36.0F / 100.0F) * 180.0F / Pi;
	EXPECT_NEAR(camera.GetEffectiveFovY(), expected, Tolerance);

	camera.SetFocalLength(0.0F);
	EXPECT_FLOAT_EQ(camera.GetEffectiveFovY(), 70.0F);
}
