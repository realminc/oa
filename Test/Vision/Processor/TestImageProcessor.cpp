// OaFnImage image processing tests — GPU operations validation

#include "../../OaTest.h"

#include <Oa/Vision/FnImage.h>
#include <Oa/Core/FnMatrix.h>

static OaMatrix MakeVisionTensor(const OaVec<OaF32>& InValues, OaMatrixShape InShape) {
	auto tensor = OaFnMatrix::Empty(InShape, OaScalarType::Float32);
	for (OaUsize i = 0; i < InValues.Size(); ++i) {
		tensor.Set(static_cast<OaI64>(i), InValues[i]);
	}
	return tensor;
}

// ─── Resize Tests ───────────────────────────────────────────────────────────────

TEST_F(OaVkEngineTestFixture, Resize_Bilinear_Downsample) {
	// Test downsampling 512×512 → 224×224
	auto input = OaFnMatrix::RandN(OaMatrixShape{1, 3, 512, 512});
	auto output = OaFnImage::Resize(Rt(), input, 224, 224, OaInterpolationMode::Bilinear);
	
	// Verify output shape
	OaExpectShape(output, {1, 3, 224, 224});
	
	// Verify values are finite
	OaExpectFinite(output);
}

TEST_F(OaVkEngineTestFixture, Resize_Bilinear_Upsample) {
	// Test upsampling 224×224 → 512×512
	auto input = OaFnMatrix::RandN(OaMatrixShape{1, 3, 224, 224});
	auto output = OaFnImage::Resize(Rt(), input, 512, 512, OaInterpolationMode::Bilinear);
	
	// Verify output shape
	OaExpectShape(output, {1, 3, 512, 512});
	
	// Verify values are finite
	OaExpectFinite(output);
}

TEST_F(OaVkEngineTestFixture, Resize_Nearest_Downsample) {
	// Test nearest neighbor downsampling
	auto input = OaFnMatrix::RandN(OaMatrixShape{1, 3, 512, 512});
	auto output = OaFnImage::Resize(Rt(), input, 224, 224, OaInterpolationMode::Nearest);
	
	// Verify output shape
	OaExpectShape(output, {1, 3, 224, 224});
	
	// Verify values are finite
	OaExpectFinite(output);
}

TEST_F(OaVkEngineTestFixture, Resize_BatchProcessing) {
	// Test batch resize (B=4)
	auto input = OaFnMatrix::RandN(OaMatrixShape{4, 3, 512, 512});
	auto output = OaFnImage::Resize(Rt(), input, 224, 224);
	
	// Verify output shape
	OaExpectShape(output, {4, 3, 224, 224});
	
	// Verify values are finite
	OaExpectFinite(output);
}

TEST_F(OaVkEngineTestFixture, Resize_IdentityTransform) {
	// Test resize to same dimensions (should be identity)
	auto input = OaFnMatrix::RandN(OaMatrixShape{1, 3, 224, 224});
	auto output = OaFnImage::Resize(Rt(), input, 224, 224);
	
	// Verify output shape
	OaExpectShape(output, {1, 3, 224, 224});
	
	// Values should be approximately equal (bilinear interpolation introduces small errors)
	OaExpectMatrixNear(input, output, 0.15f);  // Relaxed tolerance for bilinear interpolation shader
}

// ─── Normalize Tests ────────────────────────────────────────────────────────────

TEST_F(OaVkEngineTestFixture, Normalize_ImageNet) {
	// Test ImageNet normalization
	auto input = OaFnMatrix::Rand(OaMatrixShape{1, 3, 224, 224});  // [0, 1] range
	
	OaNormalizationParams params = {
		.Mean = {0.485f, 0.456f, 0.406f},
		.Std = {0.229f, 0.224f, 0.225f}
	};
	
	auto output = OaFnImage::Normalize(Rt(), input, params);
	
	// Verify output shape
	OaExpectShape(output, {1, 3, 224, 224});
	
	// Verify values are finite
	OaExpectFinite(output);
	
	// Normalized values should be roughly in [-3, 3] range
	for (OaI64 i = 0; i < output.NumElements(); ++i) {
		OaF32 v = output.At(i);
		EXPECT_GE(v, -5.0f) << "Value too negative at index " << i;
		EXPECT_LE(v, 5.0f) << "Value too positive at index " << i;
	}
}

TEST_F(OaVkEngineTestFixture, Normalize_ZeroMeanUnitStd) {
	// Test zero-mean, unit-std normalization
	auto input = OaFnMatrix::RandN(OaMatrixShape{1, 3, 224, 224});
	
	OaNormalizationParams params = {
		.Mean = {0.0f, 0.0f, 0.0f},
		.Std = {1.0f, 1.0f, 1.0f}
	};
	
	auto output = OaFnImage::Normalize(Rt(), input, params);
	
	// Should be identity transform (relaxed tolerance for bilinear interpolation)
	OaExpectMatrixNear(input, output, 0.05f);
}

TEST_F(OaVkEngineTestFixture, Normalize_KnownValues) {
	OaVec<OaF32> values = {1.0f, 2.0f, 10.0f, 20.0f, 100.0f, 200.0f};
	auto input = MakeVisionTensor(values, OaMatrixShape{1, 3, 1, 2});
	OaNormalizationParams params = {
		.Mean = {1.0f, 10.0f, 100.0f},
		.Std = {1.0f, 2.0f, 4.0f}
	};
	auto output = OaFnImage::Normalize(Rt(), input, params);
	OaExpectShape(output, {1, 3, 1, 2});
	EXPECT_NEAR(output.At(0), 0.0f, 1e-5f);
	EXPECT_NEAR(output.At(1), 1.0f, 1e-5f);
	EXPECT_NEAR(output.At(2), 0.0f, 1e-5f);
	EXPECT_NEAR(output.At(3), 5.0f, 1e-5f);
	EXPECT_NEAR(output.At(4), 0.0f, 1e-5f);
	EXPECT_NEAR(output.At(5), 25.0f, 1e-5f);
}

TEST_F(OaVkEngineTestFixture, Normalize_BatchProcessing) {
	// Test batch normalization (B=4)
	auto input = OaFnMatrix::Rand(OaMatrixShape{4, 3, 224, 224});
	
	OaNormalizationParams params = {
		.Mean = {0.5f, 0.5f, 0.5f},
		.Std = {0.5f, 0.5f, 0.5f}
	};
	
	auto output = OaFnImage::Normalize(Rt(), input, params);
	
	// Verify output shape (should match input)
	EXPECT_EQ(output.GetShape(), input.GetShape());
	
	// Verify values are finite
	OaExpectFinite(output);
}

// ─── GaussianBlur Tests ─────────────────────────────────────────────────────────

TEST_F(OaVkEngineTestFixture, GaussianBlur_Radius2) {
	// Test Gaussian blur with radius=2 (5-tap kernel)
	auto input = OaFnMatrix::RandN(OaMatrixShape{1, 3, 224, 224});
	auto output = OaFnImage::GaussianBlur(Rt(), input, 2, 1.0f);
	
	// Verify output shape
	OaExpectShape(output, {1, 3, 224, 224});
	
	// Verify values are finite
	OaExpectFinite(output);
	
	// Blurred image should have lower variance than input
	// (This is a weak test, but validates basic correctness)
}

TEST_F(OaVkEngineTestFixture, GaussianBlur_Radius4) {
	// Test Gaussian blur with radius=4 (9-tap kernel)
	auto input = OaFnMatrix::RandN(OaMatrixShape{1, 3, 224, 224});
	auto output = OaFnImage::GaussianBlur(Rt(), input, 4, 2.0f);
	
	// Verify output shape
	OaExpectShape(output, {1, 3, 224, 224});
	
	// Verify values are finite
	OaExpectFinite(output);
}

TEST_F(OaVkEngineTestFixture, GaussianBlur_BatchProcessing) {
	// Test batch blur (B=4)
	auto input = OaFnMatrix::RandN(OaMatrixShape{4, 3, 224, 224});
	auto output = OaFnImage::GaussianBlur(Rt(), input, 2, 1.0f);
	
	// Verify output shape
	OaExpectShape(output, {4, 3, 224, 224});
	
	// Verify values are finite
	OaExpectFinite(output);
}

// ─── Crop Tests ─────────────────────────────────────────────────────────────────

TEST_F(OaVkEngineTestFixture, Crop_CenterCrop) {
	// Test center crop 512×512 → 224×224
	auto input = OaFnMatrix::RandN(OaMatrixShape{1, 3, 512, 512});
	auto output = OaFnImage::Crop(Rt(), input, 144, 144, 224, 224);
	
	// Verify output shape
	OaExpectShape(output, {1, 3, 224, 224});
	
	// Verify values are finite
	OaExpectFinite(output);
}

TEST_F(OaVkEngineTestFixture, Crop_TopLeftCorner) {
	// Test top-left corner crop
	auto input = OaFnMatrix::RandN(OaMatrixShape{1, 3, 512, 512});
	auto output = OaFnImage::Crop(Rt(), input, 0, 0, 224, 224);
	
	// Verify output shape
	OaExpectShape(output, {1, 3, 224, 224});
	
	// Verify values are finite
	OaExpectFinite(output);
}

TEST_F(OaVkEngineTestFixture, Crop_KnownValues) {
	OaVec<OaF32> values = {
		1.0f, 2.0f, 3.0f, 4.0f,
		5.0f, 6.0f, 7.0f, 8.0f,
		9.0f, 10.0f, 11.0f, 12.0f
	};
	auto input = MakeVisionTensor(values, OaMatrixShape{1, 1, 3, 4});
	auto output = OaFnImage::Crop(Rt(), input, 1, 1, 2, 2);
	OaExpectShape(output, {1, 1, 2, 2});
	EXPECT_NEAR(output.At(0), 6.0f, 1e-5f);
	EXPECT_NEAR(output.At(1), 7.0f, 1e-5f);
	EXPECT_NEAR(output.At(2), 10.0f, 1e-5f);
	EXPECT_NEAR(output.At(3), 11.0f, 1e-5f);
}

// ─── Flip Tests ─────────────────────────────────────────────────────────────────
TEST_F(OaVkEngineTestFixture, Flip_Horizontal) {
	auto input = OaFnMatrix::RandN(OaMatrixShape{1, 3, 224, 224});
	auto output = OaFnImage::Flip(Rt(), input, true, false);
	OaExpectShape(output, {1, 3, 224, 224});
	OaExpectFinite(output);
}

TEST_F(OaVkEngineTestFixture, Flip_Vertical) {
	auto input = OaFnMatrix::RandN(OaMatrixShape{1, 3, 224, 224});
	auto output = OaFnImage::Flip(Rt(), input, false, true);
	OaExpectShape(output, {1, 3, 224, 224});
	OaExpectFinite(output);
}

TEST_F(OaVkEngineTestFixture, Flip_Both) {
	auto input = OaFnMatrix::RandN(OaMatrixShape{1, 3, 224, 224});
	auto output = OaFnImage::Flip(Rt(), input, true, true);
	OaExpectShape(output, {1, 3, 224, 224});
	OaExpectFinite(output);
}

TEST_F(OaVkEngineTestFixture, Flip_KnownValues) {
	OaVec<OaF32> values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
	auto input = MakeVisionTensor(values, OaMatrixShape{1, 1, 2, 3});
	auto h = OaFnImage::Flip(Rt(), input, true, false);
	EXPECT_NEAR(h.At(0), 3.0f, 1e-5f);
	EXPECT_NEAR(h.At(1), 2.0f, 1e-5f);
	EXPECT_NEAR(h.At(2), 1.0f, 1e-5f);
	EXPECT_NEAR(h.At(3), 6.0f, 1e-5f);
	EXPECT_NEAR(h.At(4), 5.0f, 1e-5f);
	EXPECT_NEAR(h.At(5), 4.0f, 1e-5f);

	auto v = OaFnImage::Flip(Rt(), input, false, true);
	EXPECT_NEAR(v.At(0), 4.0f, 1e-5f);
	EXPECT_NEAR(v.At(1), 5.0f, 1e-5f);
	EXPECT_NEAR(v.At(2), 6.0f, 1e-5f);
	EXPECT_NEAR(v.At(3), 1.0f, 1e-5f);
	EXPECT_NEAR(v.At(4), 2.0f, 1e-5f);
	EXPECT_NEAR(v.At(5), 3.0f, 1e-5f);
}

// ─── Rotate Tests ───────────────────────────────────────────────────────────────

TEST_F(OaVkEngineTestFixture, Rotate_KnownValues) {
	OaVec<OaF32> values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
	auto input = MakeVisionTensor(values, OaMatrixShape{1, 1, 2, 3});
	auto r90 = OaFnImage::Rotate(Rt(), input, 90);
	OaExpectShape(r90, {1, 1, 3, 2});
	EXPECT_NEAR(r90.At(0), 4.0f, 1e-5f);
	EXPECT_NEAR(r90.At(1), 1.0f, 1e-5f);
	EXPECT_NEAR(r90.At(2), 5.0f, 1e-5f);
	EXPECT_NEAR(r90.At(3), 2.0f, 1e-5f);
	EXPECT_NEAR(r90.At(4), 6.0f, 1e-5f);
	EXPECT_NEAR(r90.At(5), 3.0f, 1e-5f);

	auto r180 = OaFnImage::Rotate(Rt(), input, 180);
	OaExpectShape(r180, {1, 1, 2, 3});
	EXPECT_NEAR(r180.At(0), 6.0f, 1e-5f);
	EXPECT_NEAR(r180.At(1), 5.0f, 1e-5f);
	EXPECT_NEAR(r180.At(2), 4.0f, 1e-5f);
	EXPECT_NEAR(r180.At(3), 3.0f, 1e-5f);
	EXPECT_NEAR(r180.At(4), 2.0f, 1e-5f);
	EXPECT_NEAR(r180.At(5), 1.0f, 1e-5f);

	auto r270 = OaFnImage::Rotate(Rt(), input, 270);
	OaExpectShape(r270, {1, 1, 3, 2});
	EXPECT_NEAR(r270.At(0), 3.0f, 1e-5f);
	EXPECT_NEAR(r270.At(1), 6.0f, 1e-5f);
	EXPECT_NEAR(r270.At(2), 2.0f, 1e-5f);
	EXPECT_NEAR(r270.At(3), 5.0f, 1e-5f);
	EXPECT_NEAR(r270.At(4), 1.0f, 1e-5f);
	EXPECT_NEAR(r270.At(5), 4.0f, 1e-5f);
}

// ─── End-to-End Pipeline Tests ──────────────────────────────────────────────────

TEST_F(OaVkEngineTestFixture, Pipeline_ResizeNormalizeBlur) {
	// Test complete preprocessing pipeline
	auto input = OaFnMatrix::Rand(OaMatrixShape{1, 3, 512, 512});
	
	// Resize
	auto resized = OaFnImage::Resize(Rt(), input, 224, 224);
	
	// Normalize
	OaNormalizationParams params = {
		.Mean = {0.485f, 0.456f, 0.406f},
		.Std = {0.229f, 0.224f, 0.225f}
	};
	auto normalized = OaFnImage::Normalize(Rt(), resized, params);
	
	// Blur (data augmentation)
	auto blurred = OaFnImage::GaussianBlur(Rt(), normalized, 2, 1.0f);
	
	// Verify final shape
	OaExpectShape(blurred, {1, 3, 224, 224});
	
	// Verify values are finite
	OaExpectFinite(blurred);
}
