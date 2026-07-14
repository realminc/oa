// Test unified OaFnMatrix API with Vision operations
// Verifies VkImage overloads work correctly

#include "../../OaTest.h"
#include <Oa/Vision.h>

// ─── Resize Tests ───────────────────────────────────────────────────────────────

TEST_F(OaVkEngineTestFixture, FnImage_Resize) {
	// Create a simple tensor
	auto tensor = OaFnMatrix::Zeros({1, 3, 64, 64});
	
	// Resize using unified API
	auto resized = OaFnImage::Resize(tensor, 128, 128);
	
	auto shape = resized.GetShape();
	EXPECT_EQ(shape.Dims[0], 1);
	EXPECT_EQ(shape.Dims[1], 3);
	EXPECT_EQ(shape.Dims[2], 128);
	EXPECT_EQ(shape.Dims[3], 128);
}

// ─── Normalize Tests ────────────────────────────────────────────────────────────

TEST_F(OaVkEngineTestFixture, FnImage_Normalize) {
	auto tensor = OaFnMatrix::Ones({1, 3, 224, 224});
	
	OaNormalizationParams params;
	params.Mean[0] = 0.485F;
	params.Mean[1] = 0.456F;
	params.Mean[2] = 0.406F;
	params.Std[0] = 0.229F;
	params.Std[1] = 0.224F;
	params.Std[2] = 0.225F;
	
	auto normalized = OaFnImage::Normalize(tensor, params);
	
	EXPECT_EQ(normalized.GetShape(), tensor.GetShape());
}

// ─── GaussianBlur Tests ─────────────────────────────────────────────────────────

TEST_F(OaVkEngineTestFixture, FnImage_GaussianBlur) {
	auto tensor = OaFnMatrix::Rand({1, 3, 128, 128});
	auto blurred = OaFnImage::GaussianBlur(tensor, 1.5F);
	
	EXPECT_EQ(blurred.GetShape(), tensor.GetShape());
}

// ─── Crop Tests ─────────────────────────────────────────────────────────────────

TEST_F(OaVkEngineTestFixture, FnImage_Crop) {
	auto tensor = OaFnMatrix::Zeros({1, 3, 256, 256});
	auto cropped = OaFnImage::Crop(tensor, 64, 64, 128, 128);
	
	auto shape = cropped.GetShape();
	EXPECT_EQ(shape.Dims[0], 1);
	EXPECT_EQ(shape.Dims[1], 3);
	EXPECT_EQ(shape.Dims[2], 128);
	EXPECT_EQ(shape.Dims[3], 128);
}

// ─── Flip Tests ─────────────────────────────────────────────────────────────────

TEST_F(OaVkEngineTestFixture, FnImage_Flip) {
	auto tensor = OaFnMatrix::Rand({1, 3, 64, 64});
	auto flipped = OaFnImage::Flip(tensor, true, false);
	
	EXPECT_EQ(flipped.GetShape(), tensor.GetShape());
}

// ─── Rotate Tests ───────────────────────────────────────────────────────────────

TEST_F(OaVkEngineTestFixture, FnImage_Rotate) {
	auto tensor = OaFnMatrix::Rand({1, 3, 64, 64});
	auto rotated = OaFnImage::Rotate(tensor, 90);
	
	// After 90° rotation, width and height swap
	auto shape = rotated.GetShape();
	EXPECT_EQ(shape.Dims[0], 1);
	EXPECT_EQ(shape.Dims[1], 3);
	EXPECT_EQ(shape.Dims[2], 64);
	EXPECT_EQ(shape.Dims[3], 64);
}
