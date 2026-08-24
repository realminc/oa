// Test unified oa::FnMatrix API with Vision operations
// Verifies VkImage overloads work correctly

#include "../../oaTest.h"
#include <oa/vision.h>

// ─── resize Tests ───────────────────────────────────────────────────────────────

TEST_F(VkEngineTestFixture, FnImage_Resize) {
	// Create a simple tensor
	auto tensor = oa::FnMatrix::zeros({1, 3, 64, 64});
	
	// resize using unified API
	auto resized = oa::FnImage::resize(tensor, 128, 128);
	
	auto shape = resized.getShape();
	EXPECT_EQ(shape.dims[0], 1);
	EXPECT_EQ(shape.dims[1], 3);
	EXPECT_EQ(shape.dims[2], 128);
	EXPECT_EQ(shape.dims[3], 128);
}

// ─── normalize Tests ────────────────────────────────────────────────────────────

TEST_F(VkEngineTestFixture, FnImage_Normalize) {
	auto tensor = oa::FnMatrix::ones({1, 3, 224, 224});
	
	oa::NormalizationParams params;
	params.mean[0] = 0.485F;
	params.mean[1] = 0.456F;
	params.mean[2] = 0.406F;
	params.std[0] = 0.229F;
	params.std[1] = 0.224F;
	params.std[2] = 0.225F;
	
	auto normalized = oa::FnImage::normalize(tensor, params);
	
	EXPECT_EQ(normalized.getShape(), tensor.getShape());
}

// ─── GaussianBlur Tests ─────────────────────────────────────────────────────────

TEST_F(VkEngineTestFixture, FnImage_GaussianBlur) {
	auto tensor = oa::FnMatrix::rand({1, 3, 128, 128});
	auto blurred = oa::FnImage::gaussianBlur(tensor, 1.5F);
	
	EXPECT_EQ(blurred.getShape(), tensor.getShape());
}

// ─── Crop Tests ─────────────────────────────────────────────────────────────────

TEST_F(VkEngineTestFixture, FnImage_Crop) {
	auto tensor = oa::FnMatrix::zeros({1, 3, 256, 256});
	auto cropped = oa::FnImage::crop(tensor, 64, 64, 128, 128);
	
	auto shape = cropped.getShape();
	EXPECT_EQ(shape.dims[0], 1);
	EXPECT_EQ(shape.dims[1], 3);
	EXPECT_EQ(shape.dims[2], 128);
	EXPECT_EQ(shape.dims[3], 128);
}

// ─── Flip Tests ─────────────────────────────────────────────────────────────────

TEST_F(VkEngineTestFixture, FnImage_Flip) {
	auto tensor = oa::FnMatrix::rand({1, 3, 64, 64});
	auto flipped = oa::FnImage::flip(tensor, true, false);
	
	EXPECT_EQ(flipped.getShape(), tensor.getShape());
}

// ─── rotate Tests ───────────────────────────────────────────────────────────────

TEST_F(VkEngineTestFixture, FnImage_Rotate) {
	auto tensor = oa::FnMatrix::rand({1, 3, 64, 64});
	auto rotated = oa::FnImage::rotate(tensor, 90);
	
	// After 90° rotation, width and height swap
	auto shape = rotated.getShape();
	EXPECT_EQ(shape.dims[0], 1);
	EXPECT_EQ(shape.dims[1], 3);
	EXPECT_EQ(shape.dims[2], 64);
	EXPECT_EQ(shape.dims[3], 64);
}
