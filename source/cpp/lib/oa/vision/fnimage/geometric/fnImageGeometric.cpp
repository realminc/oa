// FnImageGeometric.cpp — Spatial transformation operations
//
// Implements:
// - oa::FnImage::resize  — Bilinear/Nearest interpolation
// - oa::FnImage::Crop    — Extract region of interest
// - oa::FnImage::Flip    — horizontal/vertical flip
// - oa::FnImage::rotate  — 90°/180°/270° rotation
//
// Future:
// - warp (affine transform)
// - Perspective transform
// - remap (arbitrary pixel mapping)

#include <oa/vision/fnImage.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/executionSession.h>
#include <oa/core/log.h>
#include <oa/core/matrix.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/std/limits.h>
#include <oa/core/std/scalarMath.h>

namespace {

oa::U32 divCeil(oa::U32 inA, oa::U32 inB)
{
	return (inA + inB - 1U) / inB;
}

bool validateNchwImage(const oa::Matrix& inImage, const char* inOperation)
{
	const oa::MatrixShape shape = inImage.getShape();
	if (!inImage.hasStorage() || shape.rank != 4 || shape[0] <= 0 || shape[1] <= 0 ||
		shape[2] <= 0 || shape[3] <= 0) {
		OaLogWarn(oa::LogComponent::Vision,
			"oa::FnImage::{} expects a stored, non-empty [B,C,H,W] tensor", inOperation);
		return false;
	}
	return true;
}

bool validBorder(oa::BorderMode inBorder) {
	return inBorder == oa::BorderMode::Constant || inBorder == oa::BorderMode::Replicate ||
		inBorder == oa::BorderMode::Reflect || inBorder == oa::BorderMode::Reflect101 ||
		inBorder == oa::BorderMode::Wrap;
}

bool validInterpolation(oa::InterpolationMode inInterpolation) {
	return inInterpolation == oa::InterpolationMode::Nearest ||
		inInterpolation == oa::InterpolationMode::Bilinear;
}

oa::Matrix warpPass(const oa::Matrix& inImage, const oa::Matrix& inMap,
	oa::U32 inWidth, oa::U32 inHeight, oa::U32 inOperation, oa::U32 inMapBatch,
	oa::InterpolationMode inInterpolation, oa::BorderMode inBorder,
	oa::F32 inBorderValue) {
	const auto shape = inImage.getShape();
	auto output = oa::FnMatrix::empty({shape[0], shape[1], inHeight, inWidth},
		inImage.getDtype());
	struct Push {
		oa::U32 batch, channels, inHeight, inWidth, outHeight, outWidth;
		oa::U32 operation, interpolation, border, mapBatch;
		oa::F32 borderValue;
	};
	Push push{static_cast<oa::U32>(shape[0]), static_cast<oa::U32>(shape[1]),
		static_cast<oa::U32>(shape[2]), static_cast<oa::U32>(shape[3]),
		inHeight, inWidth, inOperation, static_cast<oa::U32>(inInterpolation),
		static_cast<oa::U32>(inBorder), inMapBatch, inBorderValue};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Write};
	oa::ExecutionSession::getActive().add( "ImageWarp", {&inImage, &inMap, &output}, access,
		&push, sizeof(push), divCeil(inWidth, 16), divCeil(inHeight, 16),
		push.batch * push.channels);
	return output;
}

} // namespace

// ═══════════════════════════════════════════════════════════════════════════════
// oa::FnImage:: Geometric operations
// ═══════════════════════════════════════════════════════════════════════════════

oa::Matrix oa::FnImage::resize(
	oa::Engine& inRt,
	const oa::Matrix& inImage,
	oa::U32 inTargetWidth,
	oa::U32 inTargetHeight,
	oa::InterpolationMode inMode)
{
	if (!validateNchwImage(inImage, "resize")) {
		return inImage;
	}
	if (inTargetWidth == 0 || inTargetHeight == 0) {
		OaLogWarn(oa::LogComponent::Vision, "oa::FnImage::resize target dimensions must be non-zero");
		return inImage;
	}
	if (inMode != oa::InterpolationMode::Nearest && inMode != oa::InterpolationMode::Bilinear) {
		OaLogWarn(oa::LogComponent::Vision, "oa::FnImage::resize interpolation mode is not implemented");
		return inImage;
	}
	auto shape = inImage.getShape();
	oa::U32 B = (oa::U32)shape[0];
	oa::U32 C = (oa::U32)shape[1];
	oa::U32 H = (oa::U32)shape[2];
	oa::U32 W = (oa::U32)shape[3];

	auto out = oa::FnMatrix::empty(oa::MatrixShape{B, C, inTargetHeight, inTargetWidth}, inImage.getDtype());

	(void)inRt;
	struct ResizePush {
		oa::U32 batchSize, channels, hIn, wIn, hOut, wOut;
	} push{B, C, H, W, inTargetHeight, inTargetWidth};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
	auto& ctx = oa::ExecutionSession::getActive();
	ctx.add( inMode == oa::InterpolationMode::Nearest ? "ResizeNearest" : "ResizeBilinear",
		{&inImage, &out}, access, &push, sizeof(push),
		divCeil(inTargetWidth, 16), divCeil(inTargetHeight, 16), B * C);

	return out;
}

oa::Matrix oa::FnImage::crop(
	oa::Engine& inRt,
	const oa::Matrix& inImage,
	oa::U32 inX,
	oa::U32 inY,
	oa::U32 inWidth,
	oa::U32 inHeight)
{
	if (!validateNchwImage(inImage, "Crop")) {
		return inImage;
	}
	auto shape = inImage.getShape();
	oa::U32 B = (oa::U32)shape[0];
	oa::U32 C = (oa::U32)shape[1];
	oa::U32 H = (oa::U32)shape[2];
	oa::U32 W = (oa::U32)shape[3];

	if (inWidth == 0 || inHeight == 0 || inX >= W || inY >= H) {
		return inImage;
	}
	oa::U32 outW = inWidth;
	oa::U32 outH = inHeight;
	if (outW > W - inX) outW = W - inX;
	if (outH > H - inY) outH = H - inY;

	auto out = oa::FnMatrix::empty(oa::MatrixShape{B, C, outH, outW}, inImage.getDtype());
	(void)inRt;
	struct CropPush {
		oa::U32 batchSize, channels, inHeight, inWidth, CropX, CropY, outHeight, outWidth;
	} push{B, C, H, W, inX, inY, outH, outW};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
	auto& ctx = oa::ExecutionSession::getActive();
	ctx.add( "CropNchw", {&inImage, &out}, access, &push, sizeof(push),
		divCeil(outW, 16), divCeil(outH, 16), B * C);
	return out;
}

oa::Matrix oa::FnImage::flip(
	oa::Engine& inRt,
	const oa::Matrix& inImage,
	bool inHorizontal,
	bool inVertical)
{
	if (!inHorizontal && !inVertical) {
		return inImage;
	}
	if (!validateNchwImage(inImage, "Flip")) {
		return inImage;
	}
	
	auto shape = inImage.getShape();
	oa::U32 B = (oa::U32)shape[0];
	oa::U32 C = (oa::U32)shape[1];
	oa::U32 H = (oa::U32)shape[2];
	oa::U32 W = (oa::U32)shape[3];

	auto out = oa::FnMatrix::empty(shape, inImage.getDtype());
	(void)inRt;
	struct FlipPush {
		oa::U32 batchSize, channels, height, width, horizontal, vertical;
	} push{B, C, H, W, inHorizontal ? 1U : 0U, inVertical ? 1U : 0U};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
	auto& ctx = oa::ExecutionSession::getActive();
	ctx.add( "FlipNchw", {&inImage, &out}, access, &push, sizeof(push),
		divCeil(W, 16), divCeil(H, 16), B * C);
	return out;
}

oa::Matrix oa::FnImage::rotate(
	oa::Engine& inRt,
	const oa::Matrix& inImage,
	oa::U32 inDegrees)
{
	if (!validateNchwImage(inImage, "rotate")) {
		return inImage;
	}
	auto shape = inImage.getShape();
	oa::U32 B = (oa::U32)shape[0];
	oa::U32 C = (oa::U32)shape[1];
	oa::U32 H = (oa::U32)shape[2];
	oa::U32 W = (oa::U32)shape[3];
	oa::U32 degrees = inDegrees % 360;
	if (degrees == 0) {
		return inImage;
	}
	if (degrees != 90 && degrees != 180 && degrees != 270) {
		OaLogWarn(oa::LogComponent::Vision, "rotate: unsupported degrees {}", inDegrees);
		return inImage;
	}

	oa::U32 outH = degrees == 180 ? H : W;
	oa::U32 outW = degrees == 180 ? W : H;
	auto out = oa::FnMatrix::empty(oa::MatrixShape{B, C, outH, outW}, inImage.getDtype());
	(void)inRt;
	struct RotatePush {
		oa::U32 batchSize, channels, inHeight, inWidth, outHeight, outWidth, Degrees;
	} push{B, C, H, W, outH, outW, degrees};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
	auto& ctx = oa::ExecutionSession::getActive();
	ctx.add( "RotateNchw", {&inImage, &out}, access, &push, sizeof(push),
		divCeil(outW, 16), divCeil(outH, 16), B * C);
	return out;
}

oa::Matrix oa::FnImage::pad(oa::Engine& inRt, const oa::Matrix& inImage,
	oa::U32 inLeft, oa::U32 inRight, oa::U32 inTop, oa::U32 inBottom,
	oa::BorderMode inBorder, oa::F32 inBorderValue) {
	(void)inRt;
	if (!validateNchwImage(inImage, "pad") || !validBorder(inBorder) ||
		!oa::isFinite(inBorderValue)) return inImage;
	const auto shape = inImage.getShape();
	const oa::U64 outWidth = static_cast<oa::U64>(shape[3]) + inLeft + inRight;
	const oa::U64 outHeight = static_cast<oa::U64>(shape[2]) + inTop + inBottom;
	if (outWidth == 0 || outHeight == 0 ||
		outWidth > oa::Limits<oa::U32>::max() ||
		outHeight > oa::Limits<oa::U32>::max()) return inImage;
	auto output = oa::FnMatrix::empty({shape[0], shape[1],
		static_cast<oa::I64>(outHeight), static_cast<oa::I64>(outWidth)}, inImage.getDtype());
	struct Push {
		oa::U32 batch, channels, inHeight, inWidth, outHeight, outWidth;
		oa::U32 padTop, padLeft, border;
		oa::F32 borderValue;
	};
	Push push{static_cast<oa::U32>(shape[0]), static_cast<oa::U32>(shape[1]),
		static_cast<oa::U32>(shape[2]), static_cast<oa::U32>(shape[3]),
		static_cast<oa::U32>(outHeight), static_cast<oa::U32>(outWidth),
		inTop, inLeft, static_cast<oa::U32>(inBorder), inBorderValue};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
	oa::ExecutionSession::getActive().add( "ImagePad", {&inImage, &output}, access,
		&push, sizeof(push), divCeil(push.outWidth, 16), divCeil(push.outHeight, 16),
		push.batch * push.channels);
	return output;
}

oa::Matrix oa::FnImage::centerCrop(oa::Engine& inRt, const oa::Matrix& inImage,
	oa::U32 inWidth, oa::U32 inHeight) {
	if (!validateNchwImage(inImage, "CenterCrop")) return inImage;
	const auto shape = inImage.getShape();
	const oa::U32 width = static_cast<oa::U32>(shape[3]);
	const oa::U32 height = static_cast<oa::U32>(shape[2]);
	if (inWidth == 0 || inHeight == 0 || inWidth > width || inHeight > height) return inImage;
	return crop(inRt, inImage, (width - inWidth) / 2, (height - inHeight) / 2,
		inWidth, inHeight);
}

oa::Matrix oa::FnImage::remap(oa::Engine& inRt, const oa::Matrix& inImage,
	const oa::Matrix& inMap, oa::InterpolationMode inInterpolation,
	oa::BorderMode inBorder, oa::F32 inBorderValue) {
	(void)inRt;
	if (!validateNchwImage(inImage, "Remap") || !inMap.hasStorage() ||
		inMap.getDtype() != inImage.getDtype() || !validInterpolation(inInterpolation) ||
		!validBorder(inBorder) || !oa::isFinite(inBorderValue)) return inImage;
	const auto map = inMap.getShape();
	const auto image = inImage.getShape();
	if (map.rank != 4 || map[1] != 2 || map[2] <= 0 || map[3] <= 0 ||
		(map[0] != 1 && map[0] != image[0])) return inImage;
	return warpPass(inImage, inMap, static_cast<oa::U32>(map[3]),
		static_cast<oa::U32>(map[2]), 0, static_cast<oa::U32>(map[0]),
		inInterpolation, inBorder, inBorderValue);
}

oa::Matrix oa::FnImage::warpAffine(oa::Engine& inRt, const oa::Matrix& inImage,
	const oa::Matrix& inTransform, oa::U32 inWidth, oa::U32 inHeight,
	oa::InterpolationMode inInterpolation, oa::BorderMode inBorder,
	oa::F32 inBorderValue) {
	(void)inRt;
	const auto transform = inTransform.getShape();
	if (!validateNchwImage(inImage, "WarpAffine") || !inTransform.hasStorage() ||
		transform.rank != 2 || transform[0] != 2 || transform[1] != 3 ||
		inTransform.getDtype() != inImage.getDtype() || inWidth == 0 || inHeight == 0 ||
		!validInterpolation(inInterpolation) || !validBorder(inBorder) ||
		!oa::isFinite(inBorderValue)) return inImage;
	return warpPass(inImage, inTransform, inWidth, inHeight, 1, 1,
		inInterpolation, inBorder, inBorderValue);
}

oa::Matrix oa::FnImage::warpPerspective(oa::Engine& inRt, const oa::Matrix& inImage,
	const oa::Matrix& inTransform, oa::U32 inWidth, oa::U32 inHeight,
	oa::InterpolationMode inInterpolation, oa::BorderMode inBorder,
	oa::F32 inBorderValue) {
	(void)inRt;
	const auto transform = inTransform.getShape();
	if (!validateNchwImage(inImage, "WarpPerspective") || !inTransform.hasStorage() ||
		transform.rank != 2 || transform[0] != 3 || transform[1] != 3 ||
		inTransform.getDtype() != inImage.getDtype() || inWidth == 0 || inHeight == 0 ||
		!validInterpolation(inInterpolation) || !validBorder(inBorder) ||
		!oa::isFinite(inBorderValue)) return inImage;
	return warpPass(inImage, inTransform, inWidth, inHeight, 2, 1,
		inInterpolation, inBorder, inBorderValue);
}
