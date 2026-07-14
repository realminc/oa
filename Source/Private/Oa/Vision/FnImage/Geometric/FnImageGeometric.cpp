// FnImageGeometric.cpp — Spatial transformation operations
//
// Implements:
// - OaFnImage::Resize  — Bilinear/Nearest interpolation
// - OaFnImage::Crop    — Extract region of interest
// - OaFnImage::Flip    — Horizontal/Vertical flip
// - OaFnImage::Rotate  — 90°/180°/270° rotation
//
// Future:
// - Warp (affine transform)
// - Perspective transform
// - Remap (arbitrary pixel mapping)

#include <Oa/Vision/FnImage.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Core/Log.h>
#include <Oa/Core/Matrix.h>
#include <Oa/Core/FnMatrix.h>

#include <cmath>
#include <limits>

namespace {

OaU32 DivCeil(OaU32 InA, OaU32 InB)
{
    return (InA + InB - 1U) / InB;
}

bool ValidateNchwImage(const OaMatrix& InImage, const char* InOperation)
{
    const OaMatrixShape shape = InImage.GetShape();
    if (!InImage.HasStorage() || shape.Rank != 4 || shape[0] <= 0 || shape[1] <= 0 ||
        shape[2] <= 0 || shape[3] <= 0) {
        OA_LOG_WARN(OaLogComponent::Core,
            "OaFnImage::%s expects a stored, non-empty [B,C,H,W] tensor", InOperation);
        return false;
    }
    return true;
}

bool ValidBorder(OaBorderMode InBorder) {
    return InBorder == OaBorderMode::Constant || InBorder == OaBorderMode::Replicate ||
        InBorder == OaBorderMode::Reflect || InBorder == OaBorderMode::Reflect101 ||
        InBorder == OaBorderMode::Wrap;
}

bool ValidInterpolation(OaInterpolationMode InInterpolation) {
    return InInterpolation == OaInterpolationMode::Nearest ||
        InInterpolation == OaInterpolationMode::Bilinear;
}

OaMatrix WarpPass(const OaMatrix& InImage, const OaMatrix& InMap,
    OaU32 InWidth, OaU32 InHeight, OaU32 InOperation, OaU32 InMapBatch,
    OaInterpolationMode InInterpolation, OaBorderMode InBorder,
    OaF32 InBorderValue) {
    const auto shape = InImage.GetShape();
    auto output = OaFnMatrix::Empty({shape[0], shape[1], InHeight, InWidth},
        InImage.GetDtype());
    struct Push {
        OaU32 Batch, Channels, InHeight, InWidth, OutHeight, OutWidth;
        OaU32 Operation, Interpolation, Border, MapBatch;
        OaF32 BorderValue;
    };
    Push push{static_cast<OaU32>(shape[0]), static_cast<OaU32>(shape[1]),
        static_cast<OaU32>(shape[2]), static_cast<OaU32>(shape[3]),
        InHeight, InWidth, InOperation, static_cast<OaU32>(InInterpolation),
        static_cast<OaU32>(InBorder), InMapBatch, InBorderValue};
    OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read,
        OaBufferAccess::Write};
    OaContext::GetDefault().Add("ImageWarp", {&InImage, &InMap, &output}, access,
        &push, sizeof(push), DivCeil(InWidth, 16), DivCeil(InHeight, 16),
        push.Batch * push.Channels);
    return output;
}

} // namespace

// ═══════════════════════════════════════════════════════════════════════════════
// OaFnImage:: Geometric Operations
// ═══════════════════════════════════════════════════════════════════════════════

OaMatrix OaFnImage::Resize(
    OaEngine& InRt,
    const OaMatrix& InImage,
    OaU32 InTargetWidth,
    OaU32 InTargetHeight,
    OaInterpolationMode InMode)
{
    if (!ValidateNchwImage(InImage, "Resize")) {
        return InImage;
    }
    if (InTargetWidth == 0 || InTargetHeight == 0) {
        OA_LOG_WARN(OaLogComponent::Core, "OaFnImage::Resize target dimensions must be non-zero");
        return InImage;
    }
    if (InMode != OaInterpolationMode::Nearest && InMode != OaInterpolationMode::Bilinear) {
        OA_LOG_WARN(OaLogComponent::Core, "OaFnImage::Resize interpolation mode is not implemented");
        return InImage;
    }
    auto shape = InImage.GetShape();
    OaU32 B = (OaU32)shape[0];
    OaU32 C = (OaU32)shape[1];
    OaU32 H = (OaU32)shape[2];
    OaU32 W = (OaU32)shape[3];

    auto out = OaFnMatrix::Empty(OaMatrixShape{B, C, InTargetHeight, InTargetWidth}, InImage.GetDtype());

    (void)InRt;
    struct ResizePush {
        OaU32 BatchSize, Channels, HIn, WIn, HOut, WOut;
    } push{B, C, H, W, InTargetHeight, InTargetWidth};
    OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
    auto& ctx = OaContext::GetDefault();
    ctx.Add(InMode == OaInterpolationMode::Nearest ? "ResizeNearest" : "ResizeBilinear",
        {&InImage, &out}, access, &push, sizeof(push),
        DivCeil(InTargetWidth, 16), DivCeil(InTargetHeight, 16), B * C);

    return out;
}

OaMatrix OaFnImage::Crop(
    OaEngine& InRt,
    const OaMatrix& InImage,
    OaU32 InX,
    OaU32 InY,
    OaU32 InWidth,
    OaU32 InHeight)
{
    if (!ValidateNchwImage(InImage, "Crop")) {
        return InImage;
    }
    auto shape = InImage.GetShape();
    OaU32 B = (OaU32)shape[0];
    OaU32 C = (OaU32)shape[1];
    OaU32 H = (OaU32)shape[2];
    OaU32 W = (OaU32)shape[3];

    if (InWidth == 0 || InHeight == 0 || InX >= W || InY >= H) {
        return InImage;
    }
    OaU32 outW = InWidth;
    OaU32 outH = InHeight;
    if (outW > W - InX) outW = W - InX;
    if (outH > H - InY) outH = H - InY;

    auto out = OaFnMatrix::Empty(OaMatrixShape{B, C, outH, outW}, InImage.GetDtype());
    (void)InRt;
    struct CropPush {
        OaU32 BatchSize, Channels, InHeight, InWidth, CropX, CropY, OutHeight, OutWidth;
    } push{B, C, H, W, InX, InY, outH, outW};
    OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
    auto& ctx = OaContext::GetDefault();
    ctx.Add("CropNchw", {&InImage, &out}, access, &push, sizeof(push),
        DivCeil(outW, 16), DivCeil(outH, 16), B * C);
    return out;
}

OaMatrix OaFnImage::Flip(
    OaEngine& InRt,
    const OaMatrix& InImage,
    bool InHorizontal,
    bool InVertical)
{
    if (!InHorizontal && !InVertical) {
        return InImage;
    }
    if (!ValidateNchwImage(InImage, "Flip")) {
        return InImage;
    }
    
    auto shape = InImage.GetShape();
    OaU32 B = (OaU32)shape[0];
    OaU32 C = (OaU32)shape[1];
    OaU32 H = (OaU32)shape[2];
    OaU32 W = (OaU32)shape[3];

    auto out = OaFnMatrix::Empty(shape, InImage.GetDtype());
    (void)InRt;
    struct FlipPush {
        OaU32 BatchSize, Channels, Height, Width, Horizontal, Vertical;
    } push{B, C, H, W, InHorizontal ? 1U : 0U, InVertical ? 1U : 0U};
    OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
    auto& ctx = OaContext::GetDefault();
    ctx.Add("FlipNchw", {&InImage, &out}, access, &push, sizeof(push),
        DivCeil(W, 16), DivCeil(H, 16), B * C);
    return out;
}

OaMatrix OaFnImage::Rotate(
    OaEngine& InRt,
    const OaMatrix& InImage,
    OaU32 InDegrees)
{
    if (!ValidateNchwImage(InImage, "Rotate")) {
        return InImage;
    }
    auto shape = InImage.GetShape();
    OaU32 B = (OaU32)shape[0];
    OaU32 C = (OaU32)shape[1];
    OaU32 H = (OaU32)shape[2];
    OaU32 W = (OaU32)shape[3];
    OaU32 degrees = InDegrees % 360;
    if (degrees == 0) {
        return InImage;
    }
    if (degrees != 90 && degrees != 180 && degrees != 270) {
        OA_LOG_WARN(OaLogComponent::Core, "Rotate: unsupported degrees %u", InDegrees);
        return InImage;
    }

    OaU32 outH = degrees == 180 ? H : W;
    OaU32 outW = degrees == 180 ? W : H;
    auto out = OaFnMatrix::Empty(OaMatrixShape{B, C, outH, outW}, InImage.GetDtype());
    (void)InRt;
    struct RotatePush {
        OaU32 BatchSize, Channels, InHeight, InWidth, OutHeight, OutWidth, Degrees;
    } push{B, C, H, W, outH, outW, degrees};
    OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
    auto& ctx = OaContext::GetDefault();
    ctx.Add("RotateNchw", {&InImage, &out}, access, &push, sizeof(push),
        DivCeil(outW, 16), DivCeil(outH, 16), B * C);
    return out;
}

OaMatrix OaFnImage::Pad(OaEngine& InRt, const OaMatrix& InImage,
    OaU32 InLeft, OaU32 InRight, OaU32 InTop, OaU32 InBottom,
    OaBorderMode InBorder, OaF32 InBorderValue) {
    (void)InRt;
    if (!ValidateNchwImage(InImage, "Pad") || !ValidBorder(InBorder) ||
        !std::isfinite(InBorderValue)) return InImage;
    const auto shape = InImage.GetShape();
    const OaU64 outWidth = static_cast<OaU64>(shape[3]) + InLeft + InRight;
    const OaU64 outHeight = static_cast<OaU64>(shape[2]) + InTop + InBottom;
    if (outWidth == 0 || outHeight == 0 ||
        outWidth > std::numeric_limits<OaU32>::max() ||
        outHeight > std::numeric_limits<OaU32>::max()) return InImage;
    auto output = OaFnMatrix::Empty({shape[0], shape[1],
        static_cast<OaI64>(outHeight), static_cast<OaI64>(outWidth)}, InImage.GetDtype());
    struct Push {
        OaU32 Batch, Channels, InHeight, InWidth, OutHeight, OutWidth;
        OaU32 PadTop, PadLeft, Border;
        OaF32 BorderValue;
    };
    Push push{static_cast<OaU32>(shape[0]), static_cast<OaU32>(shape[1]),
        static_cast<OaU32>(shape[2]), static_cast<OaU32>(shape[3]),
        static_cast<OaU32>(outHeight), static_cast<OaU32>(outWidth),
        InTop, InLeft, static_cast<OaU32>(InBorder), InBorderValue};
    OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
    OaContext::GetDefault().Add("ImagePad", {&InImage, &output}, access,
        &push, sizeof(push), DivCeil(push.OutWidth, 16), DivCeil(push.OutHeight, 16),
        push.Batch * push.Channels);
    return output;
}

OaMatrix OaFnImage::CenterCrop(OaEngine& InRt, const OaMatrix& InImage,
    OaU32 InWidth, OaU32 InHeight) {
    if (!ValidateNchwImage(InImage, "CenterCrop")) return InImage;
    const auto shape = InImage.GetShape();
    const OaU32 width = static_cast<OaU32>(shape[3]);
    const OaU32 height = static_cast<OaU32>(shape[2]);
    if (InWidth == 0 || InHeight == 0 || InWidth > width || InHeight > height) return InImage;
    return Crop(InRt, InImage, (width - InWidth) / 2, (height - InHeight) / 2,
        InWidth, InHeight);
}

OaMatrix OaFnImage::Remap(OaEngine& InRt, const OaMatrix& InImage,
    const OaMatrix& InMap, OaInterpolationMode InInterpolation,
    OaBorderMode InBorder, OaF32 InBorderValue) {
    (void)InRt;
    if (!ValidateNchwImage(InImage, "Remap") || !InMap.HasStorage() ||
        InMap.GetDtype() != InImage.GetDtype() || !ValidInterpolation(InInterpolation) ||
        !ValidBorder(InBorder) || !std::isfinite(InBorderValue)) return InImage;
    const auto map = InMap.GetShape();
    const auto image = InImage.GetShape();
    if (map.Rank != 4 || map[1] != 2 || map[2] <= 0 || map[3] <= 0 ||
        (map[0] != 1 && map[0] != image[0])) return InImage;
    return WarpPass(InImage, InMap, static_cast<OaU32>(map[3]),
        static_cast<OaU32>(map[2]), 0, static_cast<OaU32>(map[0]),
        InInterpolation, InBorder, InBorderValue);
}

OaMatrix OaFnImage::WarpAffine(OaEngine& InRt, const OaMatrix& InImage,
    const OaMatrix& InTransform, OaU32 InWidth, OaU32 InHeight,
    OaInterpolationMode InInterpolation, OaBorderMode InBorder,
    OaF32 InBorderValue) {
    (void)InRt;
    const auto transform = InTransform.GetShape();
    if (!ValidateNchwImage(InImage, "WarpAffine") || !InTransform.HasStorage() ||
        transform.Rank != 2 || transform[0] != 2 || transform[1] != 3 ||
        InTransform.GetDtype() != InImage.GetDtype() || InWidth == 0 || InHeight == 0 ||
        !ValidInterpolation(InInterpolation) || !ValidBorder(InBorder) ||
        !std::isfinite(InBorderValue)) return InImage;
    return WarpPass(InImage, InTransform, InWidth, InHeight, 1, 1,
        InInterpolation, InBorder, InBorderValue);
}

OaMatrix OaFnImage::WarpPerspective(OaEngine& InRt, const OaMatrix& InImage,
    const OaMatrix& InTransform, OaU32 InWidth, OaU32 InHeight,
    OaInterpolationMode InInterpolation, OaBorderMode InBorder,
    OaF32 InBorderValue) {
    (void)InRt;
    const auto transform = InTransform.GetShape();
    if (!ValidateNchwImage(InImage, "WarpPerspective") || !InTransform.HasStorage() ||
        transform.Rank != 2 || transform[0] != 3 || transform[1] != 3 ||
        InTransform.GetDtype() != InImage.GetDtype() || InWidth == 0 || InHeight == 0 ||
        !ValidInterpolation(InInterpolation) || !ValidBorder(InBorder) ||
        !std::isfinite(InBorderValue)) return InImage;
    return WarpPass(InImage, InTransform, InWidth, InHeight, 2, 1,
        InInterpolation, InBorder, InBorderValue);
}
