// FnImageConversion.cpp — Image normalization operations
//
// Implements:
// - oa::FnImage::normalize — ImageNet-style normalization

#include <oa/vision/fnImage.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/executionSession.h>
#include <oa/core/log.h>
#include <oa/core/matrix.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/std/scalarMath.h>
#include <oa/core/validation.h>

namespace {

oa::U32 divCeil(oa::U32 inA, oa::U32 inB)
{
    return (inA + inB - 1U) / inB;
}

void reportVisionValidation(oa::ValidationSeverity inSeverity, const char* inMessage)
{
    if (oa::Validation::isEnabled()) {
        (void)oa::Validation::report(inSeverity, oa::LogComponent::Vision, "%s", inMessage);
    }
}

void reportVisionValidation2U(
    oa::ValidationSeverity inSeverity,
    const char* inFormat,
    oa::U32 inA,
    oa::U32 inB)
{
    if (oa::Validation::isEnabled()) {
        (void)oa::Validation::report(inSeverity, oa::LogComponent::Vision, inFormat, inA, inB);
    }
}

} // namespace

oa::Matrix oa::FnImage::normalize(
    oa::Engine& inRt,
    const oa::Matrix& inImage,
    const oa::NormalizationParams& inParams)
{
    auto shape = inImage.getShape();
    if (shape.rank != 4) {
        if (oa::Validation::isEnabled()) {
            (void)oa::Validation::report(
                oa::ValidationSeverity::Error,
                oa::LogComponent::Vision,
                "oa::FnImage::normalize: expected 4D [B,C,H,W] tensor, got rank %d",
                static_cast<int>(shape.rank));
        }
        return inImage;
    }

    oa::U32 B = (oa::U32)shape[0];
    oa::U32 C = (oa::U32)shape[1];
    oa::U32 H = (oa::U32)shape[2];
    oa::U32 W = (oa::U32)shape[3];
    if (B == 0 || C == 0 || H == 0 || W == 0) {
        reportVisionValidation(oa::ValidationSeverity::Error, "oa::FnImage::normalize: tensor dimensions must be non-zero");
        return inImage;
    }
    if (C > 3) {
        reportVisionValidation2U(
            oa::ValidationSeverity::Error,
            "oa::FnImage::normalize: at most 3 channels are supported, got %u (B=%u)",
            C,
            B);
        return inImage;
    }
    for (oa::U32 c = 0; c < C; ++c) {
        if (!oa::isFinite(inParams.mean[c]) ||
            !oa::isFinite(inParams.std[c]) ||
            inParams.std[c] <= 0.0F) {
            reportVisionValidation(
                oa::ValidationSeverity::Error,
                "oa::FnImage::normalize: means must be finite and standard deviations finite and positive");
            return inImage;
        }
    }

    auto result = oa::FnMatrix::empty(shape, inImage.getDtype());
    (void)inRt;
    struct NormalizePush {
        oa::U32 batchSize, channels, height, width;
        oa::F32 Mean0, Mean1, Mean2, Std0, Std1, Std2;
    } push{B, C, H, W,
        inParams.mean[0], inParams.mean[1], inParams.mean[2],
        inParams.std[0], inParams.std[1], inParams.std[2]};
    oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
    auto& ctx = oa::ExecutionSession::getActive();
    ctx.add( "NormalizeImage", {&inImage, &result}, access, &push, sizeof(push),
        divCeil(W, 16), divCeil(H, 16), B * C);

    return result;
}
