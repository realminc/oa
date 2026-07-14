// FnImageConversion.cpp — Image normalization operations
//
// Implements:
// - OaFnImage::Normalize — ImageNet-style normalization

#include <Oa/Vision/FnImage.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Core/Log.h>
#include <Oa/Core/Matrix.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/Validation.h>

#include <cmath>

namespace {

OaU32 DivCeil(OaU32 InA, OaU32 InB)
{
    return (InA + InB - 1U) / InB;
}

void ReportVisionValidation(OaValidationSeverity InSeverity, const char* InMessage)
{
    if (OaValidation::IsEnabled()) {
        (void)OaValidation::Report(InSeverity, OaLogComponent::Core, "%s", InMessage);
    }
}

void ReportVisionValidation2U(
    OaValidationSeverity InSeverity,
    const char* InFormat,
    OaU32 InA,
    OaU32 InB)
{
    if (OaValidation::IsEnabled()) {
        (void)OaValidation::Report(InSeverity, OaLogComponent::Core, InFormat, InA, InB);
    }
}

} // namespace

OaMatrix OaFnImage::Normalize(
    OaEngine& InRt,
    const OaMatrix& InImage,
    const OaNormalizationParams& InParams)
{
    auto shape = InImage.GetShape();
    if (shape.Rank != 4) {
        if (OaValidation::IsEnabled()) {
            (void)OaValidation::Report(
                OaValidationSeverity::Error,
                OaLogComponent::Core,
                "OaFnImage::Normalize: expected 4D [B,C,H,W] tensor, got rank %d",
                static_cast<int>(shape.Rank));
        }
        return InImage;
    }

    OaU32 B = (OaU32)shape[0];
    OaU32 C = (OaU32)shape[1];
    OaU32 H = (OaU32)shape[2];
    OaU32 W = (OaU32)shape[3];
    if (B == 0 || C == 0 || H == 0 || W == 0) {
        ReportVisionValidation(OaValidationSeverity::Error, "OaFnImage::Normalize: tensor dimensions must be non-zero");
        return InImage;
    }
    if (C > 3) {
        ReportVisionValidation2U(
            OaValidationSeverity::Error,
            "OaFnImage::Normalize: at most 3 channels are supported, got %u (B=%u)",
            C,
            B);
        return InImage;
    }
    for (OaU32 c = 0; c < C; ++c) {
        if (!std::isfinite(InParams.Mean[c]) ||
            !std::isfinite(InParams.Std[c]) ||
            InParams.Std[c] <= 0.0F) {
            ReportVisionValidation(
                OaValidationSeverity::Error,
                "OaFnImage::Normalize: means must be finite and standard deviations finite and positive");
            return InImage;
        }
    }

    auto result = OaFnMatrix::Empty(shape, InImage.GetDtype());
    (void)InRt;
    struct NormalizePush {
        OaU32 BatchSize, Channels, Height, Width;
        OaF32 Mean0, Mean1, Mean2, Std0, Std1, Std2;
    } push{B, C, H, W,
        InParams.Mean[0], InParams.Mean[1], InParams.Mean[2],
        InParams.Std[0], InParams.Std[1], InParams.Std[2]};
    OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
    auto& ctx = OaContext::GetDefault();
    ctx.Add("NormalizeImage", {&InImage, &result}, access, &push, sizeof(push),
        DivCeil(W, 16), DivCeil(H, 16), B * C);

    return result;
}
