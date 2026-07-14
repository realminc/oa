// GPU-native NCHW pointwise, channel and compositing operations.

#include <Oa/Vision/FnImage.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Core/BufferAccess.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/Log.h>

#include <cmath>

namespace {

OaU32 DivCeil(OaU32 InA, OaU32 InB) {
	return (InA + InB - 1U) / InB;
}

bool ValidImage(const OaMatrix& InImage, const char* InOperation) {
	const auto shape = InImage.GetShape();
	if (shape.Rank == 4 && shape[0] > 0 && shape[1] > 0 &&
		shape[2] > 0 && shape[3] > 0 && InImage.HasStorage()) return true;
	OA_LOG_WARN(OaLogComponent::Core,
		"OaFnImage::%s expects a stored non-empty [B,C,H,W] tensor", InOperation);
	return false;
}

bool SameImage(const OaMatrix& InA, const OaMatrix& InB) {
	return InA.GetShape() == InB.GetShape() &&
		InA.GetDtype() == InB.GetDtype() && InB.HasStorage();
}

OaMatrix Pointwise(const OaMatrix& InImage, OaU32 InOperation,
	OaF32 InP0 = 0.0F, OaF32 InP1 = 0.0F,
	OaF32 InP2 = 0.0F, OaF32 InP3 = 0.0F) {
	auto output = OaFnMatrix::Empty(InImage.GetShape(), InImage.GetDtype());
	struct Push { OaU32 NumElements, Operation; OaF32 P0, P1, P2, P3; };
	Push push{static_cast<OaU32>(InImage.NumElements()), InOperation,
		InP0, InP1, InP2, InP3};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	OaContext::GetDefault().Add("ImagePointwise", {&InImage, &output}, access,
		&push, sizeof(push), DivCeil(push.NumElements, 256), 1, 1);
	return output;
}

OaMatrix CompositePass(const OaMatrix& InA, const OaMatrix& InB,
	const OaMatrix& InMask, OaU32 InOperation, OaF32 InAlpha,
	OaU32 InX = 0, OaU32 InY = 0, OaU32 InWidth = 0,
	OaU32 InHeight = 0, OaF32 InValue = 0.0F) {
	const auto shape = InA.GetShape();
	auto output = OaFnMatrix::Empty(shape, InA.GetDtype());
	struct Push {
		OaU32 Batch, Channels, Height, Width, Operation, MaskChannels;
		OaU32 RectX, RectY, RectWidth, RectHeight;
		OaF32 Alpha, Value;
	};
	Push push{static_cast<OaU32>(shape[0]), static_cast<OaU32>(shape[1]),
		static_cast<OaU32>(shape[2]), static_cast<OaU32>(shape[3]), InOperation,
		static_cast<OaU32>(InMask.GetShape()[1]), InX, InY, InWidth, InHeight,
		InAlpha, InValue};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Read, OaBufferAccess::Write};
	OaContext::GetDefault().Add("ImageComposite", {&InA, &InB, &InMask, &output},
		access, &push, sizeof(push), DivCeil(push.Width, 16),
		DivCeil(push.Height, 16), push.Batch * push.Channels);
	return output;
}

} // namespace

OaMatrix OaFnImage::ThresholdBinary(OaEngine& InRt, const OaMatrix& InImage,
	OaF32 InThreshold, OaF32 InMaxValue) {
	(void)InRt;
	if (!ValidImage(InImage, "ThresholdBinary") || !std::isfinite(InThreshold) ||
		!std::isfinite(InMaxValue)) return InImage;
	return Pointwise(InImage, 0, InThreshold, InMaxValue);
}

OaMatrix OaFnImage::ThresholdBinaryInv(OaEngine& InRt, const OaMatrix& InImage,
	OaF32 InThreshold, OaF32 InMaxValue) {
	(void)InRt;
	if (!ValidImage(InImage, "ThresholdBinaryInv") || !std::isfinite(InThreshold) ||
		!std::isfinite(InMaxValue)) return InImage;
	return Pointwise(InImage, 1, InThreshold, InMaxValue);
}

OaMatrix OaFnImage::ThresholdTruncate(OaEngine& InRt, const OaMatrix& InImage,
	OaF32 InThreshold) {
	(void)InRt;
	if (!ValidImage(InImage, "ThresholdTruncate") || !std::isfinite(InThreshold)) return InImage;
	return Pointwise(InImage, 2, InThreshold);
}

OaMatrix OaFnImage::ThresholdToZero(OaEngine& InRt, const OaMatrix& InImage,
	OaF32 InThreshold) {
	(void)InRt;
	if (!ValidImage(InImage, "ThresholdToZero") || !std::isfinite(InThreshold)) return InImage;
	return Pointwise(InImage, 3, InThreshold);
}

OaMatrix OaFnImage::ThresholdToZeroInv(OaEngine& InRt, const OaMatrix& InImage,
	OaF32 InThreshold) {
	(void)InRt;
	if (!ValidImage(InImage, "ThresholdToZeroInv") || !std::isfinite(InThreshold)) return InImage;
	return Pointwise(InImage, 4, InThreshold);
}

OaMatrix OaFnImage::InRange(OaEngine& InRt, const OaMatrix& InImage,
	OaF32 InLow, OaF32 InHigh, OaF32 InTrueValue) {
	(void)InRt;
	if (!ValidImage(InImage, "InRange") || InLow > InHigh ||
		!std::isfinite(InLow) || !std::isfinite(InHigh) || !std::isfinite(InTrueValue)) return InImage;
	return Pointwise(InImage, 5, InLow, InHigh, InTrueValue);
}

OaMatrix OaFnImage::Clamp(OaEngine& InRt, const OaMatrix& InImage,
	OaF32 InLow, OaF32 InHigh) {
	(void)InRt;
	if (!ValidImage(InImage, "Clamp") || InLow > InHigh ||
		!std::isfinite(InLow) || !std::isfinite(InHigh)) return InImage;
	return Pointwise(InImage, 6, InLow, InHigh);
}

OaMatrix OaFnImage::Invert(OaEngine& InRt, const OaMatrix& InImage, OaF32 InMaxValue) {
	(void)InRt;
	if (!ValidImage(InImage, "Invert") || !std::isfinite(InMaxValue)) return InImage;
	return Pointwise(InImage, 7, InMaxValue);
}

OaMatrix OaFnImage::BrightnessContrast(OaEngine& InRt, const OaMatrix& InImage,
	OaF32 InBrightness, OaF32 InContrast) {
	(void)InRt;
	if (!ValidImage(InImage, "BrightnessContrast") || !std::isfinite(InBrightness) ||
		!std::isfinite(InContrast)) return InImage;
	return Pointwise(InImage, 8, InContrast, InBrightness);
}

OaMatrix OaFnImage::GammaContrast(OaEngine& InRt, const OaMatrix& InImage,
	OaF32 InGamma, OaF32 InGain) {
	(void)InRt;
	if (!ValidImage(InImage, "GammaContrast") || InGamma <= 0.0F ||
		!std::isfinite(InGamma) || !std::isfinite(InGain)) return InImage;
	return Pointwise(InImage, 9, InGamma, InGain);
}

OaMatrix OaFnImage::Solarize(OaEngine& InRt, const OaMatrix& InImage,
	OaF32 InThreshold, OaF32 InMaxValue) {
	(void)InRt;
	if (!ValidImage(InImage, "Solarize") || !std::isfinite(InThreshold) ||
		!std::isfinite(InMaxValue)) return InImage;
	return Pointwise(InImage, 10, InThreshold, InMaxValue);
}

OaMatrix OaFnImage::Posterize(OaEngine& InRt, const OaMatrix& InImage,
	OaU32 InLevels, OaF32 InLow, OaF32 InHigh) {
	(void)InRt;
	if (!ValidImage(InImage, "Posterize") || InLevels < 2 || InLevels > 65536 ||
		InLow >= InHigh || !std::isfinite(InLow) || !std::isfinite(InHigh)) return InImage;
	return Pointwise(InImage, 11, static_cast<OaF32>(InLevels), InLow, InHigh);
}

OaMatrix OaFnImage::Grayscale(OaEngine& InRt, const OaMatrix& InImage) {
	(void)InRt;
	if (!ValidImage(InImage, "Grayscale") || InImage.GetShape()[1] < 3) return InImage;
	const auto shape = InImage.GetShape();
	auto output = OaFnMatrix::Empty({shape[0], 1, shape[2], shape[3]}, InImage.GetDtype());
	struct Push { OaU32 Batch, InChannels, OutChannels, Height, Width, Operation;
		OaU32 Order0, Order1, Order2, Order3; };
	Push push{static_cast<OaU32>(shape[0]), static_cast<OaU32>(shape[1]), 1,
		static_cast<OaU32>(shape[2]), static_cast<OaU32>(shape[3]), 0, 0, 1, 2, 3};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	OaContext::GetDefault().Add("ImageChannelTransform", {&InImage, &output}, access,
		&push, sizeof(push), DivCeil(push.Width, 16), DivCeil(push.Height, 16), push.Batch);
	return output;
}

OaMatrix OaFnImage::ChannelReorder(OaEngine& InRt, const OaMatrix& InImage,
	OaU32 InChannel0, OaU32 InChannel1, OaU32 InChannel2, OaU32 InChannel3) {
	(void)InRt;
	if (!ValidImage(InImage, "ChannelReorder")) return InImage;
	const auto shape = InImage.GetShape();
	const OaU32 channels = static_cast<OaU32>(shape[1]);
	const OaU32 order[] = {InChannel0, InChannel1, InChannel2, InChannel3};
	if (channels > 4) return InImage;
	for (OaU32 i = 0; i < channels; ++i) if (order[i] >= channels) return InImage;
	auto output = OaFnMatrix::Empty(shape, InImage.GetDtype());
	struct Push { OaU32 Batch, InChannels, OutChannels, Height, Width, Operation;
		OaU32 Order0, Order1, Order2, Order3; };
	Push push{static_cast<OaU32>(shape[0]), channels, channels,
		static_cast<OaU32>(shape[2]), static_cast<OaU32>(shape[3]), 1,
		InChannel0, InChannel1, InChannel2, InChannel3};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	OaContext::GetDefault().Add("ImageChannelTransform", {&InImage, &output}, access,
		&push, sizeof(push), DivCeil(push.Width, 16), DivCeil(push.Height, 16),
		push.Batch * channels);
	return output;
}

OaMatrix OaFnImage::AlphaBlend(OaEngine& InRt, const OaMatrix& InA,
	const OaMatrix& InB, OaF32 InAlpha) {
	(void)InRt;
	if (!ValidImage(InA, "AlphaBlend") || !SameImage(InA, InB) ||
		!std::isfinite(InAlpha) || InAlpha < 0.0F || InAlpha > 1.0F) return InA;
	return CompositePass(InA, InB, InA, 0, InAlpha);
}

OaMatrix OaFnImage::Composite(OaEngine& InRt, const OaMatrix& InA,
	const OaMatrix& InB, const OaMatrix& InMask) {
	(void)InRt;
	if (!ValidImage(InA, "Composite") || !SameImage(InA, InB) ||
		!ValidImage(InMask, "Composite")) return InA;
	const auto a = InA.GetShape();
	const auto m = InMask.GetShape();
	if (a[0] != m[0] || a[2] != m[2] || a[3] != m[3] ||
		(m[1] != 1 && m[1] != a[1]) || InMask.GetDtype() != InA.GetDtype()) return InA;
	return CompositePass(InA, InB, InMask, 1, 0.0F);
}

OaMatrix OaFnImage::Erase(OaEngine& InRt, const OaMatrix& InImage,
	OaU32 InX, OaU32 InY, OaU32 InWidth, OaU32 InHeight, OaF32 InValue) {
	(void)InRt;
	if (!ValidImage(InImage, "Erase") || !std::isfinite(InValue)) return InImage;
	const auto shape = InImage.GetShape();
	if (InWidth == 0 || InHeight == 0 || InX >= static_cast<OaU32>(shape[3]) ||
		InY >= static_cast<OaU32>(shape[2])) return InImage;
	return CompositePass(InImage, InImage, InImage, 2, 0.0F,
		InX, InY, InWidth, InHeight, InValue);
}

OaMatrix OaFnImage::ColorTwist(OaEngine& InRt, const OaMatrix& InImage,
	const OaMatrix& InTransform) {
	(void)InRt;
	const auto transform = InTransform.GetShape();
	if (!ValidImage(InImage, "ColorTwist") || InImage.GetShape()[1] < 3 ||
		!InTransform.HasStorage() || transform.Rank != 2 || transform[0] != 3 ||
		transform[1] != 4 || InTransform.GetDtype() != InImage.GetDtype()) return InImage;
	const auto shape = InImage.GetShape();
	auto output = OaFnMatrix::Empty(shape, InImage.GetDtype());
	struct Push { OaU32 Batch, Channels, Height, Width; };
	Push push{static_cast<OaU32>(shape[0]), static_cast<OaU32>(shape[1]),
		static_cast<OaU32>(shape[2]), static_cast<OaU32>(shape[3])};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Write};
	OaContext::GetDefault().Add("ImageColorMatrix", {&InImage, &InTransform, &output},
		access, &push, sizeof(push), DivCeil(push.Width, 16), DivCeil(push.Height, 16),
		push.Batch * push.Channels);
	return output;
}

OaMatrix OaFnImage::GaussianNoise(OaEngine& InRt, const OaMatrix& InImage,
	OaF32 InMean, OaF32 InStddev, OaU64 InSeed) {
	(void)InRt;
	if (!ValidImage(InImage, "GaussianNoise") || InStddev < 0.0F ||
		!std::isfinite(InMean) || !std::isfinite(InStddev)) return InImage;
	auto noise = OaFnMatrix::PhiloxNormal(InImage, InMean, InStddev, InSeed);
	return OaFnMatrix::Add(InImage, noise);
}

OaMatrix OaFnImage::SaltPepperNoise(OaEngine& InRt, const OaMatrix& InImage,
	OaF32 InProbability, OaF32 InSaltValue, OaF32 InPepperValue, OaU64 InSeed) {
	(void)InRt;
	if (!ValidImage(InImage, "SaltPepperNoise") || InProbability < 0.0F ||
		InProbability > 1.0F || !std::isfinite(InProbability) ||
		!std::isfinite(InSaltValue) || !std::isfinite(InPepperValue)) return InImage;
	auto random = OaFnMatrix::PhiloxUniform(InImage, 0.0F, 1.0F, InSeed);
	auto output = OaFnMatrix::Empty(InImage.GetShape(), InImage.GetDtype());
	struct Push { OaU32 NumElements; OaF32 Probability, SaltValue, PepperValue; };
	Push push{static_cast<OaU32>(InImage.NumElements()), InProbability,
		InSaltValue, InPepperValue};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Write};
	OaContext::GetDefault().Add("ImageSaltPepper", {&InImage, &random, &output},
		access, &push, sizeof(push), DivCeil(push.NumElements, 256), 1, 1);
	return output;
}
