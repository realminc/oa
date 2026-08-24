// GPU-native NCHW pointwise, channel and compositing operations.

#include <oa/vision/fnImage.h>
#include <oa/runtime/executionSession.h>
#include <oa/core/bufferAccess.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/log.h>

#include <cmath>

namespace {

oa::U32 divCeil(oa::U32 inA, oa::U32 inB) {
	return (inA + inB - 1U) / inB;
}

bool validImage(const oa::Matrix& inImage, const char* inOperation) {
	const auto shape = inImage.getShape();
	if (shape.rank == 4 && shape[0] > 0 && shape[1] > 0 &&
		shape[2] > 0 && shape[3] > 0 && inImage.hasStorage()) return true;
	OaLogWarn(oa::LogComponent::Vision,
		"oa::FnImage::%s expects a stored non-empty [B,C,H,W] tensor", inOperation);
	return false;
}

bool sameImage(const oa::Matrix& inA, const oa::Matrix& inB) {
	return inA.getShape() == inB.getShape() &&
		inA.getDtype() == inB.getDtype() && inB.hasStorage();
}

oa::Matrix pointwise(const oa::Matrix& inImage, oa::U32 inOperation,
	oa::F32 inP0 = 0.0F, oa::F32 inP1 = 0.0F,
	oa::F32 inP2 = 0.0F, oa::F32 inP3 = 0.0F) {
	auto output = oa::FnMatrix::empty(inImage.getShape(), inImage.getDtype());
	struct Push { oa::U32 numElements, operation; oa::F32 p0, p1, p2, p3; };
	Push push{static_cast<oa::U32>(inImage.numElements()), inOperation,
		inP0, inP1, inP2, inP3};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
	oa::ExecutionSession::getActive().add( "ImagePointwise", {&inImage, &output}, access,
		&push, sizeof(push), divCeil(push.numElements, 256), 1, 1);
	return output;
}

oa::Matrix compositePass(const oa::Matrix& inA, const oa::Matrix& inB,
	const oa::Matrix& inMask, oa::U32 inOperation, oa::F32 inAlpha,
	oa::U32 inX = 0, oa::U32 inY = 0, oa::U32 inWidth = 0,
	oa::U32 inHeight = 0, oa::F32 inValue = 0.0F) {
	const auto shape = inA.getShape();
	auto output = oa::FnMatrix::empty(shape, inA.getDtype());
	struct Push {
		oa::U32 batch, channels, height, width, operation, maskChannels;
		oa::U32 rectX, rectY, rectWidth, rectHeight;
		oa::F32 alpha, value;
	};
	Push push{static_cast<oa::U32>(shape[0]), static_cast<oa::U32>(shape[1]),
		static_cast<oa::U32>(shape[2]), static_cast<oa::U32>(shape[3]), inOperation,
		static_cast<oa::U32>(inMask.getShape()[1]), inX, inY, inWidth, inHeight,
		inAlpha, inValue};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Write};
	oa::ExecutionSession::getActive().add( "ImageComposite", {&inA, &inB, &inMask, &output},
		access, &push, sizeof(push), divCeil(push.width, 16),
		divCeil(push.height, 16), push.batch * push.channels);
	return output;
}

} // namespace

oa::Matrix oa::FnImage::thresholdBinary(oa::Engine& inRt, const oa::Matrix& inImage,
	oa::F32 inThreshold, oa::F32 inMaxValue) {
	(void)inRt;
	if (!validImage(inImage, "ThresholdBinary") || !std::isfinite(inThreshold) ||
		!std::isfinite(inMaxValue)) return inImage;
	return pointwise(inImage, 0, inThreshold, inMaxValue);
}

oa::Matrix oa::FnImage::thresholdBinaryInv(oa::Engine& inRt, const oa::Matrix& inImage,
	oa::F32 inThreshold, oa::F32 inMaxValue) {
	(void)inRt;
	if (!validImage(inImage, "ThresholdBinaryInv") || !std::isfinite(inThreshold) ||
		!std::isfinite(inMaxValue)) return inImage;
	return pointwise(inImage, 1, inThreshold, inMaxValue);
}

oa::Matrix oa::FnImage::thresholdTruncate(oa::Engine& inRt, const oa::Matrix& inImage,
	oa::F32 inThreshold) {
	(void)inRt;
	if (!validImage(inImage, "ThresholdTruncate") || !std::isfinite(inThreshold)) return inImage;
	return pointwise(inImage, 2, inThreshold);
}

oa::Matrix oa::FnImage::thresholdToZero(oa::Engine& inRt, const oa::Matrix& inImage,
	oa::F32 inThreshold) {
	(void)inRt;
	if (!validImage(inImage, "ThresholdToZero") || !std::isfinite(inThreshold)) return inImage;
	return pointwise(inImage, 3, inThreshold);
}

oa::Matrix oa::FnImage::thresholdToZeroInv(oa::Engine& inRt, const oa::Matrix& inImage,
	oa::F32 inThreshold) {
	(void)inRt;
	if (!validImage(inImage, "ThresholdToZeroInv") || !std::isfinite(inThreshold)) return inImage;
	return pointwise(inImage, 4, inThreshold);
}

oa::Matrix oa::FnImage::inRange(oa::Engine& inRt, const oa::Matrix& inImage,
	oa::F32 inLow, oa::F32 inHigh, oa::F32 inTrueValue) {
	(void)inRt;
	if (!validImage(inImage, "inRange") || inLow > inHigh ||
		!std::isfinite(inLow) || !std::isfinite(inHigh) || !std::isfinite(inTrueValue)) return inImage;
	return pointwise(inImage, 5, inLow, inHigh, inTrueValue);
}

oa::Matrix oa::FnImage::clamp(oa::Engine& inRt, const oa::Matrix& inImage,
	oa::F32 inLow, oa::F32 inHigh) {
	(void)inRt;
	if (!validImage(inImage, "Clamp") || inLow > inHigh ||
		!std::isfinite(inLow) || !std::isfinite(inHigh)) return inImage;
	return pointwise(inImage, 6, inLow, inHigh);
}

oa::Matrix oa::FnImage::invert(oa::Engine& inRt, const oa::Matrix& inImage, oa::F32 inMaxValue) {
	(void)inRt;
	if (!validImage(inImage, "Invert") || !std::isfinite(inMaxValue)) return inImage;
	return pointwise(inImage, 7, inMaxValue);
}

oa::Matrix oa::FnImage::brightnessContrast(oa::Engine& inRt, const oa::Matrix& inImage,
	oa::F32 inBrightness, oa::F32 inContrast) {
	(void)inRt;
	if (!validImage(inImage, "BrightnessContrast") || !std::isfinite(inBrightness) ||
		!std::isfinite(inContrast)) return inImage;
	return pointwise(inImage, 8, inContrast, inBrightness);
}

oa::Matrix oa::FnImage::gammaContrast(oa::Engine& inRt, const oa::Matrix& inImage,
	oa::F32 inGamma, oa::F32 inGain) {
	(void)inRt;
	if (!validImage(inImage, "GammaContrast") || inGamma <= 0.0F ||
		!std::isfinite(inGamma) || !std::isfinite(inGain)) return inImage;
	return pointwise(inImage, 9, inGamma, inGain);
}

oa::Matrix oa::FnImage::solarize(oa::Engine& inRt, const oa::Matrix& inImage,
	oa::F32 inThreshold, oa::F32 inMaxValue) {
	(void)inRt;
	if (!validImage(inImage, "Solarize") || !std::isfinite(inThreshold) ||
		!std::isfinite(inMaxValue)) return inImage;
	return pointwise(inImage, 10, inThreshold, inMaxValue);
}

oa::Matrix oa::FnImage::posterize(oa::Engine& inRt, const oa::Matrix& inImage,
	oa::U32 inLevels, oa::F32 inLow, oa::F32 inHigh) {
	(void)inRt;
	if (!validImage(inImage, "Posterize") || inLevels < 2 || inLevels > 65536 ||
		inLow >= inHigh || !std::isfinite(inLow) || !std::isfinite(inHigh)) return inImage;
	return pointwise(inImage, 11, static_cast<oa::F32>(inLevels), inLow, inHigh);
}

oa::Matrix oa::FnImage::grayscale(oa::Engine& inRt, const oa::Matrix& inImage) {
	(void)inRt;
	if (!validImage(inImage, "Grayscale") || inImage.getShape()[1] < 3) return inImage;
	const auto shape = inImage.getShape();
	auto output = oa::FnMatrix::empty({shape[0], 1, shape[2], shape[3]}, inImage.getDtype());
	struct Push { oa::U32 batch, inChannels, outChannels, height, width, operation;
		oa::U32 order0, order1, order2, order3; };
	Push push{static_cast<oa::U32>(shape[0]), static_cast<oa::U32>(shape[1]), 1,
		static_cast<oa::U32>(shape[2]), static_cast<oa::U32>(shape[3]), 0, 0, 1, 2, 3};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
	oa::ExecutionSession::getActive().add( "ImageChannelTransform", {&inImage, &output}, access,
		&push, sizeof(push), divCeil(push.width, 16), divCeil(push.height, 16), push.batch);
	return output;
}

oa::Matrix oa::FnImage::channelReorder(oa::Engine& inRt, const oa::Matrix& inImage,
	oa::U32 inChannel0, oa::U32 inChannel1, oa::U32 inChannel2, oa::U32 inChannel3) {
	(void)inRt;
	if (!validImage(inImage, "ChannelReorder")) return inImage;
	const auto shape = inImage.getShape();
	const oa::U32 channels = static_cast<oa::U32>(shape[1]);
	const oa::U32 order[] = {inChannel0, inChannel1, inChannel2, inChannel3};
	if (channels > 4) return inImage;
	for (oa::U32 i = 0; i < channels; ++i) if (order[i] >= channels) return inImage;
	auto output = oa::FnMatrix::empty(shape, inImage.getDtype());
	struct Push { oa::U32 batch, inChannels, outChannels, height, width, operation;
		oa::U32 order0, order1, order2, order3; };
	Push push{static_cast<oa::U32>(shape[0]), channels, channels,
		static_cast<oa::U32>(shape[2]), static_cast<oa::U32>(shape[3]), 1,
		inChannel0, inChannel1, inChannel2, inChannel3};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
	oa::ExecutionSession::getActive().add( "ImageChannelTransform", {&inImage, &output}, access,
		&push, sizeof(push), divCeil(push.width, 16), divCeil(push.height, 16),
		push.batch * channels);
	return output;
}

oa::Matrix oa::FnImage::alphaBlend(oa::Engine& inRt, const oa::Matrix& inA,
	const oa::Matrix& inB, oa::F32 inAlpha) {
	(void)inRt;
	if (!validImage(inA, "AlphaBlend") || !sameImage(inA, inB) ||
		!std::isfinite(inAlpha) || inAlpha < 0.0F || inAlpha > 1.0F) return inA;
	return compositePass(inA, inB, inA, 0, inAlpha);
}

oa::Matrix oa::FnImage::composite(oa::Engine& inRt, const oa::Matrix& inA,
	const oa::Matrix& inB, const oa::Matrix& inMask) {
	(void)inRt;
	if (!validImage(inA, "Composite") || !sameImage(inA, inB) ||
		!validImage(inMask, "Composite")) return inA;
	const auto a = inA.getShape();
	const auto m = inMask.getShape();
	if (a[0] != m[0] || a[2] != m[2] || a[3] != m[3] ||
		(m[1] != 1 && m[1] != a[1]) || inMask.getDtype() != inA.getDtype()) return inA;
	return compositePass(inA, inB, inMask, 1, 0.0F);
}

oa::Matrix oa::FnImage::erase(oa::Engine& inRt, const oa::Matrix& inImage,
	oa::U32 inX, oa::U32 inY, oa::U32 inWidth, oa::U32 inHeight, oa::F32 inValue) {
	(void)inRt;
	if (!validImage(inImage, "erase") || !std::isfinite(inValue)) return inImage;
	const auto shape = inImage.getShape();
	if (inWidth == 0 || inHeight == 0 || inX >= static_cast<oa::U32>(shape[3]) ||
		inY >= static_cast<oa::U32>(shape[2])) return inImage;
	return compositePass(inImage, inImage, inImage, 2, 0.0F,
		inX, inY, inWidth, inHeight, inValue);
}

oa::Matrix oa::FnImage::colorTwist(oa::Engine& inRt, const oa::Matrix& inImage,
	const oa::Matrix& inTransform) {
	(void)inRt;
	const auto transform = inTransform.getShape();
	if (!validImage(inImage, "ColorTwist") || inImage.getShape()[1] < 3 ||
		!inTransform.hasStorage() || transform.rank != 2 || transform[0] != 3 ||
		transform[1] != 4 || inTransform.getDtype() != inImage.getDtype()) return inImage;
	const auto shape = inImage.getShape();
	auto output = oa::FnMatrix::empty(shape, inImage.getDtype());
	struct Push { oa::U32 batch, channels, height, width; };
	Push push{static_cast<oa::U32>(shape[0]), static_cast<oa::U32>(shape[1]),
		static_cast<oa::U32>(shape[2]), static_cast<oa::U32>(shape[3])};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Write};
	oa::ExecutionSession::getActive().add( "ImageColorMatrix", {&inImage, &inTransform, &output},
		access, &push, sizeof(push), divCeil(push.width, 16), divCeil(push.height, 16),
		push.batch * push.channels);
	return output;
}

oa::Matrix oa::FnImage::gaussianNoise(oa::Engine& inRt, const oa::Matrix& inImage,
	oa::F32 inMean, oa::F32 inStddev, oa::U64 inSeed) {
	(void)inRt;
	if (!validImage(inImage, "GaussianNoise") || inStddev < 0.0F ||
		!std::isfinite(inMean) || !std::isfinite(inStddev)) return inImage;
	auto noise = oa::FnMatrix::philoxNormal(inImage, inMean, inStddev, inSeed);
	return oa::FnMatrix::add(inImage, noise);
}

oa::Matrix oa::FnImage::saltPepperNoise(oa::Engine& inRt, const oa::Matrix& inImage,
	oa::F32 inProbability, oa::F32 inSaltValue, oa::F32 inPepperValue, oa::U64 inSeed) {
	(void)inRt;
	if (!validImage(inImage, "SaltPepperNoise") || inProbability < 0.0F ||
		inProbability > 1.0F || !std::isfinite(inProbability) ||
		!std::isfinite(inSaltValue) || !std::isfinite(inPepperValue)) return inImage;
	auto random = oa::FnMatrix::philoxUniform(inImage, 0.0F, 1.0F, inSeed);
	auto output = oa::FnMatrix::empty(inImage.getShape(), inImage.getDtype());
	struct Push { oa::U32 numElements; oa::F32 probability, saltValue, pepperValue; };
	Push push{static_cast<oa::U32>(inImage.numElements()), inProbability,
		inSaltValue, inPepperValue};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Write};
	oa::ExecutionSession::getActive().add( "ImageSaltPepper", {&inImage, &random, &output},
		access, &push, sizeof(push), divCeil(push.numElements, 256), 1, 1);
	return output;
}
