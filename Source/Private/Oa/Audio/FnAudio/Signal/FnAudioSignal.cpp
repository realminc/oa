// FnAudioSignal.cpp — hand-written OaFnAudio Signal implementations.
// The generated Signal wrapper (Mix) lives in FnAudioSignal.gen.cpp.
//
// Everything except Resample is composed from existing verified FnMatrix
// kernels (Scale/Clamp/Log/Abs/Max/Mean/Mul); Resample dispatches the
// windowed-sinc AudioResample kernel with a shape-changing output.

#include <Oa/Audio/FnAudio.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/Memory.h>
#include <Oa/Core/BufferAccess.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Engine.h>
#include <cmath>
#include <numeric>
#include <limits>

namespace OaFnAudio {

constexpr OaF32 kLn10Over20Inv = 8.68588963806504F;  // 20 / ln(10)

static OaF32 DbToLinear(OaF32 InDb) { return std::pow(10.0F, InDb / 20.0F); }


OaMatrix Normalize(const OaMatrix& InA, OaF32 InTargetDb, OaU8 InMode) {
	const auto& shape = InA.GetShape();
	if (shape.Rank != 2 || shape[0] <= 0 || shape[1] <= 0 ||
		InA.GetDtype() != OaScalarType::Float32 || !std::isfinite(InTargetDb) ||
		InTargetDb < -300.0F || InTargetDb > 100.0F || InMode > 1) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"OaFnAudio::Normalize: expected non-empty F32 [Channels, Samples], finite target, mode 0 or 1");
		return {};
	}
	// Global (all-channel) level. Peak: max |x|. RMS: sqrt(mean x²).
	// Uses a host readback of one scalar (OaFnMatrix::Scalar syncs) — audio
	// normalization is a pipeline-boundary op, not an inner-loop op.
	const OaF32 targetLin = DbToLinear(InTargetDb);
	OaF32 level = 0.0F;
	if (InMode == 0) {
		level = OaFnMatrix::Scalar(OaFnMatrix::Max(OaFnMatrix::Abs(InA)));
	} else {
		level = std::sqrt(OaFnMatrix::Scalar(OaFnMatrix::Mean(OaFnMatrix::Mul(InA, InA))));
	}
	if (!(level > 0.0F)) {
		return OaFnMatrix::Scale(InA, 1.0F);   // silence: return an unscaled copy
	}
	return OaFnMatrix::Scale(InA, targetLin / level);
}

OaMatrix Resample(const OaMatrix& InA, OaU32 InInRate, OaU32 InOutRate, OaU32 InFilterHalfWidth) {
	const auto& shape = InA.GetShape();
	if (shape.Rank != 2 || shape[0] <= 0 || shape[1] <= 0 || InA.GetDtype() != OaScalarType::Float32) {
		OA_LOG_ERROR(OaLogComponent::Core, "OaFnAudio::Resample: expected [Channels, Samples], rank=%d", shape.Rank);
		return {};
	}
	if (InInRate == 0 || InOutRate == 0 || InFilterHalfWidth == 0 || InFilterHalfWidth > 1024) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"OaFnAudio::Resample: rates must be > 0 and FilterHalfWidth in [1, 1024]");
		return {};
	}
	if (InInRate == InOutRate) {
		return OaFnMatrix::Scale(InA, 1.0F);   // no-op: copy
	}

	// gcd-reduce so the shader's rational source positioning stays in u32.
	const OaU32 g       = std::gcd(InInRate, InOutRate);
	const OaU32 inRateR  = InInRate  / g;
	const OaU32 outRateR = InOutRate / g;

	const OaU32 channels   = static_cast<OaU32>(shape[0]);
	const OaU64 inSamples  = static_cast<OaU64>(shape[1]);
	if (inSamples > std::numeric_limits<OaU64>::max() / outRateR) {
		OA_LOG_ERROR(OaLogComponent::Core, "OaFnAudio::Resample: output length overflows");
		return {};
	}
	const OaU64 outSamples = inSamples * outRateR / inRateR;
	if (outSamples == 0) {
		OA_LOG_ERROR(OaLogComponent::Core, "OaFnAudio::Resample: output would be empty");
		return {};
	}

	if (inSamples > std::numeric_limits<OaU32>::max() ||
		outSamples > std::numeric_limits<OaU32>::max() ||
		outSamples > std::numeric_limits<OaU32>::max() / channels) {
		OA_LOG_ERROR(OaLogComponent::Core, "OaFnAudio::Resample: dispatch exceeds u32 limits");
		return {};
	}
	OaMatrix out = OaFnMatrix::Empty(OaMatrixShape{shape[0], static_cast<OaI64>(outSamples)}, OaScalarType::Float32);
	const OaU32 count = static_cast<OaU32>(channels * outSamples);

	auto& ctx = OaContext::GetDefault();
	struct {
		OaU32 Count; OaU32 InRateR; OaU32 OutRateR; OaU32 FilterHalfWidth;
		OaU32 InSamples; OaU32 OutSamples;
	} push{
		.Count = count, .InRateR = inRateR, .OutRateR = outRateR,
		.FilterHalfWidth = InFilterHalfWidth,
		.InSamples = static_cast<OaU32>(inSamples),
		.OutSamples = static_cast<OaU32>(outSamples)
	};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("AudioResample", {&InA, &out}, access, &push, sizeof(push), (count + 255) / 256);
	return out;
}

OaMatrix Gain(const OaAudioBuffer& InBuf, OaF32 InGainDb) {
	if (!std::isfinite(InGainDb) || InGainDb < -300.0F || InGainDb > 100.0F) return {};
	return OaFnMatrix::Scale(InBuf, DbToLinear(InGainDb));
}

OaMatrix Clip(const OaAudioBuffer& InBuf, OaF32 InMin, OaF32 InMax) {
	if (!std::isfinite(InMin) || !std::isfinite(InMax) || InMin > InMax) return {};
	return OaFnMatrix::ClampMax(OaFnMatrix::ClampMin(InBuf, InMin), InMax);
}

OaMatrix AmplitudeToDb(const OaAudioBuffer& InBuf, OaF32 InFloorDb) {
	if (!std::isfinite(InFloorDb) || InFloorDb < -300.0F || InFloorDb > 0.0F) return {};
	// 20·log10(max(|x|, floor)) — floor keeps silence finite.
	const OaF32 floorLin = DbToLinear(InFloorDb);
	return OaFnMatrix::Scale(
		OaFnMatrix::Log(OaFnMatrix::ClampMin(OaFnMatrix::Abs(InBuf), floorLin)),
		kLn10Over20Inv
	);
}

OaMatrix PreEmphasis(const OaAudioBuffer& InBuf, OaF32 InAlpha) {
	// y[n] = x[n] − α·x[n−1], y[0] = x[0] (zero-padded left neighbor).
	// Composed: shift right by one via Zeros+Slice+Concat, then Sub(x, α·shift).
	const auto& shape = InBuf.GetShape();
	if (shape.Rank != 2 || shape[0] <= 0 || shape[1] <= 0 ||
		InBuf.GetDtype() != OaScalarType::Float32 || !std::isfinite(InAlpha)) {
		OA_LOG_ERROR(OaLogComponent::Core, "OaFnAudio::PreEmphasis: expected [Channels, Samples], rank=%d", shape.Rank);
		return {};
	}
	if (shape[1] < 2) {
		return OaFnMatrix::Scale(InBuf, 1.0F);   // nothing to emphasize: copy
	}
	auto z = OaFnMatrix::Zeros(OaMatrixShape{shape[0], 1}, OaScalarType::Float32);
	auto head = OaFnMatrix::Slice(InBuf, 1, 0, shape[1] - 1);   // x[0 .. S−2]
	OaMatrix parts[] = {z, head};
	auto shifted = OaFnMatrix::Concat(OaSpan<OaMatrix>(parts, 2), 1);
	return OaFnMatrix::Sub(InBuf, OaFnMatrix::Scale(shifted, InAlpha));
}

OaMatrix ToMono(const OaAudioBuffer& InBuf) {
	const auto& shape = InBuf.GetShape();
	if (shape.Rank != 2 || shape[0] <= 0 || shape[1] <= 0 || InBuf.GetDtype() != OaScalarType::Float32) {
		OA_LOG_ERROR(OaLogComponent::Core, "OaFnAudio::ToMono: expected [Channels, Samples], rank=%d", shape.Rank);
		return {};
	}
	if (shape[0] == 1) {
		return OaFnMatrix::Scale(InBuf, 1.0F);   // already mono: copy
	}
	OaMatrix mean = OaFnMatrix::Mean(InBuf, 0);  // reduce channel dim
	return OaFnMatrix::Reshape(mean, OaMatrixShape{1, shape[1]});
}

OaMatrix Fade(const OaAudioBuffer& InBuf, OaU64 InFadeInSamples, OaU64 InFadeOutSamples) {
	const auto& shape = InBuf.GetShape();
	if (shape.Rank != 2 || shape[0] <= 0 || shape[1] <= 0 || InBuf.GetDtype() != OaScalarType::Float32) {
		OA_LOG_ERROR(OaLogComponent::Core, "OaFnAudio::Fade: expected [Channels, Samples], rank=%d", shape.Rank);
		return {};
	}
	const OaU64 samples  = static_cast<OaU64>(shape[1]);
	const OaU64 fadeIn   = InFadeInSamples  < samples ? InFadeInSamples  : samples;
	const OaU64 fadeOut  = InFadeOutSamples < samples ? InFadeOutSamples : samples;
	const OaU64 count64 = static_cast<OaU64>(shape[0]) * samples;
	if (samples > std::numeric_limits<OaU32>::max() || count64 > std::numeric_limits<OaU32>::max()) {
		OA_LOG_ERROR(OaLogComponent::Core, "OaFnAudio::Fade: dispatch exceeds u32 limits");
		return {};
	}

	OaMatrix out = OaFnMatrix::Empty(shape, OaScalarType::Float32);
	struct { OaU32 Count; OaU32 Samples; OaU32 FadeIn; OaU32 FadeOut; } push{
		.Count = static_cast<OaU32>(count64),
		.Samples = static_cast<OaU32>(samples),
		.FadeIn = static_cast<OaU32>(fadeIn),
		.FadeOut = static_cast<OaU32>(fadeOut),
	};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	OaContext::GetDefault().Add(
		"AudioFade", {&InBuf, &out}, access, &push, sizeof(push), (push.Count + 255U) / 256U);
	return out;
}

} // namespace OaFnAudio
