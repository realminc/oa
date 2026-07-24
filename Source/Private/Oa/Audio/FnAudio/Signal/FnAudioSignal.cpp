// FnAudioSignal.cpp — hand-written OaFnAudio Signal implementations.
// The schema owns their declarations and contracts; this file owns the
// manual_context bodies.
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

static OaAudio WrapLike(OaMatrix InMatrix, const OaAudio& InAudio) {
	if (InMatrix.IsEmpty()) return {};
	return OaAudio(
		OaStdMove(InMatrix), InAudio.SampleRate(), InAudio.Layout());
}

OaAudio Normalize(
	const OaAudio& InAudio, OaF32 InTargetDb, OaU8 InMode) {
	if (not InAudio.Validate() || InAudio.IsEmpty()) return {};
	const OaMatrix& InA = InAudio.AsMatrix();
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
		return WrapLike(
			OaFnMatrix::Scale(InA, 1.0F), InAudio);
	}
	return WrapLike(
		OaFnMatrix::Scale(InA, targetLin / level), InAudio);
}

OaAudio Resample(
	const OaAudio& InAudio,
	OaU32 InOutRate,
	OaU32 InFilterHalfWidth) {
	if (not InAudio.Validate() || InAudio.IsEmpty()) return {};
	const OaMatrix& InA = InAudio.AsMatrix();
	const OaU32 InInRate = InAudio.SampleRate();
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
		return WrapLike(OaFnMatrix::Scale(InA, 1.0F), InAudio);
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
	return OaAudio(OaStdMove(out), InOutRate, InAudio.Layout());
}

OaAudio Mix(
	const OaAudio& InA,
	const OaAudio& InB,
	OaF32 InGainA,
	OaF32 InGainB) {
	if (not InA.Validate() || not InB.Validate()
		|| InA.IsEmpty() || InB.IsEmpty()
		|| InA.SampleRate() != InB.SampleRate()
		|| InA.Layout() != InB.Layout()
		|| InA.AsMatrix().GetShape() != InB.AsMatrix().GetShape()
		|| not std::isfinite(InGainA) || not std::isfinite(InGainB)) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"OaFnAudio::Mix: inputs must have matching valid audio contracts");
		return {};
	}
	const OaMatrix& a = InA.AsMatrix();
	const OaMatrix& b = InB.AsMatrix();
	OaMatrix out = OaFnMatrix::Empty(a.GetShape(), OaScalarType::Float32);
	const OaU64 count64 = static_cast<OaU64>(a.NumElements());
	if (count64 > std::numeric_limits<OaU32>::max()) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"OaFnAudio::Mix: dispatch exceeds u32 limits");
		return {};
	}
	struct {
		OaU32 Count;
		OaF32 GainA;
		OaF32 GainB;
	} push{
		.Count = static_cast<OaU32>(count64),
		.GainA = InGainA,
		.GainB = InGainB,
	};
	OaBufferAccess access[] = {
		OaBufferAccess::Read,
		OaBufferAccess::Read,
		OaBufferAccess::Write,
	};
	OaContext::GetDefault().Add(
		"AudioMix", {&a, &b, &out}, access, &push, sizeof(push),
		(push.Count + 255U) / 256U);
	return WrapLike(OaStdMove(out), InA);
}

OaAudio Gain(const OaAudio& InAudio, OaF32 InGainDb) {
	if (not InAudio.Validate() || InAudio.IsEmpty()) return {};
	if (!std::isfinite(InGainDb) || InGainDb < -300.0F || InGainDb > 100.0F) return {};
	return WrapLike(
		OaFnMatrix::Scale(InAudio.AsMatrix(), DbToLinear(InGainDb)), InAudio);
}

OaAudio Clip(const OaAudio& InAudio, OaF32 InMin, OaF32 InMax) {
	if (not InAudio.Validate() || InAudio.IsEmpty()) return {};
	if (!std::isfinite(InMin) || !std::isfinite(InMax) || InMin > InMax) return {};
	return WrapLike(
		OaFnMatrix::ClampMax(
			OaFnMatrix::ClampMin(InAudio.AsMatrix(), InMin), InMax),
		InAudio);
}

OaMatrix AmplitudeToDb(const OaAudio& InAudio, OaF32 InFloorDb) {
	if (not InAudio.Validate() || InAudio.IsEmpty()) return {};
	if (!std::isfinite(InFloorDb) || InFloorDb < -300.0F || InFloorDb > 0.0F) return {};
	// 20·log10(max(|x|, floor)) — floor keeps silence finite.
	const OaF32 floorLin = DbToLinear(InFloorDb);
	return OaFnMatrix::Scale(
		OaFnMatrix::Log(OaFnMatrix::ClampMin(
			OaFnMatrix::Abs(InAudio.AsMatrix()), floorLin)),
		kLn10Over20Inv
	);
}

OaAudio PreEmphasis(const OaAudio& InAudio, OaF32 InAlpha) {
	if (not InAudio.Validate() || InAudio.IsEmpty()) return {};
	const OaMatrix& InBuf = InAudio.AsMatrix();
	// y[n] = x[n] − α·x[n−1], y[0] = x[0] (zero-padded left neighbor).
	// Composed: shift right by one via Zeros+Slice+Concat, then Sub(x, α·shift).
	const auto& shape = InBuf.GetShape();
	if (shape.Rank != 2 || shape[0] <= 0 || shape[1] <= 0 ||
		InBuf.GetDtype() != OaScalarType::Float32 || !std::isfinite(InAlpha)) {
		OA_LOG_ERROR(OaLogComponent::Core, "OaFnAudio::PreEmphasis: expected [Channels, Samples], rank=%d", shape.Rank);
		return {};
	}
	if (shape[1] < 2) {
		return WrapLike(OaFnMatrix::Scale(InBuf, 1.0F), InAudio);
	}
	auto z = OaFnMatrix::Zeros(OaMatrixShape{shape[0], 1}, OaScalarType::Float32);
	auto head = OaFnMatrix::Slice(InBuf, 1, 0, shape[1] - 1);   // x[0 .. S−2]
	OaMatrix parts[] = {z, head};
	auto shifted = OaFnMatrix::Concat(OaSpan<OaMatrix>(parts, 2), 1);
	return WrapLike(
		OaFnMatrix::Sub(InBuf, OaFnMatrix::Scale(shifted, InAlpha)),
		InAudio);
}

OaAudio ToMono(const OaAudio& InAudio) {
	if (not InAudio.Validate() || InAudio.IsEmpty()) return {};
	const OaMatrix& InBuf = InAudio.AsMatrix();
	const auto& shape = InBuf.GetShape();
	if (shape.Rank != 2 || shape[0] <= 0 || shape[1] <= 0 || InBuf.GetDtype() != OaScalarType::Float32) {
		OA_LOG_ERROR(OaLogComponent::Core, "OaFnAudio::ToMono: expected [Channels, Samples], rank=%d", shape.Rank);
		return {};
	}
	if (shape[0] == 1) {
		return OaAudio(
			OaFnMatrix::Scale(InBuf, 1.0F),
			InAudio.SampleRate(),
			OaChannelLayout::Mono);
	}
	OaMatrix mean = OaFnMatrix::Mean(InBuf, 0);  // reduce channel dim
	return OaAudio(
		OaFnMatrix::Reshape(mean, OaMatrixShape{1, shape[1]}),
		InAudio.SampleRate(),
		OaChannelLayout::Mono);
}

OaAudio Fade(
	const OaAudio& InAudio,
	OaU64 InFadeInSamples,
	OaU64 InFadeOutSamples) {
	if (not InAudio.Validate() || InAudio.IsEmpty()) return {};
	const OaMatrix& InBuf = InAudio.AsMatrix();
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
	return WrapLike(OaStdMove(out), InAudio);
}

OaMatrix WaveformEnvelope(const OaAudio& InAudio, OaU32 InBins) {
	if (not InAudio.Validate() || InAudio.IsEmpty()) return {};
	const OaMatrix& InBuf = InAudio.AsMatrix();
	const auto& shape = InBuf.GetShape();
	if (shape.Rank != 2 || shape[0] <= 0 || shape[1] <= 0
		|| InBuf.GetDtype() != OaScalarType::Float32 || InBins == 0U
		|| InBins > 65'536U
		|| shape[0] > static_cast<OaI64>(std::numeric_limits<OaU32>::max())
		|| shape[1] > static_cast<OaI64>(std::numeric_limits<OaU32>::max())) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"OaFnAudio::WaveformEnvelope: expected non-empty F32 [Channels, Samples] and bins in [1, 65536]");
		return {};
	}

	OaMatrix out = OaFnMatrix::Empty(
		OaMatrixShape{static_cast<OaI64>(InBins), 2},
		OaScalarType::Float32);
	struct {
		OaU32 Channels;
		OaU32 Samples;
		OaU32 Bins;
	} push{
		.Channels = static_cast<OaU32>(shape[0]),
		.Samples = static_cast<OaU32>(shape[1]),
		.Bins = InBins,
	};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	OaContext::GetDefault().Add(
		"AudioWaveformEnvelope",
		{&InBuf, &out},
		access,
		&push,
		sizeof(push),
		(InBins + 255U) / 256U);
	return out;
}

} // namespace OaFnAudio
