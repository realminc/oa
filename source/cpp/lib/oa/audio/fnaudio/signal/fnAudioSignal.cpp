// fnAudioSignal.cpp — hand-written oa::FnAudio signal implementations.
// The schema owns their declarations and contracts; this file owns the
// manual_session bodies.
//
// Pointwise and recurrent signal operations dispatch native Audio kernels;
// simple algebraic operations compose verified FnMatrix kernels. Multi-node
// lowerings remain one schema-owned semantic operation.

#include <oa/audio/fnAudio.h>
#include <oa/core/log.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/std/memory.h>
#include <oa/core/bufferAccess.h>
#include <oa/core/op.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/engine.h>
#include <oa/core/std/algo.h>
#include <oa/core/std/array.h>
#include <oa/core/std/utility.h>
#include <oa/core/std/scalarMath.h>
#include <oa/core/std/limits.h>

namespace oa {

namespace FnAudio {

constexpr oa::F32 kLn10Over20Inv = 8.68588963806504F;  // 20 / ln(10)

static oa::F32 dbToLinear(oa::F32 inDb) { return oa::pow(10.0F, inDb / 20.0F); }

static Audio wrapLike(oa::Matrix inMatrix, const Audio& inAudio) {
	if (inMatrix.isEmpty()) return {};
	return Audio(
		oa::move(inMatrix), inAudio.sampleRate(), inAudio.layout());
}

Audio normalize(const Audio& inAudio, oa::F32 inTargetDb, oa::U8 inMode) {
	if (not inAudio.validate() || inAudio.isEmpty()) return {};
	const oa::Matrix& inA = inAudio.asMatrix();
	const auto& shape = inA.getShape();
	if (shape.rank != 2 || shape[0] <= 0 || shape[1] <= 0 ||
		inA.getDtype() != oa::ScalarType::Float32 || !oa::isFinite(inTargetDb) ||
		inTargetDb < -300.0F || inTargetDb > 100.0F || inMode > 1) {
		OaLogError(oa::LogComponent::Audio,
			"oa::FnAudio::normalize: expected non-empty F32 [channels, samples], finite target, mode 0 or 1");
		return {};
	}
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	// global (all-channel) level. Peak: max |x|. RMS: sqrt(mean x²).
	// Keep the level and gain on device so normalization remains one semantic
	// operation with no hidden execute/readback boundary.
	const oa::F32 targetLin = dbToLinear(inTargetDb);
	oa::Matrix level;
	if (inMode == 0) {
		level = oa::FnMatrix::max(oa::FnMatrix::abs(inA));
	} else {
		level = oa::FnMatrix::sqrt(oa::FnMatrix::mean(oa::FnMatrix::mul(inA, inA)));
	}
	auto safeLevel = oa::FnMatrix::clampMin(
		level, oa::Limits<oa::F32>::min());
	auto gain = oa::FnMatrix::scale(
		oa::FnMatrix::reciprocal(safeLevel), targetLin);
	Audio result = wrapLike(oa::FnMatrix::mul(inA, gain), inAudio);
	const auto status = lowering.commit(
		oa::detail::opRegistry::FnAudio::normalize,
		{&inA}, {&result.asMatrix()},
		{
			oa::OpAttribute::fromFloat("targetDb", inTargetDb),
			oa::OpAttribute::fromUnsignedInteger("mode", inMode),
		});
	if (not status.isOk()) return {};
	return result;
}

Audio resample(
	const Audio& inAudio,
	oa::U32 inOutRate,
	oa::U32 inFilterHalfWidth) {
	if (not inAudio.validate() || inAudio.isEmpty()) return {};
	const oa::Matrix& inA = inAudio.asMatrix();
	const oa::U32 inInRate = inAudio.sampleRate();
	const auto& shape = inA.getShape();
	if (shape.rank != 2 || shape[0] <= 0 || shape[1] <= 0 || inA.getDtype() != oa::ScalarType::Float32) {
		OaLogError(oa::LogComponent::Audio, "oa::FnAudio::resample: expected [channels, samples], rank={}", shape.rank);
		return {};
	}
	if (inInRate == 0 || inOutRate == 0 || inFilterHalfWidth == 0 || inFilterHalfWidth > 1024) {
		OaLogError(oa::LogComponent::Audio,
			"oa::FnAudio::resample: rates must be > 0 and filterHalfWidth in [1, 1024]");
		return {};
	}
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	if (inInRate == inOutRate) {
		Audio result = wrapLike(oa::FnMatrix::scale(inA, 1.0F), inAudio);
		const auto status = lowering.commit(
			oa::detail::opRegistry::FnAudio::resample,
			{&inA}, {&result.asMatrix()},
			{
				oa::OpAttribute::fromUnsignedInteger("outRate", inOutRate),
				oa::OpAttribute::fromUnsignedInteger(
					"filterHalfWidth", inFilterHalfWidth),
			});
		if (not status.isOk()) return {};
		return result;
	}

	// gcd-reduce so the shader's rational source positioning stays in u32.
	const oa::U32 g       = oa::gcd(inInRate, inOutRate);
	const oa::U32 inRateR  = inInRate  / g;
	const oa::U32 outRateR = inOutRate / g;

	const oa::U32 channels   = static_cast<oa::U32>(shape[0]);
	const oa::U64 inSamples  = static_cast<oa::U64>(shape[1]);
	if (inSamples > oa::Limits<oa::U64>::max() / outRateR) {
		OaLogError(oa::LogComponent::Audio, "oa::FnAudio::resample: output length overflows");
		return {};
	}
	const oa::U64 outSamples = inSamples * outRateR / inRateR;
	if (outSamples == 0) {
		OaLogError(oa::LogComponent::Audio, "oa::FnAudio::resample: output would be empty");
		return {};
	}

	if (inSamples > oa::Limits<oa::U32>::max() ||
		outSamples > oa::Limits<oa::U32>::max() ||
		outSamples > oa::Limits<oa::U32>::max() / channels) {
		OaLogError(oa::LogComponent::Audio, "oa::FnAudio::resample: dispatch exceeds u32 limits");
		return {};
	}
	oa::Matrix out = oa::FnMatrix::empty(oa::MatrixShape{shape[0], static_cast<oa::I64>(outSamples)}, oa::ScalarType::Float32);
	const oa::U32 count = static_cast<oa::U32>(channels * outSamples);

	struct {
		oa::U32 count; oa::U32 inRateR; oa::U32 outRateR; oa::U32 filterHalfWidth;
		oa::U32 inSamples; oa::U32 outSamples;
	} push{
		.count = count, .inRateR = inRateR, .outRateR = outRateR,
		.filterHalfWidth = inFilterHalfWidth,
		.inSamples = static_cast<oa::U32>(inSamples),
		.outSamples = static_cast<oa::U32>(outSamples)
	};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "AudioResample", {&inA, &out}, access, &push, sizeof(push), (count + 255) / 256);
	Audio result(oa::move(out), inOutRate, inAudio.layout());
	const auto status = lowering.commit(
		oa::detail::opRegistry::FnAudio::resample,
		{&inA}, {&result.asMatrix()},
		{
			oa::OpAttribute::fromUnsignedInteger("outRate", inOutRate),
			oa::OpAttribute::fromUnsignedInteger(
				"filterHalfWidth", inFilterHalfWidth),
		});
	if (not status.isOk()) return {};
	return result;
}

Audio mix(
	const Audio& inA,
	const Audio& inB,
	oa::F32 inGainA,
	oa::F32 inGainB) {
	if (not inA.validate() || not inB.validate()
		|| inA.isEmpty() || inB.isEmpty()
		|| inA.sampleRate() != inB.sampleRate()
		|| inA.layout() != inB.layout()
		|| inA.asMatrix().getShape() != inB.asMatrix().getShape()
		|| not oa::isFinite(inGainA) || not oa::isFinite(inGainB)) {
		OaLogError(oa::LogComponent::Audio,
			"oa::FnAudio::mix: inputs must have matching valid audio contracts");
		return {};
	}
	const oa::Matrix& a = inA.asMatrix();
	const oa::Matrix& b = inB.asMatrix();
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	oa::Matrix out = oa::FnMatrix::empty(a.getShape(), oa::ScalarType::Float32);
	const oa::U64 count64 = static_cast<oa::U64>(a.numElements());
	if (count64 > oa::Limits<oa::U32>::max()) {
		OaLogError(oa::LogComponent::Audio,
			"oa::FnAudio::mix: dispatch exceeds u32 limits");
		return {};
	}
	struct {
		oa::U32 count;
		oa::F32 gainA;
		oa::F32 gainB;
	} push{
		.count = static_cast<oa::U32>(count64),
		.gainA = inGainA,
		.gainB = inGainB,
	};
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read,
		oa::BufferAccess::Read,
		oa::BufferAccess::Write,
	};
	ctx.add(
		"AudioMix", {&a, &b, &out}, access, &push, sizeof(push),
		(push.count + 255U) / 256U);
	Audio result = wrapLike(oa::move(out), inA);
	const auto status = lowering.commit(
		oa::detail::opRegistry::FnAudio::mix,
		{&a, &b}, {&result.asMatrix()},
		{
			oa::OpAttribute::fromFloat("gainA", inGainA),
			oa::OpAttribute::fromFloat("gainB", inGainB),
		});
	if (not status.isOk()) return {};
	return result;
}

Audio gain(const Audio& inAudio, oa::F32 inGainDb) {
	if (not inAudio.validate() || inAudio.isEmpty()) return {};
	if (!oa::isFinite(inGainDb) || inGainDb < -300.0F || inGainDb > 100.0F) return {};
	const oa::Matrix& input = inAudio.asMatrix();
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	Audio result = wrapLike(
		oa::FnMatrix::scale(inAudio.asMatrix(), dbToLinear(inGainDb)), inAudio);
	const auto status = lowering.commit(
		oa::detail::opRegistry::FnAudio::gain,
		{&input}, {&result.asMatrix()},
		{oa::OpAttribute::fromFloat("gainDb", inGainDb)});
	if (not status.isOk()) return {};
	return result;
}

Audio clip(const Audio& inAudio, oa::F32 inMin, oa::F32 inMax) {
	if (not inAudio.validate() || inAudio.isEmpty()) return {};
	if (!oa::isFinite(inMin) || !oa::isFinite(inMax) || inMin > inMax) return {};
	const oa::Matrix& input = inAudio.asMatrix();
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	Audio result = wrapLike(
		oa::FnMatrix::clampMax(
			oa::FnMatrix::clampMin(inAudio.asMatrix(), inMin), inMax),
		inAudio);
	const auto status = lowering.commit(
		oa::detail::opRegistry::FnAudio::clip,
		{&input}, {&result.asMatrix()},
		{
			oa::OpAttribute::fromFloat("min", inMin),
			oa::OpAttribute::fromFloat("max", inMax),
		});
	if (not status.isOk()) return {};
	return result;
}

Audio saturate(
	const Audio& inAudio,
	oa::F32 inDriveDb,
	oa::F32 inMix) {
	if (not inAudio.validate() || inAudio.isEmpty()) return {};
	const oa::Matrix& input = inAudio.asMatrix();
	const oa::U64 count64 = static_cast<oa::U64>(input.numElements());
	if (!oa::isFinite(inDriveDb) || inDriveDb < -60.0F || inDriveDb > 60.0F
		|| !oa::isFinite(inMix) || inMix < 0.0F || inMix > 1.0F
		|| count64 == 0 || count64 > oa::Limits<oa::U32>::max()) {
		OaLogError(oa::LogComponent::Audio,
			"oa::FnAudio::saturate: expected drive in [-60, 60] dB, mix in [0, 1], and u32-addressable audio");
		return {};
	}

	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	oa::Matrix out = oa::FnMatrix::empty(input.getShape(), oa::ScalarType::Float32);
	struct {
		oa::U32 count;
		oa::F32 driveDb;
		oa::F32 mix;
	} push{
		.count = static_cast<oa::U32>(count64),
		.driveDb = inDriveDb,
		.mix = inMix,
	};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add(
		"AudioSaturate",
		{&input, &out},
		access,
		&push,
		sizeof(push),
		(push.count + 255U) / 256U);
	Audio result = wrapLike(oa::move(out), inAudio);
	const auto status = lowering.commit(
		oa::detail::opRegistry::FnAudio::saturate,
		{&input},
		{&result.asMatrix()},
		{
			oa::OpAttribute::fromFloat("driveDb", inDriveDb),
			oa::OpAttribute::fromFloat("mix", inMix),
		});
	if (not status.isOk()) return {};
	return result;
}

Audio reverb(
	const Audio& inAudio,
	oa::F32 inDecaySeconds,
	oa::F32 inWet) {
	if (not inAudio.validate() or inAudio.isEmpty()) return {};
	const oa::Matrix& input = inAudio.asMatrix();
	const auto& shape = input.getShape();
	if (shape.rank != 2 or shape[0] <= 0 or shape[1] <= 0
		or input.getDtype() != oa::ScalarType::Float32
		or not oa::isFinite(inDecaySeconds)
		or inDecaySeconds < 0.1F or inDecaySeconds > 10.0F
		or not oa::isFinite(inWet) or inWet < 0.0F or inWet > 1.0F) {
		OaLogError(oa::LogComponent::Audio,
			"oa::FnAudio::reverb: expected non-empty F32 [channels, samples], decaySeconds in [0.1, 10], and wet in [0, 1]");
		return {};
	}

	const oa::U64 channels64 = static_cast<oa::U64>(shape[0]);
	const oa::U64 inSamples64 = static_cast<oa::U64>(shape[1]);
	const oa::U64 tailSamples64 = static_cast<oa::U64>(oa::ceil(
		static_cast<oa::F64>(inAudio.sampleRate())
		* static_cast<oa::F64>(inDecaySeconds)));
	const oa::U64 maxAddressableSamples = oa::Limits<oa::U32>::max();
	if (tailSamples64 == 0U or tailSamples64 > maxAddressableSamples
		or inSamples64 > maxAddressableSamples - tailSamples64) {
		OaLogError(oa::LogComponent::Audio,
			"oa::FnAudio::reverb: rendered tail exceeds u32 sample addressing");
		return {};
	}
	const oa::U64 outSamples64 = inSamples64 + tailSamples64;
	if (channels64 > oa::Limits<oa::U32>::max()
		or outSamples64 > oa::Limits<oa::U32>::max()
		or channels64 > oa::Limits<oa::U32>::max() / outSamples64) {
		OaLogError(oa::LogComponent::Audio,
			"oa::FnAudio::reverb: output dispatch exceeds u32 addressing");
		return {};
	}

	constexpr oa::Array<oa::F64, 4> combDelaySeconds{
		0.0297, 0.0371, 0.0411, 0.0437};
	constexpr oa::Array<oa::F64, 2> allpassDelaySeconds{0.0050, 0.0017};
	constexpr oa::F32 allpassGain = 0.7F;
	const oa::U32 channels = static_cast<oa::U32>(channels64);
	const oa::U32 inSamples = static_cast<oa::U32>(inSamples64);
	const oa::U32 outSamples = static_cast<oa::U32>(outSamples64);
	const oa::U32 count = static_cast<oa::U32>(channels64 * outSamples64);

	oa::Array<oa::U32, 4> combDelays{};
	oa::Array<oa::F32, 4> combFeedback{};
	oa::F64 normalizationDenominator = 0.0;
	for (oa::Usize index = 0; index < combDelaySeconds.size(); ++index) {
		const oa::U64 delay = oa::max<oa::U64>(1U, static_cast<oa::U64>(
			oa::round(combDelaySeconds[index] * inAudio.sampleRate())));
		if (delay > oa::Limits<oa::U32>::max()
			or channels64 > oa::Limits<oa::U32>::max() / delay) {
			OaLogError(oa::LogComponent::Audio,
				"oa::FnAudio::reverb: comb dispatch exceeds u32 addressing");
			return {};
		}
		combDelays[index] = static_cast<oa::U32>(delay);
		combFeedback[index] = static_cast<oa::F32>(oa::pow(
			0.001,
			(static_cast<oa::F64>(delay) / inAudio.sampleRate())
				/ static_cast<oa::F64>(inDecaySeconds)));
		normalizationDenominator += 1.0 / (1.0 - combFeedback[index]);
	}
	oa::Array<oa::U32, 2> allpassDelays{};
	for (oa::Usize index = 0; index < allpassDelaySeconds.size(); ++index) {
		const oa::U64 delay = oa::max<oa::U64>(1U, static_cast<oa::U64>(
			oa::round(allpassDelaySeconds[index] * inAudio.sampleRate())));
		if (delay > oa::Limits<oa::U32>::max()
			or channels64 > oa::Limits<oa::U32>::max() / delay) {
			OaLogError(oa::LogComponent::Audio,
				"oa::FnAudio::reverb: all-pass dispatch exceeds u32 addressing");
			return {};
		}
		allpassDelays[index] = static_cast<oa::U32>(delay);
	}
	const oa::F32 normalization = static_cast<oa::F32>(
		1.0 / normalizationDenominator);

	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	const oa::MatrixShape outputShape{
		static_cast<oa::I64>(channels), static_cast<oa::I64>(outSamples)};
	oa::Array<oa::Matrix, 4> combs;
	for (oa::Usize index = 0; index < combs.size(); ++index) {
		combs[index] = oa::FnMatrix::empty(outputShape, oa::ScalarType::Float32);
		if (combs[index].isEmpty()) return {};
		struct {
			oa::U32 channels;
			oa::U32 inSamples;
			oa::U32 outSamples;
			oa::U32 delaySamples;
			oa::F32 feedback;
		} push{
			.channels = channels,
			.inSamples = inSamples,
			.outSamples = outSamples,
			.delaySamples = combDelays[index],
			.feedback = combFeedback[index],
		};
		oa::BufferAccess access[] = {
			oa::BufferAccess::Read, oa::BufferAccess::Write};
		const oa::U32 tasks = channels * combDelays[index];
		ctx.add(
			"AudioReverbComb", {&input, &combs[index]}, access,
			&push, sizeof(push), (tasks + 255U) / 256U);
	}

	oa::Matrix summed = oa::FnMatrix::empty(outputShape, oa::ScalarType::Float32);
	if (summed.isEmpty()) return {};
	struct {
		oa::U32 count;
		oa::F32 normalization;
	} sumPush{.count = count, .normalization = normalization};
	oa::BufferAccess sumAccess[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Write};
	ctx.add(
		"AudioReverbSum",
		{&combs[0], &combs[1], &combs[2], &combs[3], &summed},
		sumAccess, &sumPush, sizeof(sumPush), (count + 255U) / 256U);

	oa::Array<oa::Matrix, 2> diffused;
	const oa::Matrix* diffuserInput = &summed;
	for (oa::Usize index = 0; index < diffused.size(); ++index) {
		diffused[index] = oa::FnMatrix::empty(outputShape, oa::ScalarType::Float32);
		if (diffused[index].isEmpty()) return {};
		struct {
			oa::U32 channels;
			oa::U32 samples;
			oa::U32 delaySamples;
			oa::F32 gain;
		} push{
			.channels = channels,
			.samples = outSamples,
			.delaySamples = allpassDelays[index],
			.gain = allpassGain,
		};
		oa::BufferAccess access[] = {
			oa::BufferAccess::Read, oa::BufferAccess::Write};
		const oa::U32 tasks = channels * allpassDelays[index];
		ctx.add(
			"AudioReverbAllpass", {diffuserInput, &diffused[index]}, access,
			&push, sizeof(push), (tasks + 255U) / 256U);
		diffuserInput = &diffused[index];
	}

	oa::Matrix output = oa::FnMatrix::empty(outputShape, oa::ScalarType::Float32);
	if (output.isEmpty()) return {};
	struct {
		oa::U32 channels;
		oa::U32 inSamples;
		oa::U32 outSamples;
		oa::F32 wet;
	} mixPush{
		.channels = channels,
		.inSamples = inSamples,
		.outSamples = outSamples,
		.wet = inWet,
	};
	oa::BufferAccess mixAccess[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Write};
	ctx.add(
		"AudioReverbMix", {&input, diffuserInput, &output}, mixAccess,
		&mixPush, sizeof(mixPush), (count + 255U) / 256U);

	Audio result(oa::move(output), inAudio.sampleRate(), inAudio.layout());
	const auto status = lowering.commit(
		oa::detail::opRegistry::FnAudio::reverb,
		{&input}, {&result.asMatrix()},
		{
			oa::OpAttribute::fromFloat("decaySeconds", inDecaySeconds),
			oa::OpAttribute::fromFloat("wet", inWet),
		});
	if (not status.isOk()) return {};
	return result;
}

static bool isStableBiquad(const BiquadCoefficients& inCoefficients) {
	const bool finite = oa::isFinite(inCoefficients.b0)
		and oa::isFinite(inCoefficients.b1)
		and oa::isFinite(inCoefficients.b2)
		and oa::isFinite(inCoefficients.a1)
		and oa::isFinite(inCoefficients.a2);
	if (not finite) return false;

	// Jury stability test for 1 + a1*z^-1 + a2*z^-2. Strict inequalities
	// reject poles on the unit circle because this offline operation promises a
	// bounded zero-state filter, not an oscillator or unstable recurrence.
	const oa::F64 a1 = static_cast<oa::F64>(inCoefficients.a1);
	const oa::F64 a2 = static_cast<oa::F64>(inCoefficients.a2);
	return 1.0 + a1 + a2 > 0.0
		and 1.0 - a1 + a2 > 0.0
		and 1.0 - a2 > 0.0;
}

constexpr oa::U32 kBiquadBlockSize = 256U;
constexpr oa::U32 kBiquadThreads = 64U;
constexpr oa::Usize kMaximumSosSections = 64U;

struct BiquadDispatchShape {
	oa::U32 channels = 0U;
	oa::U32 samples = 0U;
	oa::U32 blocks = 0U;
	oa::U32 tasks = 0U;
};

static bool resolveBiquadDispatchShape(
	const oa::Matrix& inInput,
	BiquadDispatchShape& outShape) {
	const oa::MatrixShape shape = inInput.getShape();
	const oa::U64 channels64 = static_cast<oa::U64>(shape[0]);
	const oa::U64 samples64 = static_cast<oa::U64>(shape[1]);
	const oa::U64 blocks64 =
		(samples64 + kBiquadBlockSize - 1U) / kBiquadBlockSize;
	if (channels64 > oa::Limits<oa::U32>::max()
		or samples64 > oa::Limits<oa::U32>::max()
		or channels64 > oa::Limits<oa::U32>::max() / samples64
		or channels64 > oa::Limits<oa::U32>::max() / blocks64
		or channels64 * blocks64 > oa::Limits<oa::U32>::max() / 6U) {
		return false;
	}
	outShape = BiquadDispatchShape{
		.channels = static_cast<oa::U32>(channels64),
		.samples = static_cast<oa::U32>(samples64),
		.blocks = static_cast<oa::U32>(blocks64),
		.tasks = static_cast<oa::U32>(channels64 * blocks64),
	};
	return true;
}

static void addBiquadSection(
	oa::ExecutionSession& inContext,
	const oa::Matrix& inInput,
	oa::Matrix& inOutSummaries,
	oa::Matrix& inOutStates,
	oa::Matrix& outOutput,
	const BiquadDispatchShape& inShape,
	const BiquadCoefficients& inCoefficients) {
	struct FilterPush {
		oa::U32 channels;
		oa::U32 samples;
		oa::U32 blocks;
		oa::U32 blockSize;
		oa::F32 b0;
		oa::F32 b1;
		oa::F32 b2;
		oa::F32 a1;
		oa::F32 a2;
	} filterPush{
		.channels = inShape.channels,
		.samples = inShape.samples,
		.blocks = inShape.blocks,
		.blockSize = kBiquadBlockSize,
		.b0 = inCoefficients.b0,
		.b1 = inCoefficients.b1,
		.b2 = inCoefficients.b2,
		.a1 = inCoefficients.a1,
		.a2 = inCoefficients.a2,
	};
	oa::BufferAccess summaryAccess[] = {
		oa::BufferAccess::Read,
		oa::BufferAccess::Write,
	};
	inContext.add(
		"AudioBiquadBlockSummary",
		{&inInput, &inOutSummaries},
		summaryAccess,
		&filterPush,
		sizeof(filterPush),
		(inShape.tasks + kBiquadThreads - 1U) / kBiquadThreads);

	struct ScanPush {
		oa::U32 channels;
		oa::U32 blocks;
	} scanPush{
		.channels = inShape.channels,
		.blocks = inShape.blocks,
	};
	oa::BufferAccess scanAccess[] = {
		oa::BufferAccess::Read,
		oa::BufferAccess::Write,
	};
	inContext.add(
		"AudioBiquadBlockScan",
		{&inOutSummaries, &inOutStates},
		scanAccess,
		&scanPush,
		sizeof(scanPush),
		(inShape.channels + kBiquadThreads - 1U) / kBiquadThreads);

	// The executable graph derives same-queue compute write->read barriers for
	// summary -> scan and state -> apply from these access declarations. The
	// input remains read-only and the output is not visible before exact event
	// completion.
	oa::BufferAccess applyAccess[] = {
		oa::BufferAccess::Read,
		oa::BufferAccess::Read,
		oa::BufferAccess::Write,
	};
	inContext.add(
		"AudioBiquadApply",
		{&inInput, &inOutStates, &outOutput},
		applyAccess,
		&filterPush,
		sizeof(filterPush),
		(inShape.tasks + kBiquadThreads - 1U) / kBiquadThreads);
}

static oa::U64 appendSosHash(oa::U64 inHash, oa::F32 inValue) {
	constexpr oa::U64 prime = 1099511628211ULL;
	const oa::U32 bits = oa::bitCast<oa::U32>(inValue);
	for (oa::U32 shift = 0U; shift < 32U; shift += 8U) {
		inHash ^= static_cast<oa::U8>(bits >> shift);
		inHash *= prime;
	}
	return inHash;
}

static oa::U64 sosCoefficientHash(
	oa::Span<const BiquadCoefficients> inSections) {
	oa::U64 hash = 14695981039346656037ULL;
	for (const auto& section : inSections) {
		hash = appendSosHash(hash, section.b0);
		hash = appendSosHash(hash, section.b1);
		hash = appendSosHash(hash, section.b2);
		hash = appendSosHash(hash, section.a1);
		hash = appendSosHash(hash, section.a2);
	}
	return hash;
}

Audio biquad(
	const Audio& inAudio,
	const BiquadCoefficients& inCoefficients) {
	if (not inAudio.validate() or inAudio.isEmpty()
		or not isStableBiquad(inCoefficients)) {
		OaLogError(oa::LogComponent::Audio,
			"oa::FnAudio::biquad: expected valid F32 audio and finite stable a0-normalized coefficients");
		return {};
	}

	const oa::Matrix& input = inAudio.asMatrix();
	BiquadDispatchShape dispatchShape;
	if (not resolveBiquadDispatchShape(input, dispatchShape)) {
		OaLogError(oa::LogComponent::Audio,
			"oa::FnAudio::biquad: input or block workspace exceeds u32 shader addressing");
		return {};
	}

	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	oa::Matrix summaries = oa::FnMatrix::empty(
		oa::MatrixShape{
			static_cast<oa::I64>(dispatchShape.channels),
			static_cast<oa::I64>(dispatchShape.blocks), 6},
		oa::ScalarType::Float32);
	oa::Matrix states = oa::FnMatrix::empty(
		oa::MatrixShape{
			static_cast<oa::I64>(dispatchShape.channels),
			static_cast<oa::I64>(dispatchShape.blocks), 2},
		oa::ScalarType::Float32);
	oa::Matrix output = oa::FnMatrix::empty(
		input.getShape(), oa::ScalarType::Float32);
	if (summaries.isEmpty() or states.isEmpty() or output.isEmpty()) return {};
	addBiquadSection(
		ctx, input, summaries, states, output, dispatchShape, inCoefficients);

	Audio result = wrapLike(oa::move(output), inAudio);
	const auto status = lowering.commit(
		oa::detail::opRegistry::FnAudio::biquad,
		{&input},
		{&result.asMatrix()},
		{
			oa::OpAttribute::fromFloat("b0", inCoefficients.b0),
			oa::OpAttribute::fromFloat("b1", inCoefficients.b1),
			oa::OpAttribute::fromFloat("b2", inCoefficients.b2),
			oa::OpAttribute::fromFloat("a1", inCoefficients.a1),
			oa::OpAttribute::fromFloat("a2", inCoefficients.a2),
		});
	if (not status.isOk()) return {};
	return result;
}

Audio sosFilter(
	const Audio& inAudio,
	oa::Span<const BiquadCoefficients> inSections) {
	if (not inAudio.validate() or inAudio.isEmpty()
		or inSections.empty() or inSections.size() > kMaximumSosSections) {
		OaLogError(oa::LogComponent::Audio,
			"oa::FnAudio::sosFilter: expected valid F32 audio and one to 64 stable sections");
		return {};
	}
	for (const auto& section : inSections) {
		if (not isStableBiquad(section)) {
			OaLogError(oa::LogComponent::Audio,
				"oa::FnAudio::sosFilter: every section must be finite, stable, and a0-normalized");
			return {};
		}
	}

	const oa::Matrix& input = inAudio.asMatrix();
	BiquadDispatchShape dispatchShape;
	if (not resolveBiquadDispatchShape(input, dispatchShape)) {
		OaLogError(oa::LogComponent::Audio,
			"oa::FnAudio::sosFilter: input or block workspace exceeds u32 shader addressing");
		return {};
	}

	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	oa::Matrix summaries = oa::FnMatrix::empty(
		oa::MatrixShape{
			static_cast<oa::I64>(dispatchShape.channels),
			static_cast<oa::I64>(dispatchShape.blocks), 6},
		oa::ScalarType::Float32);
	oa::Matrix states = oa::FnMatrix::empty(
		oa::MatrixShape{
			static_cast<oa::I64>(dispatchShape.channels),
			static_cast<oa::I64>(dispatchShape.blocks), 2},
		oa::ScalarType::Float32);
	oa::Matrix primary = oa::FnMatrix::empty(
		input.getShape(), oa::ScalarType::Float32);
	oa::Matrix secondary;
	if (inSections.size() > 1U) {
		secondary = oa::FnMatrix::empty(
			input.getShape(), oa::ScalarType::Float32);
	}
	if (summaries.isEmpty() or states.isEmpty() or primary.isEmpty()
		or (inSections.size() > 1U and secondary.isEmpty())) {
		return {};
	}

	const oa::Matrix* sectionInput = &input;
	oa::Matrix* sectionOutput = &primary;
	for (const auto& section : inSections) {
		addBiquadSection(
			ctx, *sectionInput, summaries, states, *sectionOutput,
			dispatchShape, section);
		sectionInput = sectionOutput;
		sectionOutput = sectionOutput == &primary ? &secondary : &primary;
	}

	oa::Matrix output;
	if ((inSections.size() & 1U) != 0U) {
		output = oa::move(primary);
	} else {
		output = oa::move(secondary);
	}
	Audio result = wrapLike(oa::move(output), inAudio);
	const auto status = lowering.commit(
		oa::detail::opRegistry::FnAudio::sosFilter,
		{&input},
		{&result.asMatrix()},
		{
			oa::OpAttribute::fromUnsignedInteger(
				"sectionCount", static_cast<oa::U64>(inSections.size())),
			oa::OpAttribute::fromUnsignedInteger(
				"coefficientHash", sosCoefficientHash(inSections)),
		});
	if (not status.isOk()) return {};
	return result;
}

oa::Matrix amplitudeToDb(const Audio& inAudio, oa::F32 inFloorDb) {
	if (not inAudio.validate() || inAudio.isEmpty()) return {};
	if (!oa::isFinite(inFloorDb) || inFloorDb < -300.0F || inFloorDb > 0.0F) return {};
	const oa::Matrix& input = inAudio.asMatrix();
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	// 20·log10(max(|x|, floor)) — floor keeps silence finite.
	const oa::F32 floorLin = dbToLinear(inFloorDb);
	oa::Matrix result = oa::FnMatrix::scale(
		oa::FnMatrix::log(oa::FnMatrix::clampMin(
			oa::FnMatrix::abs(inAudio.asMatrix()), floorLin)),
		kLn10Over20Inv
	);
	const auto status = lowering.commit(
		oa::detail::opRegistry::FnAudio::amplitudeToDb,
		{&input}, {&result},
		{oa::OpAttribute::fromFloat("floorDb", inFloorDb)});
	if (not status.isOk()) return {};
	return result;
}

Audio preEmphasis(const Audio& inAudio, oa::F32 inAlpha) {
	if (not inAudio.validate() || inAudio.isEmpty()) return {};
	const oa::Matrix& inBuf = inAudio.asMatrix();
	// y[n] = x[n] − α·x[n−1], y[0] = x[0] (zero-padded left neighbor).
	// Composed: shift right by one via Zeros+Slice+Concat, then sub(x, α·shift).
	const auto& shape = inBuf.getShape();
	if (shape.rank != 2 || shape[0] <= 0 || shape[1] <= 0 ||
		inBuf.getDtype() != oa::ScalarType::Float32 || !oa::isFinite(inAlpha)) {
		OaLogError(oa::LogComponent::Audio, "oa::FnAudio::preEmphasis: expected [channels, samples], rank={}", shape.rank);
		return {};
	}
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	if (shape[1] < 2) {
		Audio result = wrapLike(
			oa::FnMatrix::scale(inBuf, 1.0F), inAudio);
		const auto status = lowering.commit(
			oa::detail::opRegistry::FnAudio::preEmphasis,
			{&inBuf}, {&result.asMatrix()},
			{oa::OpAttribute::fromFloat("alpha", inAlpha)});
		if (not status.isOk()) return {};
		return result;
	}
	auto z = oa::FnMatrix::zeros(oa::MatrixShape{shape[0], 1}, oa::ScalarType::Float32);
	auto head = oa::FnMatrix::slice(inBuf, 1, 0, shape[1] - 1);   // x[0 .. S−2]
	oa::Matrix parts[] = {z, head};
	auto shifted = oa::FnMatrix::concat(oa::Span<oa::Matrix>(parts, 2), 1);
	Audio result = wrapLike(
		oa::FnMatrix::sub(inBuf, oa::FnMatrix::scale(shifted, inAlpha)),
		inAudio);
	const auto status = lowering.commit(
		oa::detail::opRegistry::FnAudio::preEmphasis,
		{&inBuf}, {&result.asMatrix()},
		{oa::OpAttribute::fromFloat("alpha", inAlpha)});
	if (not status.isOk()) return {};
	return result;
}

Audio toMono(const Audio& inAudio) {
	if (not inAudio.validate() || inAudio.isEmpty()) return {};
	const oa::Matrix& inBuf = inAudio.asMatrix();
	const auto& shape = inBuf.getShape();
	if (shape.rank != 2 || shape[0] <= 0 || shape[1] <= 0 || inBuf.getDtype() != oa::ScalarType::Float32) {
		OaLogError(oa::LogComponent::Audio, "oa::FnAudio::toMono: expected [channels, samples], rank={}", shape.rank);
		return {};
	}
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	Audio result;
	if (shape[0] == 1) {
		result = Audio(
			oa::FnMatrix::scale(inBuf, 1.0F),
			inAudio.sampleRate(),
			AudioChannelLayout::Mono);
	} else {
		oa::Matrix mean = oa::FnMatrix::mean(inBuf, 0);  // reduce channel dim
		result = Audio(
			oa::FnMatrix::reshape(mean, oa::MatrixShape{1, shape[1]}),
			inAudio.sampleRate(),
			AudioChannelLayout::Mono);
	}
	const auto status = lowering.commit(
		oa::detail::opRegistry::FnAudio::toMono,
		{&inBuf}, {&result.asMatrix()});
	if (not status.isOk()) return {};
	return result;
}

Audio fade(
	const Audio& inAudio,
	oa::U64 inFadeInSamples,
	oa::U64 inFadeOutSamples) {
	if (not inAudio.validate() || inAudio.isEmpty()) return {};
	const oa::Matrix& inBuf = inAudio.asMatrix();
	const auto& shape = inBuf.getShape();
	if (shape.rank != 2 || shape[0] <= 0 || shape[1] <= 0 || inBuf.getDtype() != oa::ScalarType::Float32) {
		OaLogError(oa::LogComponent::Audio, "oa::FnAudio::fade: expected [channels, samples], rank={}", shape.rank);
		return {};
	}
	const oa::U64 samples  = static_cast<oa::U64>(shape[1]);
	const oa::U64 fadeIn   = inFadeInSamples  < samples ? inFadeInSamples  : samples;
	const oa::U64 fadeOut  = inFadeOutSamples < samples ? inFadeOutSamples : samples;
	const oa::U64 count64 = static_cast<oa::U64>(shape[0]) * samples;
	if (samples > oa::Limits<oa::U32>::max() || count64 > oa::Limits<oa::U32>::max()) {
		OaLogError(oa::LogComponent::Audio, "oa::FnAudio::fade: dispatch exceeds u32 limits");
		return {};
	}

	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	oa::Matrix out = oa::FnMatrix::empty(shape, oa::ScalarType::Float32);
	struct { oa::U32 count; oa::U32 samples; oa::U32 fadeIn; oa::U32 fadeOut; } push{
		.count = static_cast<oa::U32>(count64),
		.samples = static_cast<oa::U32>(samples),
		.fadeIn = static_cast<oa::U32>(fadeIn),
		.fadeOut = static_cast<oa::U32>(fadeOut),
	};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add(
		"AudioFade", {&inBuf, &out}, access, &push, sizeof(push), (push.count + 255U) / 256U);
	Audio result = wrapLike(oa::move(out), inAudio);
	const auto status = lowering.commit(
		oa::detail::opRegistry::FnAudio::fade,
		{&inBuf}, {&result.asMatrix()},
		{
			oa::OpAttribute::fromUnsignedInteger(
				"fadeInSamples", inFadeInSamples),
			oa::OpAttribute::fromUnsignedInteger(
				"fadeOutSamples", inFadeOutSamples),
		});
	if (not status.isOk()) return {};
	return result;
}

oa::Matrix waveformEnvelope(const Audio& inAudio, oa::U32 inBins) {
	if (not inAudio.validate() || inAudio.isEmpty()) return {};
	const oa::Matrix& inBuf = inAudio.asMatrix();
	const auto& shape = inBuf.getShape();
	if (shape.rank != 2 || shape[0] <= 0 || shape[1] <= 0
		|| inBuf.getDtype() != oa::ScalarType::Float32 || inBins == 0U
		|| inBins > 65'536U
		|| shape[0] > static_cast<oa::I64>(oa::Limits<oa::U32>::max())
		|| shape[1] > static_cast<oa::I64>(oa::Limits<oa::U32>::max())) {
		OaLogError(oa::LogComponent::Audio,
			"oa::FnAudio::waveformEnvelope: expected non-empty F32 [channels, samples] and bins in [1, 65536]");
		return {};
	}

	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	oa::Matrix out = oa::FnMatrix::empty(
		oa::MatrixShape{static_cast<oa::I64>(inBins), 2},
		oa::ScalarType::Float32);
	struct {
		oa::U32 channels;
		oa::U32 samples;
		oa::U32 bins;
	} push{
		.channels = static_cast<oa::U32>(shape[0]),
		.samples = static_cast<oa::U32>(shape[1]),
		.bins = inBins,
	};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add(
		"AudioWaveformEnvelope",
		{&inBuf, &out},
		access,
		&push,
		sizeof(push),
		(inBins + 255U) / 256U);
	const auto status = lowering.commit(
		oa::detail::opRegistry::FnAudio::waveformEnvelope,
		{&inBuf}, {&out},
		{oa::OpAttribute::fromUnsignedInteger("bins", inBins)});
	if (not status.isOk()) return {};
	return out;
}

} // namespace FnAudio

} // namespace oa
