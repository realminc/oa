// ═══════════════════════════════════════════════════════════════════════════
// TutorialNlpCommon.h — shared scaffolding for the NLP fair-comparison suite
//
// Every NLP tutorial (RNN, GRU, Transformer, MoE Transformer, Mamba-3) trains the SAME
// task on the SAME corpus with the SAME dims, so their loss/accuracy curves are
// directly comparable and the set doubles as an end-to-end regression test.
//
// The one axis that legitimately differs is the *vocabulary*:
//   - TutorialNlpByte*  → byte vocab (256), OaByteEmbedding. Universal, no tokenizer.
//   - TutorialNlpBpe*   → byte-pair encoding (320), OaEmbedding.
//   - TutorialNlpChar*  → character vocab (27 = a–z + space), OaEmbedding.
// Loss scales with ln(vocab), so compare within a vocab family; the corpus text is
// identical (lowercase + spaces) so it is valid for both.
//
// TASK: all-position dense next-token prediction. For an input window [B, S] the
// model must emit logits at *every* position [B*S, V] and predict token t+1 at
// each t. This is the real language-model objective — not the old flatten-window
// "predict one token" baselines (removed 2026-06-26). On this tiny, highly
// repetitive corpus the per-position conditional is near-deterministic, so a
// correct model still drives the averaged CE close to zero.
// ═══════════════════════════════════════════════════════════════════════════

#pragma once

#include <Oa/Ml.h>
#include <Oa/Ml/Byte.h>
#include <Oa/Runtime/Context.h>

#include <algorithm>
#include <cmath>
#include <cstring>

// ─── Shared hyperparameters (identical across the whole suite) ───────────────

inline constexpr OaI32 kContextLen = OaNlpSuiteContextLength;
inline constexpr OaI32 kDModel = OaNlpSuiteModelWidth;
inline constexpr OaI32 kHiddenDim = OaNlpSuiteHiddenWidth;
inline constexpr OaI32 kSteps = OaNlpSuiteTrainingSteps;
inline constexpr OaI32 kBatch = OaNlpSuiteBatchSize;
inline constexpr const char* kNlpGenerationPrompt = OaNlpSuiteGenerationPrompt;
inline constexpr OaI32 kNlpGenerationBytes = OaNlpSuiteGenerationSourceUnits;

// Cross-entropy is mean nats/token. Dividing by ln(2) and the represented
// source bytes/token yields tokenizer-independent bits per byte.
inline OaF64 NlpBitsPerByte(OaF64 InCrossEntropyNats, OaF64 InBytesPerToken = 1.0) {
	return InBytesPerToken > 0.0 ? InCrossEntropyNats / (std::log(2.0) * InBytesPerToken) : 0.0;
}

// ─── Shared corpus ───────────────────────────────────────────────────────────
// Lowercase letters + spaces only, so the exact same text is a valid stream for
// both the byte vocab (raw bytes) and the 27-symbol char vocab.

inline const char* NlpCorpus() {
	return OaNlpSuiteSampler::Corpus();
}

// ─── Tiny tokenizer vocab (a–z + space = 27) ────────────────────────────────
// Used by TutorialNlpChar* tutorials. Pass NlpCharEncode to the sampler.

inline constexpr OaI32 kCharVocabSize = 27;

inline OaU8 NlpCharEncode(char InChar) {
	if (InChar >= 'a' and InChar <= 'z') return static_cast<OaU8>(InChar - 'a');
	if (InChar >= 'A' and InChar <= 'Z') return static_cast<OaU8>(InChar - 'A');
	return 26;  // space / unknown
}

inline char NlpCharDecode(OaI32 InToken) {
	if (InToken >= 0 and InToken < 26) return static_cast<char>('a' + InToken);
	return ' ';
}

// ─── All-position batch sampler ──────────────────────────────────────────────
// Produces X=[batch, S] and Y=[batch, S] where Y is X shifted by one (the dense
// next-token targets). The same class serves both vocab families: pass nullptr
// for raw byte tokens, or NlpCharEncode for the 27-symbol char vocab.

class NlpAllPositionSampler {
public:
	using EncodeFn = OaU8 (*)(char);

	NlpAllPositionSampler(const char* InText, OaI32 InBatchSize, EncodeFn InEncode = nullptr)
		: BatchSize_(InBatchSize), Encode_(InEncode)
	{
		const OaI64 len = static_cast<OaI64>(std::strlen(InText));
		Tokens_.Resize(len);
		for (OaI64 i = 0; i < len; ++i) Tokens_[i] = Encode(InText[i]);
	}

	[[nodiscard]] OaI32 Encode(char InChar) const {
		return Encode_ ? static_cast<OaI32>(Encode_(InChar))
			: static_cast<OaI32>(static_cast<OaU8>(InChar));
	}

	void NextBatch(OaMatrix& OutX, OaMatrix& OutY) {
		OaVec<OaI32> x(static_cast<OaI64>(BatchSize_) * kContextLen);
		OaVec<OaI32> y(static_cast<OaI64>(BatchSize_) * kContextLen);
		const OaI64 limit = Tokens_.Size() - kContextLen - 1;
		for (OaI32 b = 0; b < BatchSize_; ++b) {
			const OaI64 start = (Cursor_ + b * 7) % limit;
			for (OaI32 t = 0; t < kContextLen; ++t) {
				x[static_cast<OaI64>(b) * kContextLen + t] = Tokens_[start + t];
				y[static_cast<OaI64>(b) * kContextLen + t] = Tokens_[start + t + 1];
			}
		}
		Cursor_ = (Cursor_ + BatchSize_) % limit;
		OutX = OaFnMatrix::FromInt32(OaSpan<const OaI32>(x.Data(), x.Size()),
			OaMatrixShape{BatchSize_, kContextLen}, OaScalarType::UInt32);
		OutY = OaFnMatrix::FromInt32(OaSpan<const OaI32>(y.Data(), y.Size()),
			OaMatrixShape{BatchSize_, kContextLen}, OaScalarType::UInt32);
	}

	// Left-aligned prompt: ids at positions 0.. (padded with 0 / space).
	[[nodiscard]] OaVec<OaI32> EncodePromptLeft(const char* InPrompt) const {
		OaVec<OaI32> out(kContextLen);
		const OaI64 len = static_cast<OaI64>(std::strlen(InPrompt));
		for (OaI32 i = 0; i < kContextLen; ++i) out[i] = Encode_ ? 26 : 0;
		for (OaI32 i = 0; i < kContextLen and i < len; ++i) out[i] = Encode(InPrompt[i]);
		return out;
	}

private:
	OaVec<OaI32> Tokens_;
	OaI32       BatchSize_;
	OaI64       Cursor_ = 0;
	EncodeFn    Encode_ = nullptr;
};

// ─── Shared evaluation: all-position argmax accuracy ─────────────────────────
// logits are [B*S, V]; targets Y are [B, S]. Compares argmax at every position.

template <class Model>
OaF32 NlpAccuracyAllPositions(Model& InModel, const OaMatrix& InX, const OaMatrix& InY, OaI32 InVocab) {
	(void)InVocab; // The class count is the logits' final dimension.
	auto logits = InModel.Forward(InX);
	return 100.0F * OaFnMetric::Accuracy(logits, InY);
}

// ─── Shared generation: greedy, all-position ─────────────────────────────────
// Feeds a [1, S] window, reads the logit row at the last filled position, takes
// the argmax, appends it, and slides the window. Greedy (deterministic) so the
// suite's sample outputs are comparable run to run. Pass a non-null EncodeFn for
// the char vocab so prompt encode + token decode use the 27-symbol mapping.

template <class Model>
OaString NlpGenerateGreedy(Model& InModel, const char* InPrompt, OaI32 InCount,
	OaI32 InVocab, NlpAllPositionSampler::EncodeFn InEncode = nullptr) {
	NlpAllPositionSampler enc(NlpCorpus(), 1, InEncode);  // only EncodePromptLeft used
	OaVec<OaI32> context = enc.EncodePromptLeft(InPrompt);
	const OaI32 promptLen = static_cast<OaI32>(std::strlen(InPrompt));
	OaI32 filled   = std::min(promptLen, kContextLen);
	OaI32 logitRow = std::max(0, filled - 1);
	OaString out(InPrompt);

	for (OaI32 i = 0; i < InCount; ++i) {
		auto x = OaFnMatrix::FromInt32(OaSpan<const OaI32>(context.Data(), context.Size()),
			OaMatrixShape{1, kContextLen}, OaScalarType::UInt32);
		auto logits = InModel.Forward(x);
		auto row = OaFnMatrix::Reshape(
			OaFnMatrix::Slice(logits, 0, logitRow, logitRow + 1),
			OaMatrixShape{InVocab});
		const OaI32 next = static_cast<OaI32>(OaFnMatrix::Argmax(row));

		out += InEncode ? NlpCharDecode(next) : static_cast<char>(static_cast<OaU8>(next));

		if (filled < kContextLen) {
			context[filled] = next;
			++filled;
			logitRow = filled - 1;
		} else {
			for (OaI32 t = 1; t < kContextLen; ++t) context[t - 1] = context[t];
			context[kContextLen - 1] = next;
			logitRow = kContextLen - 1;
		}
	}
	return out;
}

// Stateful recurrent decoding for modules whose full training scan has a
// dedicated single-token step (currently Mamba-3). Reset and prompt priming are
// synchronized deliberately: the recurrent state is mutated in place and each
// token must observe the previous token's completed state update.
template <class Model>
OaString NlpGenerateStatefulGreedy(
	Model& InModel, const char* InPrompt, OaI32 InCount, OaI32 InVocab) {
	auto& ctx = OaContext::GetDefault();
	InModel.ResetGenerationState(1);
	(void)ctx.Execute();
	(void)ctx.Sync();

	OaMatrix logits;
	const auto promptLength = static_cast<OaI32>(std::strlen(InPrompt));
	for (OaI32 index = 0; index < promptLength; ++index) {
		const OaI32 token = static_cast<OaU8>(InPrompt[index]);
		auto input = OaFnMatrix::FromInt32(
			OaSpan<const OaI32>(&token, 1),
			OaMatrixShape{1, 1},
			OaScalarType::UInt32);
		logits = InModel.ForwardGenerationStep(input);
		(void)ctx.Execute();
		(void)ctx.Sync();
	}

	OaString output(InPrompt);
	for (OaI32 index = 0; index < InCount; ++index) {
			const OaI32 next = static_cast<OaI32>(OaFnMatrix::Argmax(
			logits.Reshape(OaMatrixShape{InVocab})));
		output += static_cast<char>(static_cast<OaU8>(next));
		if (index + 1 < InCount) {
			auto input = OaFnMatrix::FromInt32(
				OaSpan<const OaI32>(&next, 1),
				OaMatrixShape{1, 1},
				OaScalarType::UInt32);
			logits = InModel.ForwardGenerationStep(input);
			(void)ctx.Execute();
			(void)ctx.Sync();
		}
	}
	return output;
}

// ─── Shared generation: temperature / top-p sampling, byte vocab only ─────
// Same sliding-window logic as NlpGenerateGreedy, but uses OaByteEncoder::Sample
// for non-deterministic output. Used by the byte-level Transformer / Mamba-3 /
// Empyrealm tutorials.

template <class Model>
OaString NlpGenerateSampled(Model& InModel, const char* InPrompt, OaI32 InCount,
	OaI32 InVocab, OaF32 InTemperature = 0.8F, OaF32 InTopP = 0.9F) {
	NlpAllPositionSampler enc(NlpCorpus(), 1);  // only EncodePromptLeft used
	OaVec<OaI32> context = enc.EncodePromptLeft(InPrompt);
	const OaI32 promptLen = static_cast<OaI32>(std::strlen(InPrompt));
	OaI32 filled = std::min(promptLen, kContextLen);
	OaI32 logitRow = std::max(0, filled - 1);
	OaString out(InPrompt);

	for (OaI32 i = 0; i < InCount; ++i) {
		auto x = OaFnMatrix::FromInt32(OaSpan<const OaI32>(context.Data(), context.Size()),
			OaMatrixShape{1, kContextLen}, OaScalarType::UInt32);
		auto logits = InModel.Forward(x);
		auto rowLogits = OaFnMatrix::Reshape(
			OaFnMatrix::Slice(logits, 0, logitRow, logitRow + 1),
			OaMatrixShape{1, InVocab});
		auto sampled = OaByteEncoder::Sample(rowLogits, InTemperature, InTopP);
		OaU8 next = sampled[0];
		out += static_cast<char>(next);

		if (filled < kContextLen) {
			context[filled] = next;
			++filled;
			logitRow = filled - 1;
		} else {
			for (OaI32 t = 1; t < kContextLen; ++t) context[t - 1] = context[t];
			context[kContextLen - 1] = next;
			logitRow = kContextLen - 1;
		}
	}
	return out;
}
