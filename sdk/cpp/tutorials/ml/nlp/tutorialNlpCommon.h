// ═══════════════════════════════════════════════════════════════════════════
// TutorialNlpCommon.h — shared scaffolding for the NLP fair-comparison suite
//
// Every NLP tutorial (RNN, GRU, Transformer, MoE Transformer, Mamba-3) trains the SAME
// task on the SAME corpus with the SAME dims, so their loss/accuracy curves are
// directly comparable and the set doubles as an end-to-end regression test.
//
// The one axis that legitimately differs is the *vocabulary*:
//   - TutorialNlpByte*  → byte vocab (256), oa::ByteEmbedding. Universal, no tokenizer.
//   - TutorialNlpBpe*   → byte-pair encoding (320), oa::Embedding.
//   - TutorialNlpChar*  → character vocab (27 = a–z + space), oa::Embedding.
// loss scales with ln(vocab), so compare within a vocab family; the corpus text is
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

#include "tutorialMl.h"

#include <oa/ml.h>
#include <oa/ml/byte.h>
#include <ml/nlpSuite.h>

#include <algorithm>
#include <cmath>
#include <cstring>

// ─── Shared hyperparameters (identical across the whole suite) ───────────────

inline constexpr oa::I32 kContextLen = oa::NlpSuiteContextLength;
inline constexpr oa::I32 kDModel = oa::NlpSuiteModelWidth;
inline constexpr oa::I32 kHiddenDim = oa::NlpSuiteHiddenWidth;
inline constexpr oa::I32 kSteps = oa::NlpSuiteTrainingSteps;
inline constexpr oa::I32 kBatch = oa::NlpSuiteBatchSize;
inline constexpr const char* kNlpGenerationPrompt = oa::NlpSuiteGenerationPrompt;
inline constexpr oa::I32 kNlpGenerationBytes = oa::NlpSuiteGenerationSourceUnits;

// Cross-entropy is mean nats/token. Dividing by ln(2) and the represented
// source bytes/token yields tokenizer-independent bits per byte.
inline oa::F64 nlpBitsPerByte(oa::F64 inCrossEntropyNats, oa::F64 inBytesPerToken = 1.0) {
	return inBytesPerToken > 0.0 ? inCrossEntropyNats / (std::log(2.0) * inBytesPerToken) : 0.0;
}

// ─── Shared corpus ───────────────────────────────────────────────────────────
// Lowercase letters + spaces only, so the exact same text is a valid stream for
// both the byte vocab (raw bytes) and the 27-symbol char vocab.

inline const char* nlpCorpus() {
	return oa::NlpSuiteSampler::corpus();
}

// ─── Tiny tokenizer vocab (a–z + space = 27) ────────────────────────────────
// Used by TutorialNlpChar* tutorials. Pass NlpCharEncode to the sampler.

inline constexpr oa::I32 kCharVocabSize = 27;

inline oa::U8 nlpCharEncode(char inChar) {
	if (inChar >= 'a' and inChar <= 'z') return static_cast<oa::U8>(inChar - 'a');
	if (inChar >= 'A' and inChar <= 'Z') return static_cast<oa::U8>(inChar - 'A');
	return 26;  // space / unknown
}

inline char nlpCharDecode(oa::I32 inToken) {
	if (inToken >= 0 and inToken < 26) return static_cast<char>('a' + inToken);
	return ' ';
}

// ─── All-position batch sampler ──────────────────────────────────────────────
// Produces X=[batch, S] and Y=[batch, S] where Y is X shifted by one (the dense
// next-token targets). The same class serves both vocab families: pass nullptr
// for raw byte tokens, or NlpCharEncode for the 27-symbol char vocab.

class NlpAllPositionSampler {
public:
	using EncodeFn = oa::U8 (*)(char);

	NlpAllPositionSampler(const char* inText, oa::I32 inBatchSize, EncodeFn inEncode = nullptr)
		: batchSize_(inBatchSize), encode_(inEncode)
	{
		const oa::I64 len = static_cast<oa::I64>(std::strlen(inText));
		tokens_.resize(len);
		for (oa::I64 i = 0; i < len; ++i) tokens_[i] = encode(inText[i]);
	}

	[[nodiscard]] oa::I32 encode(char inChar) const {
		return encode_ ? static_cast<oa::I32>(encode_(inChar))
			: static_cast<oa::I32>(static_cast<oa::U8>(inChar));
	}

	void nextBatch(oa::Matrix& outX, oa::Matrix& outY) {
		oa::Vector<oa::I32> x(static_cast<oa::I64>(batchSize_) * kContextLen);
		oa::Vector<oa::I32> y(static_cast<oa::I64>(batchSize_) * kContextLen);
		const oa::I64 limit = tokens_.size() - kContextLen - 1;
		for (oa::I32 b = 0; b < batchSize_; ++b) {
			const oa::I64 start = (cursor_ + b * 7) % limit;
			for (oa::I32 t = 0; t < kContextLen; ++t) {
				x[static_cast<oa::I64>(b) * kContextLen + t] = tokens_[start + t];
				y[static_cast<oa::I64>(b) * kContextLen + t] = tokens_[start + t + 1];
			}
		}
		cursor_ = (cursor_ + batchSize_) % limit;
		outX = oa::FnMatrix::fromInt32(oa::Span<const oa::I32>(x.data(), x.size()),
			oa::MatrixShape{batchSize_, kContextLen}, oa::ScalarType::UInt32);
		outY = oa::FnMatrix::fromInt32(oa::Span<const oa::I32>(y.data(), y.size()),
			oa::MatrixShape{batchSize_, kContextLen}, oa::ScalarType::UInt32);
	}

	// Left-aligned prompt: ids at positions 0.. (padded with 0 / space).
	[[nodiscard]] oa::Vector<oa::I32> encodePromptLeft(const char* inPrompt) const {
		oa::Vector<oa::I32> out(kContextLen);
		const oa::I64 len = static_cast<oa::I64>(std::strlen(inPrompt));
		for (oa::I32 i = 0; i < kContextLen; ++i) out[i] = encode_ ? 26 : 0;
		for (oa::I32 i = 0; i < kContextLen and i < len; ++i) out[i] = encode(inPrompt[i]);
		return out;
	}

private:
	oa::Vector<oa::I32> tokens_;
	oa::I32       batchSize_;
	oa::I64       cursor_ = 0;
	EncodeFn    encode_ = nullptr;
};

// ─── Shared evaluation: all-position argmax accuracy ─────────────────────────
// logits are [B*S, V]; targets Y are [B, S]. Compares argmax at every position.

template <class Model>
oa::F32 nlpAccuracyAllPositions(Model& inModel, const oa::Matrix& inX, const oa::Matrix& inY, oa::I32 inVocab) {
	(void)inVocab; // The class count is the logits' final dimension.
	auto logits = inModel.forward(inX);
	return 100.0F * oa::FnMetric::accuracy(logits, inY);
}

// ─── Shared generation: greedy, all-position ─────────────────────────────────
// Feeds a [1, S] window, reads the logit row at the last filled position, takes
// the argmax, appends it, and slides the window. greedy (deterministic) so the
// suite's sample outputs are comparable run to run. Pass a non-null EncodeFn for
// the char vocab so prompt encode + token decode use the 27-symbol mapping.

template <class Model>
oa::String nlpGenerateGreedy(Model& inModel, const char* inPrompt, oa::I32 inCount,
	oa::I32 inVocab, NlpAllPositionSampler::EncodeFn inEncode = nullptr) {
	NlpAllPositionSampler enc(nlpCorpus(), 1, inEncode);  // only encodePromptLeft used
	oa::Vector<oa::I32> context = enc.encodePromptLeft(inPrompt);
	const oa::I32 promptLen = static_cast<oa::I32>(std::strlen(inPrompt));
	oa::I32 filled   = std::min(promptLen, kContextLen);
	oa::I32 logitRow = std::max(0, filled - 1);
	oa::String out(inPrompt);

	for (oa::I32 i = 0; i < inCount; ++i) {
		auto x = oa::FnMatrix::fromInt32(oa::Span<const oa::I32>(context.data(), context.size()),
			oa::MatrixShape{1, kContextLen}, oa::ScalarType::UInt32);
		auto logits = inModel.forward(x);
		auto row = oa::FnMatrix::reshape(
			oa::FnMatrix::slice(logits, 0, logitRow, logitRow + 1),
			oa::MatrixShape{inVocab});
		const oa::I32 next = static_cast<oa::I32>(oa::FnMatrix::argmax(row));

		out += inEncode ? nlpCharDecode(next) : static_cast<char>(static_cast<oa::U8>(next));

		if (filled < kContextLen) {
			context[filled] = next;
			++filled;
			logitRow = filled - 1;
		} else {
			for (oa::I32 t = 1; t < kContextLen; ++t) context[t - 1] = context[t];
			context[kContextLen - 1] = next;
			logitRow = kContextLen - 1;
		}
	}
	return out;
}

// Stateful recurrent decoding for modules whose full training scan has a
// dedicated single-token step (currently Mamba-3). reset and prompt priming are
// synchronized deliberately: the recurrent state is mutated in place and each
// token must observe the previous token's completed state update.
template <class Model>
oa::String nlpGenerateStatefulGreedy(
	Model& inModel, const char* inPrompt, oa::I32 inCount, oa::I32 inVocab) {
	inModel.resetGenerationState(1);
	(void)tutorialSubmitAndWait(testEngine());

	oa::Matrix logits;
	const auto promptLength = static_cast<oa::I32>(std::strlen(inPrompt));
	for (oa::I32 index = 0; index < promptLength; ++index) {
		const oa::I32 token = static_cast<oa::U8>(inPrompt[index]);
		auto input = oa::FnMatrix::fromInt32(
			oa::Span<const oa::I32>(&token, 1),
			oa::MatrixShape{1, 1},
			oa::ScalarType::UInt32);
		logits = inModel.forwardGenerationStep(input);
		(void)tutorialSubmitAndWait(testEngine());
	}

	oa::String output(inPrompt);
	for (oa::I32 index = 0; index < inCount; ++index) {
			const oa::I32 next = static_cast<oa::I32>(oa::FnMatrix::argmax(
			logits.reshape(oa::MatrixShape{inVocab})));
		output += static_cast<char>(static_cast<oa::U8>(next));
		if (index + 1 < inCount) {
			auto input = oa::FnMatrix::fromInt32(
				oa::Span<const oa::I32>(&next, 1),
				oa::MatrixShape{1, 1},
				oa::ScalarType::UInt32);
			logits = inModel.forwardGenerationStep(input);
			(void)tutorialSubmitAndWait(testEngine());
		}
	}
	return output;
}

// ─── Shared generation: temperature / top-p sampling, byte vocab only ─────
// Same sliding-window logic as NlpGenerateGreedy, but uses oa::ByteEncoder::sample
// for non-deterministic output. Used by the byte-level Transformer / Mamba-3 /
// empyrealm tutorials.

template <class Model>
oa::String nlpGenerateSampled(Model& inModel, const char* inPrompt, oa::I32 inCount,
	oa::I32 inVocab, oa::F32 inTemperature = 0.8F, oa::F32 inTopP = 0.9F) {
	NlpAllPositionSampler enc(nlpCorpus(), 1);  // only encodePromptLeft used
	oa::Vector<oa::I32> context = enc.encodePromptLeft(inPrompt);
	const oa::I32 promptLen = static_cast<oa::I32>(std::strlen(inPrompt));
	oa::I32 filled = std::min(promptLen, kContextLen);
	oa::I32 logitRow = std::max(0, filled - 1);
	oa::String out(inPrompt);

	for (oa::I32 i = 0; i < inCount; ++i) {
		auto x = oa::FnMatrix::fromInt32(oa::Span<const oa::I32>(context.data(), context.size()),
			oa::MatrixShape{1, kContextLen}, oa::ScalarType::UInt32);
		auto logits = inModel.forward(x);
		auto rowLogits = oa::FnMatrix::reshape(
			oa::FnMatrix::slice(logits, 0, logitRow, logitRow + 1),
			oa::MatrixShape{1, inVocab});
		auto sampled = oa::ByteEncoder::sample(rowLogits, inTemperature, inTopP);
		oa::U8 next = sampled[0];
		out += static_cast<char>(next);

		if (filled < kContextLen) {
			context[filled] = next;
			++filled;
			logitRow = filled - 1;
		} else {
			for (oa::I32 t = 1; t < kContextLen; ++t) context[t - 1] = context[t];
			context[kContextLen - 1] = next;
			logitRow = kContextLen - 1;
		}
	}
	return out;
}
