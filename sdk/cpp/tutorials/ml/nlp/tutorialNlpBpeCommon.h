// Shared byte-pair tokenizer and batching helpers for the TutorialNlpBpe* suite.
// This is intentionally a small, native, dependency-free byte-level BPE: all
// 256 bytes remain representable and learned tokens are deterministic pair merges.

#pragma once

#include "tutorialNlpCommon.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <vector>

inline constexpr oa::I32 kBpeVocabSize = 320; // 256 bytes + 64 learned merges

class NlpBpeTokenizer {
public:
	struct Merge { oa::U32 left = 0; oa::U32 right = 0; };

	explicit NlpBpeTokenizer(const char* inTrainingText, oa::I32 inVocabSize = kBpeVocabSize)
		: vocabSize_(std::max<oa::I32>(inVocabSize, 256)) {
		vocab_.resize(static_cast<size_t>(vocabSize_));
		for (oa::U32 i = 0; i < 256; ++i) vocab_[i].push_back(static_cast<oa::U8>(i));

		std::vector<oa::U32> ids;
		const auto* bytes = reinterpret_cast<const oa::U8*>(inTrainingText);
		for (size_t i = 0; i < std::strlen(inTrainingText); ++i) ids.push_back(bytes[i]);

		for (oa::U32 token = 256; token < static_cast<oa::U32>(vocabSize_) && ids.size() > 1; ++token) {
			std::unordered_map<std::uint64_t, oa::I64> counts;
			for (size_t i = 0; i + 1 < ids.size(); ++i) ++counts[pairKey(ids[i], ids[i + 1])];
			std::uint64_t bestKey = std::numeric_limits<std::uint64_t>::max();
			oa::I64 bestCount = -1;
			for (const auto& [key, count] : counts) {
				if (count > bestCount || (count == bestCount && key < bestKey)) {
					bestKey = key;
					bestCount = count;
				}
			}
			const Merge merge{static_cast<oa::U32>(bestKey >> 32), static_cast<oa::U32>(bestKey)};
			merges_.push_back(merge);
			ids = applyMerge(ids, merge, token);
			vocab_[token] = vocab_[merge.left];
			vocab_[token].insert(vocab_[token].end(), vocab_[merge.right].begin(), vocab_[merge.right].end());
		}
		vocabSize_ = 256 + static_cast<oa::I32>(merges_.size());
		vocab_.resize(static_cast<size_t>(vocabSize_));
	}

	[[nodiscard]] oa::I32 vocabSize() const { return vocabSize_; }
	[[nodiscard]] oa::I32 mergeCount() const { return static_cast<oa::I32>(merges_.size()); }

	[[nodiscard]] std::vector<oa::U32> encode(const char* inText) const {
		std::vector<oa::U32> ids;
		const auto* bytes = reinterpret_cast<const oa::U8*>(inText);
		for (size_t i = 0; i < std::strlen(inText); ++i) ids.push_back(bytes[i]);
		for (size_t rank = 0; rank < merges_.size(); ++rank) {
			ids = applyMerge(ids, merges_[rank], static_cast<oa::U32>(256 + rank));
		}
		return ids;
	}

	[[nodiscard]] oa::String decode(const std::vector<oa::U32>& inIds) const {
		oa::String out;
		for (const oa::U32 id : inIds) {
			if (id >= vocab_.size()) continue;
			for (const oa::U8 byte : vocab_[id]) out += static_cast<char>(byte);
		}
		return out;
	}

	[[nodiscard]] oa::I64 tokenBytes(oa::U32 inToken) const {
		return inToken < vocab_.size() ? static_cast<oa::I64>(vocab_[inToken].size()) : 0;
	}

private:
	static std::uint64_t pairKey(oa::U32 inLeft, oa::U32 inRight) {
		return (static_cast<std::uint64_t>(inLeft) << 32) | inRight;
	}

	static std::vector<oa::U32> applyMerge(const std::vector<oa::U32>& inIds,
		const Merge& inMerge, oa::U32 inToken) {
		std::vector<oa::U32> out;
		out.reserve(inIds.size());
		for (size_t i = 0; i < inIds.size();) {
			if (i + 1 < inIds.size() && inIds[i] == inMerge.left && inIds[i + 1] == inMerge.right) {
				out.push_back(inToken);
				i += 2;
			} else {
				out.push_back(inIds[i++]);
			}
		}
		return out;
	}

	oa::I32 vocabSize_ = 256;
	std::vector<Merge> merges_;
	std::vector<std::vector<oa::U8>> vocab_;
};

class NlpBpeAllPositionSampler {
public:
	NlpBpeAllPositionSampler(const char* inText, oa::I32 inBatchSize, const NlpBpeTokenizer& inTokenizer)
		: batchSize_(inBatchSize), tokenizer_(inTokenizer), tokens_(inTokenizer.encode(inText)) {}

	void nextBatch(oa::Matrix& outX, oa::Matrix& outY) {
		std::vector<oa::U32> x(static_cast<size_t>(batchSize_) * kContextLen);
		std::vector<oa::U32> y(static_cast<size_t>(batchSize_) * kContextLen);
		const oa::I64 limit = static_cast<oa::I64>(tokens_.size()) - kContextLen - 1;
		lastBatchBytes_ = 0;
		for (oa::I32 b = 0; b < batchSize_; ++b) {
			const oa::I64 start = (cursor_ + static_cast<oa::I64>(b) * 7) % limit;
			for (oa::I32 t = 0; t < kContextLen; ++t) {
				const size_t dst = static_cast<size_t>(b) * kContextLen + t;
				x[dst] = tokens_[static_cast<size_t>(start + t)];
				y[dst] = tokens_[static_cast<size_t>(start + t + 1)];
				lastBatchBytes_ += tokenizer_.tokenBytes(y[dst]);
			}
		}
		cursor_ = (cursor_ + batchSize_) % limit;
		outX = fromU32(x);
		outY = fromU32(y);
	}

	[[nodiscard]] oa::I64 lastBatchBytes() const { return lastBatchBytes_; }
	[[nodiscard]] oa::F64 lastBatchBytesPerToken() const {
		const oa::I64 positions = static_cast<oa::I64>(batchSize_) * kContextLen;
		return positions > 0 ? static_cast<oa::F64>(lastBatchBytes_) / static_cast<oa::F64>(positions) : 0.0;
	}

private:
	oa::Matrix fromU32(const std::vector<oa::U32>& inIds) const {
		const auto* ptr = reinterpret_cast<const oa::U8*>(inIds.data());
		return oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(ptr, inIds.size() * sizeof(oa::U32)),
			oa::MatrixShape{batchSize_, kContextLen}, oa::ScalarType::UInt32);
	}

	oa::I32 batchSize_ = 1;
	const NlpBpeTokenizer& tokenizer_;
	std::vector<oa::U32> tokens_;
	oa::I64 cursor_ = 0;
	oa::I64 lastBatchBytes_ = 0;
};

template <class Model>
oa::String nlpGenerateBpeGreedy(Model& inModel, const NlpBpeTokenizer& inTokenizer,
	const char* inPrompt, oa::I32 inByteCount) {
	auto prompt = inTokenizer.encode(inPrompt);
	std::vector<oa::U32> context(kContextLen, 0);
	const oa::I32 copyCount = std::min<oa::I32>(static_cast<oa::I32>(prompt.size()), kContextLen);
	for (oa::I32 i = 0; i < copyCount; ++i) context[static_cast<size_t>(i)] = prompt[static_cast<size_t>(i)];
	oa::I32 filled = std::max(copyCount, 1);
	oa::I32 logitRow = filled - 1;
	std::vector<oa::U32> generated = prompt;
	oa::I64 generatedBytes = 0;

	// Byte/char/BPE samples request the same amount of source text. BPE tokens
	// span a variable number of bytes, so stop on decoded bytes rather than tokens.
	for (oa::I32 i = 0; i < inByteCount and generatedBytes < inByteCount; ++i) {
		const auto* ptr = reinterpret_cast<const oa::U8*>(context.data());
		auto x = oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(ptr, context.size() * sizeof(oa::U32)),
			oa::MatrixShape{1, kContextLen}, oa::ScalarType::UInt32);
		auto logits = inModel.forward(x);
		auto row = oa::FnMatrix::reshape(
			oa::FnMatrix::slice(logits, 0, logitRow, logitRow + 1),
			oa::MatrixShape{inTokenizer.vocabSize()});
		const oa::U32 next = static_cast<oa::U32>(oa::FnMatrix::argmax(row));
		generated.push_back(next);
		generatedBytes += inTokenizer.tokenBytes(next);
		if (filled < kContextLen) {
			context[static_cast<size_t>(filled++)] = next;
			logitRow = filled - 1;
		} else {
			for (oa::I32 t = 1; t < kContextLen; ++t) {
				context[static_cast<size_t>(t - 1)] = context[static_cast<size_t>(t)];
			}
			context.back() = next;
			logitRow = kContextLen - 1;
		}
	}
	const oa::String decoded = inTokenizer.decode(generated);
	const oa::Usize targetBytes = std::strlen(inPrompt) + static_cast<oa::Usize>(inByteCount);
	return decoded.size() > targetBytes ? decoded.substr(0, targetBytes) : decoded;
}
