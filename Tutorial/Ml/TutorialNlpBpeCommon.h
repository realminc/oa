// Shared byte-pair tokenizer and batching helpers for the TutorialNlpBpe* suite.
// This is intentionally a small, native, dependency-free byte-level BPE: all
// 256 bytes remain representable and learned tokens are deterministic pair merges.

#pragma once

#include "TutorialNlpCommon.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <vector>

inline constexpr OaI32 kBpeVocabSize = 320; // 256 bytes + 64 learned merges

class NlpBpeTokenizer {
public:
	struct Merge { OaU32 Left = 0; OaU32 Right = 0; };

	explicit NlpBpeTokenizer(const char* InTrainingText, OaI32 InVocabSize = kBpeVocabSize)
		: VocabSize_(std::max<OaI32>(InVocabSize, 256)) {
		Vocab_.resize(static_cast<size_t>(VocabSize_));
		for (OaU32 i = 0; i < 256; ++i) Vocab_[i].push_back(static_cast<OaU8>(i));

		std::vector<OaU32> ids;
		const auto* bytes = reinterpret_cast<const OaU8*>(InTrainingText);
		for (size_t i = 0; i < std::strlen(InTrainingText); ++i) ids.push_back(bytes[i]);

		for (OaU32 token = 256; token < static_cast<OaU32>(VocabSize_) && ids.size() > 1; ++token) {
			std::unordered_map<std::uint64_t, OaI64> counts;
			for (size_t i = 0; i + 1 < ids.size(); ++i) ++counts[PairKey(ids[i], ids[i + 1])];
			std::uint64_t bestKey = std::numeric_limits<std::uint64_t>::max();
			OaI64 bestCount = -1;
			for (const auto& [key, count] : counts) {
				if (count > bestCount || (count == bestCount && key < bestKey)) {
					bestKey = key;
					bestCount = count;
				}
			}
			const Merge merge{static_cast<OaU32>(bestKey >> 32), static_cast<OaU32>(bestKey)};
			Merges_.push_back(merge);
			ids = ApplyMerge(ids, merge, token);
			Vocab_[token] = Vocab_[merge.Left];
			Vocab_[token].insert(Vocab_[token].end(), Vocab_[merge.Right].begin(), Vocab_[merge.Right].end());
		}
		VocabSize_ = 256 + static_cast<OaI32>(Merges_.size());
		Vocab_.resize(static_cast<size_t>(VocabSize_));
	}

	[[nodiscard]] OaI32 VocabSize() const { return VocabSize_; }
	[[nodiscard]] OaI32 MergeCount() const { return static_cast<OaI32>(Merges_.size()); }

	[[nodiscard]] std::vector<OaU32> Encode(const char* InText) const {
		std::vector<OaU32> ids;
		const auto* bytes = reinterpret_cast<const OaU8*>(InText);
		for (size_t i = 0; i < std::strlen(InText); ++i) ids.push_back(bytes[i]);
		for (size_t rank = 0; rank < Merges_.size(); ++rank) {
			ids = ApplyMerge(ids, Merges_[rank], static_cast<OaU32>(256 + rank));
		}
		return ids;
	}

	[[nodiscard]] OaString Decode(const std::vector<OaU32>& InIds) const {
		OaString out;
		for (const OaU32 id : InIds) {
			if (id >= Vocab_.size()) continue;
			for (const OaU8 byte : Vocab_[id]) out += static_cast<char>(byte);
		}
		return out;
	}

	[[nodiscard]] OaI64 TokenBytes(OaU32 InToken) const {
		return InToken < Vocab_.size() ? static_cast<OaI64>(Vocab_[InToken].size()) : 0;
	}

private:
	static std::uint64_t PairKey(OaU32 InLeft, OaU32 InRight) {
		return (static_cast<std::uint64_t>(InLeft) << 32) | InRight;
	}

	static std::vector<OaU32> ApplyMerge(const std::vector<OaU32>& InIds,
		const Merge& InMerge, OaU32 InToken) {
		std::vector<OaU32> out;
		out.reserve(InIds.size());
		for (size_t i = 0; i < InIds.size();) {
			if (i + 1 < InIds.size() && InIds[i] == InMerge.Left && InIds[i + 1] == InMerge.Right) {
				out.push_back(InToken);
				i += 2;
			} else {
				out.push_back(InIds[i++]);
			}
		}
		return out;
	}

	OaI32 VocabSize_ = 256;
	std::vector<Merge> Merges_;
	std::vector<std::vector<OaU8>> Vocab_;
};

class NlpBpeAllPositionSampler {
public:
	NlpBpeAllPositionSampler(const char* InText, OaI32 InBatchSize, const NlpBpeTokenizer& InTokenizer)
		: BatchSize_(InBatchSize), Tokenizer_(InTokenizer), Tokens_(InTokenizer.Encode(InText)) {}

	void NextBatch(OaMatrix& OutX, OaMatrix& OutY) {
		std::vector<OaU32> x(static_cast<size_t>(BatchSize_) * kContextLen);
		std::vector<OaU32> y(static_cast<size_t>(BatchSize_) * kContextLen);
		const OaI64 limit = static_cast<OaI64>(Tokens_.size()) - kContextLen - 1;
		LastBatchBytes_ = 0;
		for (OaI32 b = 0; b < BatchSize_; ++b) {
			const OaI64 start = (Cursor_ + static_cast<OaI64>(b) * 7) % limit;
			for (OaI32 t = 0; t < kContextLen; ++t) {
				const size_t dst = static_cast<size_t>(b) * kContextLen + t;
				x[dst] = Tokens_[static_cast<size_t>(start + t)];
				y[dst] = Tokens_[static_cast<size_t>(start + t + 1)];
				LastBatchBytes_ += Tokenizer_.TokenBytes(y[dst]);
			}
		}
		Cursor_ = (Cursor_ + BatchSize_) % limit;
		OutX = FromU32(x);
		OutY = FromU32(y);
	}

	[[nodiscard]] OaI64 LastBatchBytes() const { return LastBatchBytes_; }
	[[nodiscard]] OaF64 LastBatchBytesPerToken() const {
		const OaI64 positions = static_cast<OaI64>(BatchSize_) * kContextLen;
		return positions > 0 ? static_cast<OaF64>(LastBatchBytes_) / static_cast<OaF64>(positions) : 0.0;
	}

private:
	OaMatrix FromU32(const std::vector<OaU32>& InIds) const {
		const auto* ptr = reinterpret_cast<const OaU8*>(InIds.data());
		return OaFnMatrix::FromBytes(OaSpan<const OaU8>(ptr, InIds.size() * sizeof(OaU32)),
			OaMatrixShape{BatchSize_, kContextLen}, OaScalarType::UInt32);
	}

	OaI32 BatchSize_ = 1;
	const NlpBpeTokenizer& Tokenizer_;
	std::vector<OaU32> Tokens_;
	OaI64 Cursor_ = 0;
	OaI64 LastBatchBytes_ = 0;
};

template <class Model>
OaString NlpGenerateBpeGreedy(Model& InModel, const NlpBpeTokenizer& InTokenizer,
	const char* InPrompt, OaI32 InByteCount) {
	auto prompt = InTokenizer.Encode(InPrompt);
	std::vector<OaU32> context(kContextLen, 0);
	const OaI32 copyCount = std::min<OaI32>(static_cast<OaI32>(prompt.size()), kContextLen);
	for (OaI32 i = 0; i < copyCount; ++i) context[static_cast<size_t>(i)] = prompt[static_cast<size_t>(i)];
	OaI32 filled = std::max(copyCount, 1);
	OaI32 logitRow = filled - 1;
	std::vector<OaU32> generated = prompt;
	OaI64 generatedBytes = 0;
	auto& ctx = OaContext::GetDefault();

	// Byte/char/BPE samples request the same amount of source text. BPE tokens
	// span a variable number of bytes, so stop on decoded bytes rather than tokens.
	for (OaI32 i = 0; i < InByteCount and generatedBytes < InByteCount; ++i) {
		const auto* ptr = reinterpret_cast<const OaU8*>(context.data());
		auto x = OaFnMatrix::FromBytes(OaSpan<const OaU8>(ptr, context.size() * sizeof(OaU32)),
			OaMatrixShape{1, kContextLen}, OaScalarType::UInt32);
		auto logits = InModel.Forward(x);
		auto row = OaFnMatrix::Reshape(
			OaFnMatrix::Slice(logits, 0, logitRow, logitRow + 1),
			OaMatrixShape{InTokenizer.VocabSize()});
		const OaU32 next = static_cast<OaU32>(OaFnMatrix::Argmax(row));
		generated.push_back(next);
		generatedBytes += InTokenizer.TokenBytes(next);
		if (filled < kContextLen) {
			context[static_cast<size_t>(filled++)] = next;
			logitRow = filled - 1;
		} else {
			for (OaI32 t = 1; t < kContextLen; ++t) {
				context[static_cast<size_t>(t - 1)] = context[static_cast<size_t>(t)];
			}
			context.back() = next;
			logitRow = kContextLen - 1;
		}
	}
	const OaString decoded = InTokenizer.Decode(generated);
	const OaUsize targetBytes = std::strlen(InPrompt) + static_cast<OaUsize>(InByteCount);
	return decoded.size() > targetBytes ? decoded.substr(0, targetBytes) : decoded;
}
