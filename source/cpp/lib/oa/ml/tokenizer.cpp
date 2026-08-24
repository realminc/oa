// OA ML - tokenizer Implementation

#include <oa/ml/tokenizer.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>

namespace {
constexpr const char* kBpeMagic = "oa_bpe_v1";

oa::U64 pairKey(oa::U32 inLeft, oa::U32 inRight) {
	return (static_cast<oa::U64>(inLeft) << 32U) | static_cast<oa::U64>(inRight);
}
} // namespace

oa::BpeTokenizer::BpeTokenizer(oa::I32 inTargetVocab)
	: targetVocab_(std::max<oa::I32>(256, inTargetVocab)) {}

oa::Vec<oa::I32> oa::BpeTokenizer::applyMerge(
	const oa::Vec<oa::I32>& inIds, const Merge& inMerge, oa::I32 inNewToken) {
	oa::Vec<oa::I32> out;
	out.reserve(inIds.size());
	for (oa::Usize i = 0; i < inIds.size();) {
		if (i + 1 < inIds.size()
			and inIds[i] == static_cast<oa::I32>(inMerge.left)
			and inIds[i + 1] == static_cast<oa::I32>(inMerge.right)) {
			out.pushBack(inNewToken);
			i += 2;
		} else {
			out.pushBack(inIds[i++]);
		}
	}
	return out;
}

void oa::BpeTokenizer::train(const char* inText, oa::I32 inNumMerges) {
	merges_.clear();
	if (inText == nullptr or inNumMerges <= 0) return;
	oa::Vec<oa::I32> ids;
	const oa::Usize len = std::strlen(inText);
	ids.reserve(len);
	const auto* bytes = reinterpret_cast<const oa::U8*>(inText);
	for (oa::Usize i = 0; i < len; ++i) ids.pushBack(bytes[i]);

	const oa::I32 maxMerges = std::min(inNumMerges, targetVocab_ - 256);
	for (oa::I32 m = 0; m < maxMerges and ids.size() > 1; ++m) {
		// std::map gives a deterministic smallest-pair tie break.
		std::map<oa::U64, oa::I64> pairCounts;
		for (oa::Usize i = 0; i + 1 < ids.size(); ++i) {
			++pairCounts[pairKey(static_cast<oa::U32>(ids[i]), static_cast<oa::U32>(ids[i + 1]))];
		}

		oa::U64 bestPair = std::numeric_limits<oa::U64>::max();
		oa::I64 bestCount = 0;
		for (const auto& [pair, count] : pairCounts) {
			if (count > bestCount) {
				bestCount = count;
				bestPair = pair;
			}
		}
		if (bestCount < 2) break;

		const Merge merge{
			static_cast<oa::U32>(bestPair >> 32U), static_cast<oa::U32>(bestPair)};
		const oa::I32 newToken = 256 + static_cast<oa::I32>(merges_.size());
		merges_.pushBack(merge);
		ids = applyMerge(ids, merge, newToken);
	}
}

oa::Vec<oa::I32> oa::BpeTokenizer::encode(const char* inText) const {
	oa::Vec<oa::I32> tokens;
	const oa::I64 len = static_cast<oa::I64>(std::strlen(inText));
	tokens.reserve(len);

	// Start with raw bytes
	for (oa::I64 i = 0; i < len; ++i) {
		tokens.pushBack(static_cast<oa::I32>(static_cast<oa::U8>(inText[i])));
	}

	for (oa::Usize rank = 0; rank < merges_.size(); ++rank) {
		tokens = applyMerge(tokens, merges_[rank], 256 + static_cast<oa::I32>(rank));
	}

	return tokens;
}

void oa::BpeTokenizer::appendDecoded(oa::I32 inToken, oa::String& outText) const {
	if (inToken < 0) return;
	if (inToken < 256) {
		outText += static_cast<char>(static_cast<oa::U8>(inToken));
		return;
	}
	const oa::I32 rank = inToken - 256;
	if (rank < 0 or rank >= static_cast<oa::I32>(merges_.size())) return;
	appendDecoded(static_cast<oa::I32>(merges_[rank].left), outText);
	appendDecoded(static_cast<oa::I32>(merges_[rank].right), outText);
}

oa::String oa::BpeTokenizer::decode(const oa::Vec<oa::I32>& inTokens) const {
	oa::String out;
	for (oa::I32 token : inTokens) appendDecoded(token, out);
	return out;
}

oa::Status oa::BpeTokenizer::save(const oa::String& inPath) const {
	std::ofstream out(inPath.cStr(), std::ios::binary | std::ios::trunc);
	if (not out) return oa::Status::error(oa::StatusCode::PermissionError,
		oa::String("oa::BpeTokenizer: cannot write ") + inPath);
	out << kBpeMagic << '\n' << merges_.size() << '\n';
	for (const auto& merge : merges_) out << merge.left << ' ' << merge.right << '\n';
	if (not out) return oa::Status::error(oa::StatusCode::DiskFull,
		oa::String("oa::BpeTokenizer: failed writing ") + inPath);
	return oa::Status::ok();
}

oa::Status oa::BpeTokenizer::load(const oa::String& inPath) {
	std::ifstream in(inPath.cStr(), std::ios::binary);
	if (not in) return oa::Status::error(oa::StatusCode::FileNotFound,
		oa::String("oa::BpeTokenizer: cannot read ") + inPath);
	std::string magic;
	oa::U64 count = 0;
	if (not std::getline(in, magic) or magic != kBpeMagic or not (in >> count)
		or count > 1000000ULL) {
		return oa::Status::error(oa::StatusCode::FileCorrupt,
			oa::String("oa::BpeTokenizer: invalid header in ") + inPath);
	}
	oa::Vec<Merge> merges;
	merges.reserve(static_cast<oa::Usize>(count));
	for (oa::U64 i = 0; i < count; ++i) {
		oa::U64 left = 0, right = 0;
		const oa::U64 nextToken = 256ULL + i;
		if (not (in >> left >> right) or left >= nextToken or right >= nextToken
			or left > std::numeric_limits<oa::U32>::max()
			or right > std::numeric_limits<oa::U32>::max()) {
			return oa::Status::error(oa::StatusCode::FileCorrupt,
				oa::String("oa::BpeTokenizer: invalid merge in ") + inPath);
		}
		merges.pushBack({static_cast<oa::U32>(left), static_cast<oa::U32>(right)});
	}
	merges_ = std::move(merges);
	targetVocab_ = vocabSize();
	return oa::Status::ok();
}

oa::Vec<oa::I32> oa::BpeTokenizer::encodePrompt(const char* inPrompt, oa::I32 inContextLen) const {
	oa::Vec<oa::I32> tokens = encode(inPrompt);
	oa::Vec<oa::I32> out(inContextLen);
	for (oa::I32 i = 0; i < inContextLen; ++i) out[i] = 0;  // PAD
	// Right-align so the most recent token sits at index inContextLen-1, matching
	// the shift-left + append-at-end autoregressive loop used in the tutorials.
	const oa::I64 len = static_cast<oa::I64>(tokens.size());
	for (oa::I32 i = 0; i < inContextLen && i < len; ++i) {
		out[inContextLen - 1 - i] = tokens[len - 1 - i];
	}
	return out;
}
