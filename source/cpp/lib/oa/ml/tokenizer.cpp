// OA ML - tokenizer Implementation

#include <oa/ml/tokenizer.h>

#include <oa/core/filesystem.h>
#include <oa/core/std/format.h>
#include <oa/core/std/hashMap.h>

namespace {
constexpr const char* kBpeMagic = "oa_bpe_v1";

oa::U64 pairKey(oa::U32 inLeft, oa::U32 inRight) {
	return (static_cast<oa::U64>(inLeft) << 32U) | static_cast<oa::U64>(inRight);
}

bool readUnsigned(oa::StringView inText, oa::Usize& inOutCursor, oa::U64& outValue) {
	while (inOutCursor < inText.size()
		and (inText[inOutCursor] == ' ' or inText[inOutCursor] == '\t'
			or inText[inOutCursor] == '\r' or inText[inOutCursor] == '\n')) {
		++inOutCursor;
	}
	if (inOutCursor == inText.size() or inText[inOutCursor] < '0'
		or inText[inOutCursor] > '9') return false;
	oa::U64 value = 0;
	while (inOutCursor < inText.size() and inText[inOutCursor] >= '0'
		and inText[inOutCursor] <= '9') {
		const oa::U64 digit = static_cast<oa::U64>(inText[inOutCursor] - '0');
		if (value > (oa::Limits<oa::U64>::max() - digit) / 10U) return false;
		value = value * 10U + digit;
		++inOutCursor;
	}
	outValue = value;
	return true;
}
} // namespace

oa::BpeTokenizer::BpeTokenizer(oa::I32 inTargetVocab)
	: targetVocab_(oa::max<oa::I32>(256, inTargetVocab)) {}

oa::Vector<oa::I32> oa::BpeTokenizer::applyMerge(
	const oa::Vector<oa::I32>& inIds, const Merge& inMerge, oa::I32 inNewToken) {
	oa::Vector<oa::I32> out;
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
	oa::Vector<oa::I32> ids;
	const oa::Usize len = oa::strlen(inText);
	ids.reserve(len);
	const auto* bytes = reinterpret_cast<const oa::U8*>(inText);
	for (oa::Usize i = 0; i < len; ++i) ids.pushBack(bytes[i]);

	const oa::I32 maxMerges = oa::min(inNumMerges, targetVocab_ - 256);
	for (oa::I32 m = 0; m < maxMerges and ids.size() > 1; ++m) {
		oa::HashMap<oa::U64, oa::I64> pairCounts;
		pairCounts.reserve(ids.size() - 1U);
		for (oa::Usize i = 0; i + 1 < ids.size(); ++i) {
			const oa::U64 key = pairKey(
				static_cast<oa::U32>(ids[i]), static_cast<oa::U32>(ids[i + 1]));
			auto found = pairCounts.find(key);
			if (found == pairCounts.end()) pairCounts.emplace(key, 1);
			else ++found->second;
		}

		oa::U64 bestPair = oa::Limits<oa::U64>::max();
		oa::I64 bestCount = 0;
		for (const auto& [pair, count] : pairCounts) {
			if (count > bestCount or (count == bestCount and pair < bestPair)) {
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

oa::Vector<oa::I32> oa::BpeTokenizer::encode(const char* inText) const {
	oa::Vector<oa::I32> tokens;
	const oa::I64 len = static_cast<oa::I64>(oa::strlen(inText));
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

oa::String oa::BpeTokenizer::decode(const oa::Vector<oa::I32>& inTokens) const {
	oa::String out;
	for (oa::I32 token : inTokens) appendDecoded(token, out);
	return out;
}

oa::Status oa::BpeTokenizer::save(const oa::String& inPath) const {
	oa::String text;
	text.append(kBpeMagic);
	text.pushBack('\n');
	text.append(oa::toString(static_cast<oa::U64>(merges_.size())).view());
	text.pushBack('\n');
	for (const auto& merge : merges_) {
		text.append(oa::toString(merge.left).view());
		text.pushBack(' ');
		text.append(oa::toString(merge.right).view());
		text.pushBack('\n');
	}
	return oa::Filesystem::writeText(oa::Path(inPath), text.view());
}

oa::Status oa::BpeTokenizer::load(const oa::String& inPath) {
	auto textResult = oa::Filesystem::readText(oa::Path(inPath));
	if (not textResult.isOk()) return textResult.getStatus();
	const oa::StringView text = textResult.getValue().view();
	const oa::Usize firstNewline = text.find('\n');
	oa::U64 count = 0;
	oa::Usize cursor = firstNewline == oa::StringView::Npos
		? text.size() : firstNewline + 1U;
	if (firstNewline == oa::StringView::Npos
		or text.subStr(0, firstNewline) != kBpeMagic
		or not readUnsigned(text, cursor, count) or count > 1000000ULL) {
		return oa::Status::error(oa::StatusCode::FileCorrupt,
			oa::String("oa::BpeTokenizer: invalid header in ") + inPath);
	}
	oa::Vector<Merge> merges;
	merges.reserve(static_cast<oa::Usize>(count));
	for (oa::U64 i = 0; i < count; ++i) {
		oa::U64 left = 0, right = 0;
		const oa::U64 nextToken = 256ULL + i;
		if (not readUnsigned(text, cursor, left) or not readUnsigned(text, cursor, right)
			or left >= nextToken or right >= nextToken
			or left > oa::Limits<oa::U32>::max()
			or right > oa::Limits<oa::U32>::max()) {
			return oa::Status::error(oa::StatusCode::FileCorrupt,
				oa::String("oa::BpeTokenizer: invalid merge in ") + inPath);
		}
		merges.pushBack({static_cast<oa::U32>(left), static_cast<oa::U32>(right)});
	}
	merges_ = oa::move(merges);
	targetVocab_ = vocabSize();
	return oa::Status::ok();
}

oa::Vector<oa::I32> oa::BpeTokenizer::encodePrompt(const char* inPrompt, oa::I32 inContextLen) const {
	oa::Vector<oa::I32> tokens = encode(inPrompt);
	oa::Vector<oa::I32> out(inContextLen);
	for (oa::I32 i = 0; i < inContextLen; ++i) out[i] = 0;  // PAD
	// Right-align so the most recent token sits at index inContextLen-1, matching
	// the shift-left + append-at-end autoregressive loop used in the tutorials.
	const oa::I64 len = static_cast<oa::I64>(tokens.size());
	for (oa::I32 i = 0; i < inContextLen && i < len; ++i) {
		out[inContextLen - 1 - i] = tokens[len - 1 - i];
	}
	return out;
}
