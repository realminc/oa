// OA ML - Tokenizer Implementation

#include <Oa/Ml/Tokenizer.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>

namespace {
constexpr const char* kBpeMagic = "oa_bpe_v1";

OaU64 PairKey(OaU32 InLeft, OaU32 InRight) {
	return (static_cast<OaU64>(InLeft) << 32U) | static_cast<OaU64>(InRight);
}
} // namespace

OaBpeTokenizer::OaBpeTokenizer(OaI32 InTargetVocab)
	: TargetVocab_(std::max<OaI32>(256, InTargetVocab)) {}

OaVec<OaI32> OaBpeTokenizer::ApplyMerge(
	const OaVec<OaI32>& InIds, const Merge& InMerge, OaI32 InNewToken) {
	OaVec<OaI32> out;
	out.Reserve(InIds.Size());
	for (OaUsize i = 0; i < InIds.Size();) {
		if (i + 1 < InIds.Size()
			and InIds[i] == static_cast<OaI32>(InMerge.Left)
			and InIds[i + 1] == static_cast<OaI32>(InMerge.Right)) {
			out.PushBack(InNewToken);
			i += 2;
		} else {
			out.PushBack(InIds[i++]);
		}
	}
	return out;
}

void OaBpeTokenizer::Train(const char* InText, OaI32 InNumMerges) {
	Merges_.Clear();
	if (InText == nullptr or InNumMerges <= 0) return;
	OaVec<OaI32> ids;
	const OaUsize len = std::strlen(InText);
	ids.Reserve(len);
	const auto* bytes = reinterpret_cast<const OaU8*>(InText);
	for (OaUsize i = 0; i < len; ++i) ids.PushBack(bytes[i]);

	const OaI32 maxMerges = std::min(InNumMerges, TargetVocab_ - 256);
	for (OaI32 m = 0; m < maxMerges and ids.Size() > 1; ++m) {
		// std::map gives a deterministic smallest-pair tie break.
		std::map<OaU64, OaI64> pairCounts;
		for (OaUsize i = 0; i + 1 < ids.Size(); ++i) {
			++pairCounts[PairKey(static_cast<OaU32>(ids[i]), static_cast<OaU32>(ids[i + 1]))];
		}

		OaU64 bestPair = std::numeric_limits<OaU64>::max();
		OaI64 bestCount = 0;
		for (const auto& [pair, count] : pairCounts) {
			if (count > bestCount) {
				bestCount = count;
				bestPair = pair;
			}
		}
		if (bestCount < 2) break;

		const Merge merge{
			static_cast<OaU32>(bestPair >> 32U), static_cast<OaU32>(bestPair)};
		const OaI32 newToken = 256 + static_cast<OaI32>(Merges_.Size());
		Merges_.PushBack(merge);
		ids = ApplyMerge(ids, merge, newToken);
	}
}

OaVec<OaI32> OaBpeTokenizer::Encode(const char* InText) const {
	OaVec<OaI32> tokens;
	const OaI64 len = static_cast<OaI64>(std::strlen(InText));
	tokens.Reserve(len);

	// Start with raw bytes
	for (OaI64 i = 0; i < len; ++i) {
		tokens.PushBack(static_cast<OaI32>(static_cast<OaU8>(InText[i])));
	}

	for (OaUsize rank = 0; rank < Merges_.Size(); ++rank) {
		tokens = ApplyMerge(tokens, Merges_[rank], 256 + static_cast<OaI32>(rank));
	}

	return tokens;
}

void OaBpeTokenizer::AppendDecoded(OaI32 InToken, OaString& OutText) const {
	if (InToken < 0) return;
	if (InToken < 256) {
		OutText += static_cast<char>(static_cast<OaU8>(InToken));
		return;
	}
	const OaI32 rank = InToken - 256;
	if (rank < 0 or rank >= static_cast<OaI32>(Merges_.Size())) return;
	AppendDecoded(static_cast<OaI32>(Merges_[rank].Left), OutText);
	AppendDecoded(static_cast<OaI32>(Merges_[rank].Right), OutText);
}

OaString OaBpeTokenizer::Decode(const OaVec<OaI32>& InTokens) const {
	OaString out;
	for (OaI32 token : InTokens) AppendDecoded(token, out);
	return out;
}

OaStatus OaBpeTokenizer::Save(const OaString& InPath) const {
	std::ofstream out(InPath.CStr(), std::ios::binary | std::ios::trunc);
	if (not out) return OaStatus::Error(OaStatusCode::PermissionError,
		OaString("OaBpeTokenizer: cannot write ") + InPath);
	out << kBpeMagic << '\n' << Merges_.Size() << '\n';
	for (const auto& merge : Merges_) out << merge.Left << ' ' << merge.Right << '\n';
	if (not out) return OaStatus::Error(OaStatusCode::DiskFull,
		OaString("OaBpeTokenizer: failed writing ") + InPath);
	return OaStatus::Ok();
}

OaStatus OaBpeTokenizer::Load(const OaString& InPath) {
	std::ifstream in(InPath.CStr(), std::ios::binary);
	if (not in) return OaStatus::Error(OaStatusCode::FileNotFound,
		OaString("OaBpeTokenizer: cannot read ") + InPath);
	std::string magic;
	OaU64 count = 0;
	if (not std::getline(in, magic) or magic != kBpeMagic or not (in >> count)
		or count > 1000000ULL) {
		return OaStatus::Error(OaStatusCode::FileCorrupt,
			OaString("OaBpeTokenizer: invalid header in ") + InPath);
	}
	OaVec<Merge> merges;
	merges.Reserve(static_cast<OaUsize>(count));
	for (OaU64 i = 0; i < count; ++i) {
		OaU64 left = 0, right = 0;
		const OaU64 nextToken = 256ULL + i;
		if (not (in >> left >> right) or left >= nextToken or right >= nextToken
			or left > std::numeric_limits<OaU32>::max()
			or right > std::numeric_limits<OaU32>::max()) {
			return OaStatus::Error(OaStatusCode::FileCorrupt,
				OaString("OaBpeTokenizer: invalid merge in ") + InPath);
		}
		merges.PushBack({static_cast<OaU32>(left), static_cast<OaU32>(right)});
	}
	Merges_ = std::move(merges);
	TargetVocab_ = VocabSize();
	return OaStatus::Ok();
}

OaVec<OaI32> OaBpeTokenizer::EncodePrompt(const char* InPrompt, OaI32 InContextLen) const {
	OaVec<OaI32> tokens = Encode(InPrompt);
	OaVec<OaI32> out(InContextLen);
	for (OaI32 i = 0; i < InContextLen; ++i) out[i] = 0;  // PAD
	// Right-align so the most recent token sits at index InContextLen-1, matching
	// the shift-left + append-at-end autoregressive loop used in the tutorials.
	const OaI64 len = static_cast<OaI64>(tokens.Size());
	for (OaI32 i = 0; i < InContextLen && i < len; ++i) {
		out[InContextLen - 1 - i] = tokens[len - 1 - i];
	}
	return out;
}
