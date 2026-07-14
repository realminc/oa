#include <Ml/Nn/Alm/ClipTokenizer.h>

#include <utf8proc.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using Pair = std::pair<std::string, std::string>;
struct PairHash {
	size_t operator()(const Pair& p) const noexcept {
		return std::hash<std::string>{}(p.first) ^ (std::hash<std::string>{}(p.second) << 1U);
	}
};

void AppendUtf8(std::string& out, OaI32 cp) {
	if (cp <= 0x7F) out.push_back(static_cast<char>(cp));
	else if (cp <= 0x7FF) {
		out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
		out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
	} else if (cp <= 0xFFFF) {
		out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
		out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
		out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
	} else {
		out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
		out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
		out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
		out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
	}
}

std::vector<OaI32> ByteOrder() {
	std::vector<OaI32> bytes;
	for (OaI32 c = '!'; c <= '~'; ++c) bytes.push_back(c);
	for (OaI32 c = 0xA1; c <= 0xAC; ++c) bytes.push_back(c);
	for (OaI32 c = 0xAE; c <= 0xFF; ++c) bytes.push_back(c);
	for (OaI32 b = 0; b < 256; ++b)
		if (std::find(bytes.begin(), bytes.end(), b) == bytes.end()) bytes.push_back(b);
	return bytes;
}

std::vector<std::string> ByteEncoder() {
	auto bytes = ByteOrder();
	std::vector<OaI32> chars = bytes;
	for (OaI32 i = 188; i < 256; ++i) chars[static_cast<size_t>(i)] = 256 + (i - 188);
	std::vector<std::string> result(256);
	for (size_t i = 0; i < bytes.size(); ++i) AppendUtf8(result[static_cast<size_t>(bytes[i])], chars[i]);
	return result;
}

bool IsWhitespace(utf8proc_int32_t cp) {
	const auto cat = utf8proc_category(cp);
	return cat == UTF8PROC_CATEGORY_ZS or cat == UTF8PROC_CATEGORY_ZL or cat == UTF8PROC_CATEGORY_ZP or
		cp == '\t' or cp == '\n' or cp == '\r' or cp == '\f' or cp == '\v';
}
bool IsLetter(utf8proc_int32_t cp) {
	const auto cat = utf8proc_category(cp);
	return cat >= UTF8PROC_CATEGORY_LU and cat <= UTF8PROC_CATEGORY_LO;
}
bool IsNumber(utf8proc_int32_t cp) {
	const auto cat = utf8proc_category(cp);
	return cat >= UTF8PROC_CATEGORY_ND and cat <= UTF8PROC_CATEGORY_NO;
}

struct Unit { std::string Bytes; utf8proc_int32_t Cp = 0; };

OaResult<std::vector<Unit>> NormalizeUnits(OaStringView text) {
	utf8proc_uint8_t* normalized = nullptr;
	const auto length = utf8proc_map(reinterpret_cast<const utf8proc_uint8_t*>(text.data()),
		static_cast<utf8proc_ssize_t>(text.size()), &normalized,
		static_cast<utf8proc_option_t>(UTF8PROC_STABLE | UTF8PROC_COMPOSE));
	if (length < 0 or normalized == nullptr) return OaStatus::InvalidArgument("CLIP prompt is not valid UTF-8");
	std::vector<Unit> result;
	for (utf8proc_ssize_t offset = 0; offset < length;) {
		utf8proc_int32_t cp = 0;
		const auto used = utf8proc_iterate(normalized + offset, length - offset, &cp);
		if (used <= 0) { std::free(normalized); return OaStatus::InvalidArgument("invalid normalized UTF-8"); }
		cp = utf8proc_tolower(cp);
		std::string bytes; AppendUtf8(bytes, cp);
		result.push_back({std::move(bytes), cp});
		offset += used;
	}
	std::free(normalized);
	return result;
}

std::vector<std::string> Pretokenize(const std::vector<Unit>& units) {
	std::vector<std::string> tokens;
	for (size_t i = 0; i < units.size();) {
		if (IsWhitespace(units[i].Cp)) { ++i; continue; }
		if (units[i].Cp == '\'' and i + 1 < units.size()) {
			static const char* suffixes[] = {"s", "t", "re", "ve", "m", "ll", "d"};
			bool matched = false;
			for (const char* suffix : suffixes) {
				const std::string candidate = std::string("'") + suffix;
				std::string actual;
				for (size_t j = i; j < units.size() and actual.size() < candidate.size(); ++j) actual += units[j].Bytes;
				if (actual == candidate) {
					tokens.push_back(candidate); i += 1 + std::strlen(suffix); matched = true; break;
				}
			}
			if (matched) continue;
		}
		const bool letter = IsLetter(units[i].Cp);
		const bool number = IsNumber(units[i].Cp);
		std::string token = units[i++].Bytes;
		if (letter) while (i < units.size() and IsLetter(units[i].Cp)) token += units[i++].Bytes;
		else if (not number) while (i < units.size() and not IsWhitespace(units[i].Cp) and
			not IsLetter(units[i].Cp) and not IsNumber(units[i].Cp)) token += units[i++].Bytes;
		tokens.push_back(std::move(token));
	}
	return tokens;
}

} // namespace

class OaClipTokenizer::Impl {
public:
	std::vector<std::string> ByteMap;
	std::unordered_map<Pair, OaI32, PairHash> Ranks;
	std::unordered_map<std::string, OaI32> Encoder;
	mutable std::unordered_map<std::string, std::vector<std::string>> Cache;
	OaI32 Bos = -1;
	OaI32 Eos = -1;

	std::vector<std::string> Bpe(const std::string& token) const {
		if (auto it = Cache.find(token); it != Cache.end()) return it->second;
		std::vector<std::string> word;
		for (size_t i = 0; i < token.size();) {
			const unsigned char lead = static_cast<unsigned char>(token[i]);
			const size_t n = lead < 0x80 ? 1 : (lead < 0xE0 ? 2 : (lead < 0xF0 ? 3 : 4));
			word.push_back(token.substr(i, n)); i += n;
		}
		if (word.empty()) return {};
		word.back() += "</w>";
		while (word.size() > 1) {
			OaI32 bestRank = std::numeric_limits<OaI32>::max(); Pair best;
			for (size_t i = 0; i + 1 < word.size(); ++i) {
				const Pair pair{word[i], word[i + 1]};
				if (auto it = Ranks.find(pair); it != Ranks.end() and it->second < bestRank) { bestRank = it->second; best = pair; }
			}
			if (bestRank == std::numeric_limits<OaI32>::max()) break;
			std::vector<std::string> merged;
			for (size_t i = 0; i < word.size();) {
				if (i + 1 < word.size() and word[i] == best.first and word[i + 1] == best.second) {
					merged.push_back(word[i] + word[i + 1]); i += 2;
				} else merged.push_back(word[i++]);
			}
			word = std::move(merged);
		}
		Cache.emplace(token, word);
		return word;
	}
};

OaClipTokenizer::OaClipTokenizer() : Impl_(OaMakeUniquePtr<Impl>()) {}
OaClipTokenizer::~OaClipTokenizer() = default;
OaClipTokenizer::OaClipTokenizer(OaClipTokenizer&&) noexcept = default;
OaClipTokenizer& OaClipTokenizer::operator=(OaClipTokenizer&&) noexcept = default;

OaStatus OaClipTokenizer::LoadMerges(const OaPath& path) {
	std::ifstream file(path.CStr(), std::ios::binary);
	if (not file) return OaStatus::NotFound(OaString("cannot open CLIP merges: ") + path.String());
	std::string bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
	return LoadMerges(OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(bytes.data()), bytes.size()));
}

OaStatus OaClipTokenizer::LoadMerges(OaSpan<const OaU8> bytes) {
	Impl_ = OaMakeUniquePtr<Impl>(); Impl_->ByteMap = ByteEncoder();
	std::vector<std::string> vocab; vocab.reserve(49408);
	for (const OaI32 byte : ByteOrder()) vocab.push_back(Impl_->ByteMap[static_cast<size_t>(byte)]);
	for (const OaI32 byte : ByteOrder()) vocab.push_back(Impl_->ByteMap[static_cast<size_t>(byte)] + "</w>");
	std::istringstream lines(std::string(reinterpret_cast<const char*>(bytes.Data()), bytes.Size()));
	std::string line; std::getline(lines, line);
	OaI32 rank = 0;
	while (rank < 48894 and std::getline(lines, line)) {
		if (line.empty()) continue;
		const auto space = line.find(' ');
		if (space == std::string::npos or space == 0 or space + 1 >= line.size()) return OaStatus::InvalidArgument("malformed CLIP merge entry");
		Pair pair{line.substr(0, space), line.substr(space + 1)};
		Impl_->Ranks.emplace(pair, rank++); vocab.push_back(pair.first + pair.second);
	}
	if (rank != 48894) return OaStatus::InvalidArgument("CLIP merges must contain 48,894 entries");
	vocab.push_back("<|startoftext|>"); vocab.push_back("<|endoftext|>");
	for (OaI32 i = 0; i < static_cast<OaI32>(vocab.size()); ++i) Impl_->Encoder.emplace(vocab[static_cast<size_t>(i)], i);
	Impl_->Bos = Impl_->Encoder.at("<|startoftext|>"); Impl_->Eos = Impl_->Encoder.at("<|endoftext|>");
	return vocab.size() == 49408 ? OaStatus::Ok() : OaStatus::InvalidArgument("CLIP vocabulary size mismatch");
}

OaResult<OaClipTokenBatch> OaClipTokenizer::Encode(OaSpan<const OaString> prompts, OaI32 context, bool truncate) const {
	if (not IsLoaded()) return OaStatus::Error(OaStatusCode::FailedPrecondition, "CLIP tokenizer is not loaded");
	if (prompts.Empty() or context < 2) return OaStatus::InvalidArgument("invalid CLIP token batch shape");
	OaClipTokenBatch out; out.Batch = static_cast<OaI32>(prompts.Size()); out.ContextLength = context;
	out.TokenIds.Resize(static_cast<OaUsize>(out.Batch) * context, Impl_->Eos); out.FlatEosRows.Resize(out.Batch);
	for (OaI32 b = 0; b < out.Batch; ++b) {
		OaVec<OaI32> ids{Impl_->Bos};
		auto units = NormalizeUnits(prompts[static_cast<OaUsize>(b)]); if (units.IsError()) return units.GetStatus();
		for (const auto& token : Pretokenize(units.GetValue())) {
			std::string encoded; for (const unsigned char byte : token) encoded += Impl_->ByteMap[byte];
			for (const auto& piece : Impl_->Bpe(encoded)) {
				auto it = Impl_->Encoder.find(piece); if (it == Impl_->Encoder.end()) return OaStatus::Error("CLIP BPE emitted an unknown piece");
				ids.PushBack(it->second);
			}
		}
		ids.PushBack(Impl_->Eos);
		if (static_cast<OaI32>(ids.Size()) > context) {
			if (not truncate) return OaStatus::Error(OaStatusCode::OutOfRange, "CLIP prompt exceeds context length");
			ids.Resize(context); ids[static_cast<OaUsize>(context - 1)] = Impl_->Eos;
		}
		const OaUsize row = static_cast<OaUsize>(b) * context;
		for (OaUsize i = 0; i < ids.Size(); ++i) out.TokenIds[row + i] = ids[i];
		out.FlatEosRows[static_cast<OaUsize>(b)] = b * context + static_cast<OaI32>(ids.Size()) - 1;
	}
	return out;
}

bool OaClipTokenizer::IsLoaded() const noexcept { return Impl_ and Impl_->Bos >= 0 and Impl_->Eos >= 0; }
OaI32 OaClipTokenizer::VocabSize() const noexcept { return IsLoaded() ? 49408 : 0; }
OaI32 OaClipTokenizer::BosToken() const noexcept { return Impl_ ? Impl_->Bos : -1; }
OaI32 OaClipTokenizer::EosToken() const noexcept { return Impl_ ? Impl_->Eos : -1; }
