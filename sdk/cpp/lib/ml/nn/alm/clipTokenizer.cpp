#include <ml/nn/alm/clipTokenizer.h>

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

void appendUtf8(std::string& out, oa::I32 cp) {
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

std::vector<oa::I32> byteOrder() {
	std::vector<oa::I32> bytes;
	for (oa::I32 c = '!'; c <= '~'; ++c) bytes.push_back(c);
	for (oa::I32 c = 0xA1; c <= 0xAC; ++c) bytes.push_back(c);
	for (oa::I32 c = 0xAE; c <= 0xFF; ++c) bytes.push_back(c);
	for (oa::I32 b = 0; b < 256; ++b)
		if (std::find(bytes.begin(), bytes.end(), b) == bytes.end()) bytes.push_back(b);
	return bytes;
}

std::vector<std::string> byteEncoder() {
	auto bytes = byteOrder();
	std::vector<oa::I32> chars = bytes;
	for (oa::I32 i = 188; i < 256; ++i) chars[static_cast<size_t>(i)] = 256 + (i - 188);
	std::vector<std::string> result(256);
	for (size_t i = 0; i < bytes.size(); ++i) appendUtf8(result[static_cast<size_t>(bytes[i])], chars[i]);
	return result;
}

bool isWhitespace(utf8proc_int32_t cp) {
	const auto cat = utf8proc_category(cp);
	return cat == UTF8PROC_CATEGORY_ZS or cat == UTF8PROC_CATEGORY_ZL or cat == UTF8PROC_CATEGORY_ZP or
		cp == '\t' or cp == '\n' or cp == '\r' or cp == '\f' or cp == '\v';
}
bool isLetter(utf8proc_int32_t cp) {
	const auto cat = utf8proc_category(cp);
	return cat >= UTF8PROC_CATEGORY_LU and cat <= UTF8PROC_CATEGORY_LO;
}
bool isNumber(utf8proc_int32_t cp) {
	const auto cat = utf8proc_category(cp);
	return cat >= UTF8PROC_CATEGORY_ND and cat <= UTF8PROC_CATEGORY_NO;
}

struct Unit { std::string bytes; utf8proc_int32_t cp = 0; };

oa::Result<std::vector<Unit>> normalizeUnits(oa::StringView text) {
	utf8proc_uint8_t* normalized = nullptr;
	const auto length = utf8proc_map(reinterpret_cast<const utf8proc_uint8_t*>(text.data()),
		static_cast<utf8proc_ssize_t>(text.size()), &normalized,
		static_cast<utf8proc_option_t>(UTF8PROC_STABLE | UTF8PROC_COMPOSE));
	if (length < 0 or normalized == nullptr) return oa::Status::invalidArgument("CLIP prompt is not valid UTF-8");
	std::vector<Unit> result;
	for (utf8proc_ssize_t offset = 0; offset < length;) {
		utf8proc_int32_t cp = 0;
		const auto used = utf8proc_iterate(normalized + offset, length - offset, &cp);
		if (used <= 0) { std::free(normalized); return oa::Status::invalidArgument("invalid normalized UTF-8"); }
		cp = utf8proc_tolower(cp);
		std::string bytes; appendUtf8(bytes, cp);
		result.push_back({std::move(bytes), cp});
		offset += used;
	}
	std::free(normalized);
	return result;
}

std::vector<std::string> pretokenize(const std::vector<Unit>& units) {
	std::vector<std::string> tokens;
	for (size_t i = 0; i < units.size();) {
		if (isWhitespace(units[i].cp)) { ++i; continue; }
		if (units[i].cp == '\'' and i + 1 < units.size()) {
			static const char* suffixes[] = {"s", "t", "re", "ve", "m", "ll", "d"};
			bool matched = false;
			for (const char* suffix : suffixes) {
				const std::string candidate = std::string("'") + suffix;
				std::string actual;
				for (size_t j = i; j < units.size() and actual.size() < candidate.size(); ++j) actual += units[j].bytes;
				if (actual == candidate) {
					tokens.push_back(candidate); i += 1 + std::strlen(suffix); matched = true; break;
				}
			}
			if (matched) continue;
		}
		const bool letter = isLetter(units[i].cp);
		const bool number = isNumber(units[i].cp);
		std::string token = units[i++].bytes;
		if (letter) while (i < units.size() and isLetter(units[i].cp)) token += units[i++].bytes;
		else if (not number) while (i < units.size() and not isWhitespace(units[i].cp) and
			not isLetter(units[i].cp) and not isNumber(units[i].cp)) token += units[i++].bytes;
		tokens.push_back(std::move(token));
	}
	return tokens;
}

} // namespace

class oa::ClipTokenizer::Impl {
public:
	std::vector<std::string> byteMap;
	std::unordered_map<::Pair, oa::I32, ::PairHash> ranks;
	std::unordered_map<std::string, oa::I32> encoder;
	mutable std::unordered_map<std::string, std::vector<std::string>> cache;
	oa::I32 bos = -1;
	oa::I32 eos = -1;

	std::vector<std::string> bpe(const std::string& token) const {
		if (auto it = cache.find(token); it != cache.end()) return it->second;
		std::vector<std::string> word;
		for (size_t i = 0; i < token.size();) {
			const unsigned char lead = static_cast<unsigned char>(token[i]);
			const size_t n = lead < 0x80 ? 1 : (lead < 0xE0 ? 2 : (lead < 0xF0 ? 3 : 4));
			word.push_back(token.substr(i, n)); i += n;
		}
		if (word.empty()) return {};
		word.back() += "</w>";
		while (word.size() > 1) {
			oa::I32 bestRank = std::numeric_limits<oa::I32>::max(); ::Pair best;
			for (size_t i = 0; i + 1 < word.size(); ++i) {
				const ::Pair pair{word[i], word[i + 1]};
				if (auto it = ranks.find(pair); it != ranks.end() and it->second < bestRank) { bestRank = it->second; best = pair; }
			}
			if (bestRank == std::numeric_limits<oa::I32>::max()) break;
			std::vector<std::string> merged;
			for (size_t i = 0; i < word.size();) {
				if (i + 1 < word.size() and word[i] == best.first and word[i + 1] == best.second) {
					merged.push_back(word[i] + word[i + 1]); i += 2;
				} else merged.push_back(word[i++]);
			}
			word = std::move(merged);
		}
		cache.emplace(token, word);
		return word;
	}
};

oa::ClipTokenizer::ClipTokenizer() : impl_(oa::makeUnique<Impl>()) {}
oa::ClipTokenizer::~ClipTokenizer() = default;
oa::ClipTokenizer::ClipTokenizer(oa::ClipTokenizer&&) noexcept = default;
oa::ClipTokenizer& oa::ClipTokenizer::operator=(oa::ClipTokenizer&&) noexcept = default;

oa::Status oa::ClipTokenizer::loadMerges(const oa::Path& path) {
	std::ifstream file(path.cStr(), std::ios::binary);
	if (not file) return oa::Status::notFound(oa::String("cannot open CLIP merges: ") + path.string());
	std::string bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
	return loadMerges(oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(bytes.data()), bytes.size()));
}

oa::Status oa::ClipTokenizer::loadMerges(oa::Span<const oa::U8> bytes) {
	impl_ = oa::makeUnique<Impl>(); impl_->byteMap = byteEncoder();
	std::vector<std::string> vocab; vocab.reserve(49408);
	for (const oa::I32 byte : byteOrder()) vocab.push_back(impl_->byteMap[static_cast<size_t>(byte)]);
	for (const oa::I32 byte : byteOrder()) vocab.push_back(impl_->byteMap[static_cast<size_t>(byte)] + "</w>");
	std::istringstream lines(std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size()));
	std::string line; std::getline(lines, line);
	oa::I32 rank = 0;
	while (rank < 48894 and std::getline(lines, line)) {
		if (line.empty()) continue;
		const auto space = line.find(' ');
		if (space == std::string::npos or space == 0 or space + 1 >= line.size()) return oa::Status::invalidArgument("malformed CLIP merge entry");
		::Pair pair{line.substr(0, space), line.substr(space + 1)};
		impl_->ranks.emplace(pair, rank++); vocab.push_back(pair.first + pair.second);
	}
	if (rank != 48894) return oa::Status::invalidArgument("CLIP merges must contain 48,894 entries");
	vocab.push_back("<|startoftext|>"); vocab.push_back("<|endoftext|>");
	for (oa::I32 i = 0; i < static_cast<oa::I32>(vocab.size()); ++i) impl_->encoder.emplace(vocab[static_cast<size_t>(i)], i);
	impl_->bos = impl_->encoder.at("<|startoftext|>"); impl_->eos = impl_->encoder.at("<|endoftext|>");
	return vocab.size() == 49408 ? oa::Status::ok() : oa::Status::invalidArgument("CLIP vocabulary size mismatch");
}

oa::Result<oa::ClipTokenBatch> oa::ClipTokenizer::encode(oa::Span<const oa::String> prompts, oa::I32 context, bool truncate) const {
	if (not isLoaded()) return oa::Status::error(oa::StatusCode::FailedPrecondition, "CLIP tokenizer is not loaded");
	if (prompts.empty() or context < 2) return oa::Status::invalidArgument("invalid CLIP token batch shape");
	oa::ClipTokenBatch out; out.batch = static_cast<oa::I32>(prompts.size()); out.contextLength = context;
	out.tokenIds.resize(static_cast<oa::Usize>(out.batch) * context, impl_->eos); out.flatEosRows.resize(out.batch);
	for (oa::I32 b = 0; b < out.batch; ++b) {
		oa::Vector<oa::I32> ids{impl_->bos};
		auto units = normalizeUnits(prompts[static_cast<oa::Usize>(b)]); if (units.isError()) return units.getStatus();
		for (const auto& token : pretokenize(units.getValue())) {
			std::string encoded; for (const unsigned char byte : token) encoded += impl_->byteMap[byte];
			for (const auto& piece : impl_->bpe(encoded)) {
				auto it = impl_->encoder.find(piece); if (it == impl_->encoder.end()) return oa::Status::error("CLIP BPE emitted an unknown piece");
				ids.pushBack(it->second);
			}
		}
		ids.pushBack(impl_->eos);
		if (static_cast<oa::I32>(ids.size()) > context) {
			if (not truncate) return oa::Status::error(oa::StatusCode::OutOfRange, "CLIP prompt exceeds context length");
			ids.resize(context); ids[static_cast<oa::Usize>(context - 1)] = impl_->eos;
		}
		const oa::Usize row = static_cast<oa::Usize>(b) * context;
		for (oa::Usize i = 0; i < ids.size(); ++i) out.tokenIds[row + i] = ids[i];
		out.flatEosRows[static_cast<oa::Usize>(b)] = b * context + static_cast<oa::I32>(ids.size()) - 1;
	}
	return out;
}

bool oa::ClipTokenizer::isLoaded() const noexcept { return impl_ and impl_->bos >= 0 and impl_->eos >= 0; }
oa::I32 oa::ClipTokenizer::vocabSize() const noexcept { return isLoaded() ? 49408 : 0; }
oa::I32 oa::ClipTokenizer::bosToken() const noexcept { return impl_ ? impl_->bos : -1; }
oa::I32 oa::ClipTokenizer::eosToken() const noexcept { return impl_ ? impl_->eos : -1; }
