#include "safeTensorsWeightSource.h"
#include <oa/core/log.h>
#include <oa/core/validation.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/std/algo.h>
#include <oa/core/std/cString.h>
#include <oa/core/std/limits.h>

namespace oa {

namespace {

constexpr const char* DTYPE_F64 = "F64";
constexpr const char* DTYPE_F32 = "F32";
constexpr const char* DTYPE_F16 = "F16";
constexpr const char* DTYPE_BF16 = "BF16";
constexpr const char* DTYPE_I64 = "I64";
constexpr const char* DTYPE_I32 = "I32";
constexpr const char* DTYPE_I16 = "I16";
constexpr const char* DTYPE_I8 = "I8";
constexpr const char* DTYPE_U8 = "U8";
constexpr const char* DTYPE_BOOL = "BOOL";

enum class JsonTokenType {
	ObjectStart, ObjectEnd, ArrayStart, ArrayEnd,
	Colon, Comma, String, Number, True, False, Null, End
	, Invalid
};

struct JsonToken {
	JsonTokenType type;
	const char* data;
	oa::Usize len;
};

class SimpleJsonLexer {
public:
	SimpleJsonLexer(const char* inData, oa::Usize inLen) : data_(inData), len_(inLen), pos_(0) {}

	JsonToken nextToken() {
		skipWhitespace();
		if (pos_ >= len_) return {.type = JsonTokenType::End, .data = nullptr, .len = 0};

		char c = data_[pos_];

		switch (c) {
			case '{': ++pos_; return {.type = JsonTokenType::ObjectStart, .data = "{", .len = 1};
			case '}': ++pos_; return {.type = JsonTokenType::ObjectEnd, .data = "}", .len = 1};
			case '[': ++pos_; return {.type = JsonTokenType::ArrayStart, .data = "[", .len = 1};
			case ']': ++pos_; return {.type = JsonTokenType::ArrayEnd, .data = "]", .len = 1};
			case ':': ++pos_; return {.type = JsonTokenType::Colon, .data = ":", .len = 1};
			case ',': ++pos_; return {.type = JsonTokenType::Comma, .data = ",", .len = 1};
			case '"': return parseString();
			case 't': return parseLiteral("true", 4, JsonTokenType::True);
			case 'f': return parseLiteral("false", 5, JsonTokenType::False);
			case 'n': return parseLiteral("null", 4, JsonTokenType::Null);
			default:
				if (c == '-' or (c >= '0' and c <= '9')) return parseNumber();
				++pos_;
				return {.type = JsonTokenType::Invalid, .data = &data_[pos_ - 1], .len = 1};
		}
	}

	bool expect(JsonTokenType inType, JsonToken& outToken) {
		outToken = nextToken();
		return outToken.type == inType;
	}

private:
	void skipWhitespace() {
		while (pos_ < len_ and (data_[pos_] == ' ' or data_[pos_] == '\t' or 
		                        data_[pos_] == '\n' or data_[pos_] == '\r')) {
			++pos_;
		}
	}

	JsonToken parseString() {
		++pos_;  // Skip opening quote
		oa::Usize start = pos_;
		while (pos_ < len_ and data_[pos_] != '"') {
			if (data_[pos_] == '\\' or static_cast<unsigned char>(data_[pos_]) < 0x20) {
				return {.type = JsonTokenType::Invalid, .data = &data_[pos_], .len = 1};
			}
			++pos_;
		}
		if (pos_ >= len_) return {.type = JsonTokenType::Invalid, .data = nullptr, .len = 0};
		JsonToken tok{.type = JsonTokenType::String, .data = &data_[start], .len = pos_ - start};
		++pos_;
		return tok;
	}

	JsonToken parseNumber() {
		oa::Usize start = pos_;
		if (data_[pos_] == '-') ++pos_;
		while (pos_ < len_ and ((data_[pos_] >= '0' and data_[pos_] <= '9') or data_[pos_] == '.' or 
		                        data_[pos_] == 'e' or data_[pos_] == 'E' or data_[pos_] == '+' or data_[pos_] == '-')) {
			++pos_;
		}
		return {.type = JsonTokenType::Number, .data = &data_[start], .len = pos_ - start};
	}

	JsonToken parseLiteral(const char* inExp, oa::Usize inLen, JsonTokenType inType) {
		if (pos_ + inLen <= len_ and oa::strncmp(&data_[pos_], inExp, inLen) == 0) {
			pos_ += inLen;
			return {.type = inType, .data = inExp, .len = inLen};
		}
		++pos_;
		return nextToken();
	}

	const char* data_;
	oa::Usize len_;
	oa::Usize pos_;
};

// ─── dtype conversion Helpers ────────────────────────────────────────────────

// BFloat16 to Float32 conversion
inline oa::F32 bf16ToF32(oa::U16 inBf16) {
	// BF16: 1 sign bit, 8 exponent bits, 7 mantissa bits
	// FP32: 1 sign bit, 8 exponent bits, 23 mantissa bits
	// BF16 is just FP32 with truncated mantissa - shift left by 16 bits
	oa::U32 bits = static_cast<oa::U32>(inBf16) << 16;
	oa::F32 result;
	oa::memcpy(&result, &bits, sizeof(oa::F32));
	return result;
}

// Float32 to BFloat16 conversion (round to nearest even)
inline oa::U16 f32ToBf16(oa::F32 inF32) {
	oa::U32 bits;
	oa::memcpy(&bits, &inF32, sizeof(oa::F32));
	
	// Round to nearest even (RNE)
	oa::U32 rounding = 0x7FFF + ((bits >> 16) & 1);
	bits += rounding;
	
	return static_cast<oa::U16>(bits >> 16);
}

// Float16 to Float32 conversion
inline oa::F32 f16ToF32(oa::U16 inF16) {
	// FP16: 1 sign bit, 5 exponent bits, 10 mantissa bits
	// FP32: 1 sign bit, 8 exponent bits, 23 mantissa bits
	
	oa::U32 sign = (inF16 & 0x8000) << 16;
	oa::U32 exponent = (inF16 & 0x7C00) >> 10;
	oa::U32 mantissa = (inF16 & 0x03FF);
	
	oa::U32 result;
	if (exponent == 0) {
		// Subnormal or zero
		if (mantissa == 0) {
			result = sign;  // Zero
		} else {
			// Subnormal - normalize it
			exponent = 1;
			while ((mantissa & 0x0400) == 0) {
				mantissa <<= 1;
				exponent--;
			}
			mantissa &= 0x03FF;
			result = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
		}
	} else if (exponent == 0x1F) {
		// Inf or NaN
		result = sign | 0x7F800000 | (mantissa << 13);
	} else {
		// Normal number
		result = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
	}
	
	oa::F32 f32;
	oa::memcpy(&f32, &result, sizeof(oa::F32));
	return f32;
}

// Float32 to Float16 conversion (round to nearest even)
inline oa::U16 f32ToF16(oa::F32 inF32) {
	oa::U32 bits;
	oa::memcpy(&bits, &inF32, sizeof(oa::F32));
	
	oa::U32 sign = (bits & 0x80000000) >> 16;
	oa::I32 exponent = ((bits & 0x7F800000) >> 23) - 127 + 15;
	oa::U32 mantissa = bits & 0x007FFFFF;
	
	if (exponent <= 0) {
		// Underflow to zero or subnormal
		if (exponent < -10) return static_cast<oa::U16>(sign);  // Too small
		
		// Subnormal
		mantissa = (mantissa | 0x00800000) >> (1 - exponent);
		return static_cast<oa::U16>(sign | (mantissa >> 13));
	} else if (exponent >= 0x1F) {
		// Overflow to infinity
		return static_cast<oa::U16>(sign | 0x7C00);
	}
	
	// Normal number - round to nearest even
	oa::U32 rounding = 0x00001000 + ((mantissa >> 13) & 1);
	mantissa += rounding;
	
	return static_cast<oa::U16>(sign | (exponent << 10) | (mantissa >> 13));
}

// convert buffer from one dtype to another
oa::Status convertDtype(
	const void* inSrc, void* outDst, oa::U64 inCount,
	oa::ScalarType inSrcDtype, oa::ScalarType inDstDtype
) {
	// Same dtype - direct copy
	if (inSrcDtype == inDstDtype) {
		oa::memcpy(outDst, inSrc, inCount * oa::scalarSize(inSrcDtype));
		return oa::Status::ok();
	}
	
	// BF16 → FP32
	if (inSrcDtype == oa::ScalarType::BFloat16 and inDstDtype == oa::ScalarType::Float32) {
		const auto* src = static_cast<const oa::U8*>(inSrc);
		auto* dst = static_cast<oa::U8*>(outDst);
		for (oa::U64 i = 0; i < inCount; ++i) {
			oa::U16 value;
			oa::memcpy(&value, src + i * sizeof(value), sizeof(value));
			const oa::F32 converted = bf16ToF32(value);
			oa::memcpy(dst + i * sizeof(converted), &converted, sizeof(converted));
		}
		return oa::Status::ok();
	}
	
	// FP32 → BF16
	if (inSrcDtype == oa::ScalarType::Float32 and inDstDtype == oa::ScalarType::BFloat16) {
		const auto* src = static_cast<const oa::U8*>(inSrc);
		auto* dst = static_cast<oa::U8*>(outDst);
		for (oa::U64 i = 0; i < inCount; ++i) {
			oa::F32 value;
			oa::memcpy(&value, src + i * sizeof(value), sizeof(value));
			const oa::U16 converted = f32ToBf16(value);
			oa::memcpy(dst + i * sizeof(converted), &converted, sizeof(converted));
		}
		return oa::Status::ok();
	}
	
	// FP16 → FP32
	if (inSrcDtype == oa::ScalarType::Float16 and inDstDtype == oa::ScalarType::Float32) {
		const auto* src = static_cast<const oa::U8*>(inSrc);
		auto* dst = static_cast<oa::U8*>(outDst);
		for (oa::U64 i = 0; i < inCount; ++i) {
			oa::U16 value;
			oa::memcpy(&value, src + i * sizeof(value), sizeof(value));
			const oa::F32 converted = f16ToF32(value);
			oa::memcpy(dst + i * sizeof(converted), &converted, sizeof(converted));
		}
		return oa::Status::ok();
	}
	
	// FP32 → FP16
	if (inSrcDtype == oa::ScalarType::Float32 and inDstDtype == oa::ScalarType::Float16) {
		const auto* src = static_cast<const oa::U8*>(inSrc);
		auto* dst = static_cast<oa::U8*>(outDst);
		for (oa::U64 i = 0; i < inCount; ++i) {
			oa::F32 value;
			oa::memcpy(&value, src + i * sizeof(value), sizeof(value));
			const oa::U16 converted = f32ToF16(value);
			oa::memcpy(dst + i * sizeof(converted), &converted, sizeof(converted));
		}
		return oa::Status::ok();
	}
	
	// BF16 → FP16 (via FP32)
	if (inSrcDtype == oa::ScalarType::BFloat16 and inDstDtype == oa::ScalarType::Float16) {
		const auto* src = static_cast<const oa::U8*>(inSrc);
		auto* dst = static_cast<oa::U8*>(outDst);
		for (oa::U64 i = 0; i < inCount; ++i) {
			oa::U16 value;
			oa::memcpy(&value, src + i * sizeof(value), sizeof(value));
			const oa::U16 converted = f32ToF16(bf16ToF32(value));
			oa::memcpy(dst + i * sizeof(converted), &converted, sizeof(converted));
		}
		return oa::Status::ok();
	}
	
	// FP16 → BF16 (via FP32)
	if (inSrcDtype == oa::ScalarType::Float16 and inDstDtype == oa::ScalarType::BFloat16) {
		const auto* src = static_cast<const oa::U8*>(inSrc);
		auto* dst = static_cast<oa::U8*>(outDst);
		for (oa::U64 i = 0; i < inCount; ++i) {
			oa::U16 value;
			oa::memcpy(&value, src + i * sizeof(value), sizeof(value));
			const oa::U16 converted = f32ToBf16(f16ToF32(value));
			oa::memcpy(dst + i * sizeof(converted), &converted, sizeof(converted));
		}
		return oa::Status::ok();
	}
	
	// Unsupported conversion
	return oa::Status::error(oa::StatusCode::Unimplemented,
		oa::String("Unsupported dtype conversion: ") +
		oa::String(oa::scalarTypeName(inSrcDtype)) + " -> " +
		oa::String(oa::scalarTypeName(inDstDtype)));
}

} // anonymous namespace

oa::Status SafeTensorsWeightSource::open(const oa::Path& inPath) {
	isOpen_ = false;
	path_ = inPath;
	entries_.clear();
	entryOrder_.clear();
	metadata_.clear();
	headerLen_ = 0;
	dataStart_ = 0;
	file_.close();

	OA_RETURN_IF_ERROR(file_.openReadOnly(inPath));
	const oa::U64 fileSize = static_cast<oa::U64>(file_.size());
	if (fileSize < 10) {
		file_.close();
		return oa::Status::error(oa::StatusCode::FileCorrupt, "file is too small to be SafeTensors");
	}

	// Read header length (first 8 bytes, little-endian uint64)
	oa::memcpy(&headerLen_, file_.data(), sizeof(headerLen_));
	if (headerLen_ < 2 || headerLen_ > 100'000'000 || headerLen_ > fileSize - 8) {
		file_.close();
		return oa::Status::error(oa::StatusCode::FileCorrupt, "Invalid SafeTensors header length");
	}
	dataStart_ = 8 + headerLen_;

	// parse header
	auto header = file_.slice(8, headerLen_);
	if (header.isError()) {
		file_.close();
		return header.getStatus();
	}
	auto parseStatus = parseHeader(header.getValue());
	if (parseStatus.isError()) {
		file_.close();
		return parseStatus;
	}
	auto validationStatus = validateEntries();
	if (validationStatus.isError()) {
		file_.close();
		return validationStatus;
	}

	isOpen_ = true;
	OaLogInfo(oa::LogComponent::Ml, "weight source: opened %s (%llu bytes, %zu entries)",
		inPath.cStr(), static_cast<unsigned long long>(fileSize), entries_.size());

	return oa::Status::ok();
}

oa::Status SafeTensorsWeightSource::parseHeader(oa::Span<const oa::U8> inHeaderData) {
	const char* data = reinterpret_cast<const char*>(inHeaderData.data());
	oa::Usize len = inHeaderData.size();

	SimpleJsonLexer lexer(data, len);
	JsonToken token;

	if (not lexer.expect(JsonTokenType::ObjectStart, token)) {
		return oa::Status::error(oa::StatusCode::FileCorrupt, "expected '{' at start of header");
	}

	auto parseUnsigned = [](const JsonToken& inToken, oa::U64& outValue) -> bool {
		if (inToken.type != JsonTokenType::Number || inToken.len == 0) return false;
		oa::U64 value = 0;
		for (oa::Usize i = 0; i < inToken.len; ++i) {
			const char c = inToken.data[i];
			if (c < '0' || c > '9') return false;
			const oa::U64 digit = static_cast<oa::U64>(c - '0');
			if (value > (oa::Limits<oa::U64>::max() - digit) / 10) return false;
			value = value * 10 + digit;
		}
		outValue = value;
		return true;
	};

	JsonToken next = lexer.nextToken();
	if (next.type == JsonTokenType::ObjectEnd) return oa::Status::ok();
	bool metadataSeen = false;

	while (true) {
		const JsonToken keyToken = next;
		if (keyToken.type != JsonTokenType::String) {
			return oa::Status::error(oa::StatusCode::FileCorrupt, "expected SafeTensors entry name");
		}

		oa::String key(keyToken.data, keyToken.len);
		if (entries_.contains(key) || (key == "__metadata__" && metadataSeen)) {
			return oa::Status::error(oa::StatusCode::FileCorrupt, oa::String("Duplicate header key: ") + key);
		}

		JsonToken colon;
		if (not lexer.expect(JsonTokenType::Colon, colon)) {
			return oa::Status::error(oa::StatusCode::FileCorrupt,
				oa::String("expected ':' after key '") + key + "'");
		}

		if (key == "__metadata__") {
			metadataSeen = true;
			// parse metadata object
			JsonToken objToken;
			if (not lexer.expect(JsonTokenType::ObjectStart, objToken)) {
				return oa::Status::error(oa::StatusCode::FileCorrupt, "expected '{' for __metadata__");
			}

			JsonToken metaNext = lexer.nextToken();
			if (metaNext.type != JsonTokenType::ObjectEnd) while (true) {
				const JsonToken metaKey = metaNext;
				if (metaKey.type != JsonTokenType::String) {
					return oa::Status::error(oa::StatusCode::FileCorrupt, "expected metadata key");
				}

				JsonToken metaColon;
				if (not lexer.expect(JsonTokenType::Colon, metaColon)) {
					return oa::Status::error(oa::StatusCode::FileCorrupt, "expected ':' after metadata key");
				}

				JsonToken metaValue;
				if (not lexer.expect(JsonTokenType::String, metaValue)) {
					return oa::Status::error(oa::StatusCode::FileCorrupt, "SafeTensors metadata values must be strings");
				}

				auto inserted = metadata_.emplace(
					oa::String(metaKey.data, metaKey.len), oa::String(metaValue.data, metaValue.len));
				if (!inserted.second) {
					return oa::Status::error(oa::StatusCode::FileCorrupt, "Duplicate SafeTensors metadata key");
				}

				JsonToken commaOrEnd = lexer.nextToken();
				if (commaOrEnd.type == JsonTokenType::ObjectEnd) break;
				if (commaOrEnd.type != JsonTokenType::Comma) {
					return oa::Status::error(oa::StatusCode::FileCorrupt, "expected ',' in SafeTensors metadata");
				}
				metaNext = lexer.nextToken();
			}
		} else {
			// parse weight entry
			JsonToken objToken;
			if (not lexer.expect(JsonTokenType::ObjectStart, objToken)) {
				return oa::Status::error(oa::StatusCode::FileCorrupt,
					oa::String("expected '{' for entry '") + key + "'");
			}

			Entry entry;
			auto& info = entry.info;
			info.name = key;
			bool hasDtype = false;
			bool hasShape = false;
			bool hasOffsets = false;

			JsonToken fieldNext = lexer.nextToken();
			if (fieldNext.type != JsonTokenType::ObjectEnd) while (true) {
				const JsonToken field = fieldNext;
				if (field.type != JsonTokenType::String) {
					return oa::Status::error(oa::StatusCode::FileCorrupt,
						oa::String("expected field name in '") + key + "'");
				}

				oa::String fieldName(field.data, field.len);

				JsonToken colon2;
				if (not lexer.expect(JsonTokenType::Colon, colon2)) {
					return oa::Status::error(oa::StatusCode::FileCorrupt, "expected ':' after entry field");
				}

				if (fieldName == "dtype") {
					if (hasDtype) return oa::Status::error(oa::StatusCode::FileCorrupt, "Duplicate dtype field");
					JsonToken dtype;
					if (!lexer.expect(JsonTokenType::String, dtype)) {
						return oa::Status::error(oa::StatusCode::FileCorrupt, "dtype must be a string");
					}
					auto parsed = parseDtype(oa::String(dtype.data, dtype.len));
					if (parsed.isError()) return parsed.getStatus();
					info.dtype = parsed.getValue();
					hasDtype = true;
				} else if (fieldName == "shape") {
					if (hasShape) return oa::Status::error(oa::StatusCode::FileCorrupt, "Duplicate shape field");
					JsonToken arr;
					if (!lexer.expect(JsonTokenType::ArrayStart, arr)) {
						return oa::Status::error(oa::StatusCode::FileCorrupt, "shape must be an array");
					}
					JsonToken dimNext = lexer.nextToken();
					if (dimNext.type != JsonTokenType::ArrayEnd) while (true) {
						oa::U64 dim = 0;
						if (!parseUnsigned(dimNext, dim) || dim > static_cast<oa::U64>(oa::Limits<oa::I64>::max())) {
							return oa::Status::error(oa::StatusCode::FileCorrupt, "shape dimensions must be non-negative integers");
						}
						info.shape.pushBack(static_cast<oa::I64>(dim));
						if (info.shape.size() > 32) {
							return oa::Status::error(oa::StatusCode::FileCorrupt, "SafeTensors rank exceeds 32");
						}
						JsonToken commaOrEnd = lexer.nextToken();
						if (commaOrEnd.type == JsonTokenType::ArrayEnd) break;
						if (commaOrEnd.type != JsonTokenType::Comma) {
							return oa::Status::error(oa::StatusCode::FileCorrupt, "expected ',' in shape");
						}
						dimNext = lexer.nextToken();
					}
					hasShape = true;
				} else if (fieldName == "data_offsets") {
					if (hasOffsets) return oa::Status::error(oa::StatusCode::FileCorrupt, "Duplicate data_offsets field");
					JsonToken arr;
					if (!lexer.expect(JsonTokenType::ArrayStart, arr)) {
						return oa::Status::error(oa::StatusCode::FileCorrupt, "data_offsets must be an array");
					}
					JsonToken beginToken = lexer.nextToken();
					JsonToken commaToken = lexer.nextToken();
					JsonToken endToken = lexer.nextToken();
					JsonToken closeToken = lexer.nextToken();
					oa::U64 begin = 0;
					oa::U64 end = 0;
					if (!parseUnsigned(beginToken, begin) || commaToken.type != JsonTokenType::Comma ||
						!parseUnsigned(endToken, end) || closeToken.type != JsonTokenType::ArrayEnd || end < begin) {
						return oa::Status::error(oa::StatusCode::FileCorrupt,
							"data_offsets must contain exactly two ordered unsigned integers");
					}
					entry.dataOffset = begin;
					info.byteSize = end - begin;
					hasOffsets = true;
				} else {
					return oa::Status::error(oa::StatusCode::FileCorrupt,
						oa::String("Unknown SafeTensors entry field: ") + fieldName);
				}

				JsonToken commaOrEnd = lexer.nextToken();
				if (commaOrEnd.type == JsonTokenType::ObjectEnd) break;
				if (commaOrEnd.type != JsonTokenType::Comma) {
					return oa::Status::error(oa::StatusCode::FileCorrupt, "expected ',' between entry fields");
				}
				fieldNext = lexer.nextToken();
			}

			if (!hasDtype || !hasShape || !hasOffsets) {
				return oa::Status::error(oa::StatusCode::FileCorrupt,
					oa::String("Entry is missing dtype, shape, or data_offsets: ") + key);
			}
			auto inserted = entries_.emplace(key, oa::move(entry));
			if (!inserted.second) {
				return oa::Status::error(oa::StatusCode::FileCorrupt, oa::String("Duplicate entry name: ") + key);
			}
			entryOrder_.pushBack(key);
		}

		JsonToken commaOrEnd = lexer.nextToken();
		if (commaOrEnd.type == JsonTokenType::ObjectEnd) break;
		if (commaOrEnd.type != JsonTokenType::Comma) {
			return oa::Status::error(oa::StatusCode::FileCorrupt, "expected ',' between SafeTensors entries");
		}
		next = lexer.nextToken();
	}
	if (lexer.nextToken().type != JsonTokenType::End) {
		return oa::Status::error(oa::StatusCode::FileCorrupt, "Unexpected data after SafeTensors header object");
	}

	return oa::Status::ok();
}

oa::Result<oa::ScalarType> SafeTensorsWeightSource::parseDtype(oa::StringView inStr) const {
	if (inStr == DTYPE_F64) return oa::ScalarType::Float64;
	if (inStr == DTYPE_F32) return oa::ScalarType::Float32;
	if (inStr == DTYPE_F16) return oa::ScalarType::Float16;
	if (inStr == DTYPE_BF16) return oa::ScalarType::BFloat16;
	if (inStr == DTYPE_I64) return oa::ScalarType::Int64;
	if (inStr == DTYPE_I32) return oa::ScalarType::Int32;
	if (inStr == DTYPE_I16) return oa::ScalarType::Int16;
	if (inStr == DTYPE_I8) return oa::ScalarType::Int8;
	if (inStr == DTYPE_U8) return oa::ScalarType::UInt8;
	if (inStr == DTYPE_BOOL) return oa::ScalarType::Bool;
	return oa::Status::error(oa::StatusCode::DtypeMismatch, oa::String("Unsupported SafeTensors dtype: ") + inStr);
}

oa::Status SafeTensorsWeightSource::validateEntries() {
	const oa::U64 dataBytes = static_cast<oa::U64>(file_.size()) - dataStart_;
	oa::Vector<const Entry*> sorted;
	sorted.reserve(entryOrder_.size());

	for (const auto& name : entryOrder_) {
		auto it = entries_.find(name);
		if (it == entries_.end()) {
			return oa::Status::error(oa::StatusCode::Internal, "SafeTensors entry order is inconsistent");
		}
		auto& entry = it->second;
		auto& info = entry.info;
		const oa::U64 dtypeSize = static_cast<oa::U64>(oa::scalarSize(info.dtype));
		if (dtypeSize == 0) {
			return oa::Status::error(oa::StatusCode::DtypeMismatch, oa::String("Invalid dtype for entry: ") + info.name);
		}

		oa::U64 count = 1;
		for (oa::I64 dim : info.shape) {
			if (dim == 0) {
				count = 0;
				break;
			}
			const oa::U64 uDim = static_cast<oa::U64>(dim);
			if (count > oa::Limits<oa::U64>::max() / uDim) {
				return oa::Status::error(oa::StatusCode::FileCorrupt,
					oa::String("Element count overflow for entry: ") + info.name);
			}
			count *= uDim;
		}
		if (info.shape.empty()) count = 1;
		info.elementCount = count;
		if (count > oa::Limits<oa::U64>::max() / dtypeSize ||
			count * dtypeSize != info.byteSize) {
			return oa::Status::error(oa::StatusCode::FileCorrupt,
				oa::String("Shape and byte length disagree for entry: ") + info.name);
		}
		if (entry.dataOffset > dataBytes || info.byteSize > dataBytes - entry.dataOffset) {
			return oa::Status::error(oa::StatusCode::FileCorrupt,
				oa::String("Entry payload is outside file bounds: ") + info.name);
		}
		sorted.pushBack(&entry);
	}

	oa::sort(sorted.begin(), sorted.end(), [](const auto* inA, const auto* inB) {
		if (inA->dataOffset != inB->dataOffset) return inA->dataOffset < inB->dataOffset;
		return inA->info.byteSize < inB->info.byteSize;
	});

	oa::U64 indexedEnd = 0;
	for (const auto* entry : sorted) {
		if (entry->info.byteSize == 0) {
			if (entry->dataOffset > indexedEnd) {
				return oa::Status::error(oa::StatusCode::FileCorrupt,
					oa::String("Unindexed gap before empty entry: ") + entry->info.name);
			}
			continue;
		}
		if (entry->dataOffset < indexedEnd) {
			return oa::Status::error(oa::StatusCode::FileCorrupt,
				oa::String("Overlapping entry payload: ") + entry->info.name);
		}
		if (entry->dataOffset > indexedEnd) {
			return oa::Status::error(oa::StatusCode::FileCorrupt,
				oa::String("SafeTensors data buffer contains a hole before entry: ") + entry->info.name);
		}
		indexedEnd = entry->dataOffset + entry->info.byteSize;
	}
	if (indexedEnd != dataBytes) {
		return oa::Status::error(oa::StatusCode::FileCorrupt, "SafeTensors data buffer is not entirely indexed");
	}
	return oa::Status::ok();
}

oa::Vector<oa::WeightInfo> SafeTensorsWeightSource::list() const {
	oa::Vector<oa::WeightInfo> result;
	result.reserve(entryOrder_.size());
	for (const auto& name : entryOrder_) {
		auto it = entries_.find(name);
		if (it != entries_.end()) result.pushBack(it->second.info);
	}
	return result;
}

const oa::WeightInfo* SafeTensorsWeightSource::find(oa::StringView inName) const {
	auto it = entries_.find(oa::String(inName));
	if (it != entries_.end()) return &it->second.info;
	return nullptr;
}

oa::Result<oa::Span<const oa::U8>> SafeTensorsWeightSource::bytes(oa::StringView inName) const {
	if (!isOpen_) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition, "SafeTensors reader is not open");
	}
	auto it = entries_.find(oa::String(inName));
	const auto* info = it == entries_.end() ? nullptr : &it->second.info;
	if (!info) return oa::Status::notFound(oa::String("Entry not found: ") + inName);
	return file_.slice(dataStart_ + it->second.dataOffset, info->byteSize);
}

oa::Status SafeTensorsWeightSource::read(
	oa::StringView inName,
	oa::Span<oa::U8> outData,
	oa::ScalarType inTargetDtype
) const {
	if (outData.data() == nullptr && outData.size() != 0) {
		return oa::Status::invalidArgument("oa::WeightSource::read: null output buffer");
	}

	const auto* info = find(inName);
	if (not info) {
		return oa::Status::error(oa::StatusCode::NotFound,
			oa::String("Entry not found: ") + inName);
	}

	// Direct copy if no conversion needed
	if (inTargetDtype == info->dtype) {
		if (outData.size() < info->byteSize) {
			return oa::Status::error(oa::StatusCode::OutOfRange,
				oa::String("output buffer is too small for entry: ") + inName);
		}

		auto bytesResult = bytes(inName);
		if (bytesResult.isError()) return bytesResult.getStatus();
		oa::memcpy(outData.data(), bytesResult->data(), info->byteSize);
		return oa::Status::ok();
	}

	// dtype conversion needed
	const oa::U64 targetDtypeSize = oa::scalarSize(inTargetDtype);
	if (targetDtypeSize == 0 || info->elementCount > oa::Limits<oa::U64>::max() / targetDtypeSize) {
		return oa::Status::error(oa::StatusCode::DtypeMismatch, "Invalid target dtype");
	}
	oa::U64 targetBytes = info->elementCount * targetDtypeSize;
	
	if (outData.size() < targetBytes) {
		return oa::Status::error(oa::StatusCode::OutOfRange,
			oa::String("output buffer is too small after dtype conversion for entry: ") + inName);
	}
	
	auto bytesResult = bytes(inName);
	if (bytesResult.isError()) return bytesResult.getStatus();
	return convertDtype(bytesResult->data(), outData.data(), info->elementCount,
		info->dtype, inTargetDtype);
}

} // namespace oa
