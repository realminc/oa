#include "SafeTensorsWeightSource.h"
#include <Oa/Core/Log.h>
#include <Oa/Core/Validation.h>
#include <Oa/Core/FnMatrix.h>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <limits>

// Minimal JSON parser for SafeTensors header
// No namespace wrapper - flat API with Oa prefix

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
	JsonTokenType Type;
	const char* Data;
	OaUsize Len;
};

class SimpleJsonLexer {
public:
	SimpleJsonLexer(const char* InData, OaUsize InLen) : Data_(InData), Len_(InLen), Pos_(0) {}

	JsonToken NextToken() {
		SkipWhitespace();
		if (Pos_ >= Len_) return {.Type = JsonTokenType::End, .Data = nullptr, .Len = 0};

		char c = Data_[Pos_];

		switch (c) {
			case '{': ++Pos_; return {.Type = JsonTokenType::ObjectStart, .Data = "{", .Len = 1};
			case '}': ++Pos_; return {.Type = JsonTokenType::ObjectEnd, .Data = "}", .Len = 1};
			case '[': ++Pos_; return {.Type = JsonTokenType::ArrayStart, .Data = "[", .Len = 1};
			case ']': ++Pos_; return {.Type = JsonTokenType::ArrayEnd, .Data = "]", .Len = 1};
			case ':': ++Pos_; return {.Type = JsonTokenType::Colon, .Data = ":", .Len = 1};
			case ',': ++Pos_; return {.Type = JsonTokenType::Comma, .Data = ",", .Len = 1};
			case '"': return ParseString();
			case 't': return ParseLiteral("true", 4, JsonTokenType::True);
			case 'f': return ParseLiteral("false", 5, JsonTokenType::False);
			case 'n': return ParseLiteral("null", 4, JsonTokenType::Null);
			default:
				if (c == '-' or (c >= '0' and c <= '9')) return ParseNumber();
				++Pos_;
				return {.Type = JsonTokenType::Invalid, .Data = &Data_[Pos_ - 1], .Len = 1};
		}
	}

	bool Expect(JsonTokenType InType, JsonToken& OutToken) {
		OutToken = NextToken();
		return OutToken.Type == InType;
	}

private:
	void SkipWhitespace() {
		while (Pos_ < Len_ and (Data_[Pos_] == ' ' or Data_[Pos_] == '\t' or 
		                        Data_[Pos_] == '\n' or Data_[Pos_] == '\r')) {
			++Pos_;
		}
	}

	JsonToken ParseString() {
		++Pos_;  // Skip opening quote
		OaUsize start = Pos_;
		while (Pos_ < Len_ and Data_[Pos_] != '"') {
			if (Data_[Pos_] == '\\' or static_cast<unsigned char>(Data_[Pos_]) < 0x20) {
				return {.Type = JsonTokenType::Invalid, .Data = &Data_[Pos_], .Len = 1};
			}
			++Pos_;
		}
		if (Pos_ >= Len_) return {.Type = JsonTokenType::Invalid, .Data = nullptr, .Len = 0};
		JsonToken tok{.Type = JsonTokenType::String, .Data = &Data_[start], .Len = Pos_ - start};
		++Pos_;
		return tok;
	}

	JsonToken ParseNumber() {
		OaUsize start = Pos_;
		if (Data_[Pos_] == '-') ++Pos_;
		while (Pos_ < Len_ and ((Data_[Pos_] >= '0' and Data_[Pos_] <= '9') or Data_[Pos_] == '.' or 
		                        Data_[Pos_] == 'e' or Data_[Pos_] == 'E' or Data_[Pos_] == '+' or Data_[Pos_] == '-')) {
			++Pos_;
		}
		return {.Type = JsonTokenType::Number, .Data = &Data_[start], .Len = Pos_ - start};
	}

	JsonToken ParseLiteral(const char* InExp, OaUsize InLen, JsonTokenType InType) {
		if (Pos_ + InLen <= Len_ and std::strncmp(&Data_[Pos_], InExp, InLen) == 0) {
			Pos_ += InLen;
			return {.Type = InType, .Data = InExp, .Len = InLen};
		}
		++Pos_;
		return NextToken();
	}

	const char* Data_;
	OaUsize Len_;
	OaUsize Pos_;
};

// ─── Dtype Conversion Helpers ────────────────────────────────────────────────

// BFloat16 to Float32 conversion
inline OaF32 Bf16ToF32(OaU16 InBf16) {
	// BF16: 1 sign bit, 8 exponent bits, 7 mantissa bits
	// FP32: 1 sign bit, 8 exponent bits, 23 mantissa bits
	// BF16 is just FP32 with truncated mantissa - shift left by 16 bits
	OaU32 bits = static_cast<OaU32>(InBf16) << 16;
	OaF32 result;
	OaMemcpy(&result, &bits, sizeof(OaF32));
	return result;
}

// Float32 to BFloat16 conversion (round to nearest even)
inline OaU16 F32ToBf16(OaF32 InF32) {
	OaU32 bits;
	OaMemcpy(&bits, &InF32, sizeof(OaF32));
	
	// Round to nearest even (RNE)
	OaU32 rounding = 0x7FFF + ((bits >> 16) & 1);
	bits += rounding;
	
	return static_cast<OaU16>(bits >> 16);
}

// Float16 to Float32 conversion
inline OaF32 F16ToF32(OaU16 InF16) {
	// FP16: 1 sign bit, 5 exponent bits, 10 mantissa bits
	// FP32: 1 sign bit, 8 exponent bits, 23 mantissa bits
	
	OaU32 sign = (InF16 & 0x8000) << 16;
	OaU32 exponent = (InF16 & 0x7C00) >> 10;
	OaU32 mantissa = (InF16 & 0x03FF);
	
	OaU32 result;
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
	
	OaF32 f32;
	OaMemcpy(&f32, &result, sizeof(OaF32));
	return f32;
}

// Float32 to Float16 conversion (round to nearest even)
inline OaU16 F32ToF16(OaF32 InF32) {
	OaU32 bits;
	OaMemcpy(&bits, &InF32, sizeof(OaF32));
	
	OaU32 sign = (bits & 0x80000000) >> 16;
	OaI32 exponent = ((bits & 0x7F800000) >> 23) - 127 + 15;
	OaU32 mantissa = bits & 0x007FFFFF;
	
	if (exponent <= 0) {
		// Underflow to zero or subnormal
		if (exponent < -10) return static_cast<OaU16>(sign);  // Too small
		
		// Subnormal
		mantissa = (mantissa | 0x00800000) >> (1 - exponent);
		return static_cast<OaU16>(sign | (mantissa >> 13));
	} else if (exponent >= 0x1F) {
		// Overflow to infinity
		return static_cast<OaU16>(sign | 0x7C00);
	}
	
	// Normal number - round to nearest even
	OaU32 rounding = 0x00001000 + ((mantissa >> 13) & 1);
	mantissa += rounding;
	
	return static_cast<OaU16>(sign | (exponent << 10) | (mantissa >> 13));
}

// Convert buffer from one dtype to another
OaStatus ConvertDtype(
	const void* InSrc, void* OutDst, OaU64 InCount,
	OaScalarType InSrcDtype, OaScalarType InDstDtype
) {
	// Same dtype - direct copy
	if (InSrcDtype == InDstDtype) {
		OaMemcpy(OutDst, InSrc, InCount * OaScalarSize(InSrcDtype));
		return OaStatus::Ok();
	}
	
	// BF16 → FP32
	if (InSrcDtype == OaScalarType::BFloat16 and InDstDtype == OaScalarType::Float32) {
		const auto* src = static_cast<const OaU8*>(InSrc);
		auto* dst = static_cast<OaU8*>(OutDst);
		for (OaU64 i = 0; i < InCount; ++i) {
			OaU16 value;
			OaMemcpy(&value, src + i * sizeof(value), sizeof(value));
			const OaF32 converted = Bf16ToF32(value);
			OaMemcpy(dst + i * sizeof(converted), &converted, sizeof(converted));
		}
		return OaStatus::Ok();
	}
	
	// FP32 → BF16
	if (InSrcDtype == OaScalarType::Float32 and InDstDtype == OaScalarType::BFloat16) {
		const auto* src = static_cast<const OaU8*>(InSrc);
		auto* dst = static_cast<OaU8*>(OutDst);
		for (OaU64 i = 0; i < InCount; ++i) {
			OaF32 value;
			OaMemcpy(&value, src + i * sizeof(value), sizeof(value));
			const OaU16 converted = F32ToBf16(value);
			OaMemcpy(dst + i * sizeof(converted), &converted, sizeof(converted));
		}
		return OaStatus::Ok();
	}
	
	// FP16 → FP32
	if (InSrcDtype == OaScalarType::Float16 and InDstDtype == OaScalarType::Float32) {
		const auto* src = static_cast<const OaU8*>(InSrc);
		auto* dst = static_cast<OaU8*>(OutDst);
		for (OaU64 i = 0; i < InCount; ++i) {
			OaU16 value;
			OaMemcpy(&value, src + i * sizeof(value), sizeof(value));
			const OaF32 converted = F16ToF32(value);
			OaMemcpy(dst + i * sizeof(converted), &converted, sizeof(converted));
		}
		return OaStatus::Ok();
	}
	
	// FP32 → FP16
	if (InSrcDtype == OaScalarType::Float32 and InDstDtype == OaScalarType::Float16) {
		const auto* src = static_cast<const OaU8*>(InSrc);
		auto* dst = static_cast<OaU8*>(OutDst);
		for (OaU64 i = 0; i < InCount; ++i) {
			OaF32 value;
			OaMemcpy(&value, src + i * sizeof(value), sizeof(value));
			const OaU16 converted = F32ToF16(value);
			OaMemcpy(dst + i * sizeof(converted), &converted, sizeof(converted));
		}
		return OaStatus::Ok();
	}
	
	// BF16 → FP16 (via FP32)
	if (InSrcDtype == OaScalarType::BFloat16 and InDstDtype == OaScalarType::Float16) {
		const auto* src = static_cast<const OaU8*>(InSrc);
		auto* dst = static_cast<OaU8*>(OutDst);
		for (OaU64 i = 0; i < InCount; ++i) {
			OaU16 value;
			OaMemcpy(&value, src + i * sizeof(value), sizeof(value));
			const OaU16 converted = F32ToF16(Bf16ToF32(value));
			OaMemcpy(dst + i * sizeof(converted), &converted, sizeof(converted));
		}
		return OaStatus::Ok();
	}
	
	// FP16 → BF16 (via FP32)
	if (InSrcDtype == OaScalarType::Float16 and InDstDtype == OaScalarType::BFloat16) {
		const auto* src = static_cast<const OaU8*>(InSrc);
		auto* dst = static_cast<OaU8*>(OutDst);
		for (OaU64 i = 0; i < InCount; ++i) {
			OaU16 value;
			OaMemcpy(&value, src + i * sizeof(value), sizeof(value));
			const OaU16 converted = F32ToBf16(F16ToF32(value));
			OaMemcpy(dst + i * sizeof(converted), &converted, sizeof(converted));
		}
		return OaStatus::Ok();
	}
	
	// Unsupported conversion
	return OaStatus::Error(OaStatusCode::Unimplemented,
		OaString("Unsupported dtype conversion: ") +
		OaString(OaScalarTypeName(InSrcDtype)) + " -> " +
		OaString(OaScalarTypeName(InDstDtype)));
}

} // anonymous namespace

OaStatus OaSafeTensorsWeightSource::Open(const OaPath& InPath) {
	IsOpen_ = false;
	Path_ = InPath;
	Entries_.Clear();
	EntryOrder_.Clear();
	Metadata_.Clear();
	HeaderLen_ = 0;
	DataStart_ = 0;
	File_.Close();

	OA_RETURN_IF_ERROR(File_.OpenReadOnly(InPath));
	const OaU64 fileSize = static_cast<OaU64>(File_.Size());
	if (fileSize < 10) {
		File_.Close();
		return OaStatus::Error(OaStatusCode::FileCorrupt, "File is too small to be SafeTensors");
	}

	// Read header length (first 8 bytes, little-endian uint64)
	OaMemcpy(&HeaderLen_, File_.Data(), sizeof(HeaderLen_));
	if (HeaderLen_ < 2 || HeaderLen_ > 100'000'000 || HeaderLen_ > fileSize - 8) {
		File_.Close();
		return OaStatus::Error(OaStatusCode::FileCorrupt, "Invalid SafeTensors header length");
	}
	DataStart_ = 8 + HeaderLen_;

	// Parse header
	auto header = File_.Slice(8, HeaderLen_);
	if (header.IsError()) {
		File_.Close();
		return header.GetStatus();
	}
	auto parseStatus = ParseHeader(header.GetValue());
	if (parseStatus.IsError()) {
		File_.Close();
		return parseStatus;
	}
	auto validationStatus = ValidateEntries();
	if (validationStatus.IsError()) {
		File_.Close();
		return validationStatus;
	}

	IsOpen_ = true;
	OA_LOG_INFO(OaLogComponent::ML, "Weight source: opened %s (%llu bytes, %zu entries)",
		InPath.CStr(), static_cast<unsigned long long>(fileSize), Entries_.Size());

	return OaStatus::Ok();
}

OaStatus OaSafeTensorsWeightSource::ParseHeader(OaSpan<const OaU8> InHeaderData) {
	const char* data = reinterpret_cast<const char*>(InHeaderData.Data());
	OaUsize len = InHeaderData.Size();

	SimpleJsonLexer lexer(data, len);
	JsonToken token;

	if (not lexer.Expect(JsonTokenType::ObjectStart, token)) {
		return OaStatus::Error(OaStatusCode::FileCorrupt, "Expected '{' at start of header");
	}

	auto ParseUnsigned = [](const JsonToken& InToken, OaU64& OutValue) -> bool {
		if (InToken.Type != JsonTokenType::Number || InToken.Len == 0) return false;
		OaU64 value = 0;
		for (OaUsize i = 0; i < InToken.Len; ++i) {
			const char c = InToken.Data[i];
			if (c < '0' || c > '9') return false;
			const OaU64 digit = static_cast<OaU64>(c - '0');
			if (value > (std::numeric_limits<OaU64>::max() - digit) / 10) return false;
			value = value * 10 + digit;
		}
		OutValue = value;
		return true;
	};

	JsonToken next = lexer.NextToken();
	if (next.Type == JsonTokenType::ObjectEnd) return OaStatus::Ok();
	bool metadataSeen = false;

	while (true) {
		const JsonToken keyToken = next;
		if (keyToken.Type != JsonTokenType::String) {
			return OaStatus::Error(OaStatusCode::FileCorrupt, "Expected SafeTensors entry name");
		}

		OaString key(keyToken.Data, keyToken.Len);
		if (Entries_.Contains(key) || (key == "__metadata__" && metadataSeen)) {
			return OaStatus::Error(OaStatusCode::FileCorrupt, OaString("Duplicate header key: ") + key);
		}

		JsonToken colon;
		if (not lexer.Expect(JsonTokenType::Colon, colon)) {
			return OaStatus::Error(OaStatusCode::FileCorrupt,
				OaString("Expected ':' after key '") + key + "'");
		}

		if (key == "__metadata__") {
			metadataSeen = true;
			// Parse metadata object
			JsonToken objToken;
			if (not lexer.Expect(JsonTokenType::ObjectStart, objToken)) {
				return OaStatus::Error(OaStatusCode::FileCorrupt, "Expected '{' for __metadata__");
			}

			JsonToken metaNext = lexer.NextToken();
			if (metaNext.Type != JsonTokenType::ObjectEnd) while (true) {
				const JsonToken metaKey = metaNext;
				if (metaKey.Type != JsonTokenType::String) {
					return OaStatus::Error(OaStatusCode::FileCorrupt, "Expected metadata key");
				}

				JsonToken metaColon;
				if (not lexer.Expect(JsonTokenType::Colon, metaColon)) {
					return OaStatus::Error(OaStatusCode::FileCorrupt, "Expected ':' after metadata key");
				}

				JsonToken metaValue;
				if (not lexer.Expect(JsonTokenType::String, metaValue)) {
					return OaStatus::Error(OaStatusCode::FileCorrupt, "SafeTensors metadata values must be strings");
				}

				auto inserted = Metadata_.Emplace(
					OaString(metaKey.Data, metaKey.Len), OaString(metaValue.Data, metaValue.Len));
				if (!inserted.second) {
					return OaStatus::Error(OaStatusCode::FileCorrupt, "Duplicate SafeTensors metadata key");
				}

				JsonToken commaOrEnd = lexer.NextToken();
				if (commaOrEnd.Type == JsonTokenType::ObjectEnd) break;
				if (commaOrEnd.Type != JsonTokenType::Comma) {
					return OaStatus::Error(OaStatusCode::FileCorrupt, "Expected ',' in SafeTensors metadata");
				}
				metaNext = lexer.NextToken();
			}
		} else {
			// Parse weight entry
			JsonToken objToken;
			if (not lexer.Expect(JsonTokenType::ObjectStart, objToken)) {
				return OaStatus::Error(OaStatusCode::FileCorrupt,
					OaString("Expected '{' for entry '") + key + "'");
			}

			Entry entry;
			auto& info = entry.Info;
			info.Name = key;
			bool hasDtype = false;
			bool hasShape = false;
			bool hasOffsets = false;

			JsonToken fieldNext = lexer.NextToken();
			if (fieldNext.Type != JsonTokenType::ObjectEnd) while (true) {
				const JsonToken field = fieldNext;
				if (field.Type != JsonTokenType::String) {
					return OaStatus::Error(OaStatusCode::FileCorrupt,
						OaString("Expected field name in '") + key + "'");
				}

				OaString fieldName(field.Data, field.Len);

				JsonToken colon2;
				if (not lexer.Expect(JsonTokenType::Colon, colon2)) {
					return OaStatus::Error(OaStatusCode::FileCorrupt, "Expected ':' after entry field");
				}

				if (fieldName == "dtype") {
					if (hasDtype) return OaStatus::Error(OaStatusCode::FileCorrupt, "Duplicate dtype field");
					JsonToken dtype;
					if (!lexer.Expect(JsonTokenType::String, dtype)) {
						return OaStatus::Error(OaStatusCode::FileCorrupt, "dtype must be a string");
					}
					auto parsed = ParseDtype(OaString(dtype.Data, dtype.Len));
					if (parsed.IsError()) return parsed.GetStatus();
					info.Dtype = parsed.GetValue();
					hasDtype = true;
				} else if (fieldName == "shape") {
					if (hasShape) return OaStatus::Error(OaStatusCode::FileCorrupt, "Duplicate shape field");
					JsonToken arr;
					if (!lexer.Expect(JsonTokenType::ArrayStart, arr)) {
						return OaStatus::Error(OaStatusCode::FileCorrupt, "shape must be an array");
					}
					JsonToken dimNext = lexer.NextToken();
					if (dimNext.Type != JsonTokenType::ArrayEnd) while (true) {
						OaU64 dim = 0;
						if (!ParseUnsigned(dimNext, dim) || dim > static_cast<OaU64>(std::numeric_limits<OaI64>::max())) {
							return OaStatus::Error(OaStatusCode::FileCorrupt, "shape dimensions must be non-negative integers");
						}
						info.Shape.PushBack(static_cast<OaI64>(dim));
						if (info.Shape.Size() > 32) {
							return OaStatus::Error(OaStatusCode::FileCorrupt, "SafeTensors rank exceeds 32");
						}
						JsonToken commaOrEnd = lexer.NextToken();
						if (commaOrEnd.Type == JsonTokenType::ArrayEnd) break;
						if (commaOrEnd.Type != JsonTokenType::Comma) {
							return OaStatus::Error(OaStatusCode::FileCorrupt, "Expected ',' in shape");
						}
						dimNext = lexer.NextToken();
					}
					hasShape = true;
				} else if (fieldName == "data_offsets") {
					if (hasOffsets) return OaStatus::Error(OaStatusCode::FileCorrupt, "Duplicate data_offsets field");
					JsonToken arr;
					if (!lexer.Expect(JsonTokenType::ArrayStart, arr)) {
						return OaStatus::Error(OaStatusCode::FileCorrupt, "data_offsets must be an array");
					}
					JsonToken beginToken = lexer.NextToken();
					JsonToken commaToken = lexer.NextToken();
					JsonToken endToken = lexer.NextToken();
					JsonToken closeToken = lexer.NextToken();
					OaU64 begin = 0;
					OaU64 end = 0;
					if (!ParseUnsigned(beginToken, begin) || commaToken.Type != JsonTokenType::Comma ||
						!ParseUnsigned(endToken, end) || closeToken.Type != JsonTokenType::ArrayEnd || end < begin) {
						return OaStatus::Error(OaStatusCode::FileCorrupt,
							"data_offsets must contain exactly two ordered unsigned integers");
					}
					entry.DataOffset = begin;
					info.ByteSize = end - begin;
					hasOffsets = true;
				} else {
					return OaStatus::Error(OaStatusCode::FileCorrupt,
						OaString("Unknown SafeTensors entry field: ") + fieldName);
				}

				JsonToken commaOrEnd = lexer.NextToken();
				if (commaOrEnd.Type == JsonTokenType::ObjectEnd) break;
				if (commaOrEnd.Type != JsonTokenType::Comma) {
					return OaStatus::Error(OaStatusCode::FileCorrupt, "Expected ',' between entry fields");
				}
				fieldNext = lexer.NextToken();
			}

			if (!hasDtype || !hasShape || !hasOffsets) {
				return OaStatus::Error(OaStatusCode::FileCorrupt,
					OaString("Entry is missing dtype, shape, or data_offsets: ") + key);
			}
			auto inserted = Entries_.Emplace(key, OaStdMove(entry));
			if (!inserted.second) {
				return OaStatus::Error(OaStatusCode::FileCorrupt, OaString("Duplicate entry name: ") + key);
			}
			EntryOrder_.PushBack(key);
		}

		JsonToken commaOrEnd = lexer.NextToken();
		if (commaOrEnd.Type == JsonTokenType::ObjectEnd) break;
		if (commaOrEnd.Type != JsonTokenType::Comma) {
			return OaStatus::Error(OaStatusCode::FileCorrupt, "Expected ',' between SafeTensors entries");
		}
		next = lexer.NextToken();
	}
	if (lexer.NextToken().Type != JsonTokenType::End) {
		return OaStatus::Error(OaStatusCode::FileCorrupt, "Unexpected data after SafeTensors header object");
	}

	return OaStatus::Ok();
}

OaResult<OaScalarType> OaSafeTensorsWeightSource::ParseDtype(OaStringView InStr) const {
	if (InStr == DTYPE_F64) return OaScalarType::Float64;
	if (InStr == DTYPE_F32) return OaScalarType::Float32;
	if (InStr == DTYPE_F16) return OaScalarType::Float16;
	if (InStr == DTYPE_BF16) return OaScalarType::BFloat16;
	if (InStr == DTYPE_I64) return OaScalarType::Int64;
	if (InStr == DTYPE_I32) return OaScalarType::Int32;
	if (InStr == DTYPE_I16) return OaScalarType::Int16;
	if (InStr == DTYPE_I8) return OaScalarType::Int8;
	if (InStr == DTYPE_U8) return OaScalarType::UInt8;
	if (InStr == DTYPE_BOOL) return OaScalarType::Bool;
	return OaStatus::Error(OaStatusCode::DtypeMismatch, OaString("Unsupported SafeTensors dtype: ") + InStr);
}

OaStatus OaSafeTensorsWeightSource::ValidateEntries() {
	const OaU64 dataBytes = static_cast<OaU64>(File_.Size()) - DataStart_;
	OaVec<const Entry*> sorted;
	sorted.Reserve(EntryOrder_.Size());

	for (const auto& name : EntryOrder_) {
		auto it = Entries_.Find(name);
		if (it == Entries_.End()) {
			return OaStatus::Error(OaStatusCode::Internal, "SafeTensors entry order is inconsistent");
		}
		auto& entry = it->second;
		auto& info = entry.Info;
		const OaU64 dtypeSize = static_cast<OaU64>(OaScalarSize(info.Dtype));
		if (dtypeSize == 0) {
			return OaStatus::Error(OaStatusCode::DtypeMismatch, OaString("Invalid dtype for entry: ") + info.Name);
		}

		OaU64 count = 1;
		for (OaI64 dim : info.Shape) {
			if (dim == 0) {
				count = 0;
				break;
			}
			const OaU64 uDim = static_cast<OaU64>(dim);
			if (count > std::numeric_limits<OaU64>::max() / uDim) {
				return OaStatus::Error(OaStatusCode::FileCorrupt,
					OaString("Element count overflow for entry: ") + info.Name);
			}
			count *= uDim;
		}
		if (info.Shape.Empty()) count = 1;
		info.ElementCount = count;
		if (count > std::numeric_limits<OaU64>::max() / dtypeSize ||
			count * dtypeSize != info.ByteSize) {
			return OaStatus::Error(OaStatusCode::FileCorrupt,
				OaString("Shape and byte length disagree for entry: ") + info.Name);
		}
		if (entry.DataOffset > dataBytes || info.ByteSize > dataBytes - entry.DataOffset) {
			return OaStatus::Error(OaStatusCode::FileCorrupt,
				OaString("Entry payload is outside file bounds: ") + info.Name);
		}
		sorted.PushBack(&entry);
	}

	std::sort(sorted.begin(), sorted.end(), [](const auto* InA, const auto* InB) {
		if (InA->DataOffset != InB->DataOffset) return InA->DataOffset < InB->DataOffset;
		return InA->Info.ByteSize < InB->Info.ByteSize;
	});

	OaU64 indexedEnd = 0;
	for (const auto* entry : sorted) {
		if (entry->Info.ByteSize == 0) {
			if (entry->DataOffset > indexedEnd) {
				return OaStatus::Error(OaStatusCode::FileCorrupt,
					OaString("Unindexed gap before empty entry: ") + entry->Info.Name);
			}
			continue;
		}
		if (entry->DataOffset < indexedEnd) {
			return OaStatus::Error(OaStatusCode::FileCorrupt,
				OaString("Overlapping entry payload: ") + entry->Info.Name);
		}
		if (entry->DataOffset > indexedEnd) {
			return OaStatus::Error(OaStatusCode::FileCorrupt,
				OaString("SafeTensors data buffer contains a hole before entry: ") + entry->Info.Name);
		}
		indexedEnd = entry->DataOffset + entry->Info.ByteSize;
	}
	if (indexedEnd != dataBytes) {
		return OaStatus::Error(OaStatusCode::FileCorrupt, "SafeTensors data buffer is not entirely indexed");
	}
	return OaStatus::Ok();
}

OaVec<OaWeightInfo> OaSafeTensorsWeightSource::List() const {
	OaVec<OaWeightInfo> result;
	result.Reserve(EntryOrder_.Size());
	for (const auto& name : EntryOrder_) {
		auto it = Entries_.Find(name);
		if (it != Entries_.End()) result.PushBack(it->second.Info);
	}
	return result;
}

const OaWeightInfo* OaSafeTensorsWeightSource::Find(OaStringView InName) const {
	auto it = Entries_.Find(OaString(InName));
	if (it != Entries_.End()) return &it->second.Info;
	return nullptr;
}

OaResult<OaSpan<const OaU8>> OaSafeTensorsWeightSource::Bytes(OaStringView InName) const {
	if (!IsOpen_) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition, "SafeTensors reader is not open");
	}
	auto it = Entries_.Find(OaString(InName));
	const auto* info = it == Entries_.End() ? nullptr : &it->second.Info;
	if (!info) return OaStatus::NotFound(OaString("Entry not found: ") + InName);
	return File_.Slice(DataStart_ + it->second.DataOffset, info->ByteSize);
}

OaStatus OaSafeTensorsWeightSource::Read(
	OaStringView InName,
	OaSpan<OaU8> OutData,
	OaScalarType InTargetDtype
) const {
	if (OutData.Data() == nullptr && OutData.Size() != 0) {
		return OaStatus::InvalidArgument("OaWeightSource::Read: null output buffer");
	}

	const auto* info = Find(InName);
	if (not info) {
		return OaStatus::Error(OaStatusCode::NotFound,
			OaString("Entry not found: ") + InName);
	}

	// Direct copy if no conversion needed
	if (InTargetDtype == info->Dtype) {
		if (OutData.Size() < info->ByteSize) {
			return OaStatus::Error(OaStatusCode::OutOfRange,
				OaString("Output buffer is too small for entry: ") + InName);
		}

		auto bytes = Bytes(InName);
		if (bytes.IsError()) return bytes.GetStatus();
		OaMemcpy(OutData.Data(), bytes->Data(), info->ByteSize);
		return OaStatus::Ok();
	}

	// Dtype conversion needed
	const OaU64 targetDtypeSize = OaScalarSize(InTargetDtype);
	if (targetDtypeSize == 0 || info->ElementCount > std::numeric_limits<OaU64>::max() / targetDtypeSize) {
		return OaStatus::Error(OaStatusCode::DtypeMismatch, "Invalid target dtype");
	}
	OaU64 targetBytes = info->ElementCount * targetDtypeSize;
	
	if (OutData.Size() < targetBytes) {
		return OaStatus::Error(OaStatusCode::OutOfRange,
			OaString("Output buffer is too small after dtype conversion for entry: ") + InName);
	}
	
	auto bytes = Bytes(InName);
	if (bytes.IsError()) return bytes.GetStatus();
	return ConvertDtype(bytes->Data(), OutData.Data(), info->ElementCount, info->Dtype, InTargetDtype);
}
