#include "satelliteProtocol.h"

#include <oa/core/memory.h>
#include <oa/core/std/limits.h>

namespace {

constexpr oa::Byte kMagic[] = {'O', 'A', 'S', 'P'};

class SatelliteMessageSchema {
public:
	oa::SatelliteMessageType type;
};

class SatelliteFieldSchema {
public:
	oa::SatelliteMessageType message;
	oa::SatelliteFieldId id;
	oa::SatelliteFieldKind kind;
	oa::Bool required;
	oa::U32 minBytes;
	oa::U32 maxBytes;
};

constexpr SatelliteMessageSchema kMessageSchemas[] = {
#define OA_SATELLITE_MESSAGE(Name, Value) {oa::SatelliteMessageType::Name},
#define OA_SATELLITE_FIELD(message, Name, Value, Kind, Required, MinBytes, MaxBytes)
#include "satelliteProtocolSchema.inl"
#undef OA_SATELLITE_FIELD
#undef OA_SATELLITE_MESSAGE
};

constexpr SatelliteFieldSchema kFieldSchemas[] = {
#define OA_SATELLITE_MESSAGE(Name, Value)
#define OA_SATELLITE_FIELD(message, Name, Value, Kind, Required, MinBytes, MaxBytes) \
	{oa::SatelliteMessageType::message, oa::SatelliteFieldId::Name, \
		oa::SatelliteFieldKind::Kind, Required, MinBytes, MaxBytes},
#include "satelliteProtocolSchema.inl"
#undef OA_SATELLITE_FIELD
#undef OA_SATELLITE_MESSAGE
};

[[nodiscard]] const SatelliteMessageSchema* findMessageSchema(
	oa::SatelliteMessageType inType)
{
	for (const auto& schema : kMessageSchemas) {
		if (schema.type == inType) return &schema;
	}
	return nullptr;
}

[[nodiscard]] const SatelliteFieldSchema* findFieldSchema(
	oa::SatelliteMessageType inMessage, oa::SatelliteFieldId inId)
{
	for (const auto& schema : kFieldSchemas) {
		if (schema.message == inMessage and schema.id == inId) return &schema;
	}
	return nullptr;
}

void appendU8(oa::Vec<oa::Byte>& out, oa::U8 inValue) {
	out.pushBack(inValue);
}

void appendU16(oa::Vec<oa::Byte>& out, oa::U16 inValue) {
	out.pushBack(static_cast<oa::Byte>(inValue & 0xffU));
	out.pushBack(static_cast<oa::Byte>((inValue >> 8U) & 0xffU));
}

void appendU32(oa::Vec<oa::Byte>& out, oa::U32 inValue) {
	for (oa::U32 shift = 0; shift < 32U; shift += 8U) {
		out.pushBack(static_cast<oa::Byte>((inValue >> shift) & 0xffU));
	}
}

void appendU64(oa::Vec<oa::Byte>& out, oa::U64 inValue) {
	for (oa::U32 shift = 0; shift < 64U; shift += 8U) {
		out.pushBack(static_cast<oa::Byte>((inValue >> shift) & 0xffU));
	}
}

[[nodiscard]] oa::U16 readU16Le(const oa::Byte* inData) {
	return static_cast<oa::U16>(static_cast<oa::U16>(inData[0])
		| (static_cast<oa::U16>(inData[1]) << 8U));
}

[[nodiscard]] oa::U32 readU32Le(const oa::Byte* inData) {
	oa::U32 value = 0;
	for (oa::U32 i = 0; i < 4U; ++i) {
		value |= static_cast<oa::U32>(inData[i]) << (i * 8U);
	}
	return value;
}

[[nodiscard]] oa::U64 readU64Le(const oa::Byte* inData) {
	oa::U64 value = 0;
	for (oa::U32 i = 0; i < 8U; ++i) {
		value |= static_cast<oa::U64>(inData[i]) << (i * 8U);
	}
	return value;
}

[[nodiscard]] oa::Bool isValidUtf8(oa::Span<const oa::Byte> inBytes) {
	oa::Usize i = 0;
	while (i < inBytes.size()) {
		const oa::U8 first = inBytes[i];
		if (first <= 0x7fU) {
			++i;
			continue;
		}
		oa::U32 continuation = 0;
		oa::U32 codepoint = 0;
		if (first >= 0xc2U and first <= 0xdfU) {
			continuation = 1;
			codepoint = first & 0x1fU;
		} else if (first >= 0xe0U and first <= 0xefU) {
			continuation = 2;
			codepoint = first & 0x0fU;
		} else if (first >= 0xf0U and first <= 0xf4U) {
			continuation = 3;
			codepoint = first & 0x07U;
		} else {
			return false;
		}
		if (i + continuation >= inBytes.size()) return false;
		for (oa::U32 j = 1; j <= continuation; ++j) {
			const oa::U8 byte = inBytes[i + j];
			if ((byte & 0xc0U) != 0x80U) return false;
			codepoint = (codepoint << 6U) | (byte & 0x3fU);
		}
		if ((continuation == 2U and codepoint < 0x800U)
			or (continuation == 3U and codepoint < 0x10000U)
			or codepoint > 0x10ffffU
			or (codepoint >= 0xd800U and codepoint <= 0xdfffU))
		{
			return false;
		}
		i += continuation + 1U;
	}
	return true;
}

[[nodiscard]] oa::Status validateFieldData(
	const oa::SatelliteField& inField, const SatelliteFieldSchema& inSchema)
{
	if (inField.kind != inSchema.kind) {
		return oa::Status::invalidArgument("satellite protocol: field kind mismatch");
	}
	const oa::Usize bytes = inField.data.size();
	if (bytes < inSchema.minBytes or bytes > inSchema.maxBytes) {
		return oa::Status::invalidArgument("satellite protocol: field size is outside schema bounds");
	}
	if ((inField.kind == oa::SatelliteFieldKind::U64Array
			or inField.kind == oa::SatelliteFieldKind::I64Array)
		and bytes % sizeof(oa::U64) != 0U)
	{
		return oa::Status::invalidArgument("satellite protocol: array field has a partial element");
	}
	if (inField.kind == oa::SatelliteFieldKind::String
		and not isValidUtf8(oa::Span<const oa::Byte>(inField.data.data(), bytes)))
	{
		return oa::Status::invalidArgument("satellite protocol: string field is not canonical UTF-8");
	}
	return oa::Status::ok();
}

[[nodiscard]] oa::Status requireKindAndBytes(
	const oa::SatelliteField& inField, oa::SatelliteFieldKind inKind, oa::Usize inBytes)
{
	if (inField.kind != inKind or inField.data.size() != inBytes) {
		return oa::Status::invalidArgument("satellite protocol: typed field access mismatch");
	}
	return oa::Status::ok();
}

[[nodiscard]] oa::Array<oa::Byte, 32> computeChecksum(
	oa::Span<const oa::Byte> inHeaderPrefix, oa::Span<const oa::Byte> inPayload)
{
	return oa::SatelliteProtocol::stableDigest(inHeaderPrefix, inPayload);
}

} // namespace

oa::SatelliteField oa::SatelliteField::u8(oa::SatelliteFieldId inId, oa::U8 inValue) {
	oa::SatelliteField field;
	field.id = inId;
	field.kind = oa::SatelliteFieldKind::U8;
	appendU8(field.data, inValue);
	return field;
}

oa::SatelliteField oa::SatelliteField::u16(oa::SatelliteFieldId inId, oa::U16 inValue) {
	oa::SatelliteField field;
	field.id = inId;
	field.kind = oa::SatelliteFieldKind::U16;
	appendU16(field.data, inValue);
	return field;
}

oa::SatelliteField oa::SatelliteField::u32(oa::SatelliteFieldId inId, oa::U32 inValue) {
	oa::SatelliteField field;
	field.id = inId;
	field.kind = oa::SatelliteFieldKind::U32;
	appendU32(field.data, inValue);
	return field;
}

oa::SatelliteField oa::SatelliteField::u64(oa::SatelliteFieldId inId, oa::U64 inValue) {
	oa::SatelliteField field;
	field.id = inId;
	field.kind = oa::SatelliteFieldKind::U64;
	appendU64(field.data, inValue);
	return field;
}

oa::SatelliteField oa::SatelliteField::f32(oa::SatelliteFieldId inId, oa::F32 inValue) {
	oa::U32 bits = 0;
	oa::memcpy(&bits, &inValue, sizeof(bits));
	oa::SatelliteField field;
	field.id = inId;
	field.kind = oa::SatelliteFieldKind::F32;
	appendU32(field.data, bits);
	return field;
}

oa::SatelliteField oa::SatelliteField::string(
	oa::SatelliteFieldId inId, oa::StringView inValue)
{
	oa::SatelliteField field;
	field.id = inId;
	field.kind = oa::SatelliteFieldKind::String;
	field.data.append(reinterpret_cast<const oa::Byte*>(inValue.data()), inValue.size());
	return field;
}

oa::SatelliteField oa::SatelliteField::bytes(
	oa::SatelliteFieldId inId, oa::Span<const oa::Byte> inValue)
{
	oa::SatelliteField field;
	field.id = inId;
	field.kind = oa::SatelliteFieldKind::Bytes;
	field.data.append(inValue.data(), inValue.size());
	return field;
}

oa::SatelliteField oa::SatelliteField::u64Array(
	oa::SatelliteFieldId inId, oa::Span<const oa::U64> inValue)
{
	oa::SatelliteField field;
	field.id = inId;
	field.kind = oa::SatelliteFieldKind::U64Array;
	for (oa::U64 value : inValue) appendU64(field.data, value);
	return field;
}

oa::SatelliteField oa::SatelliteField::i64Array(
	oa::SatelliteFieldId inId, oa::Span<const oa::I64> inValue)
{
	oa::SatelliteField field;
	field.id = inId;
	field.kind = oa::SatelliteFieldKind::I64Array;
	for (oa::I64 value : inValue) appendU64(field.data, static_cast<oa::U64>(value));
	return field;
}

oa::Status oa::SatelliteProtocol::validate(const oa::SatelliteMessage& inMessage) {
	if (inMessage.version != kVersion) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"satellite protocol: unsupported protocol version");
	}
	if (findMessageSchema(inMessage.type) == nullptr) {
		return oa::Status::invalidArgument("satellite protocol: unknown message type");
	}
	if ((inMessage.flags & ~kKnownFlags) != 0U) {
		return oa::Status::invalidArgument("satellite protocol: unknown header flags");
	}
	if (inMessage.requestId == 0U) {
		return oa::Status::invalidArgument("satellite protocol: request id must be non-zero");
	}
	if (inMessage.fields.size() > kMaxFields) {
		return oa::Status::error(oa::StatusCode::ResourceExhausted,
			"satellite protocol: too many fields");
	}

	oa::U16 previousId = 0;
	for (const auto& field : inMessage.fields) {
		const oa::U16 fieldId = static_cast<oa::U16>(field.id);
		if (fieldId == 0U or fieldId <= previousId) {
			return oa::Status::invalidArgument(
				"satellite protocol: fields must be unique and strictly ordered");
		}
		previousId = fieldId;
		const auto* schema = findFieldSchema(inMessage.type, field.id);
		if (schema == nullptr) {
			return oa::Status::invalidArgument("satellite protocol: field is not in the message schema");
		}
		const auto status = validateFieldData(field, *schema);
		if (status.isError()) return status;
	}

	for (const auto& schema : kFieldSchemas) {
		if (schema.message != inMessage.type or not schema.required) continue;
		oa::Bool found = false;
		for (const auto& field : inMessage.fields) {
			if (field.id == schema.id) {
				found = true;
				break;
			}
		}
		if (not found) {
			return oa::Status::invalidArgument("satellite protocol: required field is missing");
		}
	}
	return oa::Status::ok();
}

oa::Result<oa::Vec<oa::Byte>> oa::SatelliteProtocol::encode(
	const oa::SatelliteMessage& inMessage)
{
	const auto status = validate(inMessage);
	if (status.isError()) return status;

	oa::Vec<oa::Byte> payload;
	appendU16(payload, static_cast<oa::U16>(inMessage.fields.size()));
	for (const auto& field : inMessage.fields) {
		if (field.data.size() > oa::Limits<oa::U32>::max()) {
			return oa::Status::error(oa::StatusCode::ResourceExhausted,
				"satellite protocol: field is too large");
		}
		appendU16(payload, static_cast<oa::U16>(field.id));
		appendU8(payload, static_cast<oa::U8>(field.kind));
		appendU8(payload, 0U);
		appendU32(payload, static_cast<oa::U32>(field.data.size()));
		payload.append(field.data.data(), field.data.size());
	}
	if (payload.size() > kMaxPayloadBytes) {
		return oa::Status::error(oa::StatusCode::ResourceExhausted,
			"satellite protocol: encoded payload exceeds the session limit");
	}

	oa::Vec<oa::Byte> encoded;
	encoded.reserve(kHeaderBytes + payload.size());
	encoded.append(kMagic, sizeof(kMagic));
	appendU16(encoded, inMessage.version);
	appendU16(encoded, static_cast<oa::U16>(inMessage.type));
	appendU32(encoded, inMessage.flags);
	appendU64(encoded, inMessage.requestId);
	appendU64(encoded, inMessage.sessionEpoch);
	appendU32(encoded, static_cast<oa::U32>(payload.size()));
	const auto checksum = computeChecksum(
		oa::Span<const oa::Byte>(encoded.data(), encoded.size()),
		oa::Span<const oa::Byte>(payload.data(), payload.size()));
	encoded.append(checksum.data(), checksum.size());
	encoded.append(payload.data(), payload.size());
	return encoded;
}

oa::Result<oa::SatelliteMessage> oa::SatelliteProtocol::decode(
	oa::Span<const oa::Byte> inBytes)
{
	if (inBytes.size() < kHeaderBytes) {
		return oa::Status::error(oa::StatusCode::DataLoss,
			"satellite protocol: truncated header");
	}
	for (oa::Usize i = 0; i < sizeof(kMagic); ++i) {
		if (inBytes[i] != kMagic[i]) {
			return oa::Status::error(oa::StatusCode::DataLoss,
				"satellite protocol: invalid magic");
		}
	}
	const oa::U32 payloadBytes = readU32Le(inBytes.data() + 28U);
	if (payloadBytes > kMaxPayloadBytes) {
		return oa::Status::error(oa::StatusCode::ResourceExhausted,
			"satellite protocol: payload exceeds the session limit");
	}
	if (inBytes.size() != static_cast<oa::Usize>(kHeaderBytes) + payloadBytes) {
		return oa::Status::error(oa::StatusCode::DataLoss,
			"satellite protocol: payload length does not match the frame");
	}
	const auto payload = inBytes.subSpan(kHeaderBytes, payloadBytes);
	const auto expectedChecksum = computeChecksum(
		inBytes.first(32U), payload);
	oa::U8 checksumDifference = 0;
	for (oa::Usize i = 0; i < expectedChecksum.size(); ++i) {
		checksumDifference |= static_cast<oa::U8>(expectedChecksum[i] ^ inBytes[32U + i]);
	}
	if (checksumDifference != 0U) {
		return oa::Status::error(oa::StatusCode::DataLoss,
			"satellite protocol: checksum mismatch");
	}
	if (payload.size() < 2U) {
		return oa::Status::error(oa::StatusCode::DataLoss,
			"satellite protocol: truncated field count");
	}

	oa::SatelliteMessage message;
	message.version = readU16Le(inBytes.data() + 4U);
	message.type = static_cast<oa::SatelliteMessageType>(readU16Le(inBytes.data() + 6U));
	message.flags = readU32Le(inBytes.data() + 8U);
	message.requestId = readU64Le(inBytes.data() + 12U);
	message.sessionEpoch = readU64Le(inBytes.data() + 20U);

	const oa::U16 fieldCount = readU16Le(payload.data());
	if (fieldCount > kMaxFields) {
		return oa::Status::error(oa::StatusCode::ResourceExhausted,
			"satellite protocol: field count exceeds the session limit");
	}
	oa::Usize offset = 2U;
	message.fields.reserve(fieldCount);
	for (oa::U16 i = 0; i < fieldCount; ++i) {
		if (payload.size() - offset < 8U) {
			return oa::Status::error(oa::StatusCode::DataLoss,
				"satellite protocol: truncated field header");
		}
		const oa::U16 id = readU16Le(payload.data() + offset);
		const oa::U8 kind = payload[offset + 2U];
		const oa::U8 reserved = payload[offset + 3U];
		const oa::U32 fieldBytes = readU32Le(payload.data() + offset + 4U);
		offset += 8U;
		if (reserved != 0U) {
			return oa::Status::error(oa::StatusCode::DataLoss,
				"satellite protocol: non-zero reserved field byte");
		}
		if (fieldBytes > payload.size() - offset) {
			return oa::Status::error(oa::StatusCode::DataLoss,
				"satellite protocol: truncated field data");
		}
		oa::SatelliteField field;
		field.id = static_cast<oa::SatelliteFieldId>(id);
		field.kind = static_cast<oa::SatelliteFieldKind>(kind);
		field.data.append(payload.data() + offset, fieldBytes);
		message.fields.pushBack(oa::move(field));
		offset += fieldBytes;
	}
	if (offset != payload.size()) {
		return oa::Status::error(oa::StatusCode::DataLoss,
			"satellite protocol: trailing payload bytes");
	}
	const auto status = validate(message);
	if (status.isError()) return status;
	return message;
}

oa::Array<oa::Byte, 32> oa::SatelliteProtocol::stableDigest(
	oa::Span<const oa::Byte> inFirst,
	oa::Span<const oa::Byte> inSecond)
{
	oa::Array<oa::U64, 4> lanes{};
	lanes[0] = 0xcbf29ce484222325ULL;
	lanes[1] = 0x84222325cbf29ce4ULL;
	lanes[2] = 0x9e3779b97f4a7c15ULL;
	lanes[3] = 0xd6e8feb86659fd93ULL;
	constexpr oa::U64 primes[] = {
		0x00000100000001b3ULL,
		0x9e3779b185ebca87ULL,
		0xc2b2ae3d27d4eb4fULL,
		0x165667b19e3779f9ULL,
	};
	auto mixByte = [&](oa::Byte inByte) {
		for (oa::Usize lane = 0; lane < lanes.size(); ++lane) {
			lanes[lane] ^= static_cast<oa::U64>(inByte)
				+ (static_cast<oa::U64>(lane) << 8U);
			lanes[lane] *= primes[lane];
			lanes[lane] ^= lanes[lane] >> 32U;
		}
	};
	auto mixU64 = [&](oa::U64 inValue) {
		for (oa::U32 byte = 0; byte < 8U; ++byte) {
			mixByte(static_cast<oa::Byte>(inValue >> (byte * 8U)));
		}
	};
	auto mixSpan = [&](oa::Span<const oa::Byte> inBytes) {
		for (const oa::Byte byte : inBytes) mixByte(byte);
	};
	mixU64(inFirst.size());
	mixSpan(inFirst);
	mixByte(0xa5U);
	mixU64(inSecond.size());
	mixSpan(inSecond);

	oa::Array<oa::Byte, 32> digest{};
	for (oa::Usize lane = 0; lane < lanes.size(); ++lane) {
		oa::U64 value = lanes[lane];
		value ^= value >> 33U;
		value *= 0xff51afd7ed558ccdULL;
		value ^= value >> 33U;
		value *= 0xc4ceb9fe1a85ec53ULL;
		value ^= value >> 33U;
		for (oa::U32 byte = 0; byte < 8U; ++byte) {
			digest[lane * 8U + byte] = static_cast<oa::Byte>(value >> (byte * 8U));
		}
	}
	return digest;
}

const oa::SatelliteField* oa::SatelliteProtocol::findField(
	const oa::SatelliteMessage& inMessage, oa::SatelliteFieldId inId)
{
	for (const auto& field : inMessage.fields) {
		if (field.id == inId) return &field;
	}
	return nullptr;
}

oa::Result<oa::U8> oa::SatelliteProtocol::readU8(const oa::SatelliteField& inField) {
	const auto status = requireKindAndBytes(inField, oa::SatelliteFieldKind::U8, 1U);
	if (status.isError()) return status;
	return inField.data[0];
}

oa::Result<oa::U16> oa::SatelliteProtocol::readU16(const oa::SatelliteField& inField) {
	const auto status = requireKindAndBytes(inField, oa::SatelliteFieldKind::U16, 2U);
	if (status.isError()) return status;
	return readU16Le(inField.data.data());
}

oa::Result<oa::U32> oa::SatelliteProtocol::readU32(const oa::SatelliteField& inField) {
	const auto status = requireKindAndBytes(inField, oa::SatelliteFieldKind::U32, 4U);
	if (status.isError()) return status;
	return readU32Le(inField.data.data());
}

oa::Result<oa::U64> oa::SatelliteProtocol::readU64(const oa::SatelliteField& inField) {
	const auto status = requireKindAndBytes(inField, oa::SatelliteFieldKind::U64, 8U);
	if (status.isError()) return status;
	return readU64Le(inField.data.data());
}

oa::Result<oa::F32> oa::SatelliteProtocol::readF32(const oa::SatelliteField& inField) {
	const auto status = requireKindAndBytes(inField, oa::SatelliteFieldKind::F32, 4U);
	if (status.isError()) return status;
	const oa::U32 bits = readU32Le(inField.data.data());
	oa::F32 value = 0.0F;
	oa::memcpy(&value, &bits, sizeof(value));
	return value;
}

oa::Result<oa::String> oa::SatelliteProtocol::readString(const oa::SatelliteField& inField) {
	if (inField.kind != oa::SatelliteFieldKind::String) {
		return oa::Status::invalidArgument("satellite protocol: typed field access mismatch");
	}
	return oa::String(reinterpret_cast<const char*>(inField.data.data()), inField.data.size());
}

oa::Result<oa::Vec<oa::U64>> oa::SatelliteProtocol::readU64Array(
	const oa::SatelliteField& inField)
{
	if (inField.kind != oa::SatelliteFieldKind::U64Array
		or inField.data.size() % sizeof(oa::U64) != 0U)
	{
		return oa::Status::invalidArgument("satellite protocol: typed field access mismatch");
	}
	oa::Vec<oa::U64> values;
	values.reserve(inField.data.size() / sizeof(oa::U64));
	for (oa::Usize offset = 0; offset < inField.data.size(); offset += sizeof(oa::U64)) {
		values.pushBack(readU64Le(inField.data.data() + offset));
	}
	return values;
}

oa::Result<oa::Vec<oa::I64>> oa::SatelliteProtocol::readI64Array(
	const oa::SatelliteField& inField)
{
	if (inField.kind != oa::SatelliteFieldKind::I64Array
		or inField.data.size() % sizeof(oa::I64) != 0U)
	{
		return oa::Status::invalidArgument("satellite protocol: typed field access mismatch");
	}
	oa::Vec<oa::I64> values;
	values.reserve(inField.data.size() / sizeof(oa::I64));
	for (oa::Usize offset = 0; offset < inField.data.size(); offset += sizeof(oa::I64)) {
		values.pushBack(static_cast<oa::I64>(readU64Le(inField.data.data() + offset)));
	}
	return values;
}
