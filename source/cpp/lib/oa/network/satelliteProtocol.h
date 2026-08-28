#pragma once

#include <oa/core/status.h>
#include <oa/core/types.h>

namespace oa {

enum class SatelliteMessageType : oa::U16 {
#define OA_SATELLITE_MESSAGE(Name, Value) Name = Value,
#define OA_SATELLITE_FIELD(message, Name, Value, Kind, Required, MinBytes, MaxBytes)
#include "satelliteProtocolSchema.inl"
#undef OA_SATELLITE_FIELD
#undef OA_SATELLITE_MESSAGE
};

enum class SatelliteFieldId : oa::U16 {
#define OA_SATELLITE_MESSAGE(Name, Value)
#define OA_SATELLITE_FIELD(message, Name, Value, Kind, Required, MinBytes, MaxBytes) Name = Value,
#include "satelliteProtocolSchema.inl"
#undef OA_SATELLITE_FIELD
#undef OA_SATELLITE_MESSAGE
};

enum class SatelliteFieldKind : oa::U8 {
	U8 = 1,
	U16 = 2,
	U32 = 3,
	U64 = 4,
	F32 = 5,
	String = 6,
	Bytes = 7,
	U64Array = 8,
	I64Array = 9,
};

class SatelliteField {
public:
	SatelliteFieldId id{};
	SatelliteFieldKind kind{};
	oa::Vector<oa::Byte> data;

	[[nodiscard]] static SatelliteField u8(SatelliteFieldId inId, oa::U8 inValue);
	[[nodiscard]] static SatelliteField u16(SatelliteFieldId inId, oa::U16 inValue);
	[[nodiscard]] static SatelliteField u32(SatelliteFieldId inId, oa::U32 inValue);
	[[nodiscard]] static SatelliteField u64(SatelliteFieldId inId, oa::U64 inValue);
	[[nodiscard]] static SatelliteField f32(SatelliteFieldId inId, oa::F32 inValue);
	[[nodiscard]] static SatelliteField string(
		SatelliteFieldId inId, oa::StringView inValue);
	[[nodiscard]] static SatelliteField bytes(
		SatelliteFieldId inId, oa::Span<const oa::Byte> inValue);
	[[nodiscard]] static SatelliteField u64Array(
		SatelliteFieldId inId, oa::Span<const oa::U64> inValue);
	[[nodiscard]] static SatelliteField i64Array(
		SatelliteFieldId inId, oa::Span<const oa::I64> inValue);
};

class SatelliteMessage {
public:
	oa::U16 version = 1;
	SatelliteMessageType type = SatelliteMessageType::Error;
	oa::U32 flags = 0;
	oa::U64 requestId = 0;
	oa::U64 sessionEpoch = 0;
	oa::Vector<SatelliteField> fields;
};

class SatelliteProtocol {
public:
	static constexpr oa::U16 kVersion = 1;
	static constexpr oa::U32 kResponseFlag = 1U;
	static constexpr oa::U32 kKnownFlags = kResponseFlag;
	static constexpr oa::U32 kHeaderBytes = 64U;
	static constexpr oa::U32 kMaxPayloadBytes = 1024U * 1024U;
	static constexpr oa::U32 kMaxFields = 64U;

	[[nodiscard]] static oa::Status validate(const SatelliteMessage& inMessage);
	[[nodiscard]] static oa::Result<oa::Vector<oa::Byte>> encode(
		const SatelliteMessage& inMessage);
	[[nodiscard]] static oa::Result<SatelliteMessage> decode(
		oa::Span<const oa::Byte> inBytes);
	// Stable non-cryptographic digest used for wire corruption detection and
	// opaque content identity. Authentication is supplied separately by the
	// application-owned satellite service.
	[[nodiscard]] static oa::Array<oa::Byte, 32> stableDigest(
		oa::Span<const oa::Byte> inFirst,
		oa::Span<const oa::Byte> inSecond = {});

	[[nodiscard]] static const SatelliteField* findField(
		const SatelliteMessage& inMessage, SatelliteFieldId inId);
	[[nodiscard]] static oa::Result<oa::U8> readU8(const SatelliteField& inField);
	[[nodiscard]] static oa::Result<oa::U16> readU16(const SatelliteField& inField);
	[[nodiscard]] static oa::Result<oa::U32> readU32(const SatelliteField& inField);
	[[nodiscard]] static oa::Result<oa::U64> readU64(const SatelliteField& inField);
	[[nodiscard]] static oa::Result<oa::F32> readF32(const SatelliteField& inField);
	[[nodiscard]] static oa::Result<oa::String> readString(const SatelliteField& inField);
	[[nodiscard]] static oa::Result<oa::Vector<oa::U64>> readU64Array(
		const SatelliteField& inField);
	[[nodiscard]] static oa::Result<oa::Vector<oa::I64>> readI64Array(
		const SatelliteField& inField);
};

} // namespace oa
