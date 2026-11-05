//! Private authenticated satellite wire protocol (codec only).
//!
//! Donor: `source/cpp/lib/oa/net/satelliteProtocol.{h,cpp}` and
//! `satelliteProtocolSchema.inl`. This module owns the canonical v1 wire
//! format: a 64-byte header, schema-bounded typed fields, and a stable
//! non-cryptographic checksum for corruption detection. Peer authentication,
//! session state, and remote compute semantics belong to the session layer
//! and are not implemented here.

use crate::{Error, Result};

const MAGIC: [u8; 4] = *b"OASP";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub(crate) enum SatelliteMessageType {
	Hello = 1,
	HelloReply = 2,
	PutObject = 3,
	DropObject = 4,
	ExecuteNamed = 5,
	Wait = 6,
	Poll = 7,
	Cancel = 8,
	GetResult = 9,
	Error = 10,
	Abort = 11,
	Close = 12,
	Result = 13,
}

impl SatelliteMessageType {
	const ALL: [Self; 13] = [
		Self::Hello,
		Self::HelloReply,
		Self::PutObject,
		Self::DropObject,
		Self::ExecuteNamed,
		Self::Wait,
		Self::Poll,
		Self::Cancel,
		Self::GetResult,
		Self::Error,
		Self::Abort,
		Self::Close,
		Self::Result,
	];

	fn from_u16(value: u16) -> Option<Self> {
		Self::ALL
			.iter()
			.copied()
			.find(|message| *message as u16 == value)
	}
}

/// Wire field identities. Field ids are scoped to their owning message type,
/// matching the donor manifest where per-message ids restart at one.
pub(crate) mod field_id {
	pub const HELLO_PROTOCOL_MIN: u16 = 1;
	pub const HELLO_PROTOCOL_MAX: u16 = 2;
	pub const HELLO_CLIENT_NONCE: u16 = 3;
	pub const HELLO_AUTH_PROOF: u16 = 4;
	pub const HELLO_BUILD_HASH: u16 = 5;
	pub const HELLO_SCHEMA_HASH: u16 = 6;
	pub const HELLO_MAX_PAYLOAD_BYTES: u16 = 7;
	pub const HELLO_MAX_RESIDENT_BYTES: u16 = 8;
	pub const HELLO_MAX_OBJECTS: u16 = 9;
	pub const HELLO_MAX_INFLIGHT: u16 = 10;

	pub const HELLO_REPLY_PROTOCOL: u16 = 1;
	pub const HELLO_REPLY_SERVER_NONCE: u16 = 2;
	pub const HELLO_REPLY_AUTH_PROOF: u16 = 3;
	pub const HELLO_REPLY_SESSION_EPOCH: u16 = 4;
	pub const HELLO_REPLY_DEVICE_NAME: u16 = 5;
	pub const HELLO_REPLY_MAX_PAYLOAD_BYTES: u16 = 6;
	pub const HELLO_REPLY_MAX_RESIDENT_BYTES: u16 = 7;
	pub const HELLO_REPLY_MAX_OBJECTS: u16 = 8;
	pub const HELLO_REPLY_MAX_INFLIGHT: u16 = 9;

	pub const PUT_OBJECT_ID: u16 = 1;
	pub const PUT_OBJECT_DTYPE: u16 = 2;
	pub const PUT_OBJECT_SHAPE: u16 = 3;
	pub const PUT_OBJECT_DATA: u16 = 4;
	pub const PUT_OBJECT_CONTENT_HASH: u16 = 5;
	pub const PUT_OBJECT_VERSION: u16 = 6;

	pub const DROP_OBJECT_ID: u16 = 1;

	pub const EXECUTE_OPERATION: u16 = 1;
	pub const EXECUTE_INPUT_OBJECT_IDS: u16 = 2;
	pub const EXECUTE_OUTPUT_OBJECT_ID: u16 = 3;
	pub const EXECUTE_ARGUMENTS: u16 = 4;
	pub const EXECUTE_EXPECTED_VERSION: u16 = 5;
	pub const EXECUTE_EXPECTED_HASH: u16 = 6;

	pub const WAIT_REQUEST_ID: u16 = 1;
	pub const POLL_REQUEST_ID: u16 = 1;
	pub const CANCEL_REQUEST_ID: u16 = 1;
	pub const GET_RESULT_REQUEST_ID: u16 = 1;

	pub const ERROR_STATUS_CODE: u16 = 1;
	pub const ERROR_MESSAGE: u16 = 2;
	pub const ERROR_POISONED: u16 = 3;

	pub const ABORT_REASON: u16 = 1;

	pub const RESULT_REQUEST_ID: u16 = 1;
	pub const RESULT_COMPLETE: u16 = 2;
	pub const RESULT_STATUS_CODE: u16 = 3;
	pub const RESULT_BYTES: u16 = 4;
	pub const RESULT_PROFILE: u16 = 5;
	pub const RESULT_OBJECT_ID: u16 = 6;
	pub const RESULT_OBJECT_VERSION: u16 = 7;
	pub const RESULT_OBJECT_HASH: u16 = 8;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum SatelliteFieldKind {
	// Wire bytes outside the declared kind space decode to `Invalid`, which
	// never matches a schema kind — mirroring the donor's raw enum carry.
	Invalid = 0,
	U8 = 1,
	U16 = 2,
	U32 = 3,
	U64 = 4,
	F32 = 5,
	String = 6,
	Bytes = 7,
	U64Array = 8,
	I64Array = 9,
}

impl SatelliteFieldKind {
	/// Donor manifest spelling used by the canonical schema-identity string.
	const fn schema_name(self) -> &'static str {
		match self {
			Self::Invalid => "Invalid",
			Self::U8 => "U8",
			Self::U16 => "U16",
			Self::U32 => "U32",
			Self::U64 => "U64",
			Self::F32 => "F32",
			Self::String => "String",
			Self::Bytes => "Bytes",
			Self::U64Array => "U64Array",
			Self::I64Array => "I64Array",
		}
	}

	fn from_u8(value: u8) -> Self {
		match value {
			1 => Self::U8,
			2 => Self::U16,
			3 => Self::U32,
			4 => Self::U64,
			5 => Self::F32,
			6 => Self::String,
			7 => Self::Bytes,
			8 => Self::U64Array,
			9 => Self::I64Array,
			_ => Self::Invalid,
		}
	}
}

#[derive(Clone, Debug)]
pub(crate) struct SatelliteField {
	pub id: u16,
	pub kind: SatelliteFieldKind,
	pub data: Vec<u8>,
}

impl SatelliteField {
	pub fn u8(id: u16, value: u8) -> Self {
		Self {
			id,
			kind: SatelliteFieldKind::U8,
			data: vec![value],
		}
	}

	pub fn u16(id: u16, value: u16) -> Self {
		Self {
			id,
			kind: SatelliteFieldKind::U16,
			data: value.to_le_bytes().to_vec(),
		}
	}

	pub fn u32(id: u16, value: u32) -> Self {
		Self {
			id,
			kind: SatelliteFieldKind::U32,
			data: value.to_le_bytes().to_vec(),
		}
	}

	pub fn u64(id: u16, value: u64) -> Self {
		Self {
			id,
			kind: SatelliteFieldKind::U64,
			data: value.to_le_bytes().to_vec(),
		}
	}

	pub fn f32(id: u16, value: f32) -> Self {
		Self {
			id,
			kind: SatelliteFieldKind::F32,
			data: value.to_bits().to_le_bytes().to_vec(),
		}
	}

	pub fn string(id: u16, value: &str) -> Self {
		Self {
			id,
			kind: SatelliteFieldKind::String,
			data: value.as_bytes().to_vec(),
		}
	}

	pub fn bytes(id: u16, value: &[u8]) -> Self {
		Self {
			id,
			kind: SatelliteFieldKind::Bytes,
			data: value.to_vec(),
		}
	}

	pub fn u64_array(id: u16, value: &[u64]) -> Self {
		let mut data = Vec::with_capacity(value.len() * 8);
		for element in value {
			data.extend_from_slice(&element.to_le_bytes());
		}
		Self {
			id,
			kind: SatelliteFieldKind::U64Array,
			data,
		}
	}

	pub fn i64_array(id: u16, value: &[i64]) -> Self {
		let mut data = Vec::with_capacity(value.len() * 8);
		for element in value {
			data.extend_from_slice(&element.to_le_bytes());
		}
		Self {
			id,
			kind: SatelliteFieldKind::I64Array,
			data,
		}
	}
}

#[derive(Clone, Debug)]
pub(crate) struct SatelliteMessage {
	pub version: u16,
	pub message_type: SatelliteMessageType,
	pub flags: u32,
	pub request_id: u64,
	pub session_epoch: u64,
	pub fields: Vec<SatelliteField>,
}

impl Default for SatelliteMessage {
	fn default() -> Self {
		Self {
			version: SatelliteProtocol::VERSION,
			message_type: SatelliteMessageType::Error,
			flags: 0,
			request_id: 0,
			session_epoch: 0,
			fields: Vec::new(),
		}
	}
}

struct FieldSchema {
	message: SatelliteMessageType,
	/// Donor `.inl` message spelling used by the schema-identity string.
	message_name: &'static str,
	id: u16,
	/// Donor `.inl` field spelling used by the schema-identity string.
	name: &'static str,
	kind: SatelliteFieldKind,
	required: bool,
	min_bytes: usize,
	max_bytes: usize,
}

macro_rules! field_schemas {
	($($message:ident($message_name:literal) { $($id:ident, $name:literal, $kind:ident, $required:expr, $min:expr, $max:expr);* $(;)? })*) => {
		&[$(
			$(
				FieldSchema {
					message: SatelliteMessageType::$message,
					message_name: $message_name,
					id: field_id::$id,
					name: $name,
					kind: SatelliteFieldKind::$kind,
					required: $required,
					min_bytes: $min,
					max_bytes: $max,
				},
			)*
		)*]
	};
}

/// Ordered wire message manifest with donor `.inl` spellings. The schema
/// identity string and hash are derived from this manifest.
const MESSAGE_MANIFEST: &[(SatelliteMessageType, &str)] = &[
	(SatelliteMessageType::Hello, "Hello"),
	(SatelliteMessageType::HelloReply, "HelloReply"),
	(SatelliteMessageType::PutObject, "PutObject"),
	(SatelliteMessageType::DropObject, "dropObject"),
	(SatelliteMessageType::ExecuteNamed, "ExecuteNamed"),
	(SatelliteMessageType::Wait, "wait"),
	(SatelliteMessageType::Poll, "poll"),
	(SatelliteMessageType::Cancel, "cancel"),
	(SatelliteMessageType::GetResult, "getResult"),
	(SatelliteMessageType::Error, "Error"),
	(SatelliteMessageType::Abort, "abort"),
	(SatelliteMessageType::Close, "Close"),
	(SatelliteMessageType::Result, "result"),
];

const FIELD_SCHEMAS: &[FieldSchema] = field_schemas! {
	Hello("Hello") {
		HELLO_PROTOCOL_MIN, "HelloProtocolMin", U16, true, 2, 2;
		HELLO_PROTOCOL_MAX, "HelloProtocolMax", U16, true, 2, 2;
		HELLO_CLIENT_NONCE, "HelloClientNonce", Bytes, true, 32, 32;
		HELLO_AUTH_PROOF, "HelloAuthProof", Bytes, true, 32, 32;
		HELLO_BUILD_HASH, "HelloBuildHash", Bytes, true, 32, 32;
		HELLO_SCHEMA_HASH, "HelloSchemaHash", Bytes, true, 32, 32;
		HELLO_MAX_PAYLOAD_BYTES, "HelloMaxPayloadBytes", U32, true, 4, 4;
		HELLO_MAX_RESIDENT_BYTES, "HelloMaxResidentBytes", U64, true, 8, 8;
		HELLO_MAX_OBJECTS, "HelloMaxObjects", U32, true, 4, 4;
		HELLO_MAX_INFLIGHT, "HelloMaxInflight", U32, true, 4, 4;
	}
	HelloReply("HelloReply") {
		HELLO_REPLY_PROTOCOL, "HelloReplyProtocol", U16, true, 2, 2;
		HELLO_REPLY_SERVER_NONCE, "HelloReplyServerNonce", Bytes, true, 32, 32;
		HELLO_REPLY_AUTH_PROOF, "HelloReplyAuthProof", Bytes, true, 32, 32;
		HELLO_REPLY_SESSION_EPOCH, "HelloReplySessionEpoch", U64, true, 8, 8;
		HELLO_REPLY_DEVICE_NAME, "HelloReplyDeviceName", String, true, 1, 256;
		HELLO_REPLY_MAX_PAYLOAD_BYTES, "HelloReplyMaxPayloadBytes", U32, true, 4, 4;
		HELLO_REPLY_MAX_RESIDENT_BYTES, "HelloReplyMaxResidentBytes", U64, true, 8, 8;
		HELLO_REPLY_MAX_OBJECTS, "HelloReplyMaxObjects", U32, true, 4, 4;
		HELLO_REPLY_MAX_INFLIGHT, "HelloReplyMaxInflight", U32, true, 4, 4;
	}
	PutObject("PutObject") {
		PUT_OBJECT_ID, "PutObjectId", U64, true, 8, 8;
		PUT_OBJECT_DTYPE, "PutObjectDtype", U8, true, 1, 1;
		PUT_OBJECT_SHAPE, "PutObjectShape", I64Array, true, 8, 64;
		PUT_OBJECT_DATA, "PutObjectData", Bytes, true, 0, 1048576;
		PUT_OBJECT_CONTENT_HASH, "PutObjectContentHash", Bytes, true, 32, 32;
		PUT_OBJECT_VERSION, "PutObjectVersion", U64, true, 8, 8;
	}
	DropObject("dropObject") {
		DROP_OBJECT_ID, "DropObjectId", U64, true, 8, 8;
	}
	ExecuteNamed("ExecuteNamed") {
		EXECUTE_OPERATION, "ExecuteOperation", String, true, 1, 128;
		EXECUTE_INPUT_OBJECT_IDS, "ExecuteInputObjectIds", U64Array, true, 0, 512;
		EXECUTE_OUTPUT_OBJECT_ID, "ExecuteOutputObjectId", U64, true, 8, 8;
		EXECUTE_ARGUMENTS, "ExecuteArguments", Bytes, false, 0, 4096;
		EXECUTE_EXPECTED_VERSION, "ExecuteExpectedVersion", U64, false, 8, 8;
		EXECUTE_EXPECTED_HASH, "ExecuteExpectedHash", Bytes, false, 32, 32;
	}
	Wait("wait") {
		WAIT_REQUEST_ID, "WaitRequestId", U64, true, 8, 8;
	}
	Poll("poll") {
		POLL_REQUEST_ID, "PollRequestId", U64, true, 8, 8;
	}
	Cancel("cancel") {
		CANCEL_REQUEST_ID, "CancelRequestId", U64, true, 8, 8;
	}
	GetResult("getResult") {
		GET_RESULT_REQUEST_ID, "GetResultRequestId", U64, true, 8, 8;
	}
	Error("Error") {
		ERROR_STATUS_CODE, "ErrorStatusCode", U32, true, 4, 4;
		ERROR_MESSAGE, "ErrorMessage", String, true, 1, 1024;
		ERROR_POISONED, "ErrorPoisoned", U8, true, 1, 1;
	}
	Abort("abort") {
		ABORT_REASON, "AbortReason", String, true, 1, 1024;
	}
	Result("result") {
		RESULT_REQUEST_ID, "ResultRequestId", U64, true, 8, 8;
		RESULT_COMPLETE, "ResultComplete", U8, true, 1, 1;
		RESULT_STATUS_CODE, "ResultStatusCode", U32, true, 4, 4;
		RESULT_BYTES, "ResultBytes", Bytes, false, 0, 1048576;
		RESULT_PROFILE, "ResultProfile", Bytes, false, 0, 4096;
		RESULT_OBJECT_ID, "ResultObjectId", U64, false, 8, 8;
		RESULT_OBJECT_VERSION, "ResultObjectVersion", U64, false, 8, 8;
		RESULT_OBJECT_HASH, "ResultObjectHash", Bytes, false, 32, 32;
	}
};

fn find_field_schema(message_type: u16, id: u16) -> Option<&'static FieldSchema> {
	FIELD_SCHEMAS
		.iter()
		.find(|schema| schema.message as u16 == message_type && schema.id == id)
}

fn validate_field_data(field: &SatelliteField, schema: &FieldSchema) -> Result<()> {
	if field.kind != schema.kind {
		return Err(Error::invalid_argument(
			"satellite protocol: field kind mismatch",
		));
	}
	let bytes = field.data.len();
	if bytes < schema.min_bytes || bytes > schema.max_bytes {
		return Err(Error::invalid_argument(
			"satellite protocol: field size is outside schema bounds",
		));
	}
	if matches!(
		field.kind,
		SatelliteFieldKind::U64Array | SatelliteFieldKind::I64Array
	) && !bytes.is_multiple_of(8)
	{
		return Err(Error::invalid_argument(
			"satellite protocol: array field has a partial element",
		));
	}
	if field.kind == SatelliteFieldKind::String && std::str::from_utf8(&field.data).is_err() {
		return Err(Error::invalid_argument(
			"satellite protocol: string field is not canonical UTF-8",
		));
	}
	Ok(())
}

fn require_kind_and_bytes(
	field: &SatelliteField,
	kind: SatelliteFieldKind,
	bytes: usize,
) -> Result<()> {
	if field.kind != kind || field.data.len() != bytes {
		return Err(Error::invalid_argument(
			"satellite protocol: typed field access mismatch",
		));
	}
	Ok(())
}

pub(crate) struct SatelliteProtocol;

impl SatelliteProtocol {
	pub const VERSION: u16 = 1;
	pub const RESPONSE_FLAG: u32 = 1;
	pub const KNOWN_FLAGS: u32 = Self::RESPONSE_FLAG;
	pub const HEADER_BYTES: usize = 64;
	pub const MAX_PAYLOAD_BYTES: usize = 1024 * 1024;
	pub const MAX_FIELDS: usize = 64;

	pub fn validate(message: &SatelliteMessage) -> Result<()> {
		Self::validate_parts(
			message.version,
			message.message_type as u16,
			message.flags,
			message.request_id,
			&message.fields,
		)
	}

	/// Schema validation over the raw wire type so a decode-time unknown type
	//  is rejected by identity rather than by the aliased message's fields.
	fn validate_parts(
		version: u16,
		message_type: u16,
		flags: u32,
		request_id: u64,
		fields: &[SatelliteField],
	) -> Result<()> {
		if version != Self::VERSION {
			return Err(Error::failed_precondition(
				"satellite protocol: unsupported protocol version",
			));
		}
		if SatelliteMessageType::from_u16(message_type).is_none() {
			return Err(Error::invalid_argument(
				"satellite protocol: unknown message type",
			));
		}
		if (flags & !Self::KNOWN_FLAGS) != 0 {
			return Err(Error::invalid_argument(
				"satellite protocol: unknown header flags",
			));
		}
		if request_id == 0 {
			return Err(Error::invalid_argument(
				"satellite protocol: request id must be non-zero",
			));
		}
		if fields.len() > Self::MAX_FIELDS {
			return Err(Error::resource_exhausted(
				"satellite protocol: too many fields",
			));
		}

		let mut previous_id = 0_u16;
		for field in fields {
			if field.id == 0 || field.id <= previous_id {
				return Err(Error::invalid_argument(
					"satellite protocol: fields must be unique and strictly ordered",
				));
			}
			previous_id = field.id;
			let schema = find_field_schema(message_type, field.id).ok_or_else(|| {
				Error::invalid_argument("satellite protocol: field is not in the message schema")
			})?;
			validate_field_data(field, schema)?;
		}

		for schema in FIELD_SCHEMAS {
			if schema.message as u16 != message_type || !schema.required {
				continue;
			}
			if !fields.iter().any(|field| field.id == schema.id) {
				return Err(Error::invalid_argument(
					"satellite protocol: required field is missing",
				));
			}
		}
		Ok(())
	}

	pub fn encode(message: &SatelliteMessage) -> Result<Vec<u8>> {
		Self::validate(message)?;

		let mut payload = Vec::new();
		payload.extend_from_slice(&(message.fields.len() as u16).to_le_bytes());
		for field in &message.fields {
			if field.data.len() > u32::MAX as usize {
				return Err(Error::resource_exhausted(
					"satellite protocol: field is too large",
				));
			}
			payload.extend_from_slice(&field.id.to_le_bytes());
			payload.push(field.kind as u8);
			payload.push(0);
			payload.extend_from_slice(&(field.data.len() as u32).to_le_bytes());
			payload.extend_from_slice(&field.data);
		}
		if payload.len() > Self::MAX_PAYLOAD_BYTES {
			return Err(Error::resource_exhausted(
				"satellite protocol: encoded payload exceeds the session limit",
			));
		}

		let mut encoded = Vec::with_capacity(Self::HEADER_BYTES + payload.len());
		encoded.extend_from_slice(&MAGIC);
		encoded.extend_from_slice(&message.version.to_le_bytes());
		encoded.extend_from_slice(&(message.message_type as u16).to_le_bytes());
		encoded.extend_from_slice(&message.flags.to_le_bytes());
		encoded.extend_from_slice(&message.request_id.to_le_bytes());
		encoded.extend_from_slice(&message.session_epoch.to_le_bytes());
		encoded.extend_from_slice(&(payload.len() as u32).to_le_bytes());
		let checksum = Self::stable_digest(&encoded, &payload);
		encoded.extend_from_slice(&checksum);
		encoded.extend_from_slice(&payload);
		Ok(encoded)
	}

	pub fn decode(bytes: &[u8]) -> Result<SatelliteMessage> {
		if bytes.len() < Self::HEADER_BYTES {
			return Err(Error::data_loss("satellite protocol: truncated header"));
		}
		if bytes[..4] != MAGIC {
			return Err(Error::data_loss("satellite protocol: invalid magic"));
		}
		let payload_bytes = u32::from_le_bytes(bytes[28..32].try_into().unwrap()) as usize;
		if payload_bytes > Self::MAX_PAYLOAD_BYTES {
			return Err(Error::resource_exhausted(
				"satellite protocol: payload exceeds the session limit",
			));
		}
		if bytes.len() != Self::HEADER_BYTES + payload_bytes {
			return Err(Error::data_loss(
				"satellite protocol: payload length does not match the frame",
			));
		}
		let payload = &bytes[Self::HEADER_BYTES..];
		let expected_checksum = Self::stable_digest(&bytes[..32], payload);
		let checksum_difference = expected_checksum
			.iter()
			.zip(&bytes[32..64])
			.fold(0_u8, |difference, (expected, actual)| {
				difference | (expected ^ actual)
			});
		if checksum_difference != 0 {
			return Err(Error::data_loss("satellite protocol: checksum mismatch"));
		}
		if payload.len() < 2 {
			return Err(Error::data_loss(
				"satellite protocol: truncated field count",
			));
		}

		let version = u16::from_le_bytes(bytes[4..6].try_into().unwrap());
		let message_type = u16::from_le_bytes(bytes[6..8].try_into().unwrap());
		let flags = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
		let request_id = u64::from_le_bytes(bytes[12..20].try_into().unwrap());
		let session_epoch = u64::from_le_bytes(bytes[20..28].try_into().unwrap());

		let field_count = u16::from_le_bytes(payload[..2].try_into().unwrap()) as usize;
		if field_count > Self::MAX_FIELDS {
			return Err(Error::resource_exhausted(
				"satellite protocol: field count exceeds the session limit",
			));
		}
		let mut offset = 2_usize;
		let mut fields = Vec::with_capacity(field_count);
		for _ in 0..field_count {
			if payload.len() - offset < 8 {
				return Err(Error::data_loss(
					"satellite protocol: truncated field header",
				));
			}
			let id = u16::from_le_bytes(payload[offset..offset + 2].try_into().unwrap());
			let kind = payload[offset + 2];
			let reserved = payload[offset + 3];
			let field_bytes =
				u32::from_le_bytes(payload[offset + 4..offset + 8].try_into().unwrap()) as usize;
			offset += 8;
			if reserved != 0 {
				return Err(Error::data_loss(
					"satellite protocol: non-zero reserved field byte",
				));
			}
			if field_bytes > payload.len() - offset {
				return Err(Error::data_loss("satellite protocol: truncated field data"));
			}
			fields.push(SatelliteField {
				id,
				kind: SatelliteFieldKind::from_u8(kind),
				data: payload[offset..offset + field_bytes].to_vec(),
			});
			offset += field_bytes;
		}
		if offset != payload.len() {
			return Err(Error::data_loss(
				"satellite protocol: trailing payload bytes",
			));
		}
		Self::validate_parts(version, message_type, flags, request_id, &fields)?;
		Ok(SatelliteMessage {
			version,
			message_type: SatelliteMessageType::from_u16(message_type)
				.expect("validate_parts admits only schema message types"),
			flags,
			request_id,
			session_epoch,
			fields,
		})
	}

	/// Stable non-cryptographic digest used for wire corruption detection and
	/// opaque content identity. Authentication is supplied separately by the
	/// application-owned satellite service.
	pub fn stable_digest(first: &[u8], second: &[u8]) -> [u8; 32] {
		let mut lanes = [
			0xcbf29ce484222325_u64,
			0x84222325cbf29ce4_u64,
			0x9e3779b97f4a7c15_u64,
			0xd6e8feb86659fd93_u64,
		];
		const PRIMES: [u64; 4] = [
			0x00000100000001b3,
			0x9e3779b185ebca87,
			0xc2b2ae3d27d4eb4f,
			0x165667b19e3779f9,
		];
		let mix_byte = |lanes: &mut [u64; 4], byte: u8| {
			for (index, lane) in lanes.iter_mut().enumerate() {
				*lane ^= u64::from(byte) + ((index as u64) << 8);
				*lane = lane.wrapping_mul(PRIMES[index]);
				*lane ^= *lane >> 32;
			}
		};
		let mix_u64 = |lanes: &mut [u64; 4], value: u64| {
			for byte in 0..8 {
				mix_byte(lanes, (value >> (byte * 8)) as u8);
			}
		};
		mix_u64(&mut lanes, first.len() as u64);
		for &byte in first {
			mix_byte(&mut lanes, byte);
		}
		mix_byte(&mut lanes, 0xa5);
		mix_u64(&mut lanes, second.len() as u64);
		for &byte in second {
			mix_byte(&mut lanes, byte);
		}

		let mut digest = [0_u8; 32];
		for (index, lane) in lanes.iter_mut().enumerate() {
			let mut value = *lane;
			value ^= value >> 33;
			value = value.wrapping_mul(0xff51afd7ed558ccd);
			value ^= value >> 33;
			value = value.wrapping_mul(0xc4ceb9fe1a85ec53);
			value ^= value >> 33;
			for byte in 0..8 {
				digest[index * 8 + byte] = (value >> (byte * 8)) as u8;
			}
		}
		digest
	}

	pub fn find_field(message: &SatelliteMessage, id: u16) -> Option<&SatelliteField> {
		message.fields.iter().find(|field| field.id == id)
	}

	pub fn read_u8(field: &SatelliteField) -> Result<u8> {
		require_kind_and_bytes(field, SatelliteFieldKind::U8, 1)?;
		Ok(field.data[0])
	}

	pub fn read_u16(field: &SatelliteField) -> Result<u16> {
		require_kind_and_bytes(field, SatelliteFieldKind::U16, 2)?;
		Ok(u16::from_le_bytes(field.data[..2].try_into().unwrap()))
	}

	pub fn read_u32(field: &SatelliteField) -> Result<u32> {
		require_kind_and_bytes(field, SatelliteFieldKind::U32, 4)?;
		Ok(u32::from_le_bytes(field.data[..4].try_into().unwrap()))
	}

	pub fn read_u64(field: &SatelliteField) -> Result<u64> {
		require_kind_and_bytes(field, SatelliteFieldKind::U64, 8)?;
		Ok(u64::from_le_bytes(field.data[..8].try_into().unwrap()))
	}

	pub fn read_f32(field: &SatelliteField) -> Result<f32> {
		require_kind_and_bytes(field, SatelliteFieldKind::F32, 4)?;
		Ok(f32::from_bits(u32::from_le_bytes(
			field.data[..4].try_into().unwrap(),
		)))
	}

	pub fn read_string(field: &SatelliteField) -> Result<String> {
		if field.kind != SatelliteFieldKind::String {
			return Err(Error::invalid_argument(
				"satellite protocol: typed field access mismatch",
			));
		}
		String::from_utf8(field.data.clone())
			.map_err(|_| Error::invalid_argument("satellite protocol: string field is not UTF-8"))
	}

	pub fn read_u64_array(field: &SatelliteField) -> Result<Vec<u64>> {
		if field.kind != SatelliteFieldKind::U64Array || !field.data.len().is_multiple_of(8) {
			return Err(Error::invalid_argument(
				"satellite protocol: typed field access mismatch",
			));
		}
		Ok(
			field
				.data
				.as_chunks::<8>()
				.0
				.iter()
				.map(|chunk| u64::from_le_bytes(*chunk))
				.collect(),
		)
	}

	pub fn read_i64_array(field: &SatelliteField) -> Result<Vec<i64>> {
		if field.kind != SatelliteFieldKind::I64Array || !field.data.len().is_multiple_of(8) {
			return Err(Error::invalid_argument(
				"satellite protocol: typed field access mismatch",
			));
		}
		Ok(
			field
				.data
				.as_chunks::<8>()
				.0
				.iter()
				.map(|chunk| i64::from_le_bytes(*chunk))
				.collect(),
		)
	}
}

/// Canonical schema-identity string, byte-identical to the donor's
/// `M:Name:Value;F:message:Name:Value:Kind:Required:Min:Max;` expansion of
/// `satelliteProtocolSchema.inl`.
fn schema_manifest_string() -> String {
	let mut manifest = String::new();
	for (message, message_name) in MESSAGE_MANIFEST {
		manifest += "M:";
		manifest += message_name;
		manifest += ":";
		manifest += &(*message as u16).to_string();
		manifest += ";";
		for schema in FIELD_SCHEMAS {
			if schema.message != *message {
				continue;
			}
			manifest += "F:";
			manifest += schema.message_name;
			manifest += ":";
			manifest += schema.name;
			manifest += ":";
			manifest += &schema.id.to_string();
			manifest += ":";
			manifest += schema.kind.schema_name();
			manifest += ":";
			manifest += if schema.required { "true" } else { "false" };
			manifest += ":";
			manifest += &schema.min_bytes.to_string();
			manifest += ":";
			manifest += &schema.max_bytes.to_string();
			manifest += ";";
		}
	}
	manifest
}

/// Stable digest over the build identity, matching the donor's
/// `satelliteBuildHash` role of digesting the version string.
pub(crate) fn satellite_build_hash() -> [u8; 32] {
	SatelliteProtocol::stable_digest(env!("CARGO_PKG_VERSION").as_bytes(), &[])
}

/// Stable digest over the canonical schema manifest, matching the donor's
/// `satelliteSchemaHash`. Byte-identical to the donor hash because the
/// manifest string is produced from the same schema expansion.
pub(crate) fn satellite_schema_hash() -> [u8; 32] {
	SatelliteProtocol::stable_digest(schema_manifest_string().as_bytes(), &[])
}

pub(crate) mod nlp_coordinator;
pub(crate) mod nlp_step;
pub(crate) mod session;

#[cfg(test)]
#[path = "../../../test/rs/net/satellite_unit.rs"]
mod tests;
