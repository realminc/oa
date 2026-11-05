//! Private authenticated satellite session over loopback TCP.
//!
//! Donor: `source/cpp/lib/oa/net/satelliteSession.{h,cpp}` and
//! `test/cpp/net/satellite/testSatellite{Session,Vulkan}.cpp`. A
//! synchronous single-connection worker session: the [`SatelliteClientSession`]
//! connects, authenticates, and exchanges bounded requests; the
//! [`SatelliteServerSession`] serves one accepted stream at a time on the
//! caller's thread and never drains or waits on teardown. The optional
//! [`Engine`] borrow enables `matrix-add-f32-v1` and application named work
//! with exact submission-event provenance; the CPU oracle path needs no
//! engine.
//!
//! Wire status codes keep the donor `StatusCode` numbering; `code_kind` and
//! `kind_code` translate between that vocabulary and [`ErrorKind`]. Domain
//! codes without an `ErrorKind` counterpart decode to the nearest documented
//! kind.

use std::net::{IpAddr, Ipv4Addr};
use std::time::Duration;

use super::{
	SatelliteField, SatelliteMessage, SatelliteMessageType, SatelliteProtocol, field_id,
	satellite_build_hash, satellite_schema_hash,
};
use crate::core::memory::{equal_constant_time, zero_secure};
use crate::{DType, Engine, Error, ErrorKind, Event, Matrix, Result, TcpFramed, TcpStream};

const HELLO_CUSTOM: &str = "oa-satellite-hello-v1";
const REPLY_CUSTOM: &str = "oa-satellite-reply-v1";
const MAX_TENSOR_DIMS: usize = 8;
const RESULT_ENVELOPE_BYTES: usize = 55;
const MAX_RESULT_PROFILE_BYTES: usize = 4096;
const CPU_ADD_OPERATION: &str = "u32-add-v1";
const MATRIX_ADD_OPERATION: &str = "matrix-add-f32-v1";

/// Donor `ScalarType` wire identities admitted by the session object table.
mod wire_dtype {
	pub const F32: u8 = 0;
	pub const U32: u8 = 10;
}

/// Donor `StatusCode` numbering carried on the wire.
mod status_code {
	pub const OK: u32 = 0;
	pub const CANCELLED: u32 = 1;
	pub const UNKNOWN: u32 = 2;
	pub const INVALID_ARGUMENT: u32 = 3;
	pub const DEADLINE_EXCEEDED: u32 = 4;
	pub const NOT_FOUND: u32 = 5;
	pub const ALREADY_EXISTS: u32 = 6;
	pub const PERMISSION_DENIED: u32 = 7;
	pub const RESOURCE_EXHAUSTED: u32 = 8;
	pub const FAILED_PRECONDITION: u32 = 9;
	pub const ABORTED: u32 = 10;
	pub const OUT_OF_RANGE: u32 = 11;
	pub const UNIMPLEMENTED: u32 = 12;
	pub const INTERNAL: u32 = 13;
	pub const UNAVAILABLE: u32 = 14;
	pub const DATA_LOSS: u32 = 15;
	pub const UNAUTHENTICATED: u32 = 16;
	pub const SHAPE_MISMATCH: u32 = 161;
	pub const DTYPE_MISMATCH: u32 = 162;
}

/// Wire status code to the nearest [`ErrorKind`]. ML domain codes
/// (160..=164) surface as `InvalidArgument`, Vulkan domain codes as
/// `BackendFailure`, and transport/file domain codes as `Io`; every other
/// admitted code without an exact kind maps to `Internal`.
fn code_kind(code: u32) -> ErrorKind {
	match code {
		status_code::CANCELLED => ErrorKind::Cancelled,
		status_code::INVALID_ARGUMENT => ErrorKind::InvalidArgument,
		status_code::DEADLINE_EXCEEDED | status_code::UNAVAILABLE => ErrorKind::Unavailable,
		status_code::NOT_FOUND => ErrorKind::NotFound,
		status_code::ALREADY_EXISTS => ErrorKind::AlreadyExists,
		status_code::PERMISSION_DENIED => ErrorKind::PermissionDenied,
		status_code::RESOURCE_EXHAUSTED => ErrorKind::ResourceExhausted,
		status_code::FAILED_PRECONDITION => ErrorKind::FailedPrecondition,
		status_code::ABORTED => ErrorKind::Aborted,
		status_code::OUT_OF_RANGE => ErrorKind::OutOfRange,
		status_code::UNIMPLEMENTED => ErrorKind::Unimplemented,
		status_code::DATA_LOSS => ErrorKind::DataLoss,
		status_code::UNAUTHENTICATED => ErrorKind::Unauthenticated,
		100..=104 => ErrorKind::BackendFailure,
		status_code::SHAPE_MISMATCH => ErrorKind::ShapeMismatch,
		status_code::DTYPE_MISMATCH => ErrorKind::DtypeMismatch,
		160..=164 => ErrorKind::InvalidArgument,
		180..=183 | 200..=203 => ErrorKind::Io,
		_ => ErrorKind::Internal,
	}
}

/// [`ErrorKind`] to the donor wire status code. Kinds without a wire
/// counterpart emit `UNKNOWN` (2).
fn kind_code(kind: ErrorKind) -> u32 {
	match kind {
		ErrorKind::Cancelled => status_code::CANCELLED,
		ErrorKind::InvalidArgument => status_code::INVALID_ARGUMENT,
		ErrorKind::NotFound => status_code::NOT_FOUND,
		ErrorKind::AlreadyExists => status_code::ALREADY_EXISTS,
		ErrorKind::PermissionDenied => status_code::PERMISSION_DENIED,
		ErrorKind::ResourceExhausted => status_code::RESOURCE_EXHAUSTED,
		ErrorKind::FailedPrecondition => status_code::FAILED_PRECONDITION,
		ErrorKind::Aborted => status_code::ABORTED,
		ErrorKind::OutOfRange => status_code::OUT_OF_RANGE,
		ErrorKind::Unimplemented => status_code::UNIMPLEMENTED,
		ErrorKind::Internal => status_code::INTERNAL,
		ErrorKind::Unavailable | ErrorKind::Io => status_code::UNAVAILABLE,
		ErrorKind::DataLoss => status_code::DATA_LOSS,
		ErrorKind::Unauthenticated => status_code::UNAUTHENTICATED,
		ErrorKind::ShapeMismatch => status_code::SHAPE_MISMATCH,
		ErrorKind::DtypeMismatch => status_code::DTYPE_MISMATCH,
		ErrorKind::BackendFailure => 100,
		_ => status_code::UNKNOWN,
	}
}

/// Donor `statusCodeName` spellings for codes emitted by this session.
fn status_name(code: u32) -> &'static str {
	match code {
		status_code::OK => "OK",
		status_code::CANCELLED => "CANCELLED",
		status_code::UNKNOWN => "UNKNOWN",
		status_code::INVALID_ARGUMENT => "INVALID_ARGUMENT",
		status_code::DEADLINE_EXCEEDED => "DEADLINE_EXCEEDED",
		status_code::NOT_FOUND => "NOT_FOUND",
		status_code::ALREADY_EXISTS => "ALREADY_EXISTS",
		status_code::PERMISSION_DENIED => "PERMISSION_DENIED",
		status_code::RESOURCE_EXHAUSTED => "RESOURCE_EXHAUSTED",
		status_code::FAILED_PRECONDITION => "FAILED_PRECONDITION",
		status_code::ABORTED => "ABORTED",
		status_code::OUT_OF_RANGE => "OUT_OF_RANGE",
		status_code::UNIMPLEMENTED => "UNIMPLEMENTED",
		status_code::INTERNAL => "INTERNAL",
		status_code::UNAVAILABLE => "UNAVAILABLE",
		status_code::DATA_LOSS => "DATA_LOSS",
		status_code::UNAUTHENTICATED => "UNAUTHENTICATED",
		status_code::SHAPE_MISMATCH => "SHAPE_MISMATCH",
		status_code::DTYPE_MISMATCH => "DTYPE_MISMATCH",
		_ => "UNKNOWN",
	}
}

/// Construct the concrete error for a donor wire status code, using the
/// specific constructor when the mapped kind has one.
fn error_for_status(code: u32, message: impl Into<String>) -> Error {
	let message = message.into();
	match code_kind(code) {
		ErrorKind::Cancelled => Error::cancelled(message),
		ErrorKind::PermissionDenied => Error::permission_denied(message),
		ErrorKind::Unimplemented => Error::unimplemented(message),
		ErrorKind::Unavailable => Error::unavailable(message),
		ErrorKind::Unauthenticated => Error::unauthenticated(message),
		ErrorKind::Aborted => Error::aborted(message),
		kind => Error::with_kind(kind, message),
	}
}

fn is_known_status_code(code: u32) -> bool {
	code <= 16
		|| (100..=104).contains(&code)
		|| (120..=125).contains(&code)
		|| (140..=144).contains(&code)
		|| (160..=164).contains(&code)
		|| (180..=183).contains(&code)
		|| (200..=203).contains(&code)
}

/// Application-supplied message authentication primitive.
///
/// Computes a 32-byte MAC over `data` keyed by `secret` under the `custom`
/// label. The session's authentication contract is entirely owned by this
/// callback; the donor test contract uses KMAC-256. A plain function pointer
/// keeps [`SatelliteSessionConfig`] movable across threads.
pub(crate) type SatelliteMacFn = fn(&[u8], &[u8], &str, &mut [u8; 32]) -> Result<()>;

/// Borrowed session object handed to application named work.
#[derive(Debug)]
pub(crate) struct SatelliteObjectView<'a> {
	pub id: u64,
	pub version: u64,
	pub dtype: DType,
	pub shape: &'a [i64],
	pub data: &'a [u8],
	pub hash: [u8; 32],
}

/// Application named-work request. Network validates object identity, bounds,
/// version/hash preconditions, and the exact completion event; the callback
/// owns workload semantics.
pub(crate) struct SatelliteNamedRequest<'a> {
	pub operation: String,
	pub inputs: Vec<SatelliteObjectView<'a>>,
	pub arguments: &'a [u8],
	pub expected_version: u64,
	pub expected_hash: [u8; 32],
}

/// Application named-work result. `completion` must be an engine-owned event
/// that is already complete when the callback returns.
pub(crate) struct SatelliteNamedResult {
	pub version: u64,
	pub dtype: DType,
	pub shape: Vec<i64>,
	pub data: Vec<u8>,
	pub profile: Vec<u8>,
	pub completion: Event,
}

/// Application-owned named work injected into the private session.
pub(crate) type SatelliteNamedWorkFn =
	Box<dyn Fn(&Engine, &SatelliteNamedRequest<'_>) -> Result<SatelliteNamedResult>>;

/// 32-byte shared authentication secret, zeroized on drop.
pub(crate) struct SatelliteSecret {
	bytes: [u8; 32],
}

impl SatelliteSecret {
	pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
		let bytes: &[u8; 32] = bytes.try_into().map_err(|_| {
			Error::invalid_argument("satellite session: authentication secret must be exactly 32 bytes")
		})?;
		Ok(Self { bytes: *bytes })
	}

	pub fn bytes(&self) -> &[u8; 32] {
		&self.bytes
	}
}

impl Drop for SatelliteSecret {
	fn drop(&mut self) {
		zero_secure(&mut self.bytes);
	}
}

/// Session limit advertisement; negotiated as the pairwise minimum.
#[derive(Clone, Debug)]
pub(crate) struct SatelliteLimits {
	pub max_payload_bytes: u32,
	pub max_resident_bytes: u64,
	pub max_objects: u32,
	pub max_inflight: u32,
}

impl Default for SatelliteLimits {
	fn default() -> Self {
		Self {
			max_payload_bytes: 1024 * 1024,
			max_resident_bytes: 256 * 1024 * 1024,
			max_objects: 256,
			max_inflight: 1,
		}
	}
}

/// Session configuration. `named_operations` bounds which named operations the
/// session admits; `named_work` is invoked only for admitted names and only on
/// execute/wait, never by poll, cancel, or disconnect.
pub(crate) struct SatelliteSessionConfig {
	pub secret: SatelliteSecret,
	pub mac: SatelliteMacFn,
	pub limits: SatelliteLimits,
	pub io_timeout_ms: u32,
	pub device_name: String,
	pub named_operations: Vec<String>,
	pub named_work: Option<SatelliteNamedWorkFn>,
}

impl SatelliteSessionConfig {
	pub fn new(secret: SatelliteSecret, mac: SatelliteMacFn) -> Self {
		Self {
			secret,
			mac,
			limits: SatelliteLimits::default(),
			io_timeout_ms: 2000,
			device_name: "CPU oracle".to_owned(),
			named_operations: Vec::new(),
			named_work: None,
		}
	}
}

/// Negotiated session identity returned by a successful client handshake.
#[derive(Clone, Debug)]
pub(crate) struct SatelliteProbe {
	pub session_epoch: u64,
	pub device_name: String,
	pub limits: SatelliteLimits,
	pub build_hash: [u8; 32],
	pub schema_hash: [u8; 32],
}

enum PendingKind {
	CpuAdd,
	MatrixAddF32,
	NamedWork,
}

struct PendingOp {
	request_id: u64,
	kind: PendingKind,
	a: u32,
	b: u32,
	input_a: u64,
	input_b: u64,
	output: u64,
	operation: String,
	inputs: Vec<u64>,
	arguments: Vec<u8>,
	expected_version: u64,
	expected_hash: [u8; 32],
	complete: bool,
	/// Donor wire status code, not an `ErrorKind`.
	status: u32,
	result: Vec<u8>,
	profile: Vec<u8>,
}

struct SessionObject {
	id: u64,
	version: u64,
	dtype: DType,
	shape: Vec<i64>,
	data: Vec<u8>,
	hash: [u8; 32],
}

fn append_u32_le(out: &mut Vec<u8>, value: u32) {
	out.extend_from_slice(&value.to_le_bytes());
}

fn append_u64_le(out: &mut Vec<u8>, value: u64) {
	out.extend_from_slice(&value.to_le_bytes());
}

fn read_u32_le(data: &[u8]) -> u32 {
	u32::from_le_bytes(data[..4].try_into().unwrap())
}

fn append_f32_le(out: &mut Vec<u8>, value: f32) {
	append_u32_le(out, value.to_bits());
}

fn pending_uses_object(pending: &PendingOp, object_id: u64) -> bool {
	pending.output == object_id
		|| pending.input_a == object_id
		|| pending.input_b == object_id
		|| pending.inputs.contains(&object_id)
}

fn hash_bytes(bytes: &[u8]) -> [u8; 32] {
	SatelliteProtocol::stable_digest(bytes, &[])
}

fn find_object(objects: &[SessionObject], id: u64) -> Option<&SessionObject> {
	objects.iter().find(|object| object.id == id)
}

fn total_object_bytes(objects: &[SessionObject]) -> u64 {
	objects.iter().map(|object| object.data.len() as u64).sum()
}

fn wire_dtype_to_dtype(dtype: u8) -> Option<DType> {
	match dtype {
		wire_dtype::F32 => Some(DType::F32),
		wire_dtype::U32 => Some(DType::U32),
		_ => None,
	}
}

fn dtype_wire(dtype: DType) -> u8 {
	match dtype {
		DType::F32 => wire_dtype::F32,
		DType::U32 => wire_dtype::U32,
		_ => unreachable!("satellite session admits only FP32/UInt32 objects"),
	}
}

fn checked_element_count(shape: &[i64]) -> Result<u64> {
	if shape.is_empty() || shape.len() > MAX_TENSOR_DIMS {
		return Err(Error::invalid_argument(
			"satellite session: tensor rank is invalid",
		));
	}
	let mut count = 1_u64;
	for &dimension in shape {
		if dimension <= 0 {
			return Err(Error::invalid_argument(
				"satellite session: tensor dimensions must be positive",
			));
		}
		let value = dimension as u64;
		if count > u64::MAX / value {
			return Err(Error::out_of_range(
				"satellite session: tensor element count overflows",
			));
		}
		count *= value;
	}
	Ok(count)
}

fn matrix_shape_from_span(shape: &[i64]) -> Result<Vec<usize>> {
	checked_element_count(shape)?;
	Ok(shape.iter().map(|&dimension| dimension as usize).collect())
}

fn decode_f32(data: &[u8]) -> Result<Vec<f32>> {
	if !data.len().is_multiple_of(4) {
		return Err(Error::data_loss(
			"satellite session: FP32 object has a partial element",
		));
	}
	Ok(
		data
			.as_chunks::<4>()
			.0
			.iter()
			.map(|chunk| f32::from_bits(u32::from_le_bytes(*chunk)))
			.collect(),
	)
}

fn random_nonce() -> Result<[u8; 32]> {
	let mut nonce = [0_u8; 32];
	getrandom::fill(&mut nonce)
		.map_err(|_| Error::unavailable("satellite session: operating-system random source failed"))?;
	Ok(nonce)
}

fn non_zero_epoch(previous: u64) -> Result<u64> {
	loop {
		let nonce = random_nonce()?;
		let epoch = u64::from_le_bytes(nonce[..8].try_into().unwrap());
		if epoch != 0 && epoch != previous {
			return Ok(epoch);
		}
	}
}

fn kmac(mac: &SatelliteMacFn, secret: &[u8], data: &[u8], custom: &str) -> Result<[u8; 32]> {
	let mut out = [0_u8; 32];
	mac(secret, data, custom, &mut out)?;
	Ok(out)
}

fn admits_named_operation(config: &SatelliteSessionConfig, operation: &str) -> bool {
	config
		.named_operations
		.iter()
		.any(|admitted| admitted == operation)
}

fn required_field(message: &SatelliteMessage, id: u16) -> Result<&SatelliteField> {
	SatelliteProtocol::find_field(message, id).ok_or_else(|| {
		Error::data_loss("satellite session: required field is missing from a decoded message")
	})
}

fn write_message(stream: &mut TcpStream, message: &SatelliteMessage) -> Result<()> {
	let encoded = SatelliteProtocol::encode(message)?;
	TcpFramed::write_message(stream, &encoded)
}

fn read_message(stream: &mut TcpStream, max_payload_bytes: u32) -> Result<SatelliteMessage> {
	if max_payload_bytes > SatelliteProtocol::MAX_PAYLOAD_BYTES as u32 {
		return Err(Error::invalid_argument(
			"satellite session: invalid payload limit",
		));
	}
	let frame = TcpFramed::read_message(
		stream,
		SatelliteProtocol::HEADER_BYTES + max_payload_bytes as usize,
	)
	.map_err(|_| Error::unavailable("satellite session: framed read failed"))?;
	SatelliteProtocol::decode(&frame)
}

fn error_response(
	request: &SatelliteMessage,
	code: u32,
	message: &str,
	poisoned: bool,
) -> SatelliteMessage {
	SatelliteMessage {
		message_type: SatelliteMessageType::Error,
		flags: SatelliteProtocol::RESPONSE_FLAG,
		request_id: request.request_id,
		session_epoch: request.session_epoch,
		fields: vec![
			SatelliteField::u32(field_id::ERROR_STATUS_CODE, code),
			SatelliteField::string(
				field_id::ERROR_MESSAGE,
				if message.is_empty() {
					status_name(code)
				} else {
					message
				},
			),
			SatelliteField::u8(field_id::ERROR_POISONED, u8::from(poisoned)),
		],
		..SatelliteMessage::default()
	}
}

fn result_response(
	request: &SatelliteMessage,
	target_request: u64,
	complete: bool,
	status: u32,
	bytes: &[u8],
	profile: &[u8],
) -> SatelliteMessage {
	let mut response = SatelliteMessage {
		message_type: SatelliteMessageType::Result,
		flags: SatelliteProtocol::RESPONSE_FLAG,
		request_id: request.request_id,
		session_epoch: request.session_epoch,
		fields: vec![
			SatelliteField::u64(field_id::RESULT_REQUEST_ID, target_request),
			SatelliteField::u8(field_id::RESULT_COMPLETE, u8::from(complete)),
			SatelliteField::u32(field_id::RESULT_STATUS_CODE, status),
		],
		..SatelliteMessage::default()
	};
	if !bytes.is_empty() {
		response
			.fields
			.push(SatelliteField::bytes(field_id::RESULT_BYTES, bytes));
	}
	if !profile.is_empty() {
		response
			.fields
			.push(SatelliteField::bytes(field_id::RESULT_PROFILE, profile));
	}
	response
}

fn decode_error(message: &SatelliteMessage) -> Error {
	let Ok(code_field) = required_field(message, field_id::ERROR_STATUS_CODE) else {
		return Error::data_loss("satellite session: malformed error response");
	};
	let Ok(message_field) = required_field(message, field_id::ERROR_MESSAGE) else {
		return Error::data_loss("satellite session: malformed error response");
	};
	let (Ok(code), Ok(text)) = (
		SatelliteProtocol::read_u32(code_field),
		SatelliteProtocol::read_string(message_field),
	) else {
		return Error::data_loss("satellite session: malformed error response");
	};
	if !is_known_status_code(code) {
		return Error::data_loss("satellite session: invalid remote status code");
	}
	error_for_status(code, text)
}

fn read_hello_limits(hello: &SatelliteMessage) -> Result<SatelliteLimits> {
	Ok(SatelliteLimits {
		max_payload_bytes: SatelliteProtocol::read_u32(required_field(
			hello,
			field_id::HELLO_MAX_PAYLOAD_BYTES,
		)?)?,
		max_resident_bytes: SatelliteProtocol::read_u64(required_field(
			hello,
			field_id::HELLO_MAX_RESIDENT_BYTES,
		)?)?,
		max_objects: SatelliteProtocol::read_u32(required_field(hello, field_id::HELLO_MAX_OBJECTS)?)?,
		max_inflight: SatelliteProtocol::read_u32(required_field(
			hello,
			field_id::HELLO_MAX_INFLIGHT,
		)?)?,
	})
}

fn valid_limits(limits: &SatelliteLimits) -> bool {
	limits.max_payload_bytes > 0
		&& limits.max_payload_bytes <= SatelliteProtocol::MAX_PAYLOAD_BYTES as u32
		&& limits.max_resident_bytes > 0
		&& limits.max_objects > 0
		&& limits.max_inflight == 1
}

fn negotiate_limits(client: &SatelliteLimits, server: &SatelliteLimits) -> SatelliteLimits {
	SatelliteLimits {
		max_payload_bytes: client.max_payload_bytes.min(server.max_payload_bytes),
		max_resident_bytes: client.max_resident_bytes.min(server.max_resident_bytes),
		max_objects: client.max_objects.min(server.max_objects),
		max_inflight: 1,
	}
}

/// Write an Error response for `request` and return the corresponding `Err`.
fn send_failure(
	stream: &mut TcpStream,
	request: &SatelliteMessage,
	code: u32,
	message: &str,
	poisoned: bool,
) -> Error {
	match write_message(stream, &error_response(request, code, message, poisoned)) {
		Err(write) => write,
		Ok(()) => error_for_status(code, message),
	}
}

/// Write an Error response echoing `error`'s mapped code and message.
fn send_failure_error(
	stream: &mut TcpStream,
	request: &SatelliteMessage,
	error: Error,
	poisoned: bool,
) -> Error {
	let code = kind_code(error.kind());
	match write_message(
		stream,
		&error_response(request, code, error.message(), poisoned),
	) {
		Err(write) => write,
		Ok(()) => error,
	}
}

/// Envelope checks shared by every post-handshake request.
pub(crate) fn satellite_validate_request_envelope(
	request: &SatelliteMessage,
	session_epoch: u64,
	last_request_id: u64,
) -> Result<()> {
	if request.flags != 0 {
		return Err(Error::aborted(
			"satellite session: a request cannot carry response flags",
		));
	}
	if request.session_epoch != session_epoch {
		return Err(Error::aborted("satellite session: stale session epoch"));
	}
	if request.request_id <= last_request_id {
		return Err(Error::aborted(
			"satellite session: request id is duplicate or non-monotonic",
		));
	}
	Ok(())
}

/// Private, synchronous, single-connection worker session. The caller owns any
/// accept/progress thread and joins it explicitly; destruction only closes the
/// socket held by a client and never drains or waits for application work.
/// [`Engine`] is `!Send`, so a server borrowing an engine must serve on the
/// engine's own thread.
pub(crate) struct SatelliteServerSession<'a> {
	config: SatelliteSessionConfig,
	engine: Option<&'a Engine>,
	last_epoch: u64,
	last_gpu_event_value: u64,
	last_gpu_event_was_owned: bool,
	last_gpu_event_completed: bool,
}

impl<'a> SatelliteServerSession<'a> {
	pub fn new(config: SatelliteSessionConfig) -> Self {
		Self {
			config,
			engine: None,
			last_epoch: 0,
			last_gpu_event_value: 0,
			last_gpu_event_was_owned: false,
			last_gpu_event_completed: false,
		}
	}

	pub fn new_with_engine(engine: &'a Engine, config: SatelliteSessionConfig) -> Self {
		Self {
			config,
			engine: Some(engine),
			last_epoch: 0,
			last_gpu_event_value: 0,
			last_gpu_event_was_owned: false,
			last_gpu_event_completed: false,
		}
	}

	pub const fn last_session_epoch(&self) -> u64 {
		self.last_epoch
	}

	pub const fn last_gpu_event_was_owned(&self) -> bool {
		self.last_gpu_event_was_owned
	}

	pub const fn last_gpu_event_completed(&self) -> bool {
		self.last_gpu_event_completed
	}

	pub const fn last_gpu_event_value(&self) -> u64 {
		self.last_gpu_event_value
	}

	pub fn serve(&mut self, mut stream: TcpStream) -> Result<()> {
		if stream.remote_addr() != IpAddr::V4(Ipv4Addr::LOCALHOST) {
			return Err(Error::permission_denied(
				"satellite session: non-loopback peers require an authenticated encrypted channel",
			));
		}
		stream.set_io_timeout(Duration::from_millis(u64::from(self.config.io_timeout_ms)))?;

		let hello = read_message(&mut stream, self.config.limits.max_payload_bytes)?;
		if hello.message_type != SatelliteMessageType::Hello
			|| hello.flags != 0
			|| hello.session_epoch != 0
		{
			return Err(send_failure(
				&mut stream,
				&hello,
				status_code::FAILED_PRECONDITION,
				"satellite session: HELLO must begin epoch zero",
				false,
			));
		}

		let protocol_min =
			SatelliteProtocol::read_u16(required_field(&hello, field_id::HELLO_PROTOCOL_MIN)?)?;
		let protocol_max =
			SatelliteProtocol::read_u16(required_field(&hello, field_id::HELLO_PROTOCOL_MAX)?)?;
		if protocol_min > SatelliteProtocol::VERSION || protocol_max < SatelliteProtocol::VERSION {
			return Err(send_failure(
				&mut stream,
				&hello,
				status_code::FAILED_PRECONDITION,
				"satellite session: no common protocol version",
				false,
			));
		}
		let requested_limits = read_hello_limits(&hello)?;
		if !valid_limits(&requested_limits) || !valid_limits(&self.config.limits) {
			return Err(send_failure(
				&mut stream,
				&hello,
				status_code::INVALID_ARGUMENT,
				"satellite session: invalid advertised limits",
				false,
			));
		}
		let limits = negotiate_limits(&requested_limits, &self.config.limits);

		let client_nonce = required_field(&hello, field_id::HELLO_CLIENT_NONCE)?
			.data
			.clone();
		let proof = required_field(&hello, field_id::HELLO_AUTH_PROOF)?
			.data
			.clone();
		let client_build = required_field(&hello, field_id::HELLO_BUILD_HASH)?
			.data
			.clone();
		let client_schema = required_field(&hello, field_id::HELLO_SCHEMA_HASH)?
			.data
			.clone();
		let build_hash = satellite_build_hash();
		let schema_hash = satellite_schema_hash();
		if !equal_constant_time(&client_build, &build_hash)
			|| !equal_constant_time(&client_schema, &schema_hash)
		{
			return Err(send_failure(
				&mut stream,
				&hello,
				status_code::FAILED_PRECONDITION,
				"satellite session: build or schema hash mismatch",
				false,
			));
		}
		let expected_proof = kmac(
			&self.config.mac,
			self.config.secret.bytes(),
			&client_nonce,
			HELLO_CUSTOM,
		)?;
		if !equal_constant_time(&proof, &expected_proof) {
			return Err(send_failure(
				&mut stream,
				&hello,
				status_code::UNAUTHENTICATED,
				"satellite session: peer authentication failed",
				false,
			));
		}

		let server_nonce = random_nonce()?;
		let epoch = non_zero_epoch(self.last_epoch)?;
		self.last_epoch = epoch;
		let mut reply_proof_data = Vec::with_capacity(72);
		reply_proof_data.extend_from_slice(&client_nonce);
		reply_proof_data.extend_from_slice(&server_nonce);
		append_u64_le(&mut reply_proof_data, epoch);
		let reply_proof = kmac(
			&self.config.mac,
			self.config.secret.bytes(),
			&reply_proof_data,
			REPLY_CUSTOM,
		)?;

		let reply = SatelliteMessage {
			message_type: SatelliteMessageType::HelloReply,
			flags: SatelliteProtocol::RESPONSE_FLAG,
			request_id: hello.request_id,
			session_epoch: epoch,
			fields: vec![
				SatelliteField::u16(field_id::HELLO_REPLY_PROTOCOL, SatelliteProtocol::VERSION),
				SatelliteField::bytes(field_id::HELLO_REPLY_SERVER_NONCE, &server_nonce),
				SatelliteField::bytes(field_id::HELLO_REPLY_AUTH_PROOF, &reply_proof),
				SatelliteField::u64(field_id::HELLO_REPLY_SESSION_EPOCH, epoch),
				SatelliteField::string(field_id::HELLO_REPLY_DEVICE_NAME, &self.config.device_name),
				SatelliteField::u32(
					field_id::HELLO_REPLY_MAX_PAYLOAD_BYTES,
					limits.max_payload_bytes,
				),
				SatelliteField::u64(
					field_id::HELLO_REPLY_MAX_RESIDENT_BYTES,
					limits.max_resident_bytes,
				),
				SatelliteField::u32(field_id::HELLO_REPLY_MAX_OBJECTS, limits.max_objects),
				SatelliteField::u32(field_id::HELLO_REPLY_MAX_INFLIGHT, limits.max_inflight),
			],
			..SatelliteMessage::default()
		};
		write_message(&mut stream, &reply)?;

		let mut last_request_id = hello.request_id;
		let mut objects: Vec<SessionObject> = Vec::new();
		let mut pending: Option<PendingOp> = None;
		loop {
			let request = read_message(&mut stream, limits.max_payload_bytes)?;
			if let Err(envelope) = satellite_validate_request_envelope(&request, epoch, last_request_id) {
				return Err(send_failure_error(&mut stream, &request, envelope, true));
			}
			last_request_id = request.request_id;

			match request.message_type {
				SatelliteMessageType::PutObject => {
					if objects.len() >= limits.max_objects as usize {
						let _ = write_message(
							&mut stream,
							&error_response(
								&request,
								status_code::RESOURCE_EXHAUSTED,
								"satellite session: object quota reached",
								false,
							),
						);
						continue;
					}
					let object_id =
						SatelliteProtocol::read_u64(required_field(&request, field_id::PUT_OBJECT_ID)?)?;
					let dtype_wire =
						SatelliteProtocol::read_u8(required_field(&request, field_id::PUT_OBJECT_DTYPE)?)?;
					let shape = SatelliteProtocol::read_i64_array(required_field(
						&request,
						field_id::PUT_OBJECT_SHAPE,
					)?)?;
					let data = required_field(&request, field_id::PUT_OBJECT_DATA)?
						.data
						.clone();
					let claimed_hash = required_field(&request, field_id::PUT_OBJECT_CONTENT_HASH)?
						.data
						.clone();
					let object_version =
						SatelliteProtocol::read_u64(required_field(&request, field_id::PUT_OBJECT_VERSION)?)?;
					if object_id == 0 || find_object(&objects, object_id).is_some() {
						let _ = write_message(
							&mut stream,
							&error_response(
								&request,
								status_code::ALREADY_EXISTS,
								"satellite session: object id is zero or already exists",
								false,
							),
						);
						continue;
					}
					let Some(dtype) = wire_dtype_to_dtype(dtype_wire) else {
						let _ = write_message(
							&mut stream,
							&error_response(
								&request,
								status_code::DTYPE_MISMATCH,
								"satellite session: named work admits canonical FP32/UInt32 objects",
								false,
							),
						);
						continue;
					};
					let scalar_bytes = dtype.size_bytes();
					let elements = checked_element_count(&shape);
					if elements
						.map(|count| {
							count > u64::MAX / scalar_bytes as u64
								|| count * scalar_bytes as u64 != data.len() as u64
						})
						.unwrap_or(true)
					{
						let _ = write_message(
							&mut stream,
							&error_response(
								&request,
								status_code::SHAPE_MISMATCH,
								"satellite session: shape, dtype, and byte count disagree",
								false,
							),
						);
						continue;
					}
					let actual_hash = hash_bytes(&data);
					if !equal_constant_time(&claimed_hash, &actual_hash) {
						let _ = write_message(
							&mut stream,
							&error_response(
								&request,
								status_code::DATA_LOSS,
								"satellite session: object content hash mismatch",
								false,
							),
						);
						continue;
					}
					if data.len() as u64 > limits.max_resident_bytes
						|| total_object_bytes(&objects) > limits.max_resident_bytes - data.len() as u64
					{
						let _ = write_message(
							&mut stream,
							&error_response(
								&request,
								status_code::RESOURCE_EXHAUSTED,
								"satellite session: resident-byte quota reached",
								false,
							),
						);
						continue;
					}
					objects.push(SessionObject {
						id: object_id,
						version: object_version,
						dtype,
						shape,
						data,
						hash: actual_hash,
					});
					let mut acknowledged = result_response(
						&request,
						request.request_id,
						true,
						status_code::OK,
						&[],
						&[],
					);
					acknowledged
						.fields
						.push(SatelliteField::u64(field_id::RESULT_OBJECT_ID, object_id));
					acknowledged.fields.push(SatelliteField::u64(
						field_id::RESULT_OBJECT_VERSION,
						object_version,
					));
					acknowledged.fields.push(SatelliteField::bytes(
						field_id::RESULT_OBJECT_HASH,
						&actual_hash,
					));
					write_message(&mut stream, &acknowledged)?;
					continue;
				}
				SatelliteMessageType::DropObject => {
					let object_id =
						SatelliteProtocol::read_u64(required_field(&request, field_id::DROP_OBJECT_ID)?)?;
					if pending
						.as_ref()
						.is_some_and(|pending| !pending.complete && pending_uses_object(pending, object_id))
					{
						let _ = write_message(
							&mut stream,
							&error_response(
								&request,
								status_code::FAILED_PRECONDITION,
								"satellite session: object belongs to pending work",
								false,
							),
						);
						continue;
					}
					let index = objects.iter().position(|object| object.id == object_id);
					let Some(index) = index else {
						let _ = write_message(
							&mut stream,
							&error_response(
								&request,
								status_code::NOT_FOUND,
								"satellite session: object is not retained",
								false,
							),
						);
						continue;
					};
					objects.remove(index);
					write_message(
						&mut stream,
						&result_response(
							&request,
							request.request_id,
							true,
							status_code::OK,
							&[],
							&[],
						),
					)?;
					continue;
				}
				SatelliteMessageType::ExecuteNamed => {
					if pending.as_ref().is_some_and(|pending| !pending.complete) {
						let _ = write_message(
							&mut stream,
							&error_response(
								&request,
								status_code::RESOURCE_EXHAUSTED,
								"satellite session: in-flight quota reached",
								false,
							),
						);
						continue;
					}
					let operation =
						SatelliteProtocol::read_string(required_field(&request, field_id::EXECUTE_OPERATION)?)?;
					let inputs = SatelliteProtocol::read_u64_array(required_field(
						&request,
						field_id::EXECUTE_INPUT_OBJECT_IDS,
					)?)?;
					let output = SatelliteProtocol::read_u64(required_field(
						&request,
						field_id::EXECUTE_OUTPUT_OBJECT_ID,
					)?)?;
					let arguments = SatelliteProtocol::find_field(&request, field_id::EXECUTE_ARGUMENTS);
					let expected_version_field =
						SatelliteProtocol::find_field(&request, field_id::EXECUTE_EXPECTED_VERSION);
					let expected_hash_field =
						SatelliteProtocol::find_field(&request, field_id::EXECUTE_EXPECTED_HASH);
					let mut next = PendingOp {
						request_id: request.request_id,
						kind: PendingKind::CpuAdd,
						a: 0,
						b: 0,
						input_a: 0,
						input_b: 0,
						output: 0,
						operation: String::new(),
						inputs: Vec::new(),
						arguments: Vec::new(),
						expected_version: 0,
						expected_hash: [0; 32],
						complete: false,
						status: status_code::OK,
						result: Vec::new(),
						profile: Vec::new(),
					};
					if operation == CPU_ADD_OPERATION
						&& inputs.is_empty()
						&& output == 0
						&& arguments.is_some_and(|arguments| arguments.data.len() == 8)
					{
						let arguments = arguments.expect("checked is_some").data.clone();
						next.kind = PendingKind::CpuAdd;
						next.a = read_u32_le(&arguments);
						next.b = read_u32_le(&arguments[4..]);
					} else if operation == MATRIX_ADD_OPERATION
						&& inputs.len() == 2
						&& output != 0
						&& arguments.is_none()
						&& self.engine.is_some()
					{
						let a = find_object(&objects, inputs[0]);
						let b = find_object(&objects, inputs[1]);
						let invalid = objects.len() >= limits.max_objects as usize
							|| !matches!(
								(a, b),
								(Some(a), Some(b))
									if a.dtype == DType::F32 && b.dtype == DType::F32 && a.shape == b.shape
							) || find_object(&objects, output).is_some()
							|| a.is_none_or(|a| {
								a.data.len() as u64 > limits.max_resident_bytes
									|| total_object_bytes(&objects) > limits.max_resident_bytes - a.data.len() as u64
							});
						if invalid {
							let _ = write_message(
								&mut stream,
								&error_response(
									&request,
									status_code::FAILED_PRECONDITION,
									"satellite session: matrix inputs/output or quota are invalid",
									false,
								),
							);
							continue;
						}
						next.kind = PendingKind::MatrixAddF32;
						next.input_a = inputs[0];
						next.input_b = inputs[1];
						next.output = output;
					} else if self.config.named_work.is_some()
						&& admits_named_operation(&self.config, &operation)
						&& !inputs.is_empty()
						&& output != 0
						&& self.engine.is_some()
						&& let (Some(version_field), Some(hash_field)) =
							(expected_version_field, expected_hash_field)
					{
						let expected_version = SatelliteProtocol::read_u64(version_field)?;
						let first = find_object(&objects, inputs[0]);
						let inputs_exist = first.is_some()
							&& inputs
								.iter()
								.all(|input| find_object(&objects, *input).is_some());
						let invalid = !inputs_exist
							|| objects.len() >= limits.max_objects as usize
							|| find_object(&objects, output).is_some()
							|| hash_field.data.len() != 32
							|| expected_version != first.map_or(0, |first| first.version)
							|| !equal_constant_time(&hash_field.data, &first.map_or([0; 32], |first| first.hash));
						if invalid {
							let _ = write_message(
								&mut stream,
								&error_response(
									&request,
									status_code::FAILED_PRECONDITION,
									"satellite session: named-work inputs, output, version, or hash are invalid",
									false,
								),
							);
							continue;
						}
						next.kind = PendingKind::NamedWork;
						next.operation = operation;
						next.inputs = inputs;
						next.output = output;
						next.expected_version = expected_version;
						next.expected_hash = hash_field.data[..32].try_into().unwrap();
						if let Some(arguments) = arguments {
							next.arguments = arguments.data.clone();
						}
					} else {
						let _ = write_message(
							&mut stream,
							&error_response(
								&request,
								status_code::INVALID_ARGUMENT,
								"satellite session: operation is unknown or unavailable",
								false,
							),
						);
						continue;
					}
					pending = Some(next);
					write_message(
						&mut stream,
						&result_response(
							&request,
							request.request_id,
							false,
							status_code::OK,
							&[],
							&[],
						),
					)?;
					continue;
				}
				SatelliteMessageType::Wait
				| SatelliteMessageType::Poll
				| SatelliteMessageType::Cancel
				| SatelliteMessageType::GetResult => {
					let target_field = match request.message_type {
						SatelliteMessageType::Wait => field_id::WAIT_REQUEST_ID,
						SatelliteMessageType::Cancel => field_id::CANCEL_REQUEST_ID,
						SatelliteMessageType::GetResult => field_id::GET_RESULT_REQUEST_ID,
						_ => field_id::POLL_REQUEST_ID,
					};
					let target = SatelliteProtocol::read_u64(required_field(&request, target_field)?)?;
					let Some(active) = pending
						.as_mut()
						.filter(|pending| pending.request_id == target)
					else {
						let _ = write_message(
							&mut stream,
							&error_response(
								&request,
								status_code::NOT_FOUND,
								"satellite session: request is not retained",
								false,
							),
						);
						continue;
					};
					if request.message_type == SatelliteMessageType::Cancel && !active.complete {
						active.complete = true;
						active.status = status_code::CANCELLED;
					}
					if request.message_type == SatelliteMessageType::Wait && !active.complete {
						match active.kind {
							PendingKind::CpuAdd => {
								active.result.clear();
								append_u32_le(&mut active.result, active.a.wrapping_add(active.b));
								active.complete = true;
							}
							PendingKind::MatrixAddF32 => {
								let a_object = find_object(&objects, active.input_a);
								let b_object = find_object(&objects, active.input_b);
								let (shape, a_data, b_data) = match (a_object, b_object) {
									(Some(a), Some(b)) => (
										matrix_shape_from_span(&a.shape),
										decode_f32(&a.data),
										decode_f32(&b.data),
									),
									_ => {
										return Err(send_failure(
											&mut stream,
											&request,
											status_code::DATA_LOSS,
											"satellite session: retained matrix object is invalid",
											true,
										));
									}
								};
								let (Ok(shape), Ok(a_values), Ok(b_values)) = (shape, a_data, b_data) else {
									return Err(send_failure(
										&mut stream,
										&request,
										status_code::DATA_LOSS,
										"satellite session: retained matrix object is invalid",
										true,
									));
								};
								let Some(engine) = self.engine else {
									return Err(send_failure(
										&mut stream,
										&request,
										status_code::INTERNAL,
										"satellite session: matrix work requires an engine",
										true,
									));
								};
								let captured = engine.capture(|| {
									let a = Matrix::new_from_slice(engine, shape.clone(), &a_values)?;
									let b = Matrix::new_from_slice(engine, shape, &b_values)?;
									crate::matrix::add(&a, &b)
								});
								let Ok((plan, output_matrix)) = captured else {
									let _ = write_message(
										&mut stream,
										&error_response(
											&request,
											status_code::DATA_LOSS,
											"satellite session: retained matrix object is invalid",
											false,
										),
									);
									continue;
								};
								let event = match engine.submit(&plan) {
									Ok(event) => event,
									Err(submitted) => {
										let _ = write_message(
											&mut stream,
											&error_response(
												&request,
												kind_code(submitted.kind()),
												submitted.message(),
												false,
											),
										);
										continue;
									}
								};
								self.last_gpu_event_was_owned = engine.handle().owns_event(&event);
								self.last_gpu_event_value = event.epoch();
								let wait_status = event.wait();
								self.last_gpu_event_completed = event.is_complete().unwrap_or(false);
								if wait_status.is_err()
									|| !self.last_gpu_event_was_owned
									|| !self.last_gpu_event_completed
								{
									let failure = match wait_status {
										Err(error) => error,
										Ok(()) => {
											Error::internal("satellite session: exact GPU event provenance failed")
										}
									};
									return Err(send_failure_error(&mut stream, &request, failure, true));
								}
								let host: Vec<f32> = match output_matrix.read() {
									Ok(host) => host,
									Err(read) => {
										return Err(send_failure_error(&mut stream, &request, read, true));
									}
								};
								let mut data = Vec::with_capacity(host.len() * 4);
								for value in host {
									append_f32_le(&mut data, value);
								}
								let hash = hash_bytes(&data);
								let a_object =
									find_object(&objects, active.input_a).expect("matrix input retained");
								objects.push(SessionObject {
									id: active.output,
									version: 0,
									dtype: DType::F32,
									shape: a_object.shape.clone(),
									data,
									hash,
								});
								active.complete = true;
							}
							PendingKind::NamedWork => {
								let mut views = Vec::with_capacity(active.inputs.len());
								for input in &active.inputs {
									let Some(object) = find_object(&objects, *input) else {
										return Err(send_failure(
											&mut stream,
											&request,
											status_code::DATA_LOSS,
											"satellite session: retained named-work input disappeared",
											true,
										));
									};
									views.push(SatelliteObjectView {
										id: object.id,
										version: object.version,
										dtype: object.dtype,
										shape: &object.shape,
										data: &object.data,
										hash: object.hash,
									});
								}
								let named_request = SatelliteNamedRequest {
									operation: active.operation.clone(),
									inputs: views,
									arguments: &active.arguments,
									expected_version: active.expected_version,
									expected_hash: active.expected_hash,
								};
								let Some(engine) = self.engine else {
									return Err(send_failure(
										&mut stream,
										&request,
										status_code::INTERNAL,
										"satellite session: named work requires an engine",
										true,
									));
								};
								let named_work = self
									.config
									.named_work
									.as_ref()
									.expect("named work admitted only when configured");
								let named_result = match named_work(engine, &named_request) {
									Ok(result) => result,
									Err(error) => {
										return Err(send_failure_error(&mut stream, &request, error, true));
									}
								};
								let elements = checked_element_count(&named_result.shape);
								let scalar_bytes = named_result.dtype.size_bytes();
								let event_owned = engine.handle().owns_event(&named_result.completion);
								let event_complete = named_result.completion.is_complete().unwrap_or(false);
								let data_len = named_result.data.len();
								let profile_len = named_result.profile.len();
								let result_fits_payload = data_len <= limits.max_payload_bytes as usize
									&& profile_len <= limits.max_payload_bytes as usize - data_len
									&& RESULT_ENVELOPE_BYTES
										<= limits.max_payload_bytes as usize - data_len - profile_len;
								let invalid = elements
									.map(|count| {
										count > u64::MAX / scalar_bytes as u64
											|| count * scalar_bytes as u64 != data_len as u64
									})
									.unwrap_or(true)
									|| !result_fits_payload
									|| profile_len > MAX_RESULT_PROFILE_BYTES
									|| data_len as u64 > limits.max_resident_bytes
									|| total_object_bytes(&objects) > limits.max_resident_bytes - data_len as u64
									|| !event_owned
									|| !event_complete;
								if invalid {
									return Err(send_failure(
										&mut stream,
										&request,
										status_code::DATA_LOSS,
										"satellite session: named-work result contract is invalid",
										true,
									));
								}
								self.last_gpu_event_was_owned = event_owned;
								self.last_gpu_event_completed = event_complete;
								self.last_gpu_event_value = named_result.completion.epoch();
								objects.push(SessionObject {
									id: active.output,
									version: named_result.version,
									dtype: named_result.dtype,
									shape: named_result.shape,
									hash: hash_bytes(&named_result.data),
									data: named_result.data,
								});
								active.profile = named_result.profile;
								active.complete = true;
							}
						}
					}
					let result_bytes: &[u8] = if active.complete && active.status == status_code::OK {
						match active.kind {
							PendingKind::CpuAdd => &active.result,
							_ => find_object(&objects, active.output)
								.map(|object| object.data.as_slice())
								.unwrap_or(&[]),
						}
					} else {
						&[]
					};
					let response = result_response(
						&request,
						target,
						active.complete,
						active.status,
						result_bytes,
						&active.profile,
					);
					write_message(&mut stream, &response)?;
					continue;
				}
				SatelliteMessageType::Close => {
					write_message(
						&mut stream,
						&result_response(
							&request,
							request.request_id,
							true,
							status_code::OK,
							&[],
							&[],
						),
					)?;
					return Ok(());
				}
				SatelliteMessageType::Abort => {
					write_message(
						&mut stream,
						&result_response(
							&request,
							request.request_id,
							true,
							status_code::ABORTED,
							&[],
							&[],
						),
					)?;
					return Err(Error::aborted("satellite session: peer aborted the epoch"));
				}
				_ => {
					let _ = write_message(
						&mut stream,
						&error_response(
							&request,
							status_code::UNIMPLEMENTED,
							"satellite session: message is not implemented by the CPU oracle",
							false,
						),
					);
					continue;
				}
			}
		}
	}
}

/// Synchronous satellite client. Dropping the session closes the socket only;
/// it does not submit, wait, or drain any admitted pending operation.
pub(crate) struct SatelliteClientSession {
	stream: TcpStream,
	probe: SatelliteProbe,
	next_request_id: u64,
	open: bool,
}

impl SatelliteClientSession {
	pub fn connect(host: &str, port: u16, config: SatelliteSessionConfig) -> Result<Self> {
		if host != "127.0.0.1" {
			return Err(Error::permission_denied(
				"satellite session: only explicit IPv4 loopback is admitted",
			));
		}
		if !valid_limits(&config.limits) {
			return Err(Error::invalid_argument(
				"satellite session: invalid client secret or limits",
			));
		}
		let mut stream = TcpStream::connect(host, port)?;
		stream.set_io_timeout(Duration::from_millis(u64::from(config.io_timeout_ms)))?;

		let client_nonce = random_nonce()?;
		let proof = kmac(
			&config.mac,
			config.secret.bytes(),
			&client_nonce,
			HELLO_CUSTOM,
		)?;
		let build_hash = satellite_build_hash();
		let schema_hash = satellite_schema_hash();
		let hello = SatelliteMessage {
			message_type: SatelliteMessageType::Hello,
			request_id: 1,
			session_epoch: 0,
			fields: vec![
				SatelliteField::u16(field_id::HELLO_PROTOCOL_MIN, SatelliteProtocol::VERSION),
				SatelliteField::u16(field_id::HELLO_PROTOCOL_MAX, SatelliteProtocol::VERSION),
				SatelliteField::bytes(field_id::HELLO_CLIENT_NONCE, &client_nonce),
				SatelliteField::bytes(field_id::HELLO_AUTH_PROOF, &proof),
				SatelliteField::bytes(field_id::HELLO_BUILD_HASH, &build_hash),
				SatelliteField::bytes(field_id::HELLO_SCHEMA_HASH, &schema_hash),
				SatelliteField::u32(
					field_id::HELLO_MAX_PAYLOAD_BYTES,
					config.limits.max_payload_bytes,
				),
				SatelliteField::u64(
					field_id::HELLO_MAX_RESIDENT_BYTES,
					config.limits.max_resident_bytes,
				),
				SatelliteField::u32(field_id::HELLO_MAX_OBJECTS, config.limits.max_objects),
				SatelliteField::u32(field_id::HELLO_MAX_INFLIGHT, config.limits.max_inflight),
			],
			..SatelliteMessage::default()
		};
		write_message(&mut stream, &hello)?;
		let reply = read_message(&mut stream, config.limits.max_payload_bytes)?;
		if reply.message_type == SatelliteMessageType::Error {
			return Err(decode_error(&reply));
		}
		if reply.message_type != SatelliteMessageType::HelloReply
			|| reply.flags != SatelliteProtocol::RESPONSE_FLAG
			|| reply.request_id != hello.request_id
			|| reply.session_epoch == 0
		{
			return Err(Error::data_loss(
				"satellite session: malformed HELLO_REPLY envelope",
			));
		}
		let server_nonce = required_field(&reply, field_id::HELLO_REPLY_SERVER_NONCE)?;
		let server_proof = required_field(&reply, field_id::HELLO_REPLY_AUTH_PROOF)?;
		let epoch =
			SatelliteProtocol::read_u64(required_field(&reply, field_id::HELLO_REPLY_SESSION_EPOCH)?)?;
		if epoch != reply.session_epoch {
			return Err(Error::data_loss(
				"satellite session: reply epoch fields disagree",
			));
		}
		let mut reply_proof_data = Vec::with_capacity(72);
		reply_proof_data.extend_from_slice(&client_nonce);
		reply_proof_data.extend_from_slice(&server_nonce.data);
		append_u64_le(&mut reply_proof_data, epoch);
		let expected_reply_proof = kmac(
			&config.mac,
			config.secret.bytes(),
			&reply_proof_data,
			REPLY_CUSTOM,
		)?;
		if !equal_constant_time(&server_proof.data, &expected_reply_proof) {
			return Err(Error::unauthenticated(
				"satellite session: server authentication failed",
			));
		}

		let probe = SatelliteProbe {
			session_epoch: epoch,
			device_name: SatelliteProtocol::read_string(required_field(
				&reply,
				field_id::HELLO_REPLY_DEVICE_NAME,
			)?)?,
			limits: SatelliteLimits {
				max_payload_bytes: SatelliteProtocol::read_u32(required_field(
					&reply,
					field_id::HELLO_REPLY_MAX_PAYLOAD_BYTES,
				)?)?,
				max_resident_bytes: SatelliteProtocol::read_u64(required_field(
					&reply,
					field_id::HELLO_REPLY_MAX_RESIDENT_BYTES,
				)?)?,
				max_objects: SatelliteProtocol::read_u32(required_field(
					&reply,
					field_id::HELLO_REPLY_MAX_OBJECTS,
				)?)?,
				max_inflight: SatelliteProtocol::read_u32(required_field(
					&reply,
					field_id::HELLO_REPLY_MAX_INFLIGHT,
				)?)?,
			},
			build_hash,
			schema_hash,
		};
		if !valid_limits(&probe.limits)
			|| probe.limits.max_payload_bytes > config.limits.max_payload_bytes
			|| probe.limits.max_resident_bytes > config.limits.max_resident_bytes
			|| probe.limits.max_objects > config.limits.max_objects
		{
			return Err(Error::data_loss(
				"satellite session: server returned invalid negotiated limits",
			));
		}
		Ok(Self {
			stream,
			probe,
			next_request_id: 2,
			open: true,
		})
	}

	pub const fn probe(&self) -> &SatelliteProbe {
		&self.probe
	}

	pub fn put_f32(&mut self, object_id: u64, shape: &[i64], values: &[f32]) -> Result<()> {
		self.put_f32_versioned(object_id, 0, shape, values)
	}

	pub fn put_f32_versioned(
		&mut self,
		object_id: u64,
		version: u64,
		shape: &[i64],
		values: &[f32],
	) -> Result<()> {
		if object_id == 0 {
			return Err(Error::invalid_argument(
				"satellite session: object id must be non-zero",
			));
		}
		let elements = checked_element_count(shape)?;
		if elements != values.len() as u64 {
			return Err(Error::with_kind(
				ErrorKind::InvalidArgument,
				"satellite session: shape and value count disagree",
			));
		}
		let mut data = Vec::with_capacity(values.len() * 4);
		for &value in values {
			append_f32_le(&mut data, value);
		}
		self.put_object(object_id, version, DType::F32, shape, &data)
	}

	pub fn put_u32(&mut self, object_id: u64, shape: &[i64], values: &[u32]) -> Result<()> {
		self.put_u32_versioned(object_id, 0, shape, values)
	}

	pub fn put_u32_versioned(
		&mut self,
		object_id: u64,
		version: u64,
		shape: &[i64],
		values: &[u32],
	) -> Result<()> {
		if object_id == 0 {
			return Err(Error::invalid_argument(
				"satellite session: object id must be non-zero",
			));
		}
		let elements = checked_element_count(shape)?;
		if elements != values.len() as u64 {
			return Err(Error::with_kind(
				ErrorKind::InvalidArgument,
				"satellite session: shape and value count disagree",
			));
		}
		let mut data = Vec::with_capacity(values.len() * 4);
		for &value in values {
			append_u32_le(&mut data, value);
		}
		self.put_object(object_id, version, DType::U32, shape, &data)
	}

	fn put_object(
		&mut self,
		object_id: u64,
		version: u64,
		dtype: DType,
		shape: &[i64],
		data: &[u8],
	) -> Result<()> {
		let hash = hash_bytes(data);
		let fields = vec![
			SatelliteField::u64(field_id::PUT_OBJECT_ID, object_id),
			SatelliteField::u8(field_id::PUT_OBJECT_DTYPE, dtype_wire(dtype)),
			SatelliteField::i64_array(field_id::PUT_OBJECT_SHAPE, shape),
			SatelliteField::bytes(field_id::PUT_OBJECT_DATA, data),
			SatelliteField::bytes(field_id::PUT_OBJECT_CONTENT_HASH, &hash),
			SatelliteField::u64(field_id::PUT_OBJECT_VERSION, version),
		];
		let response = self.exchange(SatelliteMessageType::PutObject, fields)?;
		let (id_field, version_field, hash_field) = (
			SatelliteProtocol::find_field(&response, field_id::RESULT_OBJECT_ID),
			SatelliteProtocol::find_field(&response, field_id::RESULT_OBJECT_VERSION),
			SatelliteProtocol::find_field(&response, field_id::RESULT_OBJECT_HASH),
		);
		let (Some(id_field), Some(version_field), Some(hash_field)) =
			(id_field, version_field, hash_field)
		else {
			return Err(Error::data_loss(
				"satellite session: object acknowledgement is incomplete",
			));
		};
		let acknowledged_id = SatelliteProtocol::read_u64(id_field)?;
		let acknowledged_version = SatelliteProtocol::read_u64(version_field)?;
		if acknowledged_id != object_id
			|| acknowledged_version != version
			|| !equal_constant_time(&hash_field.data, &hash)
		{
			return Err(Error::data_loss(
				"satellite session: object acknowledgement identity mismatch",
			));
		}
		Ok(())
	}

	pub fn drop_object(&mut self, object_id: u64) -> Result<()> {
		let fields = vec![SatelliteField::u64(field_id::DROP_OBJECT_ID, object_id)];
		self.exchange(SatelliteMessageType::DropObject, fields)?;
		Ok(())
	}

	fn exchange(
		&mut self,
		message_type: SatelliteMessageType,
		fields: Vec<SatelliteField>,
	) -> Result<SatelliteMessage> {
		if !self.open {
			return Err(Error::failed_precondition(
				"satellite session: client is closed",
			));
		}
		let request = SatelliteMessage {
			message_type,
			request_id: self.next_request_id,
			session_epoch: self.probe.session_epoch,
			fields,
			..SatelliteMessage::default()
		};
		self.next_request_id += 1;
		write_message(&mut self.stream, &request)?;
		let response = read_message(&mut self.stream, self.probe.limits.max_payload_bytes)?;
		if response.flags != SatelliteProtocol::RESPONSE_FLAG
			|| response.request_id != request.request_id
			|| response.session_epoch != request.session_epoch
		{
			return Err(Error::data_loss(
				"satellite session: response envelope does not match request",
			));
		}
		if response.message_type == SatelliteMessageType::Error {
			return Err(decode_error(&response));
		}
		if response.message_type != SatelliteMessageType::Result {
			return Err(Error::data_loss(
				"satellite session: expected RESULT response",
			));
		}
		Ok(response)
	}

	pub fn start_cpu_add(&mut self, a: u32, b: u32) -> Result<u64> {
		let mut arguments = Vec::with_capacity(8);
		append_u32_le(&mut arguments, a);
		append_u32_le(&mut arguments, b);
		let fields = vec![
			SatelliteField::string(field_id::EXECUTE_OPERATION, CPU_ADD_OPERATION),
			SatelliteField::u64_array(field_id::EXECUTE_INPUT_OBJECT_IDS, &[]),
			SatelliteField::u64(field_id::EXECUTE_OUTPUT_OBJECT_ID, 0),
			SatelliteField::bytes(field_id::EXECUTE_ARGUMENTS, &arguments),
		];
		let request_id = self.next_request_id;
		self.exchange(SatelliteMessageType::ExecuteNamed, fields)?;
		Ok(request_id)
	}

	pub fn start_matrix_add_f32(&mut self, a: u64, b: u64, output: u64) -> Result<u64> {
		let fields = vec![
			SatelliteField::string(field_id::EXECUTE_OPERATION, MATRIX_ADD_OPERATION),
			SatelliteField::u64_array(field_id::EXECUTE_INPUT_OBJECT_IDS, &[a, b]),
			SatelliteField::u64(field_id::EXECUTE_OUTPUT_OBJECT_ID, output),
		];
		let request_id = self.next_request_id;
		self.exchange(SatelliteMessageType::ExecuteNamed, fields)?;
		Ok(request_id)
	}

	pub fn start_named(
		&mut self,
		operation: &str,
		inputs: &[u64],
		output: u64,
		arguments: &[u8],
		expected_version: u64,
		expected_hash: &[u8; 32],
	) -> Result<u64> {
		if operation.is_empty() || inputs.is_empty() || output == 0 {
			return Err(Error::invalid_argument(
				"satellite session: named operation, inputs, or output is invalid",
			));
		}
		if inputs.contains(&0) {
			return Err(Error::invalid_argument(
				"satellite session: named-operation input id must be non-zero",
			));
		}
		let mut fields = vec![
			SatelliteField::string(field_id::EXECUTE_OPERATION, operation),
			SatelliteField::u64_array(field_id::EXECUTE_INPUT_OBJECT_IDS, inputs),
			SatelliteField::u64(field_id::EXECUTE_OUTPUT_OBJECT_ID, output),
		];
		if !arguments.is_empty() {
			fields.push(SatelliteField::bytes(
				field_id::EXECUTE_ARGUMENTS,
				arguments,
			));
		}
		fields.push(SatelliteField::u64(
			field_id::EXECUTE_EXPECTED_VERSION,
			expected_version,
		));
		fields.push(SatelliteField::bytes(
			field_id::EXECUTE_EXPECTED_HASH,
			expected_hash,
		));
		let request_id = self.next_request_id;
		self.exchange(SatelliteMessageType::ExecuteNamed, fields)?;
		Ok(request_id)
	}

	pub fn poll(&mut self, request_id: u64) -> Result<SatelliteMessage> {
		let fields = vec![SatelliteField::u64(field_id::POLL_REQUEST_ID, request_id)];
		self.exchange(SatelliteMessageType::Poll, fields)
	}

	pub fn wait(&mut self, request_id: u64) -> Result<SatelliteMessage> {
		let fields = vec![SatelliteField::u64(field_id::WAIT_REQUEST_ID, request_id)];
		self.exchange(SatelliteMessageType::Wait, fields)
	}

	pub fn cancel(&mut self, request_id: u64) -> Result<SatelliteMessage> {
		let fields = vec![SatelliteField::u64(field_id::CANCEL_REQUEST_ID, request_id)];
		self.exchange(SatelliteMessageType::Cancel, fields)
	}

	pub fn get_result(&mut self, request_id: u64) -> Result<SatelliteMessage> {
		let fields = vec![SatelliteField::u64(
			field_id::GET_RESULT_REQUEST_ID,
			request_id,
		)];
		self.exchange(SatelliteMessageType::GetResult, fields)
	}

	pub fn close(&mut self) -> Result<()> {
		if !self.open {
			return Ok(());
		}
		self.exchange(SatelliteMessageType::Close, Vec::new())?;
		self.open = false;
		self.stream.close();
		Ok(())
	}

	pub fn abort(&mut self, reason: &str) -> Result<()> {
		if !self.open {
			return Ok(());
		}
		let fields = vec![SatelliteField::string(
			field_id::ABORT_REASON,
			if reason.is_empty() {
				"client abort"
			} else {
				reason
			},
		)];
		let response = self.exchange(SatelliteMessageType::Abort, fields);
		self.open = false;
		self.stream.close();
		response.map(|_| ())
	}

	pub fn read_cpu_add_result(message: &SatelliteMessage) -> Result<u32> {
		let (bytes, _) = read_result_envelope(message)?;
		let Some(bytes) = bytes else {
			return Err(Error::data_loss(
				"satellite session: CPU add result has the wrong size",
			));
		};
		if bytes.len() != 4 {
			return Err(Error::data_loss(
				"satellite session: CPU add result has the wrong size",
			));
		}
		Ok(read_u32_le(bytes))
	}

	pub fn read_f32_result(message: &SatelliteMessage) -> Result<Vec<f32>> {
		let (bytes, _) = read_result_envelope(message)?;
		let Some(bytes) = bytes else {
			return Err(Error::data_loss(
				"satellite session: FP32 result is missing",
			));
		};
		decode_f32(bytes)
	}

	pub fn read_bytes_result(message: &SatelliteMessage) -> Result<Vec<u8>> {
		let (bytes, _) = read_result_envelope(message)?;
		let Some(bytes) = bytes else {
			return Err(Error::data_loss(
				"satellite session: byte result is missing",
			));
		};
		Ok(bytes.to_vec())
	}

	pub fn read_profile(message: &SatelliteMessage) -> Result<Vec<u8>> {
		Ok(
			SatelliteProtocol::find_field(message, field_id::RESULT_PROFILE)
				.map(|profile| profile.data.clone())
				.unwrap_or_default(),
		)
	}
}

/// Shared complete/status/bytes envelope decode for the typed result readers.
fn read_result_envelope(message: &SatelliteMessage) -> Result<(Option<&[u8]>, u32)> {
	let (Ok(complete_field), Ok(status_field)) = (
		required_field(message, field_id::RESULT_COMPLETE),
		required_field(message, field_id::RESULT_STATUS_CODE),
	) else {
		return Err(Error::data_loss("satellite session: malformed RESULT"));
	};
	let complete = SatelliteProtocol::read_u8(complete_field)?;
	let status = SatelliteProtocol::read_u32(status_field)?;
	if !is_known_status_code(status) {
		return Err(Error::data_loss(
			"satellite session: invalid result status code",
		));
	}
	if complete == 0 {
		return Err(Error::failed_precondition(
			"satellite session: result is not complete",
		));
	}
	if status != status_code::OK {
		return Err(error_for_status(
			status,
			"satellite session: operation did not complete successfully",
		));
	}
	Ok((
		SatelliteProtocol::find_field(message, field_id::RESULT_BYTES)
			.map(|bytes| bytes.data.as_slice()),
		status,
	))
}

#[cfg(test)]
#[path = "../../../../test/rs/net/satellite_session_unit.rs"]
mod tests;
