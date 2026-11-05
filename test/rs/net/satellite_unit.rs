//! Donor: test/cpp/net/satellite/testSatelliteProtocol.cpp.
//! Canonical wire codec coverage for the private satellite protocol.

use super::*;
use crate::ErrorKind;

fn bytes32(seed: u8) -> [u8; 32] {
	let mut bytes = [0_u8; 32];
	for (index, byte) in bytes.iter_mut().enumerate() {
		*byte = seed.wrapping_add(index as u8);
	}
	bytes
}

fn add_bytes32(message: &mut SatelliteMessage, id: u16, seed: u8) {
	message
		.fields
		.push(SatelliteField::bytes(id, &bytes32(seed)));
}

fn make_message(message_type: SatelliteMessageType) -> SatelliteMessage {
	let mut message = SatelliteMessage {
		message_type,
		request_id: 17,
		session_epoch: 23,
		..SatelliteMessage::default()
	};
	match message_type {
		SatelliteMessageType::Hello => {
			message.session_epoch = 0;
			message
				.fields
				.push(SatelliteField::u16(field_id::HELLO_PROTOCOL_MIN, 1));
			message
				.fields
				.push(SatelliteField::u16(field_id::HELLO_PROTOCOL_MAX, 1));
			add_bytes32(&mut message, field_id::HELLO_CLIENT_NONCE, 1);
			add_bytes32(&mut message, field_id::HELLO_AUTH_PROOF, 2);
			add_bytes32(&mut message, field_id::HELLO_BUILD_HASH, 3);
			add_bytes32(&mut message, field_id::HELLO_SCHEMA_HASH, 4);
			message.fields.push(SatelliteField::u32(
				field_id::HELLO_MAX_PAYLOAD_BYTES,
				65536,
			));
			message.fields.push(SatelliteField::u64(
				field_id::HELLO_MAX_RESIDENT_BYTES,
				1048576,
			));
			message
				.fields
				.push(SatelliteField::u32(field_id::HELLO_MAX_OBJECTS, 8));
			message
				.fields
				.push(SatelliteField::u32(field_id::HELLO_MAX_INFLIGHT, 1));
		}
		SatelliteMessageType::HelloReply => {
			message.flags = SatelliteProtocol::RESPONSE_FLAG;
			message
				.fields
				.push(SatelliteField::u16(field_id::HELLO_REPLY_PROTOCOL, 1));
			add_bytes32(&mut message, field_id::HELLO_REPLY_SERVER_NONCE, 5);
			add_bytes32(&mut message, field_id::HELLO_REPLY_AUTH_PROOF, 6);
			message
				.fields
				.push(SatelliteField::u64(field_id::HELLO_REPLY_SESSION_EPOCH, 23));
			message.fields.push(SatelliteField::string(
				field_id::HELLO_REPLY_DEVICE_NAME,
				"Iris Xe",
			));
			message.fields.push(SatelliteField::u32(
				field_id::HELLO_REPLY_MAX_PAYLOAD_BYTES,
				65536,
			));
			message.fields.push(SatelliteField::u64(
				field_id::HELLO_REPLY_MAX_RESIDENT_BYTES,
				1048576,
			));
			message
				.fields
				.push(SatelliteField::u32(field_id::HELLO_REPLY_MAX_OBJECTS, 8));
			message
				.fields
				.push(SatelliteField::u32(field_id::HELLO_REPLY_MAX_INFLIGHT, 1));
		}
		SatelliteMessageType::PutObject => {
			message
				.fields
				.push(SatelliteField::u64(field_id::PUT_OBJECT_ID, 7));
			message
				.fields
				.push(SatelliteField::u8(field_id::PUT_OBJECT_DTYPE, 0));
			message.fields.push(SatelliteField::i64_array(
				field_id::PUT_OBJECT_SHAPE,
				&[2, 3],
			));
			message.fields.push(SatelliteField::bytes(
				field_id::PUT_OBJECT_DATA,
				&[1, 2, 3, 4],
			));
			add_bytes32(&mut message, field_id::PUT_OBJECT_CONTENT_HASH, 7);
			message
				.fields
				.push(SatelliteField::u64(field_id::PUT_OBJECT_VERSION, 11));
		}
		SatelliteMessageType::DropObject => {
			message
				.fields
				.push(SatelliteField::u64(field_id::DROP_OBJECT_ID, 7));
		}
		SatelliteMessageType::ExecuteNamed => {
			message.fields.push(SatelliteField::string(
				field_id::EXECUTE_OPERATION,
				"matrix-add-f32-v1",
			));
			message.fields.push(SatelliteField::u64_array(
				field_id::EXECUTE_INPUT_OBJECT_IDS,
				&[1, 2],
			));
			message
				.fields
				.push(SatelliteField::u64(field_id::EXECUTE_OUTPUT_OBJECT_ID, 3));
			message
				.fields
				.push(SatelliteField::bytes(field_id::EXECUTE_ARGUMENTS, &[9, 8]));
			message
				.fields
				.push(SatelliteField::u64(field_id::EXECUTE_EXPECTED_VERSION, 4));
			add_bytes32(&mut message, field_id::EXECUTE_EXPECTED_HASH, 8);
		}
		SatelliteMessageType::Wait => {
			message
				.fields
				.push(SatelliteField::u64(field_id::WAIT_REQUEST_ID, 11));
		}
		SatelliteMessageType::Poll => {
			message
				.fields
				.push(SatelliteField::u64(field_id::POLL_REQUEST_ID, 11));
		}
		SatelliteMessageType::Cancel => {
			message
				.fields
				.push(SatelliteField::u64(field_id::CANCEL_REQUEST_ID, 11));
		}
		SatelliteMessageType::GetResult => {
			message
				.fields
				.push(SatelliteField::u64(field_id::GET_RESULT_REQUEST_ID, 11));
		}
		SatelliteMessageType::Error => {
			message.flags = SatelliteProtocol::RESPONSE_FLAG;
			message
				.fields
				.push(SatelliteField::u32(field_id::ERROR_STATUS_CODE, 1));
			message.fields.push(SatelliteField::string(
				field_id::ERROR_MESSAGE,
				"bad request",
			));
			message
				.fields
				.push(SatelliteField::u8(field_id::ERROR_POISONED, 0));
		}
		SatelliteMessageType::Abort => {
			message.fields.push(SatelliteField::string(
				field_id::ABORT_REASON,
				"caller abort",
			));
		}
		SatelliteMessageType::Close => {}
		SatelliteMessageType::Result => {
			message.flags = SatelliteProtocol::RESPONSE_FLAG;
			message
				.fields
				.push(SatelliteField::u64(field_id::RESULT_REQUEST_ID, 11));
			message
				.fields
				.push(SatelliteField::u8(field_id::RESULT_COMPLETE, 1));
			message
				.fields
				.push(SatelliteField::u32(field_id::RESULT_STATUS_CODE, 0));
			message
				.fields
				.push(SatelliteField::bytes(field_id::RESULT_BYTES, &[4, 5, 6]));
			message
				.fields
				.push(SatelliteField::bytes(field_id::RESULT_PROFILE, &[7, 8]));
		}
	}
	message
}

fn refresh_checksum(bytes: &mut [u8]) {
	let checksum =
		SatelliteProtocol::stable_digest(&bytes[..32], &bytes[SatelliteProtocol::HEADER_BYTES..]);
	bytes[32..64].copy_from_slice(&checksum);
}

fn write_u32_le(data: &mut [u8], value: u32) {
	data[..4].copy_from_slice(&value.to_le_bytes());
}

fn next_random(state: &mut u64) -> u64 {
	*state ^= *state << 13;
	*state ^= *state >> 7;
	*state ^= *state << 17;
	*state
}

#[test]
fn every_schema_message_round_trips_canonically() {
	for message_type in SatelliteMessageType::ALL {
		let message = make_message(message_type);
		let encoded = SatelliteProtocol::encode(&message).expect("encode");
		let decoded = SatelliteProtocol::decode(&encoded).expect("decode");
		let reencoded = SatelliteProtocol::encode(&decoded).expect("re-encode");
		assert_eq!(encoded.len(), reencoded.len());
		assert_eq!(encoded, reencoded, "{message_type:?}");
	}
}

#[test]
fn typed_fields_use_canonical_little_endian_encoding() {
	let u16_field = SatelliteField::u16(field_id::HELLO_PROTOCOL_MIN, 0x1234);
	assert_eq!(u16_field.data.len(), 2);
	assert_eq!(u16_field.data[0], 0x34);
	assert_eq!(u16_field.data[1], 0x12);
	assert_eq!(
		SatelliteProtocol::read_u16(&u16_field).expect("read u16"),
		0x1234
	);

	let source = [0x0102030405060708_u64, 9];
	let array_field = SatelliteField::u64_array(field_id::EXECUTE_INPUT_OBJECT_IDS, &source);
	let values = SatelliteProtocol::read_u64_array(&array_field).expect("read u64 array");
	assert_eq!(values.len(), 2);
	assert_eq!(values[0], source[0]);
	assert_eq!(values[1], source[1]);

	assert_eq!(
		SatelliteProtocol::read_u8(&SatelliteField::u8(field_id::PUT_OBJECT_DTYPE, 3))
			.expect("read u8"),
		3
	);
	assert_eq!(
		SatelliteProtocol::read_u32(&SatelliteField::u32(
			field_id::ERROR_STATUS_CODE,
			0x89abcdef
		))
		.expect("read u32"),
		0x89abcdef
	);
	assert_eq!(
		SatelliteProtocol::read_u64(&SatelliteField::u64(
			field_id::RESULT_REQUEST_ID,
			0xfedcba9876543210
		))
		.expect("read u64"),
		0xfedcba9876543210
	);
	assert_eq!(
		SatelliteProtocol::read_f32(&SatelliteField::f32(field_id::RESULT_COMPLETE, 1.5))
			.expect("read f32"),
		1.5
	);
	assert_eq!(
		SatelliteProtocol::read_string(&SatelliteField::string(
			field_id::ABORT_REASON,
			"caller abort"
		))
		.expect("read string"),
		"caller abort"
	);
	assert_eq!(
		SatelliteProtocol::read_i64_array(&SatelliteField::i64_array(
			field_id::PUT_OBJECT_SHAPE,
			&[-2, 3]
		))
		.expect("read i64 array"),
		vec![-2_i64, 3]
	);

	let message = make_message(SatelliteMessageType::Wait);
	let field = SatelliteProtocol::find_field(&message, field_id::WAIT_REQUEST_ID)
		.expect("find request id field");
	assert_eq!(
		SatelliteProtocol::read_u64(field).expect("read request id"),
		11
	);
	assert!(SatelliteProtocol::find_field(&message, field_id::RESULT_BYTES).is_none());
}

#[test]
fn stable_digest_pins_bytes_and_span_boundary() {
	let first = *b"OA";
	let second = [0_u8, 1, 2];
	let expected = [
		0x5b, 0xd5, 0xd6, 0x1d, 0x3d, 0xf9, 0x30, 0xf6, 0x66, 0x82, 0xbd, 0xe7, 0xbc, 0x22, 0x7e, 0xae,
		0xc7, 0x56, 0x82, 0xb9, 0xf5, 0x10, 0x51, 0x7a, 0x9a, 0x26, 0x25, 0xd1, 0xb5, 0x3f, 0xe5, 0x4c,
	];
	let digest = SatelliteProtocol::stable_digest(&first, &second);
	assert_eq!(digest, expected);

	let different_first = [b'O', b'A', 0_u8];
	let different_second = [1_u8, 2];
	let different = SatelliteProtocol::stable_digest(&different_first, &different_second);
	assert_ne!(digest, different);
}

#[test]
fn validation_rejects_missing_duplicate_unsorted_and_unknown_fields() {
	let mut missing = make_message(SatelliteMessageType::DropObject);
	missing.fields.clear();
	assert!(SatelliteProtocol::validate(&missing).is_err());

	let mut duplicate = make_message(SatelliteMessageType::Hello);
	let last = duplicate.fields.last().unwrap().clone();
	duplicate.fields.push(last);
	assert!(SatelliteProtocol::validate(&duplicate).is_err());

	let mut unsorted = make_message(SatelliteMessageType::PutObject);
	unsorted.fields.swap(0, 1);
	assert!(SatelliteProtocol::validate(&unsorted).is_err());

	let mut unknown = make_message(SatelliteMessageType::DropObject);
	unknown.fields[0].id = 99;
	assert!(SatelliteProtocol::validate(&unknown).is_err());
}

#[test]
fn validation_rejects_bounds_kinds_and_malformed_utf8() {
	let mut wrong_kind = make_message(SatelliteMessageType::DropObject);
	wrong_kind.fields[0].kind = SatelliteFieldKind::Bytes;
	assert!(SatelliteProtocol::validate(&wrong_kind).is_err());

	let mut oversized = make_message(SatelliteMessageType::Abort);
	oversized.fields[0].data.resize(1025, 0);
	assert!(SatelliteProtocol::validate(&oversized).is_err());

	let mut utf8 = make_message(SatelliteMessageType::Abort);
	utf8.fields[0].data = vec![0xc0, 0x80];
	assert!(SatelliteProtocol::validate(&utf8).is_err());

	let mut partial_array = make_message(SatelliteMessageType::ExecuteNamed);
	partial_array.fields[1].data.push(0);
	assert!(SatelliteProtocol::validate(&partial_array).is_err());
}

#[test]
fn parser_rejects_every_truncation_and_checksum_damage() {
	let encoded =
		SatelliteProtocol::encode(&make_message(SatelliteMessageType::Hello)).expect("encode");
	for bytes in 0..encoded.len() {
		assert!(
			SatelliteProtocol::decode(&encoded[..bytes]).is_err(),
			"{bytes}"
		);
	}
	let mut corrupt = encoded.clone();
	*corrupt.last_mut().unwrap() ^= 0x80;
	assert_eq!(
		SatelliteProtocol::decode(&corrupt).unwrap_err().kind(),
		ErrorKind::DataLoss
	);
}

#[test]
fn parser_rejects_unknown_version_type_flags_and_trailing_bytes() {
	let encoded =
		SatelliteProtocol::encode(&make_message(SatelliteMessageType::Close)).expect("encode");

	let mut version = encoded.clone();
	version[4] = 2;
	refresh_checksum(&mut version);
	assert_eq!(
		SatelliteProtocol::decode(&version).unwrap_err().kind(),
		ErrorKind::FailedPrecondition
	);

	let mut message_type = encoded.clone();
	message_type[6] = 99;
	refresh_checksum(&mut message_type);
	assert_eq!(
		SatelliteProtocol::decode(&message_type).unwrap_err().kind(),
		ErrorKind::InvalidArgument
	);

	let mut flags = encoded.clone();
	flags[8] = 0x80;
	refresh_checksum(&mut flags);
	assert_eq!(
		SatelliteProtocol::decode(&flags).unwrap_err().kind(),
		ErrorKind::InvalidArgument
	);

	let mut trailing = encoded.clone();
	trailing.push(0);
	let payload_len = (trailing.len() - SatelliteProtocol::HEADER_BYTES) as u32;
	write_u32_le(&mut trailing[28..32], payload_len);
	refresh_checksum(&mut trailing);
	assert_eq!(
		SatelliteProtocol::decode(&trailing).unwrap_err().kind(),
		ErrorKind::DataLoss
	);
}

#[test]
fn parser_rejects_payload_allocation_limit_before_field_decode() {
	let mut header = vec![0_u8; SatelliteProtocol::HEADER_BYTES];
	header[0] = b'O';
	header[1] = b'A';
	header[2] = b'S';
	header[3] = b'P';
	write_u32_le(
		&mut header[28..32],
		SatelliteProtocol::MAX_PAYLOAD_BYTES as u32 + 1,
	);
	assert_eq!(
		SatelliteProtocol::decode(&header).unwrap_err().kind(),
		ErrorKind::ResourceExhausted
	);
}

#[test]
fn deterministic_mutations_either_reject_or_remain_canonical() {
	let types = [
		SatelliteMessageType::Hello,
		SatelliteMessageType::HelloReply,
		SatelliteMessageType::PutObject,
		SatelliteMessageType::ExecuteNamed,
		SatelliteMessageType::Error,
		SatelliteMessageType::Result,
	];
	let mut random = 0x4f41535046555a5a_u64;
	for iteration in 0..10000_u32 {
		let encoded = SatelliteProtocol::encode(&make_message(
			types[next_random(&mut random) as usize % types.len()],
		))
		.expect("encode");
		let mut mutated = encoded.clone();
		let penetrate_checksum = next_random(&mut random).is_multiple_of(3);
		let mutation_count = 1 + next_random(&mut random) % 6;
		for _ in 0..mutation_count {
			let first_byte = if penetrate_checksum {
				SatelliteProtocol::HEADER_BYTES
			} else {
				0
			};
			let index = first_byte + (next_random(&mut random) as usize % (mutated.len() - first_byte));
			mutated[index] ^= 1_u8 << (next_random(&mut random) % 8);
		}
		if penetrate_checksum {
			refresh_checksum(&mut mutated);
		}

		let Ok(decoded) = SatelliteProtocol::decode(&mutated) else {
			continue;
		};
		let reencoded = SatelliteProtocol::encode(&decoded).expect("re-encode");
		assert_eq!(mutated.len(), reencoded.len());
		assert_eq!(mutated, reencoded, "iteration {iteration}");
	}
}
