//! Persisted execution-plan descriptor tests.
//!
//! Donor: `oacpp/test/cpp/runtime/` execution-plan descriptor coverage —
//! envelope rejection (truncated header, bad magic, unsupported version,
//! invalid length, checksum mismatch) and same-plan byte-equality validation.

use super::*;

fn corrupt(mut descriptor: Vec<u8>, offset: usize, value: u8) -> Vec<u8> {
	descriptor[offset] = value;
	descriptor
}

#[test]
#[ignore = "requires an admitted Vulkan compute device"]
fn execution_plan_descriptor_roundtrips_and_revalidates() -> crate::Result<()> {
	let engine = match crate::Engine::new() {
		Ok(engine) => engine,
		Err(_) => return Ok(()),
	};
	let one = crate::matrix::ones(&engine, [2, 3])?;
	let two = crate::matrix::full(&engine, [2, 3], 2.0)?;
	let (plan, _output) = engine.capture(|| {
		let three = crate::matrix::add(&one, &two)?;
		crate::matrix::add(&three, &three)
	})?;

	let descriptor = plan.serialize_descriptor()?;
	assert!(descriptor.len() > DESCRIPTOR_HEADER_BYTES);
	assert_eq!(
		u64::from_le_bytes(descriptor[0..8].try_into().expect("header")),
		DESCRIPTOR_MAGIC
	);
	assert_eq!(
		u32::from_le_bytes(descriptor[8..12].try_into().expect("header")),
		DESCRIPTOR_VERSION
	);
	plan.validate_serialized_descriptor(&descriptor)?;

	// Serialization is deterministic: a fresh serialization and a structurally
	// identical recapture both reproduce the persisted bytes.
	assert_eq!(plan.serialize_descriptor()?, descriptor);
	let (recaptured, _output) = engine.capture(|| {
		let three = crate::matrix::add(&one, &two)?;
		crate::matrix::add(&three, &three)
	})?;
	assert_eq!(recaptured.serialize_descriptor()?, descriptor);
	recaptured.validate_serialized_descriptor(&descriptor)?;
	Ok(())
}

#[test]
#[ignore = "requires an admitted Vulkan compute device"]
fn execution_plan_descriptor_rejects_malformed_and_foreign_envelopes() -> crate::Result<()> {
	let engine = match crate::Engine::new() {
		Ok(engine) => engine,
		Err(_) => return Ok(()),
	};
	let one = crate::matrix::ones(&engine, [2, 3])?;
	let two = crate::matrix::full(&engine, [2, 3], 2.0)?;
	let (plan, _output) = engine.capture(|| crate::matrix::add(&one, &two))?;
	let descriptor = plan.serialize_descriptor()?;

	let error = plan
		.validate_serialized_descriptor(&descriptor[..8])
		.expect_err("truncated header must be rejected");
	assert_eq!(error.kind(), crate::ErrorKind::InvalidArgument);

	let error = plan
		.validate_serialized_descriptor(&corrupt(descriptor.clone(), 0, 0x42))
		.expect_err("invalid magic must be rejected");
	assert_eq!(error.kind(), crate::ErrorKind::InvalidArgument);

	let error = plan
		.validate_serialized_descriptor(&corrupt(descriptor.clone(), 8, 0xff))
		.expect_err("unsupported version must be rejected");
	assert_eq!(error.kind(), crate::ErrorKind::FailedPrecondition);

	let mut truncated_payload = descriptor.clone();
	truncated_payload.pop();
	let error = plan
		.validate_serialized_descriptor(&truncated_payload)
		.expect_err("payload length mismatch must be rejected");
	assert_eq!(error.kind(), crate::ErrorKind::DataLoss);

	let error = plan
		.validate_serialized_descriptor(&corrupt(descriptor.clone(), descriptor.len() - 1, 0xa5))
		.expect_err("checksum mismatch must be rejected");
	assert_eq!(error.kind(), crate::ErrorKind::DataLoss);

	let (foreign, _output) = engine.capture(|| crate::matrix::scale(&one, 2.0))?;
	let error = plan
		.validate_serialized_descriptor(&foreign.serialize_descriptor()?)
		.expect_err("a different compiled plan must not validate");
	assert_eq!(error.kind(), crate::ErrorKind::FailedPrecondition);
	Ok(())
}
