use super::{Entropy, consume_bytes};
use crate::{Error, ErrorKind, Result};

#[test]
fn entropy_rejects_invalid_sizes_before_invoking_source() {
	let result = Entropy::<31>::from_source(|_| panic!("invalid extent invoked source"));
	assert!(matches!(result, Err(e) if e.kind() == ErrorKind::InvalidArgument));
}

#[test]
fn entropy_consumption_erases_on_success() {
	let mut bytes = [0xa5; 32];
	let result = consume_bytes(&mut bytes, |value| {
		assert_eq!(value, &[0xa5; 32]);
		Ok(7)
	});
	assert_eq!(result.unwrap(), 7);
	assert_eq!(bytes, [0; 32]);
}

#[test]
fn entropy_consumption_erases_on_error() {
	let mut bytes = [0xa5; 64];
	let result: Result<()> = consume_bytes(&mut bytes, |_| Err(Error::internal("test failure")));
	assert!(result.is_err());
	assert_eq!(bytes, [0; 64]);
}

#[test]
fn entropy_consumption_erases_on_unwind() {
	let mut bytes = [0xa5; 64];
	let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
		let _: Result<()> = consume_bytes(&mut bytes, |_| panic!("test unwind"));
	}));
	assert!(result.is_err());
	assert_eq!(bytes, [0; 64]);
}

#[cfg(unix)]
#[test]
fn entropy_owned_staging_consumes_deterministic_test_source() {
	let entropy = Entropy::<64>::from_source(|bytes| {
		bytes.fill(0xa5);
		Ok(())
	})
	.unwrap();
	entropy
		.consume(|bytes| {
			assert_eq!(bytes, &[0xa5; 64]);
			Ok(())
		})
		.unwrap();
}

#[cfg(unix)]
#[test]
fn entropy_source_failure_is_returned_after_partial_write() {
	let result = Entropy::<32>::from_source(|bytes| {
		bytes[..13].fill(0xa5);
		Err(Error::internal("test source failure"))
	});
	assert!(matches!(result, Err(e) if e.message() == "test source failure"));
}

#[cfg(unix)]
#[test]
fn entropy_os_source_supports_both_exact_extents() {
	Entropy::<32>::generate()
		.unwrap()
		.consume(|bytes| {
			assert_eq!(bytes.len(), 32);
			Ok(())
		})
		.unwrap();
	Entropy::<64>::generate()
		.unwrap()
		.consume(|bytes| {
			assert_eq!(bytes.len(), 64);
			Ok(())
		})
		.unwrap();
}

// Deterministic entropy injection exists only in this cfg(test) module.
impl<const N: usize> Entropy<N> {
	pub(crate) fn test_from_bytes(bytes: &[u8; N]) -> Result<Self> {
		Self::from_source(|storage| {
			storage.copy_from_slice(bytes);
			Ok(())
		})
	}
}
