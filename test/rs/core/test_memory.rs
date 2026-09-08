use oa::{ErrorKind, core::memory};

#[test]
fn public_memory_contract_is_checked_and_exact() {
	let source = (0_u8..64).collect::<Vec<_>>();
	let mut destination = vec![0_u8; source.len()];
	memory::copy(&mut destination, &source).unwrap();
	assert_eq!(destination, source);
	assert!(memory::equal(&destination, &source));
	assert!(memory::equal_constant_time(&destination, &source));

	destination[31] ^= 1;
	assert!(!memory::equal(&destination, &source));
	assert!(!memory::equal_constant_time(&destination, &source));
	memory::zero_secure(&mut destination);
	assert!(destination.iter().all(|byte| *byte == 0));
}

#[test]
fn public_streaming_copy_rejects_length_mismatch() {
	let mut destination = [0xa5; 4];
	let error = memory::copy_streaming(&mut destination, &[1, 2, 3]).unwrap_err();
	assert_eq!(error.kind(), ErrorKind::InvalidArgument);
	assert_eq!(destination, [0xa5; 4]);
}
