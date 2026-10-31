use oa::cryptography::ntt::{NttConfig, NttRepresentation};

#[test]
fn rejects_invalid_field_root_and_length() {
	for (prime, root, length) in [
		(15, 2, 4),
		(17, 1, 4),
		(17, 3, 4),
		(17, 4, 3),
		(1_u64 << 62, 4, 4),
		(17, 4, 0),
	] {
		assert!(NttConfig::new(prime, root, length).is_err());
	}
	let config = NttConfig::new(17, 4, 4)
		.unwrap()
		.with_representation(NttRepresentation::Montgomery);
	assert_eq!(config.modulus(), 17);
	assert_eq!(config.root(), 4);
	assert_eq!(config.length(), 4);
	assert_eq!(config.representation(), NttRepresentation::Montgomery);
}
