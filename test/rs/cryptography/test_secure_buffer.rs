use oa::SecureBuffer;

#[test]
fn reset_erases_the_borrowed_range() {
	let mut bytes = [0xa5; 32];
	{
		let mut guarded = SecureBuffer::new(&mut bytes);
		assert!(guarded.is_valid());
		assert_eq!(guarded.size_bytes(), 32);
		guarded.reset();
		assert!(!guarded.is_valid());
	}
	assert_eq!(bytes, [0; 32]);
}

#[test]
fn drop_erases_the_borrowed_range() {
	let mut bytes = [0x5a; 17];
	{
		let _guarded = SecureBuffer::new(&mut bytes);
	}
	assert_eq!(bytes, [0; 17]);
}
