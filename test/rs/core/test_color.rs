use oa::Color;

#[test]
fn donor_pack_rounding_and_unpack_contract() {
	assert_eq!(Color::new(1.0, 0.5, -1.0, 2.0).to_u32(), 0xff8000ff);
	assert_eq!(
		Color::from_u32(0xff8000ff),
		Color::new(1.0, 128.0 / 255.0, 0.0, 1.0)
	);
}

#[test]
fn donor_arithmetic_and_lerp_contract() {
	let a = Color::new(0.0, 0.2, 0.4, 0.6);
	let b = Color::new(1.0, 0.8, 0.6, 0.4);
	assert_eq!(a.lerp(b, 0.5), Color::new(0.5, 0.5, 0.5, 0.5));
	assert_eq!((a + b).clamp(), Color::new(1.0, 1.0, 1.0, 1.0));
}
