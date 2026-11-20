//! Host contracts for the F16 storage dtype and host value type.

use oa::core::Element;
use oa::{DType, F16};

#[test]
fn f16_dtype_reports_binary16_storage() {
	assert_eq!(DType::F16.size_bytes(), 2);
	assert_eq!(DType::F16.token(), "f16");
	assert_eq!(F16::DTYPE, DType::F16);
}

#[test]
fn f16_widens_every_exactly_representable_value() {
	for (value, bits) in [
		(0.0_f32, 0x0000_u16),
		(-0.0, 0x8000),
		(1.0, 0x3c00),
		(-2.5, 0xc100),
		(0.5, 0x3800),
		(65504.0, 0x7bff),
		(6.103_515_6e-5, 0x0400),
		(5.960_464_5e-8, 0x0001),
		(-5.960_464_5e-8, 0x8001),
		(f32::INFINITY, 0x7c00),
		(f32::NEG_INFINITY, 0xfc00),
	] {
		assert_eq!(F16::from_f32(value).to_bits(), bits, "from_f32({value})");
		assert_eq!(F16::from_bits(bits).to_f32(), value, "to_f32({bits:#06x})");
	}
}

#[test]
fn f16_round_trips_all_binary16_values_without_drift() {
	for bits in 0_u32..=0xffff {
		let value = F16::from_bits(bits as u16);
		let widened = value.to_f32();
		if widened.is_nan() {
			assert!(F16::from_f32(widened).to_f32().is_nan());
			continue;
		}
		assert_eq!(
			F16::from_f32(widened).to_bits(),
			bits as u16,
			"round trip drifted for bits {bits:#06x}"
		);
	}
}

#[test]
fn f16_narrows_with_round_to_nearest_even() {
	let midpoint =
		|low: u16, high: u16| (F16::from_bits(low).to_f32() + F16::from_bits(high).to_f32()) * 0.5;
	// 1.0 + 2^-11 is exactly halfway between 1.0 and its successor; the even
	// mantissa wins.
	assert_eq!(F16::from_f32(midpoint(0x3c00, 0x3c01)).to_bits(), 0x3c00);
	// 1.0013 sits just above the 0x3C00/0x3C01 midpoint and rounds up.
	assert_eq!(F16::from_f32(1.0013).to_bits(), 0x3c01);
	// The 0x3C01/0x3C02 midpoint ties to the even mantissa.
	assert_eq!(F16::from_f32(midpoint(0x3c01, 0x3c02)).to_bits(), 0x3c02);
	// A mantissa that rounds past 0x7ff carries into the next binade: 32760 is
	// the 32752/32768 midpoint and ties to the even mantissa.
	assert_eq!(F16::from_f32(32752.0).to_bits(), 0x77ff);
	assert_eq!(F16::from_f32(32760.0).to_bits(), 0x7800);
	assert_eq!(F16::from_f32(1023.99).to_bits(), 0x6400);
	// Magnitudes above the 65520 midpoint saturate to infinity.
	assert_eq!(F16::from_f32(65519.0).to_bits(), 0x7bff);
	assert_eq!(F16::from_f32(65520.0).to_bits(), 0x7c00);
	assert_eq!(F16::from_f32(65536.0).to_bits(), 0x7c00);
	assert_eq!(F16::from_f32(1.0e30).to_bits(), 0x7c00);
	assert_eq!(F16::from_f32(-1.0e30).to_bits(), 0xfc00);
	// Magnitudes below half the smallest subnormal become signed zero.
	assert_eq!(F16::from_f32(2.0e-8).to_bits(), 0x0000);
	assert_eq!(F16::from_f32(-2.0e-8).to_bits(), 0x8000);
}

#[test]
fn f16_nan_bits_survive_the_widen_narrow_cycle() {
	let nan = F16::from_f32(f32::NAN);
	assert!(nan.to_f32().is_nan());
	assert_eq!(nan.to_bits() & 0x7c00, 0x7c00);
	let payload = F16::from_bits(0x7e00);
	assert!(payload.to_f32().is_nan());
}
