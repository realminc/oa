/// Scalar data representation of an OA value.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
	/// 8-bit unsigned integer, including byte-oriented media and cryptography.
	U8,
	/// IEEE 754 binary16 half-precision floating point. Storage only:
	/// arithmetic widens to F32 (the FP16-storage/FP32-accumulate mode).
	F16,
	/// IEEE 754 single-precision floating point.
	F32,
	/// 32-bit signed integer.
	I32,
	/// 32-bit unsigned integer.
	U32,
}

impl DType {
	/// Return the normalized schema and shader token for this dtype.
	pub const fn token(self) -> &'static str {
		match self {
			Self::U8 => "u8",
			Self::F16 => "f16",
			Self::F32 => "f32",
			Self::I32 => "i32",
			Self::U32 => "u32",
		}
	}

	/// Return the number of storage bytes occupied by one dense element.
	pub const fn size_bytes(self) -> usize {
		match self {
			Self::U8 => 1,
			Self::F16 => 2,
			Self::F32 | Self::I32 | Self::U32 => 4,
		}
	}
}

/// Rust host type that has one exact admitted dense [`DType`] representation.
///
/// This trait is sealed so OA can uphold the no-padding and bit-validity
/// requirements used by checked upload and readback.
pub trait Element: private::Sealed + Copy + Default + 'static {
	/// Dense OA dtype represented by this Rust type.
	const DTYPE: DType;
}

/// IEEE 754 binary16 (half-precision) storage value for [`DType::F16`].
///
/// `F16` is a dense storage representation, not a host arithmetic type:
/// conversions are exact in the widening direction and round-to-nearest-even
/// when narrowing, matching the device FP16-storage contract. Host-side
/// arithmetic goes through [`f32`].
///
/// `PartialEq` compares stored bits, not values; distinct NaN payloads do not
/// compare equal, which is the intended dense-storage equality.
#[repr(transparent)]
#[derive(Clone, Copy, Default)]
pub struct F16(u16);

impl F16 {
	/// Positive zero.
	pub const ZERO: Self = Self(0);

	/// Wrap raw IEEE 754 binary16 bits.
	pub const fn from_bits(bits: u16) -> Self {
		Self(bits)
	}

	/// Return the raw IEEE 754 binary16 bits.
	pub const fn to_bits(self) -> u16 {
		self.0
	}

	/// Widen to [`f32`]. Exact for every finite value; subnormals normalize,
	/// and NaN payloads are preserved with the quiet bit set.
	pub fn to_f32(self) -> f32 {
		let bits = u32::from(self.0);
		let sign = (bits & 0x8000) << 16;
		let exponent = (bits >> 10) & 0x1f;
		let mantissa = bits & 0x3ff;
		let widened = if exponent == 0 {
			if mantissa == 0 {
				sign
			} else {
				let mut e = -14_i32;
				let mut m = mantissa;
				while m & 0x400 == 0 {
					m <<= 1;
					e -= 1;
				}
				m &= 0x3ff;
				sign | (((e + 127) as u32) << 23) | (m << 13)
			}
		} else if exponent == 0x1f {
			sign | 0x7f80_0000 | (mantissa << 13)
		} else {
			sign | ((exponent + 112) << 23) | (mantissa << 13)
		};
		f32::from_bits(widened)
	}

	/// Narrow an [`f32`] with round-to-nearest-even. Magnitudes above the
	/// binary16 range become signed infinity; magnitudes below the smallest
	/// representable subnormal become signed zero; NaN payloads are
	/// preserved with the quiet bit set.
	pub fn from_f32(value: f32) -> Self {
		let bits = value.to_bits();
		let sign = ((bits >> 16) & 0x8000) as u16;
		let exponent = ((bits >> 23) & 0xff) as i32;
		let mantissa = bits & 0x007f_ffff;
		if exponent == 0xff {
			if mantissa == 0 {
				return Self(sign | 0x7c00);
			}
			let payload = (mantissa >> 13) as u16;
			return Self(sign | 0x7c00 | payload.max(1));
		}
		let mut half_exponent = exponent - 112;
		if half_exponent >= 31 {
			return Self(sign | 0x7c00);
		}
		if half_exponent < -10 {
			return Self(sign);
		}
		if half_exponent > 0 {
			let mantissa24 = mantissa | 0x0080_0000;
			let mut mantissa10 = mantissa24 >> 13;
			let remainder = mantissa24 & 0x1fff;
			if remainder > 0x1000 || (remainder == 0x1000 && mantissa10 & 1 == 1) {
				mantissa10 += 1;
			}
			// mantissa10 still carries the implicit leading bit; when rounding
			// pushes it to 0x800 the value carries into the next binade and the
			// top binade saturates to infinity.
			if mantissa10 == 0x800 {
				half_exponent += 1;
				if half_exponent >= 31 {
					return Self(sign | 0x7c00);
				}
			}
			return Self(sign | ((half_exponent as u16) << 10 | (mantissa10 as u16 & 0x3ff)));
		}
		let mantissa24 = mantissa | 0x0080_0000;
		let shift = (14 - half_exponent) as u32;
		let mut mantissa10 = mantissa24 >> shift;
		let remainder = mantissa24 & ((1 << shift) - 1);
		let halfway = 1_u32 << (shift - 1);
		if remainder > halfway || (remainder == halfway && mantissa10 & 1 == 1) {
			mantissa10 += 1;
		}
		Self(sign | mantissa10 as u16)
	}
}

impl From<f32> for F16 {
	fn from(value: f32) -> Self {
		Self::from_f32(value)
	}
}

impl From<F16> for f32 {
	fn from(value: F16) -> Self {
		value.to_f32()
	}
}

impl PartialEq for F16 {
	fn eq(&self, other: &Self) -> bool {
		self.0 == other.0
	}
}

impl core::fmt::Debug for F16 {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		write!(f, "{}", self.to_f32())
	}
}

impl Element for f32 {
	const DTYPE: DType = DType::F32;
}

impl Element for F16 {
	const DTYPE: DType = DType::F16;
}

impl Element for u8 {
	const DTYPE: DType = DType::U8;
}

impl Element for i32 {
	const DTYPE: DType = DType::I32;
}

impl Element for u32 {
	const DTYPE: DType = DType::U32;
}

mod private {
	pub trait Sealed {}

	impl Sealed for f32 {}
	impl Sealed for super::F16 {}
	impl Sealed for u8 {}
	impl Sealed for i32 {}
	impl Sealed for u32 {}
}
