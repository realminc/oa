/// Scalar data representation of an OA value.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
	/// 8-bit unsigned integer, including byte-oriented media and cryptography.
	U8,
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
			Self::F32 => "f32",
			Self::I32 => "i32",
			Self::U32 => "u32",
		}
	}

	/// Return the number of storage bytes occupied by one dense element.
	pub const fn size_bytes(self) -> usize {
		match self {
			Self::U8 => 1,
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

impl Element for f32 {
	const DTYPE: DType = DType::F32;
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
	impl Sealed for u8 {}
	impl Sealed for i32 {}
	impl Sealed for u32 {}
}
