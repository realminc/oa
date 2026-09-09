/// Scalar data representation of an OA value.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
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
			Self::F32 => "f32",
			Self::I32 => "i32",
			Self::U32 => "u32",
		}
	}

	/// Return the number of storage bytes occupied by one dense element.
	pub const fn size_bytes(self) -> usize {
		match self {
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

impl Element for i32 {
	const DTYPE: DType = DType::I32;
}

impl Element for u32 {
	const DTYPE: DType = DType::U32;
}

mod private {
	pub trait Sealed {}

	impl Sealed for f32 {}
	impl Sealed for i32 {}
	impl Sealed for u32 {}
}

#[cfg(test)]
mod tests {
	use super::{DType, Element};

	#[test]
	fn admitted_host_elements_match_dense_storage_widths() {
		assert_eq!(<f32 as Element>::DTYPE, DType::F32);
		assert_eq!(<i32 as Element>::DTYPE, DType::I32);
		assert_eq!(<u32 as Element>::DTYPE, DType::U32);
		assert_eq!(size_of::<f32>(), DType::F32.size_bytes());
		assert_eq!(size_of::<i32>(), DType::I32.size_bytes());
		assert_eq!(size_of::<u32>(), DType::U32.size_bytes());
	}
}
