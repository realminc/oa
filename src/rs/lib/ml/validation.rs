//! Shared checked Matrix ownership, dtype and shader-width validation.
//! No allocation, kernel selection, recording or execution lives here.

use crate::{DType, Error, Matrix, Result};

pub(in crate::ml) fn validate_f32_same_engine(
	operation: &'static str,
	matrices: &[&Matrix],
) -> Result<()> {
	let Some(first) = matrices.first() else {
		return Err(Error::invalid_argument(
			"matrix validation requires an input",
		));
	};
	for matrix in matrices {
		if matrix.dtype() != DType::F32 {
			return Err(Error::invalid_argument(format!(
				"{operation} requires F32 matrices; found {}",
				matrix.dtype().token()
			)));
		}
		if !first.engine_handle().same_as(matrix.engine_handle()) {
			return Err(Error::invalid_argument(format!(
				"{operation} inputs must belong to the same engine"
			)));
		}
	}
	Ok(())
}

pub(in crate::ml) fn shader_u32(value: usize, label: &str, operation: &'static str) -> Result<u32> {
	u32::try_from(value)
		.map_err(|_| Error::invalid_argument(format!("{operation} {label} exceeds u32")))
}
