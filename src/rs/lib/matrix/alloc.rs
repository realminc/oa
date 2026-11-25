//! Matrix source constructors and explicit host uploads.

use crate::{Matrix, Result, Runtime};

/// Create an FP32 matrix on this thread's default runtime.
///
/// # Errors
///
/// Returns an error when the default runtime cannot be initialized, the shape
/// does not match `values`, or allocation and upload fail.
pub fn from_f32(shape: impl Into<Vec<usize>>, values: &[f32]) -> Result<Matrix> {
	let shape = shape.into();
	from_f32_on(&Runtime::current()?, shape, values)
}
/// Create an FP32 matrix on a particular runtime.
///
/// # Errors
///
/// Returns an error when the shape does not match `values`, or allocation and
/// upload fail.
pub fn from_f32_on(
	runtime: &Runtime,
	shape: impl Into<Vec<usize>>,
	values: &[f32],
) -> Result<Matrix> {
	Matrix::new_from_f32_on(runtime, shape, values)
}

/// Create an FP32 matrix filled with ones on this thread's default runtime.
///
/// # Errors
///
/// Returns an error when the default runtime cannot be initialized, the shape or
/// storage size overflows, or allocation and upload fail.
pub fn ones(shape: impl Into<Vec<usize>>) -> Result<Matrix> {
	let shape = shape.into();
	ones_on(&Runtime::current()?, shape)
}
/// Create an FP32 matrix filled with ones on a particular runtime.
///
/// # Errors
///
/// Returns an error when the shape or storage size overflows, or allocation and
/// upload fail.
pub fn ones_on(runtime: &Runtime, shape: impl Into<Vec<usize>>) -> Result<Matrix> {
	Matrix::filled_f32(runtime, shape, 1.0)
}

/// Create an FP32 matrix filled with `value` on this thread's default runtime.
///
/// # Errors
///
/// Returns an error when the default runtime cannot be initialized, the shape or
/// storage size overflows, or allocation and upload fail.
pub fn full(shape: impl Into<Vec<usize>>, value: f32) -> Result<Matrix> {
	let shape = shape.into();
	full_on(&Runtime::current()?, shape, value)
}
/// Create an FP32 matrix filled with `value` on a particular runtime.
///
/// # Errors
///
/// Returns an error when the shape or storage size overflows, or allocation and
/// upload fail.
pub fn full_on(runtime: &Runtime, shape: impl Into<Vec<usize>>, value: f32) -> Result<Matrix> {
	Matrix::filled_f32(runtime, shape, value)
}
