//! Stateless Matrix operations.

use crate::{
	DType, Engine, Error, Matrix, Result,
	runtime::{BufferBinding, ComputeDispatch, KernelId, PushConstant},
};

/// Create an FP32 matrix filled with ones.
///
/// # Errors
///
/// Returns an error when the shape or storage size overflows, or allocation and
/// upload fail.
pub fn ones(engine: &Engine, shape: impl Into<Vec<usize>>) -> Result<Matrix> {
	Matrix::filled_f32(engine, shape, 1.0)
}

/// Create an FP32 matrix filled with `value`.
///
/// # Errors
///
/// Returns an error when the shape or storage size overflows, or allocation and
/// upload fail.
pub fn full(engine: &Engine, shape: impl Into<Vec<usize>>, value: f32) -> Result<Matrix> {
	Matrix::filled_f32(engine, shape, value)
}

fn binary(
	left: &Matrix,
	right: &Matrix,
	routes: &[(DType, KernelId)],
	operation: &'static str,
) -> Result<Matrix> {
	if left.shape() != right.shape() {
		return Err(Error::invalid_argument(format!(
			"{operation} requires equal shapes; left is {:?}, right is {:?}",
			left.shape(),
			right.shape()
		)));
	}
	if left.dtype() != right.dtype() {
		return Err(Error::invalid_argument(format!(
			"{operation} requires equal dtypes; left is {}, right is {}",
			left.dtype().token(),
			right.dtype().token()
		)));
	}
	let kernel = select_kernel(left.dtype(), routes, operation)?;
	let engine = left.engine_handle();
	if !engine.same_as(right.engine_handle()) {
		return Err(Error::invalid_argument(format!(
			"{operation} inputs must belong to the same engine"
		)));
	}
	let element_count = u32::try_from(left.element_count())
		.map_err(|_| Error::invalid_argument(format!("{operation} element count exceeds u32")))?;
	let output = Matrix::allocate(
		engine,
		left.shape().to_vec(),
		left.element_count(),
		left.dtype(),
	)?;
	if element_count != 0 {
		let buffers = [
			BufferBinding::read(left.storage()),
			BufferBinding::read(right.storage()),
			BufferBinding::write(output.storage()),
		];
		let push_constants = [PushConstant::U32(element_count)];
		engine.record(ComputeDispatch {
			operation,
			kernel,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: kernel.linear_workgroups(element_count),
		})?;
	}
	Ok(output)
}

fn unary(input: &Matrix, routes: &[(DType, KernelId)], operation: &'static str) -> Result<Matrix> {
	let kernel = select_kernel(input.dtype(), routes, operation)?;
	let engine = input.engine_handle();
	let element_count = u32::try_from(input.element_count())
		.map_err(|_| Error::invalid_argument(format!("{operation} element count exceeds u32")))?;
	let output = Matrix::allocate(
		engine,
		input.shape().to_vec(),
		input.element_count(),
		input.dtype(),
	)?;
	if element_count != 0 {
		let buffers = [
			BufferBinding::read(input.storage()),
			BufferBinding::write(output.storage()),
		];
		let push_constants = [PushConstant::U32(element_count)];
		engine.record(ComputeDispatch {
			operation,
			kernel,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: kernel.linear_workgroups(element_count),
		})?;
	}
	Ok(output)
}

fn unary_scalar(
	input: &Matrix,
	scalar: f32,
	routes: &[(DType, KernelId)],
	operation: &'static str,
) -> Result<Matrix> {
	let kernel = select_kernel(input.dtype(), routes, operation)?;
	let engine = input.engine_handle();
	let element_count = u32::try_from(input.element_count())
		.map_err(|_| Error::invalid_argument(format!("{operation} element count exceeds u32")))?;
	let output = Matrix::allocate(
		engine,
		input.shape().to_vec(),
		input.element_count(),
		input.dtype(),
	)?;
	if element_count != 0 {
		let buffers = [
			BufferBinding::read(input.storage()),
			BufferBinding::write(output.storage()),
		];
		let push_constants = [PushConstant::U32(element_count), PushConstant::F32(scalar)];
		engine.record(ComputeDispatch {
			operation,
			kernel,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: kernel.linear_workgroups(element_count),
		})?;
	}
	Ok(output)
}

fn mat_mul_nt_impl(left: &Matrix, right: &Matrix) -> Result<Matrix> {
	const OPERATION: &str = "matrix.mat_mul_nt";
	let [m, k] = left.shape() else {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires rank-two inputs; left is {:?}",
			left.shape()
		)));
	};
	let [n, right_k] = right.shape() else {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires rank-two inputs; right is {:?}",
			right.shape()
		)));
	};
	if k != right_k {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires equal K extents; left is {:?}, right is {:?}",
			left.shape(),
			right.shape()
		)));
	}
	if left.dtype() != DType::F32 || right.dtype() != DType::F32 {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires two F32 matrices; left is {}, right is {}",
			left.dtype().token(),
			right.dtype().token()
		)));
	}
	let engine = left.engine_handle();
	if !engine.same_as(right.engine_handle()) {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} inputs must belong to the same engine"
		)));
	}

	let output_count = m.checked_mul(*n).ok_or_else(|| {
		Error::invalid_argument(format!("{OPERATION} output size overflows usize"))
	})?;
	let m = u32::try_from(*m)
		.map_err(|_| Error::invalid_argument(format!("{OPERATION} M extent exceeds u32")))?;
	let n = u32::try_from(*n)
		.map_err(|_| Error::invalid_argument(format!("{OPERATION} N extent exceeds u32")))?;
	let k = u32::try_from(*k)
		.map_err(|_| Error::invalid_argument(format!("{OPERATION} K extent exceeds u32")))?;
	for (label, count) in [
		("left", left.element_count()),
		("right", right.element_count()),
		("output", output_count),
	] {
		u32::try_from(count).map_err(|_| {
			Error::invalid_argument(format!("{OPERATION} {label} element count exceeds u32"))
		})?;
	}

	let output = Matrix::allocate(
		engine,
		vec![m as usize, n as usize],
		output_count,
		DType::F32,
	)?;
	let kernel = KernelId::MatrixMatMulNtTiledF32;
	if m != 0 && n != 0 && k != 0 {
		let buffers = [
			BufferBinding::read(left.storage()),
			BufferBinding::read(right.storage()),
			BufferBinding::write(output.storage()),
		];
		let push_constants = [
			PushConstant::U32(m),
			PushConstant::U32(n),
			PushConstant::U32(k),
		];
		engine.record(ComputeDispatch {
			operation: OPERATION,
			kernel,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: kernel.output_workgroups(m, n),
		})?;
	}
	Ok(output)
}

fn select_kernel(
	dtype: DType,
	routes: &[(DType, KernelId)],
	operation: &'static str,
) -> Result<KernelId> {
	routes
		.iter()
		.find_map(|(candidate, kernel)| (*candidate == dtype).then_some(*kernel))
		.ok_or_else(|| {
			Error::invalid_argument(format!(
				"{operation} does not support dtype {}",
				dtype.token()
			))
		})
}

include!("matrix/elemwise.gen.rs");
include!("matrix/blas.gen.rs");
