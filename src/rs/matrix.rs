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
	let mut output = Matrix::allocate(
		engine,
		left.shape().to_vec(),
		left.element_count(),
		left.dtype(),
	)?;
	let event = if element_count == 0 {
		engine.checkpoint()?
	} else {
		let buffers = [
			BufferBinding::read(left.storage()),
			BufferBinding::read(right.storage()),
			BufferBinding::write(output.storage()),
		];
		let push_constants = [
			PushConstant::StorageBuffer(0),
			PushConstant::StorageBuffer(1),
			PushConstant::StorageBuffer(2),
			PushConstant::U32(element_count),
		];
		engine.submit(ComputeDispatch {
			operation,
			kernel,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: kernel.linear_workgroups(element_count),
		})?
	};
	output.mark_pending(event);
	Ok(output)
}

fn unary(input: &Matrix, routes: &[(DType, KernelId)], operation: &'static str) -> Result<Matrix> {
	let kernel = select_kernel(input.dtype(), routes, operation)?;
	let engine = input.engine_handle();
	let element_count = u32::try_from(input.element_count())
		.map_err(|_| Error::invalid_argument(format!("{operation} element count exceeds u32")))?;
	let mut output = Matrix::allocate(
		engine,
		input.shape().to_vec(),
		input.element_count(),
		input.dtype(),
	)?;
	let event = if element_count == 0 {
		engine.checkpoint()?
	} else {
		let buffers = [
			BufferBinding::read(input.storage()),
			BufferBinding::write(output.storage()),
		];
		let push_constants = [
			PushConstant::StorageBuffer(0),
			PushConstant::StorageBuffer(1),
			PushConstant::U32(element_count),
		];
		engine.submit(ComputeDispatch {
			operation,
			kernel,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: kernel.linear_workgroups(element_count),
		})?
	};
	output.mark_pending(event);
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
	let mut output = Matrix::allocate(
		engine,
		input.shape().to_vec(),
		input.element_count(),
		input.dtype(),
	)?;
	let event = if element_count == 0 {
		engine.checkpoint()?
	} else {
		let buffers = [
			BufferBinding::read(input.storage()),
			BufferBinding::write(output.storage()),
		];
		let push_constants = [
			PushConstant::StorageBuffer(0),
			PushConstant::StorageBuffer(1),
			PushConstant::U32(element_count),
			PushConstant::F32(scalar),
		];
		engine.submit(ComputeDispatch {
			operation,
			kernel,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: kernel.linear_workgroups(element_count),
		})?
	};
	output.mark_pending(event);
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
