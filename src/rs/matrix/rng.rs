use std::{cell::Cell, time::SystemTime};

use crate::{
	DType, Error, Matrix, OpAttribute, OperationContract, Result,
	core::autograd::{self, MatrixNode},
	runtime::{BufferBinding, ComputeDispatch, KernelId, PushConstant, SemanticDispatch},
};

thread_local! {
	static RNG_STATE: Cell<u64> = Cell::new(initial_seed());
}

/// Replace the current thread's implicit random seed source.
///
/// Explicit nonzero operation seeds bypass this state. Passing zero to a Philox
/// operation consumes one value from this deterministic thread-local sequence.
pub fn set_rng_seed(seed: u64) {
	RNG_STATE.set(seed);
}

/// Generate uniformly distributed FP32 values with `input`'s shape.
///
/// `input` supplies shape, dtype, engine, and semantic lineage; its values are
/// not read. A nonzero `seed` is exact. Zero consumes the thread-local seed
/// sequence controlled by [`set_rng_seed`].
///
/// # Errors
///
/// Returns an error unless `input` is FP32, `low < high`, both bounds are
/// finite, or runtime allocation/recording fails.
pub fn philox_uniform(input: &Matrix, low: f32, high: f32, seed: u64) -> Result<Matrix> {
	if input.dtype() != DType::F32 {
		return Err(Error::invalid_argument(
			"oa::matrix::philox_uniform requires an F32 input",
		));
	}
	if !low.is_finite() || !high.is_finite() || low >= high {
		return Err(Error::invalid_argument(
			"oa::matrix::philox_uniform requires finite low < high",
		));
	}
	let seed = resolve_seed(seed);
	record_rng(
		input,
		crate::core::operation::matrix::PHILOX_UNIFORM,
		KernelId::MatrixPhiloxUniformF32,
		KernelId::MatrixPhiloxUniformReplayF32,
		seed,
		low,
		high,
	)
}

/// Generate normally distributed FP32 values with `input`'s shape.
///
/// # Errors
///
/// Returns an error unless `input` is FP32, `mean` and `stddev` are finite,
/// `stddev` is nonnegative, or runtime allocation/recording fails.
pub fn philox_normal(input: &Matrix, mean: f32, stddev: f32, seed: u64) -> Result<Matrix> {
	if input.dtype() != DType::F32 {
		return Err(Error::invalid_argument(
			"oa::matrix::philox_normal requires an F32 input",
		));
	}
	if !mean.is_finite() || !stddev.is_finite() || stddev < 0.0 {
		return Err(Error::invalid_argument(
			"oa::matrix::philox_normal requires a finite mean and nonnegative finite stddev",
		));
	}
	let seed = resolve_seed(seed);
	record_rng(
		input,
		crate::core::operation::matrix::PHILOX_NORMAL,
		KernelId::MatrixPhiloxNormalF32,
		KernelId::MatrixPhiloxNormalReplayF32,
		seed,
		mean,
		stddev,
	)
}

/// Apply inverted Dropout with a Philox-generated mask.
///
/// Kept values are scaled by `1 / (1 - probability)`. The effective seed is
/// retained for the reverse pass, and captured execution advances its random
/// counter once per replay without rebuilding the graph.
///
/// # Errors
///
/// Returns an error unless `input` is FP32, `probability` is finite and in
/// `[0, 1)`, or runtime allocation/recording fails.
pub fn dropout(input: &Matrix, probability: f32, seed: u64) -> Result<Matrix> {
	if input.dtype() != DType::F32 {
		return Err(Error::invalid_argument(
			"oa::matrix::dropout requires an F32 input",
		));
	}
	if !probability.is_finite() || !(0.0..1.0).contains(&probability) {
		return Err(Error::invalid_argument(
			"oa::matrix::dropout probability must be finite and in [0, 1)",
		));
	}
	if probability == 0.0 {
		return Ok(input.clone());
	}
	let seed = resolve_seed(seed);
	let output = record_dropout(
		input,
		probability,
		seed,
		crate::core::operation::matrix::DROPOUT,
		KernelId::MatrixDropoutF32,
		KernelId::MatrixDropoutReplayF32,
	)?;
	autograd::record(MatrixNode::Dropout {
		input: input.clone(),
		output_id: output.value_id(),
		probability,
		seed,
	})?;
	Ok(output)
}

pub(crate) fn dropout_backward(gradient: &Matrix, probability: f32, seed: u64) -> Result<Matrix> {
	record_dropout(
		gradient,
		probability,
		seed,
		crate::core::operation::matrix::DROPOUT_BACKWARD,
		KernelId::MatrixDropoutBackwardF32,
		KernelId::MatrixDropoutBackwardReplayF32,
	)
}

fn record_rng(
	input: &Matrix,
	contract: OperationContract,
	eager_kernel: KernelId,
	replay_kernel: KernelId,
	seed: u64,
	first: f32,
	second: f32,
) -> Result<Matrix> {
	let engine = input.engine_handle();
	let element_count = u32::try_from(input.num_elements()).map_err(|_| {
		Error::invalid_argument(format!("{} element count exceeds u32", contract.name()))
	})?;
	let output = Matrix::allocate(
		engine,
		input.shape().to_vec(),
		input.num_elements(),
		DType::F32,
	)?;
	if element_count == 0 {
		return Ok(output);
	}
	let payload = [
		PushConstant::U32(element_count),
		PushConstant::U32(seed as u32),
		PushConstant::U32((seed >> 32) as u32),
		PushConstant::U32(0),
		PushConstant::F32(first),
		PushConstant::F32(second),
	];
	let attributes = [
		OpAttribute::Float {
			name: contract.attribute_specs()[0].name().into(),
			value: f64::from(first),
		},
		OpAttribute::Float {
			name: contract.attribute_specs()[1].name().into(),
			value: f64::from(second),
		},
		OpAttribute::UnsignedInteger {
			name: "seed".into(),
			value: seed,
		},
	];
	let inputs = [input];
	let outputs = [&output];
	if engine.capture_active() {
		let state = Matrix::from_slice_handle(engine, vec![1], &[0_u32])?;
		let buffers = [
			BufferBinding::read(input.storage()),
			BufferBinding::write(output.storage()),
			BufferBinding::read(state.storage()),
		];
		engine.record_semantic(
			ComputeDispatch {
				kernel: replay_kernel,
				buffers: &buffers,
				push_constants: &payload,
				workgroups: replay_kernel.linear_workgroups(element_count),
			},
			SemanticDispatch {
				contract,
				inputs: &inputs,
				outputs: &outputs,
				attributes: &attributes,
			},
		)?;
		advance_replay_state(&state)?;
	} else {
		let buffers = [
			BufferBinding::read(input.storage()),
			BufferBinding::write(output.storage()),
		];
		engine.record_semantic(
			ComputeDispatch {
				kernel: eager_kernel,
				buffers: &buffers,
				push_constants: &payload,
				workgroups: eager_kernel.linear_workgroups(element_count),
			},
			SemanticDispatch {
				contract,
				inputs: &inputs,
				outputs: &outputs,
				attributes: &attributes,
			},
		)?;
	}
	Ok(output)
}

fn advance_replay_state(state: &Matrix) -> Result<()> {
	let buffers = [BufferBinding::read_write(state.storage())];
	let inputs = [state];
	let outputs = [state];
	state.engine_handle().record_semantic(
		ComputeDispatch {
			kernel: KernelId::MatrixPhiloxReplayAdvanceU32,
			buffers: &buffers,
			push_constants: &[],
			workgroups: [1, 1, 1],
		},
		SemanticDispatch {
			contract: crate::core::operation::matrix::PHILOX_REPLAY_ADVANCE,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &[],
		},
	)
}

fn record_dropout(
	input: &Matrix,
	probability: f32,
	seed: u64,
	contract: OperationContract,
	eager_kernel: KernelId,
	replay_kernel: KernelId,
) -> Result<Matrix> {
	let engine = input.engine_handle();
	let element_count = u32::try_from(input.num_elements()).map_err(|_| {
		Error::invalid_argument(format!("{} element count exceeds u32", contract.name()))
	})?;
	let output = Matrix::allocate(
		engine,
		input.shape().to_vec(),
		input.num_elements(),
		DType::F32,
	)?;
	if element_count == 0 {
		return Ok(output);
	}
	let payload = [
		PushConstant::U32(element_count),
		PushConstant::U32(seed as u32),
		PushConstant::U32((seed >> 32) as u32),
		PushConstant::U32(0),
		PushConstant::F32(probability),
	];
	let attributes = [
		OpAttribute::Float {
			name: "probability".into(),
			value: f64::from(probability),
		},
		OpAttribute::UnsignedInteger {
			name: "seed".into(),
			value: seed,
		},
	];
	let inputs = [input];
	let outputs = [&output];
	if engine.capture_active() {
		let state = Matrix::from_slice_handle(engine, vec![1], &[0_u32])?;
		let buffers = [
			BufferBinding::read(input.storage()),
			BufferBinding::write(output.storage()),
			BufferBinding::read(state.storage()),
		];
		engine.record_semantic(
			ComputeDispatch {
				kernel: replay_kernel,
				buffers: &buffers,
				push_constants: &payload,
				workgroups: replay_kernel.linear_workgroups(element_count),
			},
			SemanticDispatch {
				contract,
				inputs: &inputs,
				outputs: &outputs,
				attributes: &attributes,
			},
		)?;
		advance_replay_state(&state)?;
	} else {
		let buffers = [
			BufferBinding::read(input.storage()),
			BufferBinding::write(output.storage()),
		];
		engine.record_semantic(
			ComputeDispatch {
				kernel: eager_kernel,
				buffers: &buffers,
				push_constants: &payload,
				workgroups: eager_kernel.linear_workgroups(element_count),
			},
			SemanticDispatch {
				contract,
				inputs: &inputs,
				outputs: &outputs,
				attributes: &attributes,
			},
		)?;
	}
	Ok(output)
}

fn resolve_seed(seed: u64) -> u64 {
	if seed != 0 {
		return seed;
	}
	let state = RNG_STATE.get().wrapping_add(0x9e37_79b9_7f4a_7c15);
	RNG_STATE.set(state);
	let mut value = state;
	value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
	value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
	value ^ (value >> 31)
}

fn initial_seed() -> u64 {
	SystemTime::UNIX_EPOCH
		.elapsed()
		.map_or(0x243f_6a88_85a3_08d3, |duration| duration.as_nanos() as u64)
}
