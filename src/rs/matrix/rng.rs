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

/// Sample one I32 class index from each final-axis row of FP32 logits.
///
/// Nonpositive temperature selects the first maximum deterministically.
/// Positive temperature uses the donor Philox categorical route; positive
/// `top_k` restricts candidates and `top_p` applies nucleus truncation after
/// descending TopK. Rank-one input is treated as one row.
///
/// # Errors
///
/// Returns an error when logits are not nonempty rank-one or rank-two FP32,
/// a sorted candidate count exceeds 1024, sizes do not fit the shader ABI,
/// allocation fails, or runtime recording fails.
pub fn sample_logits(
	logits: &Matrix,
	temperature: f32,
	top_k: i32,
	top_p: f32,
	seed: u64,
) -> Result<Matrix> {
	const CONTRACT: OperationContract = crate::core::operation::matrix::SAMPLE_LOGITS;
	let operation = CONTRACT.name();
	if logits.dtype() != DType::F32 || !(1..=2).contains(&logits.shape().len()) {
		return Err(Error::invalid_argument(format!(
			"{operation} requires rank-one or rank-two F32 logits"
		)));
	}
	let vocabulary = *logits.shape().last().expect("validated logits rank");
	let rows = if logits.shape().len() == 1 {
		1
	} else {
		logits.shape()[0]
	};
	if rows == 0 || vocabulary == 0 {
		return Err(Error::invalid_argument(format!(
			"{operation} requires positive row and vocabulary extents"
		)));
	}
	let rows_u32 = u32::try_from(rows)
		.map_err(|_| Error::invalid_argument(format!("{operation} row count exceeds u32")))?;
	let vocabulary_u32 = u32::try_from(vocabulary).map_err(|_| {
		Error::invalid_argument(format!("{operation} vocabulary extent exceeds u32"))
	})?;
	let candidates = if top_k > 0 {
		usize::try_from(top_k)
			.expect("positive i32 fits usize")
			.min(vocabulary)
	} else {
		vocabulary
	};
	let normalized_top_p = top_p.clamp(1.0e-7, 1.0);
	let sorted_route = top_k > 0 || normalized_top_p < 1.0;
	let effective_seed = if temperature > 0.0 {
		resolve_seed(seed)
	} else {
		seed
	};
	let output = Matrix::allocate(logits.engine_handle(), vec![rows], rows, DType::I32)?;
	let attributes = [
		OpAttribute::Float {
			name: "temperature".into(),
			value: f64::from(temperature),
		},
		OpAttribute::SignedInteger {
			name: "top_k".into(),
			value: i64::from(top_k),
		},
		OpAttribute::Float {
			name: "top_p".into(),
			value: f64::from(normalized_top_p),
		},
		OpAttribute::UnsignedInteger {
			name: "seed".into(),
			value: effective_seed,
		},
	];
	let inputs = [logits];
	let outputs = [&output];
	let semantic = SemanticDispatch {
		contract: CONTRACT,
		inputs: &inputs,
		outputs: &outputs,
		attributes: &attributes,
	};
	if temperature <= 0.0 {
		let buffers = [
			BufferBinding::read(logits.storage()),
			BufferBinding::write(output.storage()),
		];
		let push_constants = [
			PushConstant::U32(rows_u32),
			PushConstant::U32(vocabulary_u32),
		];
		logits.engine_handle().record_semantic(
			ComputeDispatch {
				kernel: KernelId::MatrixSampleLogitsGreedyF32,
				buffers: &buffers,
				push_constants: &push_constants,
				workgroups: KernelId::MatrixSampleLogitsGreedyF32.linear_workgroups(rows_u32),
			},
			semantic,
		)?;
		return Ok(output);
	}

	if sorted_route && candidates > 1024 {
		return Err(Error::invalid_argument(format!(
			"{operation} top-k/top-p candidate count exceeds 1024"
		)));
	}
	let random = Matrix::allocate(logits.engine_handle(), vec![rows], rows, DType::F32)?;
	let random_payload = [
		PushConstant::U32(rows_u32),
		PushConstant::U32(effective_seed as u32),
		PushConstant::U32((effective_seed >> 32) as u32),
		PushConstant::U32(0),
		PushConstant::F32(0.0),
		PushConstant::F32(1.0),
	];
	if !sorted_route {
		record_dense_logits_sample(
			logits,
			&random,
			&output,
			temperature,
			rows_u32,
			vocabulary_u32,
			&random_payload,
			semantic,
		)?;
	} else {
		record_sorted_logits_sample(
			logits,
			&random,
			&output,
			temperature,
			normalized_top_p,
			rows_u32,
			vocabulary_u32,
			candidates,
			&random_payload,
			semantic,
		)?;
	}
	Ok(output)
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

#[allow(
	clippy::too_many_arguments,
	reason = "preserves the donor dense categorical sampling ABI explicitly"
)]
fn record_dense_logits_sample(
	logits: &Matrix,
	random: &Matrix,
	output: &Matrix,
	temperature: f32,
	rows: u32,
	vocabulary: u32,
	random_payload: &[PushConstant],
	semantic: SemanticDispatch<'_>,
) -> Result<()> {
	let random_buffers_eager = [
		BufferBinding::read(logits.storage()),
		BufferBinding::write(random.storage()),
	];
	let sample_buffers = [
		BufferBinding::read(logits.storage()),
		BufferBinding::read(random.storage()),
		BufferBinding::write(output.storage()),
	];
	let sample_payload = [
		PushConstant::U32(rows),
		PushConstant::U32(vocabulary),
		PushConstant::F32(temperature),
	];
	let engine = logits.engine_handle();
	if engine.capture_active() {
		let state = Matrix::from_slice_handle(engine, vec![1], &[0_u32])?;
		let random_buffers = [
			BufferBinding::read(logits.storage()),
			BufferBinding::write(random.storage()),
			BufferBinding::read(state.storage()),
		];
		let advance_buffers = [BufferBinding::read_write(state.storage())];
		let dispatches = [
			ComputeDispatch {
				kernel: KernelId::MatrixPhiloxUniformReplayF32,
				buffers: &random_buffers,
				push_constants: random_payload,
				workgroups: KernelId::MatrixPhiloxUniformReplayF32.linear_workgroups(rows),
			},
			ComputeDispatch {
				kernel: KernelId::MatrixPhiloxReplayAdvanceU32,
				buffers: &advance_buffers,
				push_constants: &[],
				workgroups: [1, 1, 1],
			},
			ComputeDispatch {
				kernel: KernelId::MatrixSampleLogitsDenseF32,
				buffers: &sample_buffers,
				push_constants: &sample_payload,
				workgroups: KernelId::MatrixSampleLogitsDenseF32.linear_workgroups(rows),
			},
		];
		engine.record_split_semantic(&dispatches, semantic)
	} else {
		let dispatches = [
			ComputeDispatch {
				kernel: KernelId::MatrixPhiloxUniformF32,
				buffers: &random_buffers_eager,
				push_constants: random_payload,
				workgroups: KernelId::MatrixPhiloxUniformF32.linear_workgroups(rows),
			},
			ComputeDispatch {
				kernel: KernelId::MatrixSampleLogitsDenseF32,
				buffers: &sample_buffers,
				push_constants: &sample_payload,
				workgroups: KernelId::MatrixSampleLogitsDenseF32.linear_workgroups(rows),
			},
		];
		engine.record_split_semantic(&dispatches, semantic)
	}
}

#[allow(
	clippy::too_many_arguments,
	reason = "preserves the donor TopK plus nucleus sampling ABI explicitly"
)]
fn record_sorted_logits_sample(
	logits: &Matrix,
	random: &Matrix,
	output: &Matrix,
	temperature: f32,
	top_p: f32,
	rows: u32,
	vocabulary: u32,
	candidates: usize,
	random_payload: &[PushConstant],
	semantic: SemanticDispatch<'_>,
) -> Result<()> {
	let operation = semantic.contract.name();
	let candidates_u32 = u32::try_from(candidates)
		.map_err(|_| Error::invalid_argument(format!("{operation} candidate count exceeds u32")))?;
	let candidate_count = usize::try_from(rows)
		.expect("u32 fits usize")
		.checked_mul(candidates)
		.ok_or_else(|| Error::invalid_argument(format!("{operation} candidate size overflows")))?;
	let values = Matrix::allocate(
		logits.engine_handle(),
		vec![rows as usize, candidates],
		candidate_count,
		DType::F32,
	)?;
	let indices = Matrix::allocate(
		logits.engine_handle(),
		vec![rows as usize, candidates],
		candidate_count,
		DType::I32,
	)?;
	let top_k_buffers = [
		BufferBinding::read(logits.storage()),
		BufferBinding::write(values.storage()),
		BufferBinding::write(indices.storage()),
	];
	let top_k_payload = [
		PushConstant::U32(rows),
		PushConstant::U32(vocabulary),
		PushConstant::U32(candidates_u32),
	];
	let random_buffers_eager = [
		BufferBinding::read(logits.storage()),
		BufferBinding::write(random.storage()),
	];
	let sample_buffers = [
		BufferBinding::read(values.storage()),
		BufferBinding::read(indices.storage()),
		BufferBinding::read(random.storage()),
		BufferBinding::write(output.storage()),
	];
	let sample_payload = [
		PushConstant::U32(rows),
		PushConstant::U32(candidates_u32),
		PushConstant::F32(temperature),
		PushConstant::F32(top_p),
	];
	let engine = logits.engine_handle();
	if engine.capture_active() {
		let state = Matrix::from_slice_handle(engine, vec![1], &[0_u32])?;
		let random_buffers = [
			BufferBinding::read(logits.storage()),
			BufferBinding::write(random.storage()),
			BufferBinding::read(state.storage()),
		];
		let advance_buffers = [BufferBinding::read_write(state.storage())];
		let dispatches = [
			ComputeDispatch {
				kernel: KernelId::MatrixTopKF32,
				buffers: &top_k_buffers,
				push_constants: &top_k_payload,
				workgroups: [rows, 1, 1],
			},
			ComputeDispatch {
				kernel: KernelId::MatrixPhiloxUniformReplayF32,
				buffers: &random_buffers,
				push_constants: random_payload,
				workgroups: KernelId::MatrixPhiloxUniformReplayF32.linear_workgroups(rows),
			},
			ComputeDispatch {
				kernel: KernelId::MatrixPhiloxReplayAdvanceU32,
				buffers: &advance_buffers,
				push_constants: &[],
				workgroups: [1, 1, 1],
			},
			ComputeDispatch {
				kernel: KernelId::MatrixSampleLogitsSortedF32,
				buffers: &sample_buffers,
				push_constants: &sample_payload,
				workgroups: KernelId::MatrixSampleLogitsSortedF32.linear_workgroups(rows),
			},
		];
		engine.record_split_semantic(&dispatches, semantic)
	} else {
		let dispatches = [
			ComputeDispatch {
				kernel: KernelId::MatrixTopKF32,
				buffers: &top_k_buffers,
				push_constants: &top_k_payload,
				workgroups: [rows, 1, 1],
			},
			ComputeDispatch {
				kernel: KernelId::MatrixPhiloxUniformF32,
				buffers: &random_buffers_eager,
				push_constants: random_payload,
				workgroups: KernelId::MatrixPhiloxUniformF32.linear_workgroups(rows),
			},
			ComputeDispatch {
				kernel: KernelId::MatrixSampleLogitsSortedF32,
				buffers: &sample_buffers,
				push_constants: &sample_payload,
				workgroups: KernelId::MatrixSampleLogitsSortedF32.linear_workgroups(rows),
			},
		];
		engine.record_split_semantic(&dispatches, semantic)
	}
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
