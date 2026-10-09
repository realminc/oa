//! Vector-quantization Matrix primitives.

use crate::{
	DType, Error, Matrix, OpAttribute, Result,
	runtime::{BufferBinding, ComputeDispatch, KernelId, PushConstant, SemanticDispatch},
};

use crate::ml::validation::{shader_u32, validate_f32_same_engine};

/// Nearest-code indices and gathered code vectors.
pub struct VqAssignResult {
	/// I32 nearest-code index for every latent row.
	pub indices: Matrix,
	/// Gathered winning code for every latent row.
	pub quantized: Matrix,
}

/// Functionally updated persistent EMA state.
pub struct VqEmaState {
	/// Next per-code EMA sum.
	pub embed_sum: Matrix,
	/// Next per-code EMA population.
	pub cluster_size: Matrix,
	/// Next codebook derived from the EMA state.
	pub codebook: Matrix,
}

/// Return a zero-copy stop-gradient view.
///
/// The returned Matrix shares storage and forward values with `input`, but owns
/// a fresh semantic identity with no reverse-mode edge to `input`.
///
/// # Errors
///
/// Returns an error only if the input's existing shape metadata is invalid.
pub fn detach(input: &Matrix) -> Result<Matrix> {
	input.reshape_view(input.shape().to_vec())
}

/// Assign every FP32 latent row to its nearest FP32 codebook row.
///
/// Squared L2 distance is used and equal-distance ties select the lower code
/// index, matching OA C++. The hard assignment is detached from reverse mode.
///
/// # Errors
///
/// Returns an error unless inputs are nonempty same-engine FP32 `[N,D]` and
/// `[K,D]` matrices or recording/allocation fails.
pub fn vq_assign(latent: &Matrix, codebook: &Matrix) -> Result<VqAssignResult> {
	let contract = crate::core::operation::ml::VQ_ASSIGN;
	let operation = contract.name();
	let [rows, code_dim] = latent.shape() else {
		return Err(Error::invalid_argument(format!(
			"{operation} latent must have shape [N,D]"
		)));
	};
	let [num_codes, codebook_dim] = codebook.shape() else {
		return Err(Error::invalid_argument(format!(
			"{operation} codebook must have shape [K,D]"
		)));
	};
	if *rows == 0 || *code_dim == 0 || *num_codes == 0 || code_dim != codebook_dim {
		return Err(Error::invalid_argument(format!(
			"{operation} requires nonempty latent [N,D] and codebook [K,D]"
		)));
	}
	validate_f32_same_engine(operation, &[latent, codebook])?;
	let rows_u32 = shader_u32(*rows, "row count", operation)?;
	let code_dim_u32 = shader_u32(*code_dim, "code dimension", operation)?;
	let num_codes_u32 = shader_u32(*num_codes, "code count", operation)?;
	let indices = Matrix::allocate(latent.engine_handle(), vec![*rows], *rows, DType::I32)?;
	let quantized = Matrix::allocate(
		latent.engine_handle(),
		latent.shape().to_vec(),
		latent.num_elements(),
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(latent.storage()),
		BufferBinding::read(codebook.storage()),
		BufferBinding::write(indices.storage()),
		BufferBinding::write(quantized.storage()),
	];
	let push_constants = [
		PushConstant::U32(rows_u32),
		PushConstant::U32(code_dim_u32),
		PushConstant::U32(num_codes_u32),
	];
	let inputs = [latent, codebook];
	let outputs = [&indices, &quantized];
	latent.engine_handle().record_semantic(
		ComputeDispatch {
			kernel: KernelId::MlVqAssignF32,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: KernelId::MlVqAssignF32.linear_workgroups(rows_u32),
		},
		SemanticDispatch {
			contract,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &[],
		},
	)?;
	Ok(VqAssignResult { indices, quantized })
}

/// Gather codebook rows selected by I32 token indices.
///
/// Invalid indices yield zero rows without an out-of-bounds read. This pure
/// decode operation is detached from reverse mode because EMA owns codebook
/// learning.
///
/// # Errors
///
/// Returns an error unless the codebook is FP32 `[K,D]`, indices are I32 `[N]`,
/// both belong to one engine, or recording/allocation fails.
pub fn vq_lookup(codebook: &Matrix, indices: &Matrix) -> Result<Matrix> {
	let contract = crate::core::operation::ml::VQ_LOOKUP;
	let operation = contract.name();
	let [num_codes, code_dim] = codebook.shape() else {
		return Err(Error::invalid_argument(format!(
			"{operation} codebook must have shape [K,D]"
		)));
	};
	let [index_count] = indices.shape() else {
		return Err(Error::invalid_argument(format!(
			"{operation} indices must have shape [N]"
		)));
	};
	if *num_codes == 0
		|| *code_dim == 0
		|| codebook.dtype() != DType::F32
		|| indices.dtype() != DType::I32
		|| !codebook.engine_handle().same_as(indices.engine_handle())
	{
		return Err(Error::invalid_argument(format!(
			"{operation} requires same-engine F32 codebook [K,D] and I32 indices [N]"
		)));
	}
	let output_count = index_count
		.checked_mul(*code_dim)
		.ok_or_else(|| Error::invalid_argument(format!("{operation} output size overflows usize")))?;
	let index_count_u32 = shader_u32(*index_count, "index count", operation)?;
	let num_codes_u32 = shader_u32(*num_codes, "code count", operation)?;
	let code_dim_u32 = shader_u32(*code_dim, "code dimension", operation)?;
	let output = Matrix::allocate(
		codebook.engine_handle(),
		vec![*index_count, *code_dim],
		output_count,
		DType::F32,
	)?;
	if output_count != 0 {
		let buffers = [
			BufferBinding::read(codebook.storage()),
			BufferBinding::read(indices.storage()),
			BufferBinding::write(output.storage()),
		];
		let push_constants = [
			PushConstant::U32(index_count_u32),
			PushConstant::U32(num_codes_u32),
			PushConstant::U32(code_dim_u32),
		];
		{
			let inputs: &[&Matrix] = &[codebook, indices];
			let outputs: &[&Matrix] = &[&output];
			let attributes: &[OpAttribute] = &[];
			let dispatch = ComputeDispatch {
				kernel: KernelId::MlVqLookupF32,
				buffers: &buffers,
				push_constants: &push_constants,
				workgroups: KernelId::MlVqLookupF32.linear_workgroups(shader_u32(
					output_count,
					"output element count",
					operation,
				)?),
			};
			let contract = dispatch.kernel.semantic_contract().ok_or_else(|| {
				Error::internal(format!(
					"lowering-only kernel {} cannot own one semantic ML operation",
					dispatch.kernel.report_name()
				))
			})?;
			let engine = inputs
				.first()
				.ok_or_else(|| Error::internal("semantic ML dispatch requires an input"))?
				.engine_handle();
			engine.record_semantic(
				dispatch,
				SemanticDispatch {
					contract,
					inputs,
					outputs,
					attributes,
				},
			)
		}?;
	}
	Ok(output)
}

/// Produce the next VQ EMA state without mutating input Matrix values.
///
/// This preserves the donor update equations, optional unit-RMS codebook
/// normalization, and exact dead-code revival hash. The owning module installs
/// the returned values into its stable persistent buffer handles.
///
/// # Errors
///
/// Returns an error for incompatible shapes, dtypes, engines, invalid numerical
/// options, ABI overflow, allocation failure, or runtime recording failure.
#[allow(
	clippy::too_many_arguments,
	reason = "publicly preserves the donor VQ state transition"
)]
pub fn vq_ema_update(
	latent: &Matrix,
	indices: &Matrix,
	embed_sum: &Matrix,
	cluster_size: &Matrix,
	codebook: &Matrix,
	decay: f32,
	epsilon: f32,
	dead_threshold: f32,
	seed: u32,
	normalize: bool,
) -> Result<VqEmaState> {
	let contract = crate::core::operation::ml::VQ_EMA_UPDATE;
	let operation = contract.name();
	let [rows, code_dim] = latent.shape() else {
		return Err(Error::invalid_argument(format!(
			"{operation} latent must have shape [N,D]"
		)));
	};
	let [num_codes, codebook_dim] = codebook.shape() else {
		return Err(Error::invalid_argument(format!(
			"{operation} codebook must have shape [K,D]"
		)));
	};
	if *rows == 0
		|| *code_dim == 0
		|| *num_codes == 0
		|| code_dim != codebook_dim
		|| indices.shape() != [*rows]
		|| indices.dtype() != DType::I32
		|| embed_sum.shape() != codebook.shape()
		|| cluster_size.shape() != [*num_codes]
		|| !decay.is_finite()
		|| !(0.0..=1.0).contains(&decay)
		|| !epsilon.is_finite()
		|| epsilon <= 0.0
		|| !dead_threshold.is_finite()
		|| dead_threshold < 0.0
	{
		return Err(Error::invalid_argument(format!(
			"{operation} received invalid VQ state or options"
		)));
	}
	validate_f32_same_engine(operation, &[latent, embed_sum, cluster_size, codebook])?;
	if !latent.engine_handle().same_as(indices.engine_handle()) {
		return Err(Error::invalid_argument(format!(
			"{operation} inputs must belong to one engine"
		)));
	}
	let rows_u32 = shader_u32(*rows, "row count", operation)?;
	let code_dim_u32 = shader_u32(*code_dim, "code dimension", operation)?;
	let num_codes_u32 = shader_u32(*num_codes, "code count", operation)?;
	let next_embed_sum = Matrix::allocate(
		latent.engine_handle(),
		embed_sum.shape().to_vec(),
		embed_sum.num_elements(),
		DType::F32,
	)?;
	let next_cluster_size = Matrix::allocate(
		latent.engine_handle(),
		cluster_size.shape().to_vec(),
		cluster_size.num_elements(),
		DType::F32,
	)?;
	let next_codebook = Matrix::allocate(
		latent.engine_handle(),
		codebook.shape().to_vec(),
		codebook.num_elements(),
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(latent.storage()),
		BufferBinding::read(indices.storage()),
		BufferBinding::read(embed_sum.storage()),
		BufferBinding::read(cluster_size.storage()),
		BufferBinding::read(codebook.storage()),
		BufferBinding::write(next_embed_sum.storage()),
		BufferBinding::write(next_cluster_size.storage()),
		BufferBinding::write(next_codebook.storage()),
	];
	let push_constants = [
		PushConstant::U32(rows_u32),
		PushConstant::U32(code_dim_u32),
		PushConstant::U32(num_codes_u32),
		PushConstant::F32(decay),
		PushConstant::F32(epsilon),
		PushConstant::F32(dead_threshold),
		PushConstant::U32(seed),
		PushConstant::U32(u32::from(normalize)),
	];
	let attributes = [
		OpAttribute::Float {
			name: "decay".into(),
			value: f64::from(decay),
		},
		OpAttribute::Float {
			name: "epsilon".into(),
			value: f64::from(epsilon),
		},
		OpAttribute::Float {
			name: "dead_threshold".into(),
			value: f64::from(dead_threshold),
		},
		OpAttribute::UnsignedInteger {
			name: "seed".into(),
			value: u64::from(seed),
		},
		OpAttribute::Boolean {
			name: "normalize".into(),
			value: normalize,
		},
	];
	{
		let inputs: &[&Matrix] = &[latent, indices, embed_sum, cluster_size, codebook];
		let outputs: &[&Matrix] = &[&next_embed_sum, &next_cluster_size, &next_codebook];
		let attributes: &[OpAttribute] = &attributes;
		let dispatch = ComputeDispatch {
			kernel: KernelId::MlVqEmaUpdateF32,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: KernelId::MlVqEmaUpdateF32.linear_workgroups(num_codes_u32),
		};
		let contract = dispatch.kernel.semantic_contract().ok_or_else(|| {
			Error::internal(format!(
				"lowering-only kernel {} cannot own one semantic ML operation",
				dispatch.kernel.report_name()
			))
		})?;
		let engine = inputs
			.first()
			.ok_or_else(|| Error::internal("semantic ML dispatch requires an input"))?
			.engine_handle();
		engine.record_semantic(
			dispatch,
			SemanticDispatch {
				contract,
				inputs,
				outputs,
				attributes,
			},
		)
	}?;
	Ok(VqEmaState {
		embed_sum: next_embed_sum,
		cluster_size: next_cluster_size,
		codebook: next_codebook,
	})
}
