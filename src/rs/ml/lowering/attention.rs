//! Private lowering for batched algebra and attention operations.

use crate::runtime::{
	BufferBinding, ComputeDispatch, KernelId, OptionalSemanticDispatch, PushConstant,
};
use crate::{DType, Error, Matrix, OpAttribute, Result};

use super::common::{record_semantic, shader_u32, validate_f32_same_engine};

#[derive(Clone, Copy)]
enum BmmLayout {
	Nn,
	Nt,
	Tn,
}

struct BmmGeometry {
	batch: u32,
	rows: u32,
	inner: u32,
	columns: u32,
	output_count: u32,
}

pub(in crate::ml) struct HeadTransformResult {
	pub(in crate::ml) output: Matrix,
	pub(in crate::ml) permuted: bool,
}

pub(in crate::ml) struct SdpaResult {
	pub(in crate::ml) output: Matrix,
	pub(in crate::ml) probabilities: Matrix,
}

pub(in crate::ml) struct FlashSdpaResult {
	pub(in crate::ml) output: Matrix,
	pub(in crate::ml) log_sum_exp: Matrix,
}

pub(in crate::ml) struct SdpaGradients {
	pub(in crate::ml) query: Matrix,
	pub(in crate::ml) key: Matrix,
	pub(in crate::ml) value: Matrix,
}

struct HeadGeometry {
	batch: u32,
	sequence_length: u32,
	num_heads: u32,
	head_dim: u32,
	element_count: u32,
}

impl HeadGeometry {
	fn attributes(&self) -> [OpAttribute; 3] {
		[
			OpAttribute::UnsignedInteger {
				name: "batch".into(),
				value: u64::from(self.batch),
			},
			OpAttribute::UnsignedInteger {
				name: "sequence_length".into(),
				value: u64::from(self.sequence_length),
			},
			OpAttribute::UnsignedInteger {
				name: "num_heads".into(),
				value: u64::from(self.num_heads),
			},
		]
	}

	const fn push_constants(&self) -> [PushConstant; 4] {
		[
			PushConstant::U32(self.batch),
			PushConstant::U32(self.sequence_length),
			PushConstant::U32(self.num_heads),
			PushConstant::U32(self.head_dim),
		]
	}
}

impl BmmGeometry {
	fn resolve(
		left: &Matrix,
		right: &Matrix,
		layout: BmmLayout,
		operation: &'static str,
	) -> Result<Self> {
		let [left_batch, left_middle, left_last] = left.shape() else {
			return Err(Error::invalid_argument(format!(
				"{operation} left input must have rank three; found {:?}",
				left.shape()
			)));
		};
		let [right_batch, right_middle, right_last] = right.shape() else {
			return Err(Error::invalid_argument(format!(
				"{operation} right input must have rank three; found {:?}",
				right.shape()
			)));
		};
		let (rows, inner, right_inner, columns) = match layout {
			BmmLayout::Nn => (*left_middle, *left_last, *right_middle, *right_last),
			BmmLayout::Nt => (*left_middle, *left_last, *right_last, *right_middle),
			BmmLayout::Tn => (*left_last, *left_middle, *right_middle, *right_last),
		};
		if *left_batch == 0
			|| rows == 0
			|| inner == 0
			|| columns == 0
			|| left_batch != right_batch
			|| inner != right_inner
		{
			return Err(Error::invalid_argument(format!(
				"{operation} requires compatible nonempty rank-three batches; found {:?} and {:?}",
				left.shape(),
				right.shape()
			)));
		}
		validate_f32_same_engine(operation, &[left, right])?;
		let output_count = left_batch
			.checked_mul(rows)
			.and_then(|count| count.checked_mul(columns))
			.ok_or_else(|| {
				Error::invalid_argument(format!("{operation} output size overflows usize"))
			})?;
		Ok(Self {
			batch: shader_u32(*left_batch, "batch", operation)?,
			rows: shader_u32(rows, "row count", operation)?,
			inner: shader_u32(inner, "inner dimension", operation)?,
			columns: shader_u32(columns, "column count", operation)?,
			output_count: shader_u32(output_count, "output element count", operation)?,
		})
	}

	const fn use_tiled(&self) -> bool {
		self.batch <= 65_535 && self.rows >= 8 && self.inner >= 8 && self.columns >= 8
	}

	const fn push_constants(&self) -> [PushConstant; 4] {
		[
			PushConstant::U32(self.batch),
			PushConstant::U32(self.rows),
			PushConstant::U32(self.inner),
			PushConstant::U32(self.columns),
		]
	}
}

pub(in crate::ml) fn bmm(left: &Matrix, right: &Matrix) -> Result<Matrix> {
	bmm_impl(
		left,
		right,
		BmmLayout::Nn,
		crate::core::operation::ml::BMM.name(),
		KernelId::MlBmmF32,
		KernelId::MlBmmTiled16F32,
	)
}

pub(in crate::ml) fn bmm_nt(left: &Matrix, right: &Matrix) -> Result<Matrix> {
	bmm_impl(
		left,
		right,
		BmmLayout::Nt,
		crate::core::operation::ml::BMM_NT.name(),
		KernelId::MlBmmNtF32,
		KernelId::MlBmmNtTiled16F32,
	)
}

pub(in crate::ml) fn bmm_tn(left: &Matrix, right: &Matrix) -> Result<Matrix> {
	bmm_impl(
		left,
		right,
		BmmLayout::Tn,
		crate::core::operation::ml::BMM_TN.name(),
		KernelId::MlBmmTnF32,
		KernelId::MlBmmTnTiled16F32,
	)
}

pub(in crate::ml) fn split_heads(
	input: &Matrix,
	batch: usize,
	sequence_length: usize,
	num_heads: usize,
) -> Result<HeadTransformResult> {
	const OPERATION: &str = "oa::ml::matrix::split_heads";
	let [rows, model_width] = input.shape() else {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} input must have shape [B*S, D]; found {:?}",
			input.shape()
		)));
	};
	let expected_rows = batch.checked_mul(sequence_length).ok_or_else(|| {
		Error::invalid_argument(format!(
			"{OPERATION} batch by sequence size overflows usize"
		))
	})?;
	if batch == 0
		|| sequence_length == 0
		|| num_heads == 0
		|| *rows != expected_rows
		|| *model_width == 0
		|| model_width % num_heads != 0
	{
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires nonzero B, S, H, [B*S, D] input, and D divisible by H"
		)));
	}
	validate_f32_same_engine(OPERATION, &[input])?;
	let head_dim = model_width / num_heads;
	if num_heads == 1 {
		return Ok(HeadTransformResult {
			output: input.reshape([batch, sequence_length, head_dim])?,
			permuted: false,
		});
	}
	let batch_heads = batch.checked_mul(num_heads).ok_or_else(|| {
		Error::invalid_argument(format!("{OPERATION} batch-head count overflows usize"))
	})?;
	let geometry = HeadGeometry {
		batch: shader_u32(batch, "batch", OPERATION)?,
		sequence_length: shader_u32(sequence_length, "sequence length", OPERATION)?,
		num_heads: shader_u32(num_heads, "head count", OPERATION)?,
		head_dim: shader_u32(head_dim, "head dimension", OPERATION)?,
		element_count: shader_u32(input.element_count(), "element count", OPERATION)?,
	};
	let output = Matrix::allocate(
		input.engine_handle(),
		vec![batch_heads, sequence_length, head_dim],
		input.element_count(),
		DType::F32,
	)?;
	record_head_transform(input, &output, geometry, KernelId::MlSplitHeadsF32)?;
	Ok(HeadTransformResult {
		output,
		permuted: true,
	})
}

pub(in crate::ml) fn merge_heads(
	input: &Matrix,
	batch: usize,
	sequence_length: usize,
	num_heads: usize,
) -> Result<HeadTransformResult> {
	const OPERATION: &str = "oa::ml::matrix::merge_heads";
	let [batch_heads, input_sequence, head_dim] = input.shape() else {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} input must have shape [B*H, S, D/H]; found {:?}",
			input.shape()
		)));
	};
	let expected_batch_heads = batch.checked_mul(num_heads).ok_or_else(|| {
		Error::invalid_argument(format!("{OPERATION} batch-head count overflows usize"))
	})?;
	if batch == 0
		|| sequence_length == 0
		|| num_heads == 0
		|| *batch_heads != expected_batch_heads
		|| *input_sequence != sequence_length
		|| *head_dim == 0
	{
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires nonzero B, S, H and input [B*H, S, D/H]"
		)));
	}
	validate_f32_same_engine(OPERATION, &[input])?;
	let model_width = num_heads.checked_mul(*head_dim).ok_or_else(|| {
		Error::invalid_argument(format!("{OPERATION} model width overflows usize"))
	})?;
	let output_rows = batch.checked_mul(sequence_length).ok_or_else(|| {
		Error::invalid_argument(format!("{OPERATION} output row count overflows usize"))
	})?;
	if num_heads == 1 {
		return Ok(HeadTransformResult {
			output: input.reshape([output_rows, model_width])?,
			permuted: false,
		});
	}
	let geometry = HeadGeometry {
		batch: shader_u32(batch, "batch", OPERATION)?,
		sequence_length: shader_u32(sequence_length, "sequence length", OPERATION)?,
		num_heads: shader_u32(num_heads, "head count", OPERATION)?,
		head_dim: shader_u32(*head_dim, "head dimension", OPERATION)?,
		element_count: shader_u32(input.element_count(), "element count", OPERATION)?,
	};
	let output = Matrix::allocate(
		input.engine_handle(),
		vec![output_rows, model_width],
		input.element_count(),
		DType::F32,
	)?;
	record_head_transform(input, &output, geometry, KernelId::MlMergeHeadsF32)?;
	Ok(HeadTransformResult {
		output,
		permuted: true,
	})
}

fn record_head_transform(
	input: &Matrix,
	output: &Matrix,
	geometry: HeadGeometry,
	kernel: KernelId,
) -> Result<()> {
	let buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::write(output.storage()),
	];
	let push_constants = geometry.push_constants();
	let attributes = geometry.attributes();
	record_semantic(
		&[input],
		&[output],
		&attributes,
		kernel,
		&buffers,
		&push_constants,
		kernel.linear_workgroups(geometry.element_count),
	)
}

pub(in crate::ml) fn scaled_dot_product_attention(
	query: &Matrix,
	key: &Matrix,
	value: &Matrix,
	additive_mask: Option<&Matrix>,
	scale: f32,
	causal: bool,
) -> Result<SdpaResult> {
	const OPERATION: &str = "oa::ml::matrix::scaled_dot_product_attention";
	let [batch_heads, sequence_length, head_dim] = query.shape() else {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires Q/K/V [batch_heads, sequence, head_dim]"
		)));
	};
	if *batch_heads == 0
		|| *sequence_length == 0
		|| *head_dim == 0
		|| key.shape() != query.shape()
		|| value.shape() != query.shape()
	{
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires equal nonempty Q/K/V [batch_heads, sequence, head_dim]"
		)));
	}
	validate_f32_same_engine(OPERATION, &[query, key, value])?;
	let mask_rows = batch_heads.checked_mul(*sequence_length).ok_or_else(|| {
		Error::invalid_argument(format!("{OPERATION} mask row count overflows usize"))
	})?;
	if let Some(mask) = additive_mask {
		if mask.shape() != [mask_rows, *sequence_length] {
			return Err(Error::invalid_argument(format!(
				"{OPERATION} additive mask must be [batch_heads*sequence, sequence]; found {:?}",
				mask.shape()
			)));
		}
		validate_f32_same_engine(OPERATION, &[query, mask])?;
	}

	let score_geometry = BmmGeometry::resolve(query, key, BmmLayout::Nt, OPERATION)?;
	let scores = Matrix::allocate(
		query.engine_handle(),
		vec![*batch_heads, *sequence_length, *sequence_length],
		score_geometry.output_count as usize,
		DType::F32,
	)?;
	let probabilities = Matrix::allocate(
		query.engine_handle(),
		scores.shape().to_vec(),
		scores.element_count(),
		DType::F32,
	)?;
	let output_geometry = BmmGeometry::resolve(&probabilities, value, BmmLayout::Nn, OPERATION)?;
	let output = Matrix::allocate(
		query.engine_handle(),
		query.shape().to_vec(),
		output_geometry.output_count as usize,
		DType::F32,
	)?;

	let score_buffers = [
		BufferBinding::read(query.storage()),
		BufferBinding::read(key.storage()),
		BufferBinding::write(scores.storage()),
	];
	let score_push_constants = score_geometry.push_constants();
	let (score_kernel, score_workgroups) = select_bmm_provider(
		&score_geometry,
		KernelId::MlBmmNtF32,
		KernelId::MlBmmNtTiled16F32,
	);
	let mask_storage = additive_mask.unwrap_or(&scores).storage();
	let probability_buffers = [
		BufferBinding::read(scores.storage()),
		BufferBinding::read(mask_storage),
		BufferBinding::write(probabilities.storage()),
	];
	let rows = shader_u32(mask_rows, "Softmax row count", OPERATION)?;
	let columns = shader_u32(*sequence_length, "Softmax column count", OPERATION)?;
	let probability_push_constants = [
		PushConstant::U32(rows),
		PushConstant::U32(columns),
		PushConstant::F32(scale),
		PushConstant::U32(u32::from(additive_mask.is_some())),
		PushConstant::U32(u32::from(causal)),
	];
	let probability_kernel = if columns <= 32 {
		KernelId::MlSdpaSoftmaxN32F32
	} else {
		KernelId::MlSdpaSoftmaxF32
	};
	let output_buffers = [
		BufferBinding::read(probabilities.storage()),
		BufferBinding::read(value.storage()),
		BufferBinding::write(output.storage()),
	];
	let output_push_constants = output_geometry.push_constants();
	let (output_kernel, output_workgroups) = select_bmm_provider(
		&output_geometry,
		KernelId::MlBmmF32,
		KernelId::MlBmmTiled16F32,
	);
	let dispatches = [
		ComputeDispatch {
			kernel: score_kernel,
			buffers: &score_buffers,
			push_constants: &score_push_constants,
			workgroups: score_workgroups,
		},
		ComputeDispatch {
			kernel: probability_kernel,
			buffers: &probability_buffers,
			push_constants: &probability_push_constants,
			workgroups: [rows, 1, 1],
		},
		ComputeDispatch {
			kernel: output_kernel,
			buffers: &output_buffers,
			push_constants: &output_push_constants,
			workgroups: output_workgroups,
		},
	];
	let semantic_inputs = [Some(query), Some(key), Some(value), additive_mask];
	let semantic_outputs = [&output, &probabilities];
	let attributes = [
		OpAttribute::Float {
			name: "scale".into(),
			value: f64::from(scale),
		},
		OpAttribute::Boolean {
			name: "causal".into(),
			value: causal,
		},
	];
	query.engine_handle().record_split_optional_semantic(
		&dispatches,
		OptionalSemanticDispatch {
			contract: crate::core::operation::ml::SCALED_DOT_PRODUCT_ATTENTION,
			inputs: &semantic_inputs,
			outputs: &semantic_outputs,
			attributes: &attributes,
		},
	)?;
	Ok(SdpaResult {
		output,
		probabilities,
	})
}

pub(in crate::ml) fn flash_attention_causal(
	query: &Matrix,
	key: &Matrix,
	value: &Matrix,
	scale: f32,
) -> Result<FlashSdpaResult> {
	const OPERATION: &str = "oa::ml::matrix::scaled_dot_product_attention";
	let [batch_heads, sequence_length, head_dim] = query.shape() else {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} Flash provider requires Q/K/V [batch_heads, sequence, head_dim]"
		)));
	};
	if *batch_heads == 0
		|| *sequence_length == 0
		|| *sequence_length > 1024
		|| *head_dim == 0
		|| key.shape() != query.shape()
		|| value.shape() != query.shape()
	{
		return Err(Error::invalid_argument(format!(
			"{OPERATION} Flash provider requires equal nonempty Q/K/V and sequence length <= 1024"
		)));
	}
	validate_f32_same_engine(OPERATION, &[query, key, value])?;
	let batch_heads_u32 = shader_u32(*batch_heads, "batch-head count", OPERATION)?;
	let sequence_length_u32 = shader_u32(*sequence_length, "sequence length", OPERATION)?;
	let head_dim_u32 = shader_u32(*head_dim, "head dimension", OPERATION)?;
	let rows = batch_heads_u32
		.checked_mul(sequence_length_u32)
		.ok_or_else(|| Error::invalid_argument(format!("{OPERATION} row count exceeds u32")))?;
	let output = Matrix::allocate(
		query.engine_handle(),
		query.shape().to_vec(),
		query.element_count(),
		DType::F32,
	)?;
	let log_sum_exp = Matrix::allocate(
		query.engine_handle(),
		vec![*batch_heads, *sequence_length],
		usize::try_from(rows)
			.map_err(|_| Error::invalid_argument(format!("{OPERATION} row count exceeds usize")))?,
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(query.storage()),
		BufferBinding::read(key.storage()),
		BufferBinding::read(value.storage()),
		BufferBinding::write(output.storage()),
		BufferBinding::write(log_sum_exp.storage()),
	];
	let push_constants = [
		PushConstant::U32(batch_heads_u32),
		PushConstant::U32(sequence_length_u32),
		PushConstant::U32(head_dim_u32),
		PushConstant::F32(scale),
	];
	let attributes = [
		OpAttribute::Float {
			name: "scale".into(),
			value: f64::from(scale),
		},
		OpAttribute::Boolean {
			name: "causal".into(),
			value: true,
		},
	];
	let semantic_inputs = [Some(query), Some(key), Some(value), None];
	let semantic_outputs = [&output, &log_sum_exp];
	query.engine_handle().record_optional_semantic(
		ComputeDispatch {
			kernel: KernelId::MlFlashAttentionCausalF32,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: [rows, 1, 1],
		},
		OptionalSemanticDispatch {
			contract: crate::core::operation::ml::SCALED_DOT_PRODUCT_ATTENTION,
			inputs: &semantic_inputs,
			outputs: &semantic_outputs,
			attributes: &attributes,
		},
	)?;
	Ok(FlashSdpaResult {
		output,
		log_sum_exp,
	})
}

pub(in crate::ml) fn flash_attention_causal_backward(
	query: &Matrix,
	key: &Matrix,
	value: &Matrix,
	output: &Matrix,
	log_sum_exp: &Matrix,
	output_gradient: &Matrix,
	scale: f32,
) -> Result<SdpaGradients> {
	const OPERATION: &str = "oa::ml::matrix::scaled_dot_product_attention_backward";
	let [batch_heads, sequence_length, head_dim] = query.shape() else {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires Q/K/V [batch_heads, sequence, head_dim]"
		)));
	};
	if *batch_heads == 0
		|| *sequence_length == 0
		|| *sequence_length > 1024
		|| *head_dim == 0
		|| key.shape() != query.shape()
		|| value.shape() != query.shape()
		|| output.shape() != query.shape()
		|| output_gradient.shape() != query.shape()
		|| log_sum_exp.shape() != [*batch_heads, *sequence_length]
	{
		return Err(Error::invalid_argument(format!(
			"{OPERATION} received incompatible Flash provider state"
		)));
	}
	validate_f32_same_engine(
		OPERATION,
		&[query, key, value, output, log_sum_exp, output_gradient],
	)?;
	let batch_heads_u32 = shader_u32(*batch_heads, "batch-head count", OPERATION)?;
	let sequence_length_u32 = shader_u32(*sequence_length, "sequence length", OPERATION)?;
	let head_dim_u32 = shader_u32(*head_dim, "head dimension", OPERATION)?;
	let rows = batch_heads_u32
		.checked_mul(sequence_length_u32)
		.ok_or_else(|| Error::invalid_argument(format!("{OPERATION} row count exceeds u32")))?;
	let query_gradient = Matrix::allocate(
		query.engine_handle(),
		query.shape().to_vec(),
		query.element_count(),
		DType::F32,
	)?;
	let key_gradient = Matrix::allocate(
		query.engine_handle(),
		key.shape().to_vec(),
		key.element_count(),
		DType::F32,
	)?;
	let value_gradient = Matrix::allocate(
		query.engine_handle(),
		value.shape().to_vec(),
		value.element_count(),
		DType::F32,
	)?;
	let query_buffers = [
		BufferBinding::read(query.storage()),
		BufferBinding::read(key.storage()),
		BufferBinding::read(value.storage()),
		BufferBinding::read(output.storage()),
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::read(log_sum_exp.storage()),
		BufferBinding::write(query_gradient.storage()),
	];
	let query_push_constants = [
		PushConstant::U32(batch_heads_u32),
		PushConstant::U32(sequence_length_u32),
		PushConstant::U32(head_dim_u32),
		PushConstant::F32(scale),
	];
	let key_value_buffers = [
		BufferBinding::read(query.storage()),
		BufferBinding::read(key.storage()),
		BufferBinding::read(value.storage()),
		BufferBinding::read(output.storage()),
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::read(log_sum_exp.storage()),
		BufferBinding::write(key_gradient.storage()),
		BufferBinding::write(value_gradient.storage()),
	];
	let key_value_push_constants = query_push_constants;
	let dispatches = [
		ComputeDispatch {
			kernel: KernelId::MlFlashAttentionCausalBackwardQF32,
			buffers: &query_buffers,
			push_constants: &query_push_constants,
			workgroups: [rows, 1, 1],
		},
		ComputeDispatch {
			kernel: KernelId::MlFlashAttentionCausalBackwardKvF32,
			buffers: &key_value_buffers,
			push_constants: &key_value_push_constants,
			workgroups: [rows, 1, 1],
		},
	];
	let semantic_inputs = [
		Some(query),
		Some(key),
		Some(value),
		Some(output),
		Some(log_sum_exp),
		Some(output_gradient),
	];
	let semantic_outputs = [&query_gradient, &key_gradient, &value_gradient];
	let attributes = [
		OpAttribute::Float {
			name: "scale".into(),
			value: f64::from(scale),
		},
		OpAttribute::Boolean {
			name: "causal".into(),
			value: true,
		},
	];
	query.engine_handle().record_split_optional_semantic(
		&dispatches,
		OptionalSemanticDispatch {
			contract: crate::core::operation::ml::SCALED_DOT_PRODUCT_ATTENTION_BACKWARD,
			inputs: &semantic_inputs,
			outputs: &semantic_outputs,
			attributes: &attributes,
		},
	)?;
	Ok(SdpaGradients {
		query: query_gradient,
		key: key_gradient,
		value: value_gradient,
	})
}

fn select_bmm_provider(
	geometry: &BmmGeometry,
	generic: KernelId,
	tiled: KernelId,
) -> (KernelId, [u32; 3]) {
	if geometry.use_tiled() {
		(
			tiled,
			[
				geometry.columns.div_ceil(16),
				geometry.rows.div_ceil(16),
				geometry.batch,
			],
		)
	} else {
		(generic, generic.linear_workgroups(geometry.output_count))
	}
}

pub(in crate::ml) fn softmax_scaled_masked(
	scores: &Matrix,
	mask: &Matrix,
	scale: f32,
) -> Result<Matrix> {
	const OPERATION: &str = "oa::ml::matrix::softmax_scaled_masked";
	let (rows, columns) = softmax_geometry(scores, mask, OPERATION)?;
	let output = Matrix::allocate(
		scores.engine_handle(),
		scores.shape().to_vec(),
		scores.element_count(),
		DType::F32,
	)?;
	let kernel = if columns <= 32 {
		KernelId::MlSoftmaxScaledMaskedN32F32
	} else {
		KernelId::MlSoftmaxScaledMaskedF32
	};
	let buffers = [
		BufferBinding::read(scores.storage()),
		BufferBinding::read(mask.storage()),
		BufferBinding::write(output.storage()),
	];
	let push_constants = [
		PushConstant::U32(rows),
		PushConstant::U32(columns),
		PushConstant::F32(scale),
	];
	let attributes = [OpAttribute::Float {
		name: "scale".into(),
		value: f64::from(scale),
	}];
	record_semantic(
		&[scores, mask],
		&[&output],
		&attributes,
		kernel,
		&buffers,
		&push_constants,
		[rows, 1, 1],
	)?;
	Ok(output)
}

pub(in crate::ml) fn softmax_scaled_masked_backward(
	forward_output: &Matrix,
	output_gradient: &Matrix,
	scale: f32,
) -> Result<Matrix> {
	const OPERATION: &str = "oa::ml::matrix::softmax_scaled_masked_backward";
	let (rows, columns) = softmax_geometry(forward_output, output_gradient, OPERATION)?;
	let input_gradient = Matrix::allocate(
		forward_output.engine_handle(),
		forward_output.shape().to_vec(),
		forward_output.element_count(),
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(forward_output.storage()),
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::write(input_gradient.storage()),
	];
	let push_constants = [
		PushConstant::U32(rows),
		PushConstant::U32(columns),
		PushConstant::F32(scale),
	];
	let attributes = [OpAttribute::Float {
		name: "scale".into(),
		value: f64::from(scale),
	}];
	record_semantic(
		&[forward_output, output_gradient],
		&[&input_gradient],
		&attributes,
		KernelId::MlSoftmaxScaledMaskedBackwardF32,
		&buffers,
		&push_constants,
		[rows, 1, 1],
	)?;
	Ok(input_gradient)
}

fn softmax_geometry(input: &Matrix, other: &Matrix, operation: &'static str) -> Result<(u32, u32)> {
	if input.shape() != other.shape() || input.element_count() == 0 {
		return Err(Error::invalid_argument(format!(
			"{operation} requires equal nonempty shapes; found {:?} and {:?}",
			input.shape(),
			other.shape()
		)));
	}
	validate_f32_same_engine(operation, &[input, other])?;
	let (rows, columns) = match input.shape() {
		[rows, columns] => (*rows, *columns),
		_ => (1, input.element_count()),
	};
	Ok((
		shader_u32(rows, "row count", operation)?,
		shader_u32(columns, "column count", operation)?,
	))
}

fn bmm_impl(
	left: &Matrix,
	right: &Matrix,
	layout: BmmLayout,
	operation: &'static str,
	generic: KernelId,
	tiled: KernelId,
) -> Result<Matrix> {
	let geometry = BmmGeometry::resolve(left, right, layout, operation)?;
	let output = Matrix::allocate(
		left.engine_handle(),
		vec![
			geometry.batch as usize,
			geometry.rows as usize,
			geometry.columns as usize,
		],
		geometry.output_count as usize,
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(left.storage()),
		BufferBinding::read(right.storage()),
		BufferBinding::write(output.storage()),
	];
	let push_constants = geometry.push_constants();
	let (kernel, workgroups) = if geometry.use_tiled() {
		(
			tiled,
			[
				geometry.columns.div_ceil(16),
				geometry.rows.div_ceil(16),
				geometry.batch,
			],
		)
	} else {
		(generic, generic.linear_workgroups(geometry.output_count))
	};
	record_semantic(
		&[left, right],
		&[&output],
		&[],
		kernel,
		&buffers,
		&push_constants,
		workgroups,
	)?;
	Ok(output)
}
