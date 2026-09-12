//! Private lowering for ML Matrix operations.

use crate::{
	DType, Error, Matrix, OpAttribute, Result,
	runtime::{BufferBinding, ComputeDispatch, KernelId, PushConstant, SemanticDispatch},
};

use super::common::{record_semantic, shader_u32, validate_f32_same_engine};

pub(in crate::ml) fn linear(input: &Matrix, weight: &Matrix, bias: &Matrix) -> Result<Matrix> {
	const OPERATION: &str = crate::core::operation::ml::LINEAR.name();
	if input.shape().len() < 2 {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} input must have rank at least two; found {:?}",
			input.shape()
		)));
	}
	let input_features = *input
		.shape()
		.last()
		.ok_or_else(|| Error::invalid_argument(format!("{OPERATION} input shape is empty")))?;
	let [output_features, weight_features] = weight.shape() else {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} weight must have rank two; found {:?}",
			weight.shape()
		)));
	};
	if bias.shape() != [*output_features] || input_features != *weight_features {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires input [..., I] with rank at least two, weight [O, I], and bias [O]; found {:?}, {:?}, {:?}",
			input.shape(),
			weight.shape(),
			bias.shape()
		)));
	}
	validate_f32_same_engine(OPERATION, &[input, weight, bias])?;
	let batch = input.shape()[..input.shape().len() - 1]
		.iter()
		.try_fold(1_usize, |product, extent| product.checked_mul(*extent))
		.ok_or_else(|| {
			Error::invalid_argument(format!("{OPERATION} leading size overflows usize"))
		})?;
	let output_count = batch.checked_mul(*output_features).ok_or_else(|| {
		Error::invalid_argument(format!("{OPERATION} output size overflows usize"))
	})?;
	let mut output_shape = input.shape().to_vec();
	*output_shape
		.last_mut()
		.ok_or_else(|| Error::invalid_argument(format!("{OPERATION} input shape is empty")))? =
		*output_features;
	let batch = shader_u32(batch, "flattened row count", OPERATION)?;
	let input_features = shader_u32(input_features, "input feature count", OPERATION)?;
	let output_features = shader_u32(*output_features, "output feature count", OPERATION)?;
	if input_features == 0 || output_features == 0 {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} feature dimensions must be nonzero"
		)));
	}
	let output = Matrix::allocate(
		input.engine_handle(),
		output_shape,
		output_count,
		DType::F32,
	)?;
	if output_count != 0 {
		let buffers = [
			BufferBinding::read(input.storage()),
			BufferBinding::read(weight.storage()),
			BufferBinding::read(bias.storage()),
			BufferBinding::write(output.storage()),
		];
		let push_constants = [
			PushConstant::U32(batch),
			PushConstant::U32(input_features),
			PushConstant::U32(output_features),
		];
		let kernel = KernelId::MlLinearF32;
		record_semantic(
			&[input, weight, bias],
			&[&output],
			&[],
			kernel,
			&buffers,
			&push_constants,
			kernel.output_workgroups(batch, output_features),
		)?;
	}
	Ok(output)
}

pub(in crate::ml) fn linear_backward(
	input: &Matrix,
	weight: &Matrix,
	output_gradient: &Matrix,
) -> Result<(Matrix, Matrix, Matrix)> {
	const OPERATION: &str = crate::core::operation::ml::LINEAR_BACKWARD.name();
	if input.shape().len() < 2 {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} input must have rank at least two"
		)));
	}
	let input_features = *input
		.shape()
		.last()
		.ok_or_else(|| Error::invalid_argument(format!("{OPERATION} input shape is empty")))?;
	let [output_features, weight_features] = weight.shape() else {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} weight must have rank two"
		)));
	};
	let mut output_shape = input.shape().to_vec();
	*output_shape
		.last_mut()
		.ok_or_else(|| Error::invalid_argument(format!("{OPERATION} input shape is empty")))? =
		*output_features;
	if input_features != *weight_features || output_gradient.shape() != output_shape {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} shapes are incompatible: input {:?}, weight {:?}, output gradient {:?}",
			input.shape(),
			weight.shape(),
			output_gradient.shape()
		)));
	}
	validate_f32_same_engine(OPERATION, &[input, weight, output_gradient])?;
	let batch = input.shape()[..input.shape().len() - 1]
		.iter()
		.try_fold(1_usize, |product, extent| product.checked_mul(*extent))
		.ok_or_else(|| {
			Error::invalid_argument(format!("{OPERATION} leading size overflows usize"))
		})?;
	let batch_u32 = shader_u32(batch, "flattened row count", OPERATION)?;
	let input_features_u32 = shader_u32(input_features, "input feature count", OPERATION)?;
	let output_features_u32 = shader_u32(*output_features, "output feature count", OPERATION)?;
	let input_gradient = Matrix::allocate(
		input.engine_handle(),
		input.shape().to_vec(),
		input.num_elements(),
		DType::F32,
	)?;
	let weight_gradient = Matrix::allocate(
		input.engine_handle(),
		weight.shape().to_vec(),
		weight.num_elements(),
		DType::F32,
	)?;
	let bias_gradient = Matrix::allocate(
		input.engine_handle(),
		vec![*output_features],
		*output_features,
		DType::F32,
	)?;
	let dimensions = [
		PushConstant::U32(batch_u32),
		PushConstant::U32(input_features_u32),
		PushConstant::U32(output_features_u32),
	];
	let input_buffers = [
		BufferBinding::read(weight.storage()),
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::write(input_gradient.storage()),
	];
	if input.num_elements() != 0 {
		let input_kernel = KernelId::MlLinearBackwardF32;
		record_semantic(
			&[weight, output_gradient],
			&[&input_gradient],
			&[],
			input_kernel,
			&input_buffers,
			&dimensions,
			input_kernel.output_workgroups(batch_u32, input_features_u32),
		)?;
	}
	let parameter_buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::write(weight_gradient.storage()),
		BufferBinding::write(bias_gradient.storage()),
	];
	record_semantic(
		&[input, output_gradient],
		&[&weight_gradient, &bias_gradient],
		&[],
		KernelId::MlLinearParameterBackwardF32,
		&parameter_buffers,
		&dimensions,
		[output_features_u32, input_features_u32.div_ceil(32), 1],
	)?;
	Ok((input_gradient, weight_gradient, bias_gradient))
}

pub(in crate::ml) struct LayerNormOutput {
	pub(in crate::ml) output: Matrix,
	pub(in crate::ml) normalized: Matrix,
	pub(in crate::ml) inverse_stddev: Matrix,
}

pub(in crate::ml) fn layer_norm(
	input: &Matrix,
	weight: &Matrix,
	bias: &Matrix,
	epsilon: f32,
) -> Result<LayerNormOutput> {
	const OPERATION: &str = crate::core::operation::ml::LAYER_NORM.name();
	let (rows, columns) = validate_layer_norm_inputs(input, weight, bias, epsilon, OPERATION)?;
	let output = Matrix::allocate(
		input.engine_handle(),
		input.shape().to_vec(),
		input.num_elements(),
		DType::F32,
	)?;
	let normalized = Matrix::allocate(
		input.engine_handle(),
		input.shape().to_vec(),
		input.num_elements(),
		DType::F32,
	)?;
	let inverse_stddev = Matrix::allocate(
		input.engine_handle(),
		vec![rows as usize],
		rows as usize,
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::read(weight.storage()),
		BufferBinding::read(bias.storage()),
		BufferBinding::write(output.storage()),
		BufferBinding::write(normalized.storage()),
		BufferBinding::write(inverse_stddev.storage()),
	];
	let push_constants = [
		PushConstant::U32(rows),
		PushConstant::U32(columns),
		PushConstant::F32(epsilon),
	];
	let attributes = [OpAttribute::Float {
		name: "epsilon".into(),
		value: f64::from(epsilon),
	}];
	record_semantic(
		&[input, weight, bias],
		&[&output, &normalized, &inverse_stddev],
		&attributes,
		KernelId::MlLayerNormF32,
		&buffers,
		&push_constants,
		[rows, 1, 1],
	)?;
	Ok(LayerNormOutput {
		output,
		normalized,
		inverse_stddev,
	})
}

pub(in crate::ml) fn layer_norm_backward(
	input: &Matrix,
	weight: &Matrix,
	normalized: &Matrix,
	inverse_stddev: &Matrix,
	output_gradient: &Matrix,
) -> Result<(Matrix, Matrix, Matrix)> {
	const OPERATION: &str = crate::core::operation::ml::LAYER_NORM_BACKWARD.name();
	let Some(&columns) = input.shape().last() else {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} input must have at least one dimension"
		)));
	};
	if columns == 0
		|| input.num_elements() == 0
		|| weight.shape() != [columns]
		|| normalized.shape() != input.shape()
		|| inverse_stddev.shape() != [input.num_elements() / columns]
	{
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires nonempty input and matching weight and saved-state shapes"
		)));
	}
	if output_gradient.shape() != input.shape() {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} output gradient must match input shape; found {:?} and {:?}",
			output_gradient.shape(),
			input.shape()
		)));
	}
	validate_f32_same_engine(
		OPERATION,
		&[input, weight, normalized, inverse_stddev, output_gradient],
	)?;
	let rows = shader_u32(input.num_elements() / columns, "row count", OPERATION)?;
	let columns = shader_u32(columns, "normalized dimension", OPERATION)?;
	let input_gradient = Matrix::allocate(
		input.engine_handle(),
		input.shape().to_vec(),
		input.num_elements(),
		DType::F32,
	)?;
	let weight_contribution = Matrix::allocate(
		input.engine_handle(),
		input.shape().to_vec(),
		input.num_elements(),
		DType::F32,
	)?;
	let weight_gradient = Matrix::allocate(
		input.engine_handle(),
		weight.shape().to_vec(),
		weight.num_elements(),
		DType::F32,
	)?;
	let bias_gradient = Matrix::allocate(
		input.engine_handle(),
		weight.shape().to_vec(),
		weight.num_elements(),
		DType::F32,
	)?;
	let input_buffers = [
		BufferBinding::read(normalized.storage()),
		BufferBinding::read(inverse_stddev.storage()),
		BufferBinding::read(weight.storage()),
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::write(input_gradient.storage()),
		BufferBinding::write(weight_contribution.storage()),
	];
	let input_push_constants = [PushConstant::U32(rows), PushConstant::U32(columns)];
	let weight_buffers = [
		BufferBinding::read(weight_contribution.storage()),
		BufferBinding::write(weight_gradient.storage()),
	];
	let bias_buffers = [
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::write(bias_gradient.storage()),
	];
	let sum_push_constants = [
		PushConstant::U32(1),
		PushConstant::U32(rows),
		PushConstant::U32(columns),
	];
	let sum_workgroups = KernelId::MatrixSumAxisF32.linear_workgroups(columns);
	let dispatches = [
		ComputeDispatch {
			kernel: KernelId::MlLayerNormBackwardF32,
			buffers: &input_buffers,
			push_constants: &input_push_constants,
			workgroups: [rows, 1, 1],
		},
		ComputeDispatch {
			kernel: KernelId::MatrixSumAxisF32,
			buffers: &weight_buffers,
			push_constants: &sum_push_constants,
			workgroups: sum_workgroups,
		},
		ComputeDispatch {
			kernel: KernelId::MatrixSumAxisF32,
			buffers: &bias_buffers,
			push_constants: &sum_push_constants,
			workgroups: sum_workgroups,
		},
	];
	let inputs = [normalized, inverse_stddev, weight, output_gradient];
	let outputs = [&input_gradient, &weight_gradient, &bias_gradient];
	input.engine_handle().record_split_semantic(
		&dispatches,
		SemanticDispatch {
			contract: crate::core::operation::ml::LAYER_NORM_BACKWARD,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &[],
		},
	)?;
	Ok((input_gradient, weight_gradient, bias_gradient))
}

pub(in crate::ml) fn rms_norm(input: &Matrix, weight: &Matrix, epsilon: f32) -> Result<Matrix> {
	const OPERATION: &str = crate::core::operation::ml::RMS_NORM.name();
	let (rows, columns) = validate_rms_norm_inputs(input, weight, epsilon, OPERATION)?;
	let output = Matrix::allocate(
		input.engine_handle(),
		input.shape().to_vec(),
		input.num_elements(),
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::read(weight.storage()),
		BufferBinding::write(output.storage()),
	];
	let push_constants = [
		PushConstant::U32(rows),
		PushConstant::U32(columns),
		PushConstant::F32(epsilon),
	];
	let attributes = [OpAttribute::Float {
		name: "epsilon".into(),
		value: f64::from(epsilon),
	}];
	record_semantic(
		&[input, weight],
		&[&output],
		&attributes,
		KernelId::MlRmsNormF32,
		&buffers,
		&push_constants,
		[rows, 1, 1],
	)?;
	Ok(output)
}

pub(in crate::ml) fn rms_norm_backward(
	input: &Matrix,
	weight: &Matrix,
	output_gradient: &Matrix,
	epsilon: f32,
) -> Result<(Matrix, Matrix)> {
	const OPERATION: &str = crate::core::operation::ml::RMS_NORM_BACKWARD.name();
	let (rows, columns) = validate_rms_norm_inputs(input, weight, epsilon, OPERATION)?;
	validate_f32_same_engine(OPERATION, &[input, weight, output_gradient])?;
	if output_gradient.shape() != input.shape() {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} output gradient must match input shape; found {:?} and {:?}",
			output_gradient.shape(),
			input.shape()
		)));
	}
	let input_gradient = Matrix::allocate(
		input.engine_handle(),
		input.shape().to_vec(),
		input.num_elements(),
		DType::F32,
	)?;
	let weight_contribution = Matrix::allocate(
		input.engine_handle(),
		input.shape().to_vec(),
		input.num_elements(),
		DType::F32,
	)?;
	let weight_gradient = Matrix::allocate(
		input.engine_handle(),
		weight.shape().to_vec(),
		weight.num_elements(),
		DType::F32,
	)?;
	let backward_buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::read(weight.storage()),
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::write(input_gradient.storage()),
		BufferBinding::write(weight_contribution.storage()),
	];
	let backward_push_constants = [
		PushConstant::U32(rows),
		PushConstant::U32(columns),
		PushConstant::F32(epsilon),
	];
	let parameter_buffers = [
		BufferBinding::read(weight_contribution.storage()),
		BufferBinding::write(weight_gradient.storage()),
	];
	let parameter_push_constants = [
		PushConstant::U32(1),
		PushConstant::U32(rows),
		PushConstant::U32(columns),
	];
	let dispatches = [
		ComputeDispatch {
			kernel: KernelId::MlRmsNormBackwardF32,
			buffers: &backward_buffers,
			push_constants: &backward_push_constants,
			workgroups: [rows, 1, 1],
		},
		ComputeDispatch {
			kernel: KernelId::MatrixSumAxisF32,
			buffers: &parameter_buffers,
			push_constants: &parameter_push_constants,
			workgroups: KernelId::MatrixSumAxisF32.linear_workgroups(columns),
		},
	];
	let inputs = [input, weight, output_gradient];
	let outputs = [&input_gradient, &weight_gradient];
	let attributes = [OpAttribute::Float {
		name: "epsilon".into(),
		value: f64::from(epsilon),
	}];
	input.engine_handle().record_split_semantic(
		&dispatches,
		SemanticDispatch {
			contract: crate::core::operation::ml::RMS_NORM_BACKWARD,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok((input_gradient, weight_gradient))
}

fn validate_rms_norm_inputs(
	input: &Matrix,
	weight: &Matrix,
	epsilon: f32,
	operation: &'static str,
) -> Result<(u32, u32)> {
	let Some(&columns) = input.shape().last() else {
		return Err(Error::invalid_argument(format!(
			"{operation} input must have at least one dimension"
		)));
	};
	if columns == 0 || input.num_elements() == 0 || weight.shape() != [columns] {
		return Err(Error::invalid_argument(format!(
			"{operation} requires nonempty input and a weight vector matching its final dimension"
		)));
	}
	if !epsilon.is_finite() || epsilon <= 0.0 {
		return Err(Error::invalid_argument(format!(
			"{operation} epsilon must be finite and positive"
		)));
	}
	validate_f32_same_engine(operation, &[input, weight])?;
	let rows = shader_u32(input.num_elements() / columns, "row count", operation)?;
	let columns = shader_u32(columns, "normalized dimension", operation)?;
	Ok((rows, columns))
}

pub(in crate::ml) fn rope(
	input: &Matrix,
	num_heads: usize,
	head_dim: usize,
	theta_base: f32,
	position_offset: u32,
) -> Result<Matrix> {
	const OPERATION: &str = crate::core::operation::ml::ROPE.name();
	let (tokens, num_heads, head_dim, pair_count) =
		validate_rope(input, num_heads, head_dim, theta_base, OPERATION)?;
	let output = Matrix::allocate(
		input.engine_handle(),
		input.shape().to_vec(),
		input.num_elements(),
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::write(output.storage()),
	];
	let push_constants = [
		PushConstant::U32(tokens),
		PushConstant::U32(num_heads),
		PushConstant::U32(head_dim),
		PushConstant::F32(theta_base),
		PushConstant::U32(position_offset),
	];
	let attributes = rope_attributes(num_heads, head_dim, theta_base, position_offset);
	let kernel = KernelId::MlRopeF32;
	record_semantic(
		&[input],
		&[&output],
		&attributes,
		kernel,
		&buffers,
		&push_constants,
		kernel.linear_workgroups(pair_count),
	)?;
	Ok(output)
}

pub(in crate::ml) fn rope_backward(
	output_gradient: &Matrix,
	num_heads: usize,
	head_dim: usize,
	theta_base: f32,
	position_offset: u32,
) -> Result<Matrix> {
	const OPERATION: &str = crate::core::operation::ml::ROPE_BACKWARD.name();
	let (tokens, num_heads, head_dim, pair_count) =
		validate_rope(output_gradient, num_heads, head_dim, theta_base, OPERATION)?;
	let input_gradient = Matrix::allocate(
		output_gradient.engine_handle(),
		output_gradient.shape().to_vec(),
		output_gradient.num_elements(),
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::write(input_gradient.storage()),
	];
	let push_constants = [
		PushConstant::U32(tokens),
		PushConstant::U32(num_heads),
		PushConstant::U32(head_dim),
		PushConstant::F32(theta_base),
		PushConstant::U32(position_offset),
	];
	let attributes = rope_attributes(num_heads, head_dim, theta_base, position_offset);
	let kernel = KernelId::MlRopeBackwardF32;
	record_semantic(
		&[output_gradient],
		&[&input_gradient],
		&attributes,
		kernel,
		&buffers,
		&push_constants,
		kernel.linear_workgroups(pair_count),
	)?;
	Ok(input_gradient)
}

fn validate_rope(
	input: &Matrix,
	num_heads: usize,
	head_dim: usize,
	theta_base: f32,
	operation: &'static str,
) -> Result<(u32, u32, u32, u32)> {
	let [tokens, width] = input.shape() else {
		return Err(Error::invalid_argument(format!(
			"{operation} input must have shape [tokens, num_heads * head_dim]"
		)));
	};
	let expected_width = num_heads
		.checked_mul(head_dim)
		.ok_or_else(|| Error::invalid_argument(format!("{operation} width overflows usize")))?;
	if *tokens == 0
		|| num_heads == 0
		|| head_dim == 0
		|| !head_dim.is_multiple_of(2)
		|| *width != expected_width
	{
		return Err(Error::invalid_argument(format!(
			"{operation} requires nonempty [tokens, num_heads * even head_dim] input"
		)));
	}
	if !theta_base.is_finite() || theta_base <= 0.0 {
		return Err(Error::invalid_argument(format!(
			"{operation} theta base must be finite and positive"
		)));
	}
	validate_f32_same_engine(operation, &[input])?;
	let pair_count = tokens
		.checked_mul(num_heads)
		.and_then(|count| count.checked_mul(head_dim / 2))
		.ok_or_else(|| {
			Error::invalid_argument(format!("{operation} pair count overflows usize"))
		})?;
	Ok((
		shader_u32(*tokens, "token count", operation)?,
		shader_u32(num_heads, "head count", operation)?,
		shader_u32(head_dim, "head dimension", operation)?,
		shader_u32(pair_count, "rotary pair count", operation)?,
	))
}

fn rope_attributes(
	num_heads: u32,
	head_dim: u32,
	theta_base: f32,
	position_offset: u32,
) -> [OpAttribute; 4] {
	[
		OpAttribute::UnsignedInteger {
			name: "num_heads".into(),
			value: u64::from(num_heads),
		},
		OpAttribute::UnsignedInteger {
			name: "head_dim".into(),
			value: u64::from(head_dim),
		},
		OpAttribute::Float {
			name: "theta_base".into(),
			value: f64::from(theta_base),
		},
		OpAttribute::UnsignedInteger {
			name: "position_offset".into(),
			value: u64::from(position_offset),
		},
	]
}

pub(in crate::ml) fn gelu(input: &Matrix) -> Result<Matrix> {
	const OPERATION: &str = crate::core::operation::ml::GELU.name();
	validate_f32_same_engine(OPERATION, &[input])?;
	let element_count = shader_u32(input.num_elements(), "element count", OPERATION)?;
	let output = Matrix::allocate(
		input.engine_handle(),
		input.shape().to_vec(),
		input.num_elements(),
		DType::F32,
	)?;
	if element_count != 0 {
		let buffers = [
			BufferBinding::read(input.storage()),
			BufferBinding::write(output.storage()),
		];
		let push_constants = [PushConstant::U32(element_count)];
		let kernel = KernelId::MlGeluF32;
		record_semantic(
			&[input],
			&[&output],
			&[],
			kernel,
			&buffers,
			&push_constants,
			kernel.linear_workgroups(element_count),
		)?;
	}
	Ok(output)
}

pub(in crate::ml) fn gelu_backward(input: &Matrix, output_gradient: &Matrix) -> Result<Matrix> {
	const OPERATION: &str = crate::core::operation::ml::GELU_BACKWARD.name();
	validate_f32_same_engine(OPERATION, &[input, output_gradient])?;
	if input.shape() != output_gradient.shape() {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires matching input and gradient shapes"
		)));
	}
	let element_count = shader_u32(input.num_elements(), "element count", OPERATION)?;
	let gradient = Matrix::allocate(
		input.engine_handle(),
		input.shape().to_vec(),
		input.num_elements(),
		DType::F32,
	)?;
	if element_count != 0 {
		let buffers = [
			BufferBinding::read(input.storage()),
			BufferBinding::read(output_gradient.storage()),
			BufferBinding::write(gradient.storage()),
		];
		let push_constants = [PushConstant::U32(element_count)];
		let kernel = KernelId::MlGeluBackwardF32;
		record_semantic(
			&[input, output_gradient],
			&[&gradient],
			&[],
			kernel,
			&buffers,
			&push_constants,
			kernel.linear_workgroups(element_count),
		)?;
	}
	Ok(gradient)
}

pub(in crate::ml) fn silu(input: &Matrix) -> Result<Matrix> {
	unary_activation(input, KernelId::MlSiluF32, None)
}

pub(in crate::ml) fn silu_backward(input: &Matrix, output_gradient: &Matrix) -> Result<Matrix> {
	activation_backward(input, output_gradient, KernelId::MlSiluBackwardF32, None)
}

pub(in crate::ml) fn relu(input: &Matrix) -> Result<Matrix> {
	unary_activation(input, KernelId::MlReluF32, None)
}

pub(in crate::ml) fn relu_backward(
	saved_output: &Matrix,
	output_gradient: &Matrix,
) -> Result<Matrix> {
	activation_backward(
		saved_output,
		output_gradient,
		KernelId::MlReluBackwardF32,
		None,
	)
}

pub(in crate::ml) fn tanh(input: &Matrix) -> Result<Matrix> {
	unary_activation(input, KernelId::MlTanhF32, None)
}

pub(in crate::ml) fn tanh_backward(
	saved_output: &Matrix,
	output_gradient: &Matrix,
) -> Result<Matrix> {
	activation_backward(
		saved_output,
		output_gradient,
		KernelId::MlTanhBackwardF32,
		None,
	)
}

pub(in crate::ml) fn sigmoid(input: &Matrix) -> Result<Matrix> {
	unary_activation(input, KernelId::MlSigmoidF32, None)
}

pub(in crate::ml) fn sigmoid_backward(
	saved_output: &Matrix,
	output_gradient: &Matrix,
) -> Result<Matrix> {
	activation_backward(
		saved_output,
		output_gradient,
		KernelId::MlSigmoidBackwardF32,
		None,
	)
}

pub(in crate::ml) fn leaky_relu(input: &Matrix, alpha: f32) -> Result<Matrix> {
	unary_activation(input, KernelId::MlLeakyReluF32, Some(alpha))
}

pub(in crate::ml) fn leaky_relu_backward(
	input: &Matrix,
	output_gradient: &Matrix,
	alpha: f32,
) -> Result<Matrix> {
	activation_backward(
		input,
		output_gradient,
		KernelId::MlLeakyReluBackwardF32,
		Some(alpha),
	)
}

pub(in crate::ml) fn elu(input: &Matrix, alpha: f32) -> Result<Matrix> {
	unary_activation(input, KernelId::MlEluF32, Some(alpha))
}

pub(in crate::ml) fn elu_backward(
	saved_output: &Matrix,
	output_gradient: &Matrix,
	alpha: f32,
) -> Result<Matrix> {
	activation_backward(
		saved_output,
		output_gradient,
		KernelId::MlEluBackwardF32,
		Some(alpha),
	)
}

pub(in crate::ml) fn mish(input: &Matrix) -> Result<Matrix> {
	unary_activation(input, KernelId::MlMishF32, None)
}

pub(in crate::ml) fn mish_backward(input: &Matrix, output_gradient: &Matrix) -> Result<Matrix> {
	activation_backward(input, output_gradient, KernelId::MlMishBackwardF32, None)
}

pub(in crate::ml) fn softplus(input: &Matrix) -> Result<Matrix> {
	unary_activation(input, KernelId::MlSoftplusF32, None)
}

pub(in crate::ml) fn softplus_backward(
	saved_output: &Matrix,
	output_gradient: &Matrix,
) -> Result<Matrix> {
	activation_backward(
		saved_output,
		output_gradient,
		KernelId::MlSoftplusBackwardF32,
		None,
	)
}

fn unary_activation(input: &Matrix, kernel: KernelId, scalar: Option<f32>) -> Result<Matrix> {
	let operation = kernel
		.semantic_contract()
		.ok_or_else(|| Error::internal("activation kernel has no semantic contract"))?
		.name();
	validate_f32_same_engine(operation, &[input])?;
	let element_count = shader_u32(input.num_elements(), "element count", operation)?;
	let output = Matrix::allocate(
		input.engine_handle(),
		input.shape().to_vec(),
		input.num_elements(),
		DType::F32,
	)?;
	if element_count != 0 {
		let buffers = [
			BufferBinding::read(input.storage()),
			BufferBinding::write(output.storage()),
		];
		let mut push_constants = vec![PushConstant::U32(element_count)];
		let attributes = scalar.map(|value| OpAttribute::Float {
			name: "alpha".into(),
			value: f64::from(value),
		});
		if let Some(value) = scalar {
			push_constants.push(PushConstant::F32(value));
		}
		record_semantic(
			&[input],
			&[&output],
			attributes.as_slice(),
			kernel,
			&buffers,
			&push_constants,
			kernel.linear_workgroups(element_count),
		)?;
	}
	Ok(output)
}

fn activation_backward(
	saved: &Matrix,
	output_gradient: &Matrix,
	kernel: KernelId,
	scalar: Option<f32>,
) -> Result<Matrix> {
	let operation = kernel
		.semantic_contract()
		.ok_or_else(|| Error::internal("activation adjoint kernel has no semantic contract"))?
		.name();
	validate_f32_same_engine(operation, &[saved, output_gradient])?;
	if saved.shape() != output_gradient.shape() {
		return Err(Error::invalid_argument(format!(
			"{operation} requires matching saved-state and output-gradient shapes"
		)));
	}
	let element_count = shader_u32(saved.num_elements(), "element count", operation)?;
	let gradient = Matrix::allocate(
		saved.engine_handle(),
		saved.shape().to_vec(),
		saved.num_elements(),
		DType::F32,
	)?;
	if element_count != 0 {
		let buffers = [
			BufferBinding::read(saved.storage()),
			BufferBinding::read(output_gradient.storage()),
			BufferBinding::write(gradient.storage()),
		];
		let mut push_constants = vec![PushConstant::U32(element_count)];
		let attributes = scalar.map(|value| OpAttribute::Float {
			name: "alpha".into(),
			value: f64::from(value),
		});
		if let Some(value) = scalar {
			push_constants.push(PushConstant::F32(value));
		}
		record_semantic(
			&[saved, output_gradient],
			&[&gradient],
			attributes.as_slice(),
			kernel,
			&buffers,
			&push_constants,
			kernel.linear_workgroups(element_count),
		)?;
	}
	Ok(gradient)
}

pub(in crate::ml) fn swiglu(gate: &Matrix, up: &Matrix) -> Result<Matrix> {
	const OPERATION: &str = crate::core::operation::ml::SWIGLU.name();
	validate_f32_same_engine(OPERATION, &[gate, up])?;
	if gate.shape() != up.shape() {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires equal gate and up shapes"
		)));
	}
	let element_count = shader_u32(gate.num_elements(), "element count", OPERATION)?;
	let output = Matrix::allocate(
		gate.engine_handle(),
		gate.shape().to_vec(),
		gate.num_elements(),
		DType::F32,
	)?;
	if element_count != 0 {
		let buffers = [
			BufferBinding::read(gate.storage()),
			BufferBinding::read(up.storage()),
			BufferBinding::write(output.storage()),
		];
		let push_constants = [PushConstant::U32(element_count)];
		let kernel = KernelId::MlSwigluF32;
		record_semantic(
			&[gate, up],
			&[&output],
			&[],
			kernel,
			&buffers,
			&push_constants,
			kernel.linear_workgroups(element_count),
		)?;
	}
	Ok(output)
}

pub(in crate::ml) fn swiglu_backward(
	gate: &Matrix,
	up: &Matrix,
	output_gradient: &Matrix,
) -> Result<(Matrix, Matrix)> {
	const OPERATION: &str = crate::core::operation::ml::SWIGLU_BACKWARD.name();
	validate_f32_same_engine(OPERATION, &[gate, up, output_gradient])?;
	if gate.shape() != up.shape() || gate.shape() != output_gradient.shape() {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires equal gate, up, and gradient shapes"
		)));
	}
	let element_count = shader_u32(gate.num_elements(), "element count", OPERATION)?;
	let gate_gradient = Matrix::allocate(
		gate.engine_handle(),
		gate.shape().to_vec(),
		gate.num_elements(),
		DType::F32,
	)?;
	let up_gradient = Matrix::allocate(
		gate.engine_handle(),
		gate.shape().to_vec(),
		gate.num_elements(),
		DType::F32,
	)?;
	if element_count != 0 {
		let buffers = [
			BufferBinding::read(gate.storage()),
			BufferBinding::read(up.storage()),
			BufferBinding::read(output_gradient.storage()),
			BufferBinding::write(gate_gradient.storage()),
			BufferBinding::write(up_gradient.storage()),
		];
		let push_constants = [PushConstant::U32(element_count)];
		let kernel = KernelId::MlSwigluBackwardF32;
		record_semantic(
			&[gate, up, output_gradient],
			&[&gate_gradient, &up_gradient],
			&[],
			kernel,
			&buffers,
			&push_constants,
			kernel.linear_workgroups(element_count),
		)?;
	}
	Ok((gate_gradient, up_gradient))
}

pub(in crate::ml) fn silu_mul(input: &Matrix, intermediate_size: usize) -> Result<Matrix> {
	const OPERATION: &str = crate::core::operation::ml::SILU_MUL.name();
	validate_f32_same_engine(OPERATION, &[input])?;
	let Some(last_extent) = input.shape().last() else {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} input must have at least one dimension"
		)));
	};
	let concatenated = intermediate_size.checked_mul(2).ok_or_else(|| {
		Error::invalid_argument(format!("{OPERATION} intermediate size overflows usize"))
	})?;
	if intermediate_size == 0 || *last_extent != concatenated {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires final input extent 2 * intermediate_size"
		)));
	}
	let output_count = input.num_elements() / 2;
	let output_count_u32 = shader_u32(output_count, "output element count", OPERATION)?;
	let intermediate_u32 = shader_u32(intermediate_size, "intermediate size", OPERATION)?;
	let mut output_shape = input.shape().to_vec();
	*output_shape.last_mut().expect("validated nonempty shape") = intermediate_size;
	let output = Matrix::allocate(
		input.engine_handle(),
		output_shape,
		output_count,
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::write(output.storage()),
	];
	let push_constants = [
		PushConstant::U32(output_count_u32),
		PushConstant::U32(intermediate_u32),
	];
	let attributes = [OpAttribute::SignedInteger {
		name: "intermediate_size".into(),
		value: i64::try_from(intermediate_size).map_err(|_| {
			Error::invalid_argument(format!("{OPERATION} intermediate size exceeds i64"))
		})?,
	}];
	let kernel = KernelId::MlSiluMulF32;
	record_semantic(
		&[input],
		&[&output],
		&attributes,
		kernel,
		&buffers,
		&push_constants,
		kernel.linear_workgroups(output_count_u32),
	)?;
	Ok(output)
}

pub(in crate::ml) fn silu_mul_backward(
	input: &Matrix,
	output_gradient: &Matrix,
	intermediate_size: usize,
) -> Result<Matrix> {
	const OPERATION: &str = crate::core::operation::ml::SILU_MUL_BACKWARD.name();
	validate_f32_same_engine(OPERATION, &[input, output_gradient])?;
	let Some(last_extent) = input.shape().last() else {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} input must have at least one dimension"
		)));
	};
	let concatenated = intermediate_size.checked_mul(2).ok_or_else(|| {
		Error::invalid_argument(format!("{OPERATION} intermediate size overflows usize"))
	})?;
	let mut expected_gradient_shape = input.shape().to_vec();
	if intermediate_size == 0 || *last_extent != concatenated {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires final input extent 2 * intermediate_size"
		)));
	}
	*expected_gradient_shape
		.last_mut()
		.expect("validated nonempty shape") = intermediate_size;
	if output_gradient.shape() != expected_gradient_shape {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} output gradient shape must equal the half-width output shape"
		)));
	}
	let output_count = output_gradient.num_elements();
	let output_count_u32 = shader_u32(output_count, "output element count", OPERATION)?;
	let intermediate_u32 = shader_u32(intermediate_size, "intermediate size", OPERATION)?;
	let input_gradient = Matrix::allocate(
		input.engine_handle(),
		input.shape().to_vec(),
		input.num_elements(),
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::write(input_gradient.storage()),
	];
	let push_constants = [
		PushConstant::U32(output_count_u32),
		PushConstant::U32(intermediate_u32),
	];
	let attributes = [OpAttribute::SignedInteger {
		name: "intermediate_size".into(),
		value: i64::try_from(intermediate_size).map_err(|_| {
			Error::invalid_argument(format!("{OPERATION} intermediate size exceeds i64"))
		})?,
	}];
	let kernel = KernelId::MlSiluMulBackwardF32;
	record_semantic(
		&[input, output_gradient],
		&[&input_gradient],
		&attributes,
		kernel,
		&buffers,
		&push_constants,
		kernel.linear_workgroups(output_count_u32),
	)?;
	Ok(input_gradient)
}

pub(in crate::ml) fn embedding(weight: &Matrix, indices: &Matrix) -> Result<Matrix> {
	const OPERATION: &str = crate::core::operation::ml::EMBEDDING.name();
	let [num_embeddings, embedding_dim] = weight.shape() else {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} weight must have rank two"
		)));
	};
	if *num_embeddings == 0 || *embedding_dim == 0 || weight.dtype() != DType::F32 {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires a nonempty FP32 weight [V, D]"
		)));
	}
	if indices.dtype() != DType::U32 {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires U32 indices; found {}",
			indices.dtype().token()
		)));
	}
	if !weight.engine_handle().same_as(indices.engine_handle()) {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} inputs must belong to the same engine"
		)));
	}
	let output_count = indices
		.num_elements()
		.checked_mul(*embedding_dim)
		.ok_or_else(|| {
			Error::invalid_argument(format!("{OPERATION} output size overflows usize"))
		})?;
	let mut output_shape = indices.shape().to_vec();
	output_shape.push(*embedding_dim);
	let output = Matrix::allocate(
		weight.engine_handle(),
		output_shape,
		output_count,
		DType::F32,
	)?;
	let output_count = shader_u32(output_count, "output element count", OPERATION)?;
	let index_count = shader_u32(indices.num_elements(), "index count", OPERATION)?;
	let num_embeddings = shader_u32(*num_embeddings, "vocabulary size", OPERATION)?;
	let embedding_dim = shader_u32(*embedding_dim, "embedding dimension", OPERATION)?;
	if output_count != 0 {
		let buffers = [
			BufferBinding::read(weight.storage()),
			BufferBinding::read(indices.storage()),
			BufferBinding::write(output.storage()),
		];
		let push_constants = [
			PushConstant::U32(index_count),
			PushConstant::U32(num_embeddings),
			PushConstant::U32(embedding_dim),
		];
		let kernel = KernelId::MlEmbeddingF32;
		record_semantic(
			&[weight, indices],
			&[&output],
			&[],
			kernel,
			&buffers,
			&push_constants,
			kernel.linear_workgroups(output_count),
		)?;
	}
	Ok(output)
}

pub(in crate::ml) fn embedding_backward(
	indices: &Matrix,
	output_gradient: &Matrix,
	weight: &Matrix,
) -> Result<Matrix> {
	const OPERATION: &str = crate::core::operation::ml::EMBEDDING_BACKWARD.name();
	let [num_embeddings, embedding_dim] = weight.shape() else {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} weight must have rank two"
		)));
	};
	let mut expected_gradient_shape = indices.shape().to_vec();
	expected_gradient_shape.push(*embedding_dim);
	if indices.dtype() != DType::U32
		|| weight.dtype() != DType::F32
		|| output_gradient.dtype() != DType::F32
		|| output_gradient.shape() != expected_gradient_shape
		|| !indices.engine_handle().same_as(weight.engine_handle())
		|| !indices
			.engine_handle()
			.same_as(output_gradient.engine_handle())
	{
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires same-engine U32 indices, FP32 gradient [..., D], and FP32 weight [V, D]"
		)));
	}
	let gradient = Matrix::allocate(
		weight.engine_handle(),
		weight.shape().to_vec(),
		weight.num_elements(),
		DType::F32,
	)?;
	let weight_count = shader_u32(weight.num_elements(), "weight element count", OPERATION)?;
	let index_count = shader_u32(indices.num_elements(), "index count", OPERATION)?;
	let num_embeddings = shader_u32(*num_embeddings, "vocabulary size", OPERATION)?;
	let embedding_dim = shader_u32(*embedding_dim, "embedding dimension", OPERATION)?;
	if weight_count != 0 {
		let buffers = [
			BufferBinding::read(indices.storage()),
			BufferBinding::read(output_gradient.storage()),
			BufferBinding::write(gradient.storage()),
		];
		let push_constants = [
			PushConstant::U32(index_count),
			PushConstant::U32(num_embeddings),
			PushConstant::U32(embedding_dim),
		];
		let kernel = KernelId::MlEmbeddingBackwardF32;
		record_semantic(
			&[indices, output_gradient, weight],
			&[&gradient],
			&[],
			kernel,
			&buffers,
			&push_constants,
			kernel.linear_workgroups(weight_count),
		)?;
	}
	Ok(gradient)
}

pub(in crate::ml) struct BatchNorm2dOutput {
	pub(in crate::ml) output: Matrix,
	pub(in crate::ml) mean: Matrix,
	pub(in crate::ml) variance: Matrix,
}

#[derive(Clone, Copy)]
struct BatchNorm2dGeometry {
	batch_size: u32,
	channels: u32,
	height: u32,
	width: u32,
	element_count: u32,
}

impl BatchNorm2dGeometry {
	fn resolve(input: &Matrix, operation: &'static str) -> Result<Self> {
		let [batch_size, channels, height, width] = input.shape() else {
			return Err(Error::invalid_argument(format!(
				"{operation} input must have NCHW rank four; found {:?}",
				input.shape()
			)));
		};
		if [*batch_size, *channels, *height, *width]
			.into_iter()
			.any(|extent| extent == 0)
		{
			return Err(Error::invalid_argument(format!(
				"{operation} input extents must be nonzero"
			)));
		}
		let batch_size = shader_u32(*batch_size, "batch size", operation)?;
		let channels = shader_u32(*channels, "channel count", operation)?;
		let height = shader_u32(*height, "height", operation)?;
		let width = shader_u32(*width, "width", operation)?;
		let element_count = shader_u32(input.num_elements(), "element count", operation)?;
		batch_size.checked_mul(channels).ok_or_else(|| {
			Error::invalid_argument(format!("{operation} batch-channel count exceeds u32"))
		})?;
		Ok(Self {
			batch_size,
			channels,
			height,
			width,
			element_count,
		})
	}

	const fn dimensions(self) -> [PushConstant; 4] {
		[
			PushConstant::U32(self.batch_size),
			PushConstant::U32(self.channels),
			PushConstant::U32(self.height),
			PushConstant::U32(self.width),
		]
	}

	const fn normalization_workgroups(self) -> [u32; 3] {
		[
			self.width.div_ceil(16),
			self.height.div_ceil(16),
			self.batch_size * self.channels,
		]
	}
}

fn validate_batch_norm_affine(
	input: &Matrix,
	weight: &Matrix,
	bias: &Matrix,
	epsilon: f32,
	operation: &'static str,
) -> Result<BatchNorm2dGeometry> {
	let geometry = BatchNorm2dGeometry::resolve(input, operation)?;
	if weight.shape() != [geometry.channels as usize]
		|| bias.shape() != [geometry.channels as usize]
	{
		return Err(Error::invalid_argument(format!(
			"{operation} requires weight and bias [{}]; found {:?} and {:?}",
			geometry.channels,
			weight.shape(),
			bias.shape()
		)));
	}
	if !epsilon.is_finite() || epsilon <= 0.0 {
		return Err(Error::invalid_argument(format!(
			"{operation} epsilon must be finite and positive"
		)));
	}
	validate_f32_same_engine(operation, &[input, weight, bias])?;
	Ok(geometry)
}

fn batch_norm_attributes(epsilon: f32) -> [OpAttribute; 1] {
	[OpAttribute::Float {
		name: "epsilon".into(),
		value: f64::from(epsilon),
	}]
}

pub(in crate::ml) fn batch_norm_2d(
	input: &Matrix,
	weight: &Matrix,
	bias: &Matrix,
	epsilon: f32,
) -> Result<BatchNorm2dOutput> {
	let contract = crate::core::operation::ml::BATCH_NORM_2D;
	let operation = contract.name();
	let geometry = validate_batch_norm_affine(input, weight, bias, epsilon, operation)?;
	let output = Matrix::allocate(
		input.engine_handle(),
		input.shape().to_vec(),
		input.num_elements(),
		DType::F32,
	)?;
	let channel_shape = vec![geometry.channels as usize];
	let mean = Matrix::allocate(
		input.engine_handle(),
		channel_shape.clone(),
		geometry.channels as usize,
		DType::F32,
	)?;
	let variance = Matrix::allocate(
		input.engine_handle(),
		channel_shape,
		geometry.channels as usize,
		DType::F32,
	)?;
	let stats_buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::write(mean.storage()),
		BufferBinding::write(variance.storage()),
	];
	let stats_push_constants = geometry.dimensions();
	let normalize_buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::read(mean.storage()),
		BufferBinding::read(variance.storage()),
		BufferBinding::read(weight.storage()),
		BufferBinding::read(bias.storage()),
		BufferBinding::write(output.storage()),
	];
	let normalize_push_constants = [
		PushConstant::U32(geometry.batch_size),
		PushConstant::U32(geometry.channels),
		PushConstant::U32(geometry.height),
		PushConstant::U32(geometry.width),
		PushConstant::F32(epsilon),
	];
	let dispatches = [
		ComputeDispatch {
			kernel: KernelId::MlBatchNorm2dF32,
			buffers: &stats_buffers,
			push_constants: &stats_push_constants,
			workgroups: [geometry.channels, 1, 1],
		},
		ComputeDispatch {
			kernel: KernelId::MlBatchNorm2dWithStatsF32,
			buffers: &normalize_buffers,
			push_constants: &normalize_push_constants,
			workgroups: geometry.normalization_workgroups(),
		},
	];
	let inputs = [input, weight, bias];
	let outputs = [&output, &mean, &variance];
	let attributes = batch_norm_attributes(epsilon);
	input.engine_handle().record_split_semantic(
		&dispatches,
		SemanticDispatch {
			contract,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok(BatchNorm2dOutput {
		output,
		mean,
		variance,
	})
}

pub(in crate::ml) fn batch_norm_2d_with_stats(
	input: &Matrix,
	mean: &Matrix,
	variance: &Matrix,
	weight: &Matrix,
	bias: &Matrix,
	epsilon: f32,
) -> Result<Matrix> {
	let contract = crate::core::operation::ml::BATCH_NORM_2D_WITH_STATS;
	let operation = contract.name();
	let geometry = validate_batch_norm_affine(input, weight, bias, epsilon, operation)?;
	if mean.shape() != [geometry.channels as usize]
		|| variance.shape() != [geometry.channels as usize]
	{
		return Err(Error::invalid_argument(format!(
			"{operation} requires mean and variance [{}]; found {:?} and {:?}",
			geometry.channels,
			mean.shape(),
			variance.shape()
		)));
	}
	validate_f32_same_engine(operation, &[input, mean, variance, weight, bias])?;
	let output = Matrix::allocate(
		input.engine_handle(),
		input.shape().to_vec(),
		input.num_elements(),
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::read(mean.storage()),
		BufferBinding::read(variance.storage()),
		BufferBinding::read(weight.storage()),
		BufferBinding::read(bias.storage()),
		BufferBinding::write(output.storage()),
	];
	let push_constants = [
		PushConstant::U32(geometry.batch_size),
		PushConstant::U32(geometry.channels),
		PushConstant::U32(geometry.height),
		PushConstant::U32(geometry.width),
		PushConstant::F32(epsilon),
	];
	let attributes = batch_norm_attributes(epsilon);
	record_semantic(
		&[input, mean, variance, weight, bias],
		&[&output],
		&attributes,
		KernelId::MlBatchNorm2dWithStatsF32,
		&buffers,
		&push_constants,
		geometry.normalization_workgroups(),
	)?;
	Ok(output)
}

pub(in crate::ml) fn batch_norm_2d_running_update(
	running_mean: &Matrix,
	running_variance: &Matrix,
	batch_mean: &Matrix,
	batch_variance: &Matrix,
	momentum: f32,
) -> Result<(Matrix, Matrix)> {
	let contract = crate::core::operation::ml::BATCH_NORM_2D_RUNNING_UPDATE;
	let operation = contract.name();
	let [channels] = running_mean.shape() else {
		return Err(Error::invalid_argument(format!(
			"{operation} running mean must be rank one"
		)));
	};
	if *channels == 0
		|| running_variance.shape() != [*channels]
		|| batch_mean.shape() != [*channels]
		|| batch_variance.shape() != [*channels]
	{
		return Err(Error::invalid_argument(format!(
			"{operation} requires four equal nonempty channel vectors"
		)));
	}
	if !momentum.is_finite() || !(0.0..=1.0).contains(&momentum) {
		return Err(Error::invalid_argument(format!(
			"{operation} momentum must be finite and in [0, 1]"
		)));
	}
	validate_f32_same_engine(
		operation,
		&[running_mean, running_variance, batch_mean, batch_variance],
	)?;
	let channels = shader_u32(*channels, "channel count", operation)?;
	let updated_mean = Matrix::allocate(
		running_mean.engine_handle(),
		running_mean.shape().to_vec(),
		running_mean.num_elements(),
		DType::F32,
	)?;
	let updated_variance = Matrix::allocate(
		running_mean.engine_handle(),
		running_mean.shape().to_vec(),
		running_mean.num_elements(),
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(running_mean.storage()),
		BufferBinding::read(running_variance.storage()),
		BufferBinding::read(batch_mean.storage()),
		BufferBinding::read(batch_variance.storage()),
		BufferBinding::write(updated_mean.storage()),
		BufferBinding::write(updated_variance.storage()),
	];
	let push_constants = [PushConstant::U32(channels), PushConstant::F32(momentum)];
	let attributes = [OpAttribute::Float {
		name: "momentum".into(),
		value: f64::from(momentum),
	}];
	record_semantic(
		&[running_mean, running_variance, batch_mean, batch_variance],
		&[&updated_mean, &updated_variance],
		&attributes,
		KernelId::MlBatchNorm2dRunningUpdateF32,
		&buffers,
		&push_constants,
		KernelId::MlBatchNorm2dRunningUpdateF32.linear_workgroups(channels),
	)?;
	Ok((updated_mean, updated_variance))
}

#[allow(
	clippy::too_many_arguments,
	reason = "the adjoint consumes the exact saved BatchNorm state explicitly"
)]
pub(in crate::ml) fn batch_norm_2d_backward(
	input: &Matrix,
	weight: &Matrix,
	mean: &Matrix,
	variance: &Matrix,
	output_gradient: &Matrix,
	epsilon: f32,
	training: bool,
) -> Result<(Matrix, Matrix, Matrix)> {
	let contract = crate::core::operation::ml::BATCH_NORM_2D_BACKWARD;
	let operation = contract.name();
	let geometry = BatchNorm2dGeometry::resolve(input, operation)?;
	if weight.shape() != [geometry.channels as usize]
		|| mean.shape() != [geometry.channels as usize]
		|| variance.shape() != [geometry.channels as usize]
		|| output_gradient.shape() != input.shape()
	{
		return Err(Error::invalid_argument(format!(
			"{operation} requires input/output gradient NCHW and matching channel vectors"
		)));
	}
	if !epsilon.is_finite() || epsilon <= 0.0 {
		return Err(Error::invalid_argument(format!(
			"{operation} epsilon must be finite and positive"
		)));
	}
	validate_f32_same_engine(operation, &[input, weight, mean, variance, output_gradient])?;
	let input_gradient = Matrix::allocate(
		input.engine_handle(),
		input.shape().to_vec(),
		input.num_elements(),
		DType::F32,
	)?;
	let channel_shape = vec![geometry.channels as usize];
	let weight_gradient = Matrix::allocate(
		input.engine_handle(),
		channel_shape.clone(),
		geometry.channels as usize,
		DType::F32,
	)?;
	let bias_gradient = Matrix::allocate(
		input.engine_handle(),
		channel_shape,
		geometry.channels as usize,
		DType::F32,
	)?;
	let stats_buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::read(mean.storage()),
		BufferBinding::read(variance.storage()),
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::write(weight_gradient.storage()),
		BufferBinding::write(bias_gradient.storage()),
	];
	let stats_push_constants = [
		PushConstant::U32(geometry.batch_size),
		PushConstant::U32(geometry.channels),
		PushConstant::U32(geometry.height),
		PushConstant::U32(geometry.width),
		PushConstant::F32(epsilon),
	];
	let input_buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::read(mean.storage()),
		BufferBinding::read(variance.storage()),
		BufferBinding::read(weight.storage()),
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::read(weight_gradient.storage()),
		BufferBinding::read(bias_gradient.storage()),
		BufferBinding::write(input_gradient.storage()),
	];
	let input_push_constants = [
		PushConstant::U32(geometry.element_count),
		PushConstant::U32(geometry.channels),
		PushConstant::U32(geometry.height),
		PushConstant::U32(geometry.width),
		PushConstant::F32(epsilon),
		PushConstant::U32(if training { 1 } else { 0 }),
	];
	let dispatches = [
		ComputeDispatch {
			kernel: KernelId::MlBatchNorm2dBackwardF32,
			buffers: &stats_buffers,
			push_constants: &stats_push_constants,
			workgroups: [geometry.channels, 1, 1],
		},
		ComputeDispatch {
			kernel: KernelId::MlBatchNorm2dInputBackwardF32,
			buffers: &input_buffers,
			push_constants: &input_push_constants,
			workgroups: KernelId::MlBatchNorm2dInputBackwardF32
				.linear_workgroups(geometry.element_count),
		},
	];
	let inputs = [input, weight, mean, variance, output_gradient];
	let outputs = [&input_gradient, &weight_gradient, &bias_gradient];
	let attributes = [
		OpAttribute::Float {
			name: "epsilon".into(),
			value: f64::from(epsilon),
		},
		OpAttribute::Boolean {
			name: "training".into(),
			value: training,
		},
	];
	input.engine_handle().record_split_semantic(
		&dispatches,
		SemanticDispatch {
			contract,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok((input_gradient, weight_gradient, bias_gradient))
}

fn validate_layer_norm_inputs(
	input: &Matrix,
	weight: &Matrix,
	bias: &Matrix,
	epsilon: f32,
	operation: &'static str,
) -> Result<(u32, u32)> {
	let Some(&columns) = input.shape().last() else {
		return Err(Error::invalid_argument(format!(
			"{operation} input must have at least one dimension"
		)));
	};
	if columns == 0 || input.num_elements() == 0 {
		return Err(Error::invalid_argument(format!(
			"{operation} input dimensions must be nonzero"
		)));
	}
	if weight.shape() != [columns] || bias.shape() != [columns] {
		return Err(Error::invalid_argument(format!(
			"{operation} requires weight and bias [{columns}]; found {:?} and {:?}",
			weight.shape(),
			bias.shape()
		)));
	}
	if !epsilon.is_finite() || epsilon <= 0.0 {
		return Err(Error::invalid_argument(format!(
			"{operation} epsilon must be finite and positive"
		)));
	}
	validate_f32_same_engine(operation, &[input, weight, bias])?;
	let rows = input.num_elements() / columns;
	shader_u32(input.num_elements(), "element count", operation)?;
	Ok((
		shader_u32(rows, "row count", operation)?,
		shader_u32(columns, "normalized dimension", operation)?,
	))
}
