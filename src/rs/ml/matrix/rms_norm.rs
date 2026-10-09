//! Rms norm Matrix operation implementations.

use crate::ml::autograd;
use crate::ml::validation::{shader_u32, validate_f32_same_engine};
use crate::{
	DType, Error, Matrix, OpAttribute, Result,
	runtime::{
		BufferBinding, ComputeDispatch, KernelId, OptionalSemanticDispatch, PushConstant,
		SemanticDispatch,
	},
};

/// Complete adjoints of broadcast-affine gated RMS normalization.
pub struct RmsNormGatedBackward {
	/// Gradient of the normalized input.
	pub input: Matrix,
	/// Gradient of the broadcast affine weight.
	pub weight: Matrix,
	/// Gradient of the optional affine bias.
	pub bias: Option<Matrix>,
	/// Gradient of the SiLU gate input.
	pub gate: Matrix,
}
/// Apply weighted root-mean-square normalization over the final dimension.
///
/// # Errors
///
/// Returns an error unless `input` is a nonempty F32 Matrix, `weight` is a
/// same-engine F32 vector matching its final dimension, `epsilon` is finite and
/// positive, or runtime recording fails.
pub fn rms_norm(input: &Matrix, weight: &Matrix, epsilon: f32) -> Result<Matrix> {
	let output = rms_norm_forward(input, weight, epsilon)?;
	autograd::record_rms_norm(input, &output, None, weight.clone(), epsilon)?;
	Ok(output)
}

/// Apply per-row RMS normalization, a cyclic broadcast affine, and a SiLU gate.
///
/// Normalization is over the final input dimension. The affine value may have
/// one or more leading groups; those groups repeat over the flattened input
/// rows. An optional bias must exactly match `weight`.
///
/// # Errors
///
/// Returns an error for incompatible shape, dtype, Engine ownership or
/// epsilon, or when runtime recording fails.
pub fn rms_norm_gated(
	input: &Matrix,
	weight: &Matrix,
	bias: Option<&Matrix>,
	gate: &Matrix,
	epsilon: f32,
) -> Result<Matrix> {
	let output = rms_norm_gated_forward(input, weight, bias, gate, epsilon)?;
	autograd::record_rms_norm_gated(
		input,
		&output,
		None,
		weight.clone(),
		None,
		bias.cloned(),
		gate,
		epsilon,
	)?;
	Ok(output)
}
pub(in crate::ml) fn rms_norm_forward(
	input: &Matrix,
	weight: &Matrix,
	epsilon: f32,
) -> Result<Matrix> {
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
	{
		let inputs: &[&Matrix] = &[input, weight];
		let outputs: &[&Matrix] = &[&output];
		let attributes: &[OpAttribute] = &attributes;
		let dispatch = ComputeDispatch {
			kernel: KernelId::MlRmsNormF32,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: [rows, 1, 1],
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
pub(in crate::ml) fn rms_norm_gated_forward(
	input: &Matrix,
	weight: &Matrix,
	bias: Option<&Matrix>,
	gate: &Matrix,
	epsilon: f32,
) -> Result<Matrix> {
	const OPERATION: &str = crate::core::operation::ml::RMS_NORM_GATED.name();
	let (rows, columns, groups, _) =
		validate_rms_norm_gated_inputs(input, weight, bias, gate, epsilon, OPERATION)?;
	let output = Matrix::allocate(
		input.engine_handle(),
		input.shape().to_vec(),
		input.num_elements(),
		DType::F32,
	)?;
	let physical_bias = bias.unwrap_or(weight);
	let buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::read(weight.storage()),
		BufferBinding::read(physical_bias.storage()),
		BufferBinding::read(gate.storage()),
		BufferBinding::write(output.storage()),
	];
	let push_constants = [
		PushConstant::U32(rows),
		PushConstant::U32(columns),
		PushConstant::U32(groups),
		PushConstant::F32(epsilon),
		PushConstant::U32(u32::from(bias.is_some())),
	];
	let attributes = [
		OpAttribute::Float {
			name: "epsilon".into(),
			value: f64::from(epsilon),
		},
		OpAttribute::Boolean {
			name: "has_bias".into(),
			value: bias.is_some(),
		},
	];
	let semantic_inputs = [Some(input), Some(weight), bias, Some(gate)];
	input.engine_handle().record_optional_semantic(
		ComputeDispatch {
			kernel: KernelId::MlRmsNormGatedF32,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: [rows, 1, 1],
		},
		OptionalSemanticDispatch {
			contract: crate::core::operation::ml::RMS_NORM_GATED,
			inputs: &semantic_inputs,
			outputs: &[&output],
			attributes: &attributes,
		},
	)?;
	Ok(output)
}

/// Compute all gated RMS normalization adjoints explicitly.
///
/// # Errors
///
/// Returns an error for an incompatible output gradient or forward contract,
/// or when runtime recording fails.
pub fn rms_norm_gated_backward(
	input: &Matrix,
	weight: &Matrix,
	bias: Option<&Matrix>,
	gate: &Matrix,
	output_gradient: &Matrix,
	epsilon: f32,
) -> Result<RmsNormGatedBackward> {
	const OPERATION: &str = crate::core::operation::ml::RMS_NORM_GATED_BACKWARD.name();
	let (rows, columns, groups, outer_rows) =
		validate_rms_norm_gated_inputs(input, weight, bias, gate, epsilon, OPERATION)?;
	validate_f32_same_engine(OPERATION, &[input, weight, gate, output_gradient])?;
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
	let gate_gradient = Matrix::allocate(
		input.engine_handle(),
		gate.shape().to_vec(),
		gate.num_elements(),
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
	let bias_contribution = if bias.is_some() {
		Matrix::allocate(
			input.engine_handle(),
			input.shape().to_vec(),
			input.num_elements(),
			DType::F32,
		)?
	} else {
		Matrix::from_slice_handle(input.engine_handle(), vec![1], &[0.0_f32])?
	};
	let bias_gradient = match bias {
		Some(value) => Some(Matrix::allocate(
			input.engine_handle(),
			value.shape().to_vec(),
			value.num_elements(),
			DType::F32,
		)?),
		None => None,
	};
	let physical_bias = bias.unwrap_or(weight);
	let backward_buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::read(weight.storage()),
		BufferBinding::read(physical_bias.storage()),
		BufferBinding::read(gate.storage()),
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::write(input_gradient.storage()),
		BufferBinding::write(gate_gradient.storage()),
		BufferBinding::write(weight_contribution.storage()),
		BufferBinding::write(bias_contribution.storage()),
	];
	let backward_push_constants = [
		PushConstant::U32(rows),
		PushConstant::U32(columns),
		PushConstant::U32(groups),
		PushConstant::F32(epsilon),
		PushConstant::U32(u32::from(bias.is_some())),
	];
	let affine_elements = shader_u32(weight.num_elements(), "affine element count", OPERATION)?;
	let reduction_push_constants = [
		PushConstant::U32(1),
		PushConstant::U32(outer_rows),
		PushConstant::U32(affine_elements),
	];
	let weight_reduction_buffers = [
		BufferBinding::read(weight_contribution.storage()),
		BufferBinding::write(weight_gradient.storage()),
	];
	let backward_dispatch = ComputeDispatch {
		kernel: KernelId::MlRmsNormGatedBackwardF32,
		buffers: &backward_buffers,
		push_constants: &backward_push_constants,
		workgroups: [rows, 1, 1],
	};
	let weight_dispatch = ComputeDispatch {
		kernel: KernelId::MatrixSumAxisF32,
		buffers: &weight_reduction_buffers,
		push_constants: &reduction_push_constants,
		workgroups: KernelId::MatrixSumAxisF32.linear_workgroups(affine_elements),
	};
	let semantic_inputs = [
		Some(input),
		Some(weight),
		bias,
		Some(gate),
		Some(output_gradient),
	];
	let bias_placeholder = bias_gradient.as_ref().unwrap_or(&bias_contribution);
	let semantic_outputs = [
		&input_gradient,
		&weight_gradient,
		bias_placeholder,
		&gate_gradient,
	];
	let attributes = [
		OpAttribute::Float {
			name: "epsilon".into(),
			value: f64::from(epsilon),
		},
		OpAttribute::Boolean {
			name: "has_bias".into(),
			value: bias.is_some(),
		},
	];
	if let Some(bias_gradient) = &bias_gradient {
		let bias_reduction_buffers = [
			BufferBinding::read(bias_contribution.storage()),
			BufferBinding::write(bias_gradient.storage()),
		];
		let bias_dispatch = ComputeDispatch {
			kernel: KernelId::MatrixSumAxisF32,
			buffers: &bias_reduction_buffers,
			push_constants: &reduction_push_constants,
			workgroups: KernelId::MatrixSumAxisF32.linear_workgroups(affine_elements),
		};
		input.engine_handle().record_split_optional_semantic(
			&[backward_dispatch, weight_dispatch, bias_dispatch],
			OptionalSemanticDispatch {
				contract: crate::core::operation::ml::RMS_NORM_GATED_BACKWARD,
				inputs: &semantic_inputs,
				outputs: &semantic_outputs,
				attributes: &attributes,
			},
		)?;
	} else {
		input.engine_handle().record_split_optional_semantic(
			&[backward_dispatch, weight_dispatch],
			OptionalSemanticDispatch {
				contract: crate::core::operation::ml::RMS_NORM_GATED_BACKWARD,
				inputs: &semantic_inputs,
				outputs: &semantic_outputs,
				attributes: &attributes,
			},
		)?;
	}
	Ok(RmsNormGatedBackward {
		input: input_gradient,
		weight: weight_gradient,
		bias: bias_gradient,
		gate: gate_gradient,
	})
}

fn validate_rms_norm_gated_inputs(
	input: &Matrix,
	weight: &Matrix,
	bias: Option<&Matrix>,
	gate: &Matrix,
	epsilon: f32,
	operation: &'static str,
) -> Result<(u32, u32, u32, u32)> {
	let Some(&columns) = input.shape().last() else {
		return Err(Error::invalid_argument(format!(
			"{operation} input must have at least one dimension"
		)));
	};
	let affine_elements = weight.num_elements();
	if columns == 0
		|| input.num_elements() == 0
		|| gate.shape() != input.shape()
		|| affine_elements == 0
		|| !affine_elements.is_multiple_of(columns)
		|| bias.is_some_and(|value| value.shape() != weight.shape())
	{
		return Err(Error::invalid_argument(format!(
			"{operation} requires matching nonempty input/gate and optional bias matching a broadcast affine weight ending in the normalized dimension"
		)));
	}
	if !epsilon.is_finite() || epsilon <= 0.0 {
		return Err(Error::invalid_argument(format!(
			"{operation} epsilon must be finite and positive"
		)));
	}
	let rows = input.num_elements() / columns;
	let groups = affine_elements / columns;
	if groups == 0 || !rows.is_multiple_of(groups) {
		return Err(Error::invalid_argument(format!(
			"{operation} affine groups must divide the flattened row count"
		)));
	}
	let mut matrices = vec![input, weight, gate];
	if let Some(bias) = bias {
		matrices.push(bias);
	}
	validate_f32_same_engine(operation, &matrices)?;
	Ok((
		shader_u32(rows, "row count", operation)?,
		shader_u32(columns, "normalized dimension", operation)?,
		shader_u32(groups, "affine group count", operation)?,
		shader_u32(rows / groups, "outer row count", operation)?,
	))
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
