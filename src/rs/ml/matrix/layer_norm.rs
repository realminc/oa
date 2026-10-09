//! Layer norm Matrix operation implementations.

use crate::ml::validation::{shader_u32, validate_f32_same_engine};
use crate::{
	DType, Error, Matrix, OpAttribute, Result,
	runtime::{BufferBinding, ComputeDispatch, KernelId, PushConstant, SemanticDispatch},
};

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
	{
		let inputs: &[&Matrix] = &[input, weight, bias];
		let outputs: &[&Matrix] = &[&output, &normalized, &inverse_stddev];
		let attributes: &[OpAttribute] = &attributes;
		let dispatch = ComputeDispatch {
			kernel: KernelId::MlLayerNormF32,
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
