//! Linear Matrix operation implementations.

use crate::OpAttribute;
use crate::ml::validation::{shader_u32, validate_f32_same_engine};
use crate::runtime::ComputeDispatch;
use crate::runtime::SemanticDispatch;
use crate::{
	DType, Error, Matrix, Result,
	runtime::{BufferBinding, KernelId, PushConstant},
};

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
		.ok_or_else(|| Error::invalid_argument(format!("{OPERATION} leading size overflows usize")))?;
	let output_count = batch
		.checked_mul(*output_features)
		.ok_or_else(|| Error::invalid_argument(format!("{OPERATION} output size overflows usize")))?;
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
		{
			let inputs: &[&Matrix] = &[input, weight, bias];
			let outputs: &[&Matrix] = &[&output];
			let attributes: &[OpAttribute] = &[];
			let dispatch = ComputeDispatch {
				kernel,
				buffers: &buffers,
				push_constants: &push_constants,
				workgroups: kernel.output_workgroups(batch, output_features),
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
		.ok_or_else(|| Error::invalid_argument(format!("{OPERATION} leading size overflows usize")))?;
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
		{
			let inputs: &[&Matrix] = &[weight, output_gradient];
			let outputs: &[&Matrix] = &[&input_gradient];
			let attributes: &[OpAttribute] = &[];
			let dispatch = ComputeDispatch {
				kernel: input_kernel,
				buffers: &input_buffers,
				push_constants: &dimensions,
				workgroups: input_kernel.output_workgroups(batch_u32, input_features_u32),
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
	let parameter_buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::write(weight_gradient.storage()),
		BufferBinding::write(bias_gradient.storage()),
	];
	{
		let inputs: &[&Matrix] = &[input, output_gradient];
		let outputs: &[&Matrix] = &[&weight_gradient, &bias_gradient];
		let attributes: &[OpAttribute] = &[];
		let dispatch = ComputeDispatch {
			kernel: KernelId::MlLinearParameterBackwardF32,
			buffers: &parameter_buffers,
			push_constants: &dimensions,
			workgroups: [output_features_u32, input_features_u32.div_ceil(32), 1],
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
	Ok((input_gradient, weight_gradient, bias_gradient))
}
