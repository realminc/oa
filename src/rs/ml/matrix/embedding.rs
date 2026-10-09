//! Embedding Matrix operation implementations.

use crate::OpAttribute;
use crate::ml::validation::shader_u32;
use crate::runtime::ComputeDispatch;
use crate::runtime::SemanticDispatch;
use crate::{
	DType, Error, Matrix, Result,
	runtime::{BufferBinding, KernelId, PushConstant},
};

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
	if !matches!(indices.dtype(), DType::U8 | DType::U32 | DType::I32) {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires U8, U32, or I32 indices; found {}",
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
		.ok_or_else(|| Error::invalid_argument(format!("{OPERATION} output size overflows usize")))?;
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
		let kernel = match indices.dtype() {
			DType::U8 => KernelId::MlEmbeddingU8F32,
			DType::U32 | DType::I32 => KernelId::MlEmbeddingF32,
			_ => unreachable!("validated embedding index dtype"),
		};
		{
			let inputs: &[&Matrix] = &[weight, indices];
			let outputs: &[&Matrix] = &[&output];
			let attributes: &[OpAttribute] = &[];
			let dispatch = ComputeDispatch {
				kernel,
				buffers: &buffers,
				push_constants: &push_constants,
				workgroups: kernel.linear_workgroups(output_count),
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
	if !matches!(indices.dtype(), DType::U8 | DType::U32 | DType::I32)
		|| weight.dtype() != DType::F32
		|| output_gradient.dtype() != DType::F32
		|| output_gradient.shape() != expected_gradient_shape
		|| !indices.engine_handle().same_as(weight.engine_handle())
		|| !indices
			.engine_handle()
			.same_as(output_gradient.engine_handle())
	{
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires same-engine U8, U32, or I32 indices, FP32 gradient [..., D], and FP32 weight [V, D]"
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
		let kernel = match indices.dtype() {
			DType::U8 => KernelId::MlEmbeddingBackwardU8F32,
			DType::U32 | DType::I32 => KernelId::MlEmbeddingBackwardF32,
			_ => unreachable!("validated embedding-backward index dtype"),
		};
		{
			let inputs: &[&Matrix] = &[indices, output_gradient, weight];
			let outputs: &[&Matrix] = &[&gradient];
			let attributes: &[OpAttribute] = &[];
			let dispatch = ComputeDispatch {
				kernel,
				buffers: &buffers,
				push_constants: &push_constants,
				workgroups: kernel.linear_workgroups(weight_count),
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
	Ok(gradient)
}
