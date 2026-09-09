use crate::{
	DType, Error, Matrix, OpAttribute, OperationContract, Result,
	runtime::{BufferBinding, ComputeDispatch, KernelId, PushConstant, SemanticDispatch},
};

pub(super) fn linear(input: &Matrix, weight: &Matrix, bias: &Matrix) -> Result<Matrix> {
	const OPERATION: &str = "ml.linear";
	let [batch, input_features] = input.shape() else {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} input must have rank two; found {:?}",
			input.shape()
		)));
	};
	let [output_features, weight_features] = weight.shape() else {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} weight must have rank two; found {:?}",
			weight.shape()
		)));
	};
	if bias.shape() != [*output_features] || input_features != weight_features {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires input [B, I], weight [O, I], and bias [O]; found {:?}, {:?}, {:?}",
			input.shape(),
			weight.shape(),
			bias.shape()
		)));
	}
	validate_f32_same_engine(OPERATION, &[input, weight, bias])?;
	let output_count = batch.checked_mul(*output_features).ok_or_else(|| {
		Error::invalid_argument(format!("{OPERATION} output size overflows usize"))
	})?;
	let batch = shader_u32(*batch, "batch", OPERATION)?;
	let input_features = shader_u32(*input_features, "input feature count", OPERATION)?;
	let output_features = shader_u32(*output_features, "output feature count", OPERATION)?;
	if input_features == 0 || output_features == 0 {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} feature dimensions must be nonzero"
		)));
	}
	let output = Matrix::allocate(
		input.engine_handle(),
		vec![batch as usize, output_features as usize],
		output_count,
		DType::F32,
	)?;
	if output_count != 0 {
		let output_count = shader_u32(output_count, "output element count", OPERATION)?;
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
			crate::core::operation::ml::LINEAR,
			&[input, weight, bias],
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

pub(super) fn linear_backward(
	input: &Matrix,
	weight: &Matrix,
	output_gradient: &Matrix,
) -> Result<(Matrix, Matrix, Matrix)> {
	const OPERATION: &str = "ml.linear_backward";
	let [batch, input_features] = input.shape() else {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} input must have rank two"
		)));
	};
	let [output_features, weight_features] = weight.shape() else {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} weight must have rank two"
		)));
	};
	if input_features != weight_features || output_gradient.shape() != [*batch, *output_features] {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} shapes are incompatible: input {:?}, weight {:?}, output gradient {:?}",
			input.shape(),
			weight.shape(),
			output_gradient.shape()
		)));
	}
	validate_f32_same_engine(OPERATION, &[input, weight, output_gradient])?;
	let batch_u32 = shader_u32(*batch, "batch", OPERATION)?;
	let input_features_u32 = shader_u32(*input_features, "input feature count", OPERATION)?;
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
			crate::core::operation::ml::LINEAR_BACKWARD,
			&[weight, output_gradient],
			&[&input_gradient],
			&[],
			input_kernel,
			&input_buffers,
			&dimensions,
			input_kernel.linear_workgroups(shader_u32(
				input.num_elements(),
				"input element count",
				OPERATION,
			)?),
		)?;
	}
	let parameter_buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::write(weight_gradient.storage()),
		BufferBinding::write(bias_gradient.storage()),
	];
	record_semantic(
		crate::core::operation::ml::LINEAR_PARAMETER_BACKWARD,
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

pub(super) struct LayerNormOutput {
	pub(super) output: Matrix,
	pub(super) normalized: Matrix,
	pub(super) inverse_stddev: Matrix,
}

pub(super) fn layer_norm(
	input: &Matrix,
	weight: &Matrix,
	bias: &Matrix,
	epsilon: f32,
) -> Result<LayerNormOutput> {
	const OPERATION: &str = "ml.layer_norm";
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
		crate::core::operation::ml::LAYER_NORM,
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

pub(super) fn layer_norm_backward(
	input: &Matrix,
	weight: &Matrix,
	normalized: &Matrix,
	inverse_stddev: &Matrix,
	output_gradient: &Matrix,
) -> Result<(Matrix, Matrix, Matrix)> {
	const OPERATION: &str = "ml.layer_norm_backward";
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
	];
	let push_constants = [PushConstant::U32(rows), PushConstant::U32(columns)];
	record_semantic(
		crate::core::operation::ml::LAYER_NORM_BACKWARD,
		&[normalized, inverse_stddev, weight, output_gradient],
		&[&input_gradient],
		&[],
		KernelId::MlLayerNormBackwardF32,
		&input_buffers,
		&push_constants,
		[rows, 1, 1],
	)?;
	let parameter_buffers = [
		BufferBinding::read(normalized.storage()),
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::write(weight_gradient.storage()),
		BufferBinding::write(bias_gradient.storage()),
	];
	record_semantic(
		crate::core::operation::ml::LAYER_NORM_PARAMETER_BACKWARD,
		&[normalized, output_gradient],
		&[&weight_gradient, &bias_gradient],
		&[],
		KernelId::MlLayerNormParameterBackwardF32,
		&parameter_buffers,
		&push_constants,
		[columns.div_ceil(32), 1, 1],
	)?;
	Ok((input_gradient, weight_gradient, bias_gradient))
}

pub(super) fn gelu(input: &Matrix) -> Result<Matrix> {
	const OPERATION: &str = "ml.gelu";
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
			crate::core::operation::ml::GELU,
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

pub(super) fn gelu_backward(input: &Matrix, output_gradient: &Matrix) -> Result<Matrix> {
	const OPERATION: &str = "ml.gelu_backward";
	validate_f32_same_engine(OPERATION, &[input, output_gradient])?;
	if input.shape() != output_gradient.shape() {
		return Err(Error::invalid_argument(
			"ml.gelu_backward requires matching input and gradient shapes",
		));
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
			crate::core::operation::ml::GELU_BACKWARD,
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

pub(super) fn swiglu(gate: &Matrix, up: &Matrix) -> Result<Matrix> {
	const OPERATION: &str = "ml.swiglu";
	validate_f32_same_engine(OPERATION, &[gate, up])?;
	if gate.shape() != up.shape() {
		return Err(Error::invalid_argument(
			"ml.swiglu requires equal gate and up shapes",
		));
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
			crate::core::operation::ml::SWIGLU,
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

pub(super) fn swiglu_backward(
	gate: &Matrix,
	up: &Matrix,
	output_gradient: &Matrix,
) -> Result<(Matrix, Matrix)> {
	const OPERATION: &str = "ml.swiglu_backward";
	validate_f32_same_engine(OPERATION, &[gate, up, output_gradient])?;
	if gate.shape() != up.shape() || gate.shape() != output_gradient.shape() {
		return Err(Error::invalid_argument(
			"ml.swiglu_backward requires equal gate, up, and gradient shapes",
		));
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
			crate::core::operation::ml::SWIGLU_BACKWARD,
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

pub(super) struct AttentionOutput {
	pub(super) output: Matrix,
	pub(super) probabilities: Matrix,
}

pub(super) fn scaled_dot_product_attention_causal(
	query: &Matrix,
	key: &Matrix,
	value: &Matrix,
	sequence_length: usize,
	num_heads: usize,
) -> Result<AttentionOutput> {
	const OPERATION: &str = "ml.scaled_dot_product_attention_causal";
	let (batch, sequence, width, heads, _) =
		validate_attention(query, key, value, sequence_length, num_heads, OPERATION)?;
	let output = Matrix::allocate(
		query.engine_handle(),
		query.shape().to_vec(),
		query.num_elements(),
		DType::F32,
	)?;
	let probability_count = (batch as usize)
		.checked_mul(heads as usize)
		.and_then(|count| count.checked_mul(sequence as usize))
		.and_then(|count| count.checked_mul(sequence as usize))
		.ok_or_else(|| {
			Error::invalid_argument(format!("{OPERATION} saved state overflows usize"))
		})?;
	shader_u32(
		probability_count,
		"saved probability element count",
		OPERATION,
	)?;
	let probabilities = Matrix::allocate(
		query.engine_handle(),
		vec![
			batch as usize,
			heads as usize,
			sequence as usize,
			sequence as usize,
		],
		probability_count,
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(query.storage()),
		BufferBinding::read(key.storage()),
		BufferBinding::read(value.storage()),
		BufferBinding::write(output.storage()),
		BufferBinding::write(probabilities.storage()),
	];
	let push_constants = [
		PushConstant::U32(batch),
		PushConstant::U32(sequence),
		PushConstant::U32(width),
		PushConstant::U32(heads),
	];
	let kernel = KernelId::MlScaledDotProductAttentionCausalF32;
	let attributes = semantic_attention_attributes(sequence_length, num_heads)?;
	record_semantic(
		crate::core::operation::ml::SCALED_DOT_PRODUCT_ATTENTION_CAUSAL,
		&[query, key, value],
		&[&output, &probabilities],
		&attributes,
		kernel,
		&buffers,
		&push_constants,
		[batch * sequence * heads, 1, 1],
	)?;
	Ok(AttentionOutput {
		output,
		probabilities,
	})
}

pub(super) fn scaled_dot_product_attention_causal_backward(
	query: &Matrix,
	key: &Matrix,
	value: &Matrix,
	probabilities: &Matrix,
	output_gradient: &Matrix,
	sequence_length: usize,
	num_heads: usize,
) -> Result<(Matrix, Matrix, Matrix)> {
	const OPERATION: &str = "ml.scaled_dot_product_attention_causal_backward";
	let (batch, sequence, width, heads, element_count) =
		validate_attention(query, key, value, sequence_length, num_heads, OPERATION)?;
	validate_f32_same_engine(OPERATION, &[query, probabilities, output_gradient])?;
	if output_gradient.shape() != query.shape() {
		return Err(Error::invalid_argument(
			"causal attention output gradient must match Q/K/V shape",
		));
	}
	let probability_shape = [
		batch as usize,
		heads as usize,
		sequence as usize,
		sequence as usize,
	];
	if probabilities.shape() != probability_shape {
		return Err(Error::invalid_argument(format!(
			"causal attention saved probabilities must have shape {probability_shape:?}; found {:?}",
			probabilities.shape()
		)));
	}
	let score_gradient = Matrix::allocate(
		query.engine_handle(),
		probabilities.shape().to_vec(),
		probabilities.num_elements(),
		DType::F32,
	)?;
	let probability_buffers = [
		BufferBinding::read(probabilities.storage()),
		BufferBinding::read(value.storage()),
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::write(score_gradient.storage()),
	];
	let dimensions = [
		PushConstant::U32(batch),
		PushConstant::U32(sequence),
		PushConstant::U32(width),
		PushConstant::U32(heads),
	];
	let attributes = semantic_attention_attributes(sequence_length, num_heads)?;
	record_semantic(
		crate::core::operation::ml::SCALED_DOT_PRODUCT_ATTENTION_CAUSAL_PROBABILITY_BACKWARD,
		&[probabilities, value, output_gradient],
		&[&score_gradient],
		&attributes,
		KernelId::MlScaledDotProductAttentionCausalProbabilityBackwardF32,
		&probability_buffers,
		&dimensions,
		[batch * sequence * heads, 1, 1],
	)?;
	let allocate = || {
		Matrix::allocate(
			query.engine_handle(),
			query.shape().to_vec(),
			query.num_elements(),
			DType::F32,
		)
	};
	let query_gradient = allocate()?;
	let key_gradient = allocate()?;
	let value_gradient = allocate()?;
	let buffers = [
		BufferBinding::read(query.storage()),
		BufferBinding::read(key.storage()),
		BufferBinding::read(probabilities.storage()),
		BufferBinding::read(score_gradient.storage()),
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::write(query_gradient.storage()),
		BufferBinding::write(key_gradient.storage()),
		BufferBinding::write(value_gradient.storage()),
	];
	let kernel = KernelId::MlScaledDotProductAttentionCausalBackwardF32;
	record_semantic(
		crate::core::operation::ml::SCALED_DOT_PRODUCT_ATTENTION_CAUSAL_BACKWARD,
		&[query, key, probabilities, &score_gradient, output_gradient],
		&[&query_gradient, &key_gradient, &value_gradient],
		&attributes,
		kernel,
		&buffers,
		&dimensions,
		kernel.linear_workgroups(element_count),
	)?;
	Ok((query_gradient, key_gradient, value_gradient))
}

pub(super) fn embedding(weight: &Matrix, indices: &Matrix) -> Result<Matrix> {
	const OPERATION: &str = "ml.embedding";
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
			crate::core::operation::ml::EMBEDDING,
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

pub(super) fn embedding_backward(
	indices: &Matrix,
	output_gradient: &Matrix,
	weight: &Matrix,
) -> Result<Matrix> {
	const OPERATION: &str = "ml.embedding_backward";
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
			crate::core::operation::ml::EMBEDDING_BACKWARD,
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

pub(super) struct RnnOutput {
	pub(super) output: Matrix,
	pub(super) hidden_previous: Matrix,
}

pub(super) fn rnn(
	input: &Matrix,
	weight_ih: &Matrix,
	weight_hh: &Matrix,
	bias_ih: &Matrix,
	bias_hh: &Matrix,
) -> Result<RnnOutput> {
	const OPERATION: &str = "ml.rnn";
	const MAX_HIDDEN_SIZE: usize = 1024;
	let [batch, sequence_length, input_size] = input.shape() else {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} input must have shape [B, S, I]"
		)));
	};
	let [hidden_size, weight_input_size] = weight_ih.shape() else {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} input weight must have shape [H, I]"
		)));
	};
	if *batch == 0
		|| *sequence_length == 0
		|| *input_size == 0
		|| *hidden_size == 0
		|| *hidden_size > MAX_HIDDEN_SIZE
		|| input_size != weight_input_size
		|| weight_hh.shape() != [*hidden_size, *hidden_size]
		|| bias_ih.shape() != [*hidden_size]
		|| bias_hh.shape() != [*hidden_size]
	{
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires nonempty input [B, S, I], weights [H, I]/[H, H], biases [H], and H <= {MAX_HIDDEN_SIZE}"
		)));
	}
	validate_f32_same_engine(OPERATION, &[input, weight_ih, weight_hh, bias_ih, bias_hh])?;
	let output_count = batch
		.checked_mul(*sequence_length)
		.and_then(|count| count.checked_mul(*hidden_size))
		.ok_or_else(|| {
			Error::invalid_argument(format!("{OPERATION} output size overflows usize"))
		})?;
	let output_shape = vec![*batch, *sequence_length, *hidden_size];
	let output = Matrix::allocate(
		input.engine_handle(),
		output_shape.clone(),
		output_count,
		DType::F32,
	)?;
	let hidden_previous = Matrix::allocate(
		input.engine_handle(),
		output_shape,
		output_count,
		DType::F32,
	)?;
	let batch = shader_u32(*batch, "batch", OPERATION)?;
	let sequence_length = shader_u32(*sequence_length, "sequence length", OPERATION)?;
	let input_size = shader_u32(*input_size, "input size", OPERATION)?;
	let hidden_size = shader_u32(*hidden_size, "hidden size", OPERATION)?;
	let buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::read(weight_ih.storage()),
		BufferBinding::read(weight_hh.storage()),
		BufferBinding::read(bias_ih.storage()),
		BufferBinding::read(bias_hh.storage()),
		BufferBinding::write(output.storage()),
		BufferBinding::write(hidden_previous.storage()),
	];
	let push_constants = [
		PushConstant::U32(batch),
		PushConstant::U32(sequence_length),
		PushConstant::U32(input_size),
		PushConstant::U32(hidden_size),
	];
	record_semantic(
		crate::core::operation::ml::RNN,
		&[input, weight_ih, weight_hh, bias_ih, bias_hh],
		&[&output, &hidden_previous],
		&[],
		KernelId::MlRnnF32,
		&buffers,
		&push_constants,
		[batch, 1, 1],
	)?;
	Ok(RnnOutput {
		output,
		hidden_previous,
	})
}

pub(super) struct RnnBackwardOutput {
	pub(super) input: Matrix,
	pub(super) weight_ih: Matrix,
	pub(super) weight_hh: Matrix,
	pub(super) bias_ih: Matrix,
	pub(super) bias_hh: Matrix,
}

pub(super) fn rnn_backward(
	output_gradient: &Matrix,
	input: &Matrix,
	output: &Matrix,
	hidden_previous: &Matrix,
	weight_ih: &Matrix,
	weight_hh: &Matrix,
) -> Result<RnnBackwardOutput> {
	const OPERATION: &str = "ml.rnn_backward";
	let [batch, sequence_length, input_size] = input.shape() else {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} input must have shape [B, S, I]"
		)));
	};
	let [hidden_size, weight_input_size] = weight_ih.shape() else {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} input weight must have shape [H, I]"
		)));
	};
	let output_shape = [*batch, *sequence_length, *hidden_size];
	if input_size != weight_input_size
		|| weight_hh.shape() != [*hidden_size, *hidden_size]
		|| output.shape() != output_shape
		|| hidden_previous.shape() != output_shape
		|| output_gradient.shape() != output_shape
	{
		return Err(Error::invalid_argument(format!(
			"{OPERATION} saved values and output gradient do not match the recurrent contract"
		)));
	}
	validate_f32_same_engine(
		OPERATION,
		&[
			output_gradient,
			input,
			output,
			hidden_previous,
			weight_ih,
			weight_hh,
		],
	)?;
	let input_gradient = Matrix::allocate(
		input.engine_handle(),
		input.shape().to_vec(),
		input.num_elements(),
		DType::F32,
	)?;
	let weight_ih_gradient = Matrix::allocate(
		input.engine_handle(),
		weight_ih.shape().to_vec(),
		weight_ih.num_elements(),
		DType::F32,
	)?;
	let weight_hh_gradient = Matrix::allocate(
		input.engine_handle(),
		weight_hh.shape().to_vec(),
		weight_hh.num_elements(),
		DType::F32,
	)?;
	let bias_ih_gradient = Matrix::allocate(
		input.engine_handle(),
		vec![*hidden_size],
		*hidden_size,
		DType::F32,
	)?;
	let bias_hh_gradient = Matrix::allocate(
		input.engine_handle(),
		vec![*hidden_size],
		*hidden_size,
		DType::F32,
	)?;
	let batch = shader_u32(*batch, "batch", OPERATION)?;
	let sequence_length = shader_u32(*sequence_length, "sequence length", OPERATION)?;
	let input_size = shader_u32(*input_size, "input size", OPERATION)?;
	let hidden_size = shader_u32(*hidden_size, "hidden size", OPERATION)?;
	let buffers = [
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::read(input.storage()),
		BufferBinding::read(output.storage()),
		BufferBinding::read(hidden_previous.storage()),
		BufferBinding::read(weight_ih.storage()),
		BufferBinding::read(weight_hh.storage()),
		BufferBinding::write(input_gradient.storage()),
		BufferBinding::write(weight_ih_gradient.storage()),
		BufferBinding::write(weight_hh_gradient.storage()),
		BufferBinding::write(bias_ih_gradient.storage()),
		BufferBinding::write(bias_hh_gradient.storage()),
	];
	let push_constants = [
		PushConstant::U32(batch),
		PushConstant::U32(sequence_length),
		PushConstant::U32(input_size),
		PushConstant::U32(hidden_size),
	];
	record_semantic(
		crate::core::operation::ml::RNN_BACKWARD,
		&[
			output_gradient,
			input,
			output,
			hidden_previous,
			weight_ih,
			weight_hh,
		],
		&[
			&input_gradient,
			&weight_ih_gradient,
			&weight_hh_gradient,
			&bias_ih_gradient,
			&bias_hh_gradient,
		],
		&[],
		KernelId::MlRnnBackwardF32,
		&buffers,
		&push_constants,
		[1, 1, 1],
	)?;
	Ok(RnnBackwardOutput {
		input: input_gradient,
		weight_ih: weight_ih_gradient,
		weight_hh: weight_hh_gradient,
		bias_ih: bias_ih_gradient,
		bias_hh: bias_hh_gradient,
	})
}

pub(super) fn cross_entropy(logits: &Matrix, targets: &Matrix) -> Result<Matrix> {
	const OPERATION: &str = "ml.cross_entropy";
	let (rows, classes) = validate_cross_entropy_inputs(logits, targets, OPERATION)?;
	let output = Matrix::allocate(logits.engine_handle(), Vec::new(), 1, DType::F32)?;
	let buffers = [
		BufferBinding::read(logits.storage()),
		BufferBinding::read(targets.storage()),
		BufferBinding::write(output.storage()),
	];
	let push_constants = [PushConstant::U32(rows), PushConstant::U32(classes)];
	record_semantic(
		crate::core::operation::ml::CROSS_ENTROPY,
		&[logits, targets],
		&[&output],
		&[],
		KernelId::MlCrossEntropyF32,
		&buffers,
		&push_constants,
		[1, 1, 1],
	)?;
	Ok(output)
}

pub(super) fn cross_entropy_backward(logits: &Matrix, targets: &Matrix) -> Result<Matrix> {
	const OPERATION: &str = "ml.cross_entropy_backward";
	let (rows, classes) = validate_cross_entropy_inputs(logits, targets, OPERATION)?;
	let gradient = Matrix::allocate(
		logits.engine_handle(),
		logits.shape().to_vec(),
		logits.num_elements(),
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(logits.storage()),
		BufferBinding::read(targets.storage()),
		BufferBinding::write(gradient.storage()),
	];
	let push_constants = [PushConstant::U32(rows), PushConstant::U32(classes)];
	let kernel = KernelId::MlCrossEntropyBackwardF32;
	record_semantic(
		crate::core::operation::ml::CROSS_ENTROPY_BACKWARD,
		&[logits, targets],
		&[&gradient],
		&[],
		kernel,
		&buffers,
		&push_constants,
		[rows, 1, 1],
	)?;
	Ok(gradient)
}

pub(super) struct AdamWScalars {
	pub(super) step: u32,
	pub(super) learning_rate: f32,
	pub(super) beta1: f32,
	pub(super) beta2: f32,
	pub(super) epsilon: f32,
	pub(super) weight_decay: f32,
}

pub(super) fn adamw_graph_advance(state: &Matrix) -> Result<()> {
	const OPERATION: &str = "ml.adamw_graph_advance";
	if state.dtype() != DType::U32 || state.shape() != [6] {
		return Err(Error::invalid_argument(
			"ml.adamw_graph_advance requires U32 optimizer state [6]",
		));
	}
	let buffers = [BufferBinding::read_write(state.storage())];
	state.engine_handle().record(ComputeDispatch {
		operation: OPERATION,
		kernel: KernelId::MlAdamWGraphAdvanceU32,
		buffers: &buffers,
		push_constants: &[],
		workgroups: [1, 1, 1],
	})
}

pub(super) fn adamw_graph(
	parameter: &Matrix,
	gradient: &Matrix,
	first_moment: &Matrix,
	second_moment: &Matrix,
	state: &Matrix,
) -> Result<()> {
	const OPERATION: &str = "ml.adamw_graph";
	validate_f32_same_engine(
		OPERATION,
		&[parameter, gradient, first_moment, second_moment],
	)?;
	if state.dtype() != DType::U32 || !state.engine_handle().same_as(parameter.engine_handle()) {
		return Err(Error::invalid_argument(
			"ml.adamw_graph requires same-engine U32 optimizer state",
		));
	}
	for candidate in [gradient, first_moment, second_moment] {
		if candidate.shape() != parameter.shape() {
			return Err(Error::invalid_argument(
				"ml.adamw_graph requires identical parameter, gradient, and moment shapes",
			));
		}
	}
	if state.shape() != [6] {
		return Err(Error::invalid_argument(
			"ml.adamw_graph requires optimizer state [6]",
		));
	}
	let element_count = shader_u32(parameter.num_elements(), "element count", OPERATION)?;
	if element_count == 0 {
		return Ok(());
	}
	let buffers = [
		BufferBinding::read_write(parameter.storage()),
		BufferBinding::read(gradient.storage()),
		BufferBinding::read_write(first_moment.storage()),
		BufferBinding::read_write(second_moment.storage()),
		BufferBinding::read(state.storage()),
	];
	let push_constants = [PushConstant::U32(element_count)];
	let kernel = KernelId::MlAdamWGraphF32;
	parameter.engine_handle().record(ComputeDispatch {
		operation: OPERATION,
		kernel,
		buffers: &buffers,
		push_constants: &push_constants,
		workgroups: kernel.linear_workgroups(element_count),
	})
}

pub(super) fn adamw(
	parameter: &Matrix,
	gradient: &Matrix,
	first_moment: &Matrix,
	second_moment: &Matrix,
	scalars: AdamWScalars,
) -> Result<()> {
	const OPERATION: &str = "ml.adamw";
	validate_f32_same_engine(
		OPERATION,
		&[parameter, gradient, first_moment, second_moment],
	)?;
	for candidate in [gradient, first_moment, second_moment] {
		if candidate.shape() != parameter.shape() {
			return Err(Error::invalid_argument(format!(
				"{OPERATION} requires identical parameter, gradient, and moment shapes"
			)));
		}
	}
	let element_count = shader_u32(parameter.num_elements(), "element count", OPERATION)?;
	if element_count != 0 {
		let buffers = [
			BufferBinding::read_write(parameter.storage()),
			BufferBinding::read(gradient.storage()),
			BufferBinding::read_write(first_moment.storage()),
			BufferBinding::read_write(second_moment.storage()),
		];
		let push_constants = [
			PushConstant::U32(element_count),
			PushConstant::U32(scalars.step),
			PushConstant::F32(scalars.learning_rate),
			PushConstant::F32(scalars.beta1),
			PushConstant::F32(scalars.beta2),
			PushConstant::F32(scalars.epsilon),
			PushConstant::F32(scalars.weight_decay),
		];
		let kernel = KernelId::MlAdamWF32;
		parameter.engine_handle().record(ComputeDispatch {
			operation: OPERATION,
			kernel,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: kernel.linear_workgroups(element_count),
		})?;
	}
	Ok(())
}

fn validate_cross_entropy_inputs(
	logits: &Matrix,
	targets: &Matrix,
	operation: &'static str,
) -> Result<(u32, u32)> {
	let [rows, classes] = logits.shape() else {
		return Err(Error::invalid_argument(format!(
			"{operation} logits must have rank two; found {:?}",
			logits.shape()
		)));
	};
	if *rows == 0 || *classes == 0 || targets.shape() != [*rows] {
		return Err(Error::invalid_argument(format!(
			"{operation} requires nonempty logits [N, C] and targets [N]; found {:?} and {:?}",
			logits.shape(),
			targets.shape()
		)));
	}
	if logits.dtype() != DType::F32 || targets.dtype() != DType::U32 {
		return Err(Error::invalid_argument(format!(
			"{operation} requires F32 logits and U32 targets; found {} and {}",
			logits.dtype().token(),
			targets.dtype().token()
		)));
	}
	if !logits.engine_handle().same_as(targets.engine_handle()) {
		return Err(Error::invalid_argument(format!(
			"{operation} inputs must belong to the same engine"
		)));
	}
	Ok((
		shader_u32(*rows, "row count", operation)?,
		shader_u32(*classes, "class count", operation)?,
	))
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

fn validate_attention(
	query: &Matrix,
	key: &Matrix,
	value: &Matrix,
	sequence_length: usize,
	num_heads: usize,
	operation: &'static str,
) -> Result<(u32, u32, u32, u32, u32)> {
	const MAX_SEQUENCE_LENGTH: usize = 1024;
	let [rows, model_width] = query.shape() else {
		return Err(Error::invalid_argument(format!(
			"{operation} requires Q/K/V shaped [B*S, D]"
		)));
	};
	if *rows == 0
		|| *model_width == 0
		|| sequence_length == 0
		|| sequence_length > MAX_SEQUENCE_LENGTH
		|| num_heads == 0
		|| rows % sequence_length != 0
		|| model_width % num_heads != 0
		|| key.shape() != query.shape()
		|| value.shape() != query.shape()
	{
		return Err(Error::invalid_argument(format!(
			"{operation} requires equal nonempty Q/K/V [B*S, D], S <= {MAX_SEQUENCE_LENGTH}, rows divisible by S, and D divisible by H"
		)));
	}
	validate_f32_same_engine(operation, &[query, key, value])?;
	Ok((
		shader_u32(rows / sequence_length, "batch", operation)?,
		shader_u32(sequence_length, "sequence length", operation)?,
		shader_u32(*model_width, "model width", operation)?,
		shader_u32(num_heads, "head count", operation)?,
		shader_u32(query.num_elements(), "element count", operation)?,
	))
}

#[allow(
	clippy::too_many_arguments,
	reason = "keeps semantic values and physical dispatch fields explicit at each lowering"
)]
fn record_semantic(
	contract: OperationContract,
	inputs: &[&Matrix],
	outputs: &[&Matrix],
	attributes: &[OpAttribute],
	kernel: KernelId,
	buffers: &[BufferBinding<'_>],
	push_constants: &[PushConstant],
	workgroups: [u32; 3],
) -> Result<()> {
	let engine = inputs
		.first()
		.ok_or_else(|| Error::internal("semantic ML dispatch requires an input"))?
		.engine_handle();
	engine.record_semantic(
		ComputeDispatch {
			operation: contract.name(),
			kernel,
			buffers,
			push_constants,
			workgroups,
		},
		SemanticDispatch {
			contract,
			inputs,
			outputs,
			attributes,
		},
	)
}

fn semantic_attention_attributes(
	sequence_length: usize,
	num_heads: usize,
) -> Result<[OpAttribute; 2]> {
	let sequence_length = u64::try_from(sequence_length)
		.map_err(|_| Error::out_of_range("attention sequence length exceeds u64"))?;
	let num_heads = u64::try_from(num_heads)
		.map_err(|_| Error::out_of_range("attention head count exceeds u64"))?;
	Ok([
		OpAttribute::UnsignedInteger {
			name: "sequence_length".into(),
			value: sequence_length,
		},
		OpAttribute::UnsignedInteger {
			name: "num_heads".into(),
			value: num_heads,
		},
	])
}

fn validate_f32_same_engine(operation: &'static str, matrices: &[&Matrix]) -> Result<()> {
	let Some(first) = matrices.first() else {
		return Err(Error::invalid_argument(
			"matrix validation requires an input",
		));
	};
	for matrix in matrices {
		if matrix.dtype() != DType::F32 {
			return Err(Error::invalid_argument(format!(
				"{operation} requires F32 matrices; found {}",
				matrix.dtype().token()
			)));
		}
		if !first.engine_handle().same_as(matrix.engine_handle()) {
			return Err(Error::invalid_argument(format!(
				"{operation} inputs must belong to the same engine"
			)));
		}
	}
	Ok(())
}

fn shader_u32(value: usize, label: &str, operation: &'static str) -> Result<u32> {
	u32::try_from(value)
		.map_err(|_| Error::invalid_argument(format!("{operation} {label} exceeds u32")))
}
