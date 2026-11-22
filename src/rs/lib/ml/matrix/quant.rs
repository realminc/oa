//! Native symmetric Q4/Q8 block-plane weight operations.
//!
//! [`QuantMatrix`] is deliberately not a [`crate::DType`]: one logical value
//! owns a packed integer payload and a separate Float32 scale plane, so matrix
//! byte sizes, strides, views, and element access of a dense dtype would be
//! incorrect. The physical planes remain private and travel as one weight
//! value.
//!
//! Donor: `oacpp/source/cpp/lib/oa/ml/fnmatrix/quant/fnMatrixQuant.cpp`.

use crate::{
	DType, Error, Matrix, OpAttribute, Result,
	ml::validation::shader_u32,
	runtime::{BufferBinding, ComputeDispatch, KernelId, PushConstant, SemanticDispatch},
};

use crate::runtime::{QuantSemanticDispatch, QuantSemanticInput, QuantSemanticOutput};

/// Number of logical values in one quantization block.
const BLOCK_SIZE: usize = 32;
/// Q4 payload bytes per block: 16 packed bytes (4 words).
const Q4_BYTES_PER_BLOCK: usize = 16;
/// Q8 payload bytes per block: 32 signed bytes (8 words).
const Q8_BYTES_PER_BLOCK: usize = 32;
/// Q4 packed 32-bit words per block.
const Q4_WORDS_PER_BLOCK: usize = 4;
/// Q8 packed 32-bit words per block.
const Q8_WORDS_PER_BLOCK: usize = 8;
/// Float32 kernels address element offsets as `u32`; the byte offset must fit.
const MAX_F32_ELEMENT_COUNT: usize = u32::MAX as usize / 4;

/// OA native block quantization selector.
///
/// Both formats share one 32-value block layout and one Float32 scale per
/// block. `Q4` stores a biased 4-bit index per value (16 bytes per block);
/// `Q8` stores one signed byte per value (32 bytes per block).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Quantization {
	/// Symmetric 4-bit quantization, range `[-7, 7]` biased by +7.
	Q4,
	/// Symmetric 8-bit quantization, range `[-127, 127]`.
	Q8,
}

impl Quantization {
	const fn payload_bytes_per_block(self) -> usize {
		match self {
			Self::Q4 => Q4_BYTES_PER_BLOCK,
			Self::Q8 => Q8_BYTES_PER_BLOCK,
		}
	}

	const fn as_str(self) -> &'static str {
		match self {
			Self::Q4 => "Q4",
			Self::Q8 => "Q8",
		}
	}
}

/// A GPU-resident weight stored in OA's native Q4 or Q8 block encoding.
///
/// Constructed by [`quantize`]; the packed payload and scale planes are
/// private implementation detail and never exposed as public matrix values.
#[derive(Clone)]
pub struct QuantMatrix {
	payload: Matrix,
	scale: Matrix,
	shape: Vec<usize>,
	quantization: Quantization,
}

impl QuantMatrix {
	/// Return the logical (dequantized) weight shape.
	pub fn shape(&self) -> &[usize] {
		&self.shape
	}

	/// Return the number of logical weight elements.
	pub fn num_elements(&self) -> usize {
		self.shape.iter().product()
	}

	/// Return the block quantization format.
	pub const fn quantization(&self) -> Quantization {
		self.quantization
	}

	/// Construct one weight value from checked private planes.
	///
	/// The payload plane stores `blocks * bytes_per_block` `U8` bytes and the
	/// scale plane stores one Float32 per 32-value block. `Quantization::Q8`
	/// uses `U8` storage because packed planes never expose element semantics;
	/// the signed byte interpretation lives inside the kernels. This is the
	/// private construction bridge used by lowering and exact-plane tests, not
	/// a public construction contract — use [`quantize`].
	#[doc(hidden)]
	pub fn from_planes(
		payload: Matrix,
		scale: Matrix,
		shape: Vec<usize>,
		quantization: Quantization,
	) -> Result<Self> {
		if payload.dtype() != DType::U8 || scale.dtype() != DType::F32 {
			return Err(Error::invalid_argument(
				"quantized planes must use U8 payload and Float32 scale storage",
			));
		}
		if payload.shape().len() != 1 || scale.shape().len() != 1 {
			return Err(Error::invalid_argument(
				"payload and scale planes must both be one-dimensional",
			));
		}
		if !payload.engine_handle().same_as(scale.engine_handle()) {
			return Err(Error::invalid_argument(
				"quantized planes must belong to one engine",
			));
		}
		let elements = shape
			.iter()
			.try_fold(1_usize, |total, extent| total.checked_mul(*extent))
			.ok_or_else(|| Error::invalid_argument("quantized logical shape overflows usize"))?;
		let blocks = elements.div_ceil(BLOCK_SIZE);
		let payload_bytes = blocks
			.checked_mul(quantization.payload_bytes_per_block())
			.ok_or_else(|| Error::invalid_argument("quantized payload size overflows usize"))?;
		if payload.num_elements() != payload_bytes || scale.num_elements() != blocks {
			return Err(Error::invalid_argument(
				"payload/scale planes do not match the logical shape",
			));
		}
		Ok(Self {
			payload,
			scale,
			shape,
			quantization,
		})
	}

	/// Private test/lowering bridge to the packed payload plane.
	///
	/// The payload is `U8` storage whose packed bit layout is private; lowering
	/// code and exact-plane oracles read bytes only. This is not a public value
	/// accessor — the planes travel inside one [`QuantMatrix`].
	#[doc(hidden)]
	pub const fn payload(&self) -> &Matrix {
		&self.payload
	}

	/// Private test/lowering bridge to the Float32 scale plane.
	#[doc(hidden)]
	pub const fn scale(&self) -> &Matrix {
		&self.scale
	}

	pub(crate) fn value_id(&self) -> u64 {
		self.payload.value_id()
	}

	pub(crate) const fn engine_handle(&self) -> &crate::runtime::EngineHandle {
		self.payload.engine_handle()
	}
}

/// Quantize Float32 storage into one native OA Q4 or Q8 weight value.
///
/// The returned value retains the source shape; its physical planes remain
/// private.
///
/// # Errors
///
/// Returns an error unless the input is a non-empty Float32 matrix whose
/// element byte offsets fit the `u32` kernel ABI, or allocation/recording
/// fails.
pub fn quantize(input: &Matrix, quantization: Quantization) -> Result<QuantMatrix> {
	let operation = "oa::FnMatrix::quantize";
	let count = input.num_elements();
	if count == 0 {
		return Err(Error::invalid_argument(format!(
			"{operation} input must have non-empty Float32 storage"
		)));
	}
	if input.dtype() != DType::F32 {
		return Err(Error::invalid_argument(format!(
			"{operation} only Float32 source storage is admitted"
		)));
	}
	admit_count(count, operation)?;

	let engine = input.engine_handle();
	let lowering = engine.begin_semantic_lowering()?;
	let scale = compute_scale(input, quantization)?;
	let payload = quantize_planes(input, &scale, quantization)?;
	let output = QuantMatrix::from_planes(payload, scale, input.shape().to_vec(), quantization)?;
	lowering.commit_quant(QuantSemanticDispatch {
		contract: crate::core::operation::ml::QUANTIZE,
		inputs: &[QuantSemanticInput::Matrix(input)],
		outputs: &[QuantSemanticOutput::QuantMatrix(&output)],
		attributes: &[quantization_attribute(quantization)],
	})?;
	Ok(output)
}

/// Materialize a Float32 matrix with the logical shape retained by `input`.
///
/// # Errors
///
/// Returns an error for an empty weight, oversized element counts, or
/// allocation/recording failure.
pub fn dequantize(input: &QuantMatrix) -> Result<Matrix> {
	let operation = "oa::FnMatrix::dequantize";
	let count = input.num_elements();
	if count == 0 {
		return Err(Error::invalid_argument(format!(
			"{operation} input must contain a non-empty quantized weight"
		)));
	}
	admit_count(count, operation)?;

	let engine = input.engine_handle();
	let lowering = engine.begin_semantic_lowering()?;
	let flat = dequantize_planes(input.payload(), input.scale(), count, input.quantization())?;
	let output = flat.reshape_semantic_output(input.shape().to_vec())?;
	lowering.commit_quant(QuantSemanticDispatch {
		contract: crate::core::operation::ml::DEQUANTIZE,
		inputs: &[QuantSemanticInput::QuantMatrix(input)],
		outputs: &[QuantSemanticOutput::Matrix(&output)],
		attributes: &[quantization_attribute(input.quantization())],
	})?;
	Ok(output)
}

/// Multiply `[..., K]` Float32 activations by a quantized `[N, K]` weight
/// without materializing a Float32 weight matrix.
///
/// The result shape is `[..., N]`.
///
/// # Errors
///
/// Returns an error unless the weight has rank 2, the input has rank at least
/// 2 with a matching last dimension, all planes are contiguous base storage on
/// one engine, the flattened dimensions fit the `u32` kernel ABI, or
/// allocation/recording fails.
pub fn mat_mul_nt(input: &Matrix, weight: &QuantMatrix) -> Result<Matrix> {
	let operation = "oa::FnMatrix::matMulNtQuantized";
	if weight.shape().len() != 2 {
		return Err(Error::invalid_argument(format!(
			"{operation} quantized weight must have rank 2 [N,K]"
		)));
	}
	if input.shape().len() < 2 || *input.shape().last().unwrap_or(&0) != weight.shape()[1] {
		return Err(Error::invalid_argument(format!(
			"{operation} input last dimension must match quantized weight K"
		)));
	}
	if !input.engine_handle().same_as(weight.engine_handle()) {
		return Err(Error::invalid_argument(format!(
			"{operation} input and quantized weight must belong to one engine"
		)));
	}
	let output_features = weight.shape()[0];
	let engine = input.engine_handle();
	let lowering = engine.begin_semantic_lowering()?;
	let output = mat_mul_nt_planes(
		input,
		weight.payload(),
		weight.scale(),
		output_features,
		weight.quantization(),
	)?;
	lowering.commit_quant(QuantSemanticDispatch {
		contract: crate::core::operation::ml::MAT_MUL_NT_QUANTIZED,
		inputs: &[
			QuantSemanticInput::Matrix(input),
			QuantSemanticInput::QuantMatrix(weight),
		],
		outputs: &[QuantSemanticOutput::Matrix(&output)],
		attributes: &[quantization_attribute(weight.quantization())],
	})?;
	Ok(output)
}

fn quantization_attribute(quantization: Quantization) -> OpAttribute {
	OpAttribute::Enum {
		name: "quantization".into(),
		value: quantization.as_str().into(),
	}
}

fn admit_count(count: usize, operation: &str) -> Result<()> {
	if count > MAX_F32_ELEMENT_COUNT {
		return Err(Error::invalid_argument(format!(
			"{operation} Float32 byte offsets must fit the u32 kernel ABI"
		)));
	}
	Ok(())
}

fn compute_scale(input: &Matrix, quantization: Quantization) -> Result<Matrix> {
	let (contract, kernel) = match quantization {
		Quantization::Q4 => (
			crate::core::operation::ml::COMPUTE_SCALE_Q4,
			KernelId::MlComputeScaleQ4F32,
		),
		Quantization::Q8 => (
			crate::core::operation::ml::COMPUTE_SCALE_Q8,
			KernelId::MlComputeScaleQ8F32,
		),
	};
	let operation = contract.name();
	let count = input.num_elements();
	admit_count(count, operation)?;
	if input.dtype() != DType::F32 {
		return Err(Error::invalid_argument(format!(
			"{operation} input must use Float32 storage"
		)));
	}
	let count_u32 = shader_u32(count, "element count", operation)?;
	let blocks = count.div_ceil(BLOCK_SIZE);
	let blocks_u32 = shader_u32(blocks, "block count", operation)?;
	let scale = Matrix::allocate(input.engine_handle(), vec![blocks], blocks, DType::F32)?;
	if count_u32 == 0 {
		return Ok(scale);
	}
	let buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::write(scale.storage()),
	];
	let push_constants = [PushConstant::U32(count_u32)];
	input.engine_handle().record_semantic(
		ComputeDispatch {
			kernel,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: kernel.linear_workgroups(blocks_u32),
			indirect: None,
		},
		SemanticDispatch {
			contract,
			inputs: &[input],
			outputs: &[&scale],
			attributes: &[],
		},
	)?;
	Ok(scale)
}

fn quantize_planes(input: &Matrix, scale: &Matrix, quantization: Quantization) -> Result<Matrix> {
	let (contract, kernel, bytes_per_block, words_per_block) = match quantization {
		Quantization::Q4 => (
			crate::core::operation::ml::QUANTIZE_Q4,
			KernelId::MlQuantizeQ4F32,
			Q4_BYTES_PER_BLOCK,
			Q4_WORDS_PER_BLOCK,
		),
		Quantization::Q8 => (
			crate::core::operation::ml::QUANTIZE_Q8,
			KernelId::MlQuantizeQ8F32,
			Q8_BYTES_PER_BLOCK,
			Q8_WORDS_PER_BLOCK,
		),
	};
	let operation = contract.name();
	let count = input.num_elements();
	admit_count(count, operation)?;
	if input.dtype() != DType::F32 || scale.dtype() != DType::F32 {
		return Err(Error::invalid_argument(format!(
			"{operation} input and scale must use Float32 storage"
		)));
	}
	let count_u32 = shader_u32(count, "element count", operation)?;
	let blocks = count.div_ceil(BLOCK_SIZE);
	if scale.num_elements() != blocks {
		return Err(Error::invalid_argument(format!(
			"{operation} expected one scale per 32-value block"
		)));
	}
	let payload_bytes = blocks
		.checked_mul(bytes_per_block)
		.ok_or_else(|| Error::invalid_argument(format!("{operation} payload size overflows usize")))?;
	let payload_words = blocks
		.checked_mul(words_per_block)
		.ok_or_else(|| Error::invalid_argument(format!("{operation} payload size overflows usize")))?;
	let payload = Matrix::allocate(
		input.engine_handle(),
		vec![payload_bytes],
		payload_bytes,
		DType::U8,
	)?;
	if count_u32 == 0 {
		return Ok(payload);
	}
	let buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::write(payload.storage()),
		BufferBinding::read(scale.storage()),
	];
	let push_constants = [PushConstant::U32(count_u32)];
	input.engine_handle().record_semantic(
		ComputeDispatch {
			kernel,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: kernel.linear_workgroups(shader_u32(
				payload_words,
				"payload word count",
				operation,
			)?),
			indirect: None,
		},
		SemanticDispatch {
			contract,
			inputs: &[input, scale],
			outputs: &[&payload],
			attributes: &[],
		},
	)?;
	Ok(payload)
}

fn dequantize_planes(
	payload: &Matrix,
	scale: &Matrix,
	count: usize,
	quantization: Quantization,
) -> Result<Matrix> {
	let (contract, kernel, bytes_per_block) = match quantization {
		Quantization::Q4 => (
			crate::core::operation::ml::DEQUANTIZE_Q4,
			KernelId::MlDequantizeQ4F32,
			Q4_BYTES_PER_BLOCK,
		),
		Quantization::Q8 => (
			crate::core::operation::ml::DEQUANTIZE_Q8,
			KernelId::MlDequantizeQ8F32,
			Q8_BYTES_PER_BLOCK,
		),
	};
	let operation = contract.name();
	admit_count(count, operation)?;
	if payload.dtype() != DType::U8 || scale.dtype() != DType::F32 {
		return Err(Error::invalid_argument(format!(
			"{operation} payload must be U8 and scale must be Float32"
		)));
	}
	let count_u32 = shader_u32(count, "element count", operation)?;
	let blocks = count.div_ceil(BLOCK_SIZE);
	let payload_bytes = blocks
		.checked_mul(bytes_per_block)
		.ok_or_else(|| Error::invalid_argument(format!("{operation} payload size overflows usize")))?;
	if scale.num_elements() != blocks || payload.num_elements() != payload_bytes {
		return Err(Error::invalid_argument(format!(
			"{operation} payload/scale shape does not match element count"
		)));
	}
	let output = Matrix::allocate(payload.engine_handle(), vec![count], count, DType::F32)?;
	if count_u32 == 0 {
		return Ok(output);
	}
	let buffers = [
		BufferBinding::read(payload.storage()),
		BufferBinding::write(output.storage()),
		BufferBinding::read(scale.storage()),
	];
	let push_constants = [PushConstant::U32(count_u32)];
	payload.engine_handle().record_semantic(
		ComputeDispatch {
			kernel,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: kernel.linear_workgroups(count_u32),
			indirect: None,
		},
		SemanticDispatch {
			contract,
			inputs: &[payload, scale],
			outputs: &[&output],
			attributes: &[],
		},
	)?;
	Ok(output)
}

fn mat_mul_nt_planes(
	input: &Matrix,
	payload: &Matrix,
	scale: &Matrix,
	output_features: usize,
	quantization: Quantization,
) -> Result<Matrix> {
	let (contract, kernel, bytes_per_block) = match quantization {
		Quantization::Q4 => (
			crate::core::operation::ml::MAT_MUL_NT_Q4,
			KernelId::MlMatMulNtQ4F32,
			Q4_BYTES_PER_BLOCK,
		),
		Quantization::Q8 => (
			crate::core::operation::ml::MAT_MUL_NT_Q8,
			KernelId::MlMatMulNtQ8F32,
			Q8_BYTES_PER_BLOCK,
		),
	};
	let operation = contract.name();
	if input.shape().len() < 2 {
		return Err(Error::invalid_argument(format!(
			"{operation} input must have rank >= 2 with K in the last dimension"
		)));
	}
	if payload.shape().len() != 1 || scale.shape().len() != 1 {
		return Err(Error::invalid_argument(format!(
			"{operation} payload and scale planes must both be one-dimensional"
		)));
	}
	if input.dtype() != DType::F32 || payload.dtype() != DType::U8 || scale.dtype() != DType::F32 {
		return Err(Error::invalid_argument(format!(
			"{operation} expected Float32 input, U8 payload, and Float32 scale"
		)));
	}
	if input.semantic().view_byte_offset() != 0
		|| payload.semantic().view_byte_offset() != 0
		|| scale.semantic().view_byte_offset() != 0
	{
		return Err(Error::invalid_argument(format!(
			"{operation} input and quantization planes must be contiguous base views"
		)));
	}
	let output_features_u32 = shader_u32(output_features, "output feature count", operation)?;
	let input_features = *input.shape().last().unwrap_or(&0);
	if input_features == 0 {
		return Err(Error::invalid_argument(format!(
			"{operation} the input K dimension must be positive and fit u32"
		)));
	}
	let k_u32 = shader_u32(input_features, "input K dimension", operation)?;
	let rows = input
		.shape()
		.iter()
		.take(input.shape().len() - 1)
		.try_fold(1_usize, |total, extent| total.checked_mul(*extent))
		.ok_or_else(|| {
			Error::invalid_argument(format!(
				"{operation} flattened input row count overflows usize"
			))
		})?;
	let rows_u32 = shader_u32(rows, "flattened input row count", operation)?;

	let weight_count = output_features
		.checked_mul(input_features)
		.ok_or_else(|| Error::invalid_argument(format!("{operation} weight count overflows usize")))?;
	let input_count = rows
		.checked_mul(input_features)
		.ok_or_else(|| Error::invalid_argument(format!("{operation} input count overflows usize")))?;
	let output_count = rows
		.checked_mul(output_features)
		.ok_or_else(|| Error::invalid_argument(format!("{operation} output count overflows usize")))?;
	if weight_count > MAX_F32_ELEMENT_COUNT
		|| input_count > MAX_F32_ELEMENT_COUNT
		|| output_count > MAX_F32_ELEMENT_COUNT
	{
		return Err(Error::invalid_argument(format!(
			"{operation} matrix byte offsets exceed the u32 kernel ABI"
		)));
	}
	let blocks = weight_count.div_ceil(BLOCK_SIZE);
	let payload_bytes = blocks
		.checked_mul(bytes_per_block)
		.ok_or_else(|| Error::invalid_argument(format!("{operation} payload size overflows usize")))?;
	if payload.num_elements() != payload_bytes || scale.num_elements() != blocks {
		return Err(Error::invalid_argument(format!(
			"{operation} payload/scale shape does not match [OutputFeatures,K]"
		)));
	}

	let mut output_shape = input.shape().to_vec();
	*output_shape.last_mut().unwrap() = output_features;
	let output = Matrix::allocate(
		input.engine_handle(),
		output_shape,
		output_count,
		DType::F32,
	)?;
	if rows == 0 || output_features == 0 {
		return Ok(output);
	}
	let buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::read(payload.storage()),
		BufferBinding::read(scale.storage()),
		BufferBinding::write(output.storage()),
	];
	let push_constants = [
		PushConstant::U32(rows_u32),
		PushConstant::U32(output_features_u32),
		PushConstant::U32(k_u32),
	];
	input.engine_handle().record_semantic(
		ComputeDispatch {
			kernel,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: [output_features_u32, rows_u32, 1],
			indirect: None,
		},
		SemanticDispatch {
			contract,
			inputs: &[input, payload, scale],
			outputs: &[&output],
			attributes: &[],
		},
	)?;
	Ok(output)
}
