//! ML-owned Matrix pooling operations.

pub use crate::ml::types::{
	AdaptiveMaxPool1dResult, AdaptiveMaxPool2dResult, AdaptiveMaxPool3dResult, MaxPool1dResult,
	MaxPool2dResult, MaxPool3dResult,
};
use crate::{
	DType, Error, Matrix, OpAttribute, Result,
	runtime::{BufferBinding, ComputeDispatch, KernelId, PushConstant, SemanticDispatch},
};

use crate::ml::autograd;

use super::super::validation::{shader_u32, validate_f32_same_runtime};

// ── geometry helpers ─────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct Pool2dGeometry {
	batch: u32,
	channels: u32,
	input_height: u32,
	input_width: u32,
	output_height: u32,
	output_width: u32,
	kernel_size: u32,
	stride: u32,
	padding: u32,
}

impl Pool2dGeometry {
	fn resolve(
		input: &Matrix,
		kernel_size: usize,
		stride: usize,
		padding: usize,
		operation: &'static str,
	) -> Result<Self> {
		let [batch, channels, input_height, input_width] = input.shape() else {
			return Err(Error::invalid_argument(format!(
				"{operation} requires NCHW rank-four input; found {:?}",
				input.shape()
			)));
		};
		if *batch == 0
			|| *channels == 0
			|| *input_height == 0
			|| *input_width == 0
			|| kernel_size == 0
			|| stride == 0
		{
			return Err(Error::invalid_argument(format!(
				"{operation} requires nonzero NCHW extents, kernel size, and stride"
			)));
		}
		shader_u32(input.num_elements(), "input element count", operation)?;
		let doubled_padding = padding.checked_mul(2).ok_or_else(|| {
			Error::invalid_argument(format!("{operation} doubled padding overflows usize"))
		})?;
		let padded_height = input_height.checked_add(doubled_padding).ok_or_else(|| {
			Error::invalid_argument(format!("{operation} padded height overflows usize"))
		})?;
		let padded_width = input_width.checked_add(doubled_padding).ok_or_else(|| {
			Error::invalid_argument(format!("{operation} padded width overflows usize"))
		})?;
		if padded_height < kernel_size || padded_width < kernel_size {
			return Err(Error::invalid_argument(format!(
				"{operation} kernel does not fit the padded input"
			)));
		}
		let output_height = (padded_height - kernel_size) / stride + 1;
		let output_width = (padded_width - kernel_size) / stride + 1;
		let last_y = output_height
			.saturating_sub(1)
			.checked_mul(stride)
			.ok_or_else(|| Error::invalid_argument(format!("{operation} Y origin overflows usize")))?;
		let last_x = output_width
			.saturating_sub(1)
			.checked_mul(stride)
			.ok_or_else(|| Error::invalid_argument(format!("{operation} X origin overflows usize")))?;
		let last_y_end = last_y.checked_add(kernel_size).ok_or_else(|| {
			Error::invalid_argument(format!("{operation} final Y window overflows usize"))
		})?;
		let last_x_end = last_x.checked_add(kernel_size).ok_or_else(|| {
			Error::invalid_argument(format!("{operation} final X window overflows usize"))
		})?;
		for (label, value) in [
			("input height", *input_height),
			("input width", *input_width),
			("kernel size", kernel_size),
			("padding", padding),
			("last Y origin", last_y),
			("last X origin", last_x),
			("final Y window end", last_y_end),
			("final X window end", last_x_end),
		] {
			if value > i32::MAX as usize {
				return Err(Error::invalid_argument(format!(
					"{operation} {label} exceeds the signed shader coordinate ABI"
				)));
			}
		}
		let batch = shader_u32(*batch, "batch size", operation)?;
		let channels = shader_u32(*channels, "channel count", operation)?;
		batch.checked_mul(channels).ok_or_else(|| {
			Error::invalid_argument(format!("{operation} batch-channel count exceeds u32"))
		})?;
		Ok(Self {
			batch,
			channels,
			input_height: shader_u32(*input_height, "input height", operation)?,
			input_width: shader_u32(*input_width, "input width", operation)?,
			output_height: shader_u32(output_height, "output height", operation)?,
			output_width: shader_u32(output_width, "output width", operation)?,
			kernel_size: shader_u32(kernel_size, "kernel size", operation)?,
			stride: shader_u32(stride, "stride", operation)?,
			padding: shader_u32(padding, "padding", operation)?,
		})
	}

	fn push_constants(self) -> [PushConstant; 9] {
		[
			PushConstant::U32(self.batch),
			PushConstant::U32(self.channels),
			PushConstant::U32(self.input_height),
			PushConstant::U32(self.input_width),
			PushConstant::U32(self.output_height),
			PushConstant::U32(self.output_width),
			PushConstant::U32(self.kernel_size),
			PushConstant::U32(self.stride),
			PushConstant::U32(self.padding),
		]
	}

	fn attributes(self) -> [OpAttribute; 3] {
		[
			OpAttribute::UnsignedInteger {
				name: "kernel_size".into(),
				value: u64::from(self.kernel_size),
			},
			OpAttribute::UnsignedInteger {
				name: "stride".into(),
				value: u64::from(self.stride),
			},
			OpAttribute::UnsignedInteger {
				name: "padding".into(),
				value: u64::from(self.padding),
			},
		]
	}
}

#[derive(Clone, Copy)]
struct AdaptivePool2dGeometry {
	batch: u32,
	channels: u32,
	input_height: u32,
	input_width: u32,
	output_height: u32,
	output_width: u32,
}

impl AdaptivePool2dGeometry {
	fn resolve(
		input: &Matrix,
		output_height: usize,
		output_width: usize,
		operation: &'static str,
	) -> Result<Self> {
		let [batch, channels, input_height, input_width] = input.shape() else {
			return Err(Error::invalid_argument(format!(
				"{operation} requires NCHW rank-four input; found {:?}",
				input.shape()
			)));
		};
		if [
			*batch,
			*channels,
			*input_height,
			*input_width,
			output_height,
			output_width,
		]
		.contains(&0)
		{
			return Err(Error::invalid_argument(format!(
				"{operation} requires nonzero input and output extents"
			)));
		}
		shader_u32(input.num_elements(), "input element count", operation)?;
		let output_count = batch
			.checked_mul(*channels)
			.and_then(|value| value.checked_mul(output_height))
			.and_then(|value| value.checked_mul(output_width))
			.ok_or_else(|| Error::invalid_argument(format!("{operation} output size overflows usize")))?;
		shader_u32(output_count, "output element count", operation)?;
		for (label, output_extent, input_extent) in [
			("height", output_height, *input_height),
			("width", output_width, *input_width),
		] {
			output_extent
				.checked_mul(input_extent)
				.and_then(|value| value.checked_add(output_extent - 1))
				.and_then(|value| u32::try_from(value).ok())
				.ok_or_else(|| {
					Error::invalid_argument(format!(
						"{operation} adaptive {label} geometry exceeds the u32 shader ABI"
					))
				})?;
		}
		let batch = shader_u32(*batch, "batch size", operation)?;
		let channels = shader_u32(*channels, "channel count", operation)?;
		batch.checked_mul(channels).ok_or_else(|| {
			Error::invalid_argument(format!("{operation} batch-channel count exceeds u32"))
		})?;
		Ok(Self {
			batch,
			channels,
			input_height: shader_u32(*input_height, "input height", operation)?,
			input_width: shader_u32(*input_width, "input width", operation)?,
			output_height: shader_u32(output_height, "output height", operation)?,
			output_width: shader_u32(output_width, "output width", operation)?,
		})
	}

	fn push_constants(self) -> [PushConstant; 6] {
		[
			PushConstant::U32(self.batch),
			PushConstant::U32(self.channels),
			PushConstant::U32(self.input_height),
			PushConstant::U32(self.input_width),
			PushConstant::U32(self.output_height),
			PushConstant::U32(self.output_width),
		]
	}

	fn attributes(self) -> [OpAttribute; 2] {
		[
			OpAttribute::UnsignedInteger {
				name: "output_height".into(),
				value: u64::from(self.output_height),
			},
			OpAttribute::UnsignedInteger {
				name: "output_width".into(),
				value: u64::from(self.output_width),
			},
		]
	}
}

#[derive(Clone, Copy)]
struct Pool1dGeometry {
	batch: u32,
	channels: u32,
	input_length: u32,
	output_length: u32,
	kernel_size: u32,
	stride: u32,
	padding: u32,
}

impl Pool1dGeometry {
	fn resolve(
		input: &Matrix,
		kernel_size: usize,
		stride: usize,
		padding: usize,
		operation: &'static str,
	) -> Result<Self> {
		let [batch, channels, input_length] = input.shape() else {
			return Err(Error::invalid_argument(format!(
				"{operation} requires NCL rank-three input; found {:?}",
				input.shape()
			)));
		};
		if *batch == 0 || *channels == 0 || *input_length == 0 || kernel_size == 0 || stride == 0 {
			return Err(Error::invalid_argument(format!(
				"{operation} requires nonzero NCL extents, kernel size, and stride"
			)));
		}
		shader_u32(input.num_elements(), "input element count", operation)?;
		let doubled_padding = padding.checked_mul(2).ok_or_else(|| {
			Error::invalid_argument(format!("{operation} doubled padding overflows usize"))
		})?;
		let padded_length = input_length.checked_add(doubled_padding).ok_or_else(|| {
			Error::invalid_argument(format!("{operation} padded length overflows usize"))
		})?;
		if padded_length < kernel_size {
			return Err(Error::invalid_argument(format!(
				"{operation} kernel does not fit the padded input"
			)));
		}
		let output_length = (padded_length - kernel_size) / stride + 1;
		let last_x = output_length
			.saturating_sub(1)
			.checked_mul(stride)
			.ok_or_else(|| Error::invalid_argument(format!("{operation} X origin overflows usize")))?;
		let last_x_end = last_x.checked_add(kernel_size).ok_or_else(|| {
			Error::invalid_argument(format!("{operation} final X window overflows usize"))
		})?;
		for (label, value) in [
			("input length", *input_length),
			("kernel size", kernel_size),
			("padding", padding),
			("last X origin", last_x),
			("final X window end", last_x_end),
		] {
			if value > i32::MAX as usize {
				return Err(Error::invalid_argument(format!(
					"{operation} {label} exceeds the signed shader coordinate ABI"
				)));
			}
		}
		let batch = shader_u32(*batch, "batch size", operation)?;
		let channels = shader_u32(*channels, "channel count", operation)?;
		batch.checked_mul(channels).ok_or_else(|| {
			Error::invalid_argument(format!("{operation} batch-channel count exceeds u32"))
		})?;
		Ok(Self {
			batch,
			channels,
			input_length: shader_u32(*input_length, "input length", operation)?,
			output_length: shader_u32(output_length, "output length", operation)?,
			kernel_size: shader_u32(kernel_size, "kernel size", operation)?,
			stride: shader_u32(stride, "stride", operation)?,
			padding: shader_u32(padding, "padding", operation)?,
		})
	}

	fn push_constants(self) -> [PushConstant; 7] {
		[
			PushConstant::U32(self.batch),
			PushConstant::U32(self.channels),
			PushConstant::U32(self.input_length),
			PushConstant::U32(self.output_length),
			PushConstant::U32(self.kernel_size),
			PushConstant::U32(self.stride),
			PushConstant::U32(self.padding),
		]
	}

	fn attributes(self) -> [OpAttribute; 3] {
		[
			OpAttribute::UnsignedInteger {
				name: "kernel_size".into(),
				value: u64::from(self.kernel_size),
			},
			OpAttribute::UnsignedInteger {
				name: "stride".into(),
				value: u64::from(self.stride),
			},
			OpAttribute::UnsignedInteger {
				name: "padding".into(),
				value: u64::from(self.padding),
			},
		]
	}
}

#[derive(Clone, Copy)]
struct AdaptivePool1dGeometry {
	batch: u32,
	channels: u32,
	input_length: u32,
	output_length: u32,
}

impl AdaptivePool1dGeometry {
	fn resolve(input: &Matrix, output_length: usize, operation: &'static str) -> Result<Self> {
		let [batch, channels, input_length] = input.shape() else {
			return Err(Error::invalid_argument(format!(
				"{operation} requires NCL rank-three input; found {:?}",
				input.shape()
			)));
		};
		if [*batch, *channels, *input_length, output_length].contains(&0) {
			return Err(Error::invalid_argument(format!(
				"{operation} requires nonzero input and output extents"
			)));
		}
		shader_u32(input.num_elements(), "input element count", operation)?;
		let output_count = batch
			.checked_mul(*channels)
			.and_then(|value| value.checked_mul(output_length))
			.ok_or_else(|| Error::invalid_argument(format!("{operation} output size overflows usize")))?;
		shader_u32(output_count, "output element count", operation)?;
		output_length
			.checked_mul(*input_length)
			.and_then(|value| value.checked_add(output_length - 1))
			.and_then(|value| u32::try_from(value).ok())
			.ok_or_else(|| {
				Error::invalid_argument(format!(
					"{operation} adaptive length geometry exceeds the u32 shader ABI"
				))
			})?;
		let batch = shader_u32(*batch, "batch size", operation)?;
		let channels = shader_u32(*channels, "channel count", operation)?;
		batch.checked_mul(channels).ok_or_else(|| {
			Error::invalid_argument(format!("{operation} batch-channel count exceeds u32"))
		})?;
		Ok(Self {
			batch,
			channels,
			input_length: shader_u32(*input_length, "input length", operation)?,
			output_length: shader_u32(output_length, "output length", operation)?,
		})
	}

	fn push_constants(self) -> [PushConstant; 4] {
		[
			PushConstant::U32(self.batch),
			PushConstant::U32(self.channels),
			PushConstant::U32(self.input_length),
			PushConstant::U32(self.output_length),
		]
	}

	fn attributes(self) -> [OpAttribute; 1] {
		[OpAttribute::UnsignedInteger {
			name: "output_length".into(),
			value: u64::from(self.output_length),
		}]
	}
}

// ── public operations ─────────────────────────────────────────────────────────

/// Apply two-dimensional average pooling to an NCHW Matrix.
///
/// Padding positions do not contribute to either the sum or divisor. The
/// output spatial extent is `floor((input + 2 * padding - kernel_size) /
/// stride) + 1`, matching the OA C++ operation.
///
/// # Errors
///
/// Returns an error unless `input` is a nonempty rank-four F32 Matrix,
/// `kernel_size` and `stride` are nonzero, the kernel fits the padded input,
/// every shader extent is representable, or runtime recording fails.
pub fn avg_pool_2d(
	input: &Matrix,
	kernel_size: usize,
	stride: usize,
	padding: usize,
) -> Result<Matrix> {
	let output = avg_pool_2d_dispatch(input, kernel_size, stride, padding)?;
	autograd::record_avg_pool_2d(input, kernel_size, stride, padding, &output)?;
	Ok(output)
}

pub(in crate::ml) fn avg_pool_2d_backward(
	input: &Matrix,
	output_gradient: &Matrix,
	kernel_size: usize,
	stride: usize,
	padding: usize,
) -> Result<Matrix> {
	let contract = crate::core::operation::ml::AVG_POOL_2D_BACKWARD;
	let operation = contract.name();
	validate_f32_same_runtime(operation, &[input, output_gradient])?;
	let geometry = Pool2dGeometry::resolve(input, kernel_size, stride, padding, operation)?;
	let expected_shape = [
		geometry.batch as usize,
		geometry.channels as usize,
		geometry.output_height as usize,
		geometry.output_width as usize,
	];
	if output_gradient.shape() != expected_shape {
		return Err(Error::invalid_argument(format!(
			"{operation} output gradient must have shape {expected_shape:?}; found {:?}",
			output_gradient.shape()
		)));
	}
	let input_gradient = Matrix::allocate(
		input.runtime(),
		input.shape().to_vec(),
		input.num_elements(),
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::write(input_gradient.storage()),
	];
	let push_constants = geometry.push_constants();
	let attributes = geometry.attributes();
	let inputs = [input, output_gradient];
	let outputs = [&input_gradient];
	input.runtime().record_semantic(
		ComputeDispatch {
			kernel: KernelId::MlAvgPool2dBackwardF32,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: [
				geometry.input_height.div_ceil(16),
				geometry.input_width.div_ceil(16),
				geometry.batch * geometry.channels,
			],
			indirect: None,
		},
		SemanticDispatch {
			contract,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok(input_gradient)
}

/// Apply two-dimensional max pooling to an NCHW Matrix.
///
/// Equal maxima select the first valid input in row-major window order. The
/// returned U32 argmax indices are explicit because they are part of the donor
/// operation contract and make the deterministic adjoint reusable.
///
/// # Errors
///
/// Returns an error under the same geometry and runtime conditions as
/// [`avg_pool_2d`].
pub fn max_pool_2d(
	input: &Matrix,
	kernel_size: usize,
	stride: usize,
	padding: usize,
) -> Result<MaxPool2dResult> {
	let (output, indices) = max_pool_2d_dispatch(input, kernel_size, stride, padding)?;
	autograd::record_max_pool_2d(
		input,
		indices.clone(),
		kernel_size,
		stride,
		padding,
		&output,
	)?;
	Ok(MaxPool2dResult { output, indices })
}

pub(in crate::ml) fn max_pool_2d_backward(
	input: &Matrix,
	indices: &Matrix,
	output_gradient: &Matrix,
	kernel_size: usize,
	stride: usize,
	padding: usize,
) -> Result<Matrix> {
	let contract = crate::core::operation::ml::MAX_POOL_2D_BACKWARD;
	let operation = contract.name();
	validate_f32_same_runtime(operation, &[input, output_gradient])?;
	if indices.dtype() != DType::U32 {
		return Err(Error::invalid_argument(format!(
			"{operation} indices must use U32; found {}",
			indices.dtype().token()
		)));
	}
	if !input.runtime().same_as(indices.runtime()) {
		return Err(Error::invalid_argument(format!(
			"{operation} inputs must belong to the same runtime"
		)));
	}
	let geometry = Pool2dGeometry::resolve(input, kernel_size, stride, padding, operation)?;
	let expected_shape = [
		geometry.batch as usize,
		geometry.channels as usize,
		geometry.output_height as usize,
		geometry.output_width as usize,
	];
	if indices.shape() != expected_shape || output_gradient.shape() != expected_shape {
		return Err(Error::invalid_argument(format!(
			"{operation} indices and output gradient must have shape {expected_shape:?}; found {:?} and {:?}",
			indices.shape(),
			output_gradient.shape()
		)));
	}
	let input_gradient = Matrix::allocate(
		input.runtime(),
		input.shape().to_vec(),
		input.num_elements(),
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(indices.storage()),
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::write(input_gradient.storage()),
	];
	let push_constants = geometry.push_constants();
	let attributes = geometry.attributes();
	let inputs = [input, indices, output_gradient];
	let outputs = [&input_gradient];
	input.runtime().record_semantic(
		ComputeDispatch {
			kernel: KernelId::MlMaxPool2dBackwardF32,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: [
				geometry.input_height.div_ceil(16),
				geometry.input_width.div_ceil(16),
				geometry.batch * geometry.channels,
			],
			indirect: None,
		},
		SemanticDispatch {
			contract,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok(input_gradient)
}

/// Adaptively average-pool an NCHW Matrix to an exact spatial extent.
///
/// Each output bin independently uses floor/ceiling boundaries along height
/// and width. This supports rectangular and non-divisible geometry without
/// approximating the operation as a square fixed-window pool.
///
/// # Errors
///
/// Returns an error unless the input is nonempty rank-four F32, both output
/// extents are nonzero, all geometry fits the shader ABI, or recording fails.
pub fn adaptive_avg_pool_2d(
	input: &Matrix,
	output_height: usize,
	output_width: usize,
) -> Result<Matrix> {
	let output = adaptive_avg_pool_2d_dispatch(input, output_height, output_width)?;
	autograd::record_adaptive_avg_pool_2d(input, output_height, output_width, &output)?;
	Ok(output)
}

pub(in crate::ml) fn adaptive_avg_pool_2d_backward(
	input: &Matrix,
	output_gradient: &Matrix,
	output_height: usize,
	output_width: usize,
) -> Result<Matrix> {
	let contract = crate::core::operation::ml::ADAPTIVE_AVG_POOL_2D_BACKWARD;
	let operation = contract.name();
	validate_f32_same_runtime(operation, &[input, output_gradient])?;
	let geometry = AdaptivePool2dGeometry::resolve(input, output_height, output_width, operation)?;
	let expected_shape = [
		geometry.batch as usize,
		geometry.channels as usize,
		output_height,
		output_width,
	];
	if output_gradient.shape() != expected_shape {
		return Err(Error::invalid_argument(format!(
			"{operation} output gradient must have shape {expected_shape:?}; found {:?}",
			output_gradient.shape()
		)));
	}
	let input_gradient = Matrix::allocate(
		input.runtime(),
		input.shape().to_vec(),
		input.num_elements(),
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::write(input_gradient.storage()),
	];
	let push_constants = geometry.push_constants();
	let attributes = geometry.attributes();
	input.runtime().record_semantic(
		ComputeDispatch {
			kernel: KernelId::MlAdaptiveAvgPool2dBackwardF32,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: [
				geometry.input_height.div_ceil(16),
				geometry.input_width.div_ceil(16),
				geometry.batch * geometry.channels,
			],
			indirect: None,
		},
		SemanticDispatch {
			contract,
			inputs: &[input, output_gradient],
			outputs: &[&input_gradient],
			attributes: &attributes,
		},
	)?;
	Ok(input_gradient)
}

/// Apply one-dimensional max pooling to an NCL Matrix.
///
/// Padding positions are excluded from each window. The output extent is
/// `floor((input + 2 * padding - kernel_size) / stride) + 1`. The returned
/// record carries the pooled output and the flat input index each window
/// selected; retaining `indices` is required for reverse-mode gradients.
///
/// # Errors
///
/// Returns an error unless `input` is a nonempty rank-three F32 Matrix,
/// `kernel_size` and `stride` are nonzero, the kernel fits the padded input,
/// every shader extent is representable, or runtime recording fails.
pub fn max_pool_1d(
	input: &Matrix,
	kernel_size: usize,
	stride: usize,
	padding: usize,
) -> Result<MaxPool1dResult> {
	let (output, indices) = max_pool_1d_dispatch(input, kernel_size, stride, padding)?;
	autograd::record_max_pool_1d(
		input,
		indices.clone(),
		kernel_size,
		stride,
		padding,
		&output,
	)?;
	Ok(MaxPool1dResult { output, indices })
}

pub(in crate::ml) fn max_pool_1d_backward(
	input: &Matrix,
	indices: &Matrix,
	output_gradient: &Matrix,
	kernel_size: usize,
	stride: usize,
	padding: usize,
) -> Result<Matrix> {
	let contract = crate::core::operation::ml::MAX_POOL_1D_BACKWARD;
	let operation = contract.name();
	validate_f32_same_runtime(operation, &[input, output_gradient])?;
	if indices.dtype() != DType::U32 {
		return Err(Error::invalid_argument(format!(
			"{operation} indices must use U32; found {}",
			indices.dtype().token()
		)));
	}
	if !input.runtime().same_as(indices.runtime()) {
		return Err(Error::invalid_argument(format!(
			"{operation} inputs must belong to the same runtime"
		)));
	}
	let geometry = Pool1dGeometry::resolve(input, kernel_size, stride, padding, operation)?;
	let expected_shape = [
		geometry.batch as usize,
		geometry.channels as usize,
		geometry.output_length as usize,
	];
	if indices.shape() != expected_shape || output_gradient.shape() != expected_shape {
		return Err(Error::invalid_argument(format!(
			"{operation} indices and output gradient must have shape {expected_shape:?}; found {:?} and {:?}",
			indices.shape(),
			output_gradient.shape()
		)));
	}
	let input_gradient = Matrix::allocate(
		input.runtime(),
		input.shape().to_vec(),
		input.num_elements(),
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(indices.storage()),
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::write(input_gradient.storage()),
	];
	let push_constants = geometry.push_constants();
	let attributes = geometry.attributes();
	let inputs = [input, indices, output_gradient];
	let outputs = [&input_gradient];
	input.runtime().record_semantic(
		ComputeDispatch {
			kernel: KernelId::MlMaxPool1dBackwardF32,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: [
				geometry.input_length.div_ceil(256),
				geometry.batch * geometry.channels,
				1,
			],
			indirect: None,
		},
		SemanticDispatch {
			contract,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok(input_gradient)
}

/// Adaptively max-pool an NCL Matrix to exactly `output_length` elements.
///
/// Each output window covers the `[floor(i*L/out), ceil((i+1)*L/out))`
/// source bin, matching `nn.AdaptiveMaxPool1d`. The returned record carries
/// the pooled output and the flat input index each bin selected; retaining
/// `indices` is required for reverse-mode gradients.
///
/// # Errors
///
/// Returns an error unless `input` is a nonempty rank-three F32 Matrix and
/// `output_length` is nonzero, or runtime recording fails.
pub fn adaptive_max_pool_1d(
	input: &Matrix,
	output_length: usize,
) -> Result<AdaptiveMaxPool1dResult> {
	let (output, indices) = adaptive_max_pool_1d_dispatch(input, output_length)?;
	autograd::record_adaptive_max_pool_1d(input, indices.clone(), output_length, &output)?;
	Ok(AdaptiveMaxPool1dResult { output, indices })
}

pub(in crate::ml) fn adaptive_max_pool_1d_backward(
	input: &Matrix,
	indices: &Matrix,
	output_gradient: &Matrix,
	output_length: usize,
) -> Result<Matrix> {
	let contract = crate::core::operation::ml::ADAPTIVE_MAX_POOL_1D_BACKWARD;
	let operation = contract.name();
	validate_f32_same_runtime(operation, &[input, output_gradient])?;
	if indices.dtype() != DType::U32 {
		return Err(Error::invalid_argument(format!(
			"{operation} indices must use U32; found {}",
			indices.dtype().token()
		)));
	}
	if !input.runtime().same_as(indices.runtime()) {
		return Err(Error::invalid_argument(format!(
			"{operation} inputs must belong to the same runtime"
		)));
	}
	let geometry = AdaptivePool1dGeometry::resolve(input, output_length, operation)?;
	let expected_shape = [
		geometry.batch as usize,
		geometry.channels as usize,
		geometry.output_length as usize,
	];
	if indices.shape() != expected_shape || output_gradient.shape() != expected_shape {
		return Err(Error::invalid_argument(format!(
			"{operation} indices and output gradient must have shape {expected_shape:?}; found {:?} and {:?}",
			indices.shape(),
			output_gradient.shape()
		)));
	}
	let input_gradient = Matrix::allocate(
		input.runtime(),
		input.shape().to_vec(),
		input.num_elements(),
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(indices.storage()),
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::write(input_gradient.storage()),
	];
	let push_constants = geometry.push_constants();
	let attributes = geometry.attributes();
	let inputs = [input, indices, output_gradient];
	let outputs = [&input_gradient];
	input.runtime().record_semantic(
		ComputeDispatch {
			kernel: KernelId::MlAdaptiveMaxPool1dBackwardF32,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: [
				geometry.input_length.div_ceil(256),
				geometry.batch * geometry.channels,
				1,
			],
			indirect: None,
		},
		SemanticDispatch {
			contract,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok(input_gradient)
}

// ── private dispatch helpers ──────────────────────────────────────────────────

fn avg_pool_2d_dispatch(
	input: &Matrix,
	kernel_size: usize,
	stride: usize,
	padding: usize,
) -> Result<Matrix> {
	let contract = crate::core::operation::ml::AVG_POOL_2D;
	let operation = contract.name();
	validate_f32_same_runtime(operation, &[input])?;
	let geometry = Pool2dGeometry::resolve(input, kernel_size, stride, padding, operation)?;
	let output_shape = [
		geometry.batch as usize,
		geometry.channels as usize,
		geometry.output_height as usize,
		geometry.output_width as usize,
	];
	let output_count = output_shape
		.iter()
		.try_fold(1_usize, |product, extent| product.checked_mul(*extent))
		.ok_or_else(|| Error::invalid_argument(format!("{operation} output size overflows usize")))?;
	shader_u32(output_count, "output element count", operation)?;
	let output = Matrix::allocate(
		input.runtime(),
		output_shape.to_vec(),
		output_count,
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::write(output.storage()),
	];
	let push_constants = geometry.push_constants();
	let attributes = geometry.attributes();
	let inputs = [input];
	let outputs = [&output];
	input.runtime().record_semantic(
		ComputeDispatch {
			kernel: KernelId::MlAvgPool2dF32,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: [
				geometry.output_height.div_ceil(16),
				geometry.output_width.div_ceil(16),
				geometry.batch * geometry.channels,
			],
			indirect: None,
		},
		SemanticDispatch {
			contract,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok(output)
}

fn max_pool_2d_dispatch(
	input: &Matrix,
	kernel_size: usize,
	stride: usize,
	padding: usize,
) -> Result<(Matrix, Matrix)> {
	let contract = crate::core::operation::ml::MAX_POOL_2D;
	let operation = contract.name();
	validate_f32_same_runtime(operation, &[input])?;
	let geometry = Pool2dGeometry::resolve(input, kernel_size, stride, padding, operation)?;
	let output_shape = [
		geometry.batch as usize,
		geometry.channels as usize,
		geometry.output_height as usize,
		geometry.output_width as usize,
	];
	let output_count = output_shape
		.iter()
		.try_fold(1_usize, |product, extent| product.checked_mul(*extent))
		.ok_or_else(|| Error::invalid_argument(format!("{operation} output size overflows usize")))?;
	shader_u32(output_count, "output element count", operation)?;
	let output = Matrix::allocate(
		input.runtime(),
		output_shape.to_vec(),
		output_count,
		DType::F32,
	)?;
	let indices = Matrix::allocate(
		input.runtime(),
		output_shape.to_vec(),
		output_count,
		DType::U32,
	)?;
	let buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::write(output.storage()),
		BufferBinding::write(indices.storage()),
	];
	let push_constants = geometry.push_constants();
	let attributes = geometry.attributes();
	let inputs = [input];
	let outputs = [&output, &indices];
	input.runtime().record_semantic(
		ComputeDispatch {
			kernel: KernelId::MlMaxPool2dF32,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: [
				geometry.output_height.div_ceil(16),
				geometry.output_width.div_ceil(16),
				geometry.batch * geometry.channels,
			],
			indirect: None,
		},
		SemanticDispatch {
			contract,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok((output, indices))
}

fn adaptive_avg_pool_2d_dispatch(
	input: &Matrix,
	output_height: usize,
	output_width: usize,
) -> Result<Matrix> {
	let contract = crate::core::operation::ml::ADAPTIVE_AVG_POOL_2D;
	let operation = contract.name();
	validate_f32_same_runtime(operation, &[input])?;
	let geometry = AdaptivePool2dGeometry::resolve(input, output_height, output_width, operation)?;
	let output_shape = [
		geometry.batch as usize,
		geometry.channels as usize,
		output_height,
		output_width,
	];
	let output_count = output_shape.iter().product();
	let output = Matrix::allocate(
		input.runtime(),
		output_shape.to_vec(),
		output_count,
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::write(output.storage()),
	];
	let push_constants = geometry.push_constants();
	let attributes = geometry.attributes();
	input.runtime().record_semantic(
		ComputeDispatch {
			kernel: KernelId::MlAdaptiveAvgPool2dF32,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: [
				geometry.output_height.div_ceil(16),
				geometry.output_width.div_ceil(16),
				geometry.batch * geometry.channels,
			],
			indirect: None,
		},
		SemanticDispatch {
			contract,
			inputs: &[input],
			outputs: &[&output],
			attributes: &attributes,
		},
	)?;
	Ok(output)
}
fn max_pool_1d_dispatch(
	input: &Matrix,
	kernel_size: usize,
	stride: usize,
	padding: usize,
) -> Result<(Matrix, Matrix)> {
	let contract = crate::core::operation::ml::MAX_POOL_1D;
	let operation = contract.name();
	validate_f32_same_runtime(operation, &[input])?;
	let geometry = Pool1dGeometry::resolve(input, kernel_size, stride, padding, operation)?;
	let output_shape = [
		geometry.batch as usize,
		geometry.channels as usize,
		geometry.output_length as usize,
	];
	let output_count = output_shape
		.iter()
		.try_fold(1_usize, |product, extent| product.checked_mul(*extent))
		.ok_or_else(|| Error::invalid_argument(format!("{operation} output size overflows usize")))?;
	shader_u32(output_count, "output element count", operation)?;
	let output = Matrix::allocate(
		input.runtime(),
		output_shape.to_vec(),
		output_count,
		DType::F32,
	)?;
	let indices = Matrix::allocate(
		input.runtime(),
		output_shape.to_vec(),
		output_count,
		DType::U32,
	)?;
	let buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::write(output.storage()),
		BufferBinding::write(indices.storage()),
	];
	let push_constants = geometry.push_constants();
	let attributes = geometry.attributes();
	let inputs = [input];
	let outputs = [&output, &indices];
	input.runtime().record_semantic(
		ComputeDispatch {
			kernel: KernelId::MlMaxPool1dF32,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: [
				geometry.output_length.div_ceil(256),
				geometry.batch * geometry.channels,
				1,
			],
			indirect: None,
		},
		SemanticDispatch {
			contract,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok((output, indices))
}

fn adaptive_max_pool_1d_dispatch(input: &Matrix, output_length: usize) -> Result<(Matrix, Matrix)> {
	let contract = crate::core::operation::ml::ADAPTIVE_MAX_POOL_1D;
	let operation = contract.name();
	validate_f32_same_runtime(operation, &[input])?;
	let geometry = AdaptivePool1dGeometry::resolve(input, output_length, operation)?;
	let output_shape = [
		geometry.batch as usize,
		geometry.channels as usize,
		geometry.output_length as usize,
	];
	let output_count = output_shape
		.iter()
		.try_fold(1_usize, |product, extent| product.checked_mul(*extent))
		.ok_or_else(|| Error::invalid_argument(format!("{operation} output size overflows usize")))?;
	shader_u32(output_count, "output element count", operation)?;
	let output = Matrix::allocate(
		input.runtime(),
		output_shape.to_vec(),
		output_count,
		DType::F32,
	)?;
	let indices = Matrix::allocate(
		input.runtime(),
		output_shape.to_vec(),
		output_count,
		DType::U32,
	)?;
	let buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::write(output.storage()),
		BufferBinding::write(indices.storage()),
	];
	let push_constants = geometry.push_constants();
	let attributes = geometry.attributes();
	let inputs = [input];
	let outputs = [&output, &indices];
	input.runtime().record_semantic(
		ComputeDispatch {
			kernel: KernelId::MlAdaptiveMaxPool1dF32,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: [
				geometry.output_length.div_ceil(256),
				geometry.batch * geometry.channels,
				1,
			],
			indirect: None,
		},
		SemanticDispatch {
			contract,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok((output, indices))
}

// ── Pool3d geometry ─────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct Pool3dGeometry {
	batch: u32,
	channels: u32,
	input_depth: u32,
	input_height: u32,
	input_width: u32,
	output_depth: u32,
	output_height: u32,
	output_width: u32,
	kernel_size: u32,
	stride: u32,
	padding: u32,
}

impl Pool3dGeometry {
	fn resolve(
		input: &Matrix,
		kernel_size: usize,
		stride: usize,
		padding: usize,
		operation: &'static str,
	) -> Result<Self> {
		let [batch, channels, input_depth, input_height, input_width] = input.shape() else {
			return Err(Error::invalid_argument(format!(
				"{operation} requires NCDHW rank-five input; found {:?}",
				input.shape()
			)));
		};
		if *batch == 0
			|| *channels == 0
			|| *input_depth == 0
			|| *input_height == 0
			|| *input_width == 0
			|| kernel_size == 0
			|| stride == 0
		{
			return Err(Error::invalid_argument(format!(
				"{operation} requires nonzero NCDHW extents, kernel size, and stride"
			)));
		}
		shader_u32(input.num_elements(), "input element count", operation)?;
		let doubled_padding = padding.checked_mul(2).ok_or_else(|| {
			Error::invalid_argument(format!("{operation} doubled padding overflows usize"))
		})?;
		let padded = |extent: usize| -> Result<usize> {
			extent.checked_add(doubled_padding).ok_or_else(|| {
				Error::invalid_argument(format!("{operation} padded extent overflows usize"))
			})
		};
		let padded_depth = padded(*input_depth)?;
		let padded_height = padded(*input_height)?;
		let padded_width = padded(*input_width)?;
		if padded_depth < kernel_size || padded_height < kernel_size || padded_width < kernel_size {
			return Err(Error::invalid_argument(format!(
				"{operation} kernel does not fit the padded input"
			)));
		}
		let output_depth = (padded_depth - kernel_size) / stride + 1;
		let output_height = (padded_height - kernel_size) / stride + 1;
		let output_width = (padded_width - kernel_size) / stride + 1;
		let last = |output: usize| -> Result<usize> {
			let origin = output
				.saturating_sub(1)
				.checked_mul(stride)
				.ok_or_else(|| {
					Error::invalid_argument(format!("{operation} window origin overflows usize"))
				})?;
			origin
				.checked_add(kernel_size)
				.ok_or_else(|| Error::invalid_argument(format!("{operation} final window overflows usize")))
		};
		for (label, value) in [
			("input depth", *input_depth),
			("input height", *input_height),
			("input width", *input_width),
			("kernel size", kernel_size),
			("padding", padding),
			("final Z window end", last(output_depth)?),
			("final Y window end", last(output_height)?),
			("final X window end", last(output_width)?),
		] {
			if value > i32::MAX as usize {
				return Err(Error::invalid_argument(format!(
					"{operation} {label} exceeds the signed shader coordinate ABI"
				)));
			}
		}
		let batch = shader_u32(*batch, "batch size", operation)?;
		let channels = shader_u32(*channels, "channel count", operation)?;
		batch.checked_mul(channels).ok_or_else(|| {
			Error::invalid_argument(format!("{operation} batch-channel count exceeds u32"))
		})?;
		Ok(Self {
			batch,
			channels,
			input_depth: shader_u32(*input_depth, "input depth", operation)?,
			input_height: shader_u32(*input_height, "input height", operation)?,
			input_width: shader_u32(*input_width, "input width", operation)?,
			output_depth: shader_u32(output_depth, "output depth", operation)?,
			output_height: shader_u32(output_height, "output height", operation)?,
			output_width: shader_u32(output_width, "output width", operation)?,
			kernel_size: shader_u32(kernel_size, "kernel size", operation)?,
			stride: shader_u32(stride, "stride", operation)?,
			padding: shader_u32(padding, "padding", operation)?,
		})
	}

	fn output_shape(&self) -> [usize; 5] {
		[
			self.batch as usize,
			self.channels as usize,
			self.output_depth as usize,
			self.output_height as usize,
			self.output_width as usize,
		]
	}

	fn push_constants(self) -> [PushConstant; 11] {
		[
			PushConstant::U32(self.batch),
			PushConstant::U32(self.channels),
			PushConstant::U32(self.input_depth),
			PushConstant::U32(self.input_height),
			PushConstant::U32(self.input_width),
			PushConstant::U32(self.output_depth),
			PushConstant::U32(self.output_height),
			PushConstant::U32(self.output_width),
			PushConstant::U32(self.kernel_size),
			PushConstant::U32(self.stride),
			PushConstant::U32(self.padding),
		]
	}

	fn attributes(self) -> [OpAttribute; 3] {
		[
			OpAttribute::UnsignedInteger {
				name: "kernel_size".into(),
				value: u64::from(self.kernel_size),
			},
			OpAttribute::UnsignedInteger {
				name: "stride".into(),
				value: u64::from(self.stride),
			},
			OpAttribute::UnsignedInteger {
				name: "padding".into(),
				value: u64::from(self.padding),
			},
		]
	}
}

#[derive(Clone, Copy)]
struct AdaptivePool3dGeometry {
	batch: u32,
	channels: u32,
	input_depth: u32,
	input_height: u32,
	input_width: u32,
	output_depth: u32,
	output_height: u32,
	output_width: u32,
}

impl AdaptivePool3dGeometry {
	fn resolve(
		input: &Matrix,
		output_depth: usize,
		output_height: usize,
		output_width: usize,
		operation: &'static str,
	) -> Result<Self> {
		let [batch, channels, input_depth, input_height, input_width] = input.shape() else {
			return Err(Error::invalid_argument(format!(
				"{operation} requires NCDHW rank-five input; found {:?}",
				input.shape()
			)));
		};
		if [
			*batch,
			*channels,
			*input_depth,
			*input_height,
			*input_width,
			output_depth,
			output_height,
			output_width,
		]
		.contains(&0)
		{
			return Err(Error::invalid_argument(format!(
				"{operation} requires nonzero input and output extents"
			)));
		}
		shader_u32(input.num_elements(), "input element count", operation)?;
		let output_count = batch
			.checked_mul(*channels)
			.and_then(|value| value.checked_mul(output_depth))
			.and_then(|value| value.checked_mul(output_height))
			.and_then(|value| value.checked_mul(output_width))
			.ok_or_else(|| Error::invalid_argument(format!("{operation} output size overflows usize")))?;
		shader_u32(output_count, "output element count", operation)?;
		let batch = shader_u32(*batch, "batch size", operation)?;
		let channels = shader_u32(*channels, "channel count", operation)?;
		let output_depth_u32 = shader_u32(output_depth, "output depth", operation)?;
		batch
			.checked_mul(channels)
			.and_then(|value| value.checked_mul(output_depth_u32))
			.ok_or_else(|| {
				Error::invalid_argument(format!("{operation} batch-channel-depth count exceeds u32"))
			})?;
		Ok(Self {
			batch,
			channels,
			input_depth: shader_u32(*input_depth, "input depth", operation)?,
			input_height: shader_u32(*input_height, "input height", operation)?,
			input_width: shader_u32(*input_width, "input width", operation)?,
			output_depth: shader_u32(output_depth, "output depth", operation)?,
			output_height: shader_u32(output_height, "output height", operation)?,
			output_width: shader_u32(output_width, "output width", operation)?,
		})
	}

	fn output_shape(&self) -> [usize; 5] {
		[
			self.batch as usize,
			self.channels as usize,
			self.output_depth as usize,
			self.output_height as usize,
			self.output_width as usize,
		]
	}

	fn push_constants(self) -> [PushConstant; 8] {
		[
			PushConstant::U32(self.batch),
			PushConstant::U32(self.channels),
			PushConstant::U32(self.input_depth),
			PushConstant::U32(self.input_height),
			PushConstant::U32(self.input_width),
			PushConstant::U32(self.output_depth),
			PushConstant::U32(self.output_height),
			PushConstant::U32(self.output_width),
		]
	}

	fn attributes(self) -> [OpAttribute; 3] {
		[
			OpAttribute::UnsignedInteger {
				name: "output_depth".into(),
				value: u64::from(self.output_depth),
			},
			OpAttribute::UnsignedInteger {
				name: "output_height".into(),
				value: u64::from(self.output_height),
			},
			OpAttribute::UnsignedInteger {
				name: "output_width".into(),
				value: u64::from(self.output_width),
			},
		]
	}
}

// ── public pool-3d operations ───────────────────────────────────────────────

/// Max-pool an NCDHW Matrix with a cubic kernel, emitting argmax indices.
///
/// # Errors
///
/// Returns an error unless `input` is a nonempty rank-five F32 Matrix,
/// `kernel_size` and `stride` are nonzero, the kernel fits the padded input,
/// or runtime recording fails.
pub fn max_pool_3d(
	input: &Matrix,
	kernel_size: usize,
	stride: usize,
	padding: usize,
) -> Result<MaxPool3dResult> {
	let (output, indices) = max_pool_3d_dispatch(input, kernel_size, stride, padding)?;
	autograd::record_max_pool_3d(
		input,
		indices.clone(),
		kernel_size,
		stride,
		padding,
		&output,
	)?;
	Ok(MaxPool3dResult { output, indices })
}

pub(in crate::ml) fn max_pool_3d_backward(
	input: &Matrix,
	indices: &Matrix,
	output_gradient: &Matrix,
	kernel_size: usize,
	stride: usize,
	padding: usize,
) -> Result<Matrix> {
	let contract = crate::core::operation::ml::MAX_POOL_3D_BACKWARD;
	let operation = contract.name();
	validate_f32_same_runtime(operation, &[input, output_gradient])?;
	if indices.dtype() != DType::U32 {
		return Err(Error::invalid_argument(format!(
			"{operation} indices must use U32; found {}",
			indices.dtype().token()
		)));
	}
	if !input.runtime().same_as(indices.runtime()) {
		return Err(Error::invalid_argument(format!(
			"{operation} inputs must belong to the same runtime"
		)));
	}
	let geometry = Pool3dGeometry::resolve(input, kernel_size, stride, padding, operation)?;
	let expected_shape = geometry.output_shape();
	if indices.shape() != expected_shape || output_gradient.shape() != expected_shape {
		return Err(Error::invalid_argument(format!(
			"{operation} indices and output gradient must have shape {expected_shape:?}; found {:?} and {:?}",
			indices.shape(),
			output_gradient.shape()
		)));
	}
	let input_gradient = Matrix::allocate(
		input.runtime(),
		input.shape().to_vec(),
		input.num_elements(),
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(indices.storage()),
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::write(input_gradient.storage()),
	];
	let push_constants = geometry.push_constants();
	let attributes = geometry.attributes();
	let inputs = [input, indices, output_gradient];
	let outputs = [&input_gradient];
	input.runtime().record_semantic(
		ComputeDispatch {
			kernel: KernelId::MlMaxPool3dBackwardF32,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: [
				geometry.input_height.div_ceil(16),
				geometry.input_width.div_ceil(16),
				geometry.batch * geometry.channels * geometry.input_depth,
			],
			indirect: None,
		},
		SemanticDispatch {
			contract,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok(input_gradient)
}

/// Average-pool an NCDHW Matrix with a cubic kernel.
///
/// # Errors
///
/// Returns an error unless `input` is a nonempty rank-five F32 Matrix,
/// `kernel_size` and `stride` are nonzero, the kernel fits the padded input,
/// or runtime recording fails.
pub fn avg_pool_3d(
	input: &Matrix,
	kernel_size: usize,
	stride: usize,
	padding: usize,
) -> Result<Matrix> {
	let output = avg_pool_3d_dispatch(input, kernel_size, stride, padding)?;
	autograd::record_avg_pool_3d(input, kernel_size, stride, padding, &output)?;
	Ok(output)
}

pub(in crate::ml) fn avg_pool_3d_backward(
	input: &Matrix,
	output_gradient: &Matrix,
	kernel_size: usize,
	stride: usize,
	padding: usize,
) -> Result<Matrix> {
	let contract = crate::core::operation::ml::AVG_POOL_3D_BACKWARD;
	let operation = contract.name();
	validate_f32_same_runtime(operation, &[input, output_gradient])?;
	let geometry = Pool3dGeometry::resolve(input, kernel_size, stride, padding, operation)?;
	let expected_shape = geometry.output_shape();
	if output_gradient.shape() != expected_shape {
		return Err(Error::invalid_argument(format!(
			"{operation} output gradient must have shape {expected_shape:?}; found {:?}",
			output_gradient.shape()
		)));
	}
	let input_gradient = Matrix::allocate(
		input.runtime(),
		input.shape().to_vec(),
		input.num_elements(),
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::write(input_gradient.storage()),
	];
	let push_constants = geometry.push_constants();
	let attributes = geometry.attributes();
	let inputs = [input, output_gradient];
	let outputs = [&input_gradient];
	input.runtime().record_semantic(
		ComputeDispatch {
			kernel: KernelId::MlAvgPool3dBackwardF32,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: [
				geometry.input_height.div_ceil(16),
				geometry.input_width.div_ceil(16),
				geometry.batch * geometry.channels * geometry.input_depth,
			],
			indirect: None,
		},
		SemanticDispatch {
			contract,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok(input_gradient)
}

/// Adaptively max-pool an NCHW Matrix to exactly `output_height` ×
/// `output_width`, emitting argmax indices.
///
/// # Errors
///
/// Returns an error unless `input` is a nonempty rank-four F32 Matrix and
/// both output extents are nonzero, or runtime recording fails.
pub fn adaptive_max_pool_2d(
	input: &Matrix,
	output_height: usize,
	output_width: usize,
) -> Result<AdaptiveMaxPool2dResult> {
	let (output, indices) = adaptive_max_pool_2d_dispatch(input, output_height, output_width)?;
	autograd::record_adaptive_max_pool_2d(
		input,
		indices.clone(),
		output_height,
		output_width,
		&output,
	)?;
	Ok(AdaptiveMaxPool2dResult { output, indices })
}

pub(in crate::ml) fn adaptive_max_pool_2d_backward(
	input: &Matrix,
	indices: &Matrix,
	output_gradient: &Matrix,
	output_height: usize,
	output_width: usize,
) -> Result<Matrix> {
	let contract = crate::core::operation::ml::ADAPTIVE_MAX_POOL_2D_BACKWARD;
	let operation = contract.name();
	validate_f32_same_runtime(operation, &[input, output_gradient])?;
	if indices.dtype() != DType::U32 {
		return Err(Error::invalid_argument(format!(
			"{operation} indices must use U32; found {}",
			indices.dtype().token()
		)));
	}
	if !input.runtime().same_as(indices.runtime()) {
		return Err(Error::invalid_argument(format!(
			"{operation} inputs must belong to the same runtime"
		)));
	}
	let geometry = AdaptivePool2dGeometry::resolve(input, output_height, output_width, operation)?;
	let expected_shape = [
		geometry.batch as usize,
		geometry.channels as usize,
		geometry.output_height as usize,
		geometry.output_width as usize,
	];
	if indices.shape() != expected_shape || output_gradient.shape() != expected_shape {
		return Err(Error::invalid_argument(format!(
			"{operation} indices and output gradient must have shape {expected_shape:?}; found {:?} and {:?}",
			indices.shape(),
			output_gradient.shape()
		)));
	}
	let input_gradient = Matrix::allocate(
		input.runtime(),
		input.shape().to_vec(),
		input.num_elements(),
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(indices.storage()),
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::write(input_gradient.storage()),
	];
	let push_constants = geometry.push_constants();
	let attributes = geometry.attributes();
	let inputs = [input, indices, output_gradient];
	let outputs = [&input_gradient];
	input.runtime().record_semantic(
		ComputeDispatch {
			kernel: KernelId::MlAdaptiveMaxPool2dBackwardF32,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: [
				geometry.input_height.div_ceil(16),
				geometry.input_width.div_ceil(16),
				geometry.batch * geometry.channels,
			],
			indirect: None,
		},
		SemanticDispatch {
			contract,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok(input_gradient)
}

/// Adaptively max-pool an NCDHW Matrix to exactly `output_depth` ×
/// `output_height` × `output_width`, emitting argmax indices.
///
/// # Errors
///
/// Returns an error unless `input` is a nonempty rank-five F32 Matrix and
/// all output extents are nonzero, or runtime recording fails.
pub fn adaptive_max_pool_3d(
	input: &Matrix,
	output_depth: usize,
	output_height: usize,
	output_width: usize,
) -> Result<AdaptiveMaxPool3dResult> {
	let (output, indices) =
		adaptive_max_pool_3d_dispatch(input, output_depth, output_height, output_width)?;
	autograd::record_adaptive_max_pool_3d(
		input,
		indices.clone(),
		output_depth,
		output_height,
		output_width,
		&output,
	)?;
	Ok(AdaptiveMaxPool3dResult { output, indices })
}

pub(in crate::ml) fn adaptive_max_pool_3d_backward(
	input: &Matrix,
	indices: &Matrix,
	output_gradient: &Matrix,
	output_depth: usize,
	output_height: usize,
	output_width: usize,
) -> Result<Matrix> {
	let contract = crate::core::operation::ml::ADAPTIVE_MAX_POOL_3D_BACKWARD;
	let operation = contract.name();
	validate_f32_same_runtime(operation, &[input, output_gradient])?;
	if indices.dtype() != DType::U32 {
		return Err(Error::invalid_argument(format!(
			"{operation} indices must use U32; found {}",
			indices.dtype().token()
		)));
	}
	if !input.runtime().same_as(indices.runtime()) {
		return Err(Error::invalid_argument(format!(
			"{operation} inputs must belong to the same runtime"
		)));
	}
	let geometry =
		AdaptivePool3dGeometry::resolve(input, output_depth, output_height, output_width, operation)?;
	let expected_shape = geometry.output_shape();
	if indices.shape() != expected_shape || output_gradient.shape() != expected_shape {
		return Err(Error::invalid_argument(format!(
			"{operation} indices and output gradient must have shape {expected_shape:?}; found {:?} and {:?}",
			indices.shape(),
			output_gradient.shape()
		)));
	}
	let input_gradient = Matrix::allocate(
		input.runtime(),
		input.shape().to_vec(),
		input.num_elements(),
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(indices.storage()),
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::write(input_gradient.storage()),
	];
	let push_constants = geometry.push_constants();
	let attributes = geometry.attributes();
	let inputs = [input, indices, output_gradient];
	let outputs = [&input_gradient];
	input.runtime().record_semantic(
		ComputeDispatch {
			kernel: KernelId::MlAdaptiveMaxPool3dBackwardF32,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: [
				geometry.input_height.div_ceil(16),
				geometry.input_width.div_ceil(16),
				geometry.batch * geometry.channels * geometry.input_depth,
			],
			indirect: None,
		},
		SemanticDispatch {
			contract,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok(input_gradient)
}

/// Adaptively average-pool an NCDHW Matrix to exactly `output_depth` ×
/// `output_height` × `output_width`.
///
/// # Errors
///
/// Returns an error unless `input` is a nonempty rank-five F32 Matrix and
/// all output extents are nonzero, or runtime recording fails.
pub fn adaptive_avg_pool_3d(
	input: &Matrix,
	output_depth: usize,
	output_height: usize,
	output_width: usize,
) -> Result<Matrix> {
	let output = adaptive_avg_pool_3d_dispatch(input, output_depth, output_height, output_width)?;
	autograd::record_adaptive_avg_pool_3d(input, output_depth, output_height, output_width, &output)?;
	Ok(output)
}

pub(in crate::ml) fn adaptive_avg_pool_3d_backward(
	input: &Matrix,
	output_gradient: &Matrix,
	output_depth: usize,
	output_height: usize,
	output_width: usize,
) -> Result<Matrix> {
	let contract = crate::core::operation::ml::ADAPTIVE_AVG_POOL_3D_BACKWARD;
	let operation = contract.name();
	validate_f32_same_runtime(operation, &[input, output_gradient])?;
	let geometry =
		AdaptivePool3dGeometry::resolve(input, output_depth, output_height, output_width, operation)?;
	let expected_shape = geometry.output_shape();
	if output_gradient.shape() != expected_shape {
		return Err(Error::invalid_argument(format!(
			"{operation} output gradient must have shape {expected_shape:?}; found {:?}",
			output_gradient.shape()
		)));
	}
	let input_gradient = Matrix::allocate(
		input.runtime(),
		input.shape().to_vec(),
		input.num_elements(),
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::write(input_gradient.storage()),
	];
	let push_constants = geometry.push_constants();
	let attributes = geometry.attributes();
	let inputs = [input, output_gradient];
	let outputs = [&input_gradient];
	input.runtime().record_semantic(
		ComputeDispatch {
			kernel: KernelId::MlAdaptiveAvgPool3dBackwardF32,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: [
				geometry.input_height.div_ceil(16),
				geometry.input_width.div_ceil(16),
				geometry.batch * geometry.channels * geometry.input_depth,
			],
			indirect: None,
		},
		SemanticDispatch {
			contract,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok(input_gradient)
}

// ── public unpool operations ────────────────────────────────────────────────

/// Scatter NCL pooled values back into an image of `output_length` using the
/// recorded argmax `indices`, the adjoint scatter of [`max_pool_1d`].
///
/// Positions no pooled element selected produce zero. Overlapping windows
/// resolve to the highest pooled index.
///
/// # Errors
///
/// Returns an error unless `input` and `indices` are nonempty rank-three
/// same-shape NCL Matrices on one runtime (`indices` U32), `output_length`
/// admits the declared kernel/stride/padding geometry, or runtime recording
/// fails.
pub fn max_unpool_1d(
	input: &Matrix,
	indices: &Matrix,
	kernel_size: usize,
	stride: usize,
	padding: usize,
	output_length: usize,
) -> Result<Matrix> {
	let output = max_unpool_1d_dispatch(input, indices, kernel_size, stride, padding, output_length)?;
	autograd::record_max_unpool_1d(input, indices, &output)?;
	Ok(output)
}

pub(in crate::ml) fn max_unpool_1d_backward(
	indices: &Matrix,
	output_gradient: &Matrix,
) -> Result<Matrix> {
	let contract = crate::core::operation::ml::MAX_UNPOOL_1D_BACKWARD;
	let operation = contract.name();
	validate_f32_same_runtime(operation, &[output_gradient])?;
	if indices.dtype() != DType::U32 {
		return Err(Error::invalid_argument(format!(
			"{operation} indices must use U32; found {}",
			indices.dtype().token()
		)));
	}
	if !output_gradient.runtime().same_as(indices.runtime()) {
		return Err(Error::invalid_argument(format!(
			"{operation} inputs must belong to the same runtime"
		)));
	}
	if indices.num_elements() > output_gradient.num_elements() {
		return Err(Error::invalid_argument(format!(
			"{operation} indices cannot exceed the output gradient element count"
		)));
	}
	let input_gradient = Matrix::allocate(
		output_gradient.runtime(),
		indices.shape().to_vec(),
		indices.num_elements(),
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(indices.storage()),
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::write(input_gradient.storage()),
	];
	let push_constants = [PushConstant::U32(shader_u32(
		indices.num_elements(),
		"element count",
		operation,
	)?)];
	let inputs = [indices, output_gradient];
	let outputs = [&input_gradient];
	output_gradient.runtime().record_semantic(
		ComputeDispatch {
			kernel: KernelId::MlMaxUnpool1dBackwardF32,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: [indices.num_elements() as u32 / 256 + 1, 1, 1],
			indirect: None,
		},
		SemanticDispatch {
			contract,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &[],
		},
	)?;
	Ok(input_gradient)
}

/// Scatter NCHW pooled values back into an image of `output_height` ×
/// `output_width` using the recorded argmax `indices`, the adjoint scatter of
/// [`max_pool_2d`].
///
/// # Errors
///
/// Returns an error unless `input` and `indices` are nonempty rank-four
/// same-shape NCHW Matrices on one runtime (`indices` U32), the declared
/// output geometry admits the kernel/stride/padding window grid, or runtime
/// recording fails.
pub fn max_unpool_2d(
	input: &Matrix,
	indices: &Matrix,
	kernel_size: usize,
	stride: usize,
	padding: usize,
	output_height: usize,
	output_width: usize,
) -> Result<Matrix> {
	let output = max_unpool_2d_dispatch(
		input,
		indices,
		kernel_size,
		stride,
		padding,
		output_height,
		output_width,
	)?;
	autograd::record_max_unpool_2d(input, indices, &output)?;
	Ok(output)
}

pub(in crate::ml) fn max_unpool_2d_backward(
	indices: &Matrix,
	output_gradient: &Matrix,
) -> Result<Matrix> {
	let contract = crate::core::operation::ml::MAX_UNPOOL_2D_BACKWARD;
	let operation = contract.name();
	validate_f32_same_runtime(operation, &[output_gradient])?;
	if indices.dtype() != DType::U32 {
		return Err(Error::invalid_argument(format!(
			"{operation} indices must use U32; found {}",
			indices.dtype().token()
		)));
	}
	if !output_gradient.runtime().same_as(indices.runtime()) {
		return Err(Error::invalid_argument(format!(
			"{operation} inputs must belong to the same runtime"
		)));
	}
	if indices.num_elements() > output_gradient.num_elements() {
		return Err(Error::invalid_argument(format!(
			"{operation} indices cannot exceed the output gradient element count"
		)));
	}
	let input_gradient = Matrix::allocate(
		output_gradient.runtime(),
		indices.shape().to_vec(),
		indices.num_elements(),
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(indices.storage()),
		BufferBinding::read(output_gradient.storage()),
		BufferBinding::write(input_gradient.storage()),
	];
	let push_constants = [PushConstant::U32(shader_u32(
		indices.num_elements(),
		"element count",
		operation,
	)?)];
	let inputs = [indices, output_gradient];
	let outputs = [&input_gradient];
	output_gradient.runtime().record_semantic(
		ComputeDispatch {
			kernel: KernelId::MlMaxUnpool2dBackwardF32,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: [indices.num_elements() as u32 / 256 + 1, 1, 1],
			indirect: None,
		},
		SemanticDispatch {
			contract,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &[],
		},
	)?;
	Ok(input_gradient)
}

fn max_pool_3d_dispatch(
	input: &Matrix,
	kernel_size: usize,
	stride: usize,
	padding: usize,
) -> Result<(Matrix, Matrix)> {
	let contract = crate::core::operation::ml::MAX_POOL_3D;
	let operation = contract.name();
	validate_f32_same_runtime(operation, &[input])?;
	let geometry = Pool3dGeometry::resolve(input, kernel_size, stride, padding, operation)?;
	let output_shape = geometry.output_shape();
	let output_count = output_shape
		.iter()
		.try_fold(1_usize, |product, extent| product.checked_mul(*extent))
		.ok_or_else(|| Error::invalid_argument(format!("{operation} output size overflows usize")))?;
	shader_u32(output_count, "output element count", operation)?;
	let output = Matrix::allocate(
		input.runtime(),
		output_shape.to_vec(),
		output_count,
		DType::F32,
	)?;
	let indices = Matrix::allocate(
		input.runtime(),
		output_shape.to_vec(),
		output_count,
		DType::U32,
	)?;
	let buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::write(output.storage()),
		BufferBinding::write(indices.storage()),
	];
	let push_constants = geometry.push_constants();
	let attributes = geometry.attributes();
	let inputs = [input];
	let outputs = [&output, &indices];
	input.runtime().record_semantic(
		ComputeDispatch {
			kernel: KernelId::MlMaxPool3dF32,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: [
				geometry.output_height.div_ceil(16),
				geometry.output_width.div_ceil(16),
				geometry.batch * geometry.channels * geometry.output_depth,
			],
			indirect: None,
		},
		SemanticDispatch {
			contract,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok((output, indices))
}

fn avg_pool_3d_dispatch(
	input: &Matrix,
	kernel_size: usize,
	stride: usize,
	padding: usize,
) -> Result<Matrix> {
	let contract = crate::core::operation::ml::AVG_POOL_3D;
	let operation = contract.name();
	validate_f32_same_runtime(operation, &[input])?;
	let geometry = Pool3dGeometry::resolve(input, kernel_size, stride, padding, operation)?;
	let output_shape = geometry.output_shape();
	let output_count = output_shape
		.iter()
		.try_fold(1_usize, |product, extent| product.checked_mul(*extent))
		.ok_or_else(|| Error::invalid_argument(format!("{operation} output size overflows usize")))?;
	shader_u32(output_count, "output element count", operation)?;
	let output = Matrix::allocate(
		input.runtime(),
		output_shape.to_vec(),
		output_count,
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::write(output.storage()),
	];
	let push_constants = geometry.push_constants();
	let attributes = geometry.attributes();
	let inputs = [input];
	let outputs = [&output];
	input.runtime().record_semantic(
		ComputeDispatch {
			kernel: KernelId::MlAvgPool3dF32,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: [
				geometry.output_height.div_ceil(16),
				geometry.output_width.div_ceil(16),
				geometry.batch * geometry.channels * geometry.output_depth,
			],
			indirect: None,
		},
		SemanticDispatch {
			contract,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok(output)
}

fn adaptive_max_pool_2d_dispatch(
	input: &Matrix,
	output_height: usize,
	output_width: usize,
) -> Result<(Matrix, Matrix)> {
	let contract = crate::core::operation::ml::ADAPTIVE_MAX_POOL_2D;
	let operation = contract.name();
	validate_f32_same_runtime(operation, &[input])?;
	let geometry = AdaptivePool2dGeometry::resolve(input, output_height, output_width, operation)?;
	let output_shape = [
		geometry.batch as usize,
		geometry.channels as usize,
		geometry.output_height as usize,
		geometry.output_width as usize,
	];
	let output_count = output_shape
		.iter()
		.try_fold(1_usize, |product, extent| product.checked_mul(*extent))
		.ok_or_else(|| Error::invalid_argument(format!("{operation} output size overflows usize")))?;
	shader_u32(output_count, "output element count", operation)?;
	let output = Matrix::allocate(
		input.runtime(),
		output_shape.to_vec(),
		output_count,
		DType::F32,
	)?;
	let indices = Matrix::allocate(
		input.runtime(),
		output_shape.to_vec(),
		output_count,
		DType::U32,
	)?;
	let buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::write(output.storage()),
		BufferBinding::write(indices.storage()),
	];
	let push_constants = geometry.push_constants();
	let attributes = geometry.attributes();
	let inputs = [input];
	let outputs = [&output, &indices];
	input.runtime().record_semantic(
		ComputeDispatch {
			kernel: KernelId::MlAdaptiveMaxPool2dF32,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: [
				geometry.output_height.div_ceil(16),
				geometry.output_width.div_ceil(16),
				geometry.batch * geometry.channels,
			],
			indirect: None,
		},
		SemanticDispatch {
			contract,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok((output, indices))
}

fn adaptive_max_pool_3d_dispatch(
	input: &Matrix,
	output_depth: usize,
	output_height: usize,
	output_width: usize,
) -> Result<(Matrix, Matrix)> {
	let contract = crate::core::operation::ml::ADAPTIVE_MAX_POOL_3D;
	let operation = contract.name();
	validate_f32_same_runtime(operation, &[input])?;
	let geometry =
		AdaptivePool3dGeometry::resolve(input, output_depth, output_height, output_width, operation)?;
	let output_shape = geometry.output_shape();
	let output_count = output_shape
		.iter()
		.try_fold(1_usize, |product, extent| product.checked_mul(*extent))
		.ok_or_else(|| Error::invalid_argument(format!("{operation} output size overflows usize")))?;
	shader_u32(output_count, "output element count", operation)?;
	let output = Matrix::allocate(
		input.runtime(),
		output_shape.to_vec(),
		output_count,
		DType::F32,
	)?;
	let indices = Matrix::allocate(
		input.runtime(),
		output_shape.to_vec(),
		output_count,
		DType::U32,
	)?;
	let buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::write(output.storage()),
		BufferBinding::write(indices.storage()),
	];
	let push_constants = geometry.push_constants();
	let attributes = geometry.attributes();
	let inputs = [input];
	let outputs = [&output, &indices];
	input.runtime().record_semantic(
		ComputeDispatch {
			kernel: KernelId::MlAdaptiveMaxPool3dF32,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: [
				geometry.output_height.div_ceil(16),
				geometry.output_width.div_ceil(16),
				geometry.batch * geometry.channels * geometry.output_depth,
			],
			indirect: None,
		},
		SemanticDispatch {
			contract,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok((output, indices))
}

fn adaptive_avg_pool_3d_dispatch(
	input: &Matrix,
	output_depth: usize,
	output_height: usize,
	output_width: usize,
) -> Result<Matrix> {
	let contract = crate::core::operation::ml::ADAPTIVE_AVG_POOL_3D;
	let operation = contract.name();
	validate_f32_same_runtime(operation, &[input])?;
	let geometry =
		AdaptivePool3dGeometry::resolve(input, output_depth, output_height, output_width, operation)?;
	let output_shape = geometry.output_shape();
	let output_count = output_shape
		.iter()
		.try_fold(1_usize, |product, extent| product.checked_mul(*extent))
		.ok_or_else(|| Error::invalid_argument(format!("{operation} output size overflows usize")))?;
	shader_u32(output_count, "output element count", operation)?;
	let output = Matrix::allocate(
		input.runtime(),
		output_shape.to_vec(),
		output_count,
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::write(output.storage()),
	];
	let push_constants = geometry.push_constants();
	let attributes = geometry.attributes();
	let inputs = [input];
	let outputs = [&output];
	input.runtime().record_semantic(
		ComputeDispatch {
			kernel: KernelId::MlAdaptiveAvgPool3dF32,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: [
				geometry.output_height.div_ceil(16),
				geometry.output_width.div_ceil(16),
				geometry.batch * geometry.channels * geometry.output_depth,
			],
			indirect: None,
		},
		SemanticDispatch {
			contract,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok(output)
}

fn max_unpool_1d_dispatch(
	input: &Matrix,
	indices: &Matrix,
	kernel_size: usize,
	stride: usize,
	padding: usize,
	output_length: usize,
) -> Result<Matrix> {
	let contract = crate::core::operation::ml::MAX_UNPOOL_1D;
	let operation = contract.name();
	validate_f32_same_runtime(operation, &[input])?;
	if indices.dtype() != DType::U32 {
		return Err(Error::invalid_argument(format!(
			"{operation} indices must use U32; found {}",
			indices.dtype().token()
		)));
	}
	if !input.runtime().same_as(indices.runtime()) {
		return Err(Error::invalid_argument(format!(
			"{operation} inputs must belong to the same runtime"
		)));
	}
	let [batch, channels, input_length] = input.shape() else {
		return Err(Error::invalid_argument(format!(
			"{operation} requires NCL rank-three input; found {:?}",
			input.shape()
		)));
	};
	if indices.shape() != input.shape() {
		return Err(Error::invalid_argument(format!(
			"{operation} indices must match the pooled input shape {:?}; found {:?}",
			input.shape(),
			indices.shape()
		)));
	}
	if *batch == 0
		|| *channels == 0
		|| *input_length == 0
		|| kernel_size == 0
		|| stride == 0
		|| output_length == 0
	{
		return Err(Error::invalid_argument(format!(
			"{operation} requires nonzero extents, kernel size, stride, and output length"
		)));
	}
	let expanded = input_length
		.saturating_sub(1)
		.checked_mul(stride)
		.and_then(|extent| extent.checked_add(kernel_size))
		.ok_or_else(|| {
			Error::invalid_argument(format!("{operation} expanded length overflows usize"))
		})?;
	let doubled_padding = padding.checked_mul(2).ok_or_else(|| {
		Error::invalid_argument(format!("{operation} doubled padding overflows usize"))
	})?;
	let minimum_output = expanded.checked_sub(doubled_padding).ok_or_else(|| {
		Error::invalid_argument(format!(
			"{operation} padding removes the complete unpooled output"
		))
	})?;
	if !(minimum_output..minimum_output + stride).contains(&output_length) {
		return Err(Error::invalid_argument(format!(
			"{operation} output length {output_length} is outside the {minimum_output}..{} admitted by the window geometry",
			minimum_output + stride - 1
		)));
	}
	let output_count = batch
		.checked_mul(*channels)
		.and_then(|count| count.checked_mul(output_length))
		.ok_or_else(|| Error::invalid_argument(format!("{operation} output size overflows usize")))?;
	shader_u32(output_count, "output element count", operation)?;
	let output = Matrix::allocate(
		input.runtime(),
		vec![*batch, *channels, output_length],
		output_count,
		DType::F32,
	)?;
	let batch_u32 = shader_u32(*batch, "batch size", operation)?;
	let channels_u32 = shader_u32(*channels, "channel count", operation)?;
	batch_u32.checked_mul(channels_u32).ok_or_else(|| {
		Error::invalid_argument(format!("{operation} batch-channel count exceeds u32"))
	})?;
	let buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::read(indices.storage()),
		BufferBinding::write(output.storage()),
	];
	let push_constants = [
		PushConstant::U32(batch_u32),
		PushConstant::U32(channels_u32),
		PushConstant::U32(shader_u32(output_length, "output length", operation)?),
		PushConstant::U32(shader_u32(*input_length, "input length", operation)?),
		PushConstant::U32(shader_u32(kernel_size, "kernel size", operation)?),
		PushConstant::U32(shader_u32(stride, "stride", operation)?),
		PushConstant::U32(shader_u32(padding, "padding", operation)?),
	];
	let attributes = [
		OpAttribute::UnsignedInteger {
			name: "kernel_size".into(),
			value: kernel_size as u64,
		},
		OpAttribute::UnsignedInteger {
			name: "stride".into(),
			value: stride as u64,
		},
		OpAttribute::UnsignedInteger {
			name: "padding".into(),
			value: padding as u64,
		},
	];
	let inputs = [input, indices];
	let outputs = [&output];
	input.runtime().record_semantic(
		ComputeDispatch {
			kernel: KernelId::MlMaxUnpool1dF32,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: [
				shader_u32(output_length, "output length", operation)?.div_ceil(256),
				batch_u32 * channels_u32,
				1,
			],
			indirect: None,
		},
		SemanticDispatch {
			contract,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok(output)
}

fn max_unpool_2d_dispatch(
	input: &Matrix,
	indices: &Matrix,
	kernel_size: usize,
	stride: usize,
	padding: usize,
	output_height: usize,
	output_width: usize,
) -> Result<Matrix> {
	let contract = crate::core::operation::ml::MAX_UNPOOL_2D;
	let operation = contract.name();
	validate_f32_same_runtime(operation, &[input])?;
	if indices.dtype() != DType::U32 {
		return Err(Error::invalid_argument(format!(
			"{operation} indices must use U32; found {}",
			indices.dtype().token()
		)));
	}
	if !input.runtime().same_as(indices.runtime()) {
		return Err(Error::invalid_argument(format!(
			"{operation} inputs must belong to the same runtime"
		)));
	}
	let [batch, channels, input_height, input_width] = input.shape() else {
		return Err(Error::invalid_argument(format!(
			"{operation} requires NCHW rank-four input; found {:?}",
			input.shape()
		)));
	};
	if indices.shape() != input.shape() {
		return Err(Error::invalid_argument(format!(
			"{operation} indices must match the pooled input shape {:?}; found {:?}",
			input.shape(),
			indices.shape()
		)));
	}
	if *batch == 0
		|| *channels == 0
		|| *input_height == 0
		|| *input_width == 0
		|| kernel_size == 0
		|| stride == 0
		|| output_height == 0
		|| output_width == 0
	{
		return Err(Error::invalid_argument(format!(
			"{operation} requires nonzero extents, kernel size, stride, and output size"
		)));
	}
	let doubled_padding = padding.checked_mul(2).ok_or_else(|| {
		Error::invalid_argument(format!("{operation} doubled padding overflows usize"))
	})?;
	for (label, input_extent, output_extent) in [
		("height", *input_height, output_height),
		("width", *input_width, output_width),
	] {
		let expanded = input_extent
			.saturating_sub(1)
			.checked_mul(stride)
			.and_then(|extent| extent.checked_add(kernel_size))
			.ok_or_else(|| {
				Error::invalid_argument(format!("{operation} expanded {label} overflows usize"))
			})?;
		let minimum = expanded.checked_sub(doubled_padding).ok_or_else(|| {
			Error::invalid_argument(format!(
				"{operation} padding removes the complete unpooled {label}"
			))
		})?;
		if !(minimum..minimum + stride).contains(&output_extent) {
			return Err(Error::invalid_argument(format!(
				"{operation} output {label} {output_extent} is outside the {minimum}..{} admitted by the window geometry",
				minimum + stride - 1
			)));
		}
	}
	let output_count = batch
		.checked_mul(*channels)
		.and_then(|count| count.checked_mul(output_height))
		.and_then(|count| count.checked_mul(output_width))
		.ok_or_else(|| Error::invalid_argument(format!("{operation} output size overflows usize")))?;
	shader_u32(output_count, "output element count", operation)?;
	let output = Matrix::allocate(
		input.runtime(),
		vec![*batch, *channels, output_height, output_width],
		output_count,
		DType::F32,
	)?;
	let batch_u32 = shader_u32(*batch, "batch size", operation)?;
	let channels_u32 = shader_u32(*channels, "channel count", operation)?;
	batch_u32.checked_mul(channels_u32).ok_or_else(|| {
		Error::invalid_argument(format!("{operation} batch-channel count exceeds u32"))
	})?;
	let buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::read(indices.storage()),
		BufferBinding::write(output.storage()),
	];
	let push_constants = [
		PushConstant::U32(batch_u32),
		PushConstant::U32(channels_u32),
		PushConstant::U32(shader_u32(output_height, "output height", operation)?),
		PushConstant::U32(shader_u32(output_width, "output width", operation)?),
		PushConstant::U32(shader_u32(*input_height, "input height", operation)?),
		PushConstant::U32(shader_u32(*input_width, "input width", operation)?),
		PushConstant::U32(shader_u32(kernel_size, "kernel size", operation)?),
		PushConstant::U32(shader_u32(stride, "stride", operation)?),
		PushConstant::U32(shader_u32(padding, "padding", operation)?),
	];
	let attributes = [
		OpAttribute::UnsignedInteger {
			name: "kernel_size".into(),
			value: kernel_size as u64,
		},
		OpAttribute::UnsignedInteger {
			name: "stride".into(),
			value: stride as u64,
		},
		OpAttribute::UnsignedInteger {
			name: "padding".into(),
			value: padding as u64,
		},
	];
	let inputs = [input, indices];
	let outputs = [&output];
	input.runtime().record_semantic(
		ComputeDispatch {
			kernel: KernelId::MlMaxUnpool2dF32,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: [
				shader_u32(output_height, "output height", operation)?.div_ceil(16),
				shader_u32(output_width, "output width", operation)?.div_ceil(16),
				batch_u32 * channels_u32,
			],
			indirect: None,
		},
		SemanticDispatch {
			contract,
			inputs: &inputs,
			outputs: &outputs,
			attributes: &attributes,
		},
	)?;
	Ok(output)
}
