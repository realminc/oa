//! ML-owned Matrix unfold/fold (im2col/col2im) operations.

use crate::{
	DType, Error, Matrix, OpAttribute, Result,
	runtime::{BufferBinding, ComputeDispatch, KernelId, PushConstant, SemanticDispatch},
};

use crate::ml::autograd;

use super::super::validation::{shader_u32, validate_f32_same_engine};

#[derive(Clone, Copy)]
struct FoldGeometry {
	batch_size: u32,
	input_channels: u32,
	input_height: u32,
	input_width: u32,
	kernel_size: u32,
	stride: u32,
	padding: u32,
	dilation: u32,
	output_height: u32,
	output_width: u32,
}

impl FoldGeometry {
	fn sliding_extent(
		input: usize,
		kernel_size: usize,
		stride: usize,
		padding: usize,
		dilation: usize,
		operation: &'static str,
	) -> Result<usize> {
		let padded = input
			.checked_add(padding.checked_mul(2).ok_or_else(|| {
				Error::invalid_argument(format!("{operation} doubled padding overflows usize"))
			})?)
			.ok_or_else(|| {
				Error::invalid_argument(format!("{operation} padded extent overflows usize"))
			})?;
		let effective_kernel = dilation
			.checked_mul(kernel_size.saturating_sub(1))
			.and_then(|span| span.checked_add(1))
			.ok_or_else(|| {
				Error::invalid_argument(format!("{operation} effective kernel overflows usize"))
			})?;
		if padded < effective_kernel {
			return Err(Error::invalid_argument(format!(
				"{operation} effective kernel does not fit the padded input"
			)));
		}
		Ok((padded - effective_kernel) / stride + 1)
	}

	fn resolve(
		input_channels: usize,
		input_height: usize,
		input_width: usize,
		geometry: [usize; 4],
		batch_size: usize,
		operation: &'static str,
	) -> Result<Self> {
		let [kernel_size, stride, padding, dilation] = geometry;
		if input_channels == 0
			|| input_height == 0
			|| input_width == 0
			|| kernel_size == 0
			|| stride == 0
			|| dilation == 0
			|| batch_size == 0
		{
			return Err(Error::invalid_argument(format!(
				"{operation} requires nonzero batch, channels, spatial extents, kernel, stride, and dilation"
			)));
		}
		let output_height = Self::sliding_extent(
			input_height,
			kernel_size,
			stride,
			padding,
			dilation,
			operation,
		)?;
		let output_width = Self::sliding_extent(
			input_width,
			kernel_size,
			stride,
			padding,
			dilation,
			operation,
		)?;
		for (label, value) in [
			("input height", input_height),
			("input width", input_width),
			("output height", output_height),
			("output width", output_width),
			("kernel size", kernel_size),
			("stride", stride),
			("padding", padding),
			("dilation", dilation),
		] {
			if value > i32::MAX as usize {
				return Err(Error::invalid_argument(format!(
					"{operation} {label} exceeds the signed shader coordinate ABI"
				)));
			}
		}
		Ok(Self {
			batch_size: shader_u32(batch_size, "batch size", operation)?,
			input_channels: shader_u32(input_channels, "input channel count", operation)?,
			input_height: shader_u32(input_height, "input height", operation)?,
			input_width: shader_u32(input_width, "input width", operation)?,
			kernel_size: shader_u32(kernel_size, "kernel size", operation)?,
			stride: shader_u32(stride, "stride", operation)?,
			padding: shader_u32(padding, "padding", operation)?,
			dilation: shader_u32(dilation, "dilation", operation)?,
			output_height: shader_u32(output_height, "output height", operation)?,
			output_width: shader_u32(output_width, "output width", operation)?,
		})
	}

	fn output_length(self) -> usize {
		self.output_height as usize * self.output_width as usize
	}

	fn dimensions(self) -> [PushConstant; 10] {
		[
			PushConstant::U32(self.batch_size),
			PushConstant::U32(self.input_channels),
			PushConstant::U32(self.input_height),
			PushConstant::U32(self.input_width),
			PushConstant::U32(self.kernel_size),
			PushConstant::U32(self.stride),
			PushConstant::U32(self.padding),
			PushConstant::U32(self.dilation),
			PushConstant::U32(self.output_height),
			PushConstant::U32(self.output_width),
		]
	}

	fn unfold_attributes(self) -> [OpAttribute; 4] {
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
			OpAttribute::UnsignedInteger {
				name: "dilation".into(),
				value: u64::from(self.dilation),
			},
		]
	}

	fn fold_attributes(self) -> [OpAttribute; 6] {
		[
			OpAttribute::UnsignedInteger {
				name: "output_height".into(),
				value: u64::from(self.input_height),
			},
			OpAttribute::UnsignedInteger {
				name: "output_width".into(),
				value: u64::from(self.input_width),
			},
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
			OpAttribute::UnsignedInteger {
				name: "dilation".into(),
				value: u64::from(self.dilation),
			},
		]
	}
}

/// Extract sliding NCHW kernel windows into a `[N, C * kernel², L]` column Matrix.
///
/// Each column holds one `kernel_size × kernel_size` patch per input channel in
/// `(channel, kernel_y, kernel_x)` order, laid out over the
/// `L = output_height * output_width` sliding positions where
/// `output = floor((input + 2 * padding - dilation * (kernel - 1) - 1) / stride) + 1`
/// on every spatial axis. Out-of-bounds taps read as zero, matching the donor
/// im2col contract.
///
/// # Errors
///
/// Returns an error unless `input` is a nonempty NCHW rank-four F32 Matrix,
/// kernel/stride/dilation are nonzero, the effective kernel fits the padded
/// input, every extent fits the shader ABI, or recording fails.
pub fn unfold(
	input: &Matrix,
	kernel_size: usize,
	stride: usize,
	padding: usize,
	dilation: usize,
) -> Result<Matrix> {
	validate_f32_same_engine(crate::core::operation::ml::UNFOLD.name(), &[input])?;
	let output = unfold_dispatch(input, kernel_size, stride, padding, dilation)?;
	autograd::record_unfold(input, kernel_size, stride, padding, dilation, &output)?;
	Ok(output)
}

/// Combine `[N, C * kernel², L]` sliding columns into an overlapping NCHW Matrix.
///
/// This is the deterministic adjoint of [`unfold`]: overlapping taps are summed
/// per output element without atomics. `output_height` and `output_width` name
/// the reconstructed image extent and must reproduce exactly
/// `L = output_height_positions * output_width_positions` sliding columns under
/// the same kernel geometry.
///
/// # Errors
///
/// Returns an error unless `input` is a nonempty `[N, C * kernel², L]`
/// rank-three F32 Matrix whose channel count is divisible by `kernel²` and
/// whose column count equals the sliding extent of the requested output image,
/// kernel/stride/dilation are nonzero, extents fit the shader ABI, or
/// recording fails.
pub fn fold(
	input: &Matrix,
	output_height: usize,
	output_width: usize,
	kernel_size: usize,
	stride: usize,
	padding: usize,
	dilation: usize,
) -> Result<Matrix> {
	let columns = fold_dispatch(
		input,
		output_height,
		output_width,
		kernel_size,
		stride,
		padding,
		dilation,
	)?;
	autograd::record_fold(input, kernel_size, stride, padding, dilation, &columns)?;
	Ok(columns)
}

pub(in crate::ml) fn unfold_backward(
	input: &Matrix,
	output_gradient: &Matrix,
	kernel_size: usize,
	stride: usize,
	padding: usize,
	dilation: usize,
) -> Result<Matrix> {
	let operation = crate::core::operation::ml::FOLD.name();
	validate_f32_same_engine(operation, &[input, output_gradient])?;
	let [_, _, input_height, input_width] = input.shape() else {
		return Err(Error::invalid_argument(format!(
			"{operation} adjoint requires NCHW rank-four saved input; found {:?}",
			input.shape()
		)));
	};
	let output = fold_dispatch(
		output_gradient,
		*input_height,
		*input_width,
		kernel_size,
		stride,
		padding,
		dilation,
	)?;
	if output.shape() != input.shape() {
		return Err(Error::invalid_argument(format!(
			"{operation} reconstructed adjoint shape {:?} disagrees with saved input {:?}",
			output.shape(),
			input.shape()
		)));
	}
	Ok(output)
}

pub(in crate::ml) fn fold_backward(
	input: &Matrix,
	output_gradient: &Matrix,
	kernel_size: usize,
	stride: usize,
	padding: usize,
	dilation: usize,
) -> Result<Matrix> {
	let operation = crate::core::operation::ml::UNFOLD.name();
	validate_f32_same_engine(operation, &[input, output_gradient])?;
	if output_gradient.shape().len() != 4 {
		return Err(Error::invalid_argument(format!(
			"{operation} output gradient must be NCHW rank four; found {:?}",
			output_gradient.shape()
		)));
	}
	let columns = unfold_dispatch(output_gradient, kernel_size, stride, padding, dilation)?;
	if columns.shape() != input.shape() {
		return Err(Error::invalid_argument(format!(
			"{operation} reconstructed column shape {:?} disagrees with saved input {:?}",
			columns.shape(),
			input.shape()
		)));
	}
	Ok(columns)
}

fn unfold_dispatch(
	input: &Matrix,
	kernel_size: usize,
	stride: usize,
	padding: usize,
	dilation: usize,
) -> Result<Matrix> {
	let contract = crate::core::operation::ml::UNFOLD;
	let operation = contract.name();
	let [batch_size, input_channels, input_height, input_width] = input.shape() else {
		return Err(Error::invalid_argument(format!(
			"{operation} requires NCHW rank-four input; found {:?}",
			input.shape()
		)));
	};
	let geometry = FoldGeometry::resolve(
		*input_channels,
		*input_height,
		*input_width,
		[kernel_size, stride, padding, dilation],
		*batch_size,
		operation,
	)?;
	let column_size = input_channels
		.checked_mul(kernel_size)
		.and_then(|count| count.checked_mul(kernel_size))
		.ok_or_else(|| Error::invalid_argument(format!("{operation} column size overflows usize")))?;
	let element_count = batch_size
		.checked_mul(column_size)
		.and_then(|count| count.checked_mul(geometry.output_length()))
		.ok_or_else(|| {
			Error::invalid_argument(format!("{operation} column element count overflows usize"))
		})?;
	let output = Matrix::allocate(
		input.engine_handle(),
		vec![*batch_size, column_size, geometry.output_length()],
		element_count,
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::write(output.storage()),
	];
	let constants = geometry.dimensions();
	let attributes = geometry.unfold_attributes();
	let kernel = KernelId::MlUnfold2dF32;
	{
		let inputs: &[&Matrix] = &[input];
		let outputs: &[&Matrix] = &[&output];
		let attributes: &[OpAttribute] = &attributes;
		let dispatch = ComputeDispatch {
			kernel,
			buffers: &buffers,
			push_constants: &constants,
			workgroups: kernel.linear_workgroups(shader_u32(
				element_count,
				"column element count",
				operation,
			)?),
			indirect: None,
		};
		input.engine_handle().record_semantic(
			dispatch,
			SemanticDispatch {
				contract,
				inputs,
				outputs,
				attributes,
			},
		)?;
	}
	Ok(output)
}

fn fold_dispatch(
	input: &Matrix,
	output_height: usize,
	output_width: usize,
	kernel_size: usize,
	stride: usize,
	padding: usize,
	dilation: usize,
) -> Result<Matrix> {
	let contract = crate::core::operation::ml::FOLD;
	let operation = contract.name();
	validate_f32_same_engine(operation, &[input])?;
	let [batch_size, column_size, output_length] = input.shape() else {
		return Err(Error::invalid_argument(format!(
			"{operation} requires [N, C * kernel², L] rank-three input; found {:?}",
			input.shape()
		)));
	};
	let kernel_area = kernel_size
		.checked_mul(kernel_size)
		.ok_or_else(|| Error::invalid_argument(format!("{operation} kernel area overflows usize")))?;
	if kernel_area == 0 || !column_size.is_multiple_of(kernel_area) {
		return Err(Error::invalid_argument(format!(
			"{operation} column channels must be a nonzero multiple of kernel²; found {column_size} and kernel {kernel_size}"
		)));
	}
	let input_channels = column_size / kernel_area;
	let geometry = FoldGeometry::resolve(
		input_channels,
		output_height,
		output_width,
		[kernel_size, stride, padding, dilation],
		*batch_size,
		operation,
	)?;
	if *output_length != geometry.output_length() {
		return Err(Error::invalid_argument(format!(
			"{operation} column count {output_length} must equal the sliding extent {}",
			geometry.output_length()
		)));
	}
	let element_count = batch_size
		.checked_mul(input_channels)
		.and_then(|count| count.checked_mul(output_height))
		.and_then(|count| count.checked_mul(output_width))
		.ok_or_else(|| {
			Error::invalid_argument(format!("{operation} image element count overflows usize"))
		})?;
	let output = Matrix::allocate(
		input.engine_handle(),
		vec![*batch_size, input_channels, output_height, output_width],
		element_count,
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(input.storage()),
		BufferBinding::write(output.storage()),
	];
	let constants = geometry.dimensions();
	let attributes = geometry.fold_attributes();
	let kernel = KernelId::MlFold2dF32;
	{
		let inputs: &[&Matrix] = &[input];
		let outputs: &[&Matrix] = &[&output];
		let attributes: &[OpAttribute] = &attributes;
		let dispatch = ComputeDispatch {
			kernel,
			buffers: &buffers,
			push_constants: &constants,
			workgroups: kernel.linear_workgroups(shader_u32(
				element_count,
				"image element count",
				operation,
			)?),
			indirect: None,
		};
		input.engine_handle().record_semantic(
			dispatch,
			SemanticDispatch {
				contract,
				inputs,
				outputs,
				attributes,
			},
		)?;
	}
	Ok(output)
}
