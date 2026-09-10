use crate::{DType, Engine, Error, Matrix, Result};

use super::super::{Module, ModuleRegistry, Parameter, matrix, random};

/// Trainable FP32 one-dimensional convolution over NCL matrices.
pub struct Conv1d {
	input_channels: usize,
	output_channels: usize,
	kernel_size: usize,
	stride: usize,
	padding: usize,
	dilation: usize,
	weight: Parameter,
	bias: Parameter,
	registry: ModuleRegistry,
}

impl Conv1d {
	/// Construct a deterministically initialized Conv1d layer.
	///
	/// Forward preserves OA's im2col plus tiled-GEMM route. Weight uses a
	/// fan-in-scaled symmetric uniform initializer and bias starts at zero.
	///
	/// # Errors
	///
	/// Returns an error when dimensions, stride, or dilation are zero, shape
	/// arithmetic overflows, or allocation/upload fails.
	#[allow(
		clippy::too_many_arguments,
		reason = "the donor Conv1d contract has five geometry fields plus a seed"
	)]
	pub fn with_seed(
		engine: &Engine,
		input_channels: usize,
		output_channels: usize,
		kernel_size: usize,
		stride: usize,
		padding: usize,
		dilation: usize,
		seed: u64,
	) -> Result<Self> {
		if input_channels == 0
			|| output_channels == 0
			|| kernel_size == 0
			|| stride == 0
			|| dilation == 0
		{
			return Err(Error::invalid_argument(
				"Conv1d requires nonzero channels, kernel size, stride, and dilation",
			));
		}
		let fan_in = input_channels
			.checked_mul(kernel_size)
			.ok_or_else(|| Error::invalid_argument("Conv1d fan-in overflows usize"))?;
		let weight_count = output_channels
			.checked_mul(fan_in)
			.ok_or_else(|| Error::invalid_argument("Conv1d weight size overflows usize"))?;
		let limit = 1.0_f32 / (fan_in as f32).sqrt();
		let values = random::symmetric_uniform(weight_count, limit, seed);
		let weight = Matrix::from_f32(
			engine,
			[output_channels, input_channels, kernel_size],
			&values,
		)?;
		let bias = Matrix::from_f32(engine, [output_channels], &vec![0.0; output_channels])?;
		Self::from_matrices(weight, bias, stride, padding, dilation)
	}

	/// Construct Conv1d from exact OIK weight and channel-bias matrices.
	///
	/// # Errors
	///
	/// Returns an error unless weight is nonempty F32 `[O, I, K]`, bias is
	/// matching same-engine F32 `[O]`, and stride/dilation are nonzero.
	pub fn from_matrices(
		weight: Matrix,
		bias: Matrix,
		stride: usize,
		padding: usize,
		dilation: usize,
	) -> Result<Self> {
		let [output_channels, input_channels, kernel_size] = weight.shape() else {
			return Err(Error::invalid_argument(
				"Conv1d weight must have OIK rank three",
			));
		};
		if *output_channels == 0
			|| *input_channels == 0
			|| *kernel_size == 0
			|| stride == 0
			|| dilation == 0
			|| bias.shape() != [*output_channels]
			|| weight.dtype() != DType::F32
			|| bias.dtype() != DType::F32
			|| !weight.engine_handle().same_as(bias.engine_handle())
		{
			return Err(Error::invalid_argument(
				"Conv1d requires nonempty same-engine F32 OIK weight, matching bias, and nonzero stride/dilation",
			));
		}
		let input_channels = *input_channels;
		let output_channels = *output_channels;
		let kernel_size = *kernel_size;
		let weight = Parameter::new("weight", weight)?;
		let bias = Parameter::new("bias", bias)?;
		let mut registry = ModuleRegistry::new();
		registry.register_parameter("weight", weight.clone())?;
		registry.register_parameter("bias", bias.clone())?;
		Ok(Self {
			input_channels,
			output_channels,
			kernel_size,
			stride,
			padding,
			dilation,
			weight,
			bias,
			registry,
		})
	}

	/// Apply Conv1d without submitting or waiting.
	///
	/// # Errors
	///
	/// Returns an error from the underlying Matrix operation.
	pub fn forward(&self, input: &Matrix) -> Result<Matrix> {
		let (weight, weight_version, _) = self.weight.snapshot();
		let (bias, bias_version, _) = self.bias.snapshot();
		matrix::conv_1d_parameterized(
			input,
			(self.weight.clone(), weight, weight_version),
			(self.bias.clone(), bias, bias_version),
			self.stride,
			self.padding,
			self.dilation,
		)
	}

	/// Return the input channel count.
	pub const fn input_channels(&self) -> usize {
		self.input_channels
	}

	/// Return the output channel count.
	pub const fn output_channels(&self) -> usize {
		self.output_channels
	}

	/// Return the kernel extent.
	pub const fn kernel_size(&self) -> usize {
		self.kernel_size
	}

	/// Return the spatial stride.
	pub const fn stride(&self) -> usize {
		self.stride
	}

	/// Return the symmetric zero-padding extent.
	pub const fn padding(&self) -> usize {
		self.padding
	}

	/// Return the within-kernel dilation.
	pub const fn dilation(&self) -> usize {
		self.dilation
	}

	/// Return the stable OIK weight parameter.
	pub fn weight(&self) -> Parameter {
		self.weight.clone()
	}

	/// Return the stable channel-bias parameter.
	pub fn bias(&self) -> Parameter {
		self.bias.clone()
	}

	/// Return parameters in deterministic weight, bias order.
	pub fn parameters(&self) -> [Parameter; 2] {
		[self.weight.clone(), self.bias.clone()]
	}
}

impl Module for Conv1d {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		Conv1d::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}

/// Trainable bias-free FP32 one-dimensional transposed convolution over NCL matrices.
pub struct ConvTranspose1d {
	input_channels: usize,
	output_channels: usize,
	kernel_size: usize,
	stride: usize,
	padding: usize,
	weight: Parameter,
	registry: ModuleRegistry,
}

impl ConvTranspose1d {
	/// Construct a deterministically initialized ConvTranspose1d layer.
	///
	/// Weight uses the donor `[input_channels, output_channels, kernel]` layout,
	/// a fan-in-scaled symmetric uniform initializer, and no bias parameter.
	///
	/// # Errors
	///
	/// Returns an error when channels, kernel size, or stride are zero, shape
	/// arithmetic overflows, or parameter allocation/upload fails.
	#[allow(
		clippy::too_many_arguments,
		reason = "the donor ConvTranspose1d contract has five geometry fields plus a seed"
	)]
	pub fn with_seed(
		engine: &Engine,
		input_channels: usize,
		output_channels: usize,
		kernel_size: usize,
		stride: usize,
		padding: usize,
		seed: u64,
	) -> Result<Self> {
		if input_channels == 0 || output_channels == 0 || kernel_size == 0 || stride == 0 {
			return Err(Error::invalid_argument(
				"ConvTranspose1d requires nonzero channels, kernel size, and stride",
			));
		}
		let fan_in = output_channels
			.checked_mul(kernel_size)
			.ok_or_else(|| Error::invalid_argument("ConvTranspose1d fan-in overflows usize"))?;
		let weight_count = input_channels.checked_mul(fan_in).ok_or_else(|| {
			Error::invalid_argument("ConvTranspose1d weight size overflows usize")
		})?;
		let limit = 1.0_f32 / (fan_in as f32).sqrt();
		let values = random::symmetric_uniform(weight_count, limit, seed);
		let weight = Matrix::from_f32(
			engine,
			[input_channels, output_channels, kernel_size],
			&values,
		)?;
		Self::from_matrix(weight, stride, padding)
	}

	/// Construct ConvTranspose1d from an exact IOK weight matrix.
	///
	/// # Errors
	///
	/// Returns an error unless weight is nonempty F32 `[I, O, K]` and stride is
	/// nonzero.
	pub fn from_matrix(weight: Matrix, stride: usize, padding: usize) -> Result<Self> {
		let [input_channels, output_channels, kernel_size] = weight.shape() else {
			return Err(Error::invalid_argument(
				"ConvTranspose1d weight must have IOK rank three",
			));
		};
		if *input_channels == 0
			|| *output_channels == 0
			|| *kernel_size == 0
			|| stride == 0
			|| weight.dtype() != DType::F32
		{
			return Err(Error::invalid_argument(
				"ConvTranspose1d requires nonempty F32 IOK weight and nonzero stride",
			));
		}
		let input_channels = *input_channels;
		let output_channels = *output_channels;
		let kernel_size = *kernel_size;
		let weight = Parameter::new("weight", weight)?;
		let mut registry = ModuleRegistry::new();
		registry.register_parameter("weight", weight.clone())?;
		Ok(Self {
			input_channels,
			output_channels,
			kernel_size,
			stride,
			padding,
			weight,
			registry,
		})
	}

	/// Apply ConvTranspose1d without submitting or waiting.
	///
	/// # Errors
	///
	/// Returns an error from the underlying Matrix operation.
	pub fn forward(&self, input: &Matrix) -> Result<Matrix> {
		let (weight, weight_version, _) = self.weight.snapshot();
		matrix::conv_transpose_1d_parameterized(
			input,
			(self.weight.clone(), weight, weight_version),
			self.stride,
			self.padding,
			1,
		)
	}

	/// Return the input channel count.
	pub const fn input_channels(&self) -> usize {
		self.input_channels
	}

	/// Return the output channel count.
	pub const fn output_channels(&self) -> usize {
		self.output_channels
	}

	/// Return the kernel extent.
	pub const fn kernel_size(&self) -> usize {
		self.kernel_size
	}

	/// Return the spatial stride.
	pub const fn stride(&self) -> usize {
		self.stride
	}

	/// Return the symmetric zero-padding extent.
	pub const fn padding(&self) -> usize {
		self.padding
	}

	/// Return the stable IOK weight parameter.
	pub fn weight(&self) -> Parameter {
		self.weight.clone()
	}

	/// Return the only trainable parameter.
	pub fn parameters(&self) -> [Parameter; 1] {
		[self.weight.clone()]
	}
}

impl Module for ConvTranspose1d {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		ConvTranspose1d::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}

/// Trainable FP32 two-dimensional transposed convolution over NCHW matrices.
pub struct ConvTranspose2d {
	input_channels: usize,
	output_channels: usize,
	kernel_size: usize,
	stride: usize,
	padding: usize,
	weight: Parameter,
	bias: Parameter,
	registry: ModuleRegistry,
}

impl ConvTranspose2d {
	/// Construct a deterministically initialized ConvTranspose2d layer.
	///
	/// Weight uses `[input_channels, output_channels, kernel, kernel]` layout,
	/// a fan-in-scaled symmetric uniform initializer, and zero channel bias.
	///
	/// # Errors
	///
	/// Returns an error when channels, kernel size, or stride are zero, shape
	/// arithmetic overflows, or parameter allocation/upload fails.
	#[allow(
		clippy::too_many_arguments,
		reason = "the donor ConvTranspose2d contract has five geometry fields plus a seed"
	)]
	pub fn with_seed(
		engine: &Engine,
		input_channels: usize,
		output_channels: usize,
		kernel_size: usize,
		stride: usize,
		padding: usize,
		seed: u64,
	) -> Result<Self> {
		if input_channels == 0 || output_channels == 0 || kernel_size == 0 || stride == 0 {
			return Err(Error::invalid_argument(
				"ConvTranspose2d requires nonzero channels, kernel size, and stride",
			));
		}
		let fan_in = output_channels
			.checked_mul(kernel_size)
			.and_then(|count| count.checked_mul(kernel_size))
			.ok_or_else(|| Error::invalid_argument("ConvTranspose2d fan-in overflows usize"))?;
		let weight_count = input_channels.checked_mul(fan_in).ok_or_else(|| {
			Error::invalid_argument("ConvTranspose2d weight size overflows usize")
		})?;
		let limit = 1.0_f32 / (fan_in as f32).sqrt();
		let values = random::symmetric_uniform(weight_count, limit, seed);
		let weight = Matrix::from_f32(
			engine,
			[input_channels, output_channels, kernel_size, kernel_size],
			&values,
		)?;
		let bias = Matrix::from_f32(engine, [output_channels], &vec![0.0; output_channels])?;
		Self::from_matrices(weight, bias, stride, padding)
	}

	/// Construct ConvTranspose2d from exact IOKK weight and channel-bias matrices.
	///
	/// # Errors
	///
	/// Returns an error unless weight is nonempty F32 `[I, O, K, K]`, bias is
	/// matching same-engine F32 `[O]`, and stride is nonzero.
	pub fn from_matrices(
		weight: Matrix,
		bias: Matrix,
		stride: usize,
		padding: usize,
	) -> Result<Self> {
		let [input_channels, output_channels, kernel_height, kernel_width] = weight.shape() else {
			return Err(Error::invalid_argument(
				"ConvTranspose2d weight must have IOKK rank four",
			));
		};
		if *input_channels == 0
			|| *output_channels == 0
			|| *kernel_height == 0
			|| kernel_height != kernel_width
			|| stride == 0
			|| bias.shape() != [*output_channels]
			|| weight.dtype() != DType::F32
			|| bias.dtype() != DType::F32
			|| !weight.engine_handle().same_as(bias.engine_handle())
		{
			return Err(Error::invalid_argument(
				"ConvTranspose2d requires nonempty same-engine F32 square IOKK weight, matching bias, and nonzero stride",
			));
		}
		let input_channels = *input_channels;
		let output_channels = *output_channels;
		let kernel_size = *kernel_height;
		let weight = Parameter::new("weight", weight)?;
		let bias = Parameter::new("bias", bias)?;
		let mut registry = ModuleRegistry::new();
		registry.register_parameter("weight", weight.clone())?;
		registry.register_parameter("bias", bias.clone())?;
		Ok(Self {
			input_channels,
			output_channels,
			kernel_size,
			stride,
			padding,
			weight,
			bias,
			registry,
		})
	}

	/// Apply ConvTranspose2d without submitting or waiting.
	///
	/// # Errors
	///
	/// Returns an error from the underlying Matrix operation.
	pub fn forward(&self, input: &Matrix) -> Result<Matrix> {
		let (weight, weight_version, _) = self.weight.snapshot();
		let (bias, bias_version, _) = self.bias.snapshot();
		matrix::conv_transpose_2d_parameterized(
			input,
			(self.weight.clone(), weight, weight_version),
			(self.bias.clone(), bias, bias_version),
			self.stride,
			self.padding,
		)
	}

	/// Return the input channel count.
	pub const fn input_channels(&self) -> usize {
		self.input_channels
	}

	/// Return the output channel count.
	pub const fn output_channels(&self) -> usize {
		self.output_channels
	}

	/// Return the square kernel extent.
	pub const fn kernel_size(&self) -> usize {
		self.kernel_size
	}

	/// Return the spatial stride.
	pub const fn stride(&self) -> usize {
		self.stride
	}

	/// Return the symmetric zero-padding extent.
	pub const fn padding(&self) -> usize {
		self.padding
	}

	/// Return the stable IOKK weight parameter.
	pub fn weight(&self) -> Parameter {
		self.weight.clone()
	}

	/// Return the stable channel-bias parameter.
	pub fn bias(&self) -> Parameter {
		self.bias.clone()
	}

	/// Return parameters in deterministic weight, bias order.
	pub fn parameters(&self) -> [Parameter; 2] {
		[self.weight.clone(), self.bias.clone()]
	}
}

impl Module for ConvTranspose2d {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		ConvTranspose2d::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}

/// Trainable grouped FP32 two-dimensional convolution over NCHW matrices.
pub struct Conv2d {
	input_channels: usize,
	output_channels: usize,
	kernel_size: usize,
	stride: usize,
	padding: usize,
	groups: usize,
	weight: Parameter,
	bias: Parameter,
	registry: ModuleRegistry,
}

impl Conv2d {
	/// Construct a deterministically initialized grouped Conv2d layer.
	///
	/// The weight initializer is symmetric uniform with bound
	/// `1 / sqrt((input_channels / groups) * kernel_size²)`; bias starts at zero.
	///
	/// # Errors
	///
	/// Returns an error when dimensions are zero, channels are not divisible by
	/// `groups`, shape arithmetic overflows, or allocation/upload fails.
	#[allow(
		clippy::too_many_arguments,
		reason = "the donor Conv2d contract has six independent geometry fields plus a seed"
	)]
	pub fn with_seed(
		engine: &Engine,
		input_channels: usize,
		output_channels: usize,
		kernel_size: usize,
		stride: usize,
		padding: usize,
		groups: usize,
		seed: u64,
	) -> Result<Self> {
		if input_channels == 0
			|| output_channels == 0
			|| kernel_size == 0
			|| stride == 0
			|| groups == 0
			|| !input_channels.is_multiple_of(groups)
			|| !output_channels.is_multiple_of(groups)
		{
			return Err(Error::invalid_argument(
				"Conv2d requires nonzero dimensions and channels divisible by groups",
			));
		}
		let input_channels_per_group = input_channels / groups;
		let fan_in = input_channels_per_group
			.checked_mul(kernel_size)
			.and_then(|count| count.checked_mul(kernel_size))
			.ok_or_else(|| Error::invalid_argument("Conv2d fan-in overflows usize"))?;
		let weight_count = output_channels
			.checked_mul(fan_in)
			.ok_or_else(|| Error::invalid_argument("Conv2d weight size overflows usize"))?;
		let limit = 1.0_f32 / (fan_in as f32).sqrt();
		let weight_values = random::symmetric_uniform(weight_count, limit, seed);
		let weight = Matrix::from_f32(
			engine,
			[
				output_channels,
				input_channels_per_group,
				kernel_size,
				kernel_size,
			],
			&weight_values,
		)?;
		let bias = Matrix::from_f32(engine, [output_channels], &vec![0.0; output_channels])?;
		Self::from_matrices(weight, bias, stride, padding, groups)
	}

	/// Construct Conv2d from exact grouped OIHW weight and bias matrices.
	///
	/// # Errors
	///
	/// Returns an error unless weight is nonempty square-kernel F32 OIHW/group,
	/// bias is matching same-engine F32, stride/groups are nonzero, and output
	/// channels are divisible by groups.
	pub fn from_matrices(
		weight: Matrix,
		bias: Matrix,
		stride: usize,
		padding: usize,
		groups: usize,
	) -> Result<Self> {
		let [
			output_channels,
			input_channels_per_group,
			kernel_height,
			kernel_width,
		] = weight.shape()
		else {
			return Err(Error::invalid_argument(
				"Conv2d weight must have grouped OIHW rank four",
			));
		};
		let input_channels = input_channels_per_group
			.checked_mul(groups)
			.ok_or_else(|| Error::invalid_argument("Conv2d input channel count overflows usize"))?;
		if *output_channels == 0
			|| *input_channels_per_group == 0
			|| *kernel_height == 0
			|| kernel_height != kernel_width
			|| stride == 0
			|| groups == 0
			|| !output_channels.is_multiple_of(groups)
			|| bias.shape() != [*output_channels]
			|| weight.dtype() != DType::F32
			|| bias.dtype() != DType::F32
			|| !weight.engine_handle().same_as(bias.engine_handle())
		{
			return Err(Error::invalid_argument(
				"Conv2d requires nonempty same-engine F32 square OIHW/group weight, matching bias, nonzero stride/groups, and output channels divisible by groups",
			));
		}
		let output_channels = *output_channels;
		let kernel_size = *kernel_height;
		let weight = Parameter::new("weight", weight)?;
		let bias = Parameter::new("bias", bias)?;
		let mut registry = ModuleRegistry::new();
		registry.register_parameter("weight", weight.clone())?;
		registry.register_parameter("bias", bias.clone())?;
		Ok(Self {
			input_channels,
			output_channels,
			kernel_size,
			stride,
			padding,
			groups,
			weight,
			bias,
			registry,
		})
	}

	/// Apply grouped convolution without submitting or waiting.
	///
	/// # Errors
	///
	/// Returns an error from the underlying Matrix operation.
	pub fn forward(&self, input: &Matrix) -> Result<Matrix> {
		let (weight, weight_version, _) = self.weight.snapshot();
		let (bias, bias_version, _) = self.bias.snapshot();
		matrix::conv_2d_parameterized(
			input,
			(self.weight.clone(), weight, weight_version),
			(self.bias.clone(), bias, bias_version),
			self.stride,
			self.padding,
			self.groups,
		)
	}

	/// Return the input channel count.
	pub const fn input_channels(&self) -> usize {
		self.input_channels
	}

	/// Return the output channel count.
	pub const fn output_channels(&self) -> usize {
		self.output_channels
	}

	/// Return the square kernel extent.
	pub const fn kernel_size(&self) -> usize {
		self.kernel_size
	}

	/// Return the spatial stride.
	pub const fn stride(&self) -> usize {
		self.stride
	}

	/// Return the symmetric zero-padding extent.
	pub const fn padding(&self) -> usize {
		self.padding
	}

	/// Return the convolution group count.
	pub const fn groups(&self) -> usize {
		self.groups
	}

	/// Return the stable grouped OIHW weight parameter.
	pub fn weight(&self) -> Parameter {
		self.weight.clone()
	}

	/// Return the stable channel-bias parameter.
	pub fn bias(&self) -> Parameter {
		self.bias.clone()
	}

	/// Return parameters in deterministic weight, bias order.
	pub fn parameters(&self) -> [Parameter; 2] {
		[self.weight.clone(), self.bias.clone()]
	}
}

impl Module for Conv2d {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		Conv2d::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}
