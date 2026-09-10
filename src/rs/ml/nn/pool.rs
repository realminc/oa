use crate::{Error, Matrix, Result};

use super::super::{Module, ModuleRegistry, matrix};

/// Parameterless two-dimensional average-pooling module for NCHW matrices.
pub struct AvgPool2d {
	kernel_size: usize,
	stride: usize,
	padding: usize,
	registry: ModuleRegistry,
}

impl AvgPool2d {
	/// Construct AvgPool2d with `stride == kernel_size` and no padding.
	///
	/// # Errors
	///
	/// Returns an error when `kernel_size` is zero.
	pub fn new(kernel_size: usize) -> Result<Self> {
		Self::with_options(kernel_size, kernel_size, 0)
	}

	/// Construct AvgPool2d with explicit square-kernel geometry.
	///
	/// # Errors
	///
	/// Returns an error when `kernel_size` or `stride` is zero.
	pub fn with_options(kernel_size: usize, stride: usize, padding: usize) -> Result<Self> {
		if kernel_size == 0 || stride == 0 {
			return Err(Error::invalid_argument(
				"AvgPool2d kernel size and stride must be nonzero",
			));
		}
		Ok(Self {
			kernel_size,
			stride,
			padding,
			registry: ModuleRegistry::new(),
		})
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

	/// Pool one NCHW Matrix without submitting or waiting.
	///
	/// # Errors
	///
	/// Returns an error from the underlying Matrix operation.
	pub fn forward(&self, input: &Matrix) -> Result<Matrix> {
		matrix::avg_pool_2d(input, self.kernel_size, self.stride, self.padding)
	}
}

impl Module for AvgPool2d {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		AvgPool2d::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}

/// Parameterless two-dimensional max-pooling module for NCHW matrices.
pub struct MaxPool2d {
	kernel_size: usize,
	stride: usize,
	padding: usize,
	registry: ModuleRegistry,
}

impl MaxPool2d {
	/// Construct MaxPool2d with `stride == kernel_size` and no padding.
	///
	/// # Errors
	///
	/// Returns an error when `kernel_size` is zero.
	pub fn new(kernel_size: usize) -> Result<Self> {
		Self::with_options(kernel_size, kernel_size, 0)
	}

	/// Construct MaxPool2d with explicit square-kernel geometry.
	///
	/// # Errors
	///
	/// Returns an error when `kernel_size` or `stride` is zero.
	pub fn with_options(kernel_size: usize, stride: usize, padding: usize) -> Result<Self> {
		if kernel_size == 0 || stride == 0 {
			return Err(Error::invalid_argument(
				"MaxPool2d kernel size and stride must be nonzero",
			));
		}
		Ok(Self {
			kernel_size,
			stride,
			padding,
			registry: ModuleRegistry::new(),
		})
	}

	/// Return the square kernel extent.
	pub const fn kernel_size(&self) -> usize {
		self.kernel_size
	}

	/// Return the spatial stride.
	pub const fn stride(&self) -> usize {
		self.stride
	}

	/// Return the symmetric padding extent.
	pub const fn padding(&self) -> usize {
		self.padding
	}

	/// Pool one NCHW Matrix and retain argmax state for reverse mode.
	///
	/// # Errors
	///
	/// Returns an error from the underlying Matrix operation.
	pub fn forward(&self, input: &Matrix) -> Result<Matrix> {
		Ok(matrix::max_pool_2d(input, self.kernel_size, self.stride, self.padding)?.output)
	}
}

impl Module for MaxPool2d {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		MaxPool2d::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}

/// Parameterless adaptive average-pooling module for NCHW matrices.
pub struct AdaptiveAvgPool2d {
	output_height: usize,
	output_width: usize,
	registry: ModuleRegistry,
}

impl AdaptiveAvgPool2d {
	/// Construct AdaptiveAvgPool2d with a square output extent.
	///
	/// # Errors
	///
	/// Returns an error when `output_size` is zero.
	pub fn new(output_size: usize) -> Result<Self> {
		Self::with_output_size(output_size, output_size)
	}

	/// Construct AdaptiveAvgPool2d with independent output height and width.
	///
	/// # Errors
	///
	/// Returns an error when either output extent is zero.
	pub fn with_output_size(output_height: usize, output_width: usize) -> Result<Self> {
		if output_height == 0 || output_width == 0 {
			return Err(Error::invalid_argument(
				"AdaptiveAvgPool2d output extents must be nonzero",
			));
		}
		Ok(Self {
			output_height,
			output_width,
			registry: ModuleRegistry::new(),
		})
	}

	/// Return the requested output height.
	pub const fn output_height(&self) -> usize {
		self.output_height
	}

	/// Return the requested output width.
	pub const fn output_width(&self) -> usize {
		self.output_width
	}

	/// Adaptively pool one NCHW Matrix without submitting or waiting.
	///
	/// # Errors
	///
	/// Returns an error from the underlying Matrix operation.
	pub fn forward(&self, input: &Matrix) -> Result<Matrix> {
		matrix::adaptive_avg_pool_2d(input, self.output_height, self.output_width)
	}
}

impl Module for AdaptiveAvgPool2d {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		AdaptiveAvgPool2d::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}
