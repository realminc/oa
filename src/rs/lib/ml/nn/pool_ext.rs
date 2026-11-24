//! One-dimensional pooling composed over slice + reduce ops.

use crate::{Error, Matrix, Result, matrix};

use super::super::{Module, ModuleRegistry};

fn pool_1d(
	input: &Matrix,
	kernel_size: usize,
	stride: usize,
	padding: usize,
	operation: &str,
) -> Result<Matrix> {
	let &[batch, channels, length] = input.shape() else {
		return Err(Error::invalid_argument(format!(
			"{operation} requires [N, C, L]; found {:?}",
			input.shape(),
		)));
	};
	if kernel_size == 0 || stride == 0 {
		return Err(Error::invalid_argument(format!(
			"{operation} requires nonzero kernel and stride"
		)));
	}
	let padded = length
		.checked_add(2 * padding)
		.ok_or_else(|| Error::invalid_argument(format!("{operation} length overflows usize")))?;
	if kernel_size > padded {
		return Err(Error::invalid_argument(format!(
			"{operation} kernel {kernel_size} exceeds padded length {padded}"
		)));
	}
	let window_count = (padded - kernel_size) / stride + 1;
	let padded_input = if padding > 0 {
		let zeros = Matrix::new_from_f32(
			[batch, channels, padding],
			&vec![0.0; batch * channels * padding],
		)?;
		matrix::concat(&[zeros.clone(), input.clone(), zeros], 2)?
	} else {
		input.clone()
	};
	let mut windows = Vec::with_capacity(window_count);
	for window in 0..window_count {
		let start = window * stride;
		let region = matrix::slice(&padded_input, 2, start as i64, (start + kernel_size) as i64)?;
		windows.push(matrix::mean(&region, 2)?);
	}
	matrix::concat(&windows, 2)
}

/// `nn.AvgPool1d`: `[N, C, L]` average pooling.
///
/// Composed over per-window `slice` + `mean` — the 2d pooling kernel rejects
/// inputs shorter than the kernel, and `matrix::max` is full-reduction only,
/// so a max variant stays planned until an axis-max op exists.
pub struct AvgPool1d {
	kernel_size: usize,
	stride: usize,
	padding: usize,
	registry: ModuleRegistry,
}

impl AvgPool1d {
	/// Construct with `stride == kernel_size` and no padding.
	///
	/// # Errors
	///
	/// Returns an error when `kernel_size` is zero.
	pub fn new(kernel_size: usize) -> Result<Self> {
		Self::new_with_options(kernel_size, kernel_size, 0)
	}

	/// Construct with explicit stride and symmetric zero padding.
	///
	/// # Errors
	///
	/// Returns an error when `kernel_size` or `stride` is zero.
	pub fn new_with_options(kernel_size: usize, stride: usize, padding: usize) -> Result<Self> {
		if kernel_size == 0 || stride == 0 {
			return Err(Error::invalid_argument(
				"avg_pool1d requires nonzero kernel and stride",
			));
		}
		Ok(Self {
			kernel_size,
			stride,
			padding,
			registry: ModuleRegistry::new(),
		})
	}

	/// Pool `[N, C, L]` to `[N, C, floor((L + 2p - k) / s) + 1]`.
	///
	/// # Errors
	///
	/// Returns an error unless `input` is rank three F32, or the composed
	/// ops fail.
	pub fn forward(&self, input: &Matrix) -> Result<Matrix> {
		pool_1d(
			input,
			self.kernel_size,
			self.stride,
			self.padding,
			"avg_pool1d",
		)
	}
}

impl Module for AvgPool1d {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		AvgPool1d::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}
