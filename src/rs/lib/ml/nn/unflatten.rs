//! Unflatten (`nn.Unflatten`): expand one axis into several dimensions.

use crate::{Error, Matrix, Result};

use super::super::{Module, ModuleRegistry};

/// Expand one input axis into the configured extent list.
pub struct Unflatten {
	dim: i32,
	sizes: Vec<usize>,
	registry: ModuleRegistry,
}

impl Unflatten {
	/// Construct an unflatten expanding `dim` into `sizes`.
	///
	/// `sizes` may contain one `-1` entry, resolved from the axis extent at
	/// call time (stored as `usize::MAX` internally).
	///
	/// # Errors
	///
	/// Returns an error when `sizes` is empty or holds more than one `-1`.
	pub fn new(dim: i32, sizes: &[i64]) -> Result<Self> {
		if sizes.is_empty() {
			return Err(Error::invalid_argument(
				"Unflatten requires a nonempty size list",
			));
		}
		if sizes.iter().filter(|extent| **extent < -1).count() > 0
			|| sizes.iter().filter(|extent| **extent == -1).count() > 1
		{
			return Err(Error::invalid_argument(
				"Unflatten sizes must be nonnegative with at most one -1",
			));
		}
		Ok(Self {
			dim,
			sizes: sizes
				.iter()
				.map(|extent| {
					if *extent == -1 {
						usize::MAX
					} else {
						*extent as usize
					}
				})
				.collect(),
			registry: ModuleRegistry::new(),
		})
	}

	/// Return the configured axis.
	pub const fn dim(&self) -> i32 {
		self.dim
	}

	/// Expand `dim` of `input` into the configured extents.
	///
	/// # Errors
	///
	/// Returns an error when the axis is out of range, the extents do not
	/// multiply to the axis extent, or the reshape fails.
	pub fn forward(&self, input: &Matrix) -> Result<Matrix> {
		let operation = "Unflatten";
		let rank = input.shape().len();
		let dim = if self.dim < 0 {
			self.dim + rank as i32
		} else {
			self.dim
		};
		if dim < 0 || dim as usize >= rank {
			return Err(Error::invalid_argument(format!(
				"{operation} dim {} is outside rank {rank}",
				self.dim
			)));
		}
		let extent = input.shape()[dim as usize];
		let mut sizes = self.sizes.clone();
		if let Some(index) = sizes.iter().position(|extent| *extent == usize::MAX) {
			let product: usize = sizes
				.iter()
				.filter(|extent| **extent != usize::MAX)
				.product();
			if product == 0 || !extent.is_multiple_of(product) {
				return Err(Error::invalid_argument(format!(
					"{operation} cannot infer the -1 extent for axis size {extent}"
				)));
			}
			sizes[index] = extent / product;
		} else if sizes.iter().product::<usize>() != extent {
			return Err(Error::invalid_argument(format!(
				"{operation} sizes must multiply to the axis extent {extent}"
			)));
		}
		let mut shape = Vec::with_capacity(rank - 1 + sizes.len());
		shape.extend_from_slice(&input.shape()[..dim as usize]);
		shape.extend_from_slice(&sizes);
		shape.extend_from_slice(&input.shape()[dim as usize + 1..]);
		input.reshape(shape)
	}
}

impl Module for Unflatten {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		Unflatten::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}
