//! Donor-compatible parameterless utility modules.

use crate::{Error, Matrix, Result};

use super::super::{Module, ModuleRegistry};

/// Return the input value unchanged.
pub struct Identity {
	registry: ModuleRegistry,
}

impl Identity {
	/// Construct an identity module.
	pub const fn new() -> Self {
		Self {
			registry: ModuleRegistry::new(),
		}
	}

	/// Return a cheap handle to the same semantic Matrix value.
	pub fn forward(&self, input: &Matrix) -> Result<Matrix> {
		Ok(input.clone())
	}
}

impl Default for Identity {
	fn default() -> Self {
		Self::new()
	}
}

impl Module for Identity {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		Identity::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}

/// Flatten an inclusive range of Matrix dimensions into one dimension.
///
/// Negative dimensions are resolved from the end at call time. The donor
/// default `(1, -1)` preserves a leading batch dimension.
pub struct Flatten {
	start_dim: isize,
	end_dim: isize,
	registry: ModuleRegistry,
}

impl Flatten {
	/// Construct a flatten module with PyTorch-style dimension indexing.
	pub const fn new(start_dim: isize, end_dim: isize) -> Self {
		Self {
			start_dim,
			end_dim,
			registry: ModuleRegistry::new(),
		}
	}

	/// Return the configured inclusive start dimension.
	pub const fn start_dim(&self) -> isize {
		self.start_dim
	}

	/// Return the configured inclusive end dimension.
	pub const fn end_dim(&self) -> isize {
		self.end_dim
	}

	/// Return a differentiable zero-copy flattened Matrix view.
	///
	/// Inputs of rank zero or one and positive start dimensions beyond the input
	/// rank preserve the donor identity behavior.
	///
	/// # Errors
	///
	/// Returns an error when either resolved dimension is outside the input rank,
	/// the resolved range is reversed, or the flattened extent overflows.
	pub fn forward(&self, input: &Matrix) -> Result<Matrix> {
		let rank = input.shape().len();
		if rank <= 1 || self.start_dim >= 0 && self.start_dim as usize >= rank {
			return Ok(input.clone());
		}
		let start = resolve_dim(self.start_dim, rank, "start")?;
		let end = resolve_dim(self.end_dim, rank, "end")?;
		if start > end {
			return Err(Error::invalid_argument(format!(
				"Flatten start dimension {start} exceeds end dimension {end}"
			)));
		}
		let flattened = input.shape()[start..=end]
			.iter()
			.try_fold(1_usize, |extent, dimension| {
				extent
					.checked_mul(*dimension)
					.ok_or_else(|| Error::resource_exhausted("Flatten extent overflows usize"))
			})?;
		let mut shape = Vec::with_capacity(rank - (end - start));
		shape.extend_from_slice(&input.shape()[..start]);
		shape.push(flattened);
		shape.extend_from_slice(&input.shape()[end + 1..]);
		input.reshape(shape)
	}
}

impl Default for Flatten {
	fn default() -> Self {
		Self::new(1, -1)
	}
}

impl Module for Flatten {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		Flatten::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}

fn resolve_dim(dimension: isize, rank: usize, label: &str) -> Result<usize> {
	let rank =
		isize::try_from(rank).map_err(|_| Error::resource_exhausted("Matrix rank exceeds isize"))?;
	let resolved = if dimension < 0 {
		rank
			.checked_add(dimension)
			.ok_or_else(|| Error::invalid_argument(format!("Flatten {label} dimension is invalid")))?
	} else {
		dimension
	};
	if !(0..rank).contains(&resolved) {
		return Err(Error::invalid_argument(format!(
			"Flatten {label} dimension {dimension} is outside rank {rank}"
		)));
	}
	Ok(resolved as usize)
}
