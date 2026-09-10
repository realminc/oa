use crate::{Matrix, Result, matrix};

use super::super::{Module, ModuleRegistry};

/// Parameterless axis-aware Softmax module.
pub struct Softmax {
	dim: i32,
	registry: ModuleRegistry,
}

impl Softmax {
	/// Construct a Softmax over `dim`, where `-1` selects the last dimension.
	pub const fn new(dim: i32) -> Self {
		Self {
			dim,
			registry: ModuleRegistry::new(),
		}
	}

	/// Return the selected dimension.
	pub const fn dim(&self) -> i32 {
		self.dim
	}

	/// Apply Softmax without submitting or waiting.
	///
	/// # Errors
	///
	/// Returns an error from the underlying Matrix operation.
	pub fn forward(&self, input: &Matrix) -> Result<Matrix> {
		matrix::softmax(input, self.dim)
	}
}

impl Default for Softmax {
	fn default() -> Self {
		Self::new(-1)
	}
}

impl Module for Softmax {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		Softmax::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}

/// Parameterless axis-aware LogSoftmax module.
pub struct LogSoftmax {
	dim: i32,
	registry: ModuleRegistry,
}

impl LogSoftmax {
	/// Construct a LogSoftmax over `dim`, where `-1` selects the last dimension.
	pub const fn new(dim: i32) -> Self {
		Self {
			dim,
			registry: ModuleRegistry::new(),
		}
	}

	/// Return the selected dimension.
	pub const fn dim(&self) -> i32 {
		self.dim
	}

	/// Apply LogSoftmax without submitting or waiting.
	///
	/// # Errors
	///
	/// Returns an error from the underlying Matrix operation.
	pub fn forward(&self, input: &Matrix) -> Result<Matrix> {
		matrix::log_softmax(input, self.dim)
	}
}

impl Default for LogSoftmax {
	fn default() -> Self {
		Self::new(-1)
	}
}

impl Module for LogSoftmax {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		LogSoftmax::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}
