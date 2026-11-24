//! Rank-adapted batch normalization over the shared 2d kernel family.

use std::rc::Rc;

use crate::{Engine, Error, Matrix, Result};

use super::super::{Module, ModuleRegistry};
use super::BatchNorm2d;

/// `nn.BatchNorm1d`: `[N, C, L]` batch normalization.
///
/// The composed contract reshapes to `[N, C, 1, L]` and delegates to the
/// shared `BatchNorm2d` kernel family — batch statistics, running
/// buffers, and adjoints are identical because normalization is
/// per-`(N, C)` over flattened spatial extents.
pub struct BatchNorm1d {
	inner: Rc<BatchNorm2d>,
	registry: ModuleRegistry,
}

impl BatchNorm1d {
	/// Construct over `num_features` channels on the current engine.
	///
	/// # Errors
	///
	/// Returns an error under the same contract as [`BatchNorm2d::new_on`].
	pub fn new(num_features: usize) -> Result<Self> {
		Self::new_on(&Engine::current()?, num_features)
	}

	/// Construct on `engine`.
	///
	/// # Errors
	///
	/// Returns an error under the same contract as [`BatchNorm2d::new_on`].
	pub fn new_on(engine: &Engine, num_features: usize) -> Result<Self> {
		let inner = Rc::new(BatchNorm2d::new_on(engine, num_features)?);
		let mut registry = ModuleRegistry::new();
		registry.register_module("inner", Rc::clone(&inner) as Rc<dyn Module>)?;
		Ok(Self { inner, registry })
	}

	/// Normalize `[N, C, L]` through the shared kernel.
	///
	/// # Errors
	///
	/// Returns an error unless `input` is rank three, or the delegated
	/// forward fails.
	pub fn forward(&self, input: &Matrix) -> Result<Matrix> {
		let &[batch, channels, length] = input.shape() else {
			return Err(Error::invalid_argument(format!(
				"batch_norm1d requires [N, C, L]; found {:?}",
				input.shape(),
			)));
		};
		self
			.inner
			.forward(&input.reshape([batch, channels, 1, length])?)?
			.reshape([batch, channels, length])
	}
}

impl Module for BatchNorm1d {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		BatchNorm1d::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}

/// `nn.BatchNorm3d`: `[N, C, D, H, W]` batch normalization.
///
/// Shares the `BatchNorm2d` kernel family by flattening `[D, H, W]` to
/// `[D, H*W]` — normalization is per-`(N, C)` over spatial extents, so the
/// flat layout is equivalent.
pub struct BatchNorm3d {
	inner: Rc<BatchNorm2d>,
	registry: ModuleRegistry,
}

impl BatchNorm3d {
	/// Construct over `num_features` channels on the current engine.
	///
	/// # Errors
	///
	/// Returns an error under the same contract as [`BatchNorm2d::new_on`].
	pub fn new(num_features: usize) -> Result<Self> {
		Self::new_on(&Engine::current()?, num_features)
	}

	/// Construct on `engine`.
	///
	/// # Errors
	///
	/// Returns an error under the same contract as [`BatchNorm2d::new_on`].
	pub fn new_on(engine: &Engine, num_features: usize) -> Result<Self> {
		let inner = Rc::new(BatchNorm2d::new_on(engine, num_features)?);
		let mut registry = ModuleRegistry::new();
		registry.register_module("inner", Rc::clone(&inner) as Rc<dyn Module>)?;
		Ok(Self { inner, registry })
	}

	/// Normalize `[N, C, D, H, W]` through the shared kernel.
	///
	/// # Errors
	///
	/// Returns an error unless `input` is rank five, or the delegated
	/// forward fails.
	pub fn forward(&self, input: &Matrix) -> Result<Matrix> {
		let &[batch, channels, depth, height, width] = input.shape() else {
			return Err(Error::invalid_argument(format!(
				"batch_norm3d requires [N, C, D, H, W]; found {:?}",
				input.shape(),
			)));
		};
		let flat = height
			.checked_mul(width)
			.ok_or_else(|| Error::invalid_argument("batch_norm3d spatial size overflows usize"))?;
		self
			.inner
			.forward(&input.reshape([batch, channels, depth, flat])?)?
			.reshape([batch, channels, depth, height, width])
	}
}

impl Module for BatchNorm3d {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		BatchNorm3d::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}
