use crate::{Error, Matrix, Result};

use super::super::{Module, ModuleRegistry, matrix};

/// Llama-style split-half rotary position embedding module.
pub struct Rope {
	num_heads: usize,
	head_dim: usize,
	theta_base: f32,
	registry: ModuleRegistry,
}

impl Rope {
	/// Construct a parameterless rotary-position module.
	///
	/// # Errors
	///
	/// Returns an error unless both dimensions are nonzero, `head_dim` is even,
	/// their product fits `usize`, and `theta_base` is finite and positive.
	pub fn new(num_heads: usize, head_dim: usize, theta_base: f32) -> Result<Self> {
		if num_heads == 0
			|| head_dim == 0
			|| !head_dim.is_multiple_of(2)
			|| !theta_base.is_finite()
			|| theta_base <= 0.0
			|| num_heads.checked_mul(head_dim).is_none()
		{
			return Err(Error::invalid_argument(
				"RoPE requires nonzero heads, an even head dimension, and a finite positive theta base",
			));
		}
		Ok(Self {
			num_heads,
			head_dim,
			theta_base,
			registry: ModuleRegistry::new(),
		})
	}

	/// Rotate a sequence beginning at absolute position zero.
	///
	/// # Errors
	///
	/// Returns an error unless input is compatible F32 `[T, H * D]`, or runtime
	/// recording fails.
	pub fn forward(&self, input: &Matrix) -> Result<Matrix> {
		matrix::rope(input, self.num_heads, self.head_dim, self.theta_base, 0)
	}

	/// Return the number of independently rotated heads.
	pub const fn num_heads(&self) -> usize {
		self.num_heads
	}

	/// Return the even dimension of each head.
	pub const fn head_dim(&self) -> usize {
		self.head_dim
	}

	/// Return the rotary frequency base.
	pub const fn theta_base(&self) -> f32 {
		self.theta_base
	}
}

impl Module for Rope {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		Rope::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}
