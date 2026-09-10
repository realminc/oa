use std::rc::Rc;

use crate::{Engine, Matrix, Result, matrix as core_matrix};

use super::super::{Module, ModuleRegistry, matrix::swiglu};
use super::{Linear, RmsNorm};

/// Pre-normalized SwiGLU feed-forward network with a residual connection.
///
/// The module preserves the OA donor structure:
/// `RMSNorm -> (gate, up) -> SwiGLU -> down -> residual add`.
pub struct Ffn {
	model_width: usize,
	hidden_width: usize,
	epsilon: f32,
	norm: Rc<RmsNorm>,
	gate: Rc<Linear>,
	up: Rc<Linear>,
	down: Rc<Linear>,
	registry: ModuleRegistry,
}

impl Ffn {
	/// Construct a deterministically initialized donor-compatible FFN.
	///
	/// # Errors
	///
	/// Returns an error for invalid dimensions or epsilon, or when child
	/// construction or registration fails.
	pub fn with_seed(
		engine: &Engine,
		model_width: usize,
		hidden_width: usize,
		epsilon: f32,
		seed: u64,
	) -> Result<Self> {
		let norm = Rc::new(RmsNorm::new(engine, model_width, epsilon)?);
		let gate = Rc::new(Linear::with_seed(engine, model_width, hidden_width, seed)?);
		let up = Rc::new(Linear::with_seed(
			engine,
			model_width,
			hidden_width,
			seed.wrapping_add(1),
		)?);
		let down = Rc::new(Linear::with_seed(
			engine,
			hidden_width,
			model_width,
			seed.wrapping_add(2),
		)?);
		let mut registry = ModuleRegistry::new();
		registry.register_module("norm", norm.clone())?;
		registry.register_module("gate", gate.clone())?;
		registry.register_module("up", up.clone())?;
		registry.register_module("down", down.clone())?;
		Ok(Self {
			model_width,
			hidden_width,
			epsilon,
			norm,
			gate,
			up,
			down,
			registry,
		})
	}

	/// Apply the pre-normalized gated projection and residual connection.
	///
	/// # Errors
	///
	/// Returns an error unless the input is a compatible F32 Matrix or a child
	/// operation fails.
	pub fn forward(&self, input: &Matrix) -> Result<Matrix> {
		let normalized = self.norm.forward(input)?;
		let gate = self.gate.forward(&normalized)?;
		let up = self.up.forward(&normalized)?;
		let activated = swiglu(&gate, &up)?;
		let projected = self.down.forward(&activated)?;
		core_matrix::add(input, &projected)
	}

	/// Return the input and output feature width.
	pub const fn model_width(&self) -> usize {
		self.model_width
	}

	/// Return the gated hidden feature width.
	pub const fn hidden_width(&self) -> usize {
		self.hidden_width
	}

	/// Return the RMSNorm numerical-stability epsilon.
	pub const fn epsilon(&self) -> f32 {
		self.epsilon
	}
}

impl Module for Ffn {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		Ffn::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}
