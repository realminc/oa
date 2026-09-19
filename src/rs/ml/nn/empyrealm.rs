use std::rc::Rc;

use crate::{DType, Engine, Error, Matrix, Result};

use super::{Embedding, Mamba3, Mamba3Config, Mamba3State};
use crate::ml::{Module, ModuleRegistry};

/// OA's embedding-plus-Mamba core with one canonical Rust execution path.
///
/// The C++ donor currently carries renamed Empyrealm shader copies whose SPIR-V
/// is identical to the Mamba-3 providers. Rust preserves the public module and
/// parameter topology while reusing [`Mamba3`] until the algorithms actually
/// diverge.
pub struct EmpyrealmCore {
	registry: ModuleRegistry,
	embedding: Rc<Embedding>,
	mixer: Rc<Mamba3>,
	model_width: usize,
}

impl EmpyrealmCore {
	/// Construct a deterministic token embedding and Mamba-3 mixer.
	///
	/// # Errors
	///
	/// Returns an error when the embedding, mixer, or child registration cannot
	/// be constructed.
	pub fn with_seed(
		engine: &Engine,
		vocab_size: usize,
		config: Mamba3Config,
		seed: u64,
	) -> Result<Self> {
		let embedding = Rc::new(Embedding::with_seed(
			engine,
			vocab_size,
			config.model_width,
			seed,
		)?);
		let mixer = Rc::new(Mamba3::with_seed(engine, config, seed.wrapping_add(1))?);
		let mut registry = ModuleRegistry::new();
		registry.register_module("embed", embedding.clone())?;
		registry.register_module("mixer", mixer.clone())?;
		Ok(Self {
			registry,
			embedding,
			mixer,
			model_width: config.model_width,
		})
	}

	/// Embed a nonempty rank-two token grid and evaluate the residual mixer.
	///
	/// # Errors
	///
	/// Returns an error unless `tokens` is U8/U32 `[batch, sequence]`, or when a
	/// child operation cannot be recorded.
	pub fn forward(&self, tokens: &Matrix) -> Result<Matrix> {
		let [batch, sequence] = tokens.shape() else {
			return Err(Error::invalid_argument(
				"EmpyrealmCore tokens must have shape [batch, sequence]",
			));
		};
		if !matches!(tokens.dtype(), DType::U8 | DType::U32) || *batch == 0 || *sequence == 0 {
			return Err(Error::invalid_argument(
				"EmpyrealmCore tokens must be a nonempty U8/U32 Matrix",
			));
		}
		let embedded = self.embedding.forward(tokens)?;
		self.forward_embedded(&embedded)
	}

	/// Evaluate the residual mixer over pre-embedded `[B, S, D]` features.
	///
	/// # Errors
	///
	/// Returns an error unless the input is nonempty FP32 `[B, S, model_width]`,
	/// or when a child operation cannot be recorded.
	pub fn forward_embedded(&self, embedded: &Matrix) -> Result<Matrix> {
		let [batch, sequence, width] = embedded.shape() else {
			return Err(Error::invalid_argument(
				"EmpyrealmCore embedded input must have shape [B, S, D]",
			));
		};
		if embedded.dtype() != DType::F32 || *batch == 0 || *sequence == 0 || *width != self.model_width
		{
			return Err(Error::invalid_argument(format!(
				"EmpyrealmCore requires nonempty F32 [B, S, {}]; found {:?} {}",
				self.model_width,
				embedded.shape(),
				embedded.dtype().token()
			)));
		}
		let mixed = self.mixer.forward(embedded)?;
		crate::matrix::add(&mixed, embedded)
	}

	/// Allocate a recurrent state owned by the mixer.
	///
	/// # Errors
	///
	/// Returns an error when the requested state geometry cannot be allocated.
	pub fn new_state(&self, batch_size: usize) -> Result<Mamba3State> {
		self.mixer.new_state(batch_size)
	}

	/// Evaluate one pre-embedded recurrent position plus its residual.
	///
	/// # Errors
	///
	/// Returns an error when the input/state contract or child execution fails.
	pub fn step_embedded(&self, embedded: &Matrix, state: &mut Mamba3State) -> Result<Matrix> {
		let mixed = self.mixer.step(embedded, state)?;
		crate::matrix::add(&mixed, embedded)
	}

	/// Return the shared Mamba-3 implementation owner.
	pub fn mixer(&self) -> &Mamba3 {
		&self.mixer
	}
}

impl Module for EmpyrealmCore {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		EmpyrealmCore::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}
