use pyo3::prelude::*;

use crate::{error::python_error, matrix::PythonMatrix, runtime::PythonEngine};

// ── CategoricalActorCriticConfig ──────────────────────────────────────────────

/// Dimensions and deterministic initialization seed for the default MLP.
#[pyclass(name = "CategoricalActorCriticConfig")]
#[derive(Clone, Copy)]
pub(crate) struct PythonCategoricalActorCriticConfig {
	pub(crate) inner: oa::ml::CategoricalActorCriticConfig,
}

#[pymethods]
impl PythonCategoricalActorCriticConfig {
	#[new]
	pub fn new(observation_size: usize, action_count: usize, hidden_size: usize, seed: u64) -> Self {
		Self {
			inner: oa::ml::CategoricalActorCriticConfig {
				observation_size,
				action_count,
				hidden_size,
				seed,
			},
		}
	}

	#[getter]
	pub fn observation_size(&self) -> usize {
		self.inner.observation_size
	}

	#[getter]
	pub fn action_count(&self) -> usize {
		self.inner.action_count
	}

	#[getter]
	pub fn hidden_size(&self) -> usize {
		self.inner.hidden_size
	}

	#[getter]
	pub fn seed(&self) -> u64 {
		self.inner.seed
	}

	pub fn __repr__(&self) -> String {
		format!(
			"CategoricalActorCriticConfig(obs={}, actions={}, hidden={}, seed={})",
			self.inner.observation_size, self.inner.action_count, self.inner.hidden_size, self.inner.seed,
		)
	}
}

// ── CategoricalActorCriticOutput ──────────────────────────────────────────────

/// Policy logits and scalar critic value produced from one observation batch.
#[pyclass(name = "CategoricalActorCriticOutput", unsendable)]
pub(crate) struct PythonCategoricalActorCriticOutput {
	pub(crate) logits: oa::Matrix,
	pub(crate) value: oa::Matrix,
}

#[pymethods]
impl PythonCategoricalActorCriticOutput {
	/// FP32 categorical policy logits shaped `[batch, actions]`.
	#[getter]
	pub fn logits(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.logits.clone())
	}

	/// FP32 critic estimates shaped `[batch]`.
	#[getter]
	pub fn value(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.value.clone())
	}

	pub fn __repr__(&self) -> &str {
		"CategoricalActorCriticOutput"
	}
}

// ── CategoricalActorCritic ────────────────────────────────────────────────────

/// Default two-tower categorical actor/critic MLP.
#[pyclass(name = "CategoricalActorCritic", unsendable)]
pub(crate) struct PythonCategoricalActorCritic {
	pub(crate) inner: oa::ml::CategoricalActorCritic,
}

#[pymethods]
impl PythonCategoricalActorCritic {
	/// Construct the checked default categorical actor/critic.
	#[new]
	pub fn new(engine: &PythonEngine, config: &PythonCategoricalActorCriticConfig) -> PyResult<Self> {
		oa::ml::CategoricalActorCritic::new(&engine.inner, config.inner)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Evaluate the independent policy and value towers.
	pub fn evaluate(
		&self,
		observation: &PythonMatrix,
	) -> PyResult<PythonCategoricalActorCriticOutput> {
		self
			.inner
			.evaluate(&observation.inner)
			.map(|output| PythonCategoricalActorCriticOutput {
				logits: output.logits,
				value: output.value,
			})
			.map_err(python_error)
	}

	/// Return all trainable parameters (for passing to optimizers).
	pub fn all_parameters(&self) -> PyResult<Vec<super::autograd::PythonParameter>> {
		use oa::ml::Module as _;
		self
			.inner
			.all_parameters()
			.map(|params| {
				params
					.into_iter()
					.map(|inner| super::autograd::PythonParameter { inner })
					.collect()
			})
			.map_err(python_error)
	}

	/// Return the immutable architecture and seed.
	pub fn config(&self) -> PythonCategoricalActorCriticConfig {
		PythonCategoricalActorCriticConfig {
			inner: self.inner.config(),
		}
	}

	pub fn __repr__(&self) -> String {
		let c = self.inner.config();
		format!(
			"CategoricalActorCritic(obs={}, actions={}, hidden={})",
			c.observation_size, c.action_count, c.hidden_size,
		)
	}
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonCategoricalActorCriticConfig>()?;
	module.add_class::<PythonCategoricalActorCriticOutput>()?;
	module.add_class::<PythonCategoricalActorCritic>()?;
	Ok(())
}
