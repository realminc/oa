use pyo3::prelude::*;

use crate::{error::python_error, matrix::PythonMatrix, runtime::PythonEngine};

use super::rl::PythonGaeConfig;

// ── RolloutConfig ─────────────────────────────────────────────────────────────

/// Shape and capacity of one categorical on-policy rollout.
#[pyclass(name = "RolloutConfig")]
#[derive(Clone)]
pub(crate) struct PythonRolloutConfig {
	pub(crate) inner: oa::ml::RolloutConfig,
}

#[pymethods]
impl PythonRolloutConfig {
	#[new]
	pub fn new(time: usize, environments: usize, observation_shape: Vec<usize>) -> Self {
		Self {
			inner: oa::ml::RolloutConfig {
				time,
				environments,
				observation_shape,
			},
		}
	}

	#[getter]
	pub fn time(&self) -> usize {
		self.inner.time
	}

	#[getter]
	pub fn environments(&self) -> usize {
		self.inner.environments
	}

	#[getter]
	pub fn observation_shape(&self) -> Vec<usize> {
		self.inner.observation_shape.clone()
	}

	pub fn __repr__(&self) -> String {
		format!(
			"RolloutConfig(time={}, environments={}, observation_shape={:?})",
			self.inner.time, self.inner.environments, self.inner.observation_shape,
		)
	}
}

// ── RolloutTransition ─────────────────────────────────────────────────────────

/// One vector-environment transition whose values remain device-resident.
#[pyclass(name = "RolloutTransition", unsendable)]
pub(crate) struct PythonRolloutTransition {
	pub(crate) inner: oa::ml::RolloutTransition,
}

#[pymethods]
impl PythonRolloutTransition {
	#[new]
	#[allow(clippy::too_many_arguments)]
	pub fn new(
		observation: &PythonMatrix,
		action: &PythonMatrix,
		reward: &PythonMatrix,
		value: &PythonMatrix,
		next_value: &PythonMatrix,
		log_probability: &PythonMatrix,
		terminated: &PythonMatrix,
		truncated: &PythonMatrix,
	) -> Self {
		Self {
			inner: oa::ml::RolloutTransition::new(
				observation.inner.clone(),
				action.inner.clone(),
				reward.inner.clone(),
				value.inner.clone(),
				next_value.inner.clone(),
				log_probability.inner.clone(),
				terminated.inner.clone(),
				truncated.inner.clone(),
			),
		}
	}

	#[getter]
	pub fn observation(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.observation().clone())
	}

	#[getter]
	pub fn action(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.action().clone())
	}

	#[getter]
	pub fn reward(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.reward().clone())
	}

	#[getter]
	pub fn value(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.value().clone())
	}

	#[getter]
	pub fn next_value(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.next_value().clone())
	}

	#[getter]
	pub fn log_probability(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.log_probability().clone())
	}

	#[getter]
	pub fn terminated(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.terminated().clone())
	}

	#[getter]
	pub fn truncated(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.truncated().clone())
	}

	pub fn __repr__(&self) -> &str {
		"RolloutTransition"
	}
}

// ── RolloutBuffer ─────────────────────────────────────────────────────────────

/// Stateful fixed-capacity owner for one categorical on-policy rollout.
#[pyclass(name = "RolloutBuffer", unsendable)]
pub(crate) struct PythonRolloutBuffer {
	pub(crate) inner: oa::ml::RolloutBuffer,
}

#[pymethods]
impl PythonRolloutBuffer {
	/// Allocate one empty fixed-capacity rollout on `engine`.
	#[new]
	pub fn new(engine: &PythonEngine, config: &PythonRolloutConfig) -> PyResult<Self> {
		oa::ml::RolloutBuffer::new(&engine.inner, config.inner.clone())
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Record one fused device-side append into the current time step.
	pub fn append(&mut self, transition: &PythonRolloutTransition) -> PyResult<()> {
		self.inner.append(&transition.inner).map_err(python_error)
	}

	/// Record generalized advantage estimation into the retained batch.
	pub fn finalize(&mut self, config: &PythonGaeConfig) -> PyResult<()> {
		self.inner.finalize(config.inner).map_err(python_error)
	}

	/// Begin a new collection cycle and clear the validity mask on the device.
	pub fn reset(&mut self) -> PyResult<()> {
		self.inner.reset().map_err(python_error)
	}

	/// Return whether all configured time steps have been appended.
	pub fn is_full(&self) -> bool {
		self.inner.is_full()
	}

	/// Return whether final GAE has been recorded for this cycle.
	pub fn is_finalized(&self) -> bool {
		self.inner.is_finalized()
	}

	/// Return the number of appended time steps.
	pub fn len(&self) -> usize {
		self.inner.len()
	}

	/// Return whether no time steps have been appended.
	pub fn is_empty(&self) -> bool {
		self.inner.is_empty()
	}

	/// Return the fixed time-step capacity.
	pub fn capacity(&self) -> usize {
		self.inner.capacity()
	}

	/// Return the observation matrix from the retained batch.
	pub fn batch_observation(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.batch().observation().clone())
	}

	/// Return the action matrix from the retained batch.
	pub fn batch_action(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.batch().action().clone())
	}

	/// Return the reward matrix from the retained batch.
	pub fn batch_reward(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.batch().reward().clone())
	}

	/// Return the value matrix from the retained batch.
	pub fn batch_value(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.batch().value().clone())
	}

	/// Return the next-value matrix from the retained batch.
	pub fn batch_next_value(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.batch().next_value().clone())
	}

	/// Return the old log-probability matrix from the retained batch.
	pub fn batch_old_log_probability(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.batch().old_log_probability().clone())
	}

	/// Return the terminated matrix from the retained batch.
	pub fn batch_terminated(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.batch().terminated().clone())
	}

	/// Return the truncated matrix from the retained batch.
	pub fn batch_truncated(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.batch().truncated().clone())
	}

	/// Return the validity mask from the retained batch.
	pub fn batch_valid(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.batch().valid().clone())
	}

	/// Return the generalized advantage from the retained batch.
	pub fn batch_advantage(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.batch().advantage().clone())
	}

	/// Return the value targets (returns) from the retained batch.
	pub fn batch_returns(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.batch().returns().clone())
	}

	pub fn __repr__(&self) -> String {
		format!(
			"RolloutBuffer(len={}, capacity={}, finalized={})",
			self.inner.len(),
			self.inner.capacity(),
			self.inner.is_finalized(),
		)
	}
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonRolloutConfig>()?;
	module.add_class::<PythonRolloutTransition>()?;
	module.add_class::<PythonRolloutBuffer>()?;
	Ok(())
}
