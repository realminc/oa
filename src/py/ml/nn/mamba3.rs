use pyo3::prelude::*;

use crate::{error::python_error, matrix::PythonMatrix, runtime::PythonEngine};

use super::super::autograd::PythonParameter;

// ── Mamba3Config ──────────────────────────────────────────────────────────────

/// Construction contract for the Mamba-3 block.
#[pyclass(name = "Mamba3Config")]
#[derive(Clone)]
pub(crate) struct PythonMamba3Config {
	pub(crate) inner: oa::ml::nn::Mamba3Config,
}

#[pymethods]
impl PythonMamba3Config {
	/// Construct with OA defaults for the given model width.
	///
	/// Accepts `dim` as an alias for `model_width`.  Additional keyword
	/// arguments (`state_dim`, `dt_rank`, `dt_min`, `dt_max`,
	/// `dt_init_floor`, `seed`) are accepted for API compatibility but
	/// ignored — configure the struct fields after construction if needed.
	#[new]
	#[pyo3(signature = (model_width=None, dim=None, state_dim=None, expand=None, head_dim=None, dt_rank=None, dt_min=None, dt_max=None, dt_init_floor=None, seed=None))]
	#[allow(clippy::too_many_arguments)]
	pub fn new(
		model_width: Option<usize>,
		dim: Option<usize>,
		state_dim: Option<usize>,
		expand: Option<usize>,
		head_dim: Option<usize>,
		dt_rank: Option<&str>,
		dt_min: Option<f32>,
		dt_max: Option<f32>,
		dt_init_floor: Option<f32>,
		seed: Option<u64>,
	) -> PyResult<Self> {
		let _ = (state_dim, dt_rank, dt_min, dt_max, dt_init_floor, seed);
		let mw = model_width.or(dim).ok_or_else(|| {
			pyo3::exceptions::PyTypeError::new_err("Mamba3Config requires model_width or dim")
		})?;
		let mut inner = oa::ml::nn::Mamba3Config::new(mw);
		if let Some(e) = expand {
			inner.expand = e;
		}
		if let Some(hd) = head_dim {
			inner.head_dim = hd;
		}
		Ok(Self { inner })
	}

	#[getter]
	pub fn model_width(&self) -> usize {
		self.inner.model_width
	}
	#[setter]
	pub fn set_model_width(&mut self, value: usize) {
		self.inner.model_width = value;
	}

	#[getter]
	pub fn state_size(&self) -> usize {
		self.inner.state_size
	}
	#[setter]
	pub fn set_state_size(&mut self, value: usize) {
		self.inner.state_size = value;
	}

	#[getter]
	pub fn expand(&self) -> usize {
		self.inner.expand
	}
	#[setter]
	pub fn set_expand(&mut self, value: usize) {
		self.inner.expand = value;
	}

	#[getter]
	pub fn head_dim(&self) -> usize {
		self.inner.head_dim
	}
	#[setter]
	pub fn set_head_dim(&mut self, value: usize) {
		self.inner.head_dim = value;
	}

	#[getter]
	pub fn num_groups(&self) -> usize {
		self.inner.num_groups
	}
	#[setter]
	pub fn set_num_groups(&mut self, value: usize) {
		self.inner.num_groups = value;
	}

	#[getter]
	pub fn mimo_rank(&self) -> usize {
		self.inner.mimo_rank
	}
	#[setter]
	pub fn set_mimo_rank(&mut self, value: usize) {
		self.inner.mimo_rank = value;
	}

	#[getter]
	pub fn rope_fraction(&self) -> f32 {
		self.inner.rope_fraction
	}
	#[setter]
	pub fn set_rope_fraction(&mut self, value: f32) {
		self.inner.rope_fraction = value;
	}

	#[getter]
	pub fn dt_min(&self) -> f32 {
		self.inner.dt_min
	}
	#[setter]
	pub fn set_dt_min(&mut self, value: f32) {
		self.inner.dt_min = value;
	}

	#[getter]
	pub fn dt_max(&self) -> f32 {
		self.inner.dt_max
	}
	#[setter]
	pub fn set_dt_max(&mut self, value: f32) {
		self.inner.dt_max = value;
	}

	#[getter]
	pub fn dt_init_floor(&self) -> f32 {
		self.inner.dt_init_floor
	}
	#[setter]
	pub fn set_dt_init_floor(&mut self, value: f32) {
		self.inner.dt_init_floor = value;
	}

	#[getter]
	pub fn a_floor(&self) -> f32 {
		self.inner.a_floor
	}
	#[setter]
	pub fn set_a_floor(&mut self, value: f32) {
		self.inner.a_floor = value;
	}

	#[getter]
	pub fn output_norm(&self) -> bool {
		self.inner.output_norm
	}
	#[setter]
	pub fn set_output_norm(&mut self, value: bool) {
		self.inner.output_norm = value;
	}

	pub fn __repr__(&self) -> String {
		format!(
			"Mamba3Config(model_width={}, state_size={}, expand={}, head_dim={})",
			self.inner.model_width, self.inner.state_size, self.inner.expand, self.inner.head_dim,
		)
	}
}

// ── Mamba3State ───────────────────────────────────────────────────────────────

/// Explicit recurrent cache for one Mamba3 module and batch size.
#[pyclass(name = "Mamba3State", unsendable)]
pub(crate) struct PythonMamba3State {
	pub(crate) inner: oa::ml::nn::Mamba3State,
}

#[pymethods]
impl PythonMamba3State {
	#[getter]
	pub fn batch_size(&self) -> usize {
		self.inner.batch_size()
	}

	pub fn __repr__(&self) -> String {
		format!("Mamba3State(batch_size={})", self.inner.batch_size())
	}
}

// ── Mamba3 ────────────────────────────────────────────────────────────────────

/// Parameter-owning Mamba-3 selective state-space block.
#[pyclass(name = "Mamba3", unsendable)]
pub(crate) struct PythonMamba3 {
	pub(crate) inner: oa::ml::nn::Mamba3,
}

#[pymethods]
impl PythonMamba3 {
	/// Construct a deterministically initialized Mamba-3 block.
	#[new]
	pub fn new(engine: &PythonEngine, config: &PythonMamba3Config, seed: u64) -> PyResult<Self> {
		oa::ml::nn::Mamba3::with_seed(&engine.inner, config.inner, seed)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Evaluate one complete Mamba-3 block (sequence mode).
	pub fn forward(&self, input: &PythonMatrix) -> PyResult<PythonMatrix> {
		self
			.inner
			.forward(&input.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	/// Allocate a zero-initialized recurrent cache for one positive batch size.
	pub fn new_state(&self, batch_size: usize) -> PyResult<PythonMamba3State> {
		self
			.inner
			.new_state(batch_size)
			.map(|inner| PythonMamba3State { inner })
			.map_err(python_error)
	}

	/// Advance one autoregressive token using an explicit recurrent cache.
	pub fn step(
		&self,
		input: &PythonMatrix,
		state: &mut PythonMamba3State,
	) -> PyResult<PythonMatrix> {
		self
			.inner
			.step(&input.inner, &mut state.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	pub fn all_parameters(&self) -> PyResult<Vec<PythonParameter>> {
		use oa::ml::Module as _;
		self
			.inner
			.all_parameters()
			.map(|params| {
				params
					.into_iter()
					.map(|inner| PythonParameter { inner })
					.collect()
			})
			.map_err(python_error)
	}

	pub fn __repr__(&self) -> &str {
		"Mamba3"
	}
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonMamba3Config>()?;
	module.add_class::<PythonMamba3State>()?;
	module.add_class::<PythonMamba3>()?;
	Ok(())
}
