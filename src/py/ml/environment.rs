use pyo3::prelude::*;

use crate::{error::python_error, matrix::PythonMatrix};

// ── EnvironmentSpaceKind ──────────────────────────────────────────────────────

/// Structural domain of one reinforcement-learning environment field.
#[pyclass(name = "EnvironmentSpaceKind", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum PythonEnvironmentSpaceKind {
	Box = 0,
	Discrete = 1,
	Binary = 2,
}

impl From<oa::ml::environment::EnvironmentSpaceKind> for PythonEnvironmentSpaceKind {
	fn from(value: oa::ml::environment::EnvironmentSpaceKind) -> Self {
		match value {
			oa::ml::environment::EnvironmentSpaceKind::Box => PythonEnvironmentSpaceKind::Box,
			oa::ml::environment::EnvironmentSpaceKind::Discrete => PythonEnvironmentSpaceKind::Discrete,
			oa::ml::environment::EnvironmentSpaceKind::Binary => PythonEnvironmentSpaceKind::Binary,
		}
	}
}

// ── EnvironmentSpace ──────────────────────────────────────────────────────────

/// Checked shape, dtype, and range contract for one environment field.
#[pyclass(name = "EnvironmentSpace")]
#[derive(Clone)]
pub(crate) struct PythonEnvironmentSpace {
	pub(crate) inner: oa::ml::environment::EnvironmentSpace,
}

#[pymethods]
impl PythonEnvironmentSpace {
	/// Construct a floating-point Box space.
	#[staticmethod]
	pub fn continuous(
		name: &str,
		shape: Vec<usize>,
		dtype: &str,
		minimum: f64,
		maximum: f64,
	) -> PyResult<Self> {
		let dtype = parse_dtype(dtype)?;
		oa::ml::environment::EnvironmentSpace::continuous(name, shape, dtype, minimum, maximum)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Construct a scalar integer category space over `[0, cardinality)`.
	#[staticmethod]
	pub fn discrete(name: &str, cardinality: usize, dtype: &str) -> PyResult<Self> {
		let dtype = parse_dtype(dtype)?;
		oa::ml::environment::EnvironmentSpace::discrete(name, cardinality, dtype)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Construct a U8 binary scalar or shaped field.
	#[staticmethod]
	pub fn binary(name: &str, shape: Vec<usize>) -> PyResult<Self> {
		oa::ml::environment::EnvironmentSpace::binary(name, shape)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	#[getter]
	pub fn name(&self) -> &str {
		self.inner.name()
	}

	#[getter]
	pub fn kind(&self) -> PythonEnvironmentSpaceKind {
		self.inner.kind().into()
	}

	#[getter]
	pub fn shape(&self) -> Vec<usize> {
		self.inner.shape().to_vec()
	}

	#[getter]
	pub fn dtype(&self) -> &'static str {
		self.inner.dtype().token()
	}

	#[getter]
	pub fn minimum(&self) -> f64 {
		self.inner.minimum()
	}

	#[getter]
	pub fn maximum(&self) -> f64 {
		self.inner.maximum()
	}

	#[getter]
	pub fn cardinality(&self) -> usize {
		self.inner.cardinality()
	}

	#[getter]
	pub fn elements_per_environment(&self) -> usize {
		self.inner.elements_per_environment()
	}

	pub fn batched_shape(&self, environments: u32) -> PyResult<Vec<usize>> {
		self.inner.batched_shape(environments).map_err(python_error)
	}

	pub fn validate_matrix(&self, matrix: &PythonMatrix, environments: u32) -> PyResult<()> {
		self
			.inner
			.validate_matrix(&matrix.inner, environments)
			.map_err(python_error)
	}

	pub fn __repr__(&self) -> String {
		format!(
			"EnvironmentSpace(name={:?}, kind={:?}, dtype={})",
			self.inner.name(),
			self.inner.kind(),
			self.inner.dtype().token(),
		)
	}
}

// ── EnvironmentSpec ───────────────────────────────────────────────────────────

/// Complete single-agent environment schema.
#[pyclass(name = "EnvironmentSpec")]
#[derive(Clone)]
pub(crate) struct PythonEnvironmentSpec {
	pub(crate) inner: oa::ml::environment::EnvironmentSpec,
}

#[pymethods]
impl PythonEnvironmentSpec {
	#[new]
	pub fn new(
		observation: &PythonEnvironmentSpace,
		action: &PythonEnvironmentSpace,
		reward: &PythonEnvironmentSpace,
		terminated: &PythonEnvironmentSpace,
		truncated: &PythonEnvironmentSpace,
	) -> PyResult<Self> {
		oa::ml::environment::EnvironmentSpec::new(
			observation.inner.clone(),
			action.inner.clone(),
			reward.inner.clone(),
			terminated.inner.clone(),
			truncated.inner.clone(),
		)
		.map(|inner| Self { inner })
		.map_err(python_error)
	}

	#[getter]
	pub fn observation(&self) -> PythonEnvironmentSpace {
		PythonEnvironmentSpace {
			inner: self.inner.observation().clone(),
		}
	}

	#[getter]
	pub fn action(&self) -> PythonEnvironmentSpace {
		PythonEnvironmentSpace {
			inner: self.inner.action().clone(),
		}
	}

	#[getter]
	pub fn reward(&self) -> PythonEnvironmentSpace {
		PythonEnvironmentSpace {
			inner: self.inner.reward().clone(),
		}
	}

	#[getter]
	pub fn terminated(&self) -> PythonEnvironmentSpace {
		PythonEnvironmentSpace {
			inner: self.inner.terminated().clone(),
		}
	}

	#[getter]
	pub fn truncated(&self) -> PythonEnvironmentSpace {
		PythonEnvironmentSpace {
			inner: self.inner.truncated().clone(),
		}
	}

	pub fn validate_reset(&self, observation: &PythonMatrix, environments: u32) -> PyResult<()> {
		self
			.inner
			.validate_reset(&observation.inner, environments)
			.map_err(python_error)
	}

	pub fn validate_action(&self, action: &PythonMatrix, environments: u32) -> PyResult<()> {
		self
			.inner
			.validate_action(&action.inner, environments)
			.map_err(python_error)
	}

	pub fn __repr__(&self) -> &str {
		"EnvironmentSpec"
	}
}

// ── EnvironmentTransition ─────────────────────────────────────────────────────

/// One batched environment step result preserving termination and truncation.
#[pyclass(name = "EnvironmentTransition", unsendable)]
pub(crate) struct PythonEnvironmentTransition {
	pub(crate) inner: oa::ml::environment::EnvironmentTransition,
}

#[pymethods]
impl PythonEnvironmentTransition {
	#[new]
	pub fn new(
		observation: &PythonMatrix,
		next_observation: &PythonMatrix,
		reward: &PythonMatrix,
		terminated: &PythonMatrix,
		truncated: &PythonMatrix,
	) -> Self {
		Self {
			inner: oa::ml::environment::EnvironmentTransition::new(
				observation.inner.clone(),
				next_observation.inner.clone(),
				reward.inner.clone(),
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
	pub fn next_observation(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.next_observation().clone())
	}

	#[getter]
	pub fn reward(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.reward().clone())
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
		"EnvironmentTransition"
	}
}

// ── Environment preprocessing functions ───────────────────────────────────────

#[pyfunction]
pub(crate) fn ml_normalize_observation(
	observation: &PythonMatrix,
	mean: &PythonMatrix,
	stddev: &PythonMatrix,
	epsilon: f32,
	clip: f32,
) -> PyResult<PythonMatrix> {
	oa::ml::environment::normalize_observation(
		&observation.inner,
		&mean.inner,
		&stddev.inner,
		epsilon,
		clip,
	)
	.map(PythonMatrix::wrap)
	.map_err(python_error)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ml_scale_action(
	action: &PythonMatrix,
	source_minimum: f32,
	source_maximum: f32,
	target_minimum: f32,
	target_maximum: f32,
	clamp_source: bool,
) -> PyResult<PythonMatrix> {
	oa::ml::environment::scale_action(
		&action.inner,
		source_minimum,
		source_maximum,
		target_minimum,
		target_maximum,
		clamp_source,
	)
	.map(PythonMatrix::wrap)
	.map_err(python_error)
}

#[pyfunction]
pub(crate) fn ml_clip_reward(
	reward: &PythonMatrix,
	minimum: f32,
	maximum: f32,
) -> PyResult<PythonMatrix> {
	oa::ml::environment::clip_reward(&reward.inner, minimum, maximum)
		.map(PythonMatrix::wrap)
		.map_err(python_error)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonEnvironmentSpaceKind>()?;
	module.add_class::<PythonEnvironmentSpace>()?;
	module.add_class::<PythonEnvironmentSpec>()?;
	module.add_class::<PythonEnvironmentTransition>()?;
	macro_rules! add_functions {
		($($f:ident),+ $(,)?) => { $(module.add_function(wrap_pyfunction!($f, module)?)?;)+ };
	}
	add_functions!(ml_normalize_observation, ml_scale_action, ml_clip_reward,);
	Ok(())
}

fn parse_dtype(token: &str) -> PyResult<oa::DType> {
	match token {
		"f32" => Ok(oa::DType::F32),
		"i32" => Ok(oa::DType::I32),
		"u8" => Ok(oa::DType::U8),
		"u32" => Ok(oa::DType::U32),
		_ => Err(pyo3::exceptions::PyValueError::new_err(format!(
			"unknown dtype {:?}; expected f32, i32, u8, or u32",
			token
		))),
	}
}
