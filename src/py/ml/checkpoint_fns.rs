use pyo3::prelude::*;

use oa::{Matrix, Result as OaResult, ml};

use crate::{
	error::python_error,
	ml::{PythonAdam, PythonAdamW, PythonMuon, PythonSgd, autograd::PythonParameter},
	runtime::PythonEngine,
};

// ── ParamListModule ───────────────────────────────────────────────────────────
// A minimal `oa::ml::Module` that wraps a flat ordered `Parameter` list.
// The `forward` method is never called by `save_checkpoint` or `load_checkpoint`;
// it exists only to satisfy the trait bound.

struct ParamListModule {
	registry: ml::ModuleRegistry,
}

impl ParamListModule {
	fn from_params(params: Vec<ml::Parameter>) -> OaResult<Self> {
		let mut registry = ml::ModuleRegistry::new();
		for (i, param) in params.into_iter().enumerate() {
			registry.register_parameter(format!("p{i}"), param)?;
		}
		Ok(Self { registry })
	}
}

impl ml::Module for ParamListModule {
	fn forward(&self, _input: &Matrix) -> OaResult<Matrix> {
		Err(oa::Error::callback(
			"ParamListModule::forward is not supported",
		))
	}

	fn registry(&self) -> &ml::ModuleRegistry {
		&self.registry
	}
}

// ── save_checkpoint ───────────────────────────────────────────────────────────

/// Save an OA `.oam` checkpoint from a flat ordered parameter list and an optimizer.
///
/// `params` must be the exact complete ordered list produced by the model's
/// `all_parameters()` call. The optimizer must be `Sgd`, `Adam`, `AdamW`, or `Muon`.
#[pyfunction]
pub(crate) fn ml_save_checkpoint(
	path: &str,
	params: Vec<PyRef<'_, PythonParameter>>,
	optimizer: &Bound<'_, PyAny>,
) -> PyResult<()> {
	let rust_params: Vec<ml::Parameter> = params.iter().map(|p| p.inner.clone()).collect();
	let model = ParamListModule::from_params(rust_params).map_err(python_error)?;
	dispatch_save(path, &model, optimizer)
}

/// Load an OA `.oam` checkpoint into a flat ordered parameter list and an optimizer.
///
/// The parameter list and optimizer type must exactly match what was saved.
#[pyfunction]
pub(crate) fn ml_load_checkpoint(
	engine: &PythonEngine,
	path: &str,
	params: Vec<PyRef<'_, PythonParameter>>,
	optimizer: &Bound<'_, PyAny>,
) -> PyResult<()> {
	let rust_params: Vec<ml::Parameter> = params.iter().map(|p| p.inner.clone()).collect();
	let model = ParamListModule::from_params(rust_params).map_err(python_error)?;
	dispatch_load(&engine.inner, path, &model, optimizer)
}

fn dispatch_save(
	path: &str,
	model: &ParamListModule,
	optimizer: &Bound<'_, PyAny>,
) -> PyResult<()> {
	if let Ok(opt) = optimizer.extract::<PyRef<PythonSgd>>() {
		return ml::save_checkpoint(path, model, &opt.inner).map_err(python_error);
	}
	if let Ok(opt) = optimizer.extract::<PyRef<PythonAdam>>() {
		return ml::save_checkpoint(path, model, &opt.inner).map_err(python_error);
	}
	if let Ok(opt) = optimizer.extract::<PyRef<PythonAdamW>>() {
		return ml::save_checkpoint(path, model, &opt.inner).map_err(python_error);
	}
	if let Ok(opt) = optimizer.extract::<PyRef<PythonMuon>>() {
		return ml::save_checkpoint(path, model, &opt.inner).map_err(python_error);
	}
	Err(pyo3::exceptions::PyTypeError::new_err(
		"optimizer must be one of Sgd, Adam, AdamW, or Muon",
	))
}

fn dispatch_load(
	engine: &oa::Engine,
	path: &str,
	model: &ParamListModule,
	optimizer: &Bound<'_, PyAny>,
) -> PyResult<()> {
	if let Ok(mut opt) = optimizer.extract::<PyRefMut<PythonSgd>>() {
		return ml::load_checkpoint(engine, path, model, &mut opt.inner).map_err(python_error);
	}
	if let Ok(mut opt) = optimizer.extract::<PyRefMut<PythonAdam>>() {
		return ml::load_checkpoint(engine, path, model, &mut opt.inner).map_err(python_error);
	}
	if let Ok(mut opt) = optimizer.extract::<PyRefMut<PythonAdamW>>() {
		return ml::load_checkpoint(engine, path, model, &mut opt.inner).map_err(python_error);
	}
	if let Ok(mut opt) = optimizer.extract::<PyRefMut<PythonMuon>>() {
		return ml::load_checkpoint(engine, path, model, &mut opt.inner).map_err(python_error);
	}
	Err(pyo3::exceptions::PyTypeError::new_err(
		"optimizer must be one of Sgd, Adam, AdamW, or Muon",
	))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_function(wrap_pyfunction!(ml_save_checkpoint, module)?)?;
	module.add_function(wrap_pyfunction!(ml_load_checkpoint, module)?)?;
	Ok(())
}
