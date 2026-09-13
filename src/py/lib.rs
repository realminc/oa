use pyo3::prelude::*;

mod audio;
mod error;
mod image;
mod matrix;
mod ml;
mod runtime;

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
	audio::register(module)?;
	image::register(module)?;
	module.add_class::<runtime::PythonEngine>()?;
	module.add_class::<runtime::PythonEvent>()?;
	matrix::register(module)?;
	ml::register(module)?;
	Ok(())
}
