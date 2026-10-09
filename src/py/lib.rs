use pyo3::prelude::*;
use pyo3::types::PyType;

mod audio;
mod core;
mod cryptography;
mod error;
mod image;
mod matrix;
mod ml;
mod plot;
mod runtime;
mod video;
mod viewer;
mod vision;

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add("__version__", env!("CARGO_PKG_VERSION"))?;
	audio::register(module)?;
	core::register(module)?;
	cryptography::register(module)?;
	image::register(module)?;
	runtime::register(module)?;
	matrix::register(module)?;
	ml::register(module)?;
	plot::register(module)?;
	video::register(module)?;
	viewer::register(module)?;
	vision::register(module)?;
	// Export additional value types at root
	module.add_class::<video::PythonVideoFrame>()?;

	// PyO3 otherwise reports these extension types as members of `builtins`,
	// which breaks documentation and stub generators that discover ownership
	// through `type.__module__`.
	for (_, value) in module.dict().iter() {
		if value.is_instance_of::<PyType>() {
			value.setattr("__module__", "oa._native")?;
		}
	}
	Ok(())
}
