use pyo3::prelude::*;
use pyo3::types::PyType;

mod audio;
mod cryptography;
mod error;
mod image;
mod matrix;
mod ml;
mod render;
mod runtime;
mod video;
mod vision;

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
	audio::register(module)?;
	cryptography::register(module)?;
	image::register(module)?;
	runtime::register(module)?;
	matrix::register(module)?;
	ml::register(module)?;
	render::register(module)?;
	video::register(module)?;
	vision::register(module)?;
	// Export additional value types at root
	module.add_class::<video::PythonVideoFrame>()?;
	module.add_class::<render::PythonTexture>()?;

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
