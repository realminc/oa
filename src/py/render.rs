use pyo3::prelude::*;

use crate::{error::python_error, runtime::PythonEngine};

#[pyclass(name = "Texture", unsendable)]
pub(crate) struct PythonTexture {
	inner: oa::render::Texture,
}

impl PythonTexture {
	pub(crate) fn wrap(inner: oa::render::Texture) -> Self {
		Self { inner }
	}
}

#[pymethods]
impl PythonTexture {
	/// Create a Texture from a flat RGBA8 byte slice, width, and height.
	#[staticmethod]
	#[pyo3(signature = (engine, rgba, width, height))]
	fn from_rgba8(
		engine: &PythonEngine,
		rgba: Vec<u8>,
		width: usize,
		height: usize,
	) -> PyResult<Self> {
		oa::render::texture_from_rgba8(&engine.inner, &rgba, width, height)
			.map(Self::wrap)
			.map_err(python_error)
	}

	fn read_rgba8(&self) -> PyResult<Vec<u8>> {
		self.inner.read_rgba8().map_err(python_error)
	}

	fn width(&self) -> usize {
		self.inner.width()
	}

	fn height(&self) -> usize {
		self.inner.height()
	}

	fn __repr__(&self) -> String {
		format!(
			"Texture(width={}, height={}, dtype='{}')",
			self.inner.width(),
			self.inner.height(),
			self.inner.dtype().token(),
		)
	}
}

#[pyfunction]
#[pyo3(signature = (texture, path, quality=90))]
pub(crate) fn render_save_texture_file(
	texture: &PythonTexture,
	path: &str,
	quality: u32,
) -> PyResult<()> {
	oa::render::save_texture_file(&texture.inner, path, quality).map_err(python_error)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonTexture>()?;
	module.add_function(wrap_pyfunction!(render_save_texture_file, module)?)?;
	Ok(())
}
