//! Python bindings for OA-owned paths and filesystem operations.
//!
//! The native `oa::Path` value retains OA's named-location contract while
//! accepting Python's standard `os.PathLike` inputs through PyO3's `PathBuf`
//! extraction. Filesystem operations remain explicit static operations rather
//! than methods on `Path`.

use std::path::PathBuf;

use pyo3::{prelude::*, types::PyBytes};

use crate::error::python_error;

#[pyclass(name = "Path")]
#[derive(Clone)]
pub(crate) struct PythonPath {
	inner: oa::Path,
}

impl PythonPath {
	fn from_path_buf(path: PathBuf) -> Self {
		Self {
			inner: oa::Path::from(path),
		}
	}

	fn wrap(inner: oa::Path) -> Self {
		Self { inner }
	}

	fn text(&self) -> String {
		self.inner.display().to_string()
	}
}

#[pymethods]
impl PythonPath {
	#[new]
	#[pyo3(signature = (path=None))]
	fn new(path: Option<PathBuf>) -> Self {
		Self::from_path_buf(path.unwrap_or_default())
	}

	#[staticmethod]
	fn empty() -> Self {
		Self::wrap(oa::Path::empty())
	}

	#[staticmethod]
	fn asset() -> Self {
		Self::wrap(oa::Path::asset())
	}

	#[staticmethod]
	fn asset_rel(relative: PathBuf) -> Self {
		Self::wrap(oa::Path::asset_rel(relative))
	}

	#[staticmethod]
	fn var() -> Self {
		Self::wrap(oa::Path::var())
	}

	#[staticmethod]
	fn var_rel(relative: PathBuf) -> Self {
		Self::wrap(oa::Path::var_rel(relative))
	}

	#[staticmethod]
	fn data() -> Self {
		Self::wrap(oa::Path::data())
	}

	#[staticmethod]
	fn data_rel(relative: PathBuf) -> Self {
		Self::wrap(oa::Path::data_rel(relative))
	}

	#[staticmethod]
	fn home() -> Self {
		Self::wrap(oa::Path::home())
	}

	#[staticmethod]
	fn temp() -> Self {
		Self::wrap(oa::Path::temp())
	}

	fn join(&self, component: PathBuf) -> Self {
		Self::wrap(self.inner.join(component))
	}

	fn parent(&self) -> Option<Self> {
		self.inner.parent().map(Self::wrap)
	}

	fn name(&self) -> Option<String> {
		self
			.inner
			.file_name()
			.map(|value| value.to_string_lossy().into_owned())
	}

	fn stem(&self) -> Option<String> {
		self
			.inner
			.file_stem()
			.map(|value| value.to_string_lossy().into_owned())
	}

	fn suffix(&self) -> Option<String> {
		self
			.inner
			.extension()
			.map(|value| value.to_string_lossy().into_owned())
	}

	fn with_extension(&self, extension: String) -> Self {
		Self::wrap(self.inner.with_extension(extension))
	}

	fn is_absolute(&self) -> bool {
		self.inner.is_absolute()
	}

	fn is_relative(&self) -> bool {
		self.inner.is_relative()
	}

	fn exists(&self) -> bool {
		self.inner.exists()
	}

	fn is_file(&self) -> bool {
		self.inner.is_file()
	}

	fn is_dir(&self) -> bool {
		self.inner.is_dir()
	}

	fn __truediv__(&self, component: PathBuf) -> Self {
		Self::wrap(self.inner.join(component))
	}

	fn __eq__(&self, other: &Self) -> bool {
		self.inner == other.inner
	}

	fn __fspath__(&self) -> String {
		self.text()
	}

	fn __str__(&self) -> String {
		self.text()
	}

	fn __repr__(&self) -> String {
		format!("oa.Path({:?})", self.text())
	}
}

#[pyclass(name = "Filesystem")]
pub(crate) struct PythonFilesystem;

#[pymethods]
impl PythonFilesystem {
	#[staticmethod]
	fn exists(path: PathBuf) -> bool {
		oa::Filesystem::exists(&path)
	}

	#[staticmethod]
	fn is_file(path: PathBuf) -> bool {
		oa::Filesystem::is_file(&path)
	}

	#[staticmethod]
	fn is_directory(path: PathBuf) -> bool {
		oa::Filesystem::is_directory(&path)
	}

	#[staticmethod]
	fn get_file_size(path: PathBuf) -> PyResult<u64> {
		oa::Filesystem::get_file_size(&path).map_err(python_error)
	}

	#[staticmethod]
	fn get_last_modified(path: PathBuf) -> PyResult<i64> {
		oa::Filesystem::get_last_modified(&path).map_err(python_error)
	}

	#[staticmethod]
	fn create_directory(path: PathBuf) -> PyResult<()> {
		oa::Filesystem::create_directory(&path).map_err(python_error)
	}

	#[staticmethod]
	fn create_directories(path: PathBuf) -> PyResult<()> {
		oa::Filesystem::create_directories(&path).map_err(python_error)
	}

	#[staticmethod]
	fn remove_file(path: PathBuf) -> PyResult<()> {
		oa::Filesystem::remove_file(&path).map_err(python_error)
	}

	#[staticmethod]
	#[pyo3(signature = (path, recursive=false))]
	fn remove_directory(path: PathBuf, recursive: bool) -> PyResult<()> {
		oa::Filesystem::remove_directory(&path, recursive).map_err(python_error)
	}

	#[staticmethod]
	fn copy(from_path: PathBuf, to_path: PathBuf) -> PyResult<()> {
		oa::Filesystem::copy(&from_path, &to_path).map_err(python_error)
	}

	#[staticmethod]
	fn move_path(from_path: PathBuf, to_path: PathBuf) -> PyResult<()> {
		oa::Filesystem::move_path(&from_path, &to_path).map_err(python_error)
	}

	#[staticmethod]
	#[pyo3(signature = (directory, extension=""))]
	fn list_files(directory: PathBuf, extension: &str) -> PyResult<Vec<PythonPath>> {
		oa::Filesystem::list_files(&directory, extension)
			.map(|paths| paths.into_iter().map(PythonPath::from_path_buf).collect())
			.map_err(python_error)
	}

	#[staticmethod]
	fn list_directories(directory: PathBuf) -> PyResult<Vec<PythonPath>> {
		oa::Filesystem::list_directories(&directory)
			.map(|paths| paths.into_iter().map(PythonPath::from_path_buf).collect())
			.map_err(python_error)
	}

	#[staticmethod]
	#[pyo3(signature = (directory, recursive=false))]
	fn list_all(directory: PathBuf, recursive: bool) -> PyResult<Vec<PythonPath>> {
		oa::Filesystem::list_all(&directory, recursive)
			.map(|paths| paths.into_iter().map(PythonPath::from_path_buf).collect())
			.map_err(python_error)
	}

	#[staticmethod]
	fn read_text(path: PathBuf) -> PyResult<String> {
		oa::Filesystem::read_text(&path).map_err(python_error)
	}

	#[staticmethod]
	fn write_text(path: PathBuf, content: String) -> PyResult<()> {
		oa::Filesystem::write_text(&path, &content).map_err(python_error)
	}

	#[staticmethod]
	fn append_text(path: PathBuf, content: String) -> PyResult<()> {
		oa::Filesystem::append_text(&path, &content).map_err(python_error)
	}

	#[staticmethod]
	fn read_lines(path: PathBuf) -> PyResult<Vec<String>> {
		oa::Filesystem::read_lines(&path).map_err(python_error)
	}

	#[staticmethod]
	fn read_binary<'py>(py: Python<'py>, path: PathBuf) -> PyResult<Bound<'py, PyBytes>> {
		let bytes = oa::Filesystem::read_binary(&path).map_err(python_error)?;
		Ok(PyBytes::new(py, &bytes))
	}

	#[staticmethod]
	fn write_binary(path: PathBuf, data: Vec<u8>) -> PyResult<()> {
		oa::Filesystem::write_binary(&path, &data).map_err(python_error)
	}

	#[staticmethod]
	fn absolute(path: PathBuf) -> PyResult<PythonPath> {
		oa::Filesystem::absolute(&path)
			.map(PythonPath::from_path_buf)
			.map_err(python_error)
	}

	#[staticmethod]
	fn glob(directory: PathBuf, pattern: &str) -> PyResult<Vec<PythonPath>> {
		oa::Filesystem::glob(&directory, pattern)
			.map(|paths| paths.into_iter().map(PythonPath::from_path_buf).collect())
			.map_err(python_error)
	}
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonPath>()?;
	module.add_class::<PythonFilesystem>()?;
	Ok(())
}
