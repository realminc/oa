use pyo3::{PyErr, exceptions::PyRuntimeError};

pub(crate) fn python_error(error: oa::Error) -> PyErr {
	PyRuntimeError::new_err(error.to_string())
}
