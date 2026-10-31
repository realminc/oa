use pyo3::{
	PyErr,
	exceptions::{PyIOError, PyKeyError, PyRuntimeError, PyValueError},
};

pub(crate) fn python_error(error: oa::Error) -> PyErr {
	let message = error.to_string();
	match error.kind() {
		oa::ErrorKind::InvalidArgument | oa::ErrorKind::OutOfRange | oa::ErrorKind::DataLoss => {
			PyValueError::new_err(message)
		}
		oa::ErrorKind::NotFound => PyKeyError::new_err(message),
		oa::ErrorKind::Io => PyIOError::new_err(message),
		_ => PyRuntimeError::new_err(message),
	}
}
