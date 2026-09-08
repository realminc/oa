use std::{error::Error as StdError, fmt};

/// Stable category for an OA failure.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorKind {
	/// A caller supplied an invalid value, shape, range, or combination.
	InvalidArgument,
	/// A value or operation has not reached the required completion state.
	NotReady,
	/// The selected execution backend could not be loaded.
	BackendUnavailable,
	/// An execution-backend operation failed.
	BackendFailure,
	/// The execution backend is older than OA's minimum supported version.
	UnsupportedBackendVersion,
	/// No device satisfies the requested execution contract.
	NoSuitableDevice,
	/// A required device or runtime capability is unavailable.
	MissingCapability,
	/// A finite runtime identity, counter, or address space was exhausted.
	ResourceExhausted,
	/// An operation was requested in an incompatible lifecycle state.
	FailedPrecondition,
	/// A host filesystem or stream operation failed.
	Io,
}

/// Error returned by fallible OA operations.
///
/// Backend and dependency errors are retained as sources without exposing their
/// concrete types through OA's public API.
#[derive(Debug)]
pub struct Error {
	kind: ErrorKind,
	message: String,
	source: Option<Box<dyn StdError + Send + Sync + 'static>>,
}

impl Error {
	/// Return the stable category for this failure.
	pub const fn kind(&self) -> ErrorKind {
		self.kind
	}

	/// Return the contextual failure message.
	pub fn message(&self) -> &str {
		&self.message
	}

	pub(crate) fn backend_unavailable<E>(backend: &'static str, source: E) -> Self
	where
		E: StdError + Send + Sync + 'static,
	{
		Self {
			kind: ErrorKind::BackendUnavailable,
			message: format!("{backend} backend is unavailable"),
			source: Some(Box::new(source)),
		}
	}

	pub(crate) fn invalid_argument(message: impl Into<String>) -> Self {
		Self {
			kind: ErrorKind::InvalidArgument,
			message: message.into(),
			source: None,
		}
	}

	pub(crate) fn not_ready(message: impl Into<String>) -> Self {
		Self {
			kind: ErrorKind::NotReady,
			message: message.into(),
			source: None,
		}
	}

	pub(crate) fn backend_failure<E>(
		backend: &'static str,
		operation: &'static str,
		source: E,
	) -> Self
	where
		E: StdError + Send + Sync + 'static,
	{
		Self {
			kind: ErrorKind::BackendFailure,
			message: format!("{backend} {operation} failed"),
			source: Some(Box::new(source)),
		}
	}

	pub(crate) fn unsupported_backend_version(
		backend: &'static str,
		required: &'static str,
		available: String,
	) -> Self {
		Self {
			kind: ErrorKind::UnsupportedBackendVersion,
			message: format!(
				"{backend} {required} or newer is required; the loader provides {available}"
			),
			source: None,
		}
	}

	pub(crate) fn no_suitable_device(message: impl Into<String>) -> Self {
		Self {
			kind: ErrorKind::NoSuitableDevice,
			message: message.into(),
			source: None,
		}
	}

	pub(crate) fn missing_capability(message: impl Into<String>) -> Self {
		Self {
			kind: ErrorKind::MissingCapability,
			message: message.into(),
			source: None,
		}
	}

	pub(crate) fn resource_exhausted(message: impl Into<String>) -> Self {
		Self {
			kind: ErrorKind::ResourceExhausted,
			message: message.into(),
			source: None,
		}
	}

	pub(crate) fn failed_precondition(message: impl Into<String>) -> Self {
		Self {
			kind: ErrorKind::FailedPrecondition,
			message: message.into(),
			source: None,
		}
	}

	pub(crate) fn io(operation: &'static str, source: std::io::Error) -> Self {
		Self {
			kind: ErrorKind::Io,
			message: format!("{operation} failed"),
			source: Some(Box::new(source)),
		}
	}
}

impl fmt::Display for Error {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter.write_str(&self.message)
	}
}

impl StdError for Error {
	fn source(&self) -> Option<&(dyn StdError + 'static)> {
		self.source
			.as_deref()
			.map(|source| source as &(dyn StdError + 'static))
	}
}

/// Result returned by fallible OA operations.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
	use super::{Error, ErrorKind};
	use std::error::Error as _;

	#[test]
	fn backend_source_is_preserved_without_entering_the_public_kind() {
		let error =
			Error::backend_failure("test", "operation", std::io::Error::other("source detail"));

		assert_eq!(error.kind(), ErrorKind::BackendFailure);
		assert_eq!(error.message(), "test operation failed");
		assert_eq!(
			error.source().map(ToString::to_string).as_deref(),
			Some("source detail")
		);
	}
}
