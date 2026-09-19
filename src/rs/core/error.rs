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
	/// A referenced index or range lies outside its owning object.
	OutOfRange,
	/// A requested object or provenance record does not exist.
	NotFound,
	/// An object with the same identity already exists.
	AlreadyExists,
	/// Optimistic concurrency rejected work against a stale revision.
	Aborted,
	/// Application callback policy failed at an explicit lifecycle hook.
	CallbackFailure,
	/// A persisted OA model/checkpoint failed its wire-format integrity contract.
	CheckpointCorrupt,
	/// External encoded data is truncated, malformed, or internally inconsistent.
	DataLoss,
	/// OA detected an invalid invariant in its own state.
	Internal,
	/// A host filesystem or stream operation failed.
	Io,
}

/// Error returned by fallible OA operations.
///
/// Backend and dependency errors are retained as sources without exposing their
/// concrete types through OA's public API.
pub struct Error {
	inner: Box<ErrorData>,
}

struct ErrorData {
	kind: ErrorKind,
	message: String,
	source: Option<Box<dyn StdError + Send + Sync + 'static>>,
}

impl Error {
	fn new(data: ErrorData) -> Self {
		Self {
			inner: Box::new(data),
		}
	}

	/// Return the stable category for this failure.
	pub const fn kind(&self) -> ErrorKind {
		self.inner.kind
	}

	/// Return the contextual failure message.
	pub fn message(&self) -> &str {
		&self.inner.message
	}

	/// Construct an application callback failure without losing its message.
	///
	/// Callback implementations may use this when their failure does not already
	/// originate from another OA operation.
	pub fn callback(message: impl Into<String>) -> Self {
		Self::new(ErrorData {
			kind: ErrorKind::CallbackFailure,
			message: message.into(),
			source: None,
		})
	}

	pub(crate) fn backend_unavailable<E>(backend: &'static str, source: E) -> Self
	where
		E: StdError + Send + Sync + 'static,
	{
		Self::new(ErrorData {
			kind: ErrorKind::BackendUnavailable,
			message: format!("{backend} backend is unavailable"),
			source: Some(Box::new(source)),
		})
	}

	pub(crate) fn invalid_argument(message: impl Into<String>) -> Self {
		Self::new(ErrorData {
			kind: ErrorKind::InvalidArgument,
			message: message.into(),
			source: None,
		})
	}

	pub(crate) fn not_ready(message: impl Into<String>) -> Self {
		Self::new(ErrorData {
			kind: ErrorKind::NotReady,
			message: message.into(),
			source: None,
		})
	}

	pub(crate) fn backend_failure<E>(
		backend: &'static str,
		operation: &'static str,
		source: E,
	) -> Self
	where
		E: StdError + Send + Sync + 'static,
	{
		Self::new(ErrorData {
			kind: ErrorKind::BackendFailure,
			message: format!("{backend} {operation} failed"),
			source: Some(Box::new(source)),
		})
	}

	pub(crate) fn unsupported_backend_version(
		backend: &'static str,
		required: &'static str,
		available: String,
	) -> Self {
		Self::new(ErrorData {
			kind: ErrorKind::UnsupportedBackendVersion,
			message: format!(
				"{backend} {required} or newer is required; the loader provides {available}"
			),
			source: None,
		})
	}

	pub(crate) fn no_suitable_device(message: impl Into<String>) -> Self {
		Self::new(ErrorData {
			kind: ErrorKind::NoSuitableDevice,
			message: message.into(),
			source: None,
		})
	}

	pub(crate) fn missing_capability(message: impl Into<String>) -> Self {
		Self::new(ErrorData {
			kind: ErrorKind::MissingCapability,
			message: message.into(),
			source: None,
		})
	}

	pub(crate) fn resource_exhausted(message: impl Into<String>) -> Self {
		Self::new(ErrorData {
			kind: ErrorKind::ResourceExhausted,
			message: message.into(),
			source: None,
		})
	}

	pub(crate) fn failed_precondition(message: impl Into<String>) -> Self {
		Self::new(ErrorData {
			kind: ErrorKind::FailedPrecondition,
			message: message.into(),
			source: None,
		})
	}

	pub(crate) fn out_of_range(message: impl Into<String>) -> Self {
		Self::new(ErrorData {
			kind: ErrorKind::OutOfRange,
			message: message.into(),
			source: None,
		})
	}

	pub(crate) fn not_found(message: impl Into<String>) -> Self {
		Self::new(ErrorData {
			kind: ErrorKind::NotFound,
			message: message.into(),
			source: None,
		})
	}

	pub(crate) fn already_exists(message: impl Into<String>) -> Self {
		Self::new(ErrorData {
			kind: ErrorKind::AlreadyExists,
			message: message.into(),
			source: None,
		})
	}

	pub(crate) fn internal(message: impl Into<String>) -> Self {
		Self::new(ErrorData {
			kind: ErrorKind::Internal,
			message: message.into(),
			source: None,
		})
	}

	pub(crate) fn checkpoint_corrupt(message: impl Into<String>) -> Self {
		Self::new(ErrorData {
			kind: ErrorKind::CheckpointCorrupt,
			message: format!("corrupt .oam: {}", message.into()),
			source: None,
		})
	}

	pub(crate) fn data_loss(message: impl Into<String>) -> Self {
		Self::new(ErrorData {
			kind: ErrorKind::DataLoss,
			message: message.into(),
			source: None,
		})
	}

	pub(crate) fn io(operation: &'static str, source: std::io::Error) -> Self {
		Self::new(ErrorData {
			kind: ErrorKind::Io,
			message: format!("{operation} failed"),
			source: Some(Box::new(source)),
		})
	}
}

impl fmt::Debug for Error {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter
			.debug_struct("Error")
			.field("kind", &self.inner.kind)
			.field("message", &self.inner.message)
			.field("source", &self.inner.source)
			.finish()
	}
}

impl fmt::Display for Error {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter.write_str(&self.inner.message)
	}
}

impl StdError for Error {
	fn source(&self) -> Option<&(dyn StdError + 'static)> {
		self
			.inner
			.source
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
	fn success_results_keep_the_error_payload_off_the_stack() {
		assert_eq!(size_of::<Error>(), size_of::<usize>());
		assert_eq!(size_of::<super::Result<()>>(), size_of::<usize>());
		fn send_sync<T: Send + Sync>() {}
		send_sync::<Error>();
	}

	#[test]
	fn boxed_payload_preserves_debug_and_releases_its_source_once() {
		use std::sync::{
			Arc,
			atomic::{AtomicUsize, Ordering},
		};
		#[derive(Debug)]
		struct Source(Arc<AtomicUsize>);
		impl std::fmt::Display for Source {
			fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
				f.write_str("source")
			}
		}
		impl std::error::Error for Source {}
		impl Drop for Source {
			fn drop(&mut self) {
				self.0.fetch_add(1, Ordering::SeqCst);
			}
		}
		let drops = Arc::new(AtomicUsize::new(0));
		let error = Error::backend_failure("test", "operation", Source(Arc::clone(&drops)));
		assert!(error.source().unwrap().downcast_ref::<Source>().is_some());
		assert_eq!(drops.load(Ordering::SeqCst), 0);
		drop(error);
		assert_eq!(drops.load(Ordering::SeqCst), 1);
		let error = Error::invalid_argument("bad input");
		assert_eq!(error.to_string(), "bad input");
		assert_eq!(
			format!("{error:?}"),
			"Error { kind: InvalidArgument, message: \"bad input\", source: None }"
		);
	}

	#[test]
	fn backend_source_is_preserved_without_entering_the_public_kind() {
		let error = Error::backend_failure("test", "operation", std::io::Error::other("source detail"));

		assert_eq!(error.kind(), ErrorKind::BackendFailure);
		assert_eq!(error.message(), "test operation failed");
		assert_eq!(
			error.source().map(ToString::to_string).as_deref(),
			Some("source detail")
		);
	}
}
