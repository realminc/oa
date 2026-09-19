//! Engine-owned structured host logging.

use std::{
	cell::RefCell,
	fmt,
	fs::{self, File, OpenOptions},
	io::{self, Write},
	path::{Path, PathBuf},
	sync::{
		Arc, Mutex, Weak,
		atomic::{AtomicU8, Ordering},
	},
};

use chrono::Local;

use crate::{Error, Result};

/// Severity threshold and record level for OA diagnostics.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
	/// Fine-grained diagnostics; macro call sites compile out in release builds.
	Trace,
	/// Developer-facing state; macro call sites compile out in release builds.
	Debug,
	/// Meaningful lifecycle or capability state.
	Info,
	/// A recoverable degradation or fallback.
	Warn,
	/// A failed action that the caller still handles explicitly.
	Error,
	/// An unrecoverable condition at the current owner; this does not terminate.
	Fatal,
	/// Disable every record.
	Off,
}

impl LogLevel {
	const fn label(self) -> &'static str {
		match self {
			Self::Trace => "TRACE",
			Self::Debug => "DEBUG",
			Self::Info => "INFO ",
			Self::Warn => "WARN ",
			Self::Error => "ERROR",
			Self::Fatal => "FATAL",
			Self::Off => "OFF  ",
		}
	}

	const fn color(self) -> &'static str {
		match self {
			Self::Trace => "\x1b[90m",
			Self::Debug => "\x1b[36m",
			Self::Info => "\x1b[32m",
			Self::Warn => "\x1b[33m",
			Self::Error => "\x1b[31m",
			Self::Fatal => "\x1b[35m",
			Self::Off => "",
		}
	}
}

/// Compact one-to-four-character diagnostic component tag.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LogComponent {
	bytes: [u8; 4],
	len: u8,
}

impl LogComponent {
	/// Foundation values and backend-neutral contracts.
	pub const CORE: Self = Self::constant(*b"CORE", 4);
	/// Device, memory, queue, stream, and descriptor machinery.
	pub const RUNTIME: Self = Self::constant(*b"RT  ", 2);
	/// Engine ownership and lifecycle.
	pub const ENGINE: Self = Self::constant(*b"ENGN", 4);
	/// Graphs, kernels, dispatch, and GEMM.
	pub const COMPUTE: Self = Self::constant(*b"COMP", 4);
	/// Machine-learning operations and sessions.
	pub const ML: Self = Self::constant(*b"ML  ", 2);
	/// Dataset and serialization operations.
	pub const DATA: Self = Self::constant(*b"DATA", 4);
	/// Vision operations.
	pub const VISION: Self = Self::constant(*b"VISN", 4);
	/// Video operations and sessions.
	pub const VIDEO: Self = Self::constant(*b"VID ", 3);
	/// Audio operations and sessions.
	pub const AUDIO: Self = Self::constant(*b"AUD ", 3);
	/// Rendering operations and sessions.
	pub const RENDER: Self = Self::constant(*b"RNDR", 4);
	/// User-interface operations.
	pub const UI: Self = Self::constant(*b"UI  ", 2);
	/// Plotting operations.
	pub const PLOT: Self = Self::constant(*b"PLOT", 4);
	/// Animation operations.
	pub const ANIMATION: Self = Self::constant(*b"ANIM", 4);
	/// Network operations and sessions.
	pub const NETWORK: Self = Self::constant(*b"NET ", 3);
	/// Cryptographic operations.
	pub const CRYPTOGRAPHY: Self = Self::constant(*b"CRYP", 4);
	/// Python binding diagnostics.
	pub const PYTHON: Self = Self::constant(*b"PY  ", 2);
	/// Application and SDK diagnostics.
	pub const APP: Self = Self::constant(*b"APP ", 3);
	/// Model Context Protocol operations.
	pub const MCP: Self = Self::constant(*b"MCP ", 3);

	const fn constant(bytes: [u8; 4], len: u8) -> Self {
		Self { bytes, len }
	}

	/// Construct a custom ASCII component tag.
	///
	/// # Errors
	///
	/// Returns an error unless `tag` contains one through four printable ASCII
	/// characters.
	pub fn new(tag: &str) -> Result<Self> {
		let source = tag.as_bytes();
		if source.is_empty() || source.len() > 4 || !source.iter().all(u8::is_ascii_graphic) {
			return Err(Error::invalid_argument(
				"log component tags require one through four printable ASCII characters",
			));
		}
		let mut bytes = [b' '; 4];
		bytes[..source.len()].copy_from_slice(source);
		let len = u8::try_from(source.len())
			.map_err(|_| Error::invalid_argument("log component tag length exceeds its storage width"))?;
		Ok(Self { bytes, len })
	}

	/// Return this component's unpadded tag.
	pub fn tag(self) -> String {
		self.bytes[..usize::from(self.len)]
			.iter()
			.map(|byte| char::from(*byte))
			.collect()
	}
}

impl fmt::Display for LogComponent {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		for byte in self.bytes {
			fmt::Write::write_char(formatter, char::from(byte))?;
		}
		Ok(())
	}
}

/// Configuration for the logging session owned by an [`Engine`](super::Engine).
#[derive(Clone, Debug)]
#[must_use]
pub struct LogOptions {
	directory: Option<PathBuf>,
	prefix: String,
	minimum_level: LogLevel,
	console_output: bool,
	file_output: bool,
	color_output: bool,
}

impl Default for LogOptions {
	fn default() -> Self {
		Self {
			directory: None,
			prefix: "oa".to_owned(),
			minimum_level: if cfg!(debug_assertions) {
				LogLevel::Debug
			} else {
				LogLevel::Info
			},
			console_output: true,
			file_output: false,
			color_output: true,
		}
	}
}

impl LogOptions {
	/// Create default console logging options.
	pub fn new() -> Self {
		Self::default()
	}

	/// Select the directory used when file output is enabled.
	pub fn directory(mut self, directory: impl Into<PathBuf>) -> Self {
		self.directory = Some(directory.into());
		self
	}

	/// Select the safe filename prefix used when file output is enabled.
	pub fn prefix(mut self, prefix: impl Into<String>) -> Self {
		self.prefix = prefix.into();
		self
	}

	/// Set the minimum emitted severity.
	pub const fn minimum_level(mut self, minimum_level: LogLevel) -> Self {
		self.minimum_level = minimum_level;
		self
	}

	/// Enable or disable synchronous standard-error output.
	pub const fn console_output(mut self, enabled: bool) -> Self {
		self.console_output = enabled;
		self
	}

	/// Enable or disable append-only dated file output.
	pub const fn file_output(mut self, enabled: bool) -> Self {
		self.file_output = enabled;
		self
	}

	/// Enable or disable ANSI severity colors on console output.
	pub const fn color_output(mut self, enabled: bool) -> Self {
		self.color_output = enabled;
		self
	}

	/// Return the configured minimum severity.
	pub const fn level(&self) -> LogLevel {
		self.minimum_level
	}
}

struct StoredIoError {
	kind: io::ErrorKind,
	message: String,
}

impl StoredIoError {
	fn new(error: &io::Error) -> Self {
		Self {
			kind: error.kind(),
			message: error.to_string(),
		}
	}

	fn to_error(&self) -> Error {
		Error::io(
			"logging output",
			io::Error::new(self.kind, self.message.clone()),
		)
	}
}

struct LogState {
	file: Option<File>,
	path: Option<PathBuf>,
	console_output: bool,
	color_output: bool,
	open: bool,
	first_error: Option<StoredIoError>,
}

struct LogInner {
	minimum_level: AtomicU8,
	state: Mutex<LogState>,
}

pub(crate) struct Logger {
	inner: Arc<LogInner>,
}

impl Logger {
	pub(crate) fn new(options: LogOptions) -> Result<Self> {
		if options.prefix.is_empty()
			|| !options
				.prefix
				.bytes()
				.all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
		{
			return Err(Error::invalid_argument(
				"log filename prefix requires ASCII letters, digits, '-' or '_'",
			));
		}

		let (file, path) = if options.file_output {
			let directory = options
				.directory
				.as_deref()
				.ok_or_else(|| Error::invalid_argument("log file output requires a directory"))?;
			fs::create_dir_all(directory)
				.map_err(|source| Error::io("log directory creation", source))?;
			let path = log_path(directory, &options.prefix);
			let file = OpenOptions::new()
				.create(true)
				.append(true)
				.open(&path)
				.map_err(|source| Error::io("log file opening", source))?;
			(Some(file), Some(path))
		} else {
			(None, None)
		};

		Ok(Self {
			inner: Arc::new(LogInner {
				minimum_level: AtomicU8::new(options.minimum_level as u8),
				state: Mutex::new(LogState {
					file,
					path,
					console_output: options.console_output,
					color_output: options.color_output,
					open: true,
					first_error: None,
				}),
			}),
		})
	}

	pub(crate) fn select(&self) -> LogSelection {
		let selected = Arc::downgrade(&self.inner);
		let previous = CURRENT_LOG
			.try_with(|current| {
				current
					.try_borrow_mut()
					.map(|mut current| std::mem::replace(&mut *current, selected.clone()))
					.ok()
			})
			.ok()
			.flatten()
			.unwrap_or_default();
		LogSelection { selected, previous }
	}

	pub(crate) fn write_text(
		&self,
		level: LogLevel,
		component: LogComponent,
		text: &str,
	) -> Result<()> {
		self.inner.write_text(level, component, text)
	}

	pub(crate) fn set_level(&self, level: LogLevel) {
		self
			.inner
			.minimum_level
			.store(level as u8, Ordering::Relaxed);
	}

	pub(crate) fn level(&self) -> LogLevel {
		level_from_u8(self.inner.minimum_level.load(Ordering::Relaxed))
	}

	pub(crate) fn path(&self) -> Option<PathBuf> {
		self.inner.lock_state().path.clone()
	}

	pub(crate) fn flush(&self) -> Result<()> {
		self.inner.flush()
	}

	pub(crate) fn close(&self) -> Result<()> {
		self.inner.close()
	}
}

impl LogInner {
	fn lock_state(&self) -> std::sync::MutexGuard<'_, LogState> {
		match self.state.lock() {
			Ok(state) => state,
			Err(poisoned) => poisoned.into_inner(),
		}
	}

	fn should_write(&self, level: LogLevel) -> bool {
		level != LogLevel::Off && (level as u8) >= self.minimum_level.load(Ordering::Relaxed)
	}

	fn write_text(&self, level: LogLevel, component: LogComponent, text: &str) -> Result<()> {
		if !self.should_write(level) {
			return Ok(());
		}
		let timestamp = Local::now().format("%H:%M:%S%.3f");
		let record = format!("{timestamp} [{}] [{component}] {text}\n", level.label());
		let mut state = self.lock_state();
		if !state.open {
			return Err(Error::failed_precondition("logger is closed"));
		}

		if let Some(file) = &mut state.file
			&& let Err(error) = file.write_all(record.as_bytes())
		{
			state
				.first_error
				.get_or_insert_with(|| StoredIoError::new(&error));
		}
		if state.console_output {
			let result = if state.color_output {
				write!(io::stderr().lock(), "{}{}\x1b[0m", level.color(), record)
			} else {
				io::stderr().lock().write_all(record.as_bytes())
			};
			if let Err(error) = result {
				state
					.first_error
					.get_or_insert_with(|| StoredIoError::new(&error));
			}
		}
		match &state.first_error {
			Some(error) => Err(error.to_error()),
			None => Ok(()),
		}
	}

	fn flush(&self) -> Result<()> {
		let mut state = self.lock_state();
		if let Some(file) = &mut state.file
			&& let Err(error) = file.flush()
		{
			state
				.first_error
				.get_or_insert_with(|| StoredIoError::new(&error));
		}
		match &state.first_error {
			Some(error) => Err(error.to_error()),
			None => Ok(()),
		}
	}

	fn close(&self) -> Result<()> {
		let mut state = self.lock_state();
		if !state.open {
			return match &state.first_error {
				Some(error) => Err(error.to_error()),
				None => Ok(()),
			};
		}
		if let Some(mut file) = state.file.take()
			&& let Err(error) = file.flush()
		{
			state
				.first_error
				.get_or_insert_with(|| StoredIoError::new(&error));
		}
		state.open = false;
		match &state.first_error {
			Some(error) => Err(error.to_error()),
			None => Ok(()),
		}
	}
}

fn log_path(directory: &Path, prefix: &str) -> PathBuf {
	directory.join(format!("{prefix}_{}.log", Local::now().format("%Y%m%d")))
}

fn level_from_u8(value: u8) -> LogLevel {
	match value {
		0 => LogLevel::Trace,
		1 => LogLevel::Debug,
		2 => LogLevel::Info,
		3 => LogLevel::Warn,
		4 => LogLevel::Error,
		5 => LogLevel::Fatal,
		_ => LogLevel::Off,
	}
}

thread_local! {
	static CURRENT_LOG: RefCell<Weak<LogInner>> = const { RefCell::new(Weak::new()) };
}

pub(crate) struct LogSelection {
	selected: Weak<LogInner>,
	previous: Weak<LogInner>,
}

impl Drop for LogSelection {
	fn drop(&mut self) {
		let _ = CURRENT_LOG.try_with(|current| {
			let Ok(mut current) = current.try_borrow_mut() else {
				return;
			};
			if Weak::ptr_eq(&current, &self.selected) {
				*current = self.previous.clone();
			}
		});
	}
}

fn selected() -> Option<Arc<LogInner>> {
	CURRENT_LOG
		.try_with(|current| {
			current
				.try_borrow()
				.ok()
				.and_then(|current| current.upgrade())
		})
		.ok()
		.flatten()
}

#[doc(hidden)]
pub fn __log_should_write(level: LogLevel) -> bool {
	level != LogLevel::Off && selected().is_none_or(|logger| logger.should_write(level))
}

#[doc(hidden)]
pub fn __log_write(level: LogLevel, component: LogComponent, arguments: fmt::Arguments<'_>) {
	let text = arguments.to_string();
	if let Some(logger) = selected() {
		let _ = logger.write_text(level, component, &text);
	} else {
		let timestamp = Local::now().format("%H:%M:%S%.3f");
		let record = format!(
			"{}{} [{}] [{component}] {text}\x1b[0m\n",
			level.color(),
			timestamp,
			level.label()
		);
		let _ = io::stderr().lock().write_all(record.as_bytes());
	}
}

#[macro_export]
/// Write an informational record through the selected engine logger.
macro_rules! log_info {
	($component:expr, $($argument:tt)*) => {{
		if $crate::runtime::__log_should_write($crate::LogLevel::Info) {
			$crate::runtime::__log_write(
				$crate::LogLevel::Info,
				$component,
				format_args!($($argument)*),
			);
		}
	}};
}

#[macro_export]
/// Write a recoverable-warning record through the selected engine logger.
macro_rules! log_warn {
	($component:expr, $($argument:tt)*) => {{
		if $crate::runtime::__log_should_write($crate::LogLevel::Warn) {
			$crate::runtime::__log_write(
				$crate::LogLevel::Warn,
				$component,
				format_args!($($argument)*),
			);
		}
	}};
}

#[macro_export]
/// Write an error record without changing control flow.
macro_rules! log_error {
	($component:expr, $($argument:tt)*) => {{
		if $crate::runtime::__log_should_write($crate::LogLevel::Error) {
			$crate::runtime::__log_write(
				$crate::LogLevel::Error,
				$component,
				format_args!($($argument)*),
			);
		}
	}};
}

#[macro_export]
/// Write a fatal-severity record without terminating execution.
macro_rules! log_fatal {
	($component:expr, $($argument:tt)*) => {{
		if $crate::runtime::__log_should_write($crate::LogLevel::Fatal) {
			$crate::runtime::__log_write(
				$crate::LogLevel::Fatal,
				$component,
				format_args!($($argument)*),
			);
		}
	}};
}

#[cfg(debug_assertions)]
#[macro_export]
/// Write a trace record in debug builds; release builds do not evaluate arguments.
macro_rules! log_trace {
	($component:expr, $($argument:tt)*) => {{
		if $crate::runtime::__log_should_write($crate::LogLevel::Trace) {
			$crate::runtime::__log_write(
				$crate::LogLevel::Trace,
				$component,
				format_args!($($argument)*),
			);
		}
	}};
}

#[cfg(not(debug_assertions))]
#[macro_export]
/// Compile out a trace record without evaluating its arguments.
macro_rules! log_trace {
	($component:expr, $($argument:tt)*) => {{}};
}

#[cfg(debug_assertions)]
#[macro_export]
/// Write a debug record in debug builds; release builds do not evaluate arguments.
macro_rules! log_debug {
	($component:expr, $($argument:tt)*) => {{
		if $crate::runtime::__log_should_write($crate::LogLevel::Debug) {
			$crate::runtime::__log_write(
				$crate::LogLevel::Debug,
				$component,
				format_args!($($argument)*),
			);
		}
	}};
}

#[cfg(not(debug_assertions))]
#[macro_export]
/// Compile out a debug record without evaluating its arguments.
macro_rules! log_debug {
	($component:expr, $($argument:tt)*) => {{}};
}

#[cfg(test)]
mod tests {
	use std::{fs, path::Path, process};

	use super::{LogComponent, LogLevel, LogOptions, Logger};
	use crate::ErrorKind;

	fn test_directory(name: &str) -> std::path::PathBuf {
		std::env::temp_dir().join(format!("oars-log-{name}-{}", process::id()))
	}

	#[test]
	fn validates_custom_component_tags() -> crate::Result<()> {
		assert_eq!(LogComponent::new("DNA")?.tag(), "DNA");
		for invalid in ["", "TOO-LONG", "é"] {
			assert_eq!(
				LogComponent::new(invalid).map_err(|error| error.kind()),
				Err(ErrorKind::InvalidArgument)
			);
		}
		Ok(())
	}

	#[test]
	fn filters_flushes_and_closes_file_output() -> crate::Result<()> {
		let directory = test_directory("file");
		let _ = fs::remove_dir_all(&directory);
		let logger = Logger::new(
			LogOptions::new()
				.directory(&directory)
				.prefix("test")
				.minimum_level(LogLevel::Info)
				.console_output(false)
				.file_output(true),
		)?;
		logger.write_text(LogLevel::Debug, LogComponent::CORE, "filtered")?;
		logger.write_text(LogLevel::Info, LogComponent::CORE, "written")?;
		logger.flush()?;
		let path = logger
			.path()
			.ok_or_else(|| crate::Error::failed_precondition("file logger did not expose its path"))?;
		let contents =
			fs::read_to_string(path).map_err(|source| crate::Error::io("test log reading", source))?;
		assert!(!contents.contains("filtered"));
		assert!(contents.contains("[INFO ] [CORE] written"));
		logger.close()?;
		assert_eq!(
			logger
				.write_text(LogLevel::Info, LogComponent::CORE, "late")
				.map_err(|error| error.kind()),
			Err(ErrorKind::FailedPrecondition)
		);
		fs::remove_dir_all(directory).map_err(|source| crate::Error::io("test log cleanup", source))?;
		Ok(())
	}

	#[test]
	fn nested_thread_selection_restores_the_previous_logger() -> crate::Result<()> {
		let first_directory = test_directory("first");
		let second_directory = test_directory("second");
		let _ = fs::remove_dir_all(&first_directory);
		let _ = fs::remove_dir_all(&second_directory);
		let options = |directory: &Path| {
			LogOptions::new()
				.directory(directory)
				.console_output(false)
				.file_output(true)
		};
		let first = Logger::new(options(&first_directory).prefix("first"))?;
		let second = Logger::new(options(&second_directory).prefix("second"))?;
		let first_selection = first.select();
		crate::log_info!(LogComponent::CORE, "first-a");
		{
			let _second_selection = second.select();
			crate::log_info!(LogComponent::CORE, "second");
		}
		crate::log_info!(LogComponent::CORE, "first-b");
		drop(first_selection);
		first.flush()?;
		second.flush()?;
		let first_path = first
			.path()
			.ok_or_else(|| crate::Error::failed_precondition("first logger did not expose its path"))?;
		let second_path = second
			.path()
			.ok_or_else(|| crate::Error::failed_precondition("second logger did not expose its path"))?;
		let first_text = fs::read_to_string(first_path)
			.map_err(|source| crate::Error::io("first test log reading", source))?;
		let second_text = fs::read_to_string(second_path)
			.map_err(|source| crate::Error::io("second test log reading", source))?;
		assert!(first_text.contains("first-a"));
		assert!(first_text.contains("first-b"));
		assert!(!first_text.contains("second"));
		assert!(second_text.contains("second"));
		first.close()?;
		second.close()?;
		fs::remove_dir_all(first_directory)
			.map_err(|source| crate::Error::io("first test log cleanup", source))?;
		fs::remove_dir_all(second_directory)
			.map_err(|source| crate::Error::io("second test log cleanup", source))?;
		Ok(())
	}

	#[test]
	fn filtered_macro_does_not_evaluate_its_arguments() -> crate::Result<()> {
		let logger = Logger::new(
			LogOptions::new()
				.minimum_level(LogLevel::Off)
				.console_output(false),
		)?;
		let _selection = logger.select();
		let evaluated = std::cell::Cell::new(false);
		crate::log_info!(LogComponent::CORE, "{}", {
			evaluated.set(true);
			"side effect"
		});
		assert!(!evaluated.get());
		Ok(())
	}

	#[cfg(not(debug_assertions))]
	#[test]
	fn release_debug_macro_compiles_out_without_evaluating_arguments() {
		let evaluated = std::cell::Cell::new(false);
		crate::log_debug!(LogComponent::CORE, "{}", {
			evaluated.set(true);
			"side effect"
		});
		assert!(!evaluated.get());
	}
}
