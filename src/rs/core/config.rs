//! Shared training and logging configuration structs.
//!
//! Port provenance: `oa/core/config.h`.
//!
//! These structs are the shared config layer used by CLI applications,
//! [`crate::ml::training`] callbacks, and logging sessions. They are
//! deliberately plain-data; the application supplies YAML loading by
//! implementing the optional [`crate::Cli`] hooks.
//!
//! # Usage
//!
//! ```rust,ignore
//! let ckpt = oa::CheckpointConfig {
//!     dir:  oa::Path::var_rel("model/dev"),
//!     name: "CharMamba3".into(),
//!     ..Default::default()
//! };
//! let log = oa::LogConfig::default();
//! ```

use crate::Path;

// ─── CheckpointConfig ────────────────────────────────────────────────────────

/// Shared checkpoint-save configuration.
///
/// Used as a building block in CLI configs and passed to
/// [`crate::ml::training::CbCheckpoint`].
#[derive(Clone, Debug, PartialEq)]
pub struct CheckpointConfig {
	/// Directory that receives checkpoint files.
	pub dir: Path,
	/// Base filename stem for checkpoint files.
	pub name: String,
	/// Environment tag appended to checkpoint paths (e.g. `"dev"`, `"prod"`).
	pub env: String,
	/// Save the best-validation-loss checkpoint automatically.
	pub save_best: bool,
	/// Save the latest checkpoint at every epoch boundary.
	pub save_last: bool,
}

impl Default for CheckpointConfig {
	fn default() -> Self {
		Self {
			dir: Path::var_rel("checkpoints"),
			name: "model".into(),
			env: "dev".into(),
			save_best: true,
			save_last: true,
		}
	}
}

// ─── LogConfig ───────────────────────────────────────────────────────────────

/// Shared logging configuration.
///
/// Used as a building block in CLI configs and passed to
/// [`crate::runtime::LogOptions`] or [`crate::LogMetrics`].
#[derive(Clone, Debug, PartialEq)]
pub struct LogConfig {
	/// Directory for log files and `events.jsonl` metrics.
	pub dir: Path,
	/// Minimum level string (`"trace"`, `"debug"`, `"info"`, `"warn"`, `"error"`).
	pub level: String,
	/// Write records to stderr.
	pub console: bool,
	/// Write records to a file inside `dir`.
	pub file: bool,
	/// Write scalar metrics to `{dir}/events.jsonl`.
	pub metrics: bool,
}

impl Default for LogConfig {
	fn default() -> Self {
		Self {
			dir: Path::var_rel("log"),
			level: "info".into(),
			console: true,
			file: false,
			metrics: true,
		}
	}
}
