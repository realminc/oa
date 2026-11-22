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

pub use super::types::{CheckpointConfig, LogConfig};
use crate::Path;

// ─── CheckpointConfig ────────────────────────────────────────────────────────

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
