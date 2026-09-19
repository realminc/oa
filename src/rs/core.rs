//! Foundational OA values, metadata, and failure contracts.

pub(crate) mod autograd;
pub mod callback;
pub mod cli;
pub mod config;
pub mod constant;
mod dtype;
pub mod env_flag;
mod error;
pub mod filesystem;
mod image;
mod json;
pub mod log_metrics;
pub mod mapped_file;
mod matrix;
pub mod memory;
mod op;
pub mod operation;
pub mod path;
pub mod perf_stat;
pub mod time;
pub mod validation;
pub mod vlm;

pub use callback::{Callback, CallbackSet, IteratorContext};
pub use cli::Cli;
pub use config::{CheckpointConfig, LogConfig};
pub use constant::{COMPACT_BANNER, REALM_BANNER, VIEWPORT_TITLE, brand_viewport};
pub use dtype::{DType, Element};
pub use env_flag::{EnvFlag, NumericMode, apply_numeric_mode};
pub use error::{Error, ErrorKind, Result};
pub use filesystem::Filesystem;
pub use image::{Image, ImageFormat, ImageLayout};
pub(crate) use json::{push_format, push_json_string};
pub use log_metrics::LogMetrics;
pub use mapped_file::MappedFile;
pub use matrix::Matrix;
pub(crate) use matrix::MatrixSemantic;
pub use op::{
	BufferAccess, OpAttribute, OpAttributeKind, OpAttributeSpec, OpControlFlow, OpDTypeRule,
	OpDifferentiation, OpEffect, OpLowering, OpShapeRule, OpValueKind, OperationContract,
};
pub use path::Path;
pub use perf_stat::PerfStat;
pub use time::{Datetime, ScopedTimer, Stopwatch, Timestamp};
pub use validation::{Validation, ValidationSeverity};
