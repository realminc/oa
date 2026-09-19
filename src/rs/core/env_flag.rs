//! Environment-variable knobs for OA runtime configuration.
//!
//! Port provenance: `oa/core/envFlag.h`.
//!
//! [`EnvFlag`] is the single-source convention for reading OA env knobs.
//! [`NumericMode`] and [`apply_numeric_mode`] translate a process-wide numeric
//! execution policy into the equivalent env-knob state, called once from
//! `Engine::create`.
//!
//! # Reading pattern
//!
//! ```rust,ignore
//! use oa::EnvFlag;
//!
//! // bool toggle — true when set to anything other than 0/false/no/off
//! if EnvFlag::is_set("OA_DISABLE_COOPMAT") { ... }
//!
//! // string override — returns default when var is unset or empty
//! let prec = EnvFlag::get_string("OA_FORCE_PRECISION", "FP32");
//!
//! // integer override — returns default when unset or unparseable
//! let workers = EnvFlag::get_int("OA_SHADER_LOAD_THREADS", 0);
//! ```
//!
//! # Canonical OA env knobs
//!
//! ## Disable toggles
//! - `OA_DISABLE_COOPMAT` — skip CoopMat extension; route to scalar
//! - `OA_DISABLE_COOPMAT2` — skip VK_NV_cooperative_matrix2
//! - `OA_DISABLE_BF16` — force FP32; skip VK_KHR_shader_bfloat16
//! - `OA_DISABLE_PERSISTENT_LOOP` — force single-step submission
//! - `OA_DISABLE_INTEGER_DOT_PRODUCT` — skip VK_KHR_shader_integer_dot_product
//! - `OA_DISABLE_GRU_SCAN` — use decomposed GRU cells instead of fused scan
//! - `OA_DISABLE_GEMM_ROUTE_CACHE` — skip measured GEMM route-cache replay
//! - `OA_DISABLE_NARROW_ROW_KERNELS` — skip narrow LayerNorm/softmax schedules
//! - `OA_DISABLE_TILED_BMM` — skip the shared-memory BMM schedule
//! - `OA_DISABLE_LINEAR_PARAM_ROWS32` — skip narrow Linear parameter-adjoint schedule
//!
//! ## Force overrides
//! - `OA_FORCE_PRECISION=FP32|BF16|FP16`
//! - `OA_FORCE_COOPMAT=1` — bypass vendor-trust blocklist
//! - `OA_FORCE_COOPVEC=1` — bypass CoopVec NVIDIA-only routing gate
//! - `OA_FORCE_DEVICE_INDEX=N` — override device pick
//! - `OA_SHADER_LOAD_THREADS=0|1|N` — shader preload workers
//!
//! ## Diagnostic / opt-in logging
//! - `OA_VK_VALIDATION=1`
//! - `OA_LOG_GEMM_ROUTER=1`
//! - `OA_LOG_PIPELINE_LOAD=1`
//! - `OA_LOG_BARRIERS=1`
//! - `OA_LOG_CONTEXT_GRAPH=N`
//! - `OA_GRAPH_REPORT=path|1`
//! - `OA_LOG_COOPMAT_SHAPES=1`
//! - `OA_LOG_NUMERIC_DEVIATIONS=1`

use std::env;

// ─── EnvFlag ─────────────────────────────────────────────────────────────────

/// Single-source environment-variable reader for OA runtime knobs.
pub struct EnvFlag;

impl EnvFlag {
	/// Return `true` when the environment variable is set to a value that is
	/// **not** one of `""`, `"0"`, `"false"`, `"no"`, `"off"` (case-insensitive).
	///
	/// Mirrors `oa::EnvFlag::isSet`.
	pub fn is_set(name: &str) -> bool {
		match env::var(name) {
			Err(_) => false,
			Ok(val) => !Self::is_falsy(&val),
		}
	}

	/// Return the env value when set and non-empty, otherwise `default`.
	///
	/// Mirrors `oa::EnvFlag::getString`.
	pub fn get_string(name: &str, default: &str) -> String {
		match env::var(name) {
			Ok(val) if !val.is_empty() => val,
			_ => default.to_owned(),
		}
	}

	/// Return the env value parsed as `i64` when set, non-empty, and parseable,
	/// otherwise `default`. Only decimal notation is accepted.
	///
	/// Mirrors `oa::EnvFlag::getInt`.
	pub fn get_int(name: &str, default: i64) -> i64 {
		match env::var(name) {
			Ok(val) if !val.is_empty() => val.parse::<i64>().unwrap_or(default),
			_ => default,
		}
	}

	/// Set the environment variable to `value` if it is not already set.
	///
	/// Returns `false` when the variable was already set externally (the
	/// caller's signal that user-supplied env wins over programmatic defaults).
	///
	/// Mirrors `oa::EnvFlag::setIfUnset`. Internally used by
	/// [`apply_numeric_mode`].
	///
	/// # Safety
	///
	/// Calls `std::env::set_var` which is not signal-safe. Callers must ensure
	/// this is not called concurrently from multiple threads. Typically called
	/// once from `Engine::create` during single-threaded startup.
	pub fn set_if_unset(name: &str, value: &str) -> bool {
		if env::var(name).is_ok() {
			return false;
		}
		// SAFETY: single-threaded startup contract — see doc comment.
		unsafe { env::set_var(name, value) };
		true
	}

	// ── internal ──────────────────────────────────────────────────────────

	fn is_falsy(val: &str) -> bool {
		matches!(
			val.to_ascii_lowercase().as_str(),
			"" | "0" | "false" | "no" | "off"
		)
	}
}

// ─── NumericMode ──────────────────────────────────────────────────────────────

/// Process-wide numerical execution policy.
///
/// Core owns the vocabulary and its environment mapping; the runtime engine
/// consumes it through `EngineConfig`.
///
/// | Mode            | Effect |
/// |-----------------|--------|
/// | `Fast`          | vendor math, non-deterministic reductions |
/// | `Stable`        | FP32 accumulators, CoopMat disabled |
/// | `Deterministic` | fixed reduction order, no race-dependent atomics |
///
/// Port provenance: `oa::NumericMode` in `oa/core/envFlag.h`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum NumericMode {
	/// Vendor math, maximum performance. Non-deterministic reductions allowed.
	Fast = 0,
	/// FP32 accumulators; CoopMat and BF16 disabled. Recommended for training.
	#[default]
	Stable = 1,
	/// Fixed reduction order; persistent-loop submission disabled.
	/// Required for byte-identical reproducibility.
	Deterministic = 2,
}

// ─── apply_numeric_mode ───────────────────────────────────────────────────────

/// Translate a [`NumericMode`] into the equivalent env-knob state.
///
/// Called **once** at engine init. User-set env vars always win
/// ([`EnvFlag::set_if_unset`] checks before writing).
///
/// | Mode            | Env vars touched |
/// |-----------------|-----------------|
/// | `Fast`          | none |
/// | `Stable`        | `OA_FORCE_PRECISION=FP32`, `OA_DISABLE_COOPMAT=1` |
/// | `Deterministic` | above + `OA_DISABLE_PERSISTENT_LOOP=1` |
///
/// Port provenance: `oa::applyNumericMode` in `oa/core/envFlag.h`.
pub fn apply_numeric_mode(mode: NumericMode) {
	match mode {
		NumericMode::Fast => {}
		NumericMode::Stable => {
			EnvFlag::set_if_unset("OA_FORCE_PRECISION", "FP32");
			EnvFlag::set_if_unset("OA_DISABLE_COOPMAT", "1");
		}
		NumericMode::Deterministic => {
			EnvFlag::set_if_unset("OA_FORCE_PRECISION", "FP32");
			EnvFlag::set_if_unset("OA_DISABLE_COOPMAT", "1");
			EnvFlag::set_if_unset("OA_DISABLE_PERSISTENT_LOOP", "1");
		}
	}
}
