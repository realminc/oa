//! Contract tests for oa::CheckpointConfig and oa::LogConfig

use oa::{CheckpointConfig, LogConfig};

// ─── CheckpointConfig defaults ───────────────────────────────────────────────

#[test]
fn checkpoint_default_name() {
	let c = CheckpointConfig::default();
	assert_eq!(c.name, "model");
}

#[test]
fn checkpoint_default_env() {
	let c = CheckpointConfig::default();
	assert_eq!(c.env, "dev");
}

#[test]
fn checkpoint_default_save_best_true() {
	let c = CheckpointConfig::default();
	assert!(c.save_best);
}

#[test]
fn checkpoint_default_save_last_true() {
	let c = CheckpointConfig::default();
	assert!(c.save_last);
}

#[test]
fn checkpoint_struct_update() {
	let c = CheckpointConfig {
		name: "MyModel".into(),
		save_best: false,
		..CheckpointConfig::default()
	};
	assert_eq!(c.name, "MyModel");
	assert!(!c.save_best);
	assert!(c.save_last); // default preserved
}

// ─── LogConfig defaults ───────────────────────────────────────────────────────

#[test]
fn log_default_level_info() {
	let l = LogConfig::default();
	assert_eq!(l.level, "info");
}

#[test]
fn log_default_console_true() {
	let l = LogConfig::default();
	assert!(l.console);
}

#[test]
fn log_default_file_false() {
	let l = LogConfig::default();
	assert!(!l.file);
}

#[test]
fn log_default_metrics_true() {
	let l = LogConfig::default();
	assert!(l.metrics);
}

#[test]
fn log_struct_update() {
	let l = LogConfig {
		level: "debug".into(),
		file: true,
		..LogConfig::default()
	};
	assert_eq!(l.level, "debug");
	assert!(l.file);
	assert!(l.console); // default preserved
}

// ─── equality ────────────────────────────────────────────────────────────────

#[test]
fn two_defaults_are_equal() {
	assert_eq!(CheckpointConfig::default(), CheckpointConfig::default());
	assert_eq!(LogConfig::default(), LogConfig::default());
}

#[test]
fn modified_differs_from_default() {
	let c = CheckpointConfig {
		name: "other".into(),
		..Default::default()
	};
	assert_ne!(c, CheckpointConfig::default());
}
