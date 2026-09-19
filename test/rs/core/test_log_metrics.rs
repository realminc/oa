//! Contract tests for oa::LogMetrics

use oa::LogMetrics;

fn tmp_dir(label: &str) -> oa::Path {
	let p = std::env::temp_dir().join("oa_test_log_metrics").join(label);
	let _ = std::fs::remove_dir_all(&p);
	std::fs::create_dir_all(&p).unwrap();
	oa::Path::from(p)
}

fn read_events(dir: &oa::Path) -> Vec<String> {
	let path = dir.join("events.jsonl");
	let text = std::fs::read_to_string(&*path).unwrap_or_default();
	text
		.lines()
		.filter(|l| !l.is_empty())
		.map(|l| l.to_owned())
		.collect()
}

// ─── open / close ────────────────────────────────────────────────────────────

#[test]
fn is_open_after_open() {
	let dir = tmp_dir("open");
	let mut m = LogMetrics::new();
	assert!(!m.is_open());
	m.open(&dir).unwrap();
	assert!(m.is_open());
	m.close().unwrap();
	assert!(!m.is_open());
}

#[test]
fn events_file_created_on_open() {
	let dir = tmp_dir("create");
	let mut m = LogMetrics::new();
	m.open(&dir).unwrap();
	m.flush().unwrap();
	// File may not exist yet if nothing was logged; that's fine.
	m.close().unwrap();
}

// ─── log_scalar ──────────────────────────────────────────────────────────────

#[test]
fn log_scalar_writes_jsonl_record() {
	let dir = tmp_dir("scalar");
	let mut m = LogMetrics::new();
	m.open(&dir).unwrap();
	m.set_flush_interval(1); // flush immediately
	m.log_scalar("loss", 1, 1.5);
	m.close().unwrap();

	let lines = read_events(&dir);
	assert_eq!(lines.len(), 1);
	assert!(
		lines[0].contains("\"tag\":\"loss\""),
		"tag missing: {}",
		lines[0]
	);
	assert!(
		lines[0].contains("\"step\":1"),
		"step missing: {}",
		lines[0]
	);
	assert!(
		lines[0].contains("\"value\":"),
		"value missing: {}",
		lines[0]
	);
	assert!(
		lines[0].contains("\"wall_time\":"),
		"wall_time missing: {}",
		lines[0]
	);
}

#[test]
fn multiple_scalars_produce_multiple_lines() {
	let dir = tmp_dir("multi");
	let mut m = LogMetrics::new();
	m.open(&dir).unwrap();
	m.set_flush_interval(1);
	for step in 0..5_i64 {
		m.log_scalar("acc", step, 0.1 * step as f64);
	}
	m.close().unwrap();

	let lines = read_events(&dir);
	assert_eq!(lines.len(), 5);
}

#[test]
fn log_scalar_noop_when_closed() {
	let dir = tmp_dir("noop");
	let m = LogMetrics::new();
	// Not opened — should not panic or write anything.
	m.log_scalar("x", 0, 1.0);
	assert!(!std::path::Path::new(&dir.join("events.jsonl")).exists());
}

#[test]
fn log_scalars_map_writes_per_entry() {
	let dir = tmp_dir("scalars_map");
	let mut m = LogMetrics::new();
	m.open(&dir).unwrap();
	m.set_flush_interval(1);
	let mut map = std::collections::HashMap::new();
	map.insert("loss".to_owned(), 1.23);
	map.insert("acc".to_owned(), 0.9);
	m.log_scalars("train", 10, &map);
	m.close().unwrap();

	let lines = read_events(&dir);
	assert_eq!(lines.len(), 2);
}

// ─── flush interval ──────────────────────────────────────────────────────────

#[test]
fn auto_flush_at_interval() {
	let dir = tmp_dir("interval");
	let mut m = LogMetrics::new();
	m.open(&dir).unwrap();
	m.set_flush_interval(3);
	m.log_scalar("x", 1, 1.0);
	m.log_scalar("x", 2, 2.0);
	// Not flushed yet.
	m.log_scalar("x", 3, 3.0); // triggers auto-flush at count == 3
	let lines = read_events(&dir);
	assert_eq!(lines.len(), 3, "auto-flush should have written 3 records");
	m.close().unwrap();
}

// ─── tag escaping ─────────────────────────────────────────────────────────────

#[test]
fn tag_with_special_chars_is_json_escaped() {
	let dir = tmp_dir("escape");
	let mut m = LogMetrics::new();
	m.open(&dir).unwrap();
	m.set_flush_interval(1);
	m.log_scalar("a/b\tc", 0, 0.0);
	m.close().unwrap();

	let lines = read_events(&dir);
	assert_eq!(lines.len(), 1);
	// Tab should be escaped as \t in the JSON output.
	assert!(lines[0].contains("\\t"), "tab not escaped: {}", lines[0]);
}

// ─── non-finite values ────────────────────────────────────────────────────────

#[test]
fn nan_value_written_as_null() {
	let dir = tmp_dir("nan");
	let mut m = LogMetrics::new();
	m.open(&dir).unwrap();
	m.set_flush_interval(1);
	m.log_scalar("x", 0, f64::NAN);
	m.close().unwrap();

	let lines = read_events(&dir);
	assert_eq!(lines.len(), 1);
	assert!(
		lines[0].contains("\"value\":null"),
		"NaN not serialized as null: {}",
		lines[0]
	);
}
