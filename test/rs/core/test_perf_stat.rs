//! Contract tests for oa::PerfStat

use oa::PerfStat;

// ─── construction ────────────────────────────────────────────────────────────

#[test]
fn not_ready_before_warmup() {
	let mut s = PerfStat::new("x", 10, 5);
	for _ in 0..5 {
		s.push(1.0);
	}
	assert!(!s.is_ready(), "should not be ready during warmup");
}

#[test]
fn ready_after_warmup_plus_one() {
	let mut s = PerfStat::new("x", 10, 5);
	for _ in 0..6 {
		s.push(1.0);
	}
	assert!(s.is_ready());
}

#[test]
fn count_includes_warmup_samples() {
	let mut s = PerfStat::new("x", 10, 5);
	for _ in 0..8 {
		s.push(1.0);
	}
	assert_eq!(s.count(), 8);
}

// ─── statistics ──────────────────────────────────────────────────────────────

fn filled_stat(values: &[f64]) -> PerfStat {
	let mut s = PerfStat::new("t", values.len() * 2, 0);
	for &v in values {
		s.push(v);
	}
	s
}

#[test]
fn mean_correct() {
	let s = filled_stat(&[1.0, 2.0, 3.0, 4.0, 5.0]);
	assert!((s.mean() - 3.0).abs() < 1e-10);
}

#[test]
fn min_max() {
	let s = filled_stat(&[3.0, 1.0, 5.0, 2.0, 4.0]);
	assert!((s.min() - 1.0).abs() < 1e-10);
	assert!((s.max() - 5.0).abs() < 1e-10);
}

#[test]
fn stddev_zero_for_uniform() {
	let s = filled_stat(&[7.0, 7.0, 7.0, 7.0]);
	assert!(s.stddev() < 1e-10);
}

#[test]
fn stddev_nonzero_for_varied() {
	let s = filled_stat(&[1.0, 2.0, 3.0]);
	assert!(s.stddev() > 0.0);
}

#[test]
fn last_returns_most_recent() {
	let s = filled_stat(&[1.0, 2.0, 99.0]);
	assert!((s.last() - 99.0).abs() < 1e-10);
}

// ─── percentiles ─────────────────────────────────────────────────────────────

#[test]
fn p50_is_median() {
	let s = filled_stat(&[1.0, 2.0, 3.0, 4.0, 5.0]);
	assert!((s.p50() - 3.0).abs() < 1e-10);
}

#[test]
fn p95_at_or_above_p50() {
	let s = filled_stat(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
	assert!(s.p95() >= s.p50());
}

#[test]
fn p99_at_or_above_p95() {
	let s = filled_stat(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
	assert!(s.p99() >= s.p95());
}

// ─── rolling window ───────────────────────────────────────────────────────────

#[test]
fn window_evicts_oldest() {
	let mut s = PerfStat::new("w", 3, 0);
	s.push(100.0);
	s.push(200.0);
	s.push(300.0);
	// Window full: [100, 200, 300]
	assert!((s.mean() - 200.0).abs() < 1e-10);
	s.push(10.0);
	// Window: [200, 300, 10]
	let expected = (200.0 + 300.0 + 10.0) / 3.0;
	assert!((s.mean() - expected).abs() < 1e-10);
}

#[test]
fn reset_clears_all() {
	let mut s = PerfStat::new("r", 10, 0);
	for i in 0..8 {
		s.push(i as f64);
	}
	s.reset();
	assert!(!s.is_ready());
	assert_eq!(s.count(), 0);
	assert_eq!(s.mean(), 0.0);
}

// ─── name ────────────────────────────────────────────────────────────────────

#[test]
fn name_preserved() {
	let s = PerfStat::new("step_ms", 10, 0);
	assert_eq!(s.name(), "step_ms");
}

// ─── zero warmup ─────────────────────────────────────────────────────────────

#[test]
fn zero_warmup_ready_after_first_push() {
	let mut s = PerfStat::new("z", 10, 0);
	s.push(5.0);
	assert!(s.is_ready());
	assert!((s.mean() - 5.0).abs() < 1e-10);
}
