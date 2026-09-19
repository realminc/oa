//! Contract tests for oa::Stopwatch, oa::ScopedTimer, oa::Timestamp

use std::thread;
use std::time::Duration;

use oa::{ScopedTimer, Stopwatch, Timestamp};

// ─── Timestamp ───────────────────────────────────────────────────────────────

#[test]
fn timestamp_zero_is_not_valid() {
	assert!(!Timestamp::zero().is_valid());
}

#[test]
fn timestamp_now_is_valid() {
	assert!(Timestamp::now().is_valid());
}

#[test]
fn timestamp_now_monotone() {
	let a = Timestamp::now();
	thread::sleep(Duration::from_millis(1));
	let b = Timestamp::now();
	assert!(b > a);
}

#[test]
fn timestamp_sub_non_negative() {
	let a = Timestamp::now();
	thread::sleep(Duration::from_millis(2));
	let b = Timestamp::now();
	let diff = b - a;
	assert!(diff.as_nanos() > 0);
	assert!(diff.to_secs() >= 0.001);
}

#[test]
fn timestamp_from_secs_round_trip() {
	let t = Timestamp::from_secs_f64(1.5);
	let back = t.to_secs();
	assert!((back - 1.5).abs() < 1e-6);
}

#[test]
fn timestamp_to_ms() {
	let t = Timestamp::from_secs_f64(2.0);
	assert!((t.to_ms() - 2000.0).abs() < 1.0);
}

#[test]
fn timestamp_add_sub() {
	let a = Timestamp::from_nanos(1_000_000);
	let b = Timestamp::from_nanos(2_000_000);
	let sum = a + b;
	let diff = b - a;
	assert_eq!(sum.as_nanos(), 3_000_000);
	assert_eq!(diff.as_nanos(), 1_000_000);
}

#[test]
fn timestamp_add_assign() {
	let mut t = Timestamp::from_nanos(100);
	t += Timestamp::from_nanos(50);
	assert_eq!(t.as_nanos(), 150);
}

#[test]
fn timestamp_sub_assign() {
	let mut t = Timestamp::from_nanos(200);
	t -= Timestamp::from_nanos(50);
	assert_eq!(t.as_nanos(), 150);
}

#[test]
fn timestamp_order() {
	let a = Timestamp::from_nanos(1);
	let b = Timestamp::from_nanos(2);
	assert!(a < b);
	assert!(b > a);
	assert!(a <= a);
	assert!(a == a);
}

// ─── Stopwatch ───────────────────────────────────────────────────────────────

#[test]
fn stopwatch_starts_at_zero() {
	let sw = Stopwatch::new();
	assert!(!sw.is_running());
	assert_eq!(sw.elapsed().as_nanos(), 0);
}

#[test]
fn stopwatch_accumulates_time() {
	let mut sw = Stopwatch::new();
	sw.start();
	thread::sleep(Duration::from_millis(10));
	sw.stop();
	assert!(sw.elapsed_sec() >= 0.008);
	assert!(!sw.is_running());
}

#[test]
fn stopwatch_running_flag() {
	let mut sw = Stopwatch::new();
	assert!(!sw.is_running());
	sw.start();
	assert!(sw.is_running());
	sw.stop();
	assert!(!sw.is_running());
}

#[test]
fn stopwatch_elapsed_while_running() {
	let mut sw = Stopwatch::new();
	sw.start();
	thread::sleep(Duration::from_millis(5));
	assert!(
		sw.elapsed_ms() >= 3.0,
		"elapsed_ms should be at least 3ms while running"
	);
	sw.stop();
}

#[test]
fn stopwatch_reset_zeroes_elapsed() {
	let mut sw = Stopwatch::new();
	sw.start();
	thread::sleep(Duration::from_millis(5));
	sw.stop();
	sw.reset();
	assert_eq!(sw.elapsed().as_nanos(), 0);
	assert!(!sw.is_running());
}

#[test]
fn stopwatch_restart_runs_again() {
	let mut sw = Stopwatch::new();
	sw.start();
	thread::sleep(Duration::from_millis(5));
	sw.restart();
	assert!(sw.is_running());
	assert!(sw.elapsed().as_nanos() < 5_000_000); // less than 5ms after restart
}

#[test]
fn stopwatch_accumulates_across_start_stop() {
	let mut sw = Stopwatch::new();
	sw.start();
	thread::sleep(Duration::from_millis(5));
	sw.stop();
	let after_first = sw.elapsed_ms();
	sw.start();
	thread::sleep(Duration::from_millis(5));
	sw.stop();
	assert!(sw.elapsed_ms() > after_first, "elapsed should accumulate");
}

// ─── ScopedTimer ─────────────────────────────────────────────────────────────

#[test]
fn scoped_timer_writes_on_drop() {
	let mut elapsed = 0.0_f64;
	{
		let _t = ScopedTimer::new(&mut elapsed);
		thread::sleep(Duration::from_millis(10));
	}
	assert!(elapsed >= 0.008, "elapsed should be >= 8ms, got {elapsed}");
}

#[test]
fn scoped_timer_writes_nonzero() {
	// We can't check elapsed == 0 while the timer borrows it, but we can
	// verify that a non-trivial delay produces a positive value.
	let mut elapsed = 0.0_f64;
	{
		let _t = ScopedTimer::new(&mut elapsed);
		thread::sleep(Duration::from_millis(5));
	}
	assert!(elapsed > 0.0);
}
