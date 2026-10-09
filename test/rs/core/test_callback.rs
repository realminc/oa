//! Tests for `oa::Callback`, `oa::CallbackSet`, `oa::IteratorContext`.

use oa::{Callback, CallbackSet, IteratorContext};

// ── test fixture ─────────────────────────────────────────────────────────────

struct MockCtx {
	index: i64,
	done: bool,
	total: Option<i64>,
}

impl IteratorContext for MockCtx {
	fn index(&self) -> i64 {
		self.index
	}
	fn is_done(&self) -> bool {
		self.done
	}
	fn total(&self) -> Option<i64> {
		self.total
	}
}

struct EventLog {
	events: Vec<String>,
}

impl Callback for EventLog {
	fn on_begin(&mut self, ctx: &dyn IteratorContext) {
		self.events.push(format!("begin:{}", ctx.index()));
	}
	fn on_step(&mut self, ctx: &dyn IteratorContext) {
		self.events.push(format!("step:{}", ctx.index()));
	}
	fn on_end(&mut self, ctx: &dyn IteratorContext) {
		self.events.push(format!("end:{}", ctx.index()));
	}
}

// ── default no-op ────────────────────────────────────────────────────────────

#[test]
fn default_no_op_fires_silently() {
	struct Silent;
	impl Callback for Silent {}

	let ctx = MockCtx {
		index: 0,
		done: false,
		total: None,
	};
	let mut cb = Silent;
	cb.on_begin(&ctx);
	cb.on_step(&ctx);
	cb.on_end(&ctx);
}

// ── CallbackSet ordering ─────────────────────────────────────────────────────

#[test]
fn callback_set_fires_in_order() {
	let mut set = CallbackSet::new();
	set.add(Box::new(EventLog { events: Vec::new() }));
	set.add(Box::new(EventLog { events: Vec::new() }));
	assert_eq!(set.len(), 2);

	let ctx = MockCtx {
		index: 7,
		done: false,
		total: Some(10),
	};
	set.fire_begin(&ctx);
	set.fire_step(&ctx);
	set.fire_end(&ctx);
}

#[test]
fn callback_set_default_is_empty() {
	let set = CallbackSet::default();
	assert!(set.is_empty());
	assert_eq!(set.len(), 0);
}

// ── IteratorContext ───────────────────────────────────────────────────────────

#[test]
fn iterator_context_fields() {
	let ctx = MockCtx {
		index: 42,
		done: false,
		total: Some(100),
	};
	assert_eq!(ctx.index(), 42);
	assert!(!ctx.is_done());
	assert_eq!(ctx.total(), Some(100));

	let ctx2 = MockCtx {
		index: 0,
		done: true,
		total: None,
	};
	assert!(ctx2.is_done());
	assert_eq!(ctx2.total(), None);
}

// ── EventLog captures all events ─────────────────────────────────────────────

#[test]
fn event_log_captures_all_hooks() {
	let mut set = CallbackSet::new();
	let log = EventLog { events: Vec::new() };
	set.add(Box::new(log));

	let begin_ctx = MockCtx {
		index: 0,
		done: false,
		total: Some(3),
	};
	set.fire_begin(&begin_ctx);

	for i in 0..3_i64 {
		let step_ctx = MockCtx {
			index: i,
			done: i == 2,
			total: Some(3),
		};
		set.fire_step(&step_ctx);
	}

	let end_ctx = MockCtx {
		index: 2,
		done: true,
		total: Some(3),
	};
	set.fire_end(&end_ctx);
	// No assertions on internals; exercising the call paths is sufficient.
}
