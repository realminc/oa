//! Callback trait and iterator context for training and inference sessions.
//!
//! Port provenance: `oa/core/callback.h`, `oa/core/iterator.h`.
//!
//! [`Callback`] is the Keras-style hook interface used by `TrainingLoop` and
//! any future `ItBatch` / `ItInference` session types. All three methods have
//! default no-op implementations so callers only override what they need.
//!
//! [`IteratorContext`] is a thin object-safe view of the current iterator
//! state that callbacks receive. Decoupled from the concrete session type so
//! that callback code does not need to name it.
//!
//! # Usage
//!
//! ```rust,ignore
//! use oa::{Callback, IteratorContext};
//!
//! struct PrintProgress;
//!
//! impl Callback for PrintProgress {
//!     fn on_step(&mut self, ctx: &dyn IteratorContext) {
//!         if ctx.index() % 100 == 0 {
//!             println!("step {}/{}", ctx.index(), ctx.total().unwrap_or(0));
//!         }
//!     }
//! }
//! ```

// ─── IteratorContext ──────────────────────────────────────────────────────────

/// Object-safe view of an iterator's current state, passed to every [`Callback`]
/// hook.
///
/// The concrete iterator (e.g. `TrainingLoop`) implements this privately; the
/// callback only needs the interface.
pub trait IteratorContext {
	/// Step index, zero-based. Matches the C++ `Iterator::index()`.
	fn index(&self) -> i64;

	/// `true` when the iterator has finished all steps.
	fn is_done(&self) -> bool;

	/// Total planned steps, if known. `None` for open-ended iterators.
	fn total(&self) -> Option<i64>;
}

// ─── Callback ─────────────────────────────────────────────────────────────────

/// Keras-style hook interface for OA iterator sessions.
///
/// Attach one or more callbacks to a session via its `add_callback` method.
/// All hooks receive a `&dyn IteratorContext` so callbacks stay decoupled from
/// the concrete session type.
///
/// # Mapping from C++
///
/// | C++                       | Rust                       |
/// |---------------------------|----------------------------|
/// | `onBegin(Iterator&)`      | `on_begin(&mut self, ctx)` |
/// | `onStep(Iterator&)`       | `on_step(&mut self, ctx)`  |
/// | `onEnd(Iterator&)`        | `on_end(&mut self, ctx)`   |
///
/// C++ used virtual inheritance; Rust uses dynamic dispatch through `Box<dyn Callback>`.
/// The C++ move-constructor convention maps to Rust's normal ownership — place
/// `Box<dyn Callback>` values in a `Vec` rather than raw pointers.
pub trait Callback: Send {
	/// Fired once before the first step.
	fn on_begin(&mut self, _ctx: &dyn IteratorContext) {}

	/// Fired after each step. The iterator has already advanced to `ctx.index()`.
	fn on_step(&mut self, _ctx: &dyn IteratorContext) {}

	/// Fired once after the last step or when the session is explicitly finished.
	fn on_end(&mut self, _ctx: &dyn IteratorContext) {}
}

// ─── CallbackSet ─────────────────────────────────────────────────────────────

/// Ordered list of [`Callback`] objects attached to a session.
///
/// Callbacks fire in insertion order. This mirrors `oa::Iterator::addCallback`
/// but is an explicit value rather than a virtual base member so the concrete
/// session type can own it directly.
pub struct CallbackSet {
	callbacks: Vec<Box<dyn Callback>>,
}

impl CallbackSet {
	/// Create an empty callback set.
	pub fn new() -> Self {
		Self {
			callbacks: Vec::new(),
		}
	}

	/// Append a callback. Callbacks fire in insertion order.
	pub fn add(&mut self, cb: Box<dyn Callback>) {
		self.callbacks.push(cb);
	}

	/// Fire `on_begin` on all callbacks in order.
	pub fn fire_begin(&mut self, ctx: &dyn IteratorContext) {
		for cb in &mut self.callbacks {
			cb.on_begin(ctx);
		}
	}

	/// Fire `on_step` on all callbacks in order.
	pub fn fire_step(&mut self, ctx: &dyn IteratorContext) {
		for cb in &mut self.callbacks {
			cb.on_step(ctx);
		}
	}

	/// Fire `on_end` on all callbacks in order.
	pub fn fire_end(&mut self, ctx: &dyn IteratorContext) {
		for cb in &mut self.callbacks {
			cb.on_end(ctx);
		}
	}

	/// Number of registered callbacks.
	pub fn len(&self) -> usize {
		self.callbacks.len()
	}

	/// True when no callbacks are registered.
	pub fn is_empty(&self) -> bool {
		self.callbacks.is_empty()
	}
}

impl Default for CallbackSet {
	fn default() -> Self {
		Self::new()
	}
}
