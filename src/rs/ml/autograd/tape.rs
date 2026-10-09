use std::{
	cell::{Cell, RefCell},
	rc::{Rc, Weak},
};

use crate::ml::Parameter;
use crate::{Error, Result, matrix::autograd as matrix_autograd};

use super::node::GradNode;

#[path = "backward.rs"]
pub(in crate::ml::autograd) mod backward;

thread_local! {
	static ACTIVE_TAPES: RefCell<Vec<Weak<RecordingState>>> = const { RefCell::new(Vec::new()) };
	static NEXT_SEQUENCE: Cell<u64> = const { Cell::new(0) };
}

struct TapeEntry {
	sequence: u64,
	node: GradNode,
}

// One shared recording allocation, weakly observed by thread-local callbacks.
// This is recording lifetime state, not a second public tape or execution owner.
struct RecordingState {
	nodes: RefCell<Vec<TapeEntry>>,
	watched_parameters: RefCell<Vec<WatchedParameter>>,
	consumed: Cell<bool>,
}

struct WatchedParameter {
	parameter: Parameter,
	data_id: u64,
	version: u64,
}

/// Thread-affine reverse-mode recording scope.
///
/// Constructing a tape selects it for operations on the current thread.
/// [`GradientTape::backward`] closes the recording scope and records the
/// corresponding backward operations into the originating engine's eager batch.
#[must_use]
pub struct GradientTape {
	recording: Rc<RecordingState>,
	active: Cell<bool>,
	matrix_recording: RefCell<Option<matrix_autograd::RecordingSelection>>,
}

impl GradientTape {
	/// Begin a reverse-mode recording scope on the current thread.
	pub fn new() -> Self {
		let recording = Rc::new(RecordingState {
			nodes: RefCell::new(Vec::new()),
			watched_parameters: RefCell::new(Vec::new()),
			consumed: Cell::new(false),
		});
		ACTIVE_TAPES.with(|tapes| tapes.borrow_mut().push(Rc::downgrade(&recording)));
		let weak = Rc::downgrade(&recording);
		let matrix_recording = matrix_autograd::select(move |node| {
			let Some(tape) = weak.upgrade() else {
				return Ok(());
			};
			let node = GradNode::from(node);
			record_node_for(&tape, node)
		});
		Self {
			recording,
			active: Cell::new(true),
			matrix_recording: RefCell::new(Some(matrix_recording)),
		}
	}

	/// Close recording without constructing a backward pass.
	pub fn close(&self) {
		if !self.active.replace(false) {
			return;
		}
		self.matrix_recording.borrow_mut().take();
		ACTIVE_TAPES.with(|tapes| {
			tapes.borrow_mut().retain(|candidate| {
				candidate.strong_count() != 0 && !candidate.ptr_eq(&Rc::downgrade(&self.recording))
			});
		});
	}
}

impl Default for GradientTape {
	fn default() -> Self {
		Self::new()
	}
}

impl Drop for GradientTape {
	fn drop(&mut self) {
		self.close();
	}
}

pub(super) fn record_node(node: GradNode) -> Result<()> {
	ACTIVE_TAPES.with(|tapes| {
		let active = tapes.borrow().last().and_then(Weak::upgrade);
		if let Some(active) = active {
			record_node_for(&active, node)?;
		}
		Ok(())
	})
}

pub(in crate::ml) fn record_parameter_leaf(parameter: &Parameter) -> Result<()> {
	let (data, version, requires_grad) = parameter.snapshot();
	if !requires_grad {
		return Ok(());
	}
	ACTIVE_TAPES.with(|tapes| {
		let active = tapes.borrow().last().and_then(Weak::upgrade);
		if let Some(active) = active {
			let mut watched = active.watched_parameters.borrow_mut();
			if !watched
				.iter()
				.any(|candidate| candidate.parameter.same_as(parameter))
			{
				watched.push(WatchedParameter {
					parameter: parameter.clone(),
					data_id: data.value_id(),
					version,
				});
			}
		}
		Ok(())
	})
}

pub(in crate::ml) fn recording_active() -> bool {
	ACTIVE_TAPES.with(|tapes| {
		tapes
			.borrow()
			.last()
			.is_some_and(|candidate| candidate.strong_count() != 0)
	})
}

fn record_node_for(tape: &RecordingState, node: GradNode) -> Result<()> {
	let sequence = NEXT_SEQUENCE.with(|next| {
		let sequence = next
			.get()
			.checked_add(1)
			.ok_or_else(|| Error::resource_exhausted("autograd sequence exhausted"))?;
		next.set(sequence);
		Ok(sequence)
	})?;
	tape.nodes.borrow_mut().push(TapeEntry { sequence, node });
	Ok(())
}

#[cfg(test)]
#[path = "../../../../test/rs/ml/tape_unit.rs"]
mod tests;
