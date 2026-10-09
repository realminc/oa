//! Private bridge from Matrix operations into an active ML tape.

use std::{cell::RefCell, marker::PhantomData, rc::Rc};

use crate::Result;

#[path = "autograd/matrix.gen.rs"]
mod matrix;

#[path = "autograd/node.gen.rs"]
mod node;

include!("autograd/exports.gen.rs");
#[path = "autograd/context.rs"]
mod context;

pub(crate) use context::GradientContext;
pub(crate) use node::GradNodeMatrix;

type Recorder = Rc<dyn Fn(GradNodeMatrix) -> Result<()>>;

thread_local! {
	static RECORDERS: RefCell<Vec<Recorder>> = const { RefCell::new(Vec::new()) };
}

/// Selection of one thread-local observer, removed exactly when dropped.
pub(crate) struct RecordingSelection {
	recorder: Recorder,
	_not_send_sync: PhantomData<Rc<()>>,
}

pub(crate) fn select(
	record: impl Fn(GradNodeMatrix) -> Result<()> + 'static,
) -> RecordingSelection {
	let recorder: Recorder = Rc::new(record);
	RECORDERS.with(|recorders| recorders.borrow_mut().push(recorder.clone()));
	RecordingSelection {
		recorder,
		_not_send_sync: PhantomData,
	}
}

pub(crate) fn record(node: GradNodeMatrix) -> Result<()> {
	let recorder = RECORDERS.with(|recorders| recorders.borrow().last().cloned());
	if let Some(recorder) = recorder {
		return recorder(node);
	}
	Ok(())
}

impl Drop for RecordingSelection {
	fn drop(&mut self) {
		RECORDERS.with(|recorders| {
			recorders
				.borrow_mut()
				.retain(|candidate| !Rc::ptr_eq(candidate, &self.recorder));
		});
	}
}
