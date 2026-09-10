//! Private bridge from foundational value operations into an active ML tape.

use std::{cell::RefCell, marker::PhantomData, rc::Rc};

use super::Matrix;
use crate::Result;

/// Foundational Matrix operation whose adjoint is independent of ML policy.
pub(crate) enum MatrixNode {
	Reshape {
		input: Matrix,
		output_id: u64,
	},
	Add {
		left: Matrix,
		right: Matrix,
		output_id: u64,
	},
	Dropout {
		input: Matrix,
		output_id: u64,
		probability: f32,
		seed: u64,
	},
	Softmax {
		input: Matrix,
		output: Matrix,
		output_id: u64,
		dim: i32,
	},
	LogSoftmax {
		input: Matrix,
		output: Matrix,
		output_id: u64,
		dim: i32,
	},
	Sum {
		input: Matrix,
		output_id: u64,
		dim: i32,
	},
}

type Recorder = Rc<dyn Fn(MatrixNode) -> Result<()>>;

thread_local! {
	static RECORDERS: RefCell<Vec<Recorder>> = const { RefCell::new(Vec::new()) };
}

/// Selection of one thread-local observer, removed exactly when dropped.
pub(crate) struct RecordingSelection {
	recorder: Recorder,
	_not_send_sync: PhantomData<Rc<()>>,
}

pub(crate) fn select(record: impl Fn(MatrixNode) -> Result<()> + 'static) -> RecordingSelection {
	let recorder: Recorder = Rc::new(record);
	RECORDERS.with(|recorders| recorders.borrow_mut().push(recorder.clone()));
	RecordingSelection {
		recorder,
		_not_send_sync: PhantomData,
	}
}

pub(crate) fn record(node: MatrixNode) -> Result<()> {
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
