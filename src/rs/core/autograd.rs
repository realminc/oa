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
	Copy {
		input: Matrix,
		output_id: u64,
	},
	Add {
		left: Matrix,
		right: Matrix,
		output_id: u64,
	},
	Mul {
		left: Matrix,
		right: Matrix,
		output_id: u64,
	},
	Div {
		left: Matrix,
		right: Matrix,
		output_id: u64,
	},
	Scale {
		input: Matrix,
		output_id: u64,
		scalar: f32,
	},
	Reciprocal {
		input: Matrix,
		output: Matrix,
		output_id: u64,
	},
	Exp {
		input: Matrix,
		output: Matrix,
		output_id: u64,
	},
	Log {
		input: Matrix,
		output_id: u64,
	},
	Abs {
		input: Matrix,
		output_id: u64,
	},
	Sqrt {
		input: Matrix,
		output: Matrix,
		output_id: u64,
	},
	ClampMax {
		input: Matrix,
		output_id: u64,
		maximum: f32,
	},
	ClampMin {
		input: Matrix,
		output_id: u64,
		minimum: f32,
	},
	Sub {
		left: Matrix,
		right: Matrix,
		output_id: u64,
	},
	Slice {
		input: Matrix,
		output_id: u64,
		dim: usize,
		start: usize,
		end: usize,
	},
	RepeatInterleave {
		input: Matrix,
		output_id: u64,
		repeats: usize,
		dim: usize,
	},
	Concat {
		inputs: Vec<Matrix>,
		output_id: u64,
		dim: usize,
		sizes: Vec<usize>,
	},
	GatherLastDim {
		input: Matrix,
		indices: Matrix,
		output_id: u64,
		input_width: usize,
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
	MatMulNt {
		left: Matrix,
		right: Matrix,
		output_id: u64,
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
