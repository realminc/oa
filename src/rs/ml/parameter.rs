use std::{cell::RefCell, rc::Rc};

use crate::{DType, Error, Matrix, Result, matrix};

/// Named trainable FP32 matrix with one live accumulated gradient.
#[derive(Clone)]
pub struct Parameter {
	inner: Rc<RefCell<ParameterState>>,
}

struct ParameterState {
	name: String,
	data: Matrix,
	gradient: Option<Matrix>,
	requires_grad: bool,
	version: u64,
}

impl Parameter {
	/// Create a trainable parameter from an existing FP32 matrix.
	///
	/// # Errors
	///
	/// Returns an error when `data` is not FP32.
	pub fn new(name: impl Into<String>, data: Matrix) -> Result<Self> {
		if data.dtype() != DType::F32 {
			return Err(Error::invalid_argument(format!(
				"parameter data must be F32; found {}",
				data.dtype().token()
			)));
		}
		Ok(Self {
			inner: Rc::new(RefCell::new(ParameterState {
				name: name.into(),
				data,
				gradient: None,
				requires_grad: true,
				version: 0,
			})),
		})
	}

	/// Return this parameter's local name.
	pub fn name(&self) -> String {
		self.inner.borrow().name.clone()
	}

	/// Return a cheap handle to the current parameter storage.
	///
	/// AdamW updates this stable storage in place. Existing handles therefore
	/// observe later optimizer steps after the corresponding execution completes.
	pub fn data(&self) -> Matrix {
		self.inner.borrow().data.clone()
	}

	/// Return a cheap handle to the accumulated gradient, when present.
	pub fn gradient(&self) -> Option<Matrix> {
		self.inner.borrow().gradient.clone()
	}

	/// Enable or disable gradient accumulation for this parameter.
	pub fn set_requires_grad(&self, requires_grad: bool) {
		let mut state = self.inner.borrow_mut();
		state.requires_grad = requires_grad;
		if !requires_grad {
			state.gradient = None;
		}
	}

	/// Return whether reverse-mode differentiation accumulates this parameter.
	pub fn requires_grad(&self) -> bool {
		self.inner.borrow().requires_grad
	}

	pub(crate) fn same_as(&self, other: &Self) -> bool {
		Rc::ptr_eq(&self.inner, &other.inner)
	}

	pub(crate) fn snapshot(&self) -> (Matrix, u64, bool) {
		let state = self.inner.borrow();
		(state.data.clone(), state.version, state.requires_grad)
	}

	pub(crate) fn validate_version(&self, expected: u64) -> Result<()> {
		let state = self.inner.borrow();
		if state.version != expected {
			return Err(Error::failed_precondition(format!(
				"parameter {} changed after its value was saved for backward",
				state.name
			)));
		}
		Ok(())
	}

	pub(crate) fn accumulate_gradient(&self, gradient: Matrix) -> Result<()> {
		let previous = {
			let state = self.inner.borrow();
			if !state.requires_grad {
				return Ok(());
			}
			state.gradient.clone()
		};
		let gradient = match previous {
			Some(previous) => matrix::add(&previous, &gradient)?,
			None => gradient,
		};
		self.inner.borrow_mut().gradient = Some(gradient);
		Ok(())
	}

	pub(crate) fn replace_data(&self, data: Matrix) -> Result<()> {
		let mut state = self.inner.borrow_mut();
		if state.data.shape() != data.shape()
			|| state.data.dtype() != data.dtype()
			|| !state.data.engine_handle().same_as(data.engine_handle())
		{
			return Err(Error::invalid_argument(format!(
				"replacement data does not match parameter {}",
				state.name
			)));
		}
		state.data = data;
		state.version = state
			.version
			.checked_add(1)
			.ok_or_else(|| Error::resource_exhausted("parameter mutation version exhausted"))?;
		Ok(())
	}

	pub(crate) fn mark_updated(&self) -> Result<()> {
		let mut state = self.inner.borrow_mut();
		state.version = state
			.version
			.checked_add(1)
			.ok_or_else(|| Error::resource_exhausted("parameter mutation version exhausted"))?;
		Ok(())
	}

	pub(crate) fn validate_can_update(&self) -> Result<()> {
		if self.inner.borrow().version == u64::MAX {
			return Err(Error::resource_exhausted(
				"parameter mutation version exhausted",
			));
		}
		Ok(())
	}

	pub(crate) fn clear_gradient(&self) {
		self.inner.borrow_mut().gradient = None;
	}
}
