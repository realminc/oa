//! Dynamic gradient scaling for mixed-precision training.
//!
//! `GradScaler` mirrors the `torch.cuda.amp.GradScaler` contract: scale the
//! loss before backward, unscale parameter gradients in place on-device, skip
//! the optimizer step when any unscaled gradient is non-finite, and adapt the
//! scale between steps. Scale growth and backoff follow PyTorch defaults.
//!
//! Unscaling and the finite check are GPU operations; the only host boundary
//! is one scalar readback inside `step_optimizer`, where torch performs the
//! equivalent CPU-side gate. Combined with `OA_FORCE_PRECISION=FP16`
//! FP16-storage lowering, this is the admitted AMP route.

use crate::ml::matrix::detach;
use crate::ml::{Optimizer, Parameter};
use crate::runtime::{BufferBinding, ComputeDispatch, KernelId, PushConstant, SemanticDispatch};
use crate::{DType, Engine, Error, Matrix, OpAttribute, Result};

const DEFAULT_INIT_SCALE: f32 = 65_536.0;
const DEFAULT_GROWTH_FACTOR: f32 = 2.0;
const DEFAULT_BACKOFF_FACTOR: f32 = 0.5;
const DEFAULT_GROWTH_INTERVAL: u64 = 2000;

/// Dynamic loss scaler for mixed-precision training steps.
///
/// The scale multiplies the loss so F16-storage gradients stay in
/// representable range; `step_optimizer` skips parameter updates on the
/// first non-finite unscaled gradient and `update` then halves the scale.
/// After `growth_interval` consecutive finite steps the scale doubles.
/// Device-resident scaler state replayed inside captured programs.
///
/// `scale` and `counter` drive scale/backoff bookkeeping on the device;
/// `flag` is the finite gate consumed by gated captured optimizer kernels.
/// All three are single-element F32 scalars so `record_semantic` broadcasts
/// them against loss and gradient shapes without a view.
#[derive(Clone)]
struct ScalerDeviceState {
	scale: Matrix,
	counter: Matrix,
	flag: Matrix,
}

#[derive(Clone)]
pub struct GradScaler {
	scale: f32,
	growth_factor: f32,
	backoff_factor: f32,
	growth_interval: u64,
	growth_tracker: u64,
	found_inf: bool,
	device_state: Option<ScalerDeviceState>,
}

impl GradScaler {
	/// Create a scaler with the PyTorch default configuration
	/// (`init_scale=65536`, growth `2.0`, backoff `0.5`, interval `2000`).
	#[must_use]
	pub fn new() -> Self {
		Self {
			scale: DEFAULT_INIT_SCALE,
			growth_factor: DEFAULT_GROWTH_FACTOR,
			backoff_factor: DEFAULT_BACKOFF_FACTOR,
			growth_interval: DEFAULT_GROWTH_INTERVAL,
			growth_tracker: 0,
			found_inf: false,
			device_state: None,
		}
	}

	/// Create a scaler with an explicit dynamic-scaling configuration.
	///
	/// # Errors
	///
	/// Returns an error when `init_scale`, `growth_factor`, or
	/// `backoff_factor` is not positive and finite, or `growth_interval` is
	/// zero.
	pub fn with_config(
		init_scale: f32,
		growth_factor: f32,
		backoff_factor: f32,
		growth_interval: u64,
	) -> Result<Self> {
		if !init_scale.is_finite()
			|| init_scale <= 0.0
			|| !growth_factor.is_finite()
			|| growth_factor <= 0.0
			|| !backoff_factor.is_finite()
			|| backoff_factor <= 0.0
			|| growth_interval == 0
		{
			return Err(Error::invalid_argument(
				"GradScaler requires positive finite scale/growth/backoff and a nonzero growth interval",
			));
		}
		Ok(Self {
			scale: init_scale,
			growth_factor,
			backoff_factor,
			growth_interval,
			growth_tracker: 0,
			found_inf: false,
			device_state: None,
		})
	}

	/// Return the current scale.
	#[must_use]
	pub fn scale(&self) -> f32 {
		self.scale
	}

	/// Allocate device-resident scaler state on `engine` for captured replay.
	///
	/// After this call, `scale_loss`, `unscale_gradients`, and
	/// `record_finite_gate` all reference the device buffers instead of the
	/// host scale, so a captured program replays scale adaptation and the
	/// optimizer gate without a mid-step host decision. Bind
	/// [`GradScaler::finite_flag`] to the optimizer once through
	/// `Optimizer::set_finite_gate`.
	///
	/// # Errors
	///
	/// Returns an error when buffer allocation or upload fails.
	pub fn prepare_captured(&mut self, engine: &Engine) -> Result<()> {
		let device_state = ScalerDeviceState {
			scale: Matrix::new_from_slice_on(engine, vec![], &[self.scale])?,
			counter: Matrix::new_from_slice_on(engine, vec![], &[0.0])?,
			flag: Matrix::new_from_slice_on(engine, vec![], &[1.0])?,
		};
		self.device_state = Some(device_state);
		Ok(())
	}

	/// Return the device-resident finite flag for optimizer binding.
	///
	/// The flag is a single-element F32 matrix written by
	/// [`GradScaler::record_finite_gate`]; bind it once with
	/// `Optimizer::set_finite_gate` before the first captured step.
	///
	/// # Errors
	///
	/// Returns an error when `prepare_captured` has not run.
	pub fn finite_flag(&self) -> Result<Matrix> {
		self
			.device_state
			.as_ref()
			.map(|state| state.flag.clone())
			.ok_or_else(|| Error::failed_precondition("GradScaler finite flag requires prepare_captured"))
	}

	/// Record the device-side finite gate update for the current step.
	///
	/// Dispatches `MlGradScalerAdvanceF32`: `check` decides the flag (finite
	/// iff every unscaled gradient element is finite), the scale backs off on
	/// overflow or grows after `growth_interval` finite steps, and the growth
	/// counter advances — all in device buffers, so the record replays the
	/// full dynamic-scaling contract inside a captured training program.
	/// Must follow `unscale_gradients` in record order.
	///
	/// # Errors
	///
	/// Returns an error when `prepare_captured` has not run, `check` is not a
	/// same-engine single-element F32 matrix, or recording fails.
	pub fn record_finite_gate(&mut self, check: &Matrix) -> Result<()> {
		const OPERATION: &str = crate::core::operation::ml::GRAD_SCALER_ADVANCE.name();
		let Some(state) = &self.device_state else {
			return Err(Error::failed_precondition(format!(
				"{OPERATION} requires prepare_captured"
			)));
		};
		if check.dtype() != DType::F32 || check.num_elements() != 1 {
			return Err(Error::invalid_argument(format!(
				"{OPERATION} requires a single-element F32 check"
			)));
		}
		if !check.engine_handle().same_as(state.scale.engine_handle()) {
			return Err(Error::invalid_argument(format!(
				"{OPERATION} requires the check on the scaler engine"
			)));
		}
		let buffers = [
			BufferBinding::read(check.storage()),
			BufferBinding::read_write(state.scale.storage()),
			BufferBinding::read_write(state.counter.storage()),
			BufferBinding::read_write(state.flag.storage()),
		];
		let growth_interval = u32::try_from(self.growth_interval)
			.map_err(|_| Error::invalid_argument(format!("{OPERATION} growth interval exceeds u32")))?;
		let push_constants = [
			PushConstant::F32(self.growth_factor),
			PushConstant::F32(self.backoff_factor),
			PushConstant::U32(growth_interval),
		];
		let inputs: &[&Matrix] = &[check, &state.scale, &state.counter, &state.flag];
		let outputs: &[&Matrix] = &[&state.scale, &state.counter, &state.flag];
		let attributes: &[OpAttribute] = &[];
		let dispatch = ComputeDispatch {
			kernel: KernelId::MlGradScalerAdvanceF32,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: [1, 1, 1],
			indirect: None,
		};
		let contract = dispatch.kernel.semantic_contract().ok_or_else(|| {
			Error::internal(format!(
				"lowering-only kernel {} cannot own one semantic ML operation",
				dispatch.kernel.report_name()
			))
		})?;
		check.engine_handle().record_semantic(
			dispatch,
			SemanticDispatch {
				contract,
				inputs,
				outputs,
				attributes,
			},
		)
	}

	/// Multiply `loss` by the current scale as a recorded device operation.
	///
	/// Under `prepare_captured` the multiplier is the device scale buffer, so
	/// replayed steps apply the scale the device-side advance produced for
	/// that step rather than the value baked at capture time.
	///
	/// # Errors
	///
	/// Returns an error when runtime recording fails.
	pub fn scale_loss(&self, loss: &Matrix) -> Result<Matrix> {
		match &self.device_state {
			Some(state) => loss.mul(&state.scale),
			None => loss.scale(self.scale),
		}
	}

	/// Divide every nonempty parameter gradient by the scale in place and
	/// return a `[1]` accumulator that is finite iff every unscaled gradient
	/// element is finite.
	///
	/// The unscale and the check are device-resident: each gradient is
	/// rescaled into detached replacement storage and its full sum is folded
	/// into the accumulator, so NaN or Inf propagates into the check value.
	///
	/// # Errors
	///
	/// Returns an error when runtime recording or allocation fails.
	pub fn unscale_gradients(&self, parameters: &[Parameter]) -> Result<Matrix> {
		let inverse_scale = match &self.device_state {
			Some(state) => detach(&state.scale.reciprocal()?)?,
			None => Matrix::new_from_slice(vec![], &[1.0 / self.scale])?,
		};
		let mut check: Option<Matrix> = None;
		for parameter in parameters {
			let Some(gradient) = parameter.gradient() else {
				continue;
			};
			let unscaled = detach(&gradient.mul(&inverse_scale)?)?;
			let gradient_sum = detach(&unscaled.sum(-1)?)?;
			parameter.set_gradient(unscaled);
			check = Some(match check {
				Some(acc) => acc.add(&gradient_sum)?,
				None => gradient_sum,
			});
		}
		check.ok_or_else(|| Error::invalid_argument("GradScaler found no nonempty parameter gradients"))
	}

	/// Apply `optimizer.step()` when `check` is finite; skip the update and
	/// flag the scale for backoff when it is not.
	///
	/// This is the one host boundary of the scaler contract: the check value
	/// is read back once per step, matching the CPU-side gate in torch.
	///
	/// # Errors
	///
	/// Returns an error when the check matrix is empty, readback fails, or
	/// `optimizer.step()` fails.
	pub fn step_optimizer(&mut self, optimizer: &mut dyn Optimizer, check: &Matrix) -> Result<bool> {
		let value = check
			.read_f32()?
			.first()
			.copied()
			.ok_or_else(|| Error::invalid_argument("GradScaler check matrix is empty"))?;
		if !value.is_finite() {
			self.found_inf = true;
			return Ok(false);
		}
		optimizer.step()?;
		Ok(true)
	}

	/// Adjust the scale after `step_optimizer`: halve it on a skipped step,
	/// otherwise double it after `growth_interval` consecutive finite steps.
	///
	/// This is the eager-path companion. Under `prepare_captured` the
	/// device-side advance owns scale bookkeeping during replay and this is a
	/// no-op; `device_scale` reads the current value back when needed.
	pub fn update(&mut self) {
		if self.device_state.is_some() {
			return;
		}
		if self.found_inf {
			self.scale *= self.backoff_factor;
			self.growth_tracker = 0;
			self.found_inf = false;
		} else {
			self.growth_tracker += 1;
			if self.growth_tracker >= self.growth_interval {
				self.scale *= self.growth_factor;
				self.growth_tracker = 0;
			}
		}
	}

	/// Read the current device-resident scale back to the host.
	///
	/// Costs one scalar synchronization; intended for diagnostics and
	/// logging, not per-step control flow.
	///
	/// # Errors
	///
	/// Returns an error when `prepare_captured` has not run or the readback
	/// fails.
	pub fn device_scale(&self) -> Result<f32> {
		let Some(state) = &self.device_state else {
			return Err(Error::failed_precondition(
				"GradScaler device scale requires prepare_captured",
			));
		};
		state
			.scale
			.read_f32()?
			.first()
			.copied()
			.ok_or_else(|| Error::internal("GradScaler device scale is empty"))
	}
}

impl Default for GradScaler {
	fn default() -> Self {
		Self::new()
	}
}
