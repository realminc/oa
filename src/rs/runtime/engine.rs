use std::{cell::RefCell, marker::PhantomData, rc::Rc};

use crate::{Error, LogComponent, LogLevel, LogOptions, Result};

use super::{
	ComputeDispatch, Event, ExecutionPlan, Storage,
	log::{LogSelection, Logger},
	session::ExecutionSession,
	vk,
};

#[cfg(test)]
use super::executable_graph::ExecutableGraph;

/// Policy for selecting the local device owned by an engine.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DeviceSelection {
	/// Select the highest-ranked compatible hardware device.
	#[default]
	Automatic,
	/// Select an exact physical-device ordinal from backend enumeration.
	Index(usize),
}

/// Configuration used to construct one OA execution engine.
#[derive(Clone, Debug, Default)]
#[must_use]
pub struct EngineBuilder {
	device_selection: DeviceSelection,
	log_options: LogOptions,
}

impl EngineBuilder {
	/// Create a builder using automatic hardware-device selection.
	pub fn new() -> Self {
		Self {
			device_selection: DeviceSelection::Automatic,
			log_options: LogOptions::new(),
		}
	}

	/// Select how this engine chooses its single local device.
	pub const fn devices(mut self, selection: DeviceSelection) -> Self {
		self.device_selection = selection;
		self
	}

	/// Configure the logging session owned by the constructed engine.
	pub fn logging(mut self, options: LogOptions) -> Self {
		self.log_options = options;
		self
	}

	/// Construct the configured engine and its execution services.
	///
	/// # Errors
	///
	/// Returns an error when the backend cannot be loaded, its required API is
	/// unavailable, the requested device does not exist or lacks OA's required
	/// capabilities, logging configuration or file creation fails, or a device
	/// service cannot be constructed.
	pub fn build(self) -> Result<Engine> {
		Engine::build(self.device_selection, self.log_options)
	}
}

/// Sole owner of OA's local execution services.
///
/// The current one-device recorder is thread-affine. `Engine` intentionally
/// remains neither `Send` nor `Sync` until concurrent queue scheduling and
/// submission synchronization are implemented and verified.
pub struct Engine {
	// Restore thread-local selection before closing the logger and releasing the
	// execution handle. Rust drops fields in declaration order.
	_log_selection: LogSelection,
	logger: Logger,
	handle: EngineHandle,
	_not_send_sync: PhantomData<Rc<()>>,
}

struct EngineState {
	// Fields drop in declaration order. Retirement disconnects first and joins only
	// an already-drained host worker; an active worker retains the device and
	// transitive instance ownership until every submitted epoch completes.
	retirement: vk::RetirementService,
	session: ExecutionSession,
	device: vk::Device,
	next_epoch: u64,
	capture_active: bool,
}

/// Thread-affine shared access to the execution owner retained by its values.
#[derive(Clone)]
pub(crate) struct EngineHandle {
	state: Rc<RefCell<EngineState>>,
}

impl Engine {
	/// Begin explicit engine configuration.
	pub fn builder() -> EngineBuilder {
		EngineBuilder::new()
	}

	/// Create an engine using one compute-capable Vulkan device.
	///
	/// # Errors
	///
	/// Returns an error when Vulkan cannot be loaded, Vulkan 1.3 is unavailable,
	/// no compatible hardware device exists, or logical-device creation fails.
	pub fn new() -> Result<Self> {
		Self::builder().build()
	}

	pub(super) fn build(selection: DeviceSelection, log_options: LogOptions) -> Result<Self> {
		let logger = Logger::new(log_options)?;
		let log_selection = logger.select();
		let instance = vk::Instance::new()?;
		let physical_device = vk::PhysicalDevice::select(&instance, selection)?;
		let device = vk::Device::new(&instance, physical_device)?;
		let retirement = vk::RetirementService::new(&device);
		crate::log_info!(LogComponent::ENGINE, "initialized Vulkan compute engine");

		Ok(Self {
			_log_selection: log_selection,
			logger,
			handle: EngineHandle {
				state: Rc::new(RefCell::new(EngineState {
					retirement,
					session: ExecutionSession::new(),
					device,
					next_epoch: 0,
					capture_active: false,
				})),
			},
			_not_send_sync: PhantomData,
		})
	}

	/// Submit an explicit completion checkpoint to this engine's compute queue.
	///
	/// Pending eager operations are submitted together as one batch. When no
	/// eager work is pending, an empty checkpoint is submitted. The returned
	/// event completes after that exact queue point. Dropping it does not wait.
	///
	/// # Errors
	///
	/// Returns an error when command recording, submission, or retirement fails,
	/// or when the engine exhausts its timeline epoch space.
	pub fn checkpoint(&self) -> Result<Event> {
		self.handle.checkpoint()
	}

	/// Capture an isolated operation sequence as an immutable execution plan.
	///
	/// Capture records but never submits or waits. The returned value may include
	/// matrices produced by the plan; those matrices become observable only after
	/// [`Engine::submit`] is called for the returned plan.
	///
	/// # Errors
	///
	/// Returns an error when eager work is already pending, capture is nested, the
	/// closure fails, or the closure records no executable work.
	pub fn capture<T>(&self, capture: impl FnOnce() -> Result<T>) -> Result<(ExecutionPlan, T)> {
		let guard = CaptureGuard::begin(self.handle.clone())?;
		let output = capture()?;
		let plan = guard.finish()?;
		Ok((plan, output))
	}

	/// Submit an immutable execution plan to its originating engine.
	///
	/// Any pending eager batch is submitted first. The returned event identifies
	/// exact completion of this plan replay. Submission does not wait.
	///
	/// # Errors
	///
	/// Returns an error when the plan belongs to another engine, capture is active,
	/// eager flushing fails, or command recording, submission, or retirement fails.
	pub fn submit(&self, plan: &ExecutionPlan) -> Result<Event> {
		self.handle.submit_plan(plan, false)
	}

	/// Submit a plan with one device timestamp pair around its complete graph.
	///
	/// The returned event exposes the device duration through
	/// [`Event::try_device_duration`] and [`Event::device_duration`]. Timing adds
	/// instrumentation only to this replay and does not wait.
	///
	/// # Errors
	///
	/// Returns the errors from [`Engine::submit`], or `MissingCapability` when the
	/// selected compute queue cannot provide timestamps.
	pub fn submit_timed(&self, plan: &ExecutionPlan) -> Result<Event> {
		self.handle.submit_plan(plan, true)
	}

	/// Write one record through this engine's logging session.
	///
	/// Filtered records return successfully without formatting or output.
	///
	/// # Errors
	///
	/// Returns the first retained console or file output failure, or an error if
	/// the logging session has already been closed.
	pub fn log(
		&self,
		level: LogLevel,
		component: LogComponent,
		text: impl AsRef<str>,
	) -> Result<()> {
		self.logger.write_text(level, component, text.as_ref())
	}

	/// Change this engine logger's minimum severity.
	pub fn set_log_level(&self, level: LogLevel) {
		self.logger.set_level(level);
	}

	/// Return this engine logger's current minimum severity.
	pub fn log_level(&self) -> LogLevel {
		self.logger.level()
	}

	/// Return the active log-file path when file output is enabled.
	pub fn log_path(&self) -> Option<std::path::PathBuf> {
		self.logger.path()
	}

	/// Flush this engine's logging session and report any retained sink failure.
	///
	/// # Errors
	///
	/// Returns the first console or file output failure.
	pub fn flush_log(&self) -> Result<()> {
		self.logger.flush()
	}

	/// Explicitly close this engine's logging session before releasing the engine.
	///
	/// Existing values retain the execution services they require, but no longer
	/// have a selected engine logging session after this call consumes `self`.
	///
	/// # Errors
	///
	/// Returns the first console or file flush failure.
	pub fn close(self) -> Result<()> {
		crate::log_info!(LogComponent::ENGINE, "closing Vulkan compute engine");
		self.logger.close()
	}

	pub(crate) fn handle(&self) -> EngineHandle {
		self.handle.clone()
	}
}

impl EngineHandle {
	pub(crate) fn same_as(&self, other: &Self) -> bool {
		Rc::ptr_eq(&self.state, &other.state)
	}

	pub(crate) fn create_storage(&self, bytes: &[u8]) -> Result<Storage> {
		let device = self.state.borrow().device.clone();
		Storage::from_bytes(&device, bytes)
	}

	pub(crate) fn checkpoint(&self) -> Result<Event> {
		if let Some(event) = self.flush()? {
			return Ok(event);
		}
		let mut state = self.state.borrow_mut();
		let command = state.device.record_empty()?;
		submit_recorded(&mut state, command)
	}

	pub(crate) fn record(&self, dispatch: ComputeDispatch<'_>) -> Result<()> {
		let mut state = self.state.borrow_mut();
		let device = state.device.clone();
		state.session.record(&device, dispatch)
	}

	pub(crate) fn flush(&self) -> Result<Option<Event>> {
		let mut state = self.state.borrow_mut();
		if state.capture_active {
			return Err(Error::failed_precondition(
				"cannot submit eager work while execution capture is active",
			));
		}
		let Some(pending) = state.session.take()? else {
			return Ok(None);
		};
		match state.device.record_compute_graph(&pending.graph) {
			Ok(command) => match submit_recorded(&mut state, command) {
				Ok(event) => {
					pending.mark_submitted(&event);
					Ok(Some(event))
				}
				Err(error) => {
					pending.mark_failed();
					Err(error)
				}
			},
			Err(error) => {
				pending.mark_failed();
				Err(error)
			}
		}
	}

	fn submit_plan(&self, plan: &ExecutionPlan, timed: bool) -> Result<Event> {
		if !plan.belongs_to(self) {
			return Err(Error::invalid_argument(
				"execution plan belongs to another engine",
			));
		}
		self.flush()?;
		let mut state = self.state.borrow_mut();
		let recorded = if timed {
			state.device.record_timed_compute_graph(plan.graph())
		} else {
			state.device.record_compute_graph(plan.graph())
		};
		let command = match recorded {
			Ok(command) => command,
			Err(error) => {
				plan.mark_failed();
				return Err(error);
			}
		};
		match submit_recorded(&mut state, command) {
			Ok(event) => {
				plan.mark_submitted(&event);
				Ok(event)
			}
			Err(error) => {
				plan.mark_failed();
				Err(error)
			}
		}
	}

	#[cfg(test)]
	pub(crate) fn submit_all(&self, dispatches: &[ComputeDispatch<'_>]) -> Result<Event> {
		let mut state = self.state.borrow_mut();
		let graph = ExecutableGraph::from_dispatches(&state.device, dispatches)?;
		let command = state.device.record_compute_graph(&graph)?;
		submit_recorded(&mut state, command)
	}
}

struct CaptureGuard {
	engine: EngineHandle,
	active: bool,
}

impl CaptureGuard {
	fn begin(engine: EngineHandle) -> Result<Self> {
		{
			let mut state = engine.state.borrow_mut();
			if state.capture_active {
				return Err(Error::failed_precondition(
					"nested execution capture is not supported",
				));
			}
			if !state.session.is_empty() {
				return Err(Error::failed_precondition(
					"pending eager work must be submitted before execution capture",
				));
			}
			state.capture_active = true;
		}
		Ok(Self {
			engine,
			active: true,
		})
	}

	fn finish(mut self) -> Result<ExecutionPlan> {
		let pending = {
			let mut state = self.engine.state.borrow_mut();
			let pending = state.session.take();
			state.capture_active = false;
			self.active = false;
			pending?
		};
		let Some(pending) = pending else {
			return Err(Error::failed_precondition(
				"execution capture recorded no executable work",
			));
		};
		pending.mark_captured();
		let (graph, outputs) = pending.into_parts();
		Ok(ExecutionPlan::new(self.engine.clone(), graph, outputs))
	}
}

impl Drop for CaptureGuard {
	fn drop(&mut self) {
		if !self.active {
			return;
		}
		let mut state = self.engine.state.borrow_mut();
		state.session.abort();
		state.capture_active = false;
	}
}

fn submit_recorded(state: &mut EngineState, command: vk::RecordedCommandBuffer) -> Result<Event> {
	let Some(epoch) = state.next_epoch.checked_add(1) else {
		state.device.free(command);
		return Err(Error::resource_exhausted(
			"Vulkan submission epoch exhausted",
		));
	};
	if let Err(error) = state.retirement.prepare() {
		state.device.free(command);
		return Err(error);
	}
	if let Err(error) = state.device.submit(&command, epoch) {
		state.device.free(command);
		return Err(error);
	}
	let timing = command.timing();
	state.next_epoch = epoch;
	let retirement = state.retirement.retire(command, epoch)?;
	Ok(Event::new(state.device.clone(), epoch, retirement, timing))
}

#[cfg(test)]
mod tests {
	use super::{DeviceSelection, Engine, EngineBuilder};
	use crate::runtime::{BufferBinding, ComputeDispatch, PushConstant, shader::KernelId};

	#[test]
	fn default_builder_uses_automatic_device_selection() {
		assert_eq!(
			EngineBuilder::default().device_selection,
			DeviceSelection::Automatic
		);
	}

	#[test]
	#[ignore = "requires a hardware Vulkan 1.3 compute device"]
	fn allocates_records_and_frees_an_empty_command_buffer() -> crate::Result<()> {
		let engine = Engine::new()?;
		let device = engine.handle.state.borrow().device.clone();
		let command = device.record_empty()?;
		device.free(command);
		Ok(())
	}

	#[test]
	#[ignore = "requires a hardware Vulkan 1.3 compute device"]
	fn submits_dependent_dispatches_as_one_hazard_planned_graph() -> crate::Result<()> {
		let engine = Engine::new()?;
		let handle = engine.handle();
		let encode = |values: &[f32]| {
			values
				.iter()
				.flat_map(|value| value.to_ne_bytes())
				.collect::<Vec<_>>()
		};
		let left = handle.create_storage(&encode(&[1.0, 2.0, 3.0, 4.0]))?;
		let right = handle.create_storage(&encode(&[10.0, 20.0, 30.0, 40.0]))?;
		let intermediate = handle.create_storage(&[0_u8; 16])?;
		let output = handle.create_storage(&[0_u8; 16])?;

		let first_buffers = [
			BufferBinding::read(&left),
			BufferBinding::read(&right),
			BufferBinding::write(&intermediate),
		];
		let second_buffers = [
			BufferBinding::read(&intermediate),
			BufferBinding::read(&right),
			BufferBinding::write(&output),
		];
		let element_count = [PushConstant::U32(4)];
		let dispatches = [
			ComputeDispatch {
				operation: "test.first_add",
				kernel: KernelId::MatrixAddF32,
				buffers: &first_buffers,
				push_constants: &element_count,
				workgroups: [1, 1, 1],
			},
			ComputeDispatch {
				operation: "test.second_add",
				kernel: KernelId::MatrixAddF32,
				buffers: &second_buffers,
				push_constants: &element_count,
				workgroups: [1, 1, 1],
			},
		];

		handle.submit_all(&dispatches)?.wait()?;
		let mut bytes = [0_u8; 16];
		output.read(&mut bytes)?;
		let actual = bytes
			.as_chunks::<4>()
			.0
			.iter()
			.map(|word| f32::from_ne_bytes([word[0], word[1], word[2], word[3]]))
			.collect::<Vec<_>>();
		assert_eq!(actual, [21.0, 42.0, 63.0, 84.0]);
		Ok(())
	}
}
