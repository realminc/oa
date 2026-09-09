use std::{cell::RefCell, marker::PhantomData, rc::Rc};

use crate::{Error, LogComponent, LogLevel, LogOptions, Matrix, Result};

use super::{
	ComputeDispatch, Event, ExecutionPlan, SemanticDispatch, Storage,
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

pub(crate) enum CaptureAttempt<T> {
	Captured { plan: Box<ExecutionPlan>, output: T },
	Rejected { error: Error, output: T },
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

	pub(crate) fn owns_matrix(&self, matrix: &Matrix) -> bool {
		self.handle.same_as(matrix.engine_handle())
	}

	pub(crate) fn same_as_handle(&self, handle: &EngineHandle) -> bool {
		self.handle.same_as(handle)
	}

	pub(crate) fn begin_stable_resource_frame(&self) -> Result<()> {
		self.handle.begin_stable_resource_frame()
	}

	pub(crate) fn seal_all_stable_resources_external(&self) -> Result<()> {
		self.handle.seal_all_stable_resources_external()
	}

	pub(crate) fn seal_stable_resource_inputs(&self) -> Result<()> {
		self.handle.seal_stable_resource_inputs()
	}

	pub(crate) fn end_stable_resource_frame(&self) {
		self.handle.end_stable_resource_frame();
	}

	pub(crate) fn has_pending_work(&self) -> bool {
		self.handle.has_pending_work()
	}

	pub(crate) fn abort_pending_work(&self) {
		self.handle.abort_pending_work();
	}

	#[cfg(test)]
	pub(crate) fn force_next_plan_compilation_failure(&self) {
		super::plan::force_next_compilation_failure();
	}

	pub(super) fn build(selection: DeviceSelection, log_options: LogOptions) -> Result<Self> {
		let logger = Logger::new(log_options)?;
		let log_selection = logger.select();
		let instance = vk::Instance::new()?;
		let physical_device = vk::PhysicalDevice::select(&instance, selection)?;
		let device = vk::Device::new(&instance, physical_device)?;
		let retirement = vk::RetirementService::new(&device);
		log_engine_identity(&device);

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
		self.handle.checkpoint(false)
	}

	/// Submit pending eager work with one device timestamp pair around the batch.
	///
	/// Unlike [`Engine::checkpoint`], this requires at least one pending operation;
	/// there is no meaningful device interval for an empty queue checkpoint.
	///
	/// # Errors
	///
	/// Returns the errors from [`Engine::checkpoint`], `MissingCapability` when
	/// timestamps are unavailable, or `FailedPrecondition` for an empty batch.
	pub fn checkpoint_timed(&self) -> Result<Event> {
		self.handle.checkpoint(true)
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
		let plan = guard.finish(&[])?;
		Ok((plan, output))
	}

	pub(crate) fn capture_observed_matrix(
		&self,
		capture: impl FnOnce() -> Result<Matrix>,
	) -> Result<(ExecutionPlan, Matrix)> {
		let guard = CaptureGuard::begin(self.handle.clone())?;
		let output = capture()?;
		let plan = guard.finish(&[&output])?;
		Ok((plan, output))
	}

	pub(crate) fn capture_observed_training_matrix_preserving(
		&self,
		capture: impl FnOnce() -> Result<Matrix>,
	) -> Result<CaptureAttempt<Matrix>> {
		let guard = CaptureGuard::begin(self.handle.clone())?;
		let output = capture()?;
		Ok(match guard.finish_preserving(&[&output]) {
			Ok(plan) => CaptureAttempt::Captured {
				plan: Box::new(plan),
				output,
			},
			Err(error) => CaptureAttempt::Rejected { error, output },
		})
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

fn log_engine_identity(device: &vk::Device) {
	let physical = device.physical();
	let info = &physical.info;
	let conformance = info.conformance_version;
	let driver = if info.driver_info.is_empty() {
		info.driver_name.clone()
	} else {
		format!("{} · {}", info.driver_name, info.driver_info)
	};
	crate::log_info!(
		LogComponent::ENGINE,
		"oa engine v{} · Vulkan · 1 compute device",
		env!("CARGO_PKG_VERSION")
	);
	crate::log_info!(
		LogComponent::RUNTIME,
		"[0] ComputeDevice · {} · {} · Vulkan {} · {}",
		info.name,
		info.device_type,
		info.api_version,
		format_capacity(info.local_memory_bytes)
	);
	crate::log_info!(
		LogComponent::RUNTIME,
		"    Driver · {} · id {:?} · version 0x{:08x} · conformance {}.{}.{}.{}",
		driver,
		info.driver_id,
		info.driver_version,
		conformance.major,
		conformance.minor,
		conformance.subminor,
		conformance.patch
	);
	crate::log_info!(
		LogComponent::RUNTIME,
		"    Hardware · PCI {:04x}:{:04x} · compute queue family {}",
		info.vendor_id,
		info.device_id,
		physical.compute_queue_family
	);
}

fn format_capacity(bytes: u64) -> String {
	const GIB: u64 = 1024 * 1024 * 1024;
	const MIB: u64 = 1024 * 1024;
	if bytes >= GIB {
		format!("{:.2} GiB local memory", bytes as f64 / GIB as f64)
	} else {
		format!("{:.2} MiB local memory", bytes as f64 / MIB as f64)
	}
}

impl EngineHandle {
	pub(crate) fn same_as(&self, other: &Self) -> bool {
		Rc::ptr_eq(&self.state, &other.state)
	}

	pub(crate) fn create_storage(&self, bytes: &[u8]) -> Result<Storage> {
		let mut state = self.state.borrow_mut();
		let device = state.device.clone();
		state.session.create_storage(&device, bytes)
	}

	pub(crate) fn begin_stable_resource_frame(&self) -> Result<()> {
		self.state
			.borrow_mut()
			.session
			.begin_stable_resource_frame()
	}

	pub(crate) fn seal_all_stable_resources_external(&self) -> Result<()> {
		self.state
			.borrow_mut()
			.session
			.seal_all_stable_resources_external()
	}

	pub(crate) fn seal_stable_resource_inputs(&self) -> Result<()> {
		self.state
			.borrow_mut()
			.session
			.seal_stable_resource_inputs()
	}

	pub(crate) fn end_stable_resource_frame(&self) {
		self.state.borrow_mut().session.end_stable_resource_frame();
	}

	pub(crate) fn has_pending_work(&self) -> bool {
		!self.state.borrow().session.is_empty()
	}

	pub(crate) fn abort_pending_work(&self) {
		self.state.borrow_mut().session.abort();
	}

	pub(super) fn create_alias_arena(&self, size: usize) -> Result<vk::Buffer> {
		let device = self.state.borrow().device.clone();
		vk::Buffer::host_visible_alias_arena(&device, size)
	}

	pub(super) fn release_stable_transient_resources(&self, retired: &[vk::Buffer]) {
		self.state
			.borrow_mut()
			.session
			.release_stable_transient_resources(retired);
	}

	pub(crate) fn capture_active(&self) -> bool {
		self.state.borrow().capture_active
	}

	pub(crate) fn checkpoint(&self, timed: bool) -> Result<Event> {
		if let Some(event) = self.flush_impl(timed)? {
			return Ok(event);
		}
		if timed {
			return Err(Error::failed_precondition(
				"timed checkpoint requires pending eager work",
			));
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

	pub(crate) fn record_semantic(
		&self,
		dispatch: ComputeDispatch<'_>,
		semantic: SemanticDispatch<'_>,
	) -> Result<()> {
		let mut state = self.state.borrow_mut();
		let device = state.device.clone();
		state.session.record_semantic(&device, dispatch, semantic)
	}

	pub(crate) fn attach_semantic_autograd(
		&self,
		matrix_value: u64,
		sequence: u64,
	) -> Result<Option<(super::SemanticOpId, u64)>> {
		self.state
			.borrow_mut()
			.session
			.attach_autograd(matrix_value, sequence)
	}

	pub(crate) fn semantic_operation_count(&self) -> usize {
		self.state.borrow().session.semantic_operation_count()
	}

	pub(crate) fn complete_semantic_autograd(
		&self,
		forward: super::SemanticOpId,
		sequence: u64,
		generation: u64,
		backward_first: usize,
	) -> Result<()> {
		self.state.borrow_mut().session.complete_autograd(
			forward,
			sequence,
			generation,
			backward_first,
		)
	}

	pub(crate) fn flush(&self) -> Result<Option<Event>> {
		self.flush_impl(false)
	}

	fn flush_impl(&self, timed: bool) -> Result<Option<Event>> {
		let mut state = self.state.borrow_mut();
		if state.capture_active {
			return Err(Error::failed_precondition(
				"cannot submit eager work while execution capture is active",
			));
		}
		let Some(pending) = state.session.take(&[])? else {
			return Ok(None);
		};
		let recorded = if timed {
			state.device.record_timed_compute_graph(&pending.graph)
		} else {
			state.device.record_compute_graph(&pending.graph)
		};
		match recorded {
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
			let recorded = state.device.record_timed_compute_graph(plan.graph());
			if recorded.is_ok() {
				plan.mark_timed_recorded();
			}
			recorded
		} else {
			plan.reusable_command(&state.device)
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

	fn finish(mut self, observed_outputs: &[&Matrix]) -> Result<ExecutionPlan> {
		let pending = {
			let mut state = self.engine.state.borrow_mut();
			let pending = match state.session.take(observed_outputs) {
				Ok(pending) => pending,
				Err(error) => {
					state.session.abort();
					state.capture_active = false;
					self.active = false;
					return Err(error);
				}
			};
			state.capture_active = false;
			self.active = false;
			pending
		};
		let Some(pending) = pending else {
			return Err(Error::failed_precondition(
				"execution capture recorded no executable work",
			));
		};
		pending.mark_captured();
		ExecutionPlan::new(self.engine.clone(), pending, false)
	}

	fn finish_preserving(mut self, observed_outputs: &[&Matrix]) -> Result<ExecutionPlan> {
		let pending = {
			let mut state = self.engine.state.borrow_mut();
			let pending = state.session.snapshot(observed_outputs);
			state.capture_active = false;
			self.active = false;
			pending?
		};
		let Some(pending) = pending else {
			return Err(Error::failed_precondition(
				"execution capture recorded no executable work",
			));
		};
		let plan = ExecutionPlan::new(self.engine.clone(), pending, true)?;
		plan.validate_training_replay_safety()?;
		let mut state = self.engine.state.borrow_mut();
		plan.precompile(&state.device)?;
		state.session.commit_captured_snapshot()?;
		Ok(plan)
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
	use super::{CaptureAttempt, DeviceSelection, Engine, EngineBuilder, format_capacity};
	use crate::runtime::{BufferBinding, ComputeDispatch, PushConstant, shader::KernelId};
	use crate::{Matrix, matrix};

	#[test]
	fn default_builder_uses_automatic_device_selection() {
		assert_eq!(
			EngineBuilder::default().device_selection,
			DeviceSelection::Automatic
		);
	}

	#[test]
	fn formats_device_local_capacity_for_the_startup_banner() {
		assert_eq!(
			format_capacity(512 * 1024 * 1024),
			"512.00 MiB local memory"
		);
		assert_eq!(format_capacity(12_338_790_400), "11.49 GiB local memory");
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
	fn reuses_retired_exact_size_storage_without_exposing_old_bytes() -> crate::Result<()> {
		let engine = Engine::new()?;
		let handle = engine.handle();
		let first = handle.create_storage(&[0xa5_u8; 64])?;
		let first_buffer = first.buffer().expect("nonempty storage has a buffer");
		let first_raw = first_buffer.raw();
		let first_descriptor = first_buffer.descriptor_index();
		drop(first);

		let second = handle.create_storage(&[0_u8; 64])?;
		let second_buffer = second.buffer().expect("nonempty storage has a buffer");
		assert_eq!(second_buffer.raw(), first_raw);
		assert_eq!(second_buffer.descriptor_index(), first_descriptor);
		let mut bytes = [0xff_u8; 64];
		second.read(&mut bytes)?;
		assert_eq!(bytes, [0_u8; 64]);
		Ok(())
	}

	#[test]
	#[ignore = "requires a hardware Vulkan 1.3 compute device"]
	fn stable_resource_frames_reuse_equal_size_allocation_ordinals() -> crate::Result<()> {
		let engine = Engine::new()?;
		engine.begin_stable_resource_frame()?;
		let first = engine.handle.create_storage(&[0xa5_u8; 64])?;
		let first_raw = first.buffer().expect("nonempty storage has a buffer").raw();
		engine.seal_all_stable_resources_external()?;
		engine.end_stable_resource_frame();

		engine.begin_stable_resource_frame()?;
		let second = engine.handle.create_storage(&[0x5a_u8; 64])?;
		engine.seal_all_stable_resources_external()?;
		engine.end_stable_resource_frame();

		assert_eq!(
			second
				.buffer()
				.expect("nonempty storage has a buffer")
				.raw(),
			first_raw
		);
		let mut bytes = [0_u8; 64];
		second.read(&mut bytes)?;
		assert_eq!(bytes, [0x5a_u8; 64]);
		Ok(())
	}

	#[test]
	#[ignore = "requires a hardware Vulkan 1.3 compute device"]
	fn capture_preserves_replay_input_and_transient_lifetimes() -> crate::Result<()> {
		let engine = Engine::new()?;
		engine.begin_stable_resource_frame()?;
		let input = Matrix::from_f32(&engine, [4], &[1.0, 2.0, 3.0, 4.0])?;
		engine.seal_stable_resource_inputs()?;
		let captured = engine.capture(|| {
			let intermediate = matrix::add(&input, &input)?;
			matrix::add(&intermediate, &input)
		});
		engine.end_stable_resource_frame();
		let (plan, output) = captured?;

		let resources = plan.captured_resources();
		assert_eq!(resources.len(), 3);
		assert!(resources[0].stable_replay_input());
		assert!(!resources[0].stable_transient());
		assert_eq!(
			(resources[0].first_access(), resources[0].last_access()),
			(0, 1)
		);
		assert!(resources[1].stable_transient());
		assert_eq!(
			(resources[1].first_access(), resources[1].last_access()),
			(0, 1)
		);
		assert_eq!(resources[1].unaccounted_owner_count(), 0);
		assert!(resources[1].alias_candidate());
		assert!(resources[2].stable_transient());
		assert_eq!(
			(resources[2].first_access(), resources[2].last_access()),
			(1, 1)
		);
		assert_eq!(resources[2].unaccounted_owner_count(), 1);
		assert!(!resources[2].alias_candidate());
		let diagnostics = plan.diagnostics();
		assert_eq!(diagnostics.captured_resource_count(), 3);
		assert_eq!(diagnostics.stable_replay_input_count(), 1);
		assert_eq!(diagnostics.stable_transient_count(), 2);
		assert_eq!(diagnostics.alias_candidate_count(), 1);
		assert_eq!(diagnostics.alias_materialized_count(), 0);
		assert_eq!(diagnostics.materialized_alias_savings(), 0);

		engine.submit(&plan)?.wait()?;
		assert_eq!(output.read_f32()?, [3.0, 6.0, 9.0, 12.0]);
		Ok(())
	}

	#[test]
	#[ignore = "requires a hardware Vulkan 1.3 compute device"]
	fn capture_materializes_non_overlapping_transients_into_one_arena() -> crate::Result<()> {
		let engine = Engine::new()?;
		engine.begin_stable_resource_frame()?;
		let input = Matrix::from_f32(&engine, [4], &[1.0, 2.0, 3.0, 4.0])?;
		engine.seal_stable_resource_inputs()?;
		let captured = engine.capture(|| {
			let first = matrix::add(&input, &input)?;
			let _dead = matrix::add(&first, &input)?;
			let second = matrix::add(&input, &input)?;
			matrix::add(&second, &input)
		});
		engine.end_stable_resource_frame();
		let (plan, output) = captured?;
		let diagnostics = plan.diagnostics();
		assert_eq!(diagnostics.alias_candidate_count(), 3);
		assert_eq!(diagnostics.alias_materialized_count(), 2);
		assert_eq!(diagnostics.materialized_alias_savings(), 16);
		assert_eq!(diagnostics.captured_resource_count(), 5);
		assert_eq!(diagnostics.physical_resource_count(), 4);
		assert_eq!(diagnostics.fallback_count(), 0);
		assert_eq!(plan.alias_materialization_fallback_reason(), None);
		assert!(
			plan.graph().nodes()[0].buffers[2]
				.buffer
				.same_as(&plan.graph().nodes()[2].buffers[2].buffer)
		);
		assert_eq!(diagnostics.barrier_count(), 3);
		assert_eq!(
			plan.captured_resources()
				.iter()
				.filter(|resource| resource.alias_materialized())
				.count(),
			2
		);

		engine.submit(&plan)?.wait()?;
		assert_eq!(output.read_f32()?, [3.0, 6.0, 9.0, 12.0]);
		engine.submit(&plan)?.wait()?;
		assert_eq!(output.read_f32()?, [3.0, 6.0, 9.0, 12.0]);
		Ok(())
	}

	#[test]
	#[ignore = "requires a hardware Vulkan 1.3 compute device"]
	fn replay_safety_rejection_preserves_the_source_recording() -> crate::Result<()> {
		let engine = Engine::new()?;
		let parameter = Matrix::from_f32(&engine, [1], &[1.0])?;
		let attempt = engine.capture_observed_training_matrix_preserving(|| {
			let buffers = [BufferBinding::read_write(parameter.storage())];
			engine.handle.record(ComputeDispatch {
				operation: "ml.adamw",
				kernel: KernelId::MlAdamWF32,
				buffers: &buffers,
				push_constants: &[PushConstant::U32(1)],
				workgroups: [1, 1, 1],
			})?;
			Ok(parameter.clone())
		})?;
		let CaptureAttempt::Rejected { error, .. } = attempt else {
			panic!("host-stepped AdamW capture must be rejected");
		};
		assert_eq!(error.kind(), crate::ErrorKind::FailedPrecondition);
		assert_eq!(
			error.message(),
			"training program operation ml.adamw embeds host-stepped optimizer state; use a replay-state kernel"
		);
		let mut state = engine.handle.state.borrow_mut();
		assert!(!state.session.is_empty());
		state.session.abort();
		Ok(())
	}

	#[test]
	#[ignore = "requires a hardware Vulkan 1.3 compute device"]
	fn recorded_storage_is_not_recycled_before_command_retirement() -> crate::Result<()> {
		let engine = Engine::new()?;
		let handle = engine.handle();
		let left = handle.create_storage(&[0_u8; 16])?;
		let right = handle.create_storage(&[0_u8; 16])?;
		let output = handle.create_storage(&[0_u8; 16])?;
		let output_raw = output
			.buffer()
			.expect("nonempty storage has a buffer")
			.raw();
		let buffers = [
			BufferBinding::read(&left),
			BufferBinding::read(&right),
			BufferBinding::write(&output),
		];
		handle.record(ComputeDispatch {
			operation: "test.retained_add",
			kernel: KernelId::MatrixAddF32,
			buffers: &buffers,
			push_constants: &[PushConstant::U32(4)],
			workgroups: [1, 1, 1],
		})?;
		drop(output);

		let replacement = handle.create_storage(&[0_u8; 16])?;
		assert_ne!(
			replacement
				.buffer()
				.expect("nonempty storage has a buffer")
				.raw(),
			output_raw
		);
		handle.checkpoint(false)?.wait()?;
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
