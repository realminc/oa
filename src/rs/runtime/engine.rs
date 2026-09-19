use std::{cell::RefCell, marker::PhantomData, rc::Rc};

use crate::{Error, LogComponent, LogLevel, LogOptions, Matrix, Result};

use super::{
	AudioSemanticDispatch, ComputeDispatch, Event, ExecutionPlan, ImageSemanticDispatch,
	OptionalSemanticDispatch, SemanticDispatch, Storage,
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

/// Runtime-owned implementation for public hardware video decoder sessions.
pub(crate) struct VideoDecoderBackend {
	engine: EngineHandle,
	session: vk::DecodeSession,
	av1_ready: Vec<Option<Event>>,
	av1_host_frames: Vec<Option<Vec<u8>>>,
	vp9_ready: Vec<Option<Event>>,
	vp9_host_frames: Vec<Option<Vec<u8>>>,
}

impl VideoDecoderBackend {
	pub(crate) fn create(
		engine: &Engine,
		profile: crate::video::VideoDecodeProfile,
		coded_extent: crate::video::VideoExtent,
	) -> Result<Self> {
		let capabilities = engine.query_video_decode_capabilities(profile)?;
		let device_capabilities = engine.query_video_device_capabilities()?;
		if !device_capabilities.supports_decode_result_status_queries() {
			return Err(Error::missing_capability(
				"the public decoder requires Vulkan Video result-status queries",
			));
		}
		let max_dpb_slots = if matches!(profile, crate::video::VideoDecodeProfile::H264 { .. }) {
			capabilities.max_dpb_slots().min(16)
		} else {
			capabilities.max_dpb_slots()
		};
		let max_active_references = capabilities
			.max_active_reference_pictures()
			.min(max_dpb_slots);
		let session = engine
			.handle
			.state
			.borrow()
			.device
			.create_video_decode_session(profile, coded_extent, max_dpb_slots, max_active_references)?;
		Ok(Self {
			engine: engine.handle.clone(),
			session,
			av1_ready: vec![None; max_dpb_slots as usize],
			av1_host_frames: vec![None; max_dpb_slots as usize],
			vp9_ready: vec![None; max_dpb_slots as usize],
			vp9_host_frames: vec![None; max_dpb_slots as usize],
		})
	}

	pub(crate) fn set_h264_parameters(
		&mut self,
		sps: &crate::video::H264SequenceParameterSet,
		pps: &crate::video::H264PictureParameterSet,
	) -> Result<()> {
		self.session.set_h264_parameters(sps, pps)
	}

	pub(crate) fn set_h265_parameters(
		&mut self,
		vps: &crate::video::H265VideoParameterSet,
		sps: &crate::video::H265SequenceParameterSet,
		pps: &crate::video::H265PictureParameterSet,
	) -> Result<()> {
		self.session.set_h265_parameters(vps, sps, pps)
	}

	pub(crate) fn set_av1_parameters(
		&mut self,
		sequence: &crate::video::Av1SequenceHeader,
	) -> Result<()> {
		self.session.set_av1_parameters(sequence)
	}

	pub(crate) fn decode_h264(
		&mut self,
		access_unit: &[u8],
		sps: &crate::video::H264SequenceParameterSet,
		pps: &crate::video::H264PictureParameterSet,
		slice: &crate::video::H264SliceHeader,
	) -> Result<Vec<u8>> {
		let device = self.engine.state.borrow().device.clone();
		self.session.upload_h264_access_unit(access_unit)?;
		let command = self
			.session
			.record_h264_picture_for_readback(&device, sps, pps, slice)?;
		let event = {
			let mut state = self.engine.state.borrow_mut();
			submit_recorded(&mut state, command)?
		};
		event.wait()?;
		self.session.verify_decode_result()?;
		let command = self.session.record_decode_readback(&device)?;
		let event = {
			let mut state = self.engine.state.borrow_mut();
			submit_recorded(&mut state, command)?
		};
		event.wait()?;
		self.session.read_decode_yuv420()
	}

	pub(crate) fn decode_h264_native(
		&mut self,
		access_unit: &[u8],
		sps: &crate::video::H264SequenceParameterSet,
		pps: &crate::video::H264PictureParameterSet,
		slice: &crate::video::H264SliceHeader,
	) -> Result<crate::runtime::NativeDecodedFrame> {
		let device = self.engine.state.borrow().device.clone();
		self.session.upload_h264_access_unit(access_unit)?;
		let (command, slot) = self
			.session
			.record_h264_picture_native(&device, sps, pps, slice)?;
		let event = {
			let mut state = self.engine.state.borrow_mut();
			submit_recorded(&mut state, command)?
		};
		event.wait()?;
		self.session.verify_decode_result()?;
		self.session.native_frame(slot, event)
	}

	pub(crate) fn decode_h265(
		&mut self,
		access_unit: &[u8],
		sps: &crate::video::H265SequenceParameterSet,
		pps: &crate::video::H265PictureParameterSet,
		slice: &crate::video::H265SliceHeader,
	) -> Result<Vec<u8>> {
		let device = self.engine.state.borrow().device.clone();
		self.session.upload_h265_access_unit(access_unit)?;
		let command = self
			.session
			.record_h265_picture_for_readback(&device, sps, pps, slice)?;
		let event = {
			let mut state = self.engine.state.borrow_mut();
			submit_recorded(&mut state, command)?
		};
		event.wait()?;
		self.session.verify_decode_result()?;
		let command = self.session.record_decode_readback(&device)?;
		let event = {
			let mut state = self.engine.state.borrow_mut();
			submit_recorded(&mut state, command)?
		};
		event.wait()?;
		self.session.read_decode_yuv420()
	}

	pub(crate) fn decode_h265_native(
		&mut self,
		access_unit: &[u8],
		sps: &crate::video::H265SequenceParameterSet,
		pps: &crate::video::H265PictureParameterSet,
		slice: &crate::video::H265SliceHeader,
	) -> Result<crate::runtime::NativeDecodedFrame> {
		let device = self.engine.state.borrow().device.clone();
		self.session.upload_h265_access_unit(access_unit)?;
		let (command, slot) = self
			.session
			.record_h265_picture_native(&device, sps, pps, slice)?;
		let event = {
			let mut state = self.engine.state.borrow_mut();
			submit_recorded(&mut state, command)?
		};
		event.wait()?;
		self.session.verify_decode_result()?;
		self.session.native_frame(slot, event)
	}

	pub(crate) fn decode_av1(
		&mut self,
		access_unit: &[u8],
		picture: &crate::video::Av1Picture,
	) -> Result<Vec<u8>> {
		let tiles = picture
			.tiles
			.as_ref()
			.ok_or_else(|| Error::invalid_argument("coded AV1 picture has no tile payloads"))?;
		let device = self.engine.state.borrow().device.clone();
		self.session.upload_av1_access_unit(access_unit)?;
		let (command, slot) = self.session.record_av1_picture_for_readback(
			&device,
			&picture.sequence,
			&picture.frame,
			picture.frame_header_offset,
			tiles,
		)?;
		let decode_event = {
			let mut state = self.engine.state.borrow_mut();
			submit_recorded(&mut state, command)?
		};
		decode_event.wait()?;
		self.session.verify_decode_result()?;
		let command = self.session.record_decode_readback(&device)?;
		let event = {
			let mut state = self.engine.state.borrow_mut();
			submit_recorded(&mut state, command)?
		};
		event.wait()?;
		let decoded = self.session.read_decode_yuv420()?;
		let index =
			usize::try_from(slot).map_err(|_| Error::internal("AV1 output slot exceeds usize"))?;
		*self
			.av1_ready
			.get_mut(index)
			.ok_or_else(|| Error::internal("AV1 output slot exceeds readiness storage"))? =
			Some(decode_event);
		*self
			.av1_host_frames
			.get_mut(index)
			.ok_or_else(|| Error::internal("AV1 output slot exceeds host-frame storage"))? =
			Some(decoded.clone());
		Ok(decoded)
	}

	pub(crate) fn decode_av1_native(
		&mut self,
		access_unit: &[u8],
		picture: &crate::video::Av1Picture,
	) -> Result<crate::runtime::NativeDecodedFrame> {
		let tiles = picture
			.tiles
			.as_ref()
			.ok_or_else(|| Error::invalid_argument("coded AV1 picture has no tile payloads"))?;
		let device = self.engine.state.borrow().device.clone();
		self.session.upload_av1_access_unit(access_unit)?;
		let (command, slot) = self.session.record_av1_picture_native(
			&device,
			&picture.sequence,
			&picture.frame,
			picture.frame_header_offset,
			tiles,
		)?;
		let event = {
			let mut state = self.engine.state.borrow_mut();
			submit_recorded(&mut state, command)?
		};
		event.wait()?;
		self.session.verify_decode_result()?;
		let index =
			usize::try_from(slot).map_err(|_| Error::internal("AV1 output slot exceeds usize"))?;
		*self
			.av1_ready
			.get_mut(index)
			.ok_or_else(|| Error::internal("AV1 output slot exceeds readiness storage"))? =
			Some(event.clone());
		self.session.native_frame(slot, event)
	}

	pub(crate) fn show_existing_av1(
		&mut self,
		frame: &crate::video::Av1FrameHeader,
	) -> Result<Vec<u8>> {
		let slot = self.session.resolve_av1_show_existing_slot(frame)?;
		let index =
			usize::try_from(slot).map_err(|_| Error::internal("AV1 show-existing slot exceeds usize"))?;
		self
			.av1_host_frames
			.get(index)
			.and_then(Clone::clone)
			.ok_or_else(|| Error::failed_precondition("AV1 show-existing host frame is unavailable"))
	}

	pub(crate) fn show_existing_av1_native(
		&mut self,
		frame: &crate::video::Av1FrameHeader,
	) -> Result<crate::runtime::NativeDecodedFrame> {
		let slot = self.session.resolve_av1_show_existing_slot(frame)?;
		let index =
			usize::try_from(slot).map_err(|_| Error::internal("AV1 show-existing slot exceeds usize"))?;
		let ready = self
			.av1_ready
			.get(index)
			.and_then(Clone::clone)
			.ok_or_else(|| Error::failed_precondition("AV1 show-existing readiness is unavailable"))?;
		self.session.native_frame(slot, ready)
	}

	pub(crate) fn decode_vp9(
		&mut self,
		access_unit: &[u8],
		picture: &crate::video::Vp9Picture,
	) -> Result<Vec<u8>> {
		let device = self.engine.state.borrow().device.clone();
		self.session.upload_vp9_picture(access_unit, picture)?;
		let (command, slot) = self
			.session
			.record_vp9_picture_for_readback(&device, picture)?;
		let decode_event = {
			let mut state = self.engine.state.borrow_mut();
			submit_recorded(&mut state, command)?
		};
		decode_event.wait()?;
		self.session.verify_decode_result()?;
		let command = self.session.record_decode_readback(&device)?;
		let event = {
			let mut state = self.engine.state.borrow_mut();
			submit_recorded(&mut state, command)?
		};
		event.wait()?;
		let decoded = self.session.read_decode_yuv420()?;
		let index =
			usize::try_from(slot).map_err(|_| Error::internal("VP9 output slot exceeds usize"))?;
		*self
			.vp9_ready
			.get_mut(index)
			.ok_or_else(|| Error::internal("VP9 output slot exceeds readiness storage"))? =
			Some(decode_event);
		*self
			.vp9_host_frames
			.get_mut(index)
			.ok_or_else(|| Error::internal("VP9 output slot exceeds host-frame storage"))? =
			Some(decoded.clone());
		Ok(decoded)
	}

	pub(crate) fn decode_vp9_native(
		&mut self,
		access_unit: &[u8],
		picture: &crate::video::Vp9Picture,
	) -> Result<crate::runtime::NativeDecodedFrame> {
		let device = self.engine.state.borrow().device.clone();
		self.session.upload_vp9_picture(access_unit, picture)?;
		let (command, slot) = self.session.record_vp9_picture_native(&device, picture)?;
		let event = {
			let mut state = self.engine.state.borrow_mut();
			submit_recorded(&mut state, command)?
		};
		event.wait()?;
		self.session.verify_decode_result()?;
		let index =
			usize::try_from(slot).map_err(|_| Error::internal("VP9 output slot exceeds usize"))?;
		*self
			.vp9_ready
			.get_mut(index)
			.ok_or_else(|| Error::internal("VP9 output slot exceeds readiness storage"))? =
			Some(event.clone());
		self.session.native_frame(slot, event)
	}

	pub(crate) fn show_existing_vp9(
		&mut self,
		picture: &crate::video::Vp9Picture,
	) -> Result<Vec<u8>> {
		let slot = self.session.resolve_vp9_show_existing_slot(picture)?;
		let index =
			usize::try_from(slot).map_err(|_| Error::internal("VP9 show-existing slot exceeds usize"))?;
		self
			.vp9_host_frames
			.get(index)
			.and_then(Clone::clone)
			.ok_or_else(|| Error::failed_precondition("VP9 show-existing host frame is unavailable"))
	}

	pub(crate) fn show_existing_vp9_native(
		&mut self,
		picture: &crate::video::Vp9Picture,
	) -> Result<crate::runtime::NativeDecodedFrame> {
		let slot = self.session.resolve_vp9_show_existing_slot(picture)?;
		let index =
			usize::try_from(slot).map_err(|_| Error::internal("VP9 show-existing slot exceeds usize"))?;
		let ready = self
			.vp9_ready
			.get(index)
			.and_then(Clone::clone)
			.ok_or_else(|| Error::failed_precondition("VP9 show-existing readiness is unavailable"))?;
		self.session.native_frame(slot, ready)
	}

	pub(crate) fn read_native_yuv420(
		&mut self,
		frame: &crate::runtime::NativeDecodedFrame,
	) -> Result<Vec<u8>> {
		frame.ready().wait()?;
		let device = self.engine.state.borrow().device.clone();
		let (release, copy, acquire) = self.session.record_native_readback(&device, frame)?;
		{
			let mut state = self.engine.state.borrow_mut();
			let _release_event = submit_recorded(&mut state, release)?;
		}
		let copy_event = {
			let mut state = self.engine.state.borrow_mut();
			submit_recorded(&mut state, copy)?
		};
		let completion = if let Some(acquire) = acquire {
			let mut state = self.engine.state.borrow_mut();
			submit_recorded(&mut state, acquire)?
		} else {
			copy_event
		};
		frame.mark_consumed(&completion)?;
		completion.wait()?;
		self.session.read_decode_yuv420()
	}
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

/// Transactional ownership of one semantic operation's composite lowering.
///
/// Child operations retain executable work while their semantic identities are
/// suppressed. The outermost commit assigns every emitted node to the parent
/// contract; dropping an unfinished scope rolls back only its emitted work.
pub(crate) struct SemanticLoweringScope {
	engine: EngineHandle,
	active: bool,
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

	pub(crate) fn query_video_device_capabilities(
		&self,
	) -> Result<crate::video::VideoDeviceCapabilities> {
		self
			.handle
			.state
			.borrow()
			.device
			.video_device_capabilities()
	}

	pub(crate) fn query_video_decode_capabilities(
		&self,
		profile: crate::video::VideoDecodeProfile,
	) -> Result<crate::video::VideoDecodeCapabilities> {
		self
			.handle
			.state
			.borrow()
			.device
			.video_decode_capabilities(profile)
	}

	pub(crate) fn query_video_decode_formats(
		&self,
		profile: crate::video::VideoDecodeProfile,
	) -> Result<crate::video::VideoDecodeFormats> {
		self
			.handle
			.state
			.borrow()
			.device
			.video_decode_formats(profile)
	}

	pub(crate) fn query_video_encode_capabilities(
		&self,
		profile: crate::video::VideoEncodeProfile,
	) -> Result<crate::video::VideoEncodeCapabilities> {
		self
			.handle
			.state
			.borrow()
			.device
			.video_encode_capabilities(profile)
	}

	pub(crate) fn query_video_encode_formats(
		&self,
		profile: crate::video::VideoEncodeProfile,
	) -> Result<crate::video::VideoEncodeFormats> {
		self
			.handle
			.state
			.borrow()
			.device
			.video_encode_formats(profile)
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
	pub fn log(&self, level: LogLevel, component: LogComponent, text: impl AsRef<str>) -> Result<()> {
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
		self
			.state
			.borrow_mut()
			.session
			.begin_stable_resource_frame()
	}

	pub(crate) fn seal_all_stable_resources_external(&self) -> Result<()> {
		self
			.state
			.borrow_mut()
			.session
			.seal_all_stable_resources_external()
	}

	pub(crate) fn seal_stable_resource_inputs(&self) -> Result<()> {
		self
			.state
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
		self
			.state
			.borrow_mut()
			.session
			.release_stable_transient_resources(retired);
	}

	pub(crate) fn capture_active(&self) -> bool {
		self.state.borrow().capture_active
	}

	pub(crate) fn begin_semantic_lowering(&self) -> Result<SemanticLoweringScope> {
		self.state.borrow_mut().session.begin_semantic_lowering()?;
		Ok(SemanticLoweringScope {
			engine: self.clone(),
			active: true,
		})
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

	#[cfg(test)]
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

	pub(crate) fn record_optional_semantic(
		&self,
		dispatch: ComputeDispatch<'_>,
		semantic: OptionalSemanticDispatch<'_>,
	) -> Result<()> {
		let mut state = self.state.borrow_mut();
		let device = state.device.clone();
		state
			.session
			.record_optional_semantic(&device, dispatch, semantic)
	}

	pub(crate) fn record_fused_semantic(
		&self,
		dispatch: ComputeDispatch<'_>,
		semantics: &[SemanticDispatch<'_>],
	) -> Result<()> {
		let mut state = self.state.borrow_mut();
		let device = state.device.clone();
		state
			.session
			.record_fused_semantic(&device, dispatch, semantics)
	}

	pub(crate) fn record_split_semantic(
		&self,
		dispatches: &[ComputeDispatch<'_>],
		semantic: SemanticDispatch<'_>,
	) -> Result<()> {
		let mut state = self.state.borrow_mut();
		let device = state.device.clone();
		state
			.session
			.record_split_semantic(&device, dispatches, semantic)
	}

	pub(crate) fn record_split_optional_semantic(
		&self,
		dispatches: &[ComputeDispatch<'_>],
		semantic: OptionalSemanticDispatch<'_>,
	) -> Result<()> {
		let mut state = self.state.borrow_mut();
		let device = state.device.clone();
		state
			.session
			.record_split_optional_semantic(&device, dispatches, semantic)
	}

	pub(crate) fn record_audio_semantic(
		&self,
		dispatch: ComputeDispatch<'_>,
		semantic: AudioSemanticDispatch<'_>,
	) -> Result<()> {
		let mut state = self.state.borrow_mut();
		let device = state.device.clone();
		state
			.session
			.record_audio_semantic(&device, dispatch, semantic)
	}

	pub(crate) fn record_image_semantic(
		&self,
		dispatch: ComputeDispatch<'_>,
		semantic: ImageSemanticDispatch<'_>,
	) -> Result<()> {
		let mut state = self.state.borrow_mut();
		let device = state.device.clone();
		state
			.session
			.record_image_semantic(&device, dispatch, semantic)
	}

	pub(crate) fn record_audio_split_semantic(
		&self,
		dispatches: &[ComputeDispatch<'_>],
		semantic: AudioSemanticDispatch<'_>,
	) -> Result<()> {
		let mut state = self.state.borrow_mut();
		let device = state.device.clone();
		state
			.session
			.record_audio_split_semantic(&device, dispatches, semantic)
	}

	pub(crate) fn attach_semantic_autograd(
		&self,
		matrix_value: u64,
		sequence: u64,
	) -> Result<Option<(super::SemanticOpId, u64)>> {
		self
			.state
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
		self
			.state
			.borrow_mut()
			.session
			.complete_autograd(forward, sequence, generation, backward_first)
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

impl SemanticLoweringScope {
	/// Record one private physical dispatch owned by this composite lowering.
	pub(crate) fn record_physical(&self, dispatch: ComputeDispatch<'_>) -> Result<()> {
		let mut state = self.engine.state.borrow_mut();
		let device = state.device.clone();
		state.session.record_physical_lowering(&device, dispatch)
	}

	pub(crate) fn commit(mut self, semantic: SemanticDispatch<'_>) -> Result<()> {
		let result = self
			.engine
			.state
			.borrow_mut()
			.session
			.finish_semantic_lowering(semantic);
		self.active = false;
		result.map(|_| ())
	}
}

impl Drop for SemanticLoweringScope {
	fn drop(&mut self) {
		if !self.active {
			return;
		}
		self
			.engine
			.state
			.borrow_mut()
			.session
			.cancel_semantic_lowering();
	}
}

struct CaptureGuard {
	engine: EngineHandle,
	active: bool,
}

/// Movable private recording transaction used by stateful domain sessions.
pub(crate) struct RecordingTransaction {
	guard: Option<CaptureGuard>,
}

impl RecordingTransaction {
	pub(crate) fn begin(engine: EngineHandle) -> Result<Self> {
		Ok(Self {
			guard: Some(CaptureGuard::begin(engine)?),
		})
	}

	pub(crate) fn submit(mut self) -> Result<Event> {
		let guard = self.guard.take().expect("recording transaction guard");
		let engine = guard.engine.clone();
		let plan = guard.finish(&[])?;
		engine.submit_plan(&plan, false)
	}
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
			state.session.begin_capture()?;
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
					state.session.abort_capture();
					state.capture_active = false;
					self.active = false;
					return Err(error);
				}
			};
			state.session.end_capture();
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
			state.session.end_capture();
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
		state.session.abort_capture();
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
	#[ignore = "requires a hardware Vulkan Video decode queue"]
	fn submits_and_retires_an_empty_video_decode_command_buffer() -> crate::Result<()> {
		let engine = Engine::new()?;
		let event = {
			let mut state = engine.handle.state.borrow_mut();
			let command = state.device.record_video_decode_empty()?;
			super::submit_recorded(&mut state, command)?
		};
		event.wait()
	}

	#[test]
	#[ignore = "requires a hardware Vulkan Video decode profile"]
	fn creates_binds_and_destroys_advertised_video_decode_sessions() -> crate::Result<()> {
		let engine = Engine::new()?;
		let device = engine.handle.state.borrow().device.clone();
		let advertised = device.video_device_capabilities()?;
		let mut profiles = Vec::new();
		if advertised.supports_h264_decode() {
			profiles.push(crate::video::VideoDecodeProfile::h264_420_8bit(
				crate::video::H264Profile::High,
			));
		}
		if advertised.supports_h265_decode() {
			profiles.push(crate::video::VideoDecodeProfile::h265_420(
				crate::video::H265Profile::Main,
				crate::video::VideoComponentBitDepth::Eight,
			));
		}
		if advertised.supports_av1_decode() {
			profiles.push(crate::video::VideoDecodeProfile::av1_420(
				crate::video::Av1Profile::Main,
				crate::video::VideoComponentBitDepth::Eight,
				false,
			));
		}
		assert!(!profiles.is_empty());
		for profile in profiles {
			let capabilities = device.video_decode_capabilities(profile)?;
			let dpb_slots = capabilities.max_dpb_slots().min(4);
			let active_references = capabilities.max_active_reference_pictures().min(dpb_slots);
			let session = device.create_video_decode_session(
				profile,
				capabilities.min_coded_extent(),
				dpb_slots,
				active_references,
			)?;
			assert!(session.memory_binding_count() <= 64);
			let expected_images = if capabilities.dpb_and_output_coincide() {
				if capabilities.separate_reference_images() {
					dpb_slots as usize
				} else {
					1
				}
			} else {
				2
			};
			assert_eq!(session.image_count(), expected_images);
			drop(session);
		}
		Ok(())
	}

	#[test]
	#[ignore = "requires a hardware Vulkan AV1 profile and OA donor fixture"]
	fn submits_first_av1_picture_from_donor_sequence() -> crate::Result<()> {
		let engine = Engine::new()?;
		let device = engine.handle.state.borrow().device.clone();
		let fixture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
			.with_file_name("oa")
			.join("sdk/asset/video/clip/shibuya_720p_30fps_av1_main_8bit_420.mp4");
		let mut demuxer = crate::video::VideoDemuxer::open(fixture)?;
		let packet = demuxer
			.read_next_packet()?
			.expect("AV1 fixture must contain a first packet");
		let sequence_obu = crate::video::parse_av1_obus(packet.data())?
			.into_iter()
			.find(|obu| obu.type_() == crate::video::Av1ObuType::SequenceHeader)
			.expect("demuxed AV1 keyframe must include a sequence header");
		let sequence = crate::video::parse_av1_sequence_header(sequence_obu.payload())?;
		let profile = crate::video::VideoDecodeProfile::Av1 {
			profile: sequence.profile,
			film_grain_support: sequence.film_grain_params_present,
			chroma_subsampling: sequence.color.chroma_subsampling,
			luma_bit_depth: sequence.color.bit_depth,
			chroma_bit_depth: sequence.color.bit_depth,
		};
		let capabilities = device.video_decode_capabilities(profile)?;
		let dpb_slots = capabilities.max_dpb_slots().min(8);
		let active_references = capabilities.max_active_reference_pictures().min(dpb_slots);
		let mut session = device.create_video_decode_session(
			profile,
			crate::video::VideoExtent {
				width: sequence.coded_width(),
				height: sequence.coded_height(),
			},
			dpb_slots,
			active_references,
		)?;
		session.set_av1_parameters(&sequence)?;
		assert!(session.parameters_ready());
		let mut references = crate::video::Av1ReferenceState::default();
		let mut parsed_frame = None;
		for obu in crate::video::parse_av1_obus(packet.data())? {
			if matches!(
				obu.type_(),
				crate::video::Av1ObuType::Frame | crate::video::Av1ObuType::FrameHeader
			) {
				let frame = crate::video::parse_av1_frame_header(obu.payload(), &sequence, &references)?;
				if !frame.show_existing_frame {
					let tiles = crate::video::parse_av1_tile_group(obu, &frame)?;
					parsed_frame = Some((
						frame,
						u32::try_from(obu.header_offset())
							.map_err(|_| crate::Error::out_of_range("AV1 frame-header offset exceeds u32"))?,
						tiles,
					));
					break;
				}
				references.refresh(&frame);
			}
		}
		let (frame, frame_header_offset, tiles) =
			parsed_frame.expect("first donor packet must contain a coded AV1 picture");
		assert_eq!(
			session.upload_av1_access_unit(packet.data())?,
			packet.data().len()
		);
		let (command, _) = session.record_av1_picture_for_readback(
			&device,
			&sequence,
			&frame,
			frame_header_offset,
			&tiles,
		)?;
		let event = {
			let mut state = engine.handle.state.borrow_mut();
			super::submit_recorded(&mut state, command)?
		};
		event.wait()?;
		session.verify_decode_result()?;
		let readback = session.record_decode_readback(&device)?;
		let event = {
			let mut state = engine.handle.state.borrow_mut();
			super::submit_recorded(&mut state, readback)?
		};
		event.wait()?;
		let decoded = session.read_decode_yuv420()?;
		assert_eq!(decoded.len(), 1280_usize * 720 + 2 * 640 * 360);
		assert_eq!(
			crate::cryptography::hash(&decoded).to_hex(),
			"beb7230253bfa12918347b1e01c0b7ee3f9687ec66d71d58d25922e54bccb66d",
			"Vulkan AV1 output differs from the independently decoded FFmpeg frame"
		);
		Ok(())
	}

	#[test]
	#[ignore = "requires a hardware Vulkan H.265 profile and OA donor fixture"]
	fn submits_first_h265_idr_decode_commands() -> crate::Result<()> {
		let engine = Engine::new()?;
		let device = engine.handle.state.borrow().device.clone();
		let fixture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
			.with_file_name("oa")
			.join("sdk/asset/video/clip/shibuya_720p_30fps_h265_main_8bit_420.mp4");
		let mut demuxer = crate::video::VideoDemuxer::open(fixture)?;
		let packet = demuxer
			.read_next_packet()?
			.expect("H.265 fixture must contain a first packet");
		let nals = crate::video::parse_nal_annex_b(packet.data());
		let find = |nal_type| {
			nals
				.iter()
				.find(|nal| (nal.payload()[0] >> 1) & 0x3f == nal_type)
				.map(|nal| nal.payload())
				.expect("demuxed keyframe must include the requested H.265 parameter set")
		};
		let vps = crate::video::parse_h265_vps(find(32))?;
		let sps = crate::video::parse_h265_sps(find(33))?;
		let pps = crate::video::parse_h265_pps(find(34))?;
		let mut coded_slices = nals
			.iter()
			.filter(|nal| ((nal.payload()[0] >> 1) & 0x3f) < 32);
		let slice_nal = coded_slices
			.next()
			.expect("demuxed keyframe must include one H.265 coded slice segment");
		assert!(
			coded_slices.next().is_none(),
			"first qualification path accepts one H.265 coded slice segment"
		);
		let slice = crate::video::parse_h265_slice_header(slice_nal.payload(), &sps, &pps)?;
		let profile = crate::video::VideoDecodeProfile::h265_420(
			crate::video::H265Profile::Main,
			crate::video::VideoComponentBitDepth::Eight,
		);
		let capabilities = device.video_decode_capabilities(profile)?;
		let dpb_slots = capabilities.max_dpb_slots().min(4);
		let active_references = capabilities.max_active_reference_pictures().min(dpb_slots);
		let mut session = device.create_video_decode_session(
			profile,
			crate::video::VideoExtent {
				width: sps.coded_width,
				height: sps.coded_height,
			},
			dpb_slots,
			active_references,
		)?;
		let packed_len = session.upload_first_h265_access_unit(packet.data())?;
		let (payload_len, range) = session
			.bitstream_upload()
			.expect("uploaded H.265 packet must retain its decode buffer");
		assert_eq!(payload_len, packed_len);
		assert_eq!(payload_len, slice_nal.payload().len() + 3);
		assert!(range >= payload_len as u64);
		assert!(range.is_multiple_of(capabilities.min_bitstream_size_alignment()));
		session.set_h265_parameters(&vps, &sps, &pps)?;
		assert!(session.parameters_ready());
		let command = session.record_first_h265_idr(&device, &sps, &pps, &slice)?;
		assert!(session.first_decode_recorded());
		let event = {
			let mut state = engine.handle.state.borrow_mut();
			super::submit_recorded(&mut state, command)?
		};
		event.wait()?;
		session.verify_first_decode_result()?;
		let readback_command = session.record_first_decode_readback(&device)?;
		let readback_event = {
			let mut state = engine.handle.state.borrow_mut();
			super::submit_recorded(&mut state, readback_command)?
		};
		readback_event.wait()?;
		let decoded = session.read_first_decode_yuv420()?;
		assert_eq!(decoded.len(), 1280 * 720 * 3 / 2);
		assert_eq!(
			crate::cryptography::hash(&decoded).to_hex(),
			"eb1e3c9d708df539d31382a0329653b59058cb538c78d85b444e1b372a80d512",
			"Vulkan H.265 output differs from the independently decoded YUV420 frame"
		);
		Ok(())
	}

	#[test]
	#[ignore = "requires Vulkan H.265 hardware, OA donor fixture, and FFmpeg"]
	fn submits_complete_h265_stream_with_references() -> crate::Result<()> {
		let engine = Engine::new()?;
		let device = engine.handle.state.borrow().device.clone();
		let fixture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
			.with_file_name("oa")
			.join("sdk/asset/video/clip/shibuya_720p_30fps_h265_main_8bit_420.mp4");
		let mut demuxer = crate::video::VideoDemuxer::open(&fixture)?;
		let expected_pictures = demuxer.info().sample_count();
		let first_packet = demuxer
			.read_next_packet()?
			.expect("H.265 fixture must contain a first packet");
		let first_nals = crate::video::parse_nal_annex_b(first_packet.data());
		let find = |nal_type| {
			first_nals
				.iter()
				.find(|nal| (nal.payload()[0] >> 1) & 0x3f == nal_type)
				.map(|nal| nal.payload())
				.expect("demuxed keyframe must include the requested H.265 parameter set")
		};
		let vps = crate::video::parse_h265_vps(find(32))?;
		let sps = crate::video::parse_h265_sps(find(33))?;
		let pps = crate::video::parse_h265_pps(find(34))?;
		let profile = crate::video::VideoDecodeProfile::h265_420(
			crate::video::H265Profile::Main,
			crate::video::VideoComponentBitDepth::Eight,
		);
		let capabilities = device.video_decode_capabilities(profile)?;
		let required_dpb_slots = sps
			.max_decoded_picture_buffering_minus_1
			.first()
			.copied()
			.unwrap_or(0)
			.checked_add(1)
			.ok_or_else(|| crate::Error::data_loss("H.265 SPS DPB slot count overflows"))?;
		if required_dpb_slots > capabilities.max_dpb_slots() {
			return Err(crate::Error::missing_capability(
				"device cannot admit the donor H.265 DPB slot count",
			));
		}
		let active_references = capabilities
			.max_active_reference_pictures()
			.min(required_dpb_slots);
		let mut session = device.create_video_decode_session(
			profile,
			crate::video::VideoExtent {
				width: sps.coded_width,
				height: sps.coded_height,
			},
			required_dpb_slots,
			active_references,
		)?;
		session.set_h265_parameters(&vps, &sps, &pps)?;
		let mut picture_count = 0_u32;
		let mut decoded_by_poc = std::collections::BTreeMap::new();
		let mut decode_packet = |packet: &crate::video::VideoPacket| -> crate::Result<()> {
			let nals = crate::video::parse_nal_annex_b(packet.data());
			let mut coded = nals
				.iter()
				.filter(|nal| ((nal.payload()[0] >> 1) & 0x3f) < 32);
			let slice_nal = coded
				.next()
				.expect("every donor packet must contain one H.265 coded slice");
			assert!(
				coded.next().is_none(),
				"reusable qualification accepts one H.265 slice per picture"
			);
			let slice = crate::video::parse_h265_slice_header(slice_nal.payload(), &sps, &pps)?;
			let picture_order_count = i32::try_from(slice.picture_order_count_lsb.unwrap_or(0))
				.map_err(|_| crate::Error::data_loss("H.265 donor POC exceeds i32"))?;
			session.upload_first_h265_access_unit(packet.data())?;
			let command = session.record_h265_picture_for_readback(&device, &sps, &pps, &slice)?;
			let event = {
				let mut state = engine.handle.state.borrow_mut();
				super::submit_recorded(&mut state, command)?
			};
			event.wait()?;
			session.verify_first_decode_result()?;
			let readback_command = session.record_decode_readback(&device)?;
			let readback_event = {
				let mut state = engine.handle.state.borrow_mut();
				super::submit_recorded(&mut state, readback_command)?
			};
			readback_event.wait()?;
			let decoded = session.read_first_decode_yuv420()?;
			if decoded_by_poc
				.insert(picture_order_count, decoded)
				.is_some()
			{
				return Err(crate::Error::data_loss(
					"H.265 donor produced a duplicate picture-order count",
				));
			}
			picture_count = picture_count
				.checked_add(1)
				.expect("fixture picture count must fit u32");
			Ok(())
		};
		decode_packet(&first_packet)?;
		while let Some(packet) = demuxer.read_next_packet()? {
			decode_packet(&packet)?;
		}
		assert_eq!(picture_count, expected_pictures);
		assert_eq!(decoded_by_poc.len(), picture_count as usize);
		let reference = std::process::Command::new("ffmpeg")
			.args(["-v", "error", "-i"])
			.arg(&fixture)
			.args([
				"-map", "0:v:0", "-pix_fmt", "yuv420p", "-f", "rawvideo", "-",
			])
			.output()
			.map_err(|source| crate::Error::backend_failure("FFmpeg", "HEVC oracle", source))?;
		if !reference.status.success() {
			return Err(crate::Error::data_loss(format!(
				"FFmpeg HEVC oracle failed: {}",
				String::from_utf8_lossy(&reference.stderr)
			)));
		}
		let frame_size = usize::try_from(sps.coded_width)
			.ok()
			.and_then(|width| {
				usize::try_from(sps.coded_height)
					.ok()
					.and_then(|height| width.checked_mul(height))
			})
			.and_then(|luma| luma.checked_mul(3))
			.and_then(|size| size.checked_div(2))
			.ok_or_else(|| crate::Error::out_of_range("H.265 oracle frame size overflows"))?;
		let expected_size = frame_size
			.checked_mul(picture_count as usize)
			.ok_or_else(|| crate::Error::out_of_range("H.265 oracle stream size overflows"))?;
		assert_eq!(reference.stdout.len(), expected_size);
		for ((picture_order_count, decoded), expected) in decoded_by_poc
			.iter()
			.zip(reference.stdout.chunks_exact(frame_size))
		{
			assert_eq!(
				decoded, expected,
				"Vulkan H.265 output differs from FFmpeg at POC {picture_order_count}"
			);
		}
		Ok(())
	}

	#[test]
	#[ignore = "requires a hardware Vulkan H.264 profile and OA donor fixture"]
	fn submits_first_h264_idr_decode_commands() -> crate::Result<()> {
		let engine = Engine::new()?;
		let device = engine.handle.state.borrow().device.clone();
		let fixture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
			.with_file_name("oa")
			.join("sdk/asset/video/clip/shibuya_720p_30fps_h264_high_8bit_420.mp4");
		let mut demuxer = crate::video::VideoDemuxer::open(fixture)?;
		let packet = demuxer
			.read_next_packet()?
			.expect("H.264 fixture must contain a first packet");
		let nals = crate::video::parse_nal_annex_b(packet.data());
		let sps_nal = nals
			.iter()
			.find(|nal| nal.payload()[0] & 0x1f == 7)
			.expect("demuxed keyframe must include SPS");
		let pps_nal = nals
			.iter()
			.find(|nal| nal.payload()[0] & 0x1f == 8)
			.expect("demuxed keyframe must include PPS");
		let sps = crate::video::parse_h264_sps(sps_nal.payload())?;
		let pps = crate::video::parse_h264_pps(pps_nal.payload(), &sps)?;
		let mut coded_slices = nals
			.iter()
			.filter(|nal| matches!(nal.payload()[0] & 0x1f, 1 | 5));
		let slice_nal = coded_slices
			.next()
			.expect("demuxed keyframe must include one coded slice");
		assert!(
			coded_slices.next().is_none(),
			"first qualification path accepts one coded slice"
		);
		let slice = crate::video::parse_h264_slice_header(slice_nal.payload(), &sps, &pps)?;
		let profile = crate::video::VideoDecodeProfile::h264_420_8bit(crate::video::H264Profile::High);
		let capabilities = device.video_decode_capabilities(profile)?;
		let dpb_slots = capabilities.max_dpb_slots().min(4);
		let active_references = capabilities.max_active_reference_pictures().min(dpb_slots);
		let mut session = device.create_video_decode_session(
			profile,
			crate::video::VideoExtent {
				width: sps.coded_width()?,
				height: sps.coded_height()?,
			},
			dpb_slots,
			active_references,
		)?;
		let packed_len = session.upload_h264_access_unit(packet.data())?;
		let (payload_len, range) = session
			.bitstream_upload()
			.expect("uploaded packet must retain its decode buffer");
		assert_eq!(payload_len, packed_len);
		assert_eq!(payload_len, slice_nal.payload().len() + 3);
		assert!(range >= payload_len as u64);
		assert!(range.is_multiple_of(capabilities.min_bitstream_size_alignment()));
		session.set_h264_parameters(&sps, &pps)?;
		assert!(session.parameters_ready());
		let command = session.record_first_h264_idr(&device, &sps, &pps, &slice)?;
		assert!(session.first_decode_recorded());
		let event = {
			let mut state = engine.handle.state.borrow_mut();
			super::submit_recorded(&mut state, command)?
		};
		event.wait()?;
		session.verify_first_decode_result()?;
		let readback_command = session.record_first_decode_readback(&device)?;
		let readback_event = {
			let mut state = engine.handle.state.borrow_mut();
			super::submit_recorded(&mut state, readback_command)?
		};
		readback_event.wait()?;
		let decoded = session.read_first_decode_yuv420()?;
		assert_eq!(decoded.len(), 1280 * 720 * 3 / 2);
		assert_eq!(
			crate::cryptography::hash(&decoded).to_hex(),
			"360d151e07a39eac2314dd527bf7ca1802e5ed3b980e42409345b6668736e8fd",
			"Vulkan H.264 output differs from the independently decoded YUV420 frame"
		);
		Ok(())
	}

	#[test]
	#[ignore = "requires Vulkan H.264 hardware, OA donor fixture, and FFmpeg"]
	fn submits_complete_h264_stream_with_references() -> crate::Result<()> {
		let engine = Engine::new()?;
		let device = engine.handle.state.borrow().device.clone();
		let fixture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
			.with_file_name("oa")
			.join("sdk/asset/video/clip/shibuya_720p_30fps_h264_high_8bit_420.mp4");
		let mut demuxer = crate::video::VideoDemuxer::open(&fixture)?;
		let expected_pictures = demuxer.info().sample_count();
		let first_packet = demuxer
			.read_next_packet()?
			.expect("H.264 fixture must contain a first packet");
		let first_nals = crate::video::parse_nal_annex_b(first_packet.data());
		let sps_nal = first_nals
			.iter()
			.find(|nal| nal.nal_unit_type() == 7)
			.expect("initial AVC packet must contain SPS");
		let pps_nal = first_nals
			.iter()
			.find(|nal| nal.nal_unit_type() == 8)
			.expect("initial AVC packet must contain PPS");
		let sps = crate::video::parse_h264_sps(sps_nal.payload())?;
		let pps = crate::video::parse_h264_pps(pps_nal.payload(), &sps)?;
		let profile = crate::video::VideoDecodeProfile::h264_420_8bit(crate::video::H264Profile::High);
		let capabilities = device.video_decode_capabilities(profile)?;
		let required_dpb_slots = sps
			.max_num_ref_frames
			.checked_add(1)
			.ok_or_else(|| crate::Error::data_loss("H.264 SPS DPB slot count overflows"))?
			.min(16);
		if required_dpb_slots > capabilities.max_dpb_slots()
			|| sps.max_num_ref_frames > capabilities.max_active_reference_pictures()
		{
			return Err(crate::Error::missing_capability(
				"device cannot admit the donor H.264 DPB requirements",
			));
		}
		let mut session = device.create_video_decode_session(
			profile,
			crate::video::VideoExtent {
				width: sps.coded_width()?,
				height: sps.coded_height()?,
			},
			required_dpb_slots,
			sps.max_num_ref_frames,
		)?;
		session.set_h264_parameters(&sps, &pps)?;
		let mut picture_count = 0_u32;
		let mut decoded_by_pts = std::collections::BTreeMap::new();
		let mut decode_packet = |packet: &crate::video::VideoPacket| -> crate::Result<()> {
			let nals = crate::video::parse_nal_annex_b(packet.data());
			let mut coded = nals
				.iter()
				.filter(|nal| matches!(nal.nal_unit_type(), 1 | 5));
			let slice_nal = coded
				.next()
				.expect("every donor packet must contain one H.264 coded slice");
			assert!(
				coded.next().is_none(),
				"reusable qualification accepts one H.264 slice per picture"
			);
			let slice = crate::video::parse_h264_slice_header(slice_nal.payload(), &sps, &pps)?;
			session.upload_h264_access_unit(packet.data())?;
			let command = session.record_h264_picture_for_readback(&device, &sps, &pps, &slice)?;
			let event = {
				let mut state = engine.handle.state.borrow_mut();
				super::submit_recorded(&mut state, command)?
			};
			event.wait()?;
			session.verify_first_decode_result()?;
			let readback_command = session.record_decode_readback(&device)?;
			let readback_event = {
				let mut state = engine.handle.state.borrow_mut();
				super::submit_recorded(&mut state, readback_command)?
			};
			readback_event.wait()?;
			let decoded = session.read_first_decode_yuv420()?;
			if decoded_by_pts
				.insert(packet.presentation_timestamp(), decoded)
				.is_some()
			{
				return Err(crate::Error::data_loss(
					"H.264 donor produced a duplicate presentation timestamp",
				));
			}
			picture_count = picture_count
				.checked_add(1)
				.expect("fixture picture count must fit u32");
			Ok(())
		};
		decode_packet(&first_packet)?;
		while let Some(packet) = demuxer.read_next_packet()? {
			decode_packet(&packet)?;
		}
		assert_eq!(picture_count, expected_pictures);
		assert_eq!(decoded_by_pts.len(), picture_count as usize);
		let reference = std::process::Command::new("ffmpeg")
			.args(["-v", "error", "-i"])
			.arg(&fixture)
			.args([
				"-map", "0:v:0", "-pix_fmt", "yuv420p", "-f", "rawvideo", "-",
			])
			.output()
			.map_err(|source| crate::Error::backend_failure("FFmpeg", "AVC oracle", source))?;
		if !reference.status.success() {
			return Err(crate::Error::data_loss(format!(
				"FFmpeg AVC oracle failed: {}",
				String::from_utf8_lossy(&reference.stderr)
			)));
		}
		let coded_width = sps.coded_width()?;
		let coded_height = sps.coded_height()?;
		let frame_size = usize::try_from(coded_width)
			.ok()
			.and_then(|width| {
				usize::try_from(coded_height)
					.ok()
					.and_then(|height| width.checked_mul(height))
			})
			.and_then(|luma| luma.checked_mul(3))
			.and_then(|size| size.checked_div(2))
			.ok_or_else(|| crate::Error::out_of_range("H.264 oracle frame size overflows"))?;
		let expected_size = frame_size
			.checked_mul(picture_count as usize)
			.ok_or_else(|| crate::Error::out_of_range("H.264 oracle stream size overflows"))?;
		assert_eq!(reference.stdout.len(), expected_size);
		for ((presentation_timestamp, decoded), expected) in decoded_by_pts
			.iter()
			.zip(reference.stdout.chunks_exact(frame_size))
		{
			assert_eq!(
				decoded, expected,
				"Vulkan H.264 output differs from FFmpeg at PTS {presentation_timestamp}"
			);
		}
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
			plan
				.captured_resources()
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
			"training program operation ml.adamw.f32 embeds host-stepped optimizer state; use a replay-state kernel"
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
				kernel: KernelId::MatrixAddF32,
				buffers: &first_buffers,
				push_constants: &element_count,
				workgroups: [1, 1, 1],
			},
			ComputeDispatch {
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

	#[test]
	#[ignore = "requires a hardware Vulkan 1.3 compute device"]
	fn abandoned_semantic_lowering_rolls_back_only_its_work() -> crate::Result<()> {
		let engine = Engine::new()?;
		let left = Matrix::from_f32(&engine, [2], &[1.0, 2.0])?;
		let right = Matrix::from_f32(&engine, [2], &[3.0, 4.0])?;
		{
			let _lowering = left.engine_handle().begin_semantic_lowering()?;
			let _abandoned = matrix::add(&left, &right)?;
			assert!(engine.has_pending_work());
		}
		assert!(!engine.has_pending_work());
		assert_eq!(left.read_f32()?, [1.0, 2.0]);
		Ok(())
	}
}
