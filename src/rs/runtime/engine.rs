use std::sync::Arc;

use std::{cell::RefCell, marker::PhantomData, rc::Rc};

use vk_mem::Alloc;

use crate::{Error, LogComponent, LogLevel, LogOptions, Matrix, Result};

use super::{
	AudioSemanticDispatch, Buffer, ComputeDispatch, DecodeSession, Device, DevicePhysical, Event,
	ExecutionPlan, FlatColorPipeline, GraphicsPipeline, ImageSemanticDispatch, Instance, MeshBuffer,
	NativeRgbaImage, OptionalSemanticDispatch, PresentationSurface, RecordedCommandBuffer,
	RenderTarget, RetirementService, SceneDrawCmd, SecretBinding, SecretBuffer, SecretErasure,
	SemanticDispatch, StandardSurfaceBlendPipeline, StandardSurfaceDescriptorArena,
	StandardSurfaceDescriptorLayouts, StandardSurfacePipeline, Storage, UnlitDescriptorArena,
	UnlitDescriptorLayouts, UnlitPipeline,
	log::{LogSelection, Logger},
	session::ExecutionSession,
};

/// Policy for selecting the local device owned by an engine.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DeviceSelection {
	/// Select a compatible hardware device by queried resource capacity.
	#[default]
	Automatic,
	/// Select an exact physical-device ordinal, including software Vulkan CPU.
	/// Use [`Engine::available_devices`] to discover current ordinals.
	Index(usize),
}

/// Optional services that the selected device must provide.
///
/// Compute is always required. These predicates constrain physical-device
/// admission before the logical device is created; they do not expose kernel
/// selection or vendor policy. Presentation readiness proves a graphics queue
/// and swapchain support for an instance created with surface extensions. The
/// eventual [`Presenter`](super::Presenter) still verifies the concrete
/// window surface.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EngineRequirements {
	pub(crate) graphics: bool,
	pub(crate) presentation: bool,
	pub(crate) video_decode: Vec<crate::video::VideoDecodeProfile>,
	pub(crate) video_encode: Vec<crate::video::VideoEncodeProfile>,
}

impl EngineRequirements {
	/// Require a graphics-capable queue family.
	pub fn graphics(mut self) -> Self {
		self.graphics = true;
		self
	}

	/// Require graphics and swapchain readiness.
	///
	/// The builder must also receive the window's required instance extensions.
	pub fn presentation(mut self) -> Self {
		self.graphics = true;
		self.presentation = true;
		self
	}

	/// Require support for one exact Vulkan Video decode profile.
	pub fn video_decode(mut self, profile: crate::video::VideoDecodeProfile) -> Self {
		if !self.video_decode.contains(&profile) {
			self.video_decode.push(profile);
		}
		self
	}

	/// Require device support for one exact Vulkan Video encode profile.
	///
	/// This checks Vulkan capability only. OA encoder-session readiness is a
	/// separate runtime capability.
	pub fn video_encode(mut self, profile: crate::video::VideoEncodeProfile) -> Self {
		if !self.video_encode.contains(&profile) {
			self.video_encode.push(profile);
		}
		self
	}
}

/// Configuration used to construct one OA execution engine.
#[derive(Clone, Debug, Default)]
#[must_use]
pub struct EngineBuilder {
	device_selection: DeviceSelection,
	requirements: EngineRequirements,
	log_options: LogOptions,
	instance_extensions: Vec<String>,
}

impl EngineBuilder {
	/// Create a builder using automatic hardware-device selection.
	pub fn new() -> Self {
		Self {
			device_selection: DeviceSelection::Automatic,
			requirements: EngineRequirements::default(),
			log_options: LogOptions::new(),
			instance_extensions: Vec::new(),
		}
	}

	/// Require optional device services during physical-device admission.
	pub fn requirements(mut self, requirements: EngineRequirements) -> Self {
		self.requirements = requirements;
		self
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

	/// Require caller-selected Vulkan instance extensions.
	///
	/// This is the explicit construction seam required by a future WSI Presenter.
	/// Headless callers leave it empty. Extension names are validated before the
	/// Vulkan instance is created; no raw instance handle crosses this boundary.
	pub fn instance_extensions(
		mut self,
		extensions: impl IntoIterator<Item = impl Into<String>>,
	) -> Self {
		self.instance_extensions = extensions.into_iter().map(Into::into).collect();
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
		Engine::build(
			self.device_selection,
			self.requirements,
			self.log_options,
			self.instance_extensions,
		)
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

thread_local! {
	static DEFAULT_ENGINE: RefCell<Option<Engine>> = const { RefCell::new(None) };
}

pub(crate) enum CaptureAttempt<T> {
	Captured { plan: Box<ExecutionPlan>, output: T },
	Rejected { error: Error, output: T },
}

/// Runtime-owned implementation for public hardware video decoder sessions.
pub(crate) struct VideoDecoderBackend {
	engine: EngineHandle,
	session: DecodeSession,
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

/// Engine-tracked compose image allocation — cleaned up on `destroy_compose_image`.
struct ComposeImageEntry {
	image: ash::vk::Image,
	view: ash::vk::ImageView,
	alloc: vk_mem::Allocation,
}

struct EngineState {
	// Fields drop in declaration order. Retirement disconnects first and joins only
	// an already-drained host worker; an active worker retains the device and
	// transitive instance ownership until every submitted epoch completes.
	retirement: RetirementService,
	session: ExecutionSession,
	device: Arc<Device>,
	next_epoch: u64,
	capture_active: bool,
	/// Live compose images allocated for UI layers.
	compose_images: RefCell<Vec<ComposeImageEntry>>,
}

/// Thread-affine shared access to the execution owner retained by its values.
#[derive(Clone)]
pub(crate) struct EngineHandle {
	state: Rc<RefCell<EngineState>>,
}

/// Unique private value; only in-flight commands may share its storage lifetime.
/// No Clone, Debug, mapping, export, Matrix conversion or implicit erasure.
#[cfg_attr(
	not(test),
	expect(dead_code, reason = "private ML-KEM retained-key integration")
)]
pub(super) struct MlKemDeviceKey {
	engine: EngineHandle,
	private: Arc<SecretBuffer>,
	public: Buffer,
	k: u32,
	ready: Event,
}

/// Unique private value; only in-flight commands may share its storage lifetime.
/// No Clone, Debug, mapping, export, Matrix conversion or implicit erasure.
#[cfg_attr(
	not(test),
	expect(dead_code, reason = "private ML-DSA retained-key integration")
)]
pub(super) struct MlDsaDeviceKey {
	engine: EngineHandle,
	private: Arc<SecretBuffer>,
	public: Buffer,
	parameter: u32,
	ready: Event,
}

/// Unique private shared-secret value. Readiness retains its originating Engine;
/// command references retain storage only, with no mapping or implicit erasure.
#[cfg_attr(
	not(test),
	expect(dead_code, reason = "private ML-KEM shared-secret integration")
)]
pub(super) struct MlKemSharedSecret {
	engine: EngineHandle,
	private: Arc<SecretBuffer>,
	ready: Event,
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
	/// List every Vulkan device, including software CPU implementations.
	///
	/// Creates a temporary instance but no logical device, allocator or pipelines.
	/// A listed device may still fail OA capability admission. Ordinals are scoped
	/// to the current loader and ICD configuration, not stable across machines.
	///
	/// # Errors
	/// Returns backend initialization or enumeration errors.
	pub fn available_devices() -> Result<Vec<super::DeviceSummary>> {
		let instance = Instance::new(&[])?;
		DevicePhysical::enumerate(&instance)
	}

	/// Return queried device facts and the allocated descriptor capacities.
	pub fn device_info(&self) -> crate::DeviceInfo {
		self.handle.device_info()
	}

	/// Begin explicit engine configuration.
	pub fn builder() -> EngineBuilder {
		EngineBuilder::new()
	}

	/// Create an engine using one compute-capable Vulkan device.
	///
	/// # Errors
	///
	/// Returns an error when Vulkan cannot be loaded, neither the Strict nor
	/// Compatibility profile is available, no compatible hardware device exists,
	/// or logical-device creation fails.
	pub fn new() -> Result<Self> {
		Self::builder().build()
	}

	/// Initialize this thread's default compute engine before its first use.
	///
	/// Initialization is lazy and attempted again after a failure. The selected
	/// engine remains thread-affine and is reused by later default operations.
	///
	/// # Errors
	///
	/// Returns the same backend or device-selection error as [`Engine::new`].
	pub fn init_default() -> Result<()> {
		DEFAULT_ENGINE.with(|slot| {
			if slot.borrow().is_none() {
				let engine = Self::new()?;
				*slot.borrow_mut() = Some(engine);
			}
			Ok(())
		})
	}

	/// Access this thread's lazily initialized default compute engine.
	///
	/// # Errors
	///
	/// Returns an engine initialization error or an error from `use_engine`.
	pub fn with_default<T>(use_engine: impl FnOnce(&Self) -> Result<T>) -> Result<T> {
		Self::init_default()?;
		DEFAULT_ENGINE.with(|slot| {
			let selected = slot.borrow();
			let engine = selected.as_ref().expect("default engine initialized above");
			use_engine(engine)
		})
	}

	/// Return whether this Engine has an executable graphics path.
	///
	/// A physical graphics queue alone does not qualify the Compatibility
	/// shader/binding path. Presentation also requires WSI support and
	/// per-surface present-support negotiation.
	pub fn has_graphics(&self) -> bool {
		self.handle.state.borrow().device.has_graphics_queue()
	}

	/// Create one SDL3 Vulkan surface borrowing this Engine and window.
	pub fn create_presentation_surface<'window>(
		&self,
		window: &'window super::SdlWindow,
	) -> Result<super::PresentationSurface<'window>> {
		let state = self.handle.state.borrow();
		PresentationSurface::create(&state.device, window)
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

	pub(super) fn build(
		selection: DeviceSelection,
		requirements: EngineRequirements,
		log_options: LogOptions,
		instance_extensions: Vec<String>,
	) -> Result<Self> {
		let logger = Logger::new(log_options)?;
		let log_selection = logger.select();
		let instance = Instance::new(&instance_extensions)?;
		if requirements.presentation && !instance.surface_enabled() {
			return Err(Error::invalid_argument(
				"presentation readiness requires Vulkan surface instance extensions",
			));
		}
		let physical_device = DevicePhysical::select(&instance, selection, &requirements)?;
		let device = Device::new(&instance, physical_device)?;
		let retirement = RetirementService::new(&device);
		device.log_identity();

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
					compose_images: RefCell::new(Vec::new()),
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

impl EngineHandle {
	pub(crate) fn device_info(&self) -> crate::DeviceInfo {
		self.state.borrow().device.device_info()
	}

	pub(crate) fn owns_event(&self, event: &Event) -> bool {
		event.comes_from(&self.state.borrow().device)
	}

	pub(crate) fn upload_native_rgba_image(
		&self,
		storage: &Storage,
		width: u32,
		height: u32,
	) -> Result<Rc<NativeRgbaImage>> {
		self.checkpoint(false)?.wait()?;
		let mut state = self.state.borrow_mut();
		let image = Rc::new(NativeRgbaImage::new(&state.device, width, height)?);
		let buffer = storage
			.buffer()
			.ok_or_else(|| {
				Error::failed_precondition("native Texture upload requires non-empty storage")
			})?
			.raw();
		let handle = image.handle;
		let command = state.device.record_compute_commands(|command| {
			let range = ash::vk::ImageSubresourceRange::default()
				.aspect_mask(ash::vk::ImageAspectFlags::COLOR)
				.level_count(1)
				.layer_count(1);
			let to_transfer = ash::vk::ImageMemoryBarrier2::default()
				.src_stage_mask(ash::vk::PipelineStageFlags2::NONE)
				.dst_stage_mask(ash::vk::PipelineStageFlags2::COPY)
				.dst_access_mask(ash::vk::AccessFlags2::TRANSFER_WRITE)
				.old_layout(ash::vk::ImageLayout::UNDEFINED)
				.new_layout(ash::vk::ImageLayout::TRANSFER_DST_OPTIMAL)
				.image(handle)
				.subresource_range(range);
			let barriers = [to_transfer];
			unsafe {
				state.device.raw().cmd_pipeline_barrier2(
					command,
					&ash::vk::DependencyInfo::default().image_memory_barriers(&barriers),
				)
			};
			let copy = ash::vk::BufferImageCopy::default()
				.image_subresource(
					ash::vk::ImageSubresourceLayers::default()
						.aspect_mask(ash::vk::ImageAspectFlags::COLOR)
						.layer_count(1),
				)
				.image_extent(ash::vk::Extent3D {
					width,
					height,
					depth: 1,
				});
			unsafe {
				state.device.raw().cmd_copy_buffer_to_image(
					command,
					buffer,
					handle,
					ash::vk::ImageLayout::TRANSFER_DST_OPTIMAL,
					&[copy],
				)
			};
			let to_read = ash::vk::ImageMemoryBarrier2::default()
				.src_stage_mask(ash::vk::PipelineStageFlags2::COPY)
				.src_access_mask(ash::vk::AccessFlags2::TRANSFER_WRITE)
				.dst_stage_mask(ash::vk::PipelineStageFlags2::ALL_COMMANDS)
				.dst_access_mask(ash::vk::AccessFlags2::SHADER_SAMPLED_READ)
				.old_layout(ash::vk::ImageLayout::TRANSFER_DST_OPTIMAL)
				.new_layout(ash::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
				.image(handle)
				.subresource_range(range);
			let barriers = [to_read];
			unsafe {
				state.device.raw().cmd_pipeline_barrier2(
					command,
					&ash::vk::DependencyInfo::default().image_memory_barriers(&barriers),
				)
			};
			Ok(())
		})?;
		let event = submit_recorded(&mut state, command)?;
		drop(state);
		event.wait()?;
		Ok(image)
	}

	pub(crate) fn same_as(&self, other: &Self) -> bool {
		Rc::ptr_eq(&self.state, &other.state)
	}

	/// Create a `GraphicsPipeline` for the vertex-color lit 3D renderer.
	pub(crate) fn create_graphics_pipeline(&self) -> Result<GraphicsPipeline> {
		self.state.borrow().device.create_graphics_pipeline()
	}

	/// Allocate a host-visible persistently-mapped mesh buffer.
	pub(crate) fn allocate_mesh_buffer(
		&self,
		size: usize,
		usage: ash::vk::BufferUsageFlags,
	) -> Result<MeshBuffer> {
		self.state.borrow().device.allocate_mesh_buffer(size, usage)
	}

	/// Allocate one `RenderTarget` (color + depth + readback) for the 3D renderer.
	pub(crate) fn allocate_render_target(&self, width: u32, height: u32) -> Result<RenderTarget> {
		let state = self.state.borrow();
		RenderTarget::new(&state.device, width, height)
	}

	/// Return the shared bindless descriptor set for command recording.
	pub(crate) fn descriptor_set(&self) -> ash::vk::DescriptorSet {
		self.state.borrow().device.descriptor_set()
	}

	pub(crate) fn require_ui_bindings(&self) -> Result<()> {
		self.state.borrow().device.require_ui_bindings()
	}

	/// Return the pipeline layout shared by all UI compute pipelines.
	pub(crate) fn ui_pipeline_layout(&self) -> ash::vk::PipelineLayout {
		self.state.borrow().device.ui_pipeline_layout()
	}

	/// Return the raw Vulkan device for command recording.
	pub(crate) fn raw_device(&self) -> ash::Device {
		self.state.borrow().device.raw().clone()
	}

	/// Create a device-local R8G8B8A8_UNORM STORAGE image and register it in
	/// the bindless image heap.  Returns `(raw_image, bindless_index)`.
	pub(crate) fn create_compose_image(
		&self,
		width: u32,
		height: u32,
	) -> Result<(ash::vk::Image, u32)> {
		let state = self.state.borrow();
		let device = &state.device;
		let image_info = ash::vk::ImageCreateInfo::default()
			.image_type(ash::vk::ImageType::TYPE_2D)
			.format(ash::vk::Format::R8G8B8A8_UNORM)
			.extent(ash::vk::Extent3D {
				width,
				height,
				depth: 1,
			})
			.mip_levels(1)
			.array_layers(1)
			.samples(ash::vk::SampleCountFlags::TYPE_1)
			.tiling(ash::vk::ImageTiling::OPTIMAL)
			.usage(
				ash::vk::ImageUsageFlags::STORAGE
					| ash::vk::ImageUsageFlags::TRANSFER_SRC
					| ash::vk::ImageUsageFlags::TRANSFER_DST,
			)
			.sharing_mode(ash::vk::SharingMode::EXCLUSIVE)
			.initial_layout(ash::vk::ImageLayout::UNDEFINED);
		let alloc_info = vk_mem::AllocationCreateInfo {
			usage: vk_mem::MemoryUsage::AutoPreferDevice,
			..Default::default()
		};
		let (image, mut alloc) = unsafe { device.allocator().create_image(&image_info, &alloc_info) }
			.map_err(|source| {
			Error::backend_failure("Vulkan", "UI compose image allocation", source)
		})?;
		let view_info = ash::vk::ImageViewCreateInfo::default()
			.image(image)
			.view_type(ash::vk::ImageViewType::TYPE_2D)
			.format(ash::vk::Format::R8G8B8A8_UNORM)
			.subresource_range(
				ash::vk::ImageSubresourceRange::default()
					.aspect_mask(ash::vk::ImageAspectFlags::COLOR)
					.level_count(1)
					.layer_count(1),
			);
		let view = match unsafe { device.raw().create_image_view(&view_info, None) } {
			Ok(view) => view,
			Err(source) => {
				unsafe { device.allocator().destroy_image(image, &mut alloc) };
				return Err(Error::backend_failure(
					"Vulkan",
					"UI compose image view creation",
					source,
				));
			}
		};
		let bindless_index = match device.bind_storage_image(view) {
			Ok(idx) => idx,
			Err(error) => {
				unsafe {
					device.raw().destroy_image_view(view, None);
					device.allocator().destroy_image(image, &mut alloc);
				}
				return Err(error);
			}
		};
		// Store the allocation in the device for cleanup; for now leak the VMA
		// allocation through the stored view in the descriptor — the compose image
		// lifetime is managed by the UI layer which calls destroy_compose_image.
		// We need to stash (image, view, alloc) somewhere. Use a per-engine registry.
		state
			.compose_images
			.borrow_mut()
			.push(ComposeImageEntry { image, view, alloc });
		Ok((image, bindless_index))
	}

	/// Release the bindless slot and destroy the compose image.
	pub(crate) fn destroy_compose_image(&self, raw_image: ash::vk::Image, bindless_index: u32) {
		if let Ok(state) = self.state.try_borrow() {
			state.device.release_storage_image(bindless_index);
			let mut entries = state.compose_images.borrow_mut();
			if let Some(pos) = entries.iter().position(|e| e.image == raw_image) {
				let mut entry = entries.swap_remove(pos);
				unsafe {
					state.device.raw().destroy_image_view(entry.view, None);
					state
						.device
						.allocator()
						.destroy_image(entry.image, &mut entry.alloc);
				}
			}
		}
	}

	/// Return the bindless storage-buffer index for engine-owned storage.
	pub(crate) fn storage_descriptor_index(&self, storage: &Storage) -> Result<u32> {
		let state = self.state.borrow();
		let buffer = storage
			.buffer()
			.ok_or_else(|| Error::invalid_argument("UI source storage must not be empty"))?;
		if !storage.belongs_to(&state.device) {
			return Err(Error::invalid_argument(
				"UI source storage belongs to a different Engine",
			));
		}
		Ok(buffer.descriptor_index())
	}

	/// Record one UI compute composition followed by an exact RGBA8 readback.
	///
	/// This is the private Vulkan seam used by the backend-neutral UI API. The
	/// returned storage becomes host-observable after the returned event completes.
	pub(crate) fn submit_ui_readback(
		&self,
		image: ash::vk::Image,
		width: u32,
		height: u32,
		record: impl FnOnce(&ash::Device, ash::vk::CommandBuffer) -> Result<()>,
	) -> Result<(Event, Storage)> {
		let byte_len = usize::try_from(width)
			.ok()
			.and_then(|w| usize::try_from(height).ok().and_then(|h| w.checked_mul(h)))
			.and_then(|pixels| pixels.checked_mul(4))
			.ok_or_else(|| Error::out_of_range("UI readback byte count overflows usize"))?;
		let mut state = self.state.borrow_mut();
		let readback = Storage::from_bytes(&state.device, &vec![0_u8; byte_len])?;
		let readback_buffer = readback
			.buffer()
			.ok_or_else(|| Error::internal("non-empty UI readback has no Vulkan buffer"))?
			.raw();
		let device = state.device.clone();
		let command = device.record_compute_commands(|command_buffer| {
			record(device.raw(), command_buffer)?;
			let range = ash::vk::ImageSubresourceRange::default()
				.aspect_mask(ash::vk::ImageAspectFlags::COLOR)
				.level_count(1)
				.layer_count(1);
			let to_copy = ash::vk::ImageMemoryBarrier2::default()
				.src_stage_mask(ash::vk::PipelineStageFlags2::COMPUTE_SHADER)
				.src_access_mask(ash::vk::AccessFlags2::SHADER_STORAGE_WRITE)
				.dst_stage_mask(ash::vk::PipelineStageFlags2::COPY)
				.dst_access_mask(ash::vk::AccessFlags2::TRANSFER_READ)
				.old_layout(ash::vk::ImageLayout::GENERAL)
				.new_layout(ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
				.image(image)
				.subresource_range(range);
			unsafe {
				device.raw().cmd_pipeline_barrier2(
					command_buffer,
					&ash::vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&to_copy)),
				);
				device.raw().cmd_copy_image_to_buffer(
					command_buffer,
					image,
					ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
					readback_buffer,
					&[ash::vk::BufferImageCopy::default()
						.image_subresource(
							ash::vk::ImageSubresourceLayers::default()
								.aspect_mask(ash::vk::ImageAspectFlags::COLOR)
								.layer_count(1),
						)
						.image_extent(ash::vk::Extent3D {
							width,
							height,
							depth: 1,
						})],
				);
			}
			Ok(())
		})?;
		let event = submit_recorded(&mut state, command)?;
		readback.mark_submitted(event.clone());
		Ok((event, readback))
	}

	/// Record one mesh draw into `target` and submit it on the graphics queue.
	///
	/// `push_data` is raw push-constant bytes; `push_stages` selects which
	/// shader stages are visible to those bytes.
	#[allow(clippy::too_many_arguments)]
	pub(crate) fn record_and_submit_mesh_frame(
		&self,
		target: &mut RenderTarget,
		vertex_buf: &MeshBuffer,
		index_buf: &MeshBuffer,
		index_count: u32,
		clear_color: [f32; 4],
		push_data: &[u8],
		push_stages: ash::vk::ShaderStageFlags,
		pipeline: ash::vk::Pipeline,
		pipeline_layout: ash::vk::PipelineLayout,
		descriptor_sets: &[ash::vk::DescriptorSet],
	) -> Result<Event> {
		let mut state = self.state.borrow_mut();
		let command = state.device.record_mesh_frame(
			target,
			vertex_buf.raw(),
			index_buf.raw(),
			index_count,
			clear_color,
			push_data,
			push_stages,
			pipeline,
			pipeline_layout,
			descriptor_sets,
		)?;
		submit_recorded(&mut state, command)
	}

	/// Record one scene frame (multiple draws) and submit it on the graphics queue.
	///
	/// `draws` is sorted so that opaque entries come before blend entries by the
	/// caller. Both opaque and blend commands share `pipeline_layout`; the device
	/// function switches to `blend_pipeline` for entries with `is_blend == true`.
	#[allow(clippy::too_many_arguments)]
	pub(crate) fn record_and_submit_scene_frame(
		&self,
		target: &mut RenderTarget,
		vertex_buf: &MeshBuffer,
		index_buf: &MeshBuffer,
		clear_color: [f32; 4],
		push_stages: ash::vk::ShaderStageFlags,
		pipeline: ash::vk::Pipeline,
		blend_pipeline: ash::vk::Pipeline,
		pipeline_layout: ash::vk::PipelineLayout,
		scene_set: ash::vk::DescriptorSet,
		draws: &[SceneDrawCmd],
	) -> Result<Event> {
		let mut state = self.state.borrow_mut();
		let command = state.device.record_scene_frame(
			target,
			vertex_buf.raw(),
			index_buf.raw(),
			clear_color,
			push_stages,
			pipeline,
			blend_pipeline,
			pipeline_layout,
			scene_set,
			draws,
		)?;
		submit_recorded(&mut state, command)
	}

	/// Create the flat-color material Vulkan pipeline, borrowing the device.
	pub(crate) fn create_flat_color_pipeline(&self) -> Result<FlatColorPipeline> {
		self.state.borrow().device.create_flat_color_pipeline()
	}

	/// Create the owned descriptor set layouts for the unlit pipeline.
	pub(crate) fn create_unlit_descriptor_layouts(&self) -> Result<UnlitDescriptorLayouts> {
		self.state.borrow().device.create_unlit_descriptor_layouts()
	}

	pub(crate) fn create_unlit_descriptor_arena(
		&self,
		layout: ash::vk::DescriptorSetLayout,
		set_count: u32,
	) -> Result<UnlitDescriptorArena> {
		self
			.state
			.borrow()
			.device
			.create_unlit_descriptor_arena(layout, set_count)
	}

	/// Create the owned descriptor set layouts for the Standard Surface pipeline.
	pub(crate) fn create_standard_surface_descriptor_layouts(
		&self,
	) -> Result<StandardSurfaceDescriptorLayouts> {
		self
			.state
			.borrow()
			.device
			.create_standard_surface_descriptor_layouts()
	}

	pub(crate) fn create_standard_surface_descriptor_arena(
		&self,
		scene_layout: ash::vk::DescriptorSetLayout,
		material_layout: ash::vk::DescriptorSetLayout,
		set_count: u32,
		max_materials_per_slot: u32,
	) -> Result<StandardSurfaceDescriptorArena> {
		self
			.state
			.borrow()
			.device
			.create_standard_surface_descriptor_arena(
				scene_layout,
				material_layout,
				set_count,
				max_materials_per_slot,
			)
	}

	/// Create the unlit textured Vulkan pipeline, borrowing the device.
	pub(crate) fn create_unlit_pipeline(
		&self,
		sampler_layout: ash::vk::DescriptorSetLayout,
	) -> Result<UnlitPipeline> {
		self
			.state
			.borrow()
			.device
			.create_unlit_pipeline(sampler_layout)
	}

	/// Create the Standard Surface PBR Vulkan pipeline, borrowing the device.
	pub(crate) fn create_standard_surface_pipeline(
		&self,
		scene_layout: ash::vk::DescriptorSetLayout,
		material_layout: ash::vk::DescriptorSetLayout,
	) -> Result<StandardSurfacePipeline> {
		self
			.state
			.borrow()
			.device
			.create_standard_surface_pipeline(scene_layout, material_layout)
	}

	/// Create the Standard Surface transparent-blend pipeline, borrowing the device.
	///
	/// Reuses the `layout` from an existing [`StandardSurfacePipeline`] so the
	/// descriptor sets remain compatible.
	pub(crate) fn create_standard_surface_blend_pipeline(
		&self,
		layout: ash::vk::PipelineLayout,
	) -> Result<StandardSurfaceBlendPipeline> {
		self
			.state
			.borrow()
			.device
			.create_standard_surface_blend_pipeline(layout)
	}

	pub(crate) fn create_storage(&self, bytes: &[u8]) -> Result<Storage> {
		let mut state = self.state.borrow_mut();
		let device = state.device.clone();
		state.session.create_storage(&device, bytes)
	}

	pub(crate) fn create_output_storage(&self, byte_len: usize) -> Result<Storage> {
		let mut state = self.state.borrow_mut();
		let device = state.device.clone();
		state.session.create_output_storage(&device, byte_len)
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

	pub(super) fn create_alias_arena(&self, size: usize) -> Result<Buffer> {
		let device = self.state.borrow().device.clone();
		Buffer::host_visible_alias_arena(&device, size)
	}

	pub(super) fn release_stable_transient_resources(&self, retired: &[Buffer]) {
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

	#[cfg_attr(
		not(test),
		expect(
			dead_code,
			reason = "private PQC ownership substrate; no secret algorithm is admitted"
		)
	)]
	pub(super) fn allocate_secret_buffer(&self, bytes: usize) -> Result<SecretBuffer> {
		let state = self.state.borrow();
		if state.capture_active {
			return Err(Error::failed_precondition(
				"secret allocation cannot enter capture",
			));
		}
		SecretBuffer::new(&state.device, bytes)
	}

	#[cfg_attr(
		not(test),
		expect(
			dead_code,
			reason = "private PQC ownership substrate; no secret algorithm is admitted"
		)
	)]
	pub(super) fn erase_secret_buffer(
		&self,
		secret: SecretBuffer,
		final_consumer: &Event,
	) -> Result<SecretErasure> {
		{
			let state = self.state.borrow();
			if state.capture_active {
				return Err(Error::failed_precondition(
					"secret erasure cannot enter capture",
				));
			}
			if !secret.belongs_to(&state.device) || !final_consumer.comes_from(&state.device) {
				return Err(Error::invalid_argument(
					"secret erasure requires its originating Engine",
				));
			}
		}
		// Flush only at this explicit execution boundary. Queue submission waits
		// on the preceding engine epoch, including the exact final consumer.
		self.flush_impl(false)?;
		let mut state = self.state.borrow_mut();
		let (command, witness) = secret.record_erasure()?;
		let event = submit_recorded(&mut state, command)?;
		Ok(SecretErasure::new(event, witness))
	}

	#[cfg_attr(
		not(test),
		expect(
			dead_code,
			reason = "private entropy execution; no secret algorithm is admitted"
		)
	)]
	pub(super) fn consume_secret_entropy<const N: usize>(
		&self,
		entropy: crate::cryptography::entropy::Entropy<N>,
		consumer: impl FnOnce(&Arc<Device>, ash::vk::CommandBuffer, SecretBinding<'_>) -> Result<()>,
	) -> Result<SecretErasure> {
		if self.state.borrow().capture_active {
			return Err(Error::failed_precondition(
				"secret entropy cannot enter capture",
			));
		}
		self.flush_impl(false)?;
		entropy.consume(|bytes| {
			let mut state = self.state.borrow_mut();
			let secret = SecretBuffer::new(&state.device, N)?;
			let (command, witness) = secret.record_entropy_consumer(bytes, consumer)?;
			let event = submit_recorded(&mut state, command)?;
			Ok(SecretErasure::new(event, witness))
		})
	}

	/// Private qualification route, not public long-lived key generation.
	/// Returns only the public encoding and exact erasure of seed/private storage.
	#[cfg_attr(
		not(test),
		expect(dead_code, reason = "ML-KEM secret operation admission is planned")
	)]
	pub(super) fn prove_mlkem_keygen(
		&self,
		k: u32,
		entropy: crate::cryptography::entropy::Entropy<64>,
	) -> Result<(Buffer, SecretErasure)> {
		if !(2..=4).contains(&k) {
			return Err(Error::invalid_argument("ML-KEM k must be 2, 3 or 4"));
		}
		if self.state.borrow().capture_active {
			return Err(Error::failed_precondition(
				"ML-KEM secret execution cannot enter capture",
			));
		}
		self
			.state
			.borrow()
			.device
			.require_kernel(super::shader::KernelId::CryptographyMlKemKeygenU8)?;
		self.flush_impl(false)?;
		entropy.consume(|bytes| {
			let mut state = self.state.borrow_mut();
			let public = Buffer::host_visible_storage(
				&state.device,
				usize::try_from(384 * k + 32)
					.map_err(|_| Error::invalid_argument("ML-KEM public extent overflow"))?,
			)?;
			let private = SecretBuffer::new(
				&state.device,
				usize::try_from(768 * k + 96)
					.map_err(|_| Error::invalid_argument("ML-KEM private extent overflow"))?,
			)?;
			let seed = SecretBuffer::new(&state.device, 64)?;
			let (command, witness) = seed.record_mlkem_keygen(bytes, private, &public, k)?;
			let event = submit_recorded(&mut state, command)?;
			Ok((public, SecretErasure::new(event, witness)))
		})
	}

	/// Private asynchronous keygen: key readiness and entropy erasure are exact
	/// evidence from the same submission. Secret storage never leaves this module.
	#[cfg_attr(
		not(test),
		expect(dead_code, reason = "private ML-KEM retained-key integration")
	)]
	pub(super) fn generate_mlkem_key(
		&self,
		k: u32,
		entropy: crate::cryptography::entropy::Entropy<64>,
	) -> Result<(MlKemDeviceKey, SecretErasure)> {
		if !(2..=4).contains(&k) {
			return Err(Error::invalid_argument("ML-KEM k must be 2, 3 or 4"));
		}
		{
			let state = self.state.borrow();
			if state.capture_active {
				return Err(Error::failed_precondition(
					"ML-KEM secret execution cannot enter capture",
				));
			}
			state
				.device
				.require_kernel(super::shader::KernelId::CryptographyMlKemKeygenU8)?;
		}
		self.flush_impl(false)?;
		entropy.consume(|bytes| {
			let mut state = self.state.borrow_mut();
			let public = Buffer::host_visible_storage(
				&state.device,
				usize::try_from(384 * k + 32)
					.map_err(|_| Error::invalid_argument("ML-KEM public extent overflow"))?,
			)?;
			let private = Arc::new(SecretBuffer::new(
				&state.device,
				usize::try_from(768 * k + 96)
					.map_err(|_| Error::invalid_argument("ML-KEM private extent overflow"))?,
			)?);
			let seed = SecretBuffer::new(&state.device, 64)?;
			let (command, witness) = seed.record_mlkem_retained_keygen(bytes, &private, &public, k)?;
			let ready = submit_recorded(&mut state, command)?;
			let erasure = SecretErasure::new(ready.clone(), witness);
			Ok((
				MlKemDeviceKey {
					engine: self.clone(),
					private,
					public,
					k,
					ready,
				},
				erasure,
			))
		})
	}

	/// Explicit consuming boundary. Submission order includes key generation
	/// and all previous consumers; failure leaves storage quarantined.
	#[cfg_attr(
		not(test),
		expect(dead_code, reason = "private ML-KEM retained-key integration")
	)]
	pub(super) fn erase_mlkem_key(&self, key: MlKemDeviceKey) -> Result<SecretErasure> {
		{
			let state = self.state.borrow();
			if state.capture_active {
				return Err(Error::failed_precondition(
					"ML-KEM secret erasure cannot enter capture",
				));
			}
			if !Rc::ptr_eq(&key.engine.state, &self.state)
				|| !key.private.belongs_to(&state.device)
				|| !key.ready.comes_from(&state.device)
			{
				return Err(Error::invalid_argument(
					"ML-KEM key requires its originating Engine",
				));
			}
		}
		self.flush_impl(false)?;
		let mut state = self.state.borrow_mut();
		let (command, witness) = key.private.record_retained_erasure()?;
		let event = submit_recorded(&mut state, command)?;
		Ok(SecretErasure::new(event, witness))
	}

	#[cfg_attr(
		not(test),
		expect(dead_code, reason = "private ML-DSA retained-key integration")
	)]
	pub(super) fn generate_mldsa_key(
		&self,
		parameter: u32,
		entropy: crate::cryptography::entropy::Entropy<32>,
	) -> Result<(MlDsaDeviceKey, SecretErasure)> {
		let (seed_len, public_len, private_len) =
			super::shader::KernelId::mldsa_keygen_layout(parameter)
				.ok_or_else(|| Error::invalid_argument("ML-DSA parameter set must be 44, 65 or 87"))?;
		{
			let state = self.state.borrow();
			if state.capture_active {
				return Err(Error::failed_precondition(
					"ML-DSA secret execution cannot enter capture",
				));
			}
			state
				.device
				.require_kernel(super::shader::KernelId::CryptographyMlDsaKeygenU8)?;
		}
		self.flush_impl(false)?;
		entropy.consume(|bytes| {
			let mut state = self.state.borrow_mut();
			let public = Buffer::host_visible_storage(
				&state.device,
				usize::try_from(public_len)
					.map_err(|_| Error::invalid_argument("ML-DSA public extent overflow"))?,
			)?;
			let private = Arc::new(SecretBuffer::new(
				&state.device,
				usize::try_from(private_len)
					.map_err(|_| Error::invalid_argument("ML-DSA private extent overflow"))?,
			)?);
			let seed = SecretBuffer::new(
				&state.device,
				usize::try_from(seed_len)
					.map_err(|_| Error::invalid_argument("ML-DSA seed extent overflow"))?,
			)?;
			let (command, witness) =
				seed.record_mldsa_retained_keygen(bytes, &private, &public, parameter)?;
			let ready = submit_recorded(&mut state, command)?;
			let erasure = SecretErasure::new(ready.clone(), witness);
			Ok((
				MlDsaDeviceKey {
					engine: self.clone(),
					private,
					public,
					parameter,
					ready,
				},
				erasure,
			))
		})
	}

	/// Explicit consuming boundary. Submission order includes key generation
	/// and all previous consumers; failure leaves storage quarantined.
	#[cfg_attr(
		not(test),
		expect(dead_code, reason = "private ML-DSA retained-key integration")
	)]
	pub(super) fn erase_mldsa_key(&self, key: MlDsaDeviceKey) -> Result<SecretErasure> {
		{
			let state = self.state.borrow();
			if state.capture_active {
				return Err(Error::failed_precondition(
					"ML-DSA secret erasure cannot enter capture",
				));
			}
			if !Rc::ptr_eq(&key.engine.state, &self.state)
				|| !key.private.belongs_to(&state.device)
				|| !key.public.belongs_to(&state.device)
				|| !key.ready.comes_from(&state.device)
			{
				return Err(Error::invalid_argument(
					"ML-DSA key requires its originating Engine",
				));
			}
		}
		self.flush_impl(false)?;
		let mut state = self.state.borrow_mut();
		let (command, witness) = key.private.record_retained_erasure()?;
		let event = submit_recorded(&mut state, command)?;
		Ok(SecretErasure::new(event, witness))
	}

	/// Private FIPS 204 internal-mu interface. The 64-byte public representative
	/// is already H(tr || M', 64); raw messages require separate framing.
	/// Returns public signature storage (zero on failure), a public success word,
	/// and exact entropy-erasure completion. No rejection count is exposed.
	#[cfg_attr(
		not(test),
		expect(dead_code, reason = "private ML-DSA signing integration")
	)]
	pub(super) fn sign_mldsa_mu(
		&self,
		key: &MlDsaDeviceKey,
		mu: &Buffer,
		entropy: crate::cryptography::entropy::Entropy<32>,
	) -> Result<(Buffer, Buffer, SecretErasure)> {
		let (seed_len, _, mu_len, sig_len) = super::shader::KernelId::mldsa_sign_layout(key.parameter)
			.ok_or_else(|| Error::invalid_argument("ML-DSA parameter set must be 44, 65 or 87"))?;
		{
			let state = self.state.borrow();
			if state.capture_active {
				return Err(Error::failed_precondition(
					"ML-DSA secret execution cannot enter capture",
				));
			}
			if !Rc::ptr_eq(&key.engine.state, &self.state)
				|| !key.private.belongs_to(&state.device)
				|| !key.public.belongs_to(&state.device)
				|| !key.ready.comes_from(&state.device)
				|| !mu.belongs_to(&state.device)
				|| mu.size() != mu_len
			{
				return Err(Error::invalid_argument(
					"ML-DSA signing ownership or mu extent mismatch",
				));
			}
			state
				.device
				.require_kernel(super::shader::KernelId::CryptographyMlDsaSignU8)?;
		}
		self.flush_impl(false)?;
		entropy.consume(|bytes| {
			let mut state = self.state.borrow_mut();
			let signature = Buffer::host_visible_storage(
				&state.device,
				usize::try_from((sig_len + 3) & !3)
					.map_err(|_| Error::invalid_argument("ML-DSA signature extent overflow"))?,
			)?;
			let status = Buffer::host_visible_storage(&state.device, 4)?;
			let seed = SecretBuffer::new(
				&state.device,
				usize::try_from(seed_len)
					.map_err(|_| Error::invalid_argument("ML-DSA randomness extent overflow"))?,
			)?;
			let (command, witness) = seed.record_mldsa_sign(
				bytes,
				&key.private,
				[mu, &signature, &status],
				key.parameter,
			)?;
			let event = submit_recorded(&mut state, command)?;
			Ok((signature, status, SecretErasure::new(event, witness)))
		})
	}

	/// Private pure ML-DSA signing. Message/context storage is public, packed
	/// and padded to at least one complete word; lengths exclude padding.
	/// The device forms H(tr || 0 || context length || context || message, 64).
	/// No host key readback, internal-message mode or prehash mode is implied.
	#[cfg_attr(
		not(test),
		expect(dead_code, reason = "private ML-DSA signing integration")
	)]
	pub(super) fn sign_mldsa_message(
		&self,
		key: &MlDsaDeviceKey,
		message: &Buffer,
		context: &Buffer,
		lengths: [u32; 2],
		entropy: crate::cryptography::entropy::Entropy<32>,
	) -> Result<(Buffer, Buffer, SecretErasure)> {
		self.sign_mldsa_message_impl(key, message, context, lengths, None, entropy)
	}

	/// Private HashML-DSA signing from the original public message. Digest and
	/// mu framing stay on device; workspace and entropy share exact erasure.
	#[cfg_attr(
		not(test),
		expect(dead_code, reason = "private ML-DSA signing integration")
	)]
	pub(super) fn sign_mldsa_hash_message(
		&self,
		key: &MlDsaDeviceKey,
		message: &Buffer,
		context: &Buffer,
		lengths: [u32; 2],
		algorithm: u32,
		entropy: crate::cryptography::entropy::Entropy<32>,
	) -> Result<(Buffer, Buffer, SecretErasure)> {
		self.sign_mldsa_message_impl(key, message, context, lengths, Some(algorithm), entropy)
	}

	fn sign_mldsa_message_impl(
		&self,
		key: &MlDsaDeviceKey,
		message: &Buffer,
		context: &Buffer,
		lengths: [u32; 2],
		algorithm: Option<u32>,
		entropy: crate::cryptography::entropy::Entropy<32>,
	) -> Result<(Buffer, Buffer, SecretErasure)> {
		if let Some(algorithm) = algorithm {
			super::shader::KernelId::mldsa_prehash_bytes(algorithm)
				.ok_or_else(|| Error::invalid_argument("ML-DSA prehash identity is unknown"))?;
		}
		let kernel = if algorithm.is_some() {
			super::shader::KernelId::CryptographyMlDsaSignHashMessageU8
		} else {
			super::shader::KernelId::CryptographyMlDsaSignMessageU8
		};
		let (seed_len, _, _, sig_len) = super::shader::KernelId::mldsa_sign_layout(key.parameter)
			.ok_or_else(|| Error::invalid_argument("ML-DSA parameter set must be 44, 65 or 87"))?;
		if lengths[1] > 255 {
			return Err(Error::invalid_argument("ML-DSA context exceeds 255 bytes"));
		}
		let extents = lengths.map(|length| length.checked_add(3).map(|n| u64::from((n & !3).max(4))));
		let [Some(message_extent), Some(context_extent)] = extents else {
			return Err(Error::invalid_argument("ML-DSA input extent overflow"));
		};
		{
			let state = self.state.borrow();
			if state.capture_active {
				return Err(Error::failed_precondition(
					"ML-DSA secret execution cannot enter capture",
				));
			}
			if !Rc::ptr_eq(&key.engine.state, &self.state)
				|| !key.private.belongs_to(&state.device)
				|| !key.public.belongs_to(&state.device)
				|| !key.ready.comes_from(&state.device)
				|| !message.belongs_to(&state.device)
				|| !context.belongs_to(&state.device)
				|| message.size() != message_extent
				|| context.size() != context_extent
				|| message.raw() == context.raw()
			{
				return Err(Error::invalid_argument(
					"ML-DSA signing ownership, alias or input extent mismatch",
				));
			}
			state.device.require_kernel(kernel)?;
		}
		self.flush_impl(false)?;
		entropy.consume(|bytes| {
			let mut state = self.state.borrow_mut();
			let signature = Buffer::host_visible_storage(
				&state.device,
				usize::try_from((sig_len + 3) & !3)
					.map_err(|_| Error::invalid_argument("ML-DSA signature extent overflow"))?,
			)?;
			let status = Buffer::host_visible_storage(&state.device, 4)?;
			let seed = SecretBuffer::new(
				&state.device,
				usize::try_from(seed_len)
					.map_err(|_| Error::invalid_argument("ML-DSA randomness extent overflow"))?,
			)?;
			let (command, witness) = seed.record_mldsa_sign_message(
				bytes,
				&key.private,
				[message, context, &signature, &status],
				key.parameter,
				lengths,
				algorithm,
			)?;
			let event = submit_recorded(&mut state, command)?;
			Ok((signature, status, SecretErasure::new(event, witness)))
		})
	}

	/// Private HashML-DSA framing for a precomputed public digest. The generated
	/// NIST hash identity fixes its length and OID. This does not compute the
	/// prehash, admit a FIPS-validated module or imply full hash-family coverage.
	#[cfg_attr(
		not(test),
		expect(dead_code, reason = "private ML-DSA signing integration")
	)]
	pub(super) fn sign_mldsa_prehashed(
		&self,
		key: &MlDsaDeviceKey,
		digest: &Buffer,
		context: &Buffer,
		algorithm: u32,
		context_length: u32,
		entropy: crate::cryptography::entropy::Entropy<32>,
	) -> Result<(Buffer, Buffer, SecretErasure)> {
		let (seed_len, _, _, sig_len) = super::shader::KernelId::mldsa_sign_layout(key.parameter)
			.ok_or_else(|| Error::invalid_argument("ML-DSA parameter set must be 44, 65 or 87"))?;
		if context_length > 255 {
			return Err(Error::invalid_argument("ML-DSA context exceeds 255 bytes"));
		}
		let digest_extent = super::shader::KernelId::mldsa_prehash_bytes(algorithm)
			.ok_or_else(|| Error::invalid_argument("ML-DSA prehash identity is unknown"))?;
		let context_extent = u64::from(((context_length + 3) & !3).max(4));
		{
			let state = self.state.borrow();
			if state.capture_active {
				return Err(Error::failed_precondition(
					"ML-DSA secret execution cannot enter capture",
				));
			}
			if !Rc::ptr_eq(&key.engine.state, &self.state)
				|| !key.private.belongs_to(&state.device)
				|| !key.public.belongs_to(&state.device)
				|| !key.ready.comes_from(&state.device)
				|| !digest.belongs_to(&state.device)
				|| !context.belongs_to(&state.device)
				|| digest.size() != digest_extent
				|| context.size() != context_extent
				|| digest.raw() == context.raw()
			{
				return Err(Error::invalid_argument(
					"ML-DSA signing ownership, alias or input extent mismatch",
				));
			}
			state
				.device
				.require_kernel(super::shader::KernelId::CryptographyMlDsaSignPrehashedU8)?;
		}
		self.flush_impl(false)?;
		entropy.consume(|bytes| {
			let mut state = self.state.borrow_mut();
			let signature = Buffer::host_visible_storage(
				&state.device,
				usize::try_from((sig_len + 3) & !3)
					.map_err(|_| Error::invalid_argument("ML-DSA signature extent overflow"))?,
			)?;
			let status = Buffer::host_visible_storage(&state.device, 4)?;
			let seed = SecretBuffer::new(
				&state.device,
				usize::try_from(seed_len)
					.map_err(|_| Error::invalid_argument("ML-DSA randomness extent overflow"))?,
			)?;
			let (command, witness) = seed.record_mldsa_sign_prehashed(
				bytes,
				&key.private,
				[digest, context, &signature, &status],
				key.parameter,
				algorithm,
				context_length,
			)?;
			let event = submit_recorded(&mut state, command)?;
			Ok((signature, status, SecretErasure::new(event, witness)))
		})
	}

	fn validate_mlkem_key(&self, key: &MlKemDeviceKey) -> Result<()> {
		let state = self.state.borrow();
		if state.capture_active {
			return Err(Error::failed_precondition(
				"ML-KEM secret execution cannot enter capture",
			));
		}
		if !Rc::ptr_eq(&key.engine.state, &self.state)
			|| !key.private.belongs_to(&state.device)
			|| !key.public.belongs_to(&state.device)
			|| !key.ready.comes_from(&state.device)
		{
			return Err(Error::invalid_argument(
				"ML-KEM key requires its originating Engine",
			));
		}
		Ok(())
	}

	/// Private retained-key route; external encapsulation keys are not admitted.
	/// Status is public key canonicality, never ciphertext rejection.
	#[cfg_attr(
		not(test),
		expect(dead_code, reason = "private ML-KEM shared-secret integration")
	)]
	pub(super) fn encapsulate_mlkem(
		&self,
		key: &MlKemDeviceKey,
		entropy: crate::cryptography::entropy::Entropy<32>,
	) -> Result<(MlKemSharedSecret, Buffer, Buffer, SecretErasure)> {
		self.validate_mlkem_key(key)?;
		self
			.state
			.borrow()
			.device
			.require_kernel(super::shader::KernelId::CryptographyMlKemEncapsU8)?;
		self.flush_impl(false)?;
		entropy.consume(|bytes| {
			let mut state = self.state.borrow_mut();
			let k = key.k;
			let ciphertext_size = 32 * if k == 4 { 11 * k + 5 } else { 10 * k + 4 };
			let ciphertext = Buffer::host_visible_storage(
				&state.device,
				usize::try_from(ciphertext_size)
					.map_err(|_| Error::invalid_argument("ML-KEM ciphertext extent overflow"))?,
			)?;
			let status = Buffer::host_visible_storage(&state.device, 4)?;
			let private = Arc::new(SecretBuffer::new(&state.device, 32)?);
			let seed = SecretBuffer::new(&state.device, 32)?;
			let (command, witness) =
				seed.record_mlkem_encaps(bytes, &key.public, &private, &ciphertext, &status, k)?;
			let ready = submit_recorded(&mut state, command)?;
			let erasure = SecretErasure::new(ready.clone(), witness);
			Ok((
				MlKemSharedSecret {
					engine: self.clone(),
					private,
					ready,
				},
				ciphertext,
				status,
				erasure,
			))
		})
	}

	/// Retained generated keys only. A malformed ciphertext of the correct size
	/// follows implicit rejection entirely on device; no rejection flag escapes.
	#[cfg_attr(
		not(test),
		expect(dead_code, reason = "private ML-KEM shared-secret integration")
	)]
	pub(super) fn decapsulate_mlkem(
		&self,
		key: &MlKemDeviceKey,
		ciphertext: &Buffer,
	) -> Result<MlKemSharedSecret> {
		self.validate_mlkem_key(key)?;
		let expected = u64::from(
			32 * if key.k == 4 {
				11 * key.k + 5
			} else {
				10 * key.k + 4
			},
		);
		{
			let state = self.state.borrow();
			if !ciphertext.belongs_to(&state.device) || ciphertext.size() != expected {
				return Err(Error::invalid_argument(
					"ML-KEM ciphertext ownership or extent mismatch",
				));
			}
			state
				.device
				.require_kernel(super::shader::KernelId::CryptographyMlKemDecapsU8)?;
		}
		self.flush_impl(false)?;
		let mut state = self.state.borrow_mut();
		let private = Arc::new(SecretBuffer::new(&state.device, 32)?);
		let command = key
			.private
			.record_mlkem_decaps(ciphertext, &private, key.k)?;
		let ready = submit_recorded(&mut state, command)?;
		Ok(MlKemSharedSecret {
			engine: self.clone(),
			private,
			ready,
		})
	}

	#[cfg_attr(
		not(test),
		expect(dead_code, reason = "private ML-KEM shared-secret integration")
	)]
	pub(super) fn erase_mlkem_shared_secret(
		&self,
		secret: MlKemSharedSecret,
	) -> Result<SecretErasure> {
		{
			let state = self.state.borrow();
			if state.capture_active {
				return Err(Error::failed_precondition(
					"ML-KEM secret erasure cannot enter capture",
				));
			}
			if !Rc::ptr_eq(&secret.engine.state, &self.state)
				|| !secret.private.belongs_to(&state.device)
				|| !secret.ready.comes_from(&state.device)
			{
				return Err(Error::invalid_argument(
					"ML-KEM shared secret requires its originating Engine",
				));
			}
		}
		self.flush_impl(false)?;
		let mut state = self.state.borrow_mut();
		let (command, witness) = secret.private.record_retained_erasure()?;
		let event = submit_recorded(&mut state, command)?;
		Ok(SecretErasure::new(event, witness))
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

	/// Emit a `Texture`-kind semantic value node for `texture_value_id`.
	///
	/// Idempotent per session: the same id is not added twice.
	pub(crate) fn record_texture_semantic(
		&self,
		texture_value_id: u64,
		width: usize,
		height: usize,
	) -> Result<()> {
		self
			.state
			.borrow_mut()
			.session
			.record_texture_semantic(texture_value_id, width, height)
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

fn submit_recorded(state: &mut EngineState, command: RecordedCommandBuffer) -> Result<Event> {
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
#[path = "../../../test/rs/runtime/secret_engine_unit.rs"]
mod secret_tests;

#[cfg(test)]
#[path = "../../../test/rs/runtime/device_owner_unit.rs"]
mod device_owner_tests;
