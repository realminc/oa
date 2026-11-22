//! Vulkan Video decode session, bitstreams, and native decoded-image leases.

use super::{
	MAX_VIDEO_SESSION_MEMORY_BINDINGS,
	image::{VideoImage, create_video_image},
	profile::*,
};
use crate::runtime::device::NativeFlagBits;
use std::sync::{
	Arc, Mutex,
	atomic::{AtomicBool, Ordering},
};

use crate::{Error, Event, Result, core::memory, video};

use crate::runtime::Device;
use crate::runtime::device::logical::{command::RecordedBatch, pipeline::ComputePipeline};
#[cfg(oa_profile = "core")]
use crate::runtime::dna::kernel::{ShaderArtifact, content_ids};

mod av1;
mod h264;
mod h265;
mod hwycbcr;
mod replay;
mod transfer;
mod vp9;

use h264::record_h264_picture;
use h265::record_h265_picture;
pub(in crate::runtime) use replay::{DecodeStatus, PreparedDecode, RetainedDecode};
pub(in crate::runtime) use transfer::{DecodedPlaneCopy, PlaneCopyPhase, yuv420_copy_layout};
use transfer::{
	ImageUse, normalize_yuv420, presentation_sample_bits, record_decode_presentation,
	record_decode_readback_round_trip, record_image_consumer, record_plane_readback,
};

#[cfg(oa_profile = "core")]
static CVT_YUV420_TO_RGBA8: ShaderArtifact = ShaderArtifact::embedded(
	include_bytes!(concat!(
		env!("OA_SHADER_DIR"),
		"/video_cvt_yuv420_to_rgba8.spv"
	)),
	[16, 16, 1],
	content_ids::VIDEO_CVT_YUV420_TO_RGBA8,
);

#[cfg(oa_profile = "core")]
fn create_presentation_pipeline(device: &Arc<Device>) -> Result<ComputePipeline> {
	device.create_compute_pipeline(&CVT_YUV420_TO_RGBA8)
}

#[cfg(oa_profile = "compatibility")]
fn create_presentation_pipeline(_device: &Arc<Device>) -> Result<ComputePipeline> {
	Err(Error::missing_capability(
		"video presentation conversion requires a Core build",
	))
}

#[repr(C)]
struct PresentationPush {
	src_idx: u32,
	dst_idx: u32,
	src_width: u32,
	src_height: u32,
	dst_width: u32,
	dst_height: u32,
	uv_offset: u32,
	v_offset: u32,
	uv_layout: u32,
	color_space: u32,
	full_range: u32,
	sample_bits: u32,
}

/// Recorded commands and output for one native-frame presentation conversion.
pub(in crate::runtime) struct NativePresentation {
	pub commands: RecordedBatch,
	pub rgba: crate::runtime::Buffer,
	pub extent: video::VideoExtent,
}

/// Recorded plane transfer and its independent canvas allocation.
pub(in crate::runtime) struct NativeCanvasTransfer {
	pub commands: RecordedBatch,
	pub planes: crate::runtime::Buffer,
	pub layout: video::CanvasYuvLayout,
}

pub(in crate::runtime) struct DecodeSession {
	handle: vk::VideoSessionKHR,
	parameters: vk::VideoSessionParametersKHR,
	result_status_pool: vk::QueryPool,
	allocations: Vec<vk::mem::Allocation>,
	images: Arc<Vec<VideoImage>>,
	native_leases: Option<Arc<NativeFrameLeases>>,
	image_set: DecodeImageSet,
	bitstream: Option<DecodeBitstream>,
	readback: Option<crate::runtime::Buffer>,
	conversion: Option<Arc<ComputePipeline>>,
	hardware_ycbcr_supported: bool,
	hardware_ycbcr: Vec<Arc<hwycbcr::HardwareYcbcrColorConversion>>,
	presentation_yuv: Option<crate::runtime::Buffer>,
	bitstream_size_alignment: u64,
	coded_extent: video::VideoExtent,
	visible_extent: video::VideoExtent,
	visible_origin: [u32; 2],
	decode_recorded: bool,
	readback_recorded: bool,
	device: Arc<Device>,
	profile: video::VideoDecodeProfile,
	h264_dpb_state: Option<h264::DpbState>,
	h265_dpb_state: Option<h265::DpbState>,
	av1_dpb_state: Option<av1::DpbState>,
	vp9_dpb_state: Option<vp9::DpbState>,
	released_output_slot: Option<u32>,
	pending_decode_acquire_slot: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DecodeImageSet {
	CoincidentLayered,
	CoincidentSeparate,
	DistinctLayered,
}

struct NativeFrameLeases {
	device: Arc<Device>,
	_images: Arc<Vec<VideoImage>>,
	format: video::VideoPixelFormat,
	slots: Mutex<Vec<NativeFrameSlot>>,
	ownership_lost: AtomicBool,
}

#[derive(Default)]
struct NativeFrameSlot {
	lease_count: usize,
	consumer: Option<Event>,
}

pub(crate) struct NativeDecodedFrame {
	lease: Arc<NativeDecodedFrameLease>,
	format: video::VideoPixelFormat,
	extent: video::VideoExtent,
	ready: Event,
}

struct NativeDecodedFrameLease {
	pool: Arc<NativeFrameLeases>,
	slot: u32,
}

impl Clone for NativeDecodedFrame {
	fn clone(&self) -> Self {
		Self {
			lease: self.lease.clone(),
			format: self.format,
			extent: self.extent,
			ready: self.ready.clone(),
		}
	}
}

impl NativeDecodedFrame {
	pub(crate) const fn format(&self) -> video::VideoPixelFormat {
		self.format
	}

	pub(crate) const fn extent(&self) -> video::VideoExtent {
		self.extent
	}

	pub(crate) const fn ready(&self) -> &Event {
		&self.ready
	}

	fn slot(&self) -> u32 {
		self.lease.slot
	}

	fn belongs_to(&self, pool: &Arc<NativeFrameLeases>) -> bool {
		Arc::ptr_eq(&self.lease.pool, pool)
	}

	pub(in crate::runtime) fn invalidate_ownership(&self) {
		self
			.lease
			.pool
			.ownership_lost
			.store(true, Ordering::Release);
	}

	pub(crate) fn mark_consumed(&self, event: &Event) -> Result<()> {
		if event.epoch() < self.ready.epoch() {
			return Err(Error::invalid_argument(
				"video consumer completion precedes native frame readiness",
			));
		}
		self.lease.pool.mark_consumed(self.lease.slot, event)
	}
}

impl Drop for NativeDecodedFrameLease {
	fn drop(&mut self) {
		self.pool.release_frame(self.slot);
	}
}

impl NativeFrameLeases {
	fn new(
		device: &Arc<Device>,
		images: Arc<Vec<VideoImage>>,
		slot_count: u32,
		format: video::VideoPixelFormat,
	) -> Result<Self> {
		let slot_count = usize::try_from(slot_count)
			.map_err(|_| Error::out_of_range("native video slot count exceeds usize"))?;
		let mut slots = Vec::new();
		slots
			.try_reserve_exact(slot_count)
			.map_err(|_| Error::resource_exhausted("native video lease allocation failed"))?;
		slots.resize_with(slot_count, NativeFrameSlot::default);
		Ok(Self {
			device: device.clone(),
			_images: images,
			format,
			slots: Mutex::new(slots),
			ownership_lost: AtomicBool::new(false),
		})
	}

	fn validate_ownership(&self) -> Result<()> {
		if self.ownership_lost.load(Ordering::Acquire) {
			return Err(Error::failed_precondition(
				"video decoder image ownership was interrupted; reopen the decoder",
			));
		}
		Ok(())
	}

	fn unavailable_slots(&self) -> Result<Vec<bool>> {
		self.validate_ownership()?;
		let mut slots = self
			.slots
			.lock()
			.map_err(|_| Error::internal("native video lease state is poisoned"))?;
		let mut unavailable = Vec::new();
		unavailable
			.try_reserve_exact(slots.len())
			.map_err(|_| Error::resource_exhausted("native video lease snapshot failed"))?;
		for slot in slots.iter_mut() {
			if slot.lease_count == 0
				&& let Some(event) = &slot.consumer
				&& event.is_complete()?
			{
				slot.consumer = None;
			}
			unavailable.push(slot.lease_count != 0 || slot.consumer.is_some());
		}
		Ok(unavailable)
	}

	fn lease(
		self: &Arc<Self>,
		slot: u32,
		ready: Event,
		extent: video::VideoExtent,
	) -> Result<NativeDecodedFrame> {
		self.validate_ownership()?;
		let index =
			usize::try_from(slot).map_err(|_| Error::out_of_range("native video slot exceeds usize"))?;
		{
			let mut slots = self
				.slots
				.lock()
				.map_err(|_| Error::internal("native video lease state is poisoned"))?;
			let state = slots
				.get_mut(index)
				.ok_or_else(|| Error::internal("native video slot exceeds lease capacity"))?;
			if state.consumer.is_some() {
				return Err(Error::resource_exhausted(
					"decoded video slot has a pending consumer completion",
				));
			}
			state.lease_count = state
				.lease_count
				.checked_add(1)
				.ok_or_else(|| Error::resource_exhausted("native video lease count exhausted"))?;
		}
		Ok(NativeDecodedFrame {
			lease: Arc::new(NativeDecodedFrameLease {
				pool: self.clone(),
				slot,
			}),
			format: self.format,
			extent,
			ready,
		})
	}

	fn mark_consumed(self: &Arc<Self>, slot: u32, event: &Event) -> Result<()> {
		self.validate_ownership()?;
		if !event.comes_from(&self.device) {
			return Err(Error::invalid_argument(
				"video consumer event belongs to another engine",
			));
		}
		let index =
			usize::try_from(slot).map_err(|_| Error::out_of_range("native video slot exceeds usize"))?;
		let mut slots = self
			.slots
			.lock()
			.map_err(|_| Error::internal("native video lease state is poisoned"))?;
		let state = slots
			.get_mut(index)
			.ok_or_else(|| Error::invalid_argument("native video slot is invalid"))?;
		if state.lease_count == 0 {
			return Err(Error::failed_precondition(
				"native video frame lease is no longer active",
			));
		}
		if state
			.consumer
			.as_ref()
			.is_none_or(|current| event.epoch() > current.epoch())
		{
			state.consumer = Some(event.clone());
		}
		event.retain_until_complete(self.clone());
		Ok(())
	}

	fn validate_live_frame(&self, slot: u32) -> Result<()> {
		self.validate_ownership()?;
		let index =
			usize::try_from(slot).map_err(|_| Error::out_of_range("native video slot exceeds usize"))?;
		let slots = self
			.slots
			.lock()
			.map_err(|_| Error::internal("native video lease state is poisoned"))?;
		let state = slots
			.get(index)
			.ok_or_else(|| Error::invalid_argument("native video slot is invalid"))?;
		if state.lease_count == 0 {
			return Err(Error::failed_precondition(
				"native video frame lease is no longer active",
			));
		}
		Ok(())
	}

	fn release_frame(&self, slot: u32) {
		let Ok(index) = usize::try_from(slot) else {
			return;
		};
		let Ok(mut slots) = self.slots.lock() else {
			return;
		};
		if let Some(state) = slots.get_mut(index) {
			state.lease_count = state.lease_count.saturating_sub(1);
			if state
				.consumer
				.as_ref()
				.is_some_and(|event| event.is_complete().unwrap_or(false))
			{
				state.consumer = None;
			}
		}
	}
}

struct DecodeBitstream {
	device: Arc<Device>,
	capacity: usize,
	handle: vk::Buffer,
	allocation: vk::mem::Allocation,
	payload_len: usize,
	range: vk::DeviceSize,
}

pub(in crate::runtime::device) fn create_decode_session(
	device: &Arc<Device>,
	profile: video::VideoDecodeProfile,
	coded_extent: video::VideoExtent,
	max_dpb_slots: u32,
	max_active_references: u32,
) -> Result<DecodeSession> {
	let physical = device.physical();
	let instance = device.instance();
	let details = query_decode_details(instance, physical, profile)?;
	let limits = details.capabilities;
	if coded_extent.width < limits.min_coded_extent().width
		|| coded_extent.height < limits.min_coded_extent().height
		|| coded_extent.width > limits.max_coded_extent().width
		|| coded_extent.height > limits.max_coded_extent().height
	{
		return Err(Error::invalid_argument(
			"video session coded extent is outside the exact profile limits",
		));
	}
	if max_dpb_slots == 0
		|| max_dpb_slots > limits.max_dpb_slots()
		|| max_active_references > limits.max_active_reference_pictures()
		|| max_active_references > max_dpb_slots
	{
		return Err(Error::invalid_argument(
			"video session DPB/reference counts exceed the exact profile limits",
		));
	}
	if limits.min_bitstream_offset_alignment() == 0 || limits.min_bitstream_size_alignment() == 0 {
		return Err(Error::backend_failure(
			"Vulkan",
			"video-profile bitstream alignment",
			std::io::Error::other("driver returned a zero bitstream alignment"),
		));
	}
	with_decode_profile(profile, |vk_profile| {
		let mut output_usage =
			vk::ImageUsageFlagBits::VIDEO_DECODE_DST_BIT_KHR | vk::ImageUsageFlagBits::TRANSFER_SRC_BIT;
		let dpb_usage = vk::ImageUsageFlagBits::VIDEO_DECODE_DPB_BIT_KHR;
		if cfg!(oa_profile = "core") && device.configuration().features().sampler_ycbcr_conversion {
			let usage = output_usage
				| vk::ImageUsageFlagBits::SAMPLED_BIT
				| if limits.dpb_and_output_coincide() {
					dpb_usage
				} else {
					vk::ImageUsageFlags::empty()
				};
			let formats = query_formats(instance.raw(), physical.handle, vk_profile, usage)?;
			if formats.iter().any(|format| {
				format.image_type == vk::ImageType::_2D
					&& format.image_tiling == vk::ImageTiling::OPTIMAL
					&& !format
						.image_create_flags
						.contains(vk::ImageCreateFlagBits::DISJOINT_BIT)
					&& format.image_usage_flags.contains(usage)
					&& hwycbcr::supported(device, format.format)
			}) {
				output_usage = output_usage | vk::ImageUsageFlagBits::SAMPLED_BIT;
			}
		}
		let mut output_formats: Vec<vk::VideoFormatPropertiesKHR<'_>>;
		let dpb_formats: Vec<vk::VideoFormatPropertiesKHR<'_>>;
		let (output_format, dpb_format, coincident_images) = if limits.dpb_and_output_coincide() {
			let combined_usage = output_usage | dpb_usage;
			output_formats = query_formats(instance.raw(), physical.handle, vk_profile, combined_usage)?;
			if output_usage.contains(vk::ImageUsageFlagBits::SAMPLED_BIT) {
				output_formats.sort_by_key(|format| !hwycbcr::supported(device, format.format));
			}
			let format = select_image_format(&output_formats, combined_usage, "coincident decode")?;
			(format, format, true)
		} else if limits.dpb_and_output_distinct() {
			output_formats = query_formats(instance.raw(), physical.handle, vk_profile, output_usage)?;
			if output_usage.contains(vk::ImageUsageFlagBits::SAMPLED_BIT) {
				output_formats.sort_by_key(|format| !hwycbcr::supported(device, format.format));
			}
			dpb_formats = query_formats(instance.raw(), physical.handle, vk_profile, dpb_usage)?;
			(
				select_image_format(&output_formats, output_usage, "decode output")?,
				select_image_format(&dpb_formats, dpb_usage, "decode DPB")?,
				false,
			)
		} else {
			return Err(Error::missing_capability(
				"decode profile permits neither coincident nor distinct DPB/output images",
			));
		};
		let create_info = vk::VideoSessionCreateInfoKHR::new()
			.set_queue_family_index(
				physical
					.video
					.decode_queue_family
					.ok_or_else(|| Error::missing_capability("no Vulkan Video decode queue is enabled"))?,
			)
			.set_video_profile(vk_profile)
			.set_picture_format(output_format.format)
			.set_max_coded_extent(vk::Extent2D {
				width: coded_extent.width,
				height: coded_extent.height,
			})
			.set_reference_picture_format(dpb_format.format)
			.set_max_dpb_slots(max_dpb_slots)
			.set_max_active_reference_pictures(max_active_references)
			.set_std_header_version(&details.std_header_version);
		let h264_dpb_state = if matches!(profile, video::VideoDecodeProfile::H264 { .. }) {
			Some(h264::DpbState::new(max_dpb_slots.min(16))?)
		} else {
			None
		};
		let h265_dpb_state = if matches!(profile, video::VideoDecodeProfile::H265 { .. }) {
			Some(h265::DpbState::new(max_dpb_slots)?)
		} else {
			None
		};
		let av1_dpb_state = if matches!(profile, video::VideoDecodeProfile::Av1 { .. }) {
			Some(av1::DpbState::new(max_dpb_slots)?)
		} else {
			None
		};
		let vp9_dpb_state = if matches!(profile, video::VideoDecodeProfile::Vp9 { .. }) {
			Some(vp9::DpbState::new(max_dpb_slots)?)
		} else {
			None
		};
		let mut handle = vk::VideoSessionKHR::default();
		// SAFETY: the exact profile, supported formats, queried header version,
		// decode queue family, and bounded extent/counts remain live for this call.
		let result =
			unsafe { device.create_video_session_khr(&create_info, std::ptr::null(), &mut handle) };
		if !matches!(result, Ok(vk::Result::SUCCESS)) {
			return Err(Error::backend_dispatch_failure(
				"Vulkan",
				"video-session creation",
				result,
			));
		}
		let image_set = if coincident_images && limits.separate_reference_images() {
			DecodeImageSet::CoincidentSeparate
		} else if coincident_images {
			DecodeImageSet::CoincidentLayered
		} else {
			DecodeImageSet::DistinctLayered
		};
		let mut session = DecodeSession {
			handle,
			parameters: vk::VideoSessionParametersKHR::default(),
			result_status_pool: vk::QueryPool::default(),
			allocations: Vec::new(),
			images: Arc::new(Vec::new()),
			native_leases: None,
			image_set,
			bitstream: None,
			readback: None,
			conversion: None,
			hardware_ycbcr_supported: output_usage.contains(vk::ImageUsageFlagBits::SAMPLED_BIT)
				&& hwycbcr::supported(device, output_format.format),
			hardware_ycbcr: Vec::new(),
			presentation_yuv: None,
			bitstream_size_alignment: limits.min_bitstream_size_alignment(),
			coded_extent,
			visible_extent: coded_extent,
			visible_origin: [0, 0],
			decode_recorded: false,
			readback_recorded: false,
			device: device.clone(),
			profile,
			h264_dpb_state,
			h265_dpb_state,
			av1_dpb_state,
			vp9_dpb_state,
			released_output_slot: None,
			pending_decode_acquire_slot: None,
		};
		if physical.video.decode_result_status_queries {
			session.result_status_pool = create_decode_status_pool(&session.device, vk_profile)?;
		}
		let requirements = session.memory_requirements()?;
		let mut bindings = Vec::new();
		bindings
			.try_reserve_exact(requirements.len())
			.map_err(|_| Error::resource_exhausted("video-session binding allocation failed"))?;
		session
			.allocations
			.try_reserve_exact(requirements.len())
			.map_err(|_| Error::resource_exhausted("video-session memory ownership allocation failed"))?;
		for (index, requirement) in requirements.iter().enumerate() {
			if requirement.memory_requirements.size == 0
				|| requirement.memory_requirements.alignment == 0
				|| requirement.memory_requirements.memory_type_bits == 0
				|| requirements[..index]
					.iter()
					.any(|prior| prior.memory_bind_index == requirement.memory_bind_index)
			{
				return Err(Error::backend_failure(
					"Vulkan",
					"video-session memory requirements",
					std::io::Error::other(
						"driver returned an invalid or duplicate memory-binding requirement",
					),
				));
			}
			let allocation_info = vk::mem::AllocationCreateInfo::new()
				.set_usage(vk::mem::MemoryUsage::Explicit)
				.set_preferred_flags(vk::MemoryPropertyFlagBits::DEVICE_LOCAL_BIT)
				.set_memory_type_bits(requirement.memory_requirements.memory_type_bits);
			// SAFETY: the requirements were returned for this live session, and VMA
			// constrains selection to the driver's advertised memory-type mask.
			let allocation = unsafe {
				session.device.allocator().with_allocator(|a| {
					a.allocate(
						&vk::mem::AllocRequest::new(
							requirement.memory_requirements.size,
							requirement.memory_requirements.alignment,
							requirement.memory_requirements.memory_type_bits,
						)
						.set_required_flags(allocation_info.required_flags)
						.set_preferred_flags(allocation_info.preferred_flags),
					)
				})
			}
			.map_err(|source| {
				Error::backend_failure("Vulkan", "video-session memory allocation", source)
			})?
			.map_err(|source| {
				Error::backend_failure("Vulkan", "video-session memory allocation", source)
			})?;
			session.allocations.push(allocation);
			let allocation = session
				.allocations
				.last()
				.ok_or_else(|| Error::internal("video session allocation missing"))?;
			let info = unsafe {
				session
					.device
					.allocator()
					.with_allocator(|a| a.allocation_info(allocation))
			}
			.map_err(Error::from)?
			.map_err(Error::from)?;
			if !info
				.offset
				.is_multiple_of(requirement.memory_requirements.alignment)
				|| requirement.memory_requirements.size > info.size
			{
				return Err(Error::backend_failure(
					"Vulkan",
					"video-session memory validation",
					std::io::Error::other("allocator returned an incompatible memory range"),
				));
			}
			bindings.push(
				vk::BindVideoSessionMemoryInfoKHR::new()
					.set_memory_bind_index(requirement.memory_bind_index)
					.set_memory(info.device_memory)
					.set_memory_offset(info.offset)
					.set_memory_size(requirement.memory_requirements.size),
			);
		}
		if !bindings.is_empty() {
			// SAFETY: every binding corresponds one-to-one with a requirement for
			// this session and its allocation remains owned by `session`.
			let result = unsafe {
				session.device.bind_video_session_memory_khr(
					session.handle,
					bindings.len() as u32,
					bindings.as_ptr(),
				)
			};
			if !matches!(result, Ok(vk::Result::SUCCESS)) {
				return Err(Error::backend_dispatch_failure(
					"Vulkan",
					"video-session memory binding",
					result,
				));
			}
		}
		let images = Arc::get_mut(&mut session.images)
			.ok_or_else(|| Error::internal("decode-image storage was shared before initialization"))?;
		if image_set == DecodeImageSet::CoincidentSeparate {
			images
				.try_reserve_exact(max_dpb_slots as usize)
				.map_err(|_| Error::resource_exhausted("decode-image ownership allocation failed"))?;
			for _ in 0..max_dpb_slots {
				images.push(create_video_image(
					device,
					vk_profile,
					output_format,
					coded_extent,
					1,
					output_usage | dpb_usage,
				)?);
			}
		} else if coincident_images {
			images.push(create_video_image(
				device,
				vk_profile,
				output_format,
				coded_extent,
				max_dpb_slots,
				output_usage | dpb_usage,
			)?);
		} else {
			images.push(create_video_image(
				device,
				vk_profile,
				output_format,
				coded_extent,
				max_dpb_slots,
				output_usage,
			)?);
			images.push(create_video_image(
				device,
				vk_profile,
				dpb_format,
				coded_extent,
				max_dpb_slots,
				dpb_usage,
			)?);
		}
		let public_format = pixel_format(output_format.format).ok_or_else(|| {
			Error::missing_capability("decoded output format has no public plane identity")
		})?;
		session.native_leases = Some(Arc::new(NativeFrameLeases::new(
			device,
			session.images.clone(),
			max_dpb_slots,
			public_format,
		)?));
		Ok(session)
	})
}

fn select_image_format<'a>(
	formats: &'a [vk::VideoFormatPropertiesKHR<'static>],
	usage: vk::ImageUsageFlags,
	role: &str,
) -> Result<&'a vk::VideoFormatPropertiesKHR<'static>> {
	formats
		.iter()
		.find(|format| {
			format.image_type == vk::ImageType::_2D
				&& (format.image_usage_flags & usage) == usage
				&& !format
					.image_create_flags
					.contains(vk::ImageCreateFlagBits::DISJOINT_BIT)
		})
		.ok_or_else(|| {
			Error::missing_capability(format!(
				"decode profile exposes no directly allocatable {role} image format"
			))
		})
}

fn create_decode_bitstream(
	device: &Arc<Device>,
	profile: &vk::VideoProfileInfoKHR<'_>,
	capacity: usize,
) -> Result<DecodeBitstream> {
	let range =
		u64::try_from(capacity).map_err(|_| Error::out_of_range("bitstream exceeds DeviceSize"))?;
	let profiles = std::slice::from_ref(profile);
	let mut profile_list = vk::VideoProfileListInfoKHR::new()
		.set_profiles(profiles)
		.map_err(Error::from)?;
	let buffer_info = vk::BufferCreateInfo::new()
		.set_size(range)
		.set_usage(vk::BufferUsageFlagBits::VIDEO_DECODE_SRC_BIT_KHR)
		.set_sharing_mode(vk::SharingMode::EXCLUSIVE)
		.push_next(&mut profile_list)
		.map_err(Error::from)?;
	let allocation_info = vk::mem::AllocationCreateInfo::new()
		.set_host_access(vk::mem::HostAccess::SequentialWrite)
		.set_dedicated_memory(true)
		.set_usage(vk::mem::MemoryUsage::PreferDevice)
		.set_required_flags(vk::MemoryPropertyFlagBits::HOST_VISIBLE_BIT)
		.set_preferred_flags(vk::MemoryPropertyFlagBits::DEVICE_LOCAL_BIT);
	// SAFETY: the exact decode profile remains live in the required buffer pNext
	// chain, the aligned range is nonzero, and VMA binds compatible host-visible
	// memory to the returned buffer.
	let (handle, allocation) = unsafe {
		device
			.allocator()
			.with_allocator(|a| a.create_buffer_with_options(&(buffer_info), &allocation_info))
			.map_err(Error::from)?
	}
	.map_err(|source| Error::backend_failure("Vulkan", "decode-bitstream allocation", source))?;
	Ok(DecodeBitstream {
		device: device.clone(),
		handle,
		allocation,
		capacity,
		payload_len: 0,
		range: 0,
	})
}

impl DecodeBitstream {
	fn write(&mut self, data: &[u8], aligned_len: usize) -> Result<()> {
		if data.len() > aligned_len || aligned_len > self.capacity {
			return Err(Error::internal("decode upload exceeds its allocation"));
		}
		let range = aligned_len as u64;
		let allocator = self.device.allocator();
		// SAFETY: the preceding decode completed before the session permits reuse.
		// The mapped HOST_VISIBLE allocation covers the checked aligned range.
		let mapped = unsafe { allocator.with_allocator(|a| a.map(&self.allocation))?? }.cast::<u8>();
		unsafe {
			memory::copy_streaming_to_ptr(mapped, data.as_ptr(), data.len());
			std::ptr::write_bytes(mapped.add(data.len()), 0, aligned_len - data.len());
		}
		let flush = unsafe { allocator.with_allocator(|a| a.flush(&self.allocation, 0, range)) };
		// SAFETY: balance the successful map even when the flush fails.
		unsafe { allocator.with_allocator(|a| a.unmap(&self.allocation))?? };
		flush??;
		self.payload_len = data.len();
		self.range = range;
		Ok(())
	}
}

impl Drop for DecodeBitstream {
	fn drop(&mut self) {
		// SAFETY: the decoder waits for the exact decode event before replacement.
		unsafe {
			let _ = self
				.device
				.allocator()
				.destroy_buffer(self.handle, &mut self.allocation);
		}
	}
}

impl DecodeSession {
	fn picture_images(&self, slot: u32) -> Result<(&VideoImage, u32, &VideoImage, u32)> {
		match self.image_set {
			DecodeImageSet::CoincidentLayered => {
				let image = self.images.first().ok_or_else(|| {
					Error::failed_precondition("coincident decode image is not initialized")
				})?;
				if slot >= image.array_layers {
					return Err(Error::data_loss("video picture slot exceeds image layers"));
				}
				Ok((image, slot, image, slot))
			}
			DecodeImageSet::CoincidentSeparate => {
				let image = self
					.images
					.get(slot as usize)
					.ok_or_else(|| Error::data_loss("video picture slot exceeds separate decode images"))?;
				Ok((image, 0, image, 0))
			}
			DecodeImageSet::DistinctLayered => {
				let [output, dpb] = self.images.as_slice() else {
					return Err(Error::failed_precondition(
						"distinct decode images are not initialized",
					));
				};
				if slot >= output.array_layers || slot >= dpb.array_layers {
					return Err(Error::data_loss("video picture slot exceeds image layers"));
				}
				Ok((output, slot, dpb, slot))
			}
		}
	}

	pub(in crate::runtime) fn upload_bitstream(&mut self, data: &[u8]) -> Result<()> {
		let range = self.aligned_bitstream_len(data)?;
		if let Some(buffer) = &mut self.bitstream
			&& range <= buffer.capacity
		{
			return buffer.write(data, range);
		}
		let mut buffer = with_decode_profile(self.profile, |profile| {
			create_decode_bitstream(&self.device, profile, range)
		})?;
		buffer.write(data, range)?;
		self.bitstream = Some(buffer);
		Ok(())
	}

	fn aligned_bitstream_len(&self, data: &[u8]) -> Result<usize> {
		if data.is_empty() {
			return Err(Error::invalid_argument(
				"video decode bitstream must not be empty",
			));
		}
		// Session creation rejects zero alignment. Only the packet's aligned
		// range is submitted, even when the backing holds a larger prior packet.
		let alignment = usize::try_from(self.bitstream_size_alignment)
			.map_err(|_| Error::out_of_range("bitstream alignment exceeds usize"))?;
		data
			.len()
			.div_ceil(alignment)
			.checked_mul(alignment)
			.ok_or_else(|| Error::out_of_range("aligned video packet length overflows"))
	}

	pub(in crate::runtime) fn upload_vp9_picture(
		&mut self,
		access_unit: &[u8],
		picture: &video::Vp9Picture,
	) -> Result<usize> {
		let frame = vp9_picture_data(access_unit, picture)?;
		self.upload_bitstream(frame)?;
		Ok(frame.len())
	}

	pub(in crate::runtime) fn record_h264_picture_for_readback(
		&mut self,
		device: &Arc<Device>,
		sps: &video::H264SequenceParameterSet,
		pps: &video::H264PictureParameterSet,
		slice: &video::H264SliceHeader,
	) -> Result<crate::runtime::RecordedCommandBuffer> {
		let unavailable = self.native_unavailable_slots()?;
		self
			.record_h264_picture_impl(device, sps, pps, slice, true, &unavailable)
			.map(|(command, _)| command)
	}

	pub(in crate::runtime) fn record_h264_picture_native(
		&mut self,
		device: &Arc<Device>,
		sps: &video::H264SequenceParameterSet,
		pps: &video::H264PictureParameterSet,
		slice: &video::H264SliceHeader,
	) -> Result<(crate::runtime::RecordedCommandBuffer, u32)> {
		let unavailable = self.native_unavailable_slots()?;
		self.record_h264_picture_impl(device, sps, pps, slice, false, &unavailable)
	}

	fn record_h264_picture_impl(
		&mut self,
		device: &Arc<Device>,
		sps: &video::H264SequenceParameterSet,
		pps: &video::H264PictureParameterSet,
		slice: &video::H264SliceHeader,
		release_for_readback: bool,
		unavailable: &[bool],
	) -> Result<(crate::runtime::RecordedCommandBuffer, u32)> {
		if !device.same_as(&self.device) {
			return Err(Error::invalid_argument(
				"video session and command device differ",
			));
		}
		self.validate_h264_picture(sps, pps, slice)?;
		let bitstream = self
			.bitstream
			.as_ref()
			.ok_or_else(|| Error::failed_precondition("H.264 decode requires an uploaded bitstream"))?;
		let mut next_dpb = self.h264_dpb_state.clone().ok_or_else(|| {
			Error::failed_precondition("H.264 session has no decoded-picture-buffer state")
		})?;
		let plan = next_dpb.plan_in_place(sps, slice, unavailable)?;
		let initialize_images = !self.decode_recorded;
		let pending_acquire = self.pending_decode_acquire_slot;
		let command = device.record_video_decode(|command_buffer| {
			record_h264_picture(
				device,
				command_buffer,
				self,
				sps,
				pps,
				slice,
				self.result_status_pool,
				bitstream,
				&plan,
				initialize_images,
				pending_acquire,
				release_for_readback,
			)
		})?;
		self.h264_dpb_state = Some(next_dpb);
		self.pending_decode_acquire_slot = None;
		if release_for_readback {
			self.released_output_slot = Some(plan.setup_slot);
		}
		self.decode_recorded = true;
		Ok((command, plan.setup_slot))
	}

	fn validate_h264_picture(
		&self,
		sps: &video::H264SequenceParameterSet,
		pps: &video::H264PictureParameterSet,
		slice: &video::H264SliceHeader,
	) -> Result<()> {
		if self.parameters == vk::VideoSessionParametersKHR::default() {
			return Err(Error::failed_precondition(
				"H.264 decode requires session parameters",
			));
		}
		if self.result_status_pool == vk::QueryPool::default() {
			return Err(Error::missing_capability(
				"the selected Vulkan Video decode queue lacks result-status queries",
			));
		}
		if self.released_output_slot.is_some() {
			return Err(Error::failed_precondition(
				"the prior H.264 output must be read back before recording another picture",
			));
		}
		if pps.sequence_parameter_set_id != sps.id || slice.picture_parameter_set_id != pps.id {
			return Err(Error::invalid_argument(
				"H.264 picture parameter-set identities do not match the session",
			));
		}
		if slice.first_macroblock_in_slice != 0 || slice.field_picture {
			return Err(Error::missing_capability(
				"reusable H.264 decode accepts one complete progressive slice per picture",
			));
		}
		Ok(())
	}

	pub(in crate::runtime) fn record_h265_picture_for_readback(
		&mut self,
		device: &Arc<Device>,
		sps: &video::H265SequenceParameterSet,
		pps: &video::H265PictureParameterSet,
		slice: &video::H265SliceHeader,
	) -> Result<crate::runtime::RecordedCommandBuffer> {
		let unavailable = self.native_unavailable_slots()?;
		self
			.record_h265_picture_impl(device, sps, pps, slice, true, &unavailable)
			.map(|(command, _)| command)
	}

	pub(in crate::runtime) fn record_h265_picture_native(
		&mut self,
		device: &Arc<Device>,
		sps: &video::H265SequenceParameterSet,
		pps: &video::H265PictureParameterSet,
		slice: &video::H265SliceHeader,
	) -> Result<(crate::runtime::RecordedCommandBuffer, u32)> {
		let unavailable = self.native_unavailable_slots()?;
		self.record_h265_picture_impl(device, sps, pps, slice, false, &unavailable)
	}

	fn record_h265_picture_impl(
		&mut self,
		device: &Arc<Device>,
		sps: &video::H265SequenceParameterSet,
		pps: &video::H265PictureParameterSet,
		slice: &video::H265SliceHeader,
		release_for_readback: bool,
		unavailable: &[bool],
	) -> Result<(crate::runtime::RecordedCommandBuffer, u32)> {
		if !device.same_as(&self.device) {
			return Err(Error::invalid_argument(
				"video session and command device differ",
			));
		}
		self.validate_h265_picture(sps, pps, slice)?;
		let bitstream = self
			.bitstream
			.as_ref()
			.ok_or_else(|| Error::failed_precondition("H.265 decode requires an uploaded bitstream"))?;
		let mut next_dpb = self.h265_dpb_state.clone().ok_or_else(|| {
			Error::failed_precondition("H.265 session has no decoded-picture-buffer state")
		})?;
		let plan = next_dpb.plan_in_place(sps, slice, unavailable)?;
		let initialize_images = !self.decode_recorded;
		let pending_acquire = self.pending_decode_acquire_slot;
		let command = device.record_video_decode(|command_buffer| {
			record_h265_picture(
				device,
				command_buffer,
				self,
				sps,
				pps,
				slice,
				self.result_status_pool,
				bitstream,
				&plan,
				initialize_images,
				pending_acquire,
				release_for_readback,
			)
		})?;
		self.h265_dpb_state = Some(next_dpb);
		self.pending_decode_acquire_slot = None;
		if release_for_readback {
			self.released_output_slot = Some(plan.setup_slot);
		}
		self.decode_recorded = true;
		Ok((command, plan.setup_slot))
	}

	fn validate_h265_picture(
		&self,
		sps: &video::H265SequenceParameterSet,
		pps: &video::H265PictureParameterSet,
		slice: &video::H265SliceHeader,
	) -> Result<()> {
		if self.parameters == vk::VideoSessionParametersKHR::default() {
			return Err(Error::failed_precondition(
				"H.265 decode requires session parameters",
			));
		}
		if self.result_status_pool == vk::QueryPool::default() {
			return Err(Error::missing_capability(
				"the selected Vulkan Video decode queue lacks result-status queries",
			));
		}
		if self.released_output_slot.is_some() {
			return Err(Error::failed_precondition(
				"the prior H.265 output must be read back before recording another picture",
			));
		}
		if pps.sequence_parameter_set_id != sps.id
			|| slice.picture_parameter_set_id != pps.id
			|| slice.sequence_parameter_set_id != sps.id
			|| slice.video_parameter_set_id != sps.video_parameter_set_id
		{
			return Err(Error::invalid_argument(
				"H.265 picture parameter-set identities do not match the session",
			));
		}
		if !slice.first_slice_segment_in_picture || slice.slice_segment_address != 0 {
			return Err(Error::missing_capability(
				"reusable H.265 qualification accepts one complete slice segment per picture",
			));
		}
		if slice.short_term_current_before_delta_pocs.len() > 8
			|| slice.short_term_current_after_delta_pocs.len() > 8
		{
			return Err(Error::missing_capability(
				"H.265 current reference-picture set exceeds the standard-video list bound",
			));
		}
		Ok(())
	}

	#[allow(dead_code, reason = "consumed by the public AV1 decoder checkpoint")]
	pub(in crate::runtime) fn record_av1_picture_for_readback(
		&mut self,
		device: &Arc<Device>,
		sequence: &video::Av1SequenceHeader,
		frame: &video::Av1FrameHeader,
		frame_header_offset: u32,
		tiles: &video::Av1TileGroup,
	) -> Result<(crate::runtime::RecordedCommandBuffer, u32)> {
		let unavailable = self.native_unavailable_slots()?;
		self.record_av1_picture_impl(
			device,
			sequence,
			frame,
			frame_header_offset,
			tiles,
			true,
			&unavailable,
		)
	}

	#[allow(dead_code, reason = "consumed by the public AV1 decoder checkpoint")]
	pub(in crate::runtime) fn record_av1_picture_native(
		&mut self,
		device: &Arc<Device>,
		sequence: &video::Av1SequenceHeader,
		frame: &video::Av1FrameHeader,
		frame_header_offset: u32,
		tiles: &video::Av1TileGroup,
	) -> Result<(crate::runtime::RecordedCommandBuffer, u32)> {
		let unavailable = self.native_unavailable_slots()?;
		self.record_av1_picture_impl(
			device,
			sequence,
			frame,
			frame_header_offset,
			tiles,
			false,
			&unavailable,
		)
	}

	#[allow(clippy::too_many_arguments)]
	fn record_av1_picture_impl(
		&mut self,
		device: &Arc<Device>,
		sequence: &video::Av1SequenceHeader,
		frame: &video::Av1FrameHeader,
		frame_header_offset: u32,
		tiles: &video::Av1TileGroup,
		release_for_readback: bool,
		unavailable: &[bool],
	) -> Result<(crate::runtime::RecordedCommandBuffer, u32)> {
		if !device.same_as(&self.device) {
			return Err(Error::invalid_argument(
				"video session and command device differ",
			));
		}
		self.validate_av1_picture(sequence, frame)?;
		let bitstream = self
			.bitstream
			.as_ref()
			.ok_or_else(|| Error::failed_precondition("AV1 decode requires an uploaded bitstream"))?;
		let mut next_dpb = self.av1_dpb_state.clone().ok_or_else(|| {
			Error::failed_precondition("AV1 session has no decoded-picture-buffer state")
		})?;
		let plan = next_dpb.plan_in_place(frame, unavailable)?;
		let setup_slot = plan
			.setup_slot
			.ok_or_else(|| Error::internal("coded AV1 picture has no setup slot"))?;
		if crate::EnvFlag::is_set("OA_AV1_TRACE") {
			crate::log_info!(
				crate::LogComponent::VIDEO,
				"av1 picture type={:?} show={} existing={} order_hint={} refresh={:#04x} primary_ref={} setup={} name_slots={:?} refs={:?} gm={:?} tile_offsets={:?} tile_sizes={:?} hdr_off={} payload={}",
				frame.frame_type,
				frame.show_frame,
				frame.show_existing_frame,
				frame.order_hint,
				frame.refresh_frame_flags,
				frame.primary_reference_frame,
				setup_slot,
				plan.reference_name_slot_indices,
				plan
					.active_references
					.iter()
					.map(|(s, _)| s)
					.collect::<Vec<_>>(),
				frame
					.global_motion_types
					.iter()
					.map(|ty| *ty as u8)
					.collect::<Vec<_>>(),
				tiles.tile_offsets,
				tiles.tile_sizes,
				frame_header_offset,
				bitstream.payload_len,
			);
		}
		let initialize_images = !self.decode_recorded;
		let pending_acquire = self.pending_decode_acquire_slot;
		let command = device.record_video_decode(|command_buffer| {
			av1::record_picture(
				device,
				command_buffer,
				self,
				self.result_status_pool,
				sequence,
				frame,
				frame_header_offset,
				tiles,
				bitstream,
				&plan,
				initialize_images,
				pending_acquire,
				release_for_readback,
			)
		})?;
		self.av1_dpb_state = Some(next_dpb);
		self.pending_decode_acquire_slot = None;
		if release_for_readback {
			self.released_output_slot = Some(setup_slot);
		}
		self.decode_recorded = true;
		Ok((command, setup_slot))
	}

	fn validate_av1_picture(
		&self,
		sequence: &video::Av1SequenceHeader,
		frame: &video::Av1FrameHeader,
	) -> Result<()> {
		if self.parameters == vk::VideoSessionParametersKHR::default() {
			return Err(Error::failed_precondition(
				"AV1 decode requires session parameters",
			));
		}
		if self.result_status_pool == vk::QueryPool::default() {
			return Err(Error::missing_capability(
				"the selected Vulkan Video decode queue lacks result-status queries",
			));
		}
		if self.released_output_slot.is_some() {
			return Err(Error::failed_precondition(
				"the prior AV1 output must be read back before recording another picture",
			));
		}
		if frame.show_existing_frame {
			return Err(Error::invalid_argument(
				"show-existing AV1 frames do not record decode commands",
			));
		}
		if sequence.coded_width() != self.coded_extent.width
			|| sequence.coded_height() != self.coded_extent.height
		{
			return Err(Error::invalid_argument(
				"AV1 sequence extent differs from the decode session",
			));
		}
		Ok(())
	}

	pub(in crate::runtime) fn resolve_av1_show_existing_slot(
		&mut self,
		frame: &video::Av1FrameHeader,
	) -> Result<u32> {
		if !frame.show_existing_frame {
			return Err(Error::invalid_argument(
				"AV1 picture is not a show-existing frame",
			));
		}
		let state = self.av1_dpb_state.as_mut().ok_or_else(|| {
			Error::failed_precondition("AV1 session has no decoded-picture-buffer state")
		})?;
		state
			.plan_in_place(frame, &[])?
			.show_existing_slot
			.ok_or_else(|| Error::internal("AV1 show-existing plan has no display slot"))
	}

	pub(in crate::runtime) fn record_vp9_picture_for_readback(
		&mut self,
		device: &Arc<Device>,
		picture: &video::Vp9Picture,
	) -> Result<(crate::runtime::RecordedCommandBuffer, u32)> {
		let unavailable = self.native_unavailable_slots()?;
		self.record_vp9_picture_impl(device, picture, true, &unavailable)
	}

	pub(in crate::runtime) fn record_vp9_picture_native(
		&mut self,
		device: &Arc<Device>,
		picture: &video::Vp9Picture,
	) -> Result<(crate::runtime::RecordedCommandBuffer, u32)> {
		let unavailable = self.native_unavailable_slots()?;
		self.record_vp9_picture_impl(device, picture, false, &unavailable)
	}

	fn record_vp9_picture_impl(
		&mut self,
		device: &Arc<Device>,
		picture: &video::Vp9Picture,
		release_for_readback: bool,
		unavailable: &[bool],
	) -> Result<(crate::runtime::RecordedCommandBuffer, u32)> {
		if !device.same_as(&self.device) {
			return Err(Error::invalid_argument(
				"video session and command device differ",
			));
		}
		self.validate_vp9_picture(picture)?;
		let bitstream = self
			.bitstream
			.as_ref()
			.ok_or_else(|| Error::failed_precondition("VP9 decode requires an uploaded bitstream"))?;
		let mut next_dpb = self.vp9_dpb_state.clone().ok_or_else(|| {
			Error::failed_precondition("VP9 session has no decoded-picture-buffer state")
		})?;
		let plan = next_dpb.plan_in_place(picture, unavailable)?;
		let setup_slot = plan
			.setup_slot
			.ok_or_else(|| Error::internal("coded VP9 picture has no setup slot"))?;
		let initialize_images = !self.decode_recorded;
		let pending_acquire = self.pending_decode_acquire_slot;
		let command = device.record_video_decode(|command_buffer| {
			vp9::record_picture(
				device,
				command_buffer,
				self,
				self.result_status_pool,
				picture,
				bitstream,
				&plan,
				initialize_images,
				pending_acquire,
				release_for_readback,
			)
		})?;
		self.vp9_dpb_state = Some(next_dpb);
		self.pending_decode_acquire_slot = None;
		if release_for_readback {
			self.released_output_slot = Some(setup_slot);
		}
		self.decode_recorded = true;
		Ok((command, setup_slot))
	}

	fn validate_vp9_picture(&self, picture: &video::Vp9Picture) -> Result<()> {
		if self.result_status_pool == vk::QueryPool::default() {
			return Err(Error::missing_capability(
				"the selected Vulkan Video decode queue lacks result-status queries",
			));
		}
		if self.released_output_slot.is_some() {
			return Err(Error::failed_precondition(
				"the prior VP9 output must be read back before recording another picture",
			));
		}
		if picture.show_existing_frame {
			return Err(Error::invalid_argument(
				"show-existing VP9 frames do not record decode commands",
			));
		}
		if picture.frame_width != self.coded_extent.width
			|| picture.frame_height != self.coded_extent.height
		{
			return Err(Error::invalid_argument(
				"VP9 picture extent differs from the decode session",
			));
		}
		Ok(())
	}

	pub(in crate::runtime) fn resolve_vp9_show_existing_slot(
		&mut self,
		picture: &video::Vp9Picture,
	) -> Result<u32> {
		if !picture.show_existing_frame {
			return Err(Error::invalid_argument(
				"VP9 picture is not a show-existing frame",
			));
		}
		let state = self.vp9_dpb_state.as_mut().ok_or_else(|| {
			Error::failed_precondition("VP9 session has no decoded-picture-buffer state")
		})?;
		state
			.plan_in_place(picture, &[])?
			.show_existing_slot
			.ok_or_else(|| Error::internal("VP9 show-existing plan has no display slot"))
	}

	fn native_unavailable_slots(&self) -> Result<Vec<bool>> {
		self
			.native_leases
			.as_ref()
			.ok_or_else(|| Error::failed_precondition("native video lease pool is unavailable"))?
			.unavailable_slots()
	}

	pub(in crate::runtime) fn native_frame(
		&self,
		slot: u32,
		ready: Event,
	) -> Result<NativeDecodedFrame> {
		self
			.native_leases
			.as_ref()
			.ok_or_else(|| Error::failed_precondition("native video lease pool is unavailable"))?
			.lease(slot, ready, self.visible_extent)
	}

	pub(in crate::runtime) fn validate_ownership(&self) -> Result<()> {
		match &self.native_leases {
			Some(leases) => leases.validate_ownership(),
			None => Ok(()),
		}
	}

	pub(in crate::runtime) fn invalidate_ownership(&self) {
		if let Some(leases) = &self.native_leases {
			leases.ownership_lost.store(true, Ordering::Release);
		}
	}

	pub(in crate::runtime) fn decode_extent(&self) -> video::VideoExtent {
		self.coded_extent
	}

	pub(in crate::runtime) fn decode_profile(&self) -> video::VideoDecodeProfile {
		self.profile
	}

	pub(in crate::runtime) fn restart_h265(&mut self) {
		if let Some(state) = &mut self.h265_dpb_state {
			state.restart = true;
		}
	}

	pub(in crate::runtime) fn set_h264_parameters(
		&mut self,
		sps: &video::H264SequenceParameterSet,
		pps: &video::H264PictureParameterSet,
	) -> Result<()> {
		if !matches!(self.profile, video::VideoDecodeProfile::H264 { .. }) {
			return Err(Error::invalid_argument(
				"H.264 parameters require an H.264 video session",
			));
		}
		if self.parameters != vk::VideoSessionParametersKHR::default() {
			return Err(Error::failed_precondition(
				"video session parameters were already created",
			));
		}
		if pps.sequence_parameter_set_id != sps.id {
			return Err(Error::invalid_argument(
				"H.264 PPS does not reference the supplied SPS",
			));
		}
		let std_sps_scaling = sps.scaling_lists.as_ref().map(std_h264_scaling);
		let std_pps_scaling = pps.scaling_lists.as_ref().map(std_h264_scaling);
		let std_hrd = sps.vui.as_ref().map(std_h264_hrd).transpose()?.flatten();
		let std_vui = sps
			.vui
			.as_ref()
			.map(|vui| {
				std_h264_vui(
					vui,
					std_hrd
						.as_ref()
						.map_or(std::ptr::null(), |hrd| hrd as *const _),
				)
			})
			.transpose()?;
		let std_sps = std_h264_sps(
			sps,
			std_sps_scaling
				.as_ref()
				.map_or(std::ptr::null(), |scaling| scaling as *const _),
			std_vui
				.as_ref()
				.map_or(std::ptr::null(), |vui| vui as *const _),
		)?;
		let std_pps = std_h264_pps(
			pps,
			std_pps_scaling
				.as_ref()
				.map_or(std::ptr::null(), |scaling| scaling as *const _),
		)?;
		let std_sp_ss = [std_sps];
		let std_pp_ss = [std_pps];
		let add = vk::VideoDecodeH264SessionParametersAddInfoKHR::new()
			.set_std_sp_ss(&std_sp_ss)
			.map_err(Error::from)?
			.set_std_pp_ss(&std_pp_ss)
			.map_err(Error::from)?;
		let mut codec = vk::VideoDecodeH264SessionParametersCreateInfoKHR::new()
			.set_max_std_sps_count(32)
			.set_max_std_pps_count(256)
			.set_parameters_add_info(&add);
		let create_info = vk::VideoSessionParametersCreateInfoKHR::new()
			.set_video_session(self.handle)
			.push_next(&mut codec);
		let mut parameters = vk::VideoSessionParametersKHR::default();
		// SAFETY: the standard-video records and codec add chain remain live for
		// this call, match the session profile, and Vulkan copies their contents.
		let result = unsafe {
			self.device.create_video_session_parameters_khr(
				&(create_info).map_err(Error::from)?,
				std::ptr::null(),
				&mut parameters,
			)
		};
		if !matches!(result, Ok(vk::Result::SUCCESS)) {
			return Err(Error::backend_dispatch_failure(
				"Vulkan",
				"H.264 video-session parameter creation",
				result,
			));
		}
		self.parameters = parameters;
		(self.visible_origin, self.visible_extent) = sps.progressive_420_crop()?;
		Ok(())
	}

	pub(in crate::runtime) fn set_h265_parameters(
		&mut self,
		vps: &video::H265VideoParameterSet,
		sps: &video::H265SequenceParameterSet,
		pps: &video::H265PictureParameterSet,
	) -> Result<()> {
		if !matches!(self.profile, video::VideoDecodeProfile::H265 { .. }) {
			return Err(Error::invalid_argument(
				"H.265 parameters require an H.265 video session",
			));
		}
		if sps.chroma_format_idc != 1 || sps.width == 0 || sps.height == 0 {
			return Err(Error::missing_capability(
				"H.265 visible output requires a nonempty 4:2:0 picture",
			));
		}
		if self.parameters != vk::VideoSessionParametersKHR::default() {
			return Err(Error::failed_precondition(
				"video session parameters were already created",
			));
		}
		if sps.video_parameter_set_id != vps.id || pps.sequence_parameter_set_id != sps.id {
			return Err(Error::invalid_argument(
				"H.265 VPS, SPS, and PPS references do not form one parameter chain",
			));
		}
		if vps.hrd_parameters.len() > 1 {
			return Err(Error::missing_capability(
				"StdVideo H.265 VPS lowering supports at most one HRD table",
			));
		}

		let std_vps_profile = std_h265_profile_tier_level(&vps.profile_tier_level)?;
		let std_sps_profile = std_h265_profile_tier_level(&sps.profile_tier_level)?;
		let std_vps_dpb = std_h265_dpb(&vps.decoded_picture_buffer)?;
		let std_sps_dpb = std_h265_dpb(&video::H265DecodedPictureBuffer {
			max_decoded_picture_buffering_minus_1: sps.max_decoded_picture_buffering_minus_1,
			max_num_reorder_pictures: sps.max_num_reorder_pictures,
			max_latency_increase_plus_1: sps.max_latency_increase_plus_1,
		})?;
		let std_vps_hrd = vps
			.hrd_parameters
			.first()
			.map(|value| std_h265_hrd(&value.parameters))
			.transpose()?;
		let std_vui_hrd = sps
			.vui
			.as_ref()
			.and_then(|vui| vui.hrd.as_ref())
			.map(std_h265_hrd)
			.transpose()?;
		let std_vui = sps
			.vui
			.as_ref()
			.map(|vui| {
				std_h265_vui(
					vui,
					std_vui_hrd
						.as_ref()
						.map_or(std::ptr::null(), StdH265Hrd::as_ptr),
				)
			})
			.transpose()?;
		let std_scaling = sps.scaling_lists.as_ref().map(std_h265_scaling);
		let std_pps_scaling = pps.scaling_lists.as_ref().map(std_h265_scaling);
		let std_short_term = sps
			.short_term_reference_picture_sets
			.iter()
			.map(std_h265_short_term_reference_set)
			.collect::<Result<Vec<_>>>()?;
		let std_long_term = std_h265_long_term_references(&sps.long_term_reference_pictures)?;
		let std_vps = std_h265_vps(
			vps,
			&std_vps_profile,
			&std_vps_dpb,
			std_vps_hrd
				.as_ref()
				.map_or(std::ptr::null(), StdH265Hrd::as_ptr),
		)?;
		let std_sps = std_h265_sps(
			sps,
			&std_sps_profile,
			&std_sps_dpb,
			std_scaling
				.as_ref()
				.map_or(std::ptr::null(), |value| value as *const _),
			&std_short_term,
			std_long_term.as_ref(),
			std_vui.as_ref(),
		)?;
		let std_pps = std_h265_pps(
			pps,
			sps.video_parameter_set_id,
			std_pps_scaling
				.as_ref()
				.map_or(std::ptr::null(), |value| value as *const _),
		)?;
		let std_vp_ss = [std_vps];
		let std_sp_ss = [std_sps];
		let std_pp_ss = [std_pps];
		let add = vk::VideoDecodeH265SessionParametersAddInfoKHR::new()
			.set_std_vp_ss(&std_vp_ss)
			.map_err(Error::from)?
			.set_std_sp_ss(&std_sp_ss)
			.map_err(Error::from)?
			.set_std_pp_ss(&std_pp_ss)
			.map_err(Error::from)?;
		let mut codec = vk::VideoDecodeH265SessionParametersCreateInfoKHR::new()
			.set_max_std_vps_count(16)
			.set_max_std_sps_count(16)
			.set_max_std_pps_count(64)
			.set_parameters_add_info(&add);
		let create_info = vk::VideoSessionParametersCreateInfoKHR::new()
			.set_video_session(self.handle)
			.push_next(&mut codec);
		let mut parameters = vk::VideoSessionParametersKHR::default();
		// SAFETY: every referenced StdVideo record and backing array remains live
		// for this call, matches the H.265 session profile, and Vulkan copies it.
		let result = unsafe {
			self.device.create_video_session_parameters_khr(
				&(create_info).map_err(Error::from)?,
				std::ptr::null(),
				&mut parameters,
			)
		};
		if !matches!(result, Ok(vk::Result::SUCCESS)) {
			return Err(Error::backend_dispatch_failure(
				"Vulkan",
				"H.265 video-session parameter creation",
				result,
			));
		}
		self.parameters = parameters;
		self.visible_origin = [sps.conformance_window[0] * 2, sps.conformance_window[2] * 2];
		self.visible_extent = video::VideoExtent {
			width: sps.width,
			height: sps.height,
		};
		Ok(())
	}

	#[allow(dead_code, reason = "wired by the next AV1 frame decode checkpoint")]
	pub(in crate::runtime) fn set_av1_parameters(
		&mut self,
		sequence: &video::Av1SequenceHeader,
	) -> Result<()> {
		let video::VideoDecodeProfile::Av1 {
			profile,
			film_grain_support,
			chroma_subsampling,
			luma_bit_depth,
			chroma_bit_depth,
		} = self.profile
		else {
			return Err(Error::invalid_argument(
				"AV1 parameters require an AV1 video session",
			));
		};
		if self.parameters != vk::VideoSessionParametersKHR::default() {
			return Err(Error::failed_precondition(
				"video session parameters were already created",
			));
		}
		if profile != sequence.profile
			|| chroma_subsampling != sequence.color.chroma_subsampling
			|| luma_bit_depth != sequence.color.bit_depth
			|| chroma_bit_depth != sequence.color.bit_depth
		{
			return Err(Error::invalid_argument(
				"AV1 sequence metadata does not match the decode-session profile",
			));
		}
		if sequence.film_grain_params_present && !film_grain_support {
			return Err(Error::invalid_argument(
				"AV1 sequence permits film grain but the decode-session profile does not",
			));
		}
		if sequence.coded_width() != self.coded_extent.width
			|| sequence.coded_height() != self.coded_extent.height
		{
			return Err(Error::invalid_argument(
				"AV1 sequence coded extent does not match the decode session",
			));
		}

		let color = std_av1_color(&sequence.color);
		let timing = sequence.timing.as_ref().map(std_av1_timing);
		let std_sequence = std_av1_sequence(
			sequence,
			&color,
			timing
				.as_ref()
				.map_or(std::ptr::null(), |value| value as *const _),
		)?;
		let mut codec = vk::VideoDecodeAV1SessionParametersCreateInfoKHR::new()
			.set_std_sequence_header(&std_sequence);
		let create_info = vk::VideoSessionParametersCreateInfoKHR::new()
			.set_video_session(self.handle)
			.push_next(&mut codec);
		let mut parameters = vk::VideoSessionParametersKHR::default();
		// SAFETY: all standard-video records and optional timing/color backing
		// remain live for this call, match the AV1 session profile, and Vulkan
		// copies the sequence header into the parameter object.
		let result = unsafe {
			self.device.create_video_session_parameters_khr(
				&(create_info).map_err(Error::from)?,
				std::ptr::null(),
				&mut parameters,
			)
		};
		if !matches!(result, Ok(vk::Result::SUCCESS)) {
			return Err(Error::backend_dispatch_failure(
				"Vulkan",
				"AV1 video-session parameter creation",
				result,
			));
		}
		self.parameters = parameters;
		Ok(())
	}

	fn memory_requirements(&self) -> Result<Vec<vk::VideoSessionMemoryRequirementsKHR<'static>>> {
		let mut count = 0_u32;
		// SAFETY: the loader and session belong to the retained live device, and
		// a null output pointer requests only the element count.
		let result = unsafe {
			self.device.get_video_session_memory_requirements_khr(
				self.handle,
				&mut count,
				std::ptr::null_mut(),
			)
		};
		if !matches!(result, Ok(vk::Result::SUCCESS)) {
			return Err(Error::backend_dispatch_failure(
				"Vulkan",
				"video-session memory-requirement count query",
				result,
			));
		}
		if count > MAX_VIDEO_SESSION_MEMORY_BINDINGS {
			return Err(Error::resource_exhausted(format!(
				"Vulkan reported {count} video-session bindings, above OA's {MAX_VIDEO_SESSION_MEMORY_BINDINGS} safety bound"
			)));
		}
		let mut requirements = std::iter::repeat_with(vk::VideoSessionMemoryRequirementsKHR::default)
			.take(count as usize)
			.collect::<Vec<_>>();
		if count == 0 {
			return Ok(requirements);
		}
		// SAFETY: the vector contains `count` initialized structures with valid
		// sType values, and Vulkan writes at most the supplied count.
		let result = unsafe {
			self.device.get_video_session_memory_requirements_khr(
				self.handle,
				&mut count,
				requirements.as_mut_ptr(),
			)
		};
		if !matches!(result, Ok(vk::Result::SUCCESS)) {
			return Err(Error::backend_dispatch_failure(
				"Vulkan",
				"video-session memory-requirement enumeration",
				result,
			));
		}
		if count as usize > requirements.len() {
			return Err(Error::backend_failure(
				"Vulkan",
				"video-session memory-requirement enumeration",
				std::io::Error::other("driver increased the binding count during enumeration"),
			));
		}
		requirements.truncate(count as usize);
		Ok(requirements)
	}

	pub(in crate::runtime) fn verify_decode_result(&self) -> Result<()> {
		if !self.decode_recorded {
			return Err(Error::failed_precondition(
				"no picture decode was recorded for this session",
			));
		}
		if self.result_status_pool == vk::QueryPool::default() {
			return Err(Error::missing_capability(
				"the selected Vulkan Video decode queue lacks result-status queries",
			));
		}
		read_decode_status(&self.device, self.result_status_pool)
	}

	pub(in crate::runtime) fn record_decode_readback(
		&mut self,
		device: &Arc<Device>,
	) -> Result<crate::runtime::RecordedCommandBuffer> {
		if !device.same_as(&self.device) {
			return Err(Error::invalid_argument(
				"video session and readback device differ",
			));
		}
		let slot = self
			.released_output_slot
			.ok_or_else(|| Error::failed_precondition("no decoded output is awaiting readback"))?;
		let (format, output_layer) = {
			let (output, output_layer, _, _) = self.picture_images(slot)?;
			(output.format, output_layer)
		};
		let (readback_size, regions) = yuv420_copy_layout(
			format,
			self.visible_extent,
			output_layer,
			self.visible_origin,
		)?;
		if self.readback.is_none() {
			self.readback = Some(crate::runtime::Buffer::host_visible_storage(
				device,
				readback_size,
			)?);
		}
		let (output, _, _, _) = self.picture_images(slot)?;
		let readback = self.readback.as_ref().ok_or_else(|| {
			Error::failed_precondition("video decode readback buffer is not initialized")
		})?;
		let decode_family = self
			.device
			.physical()
			.video
			.decode_queue_family
			.ok_or_else(|| Error::missing_capability("no Vulkan Video decode queue is enabled"))?;
		let compute_family = self.device.physical().compute_queue_family;
		let decoded_layout = if self.image_set == DecodeImageSet::DistinctLayered {
			vk::ImageLayout::VIDEO_DECODE_DST_KHR
		} else {
			vk::ImageLayout::VIDEO_DECODE_DPB_KHR
		};
		let command = device.record_compute_commands(|command_buffer| {
			record_decode_readback_round_trip(
				device,
				command_buffer,
				output,
				output_layer,
				readback,
				&regions,
				(decode_family, compute_family),
				decoded_layout,
			)
		})?;
		self.released_output_slot = None;
		self.pending_decode_acquire_slot = (decode_family != compute_family).then_some(slot);
		self.readback_recorded = true;
		Ok(command)
	}

	pub(in crate::runtime) fn retain_replay_planes(&self, slot: u32) -> Result<DecodedPlaneCopy> {
		DecodedPlaneCopy::for_replay(self, slot)
	}

	pub(in crate::runtime) fn retain_plane_copy(
		&self,
		frame: &NativeDecodedFrame,
	) -> Result<DecodedPlaneCopy> {
		DecodedPlaneCopy::new(self, frame)
	}

	pub(in crate::runtime) fn record_native_readback(
		&mut self,
		device: &Arc<Device>,
		frame: &NativeDecodedFrame,
	) -> Result<RecordedBatch> {
		let NativeCanvasTransfer {
			commands, planes, ..
		} = self.record_native_plane_transfer(device, frame, false)?;
		self.readback = Some(planes);
		self.readback_recorded = true;
		Ok(commands)
	}

	pub(in crate::runtime) fn record_native_plane_transfer(
		&self,
		device: &Arc<Device>,
		frame: &NativeDecodedFrame,
		for_canvas: bool,
	) -> Result<NativeCanvasTransfer> {
		if !device.same_as(&self.device) {
			return Err(Error::invalid_argument(
				"video session and readback device differ",
			));
		}
		let leases = self
			.native_leases
			.as_ref()
			.ok_or_else(|| Error::failed_precondition("native video lease pool is unavailable"))?;
		if !frame.belongs_to(leases) {
			return Err(Error::invalid_argument(
				"native video frame belongs to another decoder",
			));
		}
		let slot = frame.slot();
		leases.validate_live_frame(slot)?;
		let (format, output_layer) = {
			let (output, output_layer, _, _) = self.picture_images(slot)?;
			(output.format, output_layer)
		};
		let (readback_size, regions) = yuv420_copy_layout(
			format,
			self.visible_extent,
			output_layer,
			self.visible_origin,
		)?;
		let planes = if let Some(buffer) = self.readback.as_ref().filter(|_| !for_canvas) {
			buffer.clone()
		} else {
			crate::runtime::Buffer::host_visible_storage(device, readback_size)?
		};
		let sample_bits = presentation_sample_bits(format)?;
		if for_canvas && (!matches!(sample_bits, 10 | 12) || u32::try_from(readback_size).is_err()) {
			return Err(Error::missing_capability(
				"video canvas plane layout exceeds the raster shader ABI",
			));
		}
		let cb_offset = regions[1].buffer_offset;
		let cr_offset = regions
			.get(2)
			.map_or(cb_offset, |region| region.buffer_offset);
		let layout = video::CanvasYuvLayout {
			extent: self.visible_extent,
			format: frame.format(),
			sample_bits,
			cb_offset: u32::try_from(cb_offset)
				.map_err(|_| Error::out_of_range("canvas chroma offset exceeds u32"))?,
			cr_offset: u32::try_from(cr_offset)
				.map_err(|_| Error::out_of_range("canvas chroma offset exceeds u32"))?,
			interleaved: regions.len() == 2,
		};
		let (output, _, _, _) = self.picture_images(slot)?;
		let decode_family = self
			.device
			.physical()
			.video
			.decode_queue_family
			.ok_or_else(|| Error::missing_capability("no Vulkan Video decode queue is enabled"))?;
		let compute_family = self.device.physical().compute_queue_family;
		let decoded_layout = if self.image_set == DecodeImageSet::DistinctLayered {
			vk::ImageLayout::VIDEO_DECODE_DST_KHR
		} else {
			vk::ImageLayout::VIDEO_DECODE_DPB_KHR
		};
		let mut commands = record_image_consumer(
			device,
			output,
			output_layer,
			decode_family,
			decoded_layout,
			ImageUse::copy(compute_family),
			|command| record_plane_readback(device, command, output, &planes, &regions, for_canvas),
		)?;
		commands.retain(Arc::new((frame.clone(), planes.clone())));
		Ok(NativeCanvasTransfer {
			commands,
			planes,
			layout,
		})
	}

	/// Record the native-plane-to-RGBA8 presentation conversion for one frame.
	///
	/// The frame's completed decode publishes the slot image; the compute-family
	/// submission packs its planes into `presentation_yuv` and dispatches the
	/// embedded `cvt_yuv420_to_rgba8` kernel into a fresh `rgba` buffer. The
	/// image returns to the decoder through the recorded ownership round trip.
	///
	/// `color_space` uses the shared `yuv_to_rgb` indices (0 BT.601, 1 BT.709,
	/// 2 BT.2020) and `full_range` selects the source component range.
	pub(in crate::runtime) fn record_native_presentation(
		&mut self,
		device: &Arc<Device>,
		frame: &NativeDecodedFrame,
		color_space: u32,
		full_range: bool,
	) -> Result<NativePresentation> {
		if !device.same_as(&self.device) {
			return Err(Error::invalid_argument(
				"video session and presentation device differ",
			));
		}
		let leases = self
			.native_leases
			.as_ref()
			.ok_or_else(|| Error::failed_precondition("native video lease pool is unavailable"))?;
		if !frame.belongs_to(leases) {
			return Err(Error::invalid_argument(
				"native video frame belongs to another decoder",
			));
		}
		let slot = frame.slot();
		leases.validate_live_frame(slot)?;
		let (format, output_layer) = {
			let (output, output_layer, _, _) = self.picture_images(slot)?;
			(output.format, output_layer)
		};
		if self.hardware_ycbcr_supported && !crate::EnvFlag::is_set("OA_DISABLE_VIDEO_YCBCR") {
			return self.record_hardware_presentation(
				device,
				slot,
				output_layer,
				color_space,
				full_range,
				frame,
			);
		}
		let (packed_size, regions) = yuv420_copy_layout(
			format,
			self.visible_extent,
			output_layer,
			self.visible_origin,
		)?;
		let sample_bits = presentation_sample_bits(format)?;
		let uv_layout = if regions.len() == 3 { 1_u32 } else { 0_u32 };
		let uv_offset = regions
			.get(1)
			.map(|region| region.buffer_offset)
			.ok_or_else(|| Error::internal("video chroma plane region is absent"))?;
		let v_offset = regions
			.get(2)
			.map(|region| region.buffer_offset)
			.unwrap_or(uv_offset);
		if self
			.presentation_yuv
			.as_ref()
			.is_none_or(|buffer| buffer.byte_len() < packed_size)
		{
			self.presentation_yuv = Some(crate::runtime::Buffer::host_visible_storage(
				device,
				packed_size,
			)?);
		}
		let extent = self.visible_extent;
		let rgba_len = extent
			.width
			.checked_mul(extent.height)
			.and_then(|pixels| pixels.checked_mul(4))
			.ok_or_else(|| Error::out_of_range("presentation rgba size overflows u32"))
			.and_then(|len| {
				usize::try_from(len)
					.map_err(|_| Error::out_of_range("presentation rgba size exceeds usize"))
			})?;
		let rgba = crate::runtime::Buffer::host_visible_storage(device, rgba_len)?;
		if self.conversion.is_none() {
			self.conversion = Some(Arc::new(create_presentation_pipeline(device)?));
			crate::log_info!(
				crate::LogComponent::VIDEO,
				"video presentation: shader fallback, format {:?}",
				format
			);
		}
		let pipeline = self.conversion.as_ref().ok_or_else(|| {
			Error::internal("video presentation conversion pipeline is not initialized")
		})?;
		let push = PresentationPush {
			src_idx: self
				.presentation_yuv
				.as_ref()
				.ok_or_else(|| Error::internal("video presentation pack buffer is absent"))?
				.descriptor_index(),
			dst_idx: rgba.descriptor_index(),
			src_width: extent.width,
			src_height: extent.height,
			dst_width: extent.width,
			dst_height: extent.height,
			uv_offset: u32::try_from(uv_offset)
				.map_err(|_| Error::out_of_range("presentation chroma offset exceeds u32"))?,
			v_offset: u32::try_from(v_offset)
				.map_err(|_| Error::out_of_range("presentation chroma offset exceeds u32"))?,
			uv_layout,
			color_space,
			full_range: u32::from(full_range),
			sample_bits,
		};
		let push_size = pipeline.push_constant_size();
		if push_size != size_of::<PresentationPush>() as u32 {
			return Err(Error::backend_failure(
				"Vulkan",
				"video presentation push-constant layout",
				std::io::Error::other(format!(
					"reflected push size {push_size} disagrees with host layout {}",
					size_of::<PresentationPush>()
				)),
			));
		}
		let groups_x = extent.width.div_ceil(16);
		let groups_y = extent.height.div_ceil(16);
		if groups_x > pipeline.max_dispatch_group_count()[0]
			|| groups_y > pipeline.max_dispatch_group_count()[1]
		{
			return Err(Error::missing_capability(
				"video presentation dispatch exceeds device workgroup limits",
			));
		}
		let decode_family = self
			.device
			.physical()
			.video
			.decode_queue_family
			.ok_or_else(|| Error::missing_capability("no Vulkan Video decode queue is enabled"))?;
		let compute_family = self.device.physical().compute_queue_family;
		let decoded_layout = if self.image_set == DecodeImageSet::DistinctLayered {
			vk::ImageLayout::VIDEO_DECODE_DST_KHR
		} else {
			vk::ImageLayout::VIDEO_DECODE_DPB_KHR
		};
		let yuv = self
			.presentation_yuv
			.as_ref()
			.ok_or_else(|| Error::internal("video presentation pack buffer is absent"))?;
		let descriptor_set = device.descriptor_set();
		let (output, _, _, _) = self.picture_images(slot)?;
		let mut commands = record_image_consumer(
			device,
			output,
			output_layer,
			decode_family,
			decoded_layout,
			ImageUse::copy(compute_family),
			|command| {
				record_decode_presentation(
					device,
					command,
					output,
					yuv,
					&rgba,
					&regions,
					pipeline,
					descriptor_set,
					&push,
					(groups_x, groups_y),
				)
			},
		)?;
		commands.retain(Arc::new((
			frame.clone(),
			yuv.clone(),
			rgba.clone(),
			pipeline.clone(),
		)));
		Ok(NativePresentation {
			commands,
			rgba,
			extent,
		})
	}

	pub(in crate::runtime) fn read_decode_yuv420(&self) -> Result<video::VideoHostPlanes> {
		if !self.readback_recorded {
			return Err(Error::failed_precondition(
				"picture readback commands have not been recorded",
			));
		}
		let output = self
			.images
			.first()
			.ok_or_else(|| Error::failed_precondition("video decode output image is not initialized"))?;
		let readback = self.readback.as_ref().ok_or_else(|| {
			Error::failed_precondition("video decode readback buffer is not initialized")
		})?;
		let mut raw = vec![0_u8; readback.byte_len()];
		readback.read(0, &mut raw)?;
		normalize_yuv420(output.format, self.visible_extent, raw)
	}
}

fn vp9_picture_data<'a>(access_unit: &'a [u8], picture: &video::Vp9Picture) -> Result<&'a [u8]> {
	let end = picture
		.frame_offset
		.checked_add(picture.frame_size)
		.ok_or_else(|| Error::out_of_range("VP9 picture range overflowed"))?;
	access_unit
		.get(picture.frame_offset..end)
		.ok_or_else(|| Error::invalid_argument("VP9 picture exceeds its access unit"))
}

fn create_decode_status_pool(
	device: &Device,
	profile: &vk::VideoProfileInfoKHR<'_>,
) -> Result<vk::QueryPool> {
	let mut info = vk::QueryPoolCreateInfo::new()
		.set_query_type(vk::QueryType::RESULT_STATUS_ONLY_KHR)
		.set_query_count(1);
	// SAFETY: the complete profile chain matches the session and is consumed
	// during this call. The selected decode family supports status queries.
	info.p_next = std::ptr::from_ref(profile).cast();
	unsafe { device.create_query_pool(&info, None) }.map_err(|source| {
		Error::backend_failure("Vulkan", "video result-status query-pool creation", source)
	})
}

fn read_decode_status(device: &Device, pool: vk::QueryPool) -> Result<()> {
	let mut raw_status = [vk::QueryResultStatusKHR::NOT_READY_KHR.as_raw()];
	// SAFETY: the pool contains one ended result-status query. The caller waits
	// for the decode event before reading this non-waiting one-i32 result.
	unsafe {
		device.get_query_pool_results(
			pool,
			0,
			1,
			std::slice::from_raw_parts_mut(
				raw_status.as_mut_ptr() as *mut u8,
				std::mem::size_of::<i32>(),
			),
			std::mem::size_of::<i32>() as vk::DeviceSize,
			vk::QueryResultFlagBits::WITH_STATUS_BIT_KHR,
		)
	}
	.map_err(|source| Error::backend_failure("Vulkan", "video decode result-status query", source))?;
	let status = vk::QueryResultStatusKHR::from_raw(raw_status[0]);
	if status != vk::QueryResultStatusKHR::COMPLETE_KHR {
		return Err(Error::data_loss(format!(
			"Vulkan video decode completed with operation status {status:?}"
		)));
	}
	Ok(())
}

fn decode_image_subresource(base_array_layer: u32, layer_count: u32) -> vk::ImageSubresourceRange {
	vk::ImageSubresourceRange {
		aspect_mask: vk::ImageAspectFlagBits::COLOR_BIT,
		base_mip_level: 0,
		level_count: 1,
		base_array_layer,
		layer_count,
	}
}

fn decode_image_barrier(
	image: &VideoImage,
	new_layout: vk::ImageLayout,
	destination_access: vk::AccessFlags2,
) -> vk::ImageMemoryBarrier2<'static> {
	vk::ImageMemoryBarrier2::new()
		.set_src_stage_mask(vk::PipelineStageFlagBits2::NONE)
		.set_src_access_mask(vk::AccessFlagBits2::NONE)
		.set_dst_stage_mask(vk::PipelineStageFlagBits2::VIDEO_DECODE_BIT_KHR)
		.set_dst_access_mask(destination_access)
		.set_old_layout(vk::ImageLayout::UNDEFINED)
		.set_new_layout(new_layout)
		.set_src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
		.set_dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
		.set_image(image.handle)
		.set_subresource_range(decode_image_subresource(0, image.array_layers))
}

impl Drop for DecodeSession {
	fn drop(&mut self) {
		if self.result_status_pool != vk::QueryPool::default() {
			// SAFETY: this private session owns the pool. Its submitted qualification
			// command is explicitly completed before session teardown in the test path.
			unsafe {
				let _ = self
					.device
					.destroy_query_pool(self.result_status_pool, None);
			}
			self.result_status_pool = vk::QueryPool::default();
		}
		if self.parameters != vk::VideoSessionParametersKHR::default() {
			// SAFETY: this wrapper owns the parameter object and destroys it before
			// the video session on which it depends.
			unsafe {
				let _ = self
					.device
					.destroy_video_session_parameters_khr(self.parameters, std::ptr::null());
			}
			self.parameters = vk::VideoSessionParametersKHR::default();
		}
		if self.handle != vk::VideoSessionKHR::default() {
			// SAFETY: this wrapper owns the session and destroys it once before its
			// bound allocations and retained device are released.
			unsafe {
				let _ = self
					.device
					.destroy_video_session_khr(self.handle, std::ptr::null());
			}
			self.handle = vk::VideoSessionKHR::default();
		}
		for allocation in &mut self.allocations {
			// SAFETY: session destruction ended every binding use, and this wrapper
			// owns each VMA allocation exactly once.
			let _ = unsafe {
				self
					.device
					.allocator()
					.with_allocator(|a| a.free(allocation))
			};
		}
	}
}
