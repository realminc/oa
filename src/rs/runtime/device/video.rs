use super::NativeFlagBits;
use std::sync::{Arc, Mutex};

use crate::{Error, Event, Result, core::memory, video};

use crate::runtime::{Device, DevicePhysical, Instance};

mod av1;
mod encode;
mod h264;
mod h265;
mod profile;
mod vp9;

pub(in crate::runtime) use encode::{EncodeSession, create_encode_session};
use h264::record_h264_picture;
use h265::record_h265_picture;
use profile::*;
pub(super) use profile::{
	query_decode_capabilities, query_decode_formats, query_encode_capabilities, query_encode_formats,
};

const MAX_VIDEO_FORMATS: u32 = 256;
const MAX_VIDEO_SESSION_MEMORY_BINDINGS: u32 = 64;

pub(in crate::runtime) struct DecodeSession {
	handle: vk::VideoSessionKHR,
	parameters: vk::VideoSessionParametersKHR,
	result_status_pool: vk::QueryPool,
	allocations: Vec<vk::mem::Allocation>,
	images: Arc<DecodeImageStorage>,
	native_leases: Option<Arc<NativeFrameLeases>>,
	image_set: DecodeImageSet,
	bitstream: Option<DecodeBitstream>,
	readback: Option<crate::runtime::Buffer>,
	bitstream_size_alignment: u64,
	coded_extent: video::VideoExtent,
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

struct DecodeImage {
	handle: vk::Image,
	view: vk::ImageView,
	allocation: vk::mem::Allocation,
	array_layers: u32,
	format: vk::Format,
}

struct DecodeImageStorage {
	device: Arc<Device>,
	images: Vec<DecodeImage>,
}

struct NativeFrameLeases {
	device: Arc<Device>,
	_images: Arc<DecodeImageStorage>,
	format: video::VideoPixelFormat,
	extent: video::VideoExtent,
	slots: Mutex<Vec<NativeFrameSlot>>,
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
		images: Arc<DecodeImageStorage>,
		slot_count: u32,
		format: video::VideoPixelFormat,
		extent: video::VideoExtent,
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
			extent,
			slots: Mutex::new(slots),
		})
	}

	fn unavailable_slots(&self) -> Result<Vec<bool>> {
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

	fn lease(self: &Arc<Self>, slot: u32, ready: Event) -> Result<NativeDecodedFrame> {
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
			extent: self.extent,
			ready,
		})
	}

	fn mark_consumed(self: &Arc<Self>, slot: u32, event: &Event) -> Result<()> {
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

impl std::ops::Deref for DecodeImageStorage {
	type Target = [DecodeImage];

	fn deref(&self) -> &Self::Target {
		&self.images
	}
}

impl Drop for DecodeImageStorage {
	fn drop(&mut self) {
		for image in self.images.drain(..) {
			// SAFETY: every frame lease retaining these images has ended. Each view
			// is destroyed before its uniquely owned image and VMA allocation.
			unsafe {
				let _ = self.device.raw().destroy_image_view(image.view, None);
			}
			let mut allocation = image.allocation;
			// SAFETY: no retained view or frame lease remains for this image.
			unsafe {
				let _ = self
					.device
					.allocator()
					.with_allocator(|a| a.destroy_image(image.handle, &mut allocation));
			}
		}
	}
}

struct DecodeBitstream {
	handle: vk::Buffer,
	allocation: vk::mem::Allocation,
	payload_len: usize,
	range: vk::DeviceSize,
}

pub(super) fn create_decode_session(
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
		let output_usage =
			vk::ImageUsageFlagBits::VIDEO_DECODE_DST_BIT_KHR | vk::ImageUsageFlagBits::TRANSFER_SRC_BIT;
		let dpb_usage = vk::ImageUsageFlagBits::VIDEO_DECODE_DPB_BIT_KHR;
		let output_formats: Vec<vk::VideoFormatPropertiesKHR<'_>>;
		let dpb_formats: Vec<vk::VideoFormatPropertiesKHR<'_>>;
		let (output_format, dpb_format, coincident_images) = if limits.dpb_and_output_coincide() {
			let combined_usage = output_usage | dpb_usage;
			output_formats = query_formats(instance.raw(), physical.handle, vk_profile, combined_usage)?;
			let format = select_image_format(&output_formats, combined_usage, "coincident decode")?;
			(format, format, true)
		} else if limits.dpb_and_output_distinct() {
			output_formats = query_formats(instance.raw(), physical.handle, vk_profile, output_usage)?;
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
		let mut handle = vk::VideoSessionKHR::default();
		// SAFETY: the exact profile, supported formats, queried header version,
		// decode queue family, and bounded extent/counts remain live for this call.
		let result = unsafe {
			device
				.raw()
				.create_video_session_khr(&create_info, std::ptr::null(), &mut handle)
		};
		if !matches!(result, Ok(vk::Result::SUCCESS)) {
			return Err(Error::backend_dispatch_failure(
				"Vulkan",
				"video-session creation",
				result,
			));
		}
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
			images: Arc::new(DecodeImageStorage {
				device: device.clone(),
				images: Vec::new(),
			}),
			native_leases: None,
			image_set,
			bitstream: None,
			readback: None,
			bitstream_size_alignment: limits.min_bitstream_size_alignment(),
			coded_extent,
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
			let mut query_profile = vk::VideoProfileInfoKHR::new()
				.set_video_codec_operation(vk_profile.video_codec_operation)
				.set_chroma_subsampling(vk_profile.chroma_subsampling)
				.set_luma_bit_depth(vk_profile.luma_bit_depth)
				.set_chroma_bit_depth(vk_profile.chroma_bit_depth);
			let query_info = vk::QueryPoolCreateInfo::new()
				.set_query_type(vk::QueryType::RESULT_STATUS_ONLY_KHR)
				.set_query_count(1)
				.push_next(&mut query_profile)
				.map_err(Error::from)?;
			// SAFETY: the selected decode queue family reports result-status support,
			// and the complete profile chain is identical to the session profile.
			session.result_status_pool =
				unsafe { session.device.raw().create_query_pool(&query_info, None) }.map_err(|source| {
					Error::backend_failure("Vulkan", "video result-status query-pool creation", source)
				})?;
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
			let info = unsafe {
				session
					.device
					.allocator()
					.with_allocator(|a| a.allocation_info(&allocation))
			}
			.map_err(Error::from)?
			.map_err(Error::from)?;
			if !info
				.offset
				.is_multiple_of(requirement.memory_requirements.alignment)
				|| requirement.memory_requirements.size > info.size
			{
				let mut allocation = allocation;
				// SAFETY: this allocation has not been published or bound.
				let _ = unsafe {
					session
						.device
						.allocator()
						.with_allocator(|a| a.free(&mut allocation))
				};
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
			session.allocations.push(allocation);
		}
		if !bindings.is_empty() {
			// SAFETY: every binding corresponds one-to-one with a requirement for
			// this session and its allocation remains owned by `session`.
			let result = unsafe {
				session.device.raw().bind_video_session_memory_khr(
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
				.images
				.try_reserve_exact(max_dpb_slots as usize)
				.map_err(|_| Error::resource_exhausted("decode-image ownership allocation failed"))?;
			for _ in 0..max_dpb_slots {
				images.images.push(create_decode_image(
					device,
					vk_profile,
					output_format,
					coded_extent,
					1,
					output_usage | dpb_usage,
				)?);
			}
		} else if coincident_images {
			images.images.push(create_decode_image(
				device,
				vk_profile,
				output_format,
				coded_extent,
				max_dpb_slots,
				output_usage | dpb_usage,
			)?);
		} else {
			images.images.push(create_decode_image(
				device,
				vk_profile,
				output_format,
				coded_extent,
				max_dpb_slots,
				output_usage,
			)?);
			images.images.push(create_decode_image(
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
			coded_extent,
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

fn create_decode_image(
	device: &Arc<Device>,
	profile: &vk::VideoProfileInfoKHR<'_>,
	properties: &vk::VideoFormatPropertiesKHR<'_>,
	extent: video::VideoExtent,
	array_layers: u32,
	usage: vk::ImageUsageFlags,
) -> Result<DecodeImage> {
	let profiles = std::slice::from_ref(profile);
	let mut profile_list = vk::VideoProfileListInfoKHR::new()
		.set_profiles(profiles)
		.map_err(Error::from)?;
	let image_info = vk::ImageCreateInfo::new()
		.set_flags(
			properties.image_create_flags & !vk::ImageCreateFlagBits::VIDEO_PROFILE_INDEPENDENT_BIT_KHR,
		)
		.set_image_type(properties.image_type)
		.set_format(properties.format)
		.set_extent(vk::Extent3D {
			width: extent.width,
			height: extent.height,
			depth: 1,
		})
		.set_mip_levels(1)
		.set_array_layers(array_layers)
		.set_samples(vk::SampleCountFlagBits::_1_BIT)
		.set_tiling(properties.image_tiling)
		.set_usage(usage)
		.set_sharing_mode(vk::SharingMode::EXCLUSIVE)
		.set_initial_layout(vk::ImageLayout::UNDEFINED)
		.push_next(&mut profile_list)
		.map_err(Error::from)?;
	let allocation_info = vk::mem::AllocationCreateInfo::new()
		.set_usage(vk::mem::MemoryUsage::PreferDevice)
		.set_required_flags(vk::MemoryPropertyFlagBits::DEVICE_LOCAL_BIT);
	// SAFETY: the exact video profile and queried format properties remain live,
	// the extent/layer count are nonzero and session-bounded, and VMA retains the
	// device/allocator used to bind the returned image allocation.
	let (handle, allocation) = unsafe {
		device
			.allocator()
			.with_allocator(|a| a.create_image_with_options(&(image_info), &allocation_info))
			.map_err(Error::from)?
	}
	.map_err(|source| Error::backend_failure("Vulkan", "decode-image allocation", source))?;
	let view_info = vk::ImageViewCreateInfo::new()
		.set_image(handle)
		.set_view_type(if array_layers == 1 {
			vk::ImageViewType::_2D
		} else {
			vk::ImageViewType::_2D_ARRAY
		})
		.set_format(properties.format)
		.set_components(properties.component_mapping)
		.set_subresource_range(vk::ImageSubresourceRange {
			aspect_mask: vk::ImageAspectFlagBits::COLOR_BIT,
			base_mip_level: 0,
			level_count: 1,
			base_array_layer: 0,
			layer_count: array_layers,
		});
	// SAFETY: `handle` is a live 2D image with the queried format, and the view
	// covers its sole mip and complete valid layer range.
	let view = match unsafe { device.raw().create_image_view(&view_info, None) } {
		Ok(view) => view,
		Err(source) => {
			let mut allocation = allocation;
			// SAFETY: image-view publication failed, so this uniquely owned image and
			// allocation can be destroyed immediately.
			unsafe {
				device
					.allocator()
					.with_allocator(|a| a.destroy_image(handle, &mut allocation))??;
			}
			return Err(Error::backend_failure(
				"Vulkan",
				"decode-image view creation",
				source,
			));
		}
	};
	Ok(DecodeImage {
		handle,
		view,
		allocation,
		array_layers,
		format: properties.format,
	})
}

fn create_decode_bitstream(
	device: &Arc<Device>,
	profile: &vk::VideoProfileInfoKHR<'_>,
	data: &[u8],
	alignment: u64,
) -> Result<DecodeBitstream> {
	let alignment = usize::try_from(alignment)
		.map_err(|_| Error::missing_capability("bitstream alignment exceeds host address space"))?;
	if alignment == 0 {
		return Err(Error::backend_failure(
			"Vulkan",
			"decode-bitstream alignment",
			std::io::Error::other("driver returned a zero size alignment"),
		));
	}
	let range_len = data
		.len()
		.div_ceil(alignment)
		.checked_mul(alignment)
		.ok_or_else(|| Error::invalid_argument("aligned video packet length overflows"))?;
	let range = u64::try_from(range_len)
		.map_err(|_| Error::invalid_argument("aligned video packet length exceeds DeviceSize"))?;
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
	let (handle, mut allocation) = unsafe {
		device
			.allocator()
			.with_allocator(|a| a.create_buffer_with_options(&(buffer_info), &allocation_info))
			.map_err(Error::from)?
	}
	.map_err(|source| Error::backend_failure("Vulkan", "decode-bitstream allocation", source))?;
	let allocator = device.allocator();
	// SAFETY: VMA selected HOST_VISIBLE memory and this allocation is uniquely
	// owned until the initialized bitstream is returned.
	let mapped = match unsafe { allocator.with_allocator(|a| a.map(&allocation)) } {
		Ok(mapped) => mapped,
		Err(source) => {
			// SAFETY: mapping failed before publication of this uniquely owned pair.
			unsafe {
				allocator.with_allocator(|a| a.destroy_buffer(handle, &mut allocation))??;
			}
			return Err(Error::backend_failure(
				"Vulkan",
				"decode-bitstream mapping",
				source,
			));
		}
	};
	// SAFETY: the VMA allocation contains `range_len` mapped bytes. The source is
	// an independent host slice; the aligned tail is initialized to zero so the
	// driver never observes stale allocation contents inside `srcBufferRange`.
	unsafe {
		let mapped_ptr = mapped.map_err(Error::from)? as *mut u8;
		memory::copy_streaming_to_ptr(mapped_ptr, data.as_ptr(), data.len());
		std::ptr::write_bytes(mapped_ptr.add(data.len()), 0, range_len - data.len());
	}
	let flush: std::result::Result<(), Error> =
		unsafe { allocator.with_allocator(|a| a.flush(&allocation, 0, range)) }
			.map_err(Error::from)
			.and_then(|result| result.map_err(Error::from));
	// SAFETY: this exactly balances the successful map after the final host write.
	unsafe {
		allocator.with_allocator(|a| a.unmap(&allocation))??;
	}
	if let Err(source) = flush {
		// SAFETY: upload failed before publication of this uniquely owned pair.
		unsafe {
			allocator.with_allocator(|a| a.destroy_buffer(handle, &mut allocation))??;
		}
		return Err(Error::backend_failure(
			"Vulkan",
			"decode-bitstream flush",
			source,
		));
	}
	Ok(DecodeBitstream {
		handle,
		allocation,
		payload_len: data.len(),
		range,
	})
}

fn yuv420_8bit_copy_layout(
	format: vk::Format,
	extent: video::VideoExtent,
	array_layer: u32,
) -> Result<(usize, Vec<vk::BufferImageCopy>)> {
	let width = u64::from(extent.width);
	let height = u64::from(extent.height);
	let chroma_width = width.div_ceil(2);
	let chroma_height = height.div_ceil(2);
	let luma_bytes = width
		.checked_mul(height)
		.ok_or_else(|| Error::out_of_range("video luma readback size overflows u64"))?;
	let chroma_plane_bytes = chroma_width
		.checked_mul(chroma_height)
		.ok_or_else(|| Error::out_of_range("video chroma readback size overflows u64"))?;
	let total_bytes = luma_bytes
		.checked_add(
			chroma_plane_bytes
				.checked_mul(2)
				.ok_or_else(|| Error::out_of_range("video chroma size overflows u64"))?,
		)
		.ok_or_else(|| Error::out_of_range("video readback size overflows u64"))?;
	let layer = |aspect_mask| vk::ImageSubresourceLayers {
		aspect_mask,
		mip_level: 0,
		base_array_layer: array_layer,
		layer_count: 1,
	};
	let region =
		|offset, aspect_mask, plane_width: u64, plane_height: u64| -> Result<vk::BufferImageCopy> {
			Ok(
				vk::BufferImageCopy::new()
					.set_buffer_offset(offset)
					.set_buffer_row_length(0)
					.set_buffer_image_height(0)
					.set_image_subresource(layer(aspect_mask))
					.set_image_extent(vk::Extent3D {
						width: u32::try_from(plane_width)
							.map_err(|_| Error::out_of_range("video plane width exceeds u32"))?,
						height: u32::try_from(plane_height)
							.map_err(|_| Error::out_of_range("video plane height exceeds u32"))?,
						depth: 1,
					}),
			)
		};
	let mut regions = vec![region(
		0,
		vk::ImageAspectFlagBits::PLANE_0_BIT,
		width,
		height,
	)?];
	match format {
		vk::Format::G8_B8R8_2PLANE_420_UNORM => {
			regions.push(region(
				luma_bytes,
				vk::ImageAspectFlagBits::PLANE_1_BIT,
				chroma_width,
				chroma_height,
			)?);
		}
		vk::Format::G8_B8_R8_3PLANE_420_UNORM => {
			regions.push(region(
				luma_bytes,
				vk::ImageAspectFlagBits::PLANE_1_BIT,
				chroma_width,
				chroma_height,
			)?);
			regions.push(region(
				luma_bytes + chroma_plane_bytes,
				vk::ImageAspectFlagBits::PLANE_2_BIT,
				chroma_width,
				chroma_height,
			)?);
		}
		_ => {
			return Err(Error::missing_capability(
				"first-picture readback requires an 8-bit 4:2:0 output format",
			));
		}
	}
	Ok((
		usize::try_from(total_bytes)
			.map_err(|_| Error::resource_exhausted("video readback exceeds usize"))?,
		regions,
	))
}

fn normalize_yuv420_8bit(
	format: vk::Format,
	extent: video::VideoExtent,
	raw: Vec<u8>,
) -> Result<Vec<u8>> {
	if format == vk::Format::G8_B8_R8_3PLANE_420_UNORM {
		return Ok(raw);
	}
	if format != vk::Format::G8_B8R8_2PLANE_420_UNORM {
		return Err(Error::missing_capability(
			"first-picture normalization requires an 8-bit 4:2:0 output format",
		));
	}
	let luma_bytes = usize::try_from(u64::from(extent.width) * u64::from(extent.height))
		.map_err(|_| Error::resource_exhausted("video luma readback exceeds usize"))?;
	let chroma_plane_bytes = raw
		.len()
		.checked_sub(luma_bytes)
		.ok_or_else(|| Error::data_loss("video readback is shorter than its luma plane"))?
		/ 2;
	if raw.len() != luma_bytes + chroma_plane_bytes * 2 {
		return Err(Error::data_loss(
			"NV12 readback has an odd interleaved chroma byte count",
		));
	}
	let mut planar = vec![0_u8; raw.len()];
	planar[..luma_bytes].copy_from_slice(&raw[..luma_bytes]);
	for (index, pair) in raw[luma_bytes..].as_chunks::<2>().0.iter().enumerate() {
		planar[luma_bytes + index] = pair[0];
		planar[luma_bytes + chroma_plane_bytes + index] = pair[1];
	}
	Ok(planar)
}

fn pack_h264_vulkan_access_unit(access_unit: &[u8]) -> Result<Vec<u8>> {
	let mut coded_slice = None;
	for nal in video::parse_nal_annex_b(access_unit) {
		if matches!(nal.nal_unit_type(), 1 | 5) && coded_slice.replace(nal.payload()).is_some() {
			return Err(Error::missing_capability(
				"the Vulkan H.264 decode path accepts one coded slice per picture",
			));
		}
	}
	let coded_slice = coded_slice
		.ok_or_else(|| Error::invalid_argument("H.264 access unit contains no coded slice"))?;
	let packed_len = coded_slice
		.len()
		.checked_add(3)
		.ok_or_else(|| Error::invalid_argument("H.264 Vulkan slice size overflows usize"))?;
	let mut packed = Vec::new();
	packed
		.try_reserve_exact(packed_len)
		.map_err(|_| Error::resource_exhausted("H.264 Vulkan slice allocation failed"))?;
	// Vulkan Video implementations consume the VCL NAL unit with the canonical
	// three-byte byte-stream prefix used by the Khronos and FFmpeg decode paths.
	packed.extend_from_slice(&[0, 0, 1]);
	packed.extend_from_slice(coded_slice);
	Ok(packed)
}

fn pack_h265_vulkan_access_unit(access_unit: &[u8]) -> Result<Vec<u8>> {
	let mut coded_slice = None;
	for nal in video::parse_nal_annex_b(access_unit) {
		let nal_type = (nal.payload()[0] >> 1) & 0x3f;
		if nal_type < 32 && coded_slice.replace(nal.payload()).is_some() {
			return Err(Error::missing_capability(
				"the Vulkan H.265 decode path accepts one coded slice segment per picture",
			));
		}
	}
	let coded_slice = coded_slice
		.ok_or_else(|| Error::invalid_argument("H.265 access unit contains no coded slice"))?;
	let packed_len = coded_slice
		.len()
		.checked_add(3)
		.ok_or_else(|| Error::invalid_argument("H.265 Vulkan slice size overflows usize"))?;
	let mut packed = Vec::new();
	packed
		.try_reserve_exact(packed_len)
		.map_err(|_| Error::resource_exhausted("H.265 Vulkan slice allocation failed"))?;
	packed.extend_from_slice(&[0, 0, 1]);
	packed.extend_from_slice(coded_slice);
	Ok(packed)
}

impl DecodeSession {
	fn picture_images(&self, slot: u32) -> Result<(&DecodeImage, u32, &DecodeImage, u32)> {
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
				let [output, dpb] = self.images.images.as_slice() else {
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

	fn upload_bitstream(&mut self, data: &[u8]) -> Result<()> {
		if data.is_empty() {
			return Err(Error::invalid_argument(
				"video decode bitstream must not be empty",
			));
		}
		let device = self.device.clone();
		let bitstream = with_decode_profile(self.profile, |profile| {
			create_decode_bitstream(&device, profile, data, self.bitstream_size_alignment)
		})?;
		if let Some(mut previous) = self.bitstream.replace(bitstream) {
			// SAFETY: callers wait for each synchronous decode/readback round trip
			// before replacement; the session uniquely owns this retired allocation.
			unsafe {
				self
					.device
					.allocator()
					.destroy_buffer(previous.handle, &mut previous.allocation)?;
			}
		}
		Ok(())
	}

	pub(in crate::runtime) fn upload_h264_access_unit(
		&mut self,
		access_unit: &[u8],
	) -> Result<usize> {
		let packed = pack_h264_vulkan_access_unit(access_unit)?;
		let packed_len = packed.len();
		self.upload_bitstream(&packed)?;
		Ok(packed_len)
	}

	pub(in crate::runtime) fn upload_h265_access_unit(
		&mut self,
		access_unit: &[u8],
	) -> Result<usize> {
		let packed = pack_h265_vulkan_access_unit(access_unit)?;
		let packed_len = packed.len();
		self.upload_bitstream(&packed)?;
		Ok(packed_len)
	}

	#[allow(dead_code, reason = "consumed by the public AV1 decoder checkpoint")]
	pub(in crate::runtime) fn upload_av1_access_unit(&mut self, access_unit: &[u8]) -> Result<usize> {
		let uploaded_len = access_unit.len();
		self.upload_bitstream(access_unit)?;
		Ok(uploaded_len)
	}

	pub(in crate::runtime) fn upload_vp9_picture(
		&mut self,
		access_unit: &[u8],
		picture: &video::Vp9Picture,
	) -> Result<usize> {
		let end = picture
			.frame_offset
			.checked_add(picture.frame_size)
			.ok_or_else(|| Error::out_of_range("VP9 picture range overflowed"))?;
		let frame = access_unit
			.get(picture.frame_offset..end)
			.ok_or_else(|| Error::invalid_argument("VP9 picture exceeds its access unit"))?;
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
		let bitstream = self
			.bitstream
			.as_ref()
			.ok_or_else(|| Error::failed_precondition("H.264 decode requires an uploaded bitstream"))?;
		let mut next_dpb = self.h264_dpb_state.clone().ok_or_else(|| {
			Error::failed_precondition("H.264 session has no decoded-picture-buffer state")
		})?;
		let plan = next_dpb.plan_with_unavailable(sps, slice, unavailable)?;
		let initialize_images = !self.decode_recorded;
		let pending_acquire = self.pending_decode_acquire_slot;
		let command = device.record_video_decode(|command_buffer| {
			record_h264_picture(
				device.raw(),
				command_buffer,
				self,
				sps,
				pps,
				slice,
				0,
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
		let bitstream = self
			.bitstream
			.as_ref()
			.ok_or_else(|| Error::failed_precondition("H.265 decode requires an uploaded bitstream"))?;
		let mut next_dpb = self.h265_dpb_state.clone().ok_or_else(|| {
			Error::failed_precondition("H.265 session has no decoded-picture-buffer state")
		})?;
		let plan = next_dpb.plan_with_unavailable(sps, slice, unavailable)?;
		let initialize_images = !self.decode_recorded;
		let pending_acquire = self.pending_decode_acquire_slot;
		let command = device.record_video_decode(|command_buffer| {
			record_h265_picture(
				device.raw(),
				command_buffer,
				self,
				sps,
				pps,
				slice,
				0,
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
		let bitstream = self
			.bitstream
			.as_ref()
			.ok_or_else(|| Error::failed_precondition("AV1 decode requires an uploaded bitstream"))?;
		let mut next_dpb = self.av1_dpb_state.clone().ok_or_else(|| {
			Error::failed_precondition("AV1 session has no decoded-picture-buffer state")
		})?;
		let plan = next_dpb.plan_with_unavailable(frame, unavailable)?;
		let setup_slot = plan
			.setup_slot
			.ok_or_else(|| Error::internal("coded AV1 picture has no setup slot"))?;
		let initialize_images = !self.decode_recorded;
		let pending_acquire = self.pending_decode_acquire_slot;
		let command = device.record_video_decode(|command_buffer| {
			av1::record_picture(
				device.raw(),
				command_buffer,
				self,
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
			.plan_with_unavailable(frame, &[])?
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
		let bitstream = self
			.bitstream
			.as_ref()
			.ok_or_else(|| Error::failed_precondition("VP9 decode requires an uploaded bitstream"))?;
		let mut next_dpb = self.vp9_dpb_state.clone().ok_or_else(|| {
			Error::failed_precondition("VP9 session has no decoded-picture-buffer state")
		})?;
		let plan = next_dpb.plan_with_unavailable(picture, unavailable)?;
		let setup_slot = plan
			.setup_slot
			.ok_or_else(|| Error::internal("coded VP9 picture has no setup slot"))?;
		let initialize_images = !self.decode_recorded;
		let pending_acquire = self.pending_decode_acquire_slot;
		let command = device.record_video_decode(|command_buffer| {
			vp9::record_picture(
				device.raw(),
				command_buffer,
				self,
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
			.plan_with_unavailable(picture, &[])?
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
			.lease(slot, ready)
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
			self.device.raw().create_video_session_parameters_khr(
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
			self.device.raw().create_video_session_parameters_khr(
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
			self.device.raw().create_video_session_parameters_khr(
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
			self.device.raw().get_video_session_memory_requirements_khr(
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
			self.device.raw().get_video_session_memory_requirements_khr(
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
		let mut raw_status = [vk::QueryResultStatusKHR::NOT_READY_KHR.as_raw()];
		// SAFETY: the pool contains one ended result-status query. The caller waits
		// for the decode event before reading this non-waiting one-i32 result.
		unsafe {
			self.device.raw().get_query_pool_results(
				self.result_status_pool,
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
		.map_err(|source| {
			Error::backend_failure("Vulkan", "video decode result-status query", source)
		})?;
		let status = vk::QueryResultStatusKHR::from_raw(raw_status[0]);
		if status != vk::QueryResultStatusKHR::COMPLETE_KHR {
			return Err(Error::data_loss(format!(
				"Vulkan video decode completed with operation status {status:?}"
			)));
		}
		Ok(())
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
		let (readback_size, regions) =
			yuv420_8bit_copy_layout(format, self.coded_extent, output_layer)?;
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
				device.raw(),
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

	pub(in crate::runtime) fn record_native_readback(
		&mut self,
		device: &Arc<Device>,
		frame: &NativeDecodedFrame,
	) -> Result<(
		crate::runtime::RecordedCommandBuffer,
		crate::runtime::RecordedCommandBuffer,
		Option<crate::runtime::RecordedCommandBuffer>,
	)> {
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
		let (readback_size, regions) =
			yuv420_8bit_copy_layout(format, self.coded_extent, output_layer)?;
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
		let release = device.record_video_decode(|command_buffer| {
			record_decode_readback_release(
				device.raw(),
				command_buffer,
				output,
				output_layer,
				(decode_family, compute_family),
				decoded_layout,
			)
		})?;
		let copy = device.record_compute_commands(|command_buffer| {
			record_decode_readback_round_trip(
				device.raw(),
				command_buffer,
				output,
				output_layer,
				readback,
				&regions,
				(decode_family, compute_family),
				decoded_layout,
			)
		})?;
		let acquire = (decode_family != compute_family)
			.then(|| {
				device.record_video_decode(|command_buffer| {
					record_decode_readback_acquire(
						device.raw(),
						command_buffer,
						output,
						output_layer,
						(decode_family, compute_family),
						decoded_layout,
					)
				})
			})
			.transpose()?;
		self.readback_recorded = true;
		Ok((release, copy, acquire))
	}

	pub(in crate::runtime) fn read_decode_yuv420(&self) -> Result<Vec<u8>> {
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
		let (size, _) = yuv420_8bit_copy_layout(output.format, self.coded_extent, 0)?;
		let mut raw = vec![0_u8; size];
		readback.read(0, &mut raw)?;
		normalize_yuv420_8bit(output.format, self.coded_extent, raw)
	}
}

fn record_decode_readback_release(
	device: &vk::Device,
	command_buffer: vk::CommandBuffer,
	output: &DecodeImage,
	output_layer: u32,
	queue_families: (u32, u32),
	decoded_layout: vk::ImageLayout,
) -> Result<()> {
	let (decode_family, compute_family) = queue_families;
	let same_family = decode_family == compute_family;
	let release = vk::ImageMemoryBarrier2::new()
		.set_src_stage_mask(vk::PipelineStageFlagBits2::VIDEO_DECODE_BIT_KHR)
		.set_src_access_mask(
			vk::AccessFlagBits2::VIDEO_DECODE_READ_BIT_KHR
				| vk::AccessFlagBits2::VIDEO_DECODE_WRITE_BIT_KHR,
		)
		.set_dst_stage_mask(if same_family {
			vk::PipelineStageFlagBits2::TRANSFER_BIT
		} else {
			vk::PipelineStageFlagBits2::NONE
		})
		.set_dst_access_mask(if same_family {
			vk::AccessFlagBits2::TRANSFER_READ_BIT
		} else {
			vk::AccessFlagBits2::NONE
		})
		.set_old_layout(decoded_layout)
		.set_new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
		.set_src_queue_family_index(if same_family {
			vk::QUEUE_FAMILY_IGNORED
		} else {
			decode_family
		})
		.set_dst_queue_family_index(if same_family {
			vk::QUEUE_FAMILY_IGNORED
		} else {
			compute_family
		})
		.set_image(output.handle)
		.set_subresource_range(decode_image_subresource(output_layer, 1));
	let dependency =
		vk::DependencyInfo::new().set_image_memory_barriers(std::slice::from_ref(&release));
	// SAFETY: the frame's producer event is complete before submission. This
	// publishes its decode writes and transfers the exact retained layer to the
	// compute family when the queue families differ.
	unsafe {
		device
			.cmd_pipeline_barrier2(command_buffer, &(dependency).map_err(Error::from)?)
			.map_err(Error::from)?;
	}
	Ok(())
}

#[allow(clippy::too_many_arguments)]
fn record_decode_readback(
	device: &vk::Device,
	command_buffer: vk::CommandBuffer,
	output: &DecodeImage,
	output_layer: u32,
	readback: &crate::runtime::Buffer,
	regions: &[vk::BufferImageCopy],
	queue_families: (u32, u32),
	decoded_layout: vk::ImageLayout,
) -> Result<()> {
	let (decode_family, compute_family) = queue_families;
	if decode_family != compute_family {
		let acquire = vk::ImageMemoryBarrier2::new()
			.set_src_stage_mask(vk::PipelineStageFlagBits2::NONE)
			.set_src_access_mask(vk::AccessFlagBits2::NONE)
			.set_dst_stage_mask(vk::PipelineStageFlagBits2::TRANSFER_BIT)
			.set_dst_access_mask(vk::AccessFlagBits2::TRANSFER_READ_BIT)
			.set_old_layout(decoded_layout)
			.set_new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
			.set_src_queue_family_index(decode_family)
			.set_dst_queue_family_index(compute_family)
			.set_image(output.handle)
			.set_subresource_range(decode_image_subresource(output_layer, 1));
		let dependency =
			vk::DependencyInfo::new().set_image_memory_barriers(std::slice::from_ref(&acquire));
		// SAFETY: this matches the release recorded on the decode queue. The engine
		// timeline wait orders this acquire after that queue's signal operation.
		unsafe {
			device
				.cmd_pipeline_barrier2(command_buffer, &(dependency).map_err(Error::from)?)
				.map_err(Error::from)?;
		}
	}
	// SAFETY: the output layer is transfer-readable and each tightly packed plane
	// region is in bounds of the retained destination buffer.
	unsafe {
		device.cmd_copy_image_to_buffer(
			command_buffer,
			output.handle,
			vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
			readback.raw(),
			regions,
		)?;
	}
	let host = vk::BufferMemoryBarrier2::new()
		.set_src_stage_mask(vk::PipelineStageFlagBits2::TRANSFER_BIT)
		.set_src_access_mask(vk::AccessFlagBits2::TRANSFER_WRITE_BIT)
		.set_dst_stage_mask(vk::PipelineStageFlagBits2::HOST_BIT)
		.set_dst_access_mask(vk::AccessFlagBits2::HOST_READ_BIT)
		.set_src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
		.set_dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
		.set_buffer(readback.raw())
		.set_offset(0)
		.set_size(readback.size());
	let dependency =
		vk::DependencyInfo::new().set_buffer_memory_barriers(std::slice::from_ref(&host));
	// SAFETY: the barrier publishes every transfer-written byte to the host domain.
	unsafe {
		device
			.cmd_pipeline_barrier2(command_buffer, &(dependency).map_err(Error::from)?)
			.map_err(Error::from)?;
	}
	Ok(())
}

#[allow(clippy::too_many_arguments)]
fn record_decode_readback_round_trip(
	device: &vk::Device,
	command_buffer: vk::CommandBuffer,
	output: &DecodeImage,
	output_layer: u32,
	readback: &crate::runtime::Buffer,
	regions: &[vk::BufferImageCopy],
	queue_families: (u32, u32),
	decoded_layout: vk::ImageLayout,
) -> Result<()> {
	record_decode_readback(
		device,
		command_buffer,
		output,
		output_layer,
		readback,
		regions,
		queue_families,
		decoded_layout,
	)?;
	let (decode_family, compute_family) = queue_families;
	let same_family = decode_family == compute_family;
	let release = vk::ImageMemoryBarrier2::new()
		.set_src_stage_mask(vk::PipelineStageFlagBits2::TRANSFER_BIT)
		.set_src_access_mask(vk::AccessFlagBits2::TRANSFER_READ_BIT)
		.set_dst_stage_mask(if same_family {
			vk::PipelineStageFlagBits2::VIDEO_DECODE_BIT_KHR
		} else {
			vk::PipelineStageFlagBits2::NONE
		})
		.set_dst_access_mask(if same_family {
			vk::AccessFlagBits2::VIDEO_DECODE_READ_BIT_KHR
				| vk::AccessFlagBits2::VIDEO_DECODE_WRITE_BIT_KHR
		} else {
			vk::AccessFlagBits2::NONE
		})
		.set_old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
		.set_new_layout(decoded_layout)
		.set_src_queue_family_index(if same_family {
			vk::QUEUE_FAMILY_IGNORED
		} else {
			compute_family
		})
		.set_dst_queue_family_index(if same_family {
			vk::QUEUE_FAMILY_IGNORED
		} else {
			decode_family
		})
		.set_image(output.handle)
		.set_subresource_range(decode_image_subresource(output_layer, 1));
	let dependency =
		vk::DependencyInfo::new().set_image_memory_barriers(std::slice::from_ref(&release));
	// SAFETY: the transfer read is complete in this command buffer. A different
	// decode family performs the matching acquire after the engine timeline wait.
	unsafe {
		device
			.cmd_pipeline_barrier2(command_buffer, &(dependency).map_err(Error::from)?)
			.map_err(Error::from)?;
	}
	Ok(())
}

fn record_decode_readback_acquire(
	device: &vk::Device,
	command_buffer: vk::CommandBuffer,
	output: &DecodeImage,
	output_layer: u32,
	queue_families: (u32, u32),
	decoded_layout: vk::ImageLayout,
) -> Result<()> {
	let (decode_family, compute_family) = queue_families;
	if decode_family == compute_family {
		return Ok(());
	}
	let acquire = vk::ImageMemoryBarrier2::new()
		.set_src_stage_mask(vk::PipelineStageFlagBits2::NONE)
		.set_src_access_mask(vk::AccessFlagBits2::NONE)
		.set_dst_stage_mask(vk::PipelineStageFlagBits2::VIDEO_DECODE_BIT_KHR)
		.set_dst_access_mask(
			vk::AccessFlagBits2::VIDEO_DECODE_READ_BIT_KHR
				| vk::AccessFlagBits2::VIDEO_DECODE_WRITE_BIT_KHR,
		)
		.set_old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
		.set_new_layout(decoded_layout)
		.set_src_queue_family_index(compute_family)
		.set_dst_queue_family_index(decode_family)
		.set_image(output.handle)
		.set_subresource_range(decode_image_subresource(output_layer, 1));
	let dependency =
		vk::DependencyInfo::new().set_image_memory_barriers(std::slice::from_ref(&acquire));
	// SAFETY: this matches the compute-family release recorded after the copy.
	// The Engine timeline orders this acquire after that submission.
	unsafe {
		device
			.cmd_pipeline_barrier2(command_buffer, &(dependency).map_err(Error::from)?)
			.map_err(Error::from)?;
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
	image: &DecodeImage,
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
		if let Some(mut bitstream) = self.bitstream.take() {
			// SAFETY: no decode submission can exist for this private qualification
			// session, which uniquely owns the buffer/allocation pair.
			unsafe {
				let _ = self
					.device
					.allocator()
					.destroy_buffer(bitstream.handle, &mut bitstream.allocation);
			}
		}
		if self.result_status_pool != vk::QueryPool::default() {
			// SAFETY: this private session owns the pool. Its submitted qualification
			// command is explicitly completed before session teardown in the test path.
			unsafe {
				let _ = self
					.device
					.raw()
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
					.raw()
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
					.raw()
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
