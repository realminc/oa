// Private Vulkan Video encode session — the device half of `VideoEncoder`.
// Donor: oacpp source/cpp/lib/oa/vision/video/encoder/videoEncoder.cpp.

use std::sync::{
	Arc,
	atomic::{AtomicBool, AtomicU64, Ordering},
};

use crate::runtime::RecordedCommandBuffer;
#[cfg(oa_profile = "core")]
use crate::runtime::dna::kernel::{ShaderArtifact, content_ids};
use crate::runtime::{Device, device::logical::pipeline::ComputePipeline};
use crate::{Error, Result, video};

use super::profile::{query_encode_details, query_formats, with_encode_profile};
use super::{
	MAX_VIDEO_SESSION_MEMORY_BINDINGS,
	image::{VideoImage, create_video_image},
};

mod h264;
mod h265;
mod replay;

pub(in crate::runtime) use replay::{
	EncodePlaneCopy, EncodeReceipt, PreparedEncode, RetainedEncode,
};

#[cfg(oa_profile = "core")]
static CVT_RGBA_TO_NV12: ShaderArtifact = ShaderArtifact::embedded(
	include_bytes!(concat!(
		env!("OA_SHADER_DIR"),
		"/video_cvt_rgba_to_nv12.spv"
	)),
	[16, 16, 1],
	content_ids::VIDEO_CVT_RGBA_TO_NV12,
);

#[cfg(oa_profile = "core")]
static CVT_RGBA64_TO_P010: ShaderArtifact = ShaderArtifact::embedded(
	include_bytes!(concat!(
		env!("OA_SHADER_DIR"),
		"/video_cvt_rgba64_to_p010.spv"
	)),
	[16, 16, 1],
	content_ids::VIDEO_CVT_RGBA64_TO_P010,
);

#[cfg(oa_profile = "core")]
fn create_conversion_pipeline(device: &Arc<Device>, wide_input: bool) -> Result<ComputePipeline> {
	device.create_compute_pipeline(if wide_input {
		&CVT_RGBA64_TO_P010
	} else {
		&CVT_RGBA_TO_NV12
	})
}

#[cfg(oa_profile = "compatibility")]
fn ensure_conversion_available() -> Result<()> {
	Err(Error::missing_capability(
		"video encode conversion requires a Core build",
	))
}

#[cfg(oa_profile = "compatibility")]
fn create_conversion_pipeline(_device: &Arc<Device>, _wide_input: bool) -> Result<ComputePipeline> {
	Err(Error::missing_capability(
		"video encode conversion requires a Core build",
	))
}

const ENCODE_BITSTREAM_CAPACITY: u64 = 4 * 1024 * 1024;

/// Source bytes per pixel and the single-plane view formats the conversion
/// kernel writes for one encode profile.
#[derive(Clone, Copy)]
struct ConversionInputLayout {
	bytes_per_pixel: usize,
	plane_view_formats: [vk::Format; 2],
}

/// Eight-bit profiles take packed RGBA8 and write R8/R8G8 (NV12) plane views;
/// ten- and twelve-bit profiles take packed RGBA64 and write R16/R16G16 views,
/// which carry the P010-style most-significant-bit packing the driver queries.
const fn input_layout(profile: video::VideoEncodeProfile) -> ConversionInputLayout {
	let luma_bit_depth = match profile {
		video::VideoEncodeProfile::H264 { luma_bit_depth, .. }
		| video::VideoEncodeProfile::H265 { luma_bit_depth, .. } => luma_bit_depth,
	};
	match luma_bit_depth {
		video::VideoComponentBitDepth::Eight => ConversionInputLayout {
			bytes_per_pixel: 4,
			plane_view_formats: [vk::Format::R8_UNORM, vk::Format::R8G8_UNORM],
		},
		_ => ConversionInputLayout {
			bytes_per_pixel: 8,
			plane_view_formats: [vk::Format::R16_UNORM, vk::Format::R16G16_UNORM],
		},
	}
}

/// Component bit depth minus eight for the std SPS fields.
const fn bit_depth_minus8(depth: video::VideoComponentBitDepth) -> u8 {
	match depth {
		video::VideoComponentBitDepth::Eight => 0,
		video::VideoComponentBitDepth::Ten => 2,
		video::VideoComponentBitDepth::Twelve => 4,
	}
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EncodeCodec {
	H264,
	H265,
}

/// Engine-owned session state shared by every encode slot.
pub(in crate::runtime) struct EncodeSession {
	handle: vk::VideoSessionKHR,
	parameters: vk::VideoSessionParametersKHR,
	allocations: Vec<vk::mem::Allocation>,
	dpb: VideoImage,
	cached_headers: Vec<u8>,
	dpb_slot_capacity: u32,
	codec: EncodeCodec,
	profile: video::VideoEncodeProfile,
	bitstream_alignment: u64,
	visible_extent: video::VideoExtent,
	coded_extent: video::VideoExtent,
	input_format: vk::Format,
	input_pixel_format: video::VideoPixelFormat,
	rate_control_mode: vk::VideoEncodeRateControlModeFlagBitsKHR,
	bitrate: u32,
	peak_bitrate: u32,
	frame_rate: u32,
	gop_size: u32,
	max_b_frames: u32,
	constant_qp: u32,
	query_result_status_supported: bool,
	frame_count: AtomicU64,
	last_keyframe_index: AtomicU64,
	current_gop_frame: AtomicU64,
	rate_control_reset: AtomicBool,
	capture_valid: AtomicBool,
	conversion: ComputePipeline,
	input_bytes_per_pixel: usize,
	device: Arc<Device>,
	slots: Vec<EncodeSlot>,
}

/// One asynchronous encode job: private input picture, staging buffer, bitstream
/// destination, and feedback query. The session and DPB stay shared.
pub(in crate::runtime) struct EncodeSlot {
	input_image: vk::Image,
	input_view: vk::ImageView,
	input_y_view: vk::ImageView,
	input_uv_view: vk::ImageView,
	input_allocation: vk::mem::Allocation,
	input_y_bindless: u32,
	input_uv_bindless: u32,
	rgba_snapshot: crate::runtime::Buffer,
	bitstream: EncodeBitstream,
	query_pool: vk::QueryPool,
	y_bindless_registered: bool,
	uv_bindless_registered: bool,
	input_initialized: AtomicBool,
	bitstream_dirty_end: AtomicU64,
	presentation_timestamp: AtomicU64,
	is_keyframe: AtomicBool,
	pending: AtomicBool,
	device: Arc<Device>,
}

struct EncodeBitstream {
	handle: vk::Buffer,
	allocation: vk::mem::Allocation,
	capacity: u64,
	device: Arc<Device>,
}

#[derive(Clone, Copy, Debug)]
pub(in crate::runtime) struct EncodeFeedback {
	pub(in crate::runtime) bitstream_offset: u32,
	pub(in crate::runtime) bytes_written: u32,
}

pub(in crate::runtime) fn create_encode_session(
	device: &Arc<Device>,
	settings: &video::VideoEncodeConfig,
	plan: video::VideoEncodePlan,
) -> Result<EncodeSession> {
	#[cfg(oa_profile = "compatibility")]
	ensure_conversion_available()?;
	let physical = device.physical();
	let instance = device.instance();
	let details = query_encode_details(instance, physical, settings.profile)?;
	let caps = &details.capabilities;
	let coded_extent = plan.coded_extent();
	let encode_family = physical
		.video
		.encode_queue_family
		.ok_or_else(|| Error::missing_capability("no Vulkan Video encode queue is enabled"))?;
	let codec = match settings.profile {
		video::VideoEncodeProfile::H264 { .. } => EncodeCodec::H264,
		video::VideoEncodeProfile::H265 { .. } => EncodeCodec::H265,
	};
	let conversion_input = input_layout(settings.profile);
	let input_bytes_per_pixel = conversion_input.bytes_per_pixel;
	let wide_input = input_bytes_per_pixel == 8;
	with_encode_profile(settings.profile, |vk_profile| {
		let src_usage = vk::ImageUsageFlagBits::VIDEO_ENCODE_SRC_BIT_KHR;
		let dpb_usage = vk::ImageUsageFlagBits::VIDEO_ENCODE_DPB_BIT_KHR;
		let input_formats = query_formats(instance.raw(), physical.handle, vk_profile, src_usage)?;
		let dpb_formats = query_formats(instance.raw(), physical.handle, vk_profile, dpb_usage)?;
		let input_format = input_formats
			.iter()
			.find(|format| {
				format.image_type == vk::ImageType::_2D
					&& (format.image_usage_flags & src_usage) == src_usage
			})
			.ok_or_else(|| {
				Error::missing_capability(
					"encode profile exposes no directly allocatable input image format",
				)
			})?;
		let dpb_format = dpb_formats
			.iter()
			.find(|format| {
				format.image_type == vk::ImageType::_2D
					&& (format.image_usage_flags & dpb_usage) == dpb_usage
			})
			.ok_or_else(|| {
				Error::missing_capability("encode profile exposes no directly allocatable DPB image format")
			})?;

		// IPP cadence: one reference (the previous P/I) is the minimum useful
		// encoder; the donor defaults to a two-slot ping-pong.
		let requested_dpb = if settings.max_dpb_slots == 0 {
			2
		} else {
			settings.max_dpb_slots
		};
		let dpb_slot_capacity = requested_dpb.min(caps.max_dpb_slots());
		if dpb_slot_capacity == 0 {
			return Err(Error::missing_capability(
				"Vulkan Video encoder reports zero DPB slots",
			));
		}
		if settings.gop_size > 1 && dpb_slot_capacity < 2 {
			return Err(Error::missing_capability(
				"P-frame encoding requires at least two DPB slots",
			));
		}
		let max_active = dpb_slot_capacity.min(caps.max_active_reference_pictures());
		let max_active = if max_active == 0 && caps.max_active_reference_pictures() > 0 {
			1
		} else {
			max_active
		};

		// The encoder never samples its DPB; ENCODE_DPB usage only, layered
		// across the slot ping-pong.
		let dpb = create_video_image(
			device,
			vk_profile,
			dpb_format,
			coded_extent,
			dpb_slot_capacity,
			dpb_usage,
		)?;
		let mut conversion = create_conversion_pipeline(device, wide_input)?;

		let create_info = vk::VideoSessionCreateInfoKHR::new()
			.set_queue_family_index(encode_family)
			.set_video_profile(vk_profile)
			.set_picture_format(input_format.format)
			.set_max_coded_extent(vk::Extent2D {
				width: coded_extent.width,
				height: coded_extent.height,
			})
			.set_reference_picture_format(dpb_format.format)
			.set_max_dpb_slots(dpb_slot_capacity)
			.set_max_active_reference_pictures(max_active)
			.set_std_header_version(&details.std_header_version);
		let mut handle = vk::VideoSessionKHR::default();
		// SAFETY: the exact profile, queried formats, header version, encode queue
		// family, and validated extent/counts remain live for this call.
		let result =
			unsafe { device.create_video_session_khr(&create_info, std::ptr::null(), &mut handle) };
		if !matches!(result, Ok(vk::Result::SUCCESS)) {
			conversion.destroy(device.raw());
			return Err(Error::backend_dispatch_failure(
				"Vulkan",
				"video-encode session creation",
				result,
			));
		}

		let rate_control_mode = match settings.rate_control {
			video::VideoRateControl::ConstantQp => {
				vk::VideoEncodeRateControlModeFlagBitsKHR::DISABLED_BIT_KHR
			}
			video::VideoRateControl::Cbr => vk::VideoEncodeRateControlModeFlagBitsKHR::CBR_BIT_KHR,
			video::VideoRateControl::Vbr => vk::VideoEncodeRateControlModeFlagBitsKHR::VBR_BIT_KHR,
		};

		let mut session = EncodeSession {
			handle,
			parameters: vk::VideoSessionParametersKHR::default(),
			allocations: Vec::new(),
			dpb,
			cached_headers: Vec::new(),
			dpb_slot_capacity,
			codec,
			profile: settings.profile,
			bitstream_alignment: caps.min_bitstream_size_alignment(),
			visible_extent: video::VideoExtent {
				width: settings.width,
				height: settings.height,
			},
			coded_extent,
			input_format: input_format.format,
			input_pixel_format: if wide_input {
				video::VideoPixelFormat::P010
			} else {
				video::VideoPixelFormat::Nv12
			},
			rate_control_mode,
			bitrate: settings.bitrate,
			peak_bitrate: plan.peak_bitrate(),
			frame_rate: settings.frame_rate,
			gop_size: settings.gop_size,
			max_b_frames: settings.max_b_frames,
			constant_qp: settings.constant_qp,
			query_result_status_supported: physical.video.encode_result_status_queries,
			frame_count: AtomicU64::new(0),
			last_keyframe_index: AtomicU64::new(0),
			current_gop_frame: AtomicU64::new(0),
			rate_control_reset: AtomicBool::new(false),
			capture_valid: AtomicBool::new(true),
			conversion,
			input_bytes_per_pixel,
			device: device.clone(),
			slots: Vec::new(),
		};
		let requirements = session.memory_requirements()?;
		if requirements.len() as u32 > MAX_VIDEO_SESSION_MEMORY_BINDINGS {
			return Err(Error::resource_exhausted(format!(
				"Vulkan reported {} video-encode session bindings, above OA's {MAX_VIDEO_SESSION_MEMORY_BINDINGS} safety bound",
				requirements.len()
			)));
		}
		let mut bindings = Vec::new();
		bindings
			.try_reserve_exact(requirements.len())
			.map_err(|_| Error::resource_exhausted("video-encode binding allocation failed"))?;
		session
			.allocations
			.try_reserve_exact(requirements.len())
			.map_err(|_| Error::resource_exhausted("video-encode memory ownership allocation failed"))?;
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
					"video-encode session memory requirements",
					std::io::Error::other(
						"driver returned an invalid or duplicate memory-binding requirement",
					),
				));
			}
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
						.set_required_flags(vk::MemoryPropertyFlags::empty())
						.set_preferred_flags(vk::MemoryPropertyFlagBits::DEVICE_LOCAL_BIT),
					)
				})
			}
			.map_err(|source| {
				Error::backend_failure("Vulkan", "video-encode session memory allocation", source)
			})?
			.map_err(|source| {
				Error::backend_failure("Vulkan", "video-encode session memory allocation", source)
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
					"video-encode session memory validation",
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
					"video-encode session memory binding",
					result,
				));
			}
		}
		session.parameters = match codec {
			EncodeCodec::H264 => h264::create_parameters(&session, settings, plan)?,
			EncodeCodec::H265 => h265::create_parameters(&session, settings, caps)?,
		};
		match codec {
			EncodeCodec::H264 => h264::cache_headers(&mut session)?,
			EncodeCodec::H265 => h265::cache_headers(&mut session)?,
		}
		let slot_count = settings.async_depth.max(1) as usize;
		session
			.slots
			.try_reserve_exact(slot_count)
			.map_err(|_| Error::resource_exhausted("encode-slot allocation failed"))?;
		for _ in 0..slot_count {
			session.slots.push(create_encode_slot(
				device,
				vk_profile,
				input_format,
				conversion_input,
				coded_extent,
				[encode_family, physical.compute_queue_family],
				caps.min_bitstream_size_alignment(),
			)?);
		}
		Ok(session)
	})
}

impl EncodeSession {
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
				"video-encode memory-requirement count query",
				result,
			));
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
				"video-encode memory-requirement enumeration",
				result,
			));
		}
		if count as usize > requirements.len() {
			return Err(Error::backend_failure(
				"Vulkan",
				"video-encode memory-requirement enumeration",
				std::io::Error::other("driver increased the binding count during enumeration"),
			));
		}
		requirements.truncate(count as usize);
		Ok(requirements)
	}

	fn create_parameter_handle(
		&self,
		create: &vk::VideoSessionParametersCreateInfoKHR<'_>,
	) -> Result<vk::VideoSessionParametersKHR> {
		let mut handle = vk::VideoSessionParametersKHR::default();
		// SAFETY: the session, manufactured standard headers, and quality chain
		// remain live for this call; the returned object belongs to this wrapper.
		let result = unsafe {
			self
				.device
				.create_video_session_parameters_khr(create, std::ptr::null(), &mut handle)
		};
		if !matches!(result, Ok(vk::Result::SUCCESS)) {
			return Err(Error::backend_dispatch_failure(
				"Vulkan",
				"video-encode session-parameter creation",
				result,
			));
		}
		Ok(handle)
	}

	// Cache SPS/PPS (AVC) or VPS/SPS/PPS (HEVC) so every IDR packet is
	// self-describing. A driver returning zero bytes is not fatal — the
	// bitstream then needs a sidecar, matching the donor contract.
	fn cache_encoded_headers(
		&mut self,
		get: &vk::VideoEncodeSessionParametersGetInfoKHR<'_>,
	) -> Result<()> {
		let mut feedback = vk::VideoEncodeSessionParametersFeedbackInfoKHR::new();
		let mut size = 0_usize;
		// SAFETY: the session-parameter object is live; null data queries size.
		let result = unsafe {
			self.device.get_encoded_video_session_parameters_khr(
				get,
				&mut feedback,
				&mut size,
				std::ptr::null_mut(),
			)
		};
		if !matches!(result, Ok(vk::Result::SUCCESS)) || size == 0 {
			return Ok(());
		}
		let mut headers = vec![0_u8; size];
		// SAFETY: `headers` is an initialized buffer of the queried byte size.
		let result = unsafe {
			self.device.get_encoded_video_session_parameters_khr(
				get,
				&mut feedback,
				&mut size,
				headers.as_mut_ptr().cast(),
			)
		};
		if matches!(result, Ok(vk::Result::SUCCESS)) {
			headers.truncate(size);
			self.cached_headers = headers;
		}
		Ok(())
	}

	pub(in crate::runtime) const fn coded_extent(&self) -> video::VideoExtent {
		self.coded_extent
	}

	pub(in crate::runtime) const fn visible_extent(&self) -> video::VideoExtent {
		self.visible_extent
	}

	pub(in crate::runtime) fn cached_headers(&self) -> &[u8] {
		&self.cached_headers
	}

	pub(in crate::runtime) fn slot_count(&self) -> usize {
		self.slots.len()
	}

	pub(in crate::runtime) fn slot_pending(&self, index: usize) -> bool {
		self.slots[index].pending.load(Ordering::Acquire)
	}

	pub(in crate::runtime) fn slot_is_keyframe(&self, index: usize) -> bool {
		self.slots[index].is_keyframe.load(Ordering::Acquire)
	}

	pub(in crate::runtime) fn slot_presentation_timestamp(&self, index: usize) -> u64 {
		self.slots[index]
			.presentation_timestamp
			.load(Ordering::Acquire)
	}

	/// Stage one visible-extent RGBA8 frame into the slot's private snapshot.
	pub(in crate::runtime) fn write_rgba(&self, index: usize, data: &[u8]) -> Result<()> {
		let visible = usize::try_from(self.visible_extent.width)
			.ok()
			.and_then(|width| {
				usize::try_from(self.visible_extent.height)
					.ok()
					.and_then(|height| {
						width
							.checked_mul(height)?
							.checked_mul(self.input_bytes_per_pixel)
					})
			})
			.ok_or_else(|| Error::invalid_argument("visible RGBA extent overflows"))?;
		if data.len() != visible {
			return Err(Error::invalid_argument(
				"RGBA input length does not match the visible frame",
			));
		}
		self.slots[index].rgba_snapshot.write(0, data)
	}

	/// Record the RGBA→NV12 conversion for a slot onto the compute queue.
	///
	/// Producer: `write_rgba` host copy (visible via `Buffer::write` flush).
	/// Consumer: this dispatch writes Y/UV plane views (SHADER_STORAGE_WRITE,
	/// GENERAL) then the frame's encode submission transitions the image to
	/// VIDEO_ENCODE_SRC. Cross-queue ordering comes from the engine timeline
	/// serializing the compute submit before the encode submit.
	pub(in crate::runtime) fn record_conversion(
		&self,
		device: &Arc<Device>,
		index: usize,
		visible: video::VideoExtent,
		color_space: u32,
		full_range: bool,
	) -> Result<RecordedCommandBuffer> {
		if !device.same_as(&self.device) {
			return Err(Error::invalid_argument(
				"encode session and conversion device differ",
			));
		}
		let (push, groups_x, groups_y) =
			self.conversion_args(index, visible, color_space, full_range)?;
		let slot = &self.slots[index];
		let pipeline = &self.conversion;
		let descriptor_set = device.descriptor_set();
		let input_image = slot.input_image;
		let was_initialized = slot.input_initialized.load(Ordering::Acquire);
		device.record_compute_commands(|command_buffer| {
			record_rgba_to_nv12(
				device,
				command_buffer,
				pipeline,
				descriptor_set,
				&push,
				input_image,
				was_initialized,
				groups_x,
				groups_y,
			)
		})
	}

	/// Record the conversion into an existing compute-family command buffer.
	/// Retained plans call this from their `VideoEncodeInput` node; the input
	/// image is already `CONCURRENT` across the compute and encode families.
	pub(in crate::runtime) fn record_conversion_into(
		&self,
		device: &Arc<Device>,
		command_buffer: vk::CommandBuffer,
		index: usize,
		visible: video::VideoExtent,
		color_space: u32,
		full_range: bool,
	) -> Result<()> {
		if !device.same_as(&self.device) {
			return Err(Error::invalid_argument(
				"encode session and conversion device differ",
			));
		}
		let (push, groups_x, groups_y) =
			self.conversion_args(index, visible, color_space, full_range)?;
		let slot = &self.slots[index];
		let was_initialized = slot.input_initialized.load(Ordering::Acquire);
		record_rgba_to_nv12(
			device,
			command_buffer,
			&self.conversion,
			device.descriptor_set(),
			&push,
			slot.input_image,
			was_initialized,
			groups_x,
			groups_y,
		)
	}

	fn conversion_args(
		&self,
		index: usize,
		visible: video::VideoExtent,
		color_space: u32,
		full_range: bool,
	) -> Result<(CvtPush, u32, u32)> {
		let slot = &self.slots[index];
		let push = CvtPush {
			rgba_idx: slot.rgba_snapshot.descriptor_index(),
			y_idx: slot.input_y_bindless,
			uv_idx: slot.input_uv_bindless,
			width: visible.width,
			height: visible.height,
			coded_width: self.coded_extent.width,
			coded_height: self.coded_extent.height,
			color_space,
			full_range: u32::from(full_range),
			pad: 0,
		};
		let groups_x = self.coded_extent.width.div_ceil(16);
		let groups_y = self.coded_extent.height.div_ceil(16);
		let push_size = self.conversion.push_constant_size();
		if push_size != size_of::<CvtPush>() as u32 {
			return Err(Error::backend_failure(
				"Vulkan",
				"video conversion push-constant layout",
				std::io::Error::other(format!(
					"reflected push size {push_size} disagrees with host layout {}",
					size_of::<CvtPush>()
				)),
			));
		}
		if groups_x > self.conversion.max_dispatch_group_count()[0]
			|| groups_y > self.conversion.max_dispatch_group_count()[1]
		{
			return Err(Error::missing_capability(
				"encode conversion dispatch exceeds device workgroup limits",
			));
		}
		Ok((push, groups_x, groups_y))
	}

	/// Record one IDR/P frame for a slot onto the video encode queue.
	///
	/// The engine timeline orders this after the slot's conversion submission;
	/// DPB read/write hazards are fenced by the in-command barrier (the donor's
	/// zero-byte Xe workaround) and GOP state advances at submission time.
	pub(in crate::runtime) fn record_frame(
		&self,
		device: &Arc<Device>,
		index: usize,
	) -> Result<RecordedCommandBuffer> {
		if !device.same_as(&self.device) {
			return Err(Error::invalid_argument(
				"encode session and command device differ",
			));
		}
		if self.parameters == vk::VideoSessionParametersKHR::default() {
			return Err(Error::failed_precondition(
				"video encode session parameters were not created",
			));
		}
		// Clear only the range the previous job wrote so a zero-byte feedback
		// fallback stays unambiguous.
		let slot = &self.slots[index];
		let dirty_end = slot.bitstream_dirty_end.swap(0, Ordering::AcqRel);
		if dirty_end > 0 {
			slot.bitstream.clear(dirty_end)?;
		}
		let frame_count = self.frame_count.load(Ordering::Acquire);
		let is_idr = self.current_gop_frame.load(Ordering::Acquire) == 0;
		let reference_depth = self.dpb_slot_capacity.max(1);
		let current_dpb_slot = (frame_count % u64::from(reference_depth)) as u32;
		let previous_dpb_slot = if frame_count == 0 {
			current_dpb_slot
		} else {
			((frame_count - 1) % u64::from(reference_depth)) as u32
		};
		let frame_index = (frame_count - self.last_keyframe_index.load(Ordering::Acquire)) as u32;
		let session = self.handle;
		let parameters = self.parameters;
		let coded = self.coded_extent;
		let dpb_view = self.dpb.view;
		let dpb_image = self.dpb.handle;
		let dpb_layers = self.dpb_slot_capacity;
		let dpb_initialized = self.rate_control_reset.load(Ordering::Acquire);
		let input_image = slot.input_image;
		let input_view = slot.input_view;
		let bitstream = slot.bitstream.handle;
		let bitstream_range = slot.bitstream.capacity;
		let query_pool = slot.query_pool;
		let codec = self.codec;
		let rate_control_mode = self.rate_control_mode;
		let rc_reset = self.rate_control_reset.load(Ordering::Acquire);
		let bitrate = self.bitrate;
		let peak = self.peak_bitrate;
		let frame_rate = self.frame_rate;
		let gop_size = self.gop_size;
		let max_b = self.max_b_frames;
		let constant_qp = self.constant_qp;
		let idr_pic_id = (frame_count & 0xFFFF) as u16;
		let first_frame = frame_count == 0;
		device.record_video_encode(|command_buffer| {
			record_encode_picture(
				device,
				command_buffer,
				EncodePictureArgs {
					session,
					parameters,
					codec,
					coded,
					input_image,
					input_view,
					dpb_image,
					dpb_view,
					dpb_layers,
					dpb_initialized,
					current_dpb_slot,
					previous_dpb_slot,
					is_idr,
					frame_index,
					idr_pic_id,
					first_frame,
					bitstream,
					bitstream_range,
					query_pool,
					rate_control_mode,
					rate_control_reset: rc_reset,
					bitrate,
					peak_bitrate: peak,
					frame_rate,
					gop_size,
					max_b_frames: max_b,
					constant_qp,
					replay_boundary: false,
				},
			)
		})
	}

	pub(in crate::runtime) fn mark_pending(&self, index: usize, pts: u64) {
		let slot = &self.slots[index];
		// GOP state advances at submission — queue order makes the shared
		// reference slot deterministic while older jobs remain unharvested.
		let frame_count = self.frame_count.load(Ordering::Acquire);
		let current_gop_frame = self.current_gop_frame.load(Ordering::Acquire);
		let is_idr = current_gop_frame == 0;
		slot.is_keyframe.store(is_idr, Ordering::Release);
		slot.input_initialized.store(true, Ordering::Release);
		if is_idr {
			self
				.last_keyframe_index
				.store(frame_count, Ordering::Release);
			self
				.current_gop_frame
				.store(u64::from(self.gop_size > 1), Ordering::Release);
		} else {
			let next = current_gop_frame + 1;
			self.current_gop_frame.store(
				if next >= u64::from(self.gop_size) {
					0
				} else {
					next
				},
				Ordering::Release,
			);
		}
		self.frame_count.store(frame_count + 1, Ordering::Release);
		self.rate_control_reset.store(true, Ordering::Release);
		slot.presentation_timestamp.store(pts, Ordering::Release);
		slot.pending.store(true, Ordering::Release);
	}

	pub(in crate::runtime) fn mark_completed(&self, index: usize) {
		self.slots[index].pending.store(false, Ordering::Release);
	}

	/// Read one slot's encode feedback after its submission event completed.
	/// Includes the donor's zero-byte Annex-B recovery for drivers that report
	/// COMPLETE without counters.
	pub(in crate::runtime) fn read_feedback(&self, index: usize) -> Result<EncodeFeedback> {
		self.feedback_at(index, true)
	}

	fn feedback_at(&self, index: usize, wait: bool) -> Result<EncodeFeedback> {
		let slot = &self.slots[index];
		let feedback = read_encode_feedback(
			&self.device,
			slot.query_pool,
			&slot.bitstream,
			self.query_result_status_supported,
			wait,
		)?;
		let dirty_end = u64::from(feedback.bitstream_offset) + u64::from(feedback.bytes_written);
		slot.bitstream_dirty_end.store(dirty_end, Ordering::Release);
		Ok(feedback)
	}

	/// Copy `buf.len()` bytes from a slot's bitstream at `offset` after the
	/// caller has proven the submission complete.
	pub(in crate::runtime) fn read_bitstream(
		&self,
		index: usize,
		offset: u64,
		output: &mut [u8],
	) -> Result<()> {
		self.slots[index].bitstream.read(offset, output)
	}
}

#[repr(C)]
struct CvtPush {
	rgba_idx: u32,
	y_idx: u32,
	uv_idx: u32,
	width: u32,
	height: u32,
	coded_width: u32,
	coded_height: u32,
	color_space: u32,
	full_range: u32,
	pad: u32,
}

fn create_encode_slot(
	device: &Arc<Device>,
	profile: &vk::VideoProfileInfoKHR<'_>,
	format: &vk::VideoFormatPropertiesKHR<'_>,
	conversion_input: ConversionInputLayout,
	coded_extent: video::VideoExtent,
	queue_families: [u32; 2],
	bitstream_size_alignment: u64,
) -> Result<EncodeSlot> {
	let [encode_family, compute_family] = queue_families;
	let profiles = std::slice::from_ref(profile);
	let mut profile_list = vk::VideoProfileListInfoKHR::new()
		.set_profiles(profiles)
		.map_err(Error::from)?;
	let view_formats = [
		format.format,
		conversion_input.plane_view_formats[0],
		conversion_input.plane_view_formats[1],
	];
	let mut format_list = vk::ImageFormatListCreateInfo::new()
		.set_view_formats(&view_formats)
		.map_err(Error::from)?;
	let usage = vk::ImageUsageFlagBits::VIDEO_ENCODE_SRC_BIT_KHR
		| vk::ImageUsageFlagBits::TRANSFER_DST_BIT
		| vk::ImageUsageFlagBits::STORAGE_BIT;
	let families = [compute_family, encode_family];
	let concurrent = compute_family != encode_family;
	let mut image_info = vk::ImageCreateInfo::new()
		.set_flags(
			format.image_create_flags & !vk::ImageCreateFlagBits::VIDEO_PROFILE_INDEPENDENT_BIT_KHR
				| vk::ImageCreateFlagBits::MUTABLE_FORMAT_BIT
				| vk::ImageCreateFlagBits::EXTENDED_USAGE_BIT,
		)
		.set_image_type(format.image_type)
		.set_format(format.format)
		.set_extent(vk::Extent3D {
			width: coded_extent.width,
			height: coded_extent.height,
			depth: 1,
		})
		.set_mip_levels(1)
		.set_array_layers(1)
		.set_samples(vk::SampleCountFlagBits::_1_BIT)
		.set_tiling(format.image_tiling)
		.set_usage(usage)
		.set_sharing_mode(if concurrent {
			vk::SharingMode::CONCURRENT
		} else {
			vk::SharingMode::EXCLUSIVE
		})
		.set_initial_layout(vk::ImageLayout::UNDEFINED)
		.push_next(&mut format_list)
		.map_err(Error::from)?
		.push_next(&mut profile_list)
		.map_err(Error::from)?;
	if concurrent {
		image_info = image_info
			.set_queue_family_indices(&families)
			.map_err(Error::from)?;
	}
	let allocation_info = vk::mem::AllocationCreateInfo::new()
		.set_usage(vk::mem::MemoryUsage::PreferDevice)
		.set_required_flags(vk::MemoryPropertyFlagBits::DEVICE_LOCAL_BIT);
	let rgba_snapshot = crate::runtime::Buffer::host_visible_storage(
		device,
		usize::try_from(coded_extent.width)
			.ok()
			.and_then(|w| {
				usize::try_from(coded_extent.height).ok().and_then(|h| {
					w.checked_mul(h)?
						.checked_mul(conversion_input.bytes_per_pixel)
				})
			})
			.ok_or_else(|| Error::out_of_range("encode RGBA snapshot size overflows"))?,
	)?;
	let bitstream = create_encode_bitstream(
		device,
		profile,
		ENCODE_BITSTREAM_CAPACITY,
		bitstream_size_alignment.max(1),
	)?;
	// SAFETY: the exact profile/format chain stays live; the slot owns the
	// returned image and allocation uniquely until Drop.
	let (input_image, input_allocation) = unsafe {
		device
			.allocator()
			.with_allocator(|a| a.create_image_with_options(&(image_info), &allocation_info))
			.map_err(Error::from)?
	}
	.map_err(|source| Error::backend_failure("Vulkan", "encode-input image allocation", source))?;
	let mut slot = EncodeSlot {
		input_image,
		input_view: vk::ImageView::default(),
		input_y_view: vk::ImageView::default(),
		input_uv_view: vk::ImageView::default(),
		input_allocation,
		input_y_bindless: 0,
		input_uv_bindless: 0,
		rgba_snapshot,
		bitstream,
		query_pool: vk::QueryPool::default(),
		y_bindless_registered: false,
		uv_bindless_registered: false,
		input_initialized: AtomicBool::new(false),
		bitstream_dirty_end: AtomicU64::new(0),
		presentation_timestamp: AtomicU64::new(0),
		is_keyframe: AtomicBool::new(false),
		pending: AtomicBool::new(false),
		device: device.clone(),
	};
	slot.query_pool = create_feedback_pool(device, profile)?;
	slot.create_views(format, conversion_input.plane_view_formats)?;
	Ok(slot)
}

impl EncodeSlot {
	fn create_views(
		&mut self,
		format: &vk::VideoFormatPropertiesKHR<'_>,
		plane_view_formats: [vk::Format; 2],
	) -> Result<()> {
		let mut encode_usage = vk::ImageViewUsageCreateInfo::new()
			.set_usage(vk::ImageUsageFlagBits::VIDEO_ENCODE_SRC_BIT_KHR);
		self.input_view = unsafe {
			self.device.create_image_view(
				&(vk::ImageViewCreateInfo::new()
					.set_image(self.input_image)
					.set_view_type(vk::ImageViewType::_2D)
					.set_format(format.format)
					.set_components(format.component_mapping)
					.set_subresource_range(vk::ImageSubresourceRange {
						aspect_mask: vk::ImageAspectFlagBits::COLOR_BIT,
						base_mip_level: 0,
						level_count: 1,
						base_array_layer: 0,
						layer_count: 1,
					})
					.push_next(&mut encode_usage)
					.map_err(Error::from)?),
				None,
			)
		}
		.map_err(|source| Error::backend_failure("Vulkan", "encode-input view creation", source))?;
		// Each plane view gets its own ImageViewUsageCreateInfo — the usage chain
		// node is consumed by the view create call and cannot be shared.
		let mut y_usage =
			vk::ImageViewUsageCreateInfo::new().set_usage(vk::ImageUsageFlagBits::STORAGE_BIT);
		let mut uv_usage =
			vk::ImageViewUsageCreateInfo::new().set_usage(vk::ImageUsageFlagBits::STORAGE_BIT);
		let plane_view = |aspect: vk::ImageAspectFlags, plane_format: vk::Format| {
			vk::ImageViewCreateInfo::new()
				.set_image(self.input_image)
				.set_view_type(vk::ImageViewType::_2D)
				.set_format(plane_format)
				.set_subresource_range(vk::ImageSubresourceRange {
					aspect_mask: aspect,
					base_mip_level: 0,
					level_count: 1,
					base_array_layer: 0,
					layer_count: 1,
				})
		};
		// SAFETY: MUTABLE_FORMAT + the explicit view-format list admit both
		// single-plane formats for this image.
		self.input_y_view = unsafe {
			self.device.create_image_view(
				&(plane_view(vk::ImageAspectFlagBits::PLANE_0_BIT, plane_view_formats[0])
					.push_next(&mut y_usage)
					.map_err(Error::from)?),
				None,
			)
		}
		.map_err(|source| {
			Error::backend_failure("Vulkan", "encode-input Y-plane view creation", source)
		})?;
		self.input_uv_view = unsafe {
			self.device.create_image_view(
				&(plane_view(vk::ImageAspectFlagBits::PLANE_1_BIT, plane_view_formats[1])
					.push_next(&mut uv_usage)
					.map_err(Error::from)?),
				None,
			)
		}
		.map_err(|source| {
			Error::backend_failure("Vulkan", "encode-input UV-plane view creation", source)
		})?;
		self.input_y_bindless = self.device.bind_storage_image(self.input_y_view)?;
		self.y_bindless_registered = true;
		self.input_uv_bindless = self.device.bind_storage_image(self.input_uv_view)?;
		self.uv_bindless_registered = true;
		Ok(())
	}
}

impl Drop for EncodeSlot {
	fn drop(&mut self) {
		if self.uv_bindless_registered {
			self.device.release_storage_image(self.input_uv_bindless);
		}
		if self.y_bindless_registered {
			self.device.release_storage_image(self.input_y_bindless);
		}
		for view in [self.input_uv_view, self.input_y_view, self.input_view] {
			if view != vk::ImageView::default() {
				// SAFETY: this slot uniquely owns each view on its private image.
				unsafe {
					let _ = self.device.destroy_image_view(view, None);
				}
			}
		}
		if self.query_pool != vk::QueryPool::default() {
			// SAFETY: pending submissions are harvested before session teardown,
			// and this slot owns the pool.
			unsafe {
				let _ = self.device.destroy_query_pool(self.query_pool, None);
			}
		}
		if self.input_image != vk::Image::default() {
			// SAFETY: views are destroyed above; the slot owns the image/allocation.
			unsafe {
				let _ = self
					.device
					.allocator()
					.with_allocator(|a| a.destroy_image(self.input_image, &mut self.input_allocation));
			}
		}
	}
}

fn read_encode_feedback(
	device: &Arc<Device>,
	pool: vk::QueryPool,
	bitstream: &EncodeBitstream,
	status_supported: bool,
	wait: bool,
) -> Result<EncodeFeedback> {
	let flags = if wait {
		vk::QueryResultFlagBits::WAIT_BIT | vk::QueryResultFlagBits::WITH_STATUS_BIT_KHR
	} else {
		vk::QueryResultFlagBits::WITH_STATUS_BIT_KHR
	};
	let (feedback, status) = if status_supported {
		#[repr(C)]
		struct WithStatus {
			values: [u32; 2],
			status: vk::QueryResultStatusKHR,
		}
		let mut result = WithStatus {
			values: [0; 2],
			status: vk::QueryResultStatusKHR::NOT_READY_KHR,
		};
		// SAFETY: the query pool and one query slot are live; the output
		// struct is exactly `2 * u32 + status` as configured at pool create.
		let vk_result = unsafe {
			device.get_query_pool_results(
				pool,
				0,
				1,
				std::slice::from_raw_parts_mut(
					std::ptr::from_mut(&mut result).cast::<u8>(),
					size_of::<WithStatus>(),
				),
				size_of::<WithStatus>() as vk::DeviceSize,
				flags,
			)
		};
		if !matches!(vk_result, Ok(vk::Result::SUCCESS)) {
			if !wait && matches!(vk_result, Ok(vk::Result::NOT_READY)) {
				return Err(Error::not_ready("encoded packet feedback is not ready"));
			}
			return Err(Error::backend_dispatch_failure(
				"Vulkan",
				"video-encode feedback query",
				vk_result,
			));
		}
		(
			EncodeFeedback {
				bitstream_offset: result.values[0],
				bytes_written: result.values[1],
			},
			result.status,
		)
	} else {
		let mut values = [0_u32; 2];
		// SAFETY: the query pool and one query slot are live; the output is
		// the two u32 counters configured at pool creation.
		let vk_result = unsafe {
			device.get_query_pool_results(
				pool,
				0,
				1,
				std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<u8>(), size_of::<[u32; 2]>()),
				size_of::<[u32; 2]>() as vk::DeviceSize,
				if wait {
					vk::QueryResultFlagBits::WAIT_BIT
				} else {
					vk::QueryResultFlagBits::empty()
				},
			)
		};
		if !matches!(vk_result, Ok(vk::Result::SUCCESS)) {
			if !wait && matches!(vk_result, Ok(vk::Result::NOT_READY)) {
				return Err(Error::not_ready("encoded packet feedback is not ready"));
			}
			return Err(Error::backend_dispatch_failure(
				"Vulkan",
				"video-encode feedback query",
				vk_result,
			));
		}
		(
			EncodeFeedback {
				bitstream_offset: values[0],
				bytes_written: values[1],
			},
			vk::QueryResultStatusKHR::COMPLETE_KHR,
		)
	};
	if !wait && status == vk::QueryResultStatusKHR::NOT_READY_KHR {
		return Err(Error::not_ready("encoded packet feedback is not ready"));
	}
	if status != vk::QueryResultStatusKHR::COMPLETE_KHR {
		return Err(Error::data_loss(format!(
			"Vulkan video encode completed with operation status {status:?}"
		)));
	}
	let mut feedback = feedback;
	if feedback.bytes_written == 0 {
		// Recovery: scan the invalidated mapping for the last nonzero byte
		// and the first Annex-B start code, exactly like the donor.
		let capacity = usize::try_from(bitstream.capacity)
			.map_err(|_| Error::out_of_range("bitstream capacity exceeds host address space"))?;
		let mut probe = vec![0_u8; capacity];
		bitstream.read(0, &mut probe)?;
		let nonzero_extent = probe
			.iter()
			.rposition(|byte| *byte != 0)
			.map_or(0, |index| index + 1);
		let mut annex_b_start = nonzero_extent;
		for idx in 0..nonzero_extent.saturating_sub(3) {
			if probe[idx] == 0
				&& probe[idx + 1] == 0
				&& (probe[idx + 2] == 1 || (probe[idx + 2] == 0 && probe[idx + 3] == 1))
			{
				annex_b_start = idx;
				break;
			}
		}
		if annex_b_start >= nonzero_extent {
			return Err(Error::data_loss(
				"Vulkan video encoder produced neither feedback nor an Annex-B payload",
			));
		}
		feedback.bitstream_offset = u32::try_from(annex_b_start)
			.map_err(|_| Error::out_of_range("Annex-B offset exceeds feedback range"))?;
		feedback.bytes_written = u32::try_from(nonzero_extent - annex_b_start)
			.map_err(|_| Error::out_of_range("Annex-B payload exceeds feedback range"))?;
	}
	let dirty_end = u64::from(feedback.bitstream_offset) + u64::from(feedback.bytes_written);
	if dirty_end > bitstream.capacity {
		return Err(Error::data_loss(
			"video encode feedback describes a range outside the bitstream buffer",
		));
	}
	Ok(feedback)
}

fn create_feedback_pool(
	device: &Arc<Device>,
	profile: &vk::VideoProfileInfoKHR<'_>,
) -> Result<vk::QueryPool> {
	let mut feedback_info = vk::QueryPoolVideoEncodeFeedbackCreateInfoKHR::new()
		.set_encode_feedback_flags(
			vk::VideoEncodeFeedbackFlagBitsKHR::BITSTREAM_BUFFER_OFFSET_BIT_KHR
				| vk::VideoEncodeFeedbackFlagBitsKHR::BITSTREAM_BYTES_WRITTEN_BIT_KHR,
		);
	// Preserve the codec-specific profile chain rather than copying only its
	// common fields. Both borrowed nodes outlive the synchronous native call.
	feedback_info.p_next = std::ptr::from_ref(profile).cast();
	let mut info = vk::QueryPoolCreateInfo::new()
		.set_query_type(vk::QueryType::VIDEO_ENCODE_FEEDBACK_KHR)
		.set_query_count(1);
	info.p_next = std::ptr::from_ref(&feedback_info).cast();
	// SAFETY: the encode profile chain is live for this call and the queue
	// family exposes video encode feedback queries.
	unsafe { device.create_query_pool(&info, None) }.map_err(|source| {
		Error::backend_failure("Vulkan", "encode feedback query-pool creation", source)
	})
}

fn create_encode_bitstream(
	device: &Arc<Device>,
	profile: &vk::VideoProfileInfoKHR<'_>,
	capacity: u64,
	size_alignment: u64,
) -> Result<EncodeBitstream> {
	let size = capacity
		.div_ceil(size_alignment)
		.checked_mul(size_alignment)
		.ok_or_else(|| Error::out_of_range("aligned encode bitstream size overflowed"))?;
	let profiles = std::slice::from_ref(profile);
	let mut profile_list = vk::VideoProfileListInfoKHR::new()
		.set_profiles(profiles)
		.map_err(Error::from)?;
	let buffer_info = vk::BufferCreateInfo::new()
		.set_size(size)
		.set_usage(vk::BufferUsageFlagBits::VIDEO_ENCODE_DST_BIT_KHR)
		.set_sharing_mode(vk::SharingMode::EXCLUSIVE)
		.push_next(&mut profile_list)
		.map_err(Error::from)?;
	let allocation_info = vk::mem::AllocationCreateInfo::new()
		.set_host_access(vk::mem::HostAccess::Random)
		.set_usage(vk::mem::MemoryUsage::PreferDevice)
		.set_required_flags(vk::MemoryPropertyFlagBits::HOST_VISIBLE_BIT)
		.set_preferred_flags(vk::MemoryPropertyFlagBits::DEVICE_LOCAL_BIT);
	// SAFETY: the profile chain is live and VMA binds compatible host-visible
	// memory to the returned buffer.
	let (handle, allocation) = unsafe {
		device
			.allocator()
			.with_allocator(|a| a.create_buffer_with_options(&buffer_info, &allocation_info))
			.map_err(Error::from)?
	}
	.map_err(|source| Error::backend_failure("Vulkan", "encode-bitstream allocation", source))?;
	let bitstream = EncodeBitstream {
		handle,
		allocation,
		capacity: size,
		device: device.clone(),
	};
	// Zero-fill once so the zero-byte feedback fallback has a clean scan target.
	bitstream.clear(size)?;
	Ok(bitstream)
}

impl EncodeBitstream {
	fn map_read(&self, offset: u64, output: &mut [u8]) -> Result<()> {
		if output.is_empty() {
			return Ok(());
		}
		let end = offset
			.checked_add(output.len() as u64)
			.ok_or_else(|| Error::invalid_argument("bitstream read range overflows"))?;
		if end > self.capacity {
			return Err(Error::invalid_argument(
				"bitstream read range exceeds the buffer",
			));
		}
		let offset_usize =
			usize::try_from(offset).map_err(|_| Error::out_of_range("bitstream offset exceeds usize"))?;
		let allocator = self.device.allocator();
		// SAFETY: VMA selected HOST_VISIBLE memory; the range was validated.
		let mapped = unsafe { allocator.with_allocator(|a| a.map(&self.allocation))?? };
		let invalidate = unsafe {
			allocator.with_allocator(|a| a.invalidate(&self.allocation, offset, output.len() as u64))
		};
		if matches!(invalidate, Ok(Ok(()))) {
			// SAFETY: the mapped range covers `output` and the slices do not alias.
			unsafe {
				crate::core::memory::copy_to_ptr(
					output.as_mut_ptr(),
					mapped.cast::<u8>().add(offset_usize),
					output.len(),
				);
			}
		}
		// SAFETY: balances the successful map above.
		unsafe {
			allocator.with_allocator(|a| a.unmap(&self.allocation))??;
		}
		invalidate
			.map_err(|source| Error::backend_failure("Vulkan", "encode-bitstream invalidation", source))?
			.map_err(|source| Error::backend_failure("Vulkan", "encode-bitstream invalidation", source))
	}

	fn read(&self, offset: u64, output: &mut [u8]) -> Result<()> {
		self.map_read(offset, output)
	}

	fn clear(&self, len: u64) -> Result<()> {
		let len = len.min(self.capacity);
		if len == 0 {
			return Ok(());
		}
		let byte_count =
			usize::try_from(len).map_err(|_| Error::out_of_range("bitstream clear exceeds usize"))?;
		let allocator = self.device.allocator();
		// SAFETY: VMA selected HOST_VISIBLE memory; `len` is within capacity.
		let mapped = unsafe { allocator.with_allocator(|a| a.map(&self.allocation))?? };
		unsafe {
			std::ptr::write_bytes(mapped.cast::<u8>(), 0, byte_count);
		}
		let flush = unsafe { allocator.with_allocator(|a| a.flush(&self.allocation, 0, len)) };
		// SAFETY: balances the successful map above.
		unsafe {
			allocator.with_allocator(|a| a.unmap(&self.allocation))??;
		}
		flush
			.map_err(|source| Error::backend_failure("Vulkan", "encode-bitstream flush", source))?
			.map_err(|source| Error::backend_failure("Vulkan", "encode-bitstream flush", source))
	}
}

impl Drop for EncodeBitstream {
	fn drop(&mut self) {
		if self.handle != vk::Buffer::default() {
			// SAFETY: the slot uniquely owns this buffer/allocation pair.
			unsafe {
				let _ = self
					.device
					.allocator()
					.with_allocator(|a| a.destroy_buffer(self.handle, &mut self.allocation));
			}
		}
	}
}

#[allow(clippy::too_many_arguments)]
fn record_rgba_to_nv12(
	device: &Device,
	command_buffer: vk::CommandBuffer,
	pipeline: &ComputePipeline,
	descriptor_set: vk::DescriptorSet,
	push: &CvtPush,
	input_image: vk::Image,
	was_initialized: bool,
	groups_x: u32,
	groups_y: u32,
) -> Result<()> {
	// Transition the NV12 picture to GENERAL for the storage-image writes;
	// reinitialize is a no-op barrier on the same layout.
	let barrier = vk::ImageMemoryBarrier2::new()
		// The previous encode read completes before slot reuse, and the Engine
		// timeline orders this compute submission after it. The compute family
		// cannot name VIDEO_ENCODE as a local pipeline stage. Images use concurrent
		// sharing across the two families; acquire the next write at COMPUTE.
		.set_src_stage_mask(vk::PipelineStageFlagBits2::NONE)
		.set_src_access_mask(vk::AccessFlagBits2::NONE)
		.set_dst_stage_mask(vk::PipelineStageFlagBits2::COMPUTE_SHADER_BIT)
		.set_dst_access_mask(vk::AccessFlagBits2::SHADER_STORAGE_WRITE_BIT)
		.set_old_layout(if was_initialized {
			vk::ImageLayout::VIDEO_ENCODE_SRC_KHR
		} else {
			vk::ImageLayout::UNDEFINED
		})
		.set_new_layout(vk::ImageLayout::GENERAL)
		.set_image(input_image)
		.set_subresource_range(vk::ImageSubresourceRange {
			aspect_mask: vk::ImageAspectFlagBits::PLANE_0_BIT | vk::ImageAspectFlagBits::PLANE_1_BIT,
			base_mip_level: 0,
			level_count: 1,
			base_array_layer: 0,
			layer_count: 1,
		});
	let barriers = [barrier];
	let dependency = vk::DependencyInfo::new()
		.set_image_memory_barriers(&barriers)
		.map_err(Error::from)?;
	// SAFETY: the command buffer is recording on this device; all named
	// resources outlive submission through RecordedCommandBuffer retention.
	unsafe {
		device.cmd_pipeline_barrier2(command_buffer, &dependency)?;
		device.cmd_bind_pipeline(
			command_buffer,
			vk::PipelineBindPoint::COMPUTE,
			pipeline.raw(),
		)?;
		device.cmd_bind_descriptor_sets(
			command_buffer,
			vk::PipelineBindPoint::COMPUTE,
			pipeline.layout(),
			0,
			&[descriptor_set],
			&[],
		)?;
		device.cmd_push_constants(
			command_buffer,
			pipeline.layout(),
			vk::ShaderStageFlagBits::COMPUTE_BIT,
			0,
			std::slice::from_raw_parts(std::ptr::from_ref(push).cast::<u8>(), size_of::<CvtPush>()),
		)?;
		device.cmd_dispatch(command_buffer, groups_x, groups_y, 1)?;
	}
	// Hand the image back to VIDEO_ENCODE_SRC so the encode submission only
	// needs the same-layout memory dependency the donor records.
	let back = vk::ImageMemoryBarrier2::new()
		.set_src_stage_mask(vk::PipelineStageFlagBits2::COMPUTE_SHADER_BIT)
		.set_src_access_mask(vk::AccessFlagBits2::SHADER_STORAGE_WRITE_BIT)
		.set_dst_stage_mask(vk::PipelineStageFlagBits2::NONE)
		.set_dst_access_mask(vk::AccessFlagBits2::NONE)
		.set_old_layout(vk::ImageLayout::GENERAL)
		.set_new_layout(vk::ImageLayout::VIDEO_ENCODE_SRC_KHR)
		.set_image(input_image)
		.set_subresource_range(vk::ImageSubresourceRange {
			aspect_mask: vk::ImageAspectFlagBits::PLANE_0_BIT | vk::ImageAspectFlagBits::PLANE_1_BIT,
			base_mip_level: 0,
			level_count: 1,
			base_array_layer: 0,
			layer_count: 1,
		});
	let barriers = [back];
	let dependency = vk::DependencyInfo::new()
		.set_image_memory_barriers(&barriers)
		.map_err(Error::from)?;
	unsafe {
		device.cmd_pipeline_barrier2(command_buffer, &dependency)?;
	}
	Ok(())
}

#[derive(Clone, Copy)]
pub(in crate::runtime) struct EncodePictureArgs {
	session: vk::VideoSessionKHR,
	parameters: vk::VideoSessionParametersKHR,
	codec: EncodeCodec,
	coded: video::VideoExtent,
	input_image: vk::Image,
	input_view: vk::ImageView,
	dpb_image: vk::Image,
	dpb_view: vk::ImageView,
	dpb_layers: u32,
	dpb_initialized: bool,
	current_dpb_slot: u32,
	previous_dpb_slot: u32,
	is_idr: bool,
	frame_index: u32,
	idr_pic_id: u16,
	first_frame: bool,
	bitstream: vk::Buffer,
	bitstream_range: u64,
	query_pool: vk::QueryPool,
	rate_control_mode: vk::VideoEncodeRateControlModeFlagBitsKHR,
	rate_control_reset: bool,
	bitrate: u32,
	peak_bitrate: u32,
	frame_rate: u32,
	gop_size: u32,
	max_b_frames: u32,
	constant_qp: u32,
	/// End-of-replay marker: the last captured frame resets the session's
	/// coding state so the next replay's first frame again sees the fresh
	/// DEFAULT rate-control state its recorded begin-info names.
	replay_boundary: bool,
}

fn record_encode_picture(
	device: &Device,
	command_buffer: vk::CommandBuffer,
	args: EncodePictureArgs,
) -> Result<()> {
	unsafe {
		device.cmd_reset_query_pool(command_buffer, args.query_pool, 0, 1)?;
	}
	// input: GENERAL (CvtRgbaToNv12 output) → encode-read on the video queue.
	// DPB: UNDEFINED on first use → VIDEO_ENCODE_DPB; afterwards a real
	// encode-write→encode-read/write dependency (the donor's Xe fix).
	let input_barrier = vk::ImageMemoryBarrier2::new()
		.set_src_stage_mask(vk::PipelineStageFlagBits2::NONE)
		.set_src_access_mask(vk::AccessFlagBits2::NONE)
		.set_dst_stage_mask(vk::PipelineStageFlagBits2::VIDEO_ENCODE_BIT_KHR)
		.set_dst_access_mask(vk::AccessFlagBits2::VIDEO_ENCODE_READ_BIT_KHR)
		.set_old_layout(vk::ImageLayout::VIDEO_ENCODE_SRC_KHR)
		.set_new_layout(vk::ImageLayout::VIDEO_ENCODE_SRC_KHR)
		.set_image(args.input_image)
		.set_subresource_range(vk::ImageSubresourceRange {
			aspect_mask: vk::ImageAspectFlagBits::PLANE_0_BIT | vk::ImageAspectFlagBits::PLANE_1_BIT,
			base_mip_level: 0,
			level_count: 1,
			base_array_layer: 0,
			layer_count: 1,
		});
	let dpb_barrier = vk::ImageMemoryBarrier2::new()
		.set_src_stage_mask(if args.dpb_initialized {
			vk::PipelineStageFlagBits2::VIDEO_ENCODE_BIT_KHR
		} else {
			vk::PipelineStageFlagBits2::NONE
		})
		.set_src_access_mask(if args.dpb_initialized {
			vk::AccessFlagBits2::VIDEO_ENCODE_READ_BIT_KHR
				| vk::AccessFlagBits2::VIDEO_ENCODE_WRITE_BIT_KHR
		} else {
			vk::AccessFlagBits2::NONE
		})
		.set_dst_stage_mask(vk::PipelineStageFlagBits2::VIDEO_ENCODE_BIT_KHR)
		.set_dst_access_mask(
			vk::AccessFlagBits2::VIDEO_ENCODE_READ_BIT_KHR
				| vk::AccessFlagBits2::VIDEO_ENCODE_WRITE_BIT_KHR,
		)
		.set_old_layout(if args.dpb_initialized {
			vk::ImageLayout::VIDEO_ENCODE_DPB_KHR
		} else {
			vk::ImageLayout::UNDEFINED
		})
		.set_new_layout(vk::ImageLayout::VIDEO_ENCODE_DPB_KHR)
		.set_image(args.dpb_image)
		.set_subresource_range(vk::ImageSubresourceRange {
			aspect_mask: vk::ImageAspectFlagBits::COLOR_BIT,
			base_mip_level: 0,
			level_count: 1,
			base_array_layer: 0,
			layer_count: args.dpb_layers,
		});
	let barriers = [input_barrier, dpb_barrier];
	let dependency = vk::DependencyInfo::new()
		.set_image_memory_barriers(&barriers)
		.map_err(Error::from)?;
	unsafe {
		device.cmd_pipeline_barrier2(command_buffer, &dependency)?;
	}

	match args.codec {
		EncodeCodec::H264 => h264::record_frame(device, command_buffer, &args),
		EncodeCodec::H265 => h265::record_frame(device, command_buffer, &args),
	}
}

fn reference_resources(args: &EncodePictureArgs) -> [vk::VideoPictureResourceInfoKHR<'static>; 2] {
	let extent = vk::Extent2D {
		width: args.coded.width,
		height: args.coded.height,
	};
	[args.current_dpb_slot, args.previous_dpb_slot].map(|layer| {
		vk::VideoPictureResourceInfoKHR::new()
			.set_coded_extent(extent)
			.set_base_array_layer(layer)
			.set_image_view_binding(args.dpb_view)
	})
}

fn record_coding<CodecRate, CodecLayer>(
	device: &Device,
	command_buffer: vk::CommandBuffer,
	args: &EncodePictureArgs,
	slots: &[vk::VideoReferenceSlotInfoKHR<'_>],
	make_rate: impl Fn() -> CodecRate,
	mut codec_layer: CodecLayer,
	record_picture: fn(
		&Device,
		vk::CommandBuffer,
		&EncodePictureArgs,
		vk::VideoPictureResourceInfoKHR<'_>,
	) -> Result<()>,
) -> Result<()>
where
	for<'a> CodecRate:
		vk::Extends<vk::VideoBeginCodingInfoKHR<'a>> + vk::Extends<vk::VideoCodingControlInfoKHR<'a>>,
	CodecLayer: for<'a> vk::Extends<vk::VideoEncodeRateControlLayerInfoKHR<'a>>,
{
	let rc_enabled = args.rate_control_mode
		!= vk::VideoEncodeRateControlModeFlagBitsKHR::DISABLED_BIT_KHR
		&& args.bitrate > 0;
	let mut configured_rc =
		vk::VideoEncodeRateControlInfoKHR::new().set_rate_control_mode(if rc_enabled {
			args.rate_control_mode
		} else {
			vk::VideoEncodeRateControlModeFlagBitsKHR::DISABLED_BIT_KHR
		});
	let configured_layer = if rc_enabled {
		vk::VideoEncodeRateControlLayerInfoKHR::new()
			.push_next(&mut codec_layer)
			.map_err(Error::from)?
			.set_average_bitrate(u64::from(args.bitrate))
			.set_max_bitrate(u64::from(
				if args.rate_control_mode == vk::VideoEncodeRateControlModeFlagBitsKHR::CBR_BIT_KHR {
					args.bitrate
				} else {
					args.peak_bitrate
				},
			))
			.set_frame_rate_numerator(args.frame_rate)
			.set_frame_rate_denominator(1)
	} else {
		vk::VideoEncodeRateControlLayerInfoKHR::new()
	};
	if rc_enabled {
		configured_rc = configured_rc
			.set_layers(std::slice::from_ref(&configured_layer))
			.map_err(Error::from)?
			.set_virtual_buffer_size_in_ms(1000)
			.set_initial_virtual_buffer_size_in_ms(500);
	}
	let mut begin_codec = make_rate();
	let mut begin = vk::VideoBeginCodingInfoKHR::new()
		.set_video_session(args.session)
		.set_video_session_parameters(args.parameters)
		.set_reference_slots(slots)
		.map_err(Error::from)?;
	if args.rate_control_reset {
		// The session's configured state is echoed verbatim, including the
		// codec rate-control structure the in-scope configuration wrote.
		begin = begin.push_next(&mut configured_rc).map_err(Error::from)?;
		if rc_enabled {
			begin = begin.push_next(&mut begin_codec).map_err(Error::from)?;
		}
	}
	// Otherwise no rate-control info is chained: its absence implies the
	// DEFAULT mode the session still holds on a fresh replay (the previous
	// replay's last scope reset it), before the in-scope control below
	// programs the configured state. Chaining the info here would demand an
	// exact echo of codec fields the session has not yet configured.

	// SAFETY: the session, parameters, slot resources, and rate-control chain
	// remain live for this recording.
	unsafe {
		device.cmd_begin_video_coding_khr(command_buffer, &begin)?;
	}

	// On the first scope, RESET and program the selected rate-control state.
	// RESET alone restores DEFAULT, which does not accept explicit slice QP.
	let mut control_codec = make_rate();
	if !args.rate_control_reset {
		let mut control = vk::VideoCodingControlInfoKHR::new()
			.set_flags(
				vk::VideoCodingControlFlagBitsKHR::RESET_BIT_KHR
					| vk::VideoCodingControlFlagBitsKHR::ENCODE_RATE_CONTROL_BIT_KHR,
			)
			.push_next(&mut configured_rc)
			.map_err(Error::from)?;
		if rc_enabled {
			control = control.push_next(&mut control_codec).map_err(Error::from)?;
		}
		// SAFETY: the scope is open and the rate-control chain remains live.
		unsafe {
			device.cmd_control_video_coding_khr(command_buffer, &control)?;
		}
	}

	let src_resource = vk::VideoPictureResourceInfoKHR::new()
		.set_coded_extent(vk::Extent2D {
			width: args.coded.width,
			height: args.coded.height,
		})
		.set_base_array_layer(0)
		.set_image_view_binding(args.input_view);
	// SAFETY: the allocated slot has one reset query and the coding scope is open.
	unsafe {
		device.cmd_begin_query(
			command_buffer,
			args.query_pool,
			0,
			vk::QueryControlFlags::empty(),
		)?;
	}
	record_picture(device, command_buffer, args, src_resource)?;
	// SAFETY: the query and coding scope above are open on the same command.
	unsafe {
		device.cmd_end_query(command_buffer, args.query_pool, 0)?;
	}
	unsafe {
		device.cmd_end_video_coding_khr(command_buffer, &vk::VideoEndCodingInfoKHR::new())?;
	}
	if args.replay_boundary {
		// Restore the session to its initial state (rate control returns to
		// DEFAULT) so the next replay's first begin always names the state the
		// previous replay left. The reset runs in its own coding scope: it must
		// not cancel the frame's just-ended feedback query, and its begin names
		// the configured state left by the frame's in-scope rate-control setup.
		let mut reset_codec = make_rate();
		let mut begin = vk::VideoBeginCodingInfoKHR::new()
			.set_video_session(args.session)
			.set_video_session_parameters(args.parameters);
		begin = begin.push_next(&mut configured_rc).map_err(Error::from)?;
		if rc_enabled {
			begin = begin.push_next(&mut reset_codec).map_err(Error::from)?;
		}
		let reset_control = vk::VideoCodingControlInfoKHR::new()
			.set_flags(vk::VideoCodingControlFlagBitsKHR::RESET_BIT_KHR);
		// SAFETY: the session is live and idle on this command buffer; the
		// begin/control/end sequence is balanced on the encode queue.
		unsafe {
			device.cmd_begin_video_coding_khr(command_buffer, &begin)?;
			device.cmd_control_video_coding_khr(command_buffer, &reset_control)?;
			device.cmd_end_video_coding_khr(command_buffer, &vk::VideoEndCodingInfoKHR::new())?;
		}
	}
	Ok(())
}

impl Drop for EncodeSession {
	fn drop(&mut self) {
		if self.parameters != vk::VideoSessionParametersKHR::default() {
			// SAFETY: this wrapper owns the parameter object and destroys it
			// before the video session it depends on.
			unsafe {
				let _ = self
					.device
					.destroy_video_session_parameters_khr(self.parameters, std::ptr::null());
			}
			self.parameters = vk::VideoSessionParametersKHR::default();
		}
		if self.handle != vk::VideoSessionKHR::default() {
			// SAFETY: this wrapper owns the session and destroys it once before
			// its bound allocations and retained device are released.
			unsafe {
				let _ = self
					.device
					.destroy_video_session_khr(self.handle, std::ptr::null());
			}
			self.handle = vk::VideoSessionKHR::default();
		}
		for allocation in &mut self.allocations {
			// SAFETY: session destruction ended every binding use, and this
			// wrapper owns each VMA allocation exactly once.
			let _ = unsafe {
				self
					.device
					.allocator()
					.with_allocator(|a| a.free(allocation))
			};
		}
		// SAFETY: the session-owned conversion pipeline is released while the
		// device is still live.
		self.conversion.destroy(self.device.raw());
	}
}
