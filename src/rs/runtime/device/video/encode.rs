// Private Vulkan Video encode session — the device half of `VideoEncoder`.
// Donor: oacpp source/cpp/lib/oa/vision/video/encoder/videoEncoder.cpp.

use std::sync::Arc;

use crate::runtime::RecordedCommandBuffer;
use crate::runtime::dna::kernel::{ShaderArtifact, content_ids};
use crate::runtime::{Device, device::pipeline::ComputePipeline};
use crate::{Error, Result, video};

use super::profile::{query_encode_details, query_formats, std_h264_level, with_encode_profile};
use super::{DecodeImage, MAX_VIDEO_SESSION_MEMORY_BINDINGS, create_decode_image};

static CVT_RGBA_TO_NV12: ShaderArtifact = ShaderArtifact::embedded(
	include_bytes!(concat!(env!("OUT_DIR"), "/video_cvt_rgba_to_nv12.spv")),
	[16, 16, 1],
	content_ids::VIDEO_CVT_RGBA_TO_NV12,
);

const ENCODE_BITSTREAM_CAPACITY: u64 = 4 * 1024 * 1024;

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
	dpb: DecodeImage,
	cached_headers: Vec<u8>,
	dpb_slot_capacity: u32,
	codec: EncodeCodec,
	visible_extent: video::VideoExtent,
	coded_extent: video::VideoExtent,
	rate_control_mode: vk::VideoEncodeRateControlModeFlagBitsKHR,
	bitrate: u32,
	peak_bitrate: u32,
	frame_rate: u32,
	gop_size: u32,
	max_b_frames: u32,
	constant_qp: u32,
	query_result_status_supported: bool,
	frame_count: u64,
	last_keyframe_index: u64,
	current_gop_frame: u32,
	rate_control_reset: bool,
	conversion: ComputePipeline,
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
	input_initialized: bool,
	bitstream_dirty_end: u64,
	presentation_timestamp: u64,
	is_keyframe: bool,
	pending: bool,
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
		let result = unsafe {
			device
				.raw()
				.create_video_session_khr(&create_info, std::ptr::null(), &mut handle)
		};
		if !matches!(result, Ok(vk::Result::SUCCESS)) {
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
		// The encoder never samples its DPB; ENCODE_DPB usage only, layered
		// across the slot ping-pong.
		let dpb = create_decode_image(
			device,
			vk_profile,
			dpb_format,
			coded_extent,
			dpb_slot_capacity,
			dpb_usage,
		)?;
		let mut session = EncodeSession {
			handle,
			parameters: vk::VideoSessionParametersKHR::default(),
			allocations: Vec::new(),
			dpb,
			cached_headers: Vec::new(),
			dpb_slot_capacity,
			codec,
			visible_extent: video::VideoExtent {
				width: settings.width,
				height: settings.height,
			},
			coded_extent,
			rate_control_mode,
			bitrate: settings.bitrate,
			peak_bitrate: plan.peak_bitrate(),
			frame_rate: settings.frame_rate,
			gop_size: settings.gop_size,
			max_b_frames: settings.max_b_frames,
			constant_qp: settings.constant_qp,
			query_result_status_supported: physical.video.encode_result_status_queries,
			frame_count: 0,
			last_keyframe_index: 0,
			current_gop_frame: 0,
			rate_control_reset: false,
			conversion: device.create_compute_pipeline(&CVT_RGBA_TO_NV12)?,
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
					"video-encode session memory binding",
					result,
				));
			}
		}
		session.create_session_parameters(settings, plan, caps)?;
		session.cache_encoded_headers(codec)?;
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
				coded_extent,
				encode_family,
				physical.compute_queue_family,
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
			self.device.raw().get_video_session_memory_requirements_khr(
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
			self.device.raw().get_video_session_memory_requirements_khr(
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

	fn create_session_parameters(
		&mut self,
		settings: &video::VideoEncodeConfig,
		plan: video::VideoEncodePlan,
		caps: &video::VideoEncodeCapabilities,
	) -> Result<()> {
		let mut quality =
			vk::VideoEncodeQualityLevelInfoKHR::new().set_quality_level(settings.quality_level);
		let handle = match self.codec {
			EncodeCodec::H264 => {
				let level = plan
					.h264_level()
					.ok_or_else(|| Error::invalid_argument("H.264 encode plan lacks a level"))?;
				let sps = build_h264_sps(self.coded_extent, level)?;
				let pps = build_h264_pps();
				let spss = [sps];
				let ppss = [pps];
				let add = vk::VideoEncodeH264SessionParametersAddInfoKHR::new()
					.set_std_sp_ss(&spss)
					.map_err(Error::from)?
					.set_std_pp_ss(&ppss)
					.map_err(Error::from)?;
				let mut codec = vk::VideoEncodeH264SessionParametersCreateInfoKHR::new()
					.set_max_std_sps_count(1)
					.set_max_std_pps_count(1)
					.set_parameters_add_info(&add);
				let create = vk::VideoSessionParametersCreateInfoKHR::new()
					.set_video_session(self.handle)
					.push_next(&mut codec)
					.map_err(Error::from)?
					.push_next(&mut quality)
					.map_err(Error::from)?;
				self.create_parameter_handle(&create)?
			}
			EncodeCodec::H265 => {
				let video::VideoEncodeCodecCapabilities::H265 {
					ctb_size_16,
					ctb_size_32,
					ctb_size_64,
					transform_size_4,
					transform_size_8,
					transform_size_16,
					transform_size_32,
					..
				} = caps.codec()
				else {
					return Err(Error::invalid_argument(
						"H.265 session parameters require H.265 codec capabilities",
					));
				};
				let mut parameters = build_h265_parameters(
					settings,
					self.coded_extent,
					self.dpb_slot_capacity,
					ctb_size_16,
					ctb_size_32,
					ctb_size_64,
					transform_size_4,
					transform_size_8,
					transform_size_16,
					transform_size_32,
				);
				// Link the pointer fields in place before copying the sets —
				// the copies then reference this stack frame for the call.
				parameters.link();
				let vpss = [parameters.vps];
				let spss = [parameters.sps];
				let ppss = [parameters.pps];
				let add = vk::VideoEncodeH265SessionParametersAddInfoKHR::new()
					.set_std_vp_ss(&vpss)
					.map_err(Error::from)?
					.set_std_sp_ss(&spss)
					.map_err(Error::from)?
					.set_std_pp_ss(&ppss)
					.map_err(Error::from)?;
				let mut codec = vk::VideoEncodeH265SessionParametersCreateInfoKHR::new()
					.set_max_std_vps_count(1)
					.set_max_std_sps_count(1)
					.set_max_std_pps_count(1)
					.set_parameters_add_info(&add);
				let create = vk::VideoSessionParametersCreateInfoKHR::new()
					.set_video_session(self.handle)
					.push_next(&mut codec)
					.map_err(Error::from)?
					.push_next(&mut quality)
					.map_err(Error::from)?;
				self.create_parameter_handle(&create)?
			}
		};
		self.parameters = handle;
		Ok(())
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
				.raw()
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
	fn cache_encoded_headers(&mut self, codec: EncodeCodec) -> Result<()> {
		let mut h264_get = vk::VideoEncodeH264SessionParametersGetInfoKHR::new()
			.set_write_std_sps(vk::TRUE)
			.set_write_std_pps(vk::TRUE)
			.set_std_sps_id(0)
			.set_std_pps_id(0);
		let mut h265_get = vk::VideoEncodeH265SessionParametersGetInfoKHR::new()
			.set_write_std_vps(vk::TRUE)
			.set_write_std_sps(vk::TRUE)
			.set_write_std_pps(vk::TRUE)
			.set_std_vps_id(0)
			.set_std_sps_id(0)
			.set_std_pps_id(0);
		let get = match codec {
			EncodeCodec::H264 => vk::VideoEncodeSessionParametersGetInfoKHR::new()
				.set_video_session_parameters(self.parameters)
				.push_next(&mut h264_get)
				.map_err(Error::from)?,
			EncodeCodec::H265 => vk::VideoEncodeSessionParametersGetInfoKHR::new()
				.set_video_session_parameters(self.parameters)
				.push_next(&mut h265_get)
				.map_err(Error::from)?,
		};
		let mut feedback = vk::VideoEncodeSessionParametersFeedbackInfoKHR::new();
		let mut size = 0_usize;
		// SAFETY: the session-parameter object is live; null data queries size.
		let result = unsafe {
			self.device.raw().get_encoded_video_session_parameters_khr(
				&get,
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
			self.device.raw().get_encoded_video_session_parameters_khr(
				&get,
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
		self.slots[index].pending
	}

	pub(in crate::runtime) fn slot_is_keyframe(&self, index: usize) -> bool {
		self.slots[index].is_keyframe
	}

	pub(in crate::runtime) fn slot_presentation_timestamp(&self, index: usize) -> u64 {
		self.slots[index].presentation_timestamp
	}

	/// Stage one visible-extent RGBA8 frame into the slot's private snapshot.
	pub(in crate::runtime) fn write_rgba(&mut self, index: usize, data: &[u8]) -> Result<()> {
		let visible = usize::try_from(self.visible_extent.width)
			.ok()
			.and_then(|width| {
				usize::try_from(self.visible_extent.height)
					.ok()
					.and_then(|height| width.checked_mul(height)?.checked_mul(4))
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
		&mut self,
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
		let slot = &mut self.slots[index];
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
		let pipeline = &self.conversion;
		let descriptor_set = device.descriptor_set();
		let input_image = slot.input_image;
		let was_initialized = slot.input_initialized;
		let groups_x = self.coded_extent.width.div_ceil(16);
		let groups_y = self.coded_extent.height.div_ceil(16);
		let push_size = pipeline.push_constant_size();
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
		if groups_x > pipeline.max_dispatch_group_count()[0]
			|| groups_y > pipeline.max_dispatch_group_count()[1]
		{
			return Err(Error::missing_capability(
				"encode conversion dispatch exceeds device workgroup limits",
			));
		}
		let command = device.record_compute_commands(|command_buffer| {
			record_rgba_to_nv12(
				device.raw(),
				command_buffer,
				pipeline,
				descriptor_set,
				&push,
				input_image,
				was_initialized,
				groups_x,
				groups_y,
			)
		})?;
		self.slots[index].input_initialized = true;
		Ok(command)
	}

	/// Record one IDR/P frame for a slot onto the video encode queue.
	///
	/// The engine timeline orders this after the slot's conversion submission;
	/// DPB read/write hazards are fenced by the in-command barrier (the donor's
	/// zero-byte Xe workaround) and GOP state advances at submission time.
	pub(in crate::runtime) fn record_frame(
		&mut self,
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
		let slot = &mut self.slots[index];
		if slot.bitstream_dirty_end > 0 {
			slot.bitstream.clear(slot.bitstream_dirty_end)?;
			slot.bitstream_dirty_end = 0;
		}
		let is_idr = self.current_gop_frame == 0;
		let reference_depth = self.dpb_slot_capacity.max(1);
		let current_dpb_slot = (self.frame_count % u64::from(reference_depth)) as u32;
		let previous_dpb_slot = if self.frame_count == 0 {
			current_dpb_slot
		} else {
			((self.frame_count - 1) % u64::from(reference_depth)) as u32
		};
		let frame_index = (self.frame_count - self.last_keyframe_index) as u32;
		let session = self.handle;
		let parameters = self.parameters;
		let coded = self.coded_extent;
		let dpb_view = self.dpb.view;
		let dpb_image = self.dpb.handle;
		let dpb_layers = self.dpb_slot_capacity;
		let dpb_initialized = self.rate_control_reset;
		let input_image = slot.input_image;
		let input_view = slot.input_view;
		let bitstream = slot.bitstream.handle;
		let bitstream_range = slot.bitstream.capacity;
		let query_pool = slot.query_pool;
		let codec = self.codec;
		let rate_control_mode = self.rate_control_mode;
		let rc_reset = self.rate_control_reset;
		let bitrate = self.bitrate;
		let peak = self.peak_bitrate;
		let frame_rate = self.frame_rate;
		let gop_size = self.gop_size;
		let max_b = self.max_b_frames;
		let constant_qp = self.constant_qp;
		let idr_pic_id = (self.frame_count & 0xFFFF) as u16;
		let first_frame = self.frame_count == 0;
		let command = device.record_video_encode(|command_buffer| {
			record_encode_picture(
				device.raw(),
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
				},
			)
		})?;
		// GOP state advances at submission — queue order makes the shared
		// reference slot deterministic while older jobs remain unharvested.
		slot.is_keyframe = is_idr;
		if is_idr {
			self.last_keyframe_index = self.frame_count;
			self.current_gop_frame = if self.gop_size > 1 { 1 } else { 0 };
		} else {
			self.current_gop_frame += 1;
			if self.current_gop_frame >= self.gop_size {
				self.current_gop_frame = 0;
			}
		}
		self.frame_count += 1;
		self.rate_control_reset = true;
		Ok(command)
	}

	pub(in crate::runtime) fn mark_pending(&mut self, index: usize, pts: u64) {
		let slot = &mut self.slots[index];
		slot.presentation_timestamp = pts;
		slot.pending = true;
	}

	pub(in crate::runtime) fn mark_completed(&mut self, index: usize) {
		self.slots[index].pending = false;
	}

	/// Read one slot's encode feedback after its submission event completed.
	/// Includes the donor's zero-byte Annex-B recovery for drivers that report
	/// COMPLETE without counters.
	pub(in crate::runtime) fn read_feedback(&mut self, index: usize) -> Result<EncodeFeedback> {
		let slot = &self.slots[index];
		let (feedback, status) = if self.query_result_status_supported {
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
				self.device.raw().get_query_pool_results(
					slot.query_pool,
					0,
					1,
					std::slice::from_raw_parts_mut(
						std::ptr::from_mut(&mut result).cast::<u8>(),
						size_of::<WithStatus>(),
					),
					size_of::<WithStatus>() as vk::DeviceSize,
					vk::QueryResultFlagBits::WAIT_BIT | vk::QueryResultFlagBits::WITH_STATUS_BIT_KHR,
				)
			};
			if !matches!(vk_result, Ok(vk::Result::SUCCESS)) {
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
				self.device.raw().get_query_pool_results(
					slot.query_pool,
					0,
					1,
					std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<u8>(), size_of::<[u32; 2]>()),
					size_of::<[u32; 2]>() as vk::DeviceSize,
					vk::QueryResultFlagBits::WAIT_BIT,
				)
			};
			if !matches!(vk_result, Ok(vk::Result::SUCCESS)) {
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
		if status != vk::QueryResultStatusKHR::COMPLETE_KHR {
			return Err(Error::data_loss(format!(
				"Vulkan video encode completed with operation status {status:?}"
			)));
		}
		let mut feedback = feedback;
		if feedback.bytes_written == 0 {
			// Recovery: scan the invalidated mapping for the last nonzero byte
			// and the first Annex-B start code, exactly like the donor.
			let capacity = usize::try_from(slot.bitstream.capacity)
				.map_err(|_| Error::out_of_range("bitstream capacity exceeds host address space"))?;
			let mut probe = vec![0_u8; capacity];
			slot.bitstream.read(0, &mut probe)?;
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
		if dirty_end > slot.bitstream.capacity {
			return Err(Error::data_loss(
				"video encode feedback describes a range outside the bitstream buffer",
			));
		}
		self.slots[index].bitstream_dirty_end = dirty_end;
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
	coded_extent: video::VideoExtent,
	encode_family: u32,
	compute_family: u32,
	bitstream_size_alignment: u64,
) -> Result<EncodeSlot> {
	let profiles = std::slice::from_ref(profile);
	let mut profile_list = vk::VideoProfileListInfoKHR::new()
		.set_profiles(profiles)
		.map_err(Error::from)?;
	let view_formats = [format.format, vk::Format::R8_UNORM, vk::Format::R8G8_UNORM];
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
		rgba_snapshot: crate::runtime::Buffer::host_visible_storage(
			device,
			usize::try_from(coded_extent.width)
				.ok()
				.and_then(|w| {
					usize::try_from(coded_extent.height)
						.ok()
						.and_then(|h| w.checked_mul(h)?.checked_mul(4))
				})
				.ok_or_else(|| Error::out_of_range("encode RGBA snapshot size overflows"))?,
		)?,
		bitstream: create_encode_bitstream(
			device,
			profile,
			ENCODE_BITSTREAM_CAPACITY,
			bitstream_size_alignment.max(1),
		)?,
		query_pool: create_feedback_pool(device, profile)?,
		y_bindless_registered: false,
		uv_bindless_registered: false,
		input_initialized: false,
		bitstream_dirty_end: 0,
		presentation_timestamp: 0,
		is_keyframe: false,
		pending: false,
		device: device.clone(),
	};
	slot.create_views(format)?;
	Ok(slot)
}

impl EncodeSlot {
	fn create_views(&mut self, format: &vk::VideoFormatPropertiesKHR<'_>) -> Result<()> {
		let mut encode_usage = vk::ImageViewUsageCreateInfo::new()
			.set_usage(vk::ImageUsageFlagBits::VIDEO_ENCODE_SRC_BIT_KHR);
		self.input_view = unsafe {
			self.device.raw().create_image_view(
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
			self.device.raw().create_image_view(
				&(plane_view(vk::ImageAspectFlagBits::PLANE_0_BIT, vk::Format::R8_UNORM)
					.push_next(&mut y_usage)
					.map_err(Error::from)?),
				None,
			)
		}
		.map_err(|source| {
			Error::backend_failure("Vulkan", "encode-input Y-plane view creation", source)
		})?;
		self.input_uv_view = unsafe {
			self.device.raw().create_image_view(
				&(plane_view(vk::ImageAspectFlagBits::PLANE_1_BIT, vk::Format::R8G8_UNORM)
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
					let _ = self.device.raw().destroy_image_view(view, None);
				}
			}
		}
		if self.query_pool != vk::QueryPool::default() {
			// SAFETY: pending submissions are harvested before session teardown,
			// and this slot owns the pool.
			unsafe {
				let _ = self.device.raw().destroy_query_pool(self.query_pool, None);
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

fn create_feedback_pool(
	device: &Arc<Device>,
	profile: &vk::VideoProfileInfoKHR<'_>,
) -> Result<vk::QueryPool> {
	let mut feedback_info = vk::QueryPoolVideoEncodeFeedbackCreateInfoKHR::new()
		.set_encode_feedback_flags(
			vk::VideoEncodeFeedbackFlagBitsKHR::BITSTREAM_BUFFER_OFFSET_BIT_KHR
				| vk::VideoEncodeFeedbackFlagBitsKHR::BITSTREAM_BYTES_WRITTEN_BIT_KHR,
		);
	// The profile node extends QueryPoolCreateInfo directly; the codec-specific
	// profile chain is not required for feedback queries.
	let mut query_profile = vk::VideoProfileInfoKHR::new()
		.set_video_codec_operation(profile.video_codec_operation)
		.set_chroma_subsampling(profile.chroma_subsampling)
		.set_luma_bit_depth(profile.luma_bit_depth)
		.set_chroma_bit_depth(profile.chroma_bit_depth);
	let info = vk::QueryPoolCreateInfo::new()
		.set_query_type(vk::QueryType::VIDEO_ENCODE_FEEDBACK_KHR)
		.set_query_count(1)
		.push_next(&mut feedback_info)
		.map_err(Error::from)?
		.push_next(&mut query_profile)
		.map_err(Error::from)?;
	// SAFETY: the encode profile chain is live for this call and the queue
	// family exposes video encode feedback queries.
	unsafe { device.raw().create_query_pool(&info, None) }.map_err(|source| {
		Error::backend_failure("Vulkan", "encode feedback query-pool creation", source)
	})
}

fn create_encode_bitstream(
	device: &Arc<Device>,
	profile: &vk::VideoProfileInfoKHR<'_>,
	capacity: u64,
	size_alignment: u64,
) -> Result<EncodeBitstream> {
	let size = capacity.div_ceil(size_alignment) * size_alignment;
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
		let allocator = self.device.allocator();
		// SAFETY: VMA selected HOST_VISIBLE memory; the range was validated.
		let mapped = unsafe { allocator.with_allocator(|a| a.map(&self.allocation)) }
			.map_err(|source| Error::backend_failure("Vulkan", "encode-bitstream mapping", source))?;
		let invalidate = unsafe {
			allocator.with_allocator(|a| a.invalidate(&self.allocation, offset, output.len() as u64))
		};
		if matches!(invalidate, Ok(Ok(()))) {
			// SAFETY: the mapped range covers `output` and the slices do not alias.
			unsafe {
				crate::core::memory::copy_to_ptr(
					output.as_mut_ptr(),
					(mapped.map_err(Error::from)? as *const u8).add(offset as usize),
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
		let allocator = self.device.allocator();
		// SAFETY: VMA selected HOST_VISIBLE memory; `len` is within capacity.
		let mapped = unsafe { allocator.with_allocator(|a| a.map(&self.allocation)) }
			.map_err(|source| Error::backend_failure("Vulkan", "encode-bitstream mapping", source))?;
		unsafe {
			std::ptr::write_bytes(mapped.map_err(Error::from)? as *mut u8, 0, len as usize);
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
	device: &vk::Device,
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
		.set_src_stage_mask(if was_initialized {
			vk::PipelineStageFlagBits2::VIDEO_ENCODE_BIT_KHR
		} else {
			vk::PipelineStageFlagBits2::NONE
		})
		.set_src_access_mask(if was_initialized {
			vk::AccessFlagBits2::VIDEO_ENCODE_READ_BIT_KHR
		} else {
			vk::AccessFlagBits2::NONE
		})
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

struct EncodePictureArgs {
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
}

#[allow(clippy::too_many_lines)]
fn record_encode_picture(
	device: &vk::Device,
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

	// Coding-scope DPB associations: the setup resource is bound inactive
	// (slotIndex -1) and the previous picture is the sole active reference.
	let coded_extent = vk::Extent2D {
		width: args.coded.width,
		height: args.coded.height,
	};
	let mut begin_current_h264_ref = vk::StdVideoEncodeH264ReferenceInfo::new();
	let mut begin_current_h264 = vk::VideoEncodeH264DpbSlotInfoKHR::new();
	let mut begin_current_h265_ref = vk::StdVideoEncodeH265ReferenceInfo::new();
	let mut begin_current_h265 = vk::VideoEncodeH265DpbSlotInfoKHR::new();
	let mut begin_previous_h264_ref = vk::StdVideoEncodeH264ReferenceInfo::new();
	let mut begin_previous_h264 = vk::VideoEncodeH264DpbSlotInfoKHR::new();
	let mut begin_previous_h265_ref = vk::StdVideoEncodeH265ReferenceInfo::new();
	let mut begin_previous_h265 = vk::VideoEncodeH265DpbSlotInfoKHR::new();
	let previous_resource = vk::VideoPictureResourceInfoKHR::new()
		.set_coded_extent(coded_extent)
		.set_base_array_layer(args.previous_dpb_slot)
		.set_image_view_binding(args.dpb_view);
	let current_resource = vk::VideoPictureResourceInfoKHR::new()
		.set_coded_extent(coded_extent)
		.set_base_array_layer(args.current_dpb_slot)
		.set_image_view_binding(args.dpb_view);
	let mut slots = [
		vk::VideoReferenceSlotInfoKHR::new(),
		vk::VideoReferenceSlotInfoKHR::new(),
	];
	match args.codec {
		EncodeCodec::H264 => {
			begin_current_h264_ref.primary_pic_type = if args.is_idr {
				vk::StdVideoH264PictureType::IDR
			} else {
				vk::StdVideoH264PictureType::P
			};
			begin_current_h264_ref.frame_num = if args.is_idr { 0 } else { args.frame_index };
			begin_current_h264_ref.pic_order_cnt = if args.is_idr {
				0
			} else {
				args.frame_index as i32
			};
			begin_current_h264 = begin_current_h264.set_std_reference_info(&begin_current_h264_ref);
			let slot0 = vk::VideoReferenceSlotInfoKHR::new()
				.set_slot_index(-1)
				.set_picture_resource(&current_resource)
				.push_next(&mut begin_current_h264)
				.map_err(Error::from)?;
			if !args.is_idr {
				begin_previous_h264_ref.primary_pic_type = vk::StdVideoH264PictureType::P;
				begin_previous_h264_ref.frame_num = args.frame_index - 1;
				begin_previous_h264_ref.pic_order_cnt = args.frame_index as i32 - 1;
				begin_previous_h264 = begin_previous_h264.set_std_reference_info(&begin_previous_h264_ref);
				let slot1 = vk::VideoReferenceSlotInfoKHR::new()
					.set_slot_index(args.previous_dpb_slot as i32)
					.set_picture_resource(&previous_resource)
					.push_next(&mut begin_previous_h264)
					.map_err(Error::from)?;
				slots[1] = slot1;
			}
			slots[0] = slot0;
		}
		EncodeCodec::H265 => {
			// The setup slot describes DPB reference state, not the NAL picture
			// type — the donor leaves it P even for the first IDR (Intel expects
			// this exact convention).
			begin_current_h265_ref.pic_type = vk::StdVideoH265PictureType::P;
			begin_current_h265_ref.pic_order_cnt_val = if args.is_idr {
				0
			} else {
				args.frame_index as i32
			};
			begin_current_h265_ref.temporal_id = 0;
			begin_current_h265 = begin_current_h265.set_std_reference_info(&begin_current_h265_ref);
			let slot0 = vk::VideoReferenceSlotInfoKHR::new()
				.set_slot_index(-1)
				.set_picture_resource(&current_resource)
				.push_next(&mut begin_current_h265)
				.map_err(Error::from)?;
			if !args.is_idr {
				begin_previous_h265_ref.pic_type = vk::StdVideoH265PictureType::P;
				begin_previous_h265_ref.pic_order_cnt_val = args.frame_index as i32 - 1;
				begin_previous_h265_ref.temporal_id = 0;
				begin_previous_h265 = begin_previous_h265.set_std_reference_info(&begin_previous_h265_ref);
				let slot1 = vk::VideoReferenceSlotInfoKHR::new()
					.set_slot_index(args.previous_dpb_slot as i32)
					.set_picture_resource(&previous_resource)
					.push_next(&mut begin_previous_h265)
					.map_err(Error::from)?;
				slots[1] = slot1;
			}
			slots[0] = slot0;
		}
	}

	// The coding scope must declare active rate-control state; constant-QP
	// uses DISABLED rather than the session's DEFAULT after the reset scope.
	let mut rc = vk::VideoEncodeRateControlInfoKHR::new();
	let mut rc_layer = vk::VideoEncodeRateControlLayerInfoKHR::new();
	let mut h264_rc = vk::VideoEncodeH264RateControlInfoKHR::new();
	let mut h264_rc_layer = vk::VideoEncodeH264RateControlLayerInfoKHR::new();
	let mut h265_rc = vk::VideoEncodeH265RateControlInfoKHR::new();
	let mut h265_rc_layer = vk::VideoEncodeH265RateControlLayerInfoKHR::new();
	let rc_enabled = args.rate_control_mode
		!= vk::VideoEncodeRateControlModeFlagBitsKHR::DISABLED_BIT_KHR
		&& args.bitrate > 0;
	rc = rc.set_rate_control_mode(if !args.rate_control_reset {
		vk::VideoEncodeRateControlModeFlagBitsKHR::DEFAULT_KHR
	} else if rc_enabled {
		args.rate_control_mode
	} else {
		vk::VideoEncodeRateControlModeFlagBitsKHR::DISABLED_BIT_KHR
	});
	if rc_enabled && args.rate_control_reset {
		match args.codec {
			EncodeCodec::H264 => {
				h264_rc_layer = h264_rc_layer
					.set_use_min_qp(vk::FALSE)
					.set_use_max_qp(vk::FALSE)
					.set_use_max_frame_size(vk::FALSE);
				h264_rc = h264_rc
					.set_gop_frame_count(args.gop_size)
					.set_idr_period(args.gop_size)
					.set_consecutive_b_frame_count(args.max_b_frames)
					.set_temporal_layer_count(1);
				rc_layer = rc_layer
					.push_next(&mut h264_rc_layer)
					.map_err(Error::from)?;
			}
			EncodeCodec::H265 => {
				h265_rc_layer = h265_rc_layer
					.set_use_min_qp(vk::FALSE)
					.set_use_max_qp(vk::FALSE)
					.set_use_max_frame_size(vk::FALSE);
				h265_rc = h265_rc
					.set_gop_frame_count(args.gop_size)
					.set_idr_period(args.gop_size)
					.set_consecutive_b_frame_count(args.max_b_frames)
					.set_sub_layer_count(1);
				rc_layer = rc_layer
					.push_next(&mut h265_rc_layer)
					.map_err(Error::from)?;
			}
		}
		rc_layer = rc_layer
			.set_average_bitrate(u64::from(args.bitrate))
			.set_max_bitrate(u64::from(
				if args.rate_control_mode == vk::VideoEncodeRateControlModeFlagBitsKHR::CBR_BIT_KHR {
					args.bitrate
				} else {
					args.peak_bitrate
				},
			))
			.set_frame_rate_numerator(args.frame_rate)
			.set_frame_rate_denominator(1);
		let layers = std::slice::from_ref(&rc_layer);
		rc = rc
			.set_layer_count(1)
			.set_layers(layers)
			.map_err(Error::from)?
			.set_virtual_buffer_size_in_ms(1000)
			.set_initial_virtual_buffer_size_in_ms(500);
	}
	let reference_count = u32::from(!args.is_idr) + 1;
	let mut begin = vk::VideoBeginCodingInfoKHR::new()
		.set_video_session(args.session)
		.set_video_session_parameters(args.parameters)
		.set_reference_slots(&slots[..reference_count as usize])
		.map_err(Error::from)?
		.push_next(&mut rc)
		.map_err(Error::from)?;
	if rc_enabled && args.rate_control_reset {
		begin = match args.codec {
			EncodeCodec::H264 => begin.push_next(&mut h264_rc).map_err(Error::from)?,
			EncodeCodec::H265 => begin.push_next(&mut h265_rc).map_err(Error::from)?,
		};
	}
	// SAFETY: the session, parameters, slot resources, and rate-control chain
	// all remain live for this recording.
	unsafe {
		device.cmd_begin_video_coding_khr(command_buffer, &begin)?;
	}

	// First scope: RESET the encoder and program the selected rate-control
	// state explicitly (RESET alone returns to DEFAULT, which rejects slice QP).
	if !args.rate_control_reset {
		let mut control_rc = vk::VideoEncodeRateControlInfoKHR::new();
		let mut control_layer = vk::VideoEncodeRateControlLayerInfoKHR::new();
		let mut control_h264_rc = vk::VideoEncodeH264RateControlInfoKHR::new();
		let mut control_h264_layer = vk::VideoEncodeH264RateControlLayerInfoKHR::new();
		let mut control_h265_rc = vk::VideoEncodeH265RateControlInfoKHR::new();
		let mut control_h265_layer = vk::VideoEncodeH265RateControlLayerInfoKHR::new();
		control_rc = control_rc.set_rate_control_mode(if rc_enabled {
			args.rate_control_mode
		} else {
			vk::VideoEncodeRateControlModeFlagBitsKHR::DISABLED_BIT_KHR
		});
		if rc_enabled {
			match args.codec {
				EncodeCodec::H264 => {
					control_h264_layer = control_h264_layer
						.set_use_min_qp(vk::FALSE)
						.set_use_max_qp(vk::FALSE)
						.set_use_max_frame_size(vk::FALSE);
					control_h264_rc = control_h264_rc
						.set_gop_frame_count(args.gop_size)
						.set_idr_period(args.gop_size)
						.set_consecutive_b_frame_count(args.max_b_frames)
						.set_temporal_layer_count(1);
					control_layer = control_layer
						.push_next(&mut control_h264_layer)
						.map_err(Error::from)?;
				}
				EncodeCodec::H265 => {
					control_h265_layer = control_h265_layer
						.set_use_min_qp(vk::FALSE)
						.set_use_max_qp(vk::FALSE)
						.set_use_max_frame_size(vk::FALSE);
					control_h265_rc = control_h265_rc
						.set_gop_frame_count(args.gop_size)
						.set_idr_period(args.gop_size)
						.set_consecutive_b_frame_count(args.max_b_frames)
						.set_sub_layer_count(1);
					control_layer = control_layer
						.push_next(&mut control_h265_layer)
						.map_err(Error::from)?;
				}
			}
			control_layer = control_layer
				.set_average_bitrate(u64::from(args.bitrate))
				.set_max_bitrate(u64::from(
					if args.rate_control_mode == vk::VideoEncodeRateControlModeFlagBitsKHR::CBR_BIT_KHR {
						args.bitrate
					} else {
						args.peak_bitrate
					},
				))
				.set_frame_rate_numerator(args.frame_rate)
				.set_frame_rate_denominator(1);
			let layers = std::slice::from_ref(&control_layer);
			control_rc = control_rc
				.set_layer_count(1)
				.set_layers(layers)
				.map_err(Error::from)?
				.set_virtual_buffer_size_in_ms(1000)
				.set_initial_virtual_buffer_size_in_ms(500);
		}
		let mut control = vk::VideoCodingControlInfoKHR::new()
			.set_flags(
				vk::VideoCodingControlFlagBitsKHR::RESET_BIT_KHR
					| vk::VideoCodingControlFlagBitsKHR::ENCODE_RATE_CONTROL_BIT_KHR,
			)
			.push_next(&mut control_rc)
			.map_err(Error::from)?;
		if rc_enabled {
			control = match args.codec {
				EncodeCodec::H264 => control
					.push_next(&mut control_h264_rc)
					.map_err(Error::from)?,
				EncodeCodec::H265 => control
					.push_next(&mut control_h265_rc)
					.map_err(Error::from)?,
			};
		}
		unsafe {
			device.cmd_control_video_coding_khr(command_buffer, &control)?;
		}
	}

	unsafe {
		device.cmd_begin_query(
			command_buffer,
			args.query_pool,
			0,
			vk::QueryControlFlags::empty(),
		)?;
	}
	let src_resource = vk::VideoPictureResourceInfoKHR::new()
		.set_coded_extent(coded_extent)
		.set_base_array_layer(0)
		.set_image_view_binding(args.input_view);
	match args.codec {
		EncodeCodec::H264 => record_h264_picture(device, command_buffer, &args, src_resource)?,
		EncodeCodec::H265 => record_h265_picture_encode(device, command_buffer, &args, src_resource)?,
	}
	unsafe {
		device.cmd_end_query(command_buffer, args.query_pool, 0)?;
		device.cmd_end_video_coding_khr(command_buffer, &vk::VideoEndCodingInfoKHR::new())?;
	}
	Ok(())
}

#[allow(clippy::field_reassign_with_default)]
fn record_h264_picture(
	device: &vk::Device,
	command_buffer: vk::CommandBuffer,
	args: &EncodePictureArgs,
	src_resource: vk::VideoPictureResourceInfoKHR<'_>,
) -> Result<()> {
	// B-picture *emission* needs display-order reorder and a second DPB
	// reference — the donor budgets maxBFrames in rate control but encodes
	// only I/P slices.
	let mut ref_lists = vk::StdVideoEncodeH264ReferenceListsInfo::new();
	ref_lists.num_ref_idx_l0_active_minus1 = 0;
	ref_lists.num_ref_idx_l1_active_minus1 = 0;
	ref_lists
		.ref_pic_list0
		.fill(vk::STD_VIDEO_H264_NO_REFERENCE_PICTURE);
	ref_lists
		.ref_pic_list1
		.fill(vk::STD_VIDEO_H264_NO_REFERENCE_PICTURE);
	if !args.is_idr {
		ref_lists.ref_pic_list0[0] = args.previous_dpb_slot as u8;
	}
	let mut pic_flags = vk::StdVideoEncodeH264PictureInfoFlags::new();
	pic_flags.set_idr_pic_flag(u32::from(args.is_idr));
	pic_flags.set_is_reference(1);
	pic_flags.set_no_output_of_prior_pics_flag(u32::from(args.is_idr));
	pic_flags.set_long_term_reference_flag(0);
	pic_flags.set_adaptive_ref_pic_marking_mode_flag(0);
	let mut pic_info = vk::StdVideoEncodeH264PictureInfo::new();
	pic_info.flags = pic_flags;
	pic_info.seq_parameter_set_id = 0;
	pic_info.pic_parameter_set_id = 0;
	pic_info.idr_pic_id = args.idr_pic_id;
	pic_info.primary_pic_type = if args.is_idr {
		vk::StdVideoH264PictureType::IDR
	} else {
		vk::StdVideoH264PictureType::P
	};
	pic_info.frame_num = if args.is_idr { 0 } else { args.frame_index };
	pic_info.pic_order_cnt = if args.is_idr {
		0
	} else {
		args.frame_index as i32
	};
	pic_info.p_ref_lists = &ref_lists;

	let mut slice_flags = vk::StdVideoEncodeH264SliceHeaderFlags::new();
	slice_flags.set_direct_spatial_mv_pred_flag(0);
	slice_flags.set_num_ref_idx_active_override_flag(0);
	let mut slice_header = vk::StdVideoEncodeH264SliceHeader::new();
	slice_header.flags = slice_flags;
	slice_header.first_mb_in_slice = 0;
	slice_header.slice_type = if args.is_idr {
		vk::StdVideoH264SliceType::I
	} else {
		vk::StdVideoH264SliceType::P
	};
	slice_header.cabac_init_idc = vk::StdVideoH264CabacInitIdc::_0;
	slice_header.disable_deblocking_filter_idc = vk::StdVideoH264DisableDeblockingFilterIdc::DISABLED;
	slice_header.slice_alpha_c0_offset_div2 = 0;
	slice_header.slice_beta_offset_div2 = 0;

	let nalu = vk::VideoEncodeH264NaluSliceInfoKHR::new()
		.set_constant_qp(args.constant_qp as i32)
		.set_std_slice_header(&slice_header);
	let nalus = std::slice::from_ref(&nalu);
	let mut h264_picture = vk::VideoEncodeH264PictureInfoKHR::new()
		.set_nalu_slice_entries(nalus)
		.map_err(Error::from)?
		.set_std_picture_info(&pic_info)
		.set_generate_prefix_nalu(vk::FALSE);

	// Setup slot: the current DPB layer being written this frame.
	let mut current_ref = vk::StdVideoEncodeH264ReferenceInfo::new();
	current_ref.primary_pic_type = if args.is_idr {
		vk::StdVideoH264PictureType::IDR
	} else {
		vk::StdVideoH264PictureType::P
	};
	current_ref.frame_num = pic_info.frame_num;
	current_ref.pic_order_cnt = pic_info.pic_order_cnt;
	let mut current_slot =
		vk::VideoEncodeH264DpbSlotInfoKHR::new().set_std_reference_info(&current_ref);
	let setup_resource = vk::VideoPictureResourceInfoKHR::new()
		.set_coded_extent(vk::Extent2D {
			width: args.coded.width,
			height: args.coded.height,
		})
		.set_base_array_layer(args.current_dpb_slot)
		.set_image_view_binding(args.dpb_view);
	let setup_slot = vk::VideoReferenceSlotInfoKHR::new()
		.set_slot_index(args.current_dpb_slot as i32)
		.set_picture_resource(&setup_resource)
		.push_next(&mut current_slot)
		.map_err(Error::from)?;

	let mut previous_ref = vk::StdVideoEncodeH264ReferenceInfo::new();
	previous_ref.primary_pic_type = vk::StdVideoH264PictureType::P;
	previous_ref.frame_num = args.frame_index.saturating_sub(1);
	previous_ref.pic_order_cnt = args.frame_index as i32 - 1;
	let mut previous_slot_info =
		vk::VideoEncodeH264DpbSlotInfoKHR::new().set_std_reference_info(&previous_ref);
	let previous_resource = vk::VideoPictureResourceInfoKHR::new()
		.set_coded_extent(vk::Extent2D {
			width: args.coded.width,
			height: args.coded.height,
		})
		.set_base_array_layer(args.previous_dpb_slot)
		.set_image_view_binding(args.dpb_view);
	let previous_slot = vk::VideoReferenceSlotInfoKHR::new()
		.set_slot_index(args.previous_dpb_slot as i32)
		.set_picture_resource(&previous_resource)
		.push_next(&mut previous_slot_info)
		.map_err(Error::from)?;
	let reference_slots = [previous_slot];
	let mut encode = vk::VideoEncodeInfoKHR::new()
		.set_dst_buffer(args.bitstream)
		.set_dst_buffer_offset(0)
		.set_dst_buffer_range(args.bitstream_range)
		.set_src_picture_resource(src_resource)
		.set_setup_reference_slot(&setup_slot)
		.push_next(&mut h264_picture)
		.map_err(Error::from)?;
	if !args.is_idr {
		encode = encode
			.set_reference_slots(&reference_slots)
			.map_err(Error::from)?;
	}
	// SAFETY: session coding scope is open; every referenced slot resource,
	// parameter chain, and bitstream buffer outlives submission via the
	// recorded-command retention on the engine timeline.
	unsafe {
		device.cmd_encode_video_khr(command_buffer, &encode)?;
	}
	Ok(())
}

#[allow(clippy::field_reassign_with_default)]
fn record_h265_picture_encode(
	device: &vk::Device,
	command_buffer: vk::CommandBuffer,
	args: &EncodePictureArgs,
	src_resource: vk::VideoPictureResourceInfoKHR<'_>,
) -> Result<()> {
	let mut ref_lists = vk::StdVideoEncodeH265ReferenceListsInfo::new();
	ref_lists
		.ref_pic_list0
		.fill(vk::STD_VIDEO_H265_NO_REFERENCE_PICTURE);
	ref_lists
		.ref_pic_list1
		.fill(vk::STD_VIDEO_H265_NO_REFERENCE_PICTURE);
	ref_lists
		.list_entry_l0
		.fill(vk::STD_VIDEO_H265_NO_REFERENCE_PICTURE);
	ref_lists
		.list_entry_l1
		.fill(vk::STD_VIDEO_H265_NO_REFERENCE_PICTURE);
	if !args.is_idr {
		ref_lists.num_ref_idx_l0_active_minus1 = 0;
		ref_lists.ref_pic_list0[0] = args.previous_dpb_slot as u8;
	}
	let picture_poc = if args.is_idr {
		0
	} else {
		args.frame_index as i32
	};
	let mut pic_flags = vk::StdVideoEncodeH265PictureInfoFlags::new();
	pic_flags.set_is_reference(1);
	pic_flags.set_irap_pic_flag(u32::from(args.is_idr));
	pic_flags.set_pic_output_flag(1);
	pic_flags.set_no_output_of_prior_pics_flag(u32::from(args.is_idr && !args.first_frame));
	pic_flags.set_short_term_ref_pic_set_sps_flag(1);
	let mut pic_info = vk::StdVideoEncodeH265PictureInfo::new();
	pic_info.flags = pic_flags;
	pic_info.pic_type = if args.is_idr {
		vk::StdVideoH265PictureType::IDR
	} else {
		vk::StdVideoH265PictureType::P
	};
	pic_info.sps_video_parameter_set_id = 0;
	pic_info.pps_seq_parameter_set_id = 0;
	pic_info.pps_pic_parameter_set_id = 0;
	pic_info.short_term_ref_pic_set_idx = 0;
	pic_info.pic_order_cnt_val = picture_poc;
	pic_info.temporal_id = 0;
	pic_info.p_ref_lists = if args.is_idr {
		std::ptr::null()
	} else {
		&ref_lists
	};

	let mut slice_flags = vk::StdVideoEncodeH265SliceSegmentHeaderFlags::new();
	slice_flags.set_first_slice_segment_in_pic_flag(1);
	slice_flags.set_slice_sao_luma_flag(1);
	slice_flags.set_slice_sao_chroma_flag(1);
	let mut slice_header = vk::StdVideoEncodeH265SliceSegmentHeader::new();
	slice_header.flags = slice_flags;
	slice_header.slice_type = if args.is_idr {
		vk::StdVideoH265SliceType::I
	} else {
		vk::StdVideoH265SliceType::P
	};
	slice_header.max_num_merge_cand = 5;

	let nalu = vk::VideoEncodeH265NaluSliceSegmentInfoKHR::new()
		.set_constant_qp(args.constant_qp as i32)
		.set_std_slice_segment_header(&slice_header);
	let nalus = std::slice::from_ref(&nalu);
	let mut h265_picture = vk::VideoEncodeH265PictureInfoKHR::new()
		.set_nalu_slice_segment_entries(nalus)
		.map_err(Error::from)?
		.set_std_picture_info(&pic_info);

	let mut current_ref = vk::StdVideoEncodeH265ReferenceInfo::new();
	current_ref.pic_type = vk::StdVideoH265PictureType::P;
	current_ref.pic_order_cnt_val = picture_poc;
	current_ref.temporal_id = 0;
	let mut current_slot =
		vk::VideoEncodeH265DpbSlotInfoKHR::new().set_std_reference_info(&current_ref);
	let setup_resource = vk::VideoPictureResourceInfoKHR::new()
		.set_coded_extent(vk::Extent2D {
			width: args.coded.width,
			height: args.coded.height,
		})
		.set_base_array_layer(args.current_dpb_slot)
		.set_image_view_binding(args.dpb_view);
	let setup_slot = vk::VideoReferenceSlotInfoKHR::new()
		.set_slot_index(args.current_dpb_slot as i32)
		.set_picture_resource(&setup_resource)
		.push_next(&mut current_slot)
		.map_err(Error::from)?;

	let mut previous_ref = vk::StdVideoEncodeH265ReferenceInfo::new();
	previous_ref.pic_type = vk::StdVideoH265PictureType::P;
	previous_ref.pic_order_cnt_val = picture_poc - 1;
	previous_ref.temporal_id = 0;
	let mut previous_slot_info =
		vk::VideoEncodeH265DpbSlotInfoKHR::new().set_std_reference_info(&previous_ref);
	let previous_resource = vk::VideoPictureResourceInfoKHR::new()
		.set_coded_extent(vk::Extent2D {
			width: args.coded.width,
			height: args.coded.height,
		})
		.set_base_array_layer(args.previous_dpb_slot)
		.set_image_view_binding(args.dpb_view);
	let previous_slot = vk::VideoReferenceSlotInfoKHR::new()
		.set_slot_index(args.previous_dpb_slot as i32)
		.set_picture_resource(&previous_resource)
		.push_next(&mut previous_slot_info)
		.map_err(Error::from)?;
	let reference_slots = [previous_slot];
	let mut encode = vk::VideoEncodeInfoKHR::new()
		.set_dst_buffer(args.bitstream)
		.set_dst_buffer_offset(0)
		.set_dst_buffer_range(args.bitstream_range)
		.set_src_picture_resource(src_resource)
		.set_setup_reference_slot(&setup_slot)
		.push_next(&mut h265_picture)
		.map_err(Error::from)?;
	if !args.is_idr {
		encode = encode
			.set_reference_slots(&reference_slots)
			.map_err(Error::from)?;
	}
	// SAFETY: identical session/scope/lifetime contract as the H.264 path.
	unsafe {
		device.cmd_encode_video_khr(command_buffer, &encode)?;
	}
	Ok(())
}

// ── Manufactured standard headers ──────────────────────────────────────

#[allow(clippy::field_reassign_with_default)]
fn build_h264_sps(
	coded: video::VideoExtent,
	level: u32,
) -> Result<vk::StdVideoH264SequenceParameterSet<'static>> {
	let mut flags = vk::StdVideoH264SpsFlags::new();
	flags.set_direct_8x8_inference_flag(1);
	flags.set_frame_mbs_only_flag(1);
	let mut sps = vk::StdVideoH264SequenceParameterSet::new();
	sps.flags = flags;
	sps.profile_idc = vk::StdVideoH264ProfileIdc::HIGH;
	sps.level_idc = std_h264_level(level)?;
	sps.chroma_format_idc = vk::StdVideoH264ChromaFormatIdc::_420;
	sps.seq_parameter_set_id = 0;
	sps.bit_depth_luma_minus8 = 0;
	sps.bit_depth_chroma_minus8 = 0;
	sps.log2_max_frame_num_minus4 = 4;
	sps.pic_order_cnt_type = vk::StdVideoH264PocType::_0;
	sps.offset_for_non_ref_pic = 0;
	sps.offset_for_top_to_bottom_field = 0;
	sps.log2_max_pic_order_cnt_lsb_minus4 = 4;
	sps.num_ref_frames_in_pic_order_cnt_cycle = 0;
	sps.max_num_ref_frames = 1;
	let width_mbs = coded.width.div_ceil(16);
	let height_mbs = coded.height.div_ceil(16);
	sps.pic_width_in_mbs_minus1 = width_mbs.saturating_sub(1);
	sps.pic_height_in_map_units_minus1 = height_mbs.saturating_sub(1);
	Ok(sps)
}

#[allow(clippy::field_reassign_with_default)]
fn build_h264_pps() -> vk::StdVideoH264PictureParameterSet<'static> {
	let mut flags = vk::StdVideoH264PpsFlags::new();
	flags.set_transform_8x8_mode_flag(1);
	flags.set_deblocking_filter_control_present_flag(1);
	flags.set_entropy_coding_mode_flag(1); // CABAC
	let mut pps = vk::StdVideoH264PictureParameterSet::new();
	pps.flags = flags;
	pps.seq_parameter_set_id = 0;
	pps.pic_parameter_set_id = 0;
	pps.num_ref_idx_l0_default_active_minus1 = 0;
	pps.num_ref_idx_l1_default_active_minus1 = 0;
	pps.weighted_bipred_idc = vk::StdVideoH264WeightedBipredIdc::DEFAULT;
	pps
}

struct H265Parameters<'a> {
	profile_tier_level: vk::StdVideoH265ProfileTierLevel,
	dpb_manager: vk::StdVideoH265DecPicBufMgr,
	short_term_ref_pic_set: vk::StdVideoH265ShortTermRefPicSet,
	vps: vk::StdVideoH265VideoParameterSet<'a>,
	sps: vk::StdVideoH265SequenceParameterSet<'a>,
	pps: vk::StdVideoH265PictureParameterSet<'a>,
}

impl H265Parameters<'_> {
	/// Point the parameter sets at this arena's leaf storage. Call after the
	/// value has reached its final stack position; the copies handed to the
	/// driver then reference `self`, not the build frame.
	fn link(&mut self) {
		self.vps.p_profile_tier_level = std::ptr::from_ref(&self.profile_tier_level);
		self.vps.p_dec_pic_buf_mgr = std::ptr::from_ref(&self.dpb_manager);
		self.sps.p_profile_tier_level = std::ptr::from_ref(&self.profile_tier_level);
		self.sps.p_dec_pic_buf_mgr = std::ptr::from_ref(&self.dpb_manager);
		self.sps.p_short_term_ref_pic_set = std::ptr::from_ref(&self.short_term_ref_pic_set);
	}
}

#[allow(clippy::too_many_arguments)]
fn build_h265_parameters(
	settings: &video::VideoEncodeConfig,
	coded: video::VideoExtent,
	dpb_slots: u32,
	ctb_16: bool,
	ctb_32: bool,
	ctb_64: bool,
	ts_4: bool,
	ts_8: bool,
	ts_16: bool,
	ts_32: bool,
) -> H265Parameters<'static> {
	let ctb_log2 = if ctb_16 {
		4
	} else if ctb_32 {
		5
	} else if ctb_64 {
		6
	} else {
		// No advertised CTB size — keep the donor's TGL-fallback geometry;
		// preflight admits at least one size on every qualified device.
		6
	};
	let min_transform_log2 = if ts_4 {
		2
	} else if ts_8 {
		3
	} else {
		4
	};
	let max_transform_log2 = if ctb_log2 >= 5 && ts_32 {
		5
	} else if ts_16 {
		4
	} else if ts_8 {
		3
	} else {
		min_transform_log2
	};
	let mut out = H265Parameters {
		profile_tier_level: vk::StdVideoH265ProfileTierLevel::new(),
		dpb_manager: vk::StdVideoH265DecPicBufMgr::new(),
		short_term_ref_pic_set: vk::StdVideoH265ShortTermRefPicSet::new(),
		vps: vk::StdVideoH265VideoParameterSet::new(),
		sps: vk::StdVideoH265SequenceParameterSet::new(),
		pps: vk::StdVideoH265PictureParameterSet::new(),
	};
	let mut ptl_flags = vk::StdVideoH265ProfileTierLevelFlags::new();
	ptl_flags.set_general_progressive_source_flag(1);
	ptl_flags.set_general_frame_only_constraint_flag(1);
	out.profile_tier_level.flags = ptl_flags;
	out.profile_tier_level.general_profile_idc = vk::StdVideoH265ProfileIdc::MAIN;
	out.profile_tier_level.general_level_idc = vk::StdVideoH265LevelIdc::_4_1;
	out.dpb_manager.max_dec_pic_buffering_minus1[0] = dpb_slots.saturating_sub(1) as u8;
	out.dpb_manager.max_num_reorder_pics[0] = 0;
	out.dpb_manager.max_latency_increase_plus1[0] = 0;
	// One flat L0 reference: the immediately preceding I/P picture.
	out.short_term_ref_pic_set.used_by_curr_pic_s0_flag = 1;
	out.short_term_ref_pic_set.num_negative_pics = 1;
	out.short_term_ref_pic_set.num_positive_pics = 0;
	out.short_term_ref_pic_set.delta_poc_s0_minus1[0] = 0;
	let mut vps_flags = vk::StdVideoH265VpsFlags::new();
	vps_flags.set_vps_temporal_id_nesting_flag(1);
	vps_flags.set_vps_sub_layer_ordering_info_present_flag(1);
	out.vps.flags = vps_flags;
	out.vps.vps_video_parameter_set_id = 0;
	out.vps.vps_max_sub_layers_minus1 = 0;
	let mut sps_flags = vk::StdVideoH265SpsFlags::new();
	sps_flags.set_sps_temporal_id_nesting_flag(1);
	sps_flags.set_sps_sub_layer_ordering_info_present_flag(1);
	sps_flags.set_amp_enabled_flag(1);
	sps_flags.set_sample_adaptive_offset_enabled_flag(1);
	sps_flags.set_conformance_window_flag(u32::from(
		coded.width != settings.width || coded.height != settings.height,
	));
	out.sps.flags = sps_flags;
	out.sps.chroma_format_idc = vk::StdVideoH265ChromaFormatIdc::_420;
	out.sps.pic_width_in_luma_samples = coded.width;
	out.sps.pic_height_in_luma_samples = coded.height;
	out.sps.sps_video_parameter_set_id = 0;
	out.sps.sps_max_sub_layers_minus1 = 0;
	out.sps.sps_seq_parameter_set_id = 0;
	out.sps.bit_depth_luma_minus8 = 0;
	out.sps.bit_depth_chroma_minus8 = 0;
	out.sps.log2_max_pic_order_cnt_lsb_minus4 = 4;
	// MinCb stays 16 while CTB/transform geometry follows the capability
	// masks — hard-coding 16 on TGL produces a valid-looking hang.
	const MIN_CODING_BLOCK_LOG2: u32 = 4;
	out.sps.log2_min_luma_coding_block_size_minus3 = (MIN_CODING_BLOCK_LOG2 - 3) as u8;
	out.sps.log2_diff_max_min_luma_coding_block_size = (ctb_log2 - MIN_CODING_BLOCK_LOG2) as u8;
	out.sps.log2_min_luma_transform_block_size_minus2 = (min_transform_log2 - 2) as u8;
	out.sps.log2_diff_max_min_luma_transform_block_size =
		(max_transform_log2 - min_transform_log2) as u8;
	out.sps.max_transform_hierarchy_depth_inter = (ctb_log2 - min_transform_log2).max(1) as u8;
	out.sps.max_transform_hierarchy_depth_intra = 3;
	out.sps.pcm_sample_bit_depth_luma_minus1 = 7;
	out.sps.pcm_sample_bit_depth_chroma_minus1 = 7;
	out.sps.log2_min_pcm_luma_coding_block_size_minus3 = (MIN_CODING_BLOCK_LOG2 - 3) as u8;
	out.sps.log2_diff_max_min_pcm_luma_coding_block_size = (ctb_log2 - MIN_CODING_BLOCK_LOG2) as u8;
	out.sps.num_short_term_ref_pic_sets = 1;
	out.sps.conf_win_right_offset = (coded.width - settings.width) / 2;
	out.sps.conf_win_bottom_offset = (coded.height - settings.height) / 2;
	let mut pps_flags = vk::StdVideoH265PpsFlags::new();
	pps_flags.set_cabac_init_present_flag(1);
	pps_flags.set_transform_skip_enabled_flag(1);
	pps_flags.set_cu_qp_delta_enabled_flag(1);
	pps_flags.set_pps_loop_filter_across_slices_enabled_flag(1);
	pps_flags.set_deblocking_filter_control_present_flag(1);
	out.pps.flags = pps_flags;
	out.pps.pps_pic_parameter_set_id = 0;
	out.pps.pps_seq_parameter_set_id = 0;
	out.pps.sps_video_parameter_set_id = 0;
	out.pps.num_ref_idx_l0_default_active_minus1 = 0;
	out.pps.num_ref_idx_l1_default_active_minus1 = 0;
	out.pps.init_qp_minus26 = 0;
	out.pps.log2_parallel_merge_level_minus2 = 0;
	out
}

impl Drop for EncodeSession {
	fn drop(&mut self) {
		if self.parameters != vk::VideoSessionParametersKHR::default() {
			// SAFETY: this wrapper owns the parameter object and destroys it
			// before the video session it depends on.
			unsafe {
				let _ = self
					.device
					.raw()
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
					.raw()
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
		if self.dpb.view != vk::ImageView::default() {
			// SAFETY: the session owns the layered DPB view/image/allocation.
			unsafe {
				let _ = self.device.raw().destroy_image_view(self.dpb.view, None);
				let _ = self
					.device
					.allocator()
					.with_allocator(|a| a.destroy_image(self.dpb.handle, &mut self.dpb.allocation));
			}
		}
	}
}
