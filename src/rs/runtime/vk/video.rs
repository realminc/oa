use std::sync::{Arc, Mutex};

use crate::{Error, Event, Result, core::memory, video};
use vk_mem::Alloc;

use super::{Device, Instance, PhysicalDevice};

mod av1;
mod vp9;

const MAX_VIDEO_FORMATS: u32 = 256;
const MAX_VIDEO_SESSION_MEMORY_BINDINGS: u32 = 64;

struct DecodeCapabilityQuery {
	capabilities: video::VideoDecodeCapabilities,
	std_header_version: ash::vk::ExtensionProperties,
}

pub(in crate::runtime) struct DecodeSession {
	handle: ash::vk::VideoSessionKHR,
	parameters: ash::vk::VideoSessionParametersKHR,
	result_status_pool: ash::vk::QueryPool,
	allocations: Vec<vk_mem::Allocation>,
	images: Arc<DecodeImageStorage>,
	native_leases: Option<Arc<NativeFrameLeases>>,
	image_set: DecodeImageSet,
	bitstream: Option<DecodeBitstream>,
	readback: Option<super::Buffer>,
	bitstream_size_alignment: u64,
	coded_extent: video::VideoExtent,
	decode_recorded: bool,
	readback_recorded: bool,
	loader: ash::khr::video_queue::Device,
	decode_loader: ash::khr::video_decode_queue::Device,
	device: Device,
	profile: video::VideoDecodeProfile,
	h264_dpb_state: Option<video::H264DpbState>,
	h265_dpb_state: Option<H265DpbState>,
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
	handle: ash::vk::Image,
	view: ash::vk::ImageView,
	allocation: vk_mem::Allocation,
	array_layers: u32,
	format: ash::vk::Format,
}

struct DecodeImageStorage {
	device: Device,
	images: Vec<DecodeImage>,
}

struct NativeFrameLeases {
	device: Device,
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
		device: &Device,
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
				self.device.raw().destroy_image_view(image.view, None);
			}
			let mut allocation = image.allocation;
			// SAFETY: no retained view or frame lease remains for this image.
			unsafe {
				self
					.device
					.allocator()
					.destroy_image(image.handle, &mut allocation);
			}
		}
	}
}

struct DecodeBitstream {
	handle: ash::vk::Buffer,
	allocation: vk_mem::Allocation,
	payload_len: usize,
	range: ash::vk::DeviceSize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct H265DpbSlotState {
	in_use: bool,
	is_reference: bool,
	picture_order_count: i32,
	decode_index: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct H265DpbState {
	slots: Vec<H265DpbSlotState>,
	previous_poc_lsb: i32,
	previous_poc_msb: i32,
	has_previous_poc: bool,
	decode_index: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct H265PicturePlan {
	picture_order_count: i32,
	setup_slot: u32,
	active_references: Vec<H265ReferencePlan>,
	current_before_slots: Vec<u8>,
	current_after_slots: Vec<u8>,
	reset_dpb: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct H265ReferencePlan {
	slot: u32,
	picture_order_count: i32,
}

impl H265DpbState {
	fn new(slot_count: u32) -> Result<Self> {
		if slot_count == 0 || slot_count > 16 {
			return Err(Error::invalid_argument(
				"H.265 DPB state requires 1..=16 slots",
			));
		}
		Ok(Self {
			slots: vec![H265DpbSlotState::default(); slot_count as usize],
			previous_poc_lsb: 0,
			previous_poc_msb: 0,
			has_previous_poc: false,
			decode_index: 0,
		})
	}

	#[cfg(test)]
	fn plan(
		&mut self,
		sps: &video::H265SequenceParameterSet,
		slice: &video::H265SliceHeader,
	) -> Result<H265PicturePlan> {
		self.plan_with_unavailable(sps, slice, &[])
	}

	fn plan_with_unavailable(
		&mut self,
		sps: &video::H265SequenceParameterSet,
		slice: &video::H265SliceHeader,
		unavailable: &[bool],
	) -> Result<H265PicturePlan> {
		if !unavailable.is_empty() && unavailable.len() != self.slots.len() {
			return Err(Error::invalid_argument(
				"H.265 unavailable-slot mask does not match DPB capacity",
			));
		}
		let mut next = self.clone();
		let plan = next.plan_in_place(sps, slice, unavailable)?;
		*self = next;
		Ok(plan)
	}

	fn plan_in_place(
		&mut self,
		sps: &video::H265SequenceParameterSet,
		slice: &video::H265SliceHeader,
		unavailable: &[bool],
	) -> Result<H265PicturePlan> {
		let poc_bits = sps
			.log2_max_pic_order_count_lsb_minus_4
			.checked_add(4)
			.ok_or_else(|| Error::data_loss("H.265 POC width overflows u32"))?;
		if !(4..=16).contains(&poc_bits) {
			return Err(Error::data_loss("H.265 POC width is outside 4..=16 bits"));
		}
		let max_poc_lsb = 1_i32
			.checked_shl(poc_bits)
			.ok_or_else(|| Error::data_loss("H.265 maximum POC LSB exceeds i32"))?;
		let poc_lsb =
			if slice.is_idr {
				0
			} else {
				i32::try_from(slice.picture_order_count_lsb.ok_or_else(|| {
					Error::data_loss("non-IDR H.265 picture omitted picture-order-count LSB")
				})?)
				.map_err(|_| Error::data_loss("H.265 picture-order-count LSB exceeds i32"))?
			};
		if poc_lsb >= max_poc_lsb {
			return Err(Error::data_loss(
				"H.265 picture-order-count LSB exceeds the SPS width",
			));
		}
		let is_bla = (16..=18).contains(&slice.nal_unit_type);
		let reset_dpb = slice.is_idr || is_bla || slice.no_output_of_prior_pictures;
		let mut poc_msb = self.previous_poc_msb;
		if reset_dpb || !self.has_previous_poc {
			poc_msb = 0;
		} else if poc_lsb < self.previous_poc_lsb && self.previous_poc_lsb - poc_lsb >= max_poc_lsb / 2
		{
			poc_msb = poc_msb
				.checked_add(max_poc_lsb)
				.ok_or_else(|| Error::out_of_range("H.265 POC MSB exceeds i32"))?;
		} else if poc_lsb > self.previous_poc_lsb && poc_lsb - self.previous_poc_lsb > max_poc_lsb / 2 {
			poc_msb = poc_msb
				.checked_sub(max_poc_lsb)
				.ok_or_else(|| Error::out_of_range("H.265 POC MSB is below i32"))?;
		}
		let picture_order_count = poc_msb
			.checked_add(poc_lsb)
			.ok_or_else(|| Error::out_of_range("H.265 picture order count exceeds i32"))?;

		if reset_dpb {
			self.slots.fill(H265DpbSlotState::default());
		}
		let retained = |poc: i32| {
			slice
				.short_term_current_before_delta_pocs
				.iter()
				.chain(&slice.short_term_current_after_delta_pocs)
				.chain(&slice.short_term_following_delta_pocs)
				.any(|delta| picture_order_count.checked_add(*delta) == Some(poc))
		};
		for slot in &mut self.slots {
			if slot.in_use && slot.is_reference && !retained(slot.picture_order_count) {
				*slot = H265DpbSlotState::default();
			}
		}

		let setup_index = self
			.slots
			.iter()
			.enumerate()
			.find(|(index, slot)| !slot.in_use && !unavailable.get(*index).copied().unwrap_or(false))
			.map(|(index, _)| index)
			.or_else(|| {
				self
					.slots
					.iter()
					.enumerate()
					.filter(|(index, slot)| {
						!slot.is_reference && !unavailable.get(*index).copied().unwrap_or(false)
					})
					.min_by_key(|(_, slot)| slot.decode_index)
					.map(|(index, _)| index)
			})
			.ok_or_else(|| Error::resource_exhausted("H.265 DPB has no unleased recyclable slot"))?;

		let resolve = |deltas: &[i32]| -> Result<Vec<u8>> {
			let mut resolved = Vec::new();
			resolved
				.try_reserve_exact(deltas.len())
				.map_err(|_| Error::resource_exhausted("H.265 reference list allocation failed"))?;
			for delta in deltas {
				let target = picture_order_count
					.checked_add(*delta)
					.ok_or_else(|| Error::out_of_range("H.265 reference POC exceeds i32"))?;
				let index = self
					.slots
					.iter()
					.position(|slot| slot.in_use && slot.is_reference && slot.picture_order_count == target)
					.ok_or_else(|| Error::data_loss("H.265 reference picture is absent from DPB"))?;
				resolved.push(
					u8::try_from(index).map_err(|_| Error::internal("H.265 DPB slot index exceeds u8"))?,
				);
			}
			Ok(resolved)
		};
		let current_before_slots = resolve(&slice.short_term_current_before_delta_pocs)?;
		let current_after_slots = resolve(&slice.short_term_current_after_delta_pocs)?;
		let active_references = self
			.slots
			.iter()
			.enumerate()
			.filter(|(_, slot)| slot.in_use && slot.is_reference)
			.map(|(index, state)| {
				Ok(H265ReferencePlan {
					slot: u32::try_from(index)
						.map_err(|_| Error::internal("H.265 DPB slot index exceeds u32"))?,
					picture_order_count: state.picture_order_count,
				})
			})
			.collect::<Result<Vec<_>>>()?;

		let is_leading = (6..=9).contains(&slice.nal_unit_type);
		if slice.temporal_id == 0 && slice.is_reference && !is_leading {
			self.previous_poc_lsb = poc_lsb;
			self.previous_poc_msb = poc_msb;
			self.has_previous_poc = true;
		}
		self.decode_index = self
			.decode_index
			.checked_add(1)
			.ok_or_else(|| Error::resource_exhausted("H.265 decode index exhausted"))?;
		self.slots[setup_index] = if slice.is_reference {
			H265DpbSlotState {
				in_use: true,
				is_reference: true,
				picture_order_count,
				decode_index: self.decode_index,
			}
		} else {
			H265DpbSlotState::default()
		};
		Ok(H265PicturePlan {
			picture_order_count,
			setup_slot: u32::try_from(setup_index)
				.map_err(|_| Error::internal("H.265 DPB slot index exceeds u32"))?,
			active_references,
			current_before_slots,
			current_after_slots,
			reset_dpb,
		})
	}
}

pub(super) fn query_decode_capabilities(
	instance: &Instance,
	physical: &PhysicalDevice,
	profile: video::VideoDecodeProfile,
) -> Result<video::VideoDecodeCapabilities> {
	Ok(query_decode_details(instance, physical, profile)?.capabilities)
}

fn query_decode_details(
	instance: &Instance,
	physical: &PhysicalDevice,
	profile: video::VideoDecodeProfile,
) -> Result<DecodeCapabilityQuery> {
	ensure_extension_advertised(physical, profile)?;
	let loader = ash::khr::video_queue::Instance::new(instance.entry(), instance.raw());

	match profile {
		video::VideoDecodeProfile::H264 {
			profile: codec_profile,
			picture_layout,
			chroma_subsampling,
			luma_bit_depth,
			chroma_bit_depth,
		} => {
			let mut codec_profile = ash::vk::VideoDecodeH264ProfileInfoKHR::default()
				.std_profile_idc(h264_profile(codec_profile))
				.picture_layout(h264_picture_layout(picture_layout));
			let vk_profile = common_profile(
				ash::vk::VideoCodecOperationFlagsKHR::DECODE_H264,
				chroma_subsampling,
				luma_bit_depth,
				chroma_bit_depth,
			)
			.push_next(&mut codec_profile);
			let mut decode = ash::vk::VideoDecodeCapabilitiesKHR::default();
			let mut codec = ash::vk::VideoDecodeH264CapabilitiesKHR::default();
			let common = {
				let mut capabilities = ash::vk::VideoCapabilitiesKHR::default()
					.push_next(&mut decode)
					.push_next(&mut codec);
				query(&loader, physical.handle, &vk_profile, &mut capabilities)?;
				CommonCapabilities::from(&capabilities)
			};
			Ok(DecodeCapabilityQuery {
				capabilities: convert_capabilities(
					profile,
					common,
					&decode,
					video::VideoDecodeLevel::H264(codec.max_level_idc),
					Some((
						codec.field_offset_granularity.x,
						codec.field_offset_granularity.y,
					)),
				),
				std_header_version: common.std_header_version,
			})
		}
		video::VideoDecodeProfile::H265 {
			profile: codec_profile,
			chroma_subsampling,
			luma_bit_depth,
			chroma_bit_depth,
		} => {
			let mut codec_profile = ash::vk::VideoDecodeH265ProfileInfoKHR::default()
				.std_profile_idc(h265_profile(codec_profile));
			let vk_profile = common_profile(
				ash::vk::VideoCodecOperationFlagsKHR::DECODE_H265,
				chroma_subsampling,
				luma_bit_depth,
				chroma_bit_depth,
			)
			.push_next(&mut codec_profile);
			let mut decode = ash::vk::VideoDecodeCapabilitiesKHR::default();
			let mut codec = ash::vk::VideoDecodeH265CapabilitiesKHR::default();
			let common = {
				let mut capabilities = ash::vk::VideoCapabilitiesKHR::default()
					.push_next(&mut decode)
					.push_next(&mut codec);
				query(&loader, physical.handle, &vk_profile, &mut capabilities)?;
				CommonCapabilities::from(&capabilities)
			};
			Ok(DecodeCapabilityQuery {
				capabilities: convert_capabilities(
					profile,
					common,
					&decode,
					video::VideoDecodeLevel::H265(codec.max_level_idc),
					None,
				),
				std_header_version: common.std_header_version,
			})
		}
		video::VideoDecodeProfile::Av1 {
			profile: codec_profile,
			film_grain_support,
			chroma_subsampling,
			luma_bit_depth,
			chroma_bit_depth,
		} => {
			let mut codec_profile = ash::vk::VideoDecodeAV1ProfileInfoKHR::default()
				.std_profile(av1_profile(codec_profile))
				.film_grain_support(film_grain_support);
			let vk_profile = common_profile(
				ash::vk::VideoCodecOperationFlagsKHR::DECODE_AV1,
				chroma_subsampling,
				luma_bit_depth,
				chroma_bit_depth,
			)
			.push_next(&mut codec_profile);
			let mut decode = ash::vk::VideoDecodeCapabilitiesKHR::default();
			let mut codec = ash::vk::VideoDecodeAV1CapabilitiesKHR::default();
			let common = {
				let mut capabilities = ash::vk::VideoCapabilitiesKHR::default()
					.push_next(&mut decode)
					.push_next(&mut codec);
				query(&loader, physical.handle, &vk_profile, &mut capabilities)?;
				CommonCapabilities::from(&capabilities)
			};
			Ok(DecodeCapabilityQuery {
				capabilities: convert_capabilities(
					profile,
					common,
					&decode,
					video::VideoDecodeLevel::Av1(codec.max_level),
					None,
				),
				std_header_version: common.std_header_version,
			})
		}
		video::VideoDecodeProfile::Vp9 {
			profile: codec_profile,
			chroma_subsampling,
			luma_bit_depth,
			chroma_bit_depth,
		} => {
			let mut codec_profile = ash_vp9::vk::VideoDecodeVP9ProfileInfoKHR::default()
				.std_profile(vp9_profile(codec_profile));
			let mut vk_profile = common_profile(
				vp9_operation(),
				chroma_subsampling,
				luma_bit_depth,
				chroma_bit_depth,
			);
			vk_profile.p_next = std::ptr::from_mut(&mut codec_profile).cast();
			let mut codec = ash_vp9::vk::VideoDecodeVP9CapabilitiesKHR::default();
			let mut decode = ash::vk::VideoDecodeCapabilitiesKHR {
				p_next: std::ptr::from_mut(&mut codec).cast(),
				..Default::default()
			};
			let common = {
				let mut capabilities = ash::vk::VideoCapabilitiesKHR {
					p_next: std::ptr::from_mut(&mut decode).cast(),
					..Default::default()
				};
				query(&loader, physical.handle, &vk_profile, &mut capabilities)?;
				CommonCapabilities::from(&capabilities)
			};
			Ok(DecodeCapabilityQuery {
				capabilities: convert_capabilities(
					profile,
					common,
					&decode,
					video::VideoDecodeLevel::Vp9(codec.max_level),
					None,
				),
				std_header_version: common.std_header_version,
			})
		}
	}
}

pub(super) fn query_decode_formats(
	instance: &Instance,
	physical: &PhysicalDevice,
	profile: video::VideoDecodeProfile,
) -> Result<video::VideoDecodeFormats> {
	ensure_extension_advertised(physical, profile)?;
	let loader = ash::khr::video_queue::Instance::new(instance.entry(), instance.raw());
	with_decode_profile(profile, |vk_profile| {
		let output = query_formats(
			&loader,
			physical.handle,
			vk_profile,
			ash::vk::ImageUsageFlags::VIDEO_DECODE_DST_KHR,
		)?;
		let dpb = query_formats(
			&loader,
			physical.handle,
			vk_profile,
			ash::vk::ImageUsageFlags::VIDEO_DECODE_DPB_KHR,
		)?;
		let (output, unrecognized_output_formats) = convert_formats(&output);
		let (dpb, unrecognized_dpb_formats) = convert_formats(&dpb);
		Ok(video::VideoDecodeFormats {
			profile,
			output,
			dpb,
			unrecognized_output_formats,
			unrecognized_dpb_formats,
		})
	})
}

pub(super) fn query_encode_capabilities(
	instance: &Instance,
	physical: &PhysicalDevice,
	profile: video::VideoEncodeProfile,
) -> Result<video::VideoEncodeCapabilities> {
	ensure_encode_extension_advertised(physical, profile)?;
	let loader = ash::khr::video_queue::Instance::new(instance.entry(), instance.raw());
	match profile {
		video::VideoEncodeProfile::H264 {
			profile: codec_profile,
			chroma_subsampling,
			luma_bit_depth,
			chroma_bit_depth,
		} => {
			let mut codec_profile = ash::vk::VideoEncodeH264ProfileInfoKHR::default()
				.std_profile_idc(h264_profile(codec_profile));
			let vk_profile = common_profile(
				ash::vk::VideoCodecOperationFlagsKHR::ENCODE_H264,
				chroma_subsampling,
				luma_bit_depth,
				chroma_bit_depth,
			)
			.push_next(&mut codec_profile);
			let mut encode = ash::vk::VideoEncodeCapabilitiesKHR::default();
			let mut codec = ash::vk::VideoEncodeH264CapabilitiesKHR::default();
			let common = {
				let mut capabilities = ash::vk::VideoCapabilitiesKHR::default()
					.push_next(&mut encode)
					.push_next(&mut codec);
				query(&loader, physical.handle, &vk_profile, &mut capabilities)?;
				CommonCapabilities::from(&capabilities)
			};
			Ok(convert_encode_capabilities(
				profile,
				common,
				&encode,
				video::VideoEncodeCodecCapabilities::H264 {
					max_level: codec.max_level_idc,
					max_slice_count: codec.max_slice_count,
					max_p_l0_references: codec.max_p_picture_l0_reference_count,
					max_b_l0_references: codec.max_b_picture_l0_reference_count,
					max_l1_references: codec.max_l1_reference_count,
					max_temporal_layers: codec.max_temporal_layer_count,
					min_qp: codec.min_qp,
					max_qp: codec.max_qp,
				},
			))
		}
		video::VideoEncodeProfile::H265 {
			profile: codec_profile,
			chroma_subsampling,
			luma_bit_depth,
			chroma_bit_depth,
		} => {
			let mut codec_profile = ash::vk::VideoEncodeH265ProfileInfoKHR::default()
				.std_profile_idc(h265_profile(codec_profile));
			let vk_profile = common_profile(
				ash::vk::VideoCodecOperationFlagsKHR::ENCODE_H265,
				chroma_subsampling,
				luma_bit_depth,
				chroma_bit_depth,
			)
			.push_next(&mut codec_profile);
			let mut encode = ash::vk::VideoEncodeCapabilitiesKHR::default();
			let mut codec = ash::vk::VideoEncodeH265CapabilitiesKHR::default();
			let common = {
				let mut capabilities = ash::vk::VideoCapabilitiesKHR::default()
					.push_next(&mut encode)
					.push_next(&mut codec);
				query(&loader, physical.handle, &vk_profile, &mut capabilities)?;
				CommonCapabilities::from(&capabilities)
			};
			Ok(convert_encode_capabilities(
				profile,
				common,
				&encode,
				video::VideoEncodeCodecCapabilities::H265 {
					max_level: codec.max_level_idc,
					max_slice_segment_count: codec.max_slice_segment_count,
					max_tiles: extent(codec.max_tiles),
					ctb_size_16: codec
						.ctb_sizes
						.contains(ash::vk::VideoEncodeH265CtbSizeFlagsKHR::TYPE_16),
					ctb_size_32: codec
						.ctb_sizes
						.contains(ash::vk::VideoEncodeH265CtbSizeFlagsKHR::TYPE_32),
					ctb_size_64: codec
						.ctb_sizes
						.contains(ash::vk::VideoEncodeH265CtbSizeFlagsKHR::TYPE_64),
					transform_size_4: codec
						.transform_block_sizes
						.contains(ash::vk::VideoEncodeH265TransformBlockSizeFlagsKHR::TYPE_4),
					transform_size_8: codec
						.transform_block_sizes
						.contains(ash::vk::VideoEncodeH265TransformBlockSizeFlagsKHR::TYPE_8),
					transform_size_16: codec
						.transform_block_sizes
						.contains(ash::vk::VideoEncodeH265TransformBlockSizeFlagsKHR::TYPE_16),
					transform_size_32: codec
						.transform_block_sizes
						.contains(ash::vk::VideoEncodeH265TransformBlockSizeFlagsKHR::TYPE_32),
					max_p_l0_references: codec.max_p_picture_l0_reference_count,
					max_b_l0_references: codec.max_b_picture_l0_reference_count,
					max_l1_references: codec.max_l1_reference_count,
					max_sub_layers: codec.max_sub_layer_count,
					min_qp: codec.min_qp,
					max_qp: codec.max_qp,
				},
			))
		}
	}
}

pub(super) fn query_encode_formats(
	instance: &Instance,
	physical: &PhysicalDevice,
	profile: video::VideoEncodeProfile,
) -> Result<video::VideoEncodeFormats> {
	ensure_encode_extension_advertised(physical, profile)?;
	let loader = ash::khr::video_queue::Instance::new(instance.entry(), instance.raw());
	with_encode_profile(profile, |vk_profile| {
		let input = query_formats(
			&loader,
			physical.handle,
			vk_profile,
			ash::vk::ImageUsageFlags::VIDEO_ENCODE_SRC_KHR,
		)?;
		let dpb = query_formats(
			&loader,
			physical.handle,
			vk_profile,
			ash::vk::ImageUsageFlags::VIDEO_ENCODE_DPB_KHR,
		)?;
		let (input, unrecognized_input_formats) = convert_formats(&input);
		let (dpb, unrecognized_dpb_formats) = convert_formats(&dpb);
		Ok(video::VideoEncodeFormats {
			profile,
			input,
			dpb,
			unrecognized_input_formats,
			unrecognized_dpb_formats,
		})
	})
}

pub(super) fn create_decode_session(
	device: &Device,
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
	let instance_loader = ash::khr::video_queue::Instance::new(instance.entry(), instance.raw());
	let loader = ash::khr::video_queue::Device::new(instance.raw(), device.raw());
	let decode_loader = ash::khr::video_decode_queue::Device::new(instance.raw(), device.raw());
	with_decode_profile(profile, |vk_profile| {
		let output_usage =
			ash::vk::ImageUsageFlags::VIDEO_DECODE_DST_KHR | ash::vk::ImageUsageFlags::TRANSFER_SRC;
		let dpb_usage = ash::vk::ImageUsageFlags::VIDEO_DECODE_DPB_KHR;
		let (output_format, dpb_format, coincident_images) = if limits.dpb_and_output_coincide() {
			let combined_usage = output_usage | dpb_usage;
			let formats = query_formats(
				&instance_loader,
				physical.handle,
				vk_profile,
				combined_usage,
			)?;
			let format = select_image_format(formats, combined_usage, "coincident decode")?;
			(format, format, true)
		} else if limits.dpb_and_output_distinct() {
			let output_formats =
				query_formats(&instance_loader, physical.handle, vk_profile, output_usage)?;
			let dpb_formats = query_formats(&instance_loader, physical.handle, vk_profile, dpb_usage)?;
			(
				select_image_format(output_formats, output_usage, "decode output")?,
				select_image_format(dpb_formats, dpb_usage, "decode DPB")?,
				false,
			)
		} else {
			return Err(Error::missing_capability(
				"decode profile permits neither coincident nor distinct DPB/output images",
			));
		};
		let create_info = ash::vk::VideoSessionCreateInfoKHR::default()
			.queue_family_index(
				physical
					.video
					.decode_queue_family
					.ok_or_else(|| Error::missing_capability("no Vulkan Video decode queue is enabled"))?,
			)
			.video_profile(vk_profile)
			.picture_format(output_format.format)
			.max_coded_extent(ash::vk::Extent2D {
				width: coded_extent.width,
				height: coded_extent.height,
			})
			.reference_picture_format(dpb_format.format)
			.max_dpb_slots(max_dpb_slots)
			.max_active_reference_pictures(max_active_references)
			.std_header_version(&details.std_header_version);
		let mut handle = ash::vk::VideoSessionKHR::null();
		// SAFETY: the exact profile, supported formats, queried header version,
		// decode queue family, and bounded extent/counts remain live for this call.
		let result = unsafe {
			(loader.fp().create_video_session_khr)(
				loader.device(),
				&create_info,
				std::ptr::null(),
				&mut handle,
			)
		};
		if result != ash::vk::Result::SUCCESS {
			return Err(Error::backend_failure(
				"Vulkan",
				"video-session creation",
				result,
			));
		}
		let h264_dpb_state = if matches!(profile, video::VideoDecodeProfile::H264 { .. }) {
			Some(video::H264DpbState::new(max_dpb_slots.min(16))?)
		} else {
			None
		};
		let h265_dpb_state = if matches!(profile, video::VideoDecodeProfile::H265 { .. }) {
			Some(H265DpbState::new(max_dpb_slots)?)
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
			parameters: ash::vk::VideoSessionParametersKHR::null(),
			result_status_pool: ash::vk::QueryPool::null(),
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
			loader: loader.clone(),
			decode_loader,
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
			let mut query_profile = *vk_profile;
			let query_info = ash::vk::QueryPoolCreateInfo::default()
				.query_type(ash::vk::QueryType::RESULT_STATUS_ONLY_KHR)
				.query_count(1)
				.push_next(&mut query_profile);
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
			let allocation_info = vk_mem::AllocationCreateInfo {
				usage: vk_mem::MemoryUsage::Unknown,
				preferred_flags: ash::vk::MemoryPropertyFlags::DEVICE_LOCAL,
				memory_type_bits: requirement.memory_requirements.memory_type_bits,
				..Default::default()
			};
			// SAFETY: the requirements were returned for this live session, and VMA
			// constrains selection to the driver's advertised memory-type mask.
			let allocation = unsafe {
				session
					.device
					.allocator()
					.allocate_memory(&requirement.memory_requirements, &allocation_info)
			}
			.map_err(|source| {
				Error::backend_failure("Vulkan", "video-session memory allocation", source)
			})?;
			let info = session.device.allocator().get_allocation_info(&allocation);
			if !info
				.offset
				.is_multiple_of(requirement.memory_requirements.alignment)
				|| requirement.memory_requirements.size > info.size
			{
				let mut allocation = allocation;
				// SAFETY: this allocation has not been published or bound.
				unsafe {
					session.device.allocator().free_memory(&mut allocation);
				}
				return Err(Error::backend_failure(
					"Vulkan",
					"video-session memory validation",
					std::io::Error::other("allocator returned an incompatible memory range"),
				));
			}
			bindings.push(
				ash::vk::BindVideoSessionMemoryInfoKHR::default()
					.memory_bind_index(requirement.memory_bind_index)
					.memory(info.device_memory)
					.memory_offset(info.offset)
					.memory_size(requirement.memory_requirements.size),
			);
			session.allocations.push(allocation);
		}
		if !bindings.is_empty() {
			// SAFETY: every binding corresponds one-to-one with a requirement for
			// this session and its allocation remains owned by `session`.
			let result = unsafe {
				(session.loader.fp().bind_video_session_memory_khr)(
					session.loader.device(),
					session.handle,
					bindings.len() as u32,
					bindings.as_ptr(),
				)
			};
			if result != ash::vk::Result::SUCCESS {
				return Err(Error::backend_failure(
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

fn select_image_format(
	formats: Vec<ash::vk::VideoFormatPropertiesKHR<'static>>,
	usage: ash::vk::ImageUsageFlags,
	role: &str,
) -> Result<ash::vk::VideoFormatPropertiesKHR<'static>> {
	formats
		.into_iter()
		.find(|format| {
			format.image_type == ash::vk::ImageType::TYPE_2D
				&& format.image_usage_flags.contains(usage)
				&& !format
					.image_create_flags
					.contains(ash::vk::ImageCreateFlags::DISJOINT)
		})
		.ok_or_else(|| {
			Error::missing_capability(format!(
				"decode profile exposes no directly allocatable {role} image format"
			))
		})
}

fn create_decode_image(
	device: &Device,
	profile: &ash::vk::VideoProfileInfoKHR<'_>,
	properties: ash::vk::VideoFormatPropertiesKHR<'_>,
	extent: video::VideoExtent,
	array_layers: u32,
	usage: ash::vk::ImageUsageFlags,
) -> Result<DecodeImage> {
	let profiles = [*profile];
	let mut profile_list = ash::vk::VideoProfileListInfoKHR::default().profiles(&profiles);
	let image_info = ash::vk::ImageCreateInfo::default()
		.flags(
			properties.image_create_flags & !ash::vk::ImageCreateFlags::VIDEO_PROFILE_INDEPENDENT_KHR,
		)
		.image_type(properties.image_type)
		.format(properties.format)
		.extent(ash::vk::Extent3D {
			width: extent.width,
			height: extent.height,
			depth: 1,
		})
		.mip_levels(1)
		.array_layers(array_layers)
		.samples(ash::vk::SampleCountFlags::TYPE_1)
		.tiling(properties.image_tiling)
		.usage(usage)
		.sharing_mode(ash::vk::SharingMode::EXCLUSIVE)
		.initial_layout(ash::vk::ImageLayout::UNDEFINED)
		.push_next(&mut profile_list);
	let allocation_info = vk_mem::AllocationCreateInfo {
		usage: vk_mem::MemoryUsage::AutoPreferDevice,
		required_flags: ash::vk::MemoryPropertyFlags::DEVICE_LOCAL,
		..Default::default()
	};
	// SAFETY: the exact video profile and queried format properties remain live,
	// the extent/layer count are nonzero and session-bounded, and VMA retains the
	// device/allocator used to bind the returned image allocation.
	let (handle, allocation) = unsafe {
		device
			.allocator()
			.create_image(&image_info, &allocation_info)
	}
	.map_err(|source| Error::backend_failure("Vulkan", "decode-image allocation", source))?;
	let view_info = ash::vk::ImageViewCreateInfo::default()
		.image(handle)
		.view_type(if array_layers == 1 {
			ash::vk::ImageViewType::TYPE_2D
		} else {
			ash::vk::ImageViewType::TYPE_2D_ARRAY
		})
		.format(properties.format)
		.components(properties.component_mapping)
		.subresource_range(ash::vk::ImageSubresourceRange {
			aspect_mask: ash::vk::ImageAspectFlags::COLOR,
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
				device.allocator().destroy_image(handle, &mut allocation);
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
	device: &Device,
	profile: &ash::vk::VideoProfileInfoKHR<'_>,
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
	let profiles = [*profile];
	let mut profile_list = ash::vk::VideoProfileListInfoKHR::default().profiles(&profiles);
	let buffer_info = ash::vk::BufferCreateInfo::default()
		.size(range)
		.usage(ash::vk::BufferUsageFlags::VIDEO_DECODE_SRC_KHR)
		.sharing_mode(ash::vk::SharingMode::EXCLUSIVE)
		.push_next(&mut profile_list);
	let allocation_info = vk_mem::AllocationCreateInfo {
		flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
			| vk_mem::AllocationCreateFlags::DEDICATED_MEMORY,
		usage: vk_mem::MemoryUsage::AutoPreferDevice,
		required_flags: ash::vk::MemoryPropertyFlags::HOST_VISIBLE,
		preferred_flags: ash::vk::MemoryPropertyFlags::DEVICE_LOCAL,
		..Default::default()
	};
	// SAFETY: the exact decode profile remains live in the required buffer pNext
	// chain, the aligned range is nonzero, and VMA binds compatible host-visible
	// memory to the returned buffer.
	let (handle, mut allocation) = unsafe {
		device
			.allocator()
			.create_buffer(&buffer_info, &allocation_info)
	}
	.map_err(|source| Error::backend_failure("Vulkan", "decode-bitstream allocation", source))?;
	let allocator = device.allocator();
	// SAFETY: VMA selected HOST_VISIBLE memory and this allocation is uniquely
	// owned until the initialized bitstream is returned.
	let mapped = match unsafe { allocator.map_memory(&mut allocation) } {
		Ok(mapped) => mapped,
		Err(source) => {
			// SAFETY: mapping failed before publication of this uniquely owned pair.
			unsafe {
				allocator.destroy_buffer(handle, &mut allocation);
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
		memory::copy_streaming_to_ptr(mapped, data.as_ptr(), data.len());
		std::ptr::write_bytes(mapped.add(data.len()), 0, range_len - data.len());
	}
	let flush = allocator.flush_allocation(&allocation, 0, range);
	// SAFETY: this exactly balances the successful map after the final host write.
	unsafe {
		allocator.unmap_memory(&mut allocation);
	}
	if let Err(source) = flush {
		// SAFETY: upload failed before publication of this uniquely owned pair.
		unsafe {
			allocator.destroy_buffer(handle, &mut allocation);
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
	format: ash::vk::Format,
	extent: video::VideoExtent,
	array_layer: u32,
) -> Result<(usize, Vec<ash::vk::BufferImageCopy>)> {
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
	let layer = |aspect_mask| ash::vk::ImageSubresourceLayers {
		aspect_mask,
		mip_level: 0,
		base_array_layer: array_layer,
		layer_count: 1,
	};
	let region = |offset, aspect_mask, plane_width: u64, plane_height: u64| {
		Ok(
			ash::vk::BufferImageCopy::default()
				.buffer_offset(offset)
				.buffer_row_length(0)
				.buffer_image_height(0)
				.image_subresource(layer(aspect_mask))
				.image_extent(ash::vk::Extent3D {
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
		ash::vk::ImageAspectFlags::PLANE_0,
		width,
		height,
	)?];
	match format {
		ash::vk::Format::G8_B8R8_2PLANE_420_UNORM => {
			regions.push(region(
				luma_bytes,
				ash::vk::ImageAspectFlags::PLANE_1,
				chroma_width,
				chroma_height,
			)?);
		}
		ash::vk::Format::G8_B8_R8_3PLANE_420_UNORM => {
			regions.push(region(
				luma_bytes,
				ash::vk::ImageAspectFlags::PLANE_1,
				chroma_width,
				chroma_height,
			)?);
			regions.push(region(
				luma_bytes + chroma_plane_bytes,
				ash::vk::ImageAspectFlags::PLANE_2,
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
	format: ash::vk::Format,
	extent: video::VideoExtent,
	raw: Vec<u8>,
) -> Result<Vec<u8>> {
	if format == ash::vk::Format::G8_B8_R8_3PLANE_420_UNORM {
		return Ok(raw);
	}
	if format != ash::vk::Format::G8_B8R8_2PLANE_420_UNORM {
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
					.destroy_buffer(previous.handle, &mut previous.allocation);
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

	#[cfg(test)]
	pub(in crate::runtime) fn upload_first_h265_access_unit(
		&mut self,
		access_unit: &[u8],
	) -> Result<usize> {
		self.upload_h265_access_unit(access_unit)
	}

	#[cfg(test)]
	pub(in crate::runtime) fn record_first_h264_idr(
		&mut self,
		device: &Device,
		sps: &video::H264SequenceParameterSet,
		pps: &video::H264PictureParameterSet,
		slice: &video::H264SliceHeader,
	) -> Result<super::RecordedCommandBuffer> {
		if !device.same_as(&self.device) {
			return Err(Error::invalid_argument(
				"video session and command device differ",
			));
		}
		if self.decode_recorded {
			return Err(Error::failed_precondition(
				"the first-picture qualification session already recorded a decode",
			));
		}
		if self.parameters == ash::vk::VideoSessionParametersKHR::null() {
			return Err(Error::failed_precondition(
				"H.264 decode requires session parameters",
			));
		}
		if self.result_status_pool == ash::vk::QueryPool::null() {
			return Err(Error::missing_capability(
				"the selected Vulkan Video decode queue lacks result-status queries",
			));
		}
		let output = self
			.images
			.first()
			.ok_or_else(|| Error::failed_precondition("H.264 decode output image is not initialized"))?;
		let (readback_size, _) = yuv420_8bit_copy_layout(output.format, self.coded_extent, 0)?;
		self.readback = Some(super::Buffer::host_visible_storage(device, readback_size)?);
		if pps.sequence_parameter_set_id != sps.id
			|| slice.picture_parameter_set_id != pps.id
			|| !slice.is_idr
			|| slice.first_macroblock_in_slice != 0
			|| slice.field_picture
			|| !matches!(slice.slice_type, video::H264SliceType::I)
		{
			return Err(Error::missing_capability(
				"the first Vulkan H.264 decode path accepts one progressive IDR I-slice",
			));
		}
		let bitstream = self
			.bitstream
			.as_ref()
			.ok_or_else(|| Error::failed_precondition("H.264 decode requires an uploaded bitstream"))?;
		let command = device.record_video_decode(|command_buffer| {
			record_first_h264_idr(
				device.raw(),
				&self.loader,
				&self.decode_loader,
				command_buffer,
				self,
				sps,
				pps,
				slice,
				0,
				bitstream,
			)
		})?;
		self.decode_recorded = true;
		Ok(command)
	}

	#[cfg(test)]
	pub(in crate::runtime) fn record_first_h265_idr(
		&mut self,
		device: &Device,
		sps: &video::H265SequenceParameterSet,
		pps: &video::H265PictureParameterSet,
		slice: &video::H265SliceHeader,
	) -> Result<super::RecordedCommandBuffer> {
		if !device.same_as(&self.device) {
			return Err(Error::invalid_argument(
				"video session and command device differ",
			));
		}
		if self.decode_recorded {
			return Err(Error::failed_precondition(
				"the first-picture qualification session already recorded a decode",
			));
		}
		if self.parameters == ash::vk::VideoSessionParametersKHR::null() {
			return Err(Error::failed_precondition(
				"H.265 decode requires session parameters",
			));
		}
		if self.result_status_pool == ash::vk::QueryPool::null() {
			return Err(Error::missing_capability(
				"the selected Vulkan Video decode queue lacks result-status queries",
			));
		}
		let output = self
			.images
			.first()
			.ok_or_else(|| Error::failed_precondition("H.265 decode output image is not initialized"))?;
		let (readback_size, _) = yuv420_8bit_copy_layout(output.format, self.coded_extent, 0)?;
		self.readback = Some(super::Buffer::host_visible_storage(device, readback_size)?);
		if pps.sequence_parameter_set_id != sps.id
			|| slice.picture_parameter_set_id != pps.id
			|| slice.video_parameter_set_id != sps.video_parameter_set_id
			|| !slice.is_idr
			|| !slice.first_slice_segment_in_picture
			|| slice.slice_segment_address != 0
			|| !matches!(slice.slice_type, video::H265SliceType::I)
		{
			return Err(Error::missing_capability(
				"the first Vulkan H.265 decode path accepts one IDR I-slice segment",
			));
		}
		let bitstream = self
			.bitstream
			.as_ref()
			.ok_or_else(|| Error::failed_precondition("H.265 decode requires an uploaded bitstream"))?;
		let command = device.record_video_decode(|command_buffer| {
			record_first_h265_idr(
				device.raw(),
				&self.loader,
				&self.decode_loader,
				command_buffer,
				self,
				sps,
				pps,
				slice,
				0,
				bitstream,
			)
		})?;
		self.decode_recorded = true;
		Ok(command)
	}

	pub(in crate::runtime) fn record_h264_picture_for_readback(
		&mut self,
		device: &Device,
		sps: &video::H264SequenceParameterSet,
		pps: &video::H264PictureParameterSet,
		slice: &video::H264SliceHeader,
	) -> Result<super::RecordedCommandBuffer> {
		let unavailable = self.native_unavailable_slots()?;
		self
			.record_h264_picture_impl(device, sps, pps, slice, true, &unavailable)
			.map(|(command, _)| command)
	}

	pub(in crate::runtime) fn record_h264_picture_native(
		&mut self,
		device: &Device,
		sps: &video::H264SequenceParameterSet,
		pps: &video::H264PictureParameterSet,
		slice: &video::H264SliceHeader,
	) -> Result<(super::RecordedCommandBuffer, u32)> {
		let unavailable = self.native_unavailable_slots()?;
		self.record_h264_picture_impl(device, sps, pps, slice, false, &unavailable)
	}

	fn record_h264_picture_impl(
		&mut self,
		device: &Device,
		sps: &video::H264SequenceParameterSet,
		pps: &video::H264PictureParameterSet,
		slice: &video::H264SliceHeader,
		release_for_readback: bool,
		unavailable: &[bool],
	) -> Result<(super::RecordedCommandBuffer, u32)> {
		if !device.same_as(&self.device) {
			return Err(Error::invalid_argument(
				"video session and command device differ",
			));
		}
		if self.parameters == ash::vk::VideoSessionParametersKHR::null() {
			return Err(Error::failed_precondition(
				"H.264 decode requires session parameters",
			));
		}
		if self.result_status_pool == ash::vk::QueryPool::null() {
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
				&self.loader,
				&self.decode_loader,
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
		device: &Device,
		sps: &video::H265SequenceParameterSet,
		pps: &video::H265PictureParameterSet,
		slice: &video::H265SliceHeader,
	) -> Result<super::RecordedCommandBuffer> {
		let unavailable = self.native_unavailable_slots()?;
		self
			.record_h265_picture_impl(device, sps, pps, slice, true, &unavailable)
			.map(|(command, _)| command)
	}

	pub(in crate::runtime) fn record_h265_picture_native(
		&mut self,
		device: &Device,
		sps: &video::H265SequenceParameterSet,
		pps: &video::H265PictureParameterSet,
		slice: &video::H265SliceHeader,
	) -> Result<(super::RecordedCommandBuffer, u32)> {
		let unavailable = self.native_unavailable_slots()?;
		self.record_h265_picture_impl(device, sps, pps, slice, false, &unavailable)
	}

	fn record_h265_picture_impl(
		&mut self,
		device: &Device,
		sps: &video::H265SequenceParameterSet,
		pps: &video::H265PictureParameterSet,
		slice: &video::H265SliceHeader,
		release_for_readback: bool,
		unavailable: &[bool],
	) -> Result<(super::RecordedCommandBuffer, u32)> {
		if !device.same_as(&self.device) {
			return Err(Error::invalid_argument(
				"video session and command device differ",
			));
		}
		if self.parameters == ash::vk::VideoSessionParametersKHR::null() {
			return Err(Error::failed_precondition(
				"H.265 decode requires session parameters",
			));
		}
		if self.result_status_pool == ash::vk::QueryPool::null() {
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
				&self.loader,
				&self.decode_loader,
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
		device: &Device,
		sequence: &video::Av1SequenceHeader,
		frame: &video::Av1FrameHeader,
		frame_header_offset: u32,
		tiles: &video::Av1TileGroup,
	) -> Result<(super::RecordedCommandBuffer, u32)> {
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
		device: &Device,
		sequence: &video::Av1SequenceHeader,
		frame: &video::Av1FrameHeader,
		frame_header_offset: u32,
		tiles: &video::Av1TileGroup,
	) -> Result<(super::RecordedCommandBuffer, u32)> {
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
		device: &Device,
		sequence: &video::Av1SequenceHeader,
		frame: &video::Av1FrameHeader,
		frame_header_offset: u32,
		tiles: &video::Av1TileGroup,
		release_for_readback: bool,
		unavailable: &[bool],
	) -> Result<(super::RecordedCommandBuffer, u32)> {
		if !device.same_as(&self.device) {
			return Err(Error::invalid_argument(
				"video session and command device differ",
			));
		}
		if self.parameters == ash::vk::VideoSessionParametersKHR::null() {
			return Err(Error::failed_precondition(
				"AV1 decode requires session parameters",
			));
		}
		if self.result_status_pool == ash::vk::QueryPool::null() {
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
				&self.loader,
				&self.decode_loader,
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
		device: &Device,
		picture: &video::Vp9Picture,
	) -> Result<(super::RecordedCommandBuffer, u32)> {
		let unavailable = self.native_unavailable_slots()?;
		self.record_vp9_picture_impl(device, picture, true, &unavailable)
	}

	pub(in crate::runtime) fn record_vp9_picture_native(
		&mut self,
		device: &Device,
		picture: &video::Vp9Picture,
	) -> Result<(super::RecordedCommandBuffer, u32)> {
		let unavailable = self.native_unavailable_slots()?;
		self.record_vp9_picture_impl(device, picture, false, &unavailable)
	}

	fn record_vp9_picture_impl(
		&mut self,
		device: &Device,
		picture: &video::Vp9Picture,
		release_for_readback: bool,
		unavailable: &[bool],
	) -> Result<(super::RecordedCommandBuffer, u32)> {
		if !device.same_as(&self.device) {
			return Err(Error::invalid_argument(
				"video session and command device differ",
			));
		}
		if self.result_status_pool == ash::vk::QueryPool::null() {
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
				&self.loader,
				&self.decode_loader,
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
		if self.parameters != ash::vk::VideoSessionParametersKHR::null() {
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
		let add = ash::vk::VideoDecodeH264SessionParametersAddInfoKHR::default()
			.std_sp_ss(&std_sp_ss)
			.std_pp_ss(&std_pp_ss);
		let mut codec = ash::vk::VideoDecodeH264SessionParametersCreateInfoKHR::default()
			.max_std_sps_count(32)
			.max_std_pps_count(256)
			.parameters_add_info(&add);
		let create_info = ash::vk::VideoSessionParametersCreateInfoKHR::default()
			.video_session(self.handle)
			.push_next(&mut codec);
		let mut parameters = ash::vk::VideoSessionParametersKHR::null();
		// SAFETY: the standard-video records and codec add chain remain live for
		// this call, match the session profile, and Vulkan copies their contents.
		let result = unsafe {
			(self.loader.fp().create_video_session_parameters_khr)(
				self.loader.device(),
				&create_info,
				std::ptr::null(),
				&mut parameters,
			)
		};
		if result != ash::vk::Result::SUCCESS {
			return Err(Error::backend_failure(
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
		if self.parameters != ash::vk::VideoSessionParametersKHR::null() {
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
		let add = ash::vk::VideoDecodeH265SessionParametersAddInfoKHR::default()
			.std_vp_ss(&std_vp_ss)
			.std_sp_ss(&std_sp_ss)
			.std_pp_ss(&std_pp_ss);
		let mut codec = ash::vk::VideoDecodeH265SessionParametersCreateInfoKHR::default()
			.max_std_vps_count(16)
			.max_std_sps_count(16)
			.max_std_pps_count(64)
			.parameters_add_info(&add);
		let create_info = ash::vk::VideoSessionParametersCreateInfoKHR::default()
			.video_session(self.handle)
			.push_next(&mut codec);
		let mut parameters = ash::vk::VideoSessionParametersKHR::null();
		// SAFETY: every referenced StdVideo record and backing array remains live
		// for this call, matches the H.265 session profile, and Vulkan copies it.
		let result = unsafe {
			(self.loader.fp().create_video_session_parameters_khr)(
				self.loader.device(),
				&create_info,
				std::ptr::null(),
				&mut parameters,
			)
		};
		if result != ash::vk::Result::SUCCESS {
			return Err(Error::backend_failure(
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
		if self.parameters != ash::vk::VideoSessionParametersKHR::null() {
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
		let mut codec = ash::vk::VideoDecodeAV1SessionParametersCreateInfoKHR::default()
			.std_sequence_header(&std_sequence);
		let create_info = ash::vk::VideoSessionParametersCreateInfoKHR::default()
			.video_session(self.handle)
			.push_next(&mut codec);
		let mut parameters = ash::vk::VideoSessionParametersKHR::null();
		// SAFETY: all standard-video records and optional timing/color backing
		// remain live for this call, match the AV1 session profile, and Vulkan
		// copies the sequence header into the parameter object.
		let result = unsafe {
			(self.loader.fp().create_video_session_parameters_khr)(
				self.loader.device(),
				&create_info,
				std::ptr::null(),
				&mut parameters,
			)
		};
		if result != ash::vk::Result::SUCCESS {
			return Err(Error::backend_failure(
				"Vulkan",
				"AV1 video-session parameter creation",
				result,
			));
		}
		self.parameters = parameters;
		Ok(())
	}

	fn memory_requirements(
		&self,
	) -> Result<Vec<ash::vk::VideoSessionMemoryRequirementsKHR<'static>>> {
		let mut count = 0_u32;
		// SAFETY: the loader and session belong to the retained live device, and
		// a null output pointer requests only the element count.
		let result = unsafe {
			(self.loader.fp().get_video_session_memory_requirements_khr)(
				self.loader.device(),
				self.handle,
				&mut count,
				std::ptr::null_mut(),
			)
		};
		if result != ash::vk::Result::SUCCESS {
			return Err(Error::backend_failure(
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
		let mut requirements =
			vec![ash::vk::VideoSessionMemoryRequirementsKHR::default(); count as usize];
		if count == 0 {
			return Ok(requirements);
		}
		// SAFETY: the vector contains `count` initialized structures with valid
		// sType values, and Vulkan writes at most the supplied count.
		let result = unsafe {
			(self.loader.fp().get_video_session_memory_requirements_khr)(
				self.loader.device(),
				self.handle,
				&mut count,
				requirements.as_mut_ptr(),
			)
		};
		if result != ash::vk::Result::SUCCESS {
			return Err(Error::backend_failure(
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

	#[cfg(test)]
	pub(in crate::runtime) fn memory_binding_count(&self) -> usize {
		self.allocations.len()
	}

	#[cfg(test)]
	pub(in crate::runtime) fn parameters_ready(&self) -> bool {
		self.parameters != ash::vk::VideoSessionParametersKHR::null()
	}

	#[cfg(test)]
	pub(in crate::runtime) fn image_count(&self) -> usize {
		self.images.len()
	}

	#[cfg(test)]
	pub(in crate::runtime) fn bitstream_upload(&self) -> Option<(usize, u64)> {
		self
			.bitstream
			.as_ref()
			.map(|bitstream| (bitstream.payload_len, bitstream.range))
	}

	#[cfg(test)]
	pub(in crate::runtime) fn first_decode_recorded(&self) -> bool {
		self.decode_recorded
	}

	pub(in crate::runtime) fn verify_decode_result(&self) -> Result<()> {
		if !self.decode_recorded {
			return Err(Error::failed_precondition(
				"no picture decode was recorded for this session",
			));
		}
		if self.result_status_pool == ash::vk::QueryPool::null() {
			return Err(Error::missing_capability(
				"the selected Vulkan Video decode queue lacks result-status queries",
			));
		}
		let mut raw_status = [ash::vk::QueryResultStatusKHR::NOT_READY.as_raw()];
		// SAFETY: the pool contains one ended result-status query. The caller waits
		// for the decode event before reading this non-waiting one-i32 result.
		unsafe {
			self.device.raw().get_query_pool_results(
				self.result_status_pool,
				0,
				&mut raw_status,
				ash::vk::QueryResultFlags::WITH_STATUS_KHR,
			)
		}
		.map_err(|source| {
			Error::backend_failure("Vulkan", "video decode result-status query", source)
		})?;
		let status = ash::vk::QueryResultStatusKHR::from_raw(raw_status[0]);
		if status != ash::vk::QueryResultStatusKHR::COMPLETE {
			return Err(Error::data_loss(format!(
				"Vulkan video decode completed with operation status {status:?}"
			)));
		}
		Ok(())
	}

	#[cfg(test)]
	pub(in crate::runtime) fn verify_first_decode_result(&self) -> Result<()> {
		self.verify_decode_result()
	}

	pub(in crate::runtime) fn record_decode_readback(
		&mut self,
		device: &Device,
	) -> Result<super::RecordedCommandBuffer> {
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
			self.readback = Some(super::Buffer::host_visible_storage(device, readback_size)?);
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
			ash::vk::ImageLayout::VIDEO_DECODE_DST_KHR
		} else {
			ash::vk::ImageLayout::VIDEO_DECODE_DPB_KHR
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
		device: &Device,
		frame: &NativeDecodedFrame,
	) -> Result<(
		super::RecordedCommandBuffer,
		super::RecordedCommandBuffer,
		Option<super::RecordedCommandBuffer>,
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
			self.readback = Some(super::Buffer::host_visible_storage(device, readback_size)?);
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
			ash::vk::ImageLayout::VIDEO_DECODE_DST_KHR
		} else {
			ash::vk::ImageLayout::VIDEO_DECODE_DPB_KHR
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

	#[cfg(test)]
	pub(in crate::runtime) fn record_first_decode_readback(
		&mut self,
		device: &Device,
	) -> Result<super::RecordedCommandBuffer> {
		if !device.same_as(&self.device) {
			return Err(Error::invalid_argument(
				"video session and readback device differ",
			));
		}
		if !self.decode_recorded || self.readback_recorded {
			return Err(Error::failed_precondition(
				"first-picture readback requires one completed, unread decode",
			));
		}
		let (output, _, _, _) = self.picture_images(0)?;
		let readback = self.readback.as_ref().ok_or_else(|| {
			Error::failed_precondition("video decode readback buffer is not initialized")
		})?;
		let (_, regions) = yuv420_8bit_copy_layout(output.format, self.coded_extent, 0)?;
		let decode_family = self
			.device
			.physical()
			.video
			.decode_queue_family
			.ok_or_else(|| Error::missing_capability("no Vulkan Video decode queue is enabled"))?;
		let compute_family = self.device.physical().compute_queue_family;
		let decoded_layout = if self.image_set == DecodeImageSet::DistinctLayered {
			ash::vk::ImageLayout::VIDEO_DECODE_DST_KHR
		} else {
			ash::vk::ImageLayout::VIDEO_DECODE_DPB_KHR
		};
		let command = device.record_compute_commands(|command_buffer| {
			record_decode_readback(
				device.raw(),
				command_buffer,
				output,
				0,
				readback,
				&regions,
				(decode_family, compute_family),
				decoded_layout,
			)
		})?;
		self.readback_recorded = true;
		Ok(command)
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

	#[cfg(test)]
	pub(in crate::runtime) fn read_first_decode_yuv420(&self) -> Result<Vec<u8>> {
		self.read_decode_yuv420()
	}
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn record_first_h264_idr(
	device: &ash::Device,
	loader: &ash::khr::video_queue::Device,
	decode_loader: &ash::khr::video_decode_queue::Device,
	command_buffer: ash::vk::CommandBuffer,
	session: &DecodeSession,
	sps: &video::H264SequenceParameterSet,
	pps: &video::H264PictureParameterSet,
	slice: &video::H264SliceHeader,
	slice_offset: u32,
	bitstream: &DecodeBitstream,
) -> Result<()> {
	let picture_order_count = i32::try_from(slice.picture_order_count_lsb.ok_or_else(|| {
		Error::missing_capability("first H.264 decode requires an explicit picture-order count")
	})?)
	.map_err(|_| Error::data_loss("H.264 picture-order count exceeds i32"))?;
	let picture_flags = ash::vk::native::StdVideoDecodeH264PictureInfoFlags {
		_bitfield_align_1: [],
		_bitfield_1: ash::vk::native::StdVideoDecodeH264PictureInfoFlags::new_bitfield_1(
			0,
			1,
			1,
			0,
			u32::from(slice.is_reference),
			0,
		),
		__bindgen_padding_0: [0; 3],
	};
	let std_picture = ash::vk::native::StdVideoDecodeH264PictureInfo {
		flags: picture_flags,
		seq_parameter_set_id: u8::try_from(sps.id)
			.map_err(|_| Error::data_loss("H.264 SPS id exceeds StdVideo picture storage"))?,
		pic_parameter_set_id: u8::try_from(pps.id)
			.map_err(|_| Error::data_loss("H.264 PPS id exceeds StdVideo picture storage"))?,
		reserved1: 0,
		reserved2: 0,
		frame_num: u16::try_from(slice.frame_number)
			.map_err(|_| Error::data_loss("H.264 frame number exceeds StdVideo storage"))?,
		idr_pic_id: u16::try_from(
			slice
				.idr_picture_id
				.ok_or_else(|| Error::data_loss("H.264 IDR slice omitted idr_pic_id"))?,
		)
		.map_err(|_| Error::data_loss("H.264 IDR picture id exceeds StdVideo storage"))?,
		PicOrderCnt: [picture_order_count; 2],
	};
	let mut h264_picture = ash::vk::VideoDecodeH264PictureInfoKHR::default()
		.std_picture_info(&std_picture)
		.slice_offsets(std::slice::from_ref(&slice_offset));

	let reference_flags = ash::vk::native::StdVideoDecodeH264ReferenceInfoFlags {
		_bitfield_align_1: [],
		_bitfield_1: ash::vk::native::StdVideoDecodeH264ReferenceInfoFlags::new_bitfield_1(
			0,
			0,
			u32::from(slice.long_term_reference),
			0,
		),
		__bindgen_padding_0: [0; 3],
	};
	let std_reference = ash::vk::native::StdVideoDecodeH264ReferenceInfo {
		flags: reference_flags,
		FrameNum: std_picture.frame_num,
		reserved: 0,
		PicOrderCnt: std_picture.PicOrderCnt,
	};

	let (output, output_layer, dpb, dpb_layer) = session.picture_images(0)?;
	let extent = ash::vk::Extent2D {
		width: session.coded_extent.width,
		height: session.coded_extent.height,
	};
	let dpb_resource = ash::vk::VideoPictureResourceInfoKHR::default()
		.coded_extent(extent)
		.base_array_layer(dpb_layer)
		.image_view_binding(dpb.view);
	let output_resource = ash::vk::VideoPictureResourceInfoKHR::default()
		.coded_extent(extent)
		.base_array_layer(output_layer)
		.image_view_binding(output.view);

	let mut begin_h264_slot =
		ash::vk::VideoDecodeH264DpbSlotInfoKHR::default().std_reference_info(&std_reference);
	let begin_slot = ash::vk::VideoReferenceSlotInfoKHR::default()
		.slot_index(-1)
		.picture_resource(&dpb_resource)
		.push_next(&mut begin_h264_slot);
	let begin_info = ash::vk::VideoBeginCodingInfoKHR::default()
		.video_session(session.handle)
		.video_session_parameters(session.parameters)
		.reference_slots(std::slice::from_ref(&begin_slot));

	let mut setup_h264_slot =
		ash::vk::VideoDecodeH264DpbSlotInfoKHR::default().std_reference_info(&std_reference);
	let setup_slot = ash::vk::VideoReferenceSlotInfoKHR::default()
		.slot_index(0)
		.picture_resource(&dpb_resource)
		.push_next(&mut setup_h264_slot);
	let decode_info = ash::vk::VideoDecodeInfoKHR::default()
		.src_buffer(bitstream.handle)
		.src_buffer_offset(0)
		.src_buffer_range(bitstream.range)
		.dst_picture_resource(output_resource)
		.setup_reference_slot(&setup_slot)
		.push_next(&mut h264_picture);

	// SAFETY: the fresh status query is reset outside the coding scope. All
	// begin-coding records and their codec-specific chains remain live for the
	// following call and name the session, parameters, and unactivated DPB slot zero.
	unsafe {
		device.cmd_reset_query_pool(command_buffer, session.result_status_pool, 0, 1);
		(loader.fp().cmd_begin_video_coding_khr)(command_buffer, &begin_info);
	}
	let control =
		ash::vk::VideoCodingControlInfoKHR::default().flags(ash::vk::VideoCodingControlFlagsKHR::RESET);
	// SAFETY: the command buffer is inside its first coding scope; RESET changes
	// the newly created session from uninitialized to initialized state.
	unsafe {
		(loader.fp().cmd_control_video_coding_khr)(command_buffer, &control);
	}

	let mut image_barriers = Vec::with_capacity(2);
	image_barriers.push(decode_image_barrier(
		dpb,
		ash::vk::ImageLayout::VIDEO_DECODE_DPB_KHR,
		ash::vk::AccessFlags2::VIDEO_DECODE_READ_KHR | ash::vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR,
	));
	if output.handle != dpb.handle {
		image_barriers.push(decode_image_barrier(
			output,
			ash::vk::ImageLayout::VIDEO_DECODE_DST_KHR,
			ash::vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR,
		));
	}
	let bitstream_barriers = [ash::vk::BufferMemoryBarrier2::default()
		.src_stage_mask(ash::vk::PipelineStageFlags2::HOST)
		.src_access_mask(ash::vk::AccessFlags2::HOST_WRITE)
		.dst_stage_mask(ash::vk::PipelineStageFlags2::VIDEO_DECODE_KHR)
		.dst_access_mask(ash::vk::AccessFlags2::VIDEO_DECODE_READ_KHR)
		.src_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
		.dst_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
		.buffer(bitstream.handle)
		.offset(0)
		.size(bitstream.range)];
	let dependency = ash::vk::DependencyInfo::default()
		.buffer_memory_barriers(&bitstream_barriers)
		.image_memory_barriers(&image_barriers);
	// SAFETY: the barriers name the complete uploaded buffer range and the complete
	// initially-undefined image subresources used by the following decode command.
	unsafe {
		device.cmd_pipeline_barrier2(command_buffer, &dependency);
	}

	// SAFETY: the initialized session is bound, all standard-video data and slice
	// offsets remain live, resources have the required usage/layouts, and the first
	// IDR has no active reference pictures beyond its reconstructed setup slot. The
	// status query is begun with this session bound and encloses exactly one decode.
	unsafe {
		device.cmd_begin_query(
			command_buffer,
			session.result_status_pool,
			0,
			ash::vk::QueryControlFlags::empty(),
		);
		(decode_loader.fp().cmd_decode_video_khr)(command_buffer, &decode_info);
		device.cmd_end_query(command_buffer, session.result_status_pool, 0);
		(loader.fp().cmd_end_video_coding_khr)(
			command_buffer,
			&ash::vk::VideoEndCodingInfoKHR::default(),
		);
	}
	let decode_family = session
		.device
		.physical()
		.video
		.decode_queue_family
		.ok_or_else(|| Error::missing_capability("no Vulkan Video decode queue is enabled"))?;
	let compute_family = session.device.physical().compute_queue_family;
	let old_layout = if output.handle == dpb.handle {
		ash::vk::ImageLayout::VIDEO_DECODE_DPB_KHR
	} else {
		ash::vk::ImageLayout::VIDEO_DECODE_DST_KHR
	};
	let same_family = decode_family == compute_family;
	let release = ash::vk::ImageMemoryBarrier2::default()
		.src_stage_mask(ash::vk::PipelineStageFlags2::VIDEO_DECODE_KHR)
		.src_access_mask(
			ash::vk::AccessFlags2::VIDEO_DECODE_READ_KHR | ash::vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR,
		)
		.dst_stage_mask(if same_family {
			ash::vk::PipelineStageFlags2::TRANSFER
		} else {
			ash::vk::PipelineStageFlags2::NONE
		})
		.dst_access_mask(if same_family {
			ash::vk::AccessFlags2::TRANSFER_READ
		} else {
			ash::vk::AccessFlags2::NONE
		})
		.old_layout(old_layout)
		.new_layout(ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
		.src_queue_family_index(if same_family {
			ash::vk::QUEUE_FAMILY_IGNORED
		} else {
			decode_family
		})
		.dst_queue_family_index(if same_family {
			ash::vk::QUEUE_FAMILY_IGNORED
		} else {
			compute_family
		})
		.image(output.handle)
		.subresource_range(ash::vk::ImageSubresourceRange {
			aspect_mask: ash::vk::ImageAspectFlags::COLOR,
			base_mip_level: 0,
			level_count: 1,
			base_array_layer: 0,
			layer_count: 1,
		});
	let release_dependency =
		ash::vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&release));
	// SAFETY: the decode write has ended. This releases layer zero to the compute
	// family, or performs the complete same-family transition, before timeline signal.
	unsafe {
		device.cmd_pipeline_barrier2(command_buffer, &release_dependency);
	}
	Ok(())
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn record_first_h265_idr(
	device: &ash::Device,
	loader: &ash::khr::video_queue::Device,
	decode_loader: &ash::khr::video_decode_queue::Device,
	command_buffer: ash::vk::CommandBuffer,
	session: &DecodeSession,
	sps: &video::H265SequenceParameterSet,
	pps: &video::H265PictureParameterSet,
	slice: &video::H265SliceHeader,
	slice_offset: u32,
	bitstream: &DecodeBitstream,
) -> Result<()> {
	let std_picture = ash::vk::native::StdVideoDecodeH265PictureInfo {
		flags: ash::vk::native::StdVideoDecodeH265PictureInfoFlags {
			_bitfield_align_1: [],
			_bitfield_1: ash::vk::native::StdVideoDecodeH265PictureInfoFlags::new_bitfield_1(
				u32::from(slice.is_irap),
				u32::from(slice.is_idr),
				u32::from(slice.is_reference),
				u32::from(slice.short_term_reference_picture_set_sps),
			),
			__bindgen_padding_0: [0; 3],
		},
		sps_video_parameter_set_id: h265_u8(sps.video_parameter_set_id, "picture VPS id")?,
		pps_seq_parameter_set_id: h265_u8(sps.id, "picture SPS id")?,
		pps_pic_parameter_set_id: h265_u8(pps.id, "picture PPS id")?,
		NumDeltaPocsOfRefRpsIdx: 0,
		PicOrderCntVal: 0,
		NumBitsForSTRefPicSetInSlice: slice.short_term_reference_picture_set_bits,
		reserved: 0,
		RefPicSetStCurrBefore: [0xff; 8],
		RefPicSetStCurrAfter: [0xff; 8],
		RefPicSetLtCurr: [0xff; 8],
	};
	let mut h265_picture = ash::vk::VideoDecodeH265PictureInfoKHR::default()
		.std_picture_info(&std_picture)
		.slice_segment_offsets(std::slice::from_ref(&slice_offset));
	let std_reference = ash::vk::native::StdVideoDecodeH265ReferenceInfo {
		flags: ash::vk::native::StdVideoDecodeH265ReferenceInfoFlags {
			_bitfield_align_1: [],
			_bitfield_1: ash::vk::native::StdVideoDecodeH265ReferenceInfoFlags::new_bitfield_1(
				0,
				u32::from(!slice.is_reference),
			),
			__bindgen_padding_0: [0; 3],
		},
		PicOrderCntVal: 0,
	};

	let (output, output_layer, dpb, dpb_layer) = session.picture_images(0)?;
	let extent = ash::vk::Extent2D {
		width: session.coded_extent.width,
		height: session.coded_extent.height,
	};
	let dpb_resource = ash::vk::VideoPictureResourceInfoKHR::default()
		.coded_extent(extent)
		.base_array_layer(dpb_layer)
		.image_view_binding(dpb.view);
	let output_resource = ash::vk::VideoPictureResourceInfoKHR::default()
		.coded_extent(extent)
		.base_array_layer(output_layer)
		.image_view_binding(output.view);

	let mut begin_h265_slot =
		ash::vk::VideoDecodeH265DpbSlotInfoKHR::default().std_reference_info(&std_reference);
	let begin_slot = ash::vk::VideoReferenceSlotInfoKHR::default()
		.slot_index(-1)
		.picture_resource(&dpb_resource)
		.push_next(&mut begin_h265_slot);
	let begin_info = ash::vk::VideoBeginCodingInfoKHR::default()
		.video_session(session.handle)
		.video_session_parameters(session.parameters)
		.reference_slots(std::slice::from_ref(&begin_slot));

	let mut setup_h265_slot =
		ash::vk::VideoDecodeH265DpbSlotInfoKHR::default().std_reference_info(&std_reference);
	let setup_slot = ash::vk::VideoReferenceSlotInfoKHR::default()
		.slot_index(0)
		.picture_resource(&dpb_resource)
		.push_next(&mut setup_h265_slot);
	let decode_info = ash::vk::VideoDecodeInfoKHR::default()
		.src_buffer(bitstream.handle)
		.src_buffer_offset(0)
		.src_buffer_range(bitstream.range)
		.dst_picture_resource(output_resource)
		.setup_reference_slot(&setup_slot)
		.push_next(&mut h265_picture);

	// SAFETY: the fresh query is reset before the coding scope. Every referenced
	// H.265 record and resource remains live through command recording.
	unsafe {
		device.cmd_reset_query_pool(command_buffer, session.result_status_pool, 0, 1);
		(loader.fp().cmd_begin_video_coding_khr)(command_buffer, &begin_info);
	}
	let control =
		ash::vk::VideoCodingControlInfoKHR::default().flags(ash::vk::VideoCodingControlFlagsKHR::RESET);
	// SAFETY: this is the first coding scope for the newly created session.
	unsafe {
		(loader.fp().cmd_control_video_coding_khr)(command_buffer, &control);
	}

	let mut image_barriers = Vec::with_capacity(2);
	image_barriers.push(decode_image_barrier(
		dpb,
		ash::vk::ImageLayout::VIDEO_DECODE_DPB_KHR,
		ash::vk::AccessFlags2::VIDEO_DECODE_READ_KHR | ash::vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR,
	));
	if output.handle != dpb.handle {
		image_barriers.push(decode_image_barrier(
			output,
			ash::vk::ImageLayout::VIDEO_DECODE_DST_KHR,
			ash::vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR,
		));
	}
	let bitstream_barriers = [ash::vk::BufferMemoryBarrier2::default()
		.src_stage_mask(ash::vk::PipelineStageFlags2::HOST)
		.src_access_mask(ash::vk::AccessFlags2::HOST_WRITE)
		.dst_stage_mask(ash::vk::PipelineStageFlags2::VIDEO_DECODE_KHR)
		.dst_access_mask(ash::vk::AccessFlags2::VIDEO_DECODE_READ_KHR)
		.src_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
		.dst_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
		.buffer(bitstream.handle)
		.offset(0)
		.size(bitstream.range)];
	let dependency = ash::vk::DependencyInfo::default()
		.buffer_memory_barriers(&bitstream_barriers)
		.image_memory_barriers(&image_barriers);
	// SAFETY: these barriers cover the uploaded buffer and complete fresh image
	// subresources consumed and produced by the following H.265 decode.
	unsafe {
		device.cmd_pipeline_barrier2(command_buffer, &dependency);
	}

	// SAFETY: the reset session, parameter object, no-reference IDR picture,
	// bitstream range, slice offset, setup slot, and images all remain valid.
	unsafe {
		device.cmd_begin_query(
			command_buffer,
			session.result_status_pool,
			0,
			ash::vk::QueryControlFlags::empty(),
		);
		(decode_loader.fp().cmd_decode_video_khr)(command_buffer, &decode_info);
		device.cmd_end_query(command_buffer, session.result_status_pool, 0);
		(loader.fp().cmd_end_video_coding_khr)(
			command_buffer,
			&ash::vk::VideoEndCodingInfoKHR::default(),
		);
	}
	let decode_family = session
		.device
		.physical()
		.video
		.decode_queue_family
		.ok_or_else(|| Error::missing_capability("no Vulkan Video decode queue is enabled"))?;
	let compute_family = session.device.physical().compute_queue_family;
	let old_layout = if output.handle == dpb.handle {
		ash::vk::ImageLayout::VIDEO_DECODE_DPB_KHR
	} else {
		ash::vk::ImageLayout::VIDEO_DECODE_DST_KHR
	};
	let same_family = decode_family == compute_family;
	let release = ash::vk::ImageMemoryBarrier2::default()
		.src_stage_mask(ash::vk::PipelineStageFlags2::VIDEO_DECODE_KHR)
		.src_access_mask(
			ash::vk::AccessFlags2::VIDEO_DECODE_READ_KHR | ash::vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR,
		)
		.dst_stage_mask(if same_family {
			ash::vk::PipelineStageFlags2::TRANSFER
		} else {
			ash::vk::PipelineStageFlags2::NONE
		})
		.dst_access_mask(if same_family {
			ash::vk::AccessFlags2::TRANSFER_READ
		} else {
			ash::vk::AccessFlags2::NONE
		})
		.old_layout(old_layout)
		.new_layout(ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
		.src_queue_family_index(if same_family {
			ash::vk::QUEUE_FAMILY_IGNORED
		} else {
			decode_family
		})
		.dst_queue_family_index(if same_family {
			ash::vk::QUEUE_FAMILY_IGNORED
		} else {
			compute_family
		})
		.image(output.handle)
		.subresource_range(ash::vk::ImageSubresourceRange {
			aspect_mask: ash::vk::ImageAspectFlags::COLOR,
			base_mip_level: 0,
			level_count: 1,
			base_array_layer: 0,
			layer_count: 1,
		});
	let release_dependency =
		ash::vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&release));
	// SAFETY: the H.265 decode write has ended and this publishes or releases
	// output layer zero before the queue's timeline signal.
	unsafe {
		device.cmd_pipeline_barrier2(command_buffer, &release_dependency);
	}
	Ok(())
}

fn h264_reference_info(
	frame_number: u32,
	picture_order_count: i32,
	long_term: bool,
) -> Result<ash::vk::native::StdVideoDecodeH264ReferenceInfo> {
	Ok(ash::vk::native::StdVideoDecodeH264ReferenceInfo {
		flags: ash::vk::native::StdVideoDecodeH264ReferenceInfoFlags {
			_bitfield_align_1: [],
			_bitfield_1: ash::vk::native::StdVideoDecodeH264ReferenceInfoFlags::new_bitfield_1(
				0,
				0,
				u32::from(long_term),
				0,
			),
			__bindgen_padding_0: [0; 3],
		},
		FrameNum: u16::try_from(frame_number)
			.map_err(|_| Error::data_loss("H.264 reference frame number exceeds u16"))?,
		reserved: 0,
		PicOrderCnt: [picture_order_count; 2],
	})
}

#[allow(clippy::too_many_arguments)]
fn record_h264_picture(
	device: &ash::Device,
	loader: &ash::khr::video_queue::Device,
	decode_loader: &ash::khr::video_decode_queue::Device,
	command_buffer: ash::vk::CommandBuffer,
	session: &DecodeSession,
	sps: &video::H264SequenceParameterSet,
	pps: &video::H264PictureParameterSet,
	slice: &video::H264SliceHeader,
	slice_offset: u32,
	bitstream: &DecodeBitstream,
	plan: &video::H264PicturePlan,
	initialize_images: bool,
	pending_acquire_slot: Option<u32>,
	release_for_readback: bool,
) -> Result<()> {
	let picture_flags = ash::vk::native::StdVideoDecodeH264PictureInfoFlags {
		_bitfield_align_1: [],
		_bitfield_1: ash::vk::native::StdVideoDecodeH264PictureInfoFlags::new_bitfield_1(
			u32::from(slice.field_picture),
			u32::from(matches!(
				slice.slice_type,
				video::H264SliceType::I | video::H264SliceType::Si
			)),
			u32::from(slice.is_idr),
			u32::from(slice.bottom_field),
			u32::from(slice.is_reference),
			0,
		),
		__bindgen_padding_0: [0; 3],
	};
	let std_picture = ash::vk::native::StdVideoDecodeH264PictureInfo {
		flags: picture_flags,
		seq_parameter_set_id: u8::try_from(sps.id)
			.map_err(|_| Error::data_loss("H.264 SPS id exceeds StdVideo picture storage"))?,
		pic_parameter_set_id: u8::try_from(pps.id)
			.map_err(|_| Error::data_loss("H.264 PPS id exceeds StdVideo picture storage"))?,
		reserved1: 0,
		reserved2: 0,
		frame_num: u16::try_from(plan.frame_number)
			.map_err(|_| Error::data_loss("H.264 frame number exceeds StdVideo storage"))?,
		idr_pic_id: u16::try_from(slice.idr_picture_id.unwrap_or(0))
			.map_err(|_| Error::data_loss("H.264 IDR picture id exceeds StdVideo storage"))?,
		PicOrderCnt: [plan.picture_order_count; 2],
	};
	let mut h264_picture = ash::vk::VideoDecodeH264PictureInfoKHR::default()
		.std_picture_info(&std_picture)
		.slice_offsets(std::slice::from_ref(&slice_offset));

	let setup_std_reference =
		h264_reference_info(plan.frame_number, plan.picture_order_count, plan.long_term)?;
	let mut setup_h264_slot =
		ash::vk::VideoDecodeH264DpbSlotInfoKHR::default().std_reference_info(&setup_std_reference);
	let (output, output_layer, dpb, dpb_layer) = session.picture_images(plan.setup_slot)?;
	let extent = ash::vk::Extent2D {
		width: session.coded_extent.width,
		height: session.coded_extent.height,
	};
	let setup_resource = ash::vk::VideoPictureResourceInfoKHR::default()
		.coded_extent(extent)
		.base_array_layer(dpb_layer)
		.image_view_binding(dpb.view);
	let setup_slot_index =
		i32::try_from(plan.setup_slot).map_err(|_| Error::internal("H.264 setup slot exceeds i32"))?;
	let setup_slot = ash::vk::VideoReferenceSlotInfoKHR::default()
		.slot_index(setup_slot_index)
		.picture_resource(&setup_resource)
		.push_next(&mut setup_h264_slot);

	let reference_count = plan.references.len();
	let mut std_references = Vec::new();
	let mut h264_reference_slots = Vec::new();
	let mut reference_resources = Vec::new();
	let mut reference_slots = Vec::new();
	std_references
		.try_reserve_exact(reference_count)
		.map_err(|_| Error::resource_exhausted("H.264 standard reference allocation failed"))?;
	h264_reference_slots
		.try_reserve_exact(reference_count)
		.map_err(|_| Error::resource_exhausted("H.264 reference chain allocation failed"))?;
	reference_resources
		.try_reserve_exact(reference_count)
		.map_err(|_| Error::resource_exhausted("H.264 reference resource allocation failed"))?;
	reference_slots
		.try_reserve_exact(reference_count + 1)
		.map_err(|_| Error::resource_exhausted("H.264 reference slot allocation failed"))?;
	for reference in &plan.references {
		let (_, _, reference_image, reference_layer) = session.picture_images(reference.slot)?;
		std_references.push(h264_reference_info(
			reference.frame_number,
			reference.picture_order_count,
			reference.long_term,
		)?);
		h264_reference_slots.push(ash::vk::VideoDecodeH264DpbSlotInfoKHR::default());
		reference_resources.push(
			ash::vk::VideoPictureResourceInfoKHR::default()
				.coded_extent(extent)
				.base_array_layer(reference_layer)
				.image_view_binding(reference_image.view),
		);
		reference_slots.push(
			ash::vk::VideoReferenceSlotInfoKHR::default().slot_index(
				i32::try_from(reference.slot)
					.map_err(|_| Error::internal("H.264 reference slot exceeds i32"))?,
			),
		);
	}
	for index in 0..reference_count {
		h264_reference_slots[index].p_std_reference_info = &std_references[index];
		reference_slots[index].p_next =
			(&h264_reference_slots[index] as *const ash::vk::VideoDecodeH264DpbSlotInfoKHR<'_>).cast();
		reference_slots[index].p_picture_resource = &reference_resources[index];
	}
	let mut inactive_setup = setup_slot;
	inactive_setup.slot_index = -1;
	reference_slots.push(inactive_setup);
	let begin_info = ash::vk::VideoBeginCodingInfoKHR::default()
		.video_session(session.handle)
		.video_session_parameters(session.parameters)
		.reference_slots(&reference_slots);
	let output_resource = ash::vk::VideoPictureResourceInfoKHR::default()
		.coded_extent(extent)
		.base_array_layer(output_layer)
		.image_view_binding(output.view);
	let decode_info = ash::vk::VideoDecodeInfoKHR::default()
		.src_buffer(bitstream.handle)
		.src_buffer_offset(0)
		.src_buffer_range(bitstream.range)
		.dst_picture_resource(output_resource)
		.setup_reference_slot(&setup_slot)
		.reference_slots(&reference_slots[..reference_count])
		.push_next(&mut h264_picture);

	let mut image_barriers = Vec::new();
	if initialize_images {
		if session.image_set == DecodeImageSet::CoincidentSeparate {
			image_barriers
				.try_reserve_exact(session.images.len())
				.map_err(|_| Error::resource_exhausted("decode-image barrier allocation failed"))?;
			for image in session.images.iter() {
				image_barriers.push(decode_image_barrier(
					image,
					ash::vk::ImageLayout::VIDEO_DECODE_DPB_KHR,
					ash::vk::AccessFlags2::VIDEO_DECODE_READ_KHR
						| ash::vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR,
				));
			}
		} else {
			image_barriers.push(decode_image_barrier(
				dpb,
				ash::vk::ImageLayout::VIDEO_DECODE_DPB_KHR,
				ash::vk::AccessFlags2::VIDEO_DECODE_READ_KHR
					| ash::vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR,
			));
			if output.handle != dpb.handle {
				image_barriers.push(decode_image_barrier(
					output,
					ash::vk::ImageLayout::VIDEO_DECODE_DST_KHR,
					ash::vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR,
				));
			}
		}
	}
	if let Some(slot) = pending_acquire_slot {
		let (acquired_output, acquired_layer, _, _) = session.picture_images(slot)?;
		let decode_family = session
			.device
			.physical()
			.video
			.decode_queue_family
			.ok_or_else(|| Error::missing_capability("no Vulkan Video decode queue is enabled"))?;
		let compute_family = session.device.physical().compute_queue_family;
		if decode_family != compute_family {
			let layout = if session.image_set == DecodeImageSet::DistinctLayered {
				ash::vk::ImageLayout::VIDEO_DECODE_DST_KHR
			} else {
				ash::vk::ImageLayout::VIDEO_DECODE_DPB_KHR
			};
			image_barriers.push(
				ash::vk::ImageMemoryBarrier2::default()
					.src_stage_mask(ash::vk::PipelineStageFlags2::NONE)
					.src_access_mask(ash::vk::AccessFlags2::NONE)
					.dst_stage_mask(ash::vk::PipelineStageFlags2::VIDEO_DECODE_KHR)
					.dst_access_mask(
						ash::vk::AccessFlags2::VIDEO_DECODE_READ_KHR
							| ash::vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR,
					)
					.old_layout(ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
					.new_layout(layout)
					.src_queue_family_index(compute_family)
					.dst_queue_family_index(decode_family)
					.image(acquired_output.handle)
					.subresource_range(decode_image_subresource(acquired_layer, 1)),
			);
		}
	}
	let bitstream_barriers = [ash::vk::BufferMemoryBarrier2::default()
		.src_stage_mask(ash::vk::PipelineStageFlags2::HOST)
		.src_access_mask(ash::vk::AccessFlags2::HOST_WRITE)
		.dst_stage_mask(ash::vk::PipelineStageFlags2::VIDEO_DECODE_KHR)
		.dst_access_mask(ash::vk::AccessFlags2::VIDEO_DECODE_READ_KHR)
		.src_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
		.dst_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
		.buffer(bitstream.handle)
		.offset(0)
		.size(bitstream.range)];
	let memory_barriers = (!initialize_images)
		.then(|| {
			ash::vk::MemoryBarrier2::default()
				.src_stage_mask(ash::vk::PipelineStageFlags2::VIDEO_DECODE_KHR)
				.src_access_mask(ash::vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR)
				.dst_stage_mask(ash::vk::PipelineStageFlags2::VIDEO_DECODE_KHR)
				.dst_access_mask(
					ash::vk::AccessFlags2::VIDEO_DECODE_READ_KHR
						| ash::vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR,
				)
		})
		.into_iter()
		.collect::<Vec<_>>();
	let dependency = ash::vk::DependencyInfo::default()
		.memory_barriers(&memory_barriers)
		.buffer_memory_barriers(&bitstream_barriers)
		.image_memory_barriers(&image_barriers);
	unsafe {
		device.cmd_reset_query_pool(command_buffer, session.result_status_pool, 0, 1);
		device.cmd_pipeline_barrier2(command_buffer, &dependency);
		(loader.fp().cmd_begin_video_coding_khr)(command_buffer, &begin_info);
	}
	if plan.reset_dpb {
		let control = ash::vk::VideoCodingControlInfoKHR::default()
			.flags(ash::vk::VideoCodingControlFlagsKHR::RESET);
		unsafe {
			(loader.fp().cmd_control_video_coding_khr)(command_buffer, &control);
		}
	}
	unsafe {
		device.cmd_begin_query(
			command_buffer,
			session.result_status_pool,
			0,
			ash::vk::QueryControlFlags::empty(),
		);
		(decode_loader.fp().cmd_decode_video_khr)(command_buffer, &decode_info);
		device.cmd_end_query(command_buffer, session.result_status_pool, 0);
		(loader.fp().cmd_end_video_coding_khr)(
			command_buffer,
			&ash::vk::VideoEndCodingInfoKHR::default(),
		);
	}
	if release_for_readback {
		let decode_family = session
			.device
			.physical()
			.video
			.decode_queue_family
			.ok_or_else(|| Error::missing_capability("no Vulkan Video decode queue is enabled"))?;
		let compute_family = session.device.physical().compute_queue_family;
		let same_family = decode_family == compute_family;
		let decoded_layout = if output.handle == dpb.handle {
			ash::vk::ImageLayout::VIDEO_DECODE_DPB_KHR
		} else {
			ash::vk::ImageLayout::VIDEO_DECODE_DST_KHR
		};
		let release = ash::vk::ImageMemoryBarrier2::default()
			.src_stage_mask(ash::vk::PipelineStageFlags2::VIDEO_DECODE_KHR)
			.src_access_mask(
				ash::vk::AccessFlags2::VIDEO_DECODE_READ_KHR
					| ash::vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR,
			)
			.dst_stage_mask(if same_family {
				ash::vk::PipelineStageFlags2::TRANSFER
			} else {
				ash::vk::PipelineStageFlags2::NONE
			})
			.dst_access_mask(if same_family {
				ash::vk::AccessFlags2::TRANSFER_READ
			} else {
				ash::vk::AccessFlags2::NONE
			})
			.old_layout(decoded_layout)
			.new_layout(ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
			.src_queue_family_index(if same_family {
				ash::vk::QUEUE_FAMILY_IGNORED
			} else {
				decode_family
			})
			.dst_queue_family_index(if same_family {
				ash::vk::QUEUE_FAMILY_IGNORED
			} else {
				compute_family
			})
			.image(output.handle)
			.subresource_range(decode_image_subresource(output_layer, 1));
		let release_dependency =
			ash::vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&release));
		unsafe {
			device.cmd_pipeline_barrier2(command_buffer, &release_dependency);
		}
	}
	Ok(())
}

#[allow(clippy::too_many_arguments)]
fn record_h265_picture(
	device: &ash::Device,
	loader: &ash::khr::video_queue::Device,
	decode_loader: &ash::khr::video_decode_queue::Device,
	command_buffer: ash::vk::CommandBuffer,
	session: &DecodeSession,
	sps: &video::H265SequenceParameterSet,
	pps: &video::H265PictureParameterSet,
	slice: &video::H265SliceHeader,
	slice_offset: u32,
	bitstream: &DecodeBitstream,
	plan: &H265PicturePlan,
	initialize_images: bool,
	pending_acquire_slot: Option<u32>,
	release_for_readback: bool,
) -> Result<()> {
	let num_delta_pocs = if slice.short_term_reference_picture_set_sps {
		let index = slice
			.short_term_reference_picture_set_index
			.ok_or_else(|| Error::data_loss("H.265 SPS reference-set selection omitted its index"))?;
		let set = sps
			.short_term_reference_picture_sets
			.get(
				usize::try_from(index)
					.map_err(|_| Error::data_loss("H.265 reference-set index exceeds host address space"))?,
			)
			.ok_or_else(|| Error::data_loss("H.265 reference-set index exceeds the SPS"))?;
		set.delta_pocs.len()
	} else {
		// video.xml requires zero when the set is carried inline in the slice.
		0
	};
	let mut std_picture = ash::vk::native::StdVideoDecodeH265PictureInfo {
		flags: ash::vk::native::StdVideoDecodeH265PictureInfoFlags {
			_bitfield_align_1: [],
			_bitfield_1: ash::vk::native::StdVideoDecodeH265PictureInfoFlags::new_bitfield_1(
				u32::from(slice.is_irap),
				u32::from(slice.is_idr),
				u32::from(slice.is_reference),
				u32::from(slice.short_term_reference_picture_set_sps),
			),
			__bindgen_padding_0: [0; 3],
		},
		sps_video_parameter_set_id: h265_u8(sps.video_parameter_set_id, "picture VPS id")?,
		pps_seq_parameter_set_id: h265_u8(sps.id, "picture SPS id")?,
		pps_pic_parameter_set_id: h265_u8(pps.id, "picture PPS id")?,
		NumDeltaPocsOfRefRpsIdx: u8::try_from(num_delta_pocs)
			.map_err(|_| Error::data_loss("H.265 reference-set delta count exceeds u8"))?,
		PicOrderCntVal: plan.picture_order_count,
		NumBitsForSTRefPicSetInSlice: slice.short_term_reference_picture_set_bits,
		reserved: 0,
		RefPicSetStCurrBefore: [0xff; 8],
		RefPicSetStCurrAfter: [0xff; 8],
		RefPicSetLtCurr: [0xff; 8],
	};
	for (destination, slot) in std_picture
		.RefPicSetStCurrBefore
		.iter_mut()
		.zip(&plan.current_before_slots)
	{
		*destination = *slot;
	}
	for (destination, slot) in std_picture
		.RefPicSetStCurrAfter
		.iter_mut()
		.zip(&plan.current_after_slots)
	{
		*destination = *slot;
	}
	let mut h265_picture = ash::vk::VideoDecodeH265PictureInfoKHR::default()
		.std_picture_info(&std_picture)
		.slice_segment_offsets(std::slice::from_ref(&slice_offset));

	let setup_std_reference = h265_reference_info(plan.picture_order_count, !slice.is_reference);
	let mut setup_h265_slot =
		ash::vk::VideoDecodeH265DpbSlotInfoKHR::default().std_reference_info(&setup_std_reference);
	let (output, output_layer, dpb, dpb_layer) = session.picture_images(plan.setup_slot)?;
	let extent = ash::vk::Extent2D {
		width: session.coded_extent.width,
		height: session.coded_extent.height,
	};
	let setup_resource = ash::vk::VideoPictureResourceInfoKHR::default()
		.coded_extent(extent)
		.base_array_layer(dpb_layer)
		.image_view_binding(dpb.view);
	let setup_slot_index =
		i32::try_from(plan.setup_slot).map_err(|_| Error::internal("H.265 setup slot exceeds i32"))?;
	let setup_slot = ash::vk::VideoReferenceSlotInfoKHR::default()
		.slot_index(setup_slot_index)
		.picture_resource(&setup_resource)
		.push_next(&mut setup_h265_slot);

	let mut std_references = Vec::new();
	let mut h265_reference_slots = Vec::new();
	let mut reference_resources = Vec::new();
	let mut reference_slots = Vec::new();
	let reference_count = plan.active_references.len();
	std_references
		.try_reserve_exact(reference_count)
		.map_err(|_| Error::resource_exhausted("H.265 standard reference allocation failed"))?;
	h265_reference_slots
		.try_reserve_exact(reference_count)
		.map_err(|_| Error::resource_exhausted("H.265 reference chain allocation failed"))?;
	reference_resources
		.try_reserve_exact(reference_count)
		.map_err(|_| Error::resource_exhausted("H.265 reference resource allocation failed"))?;
	reference_slots
		.try_reserve_exact(reference_count)
		.map_err(|_| Error::resource_exhausted("H.265 reference slot allocation failed"))?;
	for reference in &plan.active_references {
		let (_, _, reference_image, reference_layer) = session.picture_images(reference.slot)?;
		std_references.push(h265_reference_info(reference.picture_order_count, false));
		h265_reference_slots.push(ash::vk::VideoDecodeH265DpbSlotInfoKHR::default());
		reference_resources.push(
			ash::vk::VideoPictureResourceInfoKHR::default()
				.coded_extent(extent)
				.base_array_layer(reference_layer)
				.image_view_binding(reference_image.view),
		);
		reference_slots.push(
			ash::vk::VideoReferenceSlotInfoKHR::default().slot_index(
				i32::try_from(reference.slot)
					.map_err(|_| Error::internal("H.265 reference slot exceeds i32"))?,
			),
		);
	}
	for index in 0..reference_count {
		h265_reference_slots[index].p_std_reference_info = &std_references[index];
		reference_slots[index].p_next =
			(&h265_reference_slots[index] as *const ash::vk::VideoDecodeH265DpbSlotInfoKHR<'_>).cast();
		reference_slots[index].p_picture_resource = &reference_resources[index];
	}
	let mut inactive_setup = setup_slot;
	inactive_setup.slot_index = -1;
	reference_slots.push(inactive_setup);
	let begin_info = ash::vk::VideoBeginCodingInfoKHR::default()
		.video_session(session.handle)
		.video_session_parameters(session.parameters)
		.reference_slots(&reference_slots);
	let output_resource = ash::vk::VideoPictureResourceInfoKHR::default()
		.coded_extent(extent)
		.base_array_layer(output_layer)
		.image_view_binding(output.view);
	let decode_info = ash::vk::VideoDecodeInfoKHR::default()
		.src_buffer(bitstream.handle)
		.src_buffer_offset(0)
		.src_buffer_range(bitstream.range)
		.dst_picture_resource(output_resource)
		.setup_reference_slot(&setup_slot)
		.reference_slots(&reference_slots[..reference_count])
		.push_next(&mut h265_picture);

	let mut image_barriers = Vec::new();
	if initialize_images {
		if session.image_set == DecodeImageSet::CoincidentSeparate {
			image_barriers
				.try_reserve_exact(session.images.len())
				.map_err(|_| Error::resource_exhausted("decode-image barrier allocation failed"))?;
			for image in session.images.iter() {
				image_barriers.push(decode_image_barrier(
					image,
					ash::vk::ImageLayout::VIDEO_DECODE_DPB_KHR,
					ash::vk::AccessFlags2::VIDEO_DECODE_READ_KHR
						| ash::vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR,
				));
			}
		} else {
			image_barriers.push(decode_image_barrier(
				dpb,
				ash::vk::ImageLayout::VIDEO_DECODE_DPB_KHR,
				ash::vk::AccessFlags2::VIDEO_DECODE_READ_KHR
					| ash::vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR,
			));
			if output.handle != dpb.handle {
				image_barriers.push(decode_image_barrier(
					output,
					ash::vk::ImageLayout::VIDEO_DECODE_DST_KHR,
					ash::vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR,
				));
			}
		}
	}
	if let Some(slot) = pending_acquire_slot {
		let (acquired_output, acquired_layer, _, _) = session.picture_images(slot)?;
		let decode_family = session
			.device
			.physical()
			.video
			.decode_queue_family
			.ok_or_else(|| Error::missing_capability("no Vulkan Video decode queue is enabled"))?;
		let compute_family = session.device.physical().compute_queue_family;
		if decode_family != compute_family {
			let layout = if session.image_set == DecodeImageSet::DistinctLayered {
				ash::vk::ImageLayout::VIDEO_DECODE_DST_KHR
			} else {
				ash::vk::ImageLayout::VIDEO_DECODE_DPB_KHR
			};
			image_barriers.push(
				ash::vk::ImageMemoryBarrier2::default()
					.src_stage_mask(ash::vk::PipelineStageFlags2::NONE)
					.src_access_mask(ash::vk::AccessFlags2::NONE)
					.dst_stage_mask(ash::vk::PipelineStageFlags2::VIDEO_DECODE_KHR)
					.dst_access_mask(
						ash::vk::AccessFlags2::VIDEO_DECODE_READ_KHR
							| ash::vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR,
					)
					.old_layout(ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
					.new_layout(layout)
					.src_queue_family_index(compute_family)
					.dst_queue_family_index(decode_family)
					.image(acquired_output.handle)
					.subresource_range(decode_image_subresource(acquired_layer, 1)),
			);
		}
	}
	let bitstream_barriers = [ash::vk::BufferMemoryBarrier2::default()
		.src_stage_mask(ash::vk::PipelineStageFlags2::HOST)
		.src_access_mask(ash::vk::AccessFlags2::HOST_WRITE)
		.dst_stage_mask(ash::vk::PipelineStageFlags2::VIDEO_DECODE_KHR)
		.dst_access_mask(ash::vk::AccessFlags2::VIDEO_DECODE_READ_KHR)
		.src_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
		.dst_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
		.buffer(bitstream.handle)
		.offset(0)
		.size(bitstream.range)];
	let memory_barriers = (!initialize_images)
		.then(|| {
			ash::vk::MemoryBarrier2::default()
				.src_stage_mask(ash::vk::PipelineStageFlags2::VIDEO_DECODE_KHR)
				.src_access_mask(ash::vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR)
				.dst_stage_mask(ash::vk::PipelineStageFlags2::VIDEO_DECODE_KHR)
				.dst_access_mask(
					ash::vk::AccessFlags2::VIDEO_DECODE_READ_KHR
						| ash::vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR,
				)
		})
		.into_iter()
		.collect::<Vec<_>>();
	let dependency = ash::vk::DependencyInfo::default()
		.memory_barriers(&memory_barriers)
		.buffer_memory_barriers(&bitstream_barriers)
		.image_memory_barriers(&image_barriers);
	// The DPB dependency and layout transitions precede the coding scope. The
	// result-status query is reused only after the caller has completed and
	// checked the preceding picture event.
	unsafe {
		device.cmd_reset_query_pool(command_buffer, session.result_status_pool, 0, 1);
		device.cmd_pipeline_barrier2(command_buffer, &dependency);
		(loader.fp().cmd_begin_video_coding_khr)(command_buffer, &begin_info);
	}
	if plan.reset_dpb {
		let control = ash::vk::VideoCodingControlInfoKHR::default()
			.flags(ash::vk::VideoCodingControlFlagsKHR::RESET);
		unsafe {
			(loader.fp().cmd_control_video_coding_khr)(command_buffer, &control);
		}
	}
	unsafe {
		device.cmd_begin_query(
			command_buffer,
			session.result_status_pool,
			0,
			ash::vk::QueryControlFlags::empty(),
		);
		(decode_loader.fp().cmd_decode_video_khr)(command_buffer, &decode_info);
		device.cmd_end_query(command_buffer, session.result_status_pool, 0);
		(loader.fp().cmd_end_video_coding_khr)(
			command_buffer,
			&ash::vk::VideoEndCodingInfoKHR::default(),
		);
	}
	if release_for_readback {
		let decode_family = session
			.device
			.physical()
			.video
			.decode_queue_family
			.ok_or_else(|| Error::missing_capability("no Vulkan Video decode queue is enabled"))?;
		let compute_family = session.device.physical().compute_queue_family;
		let same_family = decode_family == compute_family;
		let decoded_layout = if output.handle == dpb.handle {
			ash::vk::ImageLayout::VIDEO_DECODE_DPB_KHR
		} else {
			ash::vk::ImageLayout::VIDEO_DECODE_DST_KHR
		};
		let release = ash::vk::ImageMemoryBarrier2::default()
			.src_stage_mask(ash::vk::PipelineStageFlags2::VIDEO_DECODE_KHR)
			.src_access_mask(
				ash::vk::AccessFlags2::VIDEO_DECODE_READ_KHR
					| ash::vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR,
			)
			.dst_stage_mask(if same_family {
				ash::vk::PipelineStageFlags2::TRANSFER
			} else {
				ash::vk::PipelineStageFlags2::NONE
			})
			.dst_access_mask(if same_family {
				ash::vk::AccessFlags2::TRANSFER_READ
			} else {
				ash::vk::AccessFlags2::NONE
			})
			.old_layout(decoded_layout)
			.new_layout(ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
			.src_queue_family_index(if same_family {
				ash::vk::QUEUE_FAMILY_IGNORED
			} else {
				decode_family
			})
			.dst_queue_family_index(if same_family {
				ash::vk::QUEUE_FAMILY_IGNORED
			} else {
				compute_family
			})
			.image(output.handle)
			.subresource_range(decode_image_subresource(output_layer, 1));
		let release_dependency =
			ash::vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&release));
		// SAFETY: decode owns this exact output layer. The caller submits the
		// matching compute-family acquire only after this queue signals completion.
		unsafe {
			device.cmd_pipeline_barrier2(command_buffer, &release_dependency);
		}
	}
	Ok(())
}

fn h265_reference_info(
	picture_order_count: i32,
	unused_for_reference: bool,
) -> ash::vk::native::StdVideoDecodeH265ReferenceInfo {
	ash::vk::native::StdVideoDecodeH265ReferenceInfo {
		flags: ash::vk::native::StdVideoDecodeH265ReferenceInfoFlags {
			_bitfield_align_1: [],
			_bitfield_1: ash::vk::native::StdVideoDecodeH265ReferenceInfoFlags::new_bitfield_1(
				0,
				u32::from(unused_for_reference),
			),
			__bindgen_padding_0: [0; 3],
		},
		PicOrderCntVal: picture_order_count,
	}
}

fn record_decode_readback_release(
	device: &ash::Device,
	command_buffer: ash::vk::CommandBuffer,
	output: &DecodeImage,
	output_layer: u32,
	queue_families: (u32, u32),
	decoded_layout: ash::vk::ImageLayout,
) -> Result<()> {
	let (decode_family, compute_family) = queue_families;
	let same_family = decode_family == compute_family;
	let release = ash::vk::ImageMemoryBarrier2::default()
		.src_stage_mask(ash::vk::PipelineStageFlags2::VIDEO_DECODE_KHR)
		.src_access_mask(
			ash::vk::AccessFlags2::VIDEO_DECODE_READ_KHR | ash::vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR,
		)
		.dst_stage_mask(if same_family {
			ash::vk::PipelineStageFlags2::TRANSFER
		} else {
			ash::vk::PipelineStageFlags2::NONE
		})
		.dst_access_mask(if same_family {
			ash::vk::AccessFlags2::TRANSFER_READ
		} else {
			ash::vk::AccessFlags2::NONE
		})
		.old_layout(decoded_layout)
		.new_layout(ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
		.src_queue_family_index(if same_family {
			ash::vk::QUEUE_FAMILY_IGNORED
		} else {
			decode_family
		})
		.dst_queue_family_index(if same_family {
			ash::vk::QUEUE_FAMILY_IGNORED
		} else {
			compute_family
		})
		.image(output.handle)
		.subresource_range(decode_image_subresource(output_layer, 1));
	let dependency =
		ash::vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&release));
	// SAFETY: the frame's producer event is complete before submission. This
	// publishes its decode writes and transfers the exact retained layer to the
	// compute family when the queue families differ.
	unsafe {
		device.cmd_pipeline_barrier2(command_buffer, &dependency);
	}
	Ok(())
}

#[allow(clippy::too_many_arguments)]
fn record_decode_readback(
	device: &ash::Device,
	command_buffer: ash::vk::CommandBuffer,
	output: &DecodeImage,
	output_layer: u32,
	readback: &super::Buffer,
	regions: &[ash::vk::BufferImageCopy],
	queue_families: (u32, u32),
	decoded_layout: ash::vk::ImageLayout,
) -> Result<()> {
	let (decode_family, compute_family) = queue_families;
	if decode_family != compute_family {
		let acquire = ash::vk::ImageMemoryBarrier2::default()
			.src_stage_mask(ash::vk::PipelineStageFlags2::NONE)
			.src_access_mask(ash::vk::AccessFlags2::NONE)
			.dst_stage_mask(ash::vk::PipelineStageFlags2::TRANSFER)
			.dst_access_mask(ash::vk::AccessFlags2::TRANSFER_READ)
			.old_layout(decoded_layout)
			.new_layout(ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
			.src_queue_family_index(decode_family)
			.dst_queue_family_index(compute_family)
			.image(output.handle)
			.subresource_range(decode_image_subresource(output_layer, 1));
		let dependency =
			ash::vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&acquire));
		// SAFETY: this matches the release recorded on the decode queue. The engine
		// timeline wait orders this acquire after that queue's signal operation.
		unsafe {
			device.cmd_pipeline_barrier2(command_buffer, &dependency);
		}
	}
	// SAFETY: the output layer is transfer-readable and each tightly packed plane
	// region is in bounds of the retained destination buffer.
	unsafe {
		device.cmd_copy_image_to_buffer(
			command_buffer,
			output.handle,
			ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
			readback.raw(),
			regions,
		);
	}
	let host = ash::vk::BufferMemoryBarrier2::default()
		.src_stage_mask(ash::vk::PipelineStageFlags2::TRANSFER)
		.src_access_mask(ash::vk::AccessFlags2::TRANSFER_WRITE)
		.dst_stage_mask(ash::vk::PipelineStageFlags2::HOST)
		.dst_access_mask(ash::vk::AccessFlags2::HOST_READ)
		.src_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
		.dst_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
		.buffer(readback.raw())
		.offset(0)
		.size(readback.size());
	let dependency =
		ash::vk::DependencyInfo::default().buffer_memory_barriers(std::slice::from_ref(&host));
	// SAFETY: the barrier publishes every transfer-written byte to the host domain.
	unsafe {
		device.cmd_pipeline_barrier2(command_buffer, &dependency);
	}
	Ok(())
}

#[allow(clippy::too_many_arguments)]
fn record_decode_readback_round_trip(
	device: &ash::Device,
	command_buffer: ash::vk::CommandBuffer,
	output: &DecodeImage,
	output_layer: u32,
	readback: &super::Buffer,
	regions: &[ash::vk::BufferImageCopy],
	queue_families: (u32, u32),
	decoded_layout: ash::vk::ImageLayout,
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
	let release = ash::vk::ImageMemoryBarrier2::default()
		.src_stage_mask(ash::vk::PipelineStageFlags2::TRANSFER)
		.src_access_mask(ash::vk::AccessFlags2::TRANSFER_READ)
		.dst_stage_mask(if same_family {
			ash::vk::PipelineStageFlags2::VIDEO_DECODE_KHR
		} else {
			ash::vk::PipelineStageFlags2::NONE
		})
		.dst_access_mask(if same_family {
			ash::vk::AccessFlags2::VIDEO_DECODE_READ_KHR | ash::vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR
		} else {
			ash::vk::AccessFlags2::NONE
		})
		.old_layout(ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
		.new_layout(decoded_layout)
		.src_queue_family_index(if same_family {
			ash::vk::QUEUE_FAMILY_IGNORED
		} else {
			compute_family
		})
		.dst_queue_family_index(if same_family {
			ash::vk::QUEUE_FAMILY_IGNORED
		} else {
			decode_family
		})
		.image(output.handle)
		.subresource_range(decode_image_subresource(output_layer, 1));
	let dependency =
		ash::vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&release));
	// SAFETY: the transfer read is complete in this command buffer. A different
	// decode family performs the matching acquire after the engine timeline wait.
	unsafe {
		device.cmd_pipeline_barrier2(command_buffer, &dependency);
	}
	Ok(())
}

fn record_decode_readback_acquire(
	device: &ash::Device,
	command_buffer: ash::vk::CommandBuffer,
	output: &DecodeImage,
	output_layer: u32,
	queue_families: (u32, u32),
	decoded_layout: ash::vk::ImageLayout,
) -> Result<()> {
	let (decode_family, compute_family) = queue_families;
	if decode_family == compute_family {
		return Ok(());
	}
	let acquire = ash::vk::ImageMemoryBarrier2::default()
		.src_stage_mask(ash::vk::PipelineStageFlags2::NONE)
		.src_access_mask(ash::vk::AccessFlags2::NONE)
		.dst_stage_mask(ash::vk::PipelineStageFlags2::VIDEO_DECODE_KHR)
		.dst_access_mask(
			ash::vk::AccessFlags2::VIDEO_DECODE_READ_KHR | ash::vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR,
		)
		.old_layout(ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
		.new_layout(decoded_layout)
		.src_queue_family_index(compute_family)
		.dst_queue_family_index(decode_family)
		.image(output.handle)
		.subresource_range(decode_image_subresource(output_layer, 1));
	let dependency =
		ash::vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&acquire));
	// SAFETY: this matches the compute-family release recorded after the copy.
	// The Engine timeline orders this acquire after that submission.
	unsafe {
		device.cmd_pipeline_barrier2(command_buffer, &dependency);
	}
	Ok(())
}

fn decode_image_subresource(
	base_array_layer: u32,
	layer_count: u32,
) -> ash::vk::ImageSubresourceRange {
	ash::vk::ImageSubresourceRange {
		aspect_mask: ash::vk::ImageAspectFlags::COLOR,
		base_mip_level: 0,
		level_count: 1,
		base_array_layer,
		layer_count,
	}
}

fn decode_image_barrier(
	image: &DecodeImage,
	new_layout: ash::vk::ImageLayout,
	destination_access: ash::vk::AccessFlags2,
) -> ash::vk::ImageMemoryBarrier2<'static> {
	ash::vk::ImageMemoryBarrier2::default()
		.src_stage_mask(ash::vk::PipelineStageFlags2::NONE)
		.src_access_mask(ash::vk::AccessFlags2::NONE)
		.dst_stage_mask(ash::vk::PipelineStageFlags2::VIDEO_DECODE_KHR)
		.dst_access_mask(destination_access)
		.old_layout(ash::vk::ImageLayout::UNDEFINED)
		.new_layout(new_layout)
		.src_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
		.dst_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
		.image(image.handle)
		.subresource_range(decode_image_subresource(0, image.array_layers))
}

impl Drop for DecodeSession {
	fn drop(&mut self) {
		if let Some(mut bitstream) = self.bitstream.take() {
			// SAFETY: no decode submission can exist for this private qualification
			// session, which uniquely owns the buffer/allocation pair.
			unsafe {
				self
					.device
					.allocator()
					.destroy_buffer(bitstream.handle, &mut bitstream.allocation);
			}
		}
		if self.result_status_pool != ash::vk::QueryPool::null() {
			// SAFETY: this private session owns the pool. Its submitted qualification
			// command is explicitly completed before session teardown in the test path.
			unsafe {
				self
					.device
					.raw()
					.destroy_query_pool(self.result_status_pool, None);
			}
			self.result_status_pool = ash::vk::QueryPool::null();
		}
		if self.parameters != ash::vk::VideoSessionParametersKHR::null() {
			// SAFETY: this wrapper owns the parameter object and destroys it before
			// the video session on which it depends.
			unsafe {
				(self.loader.fp().destroy_video_session_parameters_khr)(
					self.loader.device(),
					self.parameters,
					std::ptr::null(),
				);
			}
			self.parameters = ash::vk::VideoSessionParametersKHR::null();
		}
		if self.handle != ash::vk::VideoSessionKHR::null() {
			// SAFETY: this wrapper owns the session and destroys it once before its
			// bound allocations and retained device are released.
			unsafe {
				(self.loader.fp().destroy_video_session_khr)(
					self.loader.device(),
					self.handle,
					std::ptr::null(),
				);
			}
			self.handle = ash::vk::VideoSessionKHR::null();
		}
		for allocation in &mut self.allocations {
			// SAFETY: session destruction ended every binding use, and this wrapper
			// owns each VMA allocation exactly once.
			unsafe {
				self.device.allocator().free_memory(allocation);
			}
		}
	}
}

#[cfg(test)]
mod h265_dpb_tests {
	use super::H265DpbState;

	#[test]
	#[ignore = "requires the OA donor HEVC fixture"]
	fn plans_every_donor_picture_in_decode_order() -> crate::Result<()> {
		let fixture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
			.with_file_name("oa")
			.join("sdk/asset/video/clip/shibuya_720p_30fps_h265_main_8bit_420.mp4");
		let mut demuxer = crate::video::VideoDemuxer::open(fixture)?;
		let mut parameter_sets = None;
		let mut state = None;
		let mut planned = 0_u32;
		let mut reordered = false;
		let mut reset_count = 0_u32;
		let mut previous_poc = None;
		while let Some(packet) = demuxer.read_next_packet()? {
			let nals = crate::video::parse_nal_annex_b(packet.data());
			if parameter_sets.is_none() {
				let find = |nal_type| {
					nals
						.iter()
						.find(|nal| (nal.payload()[0] >> 1) & 0x3f == nal_type)
						.map(|nal| nal.payload())
				};
				if let (Some(vps), Some(sps), Some(pps)) = (find(32), find(33), find(34)) {
					parameter_sets = Some((
						crate::video::parse_h265_vps(vps)?,
						crate::video::parse_h265_sps(sps)?,
						crate::video::parse_h265_pps(pps)?,
					));
				}
			}
			let (_, sps, pps) = parameter_sets
				.as_ref()
				.expect("fixture must begin with VPS/SPS/PPS");
			let mut coded = nals
				.iter()
				.filter(|nal| (nal.payload()[0] >> 1) & 0x3f < 32);
			let Some(slice_nal) = coded.next() else {
				continue;
			};
			assert!(
				coded.next().is_none(),
				"fixture qualification expects one slice per picture"
			);
			let slice = crate::video::parse_h265_slice_header(slice_nal.payload(), sps, pps)?;
			let slot_count = sps
				.max_decoded_picture_buffering_minus_1
				.first()
				.copied()
				.unwrap_or(3)
				.saturating_add(1)
				.clamp(1, 16);
			let state = state.get_or_insert(H265DpbState::new(slot_count)?);
			let plan = state.plan(sps, &slice).unwrap_or_else(|error| {
				panic!(
					"HEVC DPB planning failed at packet {planned}: {error}; slice={slice:?}; state={state:?}"
				)
			});
			reset_count += u32::from(plan.reset_dpb);
			assert!(plan.setup_slot < slot_count);
			assert!(
				plan
					.active_references
					.iter()
					.all(|reference| reference.slot < slot_count)
			);
			assert!(
				plan
					.current_before_slots
					.iter()
					.all(|slot| u32::from(*slot) < slot_count)
			);
			assert!(
				plan
					.current_after_slots
					.iter()
					.all(|slot| u32::from(*slot) < slot_count)
			);
			if previous_poc.is_some_and(|previous| plan.picture_order_count < previous) {
				reordered = true;
			}
			previous_poc = Some(plan.picture_order_count);
			planned = planned
				.checked_add(1)
				.expect("fixture picture count must fit u32");
		}
		assert_eq!(planned, demuxer.info().sample_count());
		assert!(
			reordered,
			"fixture must exercise non-monotonic decode-order POC"
		);
		assert_eq!(
			reset_count, 1,
			"fixture must reset DPB only at its opening IDR"
		);
		Ok(())
	}
}

struct StdH265Hrd {
	_nal: Box<[ash::vk::native::StdVideoH265SubLayerHrdParameters]>,
	_vcl: Box<[ash::vk::native::StdVideoH265SubLayerHrdParameters]>,
	value: ash::vk::native::StdVideoH265HrdParameters,
}

impl StdH265Hrd {
	fn as_ptr(&self) -> *const ash::vk::native::StdVideoH265HrdParameters {
		&self.value
	}
}

fn std_h265_profile_tier_level(
	profile: &video::H265ProfileTierLevel,
) -> Result<ash::vk::native::StdVideoH265ProfileTierLevel> {
	let profile_idc = match profile.profile_idc {
		1 | 2 | 3 | 4 | 9 => profile.profile_idc,
		value => {
			return Err(Error::missing_capability(format!(
				"H.265 profile_idc {value} has no supported StdVideo mapping"
			)));
		}
	};
	Ok(ash::vk::native::StdVideoH265ProfileTierLevel {
		flags: ash::vk::native::StdVideoH265ProfileTierLevelFlags {
			_bitfield_align_1: [],
			_bitfield_1: ash::vk::native::StdVideoH265ProfileTierLevelFlags::new_bitfield_1(
				u32::from(profile.high_tier),
				u32::from(profile.progressive_source),
				u32::from(profile.interlaced_source),
				u32::from(profile.non_packed_constraint),
				u32::from(profile.frame_only_constraint),
			),
			__bindgen_padding_0: [0; 3],
		},
		general_profile_idc: profile_idc,
		general_level_idc: std_h265_level(profile.level_idc)?,
	})
}

fn std_h265_level(level_idc: u32) -> Result<ash::vk::native::StdVideoH265LevelIdc> {
	match level_idc {
		30 => Ok(0),
		60 => Ok(1),
		63 => Ok(2),
		90 => Ok(3),
		93 => Ok(4),
		120 => Ok(5),
		123 => Ok(6),
		150 => Ok(7),
		153 => Ok(8),
		156 => Ok(9),
		180 => Ok(10),
		183 => Ok(11),
		186 => Ok(12),
		value => Err(Error::missing_capability(format!(
			"H.265 level_idc {value} has no Khronos StdVideo mapping"
		))),
	}
}

fn std_h265_dpb(
	dpb: &video::H265DecodedPictureBuffer,
) -> Result<ash::vk::native::StdVideoH265DecPicBufMgr> {
	let mut max_dec_pic_buffering_minus1 = [0_u8; 7];
	let mut max_num_reorder_pics = [0_u8; 7];
	for index in 0..7 {
		max_dec_pic_buffering_minus1[index] =
			u8::try_from(dpb.max_decoded_picture_buffering_minus_1[index])
				.map_err(|_| Error::data_loss("H.265 DPB capacity exceeds StdVideo storage"))?;
		max_num_reorder_pics[index] = u8::try_from(dpb.max_num_reorder_pictures[index])
			.map_err(|_| Error::data_loss("H.265 reorder count exceeds StdVideo storage"))?;
	}
	Ok(ash::vk::native::StdVideoH265DecPicBufMgr {
		max_latency_increase_plus1: dpb.max_latency_increase_plus_1,
		max_dec_pic_buffering_minus1,
		max_num_reorder_pics,
	})
}

fn std_h265_hrd(hrd: &video::H265HrdParameters) -> Result<StdH265Hrd> {
	if hrd.sub_layers.len() > 7 {
		return Err(Error::data_loss("H.265 HRD sub-layer count exceeds seven"));
	}
	let mut cpb_cnt_minus1 = [0_u8; 7];
	let mut elemental_duration_in_tc_minus1 = [0_u16; 7];
	let mut fixed_general = 0_u32;
	let mut fixed_within = 0_u32;
	let mut low_delay = 0_u32;
	let mut nal = Vec::with_capacity(hrd.sub_layers.len());
	let mut vcl = Vec::with_capacity(hrd.sub_layers.len());
	for (index, layer) in hrd.sub_layers.iter().enumerate() {
		fixed_general |= u32::from(layer.fixed_picture_rate_general) << index;
		fixed_within |= u32::from(layer.fixed_picture_rate_within_cvs) << index;
		low_delay |= u32::from(layer.low_delay) << index;
		elemental_duration_in_tc_minus1[index] = u16::try_from(layer.elemental_duration_in_tc_minus_1)
			.map_err(|_| Error::data_loss("H.265 HRD elemental duration exceeds StdVideo storage"))?;
		let entries = if hrd.nal_parameters_present {
			&layer.nal_entries
		} else {
			&layer.vcl_entries
		};
		let count_minus_1 = entries
			.len()
			.checked_sub(1)
			.ok_or_else(|| Error::data_loss("H.265 HRD sub-layer has no CPB entries"))?;
		cpb_cnt_minus1[index] = u8::try_from(count_minus_1)
			.map_err(|_| Error::data_loss("H.265 HRD CPB count exceeds StdVideo storage"))?;
		if hrd.nal_parameters_present {
			nal.push(std_h265_sub_layer_hrd(&layer.nal_entries)?);
		}
		if hrd.vcl_parameters_present {
			if hrd.nal_parameters_present && layer.nal_entries.len() != layer.vcl_entries.len() {
				return Err(Error::data_loss(
					"H.265 NAL and VCL HRD tables use different CPB counts",
				));
			}
			vcl.push(std_h265_sub_layer_hrd(&layer.vcl_entries)?);
		}
	}
	let nal = nal.into_boxed_slice();
	let vcl = vcl.into_boxed_slice();
	let value = ash::vk::native::StdVideoH265HrdParameters {
		flags: ash::vk::native::StdVideoH265HrdFlags {
			_bitfield_align_1: [],
			_bitfield_1: ash::vk::native::StdVideoH265HrdFlags::new_bitfield_1(
				u32::from(hrd.nal_parameters_present),
				u32::from(hrd.vcl_parameters_present),
				u32::from(hrd.sub_picture_parameters_present),
				u32::from(hrd.sub_picture_cpb_parameters_in_picture_timing_sei),
				fixed_general,
				fixed_within,
				low_delay,
			),
		},
		tick_divisor_minus2: hrd.tick_divisor_minus_2,
		du_cpb_removal_delay_increment_length_minus1: hrd.du_cpb_removal_delay_increment_length_minus_1,
		dpb_output_delay_du_length_minus1: hrd.dpb_output_delay_du_length_minus_1,
		bit_rate_scale: hrd.bit_rate_scale,
		cpb_size_scale: hrd.cpb_size_scale,
		cpb_size_du_scale: hrd.cpb_size_du_scale,
		initial_cpb_removal_delay_length_minus1: hrd.initial_cpb_removal_delay_length_minus_1,
		au_cpb_removal_delay_length_minus1: hrd.au_cpb_removal_delay_length_minus_1,
		dpb_output_delay_length_minus1: hrd.dpb_output_delay_length_minus_1,
		cpb_cnt_minus1,
		elemental_duration_in_tc_minus1,
		reserved: [0; 3],
		pSubLayerHrdParametersNal: if nal.is_empty() {
			std::ptr::null()
		} else {
			nal.as_ptr()
		},
		pSubLayerHrdParametersVcl: if vcl.is_empty() {
			std::ptr::null()
		} else {
			vcl.as_ptr()
		},
	};
	Ok(StdH265Hrd {
		_nal: nal,
		_vcl: vcl,
		value,
	})
}

fn std_h265_sub_layer_hrd(
	entries: &[video::H265CpbEntry],
) -> Result<ash::vk::native::StdVideoH265SubLayerHrdParameters> {
	if entries.is_empty() || entries.len() > 32 {
		return Err(Error::data_loss("H.265 HRD CPB count is outside 1..=32"));
	}
	let mut result = ash::vk::native::StdVideoH265SubLayerHrdParameters {
		bit_rate_value_minus1: [0; 32],
		cpb_size_value_minus1: [0; 32],
		cpb_size_du_value_minus1: [0; 32],
		bit_rate_du_value_minus1: [0; 32],
		cbr_flag: 0,
	};
	for (index, entry) in entries.iter().enumerate() {
		result.bit_rate_value_minus1[index] = entry.bit_rate_value_minus_1;
		result.cpb_size_value_minus1[index] = entry.cpb_size_value_minus_1;
		result.cpb_size_du_value_minus1[index] = entry.cpb_size_du_value_minus_1;
		result.bit_rate_du_value_minus1[index] = entry.bit_rate_du_value_minus_1;
		result.cbr_flag |= u32::from(entry.constant_bit_rate) << index;
	}
	Ok(result)
}

fn std_h265_vps(
	vps: &video::H265VideoParameterSet,
	profile: &ash::vk::native::StdVideoH265ProfileTierLevel,
	dpb: &ash::vk::native::StdVideoH265DecPicBufMgr,
	hrd: *const ash::vk::native::StdVideoH265HrdParameters,
) -> Result<ash::vk::native::StdVideoH265VideoParameterSet> {
	let timing = vps.timing.unwrap_or_default();
	Ok(ash::vk::native::StdVideoH265VideoParameterSet {
		flags: ash::vk::native::StdVideoH265VpsFlags {
			_bitfield_align_1: [],
			_bitfield_1: ash::vk::native::StdVideoH265VpsFlags::new_bitfield_1(
				u32::from(vps.temporal_id_nesting),
				u32::from(vps.sub_layer_ordering_info_present),
				u32::from(vps.timing.is_some()),
				u32::from(timing.num_ticks_poc_diff_one_minus_1.is_some()),
			),
			__bindgen_padding_0: [0; 3],
		},
		vps_video_parameter_set_id: u8::try_from(vps.id)
			.map_err(|_| Error::data_loss("H.265 VPS id exceeds StdVideo storage"))?,
		vps_max_sub_layers_minus1: u8::try_from(vps.max_sub_layers_minus_1)
			.map_err(|_| Error::data_loss("H.265 VPS sub-layer count exceeds StdVideo storage"))?,
		reserved1: 0,
		reserved2: 0,
		vps_num_units_in_tick: timing.num_units_in_tick,
		vps_time_scale: timing.time_scale,
		vps_num_ticks_poc_diff_one_minus1: timing.num_ticks_poc_diff_one_minus_1.unwrap_or(0),
		reserved3: 0,
		pDecPicBufMgr: dpb,
		pHrdParameters: hrd,
		pProfileTierLevel: profile,
	})
}

fn std_h265_sps(
	sps: &video::H265SequenceParameterSet,
	profile: &ash::vk::native::StdVideoH265ProfileTierLevel,
	dpb: &ash::vk::native::StdVideoH265DecPicBufMgr,
	scaling: *const ash::vk::native::StdVideoH265ScalingLists,
	short_term: &[ash::vk::native::StdVideoH265ShortTermRefPicSet],
	long_term: Option<&ash::vk::native::StdVideoH265LongTermRefPicsSps>,
	vui: Option<&ash::vk::native::StdVideoH265SequenceParameterSetVui>,
) -> Result<ash::vk::native::StdVideoH265SequenceParameterSet> {
	let pcm = sps.pcm.unwrap_or_default();
	Ok(ash::vk::native::StdVideoH265SequenceParameterSet {
		flags: ash::vk::native::StdVideoH265SpsFlags {
			_bitfield_align_1: [],
			_bitfield_1: ash::vk::native::StdVideoH265SpsFlags::new_bitfield_1(
				u32::from(sps.temporal_id_nesting),
				u32::from(sps.separate_colour_plane),
				u32::from(sps.conformance_window != [0; 4]),
				u32::from(sps.sub_layer_ordering_info_present),
				u32::from(sps.scaling_list_enabled),
				u32::from(sps.scaling_lists.is_some()),
				u32::from(sps.asymmetric_motion_partitions_enabled),
				u32::from(sps.sample_adaptive_offset_enabled),
				u32::from(sps.pcm.is_some()),
				u32::from(pcm.loop_filter_disabled),
				u32::from(!sps.long_term_reference_pictures.is_empty()),
				u32::from(sps.temporal_mvp_enabled),
				u32::from(sps.strong_intra_smoothing_enabled),
				u32::from(sps.vui.is_some()),
				0,
				0,
				0,
				0,
				0,
				0,
				0,
				0,
				0,
				0,
				0,
				0,
				0,
				0,
				0,
				0,
			),
		},
		chroma_format_idc: sps.chroma_format_idc,
		pic_width_in_luma_samples: sps.coded_width,
		pic_height_in_luma_samples: sps.coded_height,
		sps_video_parameter_set_id: h265_u8(sps.video_parameter_set_id, "VPS id")?,
		sps_max_sub_layers_minus1: h265_u8(sps.max_sub_layers_minus_1, "sub-layer count")?,
		sps_seq_parameter_set_id: h265_u8(sps.id, "SPS id")?,
		bit_depth_luma_minus8: h265_u8(sps.bit_depth_luma_minus_8, "luma bit depth")?,
		bit_depth_chroma_minus8: h265_u8(sps.bit_depth_chroma_minus_8, "chroma bit depth")?,
		log2_max_pic_order_cnt_lsb_minus4: h265_u8(
			sps.log2_max_pic_order_count_lsb_minus_4,
			"POC width",
		)?,
		log2_min_luma_coding_block_size_minus3: h265_u8(
			sps.log2_min_luma_coding_block_size_minus_3,
			"minimum coding-block size",
		)?,
		log2_diff_max_min_luma_coding_block_size: h265_u8(
			sps.log2_diff_max_min_luma_coding_block_size,
			"coding-block size difference",
		)?,
		log2_min_luma_transform_block_size_minus2: h265_u8(
			sps.log2_min_luma_transform_block_size_minus_2,
			"minimum transform-block size",
		)?,
		log2_diff_max_min_luma_transform_block_size: h265_u8(
			sps.log2_diff_max_min_luma_transform_block_size,
			"transform-block size difference",
		)?,
		max_transform_hierarchy_depth_inter: h265_u8(
			sps.max_transform_hierarchy_depth_inter,
			"inter transform depth",
		)?,
		max_transform_hierarchy_depth_intra: h265_u8(
			sps.max_transform_hierarchy_depth_intra,
			"intra transform depth",
		)?,
		num_short_term_ref_pic_sets: u8::try_from(short_term.len())
			.map_err(|_| Error::data_loss("H.265 short-term RPS count exceeds StdVideo storage"))?,
		num_long_term_ref_pics_sps: u8::try_from(sps.long_term_reference_pictures.len())
			.map_err(|_| Error::data_loss("H.265 long-term RPS count exceeds StdVideo storage"))?,
		pcm_sample_bit_depth_luma_minus1: pcm.sample_bit_depth_luma_minus_1,
		pcm_sample_bit_depth_chroma_minus1: pcm.sample_bit_depth_chroma_minus_1,
		log2_min_pcm_luma_coding_block_size_minus3: h265_u8(
			pcm.log2_min_luma_coding_block_size_minus_3,
			"minimum PCM block size",
		)?,
		log2_diff_max_min_pcm_luma_coding_block_size: h265_u8(
			pcm.log2_diff_max_min_luma_coding_block_size,
			"PCM block-size difference",
		)?,
		reserved1: 0,
		reserved2: 0,
		palette_max_size: 0,
		delta_palette_max_predictor_size: 0,
		motion_vector_resolution_control_idc: 0,
		sps_num_palette_predictor_initializers_minus1: 0,
		conf_win_left_offset: sps.conformance_window[0],
		conf_win_right_offset: sps.conformance_window[1],
		conf_win_top_offset: sps.conformance_window[2],
		conf_win_bottom_offset: sps.conformance_window[3],
		pProfileTierLevel: profile,
		pDecPicBufMgr: dpb,
		pScalingLists: scaling,
		pShortTermRefPicSet: if short_term.is_empty() {
			std::ptr::null()
		} else {
			short_term.as_ptr()
		},
		pLongTermRefPicsSps: long_term.map_or(std::ptr::null(), |value| value as *const _),
		pSequenceParameterSetVui: vui.map_or(std::ptr::null(), |value| value as *const _),
		pPredictorPaletteEntries: std::ptr::null(),
	})
}

fn std_h265_short_term_reference_set(
	set: &video::H265ShortTermReferencePictureSet,
) -> Result<ash::vk::native::StdVideoH265ShortTermRefPicSet> {
	if set.negative_delta_poc_minus_1.len() > 16 || set.positive_delta_poc_minus_1.len() > 16 {
		return Err(Error::data_loss(
			"H.265 short-term RPS exceeds StdVideo's sixteen-entry arrays",
		));
	}
	let mut delta_poc_s0_minus1 = [0_u16; 16];
	let mut delta_poc_s1_minus1 = [0_u16; 16];
	for (output, input) in delta_poc_s0_minus1
		.iter_mut()
		.zip(&set.negative_delta_poc_minus_1)
	{
		*output = u16::try_from(*input)
			.map_err(|_| Error::data_loss("H.265 negative delta POC exceeds StdVideo storage"))?;
	}
	for (output, input) in delta_poc_s1_minus1
		.iter_mut()
		.zip(&set.positive_delta_poc_minus_1)
	{
		*output = u16::try_from(*input)
			.map_err(|_| Error::data_loss("H.265 positive delta POC exceeds StdVideo storage"))?;
	}
	Ok(ash::vk::native::StdVideoH265ShortTermRefPicSet {
		flags: ash::vk::native::StdVideoH265ShortTermRefPicSetFlags {
			_bitfield_align_1: [],
			_bitfield_1: ash::vk::native::StdVideoH265ShortTermRefPicSetFlags::new_bitfield_1(
				u32::from(set.inter_ref_pic_set_prediction),
				u32::from(set.delta_rps_sign),
			),
			__bindgen_padding_0: [0; 3],
		},
		delta_idx_minus1: set.delta_index_minus_1,
		use_delta_flag: u16::try_from(set.use_delta_mask)
			.map_err(|_| Error::data_loss("H.265 use-delta mask exceeds StdVideo storage"))?,
		abs_delta_rps_minus1: u16::try_from(set.abs_delta_rps_minus_1)
			.map_err(|_| Error::data_loss("H.265 delta RPS exceeds StdVideo storage"))?,
		used_by_curr_pic_flag: u16::try_from(set.used_by_current_mask)
			.map_err(|_| Error::data_loss("H.265 used-by-current mask exceeds StdVideo storage"))?,
		used_by_curr_pic_s0_flag: u16::try_from(set.used_by_current_negative_mask)
			.map_err(|_| Error::data_loss("H.265 negative RPS mask exceeds StdVideo storage"))?,
		used_by_curr_pic_s1_flag: u16::try_from(set.used_by_current_positive_mask)
			.map_err(|_| Error::data_loss("H.265 positive RPS mask exceeds StdVideo storage"))?,
		reserved1: 0,
		reserved2: 0,
		reserved3: 0,
		num_negative_pics: u8::try_from(set.negative_delta_poc_minus_1.len())
			.map_err(|_| Error::data_loss("H.265 negative RPS count exceeds StdVideo storage"))?,
		num_positive_pics: u8::try_from(set.positive_delta_poc_minus_1.len())
			.map_err(|_| Error::data_loss("H.265 positive RPS count exceeds StdVideo storage"))?,
		delta_poc_s0_minus1,
		delta_poc_s1_minus1,
	})
}

fn std_h265_long_term_references(
	values: &[video::H265LongTermReferencePicture],
) -> Result<Option<ash::vk::native::StdVideoH265LongTermRefPicsSps>> {
	if values.is_empty() {
		return Ok(None);
	}
	if values.len() > 32 {
		return Err(Error::data_loss("H.265 long-term RPS count exceeds 32"));
	}
	let mut used_by_curr_pic_lt_sps_flag = 0_u32;
	let mut lt_ref_pic_poc_lsb_sps = [0_u32; 32];
	for (index, value) in values.iter().enumerate() {
		used_by_curr_pic_lt_sps_flag |= u32::from(value.used_by_current) << index;
		lt_ref_pic_poc_lsb_sps[index] = value.picture_order_count_lsb;
	}
	Ok(Some(ash::vk::native::StdVideoH265LongTermRefPicsSps {
		used_by_curr_pic_lt_sps_flag,
		lt_ref_pic_poc_lsb_sps,
	}))
}

fn std_h265_scaling(
	scaling: &video::H265ScalingLists,
) -> ash::vk::native::StdVideoH265ScalingLists {
	ash::vk::native::StdVideoH265ScalingLists {
		ScalingList4x4: scaling.list_4x4,
		ScalingList8x8: scaling.list_8x8,
		ScalingList16x16: scaling.list_16x16,
		ScalingList32x32: scaling.list_32x32,
		ScalingListDCCoef16x16: scaling.dc_16x16,
		ScalingListDCCoef32x32: scaling.dc_32x32,
	}
}

fn std_h265_vui(
	vui: &video::H265VuiParameters,
	hrd: *const ash::vk::native::StdVideoH265HrdParameters,
) -> Result<ash::vk::native::StdVideoH265SequenceParameterSetVui> {
	let aspect = vui.aspect_ratio.unwrap_or_default();
	let signal = vui.video_signal.unwrap_or_default();
	let colour = signal.colour_description.unwrap_or_default();
	let chroma = vui.chroma_location.unwrap_or_default();
	let display = vui.default_display_window.unwrap_or([0; 4]);
	let timing = vui.timing.unwrap_or_default();
	let restriction = vui.bitstream_restriction.unwrap_or_default();
	Ok(ash::vk::native::StdVideoH265SequenceParameterSetVui {
		flags: ash::vk::native::StdVideoH265SpsVuiFlags {
			_bitfield_align_1: [],
			_bitfield_1: ash::vk::native::StdVideoH265SpsVuiFlags::new_bitfield_1(
				u32::from(vui.aspect_ratio.is_some()),
				u32::from(vui.overscan_appropriate.is_some()),
				u32::from(vui.overscan_appropriate.unwrap_or(false)),
				u32::from(vui.video_signal.is_some()),
				u32::from(signal.full_range),
				u32::from(signal.colour_description.is_some()),
				u32::from(vui.chroma_location.is_some()),
				u32::from(vui.neutral_chroma_indication),
				u32::from(vui.field_sequence),
				u32::from(vui.frame_field_info_present),
				u32::from(vui.default_display_window.is_some()),
				u32::from(vui.timing.is_some()),
				u32::from(timing.num_ticks_poc_diff_one_minus_1.is_some()),
				u32::from(vui.hrd.is_some()),
				u32::from(vui.bitstream_restriction.is_some()),
				u32::from(restriction.tiles_fixed_structure),
				u32::from(restriction.motion_vectors_over_picture_boundaries),
				u32::from(restriction.restricted_reference_picture_lists),
			),
			__bindgen_padding_0: 0,
		},
		aspect_ratio_idc: u32::from(aspect.idc),
		sar_width: aspect.sar_width,
		sar_height: aspect.sar_height,
		video_format: signal.video_format,
		colour_primaries: colour.colour_primaries,
		transfer_characteristics: colour.transfer_characteristics,
		matrix_coeffs: colour.matrix_coefficients,
		chroma_sample_loc_type_top_field: h265_u8(chroma.top_field, "top chroma location")?,
		chroma_sample_loc_type_bottom_field: h265_u8(chroma.bottom_field, "bottom chroma location")?,
		reserved1: 0,
		reserved2: 0,
		def_disp_win_left_offset: u16::try_from(display[0])
			.map_err(|_| Error::data_loss("H.265 display-window left offset exceeds StdVideo storage"))?,
		def_disp_win_right_offset: u16::try_from(display[1]).map_err(|_| {
			Error::data_loss("H.265 display-window right offset exceeds StdVideo storage")
		})?,
		def_disp_win_top_offset: u16::try_from(display[2])
			.map_err(|_| Error::data_loss("H.265 display-window top offset exceeds StdVideo storage"))?,
		def_disp_win_bottom_offset: u16::try_from(display[3]).map_err(|_| {
			Error::data_loss("H.265 display-window bottom offset exceeds StdVideo storage")
		})?,
		vui_num_units_in_tick: timing.num_units_in_tick,
		vui_time_scale: timing.time_scale,
		vui_num_ticks_poc_diff_one_minus1: timing.num_ticks_poc_diff_one_minus_1.unwrap_or(0),
		min_spatial_segmentation_idc: u16::try_from(restriction.min_spatial_segmentation_idc).map_err(
			|_| Error::data_loss("H.265 minimum spatial segmentation exceeds StdVideo storage"),
		)?,
		reserved3: 0,
		max_bytes_per_pic_denom: h265_u8(
			restriction.max_bytes_per_picture_denom,
			"maximum bytes-per-picture denominator",
		)?,
		max_bits_per_min_cu_denom: h265_u8(
			restriction.max_bits_per_min_coding_unit_denom,
			"maximum bits-per-min-CU denominator",
		)?,
		log2_max_mv_length_horizontal: h265_u8(
			restriction.log2_max_motion_vector_length_horizontal,
			"horizontal motion-vector length",
		)?,
		log2_max_mv_length_vertical: h265_u8(
			restriction.log2_max_motion_vector_length_vertical,
			"vertical motion-vector length",
		)?,
		pHrdParameters: hrd,
	})
}

fn std_h265_pps(
	pps: &video::H265PictureParameterSet,
	video_parameter_set_id: u32,
	scaling: *const ash::vk::native::StdVideoH265ScalingLists,
) -> Result<ash::vk::native::StdVideoH265PictureParameterSet> {
	let mut column_width_minus1 = [0_u16; 19];
	let mut row_height_minus1 = [0_u16; 21];
	for (output, input) in column_width_minus1
		.iter_mut()
		.zip(&pps.column_width_minus_1)
	{
		*output = u16::try_from(*input)
			.map_err(|_| Error::data_loss("H.265 tile-column width exceeds StdVideo storage"))?;
	}
	for (output, input) in row_height_minus1.iter_mut().zip(&pps.row_height_minus_1) {
		*output = u16::try_from(*input)
			.map_err(|_| Error::data_loss("H.265 tile-row height exceeds StdVideo storage"))?;
	}
	Ok(ash::vk::native::StdVideoH265PictureParameterSet {
		flags: ash::vk::native::StdVideoH265PpsFlags {
			_bitfield_align_1: [],
			_bitfield_1: ash::vk::native::StdVideoH265PpsFlags::new_bitfield_1(
				u32::from(pps.dependent_slice_segments_enabled),
				u32::from(pps.output_flag_present),
				u32::from(pps.sign_data_hiding_enabled),
				u32::from(pps.cabac_init_present),
				u32::from(pps.constrained_intra_pred),
				u32::from(pps.transform_skip_enabled),
				u32::from(pps.cu_qp_delta_enabled),
				u32::from(pps.slice_chroma_qp_offsets_present),
				u32::from(pps.weighted_pred),
				u32::from(pps.weighted_bipred),
				u32::from(pps.transquant_bypass_enabled),
				u32::from(pps.tiles_enabled),
				u32::from(pps.entropy_coding_sync_enabled),
				u32::from(pps.uniform_spacing),
				u32::from(pps.loop_filter_across_tiles_enabled),
				u32::from(pps.loop_filter_across_slices_enabled),
				u32::from(pps.deblocking_filter_control_present),
				u32::from(pps.deblocking_filter_override_enabled),
				u32::from(pps.deblocking_filter_disabled),
				u32::from(pps.scaling_lists.is_some()),
				u32::from(pps.lists_modification_present),
				u32::from(pps.slice_segment_header_extension_present),
				u32::from(pps.extension_present),
				0,
				0,
				0,
				0,
				0,
				0,
				0,
				0,
			),
		},
		pps_pic_parameter_set_id: h265_u8(pps.id, "PPS id")?,
		pps_seq_parameter_set_id: h265_u8(pps.sequence_parameter_set_id, "SPS id")?,
		sps_video_parameter_set_id: h265_u8(video_parameter_set_id, "VPS id")?,
		num_extra_slice_header_bits: h265_u8(pps.num_extra_slice_header_bits, "extra slice bits")?,
		num_ref_idx_l0_default_active_minus1: h265_u8(
			pps.num_ref_idx_l0_default_active_minus_1,
			"default L0 reference count",
		)?,
		num_ref_idx_l1_default_active_minus1: h265_u8(
			pps.num_ref_idx_l1_default_active_minus_1,
			"default L1 reference count",
		)?,
		init_qp_minus26: i8::try_from(pps.init_qp_minus_26)
			.map_err(|_| Error::data_loss("H.265 initial QP exceeds StdVideo storage"))?,
		diff_cu_qp_delta_depth: h265_u8(pps.diff_cu_qp_delta_depth, "CU QP delta depth")?,
		pps_cb_qp_offset: i8::try_from(pps.cb_qp_offset)
			.map_err(|_| Error::data_loss("H.265 Cb QP offset exceeds StdVideo storage"))?,
		pps_cr_qp_offset: i8::try_from(pps.cr_qp_offset)
			.map_err(|_| Error::data_loss("H.265 Cr QP offset exceeds StdVideo storage"))?,
		pps_beta_offset_div2: i8::try_from(pps.beta_offset_div_2)
			.map_err(|_| Error::data_loss("H.265 beta offset exceeds StdVideo storage"))?,
		pps_tc_offset_div2: i8::try_from(pps.tc_offset_div_2)
			.map_err(|_| Error::data_loss("H.265 tc offset exceeds StdVideo storage"))?,
		log2_parallel_merge_level_minus2: h265_u8(
			pps.log2_parallel_merge_level_minus_2,
			"parallel merge level",
		)?,
		log2_max_transform_skip_block_size_minus2: 0,
		diff_cu_chroma_qp_offset_depth: 0,
		chroma_qp_offset_list_len_minus1: 0,
		cb_qp_offset_list: [0; 6],
		cr_qp_offset_list: [0; 6],
		log2_sao_offset_scale_luma: 0,
		log2_sao_offset_scale_chroma: 0,
		pps_act_y_qp_offset_plus5: 0,
		pps_act_cb_qp_offset_plus5: 0,
		pps_act_cr_qp_offset_plus3: 0,
		pps_num_palette_predictor_initializers: 0,
		luma_bit_depth_entry_minus8: 0,
		chroma_bit_depth_entry_minus8: 0,
		num_tile_columns_minus1: h265_u8(pps.num_tile_columns_minus_1, "tile-column count")?,
		num_tile_rows_minus1: h265_u8(pps.num_tile_rows_minus_1, "tile-row count")?,
		reserved1: 0,
		reserved2: 0,
		column_width_minus1,
		row_height_minus1,
		reserved3: 0,
		pScalingLists: scaling,
		pPredictorPaletteEntries: std::ptr::null(),
	})
}

fn h265_u8(value: u32, field: &str) -> Result<u8> {
	u8::try_from(value)
		.map_err(|_| Error::data_loss(format!("H.265 {field} exceeds StdVideo storage")))
}

fn std_h264_sps(
	sps: &video::H264SequenceParameterSet,
	std_scaling: *const ash::vk::native::StdVideoH264ScalingLists,
	std_vui: *const ash::vk::native::StdVideoH264SequenceParameterSetVui,
) -> Result<ash::vk::native::StdVideoH264SequenceParameterSet> {
	let flags = ash::vk::native::StdVideoH264SpsFlags {
		_bitfield_align_1: [],
		_bitfield_1: ash::vk::native::StdVideoH264SpsFlags::new_bitfield_1(
			u32::from(sps.constraint_flags & 0x80 != 0),
			u32::from(sps.constraint_flags & 0x40 != 0),
			u32::from(sps.constraint_flags & 0x20 != 0),
			u32::from(sps.constraint_flags & 0x10 != 0),
			u32::from(sps.constraint_flags & 0x08 != 0),
			u32::from(sps.constraint_flags & 0x04 != 0),
			u32::from(sps.direct_8x8_inference),
			u32::from(sps.mb_adaptive_frame_field),
			u32::from(sps.frame_mbs_only),
			u32::from(sps.delta_pic_order_always_zero),
			u32::from(sps.separate_colour_plane),
			u32::from(sps.gaps_in_frame_num_value_allowed),
			u32::from(sps.qpprime_y_zero_transform_bypass),
			u32::from(
				sps.frame_crop_left_offset != 0
					|| sps.frame_crop_right_offset != 0
					|| sps.frame_crop_top_offset != 0
					|| sps.frame_crop_bottom_offset != 0,
			),
			u32::from(sps.scaling_lists.is_some()),
			u32::from(sps.vui.is_some()),
		),
		__bindgen_padding_0: 0,
	};
	Ok(ash::vk::native::StdVideoH264SequenceParameterSet {
		flags,
		profile_idc: sps.profile_idc,
		level_idc: std_h264_level(sps.level_idc)?,
		chroma_format_idc: sps.chroma_format_idc,
		seq_parameter_set_id: u8::try_from(sps.id)
			.map_err(|_| Error::data_loss("H.264 SPS id exceeds StdVideo storage"))?,
		bit_depth_luma_minus8: u8::try_from(sps.bit_depth_luma_minus_8)
			.map_err(|_| Error::data_loss("H.264 luma bit depth exceeds StdVideo storage"))?,
		bit_depth_chroma_minus8: u8::try_from(sps.bit_depth_chroma_minus_8)
			.map_err(|_| Error::data_loss("H.264 chroma bit depth exceeds StdVideo storage"))?,
		log2_max_frame_num_minus4: u8::try_from(sps.log2_max_frame_num_minus_4)
			.map_err(|_| Error::data_loss("H.264 frame-number width exceeds StdVideo storage"))?,
		pic_order_cnt_type: sps.pic_order_count_type,
		offset_for_non_ref_pic: sps.offset_for_non_ref_pic,
		offset_for_top_to_bottom_field: sps.offset_for_top_to_bottom_field,
		log2_max_pic_order_cnt_lsb_minus4: u8::try_from(sps.log2_max_pic_order_count_lsb_minus_4)
			.map_err(|_| Error::data_loss("H.264 POC width exceeds StdVideo storage"))?,
		num_ref_frames_in_pic_order_cnt_cycle: u8::try_from(sps.offset_for_ref_frame.len())
			.map_err(|_| Error::data_loss("H.264 POC cycle exceeds StdVideo storage"))?,
		max_num_ref_frames: u8::try_from(sps.max_num_ref_frames)
			.map_err(|_| Error::data_loss("H.264 reference count exceeds StdVideo storage"))?,
		reserved1: 0,
		pic_width_in_mbs_minus1: sps
			.width_in_macroblocks
			.checked_sub(1)
			.ok_or_else(|| Error::data_loss("H.264 SPS has zero macroblock width"))?,
		pic_height_in_map_units_minus1: sps
			.height_in_map_units
			.checked_sub(1)
			.ok_or_else(|| Error::data_loss("H.264 SPS has zero map-unit height"))?,
		frame_crop_left_offset: sps.frame_crop_left_offset,
		frame_crop_right_offset: sps.frame_crop_right_offset,
		frame_crop_top_offset: sps.frame_crop_top_offset,
		frame_crop_bottom_offset: sps.frame_crop_bottom_offset,
		reserved2: 0,
		pOffsetForRefFrame: if sps.offset_for_ref_frame.is_empty() {
			std::ptr::null()
		} else {
			sps.offset_for_ref_frame.as_ptr()
		},
		pScalingLists: std_scaling,
		pSequenceParameterSetVui: std_vui,
	})
}

fn std_h264_hrd(
	vui: &video::H264VuiParameters,
) -> Result<Option<ash::vk::native::StdVideoH264HrdParameters>> {
	let hrd = match (&vui.nal_hrd, &vui.vcl_hrd) {
		(Some(nal), Some(vcl)) if nal != vcl => {
			return Err(Error::missing_capability(
				"distinct H.264 NAL and VCL HRD parameters cannot share one StdVideo table",
			));
		}
		(Some(hrd), _) | (_, Some(hrd)) => hrd,
		(None, None) => return Ok(None),
	};
	let cpb_count_minus_1 = hrd
		.entries
		.len()
		.checked_sub(1)
		.ok_or_else(|| Error::data_loss("H.264 HRD has no CPB entries"))?;
	let mut bit_rate_value_minus1 = [0_u32; 32];
	let mut cpb_size_value_minus1 = [0_u32; 32];
	let mut cbr_flag = [0_u8; 32];
	for (index, entry) in hrd.entries.iter().enumerate() {
		let Some(bit_rate) = bit_rate_value_minus1.get_mut(index) else {
			return Err(Error::data_loss("H.264 HRD CPB count exceeds 32"));
		};
		*bit_rate = entry.bit_rate_value_minus_1;
		cpb_size_value_minus1[index] = entry.cpb_size_value_minus_1;
		cbr_flag[index] = u8::from(entry.constant_bit_rate);
	}
	Ok(Some(ash::vk::native::StdVideoH264HrdParameters {
		cpb_cnt_minus1: u8::try_from(cpb_count_minus_1)
			.map_err(|_| Error::data_loss("H.264 HRD CPB count exceeds StdVideo storage"))?,
		bit_rate_scale: hrd.bit_rate_scale,
		cpb_size_scale: hrd.cpb_size_scale,
		reserved1: 0,
		bit_rate_value_minus1,
		cpb_size_value_minus1,
		cbr_flag,
		initial_cpb_removal_delay_length_minus1: u32::from(
			hrd.initial_cpb_removal_delay_length_minus_1,
		),
		cpb_removal_delay_length_minus1: u32::from(hrd.cpb_removal_delay_length_minus_1),
		dpb_output_delay_length_minus1: u32::from(hrd.dpb_output_delay_length_minus_1),
		time_offset_length: u32::from(hrd.time_offset_length),
	}))
}

fn std_h264_vui(
	vui: &video::H264VuiParameters,
	std_hrd: *const ash::vk::native::StdVideoH264HrdParameters,
) -> Result<ash::vk::native::StdVideoH264SequenceParameterSetVui> {
	let aspect = vui.aspect_ratio.unwrap_or(video::H264AspectRatio {
		idc: 0,
		sar_width: 0,
		sar_height: 0,
	});
	let signal = vui.video_signal.unwrap_or(video::H264VideoSignal {
		video_format: 0,
		full_range: false,
		colour_description: None,
	});
	let colour = signal
		.colour_description
		.unwrap_or(video::H264ColourDescription {
			colour_primaries: 0,
			transfer_characteristics: 0,
			matrix_coefficients: 0,
		});
	let timing = vui.timing.unwrap_or(video::H264TimingInfo {
		num_units_in_tick: 0,
		time_scale: 0,
		fixed_frame_rate: false,
	});
	let chroma = vui.chroma_location.unwrap_or(video::H264ChromaLocation {
		top_field: 0,
		bottom_field: 0,
	});
	let restriction = vui
		.bitstream_restriction
		.unwrap_or(video::H264BitstreamRestriction {
			motion_vectors_over_picture_boundaries: false,
			max_bytes_per_picture_denom: 0,
			max_bits_per_macroblock_denom: 0,
			log2_max_motion_vector_length_horizontal: 0,
			log2_max_motion_vector_length_vertical: 0,
			max_num_reorder_frames: 0,
			max_dec_frame_buffering: 0,
		});
	let flags = ash::vk::native::StdVideoH264SpsVuiFlags {
		_bitfield_align_1: [],
		_bitfield_1: ash::vk::native::StdVideoH264SpsVuiFlags::new_bitfield_1(
			u32::from(vui.aspect_ratio.is_some()),
			u32::from(vui.overscan_appropriate.is_some()),
			u32::from(vui.overscan_appropriate.unwrap_or(false)),
			u32::from(vui.video_signal.is_some()),
			u32::from(signal.full_range),
			u32::from(signal.colour_description.is_some()),
			u32::from(vui.chroma_location.is_some()),
			u32::from(vui.timing.is_some()),
			u32::from(timing.fixed_frame_rate),
			u32::from(vui.bitstream_restriction.is_some()),
			u32::from(vui.nal_hrd.is_some()),
			u32::from(vui.vcl_hrd.is_some()),
		),
		__bindgen_padding_0: 0,
	};
	Ok(ash::vk::native::StdVideoH264SequenceParameterSetVui {
		flags,
		aspect_ratio_idc: u32::from(aspect.idc),
		sar_width: aspect.sar_width,
		sar_height: aspect.sar_height,
		video_format: signal.video_format,
		colour_primaries: colour.colour_primaries,
		transfer_characteristics: colour.transfer_characteristics,
		matrix_coefficients: colour.matrix_coefficients,
		num_units_in_tick: timing.num_units_in_tick,
		time_scale: timing.time_scale,
		max_num_reorder_frames: u8::try_from(restriction.max_num_reorder_frames)
			.map_err(|_| Error::data_loss("H.264 reorder count exceeds StdVideo storage"))?,
		max_dec_frame_buffering: u8::try_from(restriction.max_dec_frame_buffering)
			.map_err(|_| Error::data_loss("H.264 DPB limit exceeds StdVideo storage"))?,
		chroma_sample_loc_type_top_field: u8::try_from(chroma.top_field)
			.map_err(|_| Error::data_loss("H.264 top chroma location exceeds StdVideo storage"))?,
		chroma_sample_loc_type_bottom_field: u8::try_from(chroma.bottom_field)
			.map_err(|_| Error::data_loss("H.264 bottom chroma location exceeds StdVideo storage"))?,
		reserved1: 0,
		pHrdParameters: std_hrd,
	})
}

fn std_h264_pps(
	pps: &video::H264PictureParameterSet,
	std_scaling: *const ash::vk::native::StdVideoH264ScalingLists,
) -> Result<ash::vk::native::StdVideoH264PictureParameterSet> {
	let flags = ash::vk::native::StdVideoH264PpsFlags {
		_bitfield_align_1: [],
		_bitfield_1: ash::vk::native::StdVideoH264PpsFlags::new_bitfield_1(
			u32::from(pps.transform_8x8_mode),
			u32::from(pps.redundant_pic_count_present),
			u32::from(pps.constrained_intra_pred),
			u32::from(pps.deblocking_filter_control_present),
			u32::from(pps.weighted_pred),
			u32::from(pps.bottom_field_pic_order_in_frame_present),
			u32::from(pps.entropy_coding_mode),
			u32::from(pps.scaling_lists.is_some()),
		),
		__bindgen_padding_0: [0; 3],
	};
	Ok(ash::vk::native::StdVideoH264PictureParameterSet {
		flags,
		seq_parameter_set_id: u8::try_from(pps.sequence_parameter_set_id)
			.map_err(|_| Error::data_loss("H.264 PPS SPS id exceeds StdVideo storage"))?,
		pic_parameter_set_id: u8::try_from(pps.id)
			.map_err(|_| Error::data_loss("H.264 PPS id exceeds StdVideo storage"))?,
		num_ref_idx_l0_default_active_minus1: u8::try_from(pps.num_ref_idx_l0_default_active_minus_1)
			.map_err(|_| {
			Error::data_loss("H.264 L0 reference count exceeds StdVideo storage")
		})?,
		num_ref_idx_l1_default_active_minus1: u8::try_from(pps.num_ref_idx_l1_default_active_minus_1)
			.map_err(|_| {
			Error::data_loss("H.264 L1 reference count exceeds StdVideo storage")
		})?,
		weighted_bipred_idc: pps.weighted_bipred_idc,
		pic_init_qp_minus26: i8::try_from(pps.pic_init_qp_minus_26)
			.map_err(|_| Error::data_loss("H.264 initial QP exceeds StdVideo storage"))?,
		pic_init_qs_minus26: i8::try_from(pps.pic_init_qs_minus_26)
			.map_err(|_| Error::data_loss("H.264 initial QS exceeds StdVideo storage"))?,
		chroma_qp_index_offset: i8::try_from(pps.chroma_qp_index_offset)
			.map_err(|_| Error::data_loss("H.264 chroma QP offset exceeds StdVideo storage"))?,
		second_chroma_qp_index_offset: i8::try_from(pps.second_chroma_qp_index_offset)
			.map_err(|_| Error::data_loss("H.264 second chroma QP offset exceeds StdVideo storage"))?,
		pScalingLists: std_scaling,
	})
}

fn std_h264_scaling(
	scaling: &video::H264ScalingLists,
) -> ash::vk::native::StdVideoH264ScalingLists {
	ash::vk::native::StdVideoH264ScalingLists {
		scaling_list_present_mask: scaling.present_mask,
		use_default_scaling_matrix_mask: scaling.use_default_mask,
		ScalingList4x4: scaling.list_4x4,
		ScalingList8x8: scaling.list_8x8,
	}
}

fn std_h264_level(level_idc: u32) -> Result<ash::vk::native::StdVideoH264LevelIdc> {
	match level_idc {
		10 => Ok(0),
		11 => Ok(1),
		12 => Ok(2),
		13 => Ok(3),
		20 => Ok(4),
		21 => Ok(5),
		22 => Ok(6),
		30 => Ok(7),
		31 => Ok(8),
		32 => Ok(9),
		40 => Ok(10),
		41 => Ok(11),
		42 => Ok(12),
		50 => Ok(13),
		51 => Ok(14),
		52 => Ok(15),
		60 => Ok(16),
		61 => Ok(17),
		62 => Ok(18),
		_ => Err(Error::missing_capability(format!(
			"H.264 level_idc {level_idc} has no Khronos StdVideo mapping"
		))),
	}
}

fn with_decode_profile<T>(
	profile: video::VideoDecodeProfile,
	query: impl FnOnce(&ash::vk::VideoProfileInfoKHR<'_>) -> Result<T>,
) -> Result<T> {
	match profile {
		video::VideoDecodeProfile::H264 {
			profile,
			picture_layout,
			chroma_subsampling,
			luma_bit_depth,
			chroma_bit_depth,
		} => {
			let mut codec = ash::vk::VideoDecodeH264ProfileInfoKHR::default()
				.std_profile_idc(h264_profile(profile))
				.picture_layout(h264_picture_layout(picture_layout));
			let profile = common_profile(
				ash::vk::VideoCodecOperationFlagsKHR::DECODE_H264,
				chroma_subsampling,
				luma_bit_depth,
				chroma_bit_depth,
			)
			.push_next(&mut codec);
			query(&profile)
		}
		video::VideoDecodeProfile::H265 {
			profile,
			chroma_subsampling,
			luma_bit_depth,
			chroma_bit_depth,
		} => {
			let mut codec =
				ash::vk::VideoDecodeH265ProfileInfoKHR::default().std_profile_idc(h265_profile(profile));
			let profile = common_profile(
				ash::vk::VideoCodecOperationFlagsKHR::DECODE_H265,
				chroma_subsampling,
				luma_bit_depth,
				chroma_bit_depth,
			)
			.push_next(&mut codec);
			query(&profile)
		}
		video::VideoDecodeProfile::Av1 {
			profile,
			film_grain_support,
			chroma_subsampling,
			luma_bit_depth,
			chroma_bit_depth,
		} => {
			let mut codec = ash::vk::VideoDecodeAV1ProfileInfoKHR::default()
				.std_profile(av1_profile(profile))
				.film_grain_support(film_grain_support);
			let profile = common_profile(
				ash::vk::VideoCodecOperationFlagsKHR::DECODE_AV1,
				chroma_subsampling,
				luma_bit_depth,
				chroma_bit_depth,
			)
			.push_next(&mut codec);
			query(&profile)
		}
		video::VideoDecodeProfile::Vp9 {
			profile,
			chroma_subsampling,
			luma_bit_depth,
			chroma_bit_depth,
		} => {
			let mut codec =
				ash_vp9::vk::VideoDecodeVP9ProfileInfoKHR::default().std_profile(vp9_profile(profile));
			let mut profile = common_profile(
				vp9_operation(),
				chroma_subsampling,
				luma_bit_depth,
				chroma_bit_depth,
			);
			profile.p_next = std::ptr::from_mut(&mut codec).cast();
			query(&profile)
		}
	}
}

fn with_encode_profile<T>(
	profile: video::VideoEncodeProfile,
	query: impl FnOnce(&ash::vk::VideoProfileInfoKHR<'_>) -> Result<T>,
) -> Result<T> {
	match profile {
		video::VideoEncodeProfile::H264 {
			profile,
			chroma_subsampling,
			luma_bit_depth,
			chroma_bit_depth,
		} => {
			let mut codec =
				ash::vk::VideoEncodeH264ProfileInfoKHR::default().std_profile_idc(h264_profile(profile));
			let profile = common_profile(
				ash::vk::VideoCodecOperationFlagsKHR::ENCODE_H264,
				chroma_subsampling,
				luma_bit_depth,
				chroma_bit_depth,
			)
			.push_next(&mut codec);
			query(&profile)
		}
		video::VideoEncodeProfile::H265 {
			profile,
			chroma_subsampling,
			luma_bit_depth,
			chroma_bit_depth,
		} => {
			let mut codec =
				ash::vk::VideoEncodeH265ProfileInfoKHR::default().std_profile_idc(h265_profile(profile));
			let profile = common_profile(
				ash::vk::VideoCodecOperationFlagsKHR::ENCODE_H265,
				chroma_subsampling,
				luma_bit_depth,
				chroma_bit_depth,
			)
			.push_next(&mut codec);
			query(&profile)
		}
	}
}

fn query_formats(
	loader: &ash::khr::video_queue::Instance,
	physical: ash::vk::PhysicalDevice,
	profile: &ash::vk::VideoProfileInfoKHR<'_>,
	usage: ash::vk::ImageUsageFlags,
) -> Result<Vec<ash::vk::VideoFormatPropertiesKHR<'static>>> {
	let profiles = [*profile];
	let mut profile_list = ash::vk::VideoProfileListInfoKHR::default().profiles(&profiles);
	let format_info = ash::vk::PhysicalDeviceVideoFormatInfoKHR::default()
		.image_usage(usage)
		.push_next(&mut profile_list);
	let mut count = 0_u32;
	// SAFETY: every pointer in `format_info` remains live, and a null output
	// pointer requests only the required element count.
	let result = unsafe {
		(loader.fp().get_physical_device_video_format_properties_khr)(
			physical,
			&format_info,
			&mut count,
			std::ptr::null_mut(),
		)
	};
	if result != ash::vk::Result::SUCCESS {
		return Err(Error::backend_failure(
			"Vulkan",
			"video-format count query",
			result,
		));
	}
	if count > MAX_VIDEO_FORMATS {
		return Err(Error::resource_exhausted(format!(
			"Vulkan reported {count} video formats, above OA's {MAX_VIDEO_FORMATS} safety bound"
		)));
	}
	let mut formats = vec![ash::vk::VideoFormatPropertiesKHR::default(); count as usize];
	if count == 0 {
		return Ok(formats);
	}
	// SAFETY: `formats` contains `count` initialized output structures with valid
	// sType values; Vulkan writes at most the count passed by mutable pointer.
	let result = unsafe {
		(loader.fp().get_physical_device_video_format_properties_khr)(
			physical,
			&format_info,
			&mut count,
			formats.as_mut_ptr(),
		)
	};
	if result != ash::vk::Result::SUCCESS {
		return Err(Error::backend_failure(
			"Vulkan",
			"video-format enumeration",
			result,
		));
	}
	formats.truncate(count as usize);
	Ok(formats)
}

fn convert_formats(
	formats: &[ash::vk::VideoFormatPropertiesKHR<'_>],
) -> (Vec<video::VideoImageFormat>, usize) {
	let mut converted = Vec::new();
	let mut unrecognized = 0;
	for format in formats {
		let Some(pixel_format) = pixel_format(format.format) else {
			unrecognized += 1;
			continue;
		};
		converted.push(video::VideoImageFormat {
			pixel_format,
			optimal_tiling: format.image_tiling == ash::vk::ImageTiling::OPTIMAL,
			sampled: format
				.image_usage_flags
				.contains(ash::vk::ImageUsageFlags::SAMPLED),
			storage: format
				.image_usage_flags
				.contains(ash::vk::ImageUsageFlags::STORAGE),
			transfer_source: format
				.image_usage_flags
				.contains(ash::vk::ImageUsageFlags::TRANSFER_SRC),
			transfer_destination: format
				.image_usage_flags
				.contains(ash::vk::ImageUsageFlags::TRANSFER_DST),
		});
	}
	converted.sort_by_key(|format| format.pixel_format);
	converted.dedup();
	(converted, unrecognized)
}

fn pixel_format(format: ash::vk::Format) -> Option<video::VideoPixelFormat> {
	match format {
		ash::vk::Format::G8_B8_R8_3PLANE_420_UNORM => Some(video::VideoPixelFormat::Yuv420Planar8),
		ash::vk::Format::G8_B8R8_2PLANE_420_UNORM => Some(video::VideoPixelFormat::Nv12),
		ash::vk::Format::G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16 => {
			Some(video::VideoPixelFormat::Yuv420Planar10)
		}
		ash::vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16 => {
			Some(video::VideoPixelFormat::P010)
		}
		ash::vk::Format::G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16 => {
			Some(video::VideoPixelFormat::P012)
		}
		_ => None,
	}
}

fn ensure_extension_advertised(
	physical: &PhysicalDevice,
	profile: video::VideoDecodeProfile,
) -> Result<()> {
	let advertised = match profile {
		video::VideoDecodeProfile::H264 { .. } => physical.video.h264_decode,
		video::VideoDecodeProfile::H265 { .. } => physical.video.h265_decode,
		video::VideoDecodeProfile::Av1 { .. } => physical.video.av1_decode,
		video::VideoDecodeProfile::Vp9 { .. } => physical.video.vp9_decode,
	};
	if physical.video.decode_queue_family.is_none() || !advertised {
		return Err(Error::missing_capability(format!(
			"the selected device does not advertise the requested {profile:?} decoder extension"
		)));
	}
	Ok(())
}

fn ensure_encode_extension_advertised(
	physical: &PhysicalDevice,
	profile: video::VideoEncodeProfile,
) -> Result<()> {
	let advertised = match profile {
		video::VideoEncodeProfile::H264 { .. } => physical.video.h264_encode,
		video::VideoEncodeProfile::H265 { .. } => physical.video.h265_encode,
	};
	if physical.video.encode_queue_family.is_none() || !advertised {
		return Err(Error::missing_capability(format!(
			"the selected device does not advertise the requested {profile:?} encoder extension"
		)));
	}
	Ok(())
}

fn common_profile<'a>(
	operation: ash::vk::VideoCodecOperationFlagsKHR,
	chroma: video::VideoChromaSubsampling,
	luma_depth: video::VideoComponentBitDepth,
	chroma_depth: video::VideoComponentBitDepth,
) -> ash::vk::VideoProfileInfoKHR<'a> {
	ash::vk::VideoProfileInfoKHR::default()
		.video_codec_operation(operation)
		.chroma_subsampling(chroma_subsampling(chroma))
		.luma_bit_depth(bit_depth(luma_depth))
		.chroma_bit_depth(bit_depth(chroma_depth))
}

fn query(
	loader: &ash::khr::video_queue::Instance,
	physical: ash::vk::PhysicalDevice,
	profile: &ash::vk::VideoProfileInfoKHR<'_>,
	capabilities: &mut ash::vk::VideoCapabilitiesKHR<'_>,
) -> Result<()> {
	// SAFETY: the loader was created from the live instance that owns `physical`;
	// both root structures and their required decode/codec-specific pNext chains
	// remain live and writable for the duration of the call.
	let result = unsafe {
		(loader.fp().get_physical_device_video_capabilities_khr)(physical, profile, capabilities)
	};
	match result {
		ash::vk::Result::SUCCESS => Ok(()),
		ash::vk::Result::ERROR_VIDEO_PROFILE_OPERATION_NOT_SUPPORTED_KHR
		| ash::vk::Result::ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED_KHR
		| ash::vk::Result::ERROR_VIDEO_PROFILE_CODEC_NOT_SUPPORTED_KHR => Err(Error::missing_capability(
			format!("Vulkan rejected the requested video profile: {result}"),
		)),
		_ => Err(Error::backend_failure(
			"Vulkan",
			"video-profile capability query",
			result,
		)),
	}
}

fn convert_capabilities(
	profile: video::VideoDecodeProfile,
	capabilities: CommonCapabilities,
	decode: &ash::vk::VideoDecodeCapabilitiesKHR<'_>,
	level: video::VideoDecodeLevel,
	field_offset_granularity: Option<(i32, i32)>,
) -> video::VideoDecodeCapabilities {
	video::VideoDecodeCapabilities {
		profile,
		min_coded_extent: capabilities.min_coded_extent,
		max_coded_extent: capabilities.max_coded_extent,
		picture_access_granularity: capabilities.picture_access_granularity,
		min_bitstream_offset_alignment: capabilities.min_bitstream_buffer_offset_alignment,
		min_bitstream_size_alignment: capabilities.min_bitstream_buffer_size_alignment,
		max_dpb_slots: capabilities.max_dpb_slots,
		max_active_reference_pictures: capabilities.max_active_reference_pictures,
		level,
		field_offset_granularity,
		dpb_and_output_coincide: decode
			.flags
			.contains(ash::vk::VideoDecodeCapabilityFlagsKHR::DPB_AND_OUTPUT_COINCIDE),
		dpb_and_output_distinct: decode
			.flags
			.contains(ash::vk::VideoDecodeCapabilityFlagsKHR::DPB_AND_OUTPUT_DISTINCT),
		protected_content: capabilities
			.flags
			.contains(ash::vk::VideoCapabilityFlagsKHR::PROTECTED_CONTENT),
		separate_reference_images: capabilities
			.flags
			.contains(ash::vk::VideoCapabilityFlagsKHR::SEPARATE_REFERENCE_IMAGES),
	}
}

fn convert_encode_capabilities(
	profile: video::VideoEncodeProfile,
	capabilities: CommonCapabilities,
	encode: &ash::vk::VideoEncodeCapabilitiesKHR<'_>,
	codec: video::VideoEncodeCodecCapabilities,
) -> video::VideoEncodeCapabilities {
	video::VideoEncodeCapabilities {
		profile,
		min_coded_extent: capabilities.min_coded_extent,
		max_coded_extent: capabilities.max_coded_extent,
		picture_access_granularity: capabilities.picture_access_granularity,
		input_picture_granularity: extent(encode.encode_input_picture_granularity),
		min_bitstream_offset_alignment: capabilities.min_bitstream_buffer_offset_alignment,
		min_bitstream_size_alignment: capabilities.min_bitstream_buffer_size_alignment,
		max_dpb_slots: capabilities.max_dpb_slots,
		max_active_reference_pictures: capabilities.max_active_reference_pictures,
		max_rate_control_layers: encode.max_rate_control_layers,
		max_bitrate: encode.max_bitrate,
		max_quality_levels: encode.max_quality_levels,
		constant_qp: encode
			.rate_control_modes
			.contains(ash::vk::VideoEncodeRateControlModeFlagsKHR::DISABLED),
		cbr: encode
			.rate_control_modes
			.contains(ash::vk::VideoEncodeRateControlModeFlagsKHR::CBR),
		vbr: encode
			.rate_control_modes
			.contains(ash::vk::VideoEncodeRateControlModeFlagsKHR::VBR),
		feedback_offset: encode
			.supported_encode_feedback_flags
			.contains(ash::vk::VideoEncodeFeedbackFlagsKHR::BITSTREAM_BUFFER_OFFSET),
		feedback_bytes_written: encode
			.supported_encode_feedback_flags
			.contains(ash::vk::VideoEncodeFeedbackFlagsKHR::BITSTREAM_BYTES_WRITTEN),
		feedback_overrides: encode
			.supported_encode_feedback_flags
			.contains(ash::vk::VideoEncodeFeedbackFlagsKHR::BITSTREAM_HAS_OVERRIDES),
		protected_content: capabilities
			.flags
			.contains(ash::vk::VideoCapabilityFlagsKHR::PROTECTED_CONTENT),
		separate_reference_images: capabilities
			.flags
			.contains(ash::vk::VideoCapabilityFlagsKHR::SEPARATE_REFERENCE_IMAGES),
		codec,
	}
}

#[derive(Clone, Copy)]
struct CommonCapabilities {
	flags: ash::vk::VideoCapabilityFlagsKHR,
	std_header_version: ash::vk::ExtensionProperties,
	min_bitstream_buffer_offset_alignment: u64,
	min_bitstream_buffer_size_alignment: u64,
	picture_access_granularity: video::VideoExtent,
	min_coded_extent: video::VideoExtent,
	max_coded_extent: video::VideoExtent,
	max_dpb_slots: u32,
	max_active_reference_pictures: u32,
}

impl From<&ash::vk::VideoCapabilitiesKHR<'_>> for CommonCapabilities {
	fn from(value: &ash::vk::VideoCapabilitiesKHR<'_>) -> Self {
		Self {
			flags: value.flags,
			std_header_version: value.std_header_version,
			min_bitstream_buffer_offset_alignment: value.min_bitstream_buffer_offset_alignment,
			min_bitstream_buffer_size_alignment: value.min_bitstream_buffer_size_alignment,
			picture_access_granularity: extent(value.picture_access_granularity),
			min_coded_extent: extent(value.min_coded_extent),
			max_coded_extent: extent(value.max_coded_extent),
			max_dpb_slots: value.max_dpb_slots,
			max_active_reference_pictures: value.max_active_reference_pictures,
		}
	}
}

const fn extent(value: ash::vk::Extent2D) -> video::VideoExtent {
	video::VideoExtent {
		width: value.width,
		height: value.height,
	}
}

const fn chroma_subsampling(
	value: video::VideoChromaSubsampling,
) -> ash::vk::VideoChromaSubsamplingFlagsKHR {
	match value {
		video::VideoChromaSubsampling::Monochrome => {
			ash::vk::VideoChromaSubsamplingFlagsKHR::MONOCHROME
		}
		video::VideoChromaSubsampling::Yuv420 => ash::vk::VideoChromaSubsamplingFlagsKHR::TYPE_420,
		video::VideoChromaSubsampling::Yuv422 => ash::vk::VideoChromaSubsamplingFlagsKHR::TYPE_422,
		video::VideoChromaSubsampling::Yuv444 => ash::vk::VideoChromaSubsamplingFlagsKHR::TYPE_444,
	}
}

const fn bit_depth(
	value: video::VideoComponentBitDepth,
) -> ash::vk::VideoComponentBitDepthFlagsKHR {
	match value {
		video::VideoComponentBitDepth::Eight => ash::vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
		video::VideoComponentBitDepth::Ten => ash::vk::VideoComponentBitDepthFlagsKHR::TYPE_10,
		video::VideoComponentBitDepth::Twelve => ash::vk::VideoComponentBitDepthFlagsKHR::TYPE_12,
	}
}

const fn h264_profile(value: video::H264Profile) -> ash::vk::native::StdVideoH264ProfileIdc {
	match value {
		video::H264Profile::Baseline => {
			ash::vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_BASELINE
		}
		video::H264Profile::Main => {
			ash::vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_MAIN
		}
		video::H264Profile::High => {
			ash::vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_HIGH
		}
		video::H264Profile::High444Predictive => {
			ash::vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_HIGH_444_PREDICTIVE
		}
	}
}

const fn h264_picture_layout(
	value: video::H264PictureLayout,
) -> ash::vk::VideoDecodeH264PictureLayoutFlagsKHR {
	match value {
		video::H264PictureLayout::Progressive => {
			ash::vk::VideoDecodeH264PictureLayoutFlagsKHR::PROGRESSIVE
		}
		video::H264PictureLayout::InterlacedInterleavedLines => {
			ash::vk::VideoDecodeH264PictureLayoutFlagsKHR::INTERLACED_INTERLEAVED_LINES
		}
		video::H264PictureLayout::InterlacedSeparatePlanes => {
			ash::vk::VideoDecodeH264PictureLayoutFlagsKHR::INTERLACED_SEPARATE_PLANES
		}
	}
}

const fn h265_profile(value: video::H265Profile) -> ash::vk::native::StdVideoH265ProfileIdc {
	match value {
		video::H265Profile::Main => {
			ash::vk::native::StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN
		}
		video::H265Profile::Main10 => {
			ash::vk::native::StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN_10
		}
		video::H265Profile::MainStillPicture => {
			ash::vk::native::StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN_STILL_PICTURE
		}
		video::H265Profile::FormatRangeExtensions => {
			ash::vk::native::StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_FORMAT_RANGE_EXTENSIONS
		}
		video::H265Profile::ScreenContentCodingExtensions => {
			ash::vk::native::StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_SCC_EXTENSIONS
		}
	}
}

const fn av1_profile(value: video::Av1Profile) -> ash::vk::native::StdVideoAV1Profile {
	match value {
		video::Av1Profile::Main => ash::vk::native::StdVideoAV1Profile_STD_VIDEO_AV1_PROFILE_MAIN,
		video::Av1Profile::High => ash::vk::native::StdVideoAV1Profile_STD_VIDEO_AV1_PROFILE_HIGH,
		video::Av1Profile::Professional => {
			ash::vk::native::StdVideoAV1Profile_STD_VIDEO_AV1_PROFILE_PROFESSIONAL
		}
	}
}

const fn vp9_operation() -> ash::vk::VideoCodecOperationFlagsKHR {
	ash::vk::VideoCodecOperationFlagsKHR::from_raw(0b1000)
}

const fn vp9_profile(value: video::Vp9Profile) -> ash_vp9::vk::native::StdVideoVP9Profile {
	match value {
		video::Vp9Profile::Profile0 => ash_vp9::vk::native::StdVideoVP9Profile_STD_VIDEO_VP9_PROFILE_0,
		video::Vp9Profile::Profile1 => ash_vp9::vk::native::StdVideoVP9Profile_STD_VIDEO_VP9_PROFILE_1,
		video::Vp9Profile::Profile2 => ash_vp9::vk::native::StdVideoVP9Profile_STD_VIDEO_VP9_PROFILE_2,
		video::Vp9Profile::Profile3 => ash_vp9::vk::native::StdVideoVP9Profile_STD_VIDEO_VP9_PROFILE_3,
	}
}

#[allow(dead_code, reason = "wired by the next AV1 frame decode checkpoint")]
fn std_av1_color(color: &video::Av1ColorConfig) -> ash::vk::native::StdVideoAV1ColorConfig {
	let description = color
		.color_description
		.unwrap_or(video::Av1ColorDescription {
			color_primaries: 2,
			transfer_characteristics: 2,
			matrix_coefficients: 2,
		});
	let (subsampling_x, subsampling_y) = match color.chroma_subsampling {
		video::VideoChromaSubsampling::Monochrome | video::VideoChromaSubsampling::Yuv420 => (1, 1),
		video::VideoChromaSubsampling::Yuv422 => (1, 0),
		video::VideoChromaSubsampling::Yuv444 => (0, 0),
	};
	let chroma_sample_position = match color.chroma_sample_position {
		video::Av1ChromaSamplePosition::Unknown => 0,
		video::Av1ChromaSamplePosition::Vertical => 1,
		video::Av1ChromaSamplePosition::Colocated => 2,
		video::Av1ChromaSamplePosition::Reserved => 3,
	};
	ash::vk::native::StdVideoAV1ColorConfig {
		flags: ash::vk::native::StdVideoAV1ColorConfigFlags {
			_bitfield_align_1: [],
			_bitfield_1: ash::vk::native::StdVideoAV1ColorConfigFlags::new_bitfield_1(
				u32::from(color.monochrome),
				u32::from(color.full_range),
				u32::from(color.separate_uv_delta_q),
				u32::from(color.color_description.is_some()),
				0,
			),
		},
		BitDepth: match color.bit_depth {
			video::VideoComponentBitDepth::Eight => 8,
			video::VideoComponentBitDepth::Ten => 10,
			video::VideoComponentBitDepth::Twelve => 12,
		},
		subsampling_x,
		subsampling_y,
		reserved1: 0,
		color_primaries: u32::from(description.color_primaries),
		transfer_characteristics: u32::from(description.transfer_characteristics),
		matrix_coefficients: u32::from(description.matrix_coefficients),
		chroma_sample_position,
	}
}

#[allow(dead_code, reason = "wired by the next AV1 frame decode checkpoint")]
fn std_av1_timing(timing: &video::Av1TimingInfo) -> ash::vk::native::StdVideoAV1TimingInfo {
	ash::vk::native::StdVideoAV1TimingInfo {
		flags: ash::vk::native::StdVideoAV1TimingInfoFlags {
			_bitfield_align_1: [],
			_bitfield_1: ash::vk::native::StdVideoAV1TimingInfoFlags::new_bitfield_1(
				u32::from(timing.equal_picture_interval),
				0,
			),
		},
		num_units_in_display_tick: timing.num_units_in_display_tick,
		time_scale: timing.time_scale,
		num_ticks_per_picture_minus_1: timing.num_ticks_per_picture_minus_1.unwrap_or(0),
	}
}

#[allow(dead_code, reason = "wired by the next AV1 frame decode checkpoint")]
fn std_av1_sequence(
	sequence: &video::Av1SequenceHeader,
	color: &ash::vk::native::StdVideoAV1ColorConfig,
	timing: *const ash::vk::native::StdVideoAV1TimingInfo,
) -> Result<ash::vk::native::StdVideoAV1SequenceHeader> {
	if sequence.frame_width_bits_minus_1 > 15 || sequence.frame_height_bits_minus_1 > 15 {
		return Err(Error::data_loss(
			"AV1 frame extent bit width exceeds 16 bits",
		));
	}
	let max_width = (1_u32 << (u32::from(sequence.frame_width_bits_minus_1) + 1)) - 1;
	let max_height = (1_u32 << (u32::from(sequence.frame_height_bits_minus_1) + 1)) - 1;
	if u32::from(sequence.max_frame_width_minus_1) > max_width
		|| u32::from(sequence.max_frame_height_minus_1) > max_height
	{
		return Err(Error::data_loss(
			"AV1 coded extent exceeds its declared field width",
		));
	}
	if sequence.enable_order_hint != (sequence.order_hint_bits != 0) || sequence.order_hint_bits > 8 {
		return Err(Error::data_loss(
			"AV1 order-hint flag and bit width are inconsistent",
		));
	}
	if sequence.timing.is_none() != timing.is_null() {
		return Err(Error::internal(
			"AV1 timing pointer does not match sequence timing metadata",
		));
	}
	let tool_choice = |choice| match choice {
		video::Av1CodingToolChoice::Disabled => 0,
		video::Av1CodingToolChoice::Enabled => 1,
		video::Av1CodingToolChoice::SelectPerFrame => 2,
	};
	Ok(ash::vk::native::StdVideoAV1SequenceHeader {
		flags: ash::vk::native::StdVideoAV1SequenceHeaderFlags {
			_bitfield_align_1: [],
			_bitfield_1: ash::vk::native::StdVideoAV1SequenceHeaderFlags::new_bitfield_1(
				u32::from(sequence.still_picture),
				u32::from(sequence.reduced_still_picture_header),
				u32::from(sequence.use_128x128_superblock),
				u32::from(sequence.enable_filter_intra),
				u32::from(sequence.enable_intra_edge_filter),
				u32::from(sequence.enable_inter_intra_compound),
				u32::from(sequence.enable_masked_compound),
				u32::from(sequence.enable_warped_motion),
				u32::from(sequence.enable_dual_filter),
				u32::from(sequence.enable_order_hint),
				u32::from(sequence.enable_joint_compound),
				u32::from(sequence.enable_reference_frame_motion_vectors),
				u32::from(sequence.frame_id_numbers_present),
				u32::from(sequence.enable_superres),
				u32::from(sequence.enable_cdef),
				u32::from(sequence.enable_restoration),
				u32::from(sequence.film_grain_params_present),
				u32::from(sequence.timing.is_some()),
				u32::from(sequence.initial_display_delay_present),
				0,
			),
		},
		seq_profile: av1_profile(sequence.profile),
		frame_width_bits_minus_1: sequence.frame_width_bits_minus_1,
		frame_height_bits_minus_1: sequence.frame_height_bits_minus_1,
		max_frame_width_minus_1: sequence.max_frame_width_minus_1,
		max_frame_height_minus_1: sequence.max_frame_height_minus_1,
		delta_frame_id_length_minus_2: sequence.delta_frame_id_length_minus_2,
		additional_frame_id_length_minus_1: sequence.additional_frame_id_length_minus_1,
		order_hint_bits_minus_1: sequence.order_hint_bits.saturating_sub(1),
		seq_force_integer_mv: tool_choice(sequence.integer_motion_vectors),
		seq_force_screen_content_tools: tool_choice(sequence.screen_content_tools),
		reserved1: [0; 5],
		pColorConfig: color,
		pTimingInfo: timing,
	})
}

#[cfg(test)]
mod tests {
	use super::{
		pack_h264_vulkan_access_unit, pack_h265_vulkan_access_unit, std_h264_hrd, std_h264_scaling,
		std_h265_hrd, std_h265_scaling,
	};

	#[test]
	fn packs_one_h265_vcl_nal_for_vulkan_video() -> crate::Result<()> {
		let access_unit = [
			0, 0, 0, 1, 0x40, 0x01, 0xaa, 0, 0, 1, 0x44, 0x01, 0xbb, 0, 0, 0, 1, 0x28, 0x01, 0x9a,
		];
		assert_eq!(
			pack_h265_vulkan_access_unit(&access_unit)?,
			[0, 0, 1, 0x28, 0x01, 0x9a]
		);
		Ok(())
	}

	#[test]
	fn rejects_unsupported_h265_access_unit_shapes() {
		assert!(pack_h265_vulkan_access_unit(&[0, 0, 1, 0x40, 0x01]).is_err());
		assert!(
			pack_h265_vulkan_access_unit(&[0, 0, 1, 0x28, 0x01, 0x9a, 0, 0, 1, 0x02, 0x01, 0x99,])
				.is_err()
		);
	}

	#[test]
	fn converts_h265_hrd_and_scaling_without_losing_syntax() -> crate::Result<()> {
		let hrd = crate::video::H265HrdParameters {
			nal_parameters_present: true,
			sub_picture_parameters_present: true,
			tick_divisor_minus_2: 17,
			bit_rate_scale: 3,
			cpb_size_scale: 4,
			cpb_size_du_scale: 5,
			sub_layers: vec![crate::video::H265SubLayerHrdParameters {
				fixed_picture_rate_general: true,
				fixed_picture_rate_within_cvs: true,
				elemental_duration_in_tc_minus_1: 2,
				nal_entries: vec![crate::video::H265CpbEntry {
					bit_rate_value_minus_1: 10,
					cpb_size_value_minus_1: 20,
					cpb_size_du_value_minus_1: 30,
					bit_rate_du_value_minus_1: 40,
					constant_bit_rate: true,
				}],
				..Default::default()
			}],
			..Default::default()
		};
		let converted = std_h265_hrd(&hrd)?;
		assert_eq!(converted.value.tick_divisor_minus2, 17);
		assert_eq!(converted.value.cpb_cnt_minus1[0], 0);
		assert_eq!(converted.value.elemental_duration_in_tc_minus1[0], 2);
		assert_eq!(converted._nal[0].bit_rate_value_minus1[0], 10);
		assert_eq!(converted._nal[0].cpb_size_du_value_minus1[0], 30);
		assert_eq!(converted._nal[0].cbr_flag, 1);

		let mut scaling = crate::video::H265ScalingLists::default();
		scaling.list_4x4[0] = core::array::from_fn(|index| index as u8);
		scaling.list_32x32[1] = core::array::from_fn(|index| 63 - index as u8);
		scaling.dc_16x16[2] = 23;
		let converted = std_h265_scaling(&scaling);
		assert_eq!(converted.ScalingList4x4[0], scaling.list_4x4[0]);
		assert_eq!(converted.ScalingList32x32[1], scaling.list_32x32[1]);
		assert_eq!(converted.ScalingListDCCoef16x16[2], 23);
		Ok(())
	}

	#[test]
	fn packs_one_h264_vcl_nal_for_vulkan_video() -> crate::Result<()> {
		let access_unit = [
			0, 0, 0, 1, 0x67, 0x64, 0x1f, 0, 0, 1, 0x68, 0xef, 0, 0, 0, 1, 0x65, 0x88, 0x84,
		];
		assert_eq!(
			pack_h264_vulkan_access_unit(&access_unit)?,
			[0, 0, 1, 0x65, 0x88, 0x84]
		);
		Ok(())
	}

	#[test]
	fn rejects_unsupported_h264_access_unit_shapes() {
		assert!(pack_h264_vulkan_access_unit(&[0, 0, 1, 0x67, 0x64]).is_err());
		assert!(pack_h264_vulkan_access_unit(&[0, 0, 1, 0x65, 0x88, 0, 0, 1, 0x61, 0x99,]).is_err());
	}

	#[test]
	fn converts_h264_hrd_entries_without_losing_syntax() -> crate::Result<()> {
		let hrd = crate::video::H264HrdParameters {
			bit_rate_scale: 3,
			cpb_size_scale: 4,
			entries: vec![
				crate::video::H264CpbEntry {
					bit_rate_value_minus_1: 10,
					cpb_size_value_minus_1: 20,
					constant_bit_rate: true,
				},
				crate::video::H264CpbEntry {
					bit_rate_value_minus_1: 30,
					cpb_size_value_minus_1: 40,
					constant_bit_rate: false,
				},
			],
			initial_cpb_removal_delay_length_minus_1: 5,
			cpb_removal_delay_length_minus_1: 6,
			dpb_output_delay_length_minus_1: 7,
			time_offset_length: 8,
		};
		let vui = crate::video::H264VuiParameters {
			aspect_ratio: None,
			overscan_appropriate: None,
			video_signal: None,
			chroma_location: None,
			timing: None,
			nal_hrd: Some(hrd.clone()),
			vcl_hrd: None,
			low_delay_hrd: Some(false),
			picture_structure_present: false,
			bitstream_restriction: None,
		};
		let converted = std_h264_hrd(&vui)?.expect("HRD table");
		assert_eq!(converted.cpb_cnt_minus1, 1);
		assert_eq!(converted.bit_rate_scale, 3);
		assert_eq!(converted.cpb_size_scale, 4);
		assert_eq!(&converted.bit_rate_value_minus1[..2], &[10, 30]);
		assert_eq!(&converted.cpb_size_value_minus1[..2], &[20, 40]);
		assert_eq!(&converted.cbr_flag[..2], &[1, 0]);

		let mut incompatible = vui;
		let mut distinct = hrd;
		distinct.time_offset_length = 9;
		incompatible.vcl_hrd = Some(distinct);
		assert!(std_h264_hrd(&incompatible).is_err());
		Ok(())
	}

	#[test]
	fn converts_h264_scaling_lists_without_reordering() {
		let mut scaling = crate::video::H264ScalingLists {
			present_mask: 0b0100_0001,
			use_default_mask: 0b0100_0000,
			list_4x4: [[0; 16]; 6],
			list_8x8: [[0; 64]; 6],
		};
		scaling.list_4x4[0] = core::array::from_fn(|index| index as u8);
		scaling.list_8x8[0] = core::array::from_fn(|index| 63 - index as u8);
		let converted = std_h264_scaling(&scaling);
		assert_eq!(converted.scaling_list_present_mask, scaling.present_mask);
		assert_eq!(
			converted.use_default_scaling_matrix_mask,
			scaling.use_default_mask
		);
		assert_eq!(converted.ScalingList4x4[0], scaling.list_4x4[0]);
		assert_eq!(converted.ScalingList8x8[0], scaling.list_8x8[0]);
	}
}
