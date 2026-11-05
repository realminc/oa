//! H.265/HEVC decoded-picture-buffer planning and Vulkan standard-video lowering.
//!
//! Mirrors the AV1, VP9, and H.264 submodule pattern. Pure bitstream parsing
//! lives in `video::h265`; DPB slot management and picture planning belong here
//! because they are a Vulkan-backend concern, not a parser concern.

use crate::runtime::device::video::*;
use crate::{Error, Result, video};

// ── DPB slot state ────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct DpbSlotState {
	pub in_use: bool,
	pub is_reference: bool,
	pub picture_order_count: i32,
	pub decode_index: u64,
}

// ── Picture plan ──────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct PicturePlan {
	pub picture_order_count: i32,
	pub setup_slot: u32,
	pub active_references: Vec<ReferencePlan>,
	pub current_before_slots: Vec<u8>,
	pub current_after_slots: Vec<u8>,
	pub reset_dpb: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ReferencePlan {
	pub slot: u32,
	pub picture_order_count: i32,
}

// ── DPB state ─────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct DpbState {
	pub(super) slots: Vec<DpbSlotState>,
	previous_poc_lsb: i32,
	previous_poc_msb: i32,
	has_previous_poc: bool,
	decode_index: u64,
}

impl DpbState {
	pub(super) fn new(slot_count: u32) -> Result<Self> {
		if slot_count == 0 || slot_count > 16 {
			return Err(Error::invalid_argument(
				"H.265 DPB state requires 1..=16 slots",
			));
		}
		Ok(Self {
			slots: vec![DpbSlotState::default(); slot_count as usize],
			previous_poc_lsb: 0,
			previous_poc_msb: 0,
			has_previous_poc: false,
			decode_index: 0,
		})
	}

	pub(super) fn plan_with_unavailable(
		&mut self,
		sps: &video::H265SequenceParameterSet,
		slice: &video::H265SliceHeader,
		unavailable: &[bool],
	) -> Result<PicturePlan> {
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
	) -> Result<PicturePlan> {
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
			self.slots.fill(DpbSlotState::default());
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
				*slot = DpbSlotState::default();
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
				Ok(ReferencePlan {
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
			DpbSlotState {
				in_use: true,
				is_reference: true,
				picture_order_count,
				decode_index: self.decode_index,
			}
		} else {
			DpbSlotState::default()
		};
		Ok(PicturePlan {
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

#[allow(clippy::too_many_arguments)]
pub(super) fn record_h265_picture(
	device: &vk::Device,
	command_buffer: vk::CommandBuffer,
	session: &DecodeSession,
	sps: &video::H265SequenceParameterSet,
	pps: &video::H265PictureParameterSet,
	slice: &video::H265SliceHeader,
	slice_offset: u32,
	bitstream: &DecodeBitstream,
	plan: &h265::PicturePlan,
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
	let mut std_picture = vk::StdVideoDecodeH265PictureInfo {
		flags: {
			let mut flags = vk::StdVideoDecodeH265PictureInfoFlags::new();
			flags.set_irap_pic_flag(u32::from(slice.is_irap) as _);
			flags.set_idr_pic_flag(u32::from(slice.is_idr) as _);
			flags.set_is_reference(u32::from(slice.is_reference) as _);
			flags.set_short_term_ref_pic_set_sps_flag(u32::from(
				slice.short_term_reference_picture_set_sps,
			) as _);
			flags
		},
		sps_video_parameter_set_id: h265_u8(sps.video_parameter_set_id, "picture VPS id")?,
		pps_seq_parameter_set_id: h265_u8(sps.id, "picture SPS id")?,
		pps_pic_parameter_set_id: h265_u8(pps.id, "picture PPS id")?,
		num_delta_pocs_of_ref_rps_idx: u8::try_from(num_delta_pocs)
			.map_err(|_| Error::data_loss("H.265 reference-set delta count exceeds u8"))?,
		pic_order_cnt_val: plan.picture_order_count,
		num_bits_for_st_ref_pic_set_in_slice: slice.short_term_reference_picture_set_bits,
		reserved: 0,
		ref_pic_set_st_curr_before: [0xff; 8],
		ref_pic_set_st_curr_after: [0xff; 8],
		ref_pic_set_lt_curr: [0xff; 8],
	};
	for (destination, slot) in std_picture
		.ref_pic_set_st_curr_before
		.iter_mut()
		.zip(&plan.current_before_slots)
	{
		*destination = *slot;
	}
	for (destination, slot) in std_picture
		.ref_pic_set_st_curr_after
		.iter_mut()
		.zip(&plan.current_after_slots)
	{
		*destination = *slot;
	}
	let mut h265_picture = vk::VideoDecodeH265PictureInfoKHR::new()
		.set_std_picture_info(&std_picture)
		.set_slice_segment_offsets(std::slice::from_ref(&slice_offset))
		.map_err(Error::from)?;

	let setup_std_reference = h265_reference_info(plan.picture_order_count, !slice.is_reference);
	let mut setup_h265_slot =
		vk::VideoDecodeH265DpbSlotInfoKHR::new().set_std_reference_info(&setup_std_reference);
	let (output, output_layer, dpb, dpb_layer) = session.picture_images(plan.setup_slot)?;
	let extent = vk::Extent2D {
		width: session.coded_extent.width,
		height: session.coded_extent.height,
	};
	let setup_resource = vk::VideoPictureResourceInfoKHR::new()
		.set_coded_extent(extent)
		.set_base_array_layer(dpb_layer)
		.set_image_view_binding(dpb.view);
	let setup_slot_index =
		i32::try_from(plan.setup_slot).map_err(|_| Error::internal("H.265 setup slot exceeds i32"))?;
	let setup_slot = vk::VideoReferenceSlotInfoKHR::new()
		.set_slot_index(setup_slot_index)
		.set_picture_resource(&setup_resource)
		.push_next(&mut setup_h265_slot)
		.map_err(Error::from)?;

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
		h265_reference_slots.push(vk::VideoDecodeH265DpbSlotInfoKHR::new());
		reference_resources.push(
			vk::VideoPictureResourceInfoKHR::new()
				.set_coded_extent(extent)
				.set_base_array_layer(reference_layer)
				.set_image_view_binding(reference_image.view),
		);
		reference_slots.push(
			vk::VideoReferenceSlotInfoKHR::new().set_slot_index(
				i32::try_from(reference.slot)
					.map_err(|_| Error::internal("H.265 reference slot exceeds i32"))?,
			),
		);
	}
	for index in 0..reference_count {
		h265_reference_slots[index].p_std_reference_info = &std_references[index];
		reference_slots[index].p_next =
			(&h265_reference_slots[index] as *const vk::VideoDecodeH265DpbSlotInfoKHR<'_>).cast();
		reference_slots[index].p_picture_resource = &reference_resources[index];
	}
	let mut inactive_setup = vk::VideoReferenceSlotInfoKHR::new()
		.set_slot_index(-1)
		.set_picture_resource(&setup_resource);
	inactive_setup.p_next = setup_slot.p_next;
	reference_slots.push(inactive_setup);
	let begin_info = vk::VideoBeginCodingInfoKHR::new()
		.set_video_session(session.handle)
		.set_video_session_parameters(session.parameters)
		.set_reference_slots(&reference_slots)
		.map_err(Error::from)?;
	let output_resource = vk::VideoPictureResourceInfoKHR::new()
		.set_coded_extent(extent)
		.set_base_array_layer(output_layer)
		.set_image_view_binding(output.view);
	let decode_info = vk::VideoDecodeInfoKHR::new()
		.set_src_buffer(bitstream.handle)
		.set_src_buffer_offset(0)
		.set_src_buffer_range(bitstream.range)
		.set_dst_picture_resource(output_resource)
		.set_setup_reference_slot(&setup_slot)
		.set_reference_slots(&reference_slots[..reference_count])
		.map_err(Error::from)?
		.push_next(&mut h265_picture)
		.map_err(Error::from)?;

	let mut image_barriers = Vec::new();
	if initialize_images {
		if session.image_set == DecodeImageSet::CoincidentSeparate {
			image_barriers
				.try_reserve_exact(session.images.len())
				.map_err(|_| Error::resource_exhausted("decode-image barrier allocation failed"))?;
			for image in session.images.iter() {
				image_barriers.push(decode_image_barrier(
					image,
					vk::ImageLayout::VIDEO_DECODE_DPB_KHR,
					vk::AccessFlagBits2::VIDEO_DECODE_READ_BIT_KHR
						| vk::AccessFlagBits2::VIDEO_DECODE_WRITE_BIT_KHR,
				));
			}
		} else {
			image_barriers.push(decode_image_barrier(
				dpb,
				vk::ImageLayout::VIDEO_DECODE_DPB_KHR,
				vk::AccessFlagBits2::VIDEO_DECODE_READ_BIT_KHR
					| vk::AccessFlagBits2::VIDEO_DECODE_WRITE_BIT_KHR,
			));
			if output.handle != dpb.handle {
				image_barriers.push(decode_image_barrier(
					output,
					vk::ImageLayout::VIDEO_DECODE_DST_KHR,
					vk::AccessFlagBits2::VIDEO_DECODE_WRITE_BIT_KHR,
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
				vk::ImageLayout::VIDEO_DECODE_DST_KHR
			} else {
				vk::ImageLayout::VIDEO_DECODE_DPB_KHR
			};
			image_barriers.push(
				vk::ImageMemoryBarrier2::new()
					.set_src_stage_mask(vk::PipelineStageFlagBits2::NONE)
					.set_src_access_mask(vk::AccessFlagBits2::NONE)
					.set_dst_stage_mask(vk::PipelineStageFlagBits2::VIDEO_DECODE_BIT_KHR)
					.set_dst_access_mask(
						vk::AccessFlagBits2::VIDEO_DECODE_READ_BIT_KHR
							| vk::AccessFlagBits2::VIDEO_DECODE_WRITE_BIT_KHR,
					)
					.set_old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
					.set_new_layout(layout)
					.set_src_queue_family_index(compute_family)
					.set_dst_queue_family_index(decode_family)
					.set_image(acquired_output.handle)
					.set_subresource_range(decode_image_subresource(acquired_layer, 1)),
			);
		}
	}
	let bitstream_barriers = [vk::BufferMemoryBarrier2::new()
		.set_src_stage_mask(vk::PipelineStageFlagBits2::HOST_BIT)
		.set_src_access_mask(vk::AccessFlagBits2::HOST_WRITE_BIT)
		.set_dst_stage_mask(vk::PipelineStageFlagBits2::VIDEO_DECODE_BIT_KHR)
		.set_dst_access_mask(vk::AccessFlagBits2::VIDEO_DECODE_READ_BIT_KHR)
		.set_src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
		.set_dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
		.set_buffer(bitstream.handle)
		.set_offset(0)
		.set_size(bitstream.range)];
	let memory_barriers = (!initialize_images)
		.then(|| {
			vk::MemoryBarrier2::new()
				.set_src_stage_mask(vk::PipelineStageFlagBits2::VIDEO_DECODE_BIT_KHR)
				.set_src_access_mask(vk::AccessFlagBits2::VIDEO_DECODE_WRITE_BIT_KHR)
				.set_dst_stage_mask(vk::PipelineStageFlagBits2::VIDEO_DECODE_BIT_KHR)
				.set_dst_access_mask(
					vk::AccessFlagBits2::VIDEO_DECODE_READ_BIT_KHR
						| vk::AccessFlagBits2::VIDEO_DECODE_WRITE_BIT_KHR,
				)
		})
		.into_iter()
		.collect::<Vec<_>>();
	let dependency = vk::DependencyInfo::new()
		.set_memory_barriers(&memory_barriers)
		.map_err(Error::from)?
		.set_buffer_memory_barriers(&bitstream_barriers)
		.map_err(Error::from)?
		.set_image_memory_barriers(&image_barriers)
		.map_err(Error::from)?;
	// The DPB dependency and layout transitions precede the coding scope. The
	// result-status query is reused only after the caller has completed and
	// checked the preceding picture event.
	unsafe {
		device.cmd_reset_query_pool(command_buffer, session.result_status_pool, 0, 1)?;
		device.cmd_pipeline_barrier2(command_buffer, &dependency)?;
		device.cmd_begin_video_coding_khr(command_buffer, &begin_info)?;
	}
	if plan.reset_dpb {
		let control = vk::VideoCodingControlInfoKHR::new()
			.set_flags(vk::VideoCodingControlFlagBitsKHR::RESET_BIT_KHR);
		unsafe {
			device.cmd_control_video_coding_khr(command_buffer, &control)?;
		}
	}
	unsafe {
		device.cmd_begin_query(
			command_buffer,
			session.result_status_pool,
			0,
			vk::QueryControlFlags::empty(),
		)?;
		device.cmd_decode_video_khr(command_buffer, &decode_info)?;
		device.cmd_end_query(command_buffer, session.result_status_pool, 0)?;
		device.cmd_end_video_coding_khr(command_buffer, &vk::VideoEndCodingInfoKHR::new())?;
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
			vk::ImageLayout::VIDEO_DECODE_DPB_KHR
		} else {
			vk::ImageLayout::VIDEO_DECODE_DST_KHR
		};
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
		let release_dependency =
			vk::DependencyInfo::new().set_image_memory_barriers(std::slice::from_ref(&release));
		// safety: decode owns this exact output layer. The caller submits the
		// matching compute-family acquire only after this queue signals completion.
		unsafe {
			device
				.cmd_pipeline_barrier2(command_buffer, &(release_dependency).map_err(Error::from)?)
				.map_err(Error::from)?;
		}
	}
	Ok(())
}

pub(super) fn h265_reference_info(
	picture_order_count: i32,
	unused_for_reference: bool,
) -> vk::StdVideoDecodeH265ReferenceInfo {
	vk::StdVideoDecodeH265ReferenceInfo {
		flags: {
			let mut flags = vk::StdVideoDecodeH265ReferenceInfoFlags::new();
			flags.set_used_for_long_term_reference(0 as _);
			flags.set_unused_for_reference(u32::from(unused_for_reference) as _);
			flags
		},
		pic_order_cnt_val: picture_order_count,
	}
}
