//! H.264 decoded-picture-buffer planning and Vulkan standard-video lowering.
//!
//! Mirrors the AV1 and VP9 submodule pattern. Pure bitstream parsing lives in
//! `video::h264`; DPB slot management and picture planning belong here because
//! they are a Vulkan-backend concern, not a parser concern.

use crate::runtime::device::video::*;
use crate::{Error, Result, video};

// ── DPB slot state ────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct DpbSlot {
	pub in_use: bool,
	pub is_reference: bool,
	pub is_long_term: bool,
	pub frame_number: u32,
	pub picture_order_count: i32,
	pub decode_sequence: u64,
}

// ── Picture plan ──────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct PicturePlan {
	pub setup_slot: u32,
	pub picture_order_count: i32,
	pub frame_number: u32,
	pub long_term: bool,
	pub reset_dpb: bool,
	pub references: Vec<ReferencePlan>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ReferencePlan {
	pub slot: u32,
	pub frame_number: u32,
	pub picture_order_count: i32,
	pub long_term: bool,
}

// ── DPB state ─────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub(super) struct DpbState {
	slots: Vec<DpbSlot>,
	previous_poc_lsb: i32,
	previous_poc_msb: i32,
	decode_sequence: u64,
}

impl DpbState {
	pub(super) fn new(slot_count: u32) -> Result<Self> {
		if slot_count == 0 || slot_count > 16 {
			return Err(Error::invalid_argument(
				"H.264 DPB planner requires 1..=16 slots",
			));
		}
		Ok(Self {
			slots: vec![DpbSlot::default(); slot_count as usize],
			previous_poc_lsb: 0,
			previous_poc_msb: 0,
			decode_sequence: 0,
		})
	}

	pub(super) fn plan_with_unavailable(
		&mut self,
		sps: &video::H264SequenceParameterSet,
		slice: &video::H264SliceHeader,
		unavailable: &[bool],
	) -> Result<PicturePlan> {
		if !unavailable.is_empty() && unavailable.len() != self.slots.len() {
			return Err(Error::invalid_argument(
				"H.264 unavailable-slot mask does not match DPB capacity",
			));
		}
		let mut next = self.clone();
		let plan = next.plan_in_place(sps, slice, unavailable)?;
		*self = next;
		Ok(plan)
	}

	fn plan_in_place(
		&mut self,
		sps: &video::H264SequenceParameterSet,
		slice: &video::H264SliceHeader,
		unavailable: &[bool],
	) -> Result<PicturePlan> {
		if slice.field_picture {
			return Err(Error::missing_capability(
				"H.264 DPB planning currently admits progressive pictures only",
			));
		}
		if sps.pic_order_count_type != 0 {
			return Err(Error::missing_capability(
				"H.264 DPB planning currently admits POC type zero only",
			));
		}
		if slice.is_idr {
			self.slots.fill(DpbSlot::default());
			self.previous_poc_lsb = 0;
			self.previous_poc_msb = 0;
		}
		let poc_lsb = i32::try_from(
			slice
				.picture_order_count_lsb
				.ok_or_else(|| Error::data_loss("H.264 POC type zero slice has no coded POC LSB"))?,
		)
		.map_err(|_| Error::data_loss("H.264 POC LSB exceeds i32"))?;
		let poc_bits = sps
			.log2_max_pic_order_count_lsb_minus_4
			.checked_add(4)
			.ok_or_else(|| Error::data_loss("H.264 POC width overflows"))?;
		let max_poc_lsb = 1_i32
			.checked_shl(poc_bits)
			.ok_or_else(|| Error::data_loss("H.264 POC width exceeds i32"))?;
		let half = max_poc_lsb / 2;
		let poc_msb = if poc_lsb < self.previous_poc_lsb && self.previous_poc_lsb - poc_lsb >= half {
			self
				.previous_poc_msb
				.checked_add(max_poc_lsb)
				.ok_or_else(|| Error::data_loss("H.264 POC MSB overflows"))?
		} else if poc_lsb > self.previous_poc_lsb && poc_lsb - self.previous_poc_lsb > half {
			self
				.previous_poc_msb
				.checked_sub(max_poc_lsb)
				.ok_or_else(|| Error::data_loss("H.264 POC MSB underflows"))?
		} else {
			self.previous_poc_msb
		};
		let picture_order_count = poc_msb
			.checked_add(poc_lsb)
			.ok_or_else(|| Error::data_loss("H.264 picture order count overflows"))?;
		if slice.is_reference {
			self.previous_poc_lsb = poc_lsb;
			self.previous_poc_msb = poc_msb;
		}

		let references = self
			.slots
			.iter()
			.enumerate()
			.filter(|(_, slot)| slot.in_use && slot.is_reference)
			.map(|(index, slot)| ReferencePlan {
				slot: index as u32,
				frame_number: slot.frame_number,
				picture_order_count: slot.picture_order_count,
				long_term: slot.is_long_term,
			})
			.collect::<Vec<_>>();
		let setup_slot = self
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
					.min_by_key(|(_, slot)| slot.decode_sequence)
					.map(|(index, _)| index)
			})
			.ok_or_else(|| {
				Error::resource_exhausted("all H.264 DPB slots are references or consumer-leased")
			})?;
		self.slots[setup_slot] = DpbSlot {
			in_use: slice.is_reference,
			is_reference: slice.is_reference,
			is_long_term: slice.is_idr && slice.long_term_reference,
			frame_number: slice.frame_number,
			picture_order_count,
			decode_sequence: self.decode_sequence,
		};
		self.decode_sequence = self
			.decode_sequence
			.checked_add(1)
			.ok_or_else(|| Error::resource_exhausted("H.264 decode sequence exhausted"))?;

		if slice.is_reference && !slice.is_idr && slice.adaptive_reference_picture_marking {
			self.apply_mmco(sps, slice, setup_slot)?;
		} else if slice.is_reference && !slice.is_idr {
			self.apply_sliding_window(sps, slice.frame_number, setup_slot)?;
		}
		Ok(PicturePlan {
			setup_slot: setup_slot as u32,
			picture_order_count,
			frame_number: slice.frame_number,
			long_term: self.slots[setup_slot].is_long_term,
			reset_dpb: slice.is_idr,
			references,
		})
	}

	fn apply_sliding_window(
		&mut self,
		sps: &video::H264SequenceParameterSet,
		current_frame_number: u32,
		current_slot: usize,
	) -> Result<()> {
		let maximum = usize::try_from(sps.max_num_ref_frames.max(1))
			.map_err(|_| Error::data_loss("H.264 reference count exceeds usize"))?;
		while self.slots.iter().filter(|slot| slot.is_reference).count() > maximum {
			let frame_bits = sps
				.log2_max_frame_num_minus_4
				.checked_add(4)
				.ok_or_else(|| Error::data_loss("H.264 frame-number width overflows"))?;
			let max_frame_number = 1_i64
				.checked_shl(frame_bits)
				.ok_or_else(|| Error::data_loss("H.264 frame-number width exceeds i64"))?;
			let current = i64::from(current_frame_number);
			let oldest = self
				.slots
				.iter()
				.enumerate()
				.filter(|(index, slot)| *index != current_slot && slot.is_reference && !slot.is_long_term)
				.min_by_key(|(_, slot)| {
					let frame = i64::from(slot.frame_number);
					if frame > current {
						frame - max_frame_number
					} else {
						frame
					}
				})
				.map(|(index, _)| index)
				.ok_or_else(|| {
					Error::resource_exhausted("H.264 sliding window has no short-term victim")
				})?;
			self.slots[oldest] = DpbSlot::default();
		}
		Ok(())
	}

	fn apply_mmco(
		&mut self,
		sps: &video::H264SequenceParameterSet,
		slice: &video::H264SliceHeader,
		current_slot: usize,
	) -> Result<()> {
		let frame_bits = sps
			.log2_max_frame_num_minus_4
			.checked_add(4)
			.ok_or_else(|| Error::data_loss("H.264 frame-number width overflows"))?;
		let max_frame_number = 1_i64
			.checked_shl(frame_bits)
			.ok_or_else(|| Error::data_loss("H.264 frame-number width exceeds i64"))?;
		for command in &slice.memory_management {
			match command.operation {
				1 => {
					let difference = i64::from(command.difference_of_picture_numbers_minus_1) + 1;
					let target = i64::from(slice.frame_number) - difference;
					if let Some((_, slot)) = self.slots.iter_mut().enumerate().find(|(_, slot)| {
						if !slot.is_reference || slot.is_long_term {
							return false;
						}
						let frame = i64::from(slot.frame_number);
						let wrapped = if frame > i64::from(slice.frame_number) {
							frame - max_frame_number
						} else {
							frame
						};
						wrapped == target
					}) {
						*slot = DpbSlot::default();
					}
				}
				5 => {
					for (index, slot) in self.slots.iter_mut().enumerate() {
						if index != current_slot {
							*slot = DpbSlot::default();
						}
					}
				}
				6 => self.slots[current_slot].is_long_term = true,
				2..=4 => {
					return Err(Error::missing_capability(
						"H.264 long-term MMCO operations 2..=4 are not yet planned",
					));
				}
				_ => return Err(Error::data_loss("H.264 MMCO operation exceeds six")),
			}
		}
		Ok(())
	}
}

pub(super) fn h264_reference_info(
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
pub(super) fn record_h264_picture(
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
	plan: &h264::PicturePlan,
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
