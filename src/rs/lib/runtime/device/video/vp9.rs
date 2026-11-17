//! VP9 decoded-picture-buffer planning and Vulkan standard-video lowering.

use crate::{Error, Result, video};

use crate::runtime::{
	device::video::DecodeBitstream, device::video::DecodeImageSet, device::video::DecodeSession,
	device::video::decode_image_barrier, device::video::decode_image_subresource,
};

const LOGICAL_REFERENCE_COUNT: usize = 8;
const REFERENCE_NAME_COUNT: usize = 3;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct DpbState {
	logical_to_physical: [Option<u32>; LOGICAL_REFERENCE_COUNT],
	physical_extents: Vec<Option<(u32, u32)>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct PicturePlan {
	pub setup_slot: Option<u32>,
	pub show_existing_slot: Option<u32>,
	pub reference_name_slot_indices: [i32; REFERENCE_NAME_COUNT],
	pub active_references: Vec<u32>,
	pub reset_dpb: bool,
}

impl DpbState {
	pub fn new(slot_count: u32) -> Result<Self> {
		if slot_count == 0 || slot_count > 16 {
			return Err(Error::invalid_argument(
				"VP9 DPB state requires 1..=16 physical slots",
			));
		}
		Ok(Self {
			logical_to_physical: [None; LOGICAL_REFERENCE_COUNT],
			physical_extents: vec![None; slot_count as usize],
		})
	}

	pub fn plan_with_unavailable(
		&mut self,
		picture: &video::Vp9Picture,
		unavailable: &[bool],
	) -> Result<PicturePlan> {
		if !unavailable.is_empty() && unavailable.len() != self.physical_extents.len() {
			return Err(Error::invalid_argument(
				"VP9 unavailable-slot mask does not match DPB capacity",
			));
		}
		let mut next = self.clone();
		let plan = next.plan_in_place(picture, unavailable)?;
		*self = next;
		Ok(plan)
	}

	fn plan_in_place(
		&mut self,
		picture: &video::Vp9Picture,
		unavailable: &[bool],
	) -> Result<PicturePlan> {
		if picture.show_existing_frame {
			let logical = usize::from(picture.frame_to_show_map_index);
			let slot = self
				.logical_to_physical
				.get(logical)
				.copied()
				.flatten()
				.ok_or_else(|| Error::data_loss("VP9 show-existing reference is absent from the DPB"))?;
			return Ok(PicturePlan {
				setup_slot: None,
				show_existing_slot: Some(slot),
				reference_name_slot_indices: [-1; REFERENCE_NAME_COUNT],
				active_references: Vec::new(),
				reset_dpb: false,
			});
		}

		let reset_dpb = picture.frame_type == video::vp9::Vp9FrameType::Key;
		if reset_dpb {
			self.logical_to_physical.fill(None);
			self.physical_extents.fill(None);
		}
		let mut named = [-1_i32; REFERENCE_NAME_COUNT];
		let mut protected = vec![false; self.physical_extents.len()];
		if !picture.frame_is_intra {
			for (name, logical) in picture.reference_frame_indices.iter().copied().enumerate() {
				let slot = self.logical_to_physical[usize::from(logical)]
					.ok_or_else(|| Error::data_loss("VP9 named reference is absent from the DPB"))?;
				let index =
					usize::try_from(slot).map_err(|_| Error::internal("VP9 physical slot exceeds usize"))?;
				if self
					.physical_extents
					.get(index)
					.copied()
					.flatten()
					.is_none()
				{
					return Err(Error::internal(
						"VP9 logical reference points at an empty slot",
					));
				}
				protected[index] = true;
				named[name] =
					i32::try_from(slot).map_err(|_| Error::internal("VP9 reference slot exceeds i32"))?;
			}
		}

		let remains_mapped = |slot: u32| {
			self
				.logical_to_physical
				.iter()
				.enumerate()
				.any(|(logical, mapped)| {
					*mapped == Some(slot) && picture.refresh_frame_flags & (1 << logical) == 0
				})
		};
		let setup_index = self
			.physical_extents
			.iter()
			.enumerate()
			.find(|(index, value)| {
				value.is_none() && !protected[*index] && !unavailable.get(*index).copied().unwrap_or(false)
			})
			.map(|(index, _)| index)
			.or_else(|| {
				self
					.physical_extents
					.iter()
					.enumerate()
					.find_map(|(index, _)| {
						let slot = u32::try_from(index).ok()?;
						(!protected[index]
							&& !unavailable.get(index).copied().unwrap_or(false)
							&& !remains_mapped(slot))
						.then_some(index)
					})
			})
			.ok_or_else(|| Error::resource_exhausted("VP9 DPB has no recyclable physical slot"))?;
		let setup_slot =
			u32::try_from(setup_index).map_err(|_| Error::internal("VP9 setup slot exceeds u32"))?;

		let mut active_references = Vec::new();
		for (index, active) in protected.into_iter().enumerate() {
			if active {
				active_references.push(
					u32::try_from(index).map_err(|_| Error::internal("VP9 active reference exceeds u32"))?,
				);
			}
		}
		for (logical, mapped) in self.logical_to_physical.iter_mut().enumerate() {
			if *mapped == Some(setup_slot) || picture.refresh_frame_flags & (1 << logical) != 0 {
				*mapped = None;
			}
		}
		if picture.refresh_frame_flags != 0 {
			self.physical_extents[setup_index] = Some((picture.frame_width, picture.frame_height));
			for (logical, mapped) in self.logical_to_physical.iter_mut().enumerate() {
				if picture.refresh_frame_flags & (1 << logical) != 0 {
					*mapped = Some(setup_slot);
				}
			}
		} else {
			self.physical_extents[setup_index] = None;
		}
		for (index, extent) in self.physical_extents.iter_mut().enumerate() {
			let slot =
				u32::try_from(index).map_err(|_| Error::internal("VP9 physical slot exceeds u32"))?;
			if !self.logical_to_physical.contains(&Some(slot)) {
				*extent = None;
			}
		}

		Ok(PicturePlan {
			setup_slot: Some(setup_slot),
			show_existing_slot: None,
			reference_name_slot_indices: named,
			active_references,
			reset_dpb,
		})
	}
}

pub(super) struct PictureParameters {
	pub color: vk::StdVideoVP9ColorConfig,
	pub loop_filter: vk::StdVideoVP9LoopFilter,
	pub segmentation: vk::StdVideoVP9Segmentation,
}

impl PictureParameters {
	pub fn new(picture: &video::Vp9Picture) -> Self {
		Self {
			color: std_color(picture.color),
			loop_filter: std_loop_filter(picture.loop_filter),
			segmentation: std_segmentation(&picture.segmentation),
		}
	}

	pub fn picture_info(&self, picture: &video::Vp9Picture) -> vk::StdVideoDecodeVP9PictureInfo<'_> {
		let mut value = vk::StdVideoDecodeVP9PictureInfo::new();
		value
			.flags
			.set_error_resilient_mode(u32::from(picture.error_resilient_mode));
		value.flags.set_intra_only(u32::from(picture.intra_only));
		value
			.flags
			.set_allow_high_precision_mv(u32::from(picture.allow_high_precision_motion_vectors));
		value
			.flags
			.set_refresh_frame_context(u32::from(picture.refresh_frame_context));
		value
			.flags
			.set_frame_parallel_decoding_mode(u32::from(picture.frame_parallel_decoding_mode));
		value
			.flags
			.set_segmentation_enabled(u32::from(picture.segmentation_enabled));
		value.flags.set_show_frame(u32::from(picture.show_frame));
		value
			.flags
			.set_use_prev_frame_mvs(u32::from(picture.use_previous_frame_motion_vectors));
		value.flags.set_reserved(0);
		value.profile = std_profile(picture.profile);
		value.frame_type = std_frame_type(picture.frame_type);
		value.frame_context_idx = picture.frame_context_index;
		value.reset_frame_context = picture.reset_frame_context;
		value.refresh_frame_flags = picture.refresh_frame_flags;
		value.ref_frame_sign_bias_mask = picture.reference_frame_sign_bias_mask;
		value.interpolation_filter = std_interpolation_filter(picture.interpolation_filter);
		value.base_q_idx = picture.base_q_index;
		value.delta_q_y_dc = picture.delta_q_y_dc;
		value.delta_q_uv_dc = picture.delta_q_uv_dc;
		value.delta_q_uv_ac = picture.delta_q_uv_ac;
		value.tile_cols_log2 = picture.tile_columns_log2;
		value.tile_rows_log2 = picture.tile_rows_log2;
		value.reserved1 = [0; 3];
		value.p_color_config = &self.color;
		value.p_loop_filter = &self.loop_filter;
		value.p_segmentation = &self.segmentation;
		value
	}
}

#[allow(clippy::too_many_arguments)]
pub(super) fn record_picture(
	device: &vk::Device,
	command_buffer: vk::CommandBuffer,
	session: &DecodeSession,
	picture: &video::Vp9Picture,
	bitstream: &DecodeBitstream,
	plan: &PicturePlan,
	initialize_images: bool,
	pending_acquire_slot: Option<u32>,
	release_for_readback: bool,
) -> Result<()> {
	let setup_slot_index = plan
		.setup_slot
		.ok_or_else(|| Error::internal("coded VP9 picture has no setup slot"))?;
	if plan.show_existing_slot.is_some() {
		return Err(Error::internal(
			"show-existing VP9 plan reached command recording",
		));
	}
	if picture.compressed_header_size == 0 || picture.tile_count == 0 {
		return Err(Error::invalid_argument(
			"VP9 header or tile metadata is incomplete",
		));
	}
	let tiles_offset = usize::try_from(picture.tiles_offset)
		.map_err(|_| Error::out_of_range("VP9 tile offset exceeds usize"))?;
	if tiles_offset > bitstream.payload_len {
		return Err(Error::invalid_argument(
			"VP9 tile offset exceeds the uploaded picture",
		));
	}

	let parameters = PictureParameters::new(picture);
	let std_picture = parameters.picture_info(picture);
	let mut vp9_picture = vk::VideoDecodeVP9PictureInfoKHR::new()
		.set_std_picture_info(&std_picture)
		.set_reference_name_slot_indices(plan.reference_name_slot_indices)
		.set_uncompressed_header_offset(picture.uncompressed_header_offset)
		.set_compressed_header_offset(picture.compressed_header_offset)
		.set_tiles_offset(picture.tiles_offset);

	let (output, output_layer, dpb, dpb_layer) = session.picture_images(setup_slot_index)?;
	let extent = vk::Extent2D {
		width: session.coded_extent.width,
		height: session.coded_extent.height,
	};
	let setup_resource = vk::VideoPictureResourceInfoKHR::new()
		.set_coded_extent(extent)
		.set_base_array_layer(dpb_layer)
		.set_image_view_binding(dpb.view);
	let setup_slot = vk::VideoReferenceSlotInfoKHR::new()
		.set_slot_index(
			i32::try_from(setup_slot_index).map_err(|_| Error::internal("VP9 setup slot exceeds i32"))?,
		)
		.set_picture_resource(&setup_resource);

	let reference_count = plan.active_references.len();
	let mut reference_resources = Vec::new();
	let mut reference_slots = Vec::new();
	reference_resources
		.try_reserve_exact(reference_count)
		.map_err(|_| Error::resource_exhausted("VP9 reference resource allocation failed"))?;
	reference_slots
		.try_reserve_exact(reference_count + 1)
		.map_err(|_| Error::resource_exhausted("VP9 reference slot allocation failed"))?;
	for &slot in &plan.active_references {
		if slot == setup_slot_index {
			return Err(Error::internal(
				"VP9 setup slot aliases an active reference",
			));
		}
		let (_, _, image, layer) = session.picture_images(slot)?;
		reference_resources.push(
			vk::VideoPictureResourceInfoKHR::new()
				.set_coded_extent(extent)
				.set_base_array_layer(layer)
				.set_image_view_binding(image.view),
		);
		reference_slots.push(vk::VideoReferenceSlotInfoKHR::new().set_slot_index(
			i32::try_from(slot).map_err(|_| Error::internal("VP9 reference slot exceeds i32"))?,
		));
	}
	for index in 0..reference_count {
		reference_slots[index].p_picture_resource = &reference_resources[index];
	}
	let inactive_setup = vk::VideoReferenceSlotInfoKHR::new()
		.set_slot_index(-1)
		.set_picture_resource(&setup_resource);
	reference_slots.push(inactive_setup);
	let begin_info = vk::VideoBeginCodingInfoKHR::new()
		.set_video_session(session.handle)
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
		.push_next(&mut vp9_picture)
		.map_err(Error::from)?;

	record_decode_commands(
		device,
		command_buffer,
		session,
		bitstream,
		plan,
		initialize_images,
		pending_acquire_slot,
		release_for_readback,
		output,
		output_layer,
		dpb,
		&begin_info,
		&decode_info,
	)
}

#[allow(clippy::too_many_arguments)]
fn record_decode_commands(
	device: &vk::Device,
	command_buffer: vk::CommandBuffer,
	session: &DecodeSession,
	bitstream: &DecodeBitstream,
	plan: &PicturePlan,
	initialize_images: bool,
	pending_acquire_slot: Option<u32>,
	release_for_readback: bool,
	output: &super::DecodeImage,
	output_layer: u32,
	dpb: &super::DecodeImage,
	begin_info: &vk::VideoBeginCodingInfoKHR<'_>,
	decode_info: &vk::VideoDecodeInfoKHR<'_>,
) -> Result<()> {
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
	// safety: every chained pointer borrows storage that remains live and
	// immovable through command recording on this session's device.
	unsafe {
		device.cmd_reset_query_pool(command_buffer, session.result_status_pool, 0, 1)?;
		device.cmd_pipeline_barrier2(command_buffer, &dependency)?;
		device.cmd_begin_video_coding_khr(command_buffer, begin_info)?;
	}
	if plan.reset_dpb {
		let control = vk::VideoCodingControlInfoKHR::new()
			.set_flags(vk::VideoCodingControlFlagBitsKHR::RESET_BIT_KHR);
		// safety: the session is inside a coding scope at a VP9 keyframe boundary.
		unsafe {
			device.cmd_control_video_coding_khr(command_buffer, &control)?;
		}
	}
	// safety: all resources and the VP9 pNext payload match the selected profile.
	unsafe {
		device.cmd_begin_query(
			command_buffer,
			session.result_status_pool,
			0,
			vk::QueryControlFlags::empty(),
		)?;
		device.cmd_decode_video_khr(command_buffer, decode_info)?;
		device.cmd_end_query(command_buffer, session.result_status_pool, 0)?;
		device.cmd_end_video_coding_khr(command_buffer, &vk::VideoEndCodingInfoKHR::new())?;
	}
	if release_for_readback {
		record_release_for_readback(device, command_buffer, session, output, output_layer, dpb)?;
	}
	Ok(())
}

fn record_release_for_readback(
	device: &vk::Device,
	command_buffer: vk::CommandBuffer,
	session: &DecodeSession,
	output: &super::DecodeImage,
	output_layer: u32,
	dpb: &super::DecodeImage,
) -> Result<()> {
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
	let dependency = vk::DependencyInfo::new()
		.set_image_memory_barriers(std::slice::from_ref(&release))
		.map_err(Error::from)?;
	// safety: decode owns this exact output layer and readback records the acquire.
	unsafe {
		device.cmd_pipeline_barrier2(command_buffer, &dependency)?;
	}
	Ok(())
}

fn std_color(value: video::vp9::Vp9ColorConfig) -> vk::StdVideoVP9ColorConfig {
	let mut flags = vk::StdVideoVP9ColorConfigFlags::new();
	flags.set_color_range(u32::from(value.full_range));
	flags.set_reserved(0);
	vk::StdVideoVP9ColorConfig {
		flags,
		bit_depth: value.bit_depth,
		subsampling_x: u8::from(value.subsampling_x),
		subsampling_y: u8::from(value.subsampling_y),
		reserved1: 0,
		color_space: vk::StdVideoVP9ColorSpace(value.color_space as i32),
	}
}

fn std_loop_filter(value: video::vp9::Vp9LoopFilter) -> vk::StdVideoVP9LoopFilter {
	let mut flags = vk::StdVideoVP9LoopFilterFlags::new();
	flags.set_loop_filter_delta_enabled(u32::from(value.delta_enabled));
	flags.set_loop_filter_delta_update(u32::from(value.delta_update));
	flags.set_reserved(0);
	vk::StdVideoVP9LoopFilter {
		flags,
		loop_filter_level: value.level,
		loop_filter_sharpness: value.sharpness,
		update_ref_delta: value.update_reference_delta,
		loop_filter_ref_deltas: value.reference_deltas,
		update_mode_delta: value.update_mode_delta,
		loop_filter_mode_deltas: value.mode_deltas,
	}
}

fn std_segmentation(value: &video::vp9::Vp9Segmentation) -> vk::StdVideoVP9Segmentation {
	let mut flags = vk::StdVideoVP9SegmentationFlags::new();
	flags.set_segmentation_update_map(u32::from(value.update_map));
	flags.set_segmentation_temporal_update(u32::from(value.temporal_update));
	flags.set_segmentation_update_data(u32::from(value.update_data));
	flags.set_segmentation_abs_or_delta_update(u32::from(value.absolute_or_delta_update));
	flags.set_reserved(0);
	vk::StdVideoVP9Segmentation {
		flags,
		segmentation_tree_probs: value.tree_probabilities,
		segmentation_pred_prob: value.prediction_probabilities,
		feature_enabled: value.feature_enabled,
		feature_data: value.feature_data,
	}
}

const fn std_frame_type(value: video::vp9::Vp9FrameType) -> vk::StdVideoVP9FrameType {
	match value {
		video::vp9::Vp9FrameType::Key => vk::StdVideoVP9FrameType::KEY,
		video::vp9::Vp9FrameType::Inter => vk::StdVideoVP9FrameType::NON_KEY,
	}
}

const fn std_profile(value: u8) -> vk::StdVideoVP9Profile {
	match value {
		0 => vk::StdVideoVP9Profile::_0,
		1 => vk::StdVideoVP9Profile::_1,
		2 => vk::StdVideoVP9Profile::_2,
		3 => vk::StdVideoVP9Profile::_3,
		_ => vk::StdVideoVP9Profile::INVALID,
	}
}

const fn std_interpolation_filter(
	value: video::vp9::Vp9InterpolationFilter,
) -> vk::StdVideoVP9InterpolationFilter {
	match value {
		video::vp9::Vp9InterpolationFilter::EightTapSmooth => {
			vk::StdVideoVP9InterpolationFilter::EIGHTTAP_SMOOTH
		}
		video::vp9::Vp9InterpolationFilter::EightTap => vk::StdVideoVP9InterpolationFilter::EIGHTTAP,
		video::vp9::Vp9InterpolationFilter::EightTapSharp => {
			vk::StdVideoVP9InterpolationFilter::EIGHTTAP_SHARP
		}
		video::vp9::Vp9InterpolationFilter::Bilinear => vk::StdVideoVP9InterpolationFilter::BILINEAR,
		video::vp9::Vp9InterpolationFilter::Switchable => {
			vk::StdVideoVP9InterpolationFilter::SWITCHABLE
		}
	}
}
