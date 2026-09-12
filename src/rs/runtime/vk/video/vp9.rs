//! VP9 decoded-picture-buffer planning and Vulkan standard-video lowering.

use crate::{Error, Result, video};

use super::{
	DecodeBitstream, DecodeImageSet, DecodeSession, decode_image_barrier, decode_image_subresource,
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
				.ok_or_else(|| {
					Error::data_loss("VP9 show-existing reference is absent from the DPB")
				})?;
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
				let slot = self.logical_to_physical[usize::from(logical)].ok_or_else(|| {
					Error::data_loss("VP9 named reference is absent from the DPB")
				})?;
				let index = usize::try_from(slot)
					.map_err(|_| Error::internal("VP9 physical slot exceeds usize"))?;
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
				named[name] = i32::try_from(slot)
					.map_err(|_| Error::internal("VP9 reference slot exceeds i32"))?;
			}
		}

		let remains_mapped = |slot: u32| {
			self.logical_to_physical
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
				value.is_none()
					&& !protected[*index]
					&& !unavailable.get(*index).copied().unwrap_or(false)
			})
			.map(|(index, _)| index)
			.or_else(|| {
				self.physical_extents
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
		let setup_slot = u32::try_from(setup_index)
			.map_err(|_| Error::internal("VP9 setup slot exceeds u32"))?;

		let mut active_references = Vec::new();
		for (index, active) in protected.into_iter().enumerate() {
			if active {
				active_references.push(
					u32::try_from(index)
						.map_err(|_| Error::internal("VP9 active reference exceeds u32"))?,
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
			let slot = u32::try_from(index)
				.map_err(|_| Error::internal("VP9 physical slot exceeds u32"))?;
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
	pub color: ash_vp9::vk::native::StdVideoVP9ColorConfig,
	pub loop_filter: ash_vp9::vk::native::StdVideoVP9LoopFilter,
	pub segmentation: ash_vp9::vk::native::StdVideoVP9Segmentation,
}

impl PictureParameters {
	pub fn new(picture: &video::Vp9Picture) -> Self {
		Self {
			color: std_color(picture.color),
			loop_filter: std_loop_filter(picture.loop_filter),
			segmentation: std_segmentation(&picture.segmentation),
		}
	}

	pub fn picture_info(
		&self,
		picture: &video::Vp9Picture,
	) -> ash_vp9::vk::native::StdVideoDecodeVP9PictureInfo {
		ash_vp9::vk::native::StdVideoDecodeVP9PictureInfo {
			flags: ash_vp9::vk::native::StdVideoDecodeVP9PictureInfoFlags {
				_bitfield_align_1: [],
				_bitfield_1: ash_vp9::vk::native::StdVideoDecodeVP9PictureInfoFlags::new_bitfield_1(
					u32::from(picture.error_resilient_mode),
					u32::from(picture.intra_only),
					u32::from(picture.allow_high_precision_motion_vectors),
					u32::from(picture.refresh_frame_context),
					u32::from(picture.frame_parallel_decoding_mode),
					u32::from(picture.segmentation_enabled),
					u32::from(picture.show_frame),
					u32::from(picture.use_previous_frame_motion_vectors),
					0,
				),
			},
			profile: std_profile(picture.profile),
			frame_type: std_frame_type(picture.frame_type),
			frame_context_idx: picture.frame_context_index,
			reset_frame_context: picture.reset_frame_context,
			refresh_frame_flags: picture.refresh_frame_flags,
			ref_frame_sign_bias_mask: picture.reference_frame_sign_bias_mask,
			interpolation_filter: std_interpolation_filter(picture.interpolation_filter),
			base_q_idx: picture.base_q_index,
			delta_q_y_dc: picture.delta_q_y_dc,
			delta_q_uv_dc: picture.delta_q_uv_dc,
			delta_q_uv_ac: picture.delta_q_uv_ac,
			tile_cols_log2: picture.tile_columns_log2,
			tile_rows_log2: picture.tile_rows_log2,
			reserved1: [0; 3],
			pColorConfig: &self.color,
			pLoopFilter: &self.loop_filter,
			pSegmentation: &self.segmentation,
		}
	}
}

#[allow(clippy::too_many_arguments)]
pub(super) fn record_picture(
	device: &ash::Device,
	loader: &ash::khr::video_queue::Device,
	decode_loader: &ash::khr::video_decode_queue::Device,
	command_buffer: ash::vk::CommandBuffer,
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
	let mut vp9_picture = ash_vp9::vk::VideoDecodeVP9PictureInfoKHR::default()
		.std_picture_info(&std_picture)
		.reference_name_slot_indices(plan.reference_name_slot_indices)
		.uncompressed_header_offset(picture.uncompressed_header_offset)
		.compressed_header_offset(picture.compressed_header_offset)
		.tiles_offset(picture.tiles_offset);

	let (output, output_layer, dpb, dpb_layer) = session.picture_images(setup_slot_index)?;
	let extent = ash::vk::Extent2D {
		width: session.coded_extent.width,
		height: session.coded_extent.height,
	};
	let setup_resource = ash::vk::VideoPictureResourceInfoKHR::default()
		.coded_extent(extent)
		.base_array_layer(dpb_layer)
		.image_view_binding(dpb.view);
	let setup_slot = ash::vk::VideoReferenceSlotInfoKHR::default()
		.slot_index(
			i32::try_from(setup_slot_index)
				.map_err(|_| Error::internal("VP9 setup slot exceeds i32"))?,
		)
		.picture_resource(&setup_resource);

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
			ash::vk::VideoPictureResourceInfoKHR::default()
				.coded_extent(extent)
				.base_array_layer(layer)
				.image_view_binding(image.view),
		);
		reference_slots.push(ash::vk::VideoReferenceSlotInfoKHR::default().slot_index(
			i32::try_from(slot).map_err(|_| Error::internal("VP9 reference slot exceeds i32"))?,
		));
	}
	for index in 0..reference_count {
		reference_slots[index].p_picture_resource = &reference_resources[index];
	}
	let mut inactive_setup = setup_slot;
	inactive_setup.slot_index = -1;
	reference_slots.push(inactive_setup);
	let begin_info = ash::vk::VideoBeginCodingInfoKHR::default()
		.video_session(session.handle)
		.reference_slots(&reference_slots);
	let output_resource = ash::vk::VideoPictureResourceInfoKHR::default()
		.coded_extent(extent)
		.base_array_layer(output_layer)
		.image_view_binding(output.view);
	let mut decode_info = ash::vk::VideoDecodeInfoKHR::default()
		.src_buffer(bitstream.handle)
		.src_buffer_offset(0)
		.src_buffer_range(bitstream.range)
		.dst_picture_resource(output_resource)
		.setup_reference_slot(&setup_slot)
		.reference_slots(&reference_slots[..reference_count]);
	decode_info.p_next = std::ptr::from_mut(&mut vp9_picture).cast();

	record_decode_commands(
		device,
		loader,
		decode_loader,
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
	device: &ash::Device,
	loader: &ash::khr::video_queue::Device,
	decode_loader: &ash::khr::video_decode_queue::Device,
	command_buffer: ash::vk::CommandBuffer,
	session: &DecodeSession,
	bitstream: &DecodeBitstream,
	plan: &PicturePlan,
	initialize_images: bool,
	pending_acquire_slot: Option<u32>,
	release_for_readback: bool,
	output: &super::DecodeImage,
	output_layer: u32,
	dpb: &super::DecodeImage,
	begin_info: &ash::vk::VideoBeginCodingInfoKHR<'_>,
	decode_info: &ash::vk::VideoDecodeInfoKHR<'_>,
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
	// SAFETY: every chained pointer borrows storage that remains live and
	// immovable through command recording; the released Ash loader owns calls.
	unsafe {
		device.cmd_reset_query_pool(command_buffer, session.result_status_pool, 0, 1);
		device.cmd_pipeline_barrier2(command_buffer, &dependency);
		(loader.fp().cmd_begin_video_coding_khr)(command_buffer, begin_info);
	}
	if plan.reset_dpb {
		let control = ash::vk::VideoCodingControlInfoKHR::default()
			.flags(ash::vk::VideoCodingControlFlagsKHR::RESET);
		// SAFETY: the session is inside a coding scope at a VP9 keyframe boundary.
		unsafe {
			(loader.fp().cmd_control_video_coding_khr)(command_buffer, &control);
		}
	}
	// SAFETY: all resources and the VP9 pNext payload match the selected profile.
	unsafe {
		device.cmd_begin_query(
			command_buffer,
			session.result_status_pool,
			0,
			ash::vk::QueryControlFlags::empty(),
		);
		(decode_loader.fp().cmd_decode_video_khr)(command_buffer, decode_info);
		device.cmd_end_query(command_buffer, session.result_status_pool, 0);
		(loader.fp().cmd_end_video_coding_khr)(
			command_buffer,
			&ash::vk::VideoEndCodingInfoKHR::default(),
		);
	}
	if release_for_readback {
		record_release_for_readback(device, command_buffer, session, output, output_layer, dpb)?;
	}
	Ok(())
}

fn record_release_for_readback(
	device: &ash::Device,
	command_buffer: ash::vk::CommandBuffer,
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
	let dependency =
		ash::vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&release));
	// SAFETY: decode owns this exact output layer and readback records the acquire.
	unsafe {
		device.cmd_pipeline_barrier2(command_buffer, &dependency);
	}
	Ok(())
}

fn std_color(value: video::vp9::Vp9ColorConfig) -> ash_vp9::vk::native::StdVideoVP9ColorConfig {
	ash_vp9::vk::native::StdVideoVP9ColorConfig {
		flags: ash_vp9::vk::native::StdVideoVP9ColorConfigFlags {
			_bitfield_align_1: [],
			_bitfield_1: ash_vp9::vk::native::StdVideoVP9ColorConfigFlags::new_bitfield_1(
				u32::from(value.full_range),
				0,
			),
		},
		BitDepth: value.bit_depth,
		subsampling_x: u8::from(value.subsampling_x),
		subsampling_y: u8::from(value.subsampling_y),
		reserved1: 0,
		color_space: u32::from(value.color_space),
	}
}

fn std_loop_filter(value: video::vp9::Vp9LoopFilter) -> ash_vp9::vk::native::StdVideoVP9LoopFilter {
	ash_vp9::vk::native::StdVideoVP9LoopFilter {
		flags: ash_vp9::vk::native::StdVideoVP9LoopFilterFlags {
			_bitfield_align_1: [],
			_bitfield_1: ash_vp9::vk::native::StdVideoVP9LoopFilterFlags::new_bitfield_1(
				u32::from(value.delta_enabled),
				u32::from(value.delta_update),
				0,
			),
		},
		loop_filter_level: value.level,
		loop_filter_sharpness: value.sharpness,
		update_ref_delta: value.update_reference_delta,
		loop_filter_ref_deltas: value.reference_deltas,
		update_mode_delta: value.update_mode_delta,
		loop_filter_mode_deltas: value.mode_deltas,
	}
}

fn std_segmentation(
	value: &video::vp9::Vp9Segmentation,
) -> ash_vp9::vk::native::StdVideoVP9Segmentation {
	ash_vp9::vk::native::StdVideoVP9Segmentation {
		flags: ash_vp9::vk::native::StdVideoVP9SegmentationFlags {
			_bitfield_align_1: [],
			_bitfield_1: ash_vp9::vk::native::StdVideoVP9SegmentationFlags::new_bitfield_1(
				u32::from(value.update_map),
				u32::from(value.temporal_update),
				u32::from(value.update_data),
				u32::from(value.absolute_or_delta_update),
				0,
			),
		},
		segmentation_tree_probs: value.tree_probabilities,
		segmentation_pred_prob: value.prediction_probabilities,
		FeatureEnabled: value.feature_enabled,
		FeatureData: value.feature_data,
	}
}

const fn std_frame_type(
	value: video::vp9::Vp9FrameType,
) -> ash_vp9::vk::native::StdVideoVP9FrameType {
	match value {
		video::vp9::Vp9FrameType::Key => {
			ash_vp9::vk::native::StdVideoVP9FrameType_STD_VIDEO_VP9_FRAME_TYPE_KEY
		}
		video::vp9::Vp9FrameType::Inter => {
			ash_vp9::vk::native::StdVideoVP9FrameType_STD_VIDEO_VP9_FRAME_TYPE_NON_KEY
		}
	}
}

const fn std_profile(value: u8) -> ash_vp9::vk::native::StdVideoVP9Profile {
	match value {
		0 => ash_vp9::vk::native::StdVideoVP9Profile_STD_VIDEO_VP9_PROFILE_0,
		1 => ash_vp9::vk::native::StdVideoVP9Profile_STD_VIDEO_VP9_PROFILE_1,
		2 => ash_vp9::vk::native::StdVideoVP9Profile_STD_VIDEO_VP9_PROFILE_2,
		3 => ash_vp9::vk::native::StdVideoVP9Profile_STD_VIDEO_VP9_PROFILE_3,
		_ => ash_vp9::vk::native::StdVideoVP9Profile_STD_VIDEO_VP9_PROFILE_INVALID,
	}
}

const fn std_interpolation_filter(
	value: video::vp9::Vp9InterpolationFilter,
) -> ash_vp9::vk::native::StdVideoVP9InterpolationFilter {
	match value {
		video::vp9::Vp9InterpolationFilter::EightTapSmooth => ash_vp9::vk::native::StdVideoVP9InterpolationFilter_STD_VIDEO_VP9_INTERPOLATION_FILTER_EIGHTTAP_SMOOTH,
		video::vp9::Vp9InterpolationFilter::EightTap => ash_vp9::vk::native::StdVideoVP9InterpolationFilter_STD_VIDEO_VP9_INTERPOLATION_FILTER_EIGHTTAP,
		video::vp9::Vp9InterpolationFilter::EightTapSharp => ash_vp9::vk::native::StdVideoVP9InterpolationFilter_STD_VIDEO_VP9_INTERPOLATION_FILTER_EIGHTTAP_SHARP,
		video::vp9::Vp9InterpolationFilter::Bilinear => ash_vp9::vk::native::StdVideoVP9InterpolationFilter_STD_VIDEO_VP9_INTERPOLATION_FILTER_BILINEAR,
		video::vp9::Vp9InterpolationFilter::Switchable => ash_vp9::vk::native::StdVideoVP9InterpolationFilter_STD_VIDEO_VP9_INTERPOLATION_FILTER_SWITCHABLE,
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn key() -> video::Vp9Picture {
		video::Vp9Picture {
			frame_width: 1280,
			frame_height: 720,
			render_width: 1280,
			render_height: 720,
			refresh_frame_flags: u8::MAX,
			show_frame: true,
			frame_is_intra: true,
			compressed_header_size: 1,
			tile_count: 1,
			..Default::default()
		}
	}

	#[test]
	fn dpb_maps_named_references_and_show_existing_without_aliasing_setup() -> Result<()> {
		let mut state = DpbState::new(4)?;
		let key_plan = state.plan_with_unavailable(&key(), &[])?;
		assert_eq!(key_plan.setup_slot, Some(0));
		assert!(key_plan.reset_dpb);

		let inter = video::Vp9Picture {
			frame_type: video::vp9::Vp9FrameType::Inter,
			frame_width: 1280,
			frame_height: 720,
			render_width: 1280,
			render_height: 720,
			reference_frame_indices: [0, 3, 7],
			refresh_frame_flags: 1,
			show_frame: true,
			compressed_header_size: 1,
			tile_count: 1,
			..Default::default()
		};
		let inter_plan = state.plan_with_unavailable(&inter, &[])?;
		assert_eq!(inter_plan.reference_name_slot_indices, [0, 0, 0]);
		assert_eq!(inter_plan.active_references, [0]);
		assert_eq!(inter_plan.setup_slot, Some(1));

		let shown = video::Vp9Picture {
			show_existing_frame: true,
			frame_to_show_map_index: 0,
			..Default::default()
		};
		assert_eq!(
			state.plan_with_unavailable(&shown, &[])?.show_existing_slot,
			Some(1)
		);
		Ok(())
	}

	#[test]
	fn dpb_failure_is_transactional_when_all_recyclable_slots_are_leased() -> Result<()> {
		let mut state = DpbState::new(2)?;
		state.plan_with_unavailable(&key(), &[])?;
		let before = state.clone();
		let inter = video::Vp9Picture {
			frame_type: video::vp9::Vp9FrameType::Inter,
			frame_width: 1280,
			frame_height: 720,
			reference_frame_indices: [0; 3],
			refresh_frame_flags: 1,
			..Default::default()
		};
		assert!(state.plan_with_unavailable(&inter, &[false, true]).is_err());
		assert_eq!(state, before);
		Ok(())
	}

	#[test]
	fn lowering_populates_all_pointer_backed_tables() {
		let mut picture = key();
		picture.profile = 0;
		picture.color = video::vp9::Vp9ColorConfig {
			bit_depth: 8,
			subsampling_x: true,
			subsampling_y: true,
			full_range: false,
			color_space: 2,
		};
		picture.loop_filter.level = 31;
		picture.segmentation.feature_enabled[2] = 3;
		picture.segmentation.feature_data[2][1] = -7;
		let parameters = PictureParameters::new(&picture);
		let lowered = parameters.picture_info(&picture);
		assert_eq!(parameters.color.BitDepth, 8);
		assert_eq!(parameters.loop_filter.loop_filter_level, 31);
		assert_eq!(parameters.segmentation.FeatureData[2][1], -7);
		assert_eq!(lowered.pColorConfig, &raw const parameters.color);
		assert_eq!(lowered.pLoopFilter, &raw const parameters.loop_filter);
		assert_eq!(lowered.pSegmentation, &raw const parameters.segmentation);
	}
}
