//! AV1 decoded-picture-buffer planning and Vulkan standard-video lowering.

use crate::{Error, Result, video};

use crate::runtime::{
	device::video::DecodeBitstream, device::video::DecodeImageSet, device::video::DecodeSession,
	device::video::decode_image_barrier, device::video::decode_image_subresource,
};

const LOGICAL_REFERENCE_COUNT: usize = 8;
const REFERENCE_NAME_COUNT: usize = 7;
const MAX_TILE_COLUMNS: usize = 64;
const MAX_TILE_ROWS: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ReferenceRecord {
	pub frame_type: video::Av1FrameType,
	pub reference_frame_sign_bias: u8,
	pub order_hint: u8,
	pub saved_order_hints: [u8; LOGICAL_REFERENCE_COUNT],
	pub disable_frame_end_update_cdf: bool,
	pub segmentation_enabled: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct DpbState {
	logical_to_physical: [Option<u32>; LOGICAL_REFERENCE_COUNT],
	physical: Vec<Option<ReferenceRecord>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct PicturePlan {
	pub setup_slot: Option<u32>,
	pub show_existing_slot: Option<u32>,
	pub reference_name_slot_indices: [i32; REFERENCE_NAME_COUNT],
	pub active_references: Vec<(u32, ReferenceRecord)>,
	pub reset_dpb: bool,
}

impl DpbState {
	pub fn new(slot_count: u32) -> Result<Self> {
		if slot_count == 0 || slot_count > 16 {
			return Err(Error::invalid_argument(
				"AV1 DPB state requires 1..=16 physical slots",
			));
		}
		Ok(Self {
			logical_to_physical: [None; LOGICAL_REFERENCE_COUNT],
			physical: vec![None; slot_count as usize],
		})
	}

	pub fn plan_with_unavailable(
		&mut self,
		frame: &video::Av1FrameHeader,
		unavailable: &[bool],
	) -> Result<PicturePlan> {
		if !unavailable.is_empty() && unavailable.len() != self.physical.len() {
			return Err(Error::invalid_argument(
				"AV1 unavailable-slot mask does not match DPB capacity",
			));
		}
		let mut next = self.clone();
		let plan = next.plan_in_place(frame, unavailable)?;
		*self = next;
		Ok(plan)
	}

	fn plan_in_place(
		&mut self,
		frame: &video::Av1FrameHeader,
		unavailable: &[bool],
	) -> Result<PicturePlan> {
		if frame.show_existing_frame {
			let logical = usize::from(frame.frame_to_show_map_idx);
			let slot = self
				.logical_to_physical
				.get(logical)
				.copied()
				.flatten()
				.ok_or_else(|| {
					Error::data_loss("AV1 show-existing frame is absent from the reference map")
				})?;
			return Ok(PicturePlan {
				setup_slot: None,
				show_existing_slot: Some(slot),
				reference_name_slot_indices: [-1; REFERENCE_NAME_COUNT],
				active_references: Vec::new(),
				reset_dpb: false,
			});
		}

		let reset_dpb = frame.frame_type == video::Av1FrameType::Key;
		if reset_dpb {
			self.logical_to_physical.fill(None);
			self.physical.fill(None);
		}

		let mut reference_name_slot_indices = [-1; REFERENCE_NAME_COUNT];
		let mut protected = vec![false; self.physical.len()];
		for (name, logical) in frame
			.reference_name_slot_indices
			.iter()
			.copied()
			.enumerate()
		{
			if logical < 0 {
				continue;
			}
			let logical = usize::try_from(logical)
				.map_err(|_| Error::data_loss("AV1 logical reference index is negative"))?;
			let physical = self
				.logical_to_physical
				.get(logical)
				.copied()
				.flatten()
				.ok_or_else(|| Error::data_loss("AV1 reference is absent from the DPB"))?;
			let physical_index = usize::try_from(physical)
				.map_err(|_| Error::internal("AV1 physical reference index exceeds usize"))?;
			let state = self
				.physical
				.get(physical_index)
				.ok_or_else(|| Error::internal("AV1 logical reference exceeds physical DPB capacity"))?;
			if state.is_none() {
				return Err(Error::internal(
					"AV1 logical reference points at an empty physical slot",
				));
			}
			protected[physical_index] = true;
			reference_name_slot_indices[name] = i32::try_from(physical)
				.map_err(|_| Error::internal("AV1 physical reference index exceeds i32"))?;
		}

		let mapped_after_refresh = |physical: u32| {
			self
				.logical_to_physical
				.iter()
				.enumerate()
				.any(|(logical, mapped)| {
					*mapped == Some(physical) && frame.refresh_frame_flags & (1 << logical) == 0
				})
		};
		let setup_index = self
			.physical
			.iter()
			.enumerate()
			.find(|(index, state)| {
				state.is_none() && !protected[*index] && !unavailable.get(*index).copied().unwrap_or(false)
			})
			.map(|(index, _)| index)
			.or_else(|| {
				self.physical.iter().enumerate().find_map(|(index, _)| {
					let physical = u32::try_from(index).ok()?;
					(!protected[index]
						&& !unavailable.get(index).copied().unwrap_or(false)
						&& !mapped_after_refresh(physical))
					.then_some(index)
				})
			})
			.ok_or_else(|| Error::resource_exhausted("AV1 DPB has no recyclable physical slot"))?;
		let setup_slot =
			u32::try_from(setup_index).map_err(|_| Error::internal("AV1 setup slot exceeds u32"))?;

		let mut active_references = Vec::new();
		active_references
			.try_reserve_exact(protected.iter().filter(|value| **value).count())
			.map_err(|_| Error::resource_exhausted("AV1 reference allocation failed"))?;
		for (index, is_protected) in protected.into_iter().enumerate() {
			if !is_protected {
				continue;
			}
			let record = self.physical[index]
				.ok_or_else(|| Error::internal("AV1 protected reference points at an empty slot"))?;
			active_references.push((
				u32::try_from(index).map_err(|_| Error::internal("AV1 reference slot exceeds u32"))?,
				record,
			));
		}

		for (logical, mapped) in self.logical_to_physical.iter_mut().enumerate() {
			if *mapped == Some(setup_slot) || frame.refresh_frame_flags & (1 << logical) != 0 {
				*mapped = None;
			}
		}
		let record = ReferenceRecord::from(frame);
		if frame.refresh_frame_flags != 0 {
			self.physical[setup_index] = Some(record);
			for (logical, mapped) in self.logical_to_physical.iter_mut().enumerate() {
				if frame.refresh_frame_flags & (1 << logical) != 0 {
					*mapped = Some(setup_slot);
				}
			}
		} else {
			self.physical[setup_index] = None;
		}
		for (index, state) in self.physical.iter_mut().enumerate() {
			let physical =
				u32::try_from(index).map_err(|_| Error::internal("AV1 physical slot exceeds u32"))?;
			if !self.logical_to_physical.contains(&Some(physical)) {
				*state = None;
			}
		}

		Ok(PicturePlan {
			setup_slot: Some(setup_slot),
			show_existing_slot: None,
			reference_name_slot_indices,
			active_references,
			reset_dpb,
		})
	}
}

impl From<&video::Av1FrameHeader> for ReferenceRecord {
	fn from(frame: &video::Av1FrameHeader) -> Self {
		Self {
			frame_type: frame.frame_type,
			reference_frame_sign_bias: frame.reference_frame_sign_bias,
			order_hint: frame.order_hint,
			saved_order_hints: frame.reference_order_hints,
			disable_frame_end_update_cdf: frame.disable_frame_end_update_cdf,
			segmentation_enabled: frame.segmentation_enabled,
		}
	}
}

pub(super) struct PictureParameters {
	mi_col_starts: [u16; MAX_TILE_COLUMNS + 1],
	mi_row_starts: [u16; MAX_TILE_ROWS + 1],
	width_in_superblocks_minus_1: [u16; MAX_TILE_COLUMNS],
	height_in_superblocks_minus_1: [u16; MAX_TILE_ROWS],
	pub quantization: vk::StdVideoAV1Quantization,
	pub segmentation: vk::StdVideoAV1Segmentation,
	pub loop_filter: vk::StdVideoAV1LoopFilter,
	pub cdef: vk::StdVideoAV1CDEF,
	pub restoration: vk::StdVideoAV1LoopRestoration,
	pub global_motion: vk::StdVideoAV1GlobalMotion,
}

impl PictureParameters {
	pub fn new(sequence: &video::Av1SequenceHeader, frame: &video::Av1FrameHeader) -> Result<Self> {
		if frame.show_existing_frame {
			return Err(Error::invalid_argument(
				"show-existing AV1 headers do not carry decode picture parameters",
			));
		}
		let columns = usize::from(frame.tile_columns);
		let rows = usize::from(frame.tile_rows);
		if columns == 0 || rows == 0 || columns > MAX_TILE_COLUMNS || rows > MAX_TILE_ROWS {
			return Err(Error::data_loss(
				"AV1 tile grid exceeds standard-video limits",
			));
		}

		let mi_columns = sequence
			.coded_width()
			.checked_add(7)
			.and_then(|value| value.checked_div(8))
			.and_then(|value| value.checked_mul(2))
			.ok_or_else(|| Error::out_of_range("AV1 MI column count overflowed"))?;
		let mi_rows = sequence
			.coded_height()
			.checked_add(7)
			.and_then(|value| value.checked_div(8))
			.and_then(|value| value.checked_mul(2))
			.ok_or_else(|| Error::out_of_range("AV1 MI row count overflowed"))?;
		let superblock_shift = if sequence.use_128x128_superblock {
			5
		} else {
			4
		};
		let superblock_columns = ceil_shift(mi_columns, superblock_shift)?;
		let superblock_rows = ceil_shift(mi_rows, superblock_shift)?;
		let tile_width = ceil_div(superblock_columns, u32::from(frame.tile_columns))?;
		let tile_height = ceil_div(superblock_rows, u32::from(frame.tile_rows))?;

		let mut mi_col_starts = [0; MAX_TILE_COLUMNS + 1];
		let mut width_in_superblocks_minus_1 = [0; MAX_TILE_COLUMNS];
		fill_uniform_axis(
			&mut mi_col_starts,
			&mut width_in_superblocks_minus_1,
			columns,
			mi_columns,
			superblock_columns,
			tile_width,
			superblock_shift,
		)?;
		let mut mi_row_starts = [0; MAX_TILE_ROWS + 1];
		let mut height_in_superblocks_minus_1 = [0; MAX_TILE_ROWS];
		fill_uniform_axis(
			&mut mi_row_starts,
			&mut height_in_superblocks_minus_1,
			rows,
			mi_rows,
			superblock_rows,
			tile_height,
			superblock_shift,
		)?;

		let mut feature_enabled = [0_u8; 8];
		for (segment, mask) in feature_enabled.iter_mut().enumerate() {
			for feature in 0..8 {
				*mask |= u8::from(frame.segment_feature_enabled[segment][feature]) << feature;
			}
		}
		let update_ref_delta = frame
			.loop_filter_update_reference_delta
			.iter()
			.enumerate()
			.fold(0_u8, |mask, (index, enabled)| {
				mask | (u8::from(*enabled) << index)
			});
		let update_mode_delta = frame
			.loop_filter_update_mode_delta
			.iter()
			.enumerate()
			.fold(0_u8, |mask, (index, enabled)| {
				mask | (u8::from(*enabled) << index)
			});
		let restoration_types = frame.restoration_types.map(std_restoration_type);
		let mut global_parameters = [[0_i32; 6]; LOGICAL_REFERENCE_COUNT];
		for parameters in &mut global_parameters {
			parameters[2] = 1 << 16;
			parameters[5] = 1 << 16;
		}

		Ok(Self {
			mi_col_starts,
			mi_row_starts,
			width_in_superblocks_minus_1,
			height_in_superblocks_minus_1,
			quantization: vk::StdVideoAV1Quantization {
				flags: {
					let mut flags = vk::StdVideoAV1QuantizationFlags::default();
					flags.set_using_qmatrix(u32::from(frame.using_q_matrix) as _);
					flags.set_diff_uv_delta(u32::from(frame.different_uv_delta) as _);
					flags.set_reserved(0 as _);
					flags
				},
				base_q_idx: frame.base_q_idx,
				delta_qy_dc: frame.delta_q_y_dc,
				delta_qu_dc: frame.delta_q_u_dc,
				delta_qu_ac: frame.delta_q_u_ac,
				delta_qv_dc: frame.delta_q_v_dc,
				delta_qv_ac: frame.delta_q_v_ac,
				qm_y: frame.qm_y,
				qm_u: frame.qm_u,
				qm_v: frame.qm_v,
			},
			segmentation: vk::StdVideoAV1Segmentation {
				feature_enabled,
				feature_data: frame.segment_feature_data,
			},
			loop_filter: vk::StdVideoAV1LoopFilter {
				flags: {
					let mut flags = vk::StdVideoAV1LoopFilterFlags::default();
					flags.set_loop_filter_delta_enabled(u32::from(frame.loop_filter_delta_enabled) as _);
					flags.set_loop_filter_delta_update(u32::from(frame.loop_filter_delta_update) as _);
					flags.set_reserved(0 as _);
					flags
				},
				loop_filter_level: frame.loop_filter_levels,
				loop_filter_sharpness: frame.loop_filter_sharpness,
				update_ref_delta,
				loop_filter_ref_deltas: frame.loop_filter_reference_deltas,
				update_mode_delta,
				loop_filter_mode_deltas: frame.loop_filter_mode_deltas,
			},
			cdef: vk::StdVideoAV1CDEF {
				cdef_damping_minus_3: frame.cdef_damping_minus_3,
				cdef_bits: frame.cdef_bits,
				cdef_y_pri_strength: frame.cdef_y_primary_strength,
				cdef_y_sec_strength: frame.cdef_y_secondary_strength,
				cdef_uv_pri_strength: frame.cdef_uv_primary_strength,
				cdef_uv_sec_strength: frame.cdef_uv_secondary_strength,
			},
			restoration: vk::StdVideoAV1LoopRestoration {
				frame_restoration_type: restoration_types,
				loop_restoration_size: frame.restoration_unit_size_log2_minus_5,
			},
			global_motion: vk::StdVideoAV1GlobalMotion {
				gm_type: [0; LOGICAL_REFERENCE_COUNT],
				gm_params: global_parameters,
			},
		})
	}

	pub fn tile_info(&self, frame: &video::Av1FrameHeader) -> vk::StdVideoAV1TileInfo<'_> {
		let mut value = vk::StdVideoAV1TileInfo::default();
		value.flags.set_uniform_tile_spacing_flag(1);
		value.flags.set_reserved(0);
		value.tile_cols = frame.tile_columns;
		value.tile_rows = frame.tile_rows;
		value.context_update_tile_id = frame.context_update_tile_id;
		value.tile_size_bytes_minus_1 = frame.tile_size_bytes_minus_1;
		value.reserved1 = [0; 7];
		value.p_mi_col_starts = self.mi_col_starts.as_ptr();
		value.p_mi_row_starts = self.mi_row_starts.as_ptr();
		value.p_width_in_sbs_minus1 = self.width_in_superblocks_minus_1.as_ptr();
		value.p_height_in_sbs_minus1 = self.height_in_superblocks_minus_1.as_ptr();
		value
	}

	pub fn picture_info<'a>(
		&'a self,
		frame: &video::Av1FrameHeader,
		tile_info: &vk::StdVideoAV1TileInfo<'a>,
	) -> vk::StdVideoDecodeAV1PictureInfo<'a> {
		vk::StdVideoDecodeAV1PictureInfo {
			frame_type: std_frame_type(frame.frame_type),
			current_frame_id: 0,
			order_hint: frame.order_hint,
			primary_ref_frame: frame.primary_reference_frame,
			refresh_frame_flags: frame.refresh_frame_flags,
			reserved1: 0,
			interpolation_filter: std_interpolation_filter(frame.interpolation_filter),
			tx_mode: std_transform_mode(frame.transform_mode),
			delta_q_res: frame.delta_q_resolution,
			delta_lf_res: frame.delta_loop_filter_resolution,
			skip_mode_frame: frame.skip_mode_frame,
			coded_denom: frame.coded_denom,
			reserved2: [0; 3],
			order_hints: frame.reference_order_hints,
			expected_frame_id: [0; LOGICAL_REFERENCE_COUNT],
			p_tile_info: tile_info,
			p_quantization: &self.quantization,
			p_segmentation: &self.segmentation,
			p_loop_filter: &self.loop_filter,
			p_cdef: &self.cdef,
			p_loop_restoration: &self.restoration,
			p_global_motion: &self.global_motion,
			p_film_grain: std::ptr::null(),
			..Default::default()
		}
	}
}

pub(super) fn reference_info(record: ReferenceRecord) -> vk::StdVideoDecodeAV1ReferenceInfo {
	vk::StdVideoDecodeAV1ReferenceInfo {
		flags: {
			let mut flags = vk::StdVideoDecodeAV1ReferenceInfoFlags::default();
			flags.set_disable_frame_end_update_cdf(u32::from(record.disable_frame_end_update_cdf) as _);
			flags.set_segmentation_enabled(u32::from(record.segmentation_enabled) as _);
			flags.set_reserved(0 as _);
			flags
		},
		frame_type: std_frame_type(record.frame_type).0 as u8,
		ref_frame_sign_bias: record.reference_frame_sign_bias,
		order_hint: record.order_hint,
		saved_order_hints: record.saved_order_hints,
	}
}

#[allow(clippy::too_many_arguments)]
pub(super) fn record_picture(
	device: &vk::Device,
	command_buffer: vk::CommandBuffer,
	session: &DecodeSession,
	sequence: &video::Av1SequenceHeader,
	frame: &video::Av1FrameHeader,
	frame_header_offset: u32,
	tiles: &video::Av1TileGroup,
	bitstream: &DecodeBitstream,
	plan: &PicturePlan,
	initialize_images: bool,
	pending_acquire_slot: Option<u32>,
	release_for_readback: bool,
) -> Result<()> {
	let setup_slot_index = plan
		.setup_slot
		.ok_or_else(|| Error::internal("coded AV1 picture has no setup slot"))?;
	if plan.show_existing_slot.is_some() {
		return Err(Error::internal(
			"show-existing AV1 plan reached command recording",
		));
	}
	if tiles.tile_offsets.is_empty() || tiles.tile_offsets.len() != tiles.tile_sizes.len() {
		return Err(Error::invalid_argument("AV1 tile metadata is incomplete"));
	}
	let expected_tiles = usize::from(frame.tile_columns)
		.checked_mul(usize::from(frame.tile_rows))
		.ok_or_else(|| Error::out_of_range("AV1 tile-grid size overflowed"))?;
	if tiles.tile_offsets.len() != expected_tiles {
		return Err(Error::invalid_argument(
			"AV1 tile grid does not match tile payload count",
		));
	}
	let frame_header_offset = usize::try_from(frame_header_offset)
		.map_err(|_| Error::out_of_range("AV1 frame-header offset exceeds usize"))?;
	if frame_header_offset >= bitstream.payload_len {
		return Err(Error::invalid_argument(
			"AV1 frame-header offset exceeds the uploaded access unit",
		));
	}
	for (&offset, &size) in tiles.tile_offsets.iter().zip(&tiles.tile_sizes) {
		let end = usize::try_from(offset)
			.ok()
			.and_then(|offset| {
				usize::try_from(size)
					.ok()
					.and_then(|size| offset.checked_add(size))
			})
			.ok_or_else(|| Error::out_of_range("AV1 tile range exceeds usize"))?;
		if size == 0 || end > bitstream.payload_len {
			return Err(Error::invalid_argument(
				"AV1 tile range exceeds the uploaded access unit",
			));
		}
	}

	let parameters = PictureParameters::new(sequence, frame)?;
	let tile_info = parameters.tile_info(frame);
	let std_picture = parameters.picture_info(frame, &tile_info);
	let mut av1_picture = vk::VideoDecodeAV1PictureInfoKHR::default()
		.std_picture_info(&std_picture)
		.reference_name_slot_indices(plan.reference_name_slot_indices)
		.frame_header_offset(
			u32::try_from(frame_header_offset)
				.map_err(|_| Error::out_of_range("AV1 frame-header offset exceeds Vulkan storage"))?,
		)
		.tile_offsets(&tiles.tile_offsets, &tiles.tile_sizes)
		.map_err(Error::from)?;

	let setup_record = ReferenceRecord::from(frame);
	let setup_std_reference = reference_info(setup_record);
	let mut setup_av1_slot =
		vk::VideoDecodeAV1DpbSlotInfoKHR::default().std_reference_info(&setup_std_reference);
	let (output, output_layer, dpb, dpb_layer) = session.picture_images(setup_slot_index)?;
	let extent = vk::Extent2D {
		width: session.coded_extent.width,
		height: session.coded_extent.height,
	};
	let setup_resource = vk::VideoPictureResourceInfoKHR::default()
		.coded_extent(extent)
		.base_array_layer(dpb_layer)
		.image_view_binding(dpb.view);
	let setup_slot = vk::VideoReferenceSlotInfoKHR::default()
		.slot_index(
			i32::try_from(setup_slot_index).map_err(|_| Error::internal("AV1 setup slot exceeds i32"))?,
		)
		.picture_resource(&setup_resource)
		.push_next(&mut setup_av1_slot)
		.map_err(Error::from)?;

	let reference_count = plan.active_references.len();
	let mut std_references = Vec::new();
	let mut av1_reference_slots = Vec::new();
	let mut reference_resources = Vec::new();
	let mut reference_slots = Vec::new();
	std_references
		.try_reserve_exact(reference_count)
		.map_err(|_| Error::resource_exhausted("AV1 standard reference allocation failed"))?;
	av1_reference_slots
		.try_reserve_exact(reference_count)
		.map_err(|_| Error::resource_exhausted("AV1 reference chain allocation failed"))?;
	reference_resources
		.try_reserve_exact(reference_count)
		.map_err(|_| Error::resource_exhausted("AV1 reference resource allocation failed"))?;
	reference_slots
		.try_reserve_exact(reference_count + 1)
		.map_err(|_| Error::resource_exhausted("AV1 reference slot allocation failed"))?;
	for &(slot, record) in &plan.active_references {
		if slot == setup_slot_index {
			return Err(Error::internal(
				"AV1 setup slot aliases an active reference",
			));
		}
		let (_, _, reference_image, reference_layer) = session.picture_images(slot)?;
		std_references.push(reference_info(record));
		av1_reference_slots.push(vk::VideoDecodeAV1DpbSlotInfoKHR::default());
		reference_resources.push(
			vk::VideoPictureResourceInfoKHR::default()
				.coded_extent(extent)
				.base_array_layer(reference_layer)
				.image_view_binding(reference_image.view),
		);
		reference_slots.push(vk::VideoReferenceSlotInfoKHR::default().slot_index(
			i32::try_from(slot).map_err(|_| Error::internal("AV1 reference slot exceeds i32"))?,
		));
	}
	for index in 0..reference_count {
		av1_reference_slots[index].p_std_reference_info = &std_references[index];
		reference_slots[index].p_next =
			(&av1_reference_slots[index] as *const vk::VideoDecodeAV1DpbSlotInfoKHR<'_>).cast();
		reference_slots[index].p_picture_resource = &reference_resources[index];
	}
	let mut inactive_setup = vk::VideoReferenceSlotInfoKHR::default()
		.slot_index(-1)
		.picture_resource(&setup_resource);
	inactive_setup.p_next = setup_slot.p_next;
	reference_slots.push(inactive_setup);
	let begin_info = vk::VideoBeginCodingInfoKHR::default()
		.video_session(session.handle)
		.video_session_parameters(session.parameters)
		.reference_slots(&reference_slots)
		.map_err(Error::from)?;
	let output_resource = vk::VideoPictureResourceInfoKHR::default()
		.coded_extent(extent)
		.base_array_layer(output_layer)
		.image_view_binding(output.view);
	let decode_info = vk::VideoDecodeInfoKHR::default()
		.src_buffer(bitstream.handle)
		.src_buffer_offset(0)
		.src_buffer_range(bitstream.range)
		.dst_picture_resource(output_resource)
		.setup_reference_slot(&setup_slot)
		.reference_slots(&reference_slots[..reference_count])
		.map_err(Error::from)?
		.push_next(&mut av1_picture)
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
				vk::ImageMemoryBarrier2::default()
					.src_stage_mask(vk::PipelineStageFlagBits2::NONE)
					.src_access_mask(vk::AccessFlagBits2::NONE)
					.dst_stage_mask(vk::PipelineStageFlagBits2::VIDEO_DECODE_BIT_KHR)
					.dst_access_mask(
						vk::AccessFlagBits2::VIDEO_DECODE_READ_BIT_KHR
							| vk::AccessFlagBits2::VIDEO_DECODE_WRITE_BIT_KHR,
					)
					.old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
					.new_layout(layout)
					.src_queue_family_index(compute_family)
					.dst_queue_family_index(decode_family)
					.image(acquired_output.handle)
					.subresource_range(decode_image_subresource(acquired_layer, 1)),
			);
		}
	}
	let bitstream_barriers = [vk::BufferMemoryBarrier2::default()
		.src_stage_mask(vk::PipelineStageFlagBits2::HOST_BIT)
		.src_access_mask(vk::AccessFlagBits2::HOST_WRITE_BIT)
		.dst_stage_mask(vk::PipelineStageFlagBits2::VIDEO_DECODE_BIT_KHR)
		.dst_access_mask(vk::AccessFlagBits2::VIDEO_DECODE_READ_BIT_KHR)
		.src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
		.dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
		.buffer(bitstream.handle)
		.offset(0)
		.size(bitstream.range)];
	let memory_barriers = (!initialize_images)
		.then(|| {
			vk::MemoryBarrier2::default()
				.src_stage_mask(vk::PipelineStageFlagBits2::VIDEO_DECODE_BIT_KHR)
				.src_access_mask(vk::AccessFlagBits2::VIDEO_DECODE_WRITE_BIT_KHR)
				.dst_stage_mask(vk::PipelineStageFlagBits2::VIDEO_DECODE_BIT_KHR)
				.dst_access_mask(
					vk::AccessFlagBits2::VIDEO_DECODE_READ_BIT_KHR
						| vk::AccessFlagBits2::VIDEO_DECODE_WRITE_BIT_KHR,
				)
		})
		.into_iter()
		.collect::<Vec<_>>();
	let dependency = vk::DependencyInfo::default()
		.memory_barriers(&memory_barriers)
		.map_err(Error::from)?
		.buffer_memory_barriers(&bitstream_barriers)
		.map_err(Error::from)?
		.image_memory_barriers(&image_barriers)
		.map_err(Error::from)?;
	// safety: every pointer chained into the coding scope borrows stack or vector
	// storage that remains live and immovable through the decode command call.
	unsafe {
		device.cmd_reset_query_pool(command_buffer, session.result_status_pool, 0, 1)?;
		device
			.cmd_pipeline_barrier2(command_buffer, &dependency)
			.map_err(Error::from)?;
		device
			.cmd_begin_video_coding_khr(command_buffer, std::ptr::from_ref(&begin_info))
			.map_err(Error::from)?;
	}
	if plan.reset_dpb {
		let control = vk::VideoCodingControlInfoKHR::default()
			.flags(vk::VideoCodingControlFlagBitsKHR::RESET_BIT_KHR);
		// safety: the session is inside a coding scope and this plan starts at an
		// AV1 key-frame reset boundary.
		unsafe {
			device
				.cmd_control_video_coding_khr(command_buffer, &control)
				.map_err(Error::from)?;
		}
	}
	// safety: source, destination, setup, and distinct active references match the
	// selected AV1 profile and remain live through command recording.
	unsafe {
		device.cmd_begin_query(
			command_buffer,
			session.result_status_pool,
			0,
			vk::QueryControlFlags::empty(),
		)?;
		device
			.cmd_decode_video_khr(command_buffer, &decode_info)
			.map_err(Error::from)?;
		device.cmd_end_query(command_buffer, session.result_status_pool, 0)?;
		device.cmd_end_video_coding_khr(command_buffer, &vk::VideoEndCodingInfoKHR::default())?;
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
		let release = vk::ImageMemoryBarrier2::default()
			.src_stage_mask(vk::PipelineStageFlagBits2::VIDEO_DECODE_BIT_KHR)
			.src_access_mask(
				vk::AccessFlagBits2::VIDEO_DECODE_READ_BIT_KHR
					| vk::AccessFlagBits2::VIDEO_DECODE_WRITE_BIT_KHR,
			)
			.dst_stage_mask(if same_family {
				vk::PipelineStageFlagBits2::TRANSFER_BIT
			} else {
				vk::PipelineStageFlagBits2::NONE
			})
			.dst_access_mask(if same_family {
				vk::AccessFlagBits2::TRANSFER_READ_BIT
			} else {
				vk::AccessFlagBits2::NONE
			})
			.old_layout(decoded_layout)
			.new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
			.src_queue_family_index(if same_family {
				vk::QUEUE_FAMILY_IGNORED
			} else {
				decode_family
			})
			.dst_queue_family_index(if same_family {
				vk::QUEUE_FAMILY_IGNORED
			} else {
				compute_family
			})
			.image(output.handle)
			.subresource_range(decode_image_subresource(output_layer, 1));
		let release_dependency =
			vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&release));
		// safety: decode owns this exact output layer; the readback command records
		// the matching acquire/copy transition before host observation.
		unsafe {
			device
				.cmd_pipeline_barrier2(command_buffer, &(release_dependency).map_err(Error::from)?)
				.map_err(Error::from)?;
		}
	}
	Ok(())
}

fn fill_uniform_axis<const STARTS: usize, const WIDTHS: usize>(
	starts: &mut [u16; STARTS],
	widths_minus_1: &mut [u16; WIDTHS],
	tile_count: usize,
	mi_count: u32,
	superblock_count: u32,
	tile_size: u32,
	superblock_shift: u32,
) -> Result<()> {
	for tile in 0..=tile_count {
		let tile_u32 =
			u32::try_from(tile).map_err(|_| Error::internal("AV1 tile index exceeds u32"))?;
		let start_superblock = tile_u32.saturating_mul(tile_size).min(superblock_count);
		let start_mi = if tile == tile_count {
			mi_count
		} else {
			start_superblock
				.checked_shl(superblock_shift)
				.ok_or_else(|| Error::out_of_range("AV1 tile MI start overflowed"))?
		};
		starts[tile] =
			u16::try_from(start_mi).map_err(|_| Error::data_loss("AV1 tile MI start exceeds u16"))?;
		if tile > 0 {
			let previous = (tile_u32 - 1)
				.saturating_mul(tile_size)
				.min(superblock_count);
			widths_minus_1[tile - 1] =
				u16::try_from(start_superblock.saturating_sub(previous).saturating_sub(1))
					.map_err(|_| Error::data_loss("AV1 tile superblock span exceeds u16"))?;
		}
	}
	Ok(())
}

fn ceil_shift(value: u32, shift: u32) -> Result<u32> {
	let add = (1_u32 << shift) - 1;
	value
		.checked_add(add)
		.map(|sum| sum >> shift)
		.ok_or_else(|| Error::out_of_range("AV1 superblock count overflowed"))
}

fn ceil_div(value: u32, divisor: u32) -> Result<u32> {
	if divisor == 0 {
		return Err(Error::data_loss("AV1 tile divisor is zero"));
	}
	value
		.checked_add(divisor - 1)
		.map(|sum| sum / divisor)
		.ok_or_else(|| Error::out_of_range("AV1 tile division overflowed"))
}

const fn std_frame_type(value: video::Av1FrameType) -> vk::StdVideoAV1FrameType {
	match value {
		video::Av1FrameType::Key => vk::StdVideoAV1FrameType::KEY,
		video::Av1FrameType::Inter => vk::StdVideoAV1FrameType::INTER,
		video::Av1FrameType::IntraOnly => vk::StdVideoAV1FrameType::INTRA_ONLY,
		video::Av1FrameType::Switch => vk::StdVideoAV1FrameType::SWITCH,
	}
}

const fn std_interpolation_filter(
	value: video::Av1InterpolationFilter,
) -> vk::StdVideoAV1InterpolationFilter {
	match value {
		video::Av1InterpolationFilter::EightTap => vk::StdVideoAV1InterpolationFilter::EIGHTTAP,
		video::Av1InterpolationFilter::EightTapSmooth => {
			vk::StdVideoAV1InterpolationFilter::EIGHTTAP_SMOOTH
		}
		video::Av1InterpolationFilter::EightTapSharp => {
			vk::StdVideoAV1InterpolationFilter::EIGHTTAP_SHARP
		}
		video::Av1InterpolationFilter::Bilinear => vk::StdVideoAV1InterpolationFilter::BILINEAR,
		video::Av1InterpolationFilter::Switchable => vk::StdVideoAV1InterpolationFilter::SWITCHABLE,
	}
}

const fn std_transform_mode(value: video::Av1TransformMode) -> vk::StdVideoAV1TxMode {
	match value {
		video::Av1TransformMode::Largest => vk::StdVideoAV1TxMode::LARGEST,
		video::Av1TransformMode::Select => vk::StdVideoAV1TxMode::SELECT,
	}
}

const fn std_restoration_type(
	value: video::Av1RestorationType,
) -> vk::StdVideoAV1FrameRestorationType {
	match value {
		video::Av1RestorationType::None => vk::StdVideoAV1FrameRestorationType::NONE,
		video::Av1RestorationType::Wiener => vk::StdVideoAV1FrameRestorationType::WIENER,
		video::Av1RestorationType::Sgrproj => vk::StdVideoAV1FrameRestorationType::SGRPROJ,
		video::Av1RestorationType::Switchable => vk::StdVideoAV1FrameRestorationType::SWITCHABLE,
	}
}
