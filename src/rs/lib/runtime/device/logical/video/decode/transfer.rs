//! Vulkan decode output transfers, readback and fallback presentation conversion.

use crate::runtime::device::logical::command::RecordedBatch;

use super::*;

/// Copy offsets satisfy both plane texel alignment and the four-byte loads
/// used by the fallback shader. Padding is never exposed as a host sample.
struct Yuv420Layout {
	bits: u32,
	interleaved: bool,
	luma_bytes: usize,
	chroma_bytes: usize, // one component
	uv_offset: usize,
	v_offset: usize,
	size: usize,
}

impl Yuv420Layout {
	fn new(format: vk::Format, extent: video::VideoExtent) -> Result<Self> {
		let bits = presentation_sample_bits(format)?;
		let interleaved = matches!(
			format,
			vk::Format::G8_B8R8_2PLANE_420_UNORM
				| vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16
				| vk::Format::G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16
		);
		let (width, height) = (extent.width as usize, extent.height as usize);
		if width == 0 || height == 0 {
			return Err(Error::invalid_argument("video plane extent is empty"));
		}
		let overflow = || Error::out_of_range("video plane layout exceeds host address space");
		let sample_bytes = bits.div_ceil(8) as usize;
		let plane = |w: usize, h: usize| {
			w.checked_mul(h)
				.and_then(|n| n.checked_mul(sample_bytes))
				.ok_or_else(overflow)
		};
		let luma_bytes = plane(width, height)?;
		let chroma_bytes = plane(width.div_ceil(2), height.div_ceil(2))?;
		let uv_offset = luma_bytes
			.checked_next_multiple_of(4)
			.ok_or_else(overflow)?;
		let u_end = uv_offset.checked_add(chroma_bytes).ok_or_else(overflow)?;
		let v_start = if interleaved {
			u_end
		} else {
			u_end.checked_next_multiple_of(4).ok_or_else(overflow)?
		};
		let size = v_start
			.checked_add(chroma_bytes)
			.and_then(|n| n.checked_next_multiple_of(4))
			.ok_or_else(overflow)?;
		Ok(Self {
			bits,
			interleaved,
			luma_bytes,
			chroma_bytes,
			uv_offset,
			v_offset: if interleaved { uv_offset } else { v_start },
			size,
		})
	}
}

pub(in crate::runtime) fn yuv420_copy_layout(
	format: vk::Format,
	extent: video::VideoExtent,
	array_layer: u32,
	origin: [u32; 2],
) -> Result<(usize, Vec<vk::BufferImageCopy>)> {
	let layout = Yuv420Layout::new(format, extent)?;
	let origin = [
		i32::try_from(origin[0]).map_err(|_| Error::out_of_range("video crop x exceeds i32"))?,
		i32::try_from(origin[1]).map_err(|_| Error::out_of_range("video crop y exceeds i32"))?,
	];
	let region = |offset: usize, aspect_mask, width, height, divisor| {
		vk::BufferImageCopy::new()
			.set_buffer_offset(offset as u64)
			.set_image_subresource(vk::ImageSubresourceLayers {
				aspect_mask,
				mip_level: 0,
				base_array_layer: array_layer,
				layer_count: 1,
			})
			.set_image_offset(vk::Offset3D {
				x: origin[0] / divisor,
				y: origin[1] / divisor,
				z: 0,
			})
			.set_image_extent(vk::Extent3D {
				width,
				height,
				depth: 1,
			})
	};
	let cw = extent.width.div_ceil(2);
	let ch = extent.height.div_ceil(2);
	let mut regions = vec![
		region(
			0,
			vk::ImageAspectFlagBits::PLANE_0_BIT,
			extent.width,
			extent.height,
			1,
		),
		region(
			layout.uv_offset,
			vk::ImageAspectFlagBits::PLANE_1_BIT,
			cw,
			ch,
			2,
		),
	];
	if !layout.interleaved {
		regions.push(region(
			layout.v_offset,
			vk::ImageAspectFlagBits::PLANE_2_BIT,
			cw,
			ch,
			2,
		));
	}
	Ok((layout.size, regions))
}

pub(in crate::runtime::device) fn normalize_yuv420(
	format: vk::Format,
	extent: video::VideoExtent,
	mut raw: Vec<u8>,
) -> Result<video::VideoHostPlanes> {
	let layout = Yuv420Layout::new(format, extent)?;
	if raw.len() != layout.size {
		return Err(Error::data_loss(
			"video readback has an unexpected byte count",
		));
	}
	let y = layout.luma_bytes;
	let c = layout.chroma_bytes;
	let uv = layout.uv_offset;
	let v = layout.v_offset;
	if layout.bits == 8 {
		if layout.interleaved {
			// Separate source and destinations so both chroma lanes can be
			// unpacked together in bulk. Luma stays in the readback buffer.
			let mut chroma = Vec::new();
			chroma
				.try_reserve_exact(2 * c)
				.map_err(|_| Error::resource_exhausted("video chroma allocation failed"))?;
			chroma.resize(2 * c, 0);
			let (cb, cr) = chroma.split_at_mut(c);
			for ((cb, cr), pair) in cb
				.iter_mut()
				.zip(cr)
				.zip(raw[uv..uv + 2 * c].as_chunks::<2>().0)
			{
				*cb = pair[0];
				*cr = pair[1];
			}
			raw[y..y + 2 * c].copy_from_slice(&chroma);
		} else {
			raw.copy_within(uv..uv + c, y);
			raw.copy_within(v..v + c, y + c);
		}
		raw.truncate(y + 2 * c);
		return Ok(video::VideoHostPlanes::U8(Arc::new(raw)));
	}
	let shift = 16 - layout.bits;
	let sample = |bytes: &[u8; 2]| u16::from_le_bytes(*bytes) >> shift;
	let count = y / 2 + c;
	let mut planar = Vec::new();
	planar
		.try_reserve_exact(count)
		.map_err(|_| Error::resource_exhausted("wide video readback allocation failed"))?;
	planar.extend(raw[..y].as_chunks::<2>().0.iter().map(sample));
	if layout.interleaved {
		planar.resize(count, 0);
		let (cb, cr) = planar[y / 2..].split_at_mut(c / 2);
		for ((cb, cr), pair) in cb
			.iter_mut()
			.zip(cr)
			.zip(raw[uv..uv + 2 * c].as_chunks::<4>().0)
		{
			*cb = u16::from_le_bytes([pair[0], pair[1]]) >> shift;
			*cr = u16::from_le_bytes([pair[2], pair[3]]) >> shift;
		}
	} else {
		for offset in [uv, v] {
			planar.extend(
				raw[offset..offset + c]
					.as_chunks::<2>()
					.0
					.iter()
					.map(sample),
			);
		}
	}
	Ok(video::VideoHostPlanes::U16(Arc::new(planar)))
}

#[cfg(test)]
#[path = "../../../../../../../../test/rs/runtime/video_readback_unit.rs"]
mod tests;

#[derive(Clone, Copy)]
pub(in crate::runtime::device) struct ImageUse {
	family: u32,
	layout: vk::ImageLayout,
	stage: vk::PipelineStageFlags2,
	access: vk::AccessFlags2,
}

impl ImageUse {
	fn decode(family: u32, layout: vk::ImageLayout) -> Self {
		Self {
			family,
			layout,
			stage: vk::PipelineStageFlagBits2::VIDEO_DECODE_BIT_KHR,
			access: vk::AccessFlagBits2::VIDEO_DECODE_READ_BIT_KHR
				| vk::AccessFlagBits2::VIDEO_DECODE_WRITE_BIT_KHR,
		}
	}

	pub(in crate::runtime::device) fn copy(family: u32) -> Self {
		Self {
			family,
			layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
			stage: vk::PipelineStageFlagBits2::COPY_BIT,
			access: vk::AccessFlagBits2::TRANSFER_READ_BIT,
		}
	}

	pub(in crate::runtime::device) fn sample(family: u32) -> Self {
		Self {
			family,
			layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
			stage: vk::PipelineStageFlagBits2::COMPUTE_SHADER_BIT,
			access: vk::AccessFlagBits2::SHADER_SAMPLED_READ_BIT,
		}
	}
}

#[derive(Clone, Copy)]
struct ImageHandoff<'a> {
	output: &'a VideoImage,
	layer: u32,
	source: ImageUse,
	destination: ImageUse,
}

impl ImageHandoff<'_> {
	fn crosses_family(self) -> bool {
		self.source.family != self.destination.family
	}

	fn reverse(self) -> Self {
		Self {
			source: self.destination,
			destination: self.source,
			..self
		}
	}

	fn record(self, device: &Device, command: vk::CommandBuffer, acquire: bool) -> Result<()> {
		let cross = self.crosses_family();
		let source_scope = !acquire;
		let destination_scope = acquire || !cross;
		let barrier = vk::ImageMemoryBarrier2::new()
			.set_src_stage_mask(if source_scope {
				self.source.stage
			} else {
				vk::PipelineStageFlagBits2::NONE
			})
			.set_src_access_mask(if source_scope {
				self.source.access
			} else {
				vk::AccessFlagBits2::NONE
			})
			.set_dst_stage_mask(if destination_scope {
				self.destination.stage
			} else {
				vk::PipelineStageFlagBits2::NONE
			})
			.set_dst_access_mask(if destination_scope {
				self.destination.access
			} else {
				vk::AccessFlagBits2::NONE
			})
			.set_old_layout(self.source.layout)
			.set_new_layout(self.destination.layout)
			.set_src_queue_family_index(if cross {
				self.source.family
			} else {
				vk::QUEUE_FAMILY_IGNORED
			})
			.set_dst_queue_family_index(if cross {
				self.destination.family
			} else {
				vk::QUEUE_FAMILY_IGNORED
			})
			.set_image(self.output.handle)
			.set_subresource_range(decode_image_subresource(self.layer, 1));
		let dependency = vk::DependencyInfo::new()
			.set_image_memory_barriers(std::slice::from_ref(&barrier))
			.map_err(Error::from)?;
		// SAFETY: both sides use the exact same layer and layouts. The
		// runtime timeline orders the matching release and acquire.
		unsafe { device.cmd_pipeline_barrier2(command, &dependency) }?;
		Ok(())
	}
}

/// Retained source of a replayable image-to-device-plane copy. The lease
/// protects the selected DPB layer even after the decoder is closed.
pub(in crate::runtime) struct DecodedPlaneCopy {
	pool: Arc<NativeFrameLeases>,
	lease: Option<Arc<NativeDecodedFrameLease>>,
	slot: u32,
	image_index: usize,
	layer: u32,
	layout: vk::ImageLayout,
	families: (u32, u32),
	regions: Vec<vk::BufferImageCopy>,
	plane_bytes: Vec<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::runtime) enum PlaneCopyPhase {
	Release,
	Copy,
	Acquire,
}

impl PlaneCopyPhase {
	pub(in crate::runtime) const fn name(self) -> &'static str {
		match self {
			Self::Release => "video.plane_release",
			Self::Copy => "video.copy_planes",
			Self::Acquire => "video.plane_acquire",
		}
	}
}

impl DecodedPlaneCopy {
	pub(in crate::runtime::device) fn new(
		session: &DecodeSession,
		frame: &NativeDecodedFrame,
	) -> Result<Self> {
		let leases = session
			.native_leases
			.as_ref()
			.ok_or_else(|| Error::failed_precondition("native video lease pool is unavailable"))?;
		if !frame.belongs_to(leases) {
			return Err(Error::invalid_argument(
				"native video frame belongs to another decoder",
			));
		}
		leases.validate_live_frame(frame.slot())?;
		Self::from_slot(session, frame.slot(), Some(frame.lease.clone()))
	}

	pub(in crate::runtime::device) fn for_replay(session: &DecodeSession, slot: u32) -> Result<Self> {
		Self::from_slot(session, slot, None)
	}

	fn from_slot(
		session: &DecodeSession,
		slot: u32,
		lease: Option<Arc<NativeDecodedFrameLease>>,
	) -> Result<Self> {
		let leases = session
			.native_leases
			.as_ref()
			.ok_or_else(|| Error::failed_precondition("native video lease pool is unavailable"))?;
		leases.validate_ownership()?;
		let (image, layer, _, _) = session.picture_images(slot)?;
		let image_index = leases
			._images
			.iter()
			.position(|entry| std::ptr::eq(entry, image))
			.ok_or_else(|| Error::internal("decoded image is absent from its lease pool"))?;
		let extent = session.visible_extent;
		let plane_bytes =
			video::plane_extents(leases.format, extent.width as usize, extent.height as usize)?
				.into_iter()
				.map(|plane| {
					plane
						.width
						.checked_mul(plane.height)
						.ok_or_else(|| Error::out_of_range("decoded plane extent overflows usize"))
				})
				.collect::<Result<Vec<_>>>()?;
		let (_, mut regions) = yuv420_copy_layout(image.format, extent, layer, session.visible_origin)?;
		if regions.len() != plane_bytes.len() {
			return Err(Error::internal(
				"native video format disagrees with its copy regions",
			));
		}
		for region in &mut regions {
			region.buffer_offset = 0;
		}
		let physical = session.device.physical();
		let decode_family = physical
			.video
			.decode_queue_family
			.ok_or_else(|| Error::missing_capability("no Vulkan Video decode queue is enabled"))?;
		let layout = if session.image_set == DecodeImageSet::DistinctLayered {
			vk::ImageLayout::VIDEO_DECODE_DST_KHR
		} else {
			vk::ImageLayout::VIDEO_DECODE_DPB_KHR
		};
		Ok(Self {
			pool: leases.clone(),
			lease,
			slot,
			image_index,
			layer,
			layout,
			families: (decode_family, physical.compute_queue_family),
			regions,
			plane_bytes,
		})
	}

	pub(in crate::runtime) fn extent(&self) -> video::VideoExtent {
		video::VideoExtent {
			width: self.regions[0].image_extent.width,
			height: self.regions[0].image_extent.height,
		}
	}

	pub(in crate::runtime) fn format(&self) -> video::VideoPixelFormat {
		self.pool.format
	}

	pub(in crate::runtime) fn crosses_family(&self) -> bool {
		self.families.0 != self.families.1
	}

	pub(in crate::runtime) fn plane_bytes(&self) -> &[usize] {
		&self.plane_bytes
	}

	pub(in crate::runtime) fn validate_ownership(&self) -> Result<()> {
		if self.lease.is_some() {
			self.pool.validate_live_frame(self.slot)
		} else {
			self.pool.validate_ownership()
		}
	}

	pub(in crate::runtime) fn invalidate_ownership(&self) {
		self.pool.ownership_lost.store(true, Ordering::Release);
	}

	pub(in crate::runtime) fn descriptor_words(&self) -> impl Iterator<Item = u32> + '_ {
		[
			self.layer,
			self.layout.as_raw() as u32,
			self.families.0,
			self.families.1,
		]
		.into_iter()
		.chain(self.regions.iter().flat_map(|region| {
			[
				region.image_subresource.aspect_mask.as_raw(),
				region.image_offset.x as u32,
				region.image_offset.y as u32,
				region.image_extent.width,
				region.image_extent.height,
			]
		}))
	}

	pub(in crate::runtime) fn record(
		&self,
		device: &Device,
		command: vk::CommandBuffer,
		phase: PlaneCopyPhase,
		destination: &[crate::runtime::graph::executable::BufferUse],
	) -> Result<()> {
		let image = &self.pool._images[self.image_index];
		let handoff = copy_handoff(image, self.layer, self.families, self.layout);
		match phase {
			PlaneCopyPhase::Release => handoff.record(device, command, false),
			PlaneCopyPhase::Acquire => handoff.reverse().record(device, command, true),
			PlaneCopyPhase::Copy => {
				handoff.record(device, command, handoff.crosses_family())?;
				for (buffer, region) in destination.iter().zip(&self.regions) {
					// SAFETY: admission checked the plane count and byte
					// extents. The retained image is in its transfer layout.
					unsafe {
						device.cmd_copy_image_to_buffer(
							command,
							image.handle,
							vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
							buffer.buffer.raw(),
							std::slice::from_ref(region),
						)?;
					}
				}
				handoff.reverse().record(device, command, false)
			}
		}
	}
}

pub(in crate::runtime::device) fn record_image_consumer(
	device: &Arc<Device>,
	output: &VideoImage,
	layer: u32,
	decode_family: u32,
	decoded_layout: vk::ImageLayout,
	consumer: ImageUse,
	record: impl FnOnce(vk::CommandBuffer) -> Result<()>,
) -> Result<RecordedBatch> {
	let handoff = ImageHandoff {
		output,
		layer,
		source: ImageUse::decode(decode_family, decoded_layout),
		destination: consumer,
	};
	let cross = handoff.crosses_family();
	let mut batch = RecordedBatch::new(device);
	if cross {
		batch.push(device.record_video_decode(|command| handoff.record(device, command, false))?);
	}
	batch.push(device.record_compute_commands(|command| {
		handoff.record(device, command, cross)?;
		record(command)?;
		handoff.reverse().record(device, command, false)
	})?);
	if cross {
		batch
			.push(device.record_video_decode(|command| handoff.reverse().record(device, command, true))?);
	}
	Ok(batch)
}

// Ownership round trip for decoded-image copies.
fn copy_handoff(
	output: &VideoImage,
	layer: u32,
	families: (u32, u32),
	layout: vk::ImageLayout,
) -> ImageHandoff<'_> {
	ImageHandoff {
		output,
		layer,
		source: ImageUse::decode(families.0, layout),
		destination: ImageUse::copy(families.1),
	}
}

#[allow(clippy::too_many_arguments)]
fn record_plane_copy(
	device: &Device,
	command_buffer: vk::CommandBuffer,
	output: &VideoImage,
	destination: &crate::runtime::Buffer,
	regions: &[vk::BufferImageCopy],
	post_stage: vk::PipelineStageFlagBits2,
	post_access: vk::AccessFlagBits2,
) -> Result<()> {
	// SAFETY: the output layer is transfer-readable and each tightly packed plane
	// region is in bounds of the retained destination buffer.
	unsafe {
		device.cmd_copy_image_to_buffer(
			command_buffer,
			output.handle,
			vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
			destination.raw(),
			regions,
		)?;
	}
	let post = vk::BufferMemoryBarrier2::new()
		.set_src_stage_mask(vk::PipelineStageFlagBits2::COPY_BIT)
		.set_src_access_mask(vk::AccessFlagBits2::TRANSFER_WRITE_BIT)
		.set_dst_stage_mask(post_stage)
		.set_dst_access_mask(post_access)
		.set_src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
		.set_dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
		.set_buffer(destination.raw())
		.set_offset(0)
		.set_size(destination.size());
	let dependency =
		vk::DependencyInfo::new().set_buffer_memory_barriers(std::slice::from_ref(&post));
	// SAFETY: the barrier publishes every transfer-written byte to the named
	// consumer stage and access domain.
	unsafe {
		device
			.cmd_pipeline_barrier2(command_buffer, &(dependency).map_err(Error::from)?)
			.map_err(Error::from)?;
	}
	Ok(())
}

/// Pack the acquired planes and dispatch into an independent RGBA buffer.
#[allow(clippy::too_many_arguments)]
pub(in crate::runtime::device) fn record_decode_presentation(
	device: &Device,
	command_buffer: vk::CommandBuffer,
	output: &VideoImage,
	yuv: &crate::runtime::Buffer,
	rgba: &crate::runtime::Buffer,
	regions: &[vk::BufferImageCopy],
	pipeline: &ComputePipeline,
	descriptor_set: vk::DescriptorSet,
	push: &PresentationPush,
	workgroups: (u32, u32),
) -> Result<()> {
	record_plane_copy(
		device,
		command_buffer,
		output,
		yuv,
		regions,
		vk::PipelineStageFlagBits2::COMPUTE_SHADER_BIT,
		vk::AccessFlagBits2::SHADER_STORAGE_READ_BIT,
	)?;
	// SAFETY: the command buffer is recording on this device; the pipeline,
	// descriptor heap, and both buffers outlive submission through retention.
	unsafe {
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
			std::slice::from_raw_parts(
				std::ptr::from_ref(push).cast::<u8>(),
				size_of::<PresentationPush>(),
			),
		)?;
		device.cmd_dispatch(command_buffer, workgroups.0, workgroups.1, 1)?;
	}
	let output_ready = vk::BufferMemoryBarrier2::new()
		.set_src_stage_mask(vk::PipelineStageFlagBits2::COMPUTE_SHADER_BIT)
		.set_src_access_mask(vk::AccessFlagBits2::SHADER_STORAGE_WRITE_BIT)
		.set_dst_stage_mask(vk::PipelineStageFlagBits2::TRANSFER_BIT)
		.set_dst_access_mask(vk::AccessFlagBits2::TRANSFER_READ_BIT)
		.set_src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
		.set_dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
		.set_buffer(rgba.raw())
		.set_offset(0)
		.set_size(rgba.size());
	let dependency =
		vk::DependencyInfo::new().set_buffer_memory_barriers(std::slice::from_ref(&output_ready));
	// SAFETY: the barrier makes every dispatch-written texel available to the
	// queue-ordered Texture upload that follows this submission.
	unsafe {
		device
			.cmd_pipeline_barrier2(command_buffer, &(dependency).map_err(Error::from)?)
			.map_err(Error::from)?;
	}
	Ok(())
}

pub(in crate::runtime::device) fn presentation_sample_bits(format: vk::Format) -> Result<u32> {
	match format {
		vk::Format::G8_B8R8_2PLANE_420_UNORM | vk::Format::G8_B8_R8_3PLANE_420_UNORM => Ok(8),
		vk::Format::G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16
		| vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16 => Ok(10),
		vk::Format::G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16
		| vk::Format::G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16 => Ok(12),
		_ => Err(Error::missing_capability(
			"native presentation requires an 8-, 10-, or 12-bit 4:2:0 output format",
		)),
	}
}

/// Legacy host-readback copy. The decode command has already recorded the
/// producer release, and the next decode takes back the selected layer.
#[allow(clippy::too_many_arguments)]
pub(in crate::runtime::device) fn record_decode_readback_round_trip(
	device: &Device,
	command: vk::CommandBuffer,
	output: &VideoImage,
	layer: u32,
	readback: &crate::runtime::Buffer,
	regions: &[vk::BufferImageCopy],
	families: (u32, u32),
	layout: vk::ImageLayout,
) -> Result<()> {
	let handoff = copy_handoff(output, layer, families, layout);
	if handoff.crosses_family() {
		handoff.record(device, command, true)?;
	}
	record_plane_readback(device, command, output, readback, regions, false)?;
	handoff.reverse().record(device, command, false)
}

pub(in crate::runtime::device) fn record_plane_readback(
	device: &Device,
	command: vk::CommandBuffer,
	output: &VideoImage,
	readback: &crate::runtime::Buffer,
	regions: &[vk::BufferImageCopy],
	for_canvas: bool,
) -> Result<()> {
	let (stage, access) = if for_canvas {
		(
			vk::PipelineStageFlagBits2::FRAGMENT_SHADER_BIT,
			vk::AccessFlagBits2::SHADER_STORAGE_READ_BIT,
		)
	} else {
		(
			vk::PipelineStageFlagBits2::HOST_BIT,
			vk::AccessFlagBits2::HOST_READ_BIT,
		)
	};
	record_plane_copy(device, command, output, readback, regions, stage, access)
}
