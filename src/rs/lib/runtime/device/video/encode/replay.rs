//! Immutable codec commands for execution plans. The live encoder and the
//! plan never share a Vulkan session, input images, or bitstream slots.

use super::*;

/// Frozen per-frame encode parameters. Replay records the same IDR/P
/// decisions, DPB slots, and rate-control preamble every submission.
pub(in crate::runtime) struct PreparedEncode {
	pub(in crate::runtime) slot: usize,
	pub(in crate::runtime) args: EncodePictureArgs,
	pub(in crate::runtime) presentation_timestamp_micros: u64,
	pub(in crate::runtime) is_keyframe: bool,
}

impl PreparedEncode {
	/// Mark this frame as the last in the captured batch: its coding scope
	/// ends with a session RESET so every replay begins from identical state.
	pub(in crate::runtime) fn mark_replay_boundary(&mut self) {
		self.args.replay_boundary = true;
	}
}

/// One retained `vkCmdEncodeVideoKHR` picture: the captured session, its
/// input image, DPB, bitstream slot, and feedback pool stay live through the
/// recorded command buffer's last accepted epoch.
pub(in crate::runtime) struct RetainedEncode {
	session: Arc<EncodeSession>,
	frame: PreparedEncode,
}

/// How one slot's input image is produced on the compute family.
enum EncodeInputKind {
	/// Plane buffers are copied into the coded-extent image planes.
	Planes,
	/// The slot's RGBA snapshot is converted by the session's fixed shader.
	Convert {
		visible: video::VideoExtent,
		color_space: u32,
		full_range: bool,
	},
}

/// Retained input staging for one slot's encode frame. The input image is
/// `CONCURRENT` between the compute and encode families, so staging needs
/// no ownership transfer — a release barrier on compute pairs with the
/// acquire barrier inside `record_encode_picture` on the encode queue.
pub(in crate::runtime) struct EncodePlaneCopy {
	session: Arc<EncodeSession>,
	slot: usize,
	kind: EncodeInputKind,
	regions: Vec<vk::BufferImageCopy>,
	plane_bytes: Vec<usize>,
}

/// Completion-boundary contract for one recorded frame. The event verifies
/// feedback status at the caller's observation point; `packet` materializes
/// the access unit from the slot's bitstream afterwards.
/// Per-recording encode output: a private feedback pool and bitstream buffer.
/// One recorded command buffer writes exactly one replay's results here, so
/// harvesting an earlier event never races a later replay's reset or write.
pub(in crate::runtime) struct ReplayEncodeOutput {
	pool: vk::QueryPool,
	bitstream: EncodeBitstream,
}

impl Drop for ReplayEncodeOutput {
	fn drop(&mut self) {
		// SAFETY: the pool is only destroyed once the owning receipt leaves the
		// event's retirement set, which is after command-buffer completion.
		unsafe {
			let _ = self
				.bitstream
				.device
				.raw()
				.destroy_query_pool(self.pool, None);
		}
	}
}

pub(in crate::runtime) struct EncodeReceipt {
	session: Arc<EncodeSession>,
	output: ReplayEncodeOutput,
	presentation_timestamp_micros: u64,
	is_keyframe: bool,
}

impl EncodeSession {
	/// Fail capture consumers once the retained session can no longer be
	/// trusted: a mid-batch submission failure leaves recorded slots in an
	/// indeterminate encode position.
	pub(in crate::runtime) fn validate_ownership(&self) -> Result<()> {
		if !self.capture_valid.load(Ordering::Acquire) {
			return Err(Error::failed_precondition(
				"video encoder session state was interrupted; recapture the plan",
			));
		}
		Ok(())
	}

	pub(in crate::runtime) fn invalidate_ownership(&self) {
		self.capture_valid.store(false, Ordering::Release);
	}

	/// Prepare one frame at capture time. GOP position, DPB slot rotation,
	/// and the rate-control preamble freeze exactly the values `record_frame`
	/// plus `mark_pending` compute at immediate submission.
	pub(in crate::runtime) fn prepare_encode_frame(
		&self,
		index: usize,
		presentation_timestamp_micros: u64,
	) -> Result<PreparedEncode> {
		if self.parameters == vk::VideoSessionParametersKHR::default() {
			return Err(Error::failed_precondition(
				"video encode session parameters were not created",
			));
		}
		let slot = self
			.slots
			.get(index)
			.ok_or_else(|| Error::out_of_range("encode slot exceeds session depth"))?;
		let frame_count = self.frame_count.load(Ordering::Acquire);
		let current_gop_frame = self.current_gop_frame.load(Ordering::Acquire);
		let is_idr = current_gop_frame == 0;
		let reference_depth = self.dpb_slot_capacity.max(1);
		let current_dpb_slot = (frame_count % u64::from(reference_depth)) as u32;
		let previous_dpb_slot = if frame_count == 0 {
			current_dpb_slot
		} else {
			((frame_count - 1) % u64::from(reference_depth)) as u32
		};
		let frame_index = (frame_count - self.last_keyframe_index.load(Ordering::Acquire)) as u32;
		let rc_reset = self.rate_control_reset.load(Ordering::Acquire);
		slot.is_keyframe.store(is_idr, Ordering::Release);
		// input_initialized stays false on the replay session: every replay's
		// conversion discards through an UNDEFINED transition, the only old
		// layout that is always correct regardless of the previous replay.
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
		slot
			.presentation_timestamp
			.store(presentation_timestamp_micros, Ordering::Release);
		slot.pending.store(true, Ordering::Release);
		Ok(PreparedEncode {
			slot: index,
			args: EncodePictureArgs {
				session: self.handle,
				parameters: self.parameters,
				codec: self.codec,
				coded: self.coded_extent,
				input_image: slot.input_image,
				input_view: slot.input_view,
				dpb_image: self.dpb.handle,
				dpb_view: self.dpb.view,
				dpb_layers: self.dpb_slot_capacity,
				// The captured first frame transitions the DPB from
				// UNDEFINED; that stays legal on every replay since the GOP
				// restarts with IDR and stale reference content is dead.
				dpb_initialized: frame_count != 0,
				current_dpb_slot,
				previous_dpb_slot,
				is_idr,
				frame_index,
				idr_pic_id: (frame_count & 0xFFFF) as u16,
				first_frame: frame_count == 0,
				bitstream: slot.bitstream.handle,
				bitstream_range: slot.bitstream.capacity,
				query_pool: slot.query_pool,
				rate_control_mode: self.rate_control_mode,
				rate_control_reset: rc_reset,
				bitrate: self.bitrate,
				peak_bitrate: self.peak_bitrate,
				frame_rate: self.frame_rate,
				gop_size: self.gop_size,
				max_b_frames: self.max_b_frames,
				constant_qp: self.constant_qp,
				replay_boundary: false,
			},
			presentation_timestamp_micros,
			is_keyframe: is_idr,
		})
	}

	/// Retain the buffer-to-image staging for one slot's input planes. The
	/// copy regions cover the whole coded extent, so recorded input carries
	/// the encoder's padding rows as well as its visible picture.
	pub(in crate::runtime) fn retain_input_copy(
		self: &Arc<Self>,
		slot: usize,
	) -> Result<EncodePlaneCopy> {
		if slot >= self.slots.len() {
			return Err(Error::out_of_range("encode slot exceeds session depth"));
		}
		let (_, mut regions) = crate::runtime::device::video::decode::yuv420_copy_layout(
			self.input_format,
			self.coded_extent,
			0,
			[0, 0],
		)?;
		let plane_bytes = video::plane_extents(
			self.input_pixel_format,
			self.coded_extent.width as usize,
			self.coded_extent.height as usize,
		)?
		.into_iter()
		.map(|plane| {
			plane
				.width
				.checked_mul(plane.height)
				.ok_or_else(|| Error::out_of_range("encode plane extent overflows usize"))
		})
		.collect::<Result<Vec<_>>>()?;
		if regions.len() != plane_bytes.len() {
			return Err(Error::internal(
				"encode input format disagrees with its copy regions",
			));
		}
		for region in &mut regions {
			region.buffer_offset = 0;
		}
		Ok(EncodePlaneCopy {
			session: self.clone(),
			slot,
			kind: EncodeInputKind::Planes,
			regions,
			plane_bytes,
		})
	}

	/// Retain the RGBA→native-input conversion for a slot. No plane buffers
	/// participate; the session's host-visible snapshot feeds the shader.
	pub(in crate::runtime) fn retain_conversion(
		self: &Arc<Self>,
		slot: usize,
		visible: video::VideoExtent,
		color_space: u32,
		full_range: bool,
	) -> Result<EncodePlaneCopy> {
		if slot >= self.slots.len() {
			return Err(Error::out_of_range("encode slot exceeds session depth"));
		}
		Ok(EncodePlaneCopy {
			session: self.clone(),
			slot,
			kind: EncodeInputKind::Convert {
				visible,
				color_space,
				full_range,
			},
			regions: Vec::new(),
			plane_bytes: Vec::new(),
		})
	}
}

impl RetainedEncode {
	pub(in crate::runtime) fn new(session: &Arc<EncodeSession>, frame: PreparedEncode) -> Self {
		Self {
			session: session.clone(),
			frame,
		}
	}

	pub(in crate::runtime) fn validate_ownership(&self) -> Result<()> {
		self.session.validate_ownership()
	}

	pub(in crate::runtime) fn invalidate_ownership(&self) {
		self.session.invalidate_ownership();
	}

	pub(in crate::runtime) const fn name(&self) -> &'static str {
		match self.frame.args.codec {
			EncodeCodec::H264 => "video.encode_h264",
			EncodeCodec::H265 => "video.encode_h265",
		}
	}

	pub(in crate::runtime) fn descriptor_words(&self) -> Vec<u32> {
		let args = &self.frame.args;
		vec![
			self.frame.slot as u32,
			args.codec as u32,
			args.coded.width,
			args.coded.height,
			args.current_dpb_slot,
			args.previous_dpb_slot,
			u32::from(args.is_idr),
			args.frame_index,
			u32::from(args.idr_pic_id),
			u32::from(args.first_frame),
			self.frame.presentation_timestamp_micros as u32,
			(self.frame.presentation_timestamp_micros >> 32) as u32,
		]
	}

	/// Record the encode picture against a private output pair allocated for
	/// this recording. Each submission receives its own feedback pool and
	/// bitstream, so a later replay cannot overwrite or reset the output an
	/// earlier event still has to harvest on the host.
	pub(in crate::runtime) fn record(
		&self,
		command: vk::CommandBuffer,
	) -> Result<Arc<EncodeReceipt>> {
		let session = &self.session;
		let mut args = self.frame.args;
		let output = with_encode_profile(session.profile, |profile| {
			let pool = create_feedback_pool(&session.device, profile)?;
			let bitstream = create_encode_bitstream(
				&session.device,
				profile,
				args.bitstream_range,
				session.bitstream_alignment,
			)?;
			Ok(ReplayEncodeOutput { pool, bitstream })
		})?;
		args.query_pool = output.pool;
		args.bitstream = output.bitstream.handle;
		args.bitstream_range = output.bitstream.capacity;
		record_encode_picture(session.device.raw(), command, args)?;
		Ok(Arc::new(EncodeReceipt {
			session: session.clone(),
			output,
			presentation_timestamp_micros: self.frame.presentation_timestamp_micros,
			is_keyframe: self.frame.is_keyframe,
		}))
	}
}

impl EncodePlaneCopy {
	pub(in crate::runtime) fn validate_ownership(&self) -> Result<()> {
		self.session.validate_ownership()
	}

	pub(in crate::runtime) fn invalidate_ownership(&self) {
		self.session.invalidate_ownership();
	}

	pub(in crate::runtime) const fn name(&self) -> &'static str {
		"video.encode_input"
	}

	pub(in crate::runtime) fn plane_bytes(&self) -> &[usize] {
		&self.plane_bytes
	}

	pub(in crate::runtime) fn descriptor_words(&self) -> impl Iterator<Item = u32> + '_ {
		let kind_words: Vec<u32> = match &self.kind {
			EncodeInputKind::Planes => vec![0],
			EncodeInputKind::Convert {
				visible,
				color_space,
				full_range,
			} => vec![
				1,
				visible.width,
				visible.height,
				*color_space,
				u32::from(*full_range),
			],
		};
		[self.slot as u32]
			.into_iter()
			.chain(kind_words)
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

	/// Record plane-buffer → input-image copies on the compute queue.
	///
	/// Producer: the plane buffers' recorded writes (same-family compute hazard
	/// handling). Consumer: the encode command's internal acquire barrier
	/// (`VIDEO_ENCODE` stage, `VIDEO_ENCODE_READ` access). The input image is
	/// `CONCURRENT` between compute and encode families, so the release side
	/// carries no queue-family transfer — the engine timeline orders the
	/// compute and encode submissions.
	pub(in crate::runtime) fn record(
		&self,
		device: &vk::Device,
		command: vk::CommandBuffer,
		sources: &[crate::runtime::graph::executable::BufferUse],
	) -> Result<()> {
		if let EncodeInputKind::Convert {
			visible,
			color_space,
			full_range,
		} = &self.kind
		{
			return self.session.record_conversion_into(
				&self.session.device,
				command,
				self.slot,
				*visible,
				*color_space,
				*full_range,
			);
		}
		let slot = &self.session.slots[self.slot];
		let aspects = vk::ImageAspectFlagBits::PLANE_0_BIT | vk::ImageAspectFlagBits::PLANE_1_BIT;
		let acquire = vk::ImageMemoryBarrier2::new()
			.set_src_stage_mask(vk::PipelineStageFlagBits2::NONE)
			.set_src_access_mask(vk::AccessFlagBits2::NONE)
			.set_dst_stage_mask(vk::PipelineStageFlagBits2::TRANSFER_BIT)
			.set_dst_access_mask(vk::AccessFlagBits2::TRANSFER_WRITE_BIT)
			// UNDEFINED is legal from any prior layout and correct here: the
			// copies below overwrite the whole coded extent every replay.
			.set_old_layout(vk::ImageLayout::UNDEFINED)
			.set_new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
			.set_image(slot.input_image)
			.set_subresource_range(vk::ImageSubresourceRange {
				aspect_mask: aspects,
				base_mip_level: 0,
				level_count: 1,
				base_array_layer: 0,
				layer_count: 1,
			});
		let release = vk::ImageMemoryBarrier2::new()
			.set_src_stage_mask(vk::PipelineStageFlagBits2::TRANSFER_BIT)
			.set_src_access_mask(vk::AccessFlagBits2::TRANSFER_WRITE_BIT)
			.set_dst_stage_mask(vk::PipelineStageFlagBits2::NONE)
			.set_dst_access_mask(vk::AccessFlagBits2::NONE)
			.set_old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
			.set_new_layout(vk::ImageLayout::VIDEO_ENCODE_SRC_KHR)
			.set_image(slot.input_image)
			.set_subresource_range(vk::ImageSubresourceRange {
				aspect_mask: aspects,
				base_mip_level: 0,
				level_count: 1,
				base_array_layer: 0,
				layer_count: 1,
			});
		let dependency = vk::DependencyInfo::new()
			.set_image_memory_barriers(std::slice::from_ref(&acquire))
			.map_err(Error::from)?;
		// SAFETY: the image is session-owned and alive via `self.session`.
		unsafe { device.cmd_pipeline_barrier2(command, &dependency) }?;
		for (source, region) in sources.iter().zip(&self.regions) {
			// SAFETY: admission checked the plane count and byte extents. The
			// input image is in its transfer layout for this submission.
			unsafe {
				device.cmd_copy_buffer_to_image(
					command,
					source.buffer.raw(),
					slot.input_image,
					vk::ImageLayout::TRANSFER_DST_OPTIMAL,
					std::slice::from_ref(region),
				)?;
			}
		}
		let dependency = vk::DependencyInfo::new()
			.set_image_memory_barriers(std::slice::from_ref(&release))
			.map_err(Error::from)?;
		unsafe { device.cmd_pipeline_barrier2(command, &dependency) }?;
		Ok(())
	}
}

impl EncodeReceipt {
	fn feedback(&self) -> Result<EncodeFeedback> {
		read_encode_feedback(
			&self.session.device,
			self.output.pool,
			&self.output.bitstream,
			self.session.query_result_status_supported,
			false,
		)
	}

	/// Verify this frame's feedback completed at the completion boundary.
	pub(in crate::runtime) fn verify(&self) -> Result<()> {
		self.feedback().map(|_| ())
	}

	/// Materialize the access unit after the plan's event observed completion.
	/// Cached codec headers precede keyframe payloads, matching `harvest_slot`.
	pub(in crate::runtime) fn packet(&self) -> Result<video::EncodedVideoPacket> {
		let feedback = self.feedback()?;
		let headers = if self.is_keyframe {
			self.session.cached_headers()
		} else {
			&[]
		};
		let length = headers
			.len()
			.checked_add(feedback.bytes_written as usize)
			.ok_or_else(|| Error::out_of_range("encoded packet size overflowed"))?;
		let mut bitstream = Vec::new();
		bitstream
			.try_reserve_exact(length)
			.map_err(|_| Error::resource_exhausted("encoded packet allocation failed"))?;
		bitstream.extend_from_slice(headers);
		bitstream.resize(length, 0);
		self.output.bitstream.read(
			u64::from(feedback.bitstream_offset),
			&mut bitstream[headers.len()..],
		)?;
		video::EncodedVideoPacket::new(
			bitstream,
			self.presentation_timestamp_micros,
			self.is_keyframe,
		)
	}
}
