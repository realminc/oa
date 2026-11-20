//! Immutable codec commands for execution plans. The live decoder and the
//! plan never share a Vulkan session, DPB images, or bitstream allocations.

use super::*;

pub(in crate::runtime) struct PreparedDecode {
	bitstream: DecodeBitstream,
	codec: DecodePicture,
	initialize_images: bool,
}

enum DecodePicture {
	H264 {
		sps: Box<video::H264SequenceParameterSet>,
		pps: Box<video::H264PictureParameterSet>,
		slice: Box<video::H264SliceHeader>,
		plan: h264::PicturePlan,
	},
	H265 {
		sps: Box<video::H265SequenceParameterSet>,
		pps: Box<video::H265PictureParameterSet>,
		slice: Box<video::H265SliceHeader>,
		plan: h265::PicturePlan,
	},
	Av1 {
		picture: Box<video::Av1Picture>,
		plan: av1::PicturePlan,
	},
	Vp9 {
		picture: Box<video::Vp9Picture>,
		plan: vp9::PicturePlan,
	},
}

pub(in crate::runtime) struct RetainedDecode {
	session: Arc<DecodeSession>,
	picture: PreparedDecode,
}

pub(in crate::runtime) struct DecodeStatus {
	device: Arc<Device>,
	pool: vk::QueryPool,
}

impl DecodeSession {
	pub(in crate::runtime) fn require_capture_support(&self) -> Result<()> {
		let driver = &self.device.physical().info;
		// ANV before Mesa 26.3 updates mapped VP9 probability tables at command
		// recording time. Recording later pictures can corrupt earlier work
		// even when the device timeline and Vulkan barriers are correct.
		if matches!(self.profile, video::VideoDecodeProfile::Vp9 { .. })
			&& driver.driver_id == vk::DriverId::INTEL_OPEN_SOURCE_MESA
			&& driver.driver_version < vk::make_api_version(0, 26, 3, 0)
		{
			return Err(Error::missing_capability(
				"VP9 capture on Intel ANV requires Mesa 26.3 with GPU-ordered probability updates",
			));
		}
		Ok(())
	}

	fn prepare_decode(
		&mut self,
		data: &[u8],
		codec: DecodePicture,
		resets_session: bool,
	) -> Result<PreparedDecode> {
		let initialize_images = !self.decode_recorded;
		if initialize_images && !resets_session {
			return Err(Error::failed_precondition(
				"recorded decode must begin with a picture that resets the video session",
			));
		}
		let range = self.aligned_bitstream_len(data)?;
		let mut bitstream = with_decode_profile(self.profile, |profile| {
			create_decode_bitstream(&self.device, profile, range)
		})?;
		bitstream.write(data, range)?;
		self.decode_recorded = true;
		Ok(PreparedDecode {
			bitstream,
			codec,
			initialize_images,
		})
	}

	pub(in crate::runtime) fn prepare_h264_picture(
		&mut self,
		data: &[u8],
		sps: &video::H264SequenceParameterSet,
		pps: &video::H264PictureParameterSet,
		slice: &video::H264SliceHeader,
	) -> Result<(PreparedDecode, u32)> {
		self.validate_h264_picture(sps, pps, slice)?;
		let plan = self
			.h264_dpb_state
			.as_mut()
			.ok_or_else(|| Error::internal("H.264 session has no decoded-picture-buffer state"))?
			.plan_in_place(sps, slice, &[])?;
		let (slot, reset) = (plan.setup_slot, plan.reset_dpb);
		let codec = DecodePicture::H264 {
			sps: Box::new(sps.clone()),
			pps: Box::new(pps.clone()),
			slice: Box::new(slice.clone()),
			plan,
		};
		Ok((self.prepare_decode(data, codec, reset)?, slot))
	}

	pub(in crate::runtime) fn prepare_h265_picture(
		&mut self,
		data: &[u8],
		sps: &video::H265SequenceParameterSet,
		pps: &video::H265PictureParameterSet,
		slice: &video::H265SliceHeader,
	) -> Result<(PreparedDecode, u32)> {
		self.validate_h265_picture(sps, pps, slice)?;
		let plan = self
			.h265_dpb_state
			.as_mut()
			.ok_or_else(|| Error::internal("H.265 session has no decoded-picture-buffer state"))?
			.plan_in_place(sps, slice, &[])?;
		let (slot, reset) = (plan.setup_slot, plan.reset_dpb);
		let codec = DecodePicture::H265 {
			sps: Box::new(sps.clone()),
			pps: Box::new(pps.clone()),
			slice: Box::new(slice.clone()),
			plan,
		};
		Ok((self.prepare_decode(data, codec, reset)?, slot))
	}

	pub(in crate::runtime) fn prepare_av1_picture(
		&mut self,
		data: &[u8],
		picture: &video::Av1Picture,
	) -> Result<(PreparedDecode, u32)> {
		self.validate_av1_picture(&picture.sequence, &picture.frame)?;
		if picture.tiles.is_none() {
			return Err(Error::invalid_argument(
				"coded AV1 picture has no tile payloads",
			));
		}
		let plan = self
			.av1_dpb_state
			.as_mut()
			.ok_or_else(|| Error::internal("AV1 session has no decoded-picture-buffer state"))?
			.plan_in_place(&picture.frame, &[])?;
		let slot = plan
			.setup_slot
			.ok_or_else(|| Error::internal("coded AV1 picture has no setup slot"))?;
		let reset = plan.reset_dpb;
		let codec = DecodePicture::Av1 {
			picture: Box::new(picture.clone()),
			plan,
		};
		Ok((self.prepare_decode(data, codec, reset)?, slot))
	}

	pub(in crate::runtime) fn prepare_vp9_picture(
		&mut self,
		data: &[u8],
		picture: &video::Vp9Picture,
	) -> Result<(PreparedDecode, u32)> {
		self.validate_vp9_picture(picture)?;
		let data = vp9_picture_data(data, picture)?;
		let plan = self
			.vp9_dpb_state
			.as_mut()
			.ok_or_else(|| Error::internal("VP9 session has no decoded-picture-buffer state"))?
			.plan_in_place(picture, &[])?;
		let slot = plan
			.setup_slot
			.ok_or_else(|| Error::internal("coded VP9 picture has no setup slot"))?;
		let reset = plan.reset_dpb;
		let codec = DecodePicture::Vp9 {
			picture: Box::new(picture.clone()),
			plan,
		};
		Ok((self.prepare_decode(data, codec, reset)?, slot))
	}
}

impl RetainedDecode {
	pub(in crate::runtime) fn new(session: &Arc<DecodeSession>, picture: PreparedDecode) -> Self {
		Self {
			session: session.clone(),
			picture,
		}
	}

	pub(in crate::runtime) fn validate_ownership(&self) -> Result<()> {
		self.session.validate_ownership()
	}

	pub(in crate::runtime) fn invalidate_ownership(&self) {
		self.session.invalidate_ownership();
	}

	pub(in crate::runtime) const fn name(&self) -> &'static str {
		match self.picture.codec {
			DecodePicture::H264 { .. } => "video.decode_h264",
			DecodePicture::H265 { .. } => "video.decode_h265",
			DecodePicture::Av1 { .. } => "video.decode_av1",
			DecodePicture::Vp9 { .. } => "video.decode_vp9",
		}
	}

	pub(in crate::runtime) fn descriptor_words(&self) -> Vec<u32> {
		let picture = &self.picture;
		let mut words = vec![
			self.session.coded_extent.width,
			self.session.coded_extent.height,
			self.session.visible_extent.width,
			self.session.visible_extent.height,
			picture.bitstream.range as u32,
			(picture.bitstream.range >> 32) as u32,
			u32::from(picture.initialize_images),
		];
		match &picture.codec {
			DecodePicture::H264 { sps, pps, plan, .. } => {
				words.extend([
					plan.setup_slot,
					u32::from(plan.reset_dpb),
					sps.profile_idc,
					sps.id,
					pps.id,
					plan.frame_number,
					plan.picture_order_count as u32,
				]);
				for reference in &plan.references {
					words.extend([
						reference.slot,
						reference.frame_number,
						reference.picture_order_count as u32,
						u32::from(reference.long_term),
					]);
				}
			}
			DecodePicture::H265 { sps, pps, plan, .. } => {
				words.extend([
					plan.setup_slot,
					u32::from(plan.reset_dpb),
					sps.video_parameter_set_id,
					sps.id,
					pps.id,
					plan.picture_order_count as u32,
				]);
				for reference in &plan.active_references {
					words.extend([reference.slot, reference.picture_order_count as u32]);
				}
			}
			DecodePicture::Av1 { picture, plan } => {
				words.extend([
					plan.setup_slot.unwrap_or(u32::MAX),
					u32::from(plan.reset_dpb),
					u32::from(picture.frame.order_hint),
					picture.frame_header_offset,
				]);
				words.extend(plan.reference_name_slot_indices.map(|slot| slot as u32));
				words.extend(
					plan
						.active_references
						.iter()
						.flat_map(|(slot, record)| [*slot, u32::from(record.order_hint)]),
				);
			}
			DecodePicture::Vp9 { picture, plan } => {
				words.extend([
					plan.setup_slot.unwrap_or(u32::MAX),
					u32::from(plan.reset_dpb),
					u32::from(picture.profile),
					picture.tiles_offset,
				]);
				words.extend(plan.reference_name_slot_indices.map(|slot| slot as u32));
				words.extend(&plan.active_references);
			}
		}
		words
	}

	pub(in crate::runtime) fn record(&self, command: vk::CommandBuffer) -> Result<DecodeStatus> {
		let session = &self.session;
		let pool = with_decode_profile(session.profile, |profile| {
			create_decode_status_pool(&session.device, profile)
		})?;
		let status = DecodeStatus {
			device: session.device.clone(),
			pool,
		};
		let device = session.device.raw();
		let PreparedDecode {
			bitstream,
			codec,
			initialize_images,
		} = &self.picture;
		match codec {
			DecodePicture::H264 {
				sps,
				pps,
				slice,
				plan,
			} => record_h264_picture(
				device,
				command,
				session,
				sps,
				pps,
				slice,
				pool,
				bitstream,
				plan,
				*initialize_images,
				None,
				false,
			)?,
			DecodePicture::H265 {
				sps,
				pps,
				slice,
				plan,
			} => record_h265_picture(
				device,
				command,
				session,
				sps,
				pps,
				slice,
				pool,
				bitstream,
				plan,
				*initialize_images,
				None,
				false,
			)?,
			DecodePicture::Av1 { picture, plan } => av1::record_picture(
				device,
				command,
				session,
				pool,
				&picture.sequence,
				&picture.frame,
				picture.frame_header_offset,
				picture
					.tiles
					.as_ref()
					.ok_or_else(|| Error::internal("recorded AV1 picture lost its tiles"))?,
				bitstream,
				plan,
				*initialize_images,
				None,
				false,
			)?,
			DecodePicture::Vp9 { picture, plan } => vp9::record_picture(
				device,
				command,
				session,
				pool,
				picture,
				bitstream,
				plan,
				*initialize_images,
				None,
				false,
			)?,
		}
		Ok(status)
	}
}

impl DecodeStatus {
	pub(in crate::runtime) fn verify(&self) -> Result<()> {
		read_decode_status(&self.device, self.pool)
	}
}

impl Drop for DecodeStatus {
	fn drop(&mut self) {
		// SAFETY: the recorded command retains its private pool until retirement.
		unsafe {
			let _ = self.device.raw().destroy_query_pool(self.pool, None);
		}
	}
}
