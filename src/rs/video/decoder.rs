//! Stateful hardware video decoding.

use std::{collections::BTreeMap, time::Duration};

use crate::{Engine, Error, Result, runtime::VideoDecoderBackend};

use super::{
	Av1Parser, Av1Profile, Av1SequenceHeader, H264PictureLayout, H264PictureParameterSet,
	H264Profile, H264SequenceParameterSet, H265PictureParameterSet, H265Profile,
	H265SequenceParameterSet, VideoChromaSubsampling, VideoCodec, VideoColorInfo, VideoColorMatrix,
	VideoColorRange, VideoComponentBitDepth, VideoContainerInfo, VideoDecodeProfile, VideoFrame,
	VideoFrameTiming, VideoPacket, Vp9Parser, Vp9Profile, parse_h264_pps, parse_h264_slice_header,
	parse_h264_sps, parse_h265_pps, parse_h265_slice_header, parse_h265_sps, parse_h265_vps,
	parse_nal_annex_b,
};

/// Stateful hardware decoder for one demuxed video stream.
///
/// The Experimental codec paths accept progressive H.264 Baseline/Main/High,
/// H.265 Main, AV1 Main 8-bit 4:2:0 without film grain, or VP9 Profile 0 8-bit
/// 4:2:0. AVC and HEVC access units contain one complete slice per picture;
/// admitted AV1 and VP9 access units may contain hidden and show-existing
/// pictures. Host-planar and retained native output are synchronous; Vulkan
/// handles and queue ownership remain private.
pub struct VideoDecoder {
	state: Option<DecoderState>,
}

struct DecoderState {
	backend: VideoDecoderBackend,
	info: VideoContainerInfo,
	codec: DecoderCodecState,
	pending_display: BTreeMap<(u64, u64), VideoFrame>,
	next_decode_sequence: u64,
	reorder_depth: usize,
	require_random_access: bool,
	output_mode: Option<DecodeOutputMode>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DecodeOutputMode {
	HostPlanar,
	Native,
}

enum DecodedBacking {
	HostPlanar(Vec<u8>),
	Native(crate::runtime::NativeDecodedFrame),
}

enum DecoderCodecState {
	H264 {
		sequence_parameter_set: Option<Box<H264SequenceParameterSet>>,
		picture_parameter_set: Option<Box<H264PictureParameterSet>>,
	},
	H265 {
		sequence_parameter_set: Option<Box<H265SequenceParameterSet>>,
		picture_parameter_set: Option<Box<H265PictureParameterSet>>,
	},
	Av1 {
		parser: Av1Parser,
		parameter_sequence: Option<Box<Av1SequenceHeader>>,
	},
	Vp9 {
		parser: Vp9Parser,
	},
}

impl VideoDecoder {
	/// Create a decoder for the selected track described by `info`.
	///
	/// # Errors
	///
	/// Returns an error unless the track is representable as progressive H.264
	/// Baseline/Main/High, H.265 Main, AV1 Main 8-bit 4:2:0 without film grain,
	/// or VP9 Profile 0 8-bit 4:2:0 and the selected Engine admits the qualified
	/// Vulkan path.
	pub fn create(engine: &Engine, info: VideoContainerInfo) -> Result<Self> {
		let profile = info.decode_profile().ok_or_else(|| {
			Error::missing_capability("the stream has no Vulkan-queryable decode profile")
		})?;
		let codec = match (info.codec(), profile) {
			(
				VideoCodec::H264,
				VideoDecodeProfile::H264 {
					profile: H264Profile::Baseline | H264Profile::Main | H264Profile::High,
					picture_layout: H264PictureLayout::Progressive,
					chroma_subsampling: VideoChromaSubsampling::Yuv420,
					luma_bit_depth: VideoComponentBitDepth::Eight,
					chroma_bit_depth: VideoComponentBitDepth::Eight,
				},
			) => DecoderCodecState::H264 {
				sequence_parameter_set: None,
				picture_parameter_set: None,
			},
			(
				VideoCodec::H265,
				VideoDecodeProfile::H265 {
					profile: H265Profile::Main,
					chroma_subsampling: VideoChromaSubsampling::Yuv420,
					luma_bit_depth: VideoComponentBitDepth::Eight,
					chroma_bit_depth: VideoComponentBitDepth::Eight,
				},
			) => DecoderCodecState::H265 {
				sequence_parameter_set: None,
				picture_parameter_set: None,
			},
			(
				VideoCodec::Av1,
				VideoDecodeProfile::Av1 {
					profile: Av1Profile::Main,
					film_grain_support: false,
					chroma_subsampling: VideoChromaSubsampling::Yuv420,
					luma_bit_depth: VideoComponentBitDepth::Eight,
					chroma_bit_depth: VideoComponentBitDepth::Eight,
				},
			) => DecoderCodecState::Av1 {
				parser: Av1Parser::default(),
				parameter_sequence: None,
			},
			(
				VideoCodec::Vp9,
				VideoDecodeProfile::Vp9 {
					profile: Vp9Profile::Profile0,
					chroma_subsampling: VideoChromaSubsampling::Yuv420,
					luma_bit_depth: VideoComponentBitDepth::Eight,
					chroma_bit_depth: VideoComponentBitDepth::Eight,
				},
			) => DecoderCodecState::Vp9 {
				parser: Vp9Parser::default(),
			},
			_ => {
				return Err(Error::missing_capability(
					"the public decoder admits progressive H.264 Baseline/Main/High, H.265 Main, AV1 Main 8-bit 4:2:0 without film grain, and VP9 Profile 0 8-bit 4:2:0",
				));
			}
		};
		let backend = VideoDecoderBackend::create(
			engine,
			profile,
			super::VideoExtent {
				width: info.width(),
				height: info.height(),
			},
		)?;
		Ok(Self {
			state: Some(DecoderState {
				backend,
				info,
				codec,
				pending_display: BTreeMap::new(),
				next_decode_sequence: 0,
				reorder_depth: 0,
				require_random_access: true,
				output_mode: None,
			}),
		})
	}

	/// Decode one demuxed access unit and emit the next display-order frame when ready.
	///
	/// Packets enter in decode order. The decoder retains up to the SPS reorder
	/// depth and returns `None` while no display-order frame is safe to emit.
	/// Call [`flush`](Self::flush) at end of stream to retrieve the delayed tail.
	///
	/// # Errors
	///
	/// Returns an error after close, for a packet from another track, when the
	/// first packet after create/flush is not random access, for unsupported
	/// bitstream shapes, or when Vulkan submission, status, or readback fails.
	pub fn decode(&mut self, packet: &VideoPacket) -> Result<Option<VideoFrame>> {
		self.decode_with_mode(packet, DecodeOutputMode::HostPlanar)
	}

	/// Decode one access unit while retaining its native multi-plane output.
	///
	/// This path omits the decoder-to-host pixel copy. It currently waits for the
	/// exact decode submission and verifies its result status before publishing
	/// the frame; asynchronous delivery is a later checkpoint. The returned frame
	/// retains its decoder image slot. Register every GPU consumer through
	/// [`VideoFrame::mark_consumed`] before dropping the last frame clone.
	///
	/// A decode sequence uses either host-planar [`decode`](Self::decode) or native
	/// output until [`flush`](Self::flush) establishes a new random-access boundary.
	///
	/// # Errors
	///
	/// Returns the same stream, capability, and state errors as
	/// [`decode`](Self::decode), plus `FailedPrecondition` when output modes are
	/// mixed before a flush and `ResourceExhausted` while every recyclable DPB
	/// slot remains retained by native frames or consumer completions.
	pub fn decode_native(&mut self, packet: &VideoPacket) -> Result<Option<VideoFrame>> {
		self.decode_with_mode(packet, DecodeOutputMode::Native)
	}

	/// Read one retained native frame as tightly packed planar 8-bit YUV 4:2:0.
	///
	/// The exact frame slot is copied; this does not observe the decoder's newest
	/// slot by accident. The synchronous operation waits for producer readiness,
	/// performs explicit decode/compute queue-family ownership transfers, restores
	/// the image to its codec layout, and registers the final submission as a
	/// consumer completion before reading host bytes.
	///
	/// # Errors
	///
	/// Returns an error after close, for a non-native frame, for a frame produced
	/// by another decoder, for an unsupported native format, or when submission,
	/// synchronization, or readback fails.
	pub fn read_native_yuv420p(&mut self, frame: &VideoFrame) -> Result<Vec<u8>> {
		let state = self
			.state
			.as_mut()
			.ok_or_else(|| Error::failed_precondition("VideoDecoder is closed"))?;
		let native = frame.native_backing().ok_or_else(|| {
			Error::invalid_argument("native video readback requires a native decoded frame")
		})?;
		state.backend.read_native_yuv420(native)
	}

	fn decode_with_mode(
		&mut self,
		packet: &VideoPacket,
		output_mode: DecodeOutputMode,
	) -> Result<Option<VideoFrame>> {
		let state = self
			.state
			.as_mut()
			.ok_or_else(|| Error::failed_precondition("VideoDecoder is closed"))?;
		if packet.track_id() != state.info.track_id() {
			return Err(Error::invalid_argument(
				"video packet track does not match the decoder stream",
			));
		}
		if state.require_random_access && !packet.is_keyframe() {
			return Err(Error::failed_precondition(
				"the first packet after create or flush must be random access",
			));
		}
		if state
			.output_mode
			.is_some_and(|active| active != output_mode)
		{
			return Err(Error::failed_precondition(
				"video decoder output mode cannot change before flush",
			));
		}

		let decoded =
			match &mut state.codec {
				DecoderCodecState::H264 {
					sequence_parameter_set,
					picture_parameter_set,
				} => {
					let nals = parse_nal_annex_b(packet.data());
					let find_parameter = |nal_type| {
						nals
							.iter()
							.find(|nal| nal.nal_unit_type() == nal_type)
							.map(|nal| nal.payload())
					};
					if sequence_parameter_set.is_none() {
						let sps = parse_h264_sps(find_parameter(7).ok_or_else(|| {
							Error::failed_precondition("initial H.264 packet contains no SPS")
						})?)?;
						let pps = parse_h264_pps(
							find_parameter(8).ok_or_else(|| {
								Error::failed_precondition("initial H.264 packet contains no PPS")
							})?,
							&sps,
						)?;
						if !sps.frame_mbs_only || sps.pic_order_count_type != 0 {
							return Err(Error::missing_capability(
								"the H.264 decoder currently requires progressive POC-type-zero streams",
							));
						}
						if sps.coded_width()? != state.info.width()
							|| sps.coded_height()? != state.info.height()
						{
							return Err(Error::invalid_argument(
								"H.264 parameter-set geometry differs from container metadata",
							));
						}
						state.backend.set_h264_parameters(&sps, &pps)?;
						state.reorder_depth = sps
							.vui
							.as_ref()
							.and_then(|vui| vui.bitstream_restriction)
							.map_or(16, |restriction| {
								restriction.max_num_reorder_frames as usize
							});
						*sequence_parameter_set = Some(Box::new(sps));
						*picture_parameter_set = Some(Box::new(pps));
					}
					let sps = sequence_parameter_set.as_ref().ok_or_else(|| {
						Error::failed_precondition("H.264 decoder has no sequence parameter set")
					})?;
					let pps = picture_parameter_set.as_ref().ok_or_else(|| {
						Error::failed_precondition("H.264 decoder has no picture parameter set")
					})?;
					let mut coded = nals
						.iter()
						.filter(|nal| matches!(nal.nal_unit_type(), 1 | 5));
					let slice_nal = coded
						.next()
						.ok_or_else(|| Error::invalid_argument("H.264 access unit contains no coded slice"))?;
					if coded.next().is_some() {
						return Err(Error::missing_capability(
							"the H.264 decoder currently accepts one slice per picture",
						));
					}
					let slice = parse_h264_slice_header(slice_nal.payload(), sps, pps)?;
					let backing =
						match output_mode {
							DecodeOutputMode::HostPlanar => DecodedBacking::HostPlanar(
								state.backend.decode_h264(packet.data(), sps, pps, &slice)?,
							),
							DecodeOutputMode::Native => DecodedBacking::Native(
								state
									.backend
									.decode_h264_native(packet.data(), sps, pps, &slice)?,
							),
						};
					Some((
						backing,
						usize::try_from(sps.coded_width()?)
							.map_err(|_| Error::out_of_range("H.264 width exceeds usize"))?,
						usize::try_from(sps.coded_height()?)
							.map_err(|_| Error::out_of_range("H.264 height exceeds usize"))?,
						h264_color_info(sps),
					))
				}
				DecoderCodecState::H265 {
					sequence_parameter_set,
					picture_parameter_set,
				} => {
					let nals = parse_nal_annex_b(packet.data());
					let find_parameter = |nal_type| {
						nals
							.iter()
							.find(|nal| ((nal.payload()[0] >> 1) & 0x3f) == nal_type)
							.map(|nal| nal.payload())
					};
					if sequence_parameter_set.is_none() {
						let vps = parse_h265_vps(find_parameter(32).ok_or_else(|| {
							Error::failed_precondition("initial H.265 packet contains no VPS")
						})?)?;
						let sps = parse_h265_sps(find_parameter(33).ok_or_else(|| {
							Error::failed_precondition("initial H.265 packet contains no SPS")
						})?)?;
						let pps = parse_h265_pps(find_parameter(34).ok_or_else(|| {
							Error::failed_precondition("initial H.265 packet contains no PPS")
						})?)?;
						if sps.coded_width != state.info.width() || sps.coded_height != state.info.height() {
							return Err(Error::invalid_argument(
								"H.265 parameter-set geometry differs from container metadata",
							));
						}
						state.backend.set_h265_parameters(&vps, &sps, &pps)?;
						let sub_layer = usize::try_from(sps.max_sub_layers_minus_1)
							.map_err(|_| Error::out_of_range("H.265 sub-layer index exceeds usize"))?;
						state.reorder_depth =
							usize::try_from(*sps.max_num_reorder_pictures.get(sub_layer).ok_or_else(|| {
								Error::data_loss("H.265 SPS reorder depth has no active sub-layer")
							})?)
							.map_err(|_| Error::out_of_range("H.265 reorder depth exceeds usize"))?;
						*sequence_parameter_set = Some(Box::new(sps));
						*picture_parameter_set = Some(Box::new(pps));
					}
					let sps = sequence_parameter_set.as_ref().ok_or_else(|| {
						Error::failed_precondition("H.265 decoder has no sequence parameter set")
					})?;
					let pps = picture_parameter_set.as_ref().ok_or_else(|| {
						Error::failed_precondition("H.265 decoder has no picture parameter set")
					})?;
					let mut coded = nals
						.iter()
						.filter(|nal| ((nal.payload()[0] >> 1) & 0x3f) < 32);
					let slice_nal = coded
						.next()
						.ok_or_else(|| Error::invalid_argument("H.265 access unit contains no coded slice"))?;
					if coded.next().is_some() {
						return Err(Error::missing_capability(
							"the H.265 decoder currently accepts one slice segment per picture",
						));
					}
					let slice = parse_h265_slice_header(slice_nal.payload(), sps, pps)?;
					let backing =
						match output_mode {
							DecodeOutputMode::HostPlanar => DecodedBacking::HostPlanar(
								state.backend.decode_h265(packet.data(), sps, pps, &slice)?,
							),
							DecodeOutputMode::Native => DecodedBacking::Native(
								state
									.backend
									.decode_h265_native(packet.data(), sps, pps, &slice)?,
							),
						};
					Some((
						backing,
						usize::try_from(sps.coded_width)
							.map_err(|_| Error::out_of_range("H.265 width exceeds usize"))?,
						usize::try_from(sps.coded_height)
							.map_err(|_| Error::out_of_range("H.265 height exceeds usize"))?,
						h265_color_info(sps),
					))
				}
				DecoderCodecState::Av1 {
					parser,
					parameter_sequence,
				} => {
					let mut next_parser = parser.clone();
					let pictures = next_parser.parse_access_unit(packet.data())?;
					if pictures.is_empty() {
						return Err(Error::invalid_argument(
							"AV1 access unit contains no complete picture",
						));
					}
					if pictures
						.iter()
						.filter(|picture| picture.frame.show_existing_frame || picture.frame.show_frame)
						.count()
						> 1
					{
						return Err(Error::missing_capability(
							"one AV1 packet exposes more than one display picture",
						));
					}
					let sequence = &pictures[0].sequence;
					if pictures.iter().any(|picture| picture.sequence != *sequence) {
						return Err(Error::missing_capability(
							"AV1 sequence changes within one access unit",
						));
					}
					if pictures.iter().any(|picture| {
						picture.frame.frame_size_override
							|| picture.frame.use_superres
							|| picture.frame.render_and_frame_size_different
					}) {
						return Err(Error::missing_capability(
							"AV1 per-picture size override, super-resolution, and distinct render size are not admitted",
						));
					}
					if sequence.coded_width() != state.info.width()
						|| sequence.coded_height() != state.info.height()
					{
						return Err(Error::invalid_argument(
							"AV1 sequence geometry differs from container metadata",
						));
					}
					match parameter_sequence {
						Some(current) if current.as_ref() != sequence => {
							return Err(Error::missing_capability(
								"AV1 sequence changes require a new decoder session",
							));
						}
						Some(_) => {}
						None => {
							state.backend.set_av1_parameters(sequence)?;
							*parameter_sequence = Some(Box::new(sequence.clone()));
						}
					}
					state.reorder_depth = 0;
					let mut displayed = None;
					for picture in &pictures {
						let backing = if picture.frame.show_existing_frame {
							match output_mode {
								DecodeOutputMode::HostPlanar => {
									DecodedBacking::HostPlanar(state.backend.show_existing_av1(&picture.frame)?)
								}
								DecodeOutputMode::Native => {
									DecodedBacking::Native(state.backend.show_existing_av1_native(&picture.frame)?)
								}
							}
						} else {
							match output_mode {
								DecodeOutputMode::HostPlanar => {
									DecodedBacking::HostPlanar(state.backend.decode_av1(packet.data(), picture)?)
								}
								DecodeOutputMode::Native => {
									DecodedBacking::Native(state.backend.decode_av1_native(packet.data(), picture)?)
								}
							}
						};
						if picture.frame.show_existing_frame || picture.frame.show_frame {
							displayed = Some(backing);
						}
					}
					*parser = next_parser;
					displayed.map(|backing| {
						(
							backing,
							usize::from(sequence.max_frame_width_minus_1) + 1,
							usize::from(sequence.max_frame_height_minus_1) + 1,
							av1_color_info(sequence),
						)
					})
				}
				DecoderCodecState::Vp9 { parser } => {
					let mut next_parser = parser.clone();
					let pictures = next_parser.parse_access_unit(packet.data())?;
					if pictures.is_empty() {
						return Err(Error::invalid_argument(
							"VP9 access unit contains no complete picture",
						));
					}
					if pictures
						.iter()
						.filter(|picture| picture.show_existing_frame || picture.show_frame)
						.count()
						> 1
					{
						return Err(Error::missing_capability(
							"one VP9 packet exposes more than one display picture",
						));
					}
					if pictures.iter().any(|picture| {
						picture.profile != 0
							|| picture.color.bit_depth != 8
							|| !picture.color.subsampling_x
							|| !picture.color.subsampling_y
					}) {
						return Err(Error::missing_capability(
							"the public VP9 decoder admits Profile 0 8-bit 4:2:0 only",
						));
					}
					if pictures.iter().any(|picture| {
						picture.frame_width != state.info.width()
							|| picture.frame_height != state.info.height()
							|| picture.render_width != picture.frame_width
							|| picture.render_height != picture.frame_height
					}) {
						return Err(Error::missing_capability(
							"VP9 dynamic coded or distinct render extents are not admitted",
						));
					}
					state.reorder_depth = 0;
					let mut displayed = None;
					for picture in &pictures {
						let backing = if picture.show_existing_frame {
							match output_mode {
								DecodeOutputMode::HostPlanar => {
									DecodedBacking::HostPlanar(state.backend.show_existing_vp9(picture)?)
								}
								DecodeOutputMode::Native => {
									DecodedBacking::Native(state.backend.show_existing_vp9_native(picture)?)
								}
							}
						} else {
							match output_mode {
								DecodeOutputMode::HostPlanar => {
									DecodedBacking::HostPlanar(state.backend.decode_vp9(packet.data(), picture)?)
								}
								DecodeOutputMode::Native => {
									DecodedBacking::Native(state.backend.decode_vp9_native(packet.data(), picture)?)
								}
							}
						};
						if picture.show_existing_frame || picture.show_frame {
							displayed = Some((backing, picture.color));
						}
					}
					*parser = next_parser;
					displayed.map(|(backing, color)| {
						(
							backing,
							state.info.width() as usize,
							state.info.height() as usize,
							vp9_color_info(color),
						)
					})
				}
			};
		state.require_random_access = false;
		state.output_mode = Some(output_mode);
		let Some((backing, width, height, color)) = decoded else {
			return Ok(None);
		};
		let timing = packet_timing(packet, state.info)?;
		let frame = match backing {
			DecodedBacking::HostPlanar(bytes) => {
				VideoFrame::from_yuv420p(bytes, width, height, timing, color)?
			}
			DecodedBacking::Native(frame) => VideoFrame::from_native(frame, timing, color),
		};
		let sequence = state.next_decode_sequence;
		state.next_decode_sequence = state
			.next_decode_sequence
			.checked_add(1)
			.ok_or_else(|| Error::resource_exhausted("video decode sequence exhausted"))?;
		state
			.pending_display
			.insert((packet.presentation_timestamp(), sequence), frame);
		if state.pending_display.len() > state.reorder_depth {
			Ok(state.pending_display.pop_first().map(|(_, frame)| frame))
		} else {
			Ok(None)
		}
	}

	/// Reset stream sequencing at a seek boundary.
	///
	/// Return every delayed frame in presentation order and require the next
	/// packet to be random access. A second flush returns an empty vector.
	pub fn flush(&mut self) -> Result<Vec<VideoFrame>> {
		let state = self
			.state
			.as_mut()
			.ok_or_else(|| Error::failed_precondition("VideoDecoder is closed"))?;
		state.require_random_access = true;
		state.output_mode = None;
		Ok(
			std::mem::take(&mut state.pending_display)
				.into_values()
				.collect(),
		)
	}

	/// Close the decoder and release its session resources.
	pub fn close(&mut self) {
		self.state = None;
	}

	/// Return whether the decoder still accepts packets.
	#[must_use]
	pub const fn is_open(&self) -> bool {
		self.state.is_some()
	}

	/// Return the selected stream metadata while open.
	#[must_use]
	pub fn info(&self) -> Option<VideoContainerInfo> {
		self.state.as_ref().map(|state| state.info)
	}
}

fn packet_timing(packet: &VideoPacket, info: VideoContainerInfo) -> Result<VideoFrameTiming> {
	let time_base = info.time_base();
	let pts = ticks_to_duration(packet.presentation_timestamp(), time_base)?;
	let duration = (packet.duration() != 0)
		.then(|| ticks_to_duration(u64::from(packet.duration()), time_base))
		.transpose()?;
	VideoFrameTiming::new(pts, duration)
}

fn ticks_to_duration(ticks: u64, time_base: super::VideoTimeBase) -> Result<Duration> {
	let nanos = u128::from(ticks)
		.checked_mul(u128::from(time_base.numerator()))
		.and_then(|value| value.checked_mul(1_000_000_000))
		.ok_or_else(|| Error::out_of_range("video timestamp overflows nanoseconds"))?
		/ u128::from(time_base.denominator());
	let seconds = u64::try_from(nanos / 1_000_000_000)
		.map_err(|_| Error::out_of_range("video timestamp exceeds Duration"))?;
	let subsec_nanos = u32::try_from(nanos % 1_000_000_000)
		.map_err(|_| Error::out_of_range("video timestamp remainder exceeds u32"))?;
	Ok(Duration::new(seconds, subsec_nanos))
}

fn h264_color_info(sps: &H264SequenceParameterSet) -> VideoColorInfo {
	let Some(signal) = sps.vui.as_ref().and_then(|vui| vui.video_signal) else {
		return VideoColorInfo::unspecified();
	};
	video_color_info(
		signal
			.colour_description
			.map(|description| description.matrix_coefficients),
		signal.full_range,
	)
}

fn h265_color_info(sps: &H265SequenceParameterSet) -> VideoColorInfo {
	let Some(signal) = sps.vui.as_ref().and_then(|vui| vui.video_signal) else {
		return VideoColorInfo::unspecified();
	};
	video_color_info(
		signal
			.colour_description
			.map(|description| description.matrix_coefficients),
		signal.full_range,
	)
}

fn av1_color_info(sequence: &Av1SequenceHeader) -> VideoColorInfo {
	video_color_info(
		sequence
			.color
			.color_description
			.map(|description| description.matrix_coefficients),
		sequence.color.full_range,
	)
}

fn vp9_color_info(color: super::vp9::Vp9ColorConfig) -> VideoColorInfo {
	let matrix = match color.color_space {
		1 | 3 => VideoColorMatrix::Bt601,
		2 => VideoColorMatrix::Bt709,
		5 => VideoColorMatrix::Bt2020,
		_ => VideoColorMatrix::Unspecified,
	};
	VideoColorInfo::new(
		matrix,
		if color.full_range {
			VideoColorRange::Full
		} else {
			VideoColorRange::Limited
		},
	)
}

fn video_color_info(matrix_coefficients: Option<u8>, full_range: bool) -> VideoColorInfo {
	let matrix = match matrix_coefficients {
		Some(5 | 6) => VideoColorMatrix::Bt601,
		Some(1) => VideoColorMatrix::Bt709,
		Some(9) => VideoColorMatrix::Bt2020,
		_ => VideoColorMatrix::Unspecified,
	};
	VideoColorInfo::new(
		matrix,
		if full_range {
			VideoColorRange::Full
		} else {
			VideoColorRange::Limited
		},
	)
}
