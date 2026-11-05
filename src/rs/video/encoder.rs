//! Hardware-independent encoder settings, exact device preflight, and the
//! public hardware encoder session.
//!
//! Donor: `source/cpp/include/oa/vision/videoEncoder.h` and
//! `source/cpp/lib/oa/vision/video/encoder/videoEncoder.cpp`.

use crate::{Engine, Error, Result, runtime::VideoEncoderBackend};

use super::{
	VideoChromaSubsampling, VideoComponentBitDepth, VideoEncodeCapabilities,
	VideoEncodeCodecCapabilities, VideoEncodeProfile, VideoExtent,
};
pub use crate::video::types::VideoEncodeConfig;

/// Rate-control policy requested for a future Vulkan Video encoder session.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VideoRateControl {
	ConstantQp,
	Cbr,
	Vbr,
}

impl VideoEncodeConfig {
	/// Construct the donor's default encode settings for a visible extent.
	pub const fn new(profile: VideoEncodeProfile, width: u32, height: u32) -> Self {
		Self {
			profile,
			width,
			height,
			frame_rate: 30,
			gop_size: 30,
			rate_control: VideoRateControl::ConstantQp,
			bitrate: 4_000_000,
			max_bitrate: 0,
			constant_qp: 26,
			quality_level: 0,
			async_depth: 3,
			max_b_frames: 0,
			max_dpb_slots: 0,
		}
	}

	/// Validate this request against the exact queried codec profile.
	///
	/// The returned coded extent and resolved peak bitrate are inputs to a
	/// future session. No device resource is allocated by this method.
	///
	/// # Errors
	///
	/// Returns an error for an unsupported profile, invalid geometry or rate
	/// control, arithmetic overflow, or device limits that cannot admit the
	/// requested stream.
	pub fn validate_for_device(self, caps: VideoEncodeCapabilities) -> Result<VideoEncodePlan> {
		if self.profile != caps.profile() {
			return Err(Error::invalid_argument(
				"encode capabilities belong to a different profile",
			));
		}
		let donor_profile = matches!(
			self.profile,
			VideoEncodeProfile::H264 {
				profile: super::H264Profile::High,
				chroma_subsampling: VideoChromaSubsampling::Yuv420,
				luma_bit_depth: VideoComponentBitDepth::Eight,
				chroma_bit_depth: VideoComponentBitDepth::Eight,
			} | VideoEncodeProfile::H265 {
				profile: super::H265Profile::Main,
				chroma_subsampling: VideoChromaSubsampling::Yuv420,
				luma_bit_depth: VideoComponentBitDepth::Eight,
				chroma_bit_depth: VideoComponentBitDepth::Eight,
			}
		);
		if !donor_profile {
			return Err(Error::missing_capability(
				"encoder picture recording is limited to H.264 High and H.265 Main 8-bit 4:2:0",
			));
		}
		if !matches!(
			(self.profile, caps.codec()),
			(
				VideoEncodeProfile::H264 { .. },
				VideoEncodeCodecCapabilities::H264 { .. }
			) | (
				VideoEncodeProfile::H265 { .. },
				VideoEncodeCodecCapabilities::H265 { .. }
			)
		) {
			return Err(Error::invalid_argument(
				"encode capability codec does not match its profile",
			));
		}
		if self.width == 0 || self.height == 0 || self.frame_rate == 0 || self.gop_size == 0 {
			return Err(Error::invalid_argument(
				"encode extent, frame rate, and GOP size must be non-zero",
			));
		}
		if self.async_depth == 0 {
			return Err(Error::invalid_argument(
				"encode async depth must be non-zero",
			));
		}
		if self.max_b_frames != 0 {
			return Err(Error::missing_capability(
				"B-picture encoding requires a reorder and reference path",
			));
		}
		if self.constant_qp > 51 {
			return Err(Error::invalid_argument(
				"encode constant QP must be in 0..=51",
			));
		}
		if self.quality_level >= caps.max_quality_levels() {
			return Err(Error::missing_capability(
				"encode quality level exceeds device capabilities",
			));
		}
		let supported_rate_control = match self.rate_control {
			VideoRateControl::ConstantQp => caps.supports_constant_qp(),
			VideoRateControl::Cbr => caps.supports_cbr(),
			VideoRateControl::Vbr => caps.supports_vbr(),
		};
		if !supported_rate_control {
			return Err(Error::missing_capability(
				"requested encode rate control is unavailable",
			));
		}
		if self.rate_control != VideoRateControl::ConstantQp && caps.max_rate_control_layers() == 0 {
			return Err(Error::missing_capability(
				"device reports no encode rate-control layers",
			));
		}
		if self.rate_control != VideoRateControl::ConstantQp && self.bitrate == 0 {
			return Err(Error::invalid_argument(
				"CBR/VBR target bitrate must be non-zero",
			));
		}
		if self.max_bitrate != 0 && self.max_bitrate < self.bitrate {
			return Err(Error::invalid_argument(
				"peak bitrate must be at least target bitrate",
			));
		}
		if caps.max_bitrate() != 0
			&& (u64::from(self.bitrate) > caps.max_bitrate()
				|| (self.max_bitrate != 0 && u64::from(self.max_bitrate) > caps.max_bitrate()))
		{
			return Err(Error::missing_capability(
				"requested bitrate exceeds device capabilities",
			));
		}
		let peak_bitrate = match self.rate_control {
			VideoRateControl::ConstantQp => 0,
			VideoRateControl::Cbr => self.bitrate,
			VideoRateControl::Vbr if self.max_bitrate != 0 => self.max_bitrate,
			VideoRateControl::Vbr => self.bitrate.saturating_mul(2),
		};
		let peak_bitrate = if caps.max_bitrate() == 0 {
			peak_bitrate
		} else {
			peak_bitrate.min(caps.max_bitrate().min(u64::from(u32::MAX)) as u32)
		};
		if self.rate_control == VideoRateControl::Vbr && peak_bitrate < self.bitrate {
			return Err(Error::missing_capability(
				"device peak bitrate is below requested target",
			));
		}
		if let VideoEncodeCodecCapabilities::H265 { min_qp, max_qp, .. } = caps.codec()
			&& (self.constant_qp < min_qp.max(0) as u32 || self.constant_qp > max_qp.max(0) as u32)
		{
			return Err(Error::missing_capability(
				"H.265 QP is outside device limits",
			));
		}
		let min = caps.min_coded_extent();
		let max = caps.max_coded_extent();
		let granularity = caps.picture_access_granularity();
		let coded_width = align_up(self.width, granularity.width.max(16))?;
		let coded_height = align_up(self.height, granularity.height.max(16))?;
		if coded_width < min.width || coded_height < min.height {
			return Err(Error::missing_capability(
				"aligned encode extent is below device minimum",
			));
		}
		if coded_width > max.width || coded_height > max.height {
			return Err(Error::missing_capability(
				"aligned encode extent exceeds device maximum",
			));
		}
		if self.gop_size > 1 && (caps.max_dpb_slots() < 2 || caps.max_active_reference_pictures() == 0)
		{
			return Err(Error::missing_capability(
				"P-picture encoding requires two DPB slots and an active reference",
			));
		}
		let h264_level = if let VideoEncodeCodecCapabilities::H264 { max_level, .. } = caps.codec() {
			let level =
				h264_level_for_extent(self.width, self.height, self.frame_rate).ok_or_else(|| {
					Error::missing_capability("H.264 extent or frame rate exceeds supported levels")
				})?;
			if level > max_level {
				return Err(Error::missing_capability(
					"H.264 extent and frame rate require an unsupported device level",
				));
			}
			Some(level)
		} else {
			None
		};
		Ok(VideoEncodePlan {
			coded_extent: VideoExtent {
				width: coded_width,
				height: coded_height,
			},
			peak_bitrate,
			h264_level,
		})
	}
}

/// Validated, hardware-independent settings for session creation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VideoEncodePlan {
	coded_extent: VideoExtent,
	peak_bitrate: u32,
	h264_level: Option<u32>,
}

impl VideoEncodePlan {
	pub const fn coded_extent(self) -> VideoExtent {
		self.coded_extent
	}
	pub const fn peak_bitrate(self) -> u32 {
		self.peak_bitrate
	}
	/// Numeric H.264 level IDC selected from the donor's progressive High table.
	pub const fn h264_level(self) -> Option<u32> {
		self.h264_level
	}
}

// Donor: videoEncoder.cpp::h264LevelForExtent, Table A-1 subset. Division
// avoids overflowing while checking macroblocks per second.
fn h264_level_for_extent(width: u32, height: u32, frame_rate: u32) -> Option<u32> {
	let width_mbs = u64::from(width).div_ceil(16);
	let height_mbs = u64::from(height).div_ceil(16);
	let frame_mbs = width_mbs * height_mbs;
	const LIMITS: [(u32, u64, u64); 7] = [
		(42, 522_240, 8_704),
		(50, 589_824, 22_080),
		(51, 983_040, 36_864),
		(52, 2_073_600, 36_864),
		(60, 4_177_920, 139_264),
		(61, 8_355_840, 139_264),
		(62, 16_711_680, 139_264),
	];
	LIMITS.iter().find_map(|&(level, per_second, max_frame)| {
		(frame_mbs <= max_frame
			&& frame_mbs <= per_second / u64::from(frame_rate)
			&& width_mbs * width_mbs <= 8 * max_frame
			&& height_mbs * height_mbs <= 8 * max_frame)
			.then_some(level)
	})
}

fn align_up(value: u32, granularity: u32) -> Result<u32> {
	let blocks = value
		.checked_add(granularity - 1)
		.ok_or_else(|| Error::out_of_range("encode coded extent alignment overflow"))?
		/ granularity;
	blocks
		.checked_mul(granularity)
		.ok_or_else(|| Error::out_of_range("encode coded extent alignment overflow"))
}

/// Stateful hardware encoder session for H.264 High or H.265 Main.
///
/// One encoder owns a private Vulkan Video encode session, per-slot NV12 input
/// pictures, and the RGBA→NV12 conversion on the selected [`Engine`]'s queues.
/// Frames enter as packed RGBA8 bytes sized exactly
/// `visible_extent().width * visible_extent().height * 4`; conversion, GOP
/// cadence, rate control, DPB management, and Annex-B framing stay private.
/// Every emitted [`EncodedVideoPacket`] is one access unit; keyframe packets
/// carry the session's cached SPS/PPS (and VPS for H.265) parameter sets so
/// each random-access point is self-describing.
///
/// `encode` waits for the submitted frame and returns its packet. `flush`
/// waits for every outstanding encode job and drains the remainder — required
/// before `close` when asynchronous submissions left frames in flight.
pub struct VideoEncoder {
	state: Option<VideoEncoderBackend>,
}

impl VideoEncoder {
	/// Create a session admitted by exact capability preflight.
	///
	/// Settings are validated against the engine's queried encode capabilities;
	/// session creation then performs the Vulkan-level admission a second time.
	/// A device without an encode queue family for the requested codec profile
	/// fails here, not on the first `encode` call.
	///
	/// # Errors
	///
	/// Returns an error when settings fail preflight or the engine's device
	/// does not admit a hardware encode session for the profile.
	pub fn new(engine: &Engine, settings: VideoEncodeConfig) -> Result<Self> {
		Ok(Self {
			state: Some(VideoEncoderBackend::create(engine, settings)?),
		})
	}

	/// Encode one packed RGBA8 frame and return its access unit.
	///
	/// `rgba` must contain exactly
	/// `visible_extent().width * visible_extent().height * 4` bytes in
	/// row-major RGBA8 order. The frame is converted to the session's coded
	/// NV12 extent, queued on the hardware encode queue, and awaited before the
	/// packet is returned; presentation timestamps are in microseconds.
	///
	/// # Errors
	///
	/// Returns an error after close, for a mismatched input size, or when
	/// conversion, encode submission, feedback, or bitstream readback fails.
	pub fn encode(
		&mut self,
		rgba: &[u8],
		presentation_timestamp_micros: u64,
	) -> Result<super::EncodedVideoPacket> {
		self
			.state
			.as_mut()
			.ok_or_else(|| Error::failed_precondition("VideoEncoder is closed"))?
			.encode_rgba(rgba, presentation_timestamp_micros)
	}

	/// Wait for every outstanding encode job and drain all remaining packets.
	///
	/// Synchronous [`encode`](Self::encode) leaves nothing pending; flush is the
	/// tail drain for asynchronous submissions and returns an empty list when
	/// the session is already drained.
	///
	/// # Errors
	///
	/// Returns an error after close or when an outstanding job fails feedback
	/// or bitstream readback.
	pub fn flush(&mut self) -> Result<Vec<super::EncodedVideoPacket>> {
		self
			.state
			.as_mut()
			.ok_or_else(|| Error::failed_precondition("VideoEncoder is closed"))?
			.flush()
	}

	/// Close the encoder and release its session resources.
	///
	/// Unharvested asynchronous jobs are released without being drained;
	/// call [`flush`](Self::flush) first to keep every submitted frame.
	pub fn close(&mut self) {
		self.state = None;
	}

	/// Return whether the encoder still accepts frames.
	#[must_use]
	pub const fn is_open(&self) -> bool {
		self.state.is_some()
	}

	/// Return the aligned NV12 coded extent the driver encodes at.
	#[must_use]
	pub fn coded_extent(&self) -> Option<VideoExtent> {
		self.state.as_ref().map(VideoEncoderBackend::coded_extent)
	}

	/// Return the visible input extent expected by `encode`.
	#[must_use]
	pub fn visible_extent(&self) -> Option<VideoExtent> {
		self.state.as_ref().map(VideoEncoderBackend::visible_extent)
	}

	/// Borrow the session's cached codec parameter sets in Annex-B form.
	///
	/// H.264 sessions carry SPS+PPS and H.265 sessions carry VPS+SPS+PPS; the
	/// bytes are already prepended to every keyframe packet, so muxers that
	/// need out-of-band codec configuration should derive it from a keyframe
	/// or use this reference directly.
	#[must_use]
	pub fn codec_headers(&self) -> Option<&[u8]> {
		self.state.as_ref().map(VideoEncoderBackend::cached_headers)
	}

	/// Queue one RGBA8 frame asynchronously and harvest completed packets.
	///
	/// Crate-internal media bridge — the donor's `VideoEncoderAccess` route
	/// used by `VideoRecorder`. `color_space` selects the conversion matrix
	/// (0 = BT.601, 1 = BT.709, 2 = BT.2020); `full_range` selects PC range.
	pub(crate) fn submit_rgba(
		&mut self,
		rgba: &[u8],
		presentation_timestamp_micros: u64,
		color_space: u32,
		full_range: bool,
	) -> Result<Vec<super::EncodedVideoPacket>> {
		self
			.state
			.as_mut()
			.ok_or_else(|| Error::failed_precondition("VideoEncoder is closed"))?
			.submit_rgba(rgba, presentation_timestamp_micros, color_space, full_range)
	}
}
