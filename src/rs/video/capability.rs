//! Backend-neutral reporting for hardware video queue and codec support.

use crate::{Engine, Result};

/// Chroma sampling carried by a video codec profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VideoChromaSubsampling {
	Monochrome,
	Yuv420,
	Yuv422,
	Yuv444,
}

/// Component precision carried by a video codec profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VideoComponentBitDepth {
	Eight,
	Ten,
	Twelve,
}

/// H.264 profile identity used for exact decoder capability queries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum H264Profile {
	Baseline,
	Main,
	High,
	High444Predictive,
}

/// H.264 decoded-picture layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum H264PictureLayout {
	Progressive,
	InterlacedInterleavedLines,
	InterlacedSeparatePlanes,
}

/// H.265 profile identity used for exact decoder capability queries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum H265Profile {
	Main,
	Main10,
	MainStillPicture,
	FormatRangeExtensions,
	ScreenContentCodingExtensions,
}

/// AV1 profile identity used for exact decoder capability queries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Av1Profile {
	Main,
	High,
	Professional,
}

/// One complete decode profile presented to the selected device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VideoDecodeProfile {
	H264 {
		profile: H264Profile,
		picture_layout: H264PictureLayout,
		chroma_subsampling: VideoChromaSubsampling,
		luma_bit_depth: VideoComponentBitDepth,
		chroma_bit_depth: VideoComponentBitDepth,
	},
	H265 {
		profile: H265Profile,
		chroma_subsampling: VideoChromaSubsampling,
		luma_bit_depth: VideoComponentBitDepth,
		chroma_bit_depth: VideoComponentBitDepth,
	},
	Av1 {
		profile: Av1Profile,
		film_grain_support: bool,
		chroma_subsampling: VideoChromaSubsampling,
		luma_bit_depth: VideoComponentBitDepth,
		chroma_bit_depth: VideoComponentBitDepth,
	},
}

impl VideoDecodeProfile {
	/// Construct the common progressive H.264 4:2:0 8-bit profile.
	pub const fn h264_420_8bit(profile: H264Profile) -> Self {
		Self::H264 {
			profile,
			picture_layout: H264PictureLayout::Progressive,
			chroma_subsampling: VideoChromaSubsampling::Yuv420,
			luma_bit_depth: VideoComponentBitDepth::Eight,
			chroma_bit_depth: VideoComponentBitDepth::Eight,
		}
	}

	/// Construct the common H.265 4:2:0 profile at one component precision.
	pub const fn h265_420(profile: H265Profile, bit_depth: VideoComponentBitDepth) -> Self {
		Self::H265 {
			profile,
			chroma_subsampling: VideoChromaSubsampling::Yuv420,
			luma_bit_depth: bit_depth,
			chroma_bit_depth: bit_depth,
		}
	}

	/// Construct the common AV1 4:2:0 profile at one component precision.
	pub const fn av1_420(
		profile: Av1Profile,
		bit_depth: VideoComponentBitDepth,
		film_grain_support: bool,
	) -> Self {
		Self::Av1 {
			profile,
			film_grain_support,
			chroma_subsampling: VideoChromaSubsampling::Yuv420,
			luma_bit_depth: bit_depth,
			chroma_bit_depth: bit_depth,
		}
	}
}

/// Two-dimensional device limit expressed without Vulkan types.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VideoExtent {
	pub width: u32,
	pub height: u32,
}

/// Multi-plane YUV image layout admitted by the Video runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum VideoPixelFormat {
	/// Three-plane 8-bit 4:2:0 (I420-style plane layout).
	Yuv420Planar8,
	/// Two-plane 8-bit 4:2:0 (NV12-style plane layout).
	Nv12,
	/// Three-plane 10-bit 4:2:0 stored in 16-bit components.
	Yuv420Planar10,
	/// Two-plane 10-bit 4:2:0 stored in 16-bit components (P010).
	P010,
	/// Two-plane 12-bit 4:2:0 stored in 16-bit components (P012).
	P012,
}

/// One Vulkan-video-compatible image representation without Vulkan types.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VideoImageFormat {
	pub(crate) pixel_format: VideoPixelFormat,
	pub(crate) optimal_tiling: bool,
	pub(crate) sampled: bool,
	pub(crate) storage: bool,
	pub(crate) transfer_source: bool,
	pub(crate) transfer_destination: bool,
}

impl VideoImageFormat {
	/// Return the semantic plane and component layout.
	pub const fn pixel_format(self) -> VideoPixelFormat {
		self.pixel_format
	}

	/// Return whether the device requires or supports optimal image tiling here.
	pub const fn optimal_tiling(self) -> bool {
		self.optimal_tiling
	}

	/// Return whether the reported image usage includes sampling.
	pub const fn sampled(self) -> bool {
		self.sampled
	}

	/// Return whether the reported image usage includes storage access.
	pub const fn storage(self) -> bool {
		self.storage
	}

	/// Return whether the reported image usage includes transfer-source access.
	pub const fn transfer_source(self) -> bool {
		self.transfer_source
	}

	/// Return whether the reported image usage includes transfer-destination access.
	pub const fn transfer_destination(self) -> bool {
		self.transfer_destination
	}
}

/// Decode-output and decoded-picture-buffer formats for one exact profile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VideoDecodeFormats {
	pub(crate) profile: VideoDecodeProfile,
	pub(crate) output: Vec<VideoImageFormat>,
	pub(crate) dpb: Vec<VideoImageFormat>,
	pub(crate) unrecognized_output_formats: usize,
	pub(crate) unrecognized_dpb_formats: usize,
}

impl VideoDecodeFormats {
	/// Return the exact profile used for this query.
	pub const fn profile(&self) -> VideoDecodeProfile {
		self.profile
	}

	/// Borrow OARS-recognized decoded-output image formats.
	pub fn output(&self) -> &[VideoImageFormat] {
		&self.output
	}

	/// Borrow OARS-recognized decoded-picture-buffer image formats.
	pub fn dpb(&self) -> &[VideoImageFormat] {
		&self.dpb
	}

	/// Return how many backend output formats OARS cannot yet represent.
	pub const fn unrecognized_output_formats(&self) -> usize {
		self.unrecognized_output_formats
	}

	/// Return how many backend DPB formats OARS cannot yet represent.
	pub const fn unrecognized_dpb_formats(&self) -> usize {
		self.unrecognized_dpb_formats
	}
}

/// Codec-specific maximum level reported for a decode profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VideoDecodeLevel {
	H264(u32),
	H265(u32),
	Av1(u32),
}

/// Exact device limits for one supported decode profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VideoDecodeCapabilities {
	pub(crate) profile: VideoDecodeProfile,
	pub(crate) min_coded_extent: VideoExtent,
	pub(crate) max_coded_extent: VideoExtent,
	pub(crate) picture_access_granularity: VideoExtent,
	pub(crate) min_bitstream_offset_alignment: u64,
	pub(crate) min_bitstream_size_alignment: u64,
	pub(crate) max_dpb_slots: u32,
	pub(crate) max_active_reference_pictures: u32,
	pub(crate) level: VideoDecodeLevel,
	pub(crate) field_offset_granularity: Option<(i32, i32)>,
	pub(crate) dpb_and_output_coincide: bool,
	pub(crate) dpb_and_output_distinct: bool,
	pub(crate) protected_content: bool,
	pub(crate) separate_reference_images: bool,
}

impl VideoDecodeCapabilities {
	/// Return the exact profile used for this query.
	pub const fn profile(self) -> VideoDecodeProfile {
		self.profile
	}

	/// Return the smallest supported coded extent.
	pub const fn min_coded_extent(self) -> VideoExtent {
		self.min_coded_extent
	}

	/// Return the largest supported coded extent.
	pub const fn max_coded_extent(self) -> VideoExtent {
		self.max_coded_extent
	}

	/// Return the device's coded-picture access granularity.
	pub const fn picture_access_granularity(self) -> VideoExtent {
		self.picture_access_granularity
	}

	/// Return the minimum encoded-bitstream buffer offset alignment in bytes.
	pub const fn min_bitstream_offset_alignment(self) -> u64 {
		self.min_bitstream_offset_alignment
	}

	/// Return the minimum encoded-bitstream buffer size alignment in bytes.
	pub const fn min_bitstream_size_alignment(self) -> u64 {
		self.min_bitstream_size_alignment
	}

	/// Return the maximum number of decoded-picture-buffer slots.
	pub const fn max_dpb_slots(self) -> u32 {
		self.max_dpb_slots
	}

	/// Return the maximum number of simultaneously active reference pictures.
	pub const fn max_active_reference_pictures(self) -> u32 {
		self.max_active_reference_pictures
	}

	/// Return the codec-specific maximum supported level.
	pub const fn level(self) -> VideoDecodeLevel {
		self.level
	}

	/// Return H.264 field-offset granularity, or `None` for another codec.
	pub const fn field_offset_granularity(self) -> Option<(i32, i32)> {
		self.field_offset_granularity
	}

	/// Return whether decoded output may alias its DPB picture.
	pub const fn dpb_and_output_coincide(self) -> bool {
		self.dpb_and_output_coincide
	}

	/// Return whether decoded output may be distinct from its DPB picture.
	pub const fn dpb_and_output_distinct(self) -> bool {
		self.dpb_and_output_distinct
	}

	/// Return whether this profile supports protected video sessions.
	pub const fn protected_content(self) -> bool {
		self.protected_content
	}

	/// Return whether each DPB slot may use a separate reference image.
	pub const fn separate_reference_images(self) -> bool {
		self.separate_reference_images
	}
}

/// Hardware video queue and codec-extension capabilities for the Engine's device.
///
/// Hardware advertisement is distinct from OA session availability. The
/// Engine creation enables advertised video queue/codec extensions, while
/// [`VideoDeviceCapabilities::decoder_sessions_available`] and
/// [`VideoDeviceCapabilities::encoder_sessions_available`] remain false until
/// OA has a complete corresponding session path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VideoDeviceCapabilities {
	pub(crate) decode_queue_family: Option<u32>,
	pub(crate) encode_queue_family: Option<u32>,
	pub(crate) decode_result_status_queries: bool,
	pub(crate) encode_result_status_queries: bool,
	pub(crate) h264_decode: bool,
	pub(crate) h265_decode: bool,
	pub(crate) av1_decode: bool,
	pub(crate) vp9_decode: bool,
	pub(crate) h264_encode: bool,
	pub(crate) h265_encode: bool,
	pub(crate) av1_encode: bool,
	pub(crate) video_queues_enabled: bool,
	pub(crate) decoder_sessions_available: bool,
	pub(crate) encoder_sessions_available: bool,
}

impl VideoDeviceCapabilities {
	/// Return the first physical-device queue family advertising video decode.
	pub const fn decode_queue_family(self) -> Option<u32> {
		self.decode_queue_family
	}

	/// Return the first physical-device queue family advertising video encode.
	pub const fn encode_queue_family(self) -> Option<u32> {
		self.encode_queue_family
	}

	/// Return whether the selected video-decode queue supports operation-status queries.
	pub const fn supports_decode_result_status_queries(self) -> bool {
		self.decode_result_status_queries
	}

	/// Return whether the selected video-encode queue supports operation-status queries.
	pub const fn supports_encode_result_status_queries(self) -> bool {
		self.encode_result_status_queries
	}

	/// Return whether the physical device advertises H.264 decode.
	pub const fn supports_h264_decode(self) -> bool {
		self.h264_decode
	}

	/// Return whether the physical device advertises H.265 decode.
	pub const fn supports_h265_decode(self) -> bool {
		self.h265_decode
	}

	/// Return whether the physical device advertises AV1 decode.
	pub const fn supports_av1_decode(self) -> bool {
		self.av1_decode
	}

	/// Return whether the physical device advertises VP9 decode.
	pub const fn supports_vp9_decode(self) -> bool {
		self.vp9_decode
	}

	/// Return whether the physical device advertises H.264 encode.
	pub const fn supports_h264_encode(self) -> bool {
		self.h264_encode
	}

	/// Return whether the physical device advertises H.265 encode.
	pub const fn supports_h265_encode(self) -> bool {
		self.h265_encode
	}

	/// Return whether the physical device advertises AV1 encode.
	pub const fn supports_av1_encode(self) -> bool {
		self.av1_encode
	}

	/// Return whether the Engine enabled the advertised Vulkan Video queue extensions.
	pub const fn video_queues_enabled(self) -> bool {
		self.video_queues_enabled
	}

	/// Return whether this Engine currently has an enabled decoder-session path.
	pub const fn decoder_sessions_available(self) -> bool {
		self.decoder_sessions_available
	}

	/// Return whether this Engine currently has an enabled encoder-session path.
	pub const fn encoder_sessions_available(self) -> bool {
		self.encoder_sessions_available
	}
}

/// Query Vulkan Video queues and codec extensions on the Engine's selected device.
///
/// This query reports both physical-device advertisement and whether Engine
/// creation enabled video queues, without implying a codec session exists.
///
/// # Errors
///
/// Returns an error when Vulkan device-extension or queue-family enumeration
/// fails.
pub fn query_device_capabilities(engine: &Engine) -> Result<VideoDeviceCapabilities> {
	engine.query_video_device_capabilities()
}

/// Query the selected device's exact limits for one decoder profile.
///
/// Extension advertisement alone does not prove that a particular profile,
/// chroma layout, bit depth, or codec option is usable. This query presents the
/// complete profile to Vulkan and returns `MissingCapability` when rejected.
///
/// # Errors
///
/// Returns an error when the codec extension is unavailable, the exact profile
/// is unsupported, or the Vulkan capability query fails.
pub fn query_decode_capabilities(
	engine: &Engine,
	profile: VideoDecodeProfile,
) -> Result<VideoDecodeCapabilities> {
	engine.query_video_decode_capabilities(profile)
}

/// Enumerate decoded-output and DPB image formats for one exact profile.
///
/// # Errors
///
/// Returns an error when the profile or its codec extension is unsupported, a
/// Vulkan enumeration fails, or the driver reports an unreasonable result size.
pub fn query_decode_formats(
	engine: &Engine,
	profile: VideoDecodeProfile,
) -> Result<VideoDecodeFormats> {
	engine.query_video_decode_formats(profile)
}
