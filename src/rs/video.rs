//! Video-frame values, codecs, containers, and stateful stream sessions.
//!
//! The first admitted value path wraps one packed [`crate::Image`] while
//! preserving frame timing and source color metadata. Bounded MP4 demux and
//! streaming AVC/HEVC mux sessions own explicit close/finalize behavior. The
//! public hardware decoder admits progressive H.264 Baseline/Main/High, H.265
//! Main, AV1 Main 8-bit 4:2:0 without film grain, and VP9 Profile 0 8-bit
//! 4:2:0. It returns either host-retained planar frames or opaque retained
//! native frames.

mod av1;
mod bitstream;
mod capability;
mod decoder;
mod demux;
mod frame;
mod h264;
mod h265;
mod mux;
mod nal;
mod player;
mod texture;
pub(crate) mod vp9;

pub(crate) use av1::{Av1Parser, Av1Picture};
pub(crate) use vp9::{Vp9Parser, Vp9Picture};

pub use av1::{
	Av1AccessUnitInfo, Av1ChromaSamplePosition, Av1CodingToolChoice, Av1ColorConfig,
	Av1ColorDescription, Av1FrameHeader, Av1FrameType, Av1InterpolationFilter, Av1Obu, Av1ObuType,
	Av1OperatingPoint, Av1ReferenceState, Av1RestorationType, Av1SequenceHeader, Av1TileGroup,
	Av1TimingInfo, Av1TransformMode, inspect_av1_access_unit, parse_av1_frame_header, parse_av1_obus,
	parse_av1_sequence_header, parse_av1_tile_group,
};
pub use capability::{
	Av1Profile, H264PictureLayout, H264Profile, H265Profile, VideoChromaSubsampling,
	VideoComponentBitDepth, VideoDecodeCapabilities, VideoDecodeFormats, VideoDecodeLevel,
	VideoDecodeProfile, VideoDeviceCapabilities, VideoEncodeCapabilities,
	VideoEncodeCodecCapabilities, VideoEncodeFormats, VideoEncodeProfile, VideoExtent,
	VideoImageFormat, VideoPixelFormat, Vp9Profile, query_decode_capabilities, query_decode_formats,
	query_device_capabilities, query_encode_capabilities, query_encode_formats,
};
pub use decoder::VideoDecoder;
pub use demux::{
	VideoCodec, VideoContainerInfo, VideoContainerKind, VideoDemuxer, VideoPacket, VideoTimeBase,
	length_prefixed_to_annex_b,
};
pub use frame::{VideoColorInfo, VideoColorMatrix, VideoColorRange, VideoFrame, VideoFrameTiming};
pub use h264::{
	H264AspectRatio, H264BitstreamRestriction, H264ChromaLocation, H264ColourDescription,
	H264CpbEntry, H264HrdParameters, H264MemoryManagementControl, H264PictureParameterSet,
	H264ScalingLists, H264SequenceParameterSet, H264SliceHeader, H264SliceType, H264TimingInfo,
	H264VideoSignal, H264VuiParameters, parse_h264_pps, parse_h264_slice_header, parse_h264_sps,
};
pub use h265::{
	H265AspectRatio, H265BitstreamRestriction, H265ChromaLocation, H265ColourDescription,
	H265CpbEntry, H265DecodedPictureBuffer, H265HrdParameters, H265LongTermReferencePicture,
	H265PcmParameters, H265PictureParameterSet, H265ProfileTierLevel, H265ScalingLists,
	H265SequenceParameterSet, H265ShortTermReferencePictureSet, H265SliceHeader, H265SliceType,
	H265SubLayerHrdParameters, H265TimingInfo, H265VideoParameterSet, H265VideoSignal,
	H265VpsHrdParameters, H265VuiParameters, parse_h265_pps, parse_h265_slice_header, parse_h265_sps,
	parse_h265_vps,
};
pub use mux::{EncodedVideoPacket, VideoMuxer, VideoMuxerAudioConfig, VideoMuxerConfig};
pub use nal::{
	NalUnit, emit_nal_annex_b, extract_pps, extract_pps_h265, extract_sps, extract_sps_h265,
	extract_vps_h265, parse_nal_annex_b,
};
pub use player::{VideoPlayer, VideoPlayerConfig, VideoPlayerStats};
pub use texture::from_texture;

pub(crate) use h264::{H264DpbState, H264PicturePlan};
