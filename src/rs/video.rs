//! Video-frame values, codecs, and stateful stream sessions.
//!
//! The first admitted value path wraps one packed [`crate::Image`] while
//! preserving frame timing and source color metadata. Native multi-plane video
//! images and stateful codec sessions remain private future extensions until
//! their ownership and capability contracts ship with executable paths.

mod bitstream;
mod capability;
mod demux;
mod frame;
mod h264;
mod h265;
mod nal;

pub use capability::{
	Av1Profile, H264PictureLayout, H264Profile, H265Profile, VideoChromaSubsampling,
	VideoComponentBitDepth, VideoDecodeCapabilities, VideoDecodeFormats, VideoDecodeLevel,
	VideoDecodeProfile, VideoDeviceCapabilities, VideoExtent, VideoImageFormat, VideoPixelFormat,
	query_decode_capabilities, query_decode_formats, query_device_capabilities,
};
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
	H265LongTermReferencePicture, H265PictureParameterSet, H265ProfileTierLevel, H265ScalingLists,
	H265SequenceParameterSet, H265ShortTermReferencePictureSet, H265SliceHeader, H265SliceType,
	H265VideoParameterSet, parse_h265_pps, parse_h265_slice_header, parse_h265_sps, parse_h265_vps,
};
pub use nal::{
	NalUnit, emit_nal_annex_b, extract_pps, extract_pps_h265, extract_sps, extract_sps_h265,
	extract_vps_h265, parse_nal_annex_b,
};
