//! Private Vulkan Video integration for the Device and Runtime adapters.

mod decode;
mod encode;
mod image;
mod profile;

pub(crate) use decode::NativeDecodedFrame;
pub(in crate::runtime::device) use decode::create_decode_session;
pub(in crate::runtime) use decode::{
	DecodeSession, DecodeStatus, DecodedPlaneCopy, NativeCanvasTransfer, NativePresentation,
	PlaneCopyPhase, PreparedDecode, RetainedDecode,
};
pub(in crate::runtime) use encode::{
	EncodePlaneCopy, EncodeReceipt, EncodeSession, PreparedEncode, RetainedEncode,
	create_encode_session,
};
pub(in crate::runtime) use profile::{
	query_decode_capabilities, query_decode_formats, query_encode_capabilities, query_encode_formats,
};

const MAX_VIDEO_SESSION_MEMORY_BINDINGS: u32 = 64;
