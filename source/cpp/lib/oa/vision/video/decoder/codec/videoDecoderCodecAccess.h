#pragma once

#include "../videoDecoderImpl.h"

#include <oa/vision/videoDecoder.h>

namespace oa {

// Private codec-dispatch authority for the stateful decoder session. codec
// parsing, DPB mutation, command recording, and submission never form an
// Fn* operation surface.
struct VideoDecoderCodecAccess {
	[[nodiscard]] static oa::Status decodeH264(
		oa::VideoDecoder& inDecoder,
		const oa::Span<const oa::U8>& inBitstream,
		oa::VideoFrame& outFrame);
	[[nodiscard]] static oa::Status decodeH265(
		oa::VideoDecoder& inDecoder,
		const oa::Span<const oa::U8>& inBitstream,
		oa::VideoFrame& outFrame);
	[[nodiscard]] static oa::Status decodeAv1(
		oa::VideoDecoder& inDecoder,
		const oa::Span<const oa::U8>& inBitstream,
		oa::VideoFrame& outFrame);
	[[nodiscard]] static oa::Status decodeAv1Picture(
		oa::VideoDecoder& inDecoder,
		const oa::Span<const oa::U8>& inBitstream,
		const Av1PictureDesc& inDesc,
		oa::VideoFrame& outFrame);
	[[nodiscard]] static oa::Status decodeVp9(
		oa::VideoDecoder& inDecoder,
		const oa::Span<const oa::U8>& inBitstream,
		oa::VideoFrame& outFrame);

	static void fillDecodedOutFrame(
		oa::VideoDecoder& inDecoder,
		oa::I32 inDpbSlot,
		oa::U32 inWidth,
		oa::U32 inHeight,
		oa::U64 inPts,
		oa::VideoFrame& outFrame);
	static void resetAllDpbSlotStates(oa::VideoDecoder& inDecoder);
};

} // namespace oa
