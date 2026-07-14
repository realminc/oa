// Shared helpers for FnVideoDecoder* decode impl (friend of OaVideoDecoder).

#pragma once

#include <Oa/Vision/VideoDecoder.h>

struct OaFnVideoDecoderAccess {
	static void FillNv12OutFrame(
		OaVideoDecoder& InDecoder,
		OaI32 InDpbSlot,
		OaU32 InWidth,
		OaU32 InHeight,
		OaU64 InPts,
		OaVideoFrame& OutFrame);

	static void ResetAllDpbSlotStates(OaVideoDecoder& InDecoder);
};