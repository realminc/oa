// Shared vulkan record helpers for oa::VideoDecoder codec bodies.

#pragma once

#include "../videoDecoderImpl.h"

#include <oa/vision/videoDecoder.h>

namespace oa {

struct VideoDecoderRecordAccess {
	struct ActiveCmd {
		VkCommandBuffer cb = VK_NULL_HANDLE;
		VkFence fence = VK_NULL_HANDLE;
		const VklDeviceTable* deviceDispatch = nullptr;
	};

	static oa::Result<ActiveCmd> begin(oa::VideoDecoder& inDecoder, const char* inLabel);
	static void releaseSlot(oa::VideoDecoder& inDecoder);

	[[nodiscard]] static bool getDpbView(
		oa::VideoDecoder& inDecoder,
		oa::I32 inSlot,
		VkImageView& outView,
		oa::U32& outBaseLayer);

	static oa::Status resolveOutputView(
		oa::VideoDecoder& inDecoder,
		oa::I32 inDpbSlot,
		VkImageView inSetupDpbView,
		VkImageView& outDstView,
		bool& outHasDistinctOutput);

	static void transitionDecodeImage(
		const ActiveCmd& inCmd,
		VkImage inImage,
		VkImageLayout& inOutLayout,
		VkImageLayout inNewLayout,
		oa::U32 inBaseLayer,
		oa::U32 inLayerCount = 1);

	static void ensureDpbLayer(ActiveCmd& inCmd, oa::VideoDecoder& inDecoder, oa::I32 inSlot);
	static void ensureDistinctOutput(ActiveCmd& inCmd, oa::VideoDecoder& inDecoder, oa::I32 inDpbSlot, bool inHasDistinctOutput);

	static void resetSessionIfNeeded(const ActiveCmd& inCmd, oa::VideoDecoder& inDecoder);

	static void emitBitstreamDecodeBarrier(
		const ActiveCmd& inCmd,
		VkBuffer inBuffer,
		VkDeviceSize inOffset,
		VkDeviceSize inSize);

	struct FinishParams {
		oa::I32 dpbSlot = -1;
		bool hasDistinctOutput = false;
		bool markSlotDeviceActivated = true;
		const char* errorContext = "video decode";
	};

	static oa::Status finishAndSubmit(oa::VideoDecoder& inDecoder, const ActiveCmd& inCmd, FinishParams inParams);
};

} // namespace oa
