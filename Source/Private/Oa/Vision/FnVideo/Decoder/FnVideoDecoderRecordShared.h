// Shared Vulkan record helpers for FnVideoDecoder* Record* bodies.

#pragma once

#include <Oa/Vision/VideoDecoder.h>

struct OaFnVideoDecoderRecord {
	struct ActiveCmd {
		VkCommandBuffer Cb = VK_NULL_HANDLE;
		VkFence Fence = VK_NULL_HANDLE;
	};

	static OaResult<ActiveCmd> Begin(OaVideoDecoder& InDecoder, const char* InLabel);
	static void ReleaseSlot(OaVideoDecoder& InDecoder);

	[[nodiscard]] static bool GetDpbView(
		OaVideoDecoder& InDecoder,
		OaI32 InSlot,
		VkImageView& OutView,
		OaU32& OutBaseLayer);

	static OaStatus ResolveOutputView(
		OaVideoDecoder& InDecoder,
		OaI32 InDpbSlot,
		VkImageView InSetupDpbView,
		VkImageView& OutDstView,
		bool& OutHasDistinctOutput);

	static void TransitionDecodeImage(
		const ActiveCmd& InCmd,
		VkImage InImage,
		VkImageLayout& InOutLayout,
		VkImageLayout InNewLayout,
		OaU32 InBaseLayer,
		OaU32 InLayerCount = 1);

	static void EnsureDpbLayer(ActiveCmd& InCmd, OaVideoDecoder& InDecoder, OaI32 InSlot);
	static void EnsureDistinctOutput(ActiveCmd& InCmd, OaVideoDecoder& InDecoder, OaI32 InDpbSlot, bool InHasDistinctOutput);

	static void ResetSessionIfNeeded(const ActiveCmd& InCmd, OaVideoDecoder& InDecoder);

	static void EmitBitstreamDecodeBarrier(
		const ActiveCmd& InCmd,
		VkBuffer InBuffer,
		VkDeviceSize InOffset,
		VkDeviceSize InSize);

	struct FinishParams {
		OaI32 DpbSlot = -1;
		bool HasDistinctOutput = false;
		bool MarkSlotDeviceActivated = true;
		const char* ErrorContext = "video decode";
	};

	static OaStatus FinishAndSubmit(OaVideoDecoder& InDecoder, const ActiveCmd& InCmd, FinishParams InParams);
};