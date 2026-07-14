// VP9 vkCmdDecodeVideoKHR record path (FnVideoDecoder* impl).

#include "FnVideoDecoderVp9.h"
#include "../FnVideoDecoderRecordShared.h"
#include "../../../Video/Codec/VcpVp9.h"

OaStatus OaVideoDecoder::RecordVP9DecodeCommands(
	OaI32 InDpbSlot,
	const OaVp9PictureDesc& InDesc,
	const OaI32 InReferenceNameSlotIndices[STD_VIDEO_VP9_REFS_PER_FRAME],
	const OaVec<OaI32>& InReferenceSlots,
	const OaVec<VkExtent2D>& InReferenceExtents)
{
	OaVideoDecoder::BitstreamSlot& bitstream = BitstreamRing_[CurrentBitstreamIndex_];
	if (!Rt_ || Session_.Handle() == VK_NULL_HANDLE || !CmdBuffers_[0] || bitstream.Buffer.GetBuffer() == VK_NULL_HANDLE) {
		return OaStatus::Error("VP9 decoder command resources are not initialized");
	}
	if (!vkCmdBeginVideoCodingKHR || !vkCmdDecodeVideoKHR || !vkCmdEndVideoCodingKHR) {
		return OaStatus::Error("Vulkan Video decode command functions are not loaded");
	}
	if (InDpbSlot < 0 || static_cast<OaU32>(InDpbSlot) >= DpbSlotCapacity_) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Invalid VP9 DPB slot");
	}
	if (InDesc.Frame.Size == 0) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Invalid VP9 frame range");
	}
	if (InReferenceSlots.Size() != InReferenceExtents.Size()) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "VP9 reference slot/extent mismatch");
	}

	const OaU32 frameWidth = InDesc.FrameWidth > 0 ? InDesc.FrameWidth : Profile_.Width;
	const OaU32 frameHeight = InDesc.FrameHeight > 0 ? InDesc.FrameHeight : Profile_.Height;
	const VkExtent2D frameExtent = {frameWidth, frameHeight};

	VkImageView setupDpbView = VK_NULL_HANDLE;
	OaU32 setupDpbBaseLayer = 0;
	if (!OaFnVideoDecoderRecord::GetDpbView(*this, InDpbSlot, setupDpbView, setupDpbBaseLayer)) {
		return OaStatus::Error(OaStatusCode::Unavailable, "VP9 decode requires profile-compatible DPB image views");
	}

	VkImageView dstView = VK_NULL_HANDLE;
	bool hasDistinctOutput = false;
	OA_RETURN_IF_ERROR(OaFnVideoDecoderRecord::ResolveOutputView(*this, InDpbSlot, setupDpbView, dstView, hasDistinctOutput));

	OaFnVideoDecoderRecord::ActiveCmd cmd;
	OA_ASSIGN_OR_RETURN(cmd, OaFnVideoDecoderRecord::Begin(*this, "VP9 decode"));

	StdVideoDecodeVP9PictureInfo stdPicture = InDesc.StdPictureInfo;
	stdPicture.pColorConfig = &InDesc.ColorConfig;
	stdPicture.pLoopFilter = &InDesc.LoopFilter;
	stdPicture.pSegmentation = &InDesc.Segmentation;

	VkVideoDecodeVP9PictureInfoKHR vp9Picture = {};
	vp9Picture.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_VP9_PICTURE_INFO_KHR;
	vp9Picture.pStdPictureInfo = &stdPicture;
	for (OaU32 i = 0; i < STD_VIDEO_VP9_REFS_PER_FRAME; ++i) {
		vp9Picture.referenceNameSlotIndices[i] = InReferenceNameSlotIndices[i];
	}
	vp9Picture.uncompressedHeaderOffset = InDesc.UncompressedHeaderOffset;
	vp9Picture.compressedHeaderOffset = InDesc.CompressedHeaderOffset;
	vp9Picture.tilesOffset = InDesc.TilesOffset;

	OaArray<VkVideoPictureResourceInfoKHR, STD_VIDEO_VP9_NUM_REF_FRAMES> refResources = {};
	OaArray<VkVideoReferenceSlotInfoKHR, STD_VIDEO_VP9_NUM_REF_FRAMES> refSlots = {};
	OaU32 refCount = 0;
	auto refAlreadyBound = [&](OaI32 s) -> bool {
		for (OaU32 j = 0; j < refCount; ++j) {
			if (refSlots[j].slotIndex == s) { return true; }
		}
		return false;
	};
	for (OaUsize i = 0; i < InReferenceSlots.Size(); ++i) {
		const OaI32 refSlot = InReferenceSlots[i];
		VkImageView refView = VK_NULL_HANDLE;
		OaU32 refBaseLayer = 0;
		if (!OaFnVideoDecoderRecord::GetDpbView(*this, refSlot, refView, refBaseLayer)) {
			continue;
		}
		refResources[refCount].sType = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR;
		refResources[refCount].codedExtent = InReferenceExtents[i];
		refResources[refCount].baseArrayLayer = refBaseLayer;
		refResources[refCount].imageViewBinding = refView;
		refSlots[refCount].sType = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR;
		refSlots[refCount].slotIndex = refSlot;
		refSlots[refCount].pPictureResource = &refResources[refCount];
		++refCount;
	}
	for (OaU32 i = 0; i < STD_VIDEO_VP9_REFS_PER_FRAME; ++i) {
		if (vp9Picture.referenceNameSlotIndices[i] >= 0
			&& !refAlreadyBound(vp9Picture.referenceNameSlotIndices[i])) {
			vp9Picture.referenceNameSlotIndices[i] = -1;
		}
	}

	VkVideoPictureResourceInfoKHR setupResource = {};
	setupResource.sType = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR;
	setupResource.codedExtent = frameExtent;
	setupResource.baseArrayLayer = setupDpbBaseLayer;
	setupResource.imageViewBinding = setupDpbView;

	VkVideoReferenceSlotInfoKHR setupSlot = {};
	setupSlot.sType = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR;
	setupSlot.slotIndex = InDpbSlot;
	setupSlot.pPictureResource = &setupResource;

	VkVideoPictureResourceInfoKHR dstResource = {};
	dstResource.sType = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR;
	dstResource.codedExtent = frameExtent;
	dstResource.baseArrayLayer = hasDistinctOutput ? 0u : setupDpbBaseLayer;
	dstResource.imageViewBinding = dstView;

	if (!DpbSlots_[InDpbSlot].InUse) {
		DpbSlots_[InDpbSlot].InUse = true;
		DpbSlots_[InDpbSlot].FrameNumber = CurrentFrameNumber_;
		DpbSlots_[InDpbSlot].PicOrderCnt = static_cast<OaI32>(CurrentFrameNumber_);
	}

	OaArray<VkVideoReferenceSlotInfoKHR, STD_VIDEO_VP9_NUM_REF_FRAMES + 1> beginRefSlots = {};
	for (OaU32 i = 0; i < refCount; ++i) {
		beginRefSlots[i] = refSlots[i];
	}
	beginRefSlots[refCount] = setupSlot;
	// The setup picture is part of the begin-coding reference set with its real
	// DPB association. This matches the Vulkan Video reference decoder; -1 is
	// the invalidation sentinel, not a first-use activation mechanism.
	beginRefSlots[refCount].slotIndex = InDpbSlot;

	VkVideoBeginCodingInfoKHR beginCoding = {};
	beginCoding.sType = VK_STRUCTURE_TYPE_VIDEO_BEGIN_CODING_INFO_KHR;
	beginCoding.videoSession = Session_.Handle();
	beginCoding.videoSessionParameters = VK_NULL_HANDLE;
	beginCoding.referenceSlotCount = refCount + 1;
	beginCoding.pReferenceSlots = beginRefSlots.Data();
	vkCmdBeginVideoCodingKHR(cmd.Cb, &beginCoding);

	OaFnVideoDecoderRecord::ResetSessionIfNeeded(cmd, *this);

	// Match the other decode paths and NVIDIA's known-good ordering:
	// BeginCoding -> optional reset -> picture-resource barriers -> decode.
	OaFnVideoDecoderRecord::EnsureDpbLayer(cmd, *this, InDpbSlot);
	for (OaI32 refSlot : InReferenceSlots) {
		OaFnVideoDecoderRecord::EnsureDpbLayer(cmd, *this, refSlot);
	}
	OaFnVideoDecoderRecord::EnsureDistinctOutput(cmd, *this, InDpbSlot, hasDistinctOutput);

	const OaU64 sizeAlignment = bitstream.Buffer.GetSizeAlignment() == 0 ? 1 : bitstream.Buffer.GetSizeAlignment();
	const VkDeviceSize srcRange = static_cast<VkDeviceSize>(
		OaAlignUp(InDesc.Frame.Size, static_cast<OaUsize>(sizeAlignment)));

	VkVideoDecodeInfoKHR decodeInfo = {};
	decodeInfo.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_INFO_KHR;
	decodeInfo.pNext = &vp9Picture;
	decodeInfo.srcBuffer = bitstream.Buffer.GetBuffer();
	decodeInfo.srcBufferOffset = 0;
	decodeInfo.srcBufferRange = srcRange;
	decodeInfo.dstPictureResource = dstResource;
	decodeInfo.pSetupReferenceSlot = &setupSlot;
	decodeInfo.referenceSlotCount = refCount;
	decodeInfo.pReferenceSlots = refCount > 0 ? refSlots.Data() : nullptr;

	OaFnVideoDecoderRecord::EmitBitstreamDecodeBarrier(
		cmd,
		decodeInfo.srcBuffer,
		decodeInfo.srcBufferOffset,
		decodeInfo.srcBufferRange);

	vkCmdDecodeVideoKHR(cmd.Cb, &decodeInfo);

	VkVideoEndCodingInfoKHR endCoding = {};
	endCoding.sType = VK_STRUCTURE_TYPE_VIDEO_END_CODING_INFO_KHR;
	vkCmdEndVideoCodingKHR(cmd.Cb, &endCoding);

	return OaFnVideoDecoderRecord::FinishAndSubmit(*this, cmd, {
		.DpbSlot = InDpbSlot,
		.HasDistinctOutput = hasDistinctOutput,
		.ErrorContext = "VP9 video decode",
	});
}
