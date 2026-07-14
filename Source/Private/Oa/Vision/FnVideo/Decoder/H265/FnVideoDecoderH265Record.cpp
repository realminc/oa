// H.265 vkCmdDecodeVideoKHR record path (FnVideoDecoder* impl).

#include "FnVideoDecoderH265.h"
#include "../FnVideoDecoderRecordShared.h"
#include "../../../Video/Codec/VcpH265.h"

OaStatus OaVideoDecoder::RecordH265DecodeCommands(
	OaI32 InDpbSlot,
	const OaH265PictureDesc& InDesc,
	const OaVec<OaI32>& InRefPicList0,
	const OaVec<OaI32>& InRefPicList1)
{
	const OaH265SliceHeader& InSliceHeader = InDesc.SliceHeader;
	const OaSpan<const OaU32> InSliceOffsets(InDesc.SliceOffsets.Data(), InDesc.SliceOffsets.Size());
	OaVideoDecoder::BitstreamSlot& bitstream = BitstreamRing_[CurrentBitstreamIndex_];
	if (!Rt_ || Session_.Handle() == VK_NULL_HANDLE || SessionParams_.Handle() == VK_NULL_HANDLE || !CmdBuffers_[0] || bitstream.Buffer.GetBuffer() == VK_NULL_HANDLE) {
		return OaStatus::Error("H.265 decoder command resources are not initialized");
	}
	if (!vkCmdBeginVideoCodingKHR || !vkCmdDecodeVideoKHR || !vkCmdEndVideoCodingKHR) {
		return OaStatus::Error("Vulkan Video decode command functions are not loaded");
	}
	if (InDpbSlot < 0 || static_cast<OaU32>(InDpbSlot) >= DpbSlotCapacity_) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Invalid H.265 DPB slot");
	}
	if (InSliceOffsets.Empty()) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "H.265 decode requires at least one slice segment");
	}
	for (OaU32 offset : InSliceOffsets) {
		if (offset >= bitstream.Size || offset >= bitstream.Buffer.GetCapacity()) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Invalid H.265 slice offset");
		}
	}

	VkImageView setupDpbView = VK_NULL_HANDLE;
	OaU32 setupDpbBaseLayer = 0;
	if (!OaFnVideoDecoderRecord::GetDpbView(*this, InDpbSlot, setupDpbView, setupDpbBaseLayer)) {
		return OaStatus::Error(OaStatusCode::Unavailable, "H.265 decode requires profile-compatible DPB image views");
	}

	VkImageView dstView = VK_NULL_HANDLE;
	bool hasDistinctOutput = false;
	OA_RETURN_IF_ERROR(OaFnVideoDecoderRecord::ResolveOutputView(*this, InDpbSlot, setupDpbView, dstView, hasDistinctOutput));

	OaFnVideoDecoderRecord::ActiveCmd cmd;
	OA_ASSIGN_OR_RETURN(cmd, OaFnVideoDecoderRecord::Begin(*this, "H.265 decode"));

	StdVideoDecodeH265PictureInfo stdPicture = {};
	stdPicture.flags.IrapPicFlag = InSliceHeader.IsIrap;
	stdPicture.flags.IdrPicFlag = InSliceHeader.IsIdr;
	stdPicture.flags.IsReference = InSliceHeader.IsReference;
	stdPicture.flags.short_term_ref_pic_set_sps_flag = InSliceHeader.ShortTermRefPicSetSpsFlag;
	stdPicture.sps_video_parameter_set_id = static_cast<uint8_t>(InSliceHeader.VpsId);
	stdPicture.pps_seq_parameter_set_id = static_cast<uint8_t>(InSliceHeader.SpsId);
	stdPicture.pps_pic_parameter_set_id = static_cast<uint8_t>(InSliceHeader.PpsId);
	stdPicture.PicOrderCntVal = InSliceHeader.PicOrderCntVal;
	stdPicture.NumBitsForSTRefPicSetInSlice = InSliceHeader.NumBitsForSTRefPicSetInSlice;
	for (OaUsize i = 0; i < STD_VIDEO_DECODE_H265_REF_PIC_SET_LIST_SIZE; ++i) {
		stdPicture.RefPicSetStCurrBefore[i] = 0xffu;
		stdPicture.RefPicSetStCurrAfter[i] = 0xffu;
		stdPicture.RefPicSetLtCurr[i] = 0xffu;
	}

	VkVideoDecodeH265PictureInfoKHR h265Picture = {};
	h265Picture.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_PICTURE_INFO_KHR;
	h265Picture.pStdPictureInfo = &stdPicture;
	h265Picture.sliceSegmentCount = static_cast<uint32_t>(InSliceOffsets.Size());
	h265Picture.pSliceSegmentOffsets = InSliceOffsets.Data();

	StdVideoDecodeH265ReferenceInfo setupStdRef = {};
	setupStdRef.PicOrderCntVal = InSliceHeader.PicOrderCntVal;

	VkVideoDecodeH265DpbSlotInfoKHR setupH265Slot = {};
	setupH265Slot.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_DPB_SLOT_INFO_KHR;
	setupH265Slot.pStdReferenceInfo = &setupStdRef;

	VkVideoPictureResourceInfoKHR setupResource = {};
	setupResource.sType = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR;
	setupResource.codedExtent = {CodedWidth_, CodedHeight_};
	setupResource.baseArrayLayer = setupDpbBaseLayer;
	setupResource.imageViewBinding = setupDpbView;

	VkVideoReferenceSlotInfoKHR setupSlot = {};
	setupSlot.sType = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR;
	setupSlot.pNext = &setupH265Slot;
	setupSlot.slotIndex = InDpbSlot;
	setupSlot.pPictureResource = &setupResource;

	OaArray<StdVideoDecodeH265ReferenceInfo, 16> stdRefs = {};
	OaArray<VkVideoDecodeH265DpbSlotInfoKHR, 16> h265Slots = {};
	OaArray<VkVideoPictureResourceInfoKHR, 16> refResources = {};
	OaArray<VkVideoReferenceSlotInfoKHR, 16> refSlots = {};
	OaU32 refCount = 0;
	auto addRefSlot = [&](OaI32 slot) {
		if (slot < 0 || static_cast<OaU32>(slot) >= DpbSlotCapacity_ || refCount >= refSlots.Size()) {
			return;
		}
		VkImageView refView = VK_NULL_HANDLE;
		OaU32 refBaseLayer = 0;
		if (!OaFnVideoDecoderRecord::GetDpbView(*this, slot, refView, refBaseLayer)) {
			return;
		}
		for (OaU32 i = 0; i < refCount; ++i) {
			if (refSlots[i].slotIndex == slot) {
				return;
			}
		}
		stdRefs[refCount].PicOrderCntVal = DpbSlots_[slot].PicOrderCnt;
		h265Slots[refCount].sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_DPB_SLOT_INFO_KHR;
		h265Slots[refCount].pStdReferenceInfo = &stdRefs[refCount];
		refResources[refCount].sType = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR;
		refResources[refCount].codedExtent = {CodedWidth_, CodedHeight_};
		refResources[refCount].baseArrayLayer = refBaseLayer;
		refResources[refCount].imageViewBinding = refView;
		refSlots[refCount].sType = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR;
		refSlots[refCount].pNext = &h265Slots[refCount];
		refSlots[refCount].slotIndex = slot;
		refSlots[refCount].pPictureResource = &refResources[refCount];
		++refCount;
	};
	for (OaI32 slot = 0; slot < static_cast<OaI32>(DpbSlotCapacity_); ++slot) {
		if (DpbSlots_[slot].InUse && DpbSlots_[slot].IsReference) {
			addRefSlot(slot);
		}
	}
	for (OaUsize i = 0; i < InRefPicList0.Size() && i < STD_VIDEO_DECODE_H265_REF_PIC_SET_LIST_SIZE; ++i) {
		stdPicture.RefPicSetStCurrBefore[i] = static_cast<uint8_t>(InRefPicList0[i]);
	}
	for (OaUsize i = 0; i < InRefPicList1.Size() && i < STD_VIDEO_DECODE_H265_REF_PIC_SET_LIST_SIZE; ++i) {
		stdPicture.RefPicSetStCurrAfter[i] = static_cast<uint8_t>(InRefPicList1[i]);
	}

	VkVideoPictureResourceInfoKHR dstResource = {};
	dstResource.sType = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR;
	dstResource.codedExtent = {CodedWidth_, CodedHeight_};
	dstResource.baseArrayLayer = hasDistinctOutput ? 0u : setupDpbBaseLayer;
	dstResource.imageViewBinding = dstView;

	if (!DpbSlots_[InDpbSlot].InUse) {
		DpbSlots_[InDpbSlot].InUse = true;
		DpbSlots_[InDpbSlot].FrameNumber = CurrentFrameNumber_;
		DpbSlots_[InDpbSlot].PicOrderCnt = InSliceHeader.PicOrderCntVal;
	}

	OaArray<VkVideoReferenceSlotInfoKHR, 17> beginRefSlots = {};
	for (OaU32 i = 0; i < refCount; ++i) {
		beginRefSlots[i] = refSlots[i];
	}
	beginRefSlots[refCount] = setupSlot;
	// Bind the current reconstruction picture as an inactive reference until
	// the decode completes. The real destination slot remains in
	// decodeInfo.pSetupReferenceSlot.
	beginRefSlots[refCount].slotIndex = -1;

	VkVideoBeginCodingInfoKHR beginCoding = {};
	beginCoding.sType = VK_STRUCTURE_TYPE_VIDEO_BEGIN_CODING_INFO_KHR;
	beginCoding.videoSession = Session_.Handle();
	beginCoding.videoSessionParameters = SessionParams_.Handle();
	beginCoding.referenceSlotCount = refCount + 1;
	beginCoding.pReferenceSlots = beginRefSlots.Data();
	vkCmdBeginVideoCodingKHR(cmd.Cb, &beginCoding);

	OaFnVideoDecoderRecord::ResetSessionIfNeeded(cmd, *this);

	OaFnVideoDecoderRecord::EnsureDpbLayer(cmd, *this, InDpbSlot);
	for (OaI32 slot = 0; slot < static_cast<OaI32>(DpbSlotCapacity_); ++slot) {
		if (DpbSlots_[slot].InUse && DpbSlots_[slot].IsReference) {
			OaFnVideoDecoderRecord::EnsureDpbLayer(cmd, *this, slot);
		}
	}
	OaFnVideoDecoderRecord::EnsureDistinctOutput(cmd, *this, InDpbSlot, hasDistinctOutput);

	VkVideoDecodeInfoKHR decodeInfo = {};
	decodeInfo.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_INFO_KHR;
	decodeInfo.pNext = &h265Picture;
	decodeInfo.srcBuffer = bitstream.Buffer.GetBuffer();
	decodeInfo.srcBufferOffset = 0;
	decodeInfo.srcBufferRange = static_cast<VkDeviceSize>(OaAlignUp(
		static_cast<OaUsize>(bitstream.Size),
		static_cast<OaUsize>(bitstream.Buffer.GetSizeAlignment() == 0 ? 1 : bitstream.Buffer.GetSizeAlignment())));
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
		.ErrorContext = "H.265 video decode",
	});
}
