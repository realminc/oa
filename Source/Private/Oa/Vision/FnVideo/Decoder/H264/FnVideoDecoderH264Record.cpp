// H.264 vkCmdDecodeVideoKHR record path (FnVideoDecoder* impl).

#include "FnVideoDecoderH264.h"
#include "../FnVideoDecoderRecordShared.h"
#include "../../../Video/Codec/VcpH264.h"

OaStatus OaVideoDecoder::RecordH264DecodeCommands(
	OaI32 InDpbSlot,
	const OaH264PictureDesc& InDesc,
	const OaVec<OaI32>& InRefPicList0,
	const OaVec<OaI32>& InRefPicList1)
{
	const OaSliceHeader& InSliceHeader = InDesc.SliceHeader;
	const OaUsize InNalOffset = InDesc.SliceStartCodeOffset;
	const OaUsize InNalSize = static_cast<OaUsize>(InDesc.SliceStartCodeSize) + InDesc.SliceNalSize;
	OaVideoDecoder::BitstreamSlot& bitstream = BitstreamRing_[CurrentBitstreamIndex_];
	if (!Rt_ || Session_.Handle() == VK_NULL_HANDLE || SessionParams_.Handle() == VK_NULL_HANDLE || !CmdBuffers_[0] || bitstream.Buffer.GetBuffer() == VK_NULL_HANDLE) {
		return OaStatus::Error("H.264 decoder command resources are not initialized");
	}
	if (!vkCmdBeginVideoCodingKHR || !vkCmdDecodeVideoKHR || !vkCmdEndVideoCodingKHR) {
		return OaStatus::Error("Vulkan Video decode command functions are not loaded");
	}
	if (InDpbSlot < 0 || static_cast<OaU32>(InDpbSlot) >= DpbSlotCapacity_) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Invalid H.264 DPB slot");
	}
	if (InNalSize == 0 || InNalOffset + InNalSize > bitstream.Size) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Invalid H.264 slice range");
	}
	if (InNalOffset >= bitstream.Buffer.GetCapacity()) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "H.264 slice offset exceeds bitstream buffer capacity");
	}

	VkImageView setupDpbView = VK_NULL_HANDLE;
	OaU32 setupDpbBaseLayer = 0;
	if (!OaFnVideoDecoderRecord::GetDpbView(*this, InDpbSlot, setupDpbView, setupDpbBaseLayer)) {
		return OaStatus::Error(OaStatusCode::Unavailable, "H.264 decode requires profile-compatible DPB image views");
	}

	VkImageView dstView = VK_NULL_HANDLE;
	bool hasDistinctOutput = false;
	OA_RETURN_IF_ERROR(OaFnVideoDecoderRecord::ResolveOutputView(*this, InDpbSlot, setupDpbView, dstView, hasDistinctOutput));

	OaFnVideoDecoderRecord::ActiveCmd cmd;
	OA_ASSIGN_OR_RETURN(cmd, OaFnVideoDecoderRecord::Begin(*this, "H.264 decode"));

	StdVideoDecodeH264PictureInfo stdPicture = {};
	stdPicture.flags.IdrPicFlag = InSliceHeader.IsIdrPic;
	stdPicture.flags.is_intra = InSliceHeader.SliceType == OaH264SliceType::I ||
		InSliceHeader.SliceType == OaH264SliceType::SI;
	stdPicture.flags.is_reference = InSliceHeader.IsReference;
	stdPicture.flags.field_pic_flag = InSliceHeader.FieldPicFlag;
	stdPicture.flags.bottom_field_flag = InSliceHeader.BottomFieldFlag;
	stdPicture.pic_parameter_set_id = static_cast<uint8_t>(InSliceHeader.PpsId);
	const OaH264PpsData* pps = GetPps(InSliceHeader.PpsId);
	if (pps) {
		stdPicture.seq_parameter_set_id = static_cast<uint8_t>(pps->SpsId);
	}
	stdPicture.frame_num = static_cast<uint16_t>(InSliceHeader.FrameNum);
	stdPicture.idr_pic_id = static_cast<uint16_t>(InSliceHeader.IdrPicId);
	stdPicture.PicOrderCnt[0] = InSliceHeader.PicOrderCntLsb;
	stdPicture.PicOrderCnt[1] = InSliceHeader.PicOrderCntLsb;

	VkVideoDecodeH264PictureInfoKHR h264Picture = {};
	h264Picture.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_PICTURE_INFO_KHR;
	h264Picture.pStdPictureInfo = &stdPicture;
	// Vulkan Video H.264 slice offsets point at the Annex-B NAL start marker.
	// This matches the Khronos parser stream markers and the known-good path.
	const uint32_t sliceOffset = static_cast<uint32_t>(InNalOffset);
	h264Picture.sliceCount = 1;
	h264Picture.pSliceOffsets = &sliceOffset;

	StdVideoDecodeH264ReferenceInfo setupStdRef = {};
	setupStdRef.FrameNum = stdPicture.frame_num;
	setupStdRef.PicOrderCnt[0] = stdPicture.PicOrderCnt[0];
	setupStdRef.PicOrderCnt[1] = stdPicture.PicOrderCnt[1];
	setupStdRef.flags.top_field_flag = 0;
	setupStdRef.flags.bottom_field_flag = 0;
	setupStdRef.flags.used_for_long_term_reference = 0;
	setupStdRef.flags.is_non_existing = 0;

	VkVideoDecodeH264DpbSlotInfoKHR setupH264Slot = {};
	setupH264Slot.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_DPB_SLOT_INFO_KHR;
	setupH264Slot.pStdReferenceInfo = &setupStdRef;

	VkVideoPictureResourceInfoKHR setupResource = {};
	setupResource.sType = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR;
	setupResource.codedExtent = {CodedWidth_, CodedHeight_};
	setupResource.baseArrayLayer = setupDpbBaseLayer;
	setupResource.imageViewBinding = setupDpbView;

	VkVideoReferenceSlotInfoKHR setupSlot = {};
	setupSlot.sType = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR;
	setupSlot.pNext = &setupH264Slot;
	setupSlot.slotIndex = InDpbSlot;
	setupSlot.pPictureResource = &setupResource;

	OaArray<StdVideoDecodeH264ReferenceInfo, 16> stdRefs = {};
	OaArray<VkVideoDecodeH264DpbSlotInfoKHR, 16> h264Slots = {};
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
		stdRefs[refCount].FrameNum = static_cast<uint16_t>(DpbSlots_[slot].H264FrameNum);
		stdRefs[refCount].PicOrderCnt[0] = DpbSlots_[slot].PicOrderCnt;
		stdRefs[refCount].PicOrderCnt[1] = DpbSlots_[slot].PicOrderCnt;
		stdRefs[refCount].flags.used_for_long_term_reference = DpbSlots_[slot].IsLongTerm ? 1u : 0u;
		stdRefs[refCount].flags.top_field_flag = 0;
		stdRefs[refCount].flags.bottom_field_flag = 0;
		stdRefs[refCount].flags.is_non_existing = 0;
		h264Slots[refCount].sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_DPB_SLOT_INFO_KHR;
		h264Slots[refCount].pStdReferenceInfo = &stdRefs[refCount];
		refResources[refCount].sType = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR;
		refResources[refCount].codedExtent = {CodedWidth_, CodedHeight_};
		refResources[refCount].baseArrayLayer = refBaseLayer;
		refResources[refCount].imageViewBinding = refView;
		refSlots[refCount].sType = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR;
		refSlots[refCount].pNext = &h264Slots[refCount];
		refSlots[refCount].slotIndex = slot;
		refSlots[refCount].pPictureResource = &refResources[refCount];
		++refCount;
	};
	for (OaI32 slot : InRefPicList0) {
		addRefSlot(slot);
	}
	for (OaI32 slot : InRefPicList1) {
		addRefSlot(slot);
	}
	// The bitstream may modify the default RefPicList0/1 ordering. Bind every
	// active DPB association so the implementation can resolve those indices.
	for (OaU32 slot = 0; slot < DpbSlotCapacity_; ++slot) {
		if (DpbSlots_[slot].InUse && DpbSlots_[slot].IsReference) {
			addRefSlot(static_cast<OaI32>(slot));
		}
	}

	VkVideoPictureResourceInfoKHR dstResource = {};
	dstResource.sType = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR;
	dstResource.codedExtent = {CodedWidth_, CodedHeight_};
	dstResource.baseArrayLayer = hasDistinctOutput ? 0u : setupDpbBaseLayer;
	dstResource.imageViewBinding = dstView;

	if (!DpbSlots_[InDpbSlot].InUse) {
		DpbSlots_[InDpbSlot].InUse = true;
		DpbSlots_[InDpbSlot].FrameNumber = CurrentFrameNumber_;
		DpbSlots_[InDpbSlot].PicOrderCnt = InSliceHeader.PicOrderCntLsb;
	}
	DpbSlots_[InDpbSlot].H264FrameNum = InSliceHeader.FrameNum;
	DpbSlots_[InDpbSlot].PicOrderCnt  = InSliceHeader.PicOrderCntLsb;

	OaArray<VkVideoReferenceSlotInfoKHR, 17> beginRefSlots = {};
	for (OaU32 i = 0; i < refCount; ++i) {
		beginRefSlots[i] = refSlots[i];
	}
	beginRefSlots[refCount] = setupSlot;
	beginRefSlots[refCount].slotIndex =
		SlotDeviceActivated_[InDpbSlot] ? InDpbSlot : -1;

	VkVideoBeginCodingInfoKHR beginCoding = {};
	beginCoding.sType = VK_STRUCTURE_TYPE_VIDEO_BEGIN_CODING_INFO_KHR;
	beginCoding.videoSession = Session_.Handle();
	beginCoding.videoSessionParameters = SessionParams_.Handle();
	beginCoding.referenceSlotCount = refCount + 1;
	beginCoding.pReferenceSlots = beginRefSlots.Data();
	vkCmdBeginVideoCodingKHR(cmd.Cb, &beginCoding);

	OaFnVideoDecoderRecord::ResetSessionIfNeeded(cmd, *this);

	// Match the Vulkan Video sample ordering: establish the coding scope,
	// reset a new session, then prepare picture resources for decode.
	OaFnVideoDecoderRecord::EnsureDpbLayer(cmd, *this, InDpbSlot);
	for (OaI32 refSlot : InRefPicList0) {
		OaFnVideoDecoderRecord::EnsureDpbLayer(cmd, *this, refSlot);
	}
	for (OaI32 refSlot : InRefPicList1) {
		OaFnVideoDecoderRecord::EnsureDpbLayer(cmd, *this, refSlot);
	}
	OaFnVideoDecoderRecord::EnsureDistinctOutput(cmd, *this, InDpbSlot, hasDistinctOutput);

	VkVideoDecodeInfoKHR decodeInfo = {};
	decodeInfo.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_INFO_KHR;
	decodeInfo.pNext = &h264Picture;
	decodeInfo.srcBuffer = bitstream.Buffer.GetBuffer();
	decodeInfo.srcBufferOffset = 0;
	decodeInfo.srcBufferRange = static_cast<VkDeviceSize>(OaAlignUp(
		bitstream.Size,
		static_cast<OaUsize>(
			bitstream.Buffer.GetSizeAlignment() == 0
				? 1
				: bitstream.Buffer.GetSizeAlignment())));
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
		.MarkSlotDeviceActivated = InSliceHeader.IsReference,
		.ErrorContext = "H.264 video decode",
	});
}
