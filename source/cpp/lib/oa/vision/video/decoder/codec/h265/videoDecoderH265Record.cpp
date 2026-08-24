// H.265 vkCmdDecodeVideoKHR record path for oa::VideoDecoder.

#include "../videoDecoderRecordAccess.h"
#include "../../../codec/vcpH265.h"
#include <oa/runtime/engine/deviceAccess.h>

oa::Status oa::VideoDecoder::recordH265DecodeCommands(
	oa::I32 inDpbSlot,
	const oa::H265PictureDesc& inDesc,
	const oa::Vec<oa::I32>& inRefPicList0,
	const oa::Vec<oa::I32>& inRefPicList1)
{
	const oa::H265SliceHeader& inSliceHeader = inDesc.sliceHeader;
	const oa::Span<const oa::U32> inSliceOffsets(inDesc.sliceOffsets.data(), inDesc.sliceOffsets.size());
	oa::VideoDecoder::BitstreamSlot& bitstream = impl_->bitstreamRing[impl_->currentBitstreamIndex];
	if (!impl_->engine || impl_->session.handle() == VK_NULL_HANDLE || impl_->sessionParameters.handle() == VK_NULL_HANDLE || !impl_->commandBuffers[0] || bitstream.buffer.getBuffer() == VK_NULL_HANDLE) {
		return oa::Status::error("H.265 decoder command resources are not initialized");
	}
	const auto& deviceDispatch =
		oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch;
	if (!deviceDispatch.vkCmdBeginVideoCodingKHR
		|| !deviceDispatch.vkCmdDecodeVideoKHR
		|| !deviceDispatch.vkCmdEndVideoCodingKHR) {
		return oa::Status::error("vulkan Video decode command functions are not loaded");
	}
	if (inDpbSlot < 0 || static_cast<oa::U32>(inDpbSlot) >= impl_->dpbSlotCapacity) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "Invalid H.265 DPB slot");
	}
	if (inSliceOffsets.empty()) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "H.265 decode requires at least one slice segment");
	}
	for (oa::U32 offset : inSliceOffsets) {
		if (offset >= bitstream.size || offset >= bitstream.buffer.getCapacity()) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "Invalid H.265 slice offset");
		}
	}

	VkImageView setupDpbView = VK_NULL_HANDLE;
	oa::U32 setupDpbBaseLayer = 0;
	if (!oa::VideoDecoderRecordAccess::getDpbView(*this, inDpbSlot, setupDpbView, setupDpbBaseLayer)) {
		return oa::Status::error(oa::StatusCode::Unavailable, "H.265 decode requires profile-compatible DPB image views");
	}

	VkImageView dstView = VK_NULL_HANDLE;
	bool hasDistinctOutput = false;
	OA_RETURN_IF_ERROR(oa::VideoDecoderRecordAccess::resolveOutputView(*this, inDpbSlot, setupDpbView, dstView, hasDistinctOutput));

	oa::VideoDecoderRecordAccess::ActiveCmd cmd;
	OA_ASSIGN_OR_RETURN(cmd, oa::VideoDecoderRecordAccess::begin(*this, "H.265 decode"));

	StdVideoDecodeH265PictureInfo stdPicture = {};
	stdPicture.flags.IrapPicFlag = inSliceHeader.isIrap;
	stdPicture.flags.IdrPicFlag = inSliceHeader.isIdr;
	stdPicture.flags.IsReference = inSliceHeader.isReference;
	stdPicture.flags.short_term_ref_pic_set_sps_flag = inSliceHeader.shortTermRefPicSetSpsFlag;
	stdPicture.sps_video_parameter_set_id = static_cast<uint8_t>(inSliceHeader.vpsId);
	stdPicture.pps_seq_parameter_set_id = static_cast<uint8_t>(inSliceHeader.spsId);
	stdPicture.pps_pic_parameter_set_id = static_cast<uint8_t>(inSliceHeader.ppsId);
	stdPicture.PicOrderCntVal = inSliceHeader.picOrderCntVal;
	stdPicture.NumBitsForSTRefPicSetInSlice = inSliceHeader.numBitsForSTRefPicSetInSlice;
	for (oa::Usize i = 0; i < STD_VIDEO_DECODE_H265_REF_PIC_SET_LIST_SIZE; ++i) {
		stdPicture.RefPicSetStCurrBefore[i] = 0xffu;
		stdPicture.RefPicSetStCurrAfter[i] = 0xffu;
		stdPicture.RefPicSetLtCurr[i] = 0xffu;
	}

	VkVideoDecodeH265PictureInfoKHR h265Picture = {};
	h265Picture.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_PICTURE_INFO_KHR;
	h265Picture.pStdPictureInfo = &stdPicture;
	h265Picture.sliceSegmentCount = static_cast<uint32_t>(inSliceOffsets.size());
	h265Picture.pSliceSegmentOffsets = inSliceOffsets.data();

	StdVideoDecodeH265ReferenceInfo setupStdRef = {};
	setupStdRef.PicOrderCntVal = inSliceHeader.picOrderCntVal;

	VkVideoDecodeH265DpbSlotInfoKHR setupH265Slot = {};
	setupH265Slot.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_DPB_SLOT_INFO_KHR;
	setupH265Slot.pStdReferenceInfo = &setupStdRef;

	VkVideoPictureResourceInfoKHR setupResource = {};
	setupResource.sType = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR;
	setupResource.codedExtent = {impl_->codedWidth, impl_->codedHeight};
	setupResource.baseArrayLayer = setupDpbBaseLayer;
	setupResource.imageViewBinding = setupDpbView;

	VkVideoReferenceSlotInfoKHR setupSlot = {};
	setupSlot.sType = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR;
	setupSlot.pNext = &setupH265Slot;
	setupSlot.slotIndex = inDpbSlot;
	setupSlot.pPictureResource = &setupResource;

	oa::Array<StdVideoDecodeH265ReferenceInfo, 16> stdRefs = {};
	oa::Array<VkVideoDecodeH265DpbSlotInfoKHR, 16> h265Slots = {};
	oa::Array<VkVideoPictureResourceInfoKHR, 16> refResources = {};
	oa::Array<VkVideoReferenceSlotInfoKHR, 16> refSlots = {};
	oa::U32 refCount = 0;
	auto addRefSlot = [&](oa::I32 slot) {
		if (slot < 0 || static_cast<oa::U32>(slot) >= impl_->dpbSlotCapacity || refCount >= refSlots.size()) {
			return;
		}
		VkImageView refView = VK_NULL_HANDLE;
		oa::U32 refBaseLayer = 0;
		if (!oa::VideoDecoderRecordAccess::getDpbView(*this, slot, refView, refBaseLayer)) {
			return;
		}
		for (oa::U32 i = 0; i < refCount; ++i) {
			if (refSlots[i].slotIndex == slot) {
				return;
			}
		}
		stdRefs[refCount].PicOrderCntVal = impl_->dpbSlots[slot].picOrderCnt;
		h265Slots[refCount].sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_DPB_SLOT_INFO_KHR;
		h265Slots[refCount].pStdReferenceInfo = &stdRefs[refCount];
		refResources[refCount].sType = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR;
		refResources[refCount].codedExtent = {impl_->codedWidth, impl_->codedHeight};
		refResources[refCount].baseArrayLayer = refBaseLayer;
		refResources[refCount].imageViewBinding = refView;
		refSlots[refCount].sType = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR;
		refSlots[refCount].pNext = &h265Slots[refCount];
		refSlots[refCount].slotIndex = slot;
		refSlots[refCount].pPictureResource = &refResources[refCount];
		++refCount;
	};
	for (oa::I32 slot = 0; slot < static_cast<oa::I32>(impl_->dpbSlotCapacity); ++slot) {
		if (impl_->dpbSlots[slot].inUse && impl_->dpbSlots[slot].isReference) {
			addRefSlot(slot);
		}
	}
	for (oa::Usize i = 0; i < inRefPicList0.size() && i < STD_VIDEO_DECODE_H265_REF_PIC_SET_LIST_SIZE; ++i) {
		stdPicture.RefPicSetStCurrBefore[i] = static_cast<uint8_t>(inRefPicList0[i]);
	}
	for (oa::Usize i = 0; i < inRefPicList1.size() && i < STD_VIDEO_DECODE_H265_REF_PIC_SET_LIST_SIZE; ++i) {
		stdPicture.RefPicSetStCurrAfter[i] = static_cast<uint8_t>(inRefPicList1[i]);
	}

	VkVideoPictureResourceInfoKHR dstResource = {};
	dstResource.sType = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR;
	dstResource.codedExtent = {impl_->codedWidth, impl_->codedHeight};
	dstResource.baseArrayLayer = hasDistinctOutput ? 0u : setupDpbBaseLayer;
	dstResource.imageViewBinding = dstView;

	if (!impl_->dpbSlots[inDpbSlot].inUse) {
		impl_->dpbSlots[inDpbSlot].inUse = true;
		impl_->dpbSlots[inDpbSlot].frameNumber = impl_->currentFrameNumber;
		impl_->dpbSlots[inDpbSlot].picOrderCnt = inSliceHeader.picOrderCntVal;
	}

	oa::Array<VkVideoReferenceSlotInfoKHR, 17> beginRefSlots = {};
	for (oa::U32 i = 0; i < refCount; ++i) {
		beginRefSlots[i] = refSlots[i];
	}
	beginRefSlots[refCount] = setupSlot;
	// Bind the current reconstruction picture as an inactive reference until
	// the decode completes. The real destination slot remains in
	// decodeInfo.pSetupReferenceSlot.
	beginRefSlots[refCount].slotIndex = -1;

	VkVideoBeginCodingInfoKHR beginCoding = {};
	beginCoding.sType = VK_STRUCTURE_TYPE_VIDEO_BEGIN_CODING_INFO_KHR;
	beginCoding.videoSession = impl_->session.handle();
	beginCoding.videoSessionParameters = impl_->sessionParameters.handle();
	beginCoding.referenceSlotCount = refCount + 1;
	beginCoding.pReferenceSlots = beginRefSlots.data();
	deviceDispatch.vkCmdBeginVideoCodingKHR(cmd.cb, &beginCoding);

	oa::VideoDecoderRecordAccess::resetSessionIfNeeded(cmd, *this);

	oa::VideoDecoderRecordAccess::ensureDpbLayer(cmd, *this, inDpbSlot);
	for (oa::I32 slot = 0; slot < static_cast<oa::I32>(impl_->dpbSlotCapacity); ++slot) {
		if (impl_->dpbSlots[slot].inUse && impl_->dpbSlots[slot].isReference) {
			oa::VideoDecoderRecordAccess::ensureDpbLayer(cmd, *this, slot);
		}
	}
	oa::VideoDecoderRecordAccess::ensureDistinctOutput(cmd, *this, inDpbSlot, hasDistinctOutput);

	VkVideoDecodeInfoKHR decodeInfo = {};
	decodeInfo.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_INFO_KHR;
	decodeInfo.pNext = &h265Picture;
	decodeInfo.srcBuffer = bitstream.buffer.getBuffer();
	decodeInfo.srcBufferOffset = 0;
	decodeInfo.srcBufferRange = static_cast<VkDeviceSize>(oa::alignUp(
		static_cast<oa::Usize>(bitstream.size),
		static_cast<oa::Usize>(bitstream.buffer.getSizeAlignment() == 0 ? 1 : bitstream.buffer.getSizeAlignment())));
	decodeInfo.dstPictureResource = dstResource;
	decodeInfo.pSetupReferenceSlot = &setupSlot;
	decodeInfo.referenceSlotCount = refCount;
	decodeInfo.pReferenceSlots = refCount > 0 ? refSlots.data() : nullptr;

	oa::VideoDecoderRecordAccess::emitBitstreamDecodeBarrier(
		cmd,
		decodeInfo.srcBuffer,
		decodeInfo.srcBufferOffset,
		decodeInfo.srcBufferRange);

	deviceDispatch.vkCmdDecodeVideoKHR(cmd.cb, &decodeInfo);

	VkVideoEndCodingInfoKHR endCoding = {};
	endCoding.sType = VK_STRUCTURE_TYPE_VIDEO_END_CODING_INFO_KHR;
	deviceDispatch.vkCmdEndVideoCodingKHR(cmd.cb, &endCoding);

	return oa::VideoDecoderRecordAccess::finishAndSubmit(*this, cmd, {
		.dpbSlot = inDpbSlot,
		.hasDistinctOutput = hasDistinctOutput,
		.errorContext = "H.265 video decode",
	});
}
