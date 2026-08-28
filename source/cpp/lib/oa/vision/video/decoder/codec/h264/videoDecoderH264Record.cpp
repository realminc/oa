// H.264 vkCmdDecodeVideoKHR record path for oa::VideoDecoder.

#include "../videoDecoderRecordAccess.h"
#include "../../../codec/vcpH264.h"
#include <oa/runtime/engine/deviceAccess.h>

oa::Status oa::VideoDecoder::recordH264DecodeCommands(
	oa::I32 inDpbSlot,
	const oa::H264PictureDesc& inDesc,
	const oa::Vector<oa::I32>& inRefPicList0,
	const oa::Vector<oa::I32>& inRefPicList1)
{
	const oa::H264SliceHeader& inSliceHeader = inDesc.sliceHeader;
	const oa::Usize inNalOffset = inDesc.sliceStartCodeOffset;
	const oa::Usize inNalSize = static_cast<oa::Usize>(inDesc.sliceStartCodeSize) + inDesc.sliceNalSize;
	oa::VideoDecoder::BitstreamSlot& bitstream = impl_->bitstreamRing[impl_->currentBitstreamIndex];
	if (!impl_->engine || impl_->session.handle() == VK_NULL_HANDLE || impl_->sessionParameters.handle() == VK_NULL_HANDLE || !impl_->commandBuffers[0] || bitstream.buffer.getBuffer() == VK_NULL_HANDLE) {
		return oa::Status::error("H.264 decoder command resources are not initialized");
	}
	const auto& deviceDispatch =
		oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch;
	if (!deviceDispatch.vkCmdBeginVideoCodingKHR
		|| !deviceDispatch.vkCmdDecodeVideoKHR
		|| !deviceDispatch.vkCmdEndVideoCodingKHR) {
		return oa::Status::error("vulkan Video decode command functions are not loaded");
	}
	if (inDpbSlot < 0 || static_cast<oa::U32>(inDpbSlot) >= impl_->dpbSlotCapacity) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "Invalid H.264 DPB slot");
	}
	if (inNalSize == 0 || inNalOffset + inNalSize > bitstream.size) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "Invalid H.264 slice range");
	}
	if (inNalOffset >= bitstream.buffer.getCapacity()) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "H.264 slice offset exceeds bitstream buffer capacity");
	}

	VkImageView setupDpbView = VK_NULL_HANDLE;
	oa::U32 setupDpbBaseLayer = 0;
	if (!oa::VideoDecoderRecordAccess::getDpbView(*this, inDpbSlot, setupDpbView, setupDpbBaseLayer)) {
		return oa::Status::error(oa::StatusCode::Unavailable, "H.264 decode requires profile-compatible DPB image views");
	}

	VkImageView dstView = VK_NULL_HANDLE;
	bool hasDistinctOutput = false;
	OA_RETURN_IF_ERROR(oa::VideoDecoderRecordAccess::resolveOutputView(*this, inDpbSlot, setupDpbView, dstView, hasDistinctOutput));

	oa::VideoDecoderRecordAccess::ActiveCmd cmd;
	OA_ASSIGN_OR_RETURN(cmd, oa::VideoDecoderRecordAccess::begin(*this, "H.264 decode"));

	StdVideoDecodeH264PictureInfo stdPicture = {};
	stdPicture.flags.IdrPicFlag = inSliceHeader.isIdrPic;
	stdPicture.flags.is_intra = inSliceHeader.sliceType == oa::H264SliceType::I ||
		inSliceHeader.sliceType == oa::H264SliceType::SI;
	stdPicture.flags.is_reference = inSliceHeader.isReference;
	stdPicture.flags.field_pic_flag = inSliceHeader.fieldPicFlag;
	stdPicture.flags.bottom_field_flag = inSliceHeader.bottomFieldFlag;
	stdPicture.pic_parameter_set_id = static_cast<uint8_t>(inSliceHeader.ppsId);
	const oa::H264PpsData* pps = getPps(inSliceHeader.ppsId);
	if (pps) {
		stdPicture.seq_parameter_set_id = static_cast<uint8_t>(pps->spsId);
	}
	stdPicture.frame_num = static_cast<uint16_t>(inSliceHeader.frameNum);
	stdPicture.idr_pic_id = static_cast<uint16_t>(inSliceHeader.idrPicId);
	stdPicture.PicOrderCnt[0] = inSliceHeader.picOrderCntLsb;
	stdPicture.PicOrderCnt[1] = inSliceHeader.picOrderCntLsb;

	VkVideoDecodeH264PictureInfoKHR h264Picture = {};
	h264Picture.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_PICTURE_INFO_KHR;
	h264Picture.pStdPictureInfo = &stdPicture;
	// vulkan Video H.264 slice offsets point at the Annex-B NAL start marker.
	// This matches the Khronos parser stream markers and the known-good path.
	const uint32_t sliceOffset = static_cast<uint32_t>(inNalOffset);
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
	setupResource.codedExtent = {impl_->codedWidth, impl_->codedHeight};
	setupResource.baseArrayLayer = setupDpbBaseLayer;
	setupResource.imageViewBinding = setupDpbView;

	VkVideoReferenceSlotInfoKHR setupSlot = {};
	setupSlot.sType = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR;
	setupSlot.pNext = &setupH264Slot;
	setupSlot.slotIndex = inDpbSlot;
	setupSlot.pPictureResource = &setupResource;

	oa::Array<StdVideoDecodeH264ReferenceInfo, 16> stdRefs = {};
	oa::Array<VkVideoDecodeH264DpbSlotInfoKHR, 16> h264Slots = {};
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
		stdRefs[refCount].FrameNum = static_cast<uint16_t>(impl_->dpbSlots[slot].h264FrameNum);
		stdRefs[refCount].PicOrderCnt[0] = impl_->dpbSlots[slot].picOrderCnt;
		stdRefs[refCount].PicOrderCnt[1] = impl_->dpbSlots[slot].picOrderCnt;
		stdRefs[refCount].flags.used_for_long_term_reference = impl_->dpbSlots[slot].isLongTerm ? 1u : 0u;
		stdRefs[refCount].flags.top_field_flag = 0;
		stdRefs[refCount].flags.bottom_field_flag = 0;
		stdRefs[refCount].flags.is_non_existing = 0;
		h264Slots[refCount].sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_DPB_SLOT_INFO_KHR;
		h264Slots[refCount].pStdReferenceInfo = &stdRefs[refCount];
		refResources[refCount].sType = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR;
		refResources[refCount].codedExtent = {impl_->codedWidth, impl_->codedHeight};
		refResources[refCount].baseArrayLayer = refBaseLayer;
		refResources[refCount].imageViewBinding = refView;
		refSlots[refCount].sType = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR;
		refSlots[refCount].pNext = &h264Slots[refCount];
		refSlots[refCount].slotIndex = slot;
		refSlots[refCount].pPictureResource = &refResources[refCount];
		++refCount;
	};
	for (oa::I32 slot : inRefPicList0) {
		addRefSlot(slot);
	}
	for (oa::I32 slot : inRefPicList1) {
		addRefSlot(slot);
	}
	// The bitstream may modify the default refPicList0/1 ordering. Bind every
	// active DPB association so the implementation can resolve those indices.
	for (oa::U32 slot = 0; slot < impl_->dpbSlotCapacity; ++slot) {
		if (impl_->dpbSlots[slot].inUse && impl_->dpbSlots[slot].isReference) {
			addRefSlot(static_cast<oa::I32>(slot));
		}
	}

	VkVideoPictureResourceInfoKHR dstResource = {};
	dstResource.sType = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR;
	dstResource.codedExtent = {impl_->codedWidth, impl_->codedHeight};
	dstResource.baseArrayLayer = hasDistinctOutput ? 0u : setupDpbBaseLayer;
	dstResource.imageViewBinding = dstView;

	if (!impl_->dpbSlots[inDpbSlot].inUse) {
		impl_->dpbSlots[inDpbSlot].inUse = true;
		impl_->dpbSlots[inDpbSlot].frameNumber = impl_->currentFrameNumber;
		impl_->dpbSlots[inDpbSlot].picOrderCnt = inSliceHeader.picOrderCntLsb;
	}
	impl_->dpbSlots[inDpbSlot].h264FrameNum = inSliceHeader.frameNum;
	impl_->dpbSlots[inDpbSlot].picOrderCnt  = inSliceHeader.picOrderCntLsb;

	oa::Array<VkVideoReferenceSlotInfoKHR, 17> beginRefSlots = {};
	for (oa::U32 i = 0; i < refCount; ++i) {
		beginRefSlots[i] = refSlots[i];
	}
	beginRefSlots[refCount] = setupSlot;
	beginRefSlots[refCount].slotIndex =
		impl_->slotDeviceActivated[inDpbSlot] ? inDpbSlot : -1;

	VkVideoBeginCodingInfoKHR beginCoding = {};
	beginCoding.sType = VK_STRUCTURE_TYPE_VIDEO_BEGIN_CODING_INFO_KHR;
	beginCoding.videoSession = impl_->session.handle();
	beginCoding.videoSessionParameters = impl_->sessionParameters.handle();
	beginCoding.referenceSlotCount = refCount + 1;
	beginCoding.pReferenceSlots = beginRefSlots.data();
	deviceDispatch.vkCmdBeginVideoCodingKHR(cmd.cb, &beginCoding);

	oa::VideoDecoderRecordAccess::resetSessionIfNeeded(cmd, *this);

	// Match the vulkan Video sample ordering: establish the coding scope,
	// reset a new session, then prepare picture resources for decode.
	oa::VideoDecoderRecordAccess::ensureDpbLayer(cmd, *this, inDpbSlot);
	for (oa::I32 refSlot : inRefPicList0) {
		oa::VideoDecoderRecordAccess::ensureDpbLayer(cmd, *this, refSlot);
	}
	for (oa::I32 refSlot : inRefPicList1) {
		oa::VideoDecoderRecordAccess::ensureDpbLayer(cmd, *this, refSlot);
	}
	oa::VideoDecoderRecordAccess::ensureDistinctOutput(cmd, *this, inDpbSlot, hasDistinctOutput);

	VkVideoDecodeInfoKHR decodeInfo = {};
	decodeInfo.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_INFO_KHR;
	decodeInfo.pNext = &h264Picture;
	decodeInfo.srcBuffer = bitstream.buffer.getBuffer();
	decodeInfo.srcBufferOffset = 0;
	decodeInfo.srcBufferRange = static_cast<VkDeviceSize>(oa::alignUp(
		bitstream.size,
		static_cast<oa::Usize>(
			bitstream.buffer.getSizeAlignment() == 0
				? 1
				: bitstream.buffer.getSizeAlignment())));
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
		.markSlotDeviceActivated = inSliceHeader.isReference,
		.errorContext = "H.264 video decode",
	});
}
