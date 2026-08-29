// VP9 vkCmdDecodeVideoKHR record path for oa::VideoDecoder.

#include "../videoDecoderRecordAccess.h"
#include "../../../codec/vcpVp9.h"
#include <oa/runtime/engine/deviceAccess.h>

oa::Status oa::VideoDecoder::recordVP9DecodeCommands(
	oa::I32 inDpbSlot,
	const oa::Vp9PictureDesc& inDesc,
	const oa::I32 inReferenceNameSlotIndices[STD_VIDEO_VP9_REFS_PER_FRAME],
	const oa::Vector<oa::I32>& inReferenceSlots,
	const oa::Vector<VkExtent2D>& inReferenceExtents)
{
	oa::VideoDecoder::BitstreamSlot& bitstream = impl_->bitstreamRing[impl_->currentBitstreamIndex];
	if (!impl_->engine || impl_->session.handle() == VK_NULL_HANDLE || !impl_->commandBuffers[0] || bitstream.buffer.getBuffer() == VK_NULL_HANDLE) {
		return oa::Status::error("VP9 decoder command resources are not initialized");
	}
	const auto& deviceDispatch =
		oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch;
	if (!deviceDispatch.vkCmdBeginVideoCodingKHR
		|| !deviceDispatch.vkCmdDecodeVideoKHR
		|| !deviceDispatch.vkCmdEndVideoCodingKHR) {
		return oa::Status::error("vulkan Video decode command functions are not loaded");
	}
	if (inDpbSlot < 0 || static_cast<oa::U32>(inDpbSlot) >= impl_->dpbSlotCapacity) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "Invalid VP9 DPB slot");
	}
	if (inDesc.frame.size == 0) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "Invalid VP9 frame range");
	}
	if (inReferenceSlots.size() != inReferenceExtents.size()) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "VP9 reference slot/extent mismatch");
	}

	const oa::U32 frameWidth = inDesc.frameWidth > 0 ? inDesc.frameWidth : impl_->profile.width;
	const oa::U32 frameHeight = inDesc.frameHeight > 0 ? inDesc.frameHeight : impl_->profile.height;
	const VkExtent2D frameExtent = {frameWidth, frameHeight};

	VkImageView setupDpbView = VK_NULL_HANDLE;
	oa::U32 setupDpbBaseLayer = 0;
	if (!oa::VideoDecoderRecordAccess::getDpbView(*this, inDpbSlot, setupDpbView, setupDpbBaseLayer)) {
		return oa::Status::error(oa::StatusCode::Unavailable, "VP9 decode requires profile-compatible DPB image views");
	}

	VkImageView dstView = VK_NULL_HANDLE;
	bool hasDistinctOutput = false;
	OA_RETURN_IF_ERROR(oa::VideoDecoderRecordAccess::resolveOutputView(*this, inDpbSlot, setupDpbView, dstView, hasDistinctOutput));

	oa::VideoDecoderRecordAccess::ActiveCmd cmd;
	OA_ASSIGN_OR_RETURN(cmd, oa::VideoDecoderRecordAccess::begin(*this, "VP9 decode"));

	StdVideoDecodeVP9PictureInfo stdPicture = inDesc.stdPictureInfo;
	stdPicture.pColorConfig = &inDesc.colorConfig;
	stdPicture.pLoopFilter = &inDesc.loopFilter;
	stdPicture.pSegmentation = &inDesc.segmentation;

	VkVideoDecodeVP9PictureInfoKHR vp9Picture = {};
	vp9Picture.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_VP9_PICTURE_INFO_KHR;
	vp9Picture.pStdPictureInfo = &stdPicture;
	for (oa::U32 i = 0; i < STD_VIDEO_VP9_REFS_PER_FRAME; ++i) {
		vp9Picture.referenceNameSlotIndices[i] = inReferenceNameSlotIndices[i];
	}
	vp9Picture.uncompressedHeaderOffset = inDesc.uncompressedHeaderOffset;
	vp9Picture.compressedHeaderOffset = inDesc.compressedHeaderOffset;
	vp9Picture.tilesOffset = inDesc.tilesOffset;

	oa::Array<VkVideoPictureResourceInfoKHR, STD_VIDEO_VP9_NUM_REF_FRAMES> refResources = {};
	oa::Array<VkVideoReferenceSlotInfoKHR, STD_VIDEO_VP9_NUM_REF_FRAMES> refSlots = {};
	oa::U32 refCount = 0;
	auto refAlreadyBound = [&](oa::I32 s) -> bool {
		for (oa::U32 j = 0; j < refCount; ++j) {
			if (refSlots[j].slotIndex == s) { return true; }
		}
		return false;
	};
	for (oa::Usize i = 0; i < inReferenceSlots.size(); ++i) {
		const oa::I32 refSlot = inReferenceSlots[i];
		VkImageView refView = VK_NULL_HANDLE;
		oa::U32 refBaseLayer = 0;
		if (!oa::VideoDecoderRecordAccess::getDpbView(*this, refSlot, refView, refBaseLayer)) {
			continue;
		}
		refResources[refCount].sType = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR;
		refResources[refCount].codedExtent = inReferenceExtents[i];
		refResources[refCount].baseArrayLayer = refBaseLayer;
		refResources[refCount].imageViewBinding = refView;
		refSlots[refCount].sType = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR;
		refSlots[refCount].slotIndex = refSlot;
		refSlots[refCount].pPictureResource = &refResources[refCount];
		++refCount;
	}
	for (oa::U32 i = 0; i < STD_VIDEO_VP9_REFS_PER_FRAME; ++i) {
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
	setupSlot.slotIndex = inDpbSlot;
	setupSlot.pPictureResource = &setupResource;

	VkVideoPictureResourceInfoKHR dstResource = {};
	dstResource.sType = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR;
	dstResource.codedExtent = frameExtent;
	dstResource.baseArrayLayer = hasDistinctOutput ? 0u : setupDpbBaseLayer;
	dstResource.imageViewBinding = dstView;

	if (!impl_->dpbSlots[inDpbSlot].inUse) {
		impl_->dpbSlots[inDpbSlot].inUse = true;
		impl_->dpbSlots[inDpbSlot].frameNumber = impl_->currentFrameNumber;
		impl_->dpbSlots[inDpbSlot].picOrderCnt = static_cast<oa::I32>(impl_->currentFrameNumber);
	}

	oa::Array<VkVideoReferenceSlotInfoKHR, STD_VIDEO_VP9_NUM_REF_FRAMES + 1> beginRefSlots = {};
	for (oa::U32 i = 0; i < refCount; ++i) {
		beginRefSlots[i] = refSlots[i];
	}
	beginRefSlots[refCount] = setupSlot;
	// The current reconstruction image participates in layout/resource tracking,
	// but its DPB association is not active until this decode completes. Binding
	// a newly allocated slot here with its real index violates VUID 07239; the
	// decode command activates it through pSetupReferenceSlot below.
	beginRefSlots[refCount].slotIndex = -1;

	VkVideoBeginCodingInfoKHR beginCoding = {};
	beginCoding.sType = VK_STRUCTURE_TYPE_VIDEO_BEGIN_CODING_INFO_KHR;
	beginCoding.videoSession = impl_->session.handle();
	beginCoding.videoSessionParameters = VK_NULL_HANDLE;
	beginCoding.referenceSlotCount = refCount + 1;
	beginCoding.pReferenceSlots = beginRefSlots.data();
	deviceDispatch.vkCmdBeginVideoCodingKHR(cmd.cb, &beginCoding);

	oa::VideoDecoderRecordAccess::resetSessionIfNeeded(cmd, *this);

	// Match the other decode paths and NVIDIA's known-good ordering:
	// BeginCoding -> optional reset -> picture-resource barriers -> decode.
	oa::VideoDecoderRecordAccess::ensureDpbLayer(cmd, *this, inDpbSlot);
	for (oa::I32 refSlot : inReferenceSlots) {
		oa::VideoDecoderRecordAccess::ensureDpbLayer(cmd, *this, refSlot);
	}
	oa::VideoDecoderRecordAccess::ensureDistinctOutput(cmd, *this, inDpbSlot, hasDistinctOutput);

	const oa::U64 sizeAlignment = bitstream.buffer.getSizeAlignment() == 0 ? 1 : bitstream.buffer.getSizeAlignment();
	const VkDeviceSize srcRange = static_cast<VkDeviceSize>(
		oa::alignUp(inDesc.frame.size, static_cast<oa::Usize>(sizeAlignment)));

	VkVideoDecodeInfoKHR decodeInfo = {};
	decodeInfo.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_INFO_KHR;
	decodeInfo.pNext = &vp9Picture;
	decodeInfo.srcBuffer = bitstream.buffer.getBuffer();
	decodeInfo.srcBufferOffset = 0;
	decodeInfo.srcBufferRange = srcRange;
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
		.errorContext = "VP9 video decode",
	});
}
