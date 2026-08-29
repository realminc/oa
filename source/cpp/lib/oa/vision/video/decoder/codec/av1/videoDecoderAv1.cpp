// AV1 access-unit decode path for oa::VideoDecoder.

#include "../videoDecoderCodecAccess.h"
#include "../../../codec/vcpAv1.h"

oa::Status oa::VideoDecoderCodecAccess::decodeAv1Picture(
	oa::VideoDecoder& inDecoder,
	const oa::Span<const oa::U8>& inBitstream,
	const oa::Av1PictureDesc& desc,
	oa::VideoFrame& outFrame)
{
	if (desc.hasPicture || desc.showExistingFrame) {
		if (desc.frame.size == 0 || desc.frame.offset + desc.frame.size > inBitstream.size()) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "Invalid AV1 frame payload range");
		}
		if (desc.frameHeaderOffset > desc.frame.size) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "Invalid AV1 frame header offset");
		}
		// upload the decode OBU slice (OBU header + payload). frameHeaderOffset is the
		// OBU header size; tile offsets are rebased into this buffer.
		const oa::Usize obuOff = desc.decodeObuOffset;
		const oa::Usize obuSize = desc.decodeObuSize;
		if (obuSize == 0 || obuOff + obuSize > desc.frame.size) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "Invalid AV1 decode OBU range");
		}
		OA_RETURN_IF_ERROR(inDecoder.uploadBitstream(oa::Span<const oa::U8>(
			inBitstream.data() + desc.frame.offset + obuOff, obuSize)));
	}

	OA_RETURN_IF_ERROR(inDecoder.updateAv1SessionParametersFromSequenceHeader(desc.sequenceHeader));

	if (desc.showExistingFrame) {
		oa::I32 logical = desc.frameToShowMapIdx;
		oa::I32 dpbSlot = (logical >= 0 && static_cast<oa::U32>(logical) < STD_VIDEO_AV1_NUM_REF_FRAMES)
			? inDecoder.impl_->av1RefFrameToDpbSlot[static_cast<oa::Usize>(logical)] : -1;
		if (dpbSlot < 0) {
			return oa::Status::error(oa::StatusCode::Unavailable, "AV1 show_existing_frame references an invalid/unavailable DPB slot");
		}
		fillDecodedOutFrame(
			inDecoder,
			dpbSlot,
			inDecoder.impl_->profile.width,
			inDecoder.impl_->profile.height,
			desc.frame.timestamp,
			outFrame);
		outFrame.shown = true;  // show_existing_frame re-displays a stored slot
		return oa::Status::ok();
	}

	if (!desc.hasPicture) {
		return oa::Status::ok();
	}

	// The current Vulkan AV1 path has no qualified 10-bit loop-restoration
	// implementation.  On Intel TGL/Mesa 26.1.7, submitting an otherwise valid
	// Main-10 picture with UsesLr set stalls the video queue without a validation
	// VUID.  Reject the syntax before mutating DPB state or submitting GPU work;
	// this gate stays cross-vendor until the std-video metadata has an independent
	// oracle and the path is qualified on more than one implementation.
	if (inDecoder.impl_->profile.lumaBitDepth == oa::VideoBitDepth::Bit10
		&& desc.frameHeader.usesLr)
	{
		return oa::Status::error(
			oa::StatusCode::Unavailable,
			"AV1 Main 10-bit loop restoration is not qualified");
	}

	const bool isKeyFrame = (desc.frameHeader.frameType == STD_VIDEO_AV1_FRAME_TYPE_KEY);
	if (isKeyFrame) {
		resetAllDpbSlotStates(inDecoder);
		for (oa::I32& av1Slot : inDecoder.impl_->av1RefFrameToDpbSlot) {
			av1Slot = -1;
		}
		inDecoder.impl_->av1DpbReferenceInfos.fill({});
		inDecoder.impl_->slotDeviceActivated.fill(false);
	}

	auto releaseUnmappedSlots = [&inDecoder]() {
		for (oa::I32 dpbIndex = 0; dpbIndex < 16; ++dpbIndex) {
			bool isMapped = false;
			for (oa::I32 mapped : inDecoder.impl_->av1RefFrameToDpbSlot) {
				if (mapped == dpbIndex) {
					isMapped = true;
					break;
				}
			}
			if (!isMapped && inDecoder.impl_->dpbSlots[static_cast<oa::Usize>(dpbIndex)].inUse) {
				inDecoder.releaseDpbSlot(dpbIndex);
			}
		}
	};

	// AV1 reference lifetime is defined by the eight logical reference frame
	// slots, not by display age. Releasing a still-mapped physical DPB slot
	// leaves future inter frames pointing at overwritten contents.
	releaseUnmappedSlots();
	oa::I32 dpbSlot = inDecoder.allocateDpbSlot();
	if (dpbSlot < 0) {
		return oa::Status::error("DPB overflow - all 16 slots are reference frames");
	}

	for (oa::I32& mapped : inDecoder.impl_->av1RefFrameToDpbSlot) {
		if (mapped == dpbSlot) {
			mapped = -1;
		}
	}

	oa::I32 refNameSlotIndices[oa::Av1MaxReferencesPerFrame] = {-1, -1, -1, -1, -1, -1, -1};
	for (oa::U32 i = 0; i < oa::Av1MaxReferencesPerFrame; ++i) {
		const oa::I32 logical = desc.frameHeader.referenceNameSlotIndices[i];
		if (logical >= 0 && static_cast<oa::U32>(logical) < STD_VIDEO_AV1_NUM_REF_FRAMES) {
			refNameSlotIndices[i] = inDecoder.impl_->av1RefFrameToDpbSlot[static_cast<oa::Usize>(logical)];
		}
	}

	OA_RETURN_IF_ERROR(inDecoder.recordAV1DecodeCommands(dpbSlot, desc, refNameSlotIndices));
	// CPU reference-map bookkeeping describes the submission order; it does
	// not read decode output. FinishAndSubmit chains this job after the prior
	// decoder timeline value, and fillDecodedOutFrame publishes the new value as
	// oa::VideoFrame::Ready for downstream GPU or host consumers.

	for (oa::U32 mask = desc.frameHeader.refreshFrameFlags, refIndex = 0u;
	     mask != 0u; mask >>= 1u, ++refIndex) {
		if ((mask & 1u) != 0u && refIndex < STD_VIDEO_AV1_NUM_REF_FRAMES) {
			inDecoder.impl_->av1RefFrameToDpbSlot[refIndex] = dpbSlot;
		}
	}

	const bool keepAsReference = (desc.frameHeader.refreshFrameFlags != 0u) || isKeyFrame;
	if (keepAsReference) {
		inDecoder.markSlotAsReference(dpbSlot, static_cast<oa::I32>(desc.frameHeader.orderHint));
	} else {
		inDecoder.releaseDpbSlot(dpbSlot);
	}
	releaseUnmappedSlots();
	inDecoder.impl_->currentFrameNumber++;

	fillDecodedOutFrame(
		inDecoder,
		dpbSlot,
		inDecoder.impl_->profile.width,
		inDecoder.impl_->profile.height,
		desc.frame.timestamp,
		outFrame);
	// Hidden alt-ref frames (show_frame=0) decode into the DPB but must not be
	// presented; the reorder layer drops !shown frames. show_existing_frame is
	// handled above and is always shown.
	outFrame.shown = desc.frameHeader.showFrame;
	return oa::Status::ok();
}

oa::Status oa::VideoDecoderCodecAccess::decodeAv1(
	oa::VideoDecoder& inDecoder,
	const oa::Span<const oa::U8>& inBitstream,
	oa::VideoFrame& outFrame)
{
	outFrame = {};
	auto* parser = static_cast<oa::VcpAv1*>(inDecoder.impl_->parser.get());
	if (!parser) return oa::Status::error("AV1 parser not registered");

	oa::Vector<oa::Av1PictureDesc> pictures;
	OA_RETURN_IF_ERROR(parser->parseAccessUnitPictures(inBitstream, pictures));
	oa::VideoFrame lastDecoded = {};
	for (const oa::Av1PictureDesc& desc : pictures) {
		oa::VideoFrame decoded = {};
		OA_RETURN_IF_ERROR(decodeAv1Picture(inDecoder, inBitstream, desc, decoded));
		if (decoded.imageView != VK_NULL_HANDLE) lastDecoded = decoded;
		if (decoded.imageView != VK_NULL_HANDLE && decoded.shown) outFrame = decoded;
	}
	if (outFrame.imageView == VK_NULL_HANDLE) outFrame = lastDecoded;
	return oa::Status::ok();
}
