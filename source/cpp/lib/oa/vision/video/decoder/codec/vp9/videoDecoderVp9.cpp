// VP9 access-unit decode path for oa::VideoDecoder.

#include "../videoDecoderCodecAccess.h"
#include "../../../codec/vcpVp9.h"

oa::Status oa::VideoDecoderCodecAccess::decodeVp9(
	oa::VideoDecoder& inDecoder,
	const oa::Span<const oa::U8>& inBitstream,
	oa::VideoFrame& outFrame)
{
	auto* parser = static_cast<oa::VcpVp9*>(inDecoder.impl_->parser.get());
	if (!parser) {
		return oa::Status::error("VP9 parser not registered");
	}

	oa::Vp9PictureDesc desc;
	OA_RETURN_IF_ERROR(parser->parseAccessUnit(inBitstream, desc));
	if (desc.showExistingFrame) {
		oa::I32 logical = desc.frameToShowMapIdx;
		oa::I32 dpbSlot = (logical >= 0 && static_cast<oa::U32>(logical) < STD_VIDEO_VP9_NUM_REF_FRAMES)
			? inDecoder.impl_->vp9BufferToDpbSlot[static_cast<oa::Usize>(logical)] : -1;
		if (dpbSlot < 0) {
			return oa::Status::error(oa::StatusCode::Unavailable, "VP9 show_existing_frame references an invalid/unavailable DPB slot");
		}
		fillDecodedOutFrame(
			inDecoder,
			dpbSlot,
			desc.frameWidth > 0 ? desc.frameWidth : inDecoder.impl_->profile.width,
			desc.frameHeight > 0 ? desc.frameHeight : inDecoder.impl_->profile.height,
			desc.frame.timestamp,
			outFrame);
		outFrame.shown = true;  // show_existing_frame re-displays a stored slot
		return oa::Status::ok();
	}
	if (!desc.hasPicture) {
		return oa::Status::ok();
	}
	if (desc.frame.size == 0
		|| desc.frame.offset > inBitstream.size()
		|| desc.frame.size > inBitstream.size() - desc.frame.offset) {
		return oa::Status::error(
			oa::StatusCode::InvalidArgument,
			"VP9 frame payload is outside the input access unit");
	}
	OA_RETURN_IF_ERROR(inDecoder.uploadBitstream(oa::Span<const oa::U8>(
		inBitstream.data() + desc.frame.offset,
		desc.frame.size)));

	const bool isKeyFrame = desc.stdPictureInfo.frame_type == STD_VIDEO_VP9_FRAME_TYPE_KEY;
	if (isKeyFrame) {
		resetAllDpbSlotStates(inDecoder);
		for (oa::I32& vp9Slot : inDecoder.impl_->vp9BufferToDpbSlot) {
			vp9Slot = -1;
		}
		inDecoder.impl_->vp9BufferExtents.fill({0, 0});
	}

	auto releaseUnmappedSlots = [&inDecoder]() {
		for (oa::I32 dpbIndex = 0; dpbIndex < 16; ++dpbIndex) {
			bool isMapped = false;
			for (oa::I32 mapped : inDecoder.impl_->vp9BufferToDpbSlot) {
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

	// VP9 reference lifetime is defined by its eight logical reference buffers.
	// Age-based eviction can discard a physical slot still named by that map.
	releaseUnmappedSlots();
	oa::I32 dpbSlot = inDecoder.allocateDpbSlot();
	if (dpbSlot < 0) {
		return oa::Status::error("DPB overflow - all VP9 slots are reference frames");
	}

	for (oa::I32& mapped : inDecoder.impl_->vp9BufferToDpbSlot) {
		if (mapped == dpbSlot) {
			mapped = -1;
		}
	}

	oa::I32 refNameSlotIndices[STD_VIDEO_VP9_REFS_PER_FRAME] = {-1, -1, -1};
	oa::Vector<oa::I32> refSlots;
	oa::Vector<VkExtent2D> refExtents;
	if (!isKeyFrame) {
		for (oa::U32 i = 0; i < STD_VIDEO_VP9_REFS_PER_FRAME; ++i) {
			const oa::U8 bufferIdx = desc.refFrameIdx[i];
			const oa::I32 mappedSlot = inDecoder.impl_->vp9BufferToDpbSlot[bufferIdx];
			if (mappedSlot == dpbSlot) {
				refNameSlotIndices[i] = -1;
				continue;
			}
			refNameSlotIndices[i] = mappedSlot;
			if (mappedSlot < 0) {
				continue;
			}
			bool alreadyAdded = false;
			for (oa::Usize j = 0; j < refSlots.size(); ++j) {
				if (refSlots[j] == mappedSlot) {
					alreadyAdded = true;
					break;
				}
			}
			if (!alreadyAdded) {
				refSlots.pushBack(mappedSlot);
				refExtents.pushBack(inDecoder.impl_->vp9BufferExtents[bufferIdx]);
			}
		}
	}

	OA_RETURN_IF_ERROR(inDecoder.recordVP9DecodeCommands(
		dpbSlot,
		desc,
		refNameSlotIndices,
		refSlots,
		refExtents));
	// CPU reference-map bookkeeping describes the submission order; it does
	// not read decode output. FinishAndSubmit chains this job after the prior
	// decoder timeline value, and fillDecodedOutFrame publishes the new value as
	// oa::VideoFrame::Ready for downstream GPU or host consumers.

	for (oa::U32 mask = desc.stdPictureInfo.refresh_frame_flags, refIndex = 0u; mask != 0u; mask >>= 1u, ++refIndex) {
		if ((mask & 1u) != 0u) {
			inDecoder.impl_->vp9BufferToDpbSlot[refIndex] = dpbSlot;
			inDecoder.impl_->vp9BufferExtents[refIndex] = {desc.frameWidth, desc.frameHeight};
		}
	}

	const bool keepAsReference = (desc.stdPictureInfo.refresh_frame_flags != 0u) || isKeyFrame;
	if (keepAsReference) {
		inDecoder.markSlotAsReference(dpbSlot, static_cast<oa::I32>(inDecoder.impl_->currentFrameNumber));
	} else {
		inDecoder.releaseDpbSlot(dpbSlot);
	}
	releaseUnmappedSlots();
	inDecoder.impl_->currentFrameNumber++;

	fillDecodedOutFrame(
		inDecoder,
		dpbSlot,
		desc.frameWidth > 0 ? desc.frameWidth : inDecoder.impl_->profile.width,
		desc.frameHeight > 0 ? desc.frameHeight : inDecoder.impl_->profile.height,
		desc.frame.timestamp,
		outFrame);
	// Hidden frames (show_frame=0) decode into the DPB but are not displayed;
	// the reorder layer drops !shown frames.
	outFrame.shown = desc.stdPictureInfo.flags.show_frame != 0u;
	return oa::Status::ok();
}
