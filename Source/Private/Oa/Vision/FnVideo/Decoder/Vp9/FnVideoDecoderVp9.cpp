// VP9 access-unit decode path (FnVideoDecoder* impl).

#include "FnVideoDecoderVp9.h"
#include "../FnVideoDecoderShared.h"
#include "../../../Video/Codec/VcpVp9.h"

namespace FnVideoDecoderVp9 {

OaStatus DecodeFrame(
	OaVideoDecoder& InDecoder,
	const OaSpan<const OaU8>& InBitstream,
	OaVideoFrame& OutFrame)
{
	auto* parser = static_cast<OaVcpVp9*>(InDecoder.Parser_.Get());
	if (!parser) {
		return OaStatus::Error("VP9 parser not registered");
	}

	OaVp9PictureDesc desc;
	OA_RETURN_IF_ERROR(parser->ParseAccessUnit(InBitstream, desc));
	if (desc.ShowExistingFrame) {
		OaI32 logical = desc.FrameToShowMapIdx;
		OaI32 dpbSlot = (logical >= 0 && static_cast<OaU32>(logical) < STD_VIDEO_VP9_NUM_REF_FRAMES)
			? InDecoder.Vp9BufferToDpbSlot_[static_cast<OaUsize>(logical)] : -1;
		if (dpbSlot < 0) {
			return OaStatus::Error(OaStatusCode::Unavailable, "VP9 show_existing_frame references an invalid/unavailable DPB slot");
		}
		OaFnVideoDecoderAccess::FillNv12OutFrame(
			InDecoder,
			dpbSlot,
			desc.FrameWidth > 0 ? desc.FrameWidth : InDecoder.Profile_.Width,
			desc.FrameHeight > 0 ? desc.FrameHeight : InDecoder.Profile_.Height,
			desc.Frame.Timestamp,
			OutFrame);
		OutFrame.Shown = true;  // show_existing_frame re-displays a stored slot
		return OaStatus::Ok();
	}
	if (!desc.HasPicture) {
		return OaStatus::Ok();
	}
	if (desc.Frame.Size == 0
		|| desc.Frame.Offset > InBitstream.Size()
		|| desc.Frame.Size > InBitstream.Size() - desc.Frame.Offset) {
		return OaStatus::Error(
			OaStatusCode::InvalidArgument,
			"VP9 frame payload is outside the input access unit");
	}
	OA_RETURN_IF_ERROR(InDecoder.UploadBitstream(OaSpan<const OaU8>(
		InBitstream.Data() + desc.Frame.Offset,
		desc.Frame.Size)));

	const bool isKeyFrame = desc.StdPictureInfo.frame_type == STD_VIDEO_VP9_FRAME_TYPE_KEY;
	if (isKeyFrame) {
		OaFnVideoDecoderAccess::ResetAllDpbSlotStates(InDecoder);
		for (OaI32& vp9Slot : InDecoder.Vp9BufferToDpbSlot_) {
			vp9Slot = -1;
		}
		InDecoder.Vp9BufferExtents_.Fill({0, 0});
	}

	auto releaseUnmappedSlots = [&InDecoder]() {
		for (OaI32 dpbIndex = 0; dpbIndex < 16; ++dpbIndex) {
			bool isMapped = false;
			for (OaI32 mapped : InDecoder.Vp9BufferToDpbSlot_) {
				if (mapped == dpbIndex) {
					isMapped = true;
					break;
				}
			}
			if (!isMapped && InDecoder.DpbSlots_[static_cast<OaUsize>(dpbIndex)].InUse) {
				InDecoder.ReleaseDpbSlot(dpbIndex);
			}
		}
	};

	// VP9 reference lifetime is defined by its eight logical reference buffers.
	// Age-based eviction can discard a physical slot still named by that map.
	releaseUnmappedSlots();
	OaI32 dpbSlot = InDecoder.AllocateDpbSlot();
	if (dpbSlot < 0) {
		return OaStatus::Error("DPB overflow - all VP9 slots are reference frames");
	}

	for (OaI32& mapped : InDecoder.Vp9BufferToDpbSlot_) {
		if (mapped == dpbSlot) {
			mapped = -1;
		}
	}

	OaI32 refNameSlotIndices[STD_VIDEO_VP9_REFS_PER_FRAME] = {-1, -1, -1};
	OaVec<OaI32> refSlots;
	OaVec<VkExtent2D> refExtents;
	if (!isKeyFrame) {
		for (OaU32 i = 0; i < STD_VIDEO_VP9_REFS_PER_FRAME; ++i) {
			const OaU8 bufferIdx = desc.RefFrameIdx[i];
			const OaI32 mappedSlot = InDecoder.Vp9BufferToDpbSlot_[bufferIdx];
			if (mappedSlot == dpbSlot) {
				refNameSlotIndices[i] = -1;
				continue;
			}
			refNameSlotIndices[i] = mappedSlot;
			if (mappedSlot < 0) {
				continue;
			}
			bool alreadyAdded = false;
			for (OaUsize j = 0; j < refSlots.Size(); ++j) {
				if (refSlots[j] == mappedSlot) {
					alreadyAdded = true;
					break;
				}
			}
			if (!alreadyAdded) {
				refSlots.PushBack(mappedSlot);
				refExtents.PushBack(InDecoder.Vp9BufferExtents_[bufferIdx]);
			}
		}
	}

	OA_RETURN_IF_ERROR(InDecoder.RecordVP9DecodeCommands(
		dpbSlot,
		desc,
		refNameSlotIndices,
		refSlots,
		refExtents));
	// Surface asynchronous video-queue failures at the decode boundary. Without
	// this, a bad VP9 submission appears to succeed and the following compute
	// conversion reports the unrelated-looking VK_ERROR_DEVICE_LOST.
	OA_RETURN_IF_ERROR(InDecoder.WaitForCompletion());

	for (OaU32 mask = desc.StdPictureInfo.refresh_frame_flags, refIndex = 0u; mask != 0u; mask >>= 1u, ++refIndex) {
		if ((mask & 1u) != 0u) {
			InDecoder.Vp9BufferToDpbSlot_[refIndex] = dpbSlot;
			InDecoder.Vp9BufferExtents_[refIndex] = {desc.FrameWidth, desc.FrameHeight};
		}
	}

	const bool keepAsReference = (desc.StdPictureInfo.refresh_frame_flags != 0u) || isKeyFrame;
	if (keepAsReference) {
		InDecoder.MarkSlotAsReference(dpbSlot, static_cast<OaI32>(InDecoder.CurrentFrameNumber_));
	} else {
		InDecoder.ReleaseDpbSlot(dpbSlot);
	}
	releaseUnmappedSlots();
	InDecoder.CurrentFrameNumber_++;

	OaFnVideoDecoderAccess::FillNv12OutFrame(
		InDecoder,
		dpbSlot,
		desc.FrameWidth > 0 ? desc.FrameWidth : InDecoder.Profile_.Width,
		desc.FrameHeight > 0 ? desc.FrameHeight : InDecoder.Profile_.Height,
		desc.Frame.Timestamp,
		OutFrame);
	// Hidden frames (show_frame=0) decode into the DPB but are not displayed;
	// the reorder layer drops !Shown frames.
	OutFrame.Shown = desc.StdPictureInfo.flags.show_frame != 0u;
	return OaStatus::Ok();
}

} // namespace FnVideoDecoderVp9
