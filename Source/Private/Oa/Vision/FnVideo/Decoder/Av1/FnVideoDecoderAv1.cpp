// AV1 access-unit decode path (FnVideoDecoder* impl).

#include "FnVideoDecoderAv1.h"
#include "../FnVideoDecoderShared.h"
#include "../../../Video/Codec/VcpAv1.h"

namespace FnVideoDecoderAv1 {

OaStatus DecodePicture(
	OaVideoDecoder& InDecoder,
	const OaSpan<const OaU8>& InBitstream,
	const OaAv1PictureDesc& desc,
	OaVideoFrame& OutFrame)
{
	if (desc.HasPicture || desc.ShowExistingFrame) {
		if (desc.Frame.Size == 0 || desc.Frame.Offset + desc.Frame.Size > InBitstream.Size()) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Invalid AV1 frame payload range");
		}
		if (desc.FrameHeaderOffset > desc.Frame.Size) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Invalid AV1 frame header offset");
		}
		// Upload the decode OBU slice (OBU header + payload). frameHeaderOffset is the
		// OBU header size; tile offsets are rebased into this buffer.
		const OaUsize obuOff = desc.DecodeObuOffset;
		const OaUsize obuSize = desc.DecodeObuSize;
		if (obuSize == 0 || obuOff + obuSize > desc.Frame.Size) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Invalid AV1 decode OBU range");
		}
		OA_RETURN_IF_ERROR(InDecoder.UploadBitstream(OaSpan<const OaU8>(
			InBitstream.Data() + desc.Frame.Offset + obuOff, obuSize)));
	}

	OA_RETURN_IF_ERROR(InDecoder.UpdateAv1SessionParametersFromSequenceHeader(desc.SequenceHeader));

	if (desc.ShowExistingFrame) {
		OaI32 logical = desc.FrameToShowMapIdx;
		OaI32 dpbSlot = (logical >= 0 && static_cast<OaU32>(logical) < STD_VIDEO_AV1_NUM_REF_FRAMES)
			? InDecoder.Av1RefFrameToDpbSlot_[static_cast<OaUsize>(logical)] : -1;
		if (dpbSlot < 0) {
			return OaStatus::Error(OaStatusCode::Unavailable, "AV1 show_existing_frame references an invalid/unavailable DPB slot");
		}
		OaFnVideoDecoderAccess::FillNv12OutFrame(
			InDecoder,
			dpbSlot,
			InDecoder.Profile_.Width,
			InDecoder.Profile_.Height,
			desc.Frame.Timestamp,
			OutFrame);
		OutFrame.Shown = true;  // show_existing_frame re-displays a stored slot
		return OaStatus::Ok();
	}

	if (!desc.HasPicture) {
		return OaStatus::Ok();
	}

	const bool isKeyFrame = (desc.FrameHeader.FrameType == STD_VIDEO_AV1_FRAME_TYPE_KEY);
	if (isKeyFrame) {
		OaFnVideoDecoderAccess::ResetAllDpbSlotStates(InDecoder);
		for (OaI32& av1Slot : InDecoder.Av1RefFrameToDpbSlot_) {
			av1Slot = -1;
		}
		InDecoder.Av1DpbReferenceInfos_.Fill({});
		InDecoder.SlotDeviceActivated_.Fill(false);
	}

	auto releaseUnmappedSlots = [&InDecoder]() {
		for (OaI32 dpbIndex = 0; dpbIndex < 16; ++dpbIndex) {
			bool isMapped = false;
			for (OaI32 mapped : InDecoder.Av1RefFrameToDpbSlot_) {
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

	// AV1 reference lifetime is defined by the eight logical reference frame
	// slots, not by display age. Releasing a still-mapped physical DPB slot
	// leaves future inter frames pointing at overwritten contents.
	releaseUnmappedSlots();
	OaI32 dpbSlot = InDecoder.AllocateDpbSlot();
	if (dpbSlot < 0) {
		return OaStatus::Error("DPB overflow - all 16 slots are reference frames");
	}

	for (OaI32& mapped : InDecoder.Av1RefFrameToDpbSlot_) {
		if (mapped == dpbSlot) {
			mapped = -1;
		}
	}

	OaI32 refNameSlotIndices[OaAv1MaxReferencesPerFrame] = {-1, -1, -1, -1, -1, -1, -1};
	for (OaU32 i = 0; i < OaAv1MaxReferencesPerFrame; ++i) {
		const OaI32 logical = desc.FrameHeader.ReferenceNameSlotIndices[i];
		if (logical >= 0 && static_cast<OaU32>(logical) < STD_VIDEO_AV1_NUM_REF_FRAMES) {
			refNameSlotIndices[i] = InDecoder.Av1RefFrameToDpbSlot_[static_cast<OaUsize>(logical)];
		}
	}

	OA_RETURN_IF_ERROR(InDecoder.RecordAV1DecodeCommands(dpbSlot, desc, refNameSlotIndices));
	OA_RETURN_IF_ERROR(InDecoder.WaitForCompletion());

	for (OaU32 mask = desc.FrameHeader.RefreshFrameFlags, refIndex = 0u;
	     mask != 0u; mask >>= 1u, ++refIndex) {
		if ((mask & 1u) != 0u && refIndex < STD_VIDEO_AV1_NUM_REF_FRAMES) {
			InDecoder.Av1RefFrameToDpbSlot_[refIndex] = dpbSlot;
		}
	}

	const bool keepAsReference = (desc.FrameHeader.RefreshFrameFlags != 0u) || isKeyFrame;
	if (keepAsReference) {
		InDecoder.MarkSlotAsReference(dpbSlot, static_cast<OaI32>(desc.FrameHeader.OrderHint));
	} else {
		InDecoder.ReleaseDpbSlot(dpbSlot);
	}
	releaseUnmappedSlots();
	InDecoder.CurrentFrameNumber_++;

	OaFnVideoDecoderAccess::FillNv12OutFrame(
		InDecoder,
		dpbSlot,
		InDecoder.Profile_.Width,
		InDecoder.Profile_.Height,
		desc.Frame.Timestamp,
		OutFrame);
	// Hidden alt-ref frames (show_frame=0) decode into the DPB but must not be
	// presented; the reorder layer drops !Shown frames. show_existing_frame is
	// handled above and is always shown.
	OutFrame.Shown = desc.FrameHeader.ShowFrame;
	return OaStatus::Ok();
}

OaStatus DecodeFrame(
	OaVideoDecoder& InDecoder,
	const OaSpan<const OaU8>& InBitstream,
	OaVideoFrame& OutFrame)
{
	OutFrame = {};
	auto* parser = static_cast<OaVcpAv1*>(InDecoder.Parser_.Get());
	if (!parser) return OaStatus::Error("AV1 parser not registered");

	OaVec<OaAv1PictureDesc> pictures;
	OA_RETURN_IF_ERROR(parser->ParseAccessUnitPictures(InBitstream, pictures));
	OaVideoFrame lastDecoded = {};
	for (const OaAv1PictureDesc& desc : pictures) {
		OaVideoFrame decoded = {};
		OA_RETURN_IF_ERROR(DecodePicture(InDecoder, InBitstream, desc, decoded));
		if (decoded.ImageView != VK_NULL_HANDLE) lastDecoded = decoded;
		if (decoded.ImageView != VK_NULL_HANDLE && decoded.Shown) OutFrame = decoded;
	}
	if (OutFrame.ImageView == VK_NULL_HANDLE) OutFrame = lastDecoded;
	return OaStatus::Ok();
}

} // namespace FnVideoDecoderAv1
