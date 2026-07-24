// Shared helpers for FnVideoDecoder* decode impl.

#include "FnVideoDecoderShared.h"
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/OaVma.h>
#include "../../Video/Decoder/VideoDecoderProfile.h"
#include "../../Video/Codec/NalParser.h"

void OaFnVideoDecoderAccess::FillNv12OutFrame(
	OaVideoDecoder& InDecoder,
	OaI32 InDpbSlot,
	OaU32 InWidth,
	OaU32 InHeight,
	OaU64 InPts,
	OaVideoFrame& OutFrame)
{
	const OaUsize slot = static_cast<OaUsize>(InDpbSlot);
	OutFrame.Image = (!InDecoder.OutputImages_.Empty() && slot < InDecoder.OutputImages_.Size())
		? InDecoder.OutputImages_[slot]
		: InDecoder.Dpb_.GetImage();
	OutFrame.ImageView = (!InDecoder.OutputViews_.Empty() && slot < InDecoder.OutputViews_.Size())
		? InDecoder.OutputViews_[slot]
		: InDecoder.Dpb_.GetView();
	OutFrame.Layout = (!InDecoder.OutputImages_.Empty() && slot < InDecoder.OutputImages_.Size())
		? InDecoder.OutputImageLayouts_[slot]
		: InDecoder.DpbImageLayouts_[slot];
	OutFrame.Format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
	OutFrame.Width = InWidth;
	OutFrame.Height = InHeight;
	OutFrame.PresentationTimestamp = InPts;
	OutFrame.IsRgb = false;
	OutFrame.ArrayLayer = static_cast<OaU32>(InDpbSlot);
	InDecoder.StampFrameReady(OutFrame);
}

void OaFnVideoDecoderAccess::ResetAllDpbSlotStates(OaVideoDecoder& InDecoder)
{
	for (auto& dpbSlotState : InDecoder.DpbSlots_) {
		dpbSlotState.InUse = false;
		dpbSlotState.IsReference = false;
		dpbSlotState.IsLongTerm = false;
		dpbSlotState.PicOrderCnt = -1;
		dpbSlotState.H264FrameNum = 0;
		dpbSlotState.FrameNumber = 0;
	}
}

void OaVideoDecoder::MarkSlotAsReference(OaI32 InSlotIndex, OaI32 InPicOrderCnt)
{
	if (InSlotIndex >= 0 && InSlotIndex < 16) {
		DpbSlots_[InSlotIndex].IsReference = true;
		DpbSlots_[InSlotIndex].PicOrderCnt = InPicOrderCnt;
	}
}

void OaVideoDecoder::ReleaseDpbSlot(OaI32 InSlotIndex)
{
	if (InSlotIndex >= 0 && InSlotIndex < 16) {
		DpbSlots_[InSlotIndex].InUse = false;
		DpbSlots_[InSlotIndex].IsReference = false;
	}
}

OaStatus OaVideoDecoder::UploadBitstream(const OaSpan<const OaU8>& InBitstream)
{
	if (!Rt_) {
		return OaStatus::Error("Runtime not initialized");
	}

	CurrentBitstreamIndex_ = (CurrentBitstreamIndex_ + 1) % kBitstreamRingSize;
	BitstreamSlot& slot = BitstreamRing_[CurrentBitstreamIndex_];
	if (TimelineSem_.Semaphore != nullptr && slot.UseValue > 0) {
		OA_RETURN_IF_ERROR(TimelineSem_.Wait(
			Rt_->Device,
			slot.UseValue));
	}

	const OaU64 alignment = slot.Buffer.GetSizeAlignment() == 0 ? 1 : slot.Buffer.GetSizeAlignment();
	const OaU64 requiredSize = static_cast<OaU64>(OaAlignUp(InBitstream.Size(), static_cast<OaUsize>(alignment)));
	if (requiredSize == 0) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Video bitstream is empty");
	}

	auto& vkEngine = *Rt_;
	auto allocator = static_cast<OaVmaAllocator>(vkEngine.Allocator.Allocator);

	if (slot.Buffer.GetCapacity() < requiredSize) {
		VkVideoDecodeH264ProfileInfoKHR h264 = {};
		VkVideoDecodeH265ProfileInfoKHR h265 = {};
		VkVideoDecodeAV1ProfileInfoKHR av1 = {};
		VkVideoDecodeVP9ProfileInfoKHR vp9 = {};
		auto profileResult =
			OaVideoDecoderProfile::BuildDecodeProfile(Profile_, h264, h265, av1, vp9);
		if (not profileResult.IsOk()) {
			return profileResult.GetStatus();
		}
		const VkVideoProfileInfoKHR& profile = *profileResult;
		const OaU64 newCapacity = OaAlignUp(
			requiredSize,
			static_cast<OaU64>(4 * 1024 * 1024));
		auto replacement = OaVkVideoBitstream::Create(
			vkEngine,
			newCapacity,
			OaVkVideoBitstream::Direction::Decoder,
			slot.Buffer.GetOffsetAlignment(),
			slot.Buffer.GetSizeAlignment(),
			&profile);
		if (not replacement.IsOk()) {
			return replacement.GetStatus();
		}
		slot.Buffer = OaStdMove(*replacement);
	}

	void* mappedPtr = slot.Buffer.GetMappedPtr();
	if (!mappedPtr) {
		return OaStatus::Error("Vulkan Video bitstream buffer is not host mapped");
	}
	OaMemcpy(mappedPtr, InBitstream.Data(), InBitstream.Size());
	if (requiredSize > InBitstream.Size()) {
		auto* tail = static_cast<OaU8*>(mappedPtr) + InBitstream.Size();
		OaMemzero(tail, static_cast<OaUsize>(requiredSize - InBitstream.Size()));
	}
	VkResult flushResult = OaVmaFlushAllocation(
		allocator,
		static_cast<OaVmaAllocation>(slot.Buffer.GetAllocation()),
		0,
		requiredSize);
	if (flushResult != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::VulkanError, "Failed to flush Vulkan Video bitstream buffer");
	}
	slot.Size = InBitstream.Size();
	return OaStatus::Ok();
}

void OaVideoDecoder::BuildRefPicList0(OaI32 InCurrentPoc, OaVec<OaI32>& OutRefList)
{
	OutRefList.Clear();

	struct RefFrame { OaI32 SlotIndex; OaI32 Poc; };
	OaVec<RefFrame> candidates;

	for (OaI32 i = 0; i < 16; ++i) {
		if (DpbSlots_[i].InUse && DpbSlots_[i].IsReference) {
			OaI32 poc = DpbSlots_[i].PicOrderCnt;
			if (poc < InCurrentPoc) {
				candidates.PushBack({i, poc});
			}
		}
	}

	for (OaUsize i = 0; i < candidates.Size(); ++i) {
		for (OaUsize j = i + 1; j < candidates.Size(); ++j) {
			if (candidates[j].Poc > candidates[i].Poc) {
				RefFrame temp = candidates[i];
				candidates[i] = candidates[j];
				candidates[j] = temp;
			}
		}
	}

	for (const auto& ref : candidates) {
		OutRefList.PushBack(ref.SlotIndex);
	}
}

void OaVideoDecoder::BuildRefPicList1(OaI32 InCurrentPoc, OaVec<OaI32>& OutRefList)
{
	OutRefList.Clear();

	struct RefFrame { OaI32 SlotIndex; OaI32 Poc; };
	OaVec<RefFrame> candidates;

	for (OaI32 i = 0; i < 16; ++i) {
		if (DpbSlots_[i].InUse && DpbSlots_[i].IsReference) {
			OaI32 poc = DpbSlots_[i].PicOrderCnt;
			if (poc > InCurrentPoc) {
				candidates.PushBack({i, poc});
			}
		}
	}

	for (OaUsize i = 0; i < candidates.Size(); ++i) {
		for (OaUsize j = i + 1; j < candidates.Size(); ++j) {
			if (candidates[j].Poc < candidates[i].Poc) {
				RefFrame temp = candidates[i];
				candidates[i] = candidates[j];
				candidates[j] = temp;
			}
		}
	}

	for (const auto& ref : candidates) {
		OutRefList.PushBack(ref.SlotIndex);
	}
}

void OaVideoDecoder::BuildH264RefPicList0P(OaVec<OaI32>& OutRefList)
{
	// H.264 §8.2.4.2.1: the initial P-slice RefPicList0 lists short-term
	// references by PicNum (FrameNumWrap) descending, then long-term references
	// by LongTermFrameIdx ascending. The shared POC-keyed BuildRefPicList0 (used
	// for HEVC and H.264 B-slices) is wrong here: it filters out refs whose POC
	// is >= the current picture's and orders by display order. With more than one
	// active reference frame, the slice's ref_idx values then select the wrong
	// DPB picture for motion compensation — visible as motion smearing/glitches.
	OutRefList.Clear();

	const OaI32 maxFrameNum  = 1 << CurrentLog2MaxFrameNum_;
	const OaI32 currFrameNum = static_cast<OaI32>(CurrentH264FrameNum_);

	struct ShortRef { OaI32 SlotIndex; OaI32 PicNum; };
	OaVec<ShortRef> shortTerm;
	OaVec<OaI32> longTerm;

	for (OaI32 i = 0; i < 16; ++i) {
		if (!DpbSlots_[i].InUse || !DpbSlots_[i].IsReference) {
			continue;
		}
		if (DpbSlots_[i].IsLongTerm) {
			longTerm.PushBack(i);
			continue;
		}
		const OaI32 fn = static_cast<OaI32>(DpbSlots_[i].H264FrameNum);
		const OaI32 picNum = (fn > currFrameNum) ? (fn - maxFrameNum) : fn;
		shortTerm.PushBack({i, picNum});
	}

	// Short-term: descending PicNum (nearest preceding reference first).
	for (OaUsize i = 0; i < shortTerm.Size(); ++i) {
		for (OaUsize j = i + 1; j < shortTerm.Size(); ++j) {
			if (shortTerm[j].PicNum > shortTerm[i].PicNum) {
				ShortRef temp = shortTerm[i];
				shortTerm[i] = shortTerm[j];
				shortTerm[j] = temp;
			}
		}
	}
	for (const auto& ref : shortTerm) {
		OutRefList.PushBack(ref.SlotIndex);
	}

	// Long-term: LongTermFrameIdx ascending. We don't track the index
	// separately, so H264FrameNum is a stable proxy ordering; long-term refs
	// are rare in the streams we decode and always follow short-term refs.
	for (OaUsize i = 0; i < longTerm.Size(); ++i) {
		for (OaUsize j = i + 1; j < longTerm.Size(); ++j) {
			if (DpbSlots_[longTerm[j]].H264FrameNum < DpbSlots_[longTerm[i]].H264FrameNum) {
				OaI32 temp = longTerm[i];
				longTerm[i] = longTerm[j];
				longTerm[j] = temp;
			}
		}
	}
	for (OaI32 slot : longTerm) {
		OutRefList.PushBack(slot);
	}
}

void OaVideoDecoder::ApplySlidingWindow(OaU32 InMaxNumRefFrames)
{
	OaU32 refCount = 0;
	for (OaI32 i = 0; i < 16; ++i) {
		if (DpbSlots_[i].InUse && DpbSlots_[i].IsReference) {
			refCount++;
		}
	}

	if (Profile_.Codec != OaVideoCodec::H264 || CurrentLog2MaxFrameNum_ == 0) {
		while (refCount >= InMaxNumRefFrames) {
			OaI32 oldestSlot = -1;
			OaU64 oldestFrame = ~0ULL;
			for (OaI32 i = 0; i < 16; ++i) {
				if (DpbSlots_[i].InUse && DpbSlots_[i].IsReference && !DpbSlots_[i].IsLongTerm) {
					if (DpbSlots_[i].FrameNumber < oldestFrame) {
						oldestSlot = i;
						oldestFrame = DpbSlots_[i].FrameNumber;
					}
				}
			}
			if (oldestSlot >= 0) {
				DpbSlots_[oldestSlot].IsReference = false;
				DpbSlots_[oldestSlot].InUse = false;
				refCount--;
			} else {
				break;
			}
		}
		return;
	}

	while (refCount >= InMaxNumRefFrames) {
		const OaI32 maxFrameNum = 1 << (CurrentLog2MaxFrameNum_);
		const OaI32 currFrameNum = static_cast<OaI32>(CurrentH264FrameNum_);
		OaI32 oldestSlot = -1;
		OaI32 oldestWrap = currFrameNum;
		for (OaI32 i = 0; i < 16; ++i) {
			if (!DpbSlots_[i].InUse || !DpbSlots_[i].IsReference || DpbSlots_[i].IsLongTerm) {
				continue;
			}
			const OaI32 fn = static_cast<OaI32>(DpbSlots_[i].H264FrameNum);
			const OaI32 wrap = (fn > currFrameNum) ? (fn - maxFrameNum) : fn;
			if (oldestSlot < 0 || wrap < oldestWrap) {
				oldestSlot = i;
				oldestWrap = wrap;
			}
		}

		if (oldestSlot >= 0) {
			DpbSlots_[oldestSlot].IsReference = false;
			DpbSlots_[oldestSlot].InUse = false;
			refCount--;
		} else {
			break;
		}
	}
}

void OaVideoDecoder::ApplyMmco(
	const OaVec<OaH264MmcoCommand>& InMmcoCommands,
	OaI32 InCurrentDpbSlot)
{
	auto findShortTermByH264FrameNum = [&](OaU32 InH264FrameNum) -> OaI32 {
		for (OaI32 i = 0; i < static_cast<OaI32>(DpbSlots_.Size()); ++i) {
			if (DpbSlots_[i].InUse && DpbSlots_[i].IsReference
				&& not DpbSlots_[i].IsLongTerm
				&& DpbSlots_[i].H264FrameNum == InH264FrameNum) {
				return i;
			}
		}
		return -1;
	};

	const OaI32 maxFrameNum = 1 << CurrentLog2MaxFrameNum_;
	for (const auto& cmd : InMmcoCommands) {
		switch (cmd.Op) {
		case 1: {
			const OaI32 curr = static_cast<OaI32>(CurrentH264FrameNum_);
			const OaI32 picNumX = curr - static_cast<OaI32>(cmd.DifferenceOfPicNumsMinus1 + 1U);
			OaI32 slot = -1;
			for (OaI32 i = 0; i < static_cast<OaI32>(DpbSlots_.Size()); ++i) {
				if (!DpbSlots_[i].InUse || !DpbSlots_[i].IsReference || DpbSlots_[i].IsLongTerm) {
					continue;
				}
				const OaI32 fn = static_cast<OaI32>(DpbSlots_[i].H264FrameNum);
				const OaI32 wrap = (fn > curr) ? (fn - maxFrameNum) : fn;
				if (wrap == picNumX) {
					slot = i;
					break;
				}
			}
			if (slot >= 0) {
				DpbSlots_[slot].IsReference = false;
				DpbSlots_[slot].InUse       = false;
			}
			break;
		}
		case 2: {
			const OaI32 slot = findShortTermByH264FrameNum(cmd.LongTermPicNum);
			if (slot >= 0) {
				DpbSlots_[slot].IsReference = false;
				DpbSlots_[slot].InUse       = false;
			}
			break;
		}
		case 3:
		case 4:
			break;
		case 5: {
			for (auto& slot : DpbSlots_) {
				slot.IsReference = false;
			}
			break;
		}
		case 6: {
			if (InCurrentDpbSlot >= 0
				&& static_cast<OaUsize>(InCurrentDpbSlot) < DpbSlots_.Size()) {
				DpbSlots_[InCurrentDpbSlot].IsReference = true;
			}
			break;
		}
		default:
			break;
		}
	}
}

void OaVideoDecoder::ApplyMmco(const OaVec<OaU32>& InMmcoCommands)
{
	(void)InMmcoCommands;
}
