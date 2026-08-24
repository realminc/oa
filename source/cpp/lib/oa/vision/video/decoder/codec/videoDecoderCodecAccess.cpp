// Shared helpers for the stateful decoder codec implementations.

#include "videoDecoderCodecAccess.h"
#include <oa/runtime/engine.h>
#include <oa/runtime/engine/allocatorAccess.h>
#include <oa/runtime/engine/deviceAccess.h>
#include <oa/runtime/oaVma.h>
#include "../videoDecoderProfile.h"
#include "../../codec/nalParser.h"

void oa::VideoDecoderCodecAccess::fillNv12OutFrame(
	oa::VideoDecoder& inDecoder,
	oa::I32 inDpbSlot,
	oa::U32 inWidth,
	oa::U32 inHeight,
	oa::U64 inPts,
	oa::VideoFrame& outFrame)
{
	const oa::Usize slot = static_cast<oa::Usize>(inDpbSlot);
	outFrame.image = (!inDecoder.impl_->outputImages.empty() && slot < inDecoder.impl_->outputImages.size())
		? inDecoder.impl_->outputImages[slot]
		: inDecoder.impl_->dpb.getImage();
	outFrame.imageView = (!inDecoder.impl_->outputViews.empty() && slot < inDecoder.impl_->outputViews.size())
		? inDecoder.impl_->outputViews[slot]
		: inDecoder.impl_->dpb.getView();
	outFrame.layout = (!inDecoder.impl_->outputImages.empty() && slot < inDecoder.impl_->outputImages.size())
		? inDecoder.impl_->outputImageLayouts[slot]
		: inDecoder.impl_->dpbImageLayouts[slot];
	outFrame.format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
	outFrame.width = inWidth;
	outFrame.height = inHeight;
	outFrame.presentationTimestamp = inPts;
	outFrame.isRgb = false;
	outFrame.arrayLayer = static_cast<oa::U32>(inDpbSlot);
	inDecoder.stampFrameReady(outFrame);
}

void oa::VideoDecoderCodecAccess::resetAllDpbSlotStates(oa::VideoDecoder& inDecoder)
{
	for (auto& dpbSlotState : inDecoder.impl_->dpbSlots) {
		dpbSlotState.inUse = false;
		dpbSlotState.isReference = false;
		dpbSlotState.isLongTerm = false;
		dpbSlotState.picOrderCnt = -1;
		dpbSlotState.h264FrameNum = 0;
		dpbSlotState.frameNumber = 0;
	}
}

void oa::VideoDecoder::markSlotAsReference(oa::I32 inSlotIndex, oa::I32 inPicOrderCnt)
{
	if (inSlotIndex >= 0 && inSlotIndex < 16) {
		impl_->dpbSlots[inSlotIndex].isReference = true;
		impl_->dpbSlots[inSlotIndex].picOrderCnt = inPicOrderCnt;
	}
}

void oa::VideoDecoder::releaseDpbSlot(oa::I32 inSlotIndex)
{
	if (inSlotIndex >= 0 && inSlotIndex < 16) {
		impl_->dpbSlots[inSlotIndex].inUse = false;
		impl_->dpbSlots[inSlotIndex].isReference = false;
	}
}

oa::Status oa::VideoDecoder::uploadBitstream(const oa::Span<const oa::U8>& inBitstream)
{
	if (!impl_->engine) {
		return oa::Status::error("Runtime not initialized");
	}

	impl_->currentBitstreamIndex = (impl_->currentBitstreamIndex + 1) % kBitstreamRingSize;
	BitstreamSlot& slot = impl_->bitstreamRing[impl_->currentBitstreamIndex];
	if (impl_->timelineSemaphore.semaphore != nullptr && slot.useValue > 0) {
		OA_RETURN_IF_ERROR(impl_->timelineSemaphore.wait(
			oa::EngineDeviceAccess::get(*impl_->engine),
			slot.useValue));
	}

	const oa::U64 alignment = slot.buffer.getSizeAlignment() == 0 ? 1 : slot.buffer.getSizeAlignment();
	const oa::U64 requiredSize = static_cast<oa::U64>(oa::alignUp(inBitstream.size(), static_cast<oa::Usize>(alignment)));
	if (requiredSize == 0) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "Video bitstream is empty");
	}

	auto& vkEngine = *impl_->engine;
	auto allocator = static_cast<OaVmaAllocator>(oa::EngineAllocatorAccess::get(vkEngine).allocator);

	if (slot.buffer.getCapacity() < requiredSize) {
		VkVideoDecodeH264ProfileInfoKHR h264 = {};
		VkVideoDecodeH265ProfileInfoKHR h265 = {};
		VkVideoDecodeAV1ProfileInfoKHR av1 = {};
		VkVideoDecodeVP9ProfileInfoKHR vp9 = {};
		auto profileResult =
			oa::videoDecoderProfile::buildDecodeProfile(impl_->profile, h264, h265, av1, vp9);
		if (not profileResult.isOk()) {
			return profileResult.getStatus();
		}
		const VkVideoProfileInfoKHR& profile = *profileResult;
		const oa::U64 newCapacity = oa::alignUp(
			requiredSize,
			static_cast<oa::U64>(4 * 1024 * 1024));
		auto replacement = oavk::VideoBitstream::create(
			vkEngine,
			newCapacity,
			oavk::VideoBitstream::Direction::Decoder,
			slot.buffer.getOffsetAlignment(),
			slot.buffer.getSizeAlignment(),
			&profile);
		if (not replacement.isOk()) {
			return replacement.getStatus();
		}
		slot.buffer = oa::move(*replacement);
	}

	void* mappedPtr = slot.buffer.getMappedPtr();
	if (!mappedPtr) {
		return oa::Status::error("vulkan Video bitstream buffer is not host mapped");
	}
	oa::memcpy(mappedPtr, inBitstream.data(), inBitstream.size());
	if (requiredSize > inBitstream.size()) {
		auto* tail = static_cast<oa::U8*>(mappedPtr) + inBitstream.size();
		oa::memzero(tail, static_cast<oa::Usize>(requiredSize - inBitstream.size()));
	}
	VkResult flushResult = OaVmaFlushAllocation(
		allocator,
		static_cast<OaVmaAllocation>(slot.buffer.getAllocation()),
		0,
		requiredSize);
	if (flushResult != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::VulkanError, "Failed to flush vulkan Video bitstream buffer");
	}
	slot.size = inBitstream.size();
	return oa::Status::ok();
}

void oa::VideoDecoder::buildRefPicList0(oa::I32 inCurrentPoc, oa::Vec<oa::I32>& outRefList)
{
	outRefList.clear();

	struct RefFrame { oa::I32 slotIndex; oa::I32 poc; };
	oa::Vec<RefFrame> candidates;

	for (oa::I32 i = 0; i < 16; ++i) {
		if (impl_->dpbSlots[i].inUse && impl_->dpbSlots[i].isReference) {
			oa::I32 poc = impl_->dpbSlots[i].picOrderCnt;
			if (poc < inCurrentPoc) {
				candidates.pushBack({i, poc});
			}
		}
	}

	for (oa::Usize i = 0; i < candidates.size(); ++i) {
		for (oa::Usize j = i + 1; j < candidates.size(); ++j) {
			if (candidates[j].poc > candidates[i].poc) {
				RefFrame temp = candidates[i];
				candidates[i] = candidates[j];
				candidates[j] = temp;
			}
		}
	}

	for (const auto& ref : candidates) {
		outRefList.pushBack(ref.slotIndex);
	}
}

void oa::VideoDecoder::buildRefPicList1(oa::I32 inCurrentPoc, oa::Vec<oa::I32>& outRefList)
{
	outRefList.clear();

	struct RefFrame { oa::I32 slotIndex; oa::I32 poc; };
	oa::Vec<RefFrame> candidates;

	for (oa::I32 i = 0; i < 16; ++i) {
		if (impl_->dpbSlots[i].inUse && impl_->dpbSlots[i].isReference) {
			oa::I32 poc = impl_->dpbSlots[i].picOrderCnt;
			if (poc > inCurrentPoc) {
				candidates.pushBack({i, poc});
			}
		}
	}

	for (oa::Usize i = 0; i < candidates.size(); ++i) {
		for (oa::Usize j = i + 1; j < candidates.size(); ++j) {
			if (candidates[j].poc < candidates[i].poc) {
				RefFrame temp = candidates[i];
				candidates[i] = candidates[j];
				candidates[j] = temp;
			}
		}
	}

	for (const auto& ref : candidates) {
		outRefList.pushBack(ref.slotIndex);
	}
}

void oa::VideoDecoder::buildH264RefPicList0P(oa::Vec<oa::I32>& outRefList)
{
	// H.264 §8.2.4.2.1: the initial P-slice refPicList0 lists short-term
	// references by picNum (FrameNumWrap) descending, then long-term references
	// by longTermFrameIdx ascending. The shared POC-keyed buildRefPicList0 (used
	// for HEVC and H.264 B-slices) is wrong here: it filters out refs whose POC
	// is >= the current picture's and orders by display order. With more than one
	// active reference frame, the slice's ref_idx values then select the wrong
	// DPB picture for motion compensation — visible as motion smearing/glitches.
	outRefList.clear();

	const oa::I32 maxFrameNum  = 1 << impl_->currentLog2MaxFrameNumber;
	const oa::I32 currFrameNum = static_cast<oa::I32>(impl_->currentH264FrameNumber);

	struct ShortRef { oa::I32 slotIndex; oa::I32 picNum; };
	oa::Vec<ShortRef> shortTerm;
	oa::Vec<oa::I32> longTerm;

	for (oa::I32 i = 0; i < 16; ++i) {
		if (!impl_->dpbSlots[i].inUse || !impl_->dpbSlots[i].isReference) {
			continue;
		}
		if (impl_->dpbSlots[i].isLongTerm) {
			longTerm.pushBack(i);
			continue;
		}
		const oa::I32 fn = static_cast<oa::I32>(impl_->dpbSlots[i].h264FrameNum);
		const oa::I32 picNum = (fn > currFrameNum) ? (fn - maxFrameNum) : fn;
		shortTerm.pushBack({i, picNum});
	}

	// Short-term: descending picNum (nearest preceding reference first).
	for (oa::Usize i = 0; i < shortTerm.size(); ++i) {
		for (oa::Usize j = i + 1; j < shortTerm.size(); ++j) {
			if (shortTerm[j].picNum > shortTerm[i].picNum) {
				ShortRef temp = shortTerm[i];
				shortTerm[i] = shortTerm[j];
				shortTerm[j] = temp;
			}
		}
	}
	for (const auto& ref : shortTerm) {
		outRefList.pushBack(ref.slotIndex);
	}

	// Long-term: longTermFrameIdx ascending. We don't track the index
	// separately, so h264FrameNum is a stable proxy ordering; long-term refs
	// are rare in the streams we decode and always follow short-term refs.
	for (oa::Usize i = 0; i < longTerm.size(); ++i) {
		for (oa::Usize j = i + 1; j < longTerm.size(); ++j) {
			if (impl_->dpbSlots[longTerm[j]].h264FrameNum < impl_->dpbSlots[longTerm[i]].h264FrameNum) {
				oa::I32 temp = longTerm[i];
				longTerm[i] = longTerm[j];
				longTerm[j] = temp;
			}
		}
	}
	for (oa::I32 slot : longTerm) {
		outRefList.pushBack(slot);
	}
}

void oa::VideoDecoder::applySlidingWindow(oa::U32 inMaxNumRefFrames)
{
	oa::U32 refCount = 0;
	for (oa::I32 i = 0; i < 16; ++i) {
		if (impl_->dpbSlots[i].inUse && impl_->dpbSlots[i].isReference) {
			refCount++;
		}
	}

	if (impl_->profile.codec != oa::VideoCodec::H264 || impl_->currentLog2MaxFrameNumber == 0) {
		while (refCount >= inMaxNumRefFrames) {
			oa::I32 oldestSlot = -1;
			oa::U64 oldestFrame = ~0ULL;
			for (oa::I32 i = 0; i < 16; ++i) {
				if (impl_->dpbSlots[i].inUse && impl_->dpbSlots[i].isReference && !impl_->dpbSlots[i].isLongTerm) {
					if (impl_->dpbSlots[i].frameNumber < oldestFrame) {
						oldestSlot = i;
						oldestFrame = impl_->dpbSlots[i].frameNumber;
					}
				}
			}
			if (oldestSlot >= 0) {
				impl_->dpbSlots[oldestSlot].isReference = false;
				impl_->dpbSlots[oldestSlot].inUse = false;
				refCount--;
			} else {
				break;
			}
		}
		return;
	}

	while (refCount >= inMaxNumRefFrames) {
		const oa::I32 maxFrameNum = 1 << (impl_->currentLog2MaxFrameNumber);
		const oa::I32 currFrameNum = static_cast<oa::I32>(impl_->currentH264FrameNumber);
		oa::I32 oldestSlot = -1;
		oa::I32 oldestWrap = currFrameNum;
		for (oa::I32 i = 0; i < 16; ++i) {
			if (!impl_->dpbSlots[i].inUse || !impl_->dpbSlots[i].isReference || impl_->dpbSlots[i].isLongTerm) {
				continue;
			}
			const oa::I32 fn = static_cast<oa::I32>(impl_->dpbSlots[i].h264FrameNum);
			const oa::I32 wrap = (fn > currFrameNum) ? (fn - maxFrameNum) : fn;
			if (oldestSlot < 0 || wrap < oldestWrap) {
				oldestSlot = i;
				oldestWrap = wrap;
			}
		}

		if (oldestSlot >= 0) {
			impl_->dpbSlots[oldestSlot].isReference = false;
			impl_->dpbSlots[oldestSlot].inUse = false;
			refCount--;
		} else {
			break;
		}
	}
}

void oa::VideoDecoder::applyMmco(
	const oa::Vec<oa::H264MmcoCommand>& inMmcoCommands,
	oa::I32 inCurrentDpbSlot)
{
	auto findShortTermByH264FrameNum = [&](oa::U32 inH264FrameNum) -> oa::I32 {
		for (oa::I32 i = 0; i < static_cast<oa::I32>(impl_->dpbSlots.size()); ++i) {
			if (impl_->dpbSlots[i].inUse && impl_->dpbSlots[i].isReference
				&& not impl_->dpbSlots[i].isLongTerm
				&& impl_->dpbSlots[i].h264FrameNum == inH264FrameNum) {
				return i;
			}
		}
		return -1;
	};

	const oa::I32 maxFrameNum = 1 << impl_->currentLog2MaxFrameNumber;
	for (const auto& cmd : inMmcoCommands) {
		switch (cmd.op) {
		case 1: {
			const oa::I32 curr = static_cast<oa::I32>(impl_->currentH264FrameNumber);
			const oa::I32 picNumX = curr - static_cast<oa::I32>(cmd.differenceOfPicNumsMinus1 + 1U);
			oa::I32 slot = -1;
			for (oa::I32 i = 0; i < static_cast<oa::I32>(impl_->dpbSlots.size()); ++i) {
				if (!impl_->dpbSlots[i].inUse || !impl_->dpbSlots[i].isReference || impl_->dpbSlots[i].isLongTerm) {
					continue;
				}
				const oa::I32 fn = static_cast<oa::I32>(impl_->dpbSlots[i].h264FrameNum);
				const oa::I32 wrap = (fn > curr) ? (fn - maxFrameNum) : fn;
				if (wrap == picNumX) {
					slot = i;
					break;
				}
			}
			if (slot >= 0) {
				impl_->dpbSlots[slot].isReference = false;
				impl_->dpbSlots[slot].inUse       = false;
			}
			break;
		}
		case 2: {
			const oa::I32 slot = findShortTermByH264FrameNum(cmd.longTermPicNum);
			if (slot >= 0) {
				impl_->dpbSlots[slot].isReference = false;
				impl_->dpbSlots[slot].inUse       = false;
			}
			break;
		}
		case 3:
		case 4:
			break;
		case 5: {
			for (auto& slot : impl_->dpbSlots) {
				slot.isReference = false;
			}
			break;
		}
		case 6: {
			if (inCurrentDpbSlot >= 0
				&& static_cast<oa::Usize>(inCurrentDpbSlot) < impl_->dpbSlots.size()) {
				impl_->dpbSlots[inCurrentDpbSlot].isReference = true;
			}
			break;
		}
		default:
			break;
		}
	}
}

void oa::VideoDecoder::applyMmco(const oa::Vec<oa::U32>& inMmcoCommands)
{
	(void)inMmcoCommands;
}
