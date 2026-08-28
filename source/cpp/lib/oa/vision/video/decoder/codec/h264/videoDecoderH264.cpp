// H.264 access-unit decode path for oa::VideoDecoder.

#include "../videoDecoderCodecAccess.h"
#include "../../../codec/vcpH264.h"

#include <stdio.h>

namespace {

oa::Status normalizeAnnexB(
	const oa::Span<const oa::U8>& inBitstream,
	oa::Vector<oa::U8>& outBitstream)
{
	outBitstream.clear();
	const oa::U8* data = inBitstream.data();
	const oa::Usize size = inBitstream.size();
	oa::Usize search = 0;
	bool foundNal = false;
	while (search < size) {
		oa::Usize start = search;
		oa::Usize prefixSize = 0;
		while (start < size) {
			if (start + 3 <= size &&
				data[start] == 0 && data[start + 1] == 0 && data[start + 2] == 1) {
				prefixSize = 3;
				break;
			}
			if (start + 4 <= size &&
				data[start] == 0 && data[start + 1] == 0 &&
				data[start + 2] == 0 && data[start + 3] == 1) {
				prefixSize = 4;
				break;
			}
			++start;
		}
		if (prefixSize == 0) {
			break;
		}
		const oa::Usize nalStart = start + prefixSize;
		oa::Usize next = nalStart;
		while (next < size) {
			if ((next + 3 <= size &&
				 data[next] == 0 && data[next + 1] == 0 && data[next + 2] == 1) ||
				(next + 4 <= size &&
				 data[next] == 0 && data[next + 1] == 0 &&
				 data[next + 2] == 0 && data[next + 3] == 1)) {
				break;
			}
			++next;
		}
		if (nalStart < next) {
			outBitstream.pushBack(0);
			outBitstream.pushBack(0);
			outBitstream.pushBack(1);
			for (oa::Usize i = nalStart; i < next; ++i) {
				outBitstream.pushBack(data[i]);
			}
			foundNal = true;
		}
		search = next;
	}
	return foundNal
		? oa::Status::ok()
		: oa::Status::error(oa::StatusCode::InvalidArgument, "Invalid H.264 Annex-B access unit");
}

} // namespace

oa::Status oa::VideoDecoderCodecAccess::decodeH264(
	oa::VideoDecoder& inDecoder,
	const oa::Span<const oa::U8>& inBitstream,
	oa::VideoFrame& outFrame)
{
	auto* parser = static_cast<oa::VcpH264*>(inDecoder.impl_->parser.get());
	if (!parser) {
		return oa::Status::error("H.264 parser not registered");
	}

	oa::Vector<oa::U8> normalizedBitstream;
	OA_RETURN_IF_ERROR(normalizeAnnexB(inBitstream, normalizedBitstream));
	const oa::Span<const oa::U8> bitstream(
		normalizedBitstream.data(),
		normalizedBitstream.size());

	oa::H264PictureDesc desc;
	OA_RETURN_IF_ERROR(parser->parseAccessUnit(bitstream, desc));
	OA_RETURN_IF_ERROR(inDecoder.uploadBitstream(bitstream));

	for (oa::U32 spsId : parser->getCachedSpsIds()) {
		auto* sps = parser->getSpsData(spsId);
		if (sps) {
			inDecoder.impl_->spsCache.insert({spsId, *sps});
			OA_RETURN_IF_ERROR(inDecoder.updateH264SessionParametersFromSps(*sps));
		}
	}
	for (oa::U32 ppsId : parser->getCachedPpsIds()) {
		auto* pps = parser->getPpsData(ppsId);
		if (pps) {
			inDecoder.impl_->ppsCache.insert({ppsId, *pps});
			OA_RETURN_IF_ERROR(inDecoder.updateH264SessionParametersFromPps(*pps));
		}
	}

	if (!desc.hasPicture) {
		return oa::Status::ok();
	}

	oa::H264SliceHeader sliceHeader = desc.sliceHeader;
	const oa::H264SpsData& sps = desc.sps;
	const oa::H264PpsData& pps = desc.pps;
	(void)pps;

	if (sliceHeader.isIdrPic) {
		for (auto& dpbSlotState : inDecoder.impl_->dpbSlots) {
			dpbSlotState.inUse = false;
			dpbSlotState.isReference = false;
			dpbSlotState.isLongTerm = false;
			dpbSlotState.picOrderCnt = -1;
			dpbSlotState.h264FrameNum = 0;
		}
	}
	inDecoder.impl_->currentH264FrameNumber = sliceHeader.frameNum;
	inDecoder.impl_->currentLog2MaxFrameNumber = sps.log2MaxFrameNumMinus4 + 4U;

	const bool useMmco = sliceHeader.refPicMarkingValid
		&& sliceHeader.adaptiveRefPicMarking
		&& !sliceHeader.mmcoCommands.empty();
	oa::I32 dpbSlot = inDecoder.allocateDpbSlot();
	if (dpbSlot < 0) {
		return oa::Status::error("DPB overflow - all 16 slots are reference frames");
	}

	oa::I32 fullPoc = sliceHeader.picOrderCntLsb;
	if (sps.picOrderCntType == 0) {
		const oa::U32 maxLsb = 1U << (sps.log2MaxPicOrderCntLsbMinus4 + 4U);
		const oa::I32 lsb = sliceHeader.picOrderCntLsb;
		if (sliceHeader.isIdrPic) {
			inDecoder.impl_->previousPocLsb = 0;
			inDecoder.impl_->previousPocMsb = 0;
		}
		oa::I32 msb = inDecoder.impl_->previousPocMsb;
		if (lsb < inDecoder.impl_->previousPocLsb
			&& (inDecoder.impl_->previousPocLsb - lsb) >= static_cast<oa::I32>(maxLsb / 2U)) {
			msb = inDecoder.impl_->previousPocMsb + static_cast<oa::I32>(maxLsb);
		} else if (lsb > inDecoder.impl_->previousPocLsb
			&& (lsb - inDecoder.impl_->previousPocLsb) > static_cast<oa::I32>(maxLsb / 2U)) {
			msb = inDecoder.impl_->previousPocMsb - static_cast<oa::I32>(maxLsb);
		}
		fullPoc = msb + lsb;
		if (sliceHeader.isReference) {
			inDecoder.impl_->previousPocMsb = msb;
			inDecoder.impl_->previousPocLsb = lsb;
		}
	}
	sliceHeader.picOrderCntLsb = fullPoc;
	desc.sliceHeader.picOrderCntLsb = fullPoc;

	oa::Vector<oa::I32> refPicList0;
	oa::Vector<oa::I32> refPicList1;
	if (sliceHeader.sliceType == oa::H264SliceType::P) {
		// P-slice list0 is picNum-ordered (§8.2.4.2.1), not POC-ordered.
		inDecoder.buildH264RefPicList0P(refPicList0);
	} else if (sliceHeader.sliceType == oa::H264SliceType::B) {
		inDecoder.buildRefPicList0(sliceHeader.picOrderCntLsb, refPicList0);
		inDecoder.buildRefPicList1(sliceHeader.picOrderCntLsb, refPicList1);
	}

	OA_RETURN_IF_ERROR(inDecoder.recordH264DecodeCommands(
		dpbSlot,
		desc,
		refPicList0,
		refPicList1));

	if (sliceHeader.isReference) {
		inDecoder.markSlotAsReference(dpbSlot, sliceHeader.picOrderCntLsb);
	}
	if (useMmco) {
		inDecoder.applyMmco(sliceHeader.mmcoCommands, dpbSlot);
	} else if (sliceHeader.isReference) {
		const oa::U32 maxRefs = sps.maxNumRefFrames > 0 ? sps.maxNumRefFrames : 1U;
		inDecoder.applySlidingWindow(maxRefs + 1U);
	}
	if (!sliceHeader.isReference) {
		inDecoder.releaseDpbSlot(dpbSlot);
	}

	inDecoder.impl_->currentFrameNumber++;

	fillNv12OutFrame(
		inDecoder,
		dpbSlot,
		inDecoder.impl_->profile.width,
		inDecoder.impl_->profile.height,
		inDecoder.impl_->currentFrameNumber,
		outFrame);
	return oa::Status::ok();
}
