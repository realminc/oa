// H.265 access-unit decode path for oa::VideoDecoder.

#include "../videoDecoderCodecAccess.h"
#include "../../../codec/vcpH265.h"

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
		: oa::Status::error(oa::StatusCode::InvalidArgument, "Invalid H.265 Annex-B access unit");
}

} // namespace

oa::Status oa::VideoDecoderCodecAccess::decodeH265(
	oa::VideoDecoder& inDecoder,
	const oa::Span<const oa::U8>& inBitstream,
	oa::VideoFrame& outFrame)
{
	auto* parser = static_cast<oa::VcpH265*>(inDecoder.impl_->parser.get());
	if (!parser) {
		return oa::Status::error("H.265 parser not registered");
	}

	oa::Vector<oa::U8> normalizedBitstream;
	OA_RETURN_IF_ERROR(normalizeAnnexB(inBitstream, normalizedBitstream));
	const oa::Span<const oa::U8> bitstream(
		normalizedBitstream.data(),
		normalizedBitstream.size());

	oa::H265PictureDesc desc;
	OA_RETURN_IF_ERROR(parser->parseAccessUnit(bitstream, desc));
	OA_RETURN_IF_ERROR(inDecoder.uploadBitstream(bitstream));

	for (oa::U32 vpsId : parser->getCachedVpsIds()) {
		auto* vps = parser->getVpsData(vpsId);
		if (vps) {
			inDecoder.impl_->h265VpsCache.insert({vpsId, *vps});
			OA_RETURN_IF_ERROR(inDecoder.updateH265SessionParametersFromVps(*vps));
		}
	}
	for (oa::U32 spsId : parser->getCachedSpsIds()) {
		auto* sps = parser->getSpsData(spsId);
		if (sps) {
			inDecoder.impl_->h265SpsCache.insert({spsId, *sps});
			OA_RETURN_IF_ERROR(inDecoder.updateH265SessionParametersFromSps(*sps));
		}
	}
	for (oa::U32 ppsId : parser->getCachedPpsIds()) {
		auto* pps = parser->getPpsData(ppsId);
		if (pps) {
			inDecoder.impl_->h265PpsCache.insert({ppsId, *pps});
			OA_RETURN_IF_ERROR(inDecoder.updateH265SessionParametersFromPps(*pps));
		}
	}

	if (!desc.hasPicture) {
		return oa::Status::ok();
	}

	const oa::I32 maxPocLsb =
		1 << (desc.sps.log2MaxPicOrderCntLsbMinus4 + 4u);
	const bool isBla =
		desc.sliceHeader.nalUnitType >= 16u &&
		desc.sliceHeader.nalUnitType <= 18u;
	const bool resetsPoc =
		desc.sliceHeader.isIdr ||
		isBla ||
		desc.sliceHeader.noOutputOfPriorPics;
	oa::I32 pocMsb = inDecoder.impl_->h265PreviousPocMsb;
	if (resetsPoc || !inDecoder.impl_->h265HasPreviousPoc) {
		pocMsb = 0;
	} else if (
		static_cast<oa::I32>(desc.sliceHeader.picOrderCntLsb)
			< inDecoder.impl_->h265PreviousPocLsb &&
		inDecoder.impl_->h265PreviousPocLsb
			- static_cast<oa::I32>(desc.sliceHeader.picOrderCntLsb)
			>= maxPocLsb / 2) {
		pocMsb += maxPocLsb;
	} else if (
		static_cast<oa::I32>(desc.sliceHeader.picOrderCntLsb)
			> inDecoder.impl_->h265PreviousPocLsb &&
		static_cast<oa::I32>(desc.sliceHeader.picOrderCntLsb)
			- inDecoder.impl_->h265PreviousPocLsb
			> maxPocLsb / 2) {
		pocMsb -= maxPocLsb;
	}
	desc.sliceHeader.picOrderCntVal =
		pocMsb + static_cast<oa::I32>(desc.sliceHeader.picOrderCntLsb);

	// HEVC 8.3.1 updates the previous POC only for temporal-layer-zero
	// reference pictures outside the leading-picture NAL classes.
	const bool isLeadingPicture =
		desc.sliceHeader.nalUnitType >= 6u &&
		desc.sliceHeader.nalUnitType <= 9u;
	if (desc.sliceHeader.temporalId == 0 &&
		desc.sliceHeader.isReference &&
		!isLeadingPicture) {
		inDecoder.impl_->h265PreviousPocLsb =
			static_cast<oa::I32>(desc.sliceHeader.picOrderCntLsb);
		inDecoder.impl_->h265PreviousPocMsb = pocMsb;
		inDecoder.impl_->h265HasPreviousPoc = true;
	}

	if (resetsPoc) {
		resetAllDpbSlotStates(inDecoder);
	}

	auto isRetainedReference = [&](oa::I32 inPoc) {
		auto containsPoc = [&](const oa::Vector<oa::I32>& inDeltaPocs) {
			for (oa::I32 deltaPoc : inDeltaPocs) {
				if (desc.sliceHeader.picOrderCntVal + deltaPoc == inPoc) {
					return true;
				}
			}
			return false;
		};
		return containsPoc(desc.sliceHeader.stCurrBeforeDeltaPocs) ||
			containsPoc(desc.sliceHeader.stCurrAfterDeltaPocs) ||
			containsPoc(desc.sliceHeader.stFollDeltaPocs);
	};
	for (oa::I32 slot = 0; slot < static_cast<oa::I32>(inDecoder.impl_->dpbSlotCapacity); ++slot) {
		if (inDecoder.impl_->dpbSlots[slot].inUse &&
			inDecoder.impl_->dpbSlots[slot].isReference &&
			!isRetainedReference(inDecoder.impl_->dpbSlots[slot].picOrderCnt)) {
			inDecoder.releaseDpbSlot(slot);
		}
	}
	oa::I32 dpbSlot = inDecoder.allocateDpbSlot();
	if (dpbSlot < 0) {
		return oa::Status::error("DPB overflow - all 16 slots are reference frames");
	}

	oa::Vector<oa::I32> refPicList0;
	oa::Vector<oa::I32> refPicList1;
	auto resolveReferences = [&](const oa::Vector<oa::I32>& inDeltaPocs, oa::Vector<oa::I32>& outSlots) -> oa::Status {
		for (oa::I32 deltaPoc : inDeltaPocs) {
			const oa::I32 targetPoc = desc.sliceHeader.picOrderCntVal + deltaPoc;
			oa::I32 targetSlot = -1;
			for (oa::I32 slot = 0; slot < static_cast<oa::I32>(inDecoder.impl_->dpbSlotCapacity); ++slot) {
				if (inDecoder.impl_->dpbSlots[slot].inUse &&
					inDecoder.impl_->dpbSlots[slot].isReference &&
					inDecoder.impl_->dpbSlots[slot].picOrderCnt == targetPoc) {
					targetSlot = slot;
					break;
				}
			}
			if (targetSlot < 0) {
				return oa::Status::error(
					oa::StatusCode::Unavailable,
					"H.265 short-term reference picture is missing from the DPB");
			}
			outSlots.pushBack(targetSlot);
		}
		return oa::Status::ok();
	};
	OA_RETURN_IF_ERROR(resolveReferences(desc.sliceHeader.stCurrBeforeDeltaPocs, refPicList0));
	OA_RETURN_IF_ERROR(resolveReferences(desc.sliceHeader.stCurrAfterDeltaPocs, refPicList1));

	OA_RETURN_IF_ERROR(inDecoder.recordH265DecodeCommands(
		dpbSlot,
		desc,
		refPicList0,
		refPicList1));

	if (desc.sliceHeader.isReference) {
		inDecoder.markSlotAsReference(dpbSlot, desc.sliceHeader.picOrderCntVal);
	} else {
		inDecoder.releaseDpbSlot(dpbSlot);
	}
	inDecoder.impl_->currentFrameNumber++;

	fillDecodedOutFrame(
		inDecoder,
		dpbSlot,
		inDecoder.impl_->profile.width,
		inDecoder.impl_->profile.height,
		inDecoder.impl_->currentFrameNumber,
		outFrame);
	return oa::Status::ok();
}
