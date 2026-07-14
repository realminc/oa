// H.265 access-unit decode path (FnVideoDecoder* impl).

#include "FnVideoDecoderH265.h"
#include "../FnVideoDecoderShared.h"
#include "../../../Video/Codec/VcpH265.h"

namespace FnVideoDecoderH265 {

namespace {

OaStatus NormalizeAnnexB(
	const OaSpan<const OaU8>& InBitstream,
	OaVec<OaU8>& OutBitstream)
{
	OutBitstream.Clear();
	const OaU8* data = InBitstream.Data();
	const OaUsize size = InBitstream.Size();
	OaUsize search = 0;
	bool foundNal = false;
	while (search < size) {
		OaUsize start = search;
		OaUsize prefixSize = 0;
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
		const OaUsize nalStart = start + prefixSize;
		OaUsize next = nalStart;
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
			OutBitstream.PushBack(0);
			OutBitstream.PushBack(0);
			OutBitstream.PushBack(1);
			for (OaUsize i = nalStart; i < next; ++i) {
				OutBitstream.PushBack(data[i]);
			}
			foundNal = true;
		}
		search = next;
	}
	return foundNal
		? OaStatus::Ok()
		: OaStatus::Error(OaStatusCode::InvalidArgument, "Invalid H.265 Annex-B access unit");
}

} // namespace

OaStatus DecodeFrame(
	OaVideoDecoder& InDecoder,
	const OaSpan<const OaU8>& InBitstream,
	OaVideoFrame& OutFrame)
{
	auto* parser = static_cast<OaVcpH265*>(InDecoder.Parser_.Get());
	if (!parser) {
		return OaStatus::Error("H.265 parser not registered");
	}

	OaVec<OaU8> normalizedBitstream;
	OA_RETURN_IF_ERROR(NormalizeAnnexB(InBitstream, normalizedBitstream));
	const OaSpan<const OaU8> bitstream(
		normalizedBitstream.Data(),
		normalizedBitstream.Size());

	OaH265PictureDesc desc;
	OA_RETURN_IF_ERROR(parser->ParseAccessUnit(bitstream, desc));
	OA_RETURN_IF_ERROR(InDecoder.UploadBitstream(bitstream));

	for (OaU32 vpsId : parser->GetCachedVpsIds()) {
		auto* vps = parser->GetOaVps(vpsId);
		if (vps) {
			InDecoder.H265VpsCache_.Insert({vpsId, *vps});
			OA_RETURN_IF_ERROR(InDecoder.UpdateH265SessionParametersFromVps(*vps));
		}
	}
	for (OaU32 spsId : parser->GetCachedSpsIds()) {
		auto* sps = parser->GetOaSps(spsId);
		if (sps) {
			InDecoder.H265SpsCache_.Insert({spsId, *sps});
			OA_RETURN_IF_ERROR(InDecoder.UpdateH265SessionParametersFromSps(*sps));
		}
	}
	for (OaU32 ppsId : parser->GetCachedPpsIds()) {
		auto* pps = parser->GetOaPps(ppsId);
		if (pps) {
			InDecoder.H265PpsCache_.Insert({ppsId, *pps});
			OA_RETURN_IF_ERROR(InDecoder.UpdateH265SessionParametersFromPps(*pps));
		}
	}

	if (!desc.HasPicture) {
		return OaStatus::Ok();
	}

	const OaI32 maxPocLsb =
		1 << (desc.Sps.Log2MaxPicOrderCntLsbMinus4 + 4u);
	const bool isBla =
		desc.SliceHeader.NalUnitType >= 16u &&
		desc.SliceHeader.NalUnitType <= 18u;
	const bool resetsPoc =
		desc.SliceHeader.IsIdr ||
		isBla ||
		desc.SliceHeader.NoOutputOfPriorPics;
	OaI32 pocMsb = InDecoder.H265PrevPocMsb_;
	if (resetsPoc || !InDecoder.H265HasPrevPoc_) {
		pocMsb = 0;
	} else if (
		static_cast<OaI32>(desc.SliceHeader.PicOrderCntLsb)
			< InDecoder.H265PrevPocLsb_ &&
		InDecoder.H265PrevPocLsb_
			- static_cast<OaI32>(desc.SliceHeader.PicOrderCntLsb)
			>= maxPocLsb / 2) {
		pocMsb += maxPocLsb;
	} else if (
		static_cast<OaI32>(desc.SliceHeader.PicOrderCntLsb)
			> InDecoder.H265PrevPocLsb_ &&
		static_cast<OaI32>(desc.SliceHeader.PicOrderCntLsb)
			- InDecoder.H265PrevPocLsb_
			> maxPocLsb / 2) {
		pocMsb -= maxPocLsb;
	}
	desc.SliceHeader.PicOrderCntVal =
		pocMsb + static_cast<OaI32>(desc.SliceHeader.PicOrderCntLsb);

	// HEVC 8.3.1 updates the previous POC only for temporal-layer-zero
	// reference pictures outside the leading-picture NAL classes.
	const bool isLeadingPicture =
		desc.SliceHeader.NalUnitType >= 6u &&
		desc.SliceHeader.NalUnitType <= 9u;
	if (desc.SliceHeader.TemporalId == 0 &&
		desc.SliceHeader.IsReference &&
		!isLeadingPicture) {
		InDecoder.H265PrevPocLsb_ =
			static_cast<OaI32>(desc.SliceHeader.PicOrderCntLsb);
		InDecoder.H265PrevPocMsb_ = pocMsb;
		InDecoder.H265HasPrevPoc_ = true;
	}

	if (resetsPoc) {
		OaFnVideoDecoderAccess::ResetAllDpbSlotStates(InDecoder);
	}

	auto isRetainedReference = [&](OaI32 InPoc) {
		auto containsPoc = [&](const OaVec<OaI32>& InDeltaPocs) {
			for (OaI32 deltaPoc : InDeltaPocs) {
				if (desc.SliceHeader.PicOrderCntVal + deltaPoc == InPoc) {
					return true;
				}
			}
			return false;
		};
		return containsPoc(desc.SliceHeader.StCurrBeforeDeltaPocs) ||
			containsPoc(desc.SliceHeader.StCurrAfterDeltaPocs) ||
			containsPoc(desc.SliceHeader.StFollDeltaPocs);
	};
	for (OaI32 slot = 0; slot < static_cast<OaI32>(InDecoder.DpbSlotCapacity_); ++slot) {
		if (InDecoder.DpbSlots_[slot].InUse &&
			InDecoder.DpbSlots_[slot].IsReference &&
			!isRetainedReference(InDecoder.DpbSlots_[slot].PicOrderCnt)) {
			InDecoder.ReleaseDpbSlot(slot);
		}
	}
	OaI32 dpbSlot = InDecoder.AllocateDpbSlot();
	if (dpbSlot < 0) {
		return OaStatus::Error("DPB overflow - all 16 slots are reference frames");
	}

	OaVec<OaI32> refPicList0;
	OaVec<OaI32> refPicList1;
	auto resolveReferences = [&](const OaVec<OaI32>& InDeltaPocs, OaVec<OaI32>& OutSlots) -> OaStatus {
		for (OaI32 deltaPoc : InDeltaPocs) {
			const OaI32 targetPoc = desc.SliceHeader.PicOrderCntVal + deltaPoc;
			OaI32 targetSlot = -1;
			for (OaI32 slot = 0; slot < static_cast<OaI32>(InDecoder.DpbSlotCapacity_); ++slot) {
				if (InDecoder.DpbSlots_[slot].InUse &&
					InDecoder.DpbSlots_[slot].IsReference &&
					InDecoder.DpbSlots_[slot].PicOrderCnt == targetPoc) {
					targetSlot = slot;
					break;
				}
			}
			if (targetSlot < 0) {
				return OaStatus::Error(
					OaStatusCode::Unavailable,
					"H.265 short-term reference picture is missing from the DPB");
			}
			OutSlots.PushBack(targetSlot);
		}
		return OaStatus::Ok();
	};
	OA_RETURN_IF_ERROR(resolveReferences(desc.SliceHeader.StCurrBeforeDeltaPocs, refPicList0));
	OA_RETURN_IF_ERROR(resolveReferences(desc.SliceHeader.StCurrAfterDeltaPocs, refPicList1));

	OA_RETURN_IF_ERROR(InDecoder.RecordH265DecodeCommands(
		dpbSlot,
		desc,
		refPicList0,
		refPicList1));

	if (desc.SliceHeader.IsReference) {
		InDecoder.MarkSlotAsReference(dpbSlot, desc.SliceHeader.PicOrderCntVal);
	} else {
		InDecoder.ReleaseDpbSlot(dpbSlot);
	}
	InDecoder.CurrentFrameNumber_++;

	OaFnVideoDecoderAccess::FillNv12OutFrame(
		InDecoder,
		dpbSlot,
		InDecoder.Profile_.Width,
		InDecoder.Profile_.Height,
		InDecoder.CurrentFrameNumber_,
		OutFrame);
	return OaStatus::Ok();
}

} // namespace FnVideoDecoderH265
