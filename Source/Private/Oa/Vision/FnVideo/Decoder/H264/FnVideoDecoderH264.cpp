// H.264 access-unit decode path (FnVideoDecoder* impl).

#include "FnVideoDecoderH264.h"
#include "../FnVideoDecoderShared.h"
#include "../../../Video/Codec/VcpH264.h"

#include <cstdio>

namespace FnVideoDecoderH264 {

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
		: OaStatus::Error(OaStatusCode::InvalidArgument, "Invalid H.264 Annex-B access unit");
}

} // namespace

OaStatus DecodeFrame(
	OaVideoDecoder& InDecoder,
	const OaSpan<const OaU8>& InBitstream,
	OaVideoFrame& OutFrame)
{
	auto* parser = static_cast<OaVcpH264*>(InDecoder.Parser_.Get());
	if (!parser) {
		return OaStatus::Error("H.264 parser not registered");
	}

	OaVec<OaU8> normalizedBitstream;
	OA_RETURN_IF_ERROR(NormalizeAnnexB(InBitstream, normalizedBitstream));
	const OaSpan<const OaU8> bitstream(
		normalizedBitstream.Data(),
		normalizedBitstream.Size());

	OaH264PictureDesc desc;
	OA_RETURN_IF_ERROR(parser->ParseAccessUnit(bitstream, desc));
	OA_RETURN_IF_ERROR(InDecoder.UploadBitstream(bitstream));

	for (OaU32 spsId : parser->GetCachedSpsIds()) {
		auto* sps = parser->GetOaSps(spsId);
		if (sps) {
			InDecoder.SpsCache_.Insert({spsId, *sps});
			OA_RETURN_IF_ERROR(InDecoder.UpdateH264SessionParametersFromSps(*sps));
		}
	}
	for (OaU32 ppsId : parser->GetCachedPpsIds()) {
		auto* pps = parser->GetOaPps(ppsId);
		if (pps) {
			InDecoder.PpsCache_.Insert({ppsId, *pps});
			OA_RETURN_IF_ERROR(InDecoder.UpdateH264SessionParametersFromPps(*pps));
		}
	}

	if (!desc.HasPicture) {
		return OaStatus::Ok();
	}

	OaSliceHeader sliceHeader = desc.SliceHeader;
	const OaH264SpsData& sps = desc.Sps;
	const OaH264PpsData& pps = desc.Pps;
	(void)pps;

	if (sliceHeader.IsIdrPic) {
		for (auto& dpbSlotState : InDecoder.DpbSlots_) {
			dpbSlotState.InUse = false;
			dpbSlotState.IsReference = false;
			dpbSlotState.IsLongTerm = false;
			dpbSlotState.PicOrderCnt = -1;
			dpbSlotState.H264FrameNum = 0;
		}
	}
	InDecoder.CurrentH264FrameNum_ = sliceHeader.FrameNum;
	InDecoder.CurrentLog2MaxFrameNum_ = sps.Log2MaxFrameNumMinus4 + 4U;

	const bool useMmco = sliceHeader.RefPicMarkingValid
		&& sliceHeader.AdaptiveRefPicMarking
		&& !sliceHeader.MmcoCommands.Empty();
	OaI32 dpbSlot = InDecoder.AllocateDpbSlot();
	if (dpbSlot < 0) {
		return OaStatus::Error("DPB overflow - all 16 slots are reference frames");
	}

	OaI32 fullPoc = sliceHeader.PicOrderCntLsb;
	if (sps.PicOrderCntType == 0) {
		const OaU32 maxLsb = 1U << (sps.Log2MaxPicOrderCntLsbMinus4 + 4U);
		const OaI32 lsb = sliceHeader.PicOrderCntLsb;
		if (sliceHeader.IsIdrPic) {
			InDecoder.PrevPocLsb_ = 0;
			InDecoder.PrevPocMsb_ = 0;
		}
		OaI32 msb = InDecoder.PrevPocMsb_;
		if (lsb < InDecoder.PrevPocLsb_
			&& (InDecoder.PrevPocLsb_ - lsb) >= static_cast<OaI32>(maxLsb / 2U)) {
			msb = InDecoder.PrevPocMsb_ + static_cast<OaI32>(maxLsb);
		} else if (lsb > InDecoder.PrevPocLsb_
			&& (lsb - InDecoder.PrevPocLsb_) > static_cast<OaI32>(maxLsb / 2U)) {
			msb = InDecoder.PrevPocMsb_ - static_cast<OaI32>(maxLsb);
		}
		fullPoc = msb + lsb;
		if (sliceHeader.IsReference) {
			InDecoder.PrevPocMsb_ = msb;
			InDecoder.PrevPocLsb_ = lsb;
		}
	}
	sliceHeader.PicOrderCntLsb = fullPoc;
	desc.SliceHeader.PicOrderCntLsb = fullPoc;

	OaVec<OaI32> refPicList0;
	OaVec<OaI32> refPicList1;
	if (sliceHeader.SliceType == OaH264SliceType::P) {
		// P-slice list0 is PicNum-ordered (§8.2.4.2.1), not POC-ordered.
		InDecoder.BuildH264RefPicList0P(refPicList0);
	} else if (sliceHeader.SliceType == OaH264SliceType::B) {
		InDecoder.BuildRefPicList0(sliceHeader.PicOrderCntLsb, refPicList0);
		InDecoder.BuildRefPicList1(sliceHeader.PicOrderCntLsb, refPicList1);
	}

	OA_RETURN_IF_ERROR(InDecoder.RecordH264DecodeCommands(
		dpbSlot,
		desc,
		refPicList0,
		refPicList1));

	if (sliceHeader.IsReference) {
		InDecoder.MarkSlotAsReference(dpbSlot, sliceHeader.PicOrderCntLsb);
	}
	if (useMmco) {
		InDecoder.ApplyMmco(sliceHeader.MmcoCommands, dpbSlot);
	} else if (sliceHeader.IsReference) {
		const OaU32 maxRefs = sps.MaxNumRefFrames > 0 ? sps.MaxNumRefFrames : 1U;
		InDecoder.ApplySlidingWindow(maxRefs + 1U);
	}
	if (!sliceHeader.IsReference) {
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

} // namespace FnVideoDecoderH264
