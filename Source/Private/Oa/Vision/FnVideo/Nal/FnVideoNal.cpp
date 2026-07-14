// OaFnVideo — VideoNal category.
// CPU-only Annex-B byte stream split / emit / SPS+PPS extraction.

#include <Oa/Vision/FnVideo.h>

#include <cstring>

namespace OaFnVideo
{

namespace {

constexpr OaU8 kSps = 7;   // H.264 nal_unit_type SPS
constexpr OaU8 kPps = 8;   // H.264 nal_unit_type PPS

// Scan for the next Annex-B start code (00 00 00 01 or 00 00 01) at or after
// InOffset. Returns the offset of the first prefix byte, or InSize if none.
OaUsize FindStartCode(const OaU8* InBytes, OaUsize InSize, OaUsize InOffset, OaUsize& OutPrefixLen)
{
	for (OaUsize i = InOffset; i + 2 < InSize; ++i) {
		if (InBytes[i] == 0 && InBytes[i + 1] == 0) {
			if (InBytes[i + 2] == 1) {
				OutPrefixLen = 3;
				return i;
			}
			if (i + 3 < InSize && InBytes[i + 2] == 0 && InBytes[i + 3] == 1) {
				OutPrefixLen = 4;
				return i;
			}
		}
	}
	OutPrefixLen = 0;
	return InSize;
}

OaNalUnit MakeUnit(const OaU8* InPayload, OaUsize InPayloadLen)
{
	OaNalUnit unit{};
	if (InPayloadLen == 0) {
		return unit;
	}
	const OaU8 header = InPayload[0];
	unit.Type    = static_cast<OaU8>(header & 0x1FU);
	unit.RefIdc  = static_cast<OaU8>((header >> 5) & 0x3U);
	unit.Payload = OaSpan<const OaU8>(InPayload, InPayloadLen);
	return unit;
}

} // namespace

OaVec<OaNalUnit> ParseNalAnnexB(const OaSpan<const OaU8>& InBytes)
{
	OaVec<OaNalUnit> out;
	const OaU8* bytes = InBytes.data();
	const OaUsize size = InBytes.size();
	if (bytes == nullptr || size == 0) {
		return out;
	}

	OaUsize prefixLen = 0;
	OaUsize cur = FindStartCode(bytes, size, 0, prefixLen);
	while (cur < size) {
		const OaUsize payloadStart = cur + prefixLen;
		OaUsize nextPrefixLen = 0;
		const OaUsize next = FindStartCode(bytes, size, payloadStart, nextPrefixLen);
		const OaUsize payloadLen = next - payloadStart;
		if (payloadLen > 0) {
			out.PushBack(MakeUnit(bytes + payloadStart, payloadLen));
		}
		cur = next;
		prefixLen = nextPrefixLen;
	}
	return out;
}

OaVec<OaU8> EmitNalAnnexB(const OaSpan<const OaNalUnit>& InUnits)
{
	OaVec<OaU8> out;
	OaUsize total = 0;
	for (OaUsize i = 0; i < InUnits.size(); ++i) {
		total += 4 + InUnits[i].Payload.size();
	}
	out.Reserve(total);
	for (OaUsize i = 0; i < InUnits.size(); ++i) {
		out.PushBack(0x00);
		out.PushBack(0x00);
		out.PushBack(0x00);
		out.PushBack(0x01);
		const OaU8* payload = InUnits[i].Payload.data();
		const OaUsize len = InUnits[i].Payload.size();
		for (OaUsize j = 0; j < len; ++j) {
			out.PushBack(payload[j]);
		}
	}
	return out;
}

namespace {

OaVec<OaU8> ExtractFirstNalByType(const OaSpan<const OaU8>& InNalBytes, OaU8 InType)
{
	const auto units = ParseNalAnnexB(InNalBytes);
	for (OaUsize i = 0; i < units.Size(); ++i) {
		if (units[i].Type == InType) {
			OaVec<OaU8> out;
			out.Reserve(units[i].Payload.size());
			const OaU8* p = units[i].Payload.data();
			for (OaUsize j = 0; j < units[i].Payload.size(); ++j) {
				out.PushBack(p[j]);
			}
			return out;
		}
	}
	return OaVec<OaU8>{};
}

OaVec<OaU8> ExtractFirstH265NalByType(
	const OaSpan<const OaU8>& InNalBytes,
	OaU8 InType)
{
	const OaU8* bytes = InNalBytes.Data();
	const OaUsize size = InNalBytes.Size();
	OaUsize prefixLength = 0U;
	OaUsize start = FindStartCode(bytes, size, 0U, prefixLength);
	while (start < size) {
		const OaUsize payloadStart = start + prefixLength;
		OaUsize nextPrefixLength = 0U;
		const OaUsize next = FindStartCode(bytes, size, payloadStart, nextPrefixLength);
		OaUsize payloadEnd = next;
		while (payloadEnd > payloadStart and bytes[payloadEnd - 1U] == 0U) {
			--payloadEnd;
		}
		if (payloadEnd > payloadStart
			and static_cast<OaU8>((bytes[payloadStart] >> 1U) & 0x3FU) == InType) {
			OaVec<OaU8> out;
			out.Resize(payloadEnd - payloadStart);
			std::memcpy(out.Data(), bytes + payloadStart, out.Size());
			return out;
		}
		start = next;
		prefixLength = nextPrefixLength;
	}
	return {};
}

} // namespace

OaVec<OaU8> ExtractSps(const OaSpan<const OaU8>& InNalBytes)
{
	return ExtractFirstNalByType(InNalBytes, kSps);
}

OaVec<OaU8> ExtractPps(const OaSpan<const OaU8>& InNalBytes)
{
	return ExtractFirstNalByType(InNalBytes, kPps);
}

OaVec<OaU8> ExtractVpsH265(const OaSpan<const OaU8>& InNalBytes)
{
	return ExtractFirstH265NalByType(InNalBytes, 32U);
}

OaVec<OaU8> ExtractSpsH265(const OaSpan<const OaU8>& InNalBytes)
{
	return ExtractFirstH265NalByType(InNalBytes, 33U);
}

OaVec<OaU8> ExtractPpsH265(const OaSpan<const OaU8>& InNalBytes)
{
	return ExtractFirstH265NalByType(InNalBytes, 34U);
}

} // namespace OaFnVideo
