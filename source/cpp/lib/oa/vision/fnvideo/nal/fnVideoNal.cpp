// oa::FnVideo — VideoNal category.
// CPU-only Annex-B byte stream split / emit / SPS+PPS extraction.

#include <oa/vision/fnVideo.h>
#include <oa/core/memory.h>

namespace oa {

namespace FnVideo {

namespace {

constexpr oa::U8 kSps = 7;   // H.264 nal_unit_type SPS
constexpr oa::U8 kPps = 8;   // H.264 nal_unit_type PPS

// Scan for the next Annex-B start code (00 00 00 01 or 00 00 01) at or after
// inOffset. Returns the offset of the first prefix byte, or inSize if none.
oa::Usize findStartCode(const oa::U8* inBytes, oa::Usize inSize, oa::Usize inOffset, oa::Usize& outPrefixLen)
{
	for (oa::Usize i = inOffset; i + 2 < inSize; ++i) {
		if (inBytes[i] == 0 && inBytes[i + 1] == 0) {
			if (inBytes[i + 2] == 1) {
				outPrefixLen = 3;
				return i;
			}
			if (i + 3 < inSize && inBytes[i + 2] == 0 && inBytes[i + 3] == 1) {
				outPrefixLen = 4;
				return i;
			}
		}
	}
	outPrefixLen = 0;
	return inSize;
}

NalUnit makeUnit(const oa::U8* inPayload, oa::Usize inPayloadLen)
{
	NalUnit unit{};
	if (inPayloadLen == 0) {
		return unit;
	}
	const oa::U8 header = inPayload[0];
	unit.type    = static_cast<oa::U8>(header & 0x1FU);
	unit.refIdc  = static_cast<oa::U8>((header >> 5) & 0x3U);
	unit.payload = oa::Span<const oa::U8>(inPayload, inPayloadLen);
	return unit;
}

} // namespace

oa::Vec<NalUnit> parseNalAnnexB(const oa::Span<const oa::U8>& inBytes)
{
	oa::Vec<NalUnit> out;
	const oa::U8* bytes = inBytes.data();
	const oa::Usize size = inBytes.size();
	if (bytes == nullptr || size == 0) {
		return out;
	}

	oa::Usize prefixLen = 0;
	oa::Usize cur = findStartCode(bytes, size, 0, prefixLen);
	while (cur < size) {
		const oa::Usize payloadStart = cur + prefixLen;
		oa::Usize nextPrefixLen = 0;
		const oa::Usize next = findStartCode(bytes, size, payloadStart, nextPrefixLen);
		const oa::Usize payloadLen = next - payloadStart;
		if (payloadLen > 0) {
			out.pushBack(makeUnit(bytes + payloadStart, payloadLen));
		}
		cur = next;
		prefixLen = nextPrefixLen;
	}
	return out;
}

oa::Vec<oa::U8> emitNalAnnexB(const oa::Span<const NalUnit>& inUnits)
{
	oa::Vec<oa::U8> out;
	oa::Usize total = 0;
	for (oa::Usize i = 0; i < inUnits.size(); ++i) {
		total += 4 + inUnits[i].payload.size();
	}
	out.reserve(total);
	for (oa::Usize i = 0; i < inUnits.size(); ++i) {
		out.pushBack(0x00);
		out.pushBack(0x00);
		out.pushBack(0x00);
		out.pushBack(0x01);
		const oa::U8* payload = inUnits[i].payload.data();
		const oa::Usize len = inUnits[i].payload.size();
		for (oa::Usize j = 0; j < len; ++j) {
			out.pushBack(payload[j]);
		}
	}
	return out;
}

namespace {

oa::Vec<oa::U8> extractFirstNalByType(const oa::Span<const oa::U8>& inNalBytes, oa::U8 inType)
{
	const auto units = parseNalAnnexB(inNalBytes);
	for (oa::Usize i = 0; i < units.size(); ++i) {
		if (units[i].type == inType) {
			oa::Vec<oa::U8> out;
			out.reserve(units[i].payload.size());
			const oa::U8* p = units[i].payload.data();
			for (oa::Usize j = 0; j < units[i].payload.size(); ++j) {
				out.pushBack(p[j]);
			}
			return out;
		}
	}
	return oa::Vec<oa::U8>{};
}

oa::Vec<oa::U8> extractFirstH265NalByType(
	const oa::Span<const oa::U8>& inNalBytes,
	oa::U8 inType)
{
	const oa::U8* bytes = inNalBytes.data();
	const oa::Usize size = inNalBytes.size();
	oa::Usize prefixLength = 0U;
	oa::Usize start = findStartCode(bytes, size, 0U, prefixLength);
	while (start < size) {
		const oa::Usize payloadStart = start + prefixLength;
		oa::Usize nextPrefixLength = 0U;
		const oa::Usize next = findStartCode(bytes, size, payloadStart, nextPrefixLength);
		oa::Usize payloadEnd = next;
		while (payloadEnd > payloadStart and bytes[payloadEnd - 1U] == 0U) {
			--payloadEnd;
		}
		if (payloadEnd > payloadStart
			and static_cast<oa::U8>((bytes[payloadStart] >> 1U) & 0x3FU) == inType) {
			oa::Vec<oa::U8> out;
			out.resize(payloadEnd - payloadStart);
			oa::memcpy(out.data(), bytes + payloadStart, out.size());
			return out;
		}
		start = next;
		prefixLength = nextPrefixLength;
	}
	return {};
}

} // namespace

oa::Vec<oa::U8> extractSps(const oa::Span<const oa::U8>& inNalBytes)
{
	return extractFirstNalByType(inNalBytes, kSps);
}

oa::Vec<oa::U8> extractPps(const oa::Span<const oa::U8>& inNalBytes)
{
	return extractFirstNalByType(inNalBytes, kPps);
}

oa::Vec<oa::U8> extractVpsH265(const oa::Span<const oa::U8>& inNalBytes)
{
	return extractFirstH265NalByType(inNalBytes, 32U);
}

oa::Vec<oa::U8> extractSpsH265(const oa::Span<const oa::U8>& inNalBytes)
{
	return extractFirstH265NalByType(inNalBytes, 33U);
}

oa::Vec<oa::U8> extractPpsH265(const oa::Span<const oa::U8>& inNalBytes)
{
	return extractFirstH265NalByType(inNalBytes, 34U);
}

} // namespace FnVideo

} // namespace oa
