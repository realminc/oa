// OA Vision — bounded container demux and compressed-video packet source.

#include <oa/vision/videoDemuxer.h>
#include "codec/nalParser.h"
#include "codec/vcpAv1.h"
#include <errno.h>
#include <stdio.h>

struct oa::VideoDemuxer::MediaImpl {
	enum class NativeKind : oa::U8 { None, MpegTs };
	NativeKind kind = NativeKind::None;
	oa::U16 pmtPid = 0x1FFFU;
	oa::U16 videoPid = 0x1FFFU;
	oa::Vec<oa::U8> pes;
	oa::U64 pesPts = 0U;
	oa::U64 pesDts = 0U;
	bool pesKeyframe = false;
	bool live = false;
	bool seekable = false;
};

namespace
{

// forward declarations for MP4 box parsing
void parseMoovBox(const oa::U8* inData, oa::U64 inSize, oa::VideoDemuxer& outStream);
void parseTrakBox(const oa::U8* inData, oa::U64 inSize, oa::VideoDemuxer& outStream);
void parseMdiaBox(const oa::U8* inData, oa::U64 inSize, oa::VideoDemuxer& outStream);
void parseMinfBox(const oa::U8* inData, oa::U64 inSize, oa::VideoDemuxer& outStream);
void parseStblBox(const oa::U8* inData, oa::U64 inSize, oa::VideoDemuxer& outStream);
oa::Status parseMoofBox(const oa::U8* inData, oa::U64 inSize, oa::U64 inMoofOffset,
	oa::U64 inMoofEnd,
	oa::VideoDemuxer& outStream);

// Enough for more than 74 hours at 30 fps, while keeping hostile metadata from
// turning a tiny file into a multi-gigabyte host allocation.
constexpr oa::U32 kMaxMp4TableEntries = 8U * 1024U * 1024U;

bool mp4TableFits(oa::U64 inPayloadSize, oa::U64 inHeaderSize,
	oa::U32 inEntryCount, oa::U64 inEntryStride)
{
	return inEntryCount <= kMaxMp4TableEntries
		and inPayloadSize >= inHeaderSize
		and inEntryStride > 0U
		and static_cast<oa::U64>(inEntryCount)
			<= (inPayloadSize - inHeaderSize) / inEntryStride;
}

// Helper: read 32-bit big-endian
inline oa::U32 readU32BE(const oa::U8* inPtr) {
	return (static_cast<oa::U32>(inPtr[0]) << 24) |
	       (static_cast<oa::U32>(inPtr[1]) << 16) |
	       (static_cast<oa::U32>(inPtr[2]) << 8)  |
	       (static_cast<oa::U32>(inPtr[3]));
}

// Helper: read 64-bit big-endian
inline oa::U64 readU64BE(const oa::U8* inPtr) {
	return (static_cast<oa::U64>(inPtr[0]) << 56) |
	       (static_cast<oa::U64>(inPtr[1]) << 48) |
	       (static_cast<oa::U64>(inPtr[2]) << 40) |
	       (static_cast<oa::U64>(inPtr[3]) << 32) |
	       (static_cast<oa::U64>(inPtr[4]) << 24) |
	       (static_cast<oa::U64>(inPtr[5]) << 16) |
	       (static_cast<oa::U64>(inPtr[6]) << 8)  |
	       (static_cast<oa::U64>(inPtr[7]));
}

// Helper: read 16-bit big-endian
inline oa::U16 readU16BE(const oa::U8* inPtr)
{
	return (static_cast<oa::U16>(inPtr[0]) << 8) |
	       (static_cast<oa::U16>(inPtr[1]));
}

oa::VideoCodecProfile h264ProfileFromIdc(oa::U32 inProfileIdc)
{
	switch (inProfileIdc) {
	case 66U: return oa::VideoCodecProfile::H264Baseline;
	case 77U: return oa::VideoCodecProfile::H264Main;
	case 100U: return oa::VideoCodecProfile::H264High;
	case 244U: return oa::VideoCodecProfile::H264High444Predictive;
	default: return oa::VideoCodecProfile::Unspecified;
	}
}

oa::VideoCodecProfile h265ProfileFromIdc(oa::U32 inProfileIdc)
{
	switch (inProfileIdc) {
	case 1U: return oa::VideoCodecProfile::H265Main;
	case 2U: return oa::VideoCodecProfile::H265Main10;
	case 3U: return oa::VideoCodecProfile::H265MainStillPicture;
	case 4U: return oa::VideoCodecProfile::H265FormatRangeExtensions;
	case 9U: return oa::VideoCodecProfile::H265ScreenContentCodingExtensions;
	default: return oa::VideoCodecProfile::Unspecified;
	}
}

oa::VideoChromaSubsampling chromaFromIdc(oa::U32 inChromaIdc)
{
	switch (inChromaIdc) {
	case 0U: return oa::VideoChromaSubsampling::Monochrome;
	case 1U: return oa::VideoChromaSubsampling::Yuv420;
	case 2U: return oa::VideoChromaSubsampling::Yuv422;
	case 3U: return oa::VideoChromaSubsampling::Yuv444;
	default: return oa::VideoChromaSubsampling::Yuv420;
	}
}

oa::VideoBitDepth bitDepthFromValue(oa::U32 inBitDepth)
{
	switch (inBitDepth) {
	case 10U: return oa::VideoBitDepth::Bit10;
	case 12U: return oa::VideoBitDepth::Bit12;
	default: return oa::VideoBitDepth::Bit8;
	}
}

bool h264StdLevel(oa::U32 inLevelIdc, oa::U32& outLevel)
{
	switch (inLevelIdc) {
	case 10U: outLevel = STD_VIDEO_H264_LEVEL_IDC_1_0; return true;
	case 11U: outLevel = STD_VIDEO_H264_LEVEL_IDC_1_1; return true;
	case 12U: outLevel = STD_VIDEO_H264_LEVEL_IDC_1_2; return true;
	case 13U: outLevel = STD_VIDEO_H264_LEVEL_IDC_1_3; return true;
	case 20U: outLevel = STD_VIDEO_H264_LEVEL_IDC_2_0; return true;
	case 21U: outLevel = STD_VIDEO_H264_LEVEL_IDC_2_1; return true;
	case 22U: outLevel = STD_VIDEO_H264_LEVEL_IDC_2_2; return true;
	case 30U: outLevel = STD_VIDEO_H264_LEVEL_IDC_3_0; return true;
	case 31U: outLevel = STD_VIDEO_H264_LEVEL_IDC_3_1; return true;
	case 32U: outLevel = STD_VIDEO_H264_LEVEL_IDC_3_2; return true;
	case 40U: outLevel = STD_VIDEO_H264_LEVEL_IDC_4_0; return true;
	case 41U: outLevel = STD_VIDEO_H264_LEVEL_IDC_4_1; return true;
	case 42U: outLevel = STD_VIDEO_H264_LEVEL_IDC_4_2; return true;
	case 50U: outLevel = STD_VIDEO_H264_LEVEL_IDC_5_0; return true;
	case 51U: outLevel = STD_VIDEO_H264_LEVEL_IDC_5_1; return true;
	case 52U: outLevel = STD_VIDEO_H264_LEVEL_IDC_5_2; return true;
	case 60U: outLevel = STD_VIDEO_H264_LEVEL_IDC_6_0; return true;
	case 61U: outLevel = STD_VIDEO_H264_LEVEL_IDC_6_1; return true;
	case 62U: outLevel = STD_VIDEO_H264_LEVEL_IDC_6_2; return true;
	default: return false;
	}
}

bool h265StdLevel(oa::U32 inGeneralLevelIdc, oa::U32& outLevel)
{
	switch (inGeneralLevelIdc) {
	case 30U: outLevel = STD_VIDEO_H265_LEVEL_IDC_1_0; return true;
	case 60U: outLevel = STD_VIDEO_H265_LEVEL_IDC_2_0; return true;
	case 63U: outLevel = STD_VIDEO_H265_LEVEL_IDC_2_1; return true;
	case 90U: outLevel = STD_VIDEO_H265_LEVEL_IDC_3_0; return true;
	case 93U: outLevel = STD_VIDEO_H265_LEVEL_IDC_3_1; return true;
	case 120U: outLevel = STD_VIDEO_H265_LEVEL_IDC_4_0; return true;
	case 123U: outLevel = STD_VIDEO_H265_LEVEL_IDC_4_1; return true;
	case 150U: outLevel = STD_VIDEO_H265_LEVEL_IDC_5_0; return true;
	case 153U: outLevel = STD_VIDEO_H265_LEVEL_IDC_5_1; return true;
	case 156U: outLevel = STD_VIDEO_H265_LEVEL_IDC_5_2; return true;
	case 180U: outLevel = STD_VIDEO_H265_LEVEL_IDC_6_0; return true;
	case 183U: outLevel = STD_VIDEO_H265_LEVEL_IDC_6_1; return true;
	case 186U: outLevel = STD_VIDEO_H265_LEVEL_IDC_6_2; return true;
	default: return false;
	}
}

bool vp9StdLevel(oa::U32 inLevel, oa::U32& outLevel)
{
	switch (inLevel) {
	case 10U: outLevel = STD_VIDEO_VP9_LEVEL_1_0; return true;
	case 11U: outLevel = STD_VIDEO_VP9_LEVEL_1_1; return true;
	case 20U: outLevel = STD_VIDEO_VP9_LEVEL_2_0; return true;
	case 21U: outLevel = STD_VIDEO_VP9_LEVEL_2_1; return true;
	case 30U: outLevel = STD_VIDEO_VP9_LEVEL_3_0; return true;
	case 31U: outLevel = STD_VIDEO_VP9_LEVEL_3_1; return true;
	case 40U: outLevel = STD_VIDEO_VP9_LEVEL_4_0; return true;
	case 41U: outLevel = STD_VIDEO_VP9_LEVEL_4_1; return true;
	case 50U: outLevel = STD_VIDEO_VP9_LEVEL_5_0; return true;
	case 51U: outLevel = STD_VIDEO_VP9_LEVEL_5_1; return true;
	case 52U: outLevel = STD_VIDEO_VP9_LEVEL_5_2; return true;
	case 60U: outLevel = STD_VIDEO_VP9_LEVEL_6_0; return true;
	case 61U: outLevel = STD_VIDEO_VP9_LEVEL_6_1; return true;
	case 62U: outLevel = STD_VIDEO_VP9_LEVEL_6_2; return true;
	default: return false;
	}
}

struct EbmlElement {
	oa::U64 id = 0U;
	oa::U64 size = 0U;
	oa::U64 dataOffset = 0U;
	bool unknownSize = false;
};

bool readEbmlVint(::FILE* inFile, bool inKeepMarker,
	oa::U64& outValue, oa::U32& outLength, bool& outUnknown)
{
	const int firstInt = ::fgetc(inFile);
	if (firstInt == EOF) return false;
	const oa::U8 first = static_cast<oa::U8>(firstInt);
	oa::U8 marker = 0x80U;
	oa::U32 length = 1U;
	while (length <= 8U and (first & marker) == 0U) {
		marker >>= 1U;
		++length;
	}
	if (length > 8U or (inKeepMarker and length > 4U)) return false;
	oa::U64 value = inKeepMarker ? first : static_cast<oa::U64>(first & (marker - 1U));
	for (oa::U32 i = 1U; i < length; ++i) {
		const int byte = ::fgetc(inFile);
		if (byte == EOF) return false;
		value = (value << 8U) | static_cast<oa::U8>(byte);
	}
	const oa::U64 unknown = length == 8U
		? 0x00FFFFFFFFFFFFFFULL
		: ((1ULL << (7U * length)) - 1ULL);
	outValue = value;
	outLength = length;
	outUnknown = not inKeepMarker and value == unknown;
	return true;
}

bool readEbmlElement(::FILE* inFile, oa::U64 inLimit, EbmlElement& out)
{
	const off_t start = ::ftello(inFile);
	if (start < 0 or static_cast<oa::U64>(start) >= inLimit) return false;
	oa::U32 idLength = 0U;
	oa::U32 sizeLength = 0U;
	bool ignored = false;
	if (not readEbmlVint(inFile, true, out.id, idLength, ignored)
		or not readEbmlVint(inFile, false, out.size, sizeLength, out.unknownSize)) return false;
	const off_t data = ::ftello(inFile);
	if (data < 0) return false;
	out.dataOffset = static_cast<oa::U64>(data);
	if (out.unknownSize) out.size = inLimit - out.dataOffset;
	return out.dataOffset <= inLimit and out.size <= inLimit - out.dataOffset;
}

oa::U64 readEbmlUnsigned(::FILE* inFile, oa::U64 inSize)
{
	if (inSize == 0U or inSize > 8U) return 0U;
	oa::U64 value = 0U;
	for (oa::U64 i = 0U; i < inSize; ++i) {
		const int byte = ::fgetc(inFile);
		if (byte == EOF) return 0U;
		value = (value << 8U) | static_cast<oa::U8>(byte);
	}
	return value;
}

oa::String readEbmlString(::FILE* inFile, oa::U64 inSize)
{
	if (inSize > 1024U) return {};
	oa::Vec<char> bytes(static_cast<oa::Usize>(inSize + 1U), '\0');
	if (inSize > 0U and ::fread(bytes.data(), 1U, static_cast<oa::Usize>(inSize), inFile)
		!= inSize) return {};
	return oa::String(bytes.data());
}

bool parseAvcDecoderConfig(const oa::U8* inData, oa::U64 inSize,
	oa::VideoDemuxer::AvcConfig& out)
{
	if (inData == nullptr or inSize < 7U or inData[0] != 1U) return false;
	out = {};
	out.profile.codec = oa::VideoCodec::H264;
	out.profile.standardProfile = h264ProfileFromIdc(inData[1]);
	out.profile.hasLevel = h264StdLevel(inData[3], out.profile.level);
	out.lengthSize = static_cast<oa::U8>((inData[4] & 0x03U) + 1U);
	oa::U64 p = 6U;
	const oa::U8 startCode[4] = {0U, 0U, 0U, 1U};
	const oa::U8 spsCount = inData[5] & 0x1FU;
	for (oa::U8 i = 0U; i < spsCount; ++i) {
		if (p + 2U > inSize) return false;
		const oa::U16 size = readU16BE(inData + p); p += 2U;
		if (p + size > inSize) return false;
		if (i == 0U) {
			oa::H264SpsData sps = {};
			if (oa::NalParser::parseSPS(inData + p, size, sps)) {
				out.profile.standardProfile = h264ProfileFromIdc(sps.profileIdc);
				out.profile.chromaSubsampling = chromaFromIdc(sps.chromaFormatIdc);
				out.profile.lumaBitDepth = bitDepthFromValue(sps.bitDepthLumaMinus8 + 8U);
				out.profile.chromaBitDepth = bitDepthFromValue(sps.bitDepthChromaMinus8 + 8U);
				out.profile.h264PictureLayout = sps.frameMbsOnly ? oa::VideoH264PictureLayout::Progressive
																 : oa::VideoH264PictureLayout::InterlacedInterleavedLines;
				out.profile.hasLevel = h264StdLevel(sps.levelIdc, out.profile.level);
			}
		}
		for (oa::U8 byte : startCode) out.spsAnnexB.pushBack(byte);
		for (oa::U16 k = 0U; k < size; ++k) out.spsAnnexB.pushBack(inData[p + k]);
		p += size;
	}
	if (p >= inSize) return false;
	const oa::U8 ppsCount = inData[p++];
	for (oa::U8 i = 0U; i < ppsCount; ++i) {
		if (p + 2U > inSize) return false;
		const oa::U16 size = readU16BE(inData + p); p += 2U;
		if (p + size > inSize) return false;
		for (oa::U8 byte : startCode) out.ppsAnnexB.pushBack(byte);
		for (oa::U16 k = 0U; k < size; ++k) out.ppsAnnexB.pushBack(inData[p + k]);
		p += size;
	}
	out.valid = not out.spsAnnexB.empty() and not out.ppsAnnexB.empty();
	return out.valid;
}

bool parseHevcDecoderConfig(const oa::U8* inData, oa::U64 inSize, oa::VideoDemuxer::HvcConfig& out)
{
	if (inData == nullptr or inSize < 23U or inData[0] != 1U) return false;
	out = {};
	out.profile.codec = oa::VideoCodec::H265;
	out.profile.standardProfile = h265ProfileFromIdc(inData[1] & 0x1FU);
	out.profile.highTier = (inData[1] & 0x20U) != 0U;
	out.profile.hasLevel = h265StdLevel(inData[12], out.profile.level);
	out.profile.chromaSubsampling = chromaFromIdc(inData[16] & 0x03U);
	out.profile.lumaBitDepth = bitDepthFromValue((inData[17] & 0x07U) + 8U);
	out.profile.chromaBitDepth = bitDepthFromValue((inData[18] & 0x07U) + 8U);
	out.lengthSize = static_cast<oa::U8>((inData[21] & 0x03U) + 1U);

	const oa::U8 startCode[4] = {0U, 0U, 0U, 1U};
	const oa::U8 numArrays = inData[22];
	oa::U64 p = 23U;
	for (oa::U8 arrayIndex = 0U; arrayIndex < numArrays and p + 3U <= inSize; ++arrayIndex) {
		const oa::U8 nalType = inData[p] & 0x3FU;
		++p;
		const oa::U16 numNalus = readU16BE(inData + p);
		p += 2U;
		oa::Vec<oa::U8>* target = nullptr;
		if (nalType == 32U)
			target = &out.vpsAnnexB;
		else if (nalType == 33U)
			target = &out.spsAnnexB;
		else if (nalType == 34U)
			target = &out.ppsAnnexB;
		for (oa::U16 nalIndex = 0U; nalIndex < numNalus and p + 2U <= inSize; ++nalIndex) {
			const oa::U16 nalSize = readU16BE(inData + p);
			p += 2U;
			if (p + nalSize > inSize) return false;
			if (nalType == 32U and nalIndex == 0U) {
				oa::H265VpsData vps = {};
				if (oa::NalParser::parseH265Vps(inData + p, nalSize, vps)) {
					out.profile.standardProfile = h265ProfileFromIdc(vps.generalProfileIdc);
					out.profile.highTier = vps.generalTierFlag;
					out.profile.hasLevel = h265StdLevel(vps.generalLevelIdc, out.profile.level);
				}
			} else if (nalType == 33U and nalIndex == 0U) {
				oa::H265SpsData sps = {};
				if (oa::NalParser::parseH265Sps(inData + p, nalSize, sps)) {
					out.profile.chromaSubsampling = chromaFromIdc(sps.chromaFormatIdc);
					out.profile.lumaBitDepth = bitDepthFromValue(sps.bitDepthLumaMinus8 + 8U);
					out.profile.chromaBitDepth = bitDepthFromValue(sps.bitDepthChromaMinus8 + 8U);
				}
			}
			if (target != nullptr) {
				for (oa::U8 byte : startCode) target->pushBack(byte);
				for (oa::U16 byteIndex = 0U; byteIndex < nalSize; ++byteIndex) {
					target->pushBack(inData[p + byteIndex]);
				}
			}
			p += nalSize;
		}
	}
	out.valid = not out.vpsAnnexB.empty() and not out.spsAnnexB.empty() and not out.ppsAnnexB.empty();
	return out.valid;
}

bool parseAv1DecoderConfig(const oa::U8* inData, oa::U64 inSize, oa::VideoDemuxer::Av1Config& out)
{
	if (inData == nullptr or inSize < 4U or (inData[0] & 0x80U) == 0U or (inData[0] & 0x7FU) != 1U) {
		return false;
	}
	out = {};
	out.profile.codec = oa::VideoCodec::AV1;
	switch ((inData[1] >> 5U) & 0x07U) {
	case 0U: out.profile.standardProfile = oa::VideoCodecProfile::Av1Main; break;
	case 1U: out.profile.standardProfile = oa::VideoCodecProfile::Av1High; break;
	case 2U: out.profile.standardProfile = oa::VideoCodecProfile::Av1Professional; break;
	default: out.profile.standardProfile = oa::VideoCodecProfile::Unspecified; break;
	}
	out.profile.level = inData[1] & 0x1FU;
	out.profile.hasLevel = true;
	out.profile.highTier = (inData[2] & 0x80U) != 0U;
	const bool highBitDepth = (inData[2] & 0x40U) != 0U;
	const bool twelveBit = (inData[2] & 0x20U) != 0U;
	const bool monochrome = (inData[2] & 0x10U) != 0U;
	const bool subsamplingX = (inData[2] & 0x08U) != 0U;
	const bool subsamplingY = (inData[2] & 0x04U) != 0U;
	const oa::U32 bitDepth = not highBitDepth ? 8U : (twelveBit ? 12U : 10U);
	out.profile.lumaBitDepth = bitDepthFromValue(bitDepth);
	out.profile.chromaBitDepth = bitDepthFromValue(bitDepth);
	if (monochrome) {
		out.profile.chromaSubsampling = oa::VideoChromaSubsampling::Monochrome;
	} else if (subsamplingX and subsamplingY) {
		out.profile.chromaSubsampling = oa::VideoChromaSubsampling::Yuv420;
	} else if (subsamplingX) {
		out.profile.chromaSubsampling = oa::VideoChromaSubsampling::Yuv422;
	} else {
		out.profile.chromaSubsampling = oa::VideoChromaSubsampling::Yuv444;
	}
	for (oa::U64 byteIndex = 4U; byteIndex < inSize; ++byteIndex) {
		out.configObus.pushBack(inData[byteIndex]);
	}
	out.valid = not out.configObus.empty();
	if (out.valid) {
		oa::VcpAv1 parser;
		oa::Vec<oa::Av1PictureDesc> pictures;
		const oa::Status parseStatus =
			parser.parseAccessUnitPictures(oa::Span<const oa::U8>(out.configObus.data(), out.configObus.size()), pictures);
		if (parseStatus.isOk() and parser.hasSequenceHeader()) {
			const oa::Av1SequenceHeaderInfo& sequence = parser.getSequenceHeader();
			switch (sequence.seqProfile) {
			case STD_VIDEO_AV1_PROFILE_MAIN: out.profile.standardProfile = oa::VideoCodecProfile::Av1Main; break;
			case STD_VIDEO_AV1_PROFILE_HIGH: out.profile.standardProfile = oa::VideoCodecProfile::Av1High; break;
			case STD_VIDEO_AV1_PROFILE_PROFESSIONAL:
				out.profile.standardProfile = oa::VideoCodecProfile::Av1Professional;
				break;
			default: out.profile.standardProfile = oa::VideoCodecProfile::Unspecified; break;
			}
			out.profile.lumaBitDepth = bitDepthFromValue(sequence.colorConfig.BitDepth);
			out.profile.chromaBitDepth = out.profile.lumaBitDepth;
			if (sequence.colorConfig.flags.mono_chrome != 0U) {
				out.profile.chromaSubsampling = oa::VideoChromaSubsampling::Monochrome;
			} else if (sequence.colorConfig.subsampling_x != 0U and sequence.colorConfig.subsampling_y != 0U) {
				out.profile.chromaSubsampling = oa::VideoChromaSubsampling::Yuv420;
			} else if (sequence.colorConfig.subsampling_x != 0U) {
				out.profile.chromaSubsampling = oa::VideoChromaSubsampling::Yuv422;
			} else {
				out.profile.chromaSubsampling = oa::VideoChromaSubsampling::Yuv444;
			}
			out.profile.av1FilmGrain = sequence.filmGrainParamsPresent;
		}
	}
	return out.valid;
}

bool parseVp9DecoderConfig(const oa::U8* inData, oa::U64 inSize, oa::VideoDemuxer::Vp9Config& out)
{
	// vpcC is a FullBox. The VPCodecConfigurationRecord starts after its
	// version/flags and carries profile, level, then packed depth/chroma.
	if (inData == nullptr or inSize < 7U) return false;
	out = {};
	out.profile.codec = oa::VideoCodec::VP9;
	switch (inData[4]) {
	case 0U: out.profile.standardProfile = oa::VideoCodecProfile::Vp9Profile0; break;
	case 1U: out.profile.standardProfile = oa::VideoCodecProfile::Vp9Profile1; break;
	case 2U: out.profile.standardProfile = oa::VideoCodecProfile::Vp9Profile2; break;
	case 3U: out.profile.standardProfile = oa::VideoCodecProfile::Vp9Profile3; break;
	default: out.profile.standardProfile = oa::VideoCodecProfile::Unspecified; break;
	}
	out.profile.hasLevel = vp9StdLevel(inData[5], out.profile.level);
	out.profile.lumaBitDepth = bitDepthFromValue(inData[6] >> 4U);
	out.profile.chromaBitDepth = out.profile.lumaBitDepth;
	switch (inData[6] & 0x07U) {
	case 3U: out.profile.chromaSubsampling = oa::VideoChromaSubsampling::Yuv422; break;
	case 4U: out.profile.chromaSubsampling = oa::VideoChromaSubsampling::Yuv444; break;
	default: out.profile.chromaSubsampling = oa::VideoChromaSubsampling::Yuv420; break;
	}
	out.valid = out.profile.standardProfile != oa::VideoCodecProfile::Unspecified;
	return out.valid;
}

struct MatroskaTrack {
	oa::U64 number = 0U;
	oa::U64 type = 0U;
	oa::String codec;
	oa::Vec<oa::U8> codecPrivate;
	oa::U32 width = 0U;
	oa::U32 height = 0U;
	oa::U64 defaultDurationNs = 0U;
};

void parseMatroskaVideo(::FILE* inFile, oa::U64 inEnd, MatroskaTrack& out)
{
	while (static_cast<oa::U64>(::ftello(inFile)) < inEnd) {
		EbmlElement element;
		if (not readEbmlElement(inFile, inEnd, element)) break;
		if (element.id == 0xB0U) out.width = static_cast<oa::U32>(readEbmlUnsigned(inFile, element.size));
		else if (element.id == 0xBAU) out.height = static_cast<oa::U32>(readEbmlUnsigned(inFile, element.size));
		::fseeko(inFile, static_cast<off_t>(element.dataOffset + element.size), SEEK_SET);
	}
}

MatroskaTrack parseMatroskaTrack(::FILE* inFile, oa::U64 inEnd)
{
	MatroskaTrack track;
	while (static_cast<oa::U64>(::ftello(inFile)) < inEnd) {
		EbmlElement element;
		if (not readEbmlElement(inFile, inEnd, element)) break;
		if (element.id == 0xD7U) track.number = readEbmlUnsigned(inFile, element.size);
		else if (element.id == 0x83U) track.type = readEbmlUnsigned(inFile, element.size);
		else if (element.id == 0x86U) track.codec = readEbmlString(inFile, element.size);
		else if (element.id == 0x63A2U and element.size <= 16U * 1024U * 1024U) {
			track.codecPrivate.resize(static_cast<oa::Usize>(element.size));
			if (element.size > 0U) {
				(void)::fread(track.codecPrivate.data(), 1U, track.codecPrivate.size(), inFile);
			}
		} else if (element.id == 0x23E383U) {
			track.defaultDurationNs = readEbmlUnsigned(inFile, element.size);
		} else if (element.id == 0xE0U) {
			parseMatroskaVideo(inFile, element.dataOffset + element.size, track);
		}
		::fseeko(inFile, static_cast<off_t>(element.dataOffset + element.size), SEEK_SET);
	}
	return track;
}

oa::Status parseMatroskaCluster(::FILE* inFile, oa::U64 inEnd,
	oa::U64 inVideoTrack, oa::U64 inTimecodeScale, oa::U64 inDefaultDurationNs,
	oa::VideoDemuxer& out)
{
	oa::U64 clusterTimecode = 0U;
	while (static_cast<oa::U64>(::ftello(inFile)) < inEnd) {
		EbmlElement element;
		if (not readEbmlElement(inFile, inEnd, element)) break;
		if (element.id == 0xE7U) {
			clusterTimecode = readEbmlUnsigned(inFile, element.size);
		} else if (element.id == 0xA3U and element.size >= 4U) {
			const off_t blockStart = ::ftello(inFile);
			oa::U64 track = 0U;
			oa::U32 trackBytes = 0U;
			bool unknown = false;
			if (blockStart >= 0 and readEbmlVint(inFile, false, track, trackBytes, unknown)) {
				oa::U8 header[3] = {};
				if (::fread(header, 1U, sizeof(header), inFile) == sizeof(header)) {
					const oa::I16 relative = static_cast<oa::I16>(
						(static_cast<oa::U16>(header[0]) << 8U) | header[1]);
					const oa::U8 flags = header[2];
					const oa::U64 headerSize = trackBytes + sizeof(header);
					if (track == inVideoTrack and (flags & 0x06U) == 0U
						and element.size > headerSize) {
						const oa::I64 signedTimestamp = static_cast<oa::I64>(clusterTimecode) + relative;
						oa::VideoDemuxer::Sample sample;
						sample.offset = element.dataOffset + headerSize;
						sample.size = static_cast<oa::U32>(element.size - headerSize);
						sample.dts = static_cast<oa::U64>(oa::max<oa::I64>(0, signedTimestamp));
						sample.duration = inDefaultDurationNs > 0U and inTimecodeScale > 0U
							? oa::max<oa::U64>(1U, inDefaultDurationNs / inTimecodeScale) : 1U;
						sample.isKeyframe = (flags & 0x80U) != 0U;
						out.samples_.pushBack(sample);
					}
				}
			}
		}
		::fseeko(inFile, static_cast<off_t>(element.dataOffset + element.size), SEEK_SET);
	}
	return oa::Status::ok();
}

oa::Status parseMatroskaFile(::FILE* inFile, oa::U64 inFileSize,
	oa::StringView inPath, oa::VideoDemuxer& out)
{
	::fseeko(inFile, 0, SEEK_SET);
	EbmlElement ebml;
	if (not readEbmlElement(inFile, inFileSize, ebml) or ebml.id != 0x1A45DFA3U) {
		return oa::Status::error(oa::StatusCode::DataLoss, "Missing Matroska EBML header");
	}
	::fseeko(inFile, static_cast<off_t>(ebml.dataOffset + ebml.size), SEEK_SET);
	EbmlElement segment;
	if (not readEbmlElement(inFile, inFileSize, segment) or segment.id != 0x18538067U) {
		return oa::Status::error(oa::StatusCode::DataLoss, "Missing Matroska segment");
	}
	const oa::U64 segmentEnd = segment.unknownSize ? inFileSize : segment.dataOffset + segment.size;
	oa::U64 timecodeScale = 1'000'000U;
	MatroskaTrack selected;
	oa::U32 trackCount = 0U;
	// Matroska normally writes Tracks before cluster, but EBML does not make
	// that ordering a demuxer invariant. index cluster payload ranges first and
	// parse them only after the selected video track is known.
	oa::Vec<oa::U64> clusters;
	while (static_cast<oa::U64>(::ftello(inFile)) < segmentEnd) {
		EbmlElement element;
		if (not readEbmlElement(inFile, segmentEnd, element)) break;
		if (element.id == 0x1549A966U) { // Info
			const oa::U64 end = element.dataOffset + element.size;
			while (static_cast<oa::U64>(::ftello(inFile)) < end) {
				EbmlElement info;
				if (not readEbmlElement(inFile, end, info)) break;
				if (info.id == 0x2AD7B1U) timecodeScale = readEbmlUnsigned(inFile, info.size);
				::fseeko(inFile, static_cast<off_t>(info.dataOffset + info.size), SEEK_SET);
			}
		} else if (element.id == 0x1654AE6BU) { // Tracks
			const oa::U64 end = element.dataOffset + element.size;
			while (static_cast<oa::U64>(::ftello(inFile)) < end) {
				EbmlElement entry;
				if (not readEbmlElement(inFile, end, entry)) break;
				if (entry.id == 0xAEU) {
					++trackCount;
					MatroskaTrack track = parseMatroskaTrack(inFile, entry.dataOffset + entry.size);
					if (selected.number == 0U and track.type == 1U) selected = oa::move(track);
				}
				::fseeko(inFile, static_cast<off_t>(entry.dataOffset + entry.size), SEEK_SET);
			}
		} else if (element.id == 0x1F43B675U) { // cluster
			clusters.pushBack(element.dataOffset);
			clusters.pushBack(element.dataOffset + element.size);
		}
		::fseeko(inFile, static_cast<off_t>(element.dataOffset + element.size), SEEK_SET);
	}
	if (selected.number == 0U) return oa::Status::error(oa::StatusCode::NotFound,
		"Matroska container has no video track");
	for (oa::Usize i = 0U; i + 1U < clusters.size(); i += 2U) {
		if (::fseeko(inFile, static_cast<off_t>(clusters[i]), SEEK_SET) != 0) {
			return oa::Status::error(oa::StatusCode::DataLoss, "Cannot seek Matroska cluster");
		}
		OA_RETURN_IF_ERROR(parseMatroskaCluster(inFile, clusters[i + 1U],
			selected.number, timecodeScale, selected.defaultDurationNs, out));
	}
	const oa::StringView codec(selected.codec.data(), selected.codec.size());
	if (codec == "V_MPEG4/ISO/AVC") {
		out.info_.codec = oa::VideoCodec::H264;
		if (not parseAvcDecoderConfig(selected.codecPrivate.data(),
			selected.codecPrivate.size(), out.avc_)) {
			return oa::Status::error(oa::StatusCode::DataLoss, "Matroska AVC track has invalid codecPrivate");
		}
	} else if (codec == "V_MPEGH/ISO/HEVC") {
		out.info_.codec = oa::VideoCodec::H265;
		if (not parseHevcDecoderConfig(selected.codecPrivate.data(), selected.codecPrivate.size(), out.hvc_)) {
			return oa::Status::error(oa::StatusCode::DataLoss, "Matroska HEVC track has invalid codecPrivate");
		}
	} else if (codec == "V_AV1") {
		out.info_.codec = oa::VideoCodec::AV1;
		if (not parseAv1DecoderConfig(selected.codecPrivate.data(), selected.codecPrivate.size(), out.av1_)) {
			return oa::Status::error(oa::StatusCode::DataLoss, "Matroska AV1 track has invalid codecPrivate");
		}
	} else if (codec == "V_VP9") {
		out.info_.codec = oa::VideoCodec::VP9;
	} else return oa::Status::error(oa::StatusCode::Unimplemented,
		"Matroska video codec is not supported by OA vulkan Video");
	const oa::StringView path(inPath.data(), inPath.size());
	const bool isWebM = path.size() >= 5U and path.subStr(path.size() - 5U) == ".webm";
	out.info_.kind = isWebM ? oa::VideoContainerKind::WebM : oa::VideoContainerKind::Matroska;
	out.info_.width = selected.width;
	out.info_.height = selected.height;
	out.info_.timebaseNum = timecodeScale;
	out.info_.timebaseDen = 1'000'000'000ULL;
	out.info_.trackCount = trackCount;
	if (out.samples_.empty())
		return oa::Status::error(oa::StatusCode::DataLoss, "Matroska video track contains no supported unlaced blocks");
	return oa::Status::ok();
}

struct TsPayload {
	oa::U16 pid = 0x1FFFU;
	const oa::U8* data = nullptr;
	oa::Usize size = 0U;
	bool start = false;
};

bool parseTsPayload(const oa::U8* inPacket, TsPayload& out)
{
	if (inPacket[0] != 0x47U or (inPacket[1] & 0x80U) != 0U) return false;
	out.pid = static_cast<oa::U16>(((inPacket[1] & 0x1FU) << 8U) | inPacket[2]);
	out.start = (inPacket[1] & 0x40U) != 0U;
	const oa::U8 control = static_cast<oa::U8>((inPacket[3] >> 4U) & 0x03U);
	if (control == 0U or control == 2U) return true;
	oa::Usize offset = 4U;
	if (control == 3U) {
		offset += 1U + inPacket[4];
		if (offset > 188U) return false;
	}
	out.data = inPacket + offset;
	out.size = 188U - offset;
	return true;
}

const oa::U8* tsPsiSection(const TsPayload& inPayload, oa::Usize& outSize)
{
	if (inPayload.data == nullptr or inPayload.size == 0U) return nullptr;
	oa::Usize offset = 0U;
	if (inPayload.start) {
		offset = 1U + inPayload.data[0];
		if (offset >= inPayload.size) return nullptr;
	}
	outSize = inPayload.size - offset;
	return inPayload.data + offset;
}

bool parseTsPat(const TsPayload& inPayload, oa::U16& outPmtPid)
{
	oa::Usize size = 0U;
	const oa::U8* section = tsPsiSection(inPayload, size);
	if (section == nullptr or size < 12U or section[0] != 0x00U) return false;
	const oa::U16 sectionLength = static_cast<oa::U16>(((section[1] & 0x0FU) << 8U) | section[2]);
	if (sectionLength + 3U > size or sectionLength < 9U) return false;
	const oa::Usize end = 3U + sectionLength - 4U; // exclude CRC
	for (oa::Usize p = 8U; p + 4U <= end; p += 4U) {
		const oa::U16 program = readU16BE(section + p);
		if (program != 0U) {
			outPmtPid = static_cast<oa::U16>(((section[p + 2U] & 0x1FU) << 8U) | section[p + 3U]);
			return true;
		}
	}
	return false;
}

bool parseTsPmt(const TsPayload& inPayload, oa::U16& outVideoPid, oa::VideoCodec& outCodec)
{
	oa::Usize size = 0U;
	const oa::U8* section = tsPsiSection(inPayload, size);
	if (section == nullptr or size < 16U or section[0] != 0x02U) return false;
	const oa::U16 sectionLength = static_cast<oa::U16>(((section[1] & 0x0FU) << 8U) | section[2]);
	if (sectionLength + 3U > size or sectionLength < 13U) return false;
	const oa::U16 programInfoLength = static_cast<oa::U16>(((section[10] & 0x0FU) << 8U) | section[11]);
	const oa::Usize end = 3U + sectionLength - 4U;
	for (oa::Usize p = 12U + programInfoLength; p + 5U <= end;) {
		const oa::U8 streamType = section[p];
		const oa::U16 pid = static_cast<oa::U16>(((section[p + 1U] & 0x1FU) << 8U) | section[p + 2U]);
		const oa::U16 infoLength = static_cast<oa::U16>(((section[p + 3U] & 0x0FU) << 8U) | section[p + 4U]);
		if (streamType == 0x1BU or streamType == 0x24U) {
			outVideoPid = pid;
			outCodec = streamType == 0x1BU ? oa::VideoCodec::H264 : oa::VideoCodec::H265;
			return true;
		}
		p += 5U + infoLength;
	}
	return false;
}

oa::U64 parsePesTimestamp(const oa::U8* inData)
{
	return (static_cast<oa::U64>((inData[0] >> 1U) & 0x07U) << 30U) | (static_cast<oa::U64>(inData[1]) << 22U) |
		   (static_cast<oa::U64>((inData[2] >> 1U) & 0x7FU) << 15U) | (static_cast<oa::U64>(inData[3]) << 7U) |
		   static_cast<oa::U64>(inData[4] >> 1U);
}

bool h264AccessUnitIsKeyframe(oa::Span<const oa::U8> inData)
{
	for (oa::Usize i = 0U; i + 4U < inData.size(); ++i) {
		if (inData[i] == 0U and inData[i + 1U] == 0U and
			((inData[i + 2U] == 1U) or (inData[i + 2U] == 0U and inData[i + 3U] == 1U))) {
			const oa::Usize header = i + (inData[i + 2U] == 1U ? 3U : 4U);
			if (header < inData.size() and (inData[header] & 0x1FU) == 5U) return true;
		}
	}
	return false;
}

void parseH264AccessUnitGeometry(oa::Span<const oa::U8> inData, oa::VideoContainerInfo& out)
{
	for (oa::Usize i = 0U; i + 5U < inData.size(); ++i) {
		if (inData[i] != 0U or inData[i + 1U] != 0U) continue;
		oa::Usize header = 0U;
		if (inData[i + 2U] == 1U)
			header = i + 3U;
		else if (inData[i + 2U] == 0U and inData[i + 3U] == 1U)
			header = i + 4U;
		if (header == 0U or header >= inData.size() or (inData[header] & 0x1FU) != 7U) continue;
		oa::Usize end = inData.size();
		for (oa::Usize p = header + 1U; p + 3U < inData.size(); ++p) {
			if (inData[p] == 0U and inData[p + 1U] == 0U and
				(inData[p + 2U] == 1U or (inData[p + 2U] == 0U and inData[p + 3U] == 1U))) {
				end = p;
				break;
			}
		}
		oa::H264SpsData sps = {};
		if (not oa::NalParser::parseSPS(inData.data() + header, end - header, sps)) return;
		const oa::U32 frameFactor = sps.frameMbsOnly ? 1U : 2U;
		oa::U32 width = sps.picWidthInMbs * 16U;
		oa::U32 height = sps.picHeightInMbs * 16U * frameFactor;
		const oa::U32 cropUnitX = sps.chromaFormatIdc == 0U or sps.chromaFormatIdc == 3U ? 1U : 2U;
		const oa::U32 chromaHeight = sps.chromaFormatIdc == 1U ? 2U : 1U;
		const oa::U32 cropUnitY = sps.chromaFormatIdc == 0U ? frameFactor : chromaHeight * frameFactor;
		const oa::U64 cropX = static_cast<oa::U64>(sps.frameCropLeftOffset + sps.frameCropRightOffset) * cropUnitX;
		const oa::U64 cropY = static_cast<oa::U64>(sps.frameCropTopOffset + sps.frameCropBottomOffset) * cropUnitY;
		if (cropX < width) width -= static_cast<oa::U32>(cropX);
		if (cropY < height) height -= static_cast<oa::U32>(cropY);
		out.width = width;
		out.height = height;
		return;
	}
}

oa::Status readMpegTsPes(::FILE* inFile, oa::VideoDemuxer::MediaImpl& inMedia, oa::VideoCodec inCodec, oa::VideoPacket& out)
{
	oa::U8 packet[188] = {};
	while (::fread(packet, 1U, sizeof(packet), inFile) == sizeof(packet)) {
		TsPayload payload;
		if (not parseTsPayload(packet, payload) or payload.pid != inMedia.videoPid or payload.data == nullptr) continue;
		if (payload.start and not inMedia.pes.empty()) {
			::fseeko(inFile, -static_cast<off_t>(sizeof(packet)), SEEK_CUR);
			break;
		}
		const oa::U8* data = payload.data;
		oa::Usize size = payload.size;
		if (payload.start) {
			if (size < 9U or data[0] != 0U or data[1] != 0U or data[2] != 1U) continue;
			const oa::U8 timestampFlags = static_cast<oa::U8>((data[7] >> 6U) & 0x03U);
			const oa::U8 headerLength = data[8];
			if (9U + headerLength > size) continue;
			if ((timestampFlags & 0x02U) != 0U and headerLength >= 5U) {
				inMedia.pesPts = parsePesTimestamp(data + 9U);
				inMedia.pesDts = inMedia.pesPts;
			}
			if (timestampFlags == 0x03U and headerLength >= 10U) {
				inMedia.pesDts = parsePesTimestamp(data + 14U);
			}
			data += 9U + headerLength;
			size -= 9U + headerLength;
		}
		const oa::Usize old = inMedia.pes.size();
		inMedia.pes.resize(old + size);
		if (size > 0U) oa::memcpy(inMedia.pes.data() + old, data, size);
	}
	if (inMedia.pes.empty()) return oa::Status::error(oa::StatusCode::OutOfRange, "End of MPEG-TS stream");
	out.data = oa::move(inMedia.pes);
	out.presentationTimestamp = inMedia.pesPts;
	out.decodeTimestamp = inMedia.pesDts;
	out.isKeyframe = inCodec == oa::VideoCodec::H264
					 ? h264AccessUnitIsKeyframe(oa::Span<const oa::U8>(out.data.data(), out.data.size()))
					 : false;
	out.trackIndex = inMedia.videoPid;
	inMedia.pes.clear();
	inMedia.pesPts = 0U;
	inMedia.pesDts = 0U;
	return oa::Status::ok();
}

oa::Status initMpegTs(::FILE* inFile, oa::U64 inFileSize, oa::VideoDemuxer::MediaImpl& outMedia, oa::VideoContainerInfo& outInfo)
{
	::fseeko(inFile, 0, SEEK_SET);
	oa::U8 packet[188] = {};
	oa::VideoCodec codec = oa::VideoCodec::H264;
	const oa::U64 scanPackets = oa::min<oa::U64>(inFileSize / sizeof(packet), 8192U);
	for (oa::U64 i = 0U; i < scanPackets; ++i) {
		if (::fread(packet, 1U, sizeof(packet), inFile) != sizeof(packet)) break;
		TsPayload payload;
		if (not parseTsPayload(packet, payload)) continue;
		if (payload.pid == 0U)
			(void)parseTsPat(payload, outMedia.pmtPid);
		else if (payload.pid == outMedia.pmtPid and parseTsPmt(payload, outMedia.videoPid, codec))
			break;
	}
	if (outMedia.videoPid == 0x1FFFU)
		return oa::Status::error(oa::StatusCode::DataLoss,
							   "MPEG-TS PAT/PMT contains no supported H.264/H.265 video stream");
	outMedia.kind = oa::VideoDemuxer::MediaImpl::NativeKind::MpegTs;
	outMedia.live = false;
	outMedia.seekable = true;
	outInfo.kind = oa::VideoContainerKind::MpegTs;
	outInfo.codec = codec;
	outInfo.timebaseNum = 1U;
	outInfo.timebaseDen = 90'000U;
	outInfo.trackCount = 1U;
	::fseeko(inFile, 0, SEEK_SET);
	oa::VideoPacket first;
	if (readMpegTsPes(inFile, outMedia, codec, first).isOk()) {
		if (codec == oa::VideoCodec::H264) {
			parseH264AccessUnitGeometry(
				oa::Span<const oa::U8>(first.data.data(), first.data.size()), outInfo);
		}
	}
	outMedia.pes.clear();
	outMedia.pesPts = 0U;
	outMedia.pesDts = 0U;
	::fseeko(inFile, 0, SEEK_SET);
	return oa::Status::ok();
}

// MP4 box header
struct BoxHeader
{
	oa::U64 size = 0;
	oa::U32 type = 0;
};

// parse box header (returns true if extended size)
bool parseBoxHeader(const oa::U8* inData, oa::U64 inOffset, oa::U64 inDataSize, BoxHeader& outHeader)
{
	if (inOffset + 8 > inDataSize) {
		return false;
	}
	
	outHeader.size = readU32BE(inData + inOffset);
	outHeader.type = readU32BE(inData + inOffset + 4);
	
	// Extended size (size == 1 means 64-bit size follows)
	if (outHeader.size == 1) {
		if (inOffset + 16 > inDataSize) {
			return false;
		}
		outHeader.size = readU64BE(inData + inOffset + 8);
	}
	
	return true;
}

bool readTopLevelBoxHeader(
	::FILE* inFile,
	oa::U64 inOffset,
	oa::U64 inFileSize,
	BoxHeader& outHeader,
	oa::U64& outHeaderSize)
{
	if (inFile == nullptr || inOffset + 8U > inFileSize
		|| ::fseeko(inFile, static_cast<off_t>(inOffset), SEEK_SET) != 0) return false;
	oa::U8 bytes[16] = {};
	if (::fread(bytes, 1U, 8U, inFile) != 8U) return false;
	outHeader.size = readU32BE(bytes);
	outHeader.type = readU32BE(bytes + 4U);
	outHeaderSize = 8U;
	if (outHeader.size == 1U) {
		if (inOffset + 16U > inFileSize || ::fread(bytes + 8U, 1U, 8U, inFile) != 8U) {
			return false;
		}
		outHeader.size = readU64BE(bytes + 8U);
		outHeaderSize = 16U;
	} else if (outHeader.size == 0U) {
		outHeader.size = inFileSize - inOffset;
	}
	return outHeader.size >= outHeaderSize && inOffset + outHeader.size <= inFileSize;
}

bool isVideoMdia(const oa::U8* inData, oa::U64 inSize)
{
	oa::U64 offset = 0U;
	while (offset + 8U <= inSize) {
		const oa::U32 size = readU32BE(inData + offset);
		const oa::U32 type = readU32BE(inData + offset + 4U);
		if (size < 8U or offset + size > inSize) return false;
		// HandlerBox: header(8), version/flags(4), pre_defined(4), handler_type(4).
		if (type == oa::VideoMp4Box::Hdlr and size >= 20U) {
			return readU32BE(inData + offset + 16U) == 0x76696465U; // vide
		}
		offset += size;
	}
	return false;
}

// Detect codec from handler type
oa::VideoCodec detectCodecFromHandler(oa::U32 inHandlerType)
{
	// Common handler types: vide, soun, text, etc.
	// For video, we need to look at stsd sample entry
	return oa::VideoCodec::H264;  // Default for now
}

}  // namespace


void oa::VideoDemuxer::reset_() noexcept
{
	if (media_) {
		media_.reset();
	}
	if (file_ != nullptr) ::fclose(file_);
	file_ = nullptr;
	fileSize_ = 0U;
	info_ = {};
	sampleData_.clear();
	currentOffset_ = 0;
	eos_ = false;
	samples_.clear();
	currentSampleIndex_ = 0;
	avc_ = {};
	hvc_ = {};
	av1_ = {};
	vp9_ = {};
	fragment_ = {};
	needParameterSets_ = true;
	bufferedPictureNals_.clear();
	bufferedTimestamp_ = 0U;
	bufferedIsKeyframe_ = false;
	uri_.clear();
	config_ = {};
	stats_ = {};
	lastDecodeTimestamp_ = 0U;
	hasLastDecodeTimestamp_ = false;
}


oa::VideoDemuxer::VideoDemuxer(oa::VideoDemuxer&& inOther) noexcept
	: samples_(oa::move(inOther.samples_))
	, info_(inOther.info_)
	, avc_(oa::move(inOther.avc_))
	, hvc_(oa::move(inOther.hvc_))
	, av1_(oa::move(inOther.av1_)), vp9_(oa::move(inOther.vp9_))
	, fragment_(inOther.fragment_)
	, media_(oa::move(inOther.media_))
	, uri_(oa::move(inOther.uri_))
	, config_(oa::move(inOther.config_))
	, stats_(inOther.stats_)
	, lastDecodeTimestamp_(inOther.lastDecodeTimestamp_)
	, hasLastDecodeTimestamp_(inOther.hasLastDecodeTimestamp_)
	, file_(inOther.file_)
	, fileSize_(inOther.fileSize_)
	, sampleData_(oa::move(inOther.sampleData_))
	, currentOffset_(inOther.currentOffset_)
	, eos_(inOther.eos_)
	, currentSampleIndex_(inOther.currentSampleIndex_)
	, needParameterSets_(inOther.needParameterSets_)
	, bufferedPictureNals_(oa::move(inOther.bufferedPictureNals_))
	, bufferedTimestamp_(inOther.bufferedTimestamp_)
	, bufferedIsKeyframe_(inOther.bufferedIsKeyframe_)
{
	inOther.file_ = nullptr;
	inOther.reset_();
}


oa::VideoDemuxer& oa::VideoDemuxer::operator=(oa::VideoDemuxer&& inOther) noexcept
{
	if (this != &inOther) {
		reset_();
		info_ = inOther.info_;
		avc_ = oa::move(inOther.avc_);
		hvc_ = oa::move(inOther.hvc_);
		av1_ = oa::move(inOther.av1_);
		vp9_ = oa::move(inOther.vp9_);
		fragment_ = inOther.fragment_;
		media_ = oa::move(inOther.media_);
		uri_ = oa::move(inOther.uri_);
		config_ = oa::move(inOther.config_);
		stats_ = inOther.stats_;
		lastDecodeTimestamp_ = inOther.lastDecodeTimestamp_;
		hasLastDecodeTimestamp_ = inOther.hasLastDecodeTimestamp_;
		file_ = inOther.file_;
		fileSize_ = inOther.fileSize_;
		sampleData_ = oa::move(inOther.sampleData_);
		currentOffset_ = inOther.currentOffset_;
		eos_ = inOther.eos_;
		samples_ = oa::move(inOther.samples_);
		currentSampleIndex_ = inOther.currentSampleIndex_;
		needParameterSets_ = inOther.needParameterSets_;
		bufferedPictureNals_ = oa::move(inOther.bufferedPictureNals_);
		bufferedTimestamp_ = inOther.bufferedTimestamp_;
		bufferedIsKeyframe_ = inOther.bufferedIsKeyframe_;
		inOther.file_ = nullptr;
		inOther.reset_();
	}
	return *this;
}


oa::VideoDemuxer::~VideoDemuxer()
{
	reset_();
}


oa::Status oa::VideoDemuxer::close()
{
	int closeResult = 0;
	if (file_ != nullptr) {
		closeResult = ::fclose(file_);
		file_ = nullptr;
	}
	reset_();
	if (closeResult != 0) {
		return oa::Status::error(
			oa::StatusCode::DataLoss,
			"Failed to close video stream");
	}
	return oa::Status::ok();
}


oa::Result<oa::VideoContainerInfo> oa::VideoDemuxer::probe(const char* inPath)
{
	oa::VideoContainerInfo info = {};
	
	if (inPath == nullptr || inPath[0] == '\0') {
		return oa::Status::invalidArgument("VideoDemuxer::probe requires a path");
	}
	::FILE* file = ::fopen(inPath, "rb");
	if (file == nullptr) {
		return oa::Status::notFound(
			oa::String("Cannot open video container: ") + inPath);
	}
	oa::U8 bytes[12] = {};
	const oa::Usize read = ::fread(bytes, 1U, sizeof(bytes), file);
	::fclose(file);
	if (read < sizeof(bytes)) {
		return oa::Status::error("file too small to be a valid container");
	}
	
	// Check for ftyp box
	oa::U32 ftypSize = readU32BE(bytes);
	oa::U32 ftypType = readU32BE(bytes + 4);
	
	if (ftypType == oa::VideoMp4Box::Ftyp) {
		info.kind = oa::VideoContainerKind::Mp4;
		// Read major brand (bytes 8-11)
		oa::U32 majorBrand = readU32BE(bytes + 8);
		(void)ftypSize;
		(void)majorBrand;
		// Common brands: isom, mp42, avc1, etc.
		// For now, just mark as MP4
	} else {
		auto media = openMedia_(inPath);
		return media.isOk() ? oa::Result<oa::VideoContainerInfo>(media->getInfo())
			: oa::Result<oa::VideoContainerInfo>(media.getStatus());
	}
	
	return info;
}

oa::Result<oa::VideoDemuxer> oa::VideoDemuxer::open(oa::StringView inUri)
{
	return open(inUri, {});
}

oa::Result<oa::VideoDemuxer> oa::VideoDemuxer::open(
	oa::StringView inUri, const oa::VideoDemuxerConfig& inConfig)
{
	if (inUri.empty()) {
		return oa::Status::invalidArgument("VideoDemuxer::open requires a URI");
	}
	const oa::String uri(inUri);
	if (uri.view().find("://") != oa::StringView::Npos) {
		return openMedia_(inUri, inConfig);
	}
	// openLocal_ owns local-container routing, including the native fallback for
	// supported non-MP4 containers. Preserve its exact file and parse failures.
	auto local = openLocal_(inUri);
	if (not local.isOk()) return local.getStatus();
	local->uri_ = uri;
	local->config_ = inConfig;
	return local;
}

oa::Result<oa::VideoDemuxer> oa::VideoDemuxer::openMedia_(
	oa::StringView inUri, const oa::VideoDemuxerConfig& inConfig)
{
	const oa::String uri(inUri);
	const oa::StringView uriView = uri.view();
	if (uriView.find("://") == oa::StringView::Npos) {
		::FILE* file = ::fopen(uri.cStr(), "rb");
		if (file != nullptr) {
			oa::U8 signature[189] = {};
			const oa::Usize signatureSize = ::fread(signature, 1U, sizeof(signature), file);
			if (::fseeko(file, 0, SEEK_END) == 0) {
				const off_t end = ::ftello(file);
				if (end >= 0 and signatureSize >= 4U
					and readU32BE(signature) == 0x1A45DFA3U) {
					oa::VideoDemuxer stream;
					stream.uri_ = uri;
					stream.config_ = inConfig;
					stream.file_ = file;
					stream.fileSize_ = static_cast<oa::U64>(end);
					oa::Status parsed = parseMatroskaFile(
						file, stream.fileSize_, inUri, stream);
					if (not parsed.isOk()) return parsed;
					stream.eos_ = false;
					return oa::move(stream);
				} else if (end >= 188 and signatureSize >= 1U and signature[0] == 0x47U
					and (signatureSize < sizeof(signature) or signature[188] == 0x47U)) {
					oa::VideoDemuxer stream;
					stream.uri_ = uri;
					stream.config_ = inConfig;
					stream.file_ = file;
					stream.fileSize_ = static_cast<oa::U64>(end);
					stream.media_ = oa::makeUnique<MediaImpl>();
					oa::Status initialized = initMpegTs(
						file, stream.fileSize_, *stream.media_, stream.info_);
					if (not initialized.isOk()) return initialized;
					stream.eos_ = false;
					return oa::move(stream);
				}
			}
			::fclose(file);
		}
	}
	return oa::Status::error(oa::StatusCode::Unimplemented,
		uriView.find("://") != oa::StringView::Npos
			? "OA-native network transports are not implemented yet"
			: "Unsupported native media container or codec");
}


oa::Result<oa::VideoDemuxer> oa::VideoDemuxer::openLocal_(oa::StringView inPath)
{
	const oa::String path(inPath);
	oa::VideoDemuxer stream;
	
	// probe container first
	auto probeResult = probe(path.cStr());
	if (!probeResult.isOk()) {
		return probeResult.getStatus();
	}
	const auto& info = probeResult.getValue();
	if (info.kind != oa::VideoContainerKind::Mp4) return openMedia_(inPath);
	stream.info_.kind = info.kind;
	stream.info_.codec = info.codec;
	stream.info_.width = info.width;
	stream.info_.height = info.height;
	stream.info_.frameRate = info.frameRate;
	stream.info_.duration = info.duration;
	stream.info_.timebaseNum = info.timebaseNum;
	stream.info_.timebaseDen = info.timebaseDen;
	stream.info_.trackCount = info.trackCount;
	
	stream.file_ = ::fopen(path.cStr(), "rb");
	if (stream.file_ == nullptr) {
		return oa::Status::error(oa::StatusCode::NotFound, "Cannot open video container");
	}
	if (::fseeko(stream.file_, 0, SEEK_END) != 0) {
		return oa::Status::error("Cannot seek video container");
	}
	const off_t end = ::ftello(stream.file_);
	if (end < 0) return oa::Status::error("Cannot determine video container size");
	stream.fileSize_ = static_cast<oa::U64>(end);

	// Scan only top-level headers and load the compact moov metadata box. Media
	// samples stay on disk and are read into bounded per-packet scratch later.
	oa::U64 offset = 0U;
	while (offset + 8U <= stream.fileSize_) {
		BoxHeader header;
		oa::U64 headerSize = 0U;
		if (!readTopLevelBoxHeader(
			stream.file_, offset, stream.fileSize_, header, headerSize)) break;
		if (header.type == oa::VideoMp4Box::Moov) {
			const oa::U64 payloadSize = header.size - headerSize;
			if (payloadSize > static_cast<oa::U64>(SIZE_MAX)) {
				return oa::Status::error("MP4 moov metadata exceeds host address space");
			}
			oa::Vec<oa::U8> metadata(static_cast<oa::Usize>(payloadSize));
			if (::fseeko(stream.file_, static_cast<off_t>(offset + headerSize), SEEK_SET) != 0
				|| ::fread(metadata.data(), 1U, metadata.size(), stream.file_) != metadata.size()) {
				return oa::Status::error("Cannot read MP4 moov metadata");
			}
			parseMoovBox(metadata.data(), metadata.size(), stream);
		} else if (header.type == oa::VideoMp4Box::Moof) {
			const oa::U64 payloadSize = header.size - headerSize;
			if (payloadSize > static_cast<oa::U64>(SIZE_MAX)) {
				return oa::Status::error("MP4 fragment metadata exceeds host address space");
			}
			oa::Vec<oa::U8> fragment(static_cast<oa::Usize>(payloadSize));
			if (::fseeko(stream.file_, static_cast<off_t>(offset + headerSize), SEEK_SET) != 0
				or ::fread(fragment.data(), 1U, fragment.size(), stream.file_) != fragment.size()) {
				return oa::Status::error("Cannot read MP4 fragment metadata");
			}
			OA_RETURN_IF_ERROR(parseMoofBox(fragment.data(), fragment.size(), offset,
				offset + header.size, stream));
		}
		offset += header.size;
	}
	
	// If we found samples, we're ready to read
	if (!stream.samples_.empty()) {
		stream.eos_ = false;
		// probe-time frameRate is often unreliable for MP4; recompute from
		// the parsed sample table now that we have it.
		if (stream.info_.frameRate == 0 and stream.info_.timebaseDen > 0) {
			const auto& last = stream.samples_[stream.samples_.size() - 1];
			const oa::U64 totalTicks = last.dts + last.duration;
			if (totalTicks > 0) {
				const double seconds = static_cast<double>(totalTicks) /
					static_cast<double>(stream.info_.timebaseDen);
				if (seconds > 0.0) {
					const double fps = static_cast<double>(stream.samples_.size()) / seconds;
					stream.info_.frameRate = static_cast<oa::U32>(fps + 0.5);
				}
			}
		}
	} else return oa::Status::error(oa::StatusCode::DataLoss,
		"MP4 contains no readable video samples");

	return stream;
}


oa::Status oa::VideoDemuxer::readNextPacket(oa::VideoPacket& outPacket)
{
	if (media_) return readMediaPacket_(outPacket);
	if (eos_ || currentSampleIndex_ >= samples_.size()) {
		eos_ = true;
		return oa::Status::error("End of stream");
	}

	const Sample& sample = samples_[currentSampleIndex_];
	if (sample.offset + sample.size > fileSize_) {
		eos_ = true;
		return oa::Status::error("Sample offset/size exceeds file bounds");
	}
	sampleData_.resize(sample.size);
	if (::fseeko(file_, static_cast<off_t>(sample.offset), SEEK_SET) != 0
		|| ::fread(sampleData_.data(), 1U, sample.size, file_) != sample.size) {
		eos_ = true;
		return oa::Status::error("Failed to read compressed video sample");
	}

	outPacket.data.clear();
	outPacket.data.reserve(sample.size + 32);  // reserve a little extra for SPS/PPS

	// Check if we have buffered picture NAL units from a previous sample
	if (!bufferedPictureNals_.empty()) {
		outPacket.data = oa::move(bufferedPictureNals_);
		outPacket.presentationTimestamp = bufferedTimestamp_;
		outPacket.decodeTimestamp = bufferedTimestamp_;
		outPacket.isKeyframe = bufferedIsKeyframe_;
		outPacket.trackIndex = 0;
		bufferedPictureNals_.clear();
		return oa::Status::ok();
	}

	// MP4 stores NAL units length-prefixed; we need Annex-B (00 00 00 01 +
	// payload) for the vulkan Video decoder. Also prepend SPS+PPS on the
	// first IDR so the decoder can bring up parameter sets.
	const oa::U8* src = sampleData_.data();
	const oa::Usize srcSize = sample.size;

	const bool isMp4 = info_.kind == oa::VideoContainerKind::Mp4;
	const bool hasLengthPrefixedCodecConfig = isMp4
		or info_.kind == oa::VideoContainerKind::Matroska
		or info_.kind == oa::VideoContainerKind::WebM;
	const bool lengthPrefixedNal = hasLengthPrefixedCodecConfig
		&& (info_.codec == oa::VideoCodec::H264 || info_.codec == oa::VideoCodec::H265);
	// AV1 MP4 samples are OBU temporal units; VP9 samples are raw frame
	// bitstreams. Neither uses H.264/HEVC length-prefixed NAL packing.
	const bool rawMp4Sample = isMp4
		&& (info_.codec == oa::VideoCodec::AV1 || info_.codec == oa::VideoCodec::VP9);

	const bool prependH264Ps = hasLengthPrefixedCodecConfig && avc_.valid && sample.isKeyframe
		&& needParameterSets_ && info_.codec == oa::VideoCodec::H264;
	const bool prependH265Ps = hasLengthPrefixedCodecConfig && hvc_.valid && sample.isKeyframe
		&& needParameterSets_ && info_.codec == oa::VideoCodec::H265;
	// AV1 MP4 samples omit the sequence header (it lives in av1C). Prepend the
	// cached configOBUs to every keyframe temporal unit so the parser always
	// sees a sequence-header OBU. Inter frames rely on the parser's cached
	// sequence header from the preceding keyframe.
	const bool prependAv1Cfg = isMp4 && av1_.valid && sample.isKeyframe
		&& info_.codec == oa::VideoCodec::AV1;

	// For H.265, we need to separate parameter-set-only NAL units from picture
	// NAL units to ensure parameter sets are uploaded to the vulkan session
	// before slices are decoded
	const bool splitH265Ps = lengthPrefixedNal
		&& info_.codec == oa::VideoCodec::H265 && !rawMp4Sample;

	if (prependH264Ps) {
		for (oa::Usize i = 0; i < avc_.spsAnnexB.size(); ++i) {
			outPacket.data.pushBack(avc_.spsAnnexB[i]);
		}
		for (oa::Usize i = 0; i < avc_.ppsAnnexB.size(); ++i) {
			outPacket.data.pushBack(avc_.ppsAnnexB[i]);
		}
		needParameterSets_ = false;
	}
	if (prependH265Ps) {
		for (oa::Usize i = 0; i < hvc_.vpsAnnexB.size(); ++i) {
			outPacket.data.pushBack(hvc_.vpsAnnexB[i]);
		}
		for (oa::Usize i = 0; i < hvc_.spsAnnexB.size(); ++i) {
			outPacket.data.pushBack(hvc_.spsAnnexB[i]);
		}
		for (oa::Usize i = 0; i < hvc_.ppsAnnexB.size(); ++i) {
			outPacket.data.pushBack(hvc_.ppsAnnexB[i]);
		}
		needParameterSets_ = false;
	}

	if (rawMp4Sample) {
		if (prependAv1Cfg) {
			for (oa::Usize i = 0; i < av1_.configObus.size(); ++i) {
				outPacket.data.pushBack(av1_.configObus[i]);
			}
		}
		for (oa::Usize i = 0; i < srcSize; ++i) {
			outPacket.data.pushBack(src[i]);
		}
	} else {
	const oa::U8 lengthSize = (info_.codec == oa::VideoCodec::H265) ? hvc_.lengthSize : avc_.lengthSize;
	if (lengthPrefixedNal && lengthSize > 0) {
		// Walk the length-prefixed NAL list and rewrite each as Annex-B.
		const oa::U8 nls = lengthSize;
		oa::Usize p = 0;
		const oa::U8 startCode[4] = { 0, 0, 0, 1 };
		bool hasPictureNal = false;

		while (p + nls <= srcSize) {
			oa::U32 nalLen = 0;
			for (oa::U8 b = 0; b < nls; ++b) {
				nalLen = (nalLen << 8) | static_cast<oa::U32>(src[p + b]);
			}
			p += nls;
			if (p + nalLen > srcSize || nalLen == 0) {
				break;
			}

			// Check NAL type for H.265
			bool isParamSet = false;
			if (splitH265Ps && nalLen > 0) {
				oa::U8 nalType = (src[p] >> 1) & 0x3F;
				if (nalType == 32 || nalType == 33 || nalType == 34) {  // VPS, SPS, PPS
					isParamSet = true;
				}
			}

			// Write start code and NAL data
			for (auto byte : startCode) {
				outPacket.data.pushBack(byte);
			}
			for (oa::U32 k = 0; k < nalLen; ++k) {
				outPacket.data.pushBack(src[p + k]);
			}

				// For H.265, if we have both parameter sets and picture NALs in the
				// same sample, return parameter sets first and buffer picture NALs for
				// the next call
			if (splitH265Ps) {
				if (!isParamSet) {
					hasPictureNal = true;
				}
			}

			p += nalLen;
		}

		// If we have both parameter sets and picture NALs, split them
		if (splitH265Ps && hasPictureNal && outPacket.data.size() > 0) {
			// Scan backwards to find the last parameter set
			oa::I32 lastParamSetEnd = -1;
			oa::I32 lastStartCode = -1;
			for (oa::I32 i = static_cast<oa::I32>(outPacket.data.size()) - 4; i >= 0; --i) {
				if (outPacket.data[i] == 0 && outPacket.data[i + 1] == 0 &&
					outPacket.data[i + 2] == 0 && outPacket.data[i + 3] == 1) {
					lastStartCode = i;
					if (i + 4 < static_cast<oa::I32>(outPacket.data.size())) {
						oa::U8 nalType = (outPacket.data[i + 4] >> 1) & 0x3F;
						if (nalType == 32 || nalType == 33 || nalType == 34) {  // VPS, SPS, PPS
							lastParamSetEnd = i;
							break;
						}
					}
				}
			}

			// If we found parameter sets followed by picture NALs, split them
			if (lastParamSetEnd >= 0 && lastStartCode > lastParamSetEnd) {
				// move picture NALs to buffer
				bufferedPictureNals_.clear();
				bufferedPictureNals_.reserve(outPacket.data.size() - lastStartCode);
				for (oa::Usize i = static_cast<oa::Usize>(lastStartCode); i < outPacket.data.size(); ++i) {
					bufferedPictureNals_.pushBack(outPacket.data[i]);
				}
				outPacket.data.resize(lastStartCode);

				// store timestamp and keyframe flag for buffered picture NALs
				bufferedTimestamp_ = sample.dts + static_cast<oa::U64>(sample.ctsOffset);
				bufferedIsKeyframe_ = sample.isKeyframe;

					// Return parameter sets only (no timestamp needed for
					// parameter-set-only packet)
				outPacket.presentationTimestamp = 0;
				outPacket.decodeTimestamp = 0;
				outPacket.isKeyframe = false;
				outPacket.trackIndex = 0;
					// Sample is fully consumed; picture NALs are buffered for the next
					// call.
				++currentSampleIndex_;
				return oa::Status::ok();
			}
		}
	} else {
		// Raw Annex-B (no container): bytes are already start-code prefixed.
		for (oa::Usize i = 0; i < srcSize; ++i) {
			outPacket.data.pushBack(src[i]);
		}
	}
	}

	outPacket.presentationTimestamp = sample.dts + static_cast<oa::U64>(sample.ctsOffset);
	outPacket.decodeTimestamp = sample.dts;
	outPacket.isKeyframe = sample.isKeyframe;
	outPacket.trackIndex = 0;

	++currentSampleIndex_;
	return oa::Status::ok();
}

oa::Status oa::VideoDemuxer::readMediaPacket_(oa::VideoPacket& outPacket)
{
	if (media_ and media_->kind == MediaImpl::NativeKind::MpegTs) {
		if (eos_) return oa::Status::error(oa::StatusCode::OutOfRange, "End of MPEG-TS stream");
		oa::Status status = readMpegTsPes(file_, *media_, info_.codec, outPacket);
		if (not status.isOk()) {
			eos_ = status.getCode() == oa::StatusCode::OutOfRange;
			return status;
		}
		++currentSampleIndex_;
		return oa::Status::ok();
	}
	(void)outPacket;
	return oa::Status::error(oa::StatusCode::Unimplemented,
		"Native media packet backend is unavailable for this source");
}

oa::Status oa::VideoDemuxer::reconnectMedia_()
{
	return oa::Status::unimplemented("OA-native live reconnect is not implemented yet");
}


oa::Status oa::VideoDemuxer::seek(oa::U64 inTimestamp)
{
	if (media_) return seekMedia_(inTimestamp);
	if (samples_.empty()) {
		return oa::Status::error("VideoDemuxer::seek: no samples parsed");
	}

	// Walk the sample table backwards from the closest sample whose PTS is
	// <= inTimestamp, snapping to the nearest preceding keyframe. With no
	// keyframe table (`stss`) at all, every sample is treated as a keyframe
	// (H.264 raw or all-IDR encoded streams).
	oa::Usize target = 0;
	for (oa::Usize i = 0; i < samples_.size(); ++i) {
		const oa::U64 pts = samples_[i].dts + static_cast<oa::U64>(samples_[i].ctsOffset);
		if (pts > inTimestamp) {
			break;
		}
		target = i;
	}
	while (target > 0 && not samples_[target].isKeyframe) {
		--target;
	}

	currentSampleIndex_ = target;
	eos_ = false;
	needParameterSets_ = true;  // re-emit SPS+PPS at next keyframe
	return oa::Status::ok();
}

oa::Status oa::VideoDemuxer::seekMedia_(oa::U64 inTimestamp)
{
	if (media_ and media_->kind == MediaImpl::NativeKind::MpegTs) {
		if (inTimestamp != 0U) return oa::Status::error(oa::StatusCode::Unimplemented,
			"Indexed MPEG-TS seek is not implemented; seek to zero is supported");
		if (file_ == nullptr or ::fseeko(file_, 0, SEEK_SET) != 0) {
			return oa::Status::error(oa::StatusCode::Unavailable, "Could not rewind MPEG-TS stream");
		}
		media_->pes.clear();
		media_->pesPts = 0U;
		media_->pesDts = 0U;
		currentSampleIndex_ = 0U;
		eos_ = false;
		return oa::Status::ok();
	}
	(void)inTimestamp;
	return oa::Status::error(oa::StatusCode::Unimplemented,
		"Indexed seek is not implemented for this native media source");
}

bool oa::VideoDemuxer::isLive() const noexcept
{
	return media_ and media_->live;
}

bool oa::VideoDemuxer::isSeekable() const noexcept
{
	return media_ ? media_->seekable : file_ != nullptr;
}


oa::VideoProfile oa::VideoDemuxer::getVideoProfile() const
{
	oa::VideoProfile profile = {};
	switch (info_.codec) {
	case oa::VideoCodec::H264:
		if (avc_.valid) profile = avc_.profile;
		break;
	case oa::VideoCodec::H265:
		if (hvc_.valid) profile = hvc_.profile;
		break;
	case oa::VideoCodec::AV1:
		if (av1_.valid) profile = av1_.profile;
		break;
	case oa::VideoCodec::VP9:
		if (vp9_.valid) profile = vp9_.profile;
		break;
	}
	profile.codec = info_.codec;
	profile.width = info_.width;
	profile.height = info_.height;
	return profile;
}


namespace
{

// parse moov box (movie metadata)
void parseMoovBox(const oa::U8* inData, oa::U64 inSize, oa::VideoDemuxer& outStream)
{
	oa::U64 offset = 0;
	while (offset + 8 <= inSize) {
		oa::U32 boxSize = readU32BE(inData + offset);
		oa::U32 boxType = readU32BE(inData + offset + 4);
		
		if (boxSize == 1) {
			// Extended size - skip for now
			break;
		}
		
		if (boxSize < 8 || offset + boxSize > inSize) {
			break;
		}
		
		// parse trak box (track)
		if (boxType == oa::VideoMp4Box::Trak) {
			parseTrakBox(inData + offset + 8, boxSize - 8, outStream);
		}
		else if (boxType == oa::VideoMp4Box::Mvex) {
			const oa::U8* mvex = inData + offset + 8U;
			const oa::U64 mvexSize = boxSize - 8U;
			oa::U64 child = 0U;
			while (child + 8U <= mvexSize) {
				const oa::U32 childSize = readU32BE(mvex + child);
				const oa::U32 childType = readU32BE(mvex + child + 4U);
				if (childSize < 8U or child + childSize > mvexSize) break;
				// TrackExtendsBox: version/flags, track_ID, default description,
				// default duration, default size, default flags.
				if (childType == oa::VideoMp4Box::Trex and childSize >= 32U) {
					const oa::U8* trex = mvex + child + 8U;
					const oa::U32 trackId = readU32BE(trex + 4U);
					if (outStream.fragment_.trackId == 0U
						or trackId == outStream.fragment_.trackId) {
						outStream.fragment_.trackId = trackId;
						outStream.fragment_.defaultSampleDuration = readU32BE(trex + 12U);
						outStream.fragment_.defaultSampleSize = readU32BE(trex + 16U);
						outStream.fragment_.defaultSampleFlags = readU32BE(trex + 20U);
					}
				}
				child += childSize;
			}
		}
		
		offset += boxSize;
	}
}

// parse trak box (track metadata)
void parseTrakBox(const oa::U8* inData, oa::U64 inSize, oa::VideoDemuxer& outStream)
{
	bool videoTrack = false;
	oa::U32 trackId = 0U;
	for (oa::U64 scan = 0U; scan + 8U <= inSize;) {
		const oa::U32 size = readU32BE(inData + scan);
		const oa::U32 type = readU32BE(inData + scan + 4U);
		if (size < 8U or scan + size > inSize) break;
		if (type == oa::VideoMp4Box::Mdia) {
			videoTrack = isVideoMdia(inData + scan + 8U, size - 8U);
		} else if (type == oa::VideoMp4Box::Tkhd and size >= 8U + 16U) {
			const oa::U8* tkhd = inData + scan + 8U;
			const oa::U8 version = tkhd[0];
			const oa::U64 idOffset = version == 1U ? 20U : 12U;
			if (size - 8U >= idOffset + 4U) trackId = readU32BE(tkhd + idOffset);
		}
		scan += size;
	}
	if (not videoTrack) return;
	if (trackId != 0U) outStream.fragment_.trackId = trackId;

	oa::U64 offset = 0;
	while (offset + 8 <= inSize) {
		oa::U32 boxSize = readU32BE(inData + offset);
		oa::U32 boxType = readU32BE(inData + offset + 4);
		
		if (boxSize < 8 || offset + boxSize > inSize) {
			break;
		}
		
		// parse mdia box (media)
		if (boxType == oa::VideoMp4Box::Mdia) {
			const oa::U8* mdia = inData + offset + 8U;
			const oa::U64 mdiaSize = boxSize - 8U;
			// oa::VideoDemuxer is intentionally a video elementary-stream reader.
			// Ignore audio/text sample tables instead of letting the final track
			// overwrite the selected video geometry and packet index.
			parseMdiaBox(mdia, mdiaSize, outStream);
		}
		
		offset += boxSize;
	}
}

// parse mdia box (media information)
void parseMdiaBox(const oa::U8* inData, oa::U64 inSize, oa::VideoDemuxer& outStream)
{
	oa::U64 offset = 0;
	while (offset + 8 <= inSize) {
		oa::U32 boxSize = readU32BE(inData + offset);
		oa::U32 boxType = readU32BE(inData + offset + 4);

		if (boxSize < 8 || offset + boxSize > inSize) {
			break;
		}

		// parse mdhd (media header) — timescale lives here. Per ISO/IEC 14496-12
		// §8.4.2 the v0 layout puts timescale at +20 (after version/flags +
		// creation_time + modification_time); v1 widens the times to 64-bit so
		// timescale moves to +28.
		if (boxType == oa::VideoMp4Box::Mdhd) {
			const oa::U8* d = inData + offset + 8;
			const oa::U64 dn = boxSize - 8;
			if (dn >= 4) {
				const oa::U8 version = d[0];
				oa::U64 tsOffset = (version == 1) ? 20 : 12;
				if (dn >= tsOffset + 4) {
					outStream.info_.timebaseNum = 1;
					outStream.info_.timebaseDen = readU32BE(d + tsOffset);
				}
			}
		}
		// parse minf box (media info)
		else if (boxType == oa::VideoMp4Box::Minf) {
			parseMinfBox(inData + offset + 8, boxSize - 8, outStream);
		}

		offset += boxSize;
	}
}

// parse minf box (media info)
void parseMinfBox(const oa::U8* inData, oa::U64 inSize, oa::VideoDemuxer& outStream)
{
	oa::U64 offset = 0;
	while (offset + 8 <= inSize) {
		oa::U32 boxSize = readU32BE(inData + offset);
		oa::U32 boxType = readU32BE(inData + offset + 4);
		
		if (boxSize < 8 || offset + boxSize > inSize) {
			break;
		}
		
		// parse stbl box (sample table)
		if (boxType == oa::VideoMp4Box::Stbl) {
			parseStblBox(inData + offset + 8, boxSize - 8, outStream);
		}
		
		offset += boxSize;
	}
}

// parse stbl box (sample table) - phase C1b: Implement full parsing
void parseStblBox(const oa::U8* inData, oa::U64 inSize, oa::VideoDemuxer& outStream)
{
	// Temporary storage for sample table data
	oa::Vec<oa::U32> sampleSizes;      // From stsz
	oa::Vec<oa::U64> chunkOffsets;     // From stco
	oa::Vec<oa::U32> sttsEntries;     // From stts (count, duration pairs)
	oa::Vec<oa::U32> stscEntries;  // From stsc (firstChunk, samplesPerChunk,
							   // sampleDescriptionIndex)
	oa::Vec<oa::U32> stssEntries;     // From stss (keyframe sample indices)
	oa::Vec<oa::I32> cttsEntries;     // From ctts (sampleCount, compositionOffset)
	
	oa::U64 offset = 0;
	while (offset + 8 <= inSize) {
		oa::U32 boxSize = readU32BE(inData + offset);
		oa::U32 boxType = readU32BE(inData + offset + 4);
		
		if (boxSize < 8 || offset + boxSize > inSize) {
			break;
		}
		
		const oa::U8* boxData = inData + offset + 8;
		const oa::U64 boxDataSize = boxSize - 8;
		
		// parse stsd (sample description) — codec, width, height.
		if (boxType == oa::VideoMp4Box::Stsd) {
			if (boxDataSize < 8U) { offset += boxSize; continue; }
			oa::U32 versionFlags = readU32BE(boxData);
			oa::U32 entryCount = readU32BE(boxData + 4);
			(void)versionFlags;
			// stsd payload: 8-byte header (version+count), then per-entry
			// VisualSampleEntry whose width/height live at fixed offsets per
			// ISO/IEC 14496-12 §8.5.2 (avc1, hev1/hvc1, av01 all extend this).
			if (entryCount > 0 && boxDataSize >= 8 + 8 + 36) {
				const oa::U8* entry     = boxData + 8;
				oa::U32 entrySize       = readU32BE(entry);
				oa::U32 entryType       = readU32BE(entry + 4);
				if (entrySize < 8U + 78U
					or static_cast<oa::U64>(entrySize) > boxDataSize - 8U) {
					offset += boxSize; continue;
				}

				// Detect codec from the sample entry fourcc.
					constexpr oa::U32 kFourccAvc1 = 0x61766331; // 'avc1'
					constexpr oa::U32 kFourccAvc3 = 0x61766333; // 'avc3'
					constexpr oa::U32 kFourccHvc1 = 0x68766331; // 'hvc1'
					constexpr oa::U32 kFourccHev1 = 0x68657631; // 'hev1'
					constexpr oa::U32 kFourccAv01 = 0x61763031; // 'av01'
					constexpr oa::U32 kFourccVp09 = 0x76703039; // 'vp09'
					if (entryType == kFourccHvc1 || entryType == kFourccHev1) {
						outStream.info_.codec = oa::VideoCodec::H265;
					} else if (entryType == kFourccAv01) {
						outStream.info_.codec = oa::VideoCodec::AV1;
					} else if (entryType == kFourccVp09) {
						outStream.info_.codec = oa::VideoCodec::VP9;
					} else if (entryType == kFourccAvc1 || entryType == kFourccAvc3) {
						outStream.info_.codec = oa::VideoCodec::H264;
					}
	
					// VisualSampleEntry: width at +32, height at +34 inside the
					// entry payload (after 8-byte box header → 24 bytes
					// SampleEntry/VisualSampleEntry preamble).
					const oa::U8* visual = entry + 8;          // skip box header
					outStream.info_.width  = readU16BE(visual + 24);
					outStream.info_.height = readU16BE(visual + 26);
	
					// VisualSampleEntry payload runs 78 bytes (preamble + 32-byte
					// compressorname + depth + pre_defined). After that, codec-
					// specific config boxes follow (avcC, hvcC, ...).
					if (entryType == kFourccAvc1 || entryType == kFourccAvc3) {
						const oa::Usize visualEnd = (entry + entrySize) - boxData;
						oa::Usize cfgOffset = (visual - boxData) + 78;
						while (cfgOffset + 8 <= visualEnd && cfgOffset + 8 <= boxDataSize) {
							const oa::U32 cfgSize = readU32BE(boxData + cfgOffset);
							const oa::U32 cfgType = readU32BE(boxData + cfgOffset + 4);
							if (cfgSize < 8 || cfgOffset + cfgSize > visualEnd) {
								break;
							}
							if (cfgType == oa::VideoMp4Box::Avcc && cfgSize > 8 + 7) {
								// aVCDecoderConfigurationRecord (ISO/IEC 14496-15 §5.2.1.1).
								const oa::U8* avcc = boxData + cfgOffset + 8;
								const oa::U64 avccSize = cfgSize - 8;
								(void)parseAvcDecoderConfig(avcc, avccSize,
								outStream.avc_);
							}
							cfgOffset += cfgSize;
						}
					}
					else if (entryType == kFourccHvc1 || entryType == kFourccHev1) {
						const oa::Usize visualEnd = (entry + entrySize) - boxData;
						oa::Usize cfgOffset = (visual - boxData) + 78;
						while (cfgOffset + 8 <= visualEnd && cfgOffset + 8 <= boxDataSize) {
							const oa::U32 cfgSize = readU32BE(boxData + cfgOffset);
							const oa::U32 cfgType = readU32BE(boxData + cfgOffset + 4);
							if (cfgSize < 8 || cfgOffset + cfgSize > visualEnd) {
								break;
							}
							if (cfgType == oa::VideoMp4Box::Hvcc && cfgSize > 8 + 22) {
								// hEVCDecoderConfigurationRecord (ISO/IEC 14496-15 §8.3.3.1).
								const oa::U8* hvcc = boxData + cfgOffset + 8;
								const oa::U64 hvccSize = cfgSize - 8;
								(void)parseHevcDecoderConfig(hvcc, hvccSize,
								outStream.hvc_);
							}
							cfgOffset += cfgSize;
						}
					}
					else if (entryType == kFourccAv01) {
						// AV1SampleEntry → av1C box (AV1 ISO-BMFF §2.3). The
						// AV1CodecConfigurationRecord is a 4-byte fixed header
						// followed by configOBUs — the sequence-header OBU(s)
						// stored out-of-band. AV1 MP4 samples are OBU temporal
						// units that omit the sequence header, so we cache it
						// here and prepend it to each keyframe (like SPS/PPS).
						constexpr oa::U32 kFourccAv1c = 0x61763143; // 'av1C'
						const oa::Usize visualEnd = (entry + entrySize) - boxData;
						oa::Usize cfgOffset = (visual - boxData) + 78;
						while (cfgOffset + 8 <= visualEnd && cfgOffset + 8 <= boxDataSize) {
							const oa::U32 cfgSize = readU32BE(boxData + cfgOffset);
							const oa::U32 cfgType = readU32BE(boxData + cfgOffset + 4);
							if (cfgSize < 8 || cfgOffset + cfgSize > visualEnd) {
								break;
							}
							// 4-byte AV1CodecConfigurationRecord header precedes the
							// configOBUs, so require > 8 (box hdr) + 4.
							if (cfgType == kFourccAv1c && cfgSize > 8 + 4) {
								const oa::U8* av1c = boxData + cfgOffset + 8;
								const oa::U64 av1cSize = cfgSize - 8; (void)parseAv1DecoderConfig(av1c, av1cSize,
									outStream.av1_);
								}
							cfgOffset += cfgSize;
						}
					} else if (entryType == kFourccVp09) {
						constexpr oa::U32 kFourccVpcc = 0x76706343; // 'vpcC'
						const oa::Usize visualEnd = (entry + entrySize) - boxData;
						oa::Usize cfgOffset = (visual - boxData) + 78;
						while (cfgOffset + 8 <= visualEnd && cfgOffset + 8 <= boxDataSize) {
							const oa::U32 cfgSize = readU32BE(boxData + cfgOffset);
							const oa::U32 cfgType = readU32BE(boxData + cfgOffset + 4);
							if (cfgSize < 8 or cfgOffset + cfgSize > visualEnd) break;
							if (cfgType == kFourccVpcc and cfgSize >= 8U + 7U) {
								(void)parseVp9DecoderConfig(boxData + cfgOffset + 8, cfgSize - 8U,
								outStream.vp9_);
							}
							cfgOffset += cfgSize;
						}
					}
			}
		}
		// parse stts (time-to-sample) - duration per sample
		else if (boxType == oa::VideoMp4Box::Stts) {
			if (boxDataSize < 8U) { offset += boxSize; continue; }
			oa::U32 versionFlags = readU32BE(boxData);
			oa::U32 entryCount = readU32BE(boxData + 4);
			(void)versionFlags;
			if (not mp4TableFits(boxDataSize, 8U, entryCount, 8U)) {
				offset += boxSize; continue;
			}
			sttsEntries.resize(entryCount * 2);
			for (oa::U32 i = 0; i < entryCount; ++i) {
				sttsEntries[i * 2] = readU32BE(boxData + 8 + i * 8);     // sampleCount
				sttsEntries[i * 2 + 1] = readU32BE(boxData + 8 + i * 8 + 4); // sampleDelta
			}
		}
		// parse stsc (sample-to-chunk) - chunk grouping
		else if (boxType == oa::VideoMp4Box::Stsc) {
			if (boxDataSize < 8U) { offset += boxSize; continue; }
			oa::U32 versionFlags = readU32BE(boxData);
			oa::U32 entryCount = readU32BE(boxData + 4);
			(void)versionFlags;
			if (not mp4TableFits(boxDataSize, 8U, entryCount, 12U)) {
				offset += boxSize; continue;
			}
			stscEntries.resize(entryCount * 3);
			for (oa::U32 i = 0; i < entryCount; ++i) {
				stscEntries[i * 3] = readU32BE(boxData + 8 + i * 12);         // firstChunk
				stscEntries[i * 3 + 1] = readU32BE(boxData + 8 + i * 12 + 4);  // samplesPerChunk
				stscEntries[i * 3 + 2] = readU32BE(boxData + 8 + i * 12 + 8);  // sampleDescriptionIndex
			}
		}
		// parse stsz (sample size) - size per sample
		else if (boxType == oa::VideoMp4Box::Stsz) {
			if (boxDataSize < 12U) { offset += boxSize; continue; }
			oa::U32 versionFlags = readU32BE(boxData);
			oa::U32 sampleSize = readU32BE(boxData + 4);
			oa::U32 sampleCount = readU32BE(boxData + 8);
			(void)versionFlags;
			if (sampleCount > kMaxMp4TableEntries) {
				offset += boxSize; continue;
			}
			if (sampleSize == 0) {
				// Variable sample sizes - read table
				if (not mp4TableFits(boxDataSize, 12U, sampleCount, 4U)) {
					offset += boxSize; continue;
				}
				sampleSizes.resize(sampleCount);
				for (oa::U32 i = 0; i < sampleCount; ++i) {
					sampleSizes[i] = readU32BE(boxData + 12 + i * 4);
				}
			} else {
				// Fixed sample size
				sampleSizes.resize(sampleCount);
				for (oa::U32 i = 0; i < sampleCount; ++i) {
					sampleSizes[i] = sampleSize;
				}
			}
		}
		// parse stco (chunk offset) - offset per chunk (32-bit)
		else if (boxType == oa::VideoMp4Box::Stco) {
			if (boxDataSize < 8U) { offset += boxSize; continue; }
			oa::U32 versionFlags = readU32BE(boxData);
			oa::U32 entryCount = readU32BE(boxData + 4);
			(void)versionFlags;
			if (not mp4TableFits(boxDataSize, 8U, entryCount, 4U)) {
				offset += boxSize; continue;
			}
			chunkOffsets.resize(entryCount);
			for (oa::U32 i = 0; i < entryCount; ++i) {
				chunkOffsets[i] = readU32BE(boxData + 8 + i * 4);
			}
		}
		// parse co64 (chunk offset, 64-bit) — required for files > 4 GiB
		else if (boxType == oa::VideoMp4Box::Co64) {
			if (boxDataSize < 8U) { offset += boxSize; continue; }
			oa::U32 versionFlags = readU32BE(boxData);
			oa::U32 entryCount = readU32BE(boxData + 4);
			(void)versionFlags;
			if (not mp4TableFits(boxDataSize, 8U, entryCount, 8U)) {
				offset += boxSize; continue;
			}
			chunkOffsets.resize(entryCount);
			for (oa::U32 i = 0; i < entryCount; ++i) {
				chunkOffsets[i] = readU64BE(boxData + 8 + i * 8);
			}
		}
		// parse stss (sync sample) - keyframe indices
		else if (boxType == oa::VideoMp4Box::Stss) {
			if (boxDataSize < 8U) { offset += boxSize; continue; }
			oa::U32 versionFlags = readU32BE(boxData);
			oa::U32 entryCount = readU32BE(boxData + 4);
			(void)versionFlags;
			if (not mp4TableFits(boxDataSize, 8U, entryCount, 4U)) {
				offset += boxSize; continue;
			}
			stssEntries.resize(entryCount);
			for (oa::U32 i = 0; i < entryCount; ++i) {
				const oa::U32 sampleNumber = readU32BE(boxData + 8 + i * 4);
				stssEntries[i] = sampleNumber == 0U ? 0U : sampleNumber - 1U;
			}
		}
		// parse ctts (composition time offset) - PTS-DTS offset
		else if (boxType == oa::VideoMp4Box::Ctts) {
			if (boxDataSize < 8U) { offset += boxSize; continue; }
			oa::U32 versionFlags = readU32BE(boxData);
			oa::U32 entryCount = readU32BE(boxData + 4);
			const oa::U8 version = static_cast<oa::U8>(versionFlags >> 24U);
			if (not mp4TableFits(boxDataSize, 8U, entryCount, 8U)) {
				offset += boxSize; continue;
			}
			cttsEntries.resize(entryCount * 2);
			for (oa::U32 i = 0; i < entryCount; ++i) {
				cttsEntries[i * 2] = static_cast<oa::I32>(readU32BE(boxData + 8 + i * 8));     // sampleCount
				const oa::U32 rawOffset = readU32BE(boxData + 8 + i * 8 + 4);
				cttsEntries[i * 2 + 1] = version == 1U
					? static_cast<oa::I32>(rawOffset)
					: static_cast<oa::I32>(oa::min<oa::U32>(rawOffset, 0x7FFFFFFFU));
			}
		}
		
		offset += boxSize;
	}
	
	// Build sample table from parsed data. Properly walks stsc to find the
	// chunk each sample lives in, then sums preceding samples' sizes to get
	// each sample's absolute file offset. stss == empty means every sample
	// is treated as a sync sample (Annex-B all-IDR streams).
	if (!sampleSizes.empty() && !chunkOffsets.empty()) {
		const oa::U32 sampleCount = sampleSizes.size();
		outStream.samples_.resize(sampleCount);

		// Walk stsc to materialize samples-per-chunk for every chunk index.
		// stscEntries layout per row: [firstChunk(1-based), samplesPerChunk,
		// descIndex].
		const oa::U32 stscRows = stscEntries.size() / 3;
		const oa::U32 chunkCount = chunkOffsets.size();

		auto samplesInChunk = [&](oa::U32 chunk1Based) -> oa::U32 {
			oa::U32 result = 1;
			for (oa::U32 r = 0; r < stscRows; ++r) {
				const oa::U32 firstChunk = stscEntries[r * 3];
				if (firstChunk > chunk1Based) { break; }
				result = stscEntries[r * 3 + 1];
			}
			return result;
		};

		// Walk samples in linear order, advancing the active chunk pointer.
		oa::U32 chunkIdx        = 0;   // 0-based index into chunkOffsets
		oa::U32 inChunkIdx      = 0;   // 0-based sample index within current chunk
		oa::U64 inChunkOffset   = 0;   // cumulative byte offset within current chunk
		oa::U64 dts             = 0;
		oa::U32 sttsRowIdx      = 0;
		oa::U32 sttsRowRemaining = sttsEntries.empty()
			? 0 : sttsEntries[0];     // sampleCount of row 0
		oa::U32 sttsRowDelta    = sttsEntries.size() >= 2
			? sttsEntries[1] : 1;
		oa::U32 cttsRowIdx      = 0;
		oa::U32 cttsRowRemaining = cttsEntries.empty()
			? 0 : cttsEntries[0];     // sampleCount of row 0
		oa::I32 cttsRowOffset   = cttsEntries.size() >= 2
			? cttsEntries[1] : 0;
		oa::Usize stssIdx       = 0;

		for (oa::U32 i = 0; i < sampleCount; ++i) {
			// advance into the next chunk when we've consumed this chunk's
			// sample budget.
			if (chunkIdx >= chunkCount) {
				outStream.samples_[i] = oa::VideoDemuxer::Sample{};
				continue;
			}
			const oa::U32 budget = samplesInChunk(chunkIdx + 1);
			if (inChunkIdx >= budget) {
				++chunkIdx;
				inChunkIdx = 0;
				inChunkOffset = 0;
				if (chunkIdx >= chunkCount) {
					outStream.samples_[i] = oa::VideoDemuxer::Sample{};
					continue;
				}
			}

			outStream.samples_[i].size   = sampleSizes[i];
			outStream.samples_[i].offset = chunkOffsets[chunkIdx] + inChunkOffset;
			inChunkOffset += sampleSizes[i];
			++inChunkIdx;

			outStream.samples_[i].dts = dts;

			// stts walk — advance to the next [count, delta] row when the
			// current row is exhausted.
			if (sttsRowRemaining == 0 && sttsRowIdx + 1 < sttsEntries.size() / 2) {
				++sttsRowIdx;
				sttsRowRemaining = sttsEntries[sttsRowIdx * 2];
				sttsRowDelta = sttsEntries[sttsRowIdx * 2 + 1];
			}
			if (sttsRowRemaining > 0) {
				--sttsRowRemaining;
				dts += sttsRowDelta;
			} else {
				dts += 1;
			}

			// ctts walk — same shape.
			if (cttsRowRemaining == 0 && cttsRowIdx + 1 < cttsEntries.size() / 2) {
				++cttsRowIdx;
				cttsRowRemaining = static_cast<oa::U32>(cttsEntries[cttsRowIdx * 2]);
				cttsRowOffset = cttsEntries[cttsRowIdx * 2 + 1];
			}
			outStream.samples_[i].ctsOffset = cttsRowOffset;
			if (cttsRowRemaining > 0) {
				--cttsRowRemaining;
			}

			// Keyframe: present in stss, or stss empty (every sample IDR).
			bool isKeyframe = stssEntries.empty();
			while (stssIdx < stssEntries.size() && stssEntries[stssIdx] < i) {
				++stssIdx;
			}
			if (stssIdx < stssEntries.size() && stssEntries[stssIdx] == i) {
				isKeyframe = true;
			}
			outStream.samples_[i].isKeyframe = isKeyframe;
		}
	}
}

oa::Status parseMoofBox(const oa::U8* inData, oa::U64 inSize, oa::U64 inMoofOffset,
	oa::U64 inMoofEnd,
	oa::VideoDemuxer& outStream)
{
	for (oa::U64 offset = 0U; offset + 8U <= inSize;) {
		const oa::U32 boxSize = readU32BE(inData + offset);
		const oa::U32 boxType = readU32BE(inData + offset + 4U);
		if (boxSize < 8U or offset + boxSize > inSize) {
			return oa::Status::error(oa::StatusCode::DataLoss, "Malformed fragmented MP4 box");
		}
		if (boxType != oa::VideoMp4Box::Traf) {
			offset += boxSize;
			continue;
		}

		const oa::U8* traf = inData + offset + 8U;
		const oa::U64 trafSize = boxSize - 8U;
		oa::U32 trackId = 0U;
		oa::U32 defaultDuration = outStream.fragment_.defaultSampleDuration;
		oa::U32 defaultSize = outStream.fragment_.defaultSampleSize;
		oa::U32 defaultFlags = outStream.fragment_.defaultSampleFlags;
		oa::U64 baseDataOffset = inMoofOffset;
		oa::U64 baseDecodeTime = 0U;
		oa::Vec<oa::U64> runs;

		for (oa::U64 child = 0U; child + 8U <= trafSize;) {
			const oa::U32 childSize = readU32BE(traf + child);
			const oa::U32 childType = readU32BE(traf + child + 4U);
			if (childSize < 8U or child + childSize > trafSize) {
				return oa::Status::error(oa::StatusCode::DataLoss, "Malformed MP4 track fragment");
			}
			const oa::U8* payload = traf + child + 8U;
			const oa::U64 payloadSize = childSize - 8U;
			if (childType == oa::VideoMp4Box::Tfhd and payloadSize >= 8U) {
				const oa::U32 flags = readU32BE(payload) & 0x00FFFFFFU;
				trackId = readU32BE(payload + 4U);
				oa::U64 p = 8U;
				if ((flags & 0x000001U) != 0U) {
					if (p + 8U > payloadSize) return oa::Status::error("truncated tfhd base offset");
					baseDataOffset = readU64BE(payload + p);
					p += 8U;
				}
				if ((flags & 0x000002U) != 0U) p += 4U; // sample_description_index
				if ((flags & 0x000008U) != 0U) {
					if (p + 4U > payloadSize) return oa::Status::error("truncated tfhd duration");
					defaultDuration = readU32BE(payload + p); p += 4U;
				}
				if ((flags & 0x000010U) != 0U) {
					if (p + 4U > payloadSize) return oa::Status::error("truncated tfhd size");
					defaultSize = readU32BE(payload + p); p += 4U;
				}
				if ((flags & 0x000020U) != 0U) {
					if (p + 4U > payloadSize) return oa::Status::error("truncated tfhd flags");
					defaultFlags = readU32BE(payload + p);
				}
			} else if (childType == oa::VideoMp4Box::Tfdt and payloadSize >= 8U) {
				const oa::U8 version = payload[0];
				if (version == 1U) {
					if (payloadSize < 12U) return oa::Status::error("truncated 64-bit tfdt");
					baseDecodeTime = readU64BE(payload + 4U);
				} else {
					baseDecodeTime = readU32BE(payload + 4U);
				}
			} else if (childType == oa::VideoMp4Box::Trun) {
				runs.pushBack(child);
			}
			child += childSize;
		}

		if (outStream.fragment_.trackId != 0U and trackId != outStream.fragment_.trackId) {
			offset += boxSize;
			continue;
		}
		if (trackId != 0U) outStream.fragment_.trackId = trackId;
		oa::U64 dts = baseDecodeTime;
		oa::U64 implicitDataOffset = inMoofEnd;
		for (oa::U64 runOffset : runs) {
			const oa::U32 runSize = readU32BE(traf + runOffset);
			const oa::U8* run = traf + runOffset + 8U;
			const oa::U64 runPayloadSize = runSize - 8U;
			if (runPayloadSize < 8U) return oa::Status::error("truncated trun");
			const oa::U8 version = run[0];
			const oa::U32 flags = readU32BE(run) & 0x00FFFFFFU;
			const oa::U32 sampleCount = readU32BE(run + 4U);
			oa::U64 p = 8U;
			oa::U64 dataOffset = implicitDataOffset;
			if ((flags & 0x000001U) != 0U) {
				if (p + 4U > runPayloadSize) return oa::Status::error("truncated trun data offset");
				const oa::I32 relative = static_cast<oa::I32>(readU32BE(run + p));
				if (relative < 0 and static_cast<oa::U64>(-static_cast<oa::I64>(relative)) > baseDataOffset) {
					return oa::Status::error("Invalid negative trun data offset");
				}
				dataOffset = relative >= 0
					? baseDataOffset + static_cast<oa::U64>(relative)
					: baseDataOffset - static_cast<oa::U64>(-static_cast<oa::I64>(relative));
				p += 4U;
			}
			oa::U32 firstSampleFlags = defaultFlags;
			if ((flags & 0x000004U) != 0U) {
				if (p + 4U > runPayloadSize) return oa::Status::error("truncated first sample flags");
				firstSampleFlags = readU32BE(run + p); p += 4U;
			}
			for (oa::U32 sampleIndex = 0U; sampleIndex < sampleCount; ++sampleIndex) {
				oa::U32 duration = defaultDuration;
				oa::U32 size = defaultSize;
				oa::U32 sampleFlags = sampleIndex == 0U ? firstSampleFlags : defaultFlags;
				oa::I32 ctsOffset = 0;
				if ((flags & 0x000100U) != 0U) {
					if (p + 4U > runPayloadSize) return oa::Status::error("truncated sample duration");
					duration = readU32BE(run + p); p += 4U;
				}
				if ((flags & 0x000200U) != 0U) {
					if (p + 4U > runPayloadSize) return oa::Status::error("truncated sample size");
					size = readU32BE(run + p); p += 4U;
				}
				if ((flags & 0x000400U) != 0U) {
					if (p + 4U > runPayloadSize) return oa::Status::error("truncated sample flags");
					sampleFlags = readU32BE(run + p); p += 4U;
				}
				if ((flags & 0x000800U) != 0U) {
					if (p + 4U > runPayloadSize) return oa::Status::error("truncated composition offset");
					const oa::U32 raw = readU32BE(run + p); p += 4U;
					ctsOffset = version == 1U ? static_cast<oa::I32>(raw)
						: static_cast<oa::I32>(oa::min<oa::U32>(raw, 0x7FFFFFFFU));
				}
				if (size == 0U) return oa::Status::error("Fragment sample has no declared size");
				oa::VideoDemuxer::Sample sample;
				sample.offset = dataOffset;
				sample.size = size;
				sample.duration = duration == 0U ? 1U : duration;
				sample.dts = dts;
				sample.ctsOffset = ctsOffset;
				sample.isKeyframe = (sampleFlags & 0x00010000U) == 0U;
				outStream.samples_.pushBack(sample);
				dataOffset += size;
				dts += sample.duration;
			}
			implicitDataOffset = dataOffset;
		}
		offset += boxSize;
	}
	return oa::Status::ok();
}

}  // namespace
