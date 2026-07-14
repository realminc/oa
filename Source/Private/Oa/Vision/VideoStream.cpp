// OA Vision — Video Stream Demuxer (MP4) Implementation
// Phase C1 of OaVideo Realm-ization plan

#include <Oa/Vision/VideoStream.h>
#include "Video/Codec/NalParser.h"
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string_view>

struct OaVideoStream::MediaImpl {
	enum class NativeKind : OaU8 { None, MpegTs } Kind = NativeKind::None;
	OaU16 PmtPid = 0x1FFFU;
	OaU16 VideoPid = 0x1FFFU;
	OaVec<OaU8> Pes;
	OaU64 PesPts = 0U;
	OaU64 PesDts = 0U;
	bool PesKeyframe = false;
	bool Live = false;
	bool Seekable = false;
};

namespace
{

// Forward declarations for MP4 box parsing
void ParseMoovBox(const OaU8* InData, OaU64 InSize, OaVideoStream& OutStream);
void ParseTrakBox(const OaU8* InData, OaU64 InSize, OaVideoStream& OutStream);
void ParseMdiaBox(const OaU8* InData, OaU64 InSize, OaVideoStream& OutStream);
void ParseMinfBox(const OaU8* InData, OaU64 InSize, OaVideoStream& OutStream);
void ParseStblBox(const OaU8* InData, OaU64 InSize, OaVideoStream& OutStream);
OaStatus ParseMoofBox(const OaU8* InData, OaU64 InSize, OaU64 InMoofOffset,
	OaU64 InMoofEnd,
	OaVideoStream& OutStream);

// Enough for more than 74 hours at 30 fps, while keeping hostile metadata from
// turning a tiny file into a multi-gigabyte host allocation.
constexpr OaU32 kMaxMp4TableEntries = 8U * 1024U * 1024U;

bool Mp4TableFits(OaU64 InPayloadSize, OaU64 InHeaderSize,
	OaU32 InEntryCount, OaU64 InEntryStride)
{
	return InEntryCount <= kMaxMp4TableEntries
		and InPayloadSize >= InHeaderSize
		and InEntryStride > 0U
		and static_cast<OaU64>(InEntryCount)
			<= (InPayloadSize - InHeaderSize) / InEntryStride;
}

// Helper: read 32-bit big-endian
inline OaU32 ReadU32BE(const OaU8* InPtr) {
	return (static_cast<OaU32>(InPtr[0]) << 24) |
	       (static_cast<OaU32>(InPtr[1]) << 16) |
	       (static_cast<OaU32>(InPtr[2]) << 8)  |
	       (static_cast<OaU32>(InPtr[3]));
}

// Helper: read 64-bit big-endian
inline OaU64 ReadU64BE(const OaU8* InPtr) {
	return (static_cast<OaU64>(InPtr[0]) << 56) |
	       (static_cast<OaU64>(InPtr[1]) << 48) |
	       (static_cast<OaU64>(InPtr[2]) << 40) |
	       (static_cast<OaU64>(InPtr[3]) << 32) |
	       (static_cast<OaU64>(InPtr[4]) << 24) |
	       (static_cast<OaU64>(InPtr[5]) << 16) |
	       (static_cast<OaU64>(InPtr[6]) << 8)  |
	       (static_cast<OaU64>(InPtr[7]));
}

// Helper: read 16-bit big-endian
inline OaU16 ReadU16BE(const OaU8* InPtr)
{
	return (static_cast<OaU16>(InPtr[0]) << 8) |
	       (static_cast<OaU16>(InPtr[1]));
}

struct EbmlElement {
	OaU64 Id = 0U;
	OaU64 Size = 0U;
	OaU64 DataOffset = 0U;
	bool UnknownSize = false;
};

bool ReadEbmlVint(std::FILE* InFile, bool InKeepMarker,
	OaU64& OutValue, OaU32& OutLength, bool& OutUnknown)
{
	const int firstInt = std::fgetc(InFile);
	if (firstInt == EOF) return false;
	const OaU8 first = static_cast<OaU8>(firstInt);
	OaU8 marker = 0x80U;
	OaU32 length = 1U;
	while (length <= 8U and (first & marker) == 0U) {
		marker >>= 1U;
		++length;
	}
	if (length > 8U or (InKeepMarker and length > 4U)) return false;
	OaU64 value = InKeepMarker ? first : static_cast<OaU64>(first & (marker - 1U));
	for (OaU32 i = 1U; i < length; ++i) {
		const int byte = std::fgetc(InFile);
		if (byte == EOF) return false;
		value = (value << 8U) | static_cast<OaU8>(byte);
	}
	const OaU64 unknown = length == 8U
		? 0x00FFFFFFFFFFFFFFULL
		: ((1ULL << (7U * length)) - 1ULL);
	OutValue = value;
	OutLength = length;
	OutUnknown = not InKeepMarker and value == unknown;
	return true;
}

bool ReadEbmlElement(std::FILE* InFile, OaU64 InLimit, EbmlElement& Out)
{
	const off_t start = ::ftello(InFile);
	if (start < 0 or static_cast<OaU64>(start) >= InLimit) return false;
	OaU32 idLength = 0U;
	OaU32 sizeLength = 0U;
	bool ignored = false;
	if (not ReadEbmlVint(InFile, true, Out.Id, idLength, ignored)
		or not ReadEbmlVint(InFile, false, Out.Size, sizeLength, Out.UnknownSize)) return false;
	const off_t data = ::ftello(InFile);
	if (data < 0) return false;
	Out.DataOffset = static_cast<OaU64>(data);
	if (Out.UnknownSize) Out.Size = InLimit - Out.DataOffset;
	return Out.DataOffset <= InLimit and Out.Size <= InLimit - Out.DataOffset;
}

OaU64 ReadEbmlUnsigned(std::FILE* InFile, OaU64 InSize)
{
	if (InSize == 0U or InSize > 8U) return 0U;
	OaU64 value = 0U;
	for (OaU64 i = 0U; i < InSize; ++i) {
		const int byte = std::fgetc(InFile);
		if (byte == EOF) return 0U;
		value = (value << 8U) | static_cast<OaU8>(byte);
	}
	return value;
}

OaString ReadEbmlString(std::FILE* InFile, OaU64 InSize)
{
	if (InSize > 1024U) return {};
	OaVec<char> bytes(static_cast<OaUsize>(InSize + 1U), '\0');
	if (InSize > 0U and std::fread(bytes.Data(), 1U, static_cast<OaUsize>(InSize), InFile)
		!= InSize) return {};
	return OaString(bytes.Data());
}

bool ParseAvcDecoderConfig(const OaU8* InData, OaU64 InSize,
	OaVideoStream::AvcConfig& Out)
{
	if (InData == nullptr or InSize < 7U or InData[0] != 1U) return false;
	Out = {};
	Out.LengthSize = static_cast<OaU8>((InData[4] & 0x03U) + 1U);
	OaU64 p = 6U;
	const OaU8 startCode[4] = {0U, 0U, 0U, 1U};
	const OaU8 spsCount = InData[5] & 0x1FU;
	for (OaU8 i = 0U; i < spsCount; ++i) {
		if (p + 2U > InSize) return false;
		const OaU16 size = ReadU16BE(InData + p); p += 2U;
		if (p + size > InSize) return false;
		for (OaU8 byte : startCode) Out.SpsAnnexB.PushBack(byte);
		for (OaU16 k = 0U; k < size; ++k) Out.SpsAnnexB.PushBack(InData[p + k]);
		p += size;
	}
	if (p >= InSize) return false;
	const OaU8 ppsCount = InData[p++];
	for (OaU8 i = 0U; i < ppsCount; ++i) {
		if (p + 2U > InSize) return false;
		const OaU16 size = ReadU16BE(InData + p); p += 2U;
		if (p + size > InSize) return false;
		for (OaU8 byte : startCode) Out.PpsAnnexB.PushBack(byte);
		for (OaU16 k = 0U; k < size; ++k) Out.PpsAnnexB.PushBack(InData[p + k]);
		p += size;
	}
	Out.Valid = not Out.SpsAnnexB.Empty() and not Out.PpsAnnexB.Empty();
	return Out.Valid;
}

struct MatroskaTrack {
	OaU64 Number = 0U;
	OaU64 Type = 0U;
	OaString Codec;
	OaVec<OaU8> CodecPrivate;
	OaU32 Width = 0U;
	OaU32 Height = 0U;
	OaU64 DefaultDurationNs = 0U;
};

void ParseMatroskaVideo(std::FILE* InFile, OaU64 InEnd, MatroskaTrack& Out)
{
	while (static_cast<OaU64>(::ftello(InFile)) < InEnd) {
		EbmlElement element;
		if (not ReadEbmlElement(InFile, InEnd, element)) break;
		if (element.Id == 0xB0U) Out.Width = static_cast<OaU32>(ReadEbmlUnsigned(InFile, element.Size));
		else if (element.Id == 0xBAU) Out.Height = static_cast<OaU32>(ReadEbmlUnsigned(InFile, element.Size));
		::fseeko(InFile, static_cast<off_t>(element.DataOffset + element.Size), SEEK_SET);
	}
}

MatroskaTrack ParseMatroskaTrack(std::FILE* InFile, OaU64 InEnd)
{
	MatroskaTrack track;
	while (static_cast<OaU64>(::ftello(InFile)) < InEnd) {
		EbmlElement element;
		if (not ReadEbmlElement(InFile, InEnd, element)) break;
		if (element.Id == 0xD7U) track.Number = ReadEbmlUnsigned(InFile, element.Size);
		else if (element.Id == 0x83U) track.Type = ReadEbmlUnsigned(InFile, element.Size);
		else if (element.Id == 0x86U) track.Codec = ReadEbmlString(InFile, element.Size);
		else if (element.Id == 0x63A2U and element.Size <= 16U * 1024U * 1024U) {
			track.CodecPrivate.Resize(static_cast<OaUsize>(element.Size));
			if (element.Size > 0U) {
				(void)std::fread(track.CodecPrivate.Data(), 1U, track.CodecPrivate.Size(), InFile);
			}
		} else if (element.Id == 0x23E383U) {
			track.DefaultDurationNs = ReadEbmlUnsigned(InFile, element.Size);
		} else if (element.Id == 0xE0U) {
			ParseMatroskaVideo(InFile, element.DataOffset + element.Size, track);
		}
		::fseeko(InFile, static_cast<off_t>(element.DataOffset + element.Size), SEEK_SET);
	}
	return track;
}

OaStatus ParseMatroskaCluster(std::FILE* InFile, OaU64 InEnd,
	OaU64 InVideoTrack, OaU64 InTimecodeScale, OaU64 InDefaultDurationNs,
	OaVideoStream& Out)
{
	OaU64 clusterTimecode = 0U;
	while (static_cast<OaU64>(::ftello(InFile)) < InEnd) {
		EbmlElement element;
		if (not ReadEbmlElement(InFile, InEnd, element)) break;
		if (element.Id == 0xE7U) {
			clusterTimecode = ReadEbmlUnsigned(InFile, element.Size);
		} else if (element.Id == 0xA3U and element.Size >= 4U) {
			const off_t blockStart = ::ftello(InFile);
			OaU64 track = 0U;
			OaU32 trackBytes = 0U;
			bool unknown = false;
			if (blockStart >= 0 and ReadEbmlVint(InFile, false, track, trackBytes, unknown)) {
				OaU8 header[3] = {};
				if (std::fread(header, 1U, sizeof(header), InFile) == sizeof(header)) {
					const OaI16 relative = static_cast<OaI16>(
						(static_cast<OaU16>(header[0]) << 8U) | header[1]);
					const OaU8 flags = header[2];
					const OaU64 headerSize = trackBytes + sizeof(header);
					if (track == InVideoTrack and (flags & 0x06U) == 0U
						and element.Size > headerSize) {
						const OaI64 signedTimestamp = static_cast<OaI64>(clusterTimecode) + relative;
						OaVideoStream::Sample sample;
						sample.Offset = element.DataOffset + headerSize;
						sample.Size = static_cast<OaU32>(element.Size - headerSize);
						sample.Dts = static_cast<OaU64>(std::max<OaI64>(0, signedTimestamp));
						sample.Duration = InDefaultDurationNs > 0U and InTimecodeScale > 0U
							? std::max<OaU64>(1U, InDefaultDurationNs / InTimecodeScale) : 1U;
						sample.IsKeyframe = (flags & 0x80U) != 0U;
						Out.Samples_.PushBack(sample);
					}
				}
			}
		}
		::fseeko(InFile, static_cast<off_t>(element.DataOffset + element.Size), SEEK_SET);
	}
	return OaStatus::Ok();
}

OaStatus ParseMatroskaFile(std::FILE* InFile, OaU64 InFileSize,
	OaStringView InPath, OaVideoStream& Out)
{
	::fseeko(InFile, 0, SEEK_SET);
	EbmlElement ebml;
	if (not ReadEbmlElement(InFile, InFileSize, ebml) or ebml.Id != 0x1A45DFA3U) {
		return OaStatus::Error(OaStatusCode::DataLoss, "Missing Matroska EBML header");
	}
	::fseeko(InFile, static_cast<off_t>(ebml.DataOffset + ebml.Size), SEEK_SET);
	EbmlElement segment;
	if (not ReadEbmlElement(InFile, InFileSize, segment) or segment.Id != 0x18538067U) {
		return OaStatus::Error(OaStatusCode::DataLoss, "Missing Matroska segment");
	}
	const OaU64 segmentEnd = segment.UnknownSize ? InFileSize : segment.DataOffset + segment.Size;
	OaU64 timecodeScale = 1'000'000U;
	MatroskaTrack selected;
	OaU32 trackCount = 0U;
	// Matroska normally writes Tracks before Cluster, but EBML does not make
	// that ordering a demuxer invariant. Index cluster payload ranges first and
	// parse them only after the selected video track is known.
	OaVec<OaU64> clusters;
	while (static_cast<OaU64>(::ftello(InFile)) < segmentEnd) {
		EbmlElement element;
		if (not ReadEbmlElement(InFile, segmentEnd, element)) break;
		if (element.Id == 0x1549A966U) { // Info
			const OaU64 end = element.DataOffset + element.Size;
			while (static_cast<OaU64>(::ftello(InFile)) < end) {
				EbmlElement info;
				if (not ReadEbmlElement(InFile, end, info)) break;
				if (info.Id == 0x2AD7B1U) timecodeScale = ReadEbmlUnsigned(InFile, info.Size);
				::fseeko(InFile, static_cast<off_t>(info.DataOffset + info.Size), SEEK_SET);
			}
		} else if (element.Id == 0x1654AE6BU) { // Tracks
			const OaU64 end = element.DataOffset + element.Size;
			while (static_cast<OaU64>(::ftello(InFile)) < end) {
				EbmlElement entry;
				if (not ReadEbmlElement(InFile, end, entry)) break;
				if (entry.Id == 0xAEU) {
					++trackCount;
					MatroskaTrack track = ParseMatroskaTrack(InFile, entry.DataOffset + entry.Size);
					if (selected.Number == 0U and track.Type == 1U) selected = OaStdMove(track);
				}
				::fseeko(InFile, static_cast<off_t>(entry.DataOffset + entry.Size), SEEK_SET);
			}
		} else if (element.Id == 0x1F43B675U) { // Cluster
			clusters.PushBack(element.DataOffset);
			clusters.PushBack(element.DataOffset + element.Size);
		}
		::fseeko(InFile, static_cast<off_t>(element.DataOffset + element.Size), SEEK_SET);
	}
	if (selected.Number == 0U) return OaStatus::Error(OaStatusCode::NotFound,
		"Matroska container has no video track");
	for (OaUsize i = 0U; i + 1U < clusters.Size(); i += 2U) {
		if (::fseeko(InFile, static_cast<off_t>(clusters[i]), SEEK_SET) != 0) {
			return OaStatus::Error(OaStatusCode::DataLoss, "Cannot seek Matroska cluster");
		}
		OA_RETURN_IF_ERROR(ParseMatroskaCluster(InFile, clusters[i + 1U],
			selected.Number, timecodeScale, selected.DefaultDurationNs, Out));
	}
	const std::string_view codec(selected.Codec.Data(), selected.Codec.Size());
	if (codec == "V_MPEG4/ISO/AVC") {
		Out.Info_.Codec = OaVideoCodec::H264;
		if (not ParseAvcDecoderConfig(selected.CodecPrivate.Data(),
			selected.CodecPrivate.Size(), Out.Avc_)) {
			return OaStatus::Error(OaStatusCode::DataLoss, "Matroska AVC track has invalid CodecPrivate");
		}
	} else if (codec == "V_MPEGH/ISO/HEVC") Out.Info_.Codec = OaVideoCodec::H265;
	else if (codec == "V_AV1") Out.Info_.Codec = OaVideoCodec::AV1;
	else if (codec == "V_VP9") Out.Info_.Codec = OaVideoCodec::VP9;
	else return OaStatus::Error(OaStatusCode::Unimplemented,
		"Matroska video codec is not supported by OA Vulkan Video");
	const std::string_view path(InPath.Data(), InPath.Size());
	Out.Info_.Kind = path.ends_with(".webm") ? OaContainerKind::WebM : OaContainerKind::Matroska;
	Out.Info_.Width = selected.Width;
	Out.Info_.Height = selected.Height;
	Out.Info_.TimebaseNum = timecodeScale;
	Out.Info_.TimebaseDen = 1'000'000'000ULL;
	Out.Info_.TrackCount = trackCount;
	if (Out.Samples_.Empty()) return OaStatus::Error(OaStatusCode::DataLoss,
		"Matroska video track contains no supported unlaced blocks");
	return OaStatus::Ok();
}

struct TsPayload {
	OaU16 Pid = 0x1FFFU;
	const OaU8* Data = nullptr;
	OaUsize Size = 0U;
	bool Start = false;
};

bool ParseTsPayload(const OaU8* InPacket, TsPayload& Out)
{
	if (InPacket[0] != 0x47U or (InPacket[1] & 0x80U) != 0U) return false;
	Out.Pid = static_cast<OaU16>(((InPacket[1] & 0x1FU) << 8U) | InPacket[2]);
	Out.Start = (InPacket[1] & 0x40U) != 0U;
	const OaU8 control = static_cast<OaU8>((InPacket[3] >> 4U) & 0x03U);
	if (control == 0U or control == 2U) return true;
	OaUsize offset = 4U;
	if (control == 3U) {
		offset += 1U + InPacket[4];
		if (offset > 188U) return false;
	}
	Out.Data = InPacket + offset;
	Out.Size = 188U - offset;
	return true;
}

const OaU8* TsPsiSection(const TsPayload& InPayload, OaUsize& OutSize)
{
	if (InPayload.Data == nullptr or InPayload.Size == 0U) return nullptr;
	OaUsize offset = 0U;
	if (InPayload.Start) {
		offset = 1U + InPayload.Data[0];
		if (offset >= InPayload.Size) return nullptr;
	}
	OutSize = InPayload.Size - offset;
	return InPayload.Data + offset;
}

bool ParseTsPat(const TsPayload& InPayload, OaU16& OutPmtPid)
{
	OaUsize size = 0U;
	const OaU8* section = TsPsiSection(InPayload, size);
	if (section == nullptr or size < 12U or section[0] != 0x00U) return false;
	const OaU16 sectionLength = static_cast<OaU16>(((section[1] & 0x0FU) << 8U) | section[2]);
	if (sectionLength + 3U > size or sectionLength < 9U) return false;
	const OaUsize end = 3U + sectionLength - 4U; // exclude CRC
	for (OaUsize p = 8U; p + 4U <= end; p += 4U) {
		const OaU16 program = ReadU16BE(section + p);
		if (program != 0U) {
			OutPmtPid = static_cast<OaU16>(((section[p + 2U] & 0x1FU) << 8U)
				| section[p + 3U]);
			return true;
		}
	}
	return false;
}

bool ParseTsPmt(const TsPayload& InPayload, OaU16& OutVideoPid, OaVideoCodec& OutCodec)
{
	OaUsize size = 0U;
	const OaU8* section = TsPsiSection(InPayload, size);
	if (section == nullptr or size < 16U or section[0] != 0x02U) return false;
	const OaU16 sectionLength = static_cast<OaU16>(((section[1] & 0x0FU) << 8U) | section[2]);
	if (sectionLength + 3U > size or sectionLength < 13U) return false;
	const OaU16 programInfoLength = static_cast<OaU16>(
		((section[10] & 0x0FU) << 8U) | section[11]);
	const OaUsize end = 3U + sectionLength - 4U;
	for (OaUsize p = 12U + programInfoLength; p + 5U <= end;) {
		const OaU8 streamType = section[p];
		const OaU16 pid = static_cast<OaU16>(((section[p + 1U] & 0x1FU) << 8U)
			| section[p + 2U]);
		const OaU16 infoLength = static_cast<OaU16>(((section[p + 3U] & 0x0FU) << 8U)
			| section[p + 4U]);
		if (streamType == 0x1BU or streamType == 0x24U) {
			OutVideoPid = pid;
			OutCodec = streamType == 0x1BU ? OaVideoCodec::H264 : OaVideoCodec::H265;
			return true;
		}
		p += 5U + infoLength;
	}
	return false;
}

OaU64 ParsePesTimestamp(const OaU8* InData)
{
	return (static_cast<OaU64>((InData[0] >> 1U) & 0x07U) << 30U)
		| (static_cast<OaU64>(InData[1]) << 22U)
		| (static_cast<OaU64>((InData[2] >> 1U) & 0x7FU) << 15U)
		| (static_cast<OaU64>(InData[3]) << 7U)
		| static_cast<OaU64>(InData[4] >> 1U);
}

bool H264AccessUnitIsKeyframe(OaSpan<const OaU8> InData)
{
	for (OaUsize i = 0U; i + 4U < InData.Size(); ++i) {
		if (InData[i] == 0U and InData[i + 1U] == 0U
			and ((InData[i + 2U] == 1U)
				or (InData[i + 2U] == 0U and InData[i + 3U] == 1U))) {
			const OaUsize header = i + (InData[i + 2U] == 1U ? 3U : 4U);
			if (header < InData.Size() and (InData[header] & 0x1FU) == 5U) return true;
		}
	}
	return false;
}

void ParseH264AccessUnitGeometry(OaSpan<const OaU8> InData, OaContainerInfo& Out)
{
	for (OaUsize i = 0U; i + 5U < InData.Size(); ++i) {
		if (InData[i] != 0U or InData[i + 1U] != 0U) continue;
		OaUsize header = 0U;
		if (InData[i + 2U] == 1U) header = i + 3U;
		else if (InData[i + 2U] == 0U and InData[i + 3U] == 1U) header = i + 4U;
		if (header == 0U or header >= InData.Size() or (InData[header] & 0x1FU) != 7U) continue;
		OaUsize end = InData.Size();
		for (OaUsize p = header + 1U; p + 3U < InData.Size(); ++p) {
			if (InData[p] == 0U and InData[p + 1U] == 0U
				and (InData[p + 2U] == 1U
					or (InData[p + 2U] == 0U and InData[p + 3U] == 1U))) {
				end = p;
				break;
			}
		}
		OaH264SpsData sps = {};
		if (not OaNalParser::ParseSPS(InData.Data() + header, end - header, sps)) return;
		const OaU32 frameFactor = sps.FrameMbsOnly ? 1U : 2U;
		OaU32 width = sps.PicWidthInMbs * 16U;
		OaU32 height = sps.PicHeightInMbs * 16U * frameFactor;
		const OaU32 cropUnitX = sps.ChromaFormatIdc == 0U or sps.ChromaFormatIdc == 3U ? 1U : 2U;
		const OaU32 chromaHeight = sps.ChromaFormatIdc == 1U ? 2U : 1U;
		const OaU32 cropUnitY = sps.ChromaFormatIdc == 0U ? frameFactor
			: chromaHeight * frameFactor;
		const OaU64 cropX = static_cast<OaU64>(sps.FrameCropLeftOffset
			+ sps.FrameCropRightOffset) * cropUnitX;
		const OaU64 cropY = static_cast<OaU64>(sps.FrameCropTopOffset
			+ sps.FrameCropBottomOffset) * cropUnitY;
		if (cropX < width) width -= static_cast<OaU32>(cropX);
		if (cropY < height) height -= static_cast<OaU32>(cropY);
		Out.Width = width;
		Out.Height = height;
		return;
	}
}

OaStatus ReadMpegTsPes(std::FILE* InFile, OaVideoStream::MediaImpl& InMedia,
	OaVideoCodec InCodec, OaVideoPacket& Out)
{
	OaU8 packet[188] = {};
	while (std::fread(packet, 1U, sizeof(packet), InFile) == sizeof(packet)) {
		TsPayload payload;
		if (not ParseTsPayload(packet, payload) or payload.Pid != InMedia.VideoPid
			or payload.Data == nullptr) continue;
		if (payload.Start and not InMedia.Pes.Empty()) {
			::fseeko(InFile, -static_cast<off_t>(sizeof(packet)), SEEK_CUR);
			break;
		}
		const OaU8* data = payload.Data;
		OaUsize size = payload.Size;
		if (payload.Start) {
			if (size < 9U or data[0] != 0U or data[1] != 0U or data[2] != 1U) continue;
			const OaU8 timestampFlags = static_cast<OaU8>((data[7] >> 6U) & 0x03U);
			const OaU8 headerLength = data[8];
			if (9U + headerLength > size) continue;
			if ((timestampFlags & 0x02U) != 0U and headerLength >= 5U) {
				InMedia.PesPts = ParsePesTimestamp(data + 9U);
				InMedia.PesDts = InMedia.PesPts;
			}
			if (timestampFlags == 0x03U and headerLength >= 10U) {
				InMedia.PesDts = ParsePesTimestamp(data + 14U);
			}
			data += 9U + headerLength;
			size -= 9U + headerLength;
		}
		const OaUsize old = InMedia.Pes.Size();
		InMedia.Pes.Resize(old + size);
		if (size > 0U) OaMemcpy(InMedia.Pes.Data() + old, data, size);
	}
	if (InMedia.Pes.Empty()) return OaStatus::Error(OaStatusCode::OutOfRange,
		"End of MPEG-TS stream");
	Out.Data = OaStdMove(InMedia.Pes);
	Out.PresentationTimestamp = InMedia.PesPts;
	Out.DecodeTimestamp = InMedia.PesDts;
	Out.IsKeyframe = InCodec == OaVideoCodec::H264
		? H264AccessUnitIsKeyframe(OaSpan<const OaU8>(Out.Data.Data(), Out.Data.Size()))
		: false;
	Out.TrackIndex = InMedia.VideoPid;
	InMedia.Pes.Clear();
	InMedia.PesPts = 0U;
	InMedia.PesDts = 0U;
	return OaStatus::Ok();
}

OaStatus InitMpegTs(std::FILE* InFile, OaU64 InFileSize,
	OaVideoStream::MediaImpl& OutMedia, OaContainerInfo& OutInfo)
{
	::fseeko(InFile, 0, SEEK_SET);
	OaU8 packet[188] = {};
	OaVideoCodec codec = OaVideoCodec::H264;
	const OaU64 scanPackets = std::min<OaU64>(InFileSize / sizeof(packet), 8192U);
	for (OaU64 i = 0U; i < scanPackets; ++i) {
		if (std::fread(packet, 1U, sizeof(packet), InFile) != sizeof(packet)) break;
		TsPayload payload;
		if (not ParseTsPayload(packet, payload)) continue;
		if (payload.Pid == 0U) (void)ParseTsPat(payload, OutMedia.PmtPid);
		else if (payload.Pid == OutMedia.PmtPid
			and ParseTsPmt(payload, OutMedia.VideoPid, codec)) break;
	}
	if (OutMedia.VideoPid == 0x1FFFU) return OaStatus::Error(OaStatusCode::DataLoss,
		"MPEG-TS PAT/PMT contains no supported H.264/H.265 video stream");
	OutMedia.Kind = OaVideoStream::MediaImpl::NativeKind::MpegTs;
	OutMedia.Live = false;
	OutMedia.Seekable = true;
	OutInfo.Kind = OaContainerKind::MpegTs;
	OutInfo.Codec = codec;
	OutInfo.TimebaseNum = 1U;
	OutInfo.TimebaseDen = 90'000U;
	OutInfo.TrackCount = 1U;
	::fseeko(InFile, 0, SEEK_SET);
	OaVideoPacket first;
	if (ReadMpegTsPes(InFile, OutMedia, codec, first).IsOk()) {
		if (codec == OaVideoCodec::H264) {
			ParseH264AccessUnitGeometry(
				OaSpan<const OaU8>(first.Data.Data(), first.Data.Size()), OutInfo);
		}
	}
	OutMedia.Pes.Clear();
	OutMedia.PesPts = 0U;
	OutMedia.PesDts = 0U;
	::fseeko(InFile, 0, SEEK_SET);
	return OaStatus::Ok();
}

// MP4 box header
struct BoxHeader
{
	OaU64 Size = 0;
	OaU32 Type = 0;
};

// Parse box header (returns true if extended size)
bool ParseBoxHeader(const OaU8* InData, OaU64 InOffset, OaU64 InDataSize, BoxHeader& OutHeader)
{
	if (InOffset + 8 > InDataSize) {
		return false;
	}
	
	OutHeader.Size = ReadU32BE(InData + InOffset);
	OutHeader.Type = ReadU32BE(InData + InOffset + 4);
	
	// Extended size (size == 1 means 64-bit size follows)
	if (OutHeader.Size == 1) {
		if (InOffset + 16 > InDataSize) {
			return false;
		}
		OutHeader.Size = ReadU64BE(InData + InOffset + 8);
	}
	
	return true;
}

bool ReadTopLevelBoxHeader(
	std::FILE* InFile,
	OaU64 InOffset,
	OaU64 InFileSize,
	BoxHeader& OutHeader,
	OaU64& OutHeaderSize)
{
	if (InFile == nullptr || InOffset + 8U > InFileSize
		|| ::fseeko(InFile, static_cast<off_t>(InOffset), SEEK_SET) != 0) return false;
	OaU8 bytes[16] = {};
	if (std::fread(bytes, 1U, 8U, InFile) != 8U) return false;
	OutHeader.Size = ReadU32BE(bytes);
	OutHeader.Type = ReadU32BE(bytes + 4U);
	OutHeaderSize = 8U;
	if (OutHeader.Size == 1U) {
		if (InOffset + 16U > InFileSize || std::fread(bytes + 8U, 1U, 8U, InFile) != 8U) {
			return false;
		}
		OutHeader.Size = ReadU64BE(bytes + 8U);
		OutHeaderSize = 16U;
	} else if (OutHeader.Size == 0U) {
		OutHeader.Size = InFileSize - InOffset;
	}
	return OutHeader.Size >= OutHeaderSize && InOffset + OutHeader.Size <= InFileSize;
}

bool IsVideoMdia(const OaU8* InData, OaU64 InSize)
{
	OaU64 offset = 0U;
	while (offset + 8U <= InSize) {
		const OaU32 size = ReadU32BE(InData + offset);
		const OaU32 type = ReadU32BE(InData + offset + 4U);
		if (size < 8U or offset + size > InSize) return false;
		// HandlerBox: header(8), version/flags(4), pre_defined(4), handler_type(4).
		if (type == OaMp4Box::Hdlr and size >= 20U) {
			return ReadU32BE(InData + offset + 16U) == 0x76696465U; // vide
		}
		offset += size;
	}
	return false;
}

// Detect codec from handler type
OaVideoCodec DetectCodecFromHandler(OaU32 InHandlerType)
{
	// Common handler types: vide, soun, text, etc.
	// For video, we need to look at stsd sample entry
	return OaVideoCodec::H264;  // Default for now
}

}  // namespace


void OaVideoStream::Reset_() noexcept
{
	if (Media_) {
		Media_.Reset();
	}
	if (File_ != nullptr) std::fclose(File_);
	File_ = nullptr;
	FileSize_ = 0U;
	Info_ = {};
	SampleData_.Clear();
	CurrentOffset_ = 0;
	Eos_ = false;
	Samples_.Clear();
	CurrentSampleIndex_ = 0;
	Avc_ = {};
	Hvc_ = {};
	Av1_ = {};
	Fragment_ = {};
	NeedParameterSets_ = true;
	BufferedPictureNals_.Clear();
	BufferedTimestamp_ = 0U;
	BufferedIsKeyframe_ = false;
	Uri_.Clear();
	Options_ = {};
	Stats_ = {};
	LastDecodeTimestamp_ = 0U;
	HasLastDecodeTimestamp_ = false;
}


OaVideoStream::OaVideoStream(OaVideoStream&& InOther) noexcept
	: Samples_(std::move(InOther.Samples_))
	, Info_(InOther.Info_)
	, Avc_(std::move(InOther.Avc_))
	, Hvc_(std::move(InOther.Hvc_))
	, Av1_(std::move(InOther.Av1_))
	, Fragment_(InOther.Fragment_)
	, Media_(OaStdMove(InOther.Media_))
	, Uri_(OaStdMove(InOther.Uri_))
	, Options_(OaStdMove(InOther.Options_))
	, Stats_(InOther.Stats_)
	, LastDecodeTimestamp_(InOther.LastDecodeTimestamp_)
	, HasLastDecodeTimestamp_(InOther.HasLastDecodeTimestamp_)
	, File_(InOther.File_)
	, FileSize_(InOther.FileSize_)
	, SampleData_(std::move(InOther.SampleData_))
	, CurrentOffset_(InOther.CurrentOffset_)
	, Eos_(InOther.Eos_)
	, CurrentSampleIndex_(InOther.CurrentSampleIndex_)
	, NeedParameterSets_(InOther.NeedParameterSets_)
	, BufferedPictureNals_(OaStdMove(InOther.BufferedPictureNals_))
	, BufferedTimestamp_(InOther.BufferedTimestamp_)
	, BufferedIsKeyframe_(InOther.BufferedIsKeyframe_)
{
	InOther.File_ = nullptr;
	InOther.Reset_();
}


OaVideoStream& OaVideoStream::operator=(OaVideoStream&& InOther) noexcept
{
	if (this != &InOther) {
		Destroy();
		Info_ = InOther.Info_;
		Avc_ = std::move(InOther.Avc_);
		Hvc_ = std::move(InOther.Hvc_);
		Av1_ = std::move(InOther.Av1_);
		Fragment_ = InOther.Fragment_;
		Media_ = OaStdMove(InOther.Media_);
		Uri_ = OaStdMove(InOther.Uri_);
		Options_ = OaStdMove(InOther.Options_);
		Stats_ = InOther.Stats_;
		LastDecodeTimestamp_ = InOther.LastDecodeTimestamp_;
		HasLastDecodeTimestamp_ = InOther.HasLastDecodeTimestamp_;
		File_ = InOther.File_;
		FileSize_ = InOther.FileSize_;
		SampleData_ = std::move(InOther.SampleData_);
		CurrentOffset_ = InOther.CurrentOffset_;
		Eos_ = InOther.Eos_;
		Samples_ = std::move(InOther.Samples_);
		CurrentSampleIndex_ = InOther.CurrentSampleIndex_;
		NeedParameterSets_ = InOther.NeedParameterSets_;
		BufferedPictureNals_ = OaStdMove(InOther.BufferedPictureNals_);
		BufferedTimestamp_ = InOther.BufferedTimestamp_;
		BufferedIsKeyframe_ = InOther.BufferedIsKeyframe_;
		InOther.File_ = nullptr;
		InOther.Reset_();
	}
	return *this;
}


OaVideoStream::~OaVideoStream()
{
	Destroy();
}


void OaVideoStream::Destroy()
{
	Reset_();
}


OaResult<OaContainerInfo> OaVideoStream::Probe(const char* InPath)
{
	OaContainerInfo info = {};
	
	if (InPath == nullptr || InPath[0] == '\0') {
		return OaStatus::InvalidArgument("OaVideoStream::Probe requires a path");
	}
	std::FILE* file = std::fopen(InPath, "rb");
	if (file == nullptr) {
		auto media = OpenMedia_(InPath);
		return media.IsOk() ? OaResult<OaContainerInfo>(media->GetInfo())
			: OaResult<OaContainerInfo>(media.GetStatus());
	}
	OaU8 bytes[12] = {};
	const OaUsize read = std::fread(bytes, 1U, sizeof(bytes), file);
	std::fclose(file);
	if (read < sizeof(bytes)) {
		return OaStatus::Error("File too small to be a valid container");
	}
	
	// Check for ftyp box
	OaU32 ftypSize = ReadU32BE(bytes);
	OaU32 ftypType = ReadU32BE(bytes + 4);
	
	if (ftypType == OaMp4Box::Ftyp) {
		info.Kind = OaContainerKind::Mp4;
		// Read major brand (bytes 8-11)
		OaU32 majorBrand = ReadU32BE(bytes + 8);
		(void)ftypSize;
		(void)majorBrand;
		// Common brands: isom, mp42, avc1, etc.
		// For now, just mark as MP4
	} else {
		auto media = OpenMedia_(InPath);
		return media.IsOk() ? OaResult<OaContainerInfo>(media->GetInfo())
			: OaResult<OaContainerInfo>(media.GetStatus());
	}
	
	return info;
}

OaResult<OaVideoStream> OaVideoStream::Open(OaStringView InUri)
{
	return Open(InUri, {});
}

OaResult<OaVideoStream> OaVideoStream::Open(
	OaStringView InUri, const OaVideoStreamOptions& InOptions)
{
	if (InUri.Empty()) {
		return OaStatus::InvalidArgument("OaVideoStream::Open requires a URI");
	}
	const OaString uri(InUri);
	if (std::string_view(uri.Data(), uri.Size()).find("://") != std::string_view::npos) {
		return OpenMedia_(InUri, InOptions);
	}
	// Keep the native incremental MP4 path for local files. Network schemes
	// and non-MP4 containers go through the live-aware media backend.
	auto local = OpenFile(uri.CStr());
	if (local.IsOk()) {
		local->Uri_ = uri;
		local->Options_ = InOptions;
		return local;
	}
	return OpenMedia_(InUri, InOptions);
}

OaResult<OaVideoStream> OaVideoStream::OpenMedia_(
	OaStringView InUri, const OaVideoStreamOptions& InOptions)
{
	const OaString uri(InUri);
	const std::string_view uriView(uri.Data(), uri.Size());
	if (uriView.find("://") == std::string_view::npos) {
		std::FILE* file = std::fopen(uri.CStr(), "rb");
		if (file != nullptr) {
			OaU8 signature[189] = {};
			const OaUsize signatureSize = std::fread(signature, 1U, sizeof(signature), file);
			if (::fseeko(file, 0, SEEK_END) == 0) {
				const off_t end = ::ftello(file);
				if (end >= 0 and signatureSize >= 4U
					and ReadU32BE(signature) == 0x1A45DFA3U) {
					OaVideoStream stream;
					stream.Uri_ = uri;
					stream.Options_ = InOptions;
					stream.File_ = file;
					stream.FileSize_ = static_cast<OaU64>(end);
					OaStatus parsed = ParseMatroskaFile(
						file, stream.FileSize_, InUri, stream);
					if (not parsed.IsOk()) return parsed;
					stream.Eos_ = false;
					return OaStdMove(stream);
				} else if (end >= 188 and signatureSize >= 1U and signature[0] == 0x47U
					and (signatureSize < sizeof(signature) or signature[188] == 0x47U)) {
					OaVideoStream stream;
					stream.Uri_ = uri;
					stream.Options_ = InOptions;
					stream.File_ = file;
					stream.FileSize_ = static_cast<OaU64>(end);
					stream.Media_ = OaMakeUniquePtr<MediaImpl>();
					OaStatus initialized = InitMpegTs(
						file, stream.FileSize_, *stream.Media_, stream.Info_);
					if (not initialized.IsOk()) return initialized;
					stream.Eos_ = false;
					return OaStdMove(stream);
				}
			}
			std::fclose(file);
		}
	}
	return OaStatus::Error(OaStatusCode::Unimplemented,
		uriView.find("://") != std::string_view::npos
			? "OA-native network transports are not implemented yet"
			: "Unsupported native media container or codec");
}


OaResult<OaVideoStream> OaVideoStream::OpenFile(const char* InPath)
{
	OaVideoStream stream;
	
	// Probe container first
	auto probeResult = Probe(InPath);
	if (!probeResult.IsOk()) {
		return OpenMedia_(InPath != nullptr ? OaStringView(InPath) : OaStringView());
	}
	const auto& info = probeResult.GetValue();
	if (info.Kind != OaContainerKind::Mp4) return OpenMedia_(InPath);
	stream.Info_.Kind = info.Kind;
	stream.Info_.Codec = info.Codec;
	stream.Info_.Width = info.Width;
	stream.Info_.Height = info.Height;
	stream.Info_.FrameRate = info.FrameRate;
	stream.Info_.Duration = info.Duration;
	stream.Info_.TimebaseNum = info.TimebaseNum;
	stream.Info_.TimebaseDen = info.TimebaseDen;
	stream.Info_.TrackCount = info.TrackCount;
	
	stream.File_ = std::fopen(InPath, "rb");
	if (stream.File_ == nullptr) {
		return OaStatus::Error(OaStatusCode::NotFound, "Cannot open video container");
	}
	if (::fseeko(stream.File_, 0, SEEK_END) != 0) {
		return OaStatus::Error("Cannot seek video container");
	}
	const off_t end = ::ftello(stream.File_);
	if (end < 0) return OaStatus::Error("Cannot determine video container size");
	stream.FileSize_ = static_cast<OaU64>(end);

	// Scan only top-level headers and load the compact moov metadata box. Media
	// samples stay on disk and are read into bounded per-packet scratch later.
	OaU64 offset = 0U;
	while (offset + 8U <= stream.FileSize_) {
		BoxHeader header;
		OaU64 headerSize = 0U;
		if (!ReadTopLevelBoxHeader(
			stream.File_, offset, stream.FileSize_, header, headerSize)) break;
		if (header.Type == OaMp4Box::Moov) {
			const OaU64 payloadSize = header.Size - headerSize;
			if (payloadSize > static_cast<OaU64>(SIZE_MAX)) {
				return OaStatus::Error("MP4 moov metadata exceeds host address space");
			}
			OaVec<OaU8> metadata(static_cast<OaUsize>(payloadSize));
			if (::fseeko(stream.File_, static_cast<off_t>(offset + headerSize), SEEK_SET) != 0
				|| std::fread(metadata.Data(), 1U, metadata.Size(), stream.File_) != metadata.Size()) {
				return OaStatus::Error("Cannot read MP4 moov metadata");
			}
			ParseMoovBox(metadata.Data(), metadata.Size(), stream);
		} else if (header.Type == OaMp4Box::Moof) {
			const OaU64 payloadSize = header.Size - headerSize;
			if (payloadSize > static_cast<OaU64>(SIZE_MAX)) {
				return OaStatus::Error("MP4 fragment metadata exceeds host address space");
			}
			OaVec<OaU8> fragment(static_cast<OaUsize>(payloadSize));
			if (::fseeko(stream.File_, static_cast<off_t>(offset + headerSize), SEEK_SET) != 0
				or std::fread(fragment.Data(), 1U, fragment.Size(), stream.File_) != fragment.Size()) {
				return OaStatus::Error("Cannot read MP4 fragment metadata");
			}
			OA_RETURN_IF_ERROR(ParseMoofBox(fragment.Data(), fragment.Size(), offset,
				offset + header.Size, stream));
		}
		offset += header.Size;
	}
	
	// If we found samples, we're ready to read
	if (!stream.Samples_.Empty()) {
		stream.Eos_ = false;
		// Probe-time FrameRate is often unreliable for MP4; recompute from
		// the parsed sample table now that we have it. duration_ticks =
		// last_sample.Dts + last.Duration; frame_rate = N * timebase / duration.
		// Falls back to 30 fps if anything is degenerate.
		if (stream.Info_.FrameRate == 0 and stream.Info_.TimebaseDen > 0) {
			const auto& last = stream.Samples_[stream.Samples_.Size() - 1];
			const OaU64 totalTicks = last.Dts + last.Duration;
			if (totalTicks > 0) {
				const double seconds = static_cast<double>(totalTicks) /
					static_cast<double>(stream.Info_.TimebaseDen);
				if (seconds > 0.0) {
					const double fps = static_cast<double>(stream.Samples_.Size()) / seconds;
					stream.Info_.FrameRate = static_cast<OaU32>(fps + 0.5);
				}
			}
		}
	} else return OaStatus::Error(OaStatusCode::DataLoss,
		"MP4 contains no readable video samples");

	return stream;
}


OaStatus OaVideoStream::ReadNextPacket(OaVideoPacket& OutPacket)
{
	if (Media_) return ReadMediaPacket_(OutPacket);
	if (Eos_ || CurrentSampleIndex_ >= Samples_.Size()) {
		Eos_ = true;
		return OaStatus::Error("End of stream");
	}

	const Sample& sample = Samples_[CurrentSampleIndex_];
	if (sample.Offset + sample.Size > FileSize_) {
		Eos_ = true;
		return OaStatus::Error("Sample offset/size exceeds file bounds");
	}
	SampleData_.Resize(sample.Size);
	if (::fseeko(File_, static_cast<off_t>(sample.Offset), SEEK_SET) != 0
		|| std::fread(SampleData_.Data(), 1U, sample.Size, File_) != sample.Size) {
		Eos_ = true;
		return OaStatus::Error("Failed to read compressed video sample");
	}

	OutPacket.Data.Clear();
	OutPacket.Data.Reserve(sample.Size + 32);  // reserve a little extra for SPS/PPS

	// Check if we have buffered picture NAL units from a previous sample
	if (!BufferedPictureNals_.Empty()) {
		OutPacket.Data = OaStdMove(BufferedPictureNals_);
		OutPacket.PresentationTimestamp = BufferedTimestamp_;
		OutPacket.DecodeTimestamp = BufferedTimestamp_;
		OutPacket.IsKeyframe = BufferedIsKeyframe_;
		OutPacket.TrackIndex = 0;
		BufferedPictureNals_.Clear();
		return OaStatus::Ok();
	}

	// MP4 stores NAL units length-prefixed; we need Annex-B (00 00 00 01 +
	// payload) for the Vulkan Video decoder. Also prepend SPS+PPS on the
	// first IDR so the decoder can bring up parameter sets.
	const OaU8* src = SampleData_.Data();
	const OaUsize srcSize = sample.Size;

	const bool isMp4 = Info_.Kind == OaContainerKind::Mp4;
	const bool hasLengthPrefixedCodecConfig = isMp4
		or Info_.Kind == OaContainerKind::Matroska
		or Info_.Kind == OaContainerKind::WebM;
	const bool lengthPrefixedNal = hasLengthPrefixedCodecConfig
		&& (Info_.Codec == OaVideoCodec::H264 || Info_.Codec == OaVideoCodec::H265);
	// AV1 MP4 samples are OBU temporal units; VP9 samples are raw frame
	// bitstreams. Neither uses H.264/HEVC length-prefixed NAL packing.
	const bool rawMp4Sample = isMp4
		&& (Info_.Codec == OaVideoCodec::AV1 || Info_.Codec == OaVideoCodec::VP9);

	const bool prependH264Ps = hasLengthPrefixedCodecConfig && Avc_.Valid && sample.IsKeyframe
		&& NeedParameterSets_ && Info_.Codec == OaVideoCodec::H264;
	const bool prependH265Ps = hasLengthPrefixedCodecConfig && Hvc_.Valid && sample.IsKeyframe
		&& NeedParameterSets_ && Info_.Codec == OaVideoCodec::H265;
	// AV1 MP4 samples omit the sequence header (it lives in av1C). Prepend the
	// cached configOBUs to every keyframe temporal unit so the parser always
	// sees a sequence-header OBU. Inter frames rely on the parser's cached
	// sequence header from the preceding keyframe.
	const bool prependAv1Cfg = isMp4 && Av1_.Valid && sample.IsKeyframe
		&& Info_.Codec == OaVideoCodec::AV1;

	// For H.265, we need to separate parameter-set-only NAL units from picture NAL units
	// to ensure parameter sets are uploaded to the Vulkan session before slices are decoded
	const bool splitH265Ps = lengthPrefixedNal
		&& Info_.Codec == OaVideoCodec::H265 && !rawMp4Sample;

	if (prependH264Ps) {
		for (OaUsize i = 0; i < Avc_.SpsAnnexB.Size(); ++i) {
			OutPacket.Data.PushBack(Avc_.SpsAnnexB[i]);
		}
		for (OaUsize i = 0; i < Avc_.PpsAnnexB.Size(); ++i) {
			OutPacket.Data.PushBack(Avc_.PpsAnnexB[i]);
		}
		NeedParameterSets_ = false;
	}
	if (prependH265Ps) {
		for (OaUsize i = 0; i < Hvc_.VpsAnnexB.Size(); ++i) {
			OutPacket.Data.PushBack(Hvc_.VpsAnnexB[i]);
		}
		for (OaUsize i = 0; i < Hvc_.SpsAnnexB.Size(); ++i) {
			OutPacket.Data.PushBack(Hvc_.SpsAnnexB[i]);
		}
		for (OaUsize i = 0; i < Hvc_.PpsAnnexB.Size(); ++i) {
			OutPacket.Data.PushBack(Hvc_.PpsAnnexB[i]);
		}
		NeedParameterSets_ = false;
	}

	if (rawMp4Sample) {
		if (prependAv1Cfg) {
			for (OaUsize i = 0; i < Av1_.ConfigObus.Size(); ++i) {
				OutPacket.Data.PushBack(Av1_.ConfigObus[i]);
			}
		}
		for (OaUsize i = 0; i < srcSize; ++i) {
			OutPacket.Data.PushBack(src[i]);
		}
	} else {
	const OaU8 lengthSize = (Info_.Codec == OaVideoCodec::H265) ? Hvc_.LengthSize : Avc_.LengthSize;
	if (lengthPrefixedNal && lengthSize > 0) {
		// Walk the length-prefixed NAL list and rewrite each as Annex-B.
		const OaU8 nls = lengthSize;
		OaUsize p = 0;
		const OaU8 startCode[4] = { 0, 0, 0, 1 };
		bool hasPictureNal = false;

		while (p + nls <= srcSize) {
			OaU32 nalLen = 0;
			for (OaU8 b = 0; b < nls; ++b) {
				nalLen = (nalLen << 8) | static_cast<OaU32>(src[p + b]);
			}
			p += nls;
			if (p + nalLen > srcSize || nalLen == 0) {
				break;
			}

			// Check NAL type for H.265
			bool isParamSet = false;
			if (splitH265Ps && nalLen > 0) {
				OaU8 nalType = (src[p] >> 1) & 0x3F;
				if (nalType == 32 || nalType == 33 || nalType == 34) {  // VPS, SPS, PPS
					isParamSet = true;
				}
			}

			// Write start code and NAL data
			for (auto byte : startCode) {
				OutPacket.Data.PushBack(byte);
			}
			for (OaU32 k = 0; k < nalLen; ++k) {
				OutPacket.Data.PushBack(src[p + k]);
			}

			// For H.265, if we have both parameter sets and picture NALs in the same sample,
			// return parameter sets first and buffer picture NALs for the next call
			if (splitH265Ps) {
				if (!isParamSet) {
					hasPictureNal = true;
				}
			}

			p += nalLen;
		}

		// If we have both parameter sets and picture NALs, split them
		if (splitH265Ps && hasPictureNal && OutPacket.Data.Size() > 0) {
			// Scan backwards to find the last parameter set
			OaI32 lastParamSetEnd = -1;
			OaI32 lastStartCode = -1;
			for (OaI32 i = static_cast<OaI32>(OutPacket.Data.Size()) - 4; i >= 0; --i) {
				if (OutPacket.Data[i] == 0 && OutPacket.Data[i + 1] == 0 &&
					OutPacket.Data[i + 2] == 0 && OutPacket.Data[i + 3] == 1) {
					lastStartCode = i;
					if (i + 4 < static_cast<OaI32>(OutPacket.Data.Size())) {
						OaU8 nalType = (OutPacket.Data[i + 4] >> 1) & 0x3F;
						if (nalType == 32 || nalType == 33 || nalType == 34) {  // VPS, SPS, PPS
							lastParamSetEnd = i;
							break;
						}
					}
				}
			}

			// If we found parameter sets followed by picture NALs, split them
			if (lastParamSetEnd >= 0 && lastStartCode > lastParamSetEnd) {
				// Move picture NALs to buffer
				BufferedPictureNals_.Clear();
				BufferedPictureNals_.Reserve(OutPacket.Data.Size() - lastStartCode);
				for (OaUsize i = static_cast<OaUsize>(lastStartCode); i < OutPacket.Data.Size(); ++i) {
					BufferedPictureNals_.PushBack(OutPacket.Data[i]);
				}
				OutPacket.Data.Resize(lastStartCode);

				// Store timestamp and keyframe flag for buffered picture NALs
				BufferedTimestamp_ = sample.Dts + static_cast<OaU64>(sample.CtsOffset);
				BufferedIsKeyframe_ = sample.IsKeyframe;

				// Return parameter sets only (no timestamp needed for parameter-set-only packet)
				OutPacket.PresentationTimestamp = 0;
				OutPacket.DecodeTimestamp = 0;
				OutPacket.IsKeyframe = false;
				OutPacket.TrackIndex = 0;
				// Sample is fully consumed; picture NALs are buffered for the next call.
				++CurrentSampleIndex_;
				return OaStatus::Ok();
			}
		}
	} else {
		// Raw Annex-B (no container): bytes are already start-code prefixed.
		for (OaUsize i = 0; i < srcSize; ++i) {
			OutPacket.Data.PushBack(src[i]);
		}
	}
	}

	OutPacket.PresentationTimestamp = sample.Dts + static_cast<OaU64>(sample.CtsOffset);
	OutPacket.DecodeTimestamp = sample.Dts;
	OutPacket.IsKeyframe = sample.IsKeyframe;
	OutPacket.TrackIndex = 0;

	++CurrentSampleIndex_;
	return OaStatus::Ok();
}

OaStatus OaVideoStream::ReadMediaPacket_(OaVideoPacket& OutPacket)
{
	if (Media_ and Media_->Kind == MediaImpl::NativeKind::MpegTs) {
		if (Eos_) return OaStatus::Error(OaStatusCode::OutOfRange, "End of MPEG-TS stream");
		OaStatus status = ReadMpegTsPes(File_, *Media_, Info_.Codec, OutPacket);
		if (not status.IsOk()) {
			Eos_ = status.GetCode() == OaStatusCode::OutOfRange;
			return status;
		}
		++CurrentSampleIndex_;
		return OaStatus::Ok();
	}
	(void)OutPacket;
	return OaStatus::Error(OaStatusCode::Unimplemented,
		"Native media packet backend is unavailable for this source");
}

OaStatus OaVideoStream::ReconnectMedia_()
{
	return OaStatus::Unimplemented("OA-native live reconnect is not implemented yet");
}


OaStatus OaVideoStream::Seek(OaU64 InTimestamp)
{
	if (Media_) return SeekMedia_(InTimestamp);
	if (Samples_.Empty()) {
		return OaStatus::Error("OaVideoStream::Seek: no samples parsed");
	}

	// Walk the sample table backwards from the closest sample whose PTS is
	// <= InTimestamp, snapping to the nearest preceding keyframe. With no
	// keyframe table (`stss`) at all, every sample is treated as a keyframe
	// (H.264 raw or all-IDR encoded streams).
	OaUsize target = 0;
	for (OaUsize i = 0; i < Samples_.Size(); ++i) {
		const OaU64 pts = Samples_[i].Dts + static_cast<OaU64>(Samples_[i].CtsOffset);
		if (pts > InTimestamp) {
			break;
		}
		target = i;
	}
	while (target > 0 && not Samples_[target].IsKeyframe) {
		--target;
	}

	CurrentSampleIndex_ = target;
	Eos_ = false;
	NeedParameterSets_ = true;  // re-emit SPS+PPS at next keyframe
	return OaStatus::Ok();
}

OaStatus OaVideoStream::SeekMedia_(OaU64 InTimestamp)
{
	if (Media_ and Media_->Kind == MediaImpl::NativeKind::MpegTs) {
		if (InTimestamp != 0U) return OaStatus::Error(OaStatusCode::Unimplemented,
			"Indexed MPEG-TS seek is not implemented; seek to zero is supported");
		if (File_ == nullptr or ::fseeko(File_, 0, SEEK_SET) != 0) {
			return OaStatus::Error(OaStatusCode::Unavailable, "Could not rewind MPEG-TS stream");
		}
		Media_->Pes.Clear();
		Media_->PesPts = 0U;
		Media_->PesDts = 0U;
		CurrentSampleIndex_ = 0U;
		Eos_ = false;
		return OaStatus::Ok();
	}
	(void)InTimestamp;
	return OaStatus::Error(OaStatusCode::Unimplemented,
		"Indexed seek is not implemented for this native media source");
}

bool OaVideoStream::IsLive() const noexcept
{
	return Media_ and Media_->Live;
}

bool OaVideoStream::IsSeekable() const noexcept
{
	return Media_ ? Media_->Seekable : File_ != nullptr;
}


OaVideoProfile OaVideoStream::GetVideoProfile() const
{
	OaVideoProfile profile = {};
	profile.Codec = Info_.Codec;
	profile.Width = Info_.Width;
	profile.Height = Info_.Height;
	return profile;
}


namespace
{

// Parse moov box (movie metadata)
void ParseMoovBox(const OaU8* InData, OaU64 InSize, OaVideoStream& OutStream)
{
	OaU64 offset = 0;
	while (offset + 8 <= InSize) {
		OaU32 boxSize = ReadU32BE(InData + offset);
		OaU32 boxType = ReadU32BE(InData + offset + 4);
		
		if (boxSize == 1) {
			// Extended size - skip for now
			break;
		}
		
		if (boxSize < 8 || offset + boxSize > InSize) {
			break;
		}
		
		// Parse trak box (track)
		if (boxType == OaMp4Box::Trak) {
			ParseTrakBox(InData + offset + 8, boxSize - 8, OutStream);
		}
		else if (boxType == OaMp4Box::Mvex) {
			const OaU8* mvex = InData + offset + 8U;
			const OaU64 mvexSize = boxSize - 8U;
			OaU64 child = 0U;
			while (child + 8U <= mvexSize) {
				const OaU32 childSize = ReadU32BE(mvex + child);
				const OaU32 childType = ReadU32BE(mvex + child + 4U);
				if (childSize < 8U or child + childSize > mvexSize) break;
				// TrackExtendsBox: version/flags, track_ID, default description,
				// default duration, default size, default flags.
				if (childType == OaMp4Box::Trex and childSize >= 32U) {
					const OaU8* trex = mvex + child + 8U;
					const OaU32 trackId = ReadU32BE(trex + 4U);
					if (OutStream.Fragment_.TrackId == 0U
						or trackId == OutStream.Fragment_.TrackId) {
						OutStream.Fragment_.TrackId = trackId;
						OutStream.Fragment_.DefaultSampleDuration = ReadU32BE(trex + 12U);
						OutStream.Fragment_.DefaultSampleSize = ReadU32BE(trex + 16U);
						OutStream.Fragment_.DefaultSampleFlags = ReadU32BE(trex + 20U);
					}
				}
				child += childSize;
			}
		}
		
		offset += boxSize;
	}
}

// Parse trak box (track metadata)
void ParseTrakBox(const OaU8* InData, OaU64 InSize, OaVideoStream& OutStream)
{
	bool videoTrack = false;
	OaU32 trackId = 0U;
	for (OaU64 scan = 0U; scan + 8U <= InSize;) {
		const OaU32 size = ReadU32BE(InData + scan);
		const OaU32 type = ReadU32BE(InData + scan + 4U);
		if (size < 8U or scan + size > InSize) break;
		if (type == OaMp4Box::Mdia) {
			videoTrack = IsVideoMdia(InData + scan + 8U, size - 8U);
		} else if (type == OaMp4Box::Tkhd and size >= 8U + 16U) {
			const OaU8* tkhd = InData + scan + 8U;
			const OaU8 version = tkhd[0];
			const OaU64 idOffset = version == 1U ? 20U : 12U;
			if (size - 8U >= idOffset + 4U) trackId = ReadU32BE(tkhd + idOffset);
		}
		scan += size;
	}
	if (not videoTrack) return;
	if (trackId != 0U) OutStream.Fragment_.TrackId = trackId;

	OaU64 offset = 0;
	while (offset + 8 <= InSize) {
		OaU32 boxSize = ReadU32BE(InData + offset);
		OaU32 boxType = ReadU32BE(InData + offset + 4);
		
		if (boxSize < 8 || offset + boxSize > InSize) {
			break;
		}
		
		// Parse mdia box (media)
		if (boxType == OaMp4Box::Mdia) {
			const OaU8* mdia = InData + offset + 8U;
			const OaU64 mdiaSize = boxSize - 8U;
			// OaVideoStream is intentionally a video elementary-stream reader.
			// Ignore audio/text sample tables instead of letting the final track
			// overwrite the selected video geometry and packet index.
			ParseMdiaBox(mdia, mdiaSize, OutStream);
		}
		
		offset += boxSize;
	}
}

// Parse mdia box (media information)
void ParseMdiaBox(const OaU8* InData, OaU64 InSize, OaVideoStream& OutStream)
{
	OaU64 offset = 0;
	while (offset + 8 <= InSize) {
		OaU32 boxSize = ReadU32BE(InData + offset);
		OaU32 boxType = ReadU32BE(InData + offset + 4);

		if (boxSize < 8 || offset + boxSize > InSize) {
			break;
		}

		// Parse mdhd (media header) — timescale lives here. Per ISO/IEC 14496-12
		// §8.4.2 the v0 layout puts timescale at +20 (after version/flags +
		// creation_time + modification_time); v1 widens the times to 64-bit so
		// timescale moves to +28.
		if (boxType == OaMp4Box::Mdhd) {
			const OaU8* d = InData + offset + 8;
			const OaU64 dn = boxSize - 8;
			if (dn >= 4) {
				const OaU8 version = d[0];
				OaU64 tsOffset = (version == 1) ? 20 : 12;
				if (dn >= tsOffset + 4) {
					OutStream.Info_.TimebaseNum = 1;
					OutStream.Info_.TimebaseDen = ReadU32BE(d + tsOffset);
				}
			}
		}
		// Parse minf box (media info)
		else if (boxType == OaMp4Box::Minf) {
			ParseMinfBox(InData + offset + 8, boxSize - 8, OutStream);
		}

		offset += boxSize;
	}
}

// Parse minf box (media info)
void ParseMinfBox(const OaU8* InData, OaU64 InSize, OaVideoStream& OutStream)
{
	OaU64 offset = 0;
	while (offset + 8 <= InSize) {
		OaU32 boxSize = ReadU32BE(InData + offset);
		OaU32 boxType = ReadU32BE(InData + offset + 4);
		
		if (boxSize < 8 || offset + boxSize > InSize) {
			break;
		}
		
		// Parse stbl box (sample table)
		if (boxType == OaMp4Box::Stbl) {
			ParseStblBox(InData + offset + 8, boxSize - 8, OutStream);
		}
		
		offset += boxSize;
	}
}

// Parse stbl box (sample table) - Phase C1b: Implement full parsing
void ParseStblBox(const OaU8* InData, OaU64 InSize, OaVideoStream& OutStream)
{
	// Temporary storage for sample table data
	OaVec<OaU32> sampleSizes;      // From stsz
	OaVec<OaU64> chunkOffsets;     // From stco
	OaVec<OaU32> sttsEntries;     // From stts (count, duration pairs)
	OaVec<OaU32> stscEntries;     // From stsc (firstChunk, samplesPerChunk, sampleDescriptionIndex)
	OaVec<OaU32> stssEntries;     // From stss (keyframe sample indices)
	OaVec<OaI32> cttsEntries;     // From ctts (sampleCount, compositionOffset)
	
	OaU64 offset = 0;
	while (offset + 8 <= InSize) {
		OaU32 boxSize = ReadU32BE(InData + offset);
		OaU32 boxType = ReadU32BE(InData + offset + 4);
		
		if (boxSize < 8 || offset + boxSize > InSize) {
			break;
		}
		
		const OaU8* boxData = InData + offset + 8;
		const OaU64 boxDataSize = boxSize - 8;
		
		// Parse stsd (sample description) — codec, width, height.
		if (boxType == OaMp4Box::Stsd) {
			if (boxDataSize < 8U) { offset += boxSize; continue; }
			OaU32 versionFlags = ReadU32BE(boxData);
			OaU32 entryCount = ReadU32BE(boxData + 4);
			(void)versionFlags;
			// stsd payload: 8-byte header (version+count), then per-entry
			// VisualSampleEntry whose width/height live at fixed offsets per
			// ISO/IEC 14496-12 §8.5.2 (avc1, hev1/hvc1, av01 all extend this).
			if (entryCount > 0 && boxDataSize >= 8 + 8 + 36) {
				const OaU8* entry     = boxData + 8;
				OaU32 entrySize       = ReadU32BE(entry);
				OaU32 entryType       = ReadU32BE(entry + 4);
				if (entrySize < 8U + 78U
					or static_cast<OaU64>(entrySize) > boxDataSize - 8U) {
					offset += boxSize; continue;
				}

				// Detect codec from the sample entry fourcc.
				constexpr OaU32 kFourccAvc1 = 0x61766331; // 'avc1'
				constexpr OaU32 kFourccAvc3 = 0x61766333; // 'avc3'
				constexpr OaU32 kFourccHvc1 = 0x68766331; // 'hvc1'
				constexpr OaU32 kFourccHev1 = 0x68657631; // 'hev1'
				constexpr OaU32 kFourccAv01 = 0x61763031; // 'av01'
				constexpr OaU32 kFourccVp09 = 0x76703039; // 'vp09'
				if (entryType == kFourccHvc1 || entryType == kFourccHev1) {
					OutStream.Info_.Codec = OaVideoCodec::H265;
				} else if (entryType == kFourccAv01) {
					OutStream.Info_.Codec = OaVideoCodec::AV1;
				} else if (entryType == kFourccVp09) {
					OutStream.Info_.Codec = OaVideoCodec::VP9;
				} else if (entryType == kFourccAvc1 || entryType == kFourccAvc3) {
					OutStream.Info_.Codec = OaVideoCodec::H264;
				}

				// VisualSampleEntry: width at +32, height at +34 inside the
				// entry payload (after 8-byte box header → 24 bytes
				// SampleEntry/VisualSampleEntry preamble).
				const OaU8* visual = entry + 8;          // skip box header
				OutStream.Info_.Width  = ReadU16BE(visual + 24);
				OutStream.Info_.Height = ReadU16BE(visual + 26);

				// VisualSampleEntry payload runs 78 bytes (preamble + 32-byte
				// compressorname + depth + pre_defined). After that, codec-
				// specific config boxes follow (avcC, hvcC, ...).
				if (entryType == kFourccAvc1 || entryType == kFourccAvc3) {
					const OaUsize visualEnd = (entry + entrySize) - boxData;
					OaUsize cfgOffset = (visual - boxData) + 78;
					while (cfgOffset + 8 <= visualEnd && cfgOffset + 8 <= boxDataSize) {
						const OaU32 cfgSize = ReadU32BE(boxData + cfgOffset);
						const OaU32 cfgType = ReadU32BE(boxData + cfgOffset + 4);
						if (cfgSize < 8 || cfgOffset + cfgSize > visualEnd) {
							break;
						}
						if (cfgType == OaMp4Box::Avcc && cfgSize > 8 + 7) {
							// AVCDecoderConfigurationRecord (ISO/IEC 14496-15 §5.2.1.1).
							const OaU8* avcc = boxData + cfgOffset + 8;
							const OaU64 avccSize = cfgSize - 8;
							OutStream.Avc_.LengthSize =
								static_cast<OaU8>((avcc[4] & 0x03U) + 1U);
							const OaU8 numSps = avcc[5] & 0x1FU;
							OaUsize p = 6;
							const OaU8 startCode[4] = { 0, 0, 0, 1 };
							for (OaU8 i = 0; i < numSps && p + 2 <= avccSize; ++i) {
								OaU16 spsLen = ReadU16BE(avcc + p);
								p += 2;
								if (p + spsLen > avccSize) {
									break;
								}
								for (auto byte : startCode) {
									OutStream.Avc_.SpsAnnexB.PushBack(byte);
								}
								for (OaU16 k = 0; k < spsLen; ++k) {
									OutStream.Avc_.SpsAnnexB.PushBack(avcc[p + k]);
								}
								p += spsLen;
							}
							if (p < avccSize) {
								const OaU8 numPps = avcc[p];
								++p;
								for (OaU8 i = 0; i < numPps && p + 2 <= avccSize; ++i) {
									OaU16 ppsLen = ReadU16BE(avcc + p);
									p += 2;
									if (p + ppsLen > avccSize) {
										break;
									}
									for (auto byte : startCode) {
										OutStream.Avc_.PpsAnnexB.PushBack(byte);
									}
									for (OaU16 k = 0; k < ppsLen; ++k) {
										OutStream.Avc_.PpsAnnexB.PushBack(avcc[p + k]);
									}
									p += ppsLen;
								}
							}
							OutStream.Avc_.Valid = OutStream.Avc_.SpsAnnexB.Size() > 0
								&& OutStream.Avc_.PpsAnnexB.Size() > 0;
						}
						cfgOffset += cfgSize;
					}
				}
				else if (entryType == kFourccHvc1 || entryType == kFourccHev1) {
					const OaUsize visualEnd = (entry + entrySize) - boxData;
					OaUsize cfgOffset = (visual - boxData) + 78;
					while (cfgOffset + 8 <= visualEnd && cfgOffset + 8 <= boxDataSize) {
						const OaU32 cfgSize = ReadU32BE(boxData + cfgOffset);
						const OaU32 cfgType = ReadU32BE(boxData + cfgOffset + 4);
						if (cfgSize < 8 || cfgOffset + cfgSize > visualEnd) {
							break;
						}
						if (cfgType == OaMp4Box::Hvcc && cfgSize > 8 + 22) {
							// HEVCDecoderConfigurationRecord (ISO/IEC 14496-15 §8.3.3.1).
							const OaU8* hvcc = boxData + cfgOffset + 8;
							const OaU64 hvccSize = cfgSize - 8;
							OutStream.Hvc_.LengthSize =
								static_cast<OaU8>((hvcc[21] & 0x03U) + 1U);
							// Parse NAL arrays (VPS=32, SPS=33, PPS=34)
							if (hvccSize > 26) {
								const OaU8 numArrays = hvcc[22];
								OaUsize p = 23;
								const OaU8 startCode[4] = { 0, 0, 0, 1 };
								for (OaU8 a = 0; a < numArrays && p + 3 <= hvccSize; ++a) {
									const OaU8 nalType = hvcc[p] & 0x3FU;
									++p;
									const OaU16 numNalus = ReadU16BE(hvcc + p);
									p += 2;
									OaVec<OaU8>* target = nullptr;
									if (nalType == 32) { target = &OutStream.Hvc_.VpsAnnexB; }
									else if (nalType == 33) { target = &OutStream.Hvc_.SpsAnnexB; }
									else if (nalType == 34) { target = &OutStream.Hvc_.PpsAnnexB; }
									for (OaU16 n = 0; n < numNalus && p + 2 <= hvccSize; ++n) {
										const OaU16 nalLen = ReadU16BE(hvcc + p);
										p += 2;
										if (p + nalLen > hvccSize) { break; }
										if (target) {
											for (auto byte : startCode) {
												target->PushBack(byte);
											}
											for (OaU16 k = 0; k < nalLen; ++k) {
												target->PushBack(hvcc[p + k]);
											}
										}
										p += nalLen;
									}
								}
								OutStream.Hvc_.Valid = OutStream.Hvc_.VpsAnnexB.Size() > 0
									&& OutStream.Hvc_.SpsAnnexB.Size() > 0
									&& OutStream.Hvc_.PpsAnnexB.Size() > 0;
							}
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
					constexpr OaU32 kFourccAv1c = 0x61763143; // 'av1C'
					const OaUsize visualEnd = (entry + entrySize) - boxData;
					OaUsize cfgOffset = (visual - boxData) + 78;
					while (cfgOffset + 8 <= visualEnd && cfgOffset + 8 <= boxDataSize) {
						const OaU32 cfgSize = ReadU32BE(boxData + cfgOffset);
						const OaU32 cfgType = ReadU32BE(boxData + cfgOffset + 4);
						if (cfgSize < 8 || cfgOffset + cfgSize > visualEnd) {
							break;
						}
						// 4-byte AV1CodecConfigurationRecord header precedes the
						// configOBUs, so require > 8 (box hdr) + 4.
						if (cfgType == kFourccAv1c && cfgSize > 8 + 4) {
							const OaU8* av1c = boxData + cfgOffset + 8;
							const OaU64 av1cSize = cfgSize - 8;
							// configOBUs = everything after the 4-byte record
							// header; copied verbatim (raw low-overhead OBUs).
							for (OaU64 k = 4; k < av1cSize; ++k) {
								OutStream.Av1_.ConfigObus.PushBack(av1c[k]);
							}
							OutStream.Av1_.Valid = OutStream.Av1_.ConfigObus.Size() > 0;
						}
						cfgOffset += cfgSize;
					}
				}
			}
		}
		// Parse stts (time-to-sample) - duration per sample
		else if (boxType == OaMp4Box::Stts) {
			if (boxDataSize < 8U) { offset += boxSize; continue; }
			OaU32 versionFlags = ReadU32BE(boxData);
			OaU32 entryCount = ReadU32BE(boxData + 4);
			(void)versionFlags;
			if (not Mp4TableFits(boxDataSize, 8U, entryCount, 8U)) {
				offset += boxSize; continue;
			}
			sttsEntries.Resize(entryCount * 2);
			for (OaU32 i = 0; i < entryCount; ++i) {
				sttsEntries[i * 2] = ReadU32BE(boxData + 8 + i * 8);     // sampleCount
				sttsEntries[i * 2 + 1] = ReadU32BE(boxData + 8 + i * 8 + 4); // sampleDelta
			}
		}
		// Parse stsc (sample-to-chunk) - chunk grouping
		else if (boxType == OaMp4Box::Stsc) {
			if (boxDataSize < 8U) { offset += boxSize; continue; }
			OaU32 versionFlags = ReadU32BE(boxData);
			OaU32 entryCount = ReadU32BE(boxData + 4);
			(void)versionFlags;
			if (not Mp4TableFits(boxDataSize, 8U, entryCount, 12U)) {
				offset += boxSize; continue;
			}
			stscEntries.Resize(entryCount * 3);
			for (OaU32 i = 0; i < entryCount; ++i) {
				stscEntries[i * 3] = ReadU32BE(boxData + 8 + i * 12);         // firstChunk
				stscEntries[i * 3 + 1] = ReadU32BE(boxData + 8 + i * 12 + 4);  // samplesPerChunk
				stscEntries[i * 3 + 2] = ReadU32BE(boxData + 8 + i * 12 + 8);  // sampleDescriptionIndex
			}
		}
		// Parse stsz (sample size) - size per sample
		else if (boxType == OaMp4Box::Stsz) {
			if (boxDataSize < 12U) { offset += boxSize; continue; }
			OaU32 versionFlags = ReadU32BE(boxData);
			OaU32 sampleSize = ReadU32BE(boxData + 4);
			OaU32 sampleCount = ReadU32BE(boxData + 8);
			(void)versionFlags;
			if (sampleCount > kMaxMp4TableEntries) {
				offset += boxSize; continue;
			}
			if (sampleSize == 0) {
				// Variable sample sizes - read table
				if (not Mp4TableFits(boxDataSize, 12U, sampleCount, 4U)) {
					offset += boxSize; continue;
				}
				sampleSizes.Resize(sampleCount);
				for (OaU32 i = 0; i < sampleCount; ++i) {
					sampleSizes[i] = ReadU32BE(boxData + 12 + i * 4);
				}
			} else {
				// Fixed sample size
				sampleSizes.Resize(sampleCount);
				for (OaU32 i = 0; i < sampleCount; ++i) {
					sampleSizes[i] = sampleSize;
				}
			}
		}
		// Parse stco (chunk offset) - offset per chunk (32-bit)
		else if (boxType == OaMp4Box::Stco) {
			if (boxDataSize < 8U) { offset += boxSize; continue; }
			OaU32 versionFlags = ReadU32BE(boxData);
			OaU32 entryCount = ReadU32BE(boxData + 4);
			(void)versionFlags;
			if (not Mp4TableFits(boxDataSize, 8U, entryCount, 4U)) {
				offset += boxSize; continue;
			}
			chunkOffsets.Resize(entryCount);
			for (OaU32 i = 0; i < entryCount; ++i) {
				chunkOffsets[i] = ReadU32BE(boxData + 8 + i * 4);
			}
		}
		// Parse co64 (chunk offset, 64-bit) — required for files > 4 GiB
		else if (boxType == OaMp4Box::Co64) {
			if (boxDataSize < 8U) { offset += boxSize; continue; }
			OaU32 versionFlags = ReadU32BE(boxData);
			OaU32 entryCount = ReadU32BE(boxData + 4);
			(void)versionFlags;
			if (not Mp4TableFits(boxDataSize, 8U, entryCount, 8U)) {
				offset += boxSize; continue;
			}
			chunkOffsets.Resize(entryCount);
			for (OaU32 i = 0; i < entryCount; ++i) {
				chunkOffsets[i] = ReadU64BE(boxData + 8 + i * 8);
			}
		}
		// Parse stss (sync sample) - keyframe indices
		else if (boxType == OaMp4Box::Stss) {
			if (boxDataSize < 8U) { offset += boxSize; continue; }
			OaU32 versionFlags = ReadU32BE(boxData);
			OaU32 entryCount = ReadU32BE(boxData + 4);
			(void)versionFlags;
			if (not Mp4TableFits(boxDataSize, 8U, entryCount, 4U)) {
				offset += boxSize; continue;
			}
			stssEntries.Resize(entryCount);
			for (OaU32 i = 0; i < entryCount; ++i) {
				const OaU32 sampleNumber = ReadU32BE(boxData + 8 + i * 4);
				stssEntries[i] = sampleNumber == 0U ? 0U : sampleNumber - 1U;
			}
		}
		// Parse ctts (composition time offset) - PTS-DTS offset
		else if (boxType == OaMp4Box::Ctts) {
			if (boxDataSize < 8U) { offset += boxSize; continue; }
			OaU32 versionFlags = ReadU32BE(boxData);
			OaU32 entryCount = ReadU32BE(boxData + 4);
			const OaU8 version = static_cast<OaU8>(versionFlags >> 24U);
			if (not Mp4TableFits(boxDataSize, 8U, entryCount, 8U)) {
				offset += boxSize; continue;
			}
			cttsEntries.Resize(entryCount * 2);
			for (OaU32 i = 0; i < entryCount; ++i) {
				cttsEntries[i * 2] = static_cast<OaI32>(ReadU32BE(boxData + 8 + i * 8));     // sampleCount
				const OaU32 rawOffset = ReadU32BE(boxData + 8 + i * 8 + 4);
				cttsEntries[i * 2 + 1] = version == 1U
					? static_cast<OaI32>(rawOffset)
					: static_cast<OaI32>(std::min<OaU32>(rawOffset, 0x7FFFFFFFU));
			}
		}
		
		offset += boxSize;
	}
	
	// Build sample table from parsed data. Properly walks stsc to find the
	// chunk each sample lives in, then sums preceding samples' sizes to get
	// each sample's absolute file offset. stss == empty means every sample
	// is treated as a sync sample (Annex-B all-IDR streams).
	if (!sampleSizes.Empty() && !chunkOffsets.Empty()) {
		const OaU32 sampleCount = sampleSizes.Size();
		OutStream.Samples_.Resize(sampleCount);

		// Walk stsc to materialize samples-per-chunk for every chunk index.
		// stscEntries layout per row: [firstChunk(1-based), samplesPerChunk, descIndex].
		const OaU32 stscRows = stscEntries.Size() / 3;
		const OaU32 chunkCount = chunkOffsets.Size();

		auto samplesInChunk = [&](OaU32 chunk1Based) -> OaU32 {
			OaU32 result = 1;
			for (OaU32 r = 0; r < stscRows; ++r) {
				const OaU32 firstChunk = stscEntries[r * 3];
				if (firstChunk > chunk1Based) { break; }
				result = stscEntries[r * 3 + 1];
			}
			return result;
		};

		// Walk samples in linear order, advancing the active chunk pointer.
		OaU32 chunkIdx        = 0;   // 0-based index into chunkOffsets
		OaU32 inChunkIdx      = 0;   // 0-based sample index within current chunk
		OaU64 inChunkOffset   = 0;   // cumulative byte offset within current chunk
		OaU64 dts             = 0;
		OaU32 sttsRowIdx      = 0;
		OaU32 sttsRowRemaining = sttsEntries.Empty()
			? 0 : sttsEntries[0];     // sampleCount of row 0
		OaU32 sttsRowDelta    = sttsEntries.Size() >= 2
			? sttsEntries[1] : 1;
		OaU32 cttsRowIdx      = 0;
		OaU32 cttsRowRemaining = cttsEntries.Empty()
			? 0 : cttsEntries[0];     // sampleCount of row 0
		OaI32 cttsRowOffset   = cttsEntries.Size() >= 2
			? cttsEntries[1] : 0;
		OaUsize stssIdx       = 0;

		for (OaU32 i = 0; i < sampleCount; ++i) {
			// Advance into the next chunk when we've consumed this chunk's
			// sample budget.
			if (chunkIdx >= chunkCount) {
				OutStream.Samples_[i] = OaVideoStream::Sample{};
				continue;
			}
			const OaU32 budget = samplesInChunk(chunkIdx + 1);
			if (inChunkIdx >= budget) {
				++chunkIdx;
				inChunkIdx = 0;
				inChunkOffset = 0;
				if (chunkIdx >= chunkCount) {
					OutStream.Samples_[i] = OaVideoStream::Sample{};
					continue;
				}
			}

			OutStream.Samples_[i].Size   = sampleSizes[i];
			OutStream.Samples_[i].Offset = chunkOffsets[chunkIdx] + inChunkOffset;
			inChunkOffset += sampleSizes[i];
			++inChunkIdx;

			OutStream.Samples_[i].Dts = dts;

			// stts walk — advance to the next [count, delta] row when the
			// current row is exhausted.
			if (sttsRowRemaining == 0 && sttsRowIdx + 1 < sttsEntries.Size() / 2) {
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
			if (cttsRowRemaining == 0 && cttsRowIdx + 1 < cttsEntries.Size() / 2) {
				++cttsRowIdx;
				cttsRowRemaining = static_cast<OaU32>(cttsEntries[cttsRowIdx * 2]);
				cttsRowOffset = cttsEntries[cttsRowIdx * 2 + 1];
			}
			OutStream.Samples_[i].CtsOffset = cttsRowOffset;
			if (cttsRowRemaining > 0) {
				--cttsRowRemaining;
			}

			// Keyframe: present in stss, or stss empty (every sample IDR).
			bool isKeyframe = stssEntries.Empty();
			while (stssIdx < stssEntries.Size() && stssEntries[stssIdx] < i) {
				++stssIdx;
			}
			if (stssIdx < stssEntries.Size() && stssEntries[stssIdx] == i) {
				isKeyframe = true;
			}
			OutStream.Samples_[i].IsKeyframe = isKeyframe;
		}
	}
}

OaStatus ParseMoofBox(const OaU8* InData, OaU64 InSize, OaU64 InMoofOffset,
	OaU64 InMoofEnd,
	OaVideoStream& OutStream)
{
	for (OaU64 offset = 0U; offset + 8U <= InSize;) {
		const OaU32 boxSize = ReadU32BE(InData + offset);
		const OaU32 boxType = ReadU32BE(InData + offset + 4U);
		if (boxSize < 8U or offset + boxSize > InSize) {
			return OaStatus::Error(OaStatusCode::DataLoss, "Malformed fragmented MP4 box");
		}
		if (boxType != OaMp4Box::Traf) {
			offset += boxSize;
			continue;
		}

		const OaU8* traf = InData + offset + 8U;
		const OaU64 trafSize = boxSize - 8U;
		OaU32 trackId = 0U;
		OaU32 defaultDuration = OutStream.Fragment_.DefaultSampleDuration;
		OaU32 defaultSize = OutStream.Fragment_.DefaultSampleSize;
		OaU32 defaultFlags = OutStream.Fragment_.DefaultSampleFlags;
		OaU64 baseDataOffset = InMoofOffset;
		OaU64 baseDecodeTime = 0U;
		OaVec<OaU64> runs;

		for (OaU64 child = 0U; child + 8U <= trafSize;) {
			const OaU32 childSize = ReadU32BE(traf + child);
			const OaU32 childType = ReadU32BE(traf + child + 4U);
			if (childSize < 8U or child + childSize > trafSize) {
				return OaStatus::Error(OaStatusCode::DataLoss, "Malformed MP4 track fragment");
			}
			const OaU8* payload = traf + child + 8U;
			const OaU64 payloadSize = childSize - 8U;
			if (childType == OaMp4Box::Tfhd and payloadSize >= 8U) {
				const OaU32 flags = ReadU32BE(payload) & 0x00FFFFFFU;
				trackId = ReadU32BE(payload + 4U);
				OaU64 p = 8U;
				if ((flags & 0x000001U) != 0U) {
					if (p + 8U > payloadSize) return OaStatus::Error("Truncated tfhd base offset");
					baseDataOffset = ReadU64BE(payload + p);
					p += 8U;
				}
				if ((flags & 0x000002U) != 0U) p += 4U; // sample_description_index
				if ((flags & 0x000008U) != 0U) {
					if (p + 4U > payloadSize) return OaStatus::Error("Truncated tfhd duration");
					defaultDuration = ReadU32BE(payload + p); p += 4U;
				}
				if ((flags & 0x000010U) != 0U) {
					if (p + 4U > payloadSize) return OaStatus::Error("Truncated tfhd size");
					defaultSize = ReadU32BE(payload + p); p += 4U;
				}
				if ((flags & 0x000020U) != 0U) {
					if (p + 4U > payloadSize) return OaStatus::Error("Truncated tfhd flags");
					defaultFlags = ReadU32BE(payload + p);
				}
			} else if (childType == OaMp4Box::Tfdt and payloadSize >= 8U) {
				const OaU8 version = payload[0];
				if (version == 1U) {
					if (payloadSize < 12U) return OaStatus::Error("Truncated 64-bit tfdt");
					baseDecodeTime = ReadU64BE(payload + 4U);
				} else {
					baseDecodeTime = ReadU32BE(payload + 4U);
				}
			} else if (childType == OaMp4Box::Trun) {
				runs.PushBack(child);
			}
			child += childSize;
		}

		if (OutStream.Fragment_.TrackId != 0U and trackId != OutStream.Fragment_.TrackId) {
			offset += boxSize;
			continue;
		}
		if (trackId != 0U) OutStream.Fragment_.TrackId = trackId;
		OaU64 dts = baseDecodeTime;
		OaU64 implicitDataOffset = InMoofEnd;
		for (OaU64 runOffset : runs) {
			const OaU32 runSize = ReadU32BE(traf + runOffset);
			const OaU8* run = traf + runOffset + 8U;
			const OaU64 runPayloadSize = runSize - 8U;
			if (runPayloadSize < 8U) return OaStatus::Error("Truncated trun");
			const OaU8 version = run[0];
			const OaU32 flags = ReadU32BE(run) & 0x00FFFFFFU;
			const OaU32 sampleCount = ReadU32BE(run + 4U);
			OaU64 p = 8U;
			OaU64 dataOffset = implicitDataOffset;
			if ((flags & 0x000001U) != 0U) {
				if (p + 4U > runPayloadSize) return OaStatus::Error("Truncated trun data offset");
				const OaI32 relative = static_cast<OaI32>(ReadU32BE(run + p));
				if (relative < 0 and static_cast<OaU64>(-static_cast<OaI64>(relative)) > baseDataOffset) {
					return OaStatus::Error("Invalid negative trun data offset");
				}
				dataOffset = relative >= 0
					? baseDataOffset + static_cast<OaU64>(relative)
					: baseDataOffset - static_cast<OaU64>(-static_cast<OaI64>(relative));
				p += 4U;
			}
			OaU32 firstSampleFlags = defaultFlags;
			if ((flags & 0x000004U) != 0U) {
				if (p + 4U > runPayloadSize) return OaStatus::Error("Truncated first sample flags");
				firstSampleFlags = ReadU32BE(run + p); p += 4U;
			}
			for (OaU32 sampleIndex = 0U; sampleIndex < sampleCount; ++sampleIndex) {
				OaU32 duration = defaultDuration;
				OaU32 size = defaultSize;
				OaU32 sampleFlags = sampleIndex == 0U ? firstSampleFlags : defaultFlags;
				OaI32 ctsOffset = 0;
				if ((flags & 0x000100U) != 0U) {
					if (p + 4U > runPayloadSize) return OaStatus::Error("Truncated sample duration");
					duration = ReadU32BE(run + p); p += 4U;
				}
				if ((flags & 0x000200U) != 0U) {
					if (p + 4U > runPayloadSize) return OaStatus::Error("Truncated sample size");
					size = ReadU32BE(run + p); p += 4U;
				}
				if ((flags & 0x000400U) != 0U) {
					if (p + 4U > runPayloadSize) return OaStatus::Error("Truncated sample flags");
					sampleFlags = ReadU32BE(run + p); p += 4U;
				}
				if ((flags & 0x000800U) != 0U) {
					if (p + 4U > runPayloadSize) return OaStatus::Error("Truncated composition offset");
					const OaU32 raw = ReadU32BE(run + p); p += 4U;
					ctsOffset = version == 1U ? static_cast<OaI32>(raw)
						: static_cast<OaI32>(std::min<OaU32>(raw, 0x7FFFFFFFU));
				}
				if (size == 0U) return OaStatus::Error("Fragment sample has no declared size");
				OaVideoStream::Sample sample;
				sample.Offset = dataOffset;
				sample.Size = size;
				sample.Duration = duration == 0U ? 1U : duration;
				sample.Dts = dts;
				sample.CtsOffset = ctsOffset;
				sample.IsKeyframe = (sampleFlags & 0x00010000U) == 0U;
				OutStream.Samples_.PushBack(sample);
				dataOffset += size;
				dts += sample.Duration;
			}
			implicitDataOffset = dataOffset;
		}
		offset += boxSize;
	}
	return OaStatus::Ok();
}

}  // namespace
