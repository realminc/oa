// OA Vision — streaming MP4 muxer implementation.

#include <Oa/Vision/VideoMuxer.h>
#include <Oa/Core/Filesystem.h>
#include <algorithm>
#include <cstring>
#include <limits>

namespace
{

// Helper: write 16-bit big-endian
inline void WriteU16BE(OaU8* OutPtr, OaU16 InValue)
{
	OutPtr[0] = static_cast<OaU8>((InValue >> 8) & 0xFF);
	OutPtr[1] = static_cast<OaU8>(InValue & 0xFF);
}

// Helper: write 32-bit big-endian
inline void WriteU32BE(OaU8* OutPtr, OaU32 InValue)
{
	OutPtr[0] = static_cast<OaU8>((InValue >> 24) & 0xFF);
	OutPtr[1] = static_cast<OaU8>((InValue >> 16) & 0xFF);
	OutPtr[2] = static_cast<OaU8>((InValue >> 8) & 0xFF);
	OutPtr[3] = static_cast<OaU8>(InValue & 0xFF);
}

inline void WriteU64BE(OaU8* OutPtr, OaU64 InValue)
{
	WriteU32BE(OutPtr, static_cast<OaU32>(InValue >> 32U));
	WriteU32BE(OutPtr + 4U, static_cast<OaU32>(InValue));
}

bool WriteFile(std::FILE* InFile, const void* InData, OaUsize InSize)
{
	return InSize == 0U or std::fwrite(InData, 1U, InSize, InFile) == InSize;
}

// Helper: write MP4 box header
inline void WriteBoxHeader(OaU8* OutPtr, OaU32 InSize, OaU32 InType)
{
	WriteU32BE(OutPtr, InSize);
	WriteU32BE(OutPtr + 4, InType);
}

OaUsize FindAnnexBStartCode(const OaU8* InData, OaUsize InSize, OaUsize InFrom, OaUsize& OutLength)
{
	for (OaUsize i = InFrom; i + 3U <= InSize; ++i) {
		if (InData[i] != 0U or InData[i + 1U] != 0U) continue;
		if (InData[i + 2U] == 1U) {
			OutLength = 3U;
			return i;
		}
		if (i + 4U <= InSize and InData[i + 2U] == 0U and InData[i + 3U] == 1U) {
			OutLength = 4U;
			return i;
		}
	}
	OutLength = 0U;
	return InSize;
}

// MP4 avc1 samples use length-prefixed NAL units. Vulkan encoders normally
// return Annex-B. Convert every access unit at the container boundary; keeping
// Annex-B in mdat makes the avcC lengthSizeMinusOne declaration a lie.
OaResult<OaVec<OaU8>> AnnexBToLengthPrefixed(const OaSpan<const OaU8>& InBytes)
{
	OaVec<OaU8> output;
	if (InBytes.Empty()) return output;
	const OaU8* data = InBytes.Data();
	const OaUsize size = InBytes.Size();
	OaUsize startLength = 0U;
	OaUsize start = FindAnnexBStartCode(data, size, 0U, startLength);
	if (start == size) {
		return OaStatus::Error(OaStatusCode::DataLoss,
			"MP4 video packet is not Annex-B framed");
	}
	while (start < size) {
		const OaUsize payloadStart = start + startLength;
		OaUsize nextLength = 0U;
		const OaUsize next = FindAnnexBStartCode(data, size, payloadStart, nextLength);
		OaUsize payloadEnd = next;
		while (payloadEnd > payloadStart and data[payloadEnd - 1U] == 0U) --payloadEnd;
		const OaUsize payloadSize = payloadEnd - payloadStart;
		if (payloadSize > 0xFFFFFFFFULL) {
			return OaStatus::Error(OaStatusCode::OutOfRange, "Video NAL exceeds MP4 length field");
		}
		if (payloadSize > 0U) {
			const OaUsize oldSize = output.Size();
			output.Resize(oldSize + 4U + payloadSize);
			WriteU32BE(output.Data() + oldSize, static_cast<OaU32>(payloadSize));
			std::memcpy(output.Data() + oldSize + 4U, data + payloadStart, payloadSize);
		}
		start = next;
		startLength = nextLength;
	}
	if (output.Empty()) {
		return OaStatus::Error(OaStatusCode::DataLoss, "Video access unit contains no NAL units");
	}
	return output;
}


OaVec<OaU8> BuildHvcc(
	const OaVec<OaU8>& InVps,
	const OaVec<OaU8>& InSps,
	const OaVec<OaU8>& InPps)
{
	// HEVCDecoderConfigurationRecord (ISO/IEC 14496-15 section 8.3.3.1):
	// 23-byte fixed record followed by complete VPS/SPS/PPS arrays. OA emits
	// Main/8-bit/4:2:0 with one temporal layer and four-byte sample lengths.
	const OaU32 payloadSize = 23U
		+ 3U + 2U + static_cast<OaU32>(InVps.Size())
		+ 3U + 2U + static_cast<OaU32>(InSps.Size())
		+ 3U + 2U + static_cast<OaU32>(InPps.Size());
	OaVec<OaU8> box;
	box.Resize(8U + payloadSize);
	std::memset(box.Data(), 0, box.Size());
	WriteBoxHeader(box.Data(), static_cast<OaU32>(box.Size()), 0x68766343U); // hvcC
	OaU8* record = box.Data() + 8U;
	record[0] = 1U;          // configurationVersion
	record[1] = 1U;          // profile_space=0, tier=0, profile_idc=Main(1)
	WriteU32BE(record + 2U, 0x60000000U); // Main-compatible profile flags
	// general_constraint_indicator_flags[6] stay zero.
	record[12] = 123U;       // general_level_idc = level 4.1
	record[13] = 0xF0U;      // reserved + min_spatial_segmentation_idc=0
	record[14] = 0U;
	record[15] = 0xFCU;      // parallelismType unknown
	record[16] = 0xFDU;      // chromaFormat = 4:2:0 (1)
	record[17] = 0xF8U;      // bitDepthLumaMinus8 = 0
	record[18] = 0xF8U;      // bitDepthChromaMinus8 = 0
	record[19] = 0U;         // avgFrameRate unknown
	record[20] = 0U;
	record[21] = 0x0FU;      // 1 temporal layer, nested, lengthSizeMinusOne=3
	record[22] = 3U;         // numOfArrays

	OaU32 offset = 23U;
	auto WriteArray = [&](OaU8 InNalType, const OaVec<OaU8>& InNal) {
		record[offset++] = static_cast<OaU8>(0x80U | InNalType);
		WriteU16BE(record + offset, 1U);
		offset += 2U;
		WriteU16BE(record + offset, static_cast<OaU16>(InNal.Size()));
		offset += 2U;
		std::memcpy(record + offset, InNal.Data(), InNal.Size());
		offset += static_cast<OaU32>(InNal.Size());
	};
	WriteArray(32U, InVps);
	WriteArray(33U, InSps);
	WriteArray(34U, InPps);
	return box;
}

void AppendBytes(OaVec<OaU8>& Out, const void* InData, OaUsize InSize)
{
	const OaUsize offset = Out.Size();
	Out.Resize(offset + InSize);
	if (InSize > 0U) std::memcpy(Out.Data() + offset, InData, InSize);
}

OaVec<OaU8> WrapBox(OaU32 InType, const OaVec<OaU8>& InPayload)
{
	OaVec<OaU8> box;
	box.Resize(8U + InPayload.Size());
	WriteBoxHeader(box.Data(), static_cast<OaU32>(box.Size()), InType);
	if (!InPayload.Empty()) std::memcpy(box.Data() + 8U, InPayload.Data(), InPayload.Size());
	return box;
}

OaVec<OaU8> BuildAudioTrack(
	const OaVideoMuxer::CreateInfo& InInfo,
	const OaVec<OaU64>& InOffsets,
	const OaVec<OaU32>& InDurations,
	OaU32 InMovieTimescale)
{
	OaU64 mediaDuration64 = 0U;
	for (OaU32 duration : InDurations) mediaDuration64 += duration;
	const OaU32 mediaDuration = static_cast<OaU32>(
		std::min<OaU64>(mediaDuration64, std::numeric_limits<OaU32>::max()));
	const OaU64 audibleFrames = mediaDuration64 > InInfo.AudioPrimingFrames
		? mediaDuration64 - InInfo.AudioPrimingFrames : 0U;
	const OaU32 movieDuration = static_cast<OaU32>(std::min<OaU64>(
		(audibleFrames * InMovieTimescale + InInfo.AudioSampleRate / 2U)
			/ InInfo.AudioSampleRate,
		std::numeric_limits<OaU32>::max()));

	OaVec<OaU8> trakPayload;
	OaU8 tkhd[92] = {};
	WriteBoxHeader(tkhd, 92U, 0x746b6864U);
	WriteU32BE(tkhd + 8U, 0x00000007U);
	WriteU32BE(tkhd + 20U, 2U);
	WriteU32BE(tkhd + 28U, movieDuration);
	WriteU16BE(tkhd + 44U, 0x0100U); // audio track volume 1.0
	WriteU32BE(tkhd + 48U, 0x00010000U);
	WriteU32BE(tkhd + 64U, 0x00010000U);
	WriteU32BE(tkhd + 80U, 0x40000000U);
	AppendBytes(trakPayload, tkhd, sizeof(tkhd));

	if (InInfo.AudioPrimingFrames > 0U) {
		OaVec<OaU8> elstPayload(20U, 0U);
		WriteU32BE(elstPayload.Data() + 4U, 1U);
		WriteU32BE(elstPayload.Data() + 8U, movieDuration);
		WriteU32BE(elstPayload.Data() + 12U, InInfo.AudioPrimingFrames);
		WriteU16BE(elstPayload.Data() + 16U, 1U);
		auto elst = WrapBox(0x656c7374U, elstPayload);
		auto edts = WrapBox(0x65647473U, elst);
		AppendBytes(trakPayload, edts.Data(), edts.Size());
	}

	OaVec<OaU8> mdiaPayload;
	OaU8 mdhd[32] = {};
	WriteBoxHeader(mdhd, 32U, 0x6d646864U);
	WriteU32BE(mdhd + 20U, InInfo.AudioSampleRate);
	WriteU32BE(mdhd + 24U, mediaDuration);
	WriteU16BE(mdhd + 28U, 0x55C4U);
	AppendBytes(mdiaPayload, mdhd, sizeof(mdhd));

	constexpr char kName[] = "SoundHandler";
	OaVec<OaU8> hdlr(32U + sizeof(kName), 0U);
	WriteBoxHeader(hdlr.Data(), static_cast<OaU32>(hdlr.Size()), 0x68646c72U);
	WriteU32BE(hdlr.Data() + 16U, 0x736f756eU); // soun
	std::memcpy(hdlr.Data() + 32U, kName, sizeof(kName));
	AppendBytes(mdiaPayload, hdlr.Data(), hdlr.Size());

	OaVec<OaU8> minfPayload;
	OaU8 smhd[16] = {};
	WriteBoxHeader(smhd, 16U, 0x736d6864U);
	AppendBytes(minfPayload, smhd, sizeof(smhd));

	OaVec<OaU8> stblPayload;
	// ISO/IEC 23003-5 uncompressed audio: signed 16-bit little-endian PCM.
	// `ipcm` is the sample entry and `pcmC` carries byte order/sample width.
	constexpr OaU32 kPcmcSize = 14U;
	constexpr OaU32 kSampleEntrySize = 36U + kPcmcSize;
	const OaU32 sampleEntrySize = kSampleEntrySize;
	OaVec<OaU8> stsd(16U + sampleEntrySize, 0U);
	WriteBoxHeader(stsd.Data(), static_cast<OaU32>(stsd.Size()), 0x73747364U);
	WriteU32BE(stsd.Data() + 12U, 1U);
	OaU8* ipcm = stsd.Data() + 16U;
	WriteBoxHeader(ipcm, sampleEntrySize, 0x6970636dU); // ipcm
	WriteU16BE(ipcm + 14U, 1U);
	WriteU16BE(ipcm + 24U, static_cast<OaU16>(InInfo.AudioChannelCount));
	WriteU16BE(ipcm + 26U, 16U);
	WriteU32BE(ipcm + 32U, InInfo.AudioSampleRate << 16U);
	OaU8* pcmc = ipcm + 36U;
	WriteBoxHeader(pcmc, kPcmcSize, 0x70636d43U); // pcmC
	pcmc[12U] = 1U; // little-endian
	pcmc[13U] = 16U;
	AppendBytes(stblPayload, stsd.Data(), stsd.Size());

	// One PCM frame is one MP4 sample and lasts one audio-timescale tick.
	OaVec<OaU8> stts(24U, 0U);
	WriteBoxHeader(stts.Data(), static_cast<OaU32>(stts.Size()), 0x73747473U);
	WriteU32BE(stts.Data() + 12U, 1U);
	WriteU32BE(stts.Data() + 16U, mediaDuration);
	WriteU32BE(stts.Data() + 20U, 1U);
	AppendBytes(stblPayload, stts.Data(), stts.Size());

	struct ChunkRun { OaU32 FirstChunk; OaU32 SamplesPerChunk; };
	OaVec<ChunkRun> chunkRuns;
	for (OaU32 i = 0U; i < InDurations.Size(); ++i) {
		if (chunkRuns.Empty() or chunkRuns.Back().SamplesPerChunk != InDurations[i]) {
			chunkRuns.PushBack({i + 1U, InDurations[i]});
		}
	}
	OaVec<OaU8> stsc(16U + chunkRuns.Size() * 12U, 0U);
	WriteBoxHeader(stsc.Data(), static_cast<OaU32>(stsc.Size()), 0x73747363U);
	WriteU32BE(stsc.Data() + 12U, static_cast<OaU32>(chunkRuns.Size()));
	for (OaU32 i = 0U; i < chunkRuns.Size(); ++i) {
		WriteU32BE(stsc.Data() + 16U + i * 12U, chunkRuns[i].FirstChunk);
		WriteU32BE(stsc.Data() + 20U + i * 12U, chunkRuns[i].SamplesPerChunk);
		WriteU32BE(stsc.Data() + 24U + i * 12U, 1U);
	}
	AppendBytes(stblPayload, stsc.Data(), stsc.Size());

	OaVec<OaU8> stsz(20U, 0U);
	WriteBoxHeader(stsz.Data(), static_cast<OaU32>(stsz.Size()), 0x7374737aU);
	WriteU32BE(stsz.Data() + 12U, InInfo.AudioChannelCount * sizeof(OaI16));
	WriteU32BE(stsz.Data() + 16U, mediaDuration);
	AppendBytes(stblPayload, stsz.Data(), stsz.Size());

	const bool largeOffsets = std::any_of(InOffsets.begin(), InOffsets.end(),
		[](OaU64 InOffset) { return InOffset > 0xFFFFFFFFULL; });
	OaVec<OaU8> offsets(16U + InOffsets.Size() * (largeOffsets ? 8U : 4U), 0U);
	WriteBoxHeader(offsets.Data(), static_cast<OaU32>(offsets.Size()),
		largeOffsets ? 0x636f3634U : 0x7374636fU); // co64 / stco
	WriteU32BE(offsets.Data() + 12U, static_cast<OaU32>(InOffsets.Size()));
	for (OaU32 i = 0U; i < InOffsets.Size(); ++i) {
		if (largeOffsets) WriteU64BE(offsets.Data() + 16U + i * 8U, InOffsets[i]);
		else WriteU32BE(offsets.Data() + 16U + i * 4U, static_cast<OaU32>(InOffsets[i]));
	}
	AppendBytes(stblPayload, offsets.Data(), offsets.Size());

	auto stbl = WrapBox(0x7374626cU, stblPayload);
	AppendBytes(minfPayload, stbl.Data(), stbl.Size());
	auto minf = WrapBox(0x6d696e66U, minfPayload);
	AppendBytes(mdiaPayload, minf.Data(), minf.Size());
	auto mdia = WrapBox(0x6d646961U, mdiaPayload);
	AppendBytes(trakPayload, mdia.Data(), mdia.Size());
	return WrapBox(0x7472616bU, trakPayload);
}

}  // namespace


void OaVideoMuxer::Reset_() noexcept
{
	if (OutputFile_ != nullptr) std::fclose(OutputFile_);
	OutputFile_ = nullptr;
	MdatPayloadBytes_ = 0U;
	Info_ = {};
	MdatData_.Clear();
	PacketOffsets_.Clear();
	PacketSizes_.Clear();
	PacketDts_.Clear();
	PacketKeyframe_.Clear();
	AudioPacketOffsets_.Clear();
	AudioPacketSizes_.Clear();
	AudioPacketDurations_.Clear();
	AudioCodecConfig_.Clear();
	PacketCount_ = 0;
	Finalized_ = false;
	Vps_.Clear();
	Sps_.Clear();
	Pps_.Clear();
}


OaVideoMuxer::OaVideoMuxer(OaVideoMuxer&& InOther) noexcept
	: Info_(InOther.Info_)
	, MdatData_(std::move(InOther.MdatData_))
	, OutputFile_(InOther.OutputFile_)
	, MdatPayloadBytes_(InOther.MdatPayloadBytes_)
	, PacketOffsets_(std::move(InOther.PacketOffsets_))
	, PacketSizes_(std::move(InOther.PacketSizes_))
	, PacketDts_(std::move(InOther.PacketDts_))
	, PacketKeyframe_(std::move(InOther.PacketKeyframe_))
	, AudioPacketOffsets_(std::move(InOther.AudioPacketOffsets_))
	, AudioPacketSizes_(std::move(InOther.AudioPacketSizes_))
	, AudioPacketDurations_(std::move(InOther.AudioPacketDurations_))
	, AudioCodecConfig_(std::move(InOther.AudioCodecConfig_))
	, PacketCount_(InOther.PacketCount_)
	, Finalized_(InOther.Finalized_)
	, Vps_(std::move(InOther.Vps_))
	, Sps_(std::move(InOther.Sps_))
	, Pps_(std::move(InOther.Pps_))
{
	InOther.OutputFile_ = nullptr;
	InOther.Reset_();
}


OaVideoMuxer& OaVideoMuxer::operator=(OaVideoMuxer&& InOther) noexcept
{
	if (this != &InOther) {
		Destroy();
		Info_ = InOther.Info_;
		MdatData_ = std::move(InOther.MdatData_);
		OutputFile_ = InOther.OutputFile_;
		MdatPayloadBytes_ = InOther.MdatPayloadBytes_;
		PacketOffsets_ = std::move(InOther.PacketOffsets_);
		PacketSizes_ = std::move(InOther.PacketSizes_);
		PacketDts_ = std::move(InOther.PacketDts_);
		PacketKeyframe_ = std::move(InOther.PacketKeyframe_);
		AudioPacketOffsets_ = std::move(InOther.AudioPacketOffsets_);
		AudioPacketSizes_ = std::move(InOther.AudioPacketSizes_);
		AudioPacketDurations_ = std::move(InOther.AudioPacketDurations_);
		AudioCodecConfig_ = std::move(InOther.AudioCodecConfig_);
		PacketCount_ = InOther.PacketCount_;
		Finalized_ = InOther.Finalized_;
		Vps_ = std::move(InOther.Vps_);
		Sps_ = std::move(InOther.Sps_);
		Pps_ = std::move(InOther.Pps_);
		InOther.OutputFile_ = nullptr;
		InOther.Reset_();
	}
	return *this;
}


OaVideoMuxer::~OaVideoMuxer()
{
	Destroy();
}


void OaVideoMuxer::Destroy()
{
	Reset_();
}


OaResult<OaVideoMuxer> OaVideoMuxer::Create(const char* InPath, const CreateInfo& InInfo)
{
	if (InPath == nullptr or *InPath == '\0') {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Video muxer path is empty");
	}
	if (InInfo.Width == 0U or InInfo.Height == 0U or InInfo.FrameRate == 0U
		or InInfo.TimebaseNum == 0U or InInfo.TimebaseDen == 0U) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"Video muxer requires extent, frame rate and a valid timebase");
	}
	if (InInfo.Codec != OaVideoCodec::H264 and InInfo.Codec != OaVideoCodec::H265) {
		return OaStatus::Error(OaStatusCode::Unimplemented,
			"MP4 muxing is implemented for H.264 and H.265");
	}
	if (InInfo.AudioEnabled and (InInfo.AudioCodec != OaAudioCodec::PcmS16
		or InInfo.AudioSampleRate == 0U or InInfo.AudioSampleRate > 65'535U
		or InInfo.AudioChannelCount == 0U or InInfo.AudioChannelCount > 8U)) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"MP4 audio requires native PcmS16, sample rate <= 65535 and 1..8 channels");
	}
	OaVideoMuxer muxer;
	muxer.Info_ = InInfo;
	
	muxer.OutputFile_ = std::fopen(InPath, "wb+");
	if (muxer.OutputFile_ == nullptr) {
		return OaStatus::Error(OaStatusCode::Unavailable,
			"Cannot open MP4 output for streaming");
	}
	const OaStatus headerStatus = muxer.WriteFtypBox();
	if (not headerStatus.IsOk()) return headerStatus;
	
	return muxer;
}


OaStatus OaVideoMuxer::WritePacket(const OaEncodedFrame& InFrame)
{
	if (Finalized_) {
		return OaStatus::Error("Muxer already finalized");
	}
	
	if (InFrame.FrameSize == 0U or InFrame.FrameSize != InFrame.Bitstream.Size()) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"Encoded frame size does not match its bitstream");
	}
	if (not PacketDts_.Empty() and InFrame.PresentationTimestamp <= PacketDts_.Back()) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"Video packet timestamps must be strictly increasing");
	}
	auto sampleResult = AnnexBToLengthPrefixed(
		OaSpan<const OaU8>(InFrame.Bitstream.Data(), InFrame.Bitstream.Size()));
	if (not sampleResult.IsOk()) return sampleResult.GetStatus();
	OaVec<OaU8> sample = OaStdMove(*sampleResult);

	if (OutputFile_ == nullptr or not WriteFile(OutputFile_, sample.Data(), sample.Size())) {
		return OaStatus::Error(OaStatusCode::DataLoss, "Failed to stream video sample to MP4");
	}
	const OaU64 offset = 32U + MdatPayloadBytes_;
	MdatPayloadBytes_ += sample.Size();
	
	// Record packet metadata
	PacketOffsets_.PushBack(offset);
	PacketSizes_.PushBack(static_cast<OaU32>(sample.Size()));
	PacketDts_.PushBack(InFrame.PresentationTimestamp);
	PacketKeyframe_.PushBack(InFrame.IsKeyframe);
	++PacketCount_;
	
	return OaStatus::Ok();
}


OaStatus OaVideoMuxer::WriteAudioPacket(const OaEncodedAudioPacket& InPacket)
{
	if (Finalized_) return OaStatus::Error("Muxer already finalized");
	if (not Info_.AudioEnabled) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"Muxer was not created with an audio track");
	}
	if (InPacket.Bitstream.Empty() or InPacket.DurationFrames == 0U) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"Audio packet must contain PCM frames and a non-zero duration");
	}
	const OaU64 expectedBytes = static_cast<OaU64>(InPacket.DurationFrames)
		* Info_.AudioChannelCount * sizeof(OaI16);
	if (InPacket.Bitstream.Size() != expectedBytes) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"PcmS16 packet byte count does not match its duration and channel count");
	}
	if (OutputFile_ == nullptr or not WriteFile(
		OutputFile_, InPacket.Bitstream.Data(), InPacket.Bitstream.Size())) {
		return OaStatus::Error(OaStatusCode::DataLoss, "Failed to stream audio sample to MP4");
	}
	const OaU64 offset = 32U + MdatPayloadBytes_;
	MdatPayloadBytes_ += InPacket.Bitstream.Size();
	AudioPacketOffsets_.PushBack(offset);
	AudioPacketSizes_.PushBack(static_cast<OaU32>(InPacket.Bitstream.Size()));
	AudioPacketDurations_.PushBack(InPacket.DurationFrames);
	return OaStatus::Ok();
}


void OaVideoMuxer::SetAudioCodecConfig(OaSpan<const OaU8> InAudioSpecificConfig)
{
	AudioCodecConfig_.Resize(InAudioSpecificConfig.Size());
	if (!InAudioSpecificConfig.Empty()) {
		std::memcpy(AudioCodecConfig_.Data(), InAudioSpecificConfig.Data(),
			InAudioSpecificConfig.Size());
	}
}


void OaVideoMuxer::SetCodecConfig(const OaVec<OaU8>& InSps, const OaVec<OaU8>& InPps)
{
	Sps_ = InSps;
	Pps_ = InPps;
}


void OaVideoMuxer::SetCodecConfig(
	const OaVec<OaU8>& InVps,
	const OaVec<OaU8>& InSps,
	const OaVec<OaU8>& InPps)
{
	Vps_ = InVps;
	Sps_ = InSps;
	Pps_ = InPps;
}


OaStatus OaVideoMuxer::Finalize()
{
	if (Finalized_) {
		return OaStatus::Error("Muxer already finalized");
	}
	if (PacketCount_ == 0U) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"Cannot finalize an empty video");
	}
	if (Sps_.Empty() or Pps_.Empty()
		or (Info_.Codec == OaVideoCodec::H265 and Vps_.Empty())) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			Info_.Codec == OaVideoCodec::H264
				? "H.264 MP4 requires SPS and PPS codec configuration"
				: "H.265 MP4 requires VPS, SPS and PPS codec configuration");
	}
	if (OutputFile_ == nullptr) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition, "MP4 output is not open");
	}
	// Samples were streamed after a 16-byte ftyp and 16-byte extended-size
	// mdat header, so their absolute offsets are final from the moment written.
	// Patch the mdat largesize, append moov, flush and close.
	OaU8 largeSize[8] = {};
	WriteU64BE(largeSize, 16U + MdatPayloadBytes_);
	if (std::fseek(OutputFile_, 24L, SEEK_SET) != 0
		or not WriteFile(OutputFile_, largeSize, sizeof(largeSize))) {
		return OaStatus::Error(OaStatusCode::DataLoss, "Failed to finalize MP4 mdat size");
	}
	MdatData_.Clear();
	WriteMoovBox();
	OaVec<OaU8> moovBuf = std::move(MdatData_);
	if (std::fseek(OutputFile_, 0L, SEEK_END) != 0
		or not WriteFile(OutputFile_, moovBuf.Data(), moovBuf.Size())
		or std::fflush(OutputFile_) != 0) {
		return OaStatus::Error(OaStatusCode::DataLoss, "Failed to append MP4 movie metadata");
	}
	std::fclose(OutputFile_);
	OutputFile_ = nullptr;

	Finalized_ = true;
	return OaStatus::Ok();
}


OaStatus OaVideoMuxer::WriteFtypBox()
{
	OaU8 header[32] = {};
	WriteBoxHeader(header, 16U, 0x66747970U); // ftyp
	WriteU32BE(header + 8U, 0x69736f6dU);     // isom
	WriteU32BE(header + 12U, 512U);
	WriteU32BE(header + 16U, 1U);             // extended-size mdat
	WriteU32BE(header + 20U, 0x6d646174U);
	// largesize at +24 is patched during Finalize().
	if (OutputFile_ == nullptr or not WriteFile(OutputFile_, header, sizeof(header))) {
		return OaStatus::Error(OaStatusCode::DataLoss, "Failed to write MP4 stream header");
	}
	return OaStatus::Ok();
}


void OaVideoMuxer::WriteMoovBox()
{
	// Build complete movie metadata after all streamed samples are known.
	
	OaVec<OaU8> moovData;
	
	const OaU32 timescale = static_cast<OaU32>(Info_.TimebaseDen);
	auto PtsToTicks = [&](OaU64 InPtsUs) -> OaU32 {
		const OaU64 denominator = 1'000'000ULL * Info_.TimebaseNum;
		const OaU64 ticks = (InPtsUs * Info_.TimebaseDen + denominator / 2ULL) / denominator;
		return static_cast<OaU32>(std::min<OaU64>(ticks, 0xFFFFFFFFULL));
	};
	OaVec<OaU32> sampleDeltas;
	sampleDeltas.Resize(PacketCount_);
	const OaU32 nominalDelta = std::max(1U, timescale / Info_.FrameRate);
	OaU64 durationTicks64 = 0U;
	for (OaU32 i = 0; i < PacketCount_; ++i) {
		OaU32 delta = nominalDelta;
		if (i + 1U < PacketCount_) {
			const OaU32 current = PtsToTicks(PacketDts_[i]);
			const OaU32 next = PtsToTicks(PacketDts_[i + 1U]);
			delta = next > current ? next - current : 1U;
		} else if (i > 0U) {
			delta = sampleDeltas[i - 1U];
		}
		sampleDeltas[i] = delta;
		durationTicks64 += delta;
	}
	const OaU32 videoDurationTicks = static_cast<OaU32>(
		std::min<OaU64>(durationTicks64, 0xFFFFFFFFULL));
	OaU64 audioFrames = 0U;
	for (OaU32 duration : AudioPacketDurations_) audioFrames += duration;
	const OaU64 audibleAudioFrames = audioFrames > Info_.AudioPrimingFrames
		? audioFrames - Info_.AudioPrimingFrames : 0U;
	const OaU32 audioDurationTicks = Info_.AudioSampleRate == 0U ? 0U
		: static_cast<OaU32>(std::min<OaU64>(
			(audibleAudioFrames * timescale + Info_.AudioSampleRate / 2U)
				/ Info_.AudioSampleRate,
			0xFFFFFFFFULL));
	const OaU32 movieDurationTicks = std::max(videoDurationTicks, audioDurationTicks);

	// MovieHeaderBox version 0 is exactly 108 bytes. The previous 28-byte
	// array was written through offset 96 and corrupted the stack.
	constexpr OaU32 kMvhdSize = 108U;
	OaU8 mvhd[kMvhdSize] = {};
	WriteBoxHeader(mvhd, kMvhdSize, 0x6d766864);  // 'mvhd'
	WriteU32BE(mvhd + 8, 0);               // version + flags
	WriteU32BE(mvhd + 12, 0);              // creation_time
	WriteU32BE(mvhd + 16, 0);              // modification_time
	WriteU32BE(mvhd + 20, timescale);
	WriteU32BE(mvhd + 24, movieDurationTicks);
	WriteU32BE(mvhd + 28, 0x00010000);     // rate (1.0 = 0x00010000)
	WriteU16BE(mvhd + 32, 0x0100);         // volume (1.0)
	WriteU16BE(mvhd + 34, 0);              // reserved
	WriteU32BE(mvhd + 36, 0);              // reserved[0]
	WriteU32BE(mvhd + 40, 0);              // reserved[1]
	// matrix (9 values, all identity except [2][2] = 0x40000000)
	WriteU32BE(mvhd + 44, 0x00010000);
	WriteU32BE(mvhd + 48, 0);
	WriteU32BE(mvhd + 52, 0);
	WriteU32BE(mvhd + 56, 0);
	WriteU32BE(mvhd + 60, 0x00010000);
	WriteU32BE(mvhd + 64, 0);
	WriteU32BE(mvhd + 68, 0);
	WriteU32BE(mvhd + 72, 0);
	WriteU32BE(mvhd + 76, 0x40000000);
	// pre_defined + reserved (6 values)
	for (int i = 0; i < 6; ++i) {
		WriteU32BE(mvhd + 80 + i * 4, 0);
	}
	WriteU32BE(mvhd + 104, AudioPacketSizes_.Empty() ? 2U : 3U); // next_track_ID
	
	moovData.Resize(moovData.Size() + kMvhdSize);
	std::memcpy(moovData.Data() + moovData.Size() - kMvhdSize, mvhd, kMvhdSize);
	
	// Build the video track; an optional audio track is appended below.
	
	// Write tkhd (track header) - simplified
	OaU8 tkhd[92] = {};
	WriteBoxHeader(tkhd, 92, 0x746b6864);  // 'tkhd'
	WriteU32BE(tkhd + 8, 0x00000007);      // version + flags (track enabled, in movie, in preview)
	WriteU32BE(tkhd + 12, 0);              // creation_time
	WriteU32BE(tkhd + 16, 0);              // modification_time
	WriteU32BE(tkhd + 20, 1);              // track_ID
	WriteU32BE(tkhd + 24, 0);              // reserved
	WriteU32BE(tkhd + 28, videoDurationTicks);
	WriteU32BE(tkhd + 32, 0);              // reserved
	WriteU32BE(tkhd + 36, 0);              // reserved
	WriteU16BE(reinterpret_cast<OaU8*>(tkhd + 40), 0);       // layer
	WriteU16BE(reinterpret_cast<OaU8*>(tkhd + 42), 0);       // alternate_group
	WriteU16BE(reinterpret_cast<OaU8*>(tkhd + 44), 0);       // volume
	WriteU16BE(reinterpret_cast<OaU8*>(tkhd + 46), 0);       // reserved
	// matrix (same as mvhd)
	for (int i = 0; i < 9; ++i) {
		if (i == 4) {
			WriteU32BE(tkhd + 48 + i * 4, 0x00010000);
		} else if (i == 8) {
			WriteU32BE(tkhd + 48 + i * 4, 0x40000000);
		} else {
			WriteU32BE(tkhd + 48 + i * 4, 0);
		}
	}
	WriteU32BE(tkhd + 84, Info_.Width << 16);  // width (fixed-point 16.16)
	WriteU32BE(tkhd + 88, Info_.Height << 16); // height (fixed-point 16.16)

	// trak payload accumulates {tkhd, mdia}. We wrap it with a 'trak' box
	// header at the end and append to moovData; that way the demuxer's
	// moov → trak → mdia recursion actually reaches the sample table.
	OaVec<OaU8> trakData;
	trakData.Resize(trakData.Size() + 92);
	std::memcpy(trakData.Data() + trakData.Size() - 92, tkhd, 92);

	// Write mdia with mdhd, hdlr, minf and stbl.
	OaVec<OaU8> mdiaData;
	
	// MediaHeaderBox version 0 is 32 bytes.
	constexpr OaU32 kMdhdSize = 32U;
	OaU8 mdhd[kMdhdSize] = {};
	WriteBoxHeader(mdhd, kMdhdSize, 0x6d646864);  // 'mdhd'
	WriteU32BE(mdhd + 8, 0);               // version + flags
	WriteU32BE(mdhd + 12, 0);              // creation_time
	WriteU32BE(mdhd + 16, 0);              // modification_time
	WriteU32BE(mdhd + 20, timescale);
	WriteU32BE(mdhd + 24, videoDurationTicks);
	WriteU16BE(mdhd + 28, 0x55C4);          // language (und)
	WriteU16BE(mdhd + 30, 0);               // pre_defined
	
	mdiaData.Resize(mdiaData.Size() + kMdhdSize);
	std::memcpy(mdiaData.Data() + mdiaData.Size() - kMdhdSize, mdhd, kMdhdSize);
	
	// hdlr (HandlerBox §8.4.3): 8 header + 4 version_flags + 4 pre_defined
	//   + 4 handler_type + 12 reserved + Pascal-style name string.
	// "VideoHandler" + null terminator = 13 bytes.
	constexpr const char* kHandlerName = "VideoHandler";
	const OaUsize kHandlerNameLen      = 12;  // strlen("VideoHandler")
	const OaU32 kHdlrSize              = 8U + 4U + 4U + 4U + 12U + static_cast<OaU32>(kHandlerNameLen) + 1U;
	OaU8 hdlr[64] = {};                        // generous, > kHdlrSize
	WriteBoxHeader(hdlr, kHdlrSize, 0x68646c72);  // 'hdlr'
	WriteU32BE(hdlr + 8, 0);                   // version + flags
	WriteU32BE(hdlr + 12, 0);                  // pre_defined
	WriteU32BE(hdlr + 16, 0x76696465U);        // handler_type 'vide'
	WriteU32BE(hdlr + 20, 0);                  // reserved[0]
	WriteU32BE(hdlr + 24, 0);                  // reserved[1]
	WriteU32BE(hdlr + 28, 0);                  // reserved[2]
	std::memcpy(hdlr + 32, kHandlerName, kHandlerNameLen);
	hdlr[32 + kHandlerNameLen] = 0;            // null terminator

	mdiaData.Resize(mdiaData.Size() + kHdlrSize);
	std::memcpy(mdiaData.Data() + mdiaData.Size() - kHdlrSize, hdlr, kHdlrSize);
	
	// Write minf box with vmhd and stbl
	OaVec<OaU8> minfData;
	
	// vmhd (VideoMediaHeaderBox §12.1.2): 8 header + 4 version_flags
	//   + 2 graphicsmode + 2*3 opcolor = 20 bytes.
	constexpr OaU32 kVmhdSize = 20;
	OaU8 vmhd[20] = {};
	WriteBoxHeader(vmhd, kVmhdSize, 0x766d6864);    // 'vmhd'
	WriteU32BE(vmhd + 8, 0x00000001U);              // version=0, flags=1 (required)
	WriteU16BE(vmhd + 12, 0);                       // graphicsmode
	WriteU16BE(vmhd + 14, 0);                       // opcolor[0]
	WriteU16BE(vmhd + 16, 0);                       // opcolor[1]
	WriteU16BE(vmhd + 18, 0);                       // opcolor[2]

	minfData.Resize(minfData.Size() + kVmhdSize);
	std::memcpy(minfData.Data() + minfData.Size() - kVmhdSize, vmhd, kVmhdSize);
	
	// Write stbl box with sample table boxes
	OaVec<OaU8> stblData;
	
	// Write stsd sample description with avcC or hvcC codec configuration.
	OaVec<OaU8> stsdData;
	
	// Build avcC box if SPS/PPS are available
	OaVec<OaU8> avcCData;
	if (!Sps_.Empty() && !Pps_.Empty()) {
		// AVCDecoderConfigurationRecord (ISO/IEC 14496-15 §5.2.1.1):
		//   8  bytes  box header (size + type)
		//   1  byte   configurationVersion
		//   3  bytes  profile/compat/level (lifted from SPS)
		//   1  byte   lengthSizeMinusOne
		//   1  byte   numOfSPS (low 5 bits)
		//   2+sps    each SPS: length + data
		//   1  byte   numOfPPS
		//   2+pps    each PPS: length + data
		const OaU32 avcCSize =
			8U + 1U + 3U + 1U + 1U + 2U + static_cast<OaU32>(Sps_.Size())
			+ 1U + 2U + static_cast<OaU32>(Pps_.Size());
		avcCData.Resize(avcCSize);
		for (OaU32 i = 0; i < avcCSize; ++i) { avcCData[i] = 0; }

		WriteBoxHeader(avcCData.Data(), avcCSize, 0x61766343);  // 'avcC'
		avcCData[8]  = 1;       // configurationVersion
		// Lift profile/compat/level from the SPS itself (bytes 1..3 of the
		// SPS NAL body, which our encoder hands us without start code).
		if (Sps_.Size() >= 4) {
			avcCData[9]  = Sps_[1];   // profile_idc
			avcCData[10] = Sps_[2];   // constraint_set flags / profile_compat
			avcCData[11] = Sps_[3];   // level_idc
		} else {
			avcCData[9] = 0x42; avcCData[10] = 0xE0; avcCData[11] = 0x1E;
		}
		avcCData[12] = 0xFF;    // lengthSizeMinusOne = 3 + reserved bits
		avcCData[13] = 0xE1;    // numOfSPS = 1 + reserved bits
		WriteU16BE(avcCData.Data() + 14, static_cast<OaU16>(Sps_.Size()));
		std::memcpy(avcCData.Data() + 16, Sps_.Data(), Sps_.Size());
		const OaU32 ppsOffset = 16U + static_cast<OaU32>(Sps_.Size());
		avcCData[ppsOffset] = 1;  // numOfPPS
		WriteU16BE(avcCData.Data() + ppsOffset + 1, static_cast<OaU16>(Pps_.Size()));
		std::memcpy(avcCData.Data() + ppsOffset + 3, Pps_.Data(), Pps_.Size());
	}
	OaVec<OaU8> codecConfig = Info_.Codec == OaVideoCodec::H264
		? OaStdMove(avcCData) : BuildHvcc(Vps_, Sps_, Pps_);
	
	// VisualSampleEntry preamble (ISO/IEC 14496-12 §8.5.2):
	//   8 bytes  box header (size + type)
	//   6 bytes  SampleEntry::reserved
	//   2 bytes  SampleEntry::data_reference_index
	//   2 bytes  pre_defined
	//   2 bytes  reserved
	//   12 bytes pre_defined[3] (u32 × 3)
	//   2 bytes  width
	//   2 bytes  height
	//   4 bytes  horizresolution
	//   4 bytes  vertresolution
	//   4 bytes  reserved
	//   2 bytes  frame_count
	//   32 bytes compressorname
	//   2 bytes  depth
	//   2 bytes  pre_defined
	// Total: 86 bytes (8 header + 78 preamble). Codec config follows.
	const OaU32 sampleEntrySize = 86U + static_cast<OaU32>(codecConfig.Size());
	// stsd payload: 8 bytes (box header) + 4 bytes (version+flags) +
	// 4 bytes (entry_count) + sampleEntrySize.
	const OaU32 stsdSize = 8 + 4 + 4 + sampleEntrySize;

	stsdData.Resize(stsdSize);
	for (OaU32 i = 0; i < stsdSize; ++i) { stsdData[i] = 0; }
	WriteBoxHeader(stsdData.Data(), stsdSize, 0x73747364);  // 'stsd'
	WriteU32BE(stsdData.Data() + 8, 0);                     // version + flags
	WriteU32BE(stsdData.Data() + 12, 1);                    // entry_count

	OaU8* sampleEntry = stsdData.Data() + 16;
	WriteU32BE(sampleEntry + 0, sampleEntrySize);
	WriteU32BE(sampleEntry + 4,
		Info_.Codec == OaVideoCodec::H264 ? 0x61766331U : 0x68766331U); // avc1/hvc1
	// avc1 + 8..13 reserved zero (already memset)
	WriteU16BE(sampleEntry + 14, 1);             // data_reference_index
	// avc1 + 16..17 pre_defined (u16 = 0)
	// avc1 + 18..19 reserved (u16 = 0)
	// avc1 + 20..31 pre_defined[3] (u32 × 3 = 0)
	WriteU16BE(sampleEntry + 32, static_cast<OaU16>(Info_.Width));
	WriteU16BE(sampleEntry + 34, static_cast<OaU16>(Info_.Height));
	WriteU32BE(sampleEntry + 36, 0x00480000U);   // horizresolution 72 DPI
	WriteU32BE(sampleEntry + 40, 0x00480000U);   // vertresolution 72 DPI
	// avc1 + 44..47 reserved (u32 = 0)
	WriteU16BE(sampleEntry + 48, 1);             // frame_count
	// avc1 + 50..81 compressorname (32 bytes, zero-filled by memset above)
	WriteU16BE(sampleEntry + 82, 0x0018U);       // depth = 24
	WriteU16BE(sampleEntry + 84, 0xFFFFU);       // pre_defined

	// Append codec config inside the visual sample entry.
	if (!codecConfig.Empty()) {
		std::memcpy(sampleEntry + 86, codecConfig.Data(), codecConfig.Size());
	}
	
	stblData.Resize(stblData.Size() + stsdSize);
	std::memcpy(stblData.Data() + stblData.Size() - stsdSize, stsdData.Data(), stsdSize);
	
	// stts: run-length encode actual capture timestamp deltas. This supports
	// variable-rate PipeWire input while remaining compact for fixed-rate video.
	{
		struct SttsRun { OaU32 Count; OaU32 Delta; };
		OaVec<SttsRun> runs;
		for (OaU32 delta : sampleDeltas) {
			if (runs.Empty() or runs.Back().Delta != delta) runs.PushBack({1U, delta});
			else ++runs.Back().Count;
		}
		const OaU32 sttsSize = 16U + static_cast<OaU32>(runs.Size()) * 8U;
		OaVec<OaU8> stts;
		stts.Resize(sttsSize);
		std::memset(stts.Data(), 0, stts.Size());
		WriteBoxHeader(stts.Data(), sttsSize, 0x73747473);
		WriteU32BE(stts.Data() + 8, 0);
		WriteU32BE(stts.Data() + 12, static_cast<OaU32>(runs.Size()));
		for (OaU32 i = 0; i < runs.Size(); ++i) {
			WriteU32BE(stts.Data() + 16U + i * 8U, runs[i].Count);
			WriteU32BE(stts.Data() + 20U + i * 8U, runs[i].Delta);
		}
		stblData.Resize(stblData.Size() + stts.Size());
		std::memcpy(stblData.Data() + stblData.Size() - stts.Size(), stts.Data(), stts.Size());
	}

	// stsc (SampleToChunkBox §8.7.4): 8 header + 4 vf + 4 entry_count +
	//                                 12*entry (first_chunk, samples_per_chunk, sdix).
	{
		constexpr OaU32 kStscSize = 28;
		OaU8 stsc[kStscSize] = {};
		WriteBoxHeader(stsc, kStscSize, 0x73747363);  // 'stsc'
		WriteU32BE(stsc + 8,  0);                     // version + flags
		WriteU32BE(stsc + 12, 1);                     // entry_count
		WriteU32BE(stsc + 16, 1);                     // first_chunk
		WriteU32BE(stsc + 20, 1);                     // samples_per_chunk
		WriteU32BE(stsc + 24, 1);                     // sample_description_index
		stblData.Resize(stblData.Size() + kStscSize);
		std::memcpy(stblData.Data() + stblData.Size() - kStscSize, stsc, kStscSize);
	}

	// stsz (SampleSizeBox §8.7.3.2): 8 header + 4 vf + 4 sample_size
	//                                + 4 sample_count + 4*N entry sizes.
	{
		const OaU32 stszSize = 8 + 4 + 4 + 4 + 4 * PacketCount_;
		OaVec<OaU8> stsz;
		stsz.Resize(stszSize);
		for (OaU32 i = 0; i < stszSize; ++i) { stsz[i] = 0; }
		WriteBoxHeader(stsz.Data(), stszSize, 0x7374737a);  // 'stsz'
		WriteU32BE(stsz.Data() + 8,  0);              // version + flags
		WriteU32BE(stsz.Data() + 12, 0);              // sample_size (0 = variable)
		WriteU32BE(stsz.Data() + 16, PacketCount_);   // sample_count
		for (OaU32 i = 0; i < PacketCount_; ++i) {
			WriteU32BE(stsz.Data() + 20 + i * 4, PacketSizes_[i]);
		}
		stblData.Resize(stblData.Size() + stszSize);
		std::memcpy(stblData.Data() + stblData.Size() - stszSize, stsz.Data(), stszSize);
	}

	// stco/co64: use 64-bit chunk offsets automatically once a streamed file
	// crosses the 4 GiB boundary.
	{
		const bool largeOffsets = std::any_of(PacketOffsets_.begin(), PacketOffsets_.end(),
			[](OaU64 InOffset) { return InOffset > 0xFFFFFFFFULL; });
		const OaU32 stride = largeOffsets ? 8U : 4U;
		const OaU32 offsetSize = 16U + stride * PacketCount_;
		OaVec<OaU8> offsets(offsetSize, 0U);
		WriteBoxHeader(offsets.Data(), offsetSize,
			largeOffsets ? 0x636f3634U : 0x7374636fU); // co64/stco
		WriteU32BE(offsets.Data() + 12, PacketCount_);
		for (OaU32 i = 0; i < PacketCount_; ++i) {
			if (largeOffsets) WriteU64BE(offsets.Data() + 16U + i * 8U, PacketOffsets_[i]);
			else WriteU32BE(offsets.Data() + 16U + i * 4U,
				static_cast<OaU32>(PacketOffsets_[i]));
		}
		AppendBytes(stblData, offsets.Data(), offsets.Size());
	}

	// stss (SyncSampleBox §8.6.2.2): 8 header + 4 vf + 4 entry_count + 4*N indices.
	OaU32 keyframeCount = 0;
	for (OaU32 i = 0; i < PacketCount_; ++i) {
		if (PacketKeyframe_[i]) {
			++keyframeCount;
		}
	}

	if (keyframeCount > 0) {
		const OaU32 stssSize = 8 + 4 + 4 + 4 * keyframeCount;
		OaVec<OaU8> stss;
		stss.Resize(stssSize);
		for (OaU32 i = 0; i < stssSize; ++i) { stss[i] = 0; }
		WriteBoxHeader(stss.Data(), stssSize, 0x73747373);  // 'stss'
		WriteU32BE(stss.Data() + 8,  0);
		WriteU32BE(stss.Data() + 12, keyframeCount);
		
		OaU32 idx = 0;
		for (OaU32 i = 0; i < PacketCount_; ++i) {
			if (PacketKeyframe_[i]) {
				WriteU32BE(stss.Data() + 16 + idx * 4, i + 1);  // 1-based
				++idx;
			}
		}
		
		stblData.Resize(stblData.Size() + stssSize);
		std::memcpy(stblData.Data() + stblData.Size() - stssSize, stss.Data(), stssSize);
	}
	
	// Write stbl box header
	OaU32 stblSize = static_cast<OaU32>(stblData.Size() + 8);
	OaU8 stblHeader[8];
	WriteBoxHeader(stblHeader, stblSize, 0x7374626c);  // 'stbl'
	
	OaVec<OaU8> finalStbl;
	finalStbl.Resize(stblData.Size() + 8);
	std::memcpy(finalStbl.Data(), stblHeader, 8);
	std::memcpy(finalStbl.Data() + 8, stblData.Data(), stblData.Size());
	
	minfData.Resize(minfData.Size() + finalStbl.Size());
	std::memcpy(minfData.Data() + minfData.Size() - finalStbl.Size(), finalStbl.Data(), finalStbl.Size());
	
	// Write minf box header
	OaU32 minfSize = static_cast<OaU32>(minfData.Size() + 8);
	OaU8 minfHeader[8];
	WriteBoxHeader(minfHeader, minfSize, 0x6d696e66);  // 'minf'
	
	OaVec<OaU8> finalMinf;
	finalMinf.Resize(minfData.Size() + 8);
	std::memcpy(finalMinf.Data(), minfHeader, 8);
	std::memcpy(finalMinf.Data() + 8, minfData.Data(), minfData.Size());
	
	mdiaData.Resize(mdiaData.Size() + finalMinf.Size());
	std::memcpy(mdiaData.Data() + mdiaData.Size() - finalMinf.Size(), finalMinf.Data(), finalMinf.Size());
	
	// Write mdia box header
	OaU32 mdiaSize = static_cast<OaU32>(mdiaData.Size() + 8);
	OaU8 mdiaHeader[8];
	WriteBoxHeader(mdiaHeader, mdiaSize, 0x6d646961);  // 'mdia'
	
	OaVec<OaU8> finalMdia;
	finalMdia.Resize(mdiaData.Size() + 8);
	std::memcpy(finalMdia.Data(), mdiaHeader, 8);
	std::memcpy(finalMdia.Data() + 8, mdiaData.Data(), mdiaData.Size());

	// Append mdia into the trak payload (after tkhd) and wrap with a trak
	// box header before appending to moov. ParseTrakBox in the demuxer
	// recurses moov → trak → mdia → minf → stbl, so the trak wrap is
	// essential — without it Width / Height / sample table stay empty.
	trakData.Resize(trakData.Size() + finalMdia.Size());
	std::memcpy(trakData.Data() + trakData.Size() - finalMdia.Size(),
		finalMdia.Data(), finalMdia.Size());

	OaU32 trakSize = static_cast<OaU32>(trakData.Size() + 8);
	OaU8 trakHeader[8];
	WriteBoxHeader(trakHeader, trakSize, 0x7472616bU);  // 'trak'
	OaVec<OaU8> finalTrak;
	finalTrak.Resize(trakData.Size() + 8);
	std::memcpy(finalTrak.Data(), trakHeader, 8);
	std::memcpy(finalTrak.Data() + 8, trakData.Data(), trakData.Size());

	moovData.Resize(moovData.Size() + finalTrak.Size());
	std::memcpy(moovData.Data() + moovData.Size() - finalTrak.Size(),
		finalTrak.Data(), finalTrak.Size());

	if (!AudioPacketSizes_.Empty()) {
		auto audioTrak = BuildAudioTrack(Info_, AudioPacketOffsets_,
			AudioPacketDurations_, timescale);
		AppendBytes(moovData, audioTrak.Data(), audioTrak.Size());
	}
	
	// Write moov box header with total size
	OaU32 moovSize = static_cast<OaU32>(moovData.Size() + 8);
	OaU8 moovHeader[8];
	WriteBoxHeader(moovHeader, moovSize, 0x6d6f6f76);  // 'moov'
	
	// Prepend moov header to moov data
	OaVec<OaU8> finalMoov;
	finalMoov.Resize(moovData.Size() + 8);
	std::memcpy(finalMoov.Data(), moovHeader, 8);
	std::memcpy(finalMoov.Data() + 8, moovData.Data(), moovData.Size());
	
	// Prepend moov to mdat data
	OaVec<OaU8> finalData;
	finalData.Resize(MdatData_.Size() + finalMoov.Size());
	std::memcpy(finalData.Data(), finalMoov.Data(), finalMoov.Size());
	if (not MdatData_.Empty()) {
		std::memcpy(finalData.Data() + finalMoov.Size(),
			MdatData_.Data(), MdatData_.Size());
	}
	MdatData_ = std::move(finalData);
}
