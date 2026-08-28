// OA Vision — streaming MP4 muxer implementation.

#include <oa/vision/videoMuxer.h>
#include <oa/core/filesystem.h>
#include <oa/core/std/algo.h>

#include <stdio.h>

namespace
{

// Helper: write 16-bit big-endian
inline void writeU16BE(oa::U8* outPtr, oa::U16 inValue)
{
	outPtr[0] = static_cast<oa::U8>((inValue >> 8) & 0xFF);
	outPtr[1] = static_cast<oa::U8>(inValue & 0xFF);
}

// Helper: write 32-bit big-endian
inline void writeU32BE(oa::U8* outPtr, oa::U32 inValue)
{
	outPtr[0] = static_cast<oa::U8>((inValue >> 24) & 0xFF);
	outPtr[1] = static_cast<oa::U8>((inValue >> 16) & 0xFF);
	outPtr[2] = static_cast<oa::U8>((inValue >> 8) & 0xFF);
	outPtr[3] = static_cast<oa::U8>(inValue & 0xFF);
}

inline void writeU64BE(oa::U8* outPtr, oa::U64 inValue)
{
	writeU32BE(outPtr, static_cast<oa::U32>(inValue >> 32U));
	writeU32BE(outPtr + 4U, static_cast<oa::U32>(inValue));
}

bool writeFile(::FILE* inFile, const void* inData, oa::Usize inSize)
{
	return inSize == 0U or ::fwrite(inData, 1U, inSize, inFile) == inSize;
}

// Helper: write MP4 box header
inline void writeBoxHeader(oa::U8* outPtr, oa::U32 inSize, oa::U32 inType)
{
	writeU32BE(outPtr, inSize);
	writeU32BE(outPtr + 4, inType);
}

oa::Usize findAnnexBStartCode(const oa::U8* inData, oa::Usize inSize, oa::Usize inFrom, oa::Usize& outLength)
{
	for (oa::Usize i = inFrom; i + 3U <= inSize; ++i) {
		if (inData[i] != 0U or inData[i + 1U] != 0U) continue;
		if (inData[i + 2U] == 1U) {
			outLength = 3U;
			return i;
		}
		if (i + 4U <= inSize and inData[i + 2U] == 0U and inData[i + 3U] == 1U) {
			outLength = 4U;
			return i;
		}
	}
	outLength = 0U;
	return inSize;
}

// MP4 avc1 samples use length-prefixed NAL units. vulkan encoders normally
// return Annex-B. convert every access unit at the container boundary; keeping
// Annex-B in mdat makes the avcC lengthSizeMinusOne declaration a lie.
oa::Result<oa::Vector<oa::U8>> annexBToLengthPrefixed(const oa::Span<const oa::U8>& inBytes)
{
	oa::Vector<oa::U8> output;
	if (inBytes.empty()) return output;
	const oa::U8* data = inBytes.data();
	const oa::Usize size = inBytes.size();
	oa::Usize startLength = 0U;
	oa::Usize start = findAnnexBStartCode(data, size, 0U, startLength);
	if (start == size) {
		return oa::Status::error(oa::StatusCode::DataLoss,
			"MP4 video packet is not Annex-B framed");
	}
	while (start < size) {
		const oa::Usize payloadStart = start + startLength;
		oa::Usize nextLength = 0U;
		const oa::Usize next = findAnnexBStartCode(data, size, payloadStart, nextLength);
		oa::Usize payloadEnd = next;
		while (payloadEnd > payloadStart and data[payloadEnd - 1U] == 0U) --payloadEnd;
		const oa::Usize payloadSize = payloadEnd - payloadStart;
		if (payloadSize > 0xFFFFFFFFULL) {
			return oa::Status::error(oa::StatusCode::OutOfRange, "Video NAL exceeds MP4 length field");
		}
		if (payloadSize > 0U) {
			const oa::Usize oldSize = output.size();
			output.resize(oldSize + 4U + payloadSize);
			writeU32BE(output.data() + oldSize, static_cast<oa::U32>(payloadSize));
			oa::memcpy(output.data() + oldSize + 4U, data + payloadStart, payloadSize);
		}
		start = next;
		startLength = nextLength;
	}
	if (output.empty()) {
		return oa::Status::error(oa::StatusCode::DataLoss, "Video access unit contains no NAL units");
	}
	return output;
}


oa::Vector<oa::U8> buildHvcc(
	const oa::Vector<oa::U8>& inVps,
	const oa::Vector<oa::U8>& inSps,
	const oa::Vector<oa::U8>& inPps)
{
	// hEVCDecoderConfigurationRecord (ISO/IEC 14496-15 section 8.3.3.1):
	// 23-byte fixed record followed by complete VPS/SPS/PPS arrays. OA emits
	// main/8-bit/4:2:0 with one temporal layer and four-byte sample lengths.
	const oa::U32 payloadSize = 23U
		+ 3U + 2U + static_cast<oa::U32>(inVps.size())
		+ 3U + 2U + static_cast<oa::U32>(inSps.size())
		+ 3U + 2U + static_cast<oa::U32>(inPps.size());
	oa::Vector<oa::U8> box;
	box.resize(8U + payloadSize);
	oa::memset(box.data(), 0, box.size());
	writeBoxHeader(box.data(), static_cast<oa::U32>(box.size()), 0x68766343U); // hvcC
	oa::U8* record = box.data() + 8U;
	record[0] = 1U;          // configurationVersion
	record[1] = 1U;          // profile_space=0, tier=0, profile_idc=main(1)
	writeU32BE(record + 2U, 0x60000000U); // main-compatible profile flags
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

	oa::U32 offset = 23U;
	auto writeArray = [&](oa::U8 inNalType, const oa::Vector<oa::U8>& inNal) {
		record[offset++] = static_cast<oa::U8>(0x80U | inNalType);
		writeU16BE(record + offset, 1U);
		offset += 2U;
		writeU16BE(record + offset, static_cast<oa::U16>(inNal.size()));
		offset += 2U;
		oa::memcpy(record + offset, inNal.data(), inNal.size());
		offset += static_cast<oa::U32>(inNal.size());
	};
	writeArray(32U, inVps);
	writeArray(33U, inSps);
	writeArray(34U, inPps);
	return box;
}

void appendBytes(oa::Vector<oa::U8>& out, const void* inData, oa::Usize inSize)
{
	const oa::Usize offset = out.size();
	out.resize(offset + inSize);
	if (inSize > 0U) oa::memcpy(out.data() + offset, inData, inSize);
}

oa::Vector<oa::U8> wrapBox(oa::U32 inType, const oa::Vector<oa::U8>& inPayload)
{
	oa::Vector<oa::U8> box;
	box.resize(8U + inPayload.size());
	writeBoxHeader(box.data(), static_cast<oa::U32>(box.size()), inType);
	if (!inPayload.empty()) oa::memcpy(box.data() + 8U, inPayload.data(), inPayload.size());
	return box;
}

oa::Vector<oa::U8> buildAudioTrack(
	const oa::VideoMuxerConfig& inConfig,
	const oa::Vector<oa::U64>& inOffsets,
	const oa::Vector<oa::U32>& inDurations,
	oa::U32 inMovieTimescale)
{
	oa::U64 mediaDuration64 = 0U;
	for (oa::U32 duration : inDurations) mediaDuration64 += duration;
	const oa::U32 mediaDuration = static_cast<oa::U32>(
		oa::min<oa::U64>(mediaDuration64, oa::Limits<oa::U32>::max()));
	const oa::U64 audibleFrames = mediaDuration64 > inConfig.audioPrimingFrames
		? mediaDuration64 - inConfig.audioPrimingFrames : 0U;
	const oa::U32 movieDuration = static_cast<oa::U32>(oa::min<oa::U64>(
		(audibleFrames * inMovieTimescale + inConfig.audioSampleRate / 2U)
			/ inConfig.audioSampleRate,
		oa::Limits<oa::U32>::max()));

	oa::Vector<oa::U8> trakPayload;
	oa::U8 tkhd[92] = {};
	writeBoxHeader(tkhd, 92U, 0x746b6864U);
	writeU32BE(tkhd + 8U, 0x00000007U);
	writeU32BE(tkhd + 20U, 2U);
	writeU32BE(tkhd + 28U, movieDuration);
	writeU16BE(tkhd + 44U, 0x0100U); // audio track volume 1.0
	writeU32BE(tkhd + 48U, 0x00010000U);
	writeU32BE(tkhd + 64U, 0x00010000U);
	writeU32BE(tkhd + 80U, 0x40000000U);
	appendBytes(trakPayload, tkhd, sizeof(tkhd));

	if (inConfig.audioPrimingFrames > 0U) {
		oa::Vector<oa::U8> elstPayload(20U, 0U);
		writeU32BE(elstPayload.data() + 4U, 1U);
		writeU32BE(elstPayload.data() + 8U, movieDuration);
		writeU32BE(elstPayload.data() + 12U, inConfig.audioPrimingFrames);
		writeU16BE(elstPayload.data() + 16U, 1U);
		auto elst = wrapBox(0x656c7374U, elstPayload);
		auto edts = wrapBox(0x65647473U, elst);
		appendBytes(trakPayload, edts.data(), edts.size());
	}

	oa::Vector<oa::U8> mdiaPayload;
	oa::U8 mdhd[32] = {};
	writeBoxHeader(mdhd, 32U, 0x6d646864U);
	writeU32BE(mdhd + 20U, inConfig.audioSampleRate);
	writeU32BE(mdhd + 24U, mediaDuration);
	writeU16BE(mdhd + 28U, 0x55C4U);
	appendBytes(mdiaPayload, mdhd, sizeof(mdhd));

	constexpr char kName[] = "SoundHandler";
	oa::Vector<oa::U8> hdlr(32U + sizeof(kName), 0U);
	writeBoxHeader(hdlr.data(), static_cast<oa::U32>(hdlr.size()), 0x68646c72U);
	writeU32BE(hdlr.data() + 16U, 0x736f756eU); // soun
	oa::memcpy(hdlr.data() + 32U, kName, sizeof(kName));
	appendBytes(mdiaPayload, hdlr.data(), hdlr.size());

	oa::Vector<oa::U8> minfPayload;
	oa::U8 smhd[16] = {};
	writeBoxHeader(smhd, 16U, 0x736d6864U);
	appendBytes(minfPayload, smhd, sizeof(smhd));

	oa::Vector<oa::U8> stblPayload;
	// ISO/IEC 23003-5 uncompressed audio: signed 16-bit little-endian PCM.
	// `ipcm` is the sample entry and `pcmC` carries byte order/sample width.
	constexpr oa::U32 kPcmcSize = 14U;
	constexpr oa::U32 kSampleEntrySize = 36U + kPcmcSize;
	const oa::U32 sampleEntrySize = kSampleEntrySize;
	oa::Vector<oa::U8> stsd(16U + sampleEntrySize, 0U);
	writeBoxHeader(stsd.data(), static_cast<oa::U32>(stsd.size()), 0x73747364U);
	writeU32BE(stsd.data() + 12U, 1U);
	oa::U8* ipcm = stsd.data() + 16U;
	writeBoxHeader(ipcm, sampleEntrySize, 0x6970636dU); // ipcm
	writeU16BE(ipcm + 14U, 1U);
	writeU16BE(ipcm + 24U, static_cast<oa::U16>(inConfig.audioChannelCount));
	writeU16BE(ipcm + 26U, 16U);
	writeU32BE(ipcm + 32U, inConfig.audioSampleRate << 16U);
	oa::U8* pcmc = ipcm + 36U;
	writeBoxHeader(pcmc, kPcmcSize, 0x70636d43U); // pcmC
	pcmc[12U] = 1U; // little-endian
	pcmc[13U] = 16U;
	appendBytes(stblPayload, stsd.data(), stsd.size());

	// One PCM frame is one MP4 sample and lasts one audio-timescale tick.
	oa::Vector<oa::U8> stts(24U, 0U);
	writeBoxHeader(stts.data(), static_cast<oa::U32>(stts.size()), 0x73747473U);
	writeU32BE(stts.data() + 12U, 1U);
	writeU32BE(stts.data() + 16U, mediaDuration);
	writeU32BE(stts.data() + 20U, 1U);
	appendBytes(stblPayload, stts.data(), stts.size());

	struct ChunkRun { oa::U32 firstChunk; oa::U32 samplesPerChunk; };
	oa::Vector<ChunkRun> chunkRuns;
	for (oa::U32 i = 0U; i < inDurations.size(); ++i) {
		if (chunkRuns.empty() or chunkRuns.back().samplesPerChunk != inDurations[i]) {
			chunkRuns.pushBack({i + 1U, inDurations[i]});
		}
	}
	oa::Vector<oa::U8> stsc(16U + chunkRuns.size() * 12U, 0U);
	writeBoxHeader(stsc.data(), static_cast<oa::U32>(stsc.size()), 0x73747363U);
	writeU32BE(stsc.data() + 12U, static_cast<oa::U32>(chunkRuns.size()));
	for (oa::U32 i = 0U; i < chunkRuns.size(); ++i) {
		writeU32BE(stsc.data() + 16U + i * 12U, chunkRuns[i].firstChunk);
		writeU32BE(stsc.data() + 20U + i * 12U, chunkRuns[i].samplesPerChunk);
		writeU32BE(stsc.data() + 24U + i * 12U, 1U);
	}
	appendBytes(stblPayload, stsc.data(), stsc.size());

	oa::Vector<oa::U8> stsz(20U, 0U);
	writeBoxHeader(stsz.data(), static_cast<oa::U32>(stsz.size()), 0x7374737aU);
	writeU32BE(stsz.data() + 12U, inConfig.audioChannelCount * sizeof(oa::I16));
	writeU32BE(stsz.data() + 16U, mediaDuration);
	appendBytes(stblPayload, stsz.data(), stsz.size());

	const bool largeOffsets = oa::anyOf(inOffsets.begin(), inOffsets.end(),
		[](oa::U64 inOffset) { return inOffset > 0xFFFFFFFFULL; });
	oa::Vector<oa::U8> offsets(16U + inOffsets.size() * (largeOffsets ? 8U : 4U), 0U);
	writeBoxHeader(offsets.data(), static_cast<oa::U32>(offsets.size()),
		largeOffsets ? 0x636f3634U : 0x7374636fU); // co64 / stco
	writeU32BE(offsets.data() + 12U, static_cast<oa::U32>(inOffsets.size()));
	for (oa::U32 i = 0U; i < inOffsets.size(); ++i) {
		if (largeOffsets) writeU64BE(offsets.data() + 16U + i * 8U, inOffsets[i]);
		else writeU32BE(offsets.data() + 16U + i * 4U, static_cast<oa::U32>(inOffsets[i]));
	}
	appendBytes(stblPayload, offsets.data(), offsets.size());

	auto stbl = wrapBox(0x7374626cU, stblPayload);
	appendBytes(minfPayload, stbl.data(), stbl.size());
	auto minf = wrapBox(0x6d696e66U, minfPayload);
	appendBytes(mdiaPayload, minf.data(), minf.size());
	auto mdia = wrapBox(0x6d646961U, mdiaPayload);
	appendBytes(trakPayload, mdia.data(), mdia.size());
	return wrapBox(0x7472616bU, trakPayload);
}

}  // namespace


void oa::VideoMuxer::reset_() noexcept
{
	if (outputFile_ != nullptr) ::fclose(outputFile_);
	outputFile_ = nullptr;
	mdatPayloadBytes_ = 0U;
	config_ = {};
	mdatData_.clear();
	packetOffsets_.clear();
	packetSizes_.clear();
	packetDts_.clear();
	packetKeyframe_.clear();
	audioPacketOffsets_.clear();
	audioPacketSizes_.clear();
	audioPacketDurations_.clear();
	audioCodecConfig_.clear();
	packetCount_ = 0;
	finalized_ = false;
	vps_.clear();
	sps_.clear();
	pps_.clear();
}


oa::VideoMuxer::VideoMuxer(oa::VideoMuxer&& inOther) noexcept
	: config_(inOther.config_)
	, mdatData_(oa::move(inOther.mdatData_))
	, outputFile_(inOther.outputFile_)
	, mdatPayloadBytes_(inOther.mdatPayloadBytes_)
	, packetOffsets_(oa::move(inOther.packetOffsets_))
	, packetSizes_(oa::move(inOther.packetSizes_))
	, packetDts_(oa::move(inOther.packetDts_))
	, packetKeyframe_(oa::move(inOther.packetKeyframe_))
	, audioPacketOffsets_(oa::move(inOther.audioPacketOffsets_))
	, audioPacketSizes_(oa::move(inOther.audioPacketSizes_))
	, audioPacketDurations_(oa::move(inOther.audioPacketDurations_))
	, audioCodecConfig_(oa::move(inOther.audioCodecConfig_))
	, packetCount_(inOther.packetCount_)
	, finalized_(inOther.finalized_)
	, vps_(oa::move(inOther.vps_))
	, sps_(oa::move(inOther.sps_))
	, pps_(oa::move(inOther.pps_))
{
	inOther.outputFile_ = nullptr;
	inOther.reset_();
}


oa::VideoMuxer& oa::VideoMuxer::operator=(oa::VideoMuxer&& inOther) noexcept
{
	if (this != &inOther) {
		reset_();
		config_ = inOther.config_;
		mdatData_ = oa::move(inOther.mdatData_);
		outputFile_ = inOther.outputFile_;
		mdatPayloadBytes_ = inOther.mdatPayloadBytes_;
		packetOffsets_ = oa::move(inOther.packetOffsets_);
		packetSizes_ = oa::move(inOther.packetSizes_);
		packetDts_ = oa::move(inOther.packetDts_);
		packetKeyframe_ = oa::move(inOther.packetKeyframe_);
		audioPacketOffsets_ = oa::move(inOther.audioPacketOffsets_);
		audioPacketSizes_ = oa::move(inOther.audioPacketSizes_);
		audioPacketDurations_ = oa::move(inOther.audioPacketDurations_);
		audioCodecConfig_ = oa::move(inOther.audioCodecConfig_);
		packetCount_ = inOther.packetCount_;
		finalized_ = inOther.finalized_;
		vps_ = oa::move(inOther.vps_);
		sps_ = oa::move(inOther.sps_);
		pps_ = oa::move(inOther.pps_);
		inOther.outputFile_ = nullptr;
		inOther.reset_();
	}
	return *this;
}


oa::VideoMuxer::~VideoMuxer()
{
	reset_();
}


oa::Status oa::VideoMuxer::close()
{
	int closeResult = 0;
	if (outputFile_ != nullptr) {
		closeResult = ::fclose(outputFile_);
		outputFile_ = nullptr;
	}
	reset_();
	if (closeResult != 0) {
		return oa::Status::error(oa::StatusCode::DataLoss,
			"Failed to close MP4 output");
	}
	return oa::Status::ok();
}


oa::Result<oa::VideoMuxer> oa::VideoMuxer::create(
	const oa::VideoMuxerConfig& inConfig)
{
	if (inConfig.outputPath.empty()) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "Video muxer path is empty");
	}
	if (inConfig.width == 0U or inConfig.height == 0U or inConfig.frameRate == 0U
		or inConfig.timebaseNum == 0U or inConfig.timebaseDen == 0U) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"Video muxer requires extent, frame rate and a valid timebase");
	}
	if (inConfig.codec != oa::VideoCodec::H264 and inConfig.codec != oa::VideoCodec::H265) {
		return oa::Status::error(oa::StatusCode::Unimplemented,
			"MP4 muxing is implemented for H.264 and H.265");
	}
	if (inConfig.audioEnabled and (inConfig.audioCodec != oa::AudioCodec::PcmS16
		or inConfig.audioSampleRate == 0U or inConfig.audioSampleRate > 65'535U
		or inConfig.audioChannelCount == 0U or inConfig.audioChannelCount > 8U)) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"MP4 audio requires native PcmS16, sample rate <= 65535 and 1..8 channels");
	}
	oa::VideoMuxer muxer;
	muxer.config_ = inConfig;
	
	muxer.outputFile_ = ::fopen(inConfig.outputPath.cStr(), "wb+");
	if (muxer.outputFile_ == nullptr) {
		return oa::Status::error(oa::StatusCode::Unavailable,
			"Cannot open MP4 output for streaming");
	}
	const oa::Status headerStatus = muxer.writeFtypBox();
	if (not headerStatus.isOk()) return headerStatus;
	
	return muxer;
}


oa::Status oa::VideoMuxer::writePacket(const oa::EncodedVideoPacket& inFrame)
{
	if (finalized_) {
		return oa::Status::error("Muxer already finalized");
	}
	
	if (inFrame.frameSize == 0U or inFrame.frameSize != inFrame.bitstream.size()) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"Encoded frame size does not match its bitstream");
	}
	if (not packetDts_.empty() and inFrame.presentationTimestamp <= packetDts_.back()) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"Video packet timestamps must be strictly increasing");
	}
	auto sampleResult = annexBToLengthPrefixed(
		oa::Span<const oa::U8>(inFrame.bitstream.data(), inFrame.bitstream.size()));
	if (not sampleResult.isOk()) return sampleResult.getStatus();
	oa::Vector<oa::U8> sample = oa::move(*sampleResult);

	if (outputFile_ == nullptr or not writeFile(outputFile_, sample.data(), sample.size())) {
		return oa::Status::error(oa::StatusCode::DataLoss, "Failed to stream video sample to MP4");
	}
	const oa::U64 offset = 32U + mdatPayloadBytes_;
	mdatPayloadBytes_ += sample.size();
	
	// Record packet metadata
	packetOffsets_.pushBack(offset);
	packetSizes_.pushBack(static_cast<oa::U32>(sample.size()));
	packetDts_.pushBack(inFrame.presentationTimestamp);
	packetKeyframe_.pushBack(inFrame.isKeyframe);
	++packetCount_;
	
	return oa::Status::ok();
}


oa::Status oa::VideoMuxer::writeAudioPacket(const oa::EncodedAudioPacket& inPacket)
{
	if (finalized_) return oa::Status::error("Muxer already finalized");
	if (not config_.audioEnabled) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"Muxer was not created with an audio track");
	}
	if (inPacket.bitstream.empty() or inPacket.durationFrames == 0U) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"Audio packet must contain PCM frames and a non-zero duration");
	}
	const oa::U64 expectedBytes = static_cast<oa::U64>(inPacket.durationFrames)
		* config_.audioChannelCount * sizeof(oa::I16);
	if (inPacket.bitstream.size() != expectedBytes) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"PcmS16 packet byte count does not match its duration and channel count");
	}
	if (outputFile_ == nullptr or not writeFile(
		outputFile_, inPacket.bitstream.data(), inPacket.bitstream.size())) {
		return oa::Status::error(oa::StatusCode::DataLoss, "Failed to stream audio sample to MP4");
	}
	const oa::U64 offset = 32U + mdatPayloadBytes_;
	mdatPayloadBytes_ += inPacket.bitstream.size();
	audioPacketOffsets_.pushBack(offset);
	audioPacketSizes_.pushBack(static_cast<oa::U32>(inPacket.bitstream.size()));
	audioPacketDurations_.pushBack(inPacket.durationFrames);
	return oa::Status::ok();
}


void oa::VideoMuxer::setAudioCodecConfig(oa::Span<const oa::U8> inAudioSpecificConfig)
{
	audioCodecConfig_.resize(inAudioSpecificConfig.size());
	if (!inAudioSpecificConfig.empty()) {
		oa::memcpy(audioCodecConfig_.data(), inAudioSpecificConfig.data(),
			inAudioSpecificConfig.size());
	}
}


void oa::VideoMuxer::setCodecConfig(const oa::Vector<oa::U8>& inSps, const oa::Vector<oa::U8>& inPps)
{
	sps_ = inSps;
	pps_ = inPps;
}


void oa::VideoMuxer::setCodecConfig(
	const oa::Vector<oa::U8>& inVps,
	const oa::Vector<oa::U8>& inSps,
	const oa::Vector<oa::U8>& inPps)
{
	vps_ = inVps;
	sps_ = inSps;
	pps_ = inPps;
}


oa::Status oa::VideoMuxer::finalize()
{
	if (finalized_) {
		return oa::Status::error("Muxer already finalized");
	}
	if (packetCount_ == 0U) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"Cannot finalize an empty video");
	}
	if (sps_.empty() or pps_.empty()
		or (config_.codec == oa::VideoCodec::H265 and vps_.empty())) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			config_.codec == oa::VideoCodec::H264
				? "H.264 MP4 requires SPS and PPS codec configuration"
				: "H.265 MP4 requires VPS, SPS and PPS codec configuration");
	}
	if (outputFile_ == nullptr) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition, "MP4 output is not open");
	}
	// samples were streamed after a 16-byte ftyp and 16-byte extended-size
	// mdat header, so their absolute offsets are final from the moment written.
	// patch the mdat largesize, append moov, flush and close.
	oa::U8 largeSize[8] = {};
	writeU64BE(largeSize, 16U + mdatPayloadBytes_);
	if (::fseek(outputFile_, 24L, SEEK_SET) != 0
		or not writeFile(outputFile_, largeSize, sizeof(largeSize))) {
		return oa::Status::error(oa::StatusCode::DataLoss, "Failed to finalize MP4 mdat size");
	}
	mdatData_.clear();
	writeMoovBox();
	oa::Vector<oa::U8> moovBuf = oa::move(mdatData_);
	if (::fseek(outputFile_, 0L, SEEK_END) != 0
		or not writeFile(outputFile_, moovBuf.data(), moovBuf.size())
		or ::fflush(outputFile_) != 0) {
		return oa::Status::error(oa::StatusCode::DataLoss, "Failed to append MP4 movie metadata");
	}
	::fclose(outputFile_);
	outputFile_ = nullptr;

	finalized_ = true;
	return oa::Status::ok();
}


oa::Status oa::VideoMuxer::writeFtypBox()
{
	oa::U8 header[32] = {};
	writeBoxHeader(header, 16U, 0x66747970U); // ftyp
	writeU32BE(header + 8U, 0x69736f6dU);     // isom
	writeU32BE(header + 12U, 512U);
	writeU32BE(header + 16U, 1U);             // extended-size mdat
	writeU32BE(header + 20U, 0x6d646174U);
	// largesize at +24 is patched during finalize().
	if (outputFile_ == nullptr or not writeFile(outputFile_, header, sizeof(header))) {
		return oa::Status::error(oa::StatusCode::DataLoss, "Failed to write MP4 stream header");
	}
	return oa::Status::ok();
}


void oa::VideoMuxer::writeMoovBox()
{
	// Build complete movie metadata after all streamed samples are known.
	
	oa::Vector<oa::U8> moovData;
	
	const oa::U32 timescale = static_cast<oa::U32>(config_.timebaseDen);
	auto ptsToTicks = [&](oa::U64 inPtsUs) -> oa::U32 {
		const oa::U64 denominator = 1'000'000ULL * config_.timebaseNum;
		const oa::U64 ticks = (inPtsUs * config_.timebaseDen + denominator / 2ULL) / denominator;
		return static_cast<oa::U32>(oa::min<oa::U64>(ticks, 0xFFFFFFFFULL));
	};
	oa::Vector<oa::U32> sampleDeltas;
	sampleDeltas.resize(packetCount_);
	const oa::U32 nominalDelta = oa::max(1U, timescale / config_.frameRate);
	oa::U64 durationTicks64 = 0U;
	for (oa::U32 i = 0; i < packetCount_; ++i) {
		oa::U32 delta = nominalDelta;
		if (i + 1U < packetCount_) {
			const oa::U32 current = ptsToTicks(packetDts_[i]);
			const oa::U32 next = ptsToTicks(packetDts_[i + 1U]);
			delta = next > current ? next - current : 1U;
		} else if (i > 0U) {
			delta = sampleDeltas[i - 1U];
		}
		sampleDeltas[i] = delta;
		durationTicks64 += delta;
	}
	const oa::U32 videoDurationTicks = static_cast<oa::U32>(
		oa::min<oa::U64>(durationTicks64, 0xFFFFFFFFULL));
	oa::U64 audioFrames = 0U;
	for (oa::U32 duration : audioPacketDurations_) audioFrames += duration;
	const oa::U64 audibleAudioFrames = audioFrames > config_.audioPrimingFrames
		? audioFrames - config_.audioPrimingFrames : 0U;
	const oa::U32 audioDurationTicks = config_.audioSampleRate == 0U ? 0U
		: static_cast<oa::U32>(oa::min<oa::U64>(
			(audibleAudioFrames * timescale + config_.audioSampleRate / 2U)
				/ config_.audioSampleRate,
			0xFFFFFFFFULL));
	const oa::U32 movieDurationTicks = oa::max(videoDurationTicks, audioDurationTicks);

	// MovieHeaderBox version 0 is exactly 108 bytes. The previous 28-byte
	// array was written through offset 96 and corrupted the stack.
	constexpr oa::U32 kMvhdSize = 108U;
	oa::U8 mvhd[kMvhdSize] = {};
	writeBoxHeader(mvhd, kMvhdSize, 0x6d766864);  // 'mvhd'
	writeU32BE(mvhd + 8, 0);               // version + flags
	writeU32BE(mvhd + 12, 0);              // creation_time
	writeU32BE(mvhd + 16, 0);              // modification_time
	writeU32BE(mvhd + 20, timescale);
	writeU32BE(mvhd + 24, movieDurationTicks);
	writeU32BE(mvhd + 28, 0x00010000);     // rate (1.0 = 0x00010000)
	writeU16BE(mvhd + 32, 0x0100);         // volume (1.0)
	writeU16BE(mvhd + 34, 0);              // reserved
	writeU32BE(mvhd + 36, 0);              // reserved[0]
	writeU32BE(mvhd + 40, 0);              // reserved[1]
	// matrix (9 values, all identity except [2][2] = 0x40000000)
	writeU32BE(mvhd + 44, 0x00010000);
	writeU32BE(mvhd + 48, 0);
	writeU32BE(mvhd + 52, 0);
	writeU32BE(mvhd + 56, 0);
	writeU32BE(mvhd + 60, 0x00010000);
	writeU32BE(mvhd + 64, 0);
	writeU32BE(mvhd + 68, 0);
	writeU32BE(mvhd + 72, 0);
	writeU32BE(mvhd + 76, 0x40000000);
	// pre_defined + reserved (6 values)
	for (int i = 0; i < 6; ++i) {
		writeU32BE(mvhd + 80 + i * 4, 0);
	}
	writeU32BE(mvhd + 104, audioPacketSizes_.empty() ? 2U : 3U); // next_track_ID
	
	moovData.resize(moovData.size() + kMvhdSize);
	oa::memcpy(moovData.data() + moovData.size() - kMvhdSize, mvhd, kMvhdSize);
	
	// Build the video track; an optional audio track is appended below.
	
	// Write tkhd (track header) - simplified
	oa::U8 tkhd[92] = {};
	writeBoxHeader(tkhd, 92, 0x746b6864);  // 'tkhd'
	writeU32BE(tkhd + 8, 0x00000007);      // version + flags (track enabled, in movie, in preview)
	writeU32BE(tkhd + 12, 0);              // creation_time
	writeU32BE(tkhd + 16, 0);              // modification_time
	writeU32BE(tkhd + 20, 1);              // track_ID
	writeU32BE(tkhd + 24, 0);              // reserved
	writeU32BE(tkhd + 28, videoDurationTicks);
	writeU32BE(tkhd + 32, 0);              // reserved
	writeU32BE(tkhd + 36, 0);              // reserved
	writeU16BE(reinterpret_cast<oa::U8*>(tkhd + 40), 0);       // layer
	writeU16BE(reinterpret_cast<oa::U8*>(tkhd + 42), 0);       // alternate_group
	writeU16BE(reinterpret_cast<oa::U8*>(tkhd + 44), 0);       // volume
	writeU16BE(reinterpret_cast<oa::U8*>(tkhd + 46), 0);       // reserved
	// matrix (same as mvhd)
	for (int i = 0; i < 9; ++i) {
		if (i == 4) {
			writeU32BE(tkhd + 48 + i * 4, 0x00010000);
		} else if (i == 8) {
			writeU32BE(tkhd + 48 + i * 4, 0x40000000);
		} else {
			writeU32BE(tkhd + 48 + i * 4, 0);
		}
	}
	writeU32BE(tkhd + 84, config_.width << 16);  // width (fixed-point 16.16)
	writeU32BE(tkhd + 88, config_.height << 16); // height (fixed-point 16.16)

	// trak payload accumulates {tkhd, mdia}. We wrap it with a 'trak' box
	// header at the end and append to moovData; that way the demuxer's
	// moov → trak → mdia recursion actually reaches the sample table.
	oa::Vector<oa::U8> trakData;
	trakData.resize(trakData.size() + 92);
	oa::memcpy(trakData.data() + trakData.size() - 92, tkhd, 92);

	// Write mdia with mdhd, hdlr, minf and stbl.
	oa::Vector<oa::U8> mdiaData;
	
	// MediaHeaderBox version 0 is 32 bytes.
	constexpr oa::U32 kMdhdSize = 32U;
	oa::U8 mdhd[kMdhdSize] = {};
	writeBoxHeader(mdhd, kMdhdSize, 0x6d646864);  // 'mdhd'
	writeU32BE(mdhd + 8, 0);               // version + flags
	writeU32BE(mdhd + 12, 0);              // creation_time
	writeU32BE(mdhd + 16, 0);              // modification_time
	writeU32BE(mdhd + 20, timescale);
	writeU32BE(mdhd + 24, videoDurationTicks);
	writeU16BE(mdhd + 28, 0x55C4);          // language (und)
	writeU16BE(mdhd + 30, 0);               // pre_defined
	
	mdiaData.resize(mdiaData.size() + kMdhdSize);
	oa::memcpy(mdiaData.data() + mdiaData.size() - kMdhdSize, mdhd, kMdhdSize);
	
	// hdlr (HandlerBox §8.4.3): 8 header + 4 version_flags + 4 pre_defined
	//   + 4 handler_type + 12 reserved + Pascal-style name string.
	// "VideoHandler" + null terminator = 13 bytes.
	constexpr const char* kHandlerName = "VideoHandler";
	const oa::Usize kHandlerNameLen      = 12;  // strlen("VideoHandler")
	const oa::U32 kHdlrSize              = 8U + 4U + 4U + 4U + 12U + static_cast<oa::U32>(kHandlerNameLen) + 1U;
	oa::U8 hdlr[64] = {};                        // generous, > kHdlrSize
	writeBoxHeader(hdlr, kHdlrSize, 0x68646c72);  // 'hdlr'
	writeU32BE(hdlr + 8, 0);                   // version + flags
	writeU32BE(hdlr + 12, 0);                  // pre_defined
	writeU32BE(hdlr + 16, 0x76696465U);        // handler_type 'vide'
	writeU32BE(hdlr + 20, 0);                  // reserved[0]
	writeU32BE(hdlr + 24, 0);                  // reserved[1]
	writeU32BE(hdlr + 28, 0);                  // reserved[2]
	oa::memcpy(hdlr + 32, kHandlerName, kHandlerNameLen);
	hdlr[32 + kHandlerNameLen] = 0;            // null terminator

	mdiaData.resize(mdiaData.size() + kHdlrSize);
	oa::memcpy(mdiaData.data() + mdiaData.size() - kHdlrSize, hdlr, kHdlrSize);
	
	// Write minf box with vmhd and stbl
	oa::Vector<oa::U8> minfData;
	
	// vmhd (VideoMediaHeaderBox §12.1.2): 8 header + 4 version_flags
	//   + 2 graphicsmode + 2*3 opcolor = 20 bytes.
	constexpr oa::U32 kVmhdSize = 20;
	oa::U8 vmhd[20] = {};
	writeBoxHeader(vmhd, kVmhdSize, 0x766d6864);    // 'vmhd'
	writeU32BE(vmhd + 8, 0x00000001U);              // version=0, flags=1 (required)
	writeU16BE(vmhd + 12, 0);                       // graphicsmode
	writeU16BE(vmhd + 14, 0);                       // opcolor[0]
	writeU16BE(vmhd + 16, 0);                       // opcolor[1]
	writeU16BE(vmhd + 18, 0);                       // opcolor[2]

	minfData.resize(minfData.size() + kVmhdSize);
	oa::memcpy(minfData.data() + minfData.size() - kVmhdSize, vmhd, kVmhdSize);
	
	// Write stbl box with sample table boxes
	oa::Vector<oa::U8> stblData;
	
	// Write stsd sample description with avcC or hvcC codec configuration.
	oa::Vector<oa::U8> stsdData;
	
	// Build avcC box if SPS/PPS are available
	oa::Vector<oa::U8> avcCData;
	if (!sps_.empty() && !pps_.empty()) {
		// aVCDecoderConfigurationRecord (ISO/IEC 14496-15 §5.2.1.1):
		//   8  bytes  box header (size + type)
		//   1  byte   configurationVersion
		//   3  bytes  profile/compat/level (lifted from SPS)
		//   1  byte   lengthSizeMinusOne
		//   1  byte   numOfSPS (low 5 bits)
		//   2+sps    each SPS: length + data
		//   1  byte   numOfPPS
		//   2+pps    each PPS: length + data
		const oa::U32 avcCSize =
			8U + 1U + 3U + 1U + 1U + 2U + static_cast<oa::U32>(sps_.size())
			+ 1U + 2U + static_cast<oa::U32>(pps_.size());
		avcCData.resize(avcCSize);
		for (oa::U32 i = 0; i < avcCSize; ++i) { avcCData[i] = 0; }

		writeBoxHeader(avcCData.data(), avcCSize, 0x61766343);  // 'avcC'
		avcCData[8]  = 1;       // configurationVersion
		// Lift profile/compat/level from the SPS itself (bytes 1..3 of the
		// SPS NAL body, which our encoder hands us without start code).
		if (sps_.size() >= 4) {
			avcCData[9]  = sps_[1];   // profile_idc
			avcCData[10] = sps_[2];   // constraint_set flags / profile_compat
			avcCData[11] = sps_[3];   // level_idc
		} else {
			avcCData[9] = 0x42; avcCData[10] = 0xE0; avcCData[11] = 0x1E;
		}
		avcCData[12] = 0xFF;    // lengthSizeMinusOne = 3 + reserved bits
		avcCData[13] = 0xE1;    // numOfSPS = 1 + reserved bits
		writeU16BE(avcCData.data() + 14, static_cast<oa::U16>(sps_.size()));
		oa::memcpy(avcCData.data() + 16, sps_.data(), sps_.size());
		const oa::U32 ppsOffset = 16U + static_cast<oa::U32>(sps_.size());
		avcCData[ppsOffset] = 1;  // numOfPPS
		writeU16BE(avcCData.data() + ppsOffset + 1, static_cast<oa::U16>(pps_.size()));
		oa::memcpy(avcCData.data() + ppsOffset + 3, pps_.data(), pps_.size());
	}
	oa::Vector<oa::U8> codecConfig = config_.codec == oa::VideoCodec::H264
		? oa::move(avcCData) : buildHvcc(vps_, sps_, pps_);
	
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
	// total: 86 bytes (8 header + 78 preamble). codec config follows.
	const oa::U32 sampleEntrySize = 86U + static_cast<oa::U32>(codecConfig.size());
	// stsd payload: 8 bytes (box header) + 4 bytes (version+flags) +
	// 4 bytes (entry_count) + sampleEntrySize.
	const oa::U32 stsdSize = 8 + 4 + 4 + sampleEntrySize;

	stsdData.resize(stsdSize);
	for (oa::U32 i = 0; i < stsdSize; ++i) { stsdData[i] = 0; }
	writeBoxHeader(stsdData.data(), stsdSize, 0x73747364);  // 'stsd'
	writeU32BE(stsdData.data() + 8, 0);                     // version + flags
	writeU32BE(stsdData.data() + 12, 1);                    // entry_count

	oa::U8* sampleEntry = stsdData.data() + 16;
	writeU32BE(sampleEntry + 0, sampleEntrySize);
	writeU32BE(sampleEntry + 4,
		config_.codec == oa::VideoCodec::H264 ? 0x61766331U : 0x68766331U); // avc1/hvc1
	// avc1 + 8..13 reserved zero (already memset)
	writeU16BE(sampleEntry + 14, 1);             // data_reference_index
	// avc1 + 16..17 pre_defined (u16 = 0)
	// avc1 + 18..19 reserved (u16 = 0)
	// avc1 + 20..31 pre_defined[3] (u32 × 3 = 0)
	writeU16BE(sampleEntry + 32, static_cast<oa::U16>(config_.width));
	writeU16BE(sampleEntry + 34, static_cast<oa::U16>(config_.height));
	writeU32BE(sampleEntry + 36, 0x00480000U);   // horizresolution 72 DPI
	writeU32BE(sampleEntry + 40, 0x00480000U);   // vertresolution 72 DPI
	// avc1 + 44..47 reserved (u32 = 0)
	writeU16BE(sampleEntry + 48, 1);             // frame_count
	// avc1 + 50..81 compressorname (32 bytes, zero-filled by memset above)
	writeU16BE(sampleEntry + 82, 0x0018U);       // depth = 24
	writeU16BE(sampleEntry + 84, 0xFFFFU);       // pre_defined

	// append codec config inside the visual sample entry.
	if (!codecConfig.empty()) {
		oa::memcpy(sampleEntry + 86, codecConfig.data(), codecConfig.size());
	}
	
	stblData.resize(stblData.size() + stsdSize);
	oa::memcpy(stblData.data() + stblData.size() - stsdSize, stsdData.data(), stsdSize);
	
	// stts: run-length encode actual capture timestamp deltas. This supports
	// variable-rate PipeWire input while remaining compact for fixed-rate video.
	{
		struct SttsRun { oa::U32 count; oa::U32 delta; };
		oa::Vector<SttsRun> runs;
		for (oa::U32 delta : sampleDeltas) {
			if (runs.empty() or runs.back().delta != delta) runs.pushBack({1U, delta});
			else ++runs.back().count;
		}
		const oa::U32 sttsSize = 16U + static_cast<oa::U32>(runs.size()) * 8U;
		oa::Vector<oa::U8> stts;
		stts.resize(sttsSize);
		oa::memset(stts.data(), 0, stts.size());
		writeBoxHeader(stts.data(), sttsSize, 0x73747473);
		writeU32BE(stts.data() + 8, 0);
		writeU32BE(stts.data() + 12, static_cast<oa::U32>(runs.size()));
		for (oa::U32 i = 0; i < runs.size(); ++i) {
			writeU32BE(stts.data() + 16U + i * 8U, runs[i].count);
			writeU32BE(stts.data() + 20U + i * 8U, runs[i].delta);
		}
		stblData.resize(stblData.size() + stts.size());
		oa::memcpy(stblData.data() + stblData.size() - stts.size(), stts.data(), stts.size());
	}

	// stsc (SampleToChunkBox §8.7.4): 8 header + 4 vf + 4 entry_count +
	//                                 12*entry (first_chunk, samples_per_chunk, sdix).
	{
		constexpr oa::U32 kStscSize = 28;
		oa::U8 stsc[kStscSize] = {};
		writeBoxHeader(stsc, kStscSize, 0x73747363);  // 'stsc'
		writeU32BE(stsc + 8,  0);                     // version + flags
		writeU32BE(stsc + 12, 1);                     // entry_count
		writeU32BE(stsc + 16, 1);                     // first_chunk
		writeU32BE(stsc + 20, 1);                     // samples_per_chunk
		writeU32BE(stsc + 24, 1);                     // sample_description_index
		stblData.resize(stblData.size() + kStscSize);
		oa::memcpy(stblData.data() + stblData.size() - kStscSize, stsc, kStscSize);
	}

	// stsz (SampleSizeBox §8.7.3.2): 8 header + 4 vf + 4 sample_size
	//                                + 4 sample_count + 4*N entry sizes.
	{
		const oa::U32 stszSize = 8 + 4 + 4 + 4 + 4 * packetCount_;
		oa::Vector<oa::U8> stsz;
		stsz.resize(stszSize);
		for (oa::U32 i = 0; i < stszSize; ++i) { stsz[i] = 0; }
		writeBoxHeader(stsz.data(), stszSize, 0x7374737a);  // 'stsz'
		writeU32BE(stsz.data() + 8,  0);              // version + flags
		writeU32BE(stsz.data() + 12, 0);              // sample_size (0 = variable)
		writeU32BE(stsz.data() + 16, packetCount_);   // sample_count
		for (oa::U32 i = 0; i < packetCount_; ++i) {
			writeU32BE(stsz.data() + 20 + i * 4, packetSizes_[i]);
		}
		stblData.resize(stblData.size() + stszSize);
		oa::memcpy(stblData.data() + stblData.size() - stszSize, stsz.data(), stszSize);
	}

	// stco/co64: use 64-bit chunk offsets automatically once a streamed file
	// crosses the 4 GiB boundary.
	{
		const bool largeOffsets = oa::anyOf(packetOffsets_.begin(), packetOffsets_.end(),
			[](oa::U64 inOffset) { return inOffset > 0xFFFFFFFFULL; });
		const oa::U32 stride = largeOffsets ? 8U : 4U;
		const oa::U32 offsetSize = 16U + stride * packetCount_;
		oa::Vector<oa::U8> offsets(offsetSize, 0U);
		writeBoxHeader(offsets.data(), offsetSize,
			largeOffsets ? 0x636f3634U : 0x7374636fU); // co64/stco
		writeU32BE(offsets.data() + 12, packetCount_);
		for (oa::U32 i = 0; i < packetCount_; ++i) {
			if (largeOffsets) writeU64BE(offsets.data() + 16U + i * 8U, packetOffsets_[i]);
			else writeU32BE(offsets.data() + 16U + i * 4U,
				static_cast<oa::U32>(packetOffsets_[i]));
		}
		appendBytes(stblData, offsets.data(), offsets.size());
	}

	// stss (SyncSampleBox §8.6.2.2): 8 header + 4 vf + 4 entry_count + 4*N indices.
	oa::U32 keyframeCount = 0;
	for (oa::U32 i = 0; i < packetCount_; ++i) {
		if (packetKeyframe_[i]) {
			++keyframeCount;
		}
	}

	if (keyframeCount > 0) {
		const oa::U32 stssSize = 8 + 4 + 4 + 4 * keyframeCount;
		oa::Vector<oa::U8> stss;
		stss.resize(stssSize);
		for (oa::U32 i = 0; i < stssSize; ++i) { stss[i] = 0; }
		writeBoxHeader(stss.data(), stssSize, 0x73747373);  // 'stss'
		writeU32BE(stss.data() + 8,  0);
		writeU32BE(stss.data() + 12, keyframeCount);
		
		oa::U32 idx = 0;
		for (oa::U32 i = 0; i < packetCount_; ++i) {
			if (packetKeyframe_[i]) {
				writeU32BE(stss.data() + 16 + idx * 4, i + 1);  // 1-based
				++idx;
			}
		}
		
		stblData.resize(stblData.size() + stssSize);
		oa::memcpy(stblData.data() + stblData.size() - stssSize, stss.data(), stssSize);
	}
	
	// Write stbl box header
	oa::U32 stblSize = static_cast<oa::U32>(stblData.size() + 8);
	oa::U8 stblHeader[8];
	writeBoxHeader(stblHeader, stblSize, 0x7374626c);  // 'stbl'
	
	oa::Vector<oa::U8> finalStbl;
	finalStbl.resize(stblData.size() + 8);
	oa::memcpy(finalStbl.data(), stblHeader, 8);
	oa::memcpy(finalStbl.data() + 8, stblData.data(), stblData.size());
	
	minfData.resize(minfData.size() + finalStbl.size());
	oa::memcpy(minfData.data() + minfData.size() - finalStbl.size(), finalStbl.data(), finalStbl.size());
	
	// Write minf box header
	oa::U32 minfSize = static_cast<oa::U32>(minfData.size() + 8);
	oa::U8 minfHeader[8];
	writeBoxHeader(minfHeader, minfSize, 0x6d696e66);  // 'minf'
	
	oa::Vector<oa::U8> finalMinf;
	finalMinf.resize(minfData.size() + 8);
	oa::memcpy(finalMinf.data(), minfHeader, 8);
	oa::memcpy(finalMinf.data() + 8, minfData.data(), minfData.size());
	
	mdiaData.resize(mdiaData.size() + finalMinf.size());
	oa::memcpy(mdiaData.data() + mdiaData.size() - finalMinf.size(), finalMinf.data(), finalMinf.size());
	
	// Write mdia box header
	oa::U32 mdiaSize = static_cast<oa::U32>(mdiaData.size() + 8);
	oa::U8 mdiaHeader[8];
	writeBoxHeader(mdiaHeader, mdiaSize, 0x6d646961);  // 'mdia'
	
	oa::Vector<oa::U8> finalMdia;
	finalMdia.resize(mdiaData.size() + 8);
	oa::memcpy(finalMdia.data(), mdiaHeader, 8);
	oa::memcpy(finalMdia.data() + 8, mdiaData.data(), mdiaData.size());

	// append mdia into the trak payload (after tkhd) and wrap with a trak
	// box header before appending to moov. ParseTrakBox in the demuxer
	// recurses moov → trak → mdia → minf → stbl, so the trak wrap is
	// essential — without it width / height / sample table stay empty.
	trakData.resize(trakData.size() + finalMdia.size());
	oa::memcpy(trakData.data() + trakData.size() - finalMdia.size(),
		finalMdia.data(), finalMdia.size());

	oa::U32 trakSize = static_cast<oa::U32>(trakData.size() + 8);
	oa::U8 trakHeader[8];
	writeBoxHeader(trakHeader, trakSize, 0x7472616bU);  // 'trak'
	oa::Vector<oa::U8> finalTrak;
	finalTrak.resize(trakData.size() + 8);
	oa::memcpy(finalTrak.data(), trakHeader, 8);
	oa::memcpy(finalTrak.data() + 8, trakData.data(), trakData.size());

	moovData.resize(moovData.size() + finalTrak.size());
	oa::memcpy(moovData.data() + moovData.size() - finalTrak.size(),
		finalTrak.data(), finalTrak.size());

	if (!audioPacketSizes_.empty()) {
		auto audioTrak = buildAudioTrack(config_, audioPacketOffsets_,
			audioPacketDurations_, timescale);
		appendBytes(moovData, audioTrak.data(), audioTrak.size());
	}
	
	// Write moov box header with total size
	oa::U32 moovSize = static_cast<oa::U32>(moovData.size() + 8);
	oa::U8 moovHeader[8];
	writeBoxHeader(moovHeader, moovSize, 0x6d6f6f76);  // 'moov'
	
	// Prepend moov header to moov data
	oa::Vector<oa::U8> finalMoov;
	finalMoov.resize(moovData.size() + 8);
	oa::memcpy(finalMoov.data(), moovHeader, 8);
	oa::memcpy(finalMoov.data() + 8, moovData.data(), moovData.size());
	
	// Prepend moov to mdat data
	oa::Vector<oa::U8> finalData;
	finalData.resize(mdatData_.size() + finalMoov.size());
	oa::memcpy(finalData.data(), finalMoov.data(), finalMoov.size());
	if (not mdatData_.empty()) {
		oa::memcpy(finalData.data() + finalMoov.size(),
			mdatData_.data(), mdatData_.size());
	}
	mdatData_ = oa::move(finalData);
}
