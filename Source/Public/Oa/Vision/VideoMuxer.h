// OA Vision — Video Stream Muxer (MP4)
// Muxes encoded video packets into MP4 containers.
//
// Supports streaming MP4 with H.264/H.265 video and optional native PCM audio.
// Media payload is written incrementally; sample tables are finalized in moov.

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Vision/VideoEncoder.h>
#include <Oa/Audio/AudioEncoder.h>

#include <cstdio>


// Video muxer - writes encoded packets to MP4 container
class OaVideoMuxer
{
public:
	struct CreateInfo
	{
		OaVideoCodec Codec = OaVideoCodec::H264;
		OaU32 Width = 0;
		OaU32 Height = 0;
		OaU32 FrameRate = 30;  // FPS
		OaU64 TimebaseNum = 1;
		OaU64 TimebaseDen = 90000;  // Common video timebase
		bool AudioEnabled = false;
		OaAudioCodec AudioCodec = OaAudioCodec::PcmS16;
		OaU32 AudioSampleRate = 48'000U;
		OaU32 AudioChannelCount = 2U;
		OaU32 AudioPrimingFrames = 0U;
	};
	
	static OaResult<OaVideoMuxer> Create(const char* InPath, const CreateInfo& InInfo);
	
	OaVideoMuxer() = default;
	OaVideoMuxer(OaVideoMuxer&&) noexcept;
	OaVideoMuxer& operator=(OaVideoMuxer&&) noexcept;
	OaVideoMuxer(const OaVideoMuxer&) = delete;
	~OaVideoMuxer();
	
	void Destroy();
	
	// Write an encoded packet to the muxer
	OaStatus WritePacket(const OaEncodedFrame& InFrame);
	OaStatus WriteAudioPacket(const OaEncodedAudioPacket& InPacket);
	void SetAudioCodecConfig(OaSpan<const OaU8> InCodecConfig);
	
	// Set AVC SPS/PPS data for the avcC decoder configuration box.
	void SetCodecConfig(const OaVec<OaU8>& InSps, const OaVec<OaU8>& InPps);
	// Set HEVC VPS/SPS/PPS data for the hvcC decoder configuration box.
	void SetCodecConfig(
		const OaVec<OaU8>& InVps,
		const OaVec<OaU8>& InSps,
		const OaVec<OaU8>& InPps);
	
	// Finalize the MP4 file (write moov box and flush)
	OaStatus Finalize();
	
	// Get muxer info
	[[nodiscard]] const CreateInfo& GetInfo() const noexcept { return Info_; }
	
	// Get packet count
	[[nodiscard]] OaU32 GetPacketCount() const noexcept { return PacketCount_; }

private:
	void Reset_() noexcept;

	// Write ftyp box
	OaStatus WriteFtypBox();
	
	// Build moov (trak/mdia/minf/stbl) into the small metadata scratch buffer.
	void WriteMoovBox();
	
	CreateInfo Info_ = {};
	OaVec<OaU8> MdatData_;  // temporary MP4 box construction scratch
	std::FILE* OutputFile_ = nullptr;
	OaU64 MdatPayloadBytes_ = 0U;
	OaVec<OaU64> PacketOffsets_;  // absolute file offset of each packet
	OaVec<OaU32> PacketSizes_;    // Size of each packet
	OaVec<OaU64> PacketDts_;      // DTS of each packet
	OaVec<bool> PacketKeyframe_; // Keyframe flag for each packet
	OaVec<OaU64> AudioPacketOffsets_;
	OaVec<OaU32> AudioPacketSizes_;
	OaVec<OaU32> AudioPacketDurations_;
	OaVec<OaU8> AudioCodecConfig_;
	OaU32 PacketCount_ = 0;
	bool Finalized_ = false;
	
	// Codec configuration for avcC / hvcC boxes.
	OaVec<OaU8> Vps_;
	OaVec<OaU8> Sps_;
	OaVec<OaU8> Pps_;
};
