// OA Vision — incremental compressed-video packet source.
//
// Supports:
//   - Native bounded-memory MP4 sample-table path
//   - OA-native fragmented MP4, Matroska/WebM and MPEG-TS demuxers
//   - H.264, H.265, AV1 and VP9 elementary packets for Vulkan Video
//
// Incremental container demux and Vulkan Video packet delivery.

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Std/UniquePtr.h>
#include <Oa/Vision/VideoDecoder.h>

#include <cstdio>


// Container kind enumeration
enum class OaContainerKind : OaU32 {
	Unknown = 0,
	Mp4     = 1,  // ISO Base Media File Format
	WebM    = 2,
	MpegTs  = 3,  // MPEG-2 Transport Stream
	Matroska = 4,
};


// Video packet extracted from container
struct OaVideoPacket {
	OaVec<OaU8> Data;           // Raw NAL bytes (Annex-B format)
	OaU64 PresentationTimestamp = 0;  // PTS in stream timebase
	OaU64 DecodeTimestamp      = 0;  // DTS in stream timebase
	bool IsKeyframe            = false;
	OaU32 TrackIndex           = 0;    // Which track this packet belongs to
};


// Container information (probed from file header)
struct OaContainerInfo {
	OaContainerKind Kind = OaContainerKind::Unknown;
	OaVideoCodec Codec = OaVideoCodec::H264;  // Default to H.264
	OaU32 Width = 0;
	OaU32 Height = 0;
	OaU32 FrameRate = 0;  // Approximate FPS from track
	OaU64 Duration = 0;    // Duration in stream timebase units
	OaU64 TimebaseNum = 0; // Timebase numerator
	OaU64 TimebaseDen = 1; // Timebase denominator
	OaU32 TrackCount = 0;
};

// Network/live-source policy. Local files ignore reconnect and jitter fields.
// Live transports use the same bounded packet/reconnect policy as file-backed
// demuxers. Protocol support is capability-gated by OA rather than inherited
// from a process-global third-party media registry.
struct OaVideoStreamOptions {
	bool Reconnect = true;
	OaU32 MaxReconnectAttempts = 8;
	OaU32 ReconnectBackoffMs = 100;
	OaU32 ReadTimeoutMs = 3000;
	OaU32 JitterBufferMs = 250;
	OaU32 ReorderQueuePackets = 64;
	OaU32 MaxTimestampDiscontinuityMs = 2000;
	OaString RtspTransport = "tcp";
};

struct OaVideoStreamStats {
	OaU64 ReconnectCount = 0U;
	OaU64 TimestampDiscontinuities = 0U;
	OaU64 FormatGeneration = 1U;
};


// MP4 box types (32-bit FourCC)
namespace OaMp4Box {
	constexpr OaU32 Ftyp = 0x66747970;  // 'ftyp'
	constexpr OaU32 Moov = 0x6d6f6f76;  // 'moov'
	constexpr OaU32 Mdat = 0x6d646174;  // 'mdat'
	constexpr OaU32 Moof = 0x6d6f6f66;  // 'moof' (fragmented)
	constexpr OaU32 Mvex = 0x6d766578;  // 'mvex' (movie extends)
	constexpr OaU32 Trex = 0x74726578;  // 'trex' (track extends)
	constexpr OaU32 Free = 0x66726565;  // 'free'
	constexpr OaU32 Skip = 0x736b6970;  // 'skip'
	constexpr OaU32 Wide = 0x77696465;  // 'wide'
	constexpr OaU32 Mvhd = 0x6d766864;  // 'mvhd' (movie header)
	constexpr OaU32 Trak = 0x7472616b;  // 'trak' (track)
	constexpr OaU32 Tkhd = 0x746b6864;  // 'tkhd' (track header)
	constexpr OaU32 Mdia = 0x6d646961;  // 'mdia' (media)
	constexpr OaU32 Mdhd = 0x6d646864;  // 'mdhd' (media header)
	constexpr OaU32 Hdlr = 0x68646c72;  // 'hdlr' (handler)
	constexpr OaU32 Minf = 0x6d696e66;  // 'minf' (media info)
	constexpr OaU32 Stbl = 0x7374626c;  // 'stbl' (sample table)
	constexpr OaU32 Stsd = 0x73747364;  // 'stsd' (sample description)
	constexpr OaU32 Stts = 0x73747473;  // 'stts' (time-to-sample)
	constexpr OaU32 Stsc = 0x73747363;  // 'stsc' (sample-to-chunk)
	constexpr OaU32 Stsz = 0x7374737a;  // 'stsz' (sample size)
	constexpr OaU32 Stco = 0x7374636f;  // 'stco' (chunk offset)
	constexpr OaU32 Co64 = 0x636f3634;  // 'co64' (64-bit chunk offset)
	constexpr OaU32 Stss = 0x73747373;  // 'stss' (sync sample / keyframe)
	constexpr OaU32 Ctts = 0x63747473;  // 'ctts' (composition time offset)
	constexpr OaU32 Traf = 0x74726166;  // 'traf' (track fragment)
	constexpr OaU32 Tfhd = 0x74666864;  // 'tfhd' (track fragment header)
	constexpr OaU32 Trun = 0x7472756e;  // 'trun' (track fragment run)
	constexpr OaU32 Tfdt = 0x74666474;  // 'tfdt' (track fragment decode time)
	constexpr OaU32 Avcc = 0x61766343;  // 'avcC' (H.264 decoder config)
	constexpr OaU32 Hvcc = 0x68766343;  // 'hvcC' (H.265 decoder config)
	constexpr OaU32 Av01 = 0x61763031;  // 'av01' (AV1 codec config)
}


// Video stream demuxer
class OaVideoStream {
public:
	struct MediaImpl;
	// Open either a local path or a URL. OpenFile is retained as a source-
	// compatible alias; both select the same incremental backend.
	static OaResult<OaVideoStream> Open(OaStringView InUri);
	static OaResult<OaVideoStream> Open(
		OaStringView InUri, const OaVideoStreamOptions& InOptions);
	static OaResult<OaVideoStream> OpenFile(const char* InPath);
	
	OaVideoStream() = default;
	OaVideoStream(OaVideoStream&&) noexcept;
	OaVideoStream& operator=(OaVideoStream&&) noexcept;
	OaVideoStream(const OaVideoStream&) = delete;
	~OaVideoStream();
	
	void Destroy();
	
	// Probe container info without opening full file
	static OaResult<OaContainerInfo> Probe(const char* InPath);
	
	// Read next packet from stream
	OaStatus ReadNextPacket(OaVideoPacket& OutPacket);
	
	// Seek to a specific timestamp (in stream timebase)
	OaStatus Seek(OaU64 InTimestamp);
	
	// Get container info
	[[nodiscard]] const OaContainerInfo& GetInfo() const noexcept { return Info_; }
	
	// Check if end of stream
	[[nodiscard]] bool IsEos() const noexcept { return Eos_; }
	[[nodiscard]] bool IsLive() const noexcept;
	[[nodiscard]] bool IsSeekable() const noexcept;
	[[nodiscard]] const OaVideoStreamStats& GetStats() const noexcept { return Stats_; }
	[[nodiscard]] OaU64 FormatGeneration() const noexcept { return Stats_.FormatGeneration; }
	
	// Get video profile for decoder creation
	[[nodiscard]] OaVideoProfile GetVideoProfile() const;

	// MP4 sample entry (used by parser)
	struct Sample	{
		OaU64 Offset = 0;
		OaU32 Size = 0;
		OaU64 Duration = 0;
		OaU64 Dts = 0;
		OaI32 CtsOffset = 0;
		bool IsKeyframe = false;
	};

	// Sample table (populated by MP4 parser)
	OaVec<Sample> Samples_;

	// Container metadata (codec/width/height/duration). Public so the free
	// helpers in VideoStream.cpp can write into it during box parsing.
	OaContainerInfo Info_ = {};

	// Codec config parsed from avcC (H.264): SPS + PPS as Annex-B bytes,
	// ready to prepend to the first IDR packet. Also exposes the NAL length
	// field width so ReadNextPacket can rewrite length-prefix → start codes.
	struct AvcConfig {
		OaVec<OaU8> SpsAnnexB;   // 00 00 00 01 + SPS NAL
		OaVec<OaU8> PpsAnnexB;   // 00 00 00 01 + PPS NAL
		OaVideoProfile Profile = {};
		OaU8 LengthSize = 4;     // bytes per NAL length field (1, 2 or 4)
		bool Valid = false;
	};
	AvcConfig Avc_;

	// Codec config parsed from hvcC (H.265): VPS + SPS + PPS as Annex-B bytes
	struct HvcConfig {
		OaVec<OaU8> VpsAnnexB;   // 00 00 00 01 + VPS NAL
		OaVec<OaU8> SpsAnnexB;   // 00 00 00 01 + SPS NAL
		OaVec<OaU8> PpsAnnexB;   // 00 00 00 01 + PPS NAL
		OaVideoProfile Profile = {};
		OaU8 LengthSize = 4;     // bytes per NAL length field (1, 2 or 4)
		bool Valid = false;
	};
	HvcConfig Hvc_;

	// Codec config parsed from av1C (AV1 ISO-BMFF §2.3): the configOBUs blob,
	// which carries the sequence-header OBU out-of-band. AV1 MP4 samples are
	// OBU temporal units that omit the sequence header, so we prepend this to
	// every keyframe temporal unit (analogous to SPS/PPS for H.264/H.265).
	struct Av1Config {
		OaVec<OaU8> ConfigObus;  // raw low-overhead OBU bytes (sequence header)
		OaVideoProfile Profile = {};
		bool Valid = false;
	};
	Av1Config Av1_;

	struct Vp9Config {
		OaVideoProfile Profile = {};
		bool Valid = false;
	};
	Vp9Config Vp9_;

	// ISO-BMFF fragment defaults. `moov/mvex/trex` establishes this state;
	// later `moof/traf/trun` boxes use it without retaining media payloads.
	struct FragmentConfig {
		OaU32 TrackId = 0U;
		OaU32 DefaultSampleDuration = 0U;
		OaU32 DefaultSampleSize = 0U;
		OaU32 DefaultSampleFlags = 0U;
	};
	FragmentConfig Fragment_;

	// Returns SPS+PPS concatenated as Annex-B (empty if not parsed). Caller
	// prepends to the first IDR packet to bootstrap the decoder.
	[[nodiscard]] const AvcConfig& GetAvcConfig() const noexcept { return Avc_; }
	[[nodiscard]] const HvcConfig& GetHvcConfig() const noexcept { return Hvc_; }
	[[nodiscard]] const Av1Config& GetAv1Config() const noexcept { return Av1_; }
	[[nodiscard]] const Vp9Config& GetVp9Config() const noexcept { return Vp9_; }

	// Index of the next sample ReadNextPacket() will read (0-based). After
	// decoding sample N, this equals N+1. Used by OaVideo for step-back.
	[[nodiscard]] OaU32 GetCurrentSampleIndex() const noexcept { return CurrentSampleIndex_; }

private:
	void Reset_() noexcept;
	static OaResult<OaVideoStream> OpenMedia_(
		OaStringView InUri, const OaVideoStreamOptions& InOptions = {});
	OaStatus ReadMediaPacket_(OaVideoPacket& OutPacket);
	OaStatus SeekMedia_(OaU64 InTimestamp);
	OaStatus ReconnectMedia_();

	OaUniquePtr<MediaImpl> Media_;
	OaString Uri_;
	OaVideoStreamOptions Options_ = {};
	OaVideoStreamStats Stats_ = {};
	OaU64 LastDecodeTimestamp_ = 0U;
	bool HasLastDecodeTimestamp_ = false;
	std::FILE* File_ = nullptr;
	OaU64 FileSize_ = 0U;
	OaVec<OaU8> SampleData_; // bounded scratch for the current compressed sample
	OaU64 CurrentOffset_ = 0;
	bool Eos_ = false;
	OaU32 CurrentSampleIndex_ = 0;
	bool NeedParameterSets_ = true;  // prepend SPS+PPS on next IDR
	OaVec<OaU8> BufferedPictureNals_;  // Picture NAL units buffered for next packet
	OaU64 BufferedTimestamp_ = 0;  // Timestamp for buffered picture NAL units
	bool BufferedIsKeyframe_ = false;  // Keyframe flag for buffered picture NAL units
};
