// oa::VideoDemuxer — incremental compressed-video packet source.
//
// Supports:
//   - Native bounded-memory MP4 sample-table path
//   - OA-native fragmented MP4, Matroska/WebM and MPEG-TS demuxers
//   - H.264, H.265, AV1 and VP9 elementary packets for vulkan Video
//
// Incremental container demux and vulkan Video packet delivery.

#pragma once

#include <oa/core/status.h>
#include <oa/core/std/uniquePtr.h>
#include <oa/core/types.h>
#include <oa/vision/videoDecoder.h>

#include <stdio.h>

namespace oa {

// Container kind enumeration
enum class VideoContainerKind : oa::U32 {
  Unknown = 0,
  Mp4 = 1, // ISO base Media file format
  WebM = 2,
  MpegTs = 3, // MPEG-2 Transport stream
  Matroska = 4,
};

// Video packet extracted from container
struct VideoPacket {
  oa::Vector<oa::U8> data;              // Raw NAL bytes (Annex-B format)
  oa::U64 presentationTimestamp = 0; // PTS in stream timebase
  oa::U64 decodeTimestamp = 0;       // DTS in stream timebase
  bool isKeyframe = false;
  oa::U32 trackIndex = 0; // Which track this packet belongs to
};

// Container information (probed from file header)
struct VideoContainerInfo {
  VideoContainerKind kind = VideoContainerKind::Unknown;
  VideoCodec codec = VideoCodec::H264; // Default to H.264
  oa::U32 width = 0;
  oa::U32 height = 0;
  oa::U32 frameRate = 0;   // Approximate FPS from track
  oa::U64 duration = 0;    // Duration in stream timebase units
  oa::U64 timebaseNum = 0; // Timebase numerator
  oa::U64 timebaseDen = 1; // Timebase denominator
  oa::U32 trackCount = 0;
};

// Network/live-source policy. local files ignore reconnect and jitter fields.
// Live transports use the same bounded packet/reconnect policy as file-backed
// demuxers. Protocol support is capability-gated by OA rather than inherited
// from a process-global third-party media registry.
struct VideoDemuxerConfig {
  bool reconnect = true;
  oa::U32 maxReconnectAttempts = 8;
  oa::U32 reconnectBackoffMs = 100;
  oa::U32 readTimeoutMs = 3000;
  oa::U32 jitterBufferMs = 250;
  oa::U32 reorderQueuePackets = 64;
  oa::U32 maxTimestampDiscontinuityMs = 2000;
  oa::String rtspTransport = "tcp";
};

struct VideoDemuxerStats {
  oa::U64 reconnectCount = 0U;
  oa::U64 timestampDiscontinuities = 0U;
  oa::U64 formatGeneration = 1U;
};

// MP4 box types (32-bit FourCC)
namespace VideoMp4Box {
constexpr oa::U32 Ftyp = 0x66747970; // 'ftyp'
constexpr oa::U32 Moov = 0x6d6f6f76; // 'moov'
constexpr oa::U32 Mdat = 0x6d646174; // 'mdat'
constexpr oa::U32 Moof = 0x6d6f6f66; // 'moof' (fragmented)
constexpr oa::U32 Mvex = 0x6d766578; // 'mvex' (movie extends)
constexpr oa::U32 Trex = 0x74726578; // 'trex' (track extends)
constexpr oa::U32 Free = 0x66726565; // 'free'
constexpr oa::U32 Skip = 0x736b6970; // 'skip'
constexpr oa::U32 Wide = 0x77696465; // 'wide'
constexpr oa::U32 Mvhd = 0x6d766864; // 'mvhd' (movie header)
constexpr oa::U32 Trak = 0x7472616b; // 'trak' (track)
constexpr oa::U32 Tkhd = 0x746b6864; // 'tkhd' (track header)
constexpr oa::U32 Mdia = 0x6d646961; // 'mdia' (media)
constexpr oa::U32 Mdhd = 0x6d646864; // 'mdhd' (media header)
constexpr oa::U32 Hdlr = 0x68646c72; // 'hdlr' (handler)
constexpr oa::U32 Minf = 0x6d696e66; // 'minf' (media info)
constexpr oa::U32 Stbl = 0x7374626c; // 'stbl' (sample table)
constexpr oa::U32 Stsd = 0x73747364; // 'stsd' (sample description)
constexpr oa::U32 Stts = 0x73747473; // 'stts' (time-to-sample)
constexpr oa::U32 Stsc = 0x73747363; // 'stsc' (sample-to-chunk)
constexpr oa::U32 Stsz = 0x7374737a; // 'stsz' (sample size)
constexpr oa::U32 Stco = 0x7374636f; // 'stco' (chunk offset)
constexpr oa::U32 Co64 = 0x636f3634; // 'co64' (64-bit chunk offset)
constexpr oa::U32 Stss = 0x73747373; // 'stss' (sync sample / keyframe)
constexpr oa::U32 Ctts = 0x63747473; // 'ctts' (composition time offset)
constexpr oa::U32 Traf = 0x74726166; // 'traf' (track fragment)
constexpr oa::U32 Tfhd = 0x74666864; // 'tfhd' (track fragment header)
constexpr oa::U32 Trun = 0x7472756e; // 'trun' (track fragment run)
constexpr oa::U32 Tfdt = 0x74666474; // 'tfdt' (track fragment decode time)
constexpr oa::U32 Avcc = 0x61766343; // 'avcC' (H.264 decoder config)
constexpr oa::U32 Hvcc = 0x68766343; // 'hvcC' (H.265 decoder config)
constexpr oa::U32 Av01 = 0x61763031; // 'av01' (AV1 codec config)
} // namespace VideoMp4Box

// Video stream demuxer
class VideoDemuxer {
public:
  struct MediaImpl;
  // open either a local path or a URL through one canonical source boundary.
  static oa::Result<VideoDemuxer> open(oa::StringView inUri);
  static oa::Result<VideoDemuxer> open(oa::StringView inUri,
                                       const VideoDemuxerConfig &inConfig);

  VideoDemuxer();
  VideoDemuxer(VideoDemuxer &&) noexcept;
  VideoDemuxer &operator=(VideoDemuxer &&) noexcept;
  VideoDemuxer(const VideoDemuxer &) = delete;
  ~VideoDemuxer();

  // Explicit resource-release boundary. Reports local file close failures;
  // pending packets are discarded and no container data is manufactured.
  [[nodiscard]] oa::Status close();

  // probe container info without opening full file
  static oa::Result<VideoContainerInfo> probe(const char *inPath);

  // Read next packet from stream
  oa::Status readNextPacket(VideoPacket &outPacket);

  // seek to a specific timestamp (in stream timebase)
  oa::Status seek(oa::U64 inTimestamp);

  // get container info
  [[nodiscard]] const VideoContainerInfo &getInfo() const noexcept {
    return info_;
  }

  // Check if end of stream
  [[nodiscard]] bool isEos() const noexcept { return eos_; }
  [[nodiscard]] bool isLive() const noexcept;
  [[nodiscard]] bool isSeekable() const noexcept;
  [[nodiscard]] const VideoDemuxerStats &getStats() const noexcept {
    return stats_;
  }
  [[nodiscard]] oa::U64 formatGeneration() const noexcept {
    return stats_.formatGeneration;
  }

  // Return one indexed sample's presentation timestamp without converting a
  // signed composition offset to unsigned first. Malformed tables whose
  // offset underflows DTS or whose positive offset overflows fail closed.
  [[nodiscard]] oa::Result<oa::U64>
  samplePresentationTimestamp(oa::Usize inSampleIndex) const;

  // get video profile for decoder creation
  [[nodiscard]] VideoProfile getVideoProfile() const;

  // MP4 sample entry (used by parser)
  struct Sample {
    oa::U64 offset = 0;
    oa::U32 size = 0;
    oa::U64 duration = 0;
    oa::U64 dts = 0;
    oa::I32 ctsOffset = 0;
    bool isKeyframe = false;
  };

  // Sample table (populated by MP4 parser)
  oa::Vector<Sample> samples_;

  // Container metadata (codec/width/height/duration). Public so the free
  // helpers in videoDemuxer.cpp can write into it during box parsing.
  VideoContainerInfo info_ = {};

  // codec config parsed from avcC (H.264): SPS + PPS as Annex-B bytes,
  // ready to prepend to the first IDR packet. Also exposes the NAL length
  // field width so readNextPacket can rewrite length-prefix → start codes.
  struct AvcConfig {
    oa::Vector<oa::U8> spsAnnexB; // 00 00 00 01 + SPS NAL
    oa::Vector<oa::U8> ppsAnnexB; // 00 00 00 01 + PPS NAL
    VideoProfile profile = {};
    oa::U8 lengthSize = 4; // bytes per NAL length field (1, 2 or 4)
    bool valid = false;
  };
  AvcConfig avc_;

  // codec config parsed from hvcC (H.265): VPS + SPS + PPS as Annex-B bytes
  struct HvcConfig {
    oa::Vector<oa::U8> vpsAnnexB; // 00 00 00 01 + VPS NAL
    oa::Vector<oa::U8> spsAnnexB; // 00 00 00 01 + SPS NAL
    oa::Vector<oa::U8> ppsAnnexB; // 00 00 00 01 + PPS NAL
    VideoProfile profile = {};
    oa::U8 lengthSize = 4; // bytes per NAL length field (1, 2 or 4)
    bool valid = false;
  };
  HvcConfig hvc_;

  // codec config parsed from av1C (AV1 ISO-BMFF §2.3): the configOBUs blob,
  // which carries the sequence-header OBU out-of-band. AV1 MP4 samples are
  // OBU temporal units that omit the sequence header, so we prepend this to
  // every keyframe temporal unit (analogous to SPS/PPS for H.264/H.265).
  struct Av1Config {
    oa::Vector<oa::U8> configObus; // raw low-overhead OBU bytes (sequence header)
    VideoProfile profile = {};
    bool valid = false;
  };
  Av1Config av1_;

  struct Vp9Config {
    VideoProfile profile = {};
    bool valid = false;
  };
  Vp9Config vp9_;

  // ISO-BMFF fragment defaults. `moov/mvex/trex` establishes this state;
  // later `moof/traf/trun` boxes use it without retaining media payloads.
  struct FragmentConfig {
    oa::U32 trackId = 0U;
    oa::U32 defaultSampleDuration = 0U;
    oa::U32 defaultSampleSize = 0U;
    oa::U32 defaultSampleFlags = 0U;
  };
  FragmentConfig fragment_;

  // Returns SPS+PPS concatenated as Annex-B (empty if not parsed). Caller
  // prepends to the first IDR packet to bootstrap the decoder.
  [[nodiscard]] const AvcConfig &getAvcConfig() const noexcept { return avc_; }
  [[nodiscard]] const HvcConfig &getHvcConfig() const noexcept { return hvc_; }
  [[nodiscard]] const Av1Config &getAv1Config() const noexcept { return av1_; }
  [[nodiscard]] const Vp9Config &getVp9Config() const noexcept { return vp9_; }

  // index of the next sample readNextPacket() will read (0-based). After
  // decoding sample N, this equals N+1. Used by Video for step-back.
  [[nodiscard]] oa::U32 getCurrentSampleIndex() const noexcept {
    return currentSampleIndex_;
  }

private:
  void reset_() noexcept;
  static oa::Result<VideoDemuxer> openLocal_(oa::StringView inPath);
  static oa::Result<VideoDemuxer>
  openMedia_(oa::StringView inUri, const VideoDemuxerConfig &inConfig = {});
  oa::Status readMediaPacket_(VideoPacket &outPacket);
  oa::Status seekMedia_(oa::U64 inTimestamp);
  oa::Status reconnectMedia_();

  oa::UniquePtr<MediaImpl> media_;
  oa::String uri_;
  VideoDemuxerConfig config_ = {};
  VideoDemuxerStats stats_ = {};
  oa::U64 lastDecodeTimestamp_ = 0U;
  bool hasLastDecodeTimestamp_ = false;
  FILE* file_ = nullptr;
  oa::U64 fileSize_ = 0U;
  oa::Vector<oa::U8>
      sampleData_; // bounded scratch for the current compressed sample
  oa::U64 currentOffset_ = 0;
  bool eos_ = false;
  oa::U32 currentSampleIndex_ = 0;
  bool needParameterSets_ = true; // prepend SPS+PPS on next IDR
  oa::Vector<oa::U8>
      bufferedPictureNals_;       // Picture NAL units buffered for next packet
  oa::U64 bufferedTimestamp_ = 0; // timestamp for buffered picture NAL units
  oa::U64 bufferedDecodeTimestamp_ = 0;
  bool bufferedIsKeyframe_ =
      false; // Keyframe flag for buffered picture NAL units
};

} // namespace oa
