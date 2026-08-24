// oa::VideoMuxer — Video stream muxer (MP4)
// Muxes encoded video packets into MP4 containers.
//
// Supports streaming MP4 with H.264/H.265 video and optional native PCM audio.
// Media payload is written incrementally; sample tables are finalized in moov.

#pragma once

#include <oa/audio/audioEncoder.h>
#include <oa/core/status.h>
#include <oa/core/types.h>
#include <oa/vision/videoEncoder.h>

#include <cstdio>

namespace oa {

struct VideoMuxerConfig {
  oa::String outputPath;
  VideoCodec codec = VideoCodec::H264;
  oa::U32 width = 0;
  oa::U32 height = 0;
  oa::U32 frameRate = 30;
  oa::U64 timebaseNum = 1;
  oa::U64 timebaseDen = 90'000;
  bool audioEnabled = false;
  AudioCodec audioCodec = AudioCodec::PcmS16;
  oa::U32 audioSampleRate = 48'000U;
  oa::U32 audioChannelCount = 2U;
  oa::U32 audioPrimingFrames = 0U;
};

class VideoMuxer {
public:
  static oa::Result<VideoMuxer> create(const VideoMuxerConfig &inConfig);

  VideoMuxer() = default;
  VideoMuxer(VideoMuxer &&) noexcept;
  VideoMuxer &operator=(VideoMuxer &&) noexcept;
  VideoMuxer(const VideoMuxer &) = delete;
  ~VideoMuxer();

  // Closes the output without writing a movie trailer. call finalize() first
  // when a playable container is required.
  [[nodiscard]] oa::Status close();

  // Write an encoded packet to the muxer.
  oa::Status writePacket(const EncodedVideoPacket &inFrame);
  oa::Status writeAudioPacket(const EncodedAudioPacket &inPacket);
  void setAudioCodecConfig(oa::Span<const oa::U8> inCodecConfig);

  // set AVC SPS/PPS data for the avcC decoder configuration box.
  void setCodecConfig(const oa::Vec<oa::U8> &inSps,
                      const oa::Vec<oa::U8> &inPps);
  // set HEVC VPS/SPS/PPS data for the hvcC decoder configuration box.
  void setCodecConfig(const oa::Vec<oa::U8> &inVps,
                      const oa::Vec<oa::U8> &inSps,
                      const oa::Vec<oa::U8> &inPps);

  // finalize the MP4 file (write moov box and flush).
  oa::Status finalize();

  [[nodiscard]] const VideoMuxerConfig &getConfig() const noexcept {
    return config_;
  }
  [[nodiscard]] oa::U32 getPacketCount() const noexcept { return packetCount_; }

private:
  void reset_() noexcept;
  oa::Status writeFtypBox();
  void writeMoovBox();

  VideoMuxerConfig config_ = {};
  oa::Vec<oa::U8> mdatData_;
  std::FILE *outputFile_ = nullptr;
  oa::U64 mdatPayloadBytes_ = 0U;
  oa::Vec<oa::U64> packetOffsets_;
  oa::Vec<oa::U32> packetSizes_;
  oa::Vec<oa::U64> packetDts_;
  oa::Vec<bool> packetKeyframe_;
  oa::Vec<oa::U64> audioPacketOffsets_;
  oa::Vec<oa::U32> audioPacketSizes_;
  oa::Vec<oa::U32> audioPacketDurations_;
  oa::Vec<oa::U8> audioCodecConfig_;
  oa::U32 packetCount_ = 0;
  bool finalized_ = false;

  oa::Vec<oa::U8> vps_;
  oa::Vec<oa::U8> sps_;
  oa::Vec<oa::U8> pps_;
};

} // namespace oa
