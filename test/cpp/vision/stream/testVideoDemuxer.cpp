// OA Vision — Video stream (MP4 demuxer) Tests
// Smoke-test the VideoDemuxer MP4 box parser + sample table walker + seek
// against the shibuya dataset. These tests skip when the dataset isn't on
// disk so CI without the dataset still passes.

#include "../../oaTest.h"
#include "../videoTestSupport.h"

#include <oa/core/filesystem.h>
#include <oa/runtime/engine.h>
#include <oa/vision/fnVideo.h>
#include <oa/vision/video/decoder/videoDecoderInternal.h>
#include <oa/vision/videoDecoder.h>
#include <oa/vision/videoDemuxer.h>
#include <oa/vision/videoPlayer.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

const std::string kShibuyaH264Storage = testStdString(oa::Paths::data(
    "video/shibuya_crossing_1080p30_h264.mp4").string());
const std::string kShibuyaH265Storage = testStdString(oa::Paths::data(
    "video/shibuya_crossing_1080p30_h265.mp4").string());
const std::string kShibuyaAv1Storage = testStdString(oa::Paths::data(
    "video/shibuya_crossing_1080p30_av1.mp4").string());
const std::string kShibuyaVp9Storage = testStdString(oa::Paths::data(
    "video/shibuya_crossing_1080p30_vp9.mp4").string());
const char* kShibuyaH264 = kShibuyaH264Storage.c_str();
const char* kShibuyaH265 = kShibuyaH265Storage.c_str();
const char* kShibuyaAv1 = kShibuyaAv1Storage.c_str();
const char* kShibuyaVp9 = kShibuyaVp9Storage.c_str();

bool datasetAvailable(const char *inPath) {
  auto status = oa::Filesystem::readBinary(oa::Path(inPath));
  return status.isOk();
}

TEST(VideoDemuxer, EmptyEventIsAlreadyComplete) {
  oa::Event token;
  EXPECT_FALSE(token.isValid());
  EXPECT_TRUE(token.isComplete());
  EXPECT_TRUE(token.wait().isOk());
}

} // namespace

TEST(VideoDemuxer, OpenShibuyaMp4) {
  if (not datasetAvailable(kShibuyaH264)) {
    GTEST_SKIP() << "Shibuya MP4 dataset not present at " << kShibuyaH264;
  }
  auto demuxerResult = oa::VideoDemuxer::open(kShibuyaH264);
  ASSERT_TRUE(demuxerResult.isOk())
      << "open failed: " << demuxerResult.getStatus().toString().cStr();

  auto profile = demuxerResult->getVideoProfile();
  EXPECT_EQ(profile.codec, oa::VideoCodec::H264);
  EXPECT_GT(profile.width, 0U);
  EXPECT_GT(profile.height, 0U);

  // avcC must be parsed: SPS + PPS available, length size = 4 (typical).
  const auto &avc = demuxerResult->getAvcConfig();
  EXPECT_TRUE(avc.valid);
  EXPECT_GT(avc.spsAnnexB.size(), 4U);
  EXPECT_GT(avc.ppsAnnexB.size(), 4U);
  EXPECT_GE(avc.lengthSize, 1U);
  EXPECT_LE(avc.lengthSize, 4U);
}

TEST(VideoDemuxer, MissingLocalFileReportsNotFound) {
  auto demuxerResult = oa::VideoDemuxer::open(
      "/oa-test/does-not-exist/missing-video.mp4");
  ASSERT_TRUE(demuxerResult.isError());
  EXPECT_EQ(demuxerResult.getStatus().getCode(), oa::StatusCode::NotFound);
  EXPECT_NE(demuxerResult.getStatus().getMessage().find("missing-video.mp4"),
            oa::String::Npos);
}

TEST(VideoDemuxer, ReadFirstPacketNonEmptyAv1Vp9H265) {
  struct Case {
    const char *path;
    oa::VideoCodec codec;
  };
  const Case cases[] = {
      {kShibuyaH265, oa::VideoCodec::H265},
      {kShibuyaAv1, oa::VideoCodec::AV1},
      {kShibuyaVp9, oa::VideoCodec::VP9},
  };
  for (const Case &c : cases) {
    if (not datasetAvailable(c.path)) {
      GTEST_SKIP() << "Dataset not present: " << c.path;
    }
    auto demuxerResult = oa::VideoDemuxer::open(c.path);
    ASSERT_TRUE(demuxerResult.isOk())
        << c.path << ": " << demuxerResult.getStatus().toString().cStr();
    EXPECT_EQ(demuxerResult->getVideoProfile().codec, c.codec) << c.path;

    oa::VideoPacket pkt{};
    ASSERT_TRUE(demuxerResult->readNextPacket(pkt).isOk()) << c.path;
    EXPECT_GT(pkt.data.size(), 0U)
        << c.path << " first packet must not be empty";
  }
}

TEST(VideoDemuxer, ReadFirstFivePackets) {
  if (not datasetAvailable(kShibuyaH264)) {
    GTEST_SKIP() << "Shibuya MP4 dataset not present";
  }
  auto demuxerResult = oa::VideoDemuxer::open(kShibuyaH264);
  ASSERT_TRUE(demuxerResult.isOk());

  oa::U64 lastDts = 0;
  for (int i = 0; i < 5; ++i) {
    oa::VideoPacket pkt{};
    auto status = demuxerResult->readNextPacket(pkt);
    ASSERT_TRUE(status.isOk())
        << "packet " << i << ": " << status.toString().cStr();
    ASSERT_GE(pkt.data.size(), 4U)
        << "packet " << i << " too short to be Annex-B";
    // Every emitted packet must begin with a 4-byte Annex-B start code.
    EXPECT_EQ(pkt.data[0], 0U);
    EXPECT_EQ(pkt.data[1], 0U);
    EXPECT_EQ(pkt.data[2], 0U);
    EXPECT_EQ(pkt.data[3], 1U);
    if (i == 0) {
      EXPECT_TRUE(pkt.isKeyframe) << "first packet should be a keyframe";
      // Bootstrap packet must contain SPS (nal_unit_type 7) before the
      // slice data, since the MP4 stores parameter sets out-of-band.
      bool sawSps = false;
      for (oa::Usize p = 0; p + 4 < pkt.data.size(); ++p) {
        if (pkt.data[p] == 0 && pkt.data[p + 1] == 0 && pkt.data[p + 2] == 0 &&
            pkt.data[p + 3] == 1) {
          const oa::U8 nalType = pkt.data[p + 4] & 0x1F;
          if (nalType == 7) {
            sawSps = true;
            break;
          }
        }
      }
      EXPECT_TRUE(sawSps) << "first keyframe packet must include SPS from avcC";
    } else {
      EXPECT_GE(pkt.decodeTimestamp, lastDts) << "DTS must be monotonic";
    }
    lastDts = pkt.decodeTimestamp;
  }
}

TEST(VideoDemuxer, SeekSnapsToKeyframe) {
  if (not datasetAvailable(kShibuyaH264)) {
    GTEST_SKIP() << "Shibuya MP4 dataset not present";
  }
  auto demuxerResult = oa::VideoDemuxer::open(kShibuyaH264);
  ASSERT_TRUE(demuxerResult.isOk());

  // seek back to the start — should always succeed and the next packet
  // should be the first keyframe.
  ASSERT_TRUE(demuxerResult->seek(0).isOk());
  oa::VideoPacket pkt{};
  ASSERT_TRUE(demuxerResult->readNextPacket(pkt).isOk());
  EXPECT_TRUE(pkt.isKeyframe);
}

// Exercise every bounded container backend with real packet data. These are
// remuxes (stream copy), not transcodes, so the test validates the demux and
// codec-configuration paths without spending time or changing the source
// elementary stream. FFmpeg is the fixture builder and OA remains the system
// under test.
TEST(VideoDemuxer, NativeContainerCoverage) {
#if defined(_WIN32)
  GTEST_SKIP() << "fixture remux commands are currently defined for Unix hosts";
#else
  if (not datasetAvailable(kShibuyaH264) or not datasetAvailable(kShibuyaVp9)) {
    GTEST_SKIP() << "Shibuya H.264/VP9 datasets are not present";
  }
  if (std::system("ffmpeg -version >/dev/null 2>&1") != 0) {
    GTEST_SKIP() << "ffmpeg is unavailable for container fixture creation";
  }

  struct Fixture {
    const char *path;
    std::string command;
    oa::VideoContainerKind kind;
    oa::VideoCodec codec;
    bool expectAnnexB;
  };
  const Fixture fixtures[] = {
      {
          "/tmp/oa_video_demuxer_h264.mkv",
          std::string("ffmpeg -v error -y -i \"") + kShibuyaH264 +
          "\" -map 0:v:0 -c copy -an /tmp/oa_video_demuxer_h264.mkv",
          oa::VideoContainerKind::Matroska,
          oa::VideoCodec::H264,
          true,
      },
      {
          "/tmp/oa_video_demuxer_h264.ts",
          std::string("ffmpeg -v error -y -i \"") + kShibuyaH264 +
          "\" -map 0:v:0 -c copy -an -bsf:v h264_mp4toannexb -f mpegts "
          "/tmp/oa_video_demuxer_h264.ts",
          oa::VideoContainerKind::MpegTs,
          oa::VideoCodec::H264,
          true,
      },
      {
          "/tmp/oa_video_demuxer_h264_fragmented.mp4",
          std::string("ffmpeg -v error -y -i \"") + kShibuyaH264 +
          "\" -map 0:v:0 -c copy -an "
          "-movflags frag_keyframe+empty_moov+default_base_moof "
          "/tmp/oa_video_demuxer_h264_fragmented.mp4",
          oa::VideoContainerKind::Mp4,
          oa::VideoCodec::H264,
          true,
      },
      {
          "/tmp/oa_video_demuxer_vp9.webm",
          std::string("ffmpeg -v error -y -i \"") + kShibuyaVp9 +
          "\" -map 0:v:0 -c copy -an /tmp/oa_video_demuxer_vp9.webm",
          oa::VideoContainerKind::WebM,
          oa::VideoCodec::VP9,
          false,
      },
  };

  for (const Fixture &fixture : fixtures) {
    ASSERT_EQ(std::system(fixture.command.c_str()), 0) << fixture.path;
    auto demuxerResult = oa::VideoDemuxer::open(fixture.path);
    ASSERT_TRUE(demuxerResult.isOk())
        << fixture.path << ": " << demuxerResult.getStatus().toString().cStr();
    EXPECT_EQ(demuxerResult->getInfo().kind, fixture.kind) << fixture.path;
    EXPECT_EQ(demuxerResult->getInfo().codec, fixture.codec) << fixture.path;
    EXPECT_GT(demuxerResult->getInfo().width, 0U) << fixture.path;
    EXPECT_GT(demuxerResult->getInfo().height, 0U) << fixture.path;

    oa::VideoPacket packet{};
    ASSERT_TRUE(demuxerResult->readNextPacket(packet).isOk()) << fixture.path;
    ASSERT_GT(packet.data.size(), 4U) << fixture.path;
    if (fixture.expectAnnexB) {
      const bool startCode3 = packet.data[0] == 0U and packet.data[1] == 0U and
                              packet.data[2] == 1U;
      const bool startCode4 = packet.data[0] == 0U and packet.data[1] == 0U and
                              packet.data[2] == 0U and packet.data[3] == 1U;
      EXPECT_TRUE(startCode3 or startCode4)
          << fixture.path << " did not emit Annex-B H.264";
    }
    if (demuxerResult->isSeekable()) {
      EXPECT_TRUE(demuxerResult->seek(0U).isOk()) << fixture.path;
    }
    EXPECT_TRUE(demuxerResult->close().isOk()) << fixture.path;
    EXPECT_TRUE(demuxerResult->close().isOk()) << fixture.path;
    std::remove(fixture.path);
  }
#endif
}

TEST(VideoDemuxer, UnifiedVideoOpenNextSeekFlush) {
  if (not datasetAvailable(kShibuyaH264)) {
    GTEST_SKIP() << "Shibuya H.264 dataset not present";
  }
  auto *engine = testEnginePtr();
  if (engine == nullptr or
      not testVideoDecodeSupported(*engine, oa::VideoCodec::H264)) {
    GTEST_SKIP() << "vulkan Video H.264 decode not supported";
  }
  oa::VideoPlayerConfig config;
  config.uri = kShibuyaH264;
  config.audio = false;
  config.startPlaying = false;
  config.loop = false;
  auto opened = oa::VideoPlayer::open(*engine, config);
  ASSERT_TRUE(opened.isOk()) << opened.getStatus().toString();
  oa::VideoPlayer video = oa::move(*opened);
  EXPECT_NE(video.currentFrame().imageView, VK_NULL_HANDLE);
  const oa::U64 durationUs = video.durationUs();
  ASSERT_GT(durationUs, 0U);
  EXPECT_LE(video.positionUs(), durationUs);
  video.setLoop(true);
  EXPECT_TRUE(video.isLooping());
  video.setLoop(false);
  EXPECT_FALSE(video.isLooping());
  const oa::I64 firstIndex = video.index();
  ASSERT_TRUE(video.next().isOk());
  EXPECT_GT(video.index(), firstIndex);
  ASSERT_TRUE(video.seekUs(durationUs / 2U).isOk());
  EXPECT_NEAR(static_cast<double>(video.positionUs()),
              static_cast<double>(durationUs / 2U), 1'000'000.0);
  ASSERT_TRUE(video.seek(0U).isOk());
  EXPECT_NE(video.currentFrame().imageView, VK_NULL_HANDLE);
  ASSERT_TRUE(video.flush().isOk());
  EXPECT_TRUE(video.close().isOk());
  EXPECT_TRUE(video.close().isOk());
}

// Multi-frame H.264 decode against the shibuya MP4: opens the file, brings
// up a vulkan decoder using the SPS-derived profile, and decodes 30 frames
// (one GOP-ish window) including any B-frames the encoder emitted. Failures
// here flag missing pieces in the IDR-P-B reference picture path.
TEST(VideoDemuxer, DecodesShibuyaH265FirstFrame) {
  if (not datasetAvailable(kShibuyaH265)) {
    GTEST_SKIP() << "Shibuya H.265 MP4 dataset not present";
  }
  auto demuxerResult = oa::VideoDemuxer::open(kShibuyaH265);
  ASSERT_TRUE(demuxerResult.isOk());

  if (!testEnginePtr()) {
    GTEST_SKIP() << "No vulkan compute engine available";
  }
  auto &engine = testEngine();
  if (not testVideoDecodeSupported(engine, oa::VideoCodec::H265)) {
    GTEST_SKIP() << "vulkan Video H.265 decode not supported";
  }

  auto profile = demuxerResult->getVideoProfile();
  profile.maxDpbSlots = 16;
  auto decoderResult = oa::VideoDecoder::create(engine, profile);
  ASSERT_TRUE(decoderResult.isOk()) << decoderResult.getStatus().toString();
  auto decoder = oa::move(*decoderResult);

  oa::VideoPacket pkt{};
  ASSERT_TRUE(demuxerResult->readNextPacket(pkt).isOk());
  ASSERT_GT(pkt.data.size(), 0U);

  oa::VideoFrame frame{};
  ASSERT_TRUE(oa::VideoDecoderInternal::decodeFrame(
                  decoder,
                  oa::Span<const oa::U8>(pkt.data.data(), pkt.data.size()),
                  frame)
                  .isOk());
  EXPECT_NE(frame.imageView, VK_NULL_HANDLE);
  EXPECT_EQ(oa::VideoDecoderInternal::getCachedH265VpsCount(decoder), 1u);
  EXPECT_EQ(oa::VideoDecoderInternal::getCachedH265SpsCount(decoder), 1u);
  EXPECT_EQ(oa::VideoDecoderInternal::getCachedH265PpsCount(decoder), 1u);
  EXPECT_TRUE(decoder.close().isOk());
}

TEST(VideoDemuxer, DecodesEntireShibuyaH265) {
  if (not datasetAvailable(kShibuyaH265)) {
    GTEST_SKIP() << "Shibuya H.265 MP4 dataset not present";
  }
  auto demuxerResult = oa::VideoDemuxer::open(kShibuyaH265);
  ASSERT_TRUE(demuxerResult.isOk());

  if (!testEnginePtr()) {
    GTEST_SKIP() << "No vulkan compute engine available";
  }
  auto &engine = testEngine();
  if (not testVideoDecodeSupported(engine, oa::VideoCodec::H265)) {
    GTEST_SKIP() << "vulkan Video H.265 decode not supported";
  }

  auto profile = demuxerResult->getVideoProfile();
  profile.maxDpbSlots = 16;
  auto decoderResult = oa::VideoDecoder::create(engine, profile);
  ASSERT_TRUE(decoderResult.isOk()) << decoderResult.getStatus().toString();
  auto decoder = oa::move(*decoderResult);

  oa::U32 decoded = 0;
  constexpr oa::U32 frameCount = 1607;
  for (oa::U32 i = 0; i < frameCount; ++i) {
    oa::VideoPacket pkt{};
    ASSERT_TRUE(demuxerResult->readNextPacket(pkt).isOk()) << "packet " << i;
    oa::VideoFrame frame{};
    const oa::Status decodeStatus = oa::VideoDecoderInternal::decodeFrame(
        decoder, oa::Span<const oa::U8>(pkt.data.data(), pkt.data.size()),
        frame);
    ASSERT_TRUE(decodeStatus.isOk())
        << "frame " << i << ": " << decodeStatus.toString();
    if (frame.imageView != VK_NULL_HANDLE) {
      ++decoded;
    }
  }
  EXPECT_EQ(decoded, frameCount);
  EXPECT_TRUE(decoder.close().isOk());
}

TEST(VideoDemuxer, DecodesShibuyaGopH264) {
  if (not datasetAvailable(kShibuyaH264)) {
    GTEST_SKIP() << "Shibuya MP4 dataset not present";
  }
  auto demuxerResult = oa::VideoDemuxer::open(kShibuyaH264);
  ASSERT_TRUE(demuxerResult.isOk());

  if (!testEnginePtr()) {
    GTEST_SKIP() << "No vulkan compute engine available";
  }
  auto &engine = testEngine();
  if (not testVideoDecodeSupported(engine, oa::VideoCodec::H264)) {
    GTEST_SKIP() << "vulkan Video H.264 decode not supported";
  }

  auto profile = demuxerResult->getVideoProfile();
  profile.maxDpbSlots = 16;
  auto decoderResult = oa::VideoDecoder::create(engine, profile);
  ASSERT_TRUE(decoderResult.isOk()) << decoderResult.getStatus().toString();
  auto decoder = oa::move(*decoderResult);
  EXPECT_EQ(oa::VideoDecoderInternal::getBitstreamRingSize(decoder), 4U);

  oa::U32 decoded = 0;
  oa::U32 keyframes = 0;
  for (int i = 0; i < 30; ++i) {
    oa::VideoPacket pkt{};
    auto packetStatus = demuxerResult->readNextPacket(pkt);
    ASSERT_TRUE(packetStatus.isOk()) << "packet " << i;

    oa::VideoFrame frame{};
    auto decodeStatus = oa::VideoDecoderInternal::decodeFrame(
        decoder, oa::Span<const oa::U8>(pkt.data.data(), pkt.data.size()),
        frame);
    ASSERT_TRUE(decodeStatus.isOk())
        << "frame " << i << " (keyframe=" << pkt.isKeyframe
        << "): " << decodeStatus.toString();
    ++decoded;
    if (pkt.isKeyframe) {
      ++keyframes;
    }
  }
  EXPECT_EQ(decoded, 30U);
  EXPECT_GE(keyframes, 1U); // at least the bootstrap IDR
}

TEST(VideoDemuxer, DecodesShibuyaVp9FirstFrame) {
  if (not datasetAvailable(kShibuyaVp9)) {
    GTEST_SKIP() << "Shibuya VP9 MP4 dataset not present";
  }
  auto demuxerResult = oa::VideoDemuxer::open(kShibuyaVp9);
  ASSERT_TRUE(demuxerResult.isOk());

  if (!testEnginePtr()) {
    GTEST_SKIP() << "No vulkan compute engine available";
  }
  auto &engine = testEngine();
  if (not testVideoDecodeSupported(engine, oa::VideoCodec::VP9)) {
    GTEST_SKIP() << "vulkan Video VP9 decode not supported";
  }

  auto profile = demuxerResult->getVideoProfile();
  profile.maxDpbSlots = 9;
  auto capabilities =
      oa::VideoDecoder::queryDecodeCapabilities(engine, profile);
  ASSERT_TRUE(capabilities.isOk()) << capabilities.getStatus().toString();
  if (not capabilities->supported) {
    GTEST_SKIP() << "exact VP9 stream profile unavailable: requested level "
                 << profile.level << ", device maximum "
                 << capabilities->maxLevel;
  }
  auto decoderResult = oa::VideoDecoder::create(engine, profile);
  ASSERT_TRUE(decoderResult.isOk()) << decoderResult.getStatus().toString();
  auto decoder = oa::move(*decoderResult);

  oa::VideoPacket pkt{};
  ASSERT_TRUE(demuxerResult->readNextPacket(pkt).isOk());
  ASSERT_GT(pkt.data.size(), 0U);

  oa::VideoFrame frame{};
  ASSERT_TRUE(oa::VideoDecoderInternal::decodeFrame(
                  decoder,
                  oa::Span<const oa::U8>(pkt.data.data(), pkt.data.size()),
                  frame)
                  .isOk());
  EXPECT_NE(frame.imageView, VK_NULL_HANDLE);
  EXPECT_TRUE(decoder.close().isOk());
}

TEST(VideoDemuxer, LongPlaybackLoopResetAndBackStep) {
  if (not datasetAvailable(kShibuyaH264)) {
    GTEST_SKIP() << "Shibuya MP4 dataset not present";
  }
  auto *engine = testEnginePtr();
  if (engine == nullptr or
      not testVideoDecodeSupported(*engine, oa::VideoCodec::H264)) {
    GTEST_SKIP() << "vulkan Video H.264 decode not supported";
  }

  oa::VideoPlayerConfig cfg;
  cfg.uri = kShibuyaH264;
  cfg.loop = true;
  cfg.startPlaying = false;
  cfg.reorderDepth = 4;
  cfg.preferHardwareYCbCr = true;
  auto videoResult = oa::VideoPlayer::open(*engine, cfg);
  ASSERT_TRUE(videoResult.isOk()) << videoResult.getStatus().toString();
  oa::VideoPlayer video = oa::move(*videoResult);

  oa::U64 priorPts = video.currentFrame().presentationTimestamp;
  for (oa::U32 i = 0; i < 180; ++i) {
    ASSERT_TRUE(video.next().isOk()) << "step " << i;
    const oa::U64 pts = video.currentFrame().presentationTimestamp;
    EXPECT_GE(pts, priorPts);
    priorPts = pts;
  }

  const oa::I64 beforeScrub = video.index();
  auto beforeScrubRgba = video.readbackCurrentRgba();
  ASSERT_TRUE(beforeScrubRgba.isOk()) << beforeScrubRgba.getStatus().toString();
  ASSERT_TRUE(video.stepFrames(5).isOk());
  EXPECT_EQ(video.index(), beforeScrub + 5);
  ASSERT_TRUE(video.stepFrames(-5).isOk());
  EXPECT_EQ(video.index(), beforeScrub);
  auto afterScrubRgba = video.readbackCurrentRgba();
  ASSERT_TRUE(afterScrubRgba.isOk()) << afterScrubRgba.getStatus().toString();
  ASSERT_EQ(afterScrubRgba->size(), beforeScrubRgba->size());
  oa::U64 scrubAbsoluteError = 0U;
  for (oa::Usize i = 0U; i < beforeScrubRgba->size(); ++i) {
    scrubAbsoluteError +=
        static_cast<oa::U64>(std::abs(static_cast<int>((*afterScrubRgba)[i]) -
                                      static_cast<int>((*beforeScrubRgba)[i])));
  }
  EXPECT_LT(static_cast<oa::F64>(scrubAbsoluteError) /
                static_cast<oa::F64>(beforeScrubRgba->size()),
            0.01)
      << "backward seek did not reconstruct the same display frame";
  ASSERT_TRUE(video.stepBackward().isOk());
  EXPECT_EQ(video.index(), beforeScrub - 1);
  ASSERT_TRUE(video.next().isOk());
  EXPECT_EQ(video.index(), beforeScrub);
  for (oa::U32 i = 0; i < 12; ++i) {
    ASSERT_TRUE(video.stepFrames(-5).isOk()) << "repeat backward scrub " << i;
    EXPECT_EQ(video.index(), beforeScrub - 5);
    ASSERT_TRUE(video.stepFrames(5).isOk()) << "repeat forward scrub " << i;
    EXPECT_EQ(video.index(), beforeScrub);
  }

  video.reset();
  ASSERT_TRUE(video.next().isOk());
  EXPECT_EQ(video.index(), 1);
  const oa::Usize frameCount = video.frameCount();
  for (oa::Usize i = 1; i < frameCount; ++i) {
    ASSERT_TRUE(video.next().isOk()) << "loop step " << i;
  }
  EXPECT_EQ(video.index(), static_cast<oa::I64>(frameCount));
  ASSERT_TRUE(video.next().isOk());
  EXPECT_EQ(video.index(), 1);
  EXPECT_FALSE(video.isDone());
  ASSERT_TRUE(video.close().isOk());
}

TEST(VideoDemuxer, BackwardSeekReconstructsFrameAcrossCodecs) {
  struct Case {
    const char *path;
    oa::VideoCodec codec;
  };
  const Case cases[] = {
      {kShibuyaH265, oa::VideoCodec::H265},
      {kShibuyaAv1, oa::VideoCodec::AV1},
      {kShibuyaVp9, oa::VideoCodec::VP9},
  };
  auto *engine = testEnginePtr();
  ASSERT_NE(engine, nullptr);
  oa::U32 exercised = 0U;
  for (const Case &testCase : cases) {
    if (!datasetAvailable(testCase.path) ||
        !testVideoDecodeSupported(*engine, testCase.codec)) {
      continue;
    }
    auto stream = oa::VideoDemuxer::open(testCase.path);
    ASSERT_TRUE(stream.isOk())
        << testCase.path << ": " << stream.getStatus().toString();
    auto capabilities = oa::VideoDecoder::queryDecodeCapabilities(
        *engine, stream->getVideoProfile());
    ASSERT_TRUE(capabilities.isOk())
        << testCase.path << ": " << capabilities.getStatus().toString();
    if (not capabilities->supported)
      continue;
    ++exercised;
    oa::VideoPlayerConfig config;
    config.uri = testCase.path;
    config.audio = false;
    config.loop = false;
    config.startPlaying = false;
    config.reorderDepth = 4;
    config.preferHardwareYCbCr = true;
    auto opened = oa::VideoPlayer::open(*engine, config);
    ASSERT_TRUE(opened.isOk())
        << testCase.path << ": " << opened.getStatus().toString();
    oa::VideoPlayer video = oa::move(*opened);
    for (oa::U32 frame = 0U; frame < 24U; ++frame) {
      ASSERT_TRUE(video.next().isOk()) << testCase.path << " frame " << frame;
    }
    const oa::I64 referenceIndex = video.index();
    auto reference = video.readbackCurrentRgba();
    ASSERT_TRUE(reference.isOk()) << testCase.path;
    ASSERT_TRUE(video.stepFrames(3).isOk()) << testCase.path;
    ASSERT_TRUE(video.stepFrames(-3).isOk()) << testCase.path;
    EXPECT_EQ(video.index(), referenceIndex) << testCase.path;
    auto reconstructed = video.readbackCurrentRgba();
    ASSERT_TRUE(reconstructed.isOk()) << testCase.path;
    ASSERT_EQ(reconstructed->size(), reference->size()) << testCase.path;
    oa::U64 absoluteError = 0U;
    for (oa::Usize i = 0U; i < reference->size(); ++i) {
      absoluteError +=
          static_cast<oa::U64>(std::abs(static_cast<int>((*reconstructed)[i]) -
                                        static_cast<int>((*reference)[i])));
    }
    EXPECT_LT(static_cast<oa::F64>(absoluteError) /
                  static_cast<oa::F64>(reference->size()),
              0.01)
        << testCase.path << " backward seek changed the display frame";
    ASSERT_TRUE(video.close().isOk());
  }
  if (exercised == 0U) {
    GTEST_SKIP() << "No H.265, AV1 or VP9 decode path available";
  }
}

TEST(VideoDemuxer, FirstFrameMatchesFfmpegReference) {
  if (not datasetAvailable(kShibuyaH264)) {
    GTEST_SKIP() << "Shibuya MP4 dataset not present";
  }
  if (std::system("command -v ffmpeg >/dev/null 2>&1") != 0) {
    GTEST_SKIP() << "ffmpeg is not installed";
  }
  auto *engine = testEnginePtr();
  if (engine == nullptr or
      not testVideoDecodeSupported(*engine, oa::VideoCodec::H264)) {
    GTEST_SKIP() << "vulkan Video H.264 decode not supported";
  }

  auto demuxerResult = oa::VideoDemuxer::open(kShibuyaH264);
  ASSERT_TRUE(demuxerResult.isOk());
  auto profile = demuxerResult->getVideoProfile();
  profile.maxDpbSlots = 16;
  auto decoderResult = oa::VideoDecoder::create(*engine, profile);
  ASSERT_TRUE(decoderResult.isOk()) << decoderResult.getStatus().toString();
  auto decoder = oa::move(*decoderResult);

  oa::VideoPacket packet;
  ASSERT_TRUE(demuxerResult->readNextPacket(packet).isOk());
  oa::VideoConversionOptions options;
  options.preferHardwareYCbCr = false;
  options.filter = oa::Filter::Nearest;
  oa::VideoFrame frame;
  ASSERT_TRUE(
      oa::VideoDecoderInternal::decodeFrameWithConversion(
          decoder,
          oa::Span<const oa::U8>(packet.data.data(), packet.data.size()),
          options, frame)
          .isOk());
  auto rgba = decoder.readbackRgba(frame);
  ASSERT_TRUE(rgba.isOk()) << rgba.getStatus().toString();

  const char *refPath = "/tmp/oa_shibuya_frame0_rgba.bin";
  const oa::String command =
      oa::String("ffmpeg -v error -y -i \"") + kShibuyaH264 +
      "\" -frames:v 1 -f rawvideo -pix_fmt rgba " + refPath;
  ASSERT_EQ(std::system(command.cStr()), 0);
  auto reference = oa::Filesystem::readBinary(oa::Path(refPath));
  std::remove(refPath);
  ASSERT_TRUE(reference.isOk());
  ASSERT_EQ(reference->size(), rgba->size());

  oa::U64 absoluteError = 0;
  oa::U8 maxError = 0;
  for (oa::Usize i = 0; i < rgba->size(); ++i) {
    const oa::U8 error = static_cast<oa::U8>(std::abs(
        static_cast<int>((*rgba)[i]) - static_cast<int>((*reference)[i])));
    absoluteError += error;
    if (error > maxError) {
      maxError = error;
    }
  }
  const oa::F64 meanAbsoluteError =
      static_cast<oa::F64>(absoluteError) / static_cast<oa::F64>(rgba->size());
  EXPECT_LT(meanAbsoluteError, 3.0);
  EXPECT_LT(maxError, 32U);
}

TEST(VideoDemuxer, FirstFrameNv12MatchesFfmpegReference) {
  if (not datasetAvailable(kShibuyaH264)) {
    GTEST_SKIP() << "Shibuya MP4 dataset not present";
  }
  if (std::system("command -v ffmpeg >/dev/null 2>&1") != 0) {
    GTEST_SKIP() << "ffmpeg is not installed";
  }
  auto *engine = testEnginePtr();
  if (engine == nullptr or
      not testVideoDecodeSupported(*engine, oa::VideoCodec::H264)) {
    GTEST_SKIP() << "vulkan Video H.264 decode not supported";
  }

  auto demuxerResult = oa::VideoDemuxer::open(kShibuyaH264);
  ASSERT_TRUE(demuxerResult.isOk());
  auto profile = demuxerResult->getVideoProfile();
  profile.maxDpbSlots = 16;
  auto decoderResult = oa::VideoDecoder::create(*engine, profile);
  ASSERT_TRUE(decoderResult.isOk()) << decoderResult.getStatus().toString();
  auto decoder = oa::move(*decoderResult);

  oa::VideoPacket packet;
  ASSERT_TRUE(demuxerResult->readNextPacket(packet).isOk());
  oa::VideoFrame frame;
  ASSERT_TRUE(
      oa::VideoDecoderInternal::decodeFrame(
          decoder,
          oa::Span<const oa::U8>(packet.data.data(), packet.data.size()), frame)
          .isOk());
  auto nv12 = decoder.readbackNv12(frame);
  ASSERT_TRUE(nv12.isOk()) << nv12.getStatus().toString();

  const char *refPath = "/tmp/oa_shibuya_frame0_nv12.bin";
  const oa::String command =
      oa::String("ffmpeg -v error -y -i \"") + kShibuyaH264 +
      "\" -frames:v 1 -f rawvideo -pix_fmt nv12 " + refPath;
  ASSERT_EQ(std::system(command.cStr()), 0);
  auto reference = oa::Filesystem::readBinary(oa::Path(refPath));
  std::remove(refPath);
  ASSERT_TRUE(reference.isOk());
  ASSERT_EQ(reference->size(), nv12->size());

  const oa::Usize lumaBytes = static_cast<oa::Usize>(profile.width) *
                              static_cast<oa::Usize>(profile.height);
  oa::U64 lumaError = 0;
  oa::U64 chromaError = 0;
  oa::U64 decodedLumaSum = 0;
  oa::U64 referenceLumaSum = 0;
  for (oa::Usize i = 0; i < nv12->size(); ++i) {
    const oa::U64 error = static_cast<oa::U64>(std::abs(
        static_cast<int>((*nv12)[i]) - static_cast<int>((*reference)[i])));
    if (i < lumaBytes) {
      lumaError += error;
      decodedLumaSum += (*nv12)[i];
      referenceLumaSum += (*reference)[i];
    } else {
      chromaError += error;
    }
  }
  const oa::F64 lumaMae =
      static_cast<oa::F64>(lumaError) / static_cast<oa::F64>(lumaBytes);
  const oa::F64 chromaMae = static_cast<oa::F64>(chromaError) /
                            static_cast<oa::F64>(nv12->size() - lumaBytes);
  oa::F64 bestShiftedLumaMae = lumaMae;
  oa::I32 bestRowShift = 0;
  for (oa::I32 rowShift = -16; rowShift <= 16; ++rowShift) {
    oa::U64 shiftedError = 0;
    oa::U64 shiftedPixels = 0;
    for (oa::I32 y = 0; y < static_cast<oa::I32>(profile.height); ++y) {
      const oa::I32 referenceY = y + rowShift;
      if (referenceY < 0 ||
          referenceY >= static_cast<oa::I32>(profile.height)) {
        continue;
      }
      for (oa::U32 x = 0; x < profile.width; ++x) {
        const oa::Usize decodedIndex =
            static_cast<oa::Usize>(y) * profile.width + x;
        const oa::Usize referenceIndex =
            static_cast<oa::Usize>(referenceY) * profile.width + x;
        shiftedError += static_cast<oa::U64>(
            std::abs(static_cast<int>((*nv12)[decodedIndex]) -
                     static_cast<int>((*reference)[referenceIndex])));
        ++shiftedPixels;
      }
    }
    const oa::F64 shiftedMae = static_cast<oa::F64>(shiftedError) /
                               static_cast<oa::F64>(shiftedPixels);
    if (shiftedMae < bestShiftedLumaMae) {
      bestShiftedLumaMae = shiftedMae;
      bestRowShift = rowShift;
    }
  }
  EXPECT_LT(lumaMae, 3.0) << "decodedMean="
                          << static_cast<oa::F64>(decodedLumaSum) /
                                 static_cast<oa::F64>(lumaBytes)
                          << " referenceMean="
                          << static_cast<oa::F64>(referenceLumaSum) /
                                 static_cast<oa::F64>(lumaBytes)
                          << " bestRowShift=" << bestRowShift
                          << " shiftedMae=" << bestShiftedLumaMae;
  EXPECT_LT(chromaMae, 3.0);
  EXPECT_TRUE(decoder.close().isOk());
}

TEST(VideoDemuxer, FirstTwelveDisplayFramesMatchFfmpegReference) {
  if (not datasetAvailable(kShibuyaH264)) {
    GTEST_SKIP() << "Shibuya MP4 dataset not present";
  }
  if (std::system("command -v ffmpeg >/dev/null 2>&1") != 0) {
    GTEST_SKIP() << "ffmpeg is not installed";
  }
  auto *engine = testEnginePtr();
  if (engine == nullptr or
      not testVideoDecodeSupported(*engine, oa::VideoCodec::H264)) {
    GTEST_SKIP() << "vulkan Video H.264 decode not supported";
  }

  constexpr oa::Usize frameCount = 12;
  constexpr oa::Usize width = 1920;
  constexpr oa::Usize height = 1080;
  constexpr oa::Usize frameBytes = width * height * 4;
  const char *refPath = "/tmp/oa_shibuya_first12_rgba.bin";
  const oa::String command =
      oa::String("ffmpeg -v error -y -i \"") + kShibuyaH264 +
      "\" -frames:v 12 -f rawvideo -pix_fmt rgba " + refPath;
  ASSERT_EQ(std::system(command.cStr()), 0);
  auto reference = oa::Filesystem::readBinary(oa::Path(refPath));
  std::remove(refPath);
  ASSERT_TRUE(reference.isOk());
  ASSERT_EQ(reference->size(), frameBytes * frameCount);

  oa::VideoPlayerConfig cfg;
  cfg.uri = kShibuyaH264;
  cfg.startPlaying = false;
  cfg.loop = false;
  cfg.preferHardwareYCbCr = false;
  cfg.filter = oa::Filter::Nearest;
  auto videoResult = oa::VideoPlayer::open(*engine, cfg);
  ASSERT_TRUE(videoResult.isOk()) << videoResult.getStatus().toString();
  oa::VideoPlayer video = oa::move(*videoResult);

  for (oa::Usize frameIndex = 0; frameIndex < frameCount; ++frameIndex) {
    if (frameIndex > 0) {
      ASSERT_TRUE(video.next().isOk()) << "frame " << frameIndex;
    }
    auto rgba = video.readbackCurrentRgba();
    ASSERT_TRUE(rgba.isOk())
        << "frame " << frameIndex << ": " << rgba.getStatus().toString();
    ASSERT_EQ(rgba->size(), frameBytes);

    oa::U64 absoluteError = 0;
    oa::U8 maxError = 0;
    const oa::U8 *expected = reference->data() + frameIndex * frameBytes;
    for (oa::Usize i = 0; i < frameBytes; ++i) {
      const oa::U8 error = static_cast<oa::U8>(std::abs(
          static_cast<int>((*rgba)[i]) - static_cast<int>(expected[i])));
      absoluteError += error;
      maxError = std::max(maxError, error);
    }
    const oa::F64 meanAbsoluteError =
        static_cast<oa::F64>(absoluteError) / static_cast<oa::F64>(frameBytes);
    EXPECT_LT(meanAbsoluteError, 3.0)
        << "display frame " << frameIndex
        << " PTS=" << video.currentFrame().presentationTimestamp
        << " maxError=" << static_cast<oa::U32>(maxError);
  }
  ASSERT_TRUE(video.close().isOk());
}

TEST(VideoDemuxer, SustainedH264DisplayFramesMatchFfmpegReference) {
  if (not datasetAvailable(kShibuyaH264)) {
    GTEST_SKIP() << "Shibuya MP4 dataset not present";
  }
  if (std::system("command -v ffmpeg >/dev/null 2>&1") != 0) {
    GTEST_SKIP() << "ffmpeg is not installed";
  }
  auto *engine = testEnginePtr();
  if (engine == nullptr or
      not testVideoDecodeSupported(*engine, oa::VideoCodec::H264)) {
    GTEST_SKIP() << "vulkan Video H.264 decode not supported";
  }

  constexpr oa::Usize width = 1920;
  constexpr oa::Usize height = 1080;
  constexpr oa::Usize frameBytes = width * height * 4;
  constexpr oa::U32 sampleFrames[] = {
      0,  1,  2,  3,  4,  5,  11,  12,  13,  29,  30,  31,
      59, 60, 61, 89, 90, 91, 119, 120, 121, 179, 239, 299,
  };
  constexpr oa::Usize sampleCount =
      sizeof(sampleFrames) / sizeof(sampleFrames[0]);
  const char *refPath = "/tmp/oa_shibuya_h264_sustained_rgba.bin";
  oa::String select = "select='";
  for (oa::Usize i = 0; i < sampleCount; ++i) {
    if (i > 0) {
      select += "+";
    }
    select += "eq(n\\,";
    const std::string frameText = std::to_string(sampleFrames[i]);
    select += oa::StringView(frameText.data(), frameText.size());
    select += ")";
  }
  select += "'";
  const oa::String command =
      oa::String("ffmpeg -v error -y -i \"") + kShibuyaH264 + "\" -vf \"" +
      select + "\" -fps_mode passthrough -f rawvideo -pix_fmt rgba " + refPath;
  ASSERT_EQ(std::system(command.cStr()), 0);
  auto reference = oa::Filesystem::readBinary(oa::Path(refPath));
  std::remove(refPath);
  ASSERT_TRUE(reference.isOk());
  ASSERT_EQ(reference->size(), frameBytes * sampleCount);

  oa::VideoPlayerConfig cfg;
  cfg.uri = kShibuyaH264;
  cfg.startPlaying = false;
  cfg.loop = false;
  cfg.preferHardwareYCbCr = false;
  cfg.filter = oa::Filter::Nearest;
  auto videoResult = oa::VideoPlayer::open(*engine, cfg);
  ASSERT_TRUE(videoResult.isOk()) << videoResult.getStatus().toString();
  oa::VideoPlayer video = oa::move(*videoResult);

  oa::U64 priorPts = video.currentFrame().presentationTimestamp;
  oa::Usize sampleIndex = 0;
  for (oa::U32 frameIndex = 0; frameIndex <= sampleFrames[sampleCount - 1];
       ++frameIndex) {
    if (frameIndex > 0) {
      ASSERT_TRUE(video.next().isOk()) << "frame " << frameIndex;
      EXPECT_GT(video.currentFrame().presentationTimestamp, priorPts)
          << "display PTS repeated/regressed at frame " << frameIndex;
      priorPts = video.currentFrame().presentationTimestamp;
    }
    if (sampleIndex >= sampleCount or frameIndex != sampleFrames[sampleIndex]) {
      continue;
    }

    auto rgba = video.readbackCurrentRgba();
    ASSERT_TRUE(rgba.isOk())
        << "frame " << frameIndex << ": " << rgba.getStatus().toString();
    ASSERT_EQ(rgba->size(), frameBytes);

    oa::U64 absoluteError = 0;
    oa::U8 maxError = 0;
    const oa::U8 *expected = reference->data() + sampleIndex * frameBytes;
    for (oa::Usize i = 0; i < frameBytes; ++i) {
      const oa::U8 error = static_cast<oa::U8>(std::abs(
          static_cast<int>((*rgba)[i]) - static_cast<int>(expected[i])));
      absoluteError += error;
      maxError = std::max(maxError, error);
    }
    const oa::F64 meanAbsoluteError =
        static_cast<oa::F64>(absoluteError) / static_cast<oa::F64>(frameBytes);
    EXPECT_LT(meanAbsoluteError, 3.0)
        << "display frame " << frameIndex
        << " PTS=" << video.currentFrame().presentationTimestamp
        << " maxError=" << static_cast<oa::U32>(maxError);
    ++sampleIndex;
  }
  EXPECT_EQ(sampleIndex, sampleCount);
  ASSERT_TRUE(video.close().isOk());
}

void expectCodecPlaybackMatchesFirstTwelveFfmpegFrames(const char *inName,
                                                       const char *inPath,
                                                       oa::VideoCodec inCodec) {
  auto *engine = testEnginePtr();
  if (engine == nullptr) {
    GTEST_SKIP() << "No vulkan compute engine available";
  }
  if (not datasetAvailable(inPath) or
      not testVideoDecodeSupported(*engine, inCodec)) {
    GTEST_SKIP() << inName << " fixture or vulkan Video support unavailable";
  }

  auto demuxerResult = oa::VideoDemuxer::open(inPath);
  ASSERT_TRUE(demuxerResult.isOk()) << inName;
  auto capabilities = oa::VideoDecoder::queryDecodeCapabilities(
      *engine, demuxerResult->getVideoProfile());
  ASSERT_TRUE(capabilities.isOk())
      << inName << ": " << capabilities.getStatus().toString();
  if (not capabilities->supported) {
    GTEST_SKIP() << inName << ": exact stream profile unavailable";
  }
  const oa::Usize frameBytes =
      static_cast<oa::Usize>(demuxerResult->getInfo().width) *
      demuxerResult->getInfo().height * 4U;
  const oa::String refPath =
      oa::String("/tmp/oa_shibuya_") + inName + "_first12_rgba.bin";
  const oa::String command =
      oa::String("ffmpeg -v error -y -i \"") + inPath +
      "\" -vf \"select='between(n\\,0\\,11)'\"" +
      " -fps_mode passthrough -f rawvideo -pix_fmt rgba " + refPath;
  ASSERT_EQ(std::system(command.cStr()), 0) << inName;
  auto reference = oa::Filesystem::readBinary(oa::Path(refPath));
  std::remove(refPath.cStr());
  ASSERT_TRUE(reference.isOk()) << inName;
  ASSERT_EQ(reference->size(), frameBytes * 12U) << inName;

  oa::VideoPlayerConfig cfg;
  cfg.uri = inPath;
  cfg.startPlaying = false;
  cfg.loop = false;
  cfg.preferHardwareYCbCr = false;
  cfg.filter = oa::Filter::Nearest;
  auto videoResult = oa::VideoPlayer::open(*engine, cfg);
  ASSERT_TRUE(videoResult.isOk())
      << inName << ": " << videoResult.getStatus().toString();
  oa::VideoPlayer video = oa::move(*videoResult);

  oa::U64 priorPts = video.currentFrame().presentationTimestamp;
  for (oa::U32 frameIndex = 0; frameIndex < 12U; ++frameIndex) {
    if (frameIndex > 0) {
      ASSERT_TRUE(video.next().isOk()) << inName << " frame " << frameIndex;
      EXPECT_GT(video.currentFrame().presentationTimestamp, priorPts)
          << inName << " frame " << frameIndex;
      priorPts = video.currentFrame().presentationTimestamp;
    }
    auto rgba = video.readbackCurrentRgba();
    ASSERT_TRUE(rgba.isOk()) << inName << " frame " << frameIndex;
    ASSERT_EQ(rgba->size(), frameBytes) << inName;

    oa::U64 absoluteError = 0;
    const oa::U8 *expected = reference->data() + frameIndex * frameBytes;
    for (oa::Usize i = 0; i < frameBytes; ++i) {
      absoluteError += static_cast<oa::U64>(std::abs(
          static_cast<int>((*rgba)[i]) - static_cast<int>(expected[i])));
    }
    const oa::F64 meanAbsoluteError =
        static_cast<oa::F64>(absoluteError) / static_cast<oa::F64>(frameBytes);
    EXPECT_LT(meanAbsoluteError, 3.0) << inName << " frame " << frameIndex;
  }
  ASSERT_TRUE(video.close().isOk());
}

TEST(VideoDemuxer, H264PlaybackMatchesFirstTwelveFfmpegFrames) {
  expectCodecPlaybackMatchesFirstTwelveFfmpegFrames("h264", kShibuyaH264,
                                                    oa::VideoCodec::H264);
}

TEST(VideoDemuxer, H265PlaybackMatchesFirstTwelveFfmpegFrames) {
  expectCodecPlaybackMatchesFirstTwelveFfmpegFrames("h265", kShibuyaH265,
                                                    oa::VideoCodec::H265);
}

TEST(VideoDemuxer, H265PlaybackMatchesFfmpegAcrossEntireVideo) {
  auto *engine = testEnginePtr();
  if (engine == nullptr) {
    GTEST_SKIP() << "No vulkan compute engine available";
  }
  if (not datasetAvailable(kShibuyaH265) ||
      not testVideoDecodeSupported(*engine, oa::VideoCodec::H265)) {
    GTEST_SKIP() << "H.265 fixture or vulkan Video support unavailable";
  }

  constexpr oa::U32 sampleFrames[] = {0u,   255u, 256u, 257u,  470u,  474u,
                                      480u, 720u, 899u, 1200u, 1500u, 1606u};
  constexpr oa::Usize sampleCount =
      sizeof(sampleFrames) / sizeof(sampleFrames[0]);
  const oa::String refPath = "/tmp/oa_shibuya_h265_sustained_rgba.bin";
  const oa::String command =
      oa::String("ffmpeg -v error -y -i \"") + kShibuyaH265 +
      "\" -vf \"select='" + "eq(n\\,0)+eq(n\\,255)+eq(n\\,256)+eq(n\\,257)+" +
      "eq(n\\,470)+eq(n\\,474)+eq(n\\,480)+eq(n\\,720)+eq(n\\,899)+" +
      "eq(n\\,1200)+eq(n\\,1500)+eq(n\\,1606)'\"" +
      " -fps_mode passthrough -f rawvideo -pix_fmt rgba " + refPath;
  ASSERT_EQ(std::system(command.cStr()), 0);

  auto demuxerResult = oa::VideoDemuxer::open(kShibuyaH265);
  ASSERT_TRUE(demuxerResult.isOk());
  const oa::Usize frameBytes =
      static_cast<oa::Usize>(demuxerResult->getInfo().width) *
      demuxerResult->getInfo().height * 4u;
  auto reference = oa::Filesystem::readBinary(oa::Path(refPath));
  std::remove(refPath.cStr());
  ASSERT_TRUE(reference.isOk());
  ASSERT_EQ(reference->size(), frameBytes * sampleCount);

  oa::VideoPlayerConfig cfg;
  cfg.uri = kShibuyaH265;
  cfg.startPlaying = false;
  cfg.loop = false;
  cfg.preferHardwareYCbCr = false;
  cfg.filter = oa::Filter::Nearest;
  auto videoResult = oa::VideoPlayer::open(*engine, cfg);
  ASSERT_TRUE(videoResult.isOk()) << videoResult.getStatus().toString();
  oa::VideoPlayer video = oa::move(*videoResult);

  oa::Usize sampleIndex = 0;
  for (oa::U32 frameIndex = 0; frameIndex <= sampleFrames[sampleCount - 1u];
       ++frameIndex) {
    if (frameIndex > 0) {
      const oa::Status stepStatus = video.next();
      ASSERT_TRUE(stepStatus.isOk())
          << "display frame " << frameIndex << ": " << stepStatus.toString();
    }
    if (frameIndex != sampleFrames[sampleIndex]) {
      continue;
    }
    auto rgba = video.readbackCurrentRgba();
    ASSERT_TRUE(rgba.isOk()) << "display frame " << frameIndex;
    ASSERT_EQ(rgba->size(), frameBytes);

    oa::U64 absoluteError = 0;
    const oa::U8 *expected = reference->data() + sampleIndex * frameBytes;
    for (oa::Usize i = 0; i < frameBytes; ++i) {
      absoluteError += static_cast<oa::U64>(std::abs(
          static_cast<int>((*rgba)[i]) - static_cast<int>(expected[i])));
    }
    const oa::F64 meanAbsoluteError =
        static_cast<oa::F64>(absoluteError) / static_cast<oa::F64>(frameBytes);
    EXPECT_LT(meanAbsoluteError, 3.0) << "display frame " << frameIndex;
    ++sampleIndex;
    if (sampleIndex == sampleCount) {
      break;
    }
  }
  EXPECT_EQ(sampleIndex, sampleCount);
  ASSERT_TRUE(video.close().isOk());
}

TEST(VideoDemuxer, Av1PlaybackMatchesFirstTwelveFfmpegFrames) {
  expectCodecPlaybackMatchesFirstTwelveFfmpegFrames("av1", kShibuyaAv1,
                                                    oa::VideoCodec::AV1);
}

TEST(VideoDemuxer, Vp9PlaybackMatchesFirstTwelveFfmpegFrames) {
  expectCodecPlaybackMatchesFirstTwelveFfmpegFrames("vp9", kShibuyaVp9,
                                                    oa::VideoCodec::VP9);
}

TEST(VideoDemuxer, SustainedH264StepLatencyStaysBelowFrameBudget) {
  if (not datasetAvailable(kShibuyaH264)) {
    GTEST_SKIP() << "Shibuya MP4 dataset not present";
  }
  auto *engine = testEnginePtr();
  if (engine == nullptr or
      not testVideoDecodeSupported(*engine, oa::VideoCodec::H264)) {
    GTEST_SKIP() << "vulkan Video H.264 decode not supported";
  }

  oa::VideoPlayerConfig cfg;
  cfg.uri = kShibuyaH264;
  cfg.startPlaying = false;
  cfg.loop = false;
  cfg.preferHardwareYCbCr = false;
  auto videoResult = oa::VideoPlayer::open(*engine, cfg);
  ASSERT_TRUE(videoResult.isOk()) << videoResult.getStatus().toString();
  oa::VideoPlayer video = oa::move(*videoResult);

  for (oa::U32 i = 0; i < 12U; ++i) {
    ASSERT_TRUE(video.next().isOk());
  }
  std::vector<oa::F64> stepSamplesMs;
  oa::F64 totalStepMs = 0.0;
  constexpr oa::U32 sampleCount = 180;
  stepSamplesMs.reserve(sampleCount);
  for (oa::U32 i = 0; i < sampleCount; ++i) {
    const auto start = std::chrono::steady_clock::now();
    ASSERT_TRUE(video.next().isOk()) << "step " << i;
    const auto end = std::chrono::steady_clock::now();
    const oa::F64 stepMs =
        std::chrono::duration<oa::F64, std::milli>(end - start).count();
    stepSamplesMs.push_back(stepMs);
    totalStepMs += stepMs;
  }
  std::sort(stepSamplesMs.begin(), stepSamplesMs.end());
  constexpr oa::Usize p99Index = (sampleCount * 99U / 100U) - 1U;
  const oa::F64 p99StepMs = stepSamplesMs[p99Index];
  const oa::F64 maxStepMs = stepSamplesMs.back();
  EXPECT_LT(totalStepMs / sampleCount, video.frameIntervalMs() * 0.5)
      << "average decode/convert step exceeds half of the frame budget";
  EXPECT_LT(p99StepMs, video.frameIntervalMs())
      << "sustained decode/convert latency exceeded the full frame budget";
  EXPECT_LT(maxStepMs, video.frameIntervalMs() * 2.0)
      << "a decode/convert step exceeded twice the full frame budget";
  ASSERT_TRUE(video.close().isOk());
}

TEST(FnVideoNal, ParseAndEmitRoundTrip) {
  // 4-byte start code + tiny SPS-like payload, twice.
  const oa::U8 kSps[] = {0x67, 0x42, 0xC0, 0x1E};
  const oa::U8 kPps[] = {0x68, 0xCE, 0x38, 0x80};

  oa::Vector<oa::U8> stream;
  const oa::U8 startCode[4] = {0, 0, 0, 1};
  for (auto byte : startCode) {
    stream.pushBack(byte);
  }
  for (auto byte : kSps) {
    stream.pushBack(byte);
  }
  for (auto byte : startCode) {
    stream.pushBack(byte);
  }
  for (auto byte : kPps) {
    stream.pushBack(byte);
  }

  auto units = oa::FnVideo::parseNalAnnexB(
      oa::Span<const oa::U8>(stream.data(), stream.size()));
  ASSERT_EQ(units.size(), 2U);
  EXPECT_EQ(units[0].type, 7U); // SPS
  EXPECT_EQ(units[1].type, 8U); // PPS

  auto roundTrip = oa::FnVideo::emitNalAnnexB(
      oa::Span<const oa::NalUnit>(units.data(), units.size()));
  ASSERT_EQ(roundTrip.size(), stream.size());
  for (oa::Usize i = 0; i < stream.size(); ++i) {
    EXPECT_EQ(roundTrip[i], stream[i]) << "byte " << i << " differs";
  }

  auto extractedSps = oa::FnVideo::extractSps(
      oa::Span<const oa::U8>(stream.data(), stream.size()));
  auto extractedPps = oa::FnVideo::extractPps(
      oa::Span<const oa::U8>(stream.data(), stream.size()));
  ASSERT_EQ(extractedSps.size(), sizeof(kSps));
  ASSERT_EQ(extractedPps.size(), sizeof(kPps));
}
