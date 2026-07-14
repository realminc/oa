// OA Vision — Video Stream (MP4 demuxer) Tests
// Smoke-test the OaVideoStream MP4 box parser + sample table walker + Seek
// against the shibuya dataset. These tests skip when the dataset isn't on
// disk so CI without the dataset still passes.

#include "../../OaTest.h"

#include <Oa/Vision/VideoStream.h>
#include <Oa/Vision/FnVideo.h>
#include <Oa/Vision/VideoDecoder.h>
#include "../../../Source/Private/Oa/Vision/Video/Decoder/VideoDecoderInternal.h"
#include <Oa/Vision/ItVideo.h>
#include <Oa/Vision/Video.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Sync.h>
#include <Oa/Core/FileIo.h>

#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace {

#if defined(_WIN32)
constexpr const char* kShibuyaH264 =
	"../dataset/video/shibuya_crossing_1080p30_h264.mp4";
constexpr const char* kShibuyaH265 =
	"../dataset/video/shibuya_crossing_1080p30_h265.mp4";
constexpr const char* kShibuyaAv1 =
	"../dataset/video/shibuya_crossing_1080p30_av1.mp4";
constexpr const char* kShibuyaVp9 =
	"../dataset/video/shibuya_crossing_1080p30_vp9.mp4";
#else
constexpr const char* kShibuyaH264 =
	"../dataset/video/shibuya_crossing_1080p30_h264.mp4";
constexpr const char* kShibuyaH265 =
	"../dataset/video/shibuya_crossing_1080p30_h265.mp4";
constexpr const char* kShibuyaAv1 =
	"../dataset/video/shibuya_crossing_1080p30_av1.mp4";
constexpr const char* kShibuyaVp9 =
	"../dataset/video/shibuya_crossing_1080p30_vp9.mp4";
#endif

bool DatasetAvailable(const char* InPath)
{
	auto status = OaFileIo::ReadBinary(OaPath(InPath));
	return status.IsOk();
}


TEST(OaVideoStream, EmptyCompletionTokenIsAlreadyComplete)
{
	OaCompletionToken token;
	EXPECT_FALSE(token.IsValid());
	EXPECT_TRUE(token.IsComplete());
	EXPECT_TRUE(token.Wait().IsOk());
}

} // namespace


TEST(OaVideoStream, OpenShibuyaMp4)
{
	if (not DatasetAvailable(kShibuyaH264)) {
		GTEST_SKIP() << "Shibuya MP4 dataset not present at " << kShibuyaH264;
	}
	auto streamResult = OaVideoStream::OpenFile(kShibuyaH264);
	ASSERT_TRUE(streamResult.IsOk()) << "OpenFile failed: "
		<< streamResult.GetStatus().ToString().c_str();

	auto profile = streamResult->GetVideoProfile();
	EXPECT_EQ(profile.Codec, OaVideoCodec::H264);
	EXPECT_GT(profile.Width,  0U);
	EXPECT_GT(profile.Height, 0U);

	// avcC must be parsed: SPS + PPS available, length size = 4 (typical).
	const auto& avc = streamResult->GetAvcConfig();
	EXPECT_TRUE(avc.Valid);
	EXPECT_GT(avc.SpsAnnexB.Size(), 4U);
	EXPECT_GT(avc.PpsAnnexB.Size(), 4U);
	EXPECT_GE(avc.LengthSize, 1U);
	EXPECT_LE(avc.LengthSize, 4U);
}


TEST(OaVideoStream, ReadFirstPacketNonEmptyAv1Vp9H265)
{
	struct Case {
		const char* Path;
		OaVideoCodec Codec;
	};
	const Case cases[] = {
		{kShibuyaH265, OaVideoCodec::H265},
		{kShibuyaAv1, OaVideoCodec::AV1},
		{kShibuyaVp9, OaVideoCodec::VP9},
	};
	for (const Case& c : cases) {
		if (not DatasetAvailable(c.Path)) {
			GTEST_SKIP() << "Dataset not present: " << c.Path;
		}
		auto streamResult = OaVideoStream::OpenFile(c.Path);
		ASSERT_TRUE(streamResult.IsOk()) << c.Path << ": "
			<< streamResult.GetStatus().ToString().c_str();
		EXPECT_EQ(streamResult->GetVideoProfile().Codec, c.Codec) << c.Path;

		OaVideoPacket pkt{};
		ASSERT_TRUE(streamResult->ReadNextPacket(pkt).IsOk()) << c.Path;
		EXPECT_GT(pkt.Data.Size(), 0U) << c.Path << " first packet must not be empty";
	}
}


TEST(OaVideoStream, ReadFirstFivePackets)
{
	if (not DatasetAvailable(kShibuyaH264)) {
		GTEST_SKIP() << "Shibuya MP4 dataset not present";
	}
	auto streamResult = OaVideoStream::OpenFile(kShibuyaH264);
	ASSERT_TRUE(streamResult.IsOk());

	OaU64 lastDts = 0;
	for (int i = 0; i < 5; ++i) {
		OaVideoPacket pkt{};
		auto status = streamResult->ReadNextPacket(pkt);
		ASSERT_TRUE(status.IsOk()) << "packet " << i << ": " << status.ToString().c_str();
		ASSERT_GE(pkt.Data.Size(), 4U) << "packet " << i << " too short to be Annex-B";
		// Every emitted packet must begin with a 4-byte Annex-B start code.
		EXPECT_EQ(pkt.Data[0], 0U);
		EXPECT_EQ(pkt.Data[1], 0U);
		EXPECT_EQ(pkt.Data[2], 0U);
		EXPECT_EQ(pkt.Data[3], 1U);
		if (i == 0) {
			EXPECT_TRUE(pkt.IsKeyframe) << "first packet should be a keyframe";
			// Bootstrap packet must contain SPS (nal_unit_type 7) before the
			// slice data, since the MP4 stores parameter sets out-of-band.
			bool sawSps = false;
			for (OaUsize p = 0; p + 4 < pkt.Data.Size(); ++p) {
				if (pkt.Data[p] == 0 && pkt.Data[p+1] == 0
				    && pkt.Data[p+2] == 0 && pkt.Data[p+3] == 1) {
					const OaU8 nalType = pkt.Data[p+4] & 0x1F;
					if (nalType == 7) { sawSps = true; break; }
				}
			}
			EXPECT_TRUE(sawSps) << "first keyframe packet must include SPS from avcC";
		} else {
			EXPECT_GE(pkt.DecodeTimestamp, lastDts) << "DTS must be monotonic";
		}
		lastDts = pkt.DecodeTimestamp;
	}
}


TEST(OaVideoStream, SeekSnapsToKeyframe)
{
	if (not DatasetAvailable(kShibuyaH264)) {
		GTEST_SKIP() << "Shibuya MP4 dataset not present";
	}
	auto streamResult = OaVideoStream::OpenFile(kShibuyaH264);
	ASSERT_TRUE(streamResult.IsOk());

	// Seek back to the start — should always succeed and the next packet
	// should be the first keyframe.
	ASSERT_TRUE(streamResult->Seek(0).IsOk());
	OaVideoPacket pkt{};
	ASSERT_TRUE(streamResult->ReadNextPacket(pkt).IsOk());
	EXPECT_TRUE(pkt.IsKeyframe);
}


// Exercise every bounded container backend with real packet data. These are
// remuxes (stream copy), not transcodes, so the test validates the demux and
// codec-configuration paths without spending time or changing the source
// elementary stream. FFmpeg is the fixture builder and OA remains the system
// under test.
TEST(OaVideoStream, NativeContainerCoverage)
{
#if defined(_WIN32)
	GTEST_SKIP() << "fixture remux commands are currently defined for Unix hosts";
#else
	if (not DatasetAvailable(kShibuyaH264)
		or not DatasetAvailable(kShibuyaVp9)) {
		GTEST_SKIP() << "Shibuya H.264/VP9 datasets are not present";
	}
	if (std::system("ffmpeg -version >/dev/null 2>&1") != 0) {
		GTEST_SKIP() << "ffmpeg is unavailable for container fixture creation";
	}

	struct Fixture {
		const char* Path;
		const char* Command;
		OaContainerKind Kind;
		OaVideoCodec Codec;
		bool ExpectAnnexB;
	};
	const Fixture fixtures[] = {
		{
			"/tmp/oa_video_stream_h264.mkv",
			"ffmpeg -v error -y -i ../dataset/video/shibuya_crossing_1080p30_h264.mp4 -map 0:v:0 -c copy -an /tmp/oa_video_stream_h264.mkv",
			OaContainerKind::Matroska, OaVideoCodec::H264, true,
		},
		{
			"/tmp/oa_video_stream_h264.ts",
			"ffmpeg -v error -y -i ../dataset/video/shibuya_crossing_1080p30_h264.mp4 -map 0:v:0 -c copy -an -bsf:v h264_mp4toannexb -f mpegts /tmp/oa_video_stream_h264.ts",
			OaContainerKind::MpegTs, OaVideoCodec::H264, true,
		},
		{
			"/tmp/oa_video_stream_h264_fragmented.mp4",
			"ffmpeg -v error -y -i ../dataset/video/shibuya_crossing_1080p30_h264.mp4 -map 0:v:0 -c copy -an -movflags frag_keyframe+empty_moov+default_base_moof /tmp/oa_video_stream_h264_fragmented.mp4",
			OaContainerKind::Mp4, OaVideoCodec::H264, true,
		},
		{
			"/tmp/oa_video_stream_vp9.webm",
			"ffmpeg -v error -y -i ../dataset/video/shibuya_crossing_1080p30_vp9.mp4 -map 0:v:0 -c copy -an /tmp/oa_video_stream_vp9.webm",
			OaContainerKind::WebM, OaVideoCodec::VP9, false,
		},
	};

	for (const Fixture& fixture : fixtures) {
		ASSERT_EQ(std::system(fixture.Command), 0) << fixture.Path;
		auto streamResult = OaVideoStream::Open(fixture.Path);
		ASSERT_TRUE(streamResult.IsOk())
			<< fixture.Path << ": "
			<< streamResult.GetStatus().ToString().c_str();
		EXPECT_EQ(streamResult->GetInfo().Kind, fixture.Kind) << fixture.Path;
		EXPECT_EQ(streamResult->GetInfo().Codec, fixture.Codec) << fixture.Path;
		EXPECT_GT(streamResult->GetInfo().Width, 0U) << fixture.Path;
		EXPECT_GT(streamResult->GetInfo().Height, 0U) << fixture.Path;

		OaVideoPacket packet{};
		ASSERT_TRUE(streamResult->ReadNextPacket(packet).IsOk()) << fixture.Path;
		ASSERT_GT(packet.Data.Size(), 4U) << fixture.Path;
		if (fixture.ExpectAnnexB) {
			const bool startCode3 = packet.Data[0] == 0U
				and packet.Data[1] == 0U and packet.Data[2] == 1U;
			const bool startCode4 = packet.Data[0] == 0U
				and packet.Data[1] == 0U and packet.Data[2] == 0U
				and packet.Data[3] == 1U;
			EXPECT_TRUE(startCode3 or startCode4)
				<< fixture.Path << " did not emit Annex-B H.264";
		}
		if (streamResult->IsSeekable()) {
			EXPECT_TRUE(streamResult->Seek(0U).IsOk()) << fixture.Path;
		}
		streamResult->Destroy();
		std::remove(fixture.Path);
	}
#endif
}


TEST(OaVideoStream, UnifiedVideoOpenNextSeekFlush)
{
	if (not DatasetAvailable(kShibuyaH264)) {
		GTEST_SKIP() << "Shibuya H.264 dataset not present";
	}
	auto* engine = OaComputeEngine::GetGlobal();
	if (engine == nullptr
		or not OaVideoDecoder::IsCodecSupported(*engine, OaVideoCodec::H264)) {
		GTEST_SKIP() << "Vulkan Video H.264 decode not supported";
	}
	OaVideoConfig config;
	config.Uri = kShibuyaH264;
	config.Audio = false;
	config.StartPlaying = false;
	config.Loop = false;
	auto opened = OaVideo::Open(*engine, config);
	ASSERT_TRUE(opened.IsOk()) << opened.GetStatus().ToString();
	OaVideo video = OaStdMove(*opened);
	EXPECT_NE(video.CurrentFrame().ImageView, VK_NULL_HANDLE);
	const OaI64 firstIndex = video.Index();
	ASSERT_TRUE(video.Next().IsOk());
	EXPECT_GT(video.Index(), firstIndex);
	ASSERT_TRUE(video.Seek(0U).IsOk());
	EXPECT_NE(video.CurrentFrame().ImageView, VK_NULL_HANDLE);
	ASSERT_TRUE(video.Flush().IsOk());
	video.Destroy();
}


// Multi-frame H.264 decode against the shibuya MP4: opens the file, brings
// up a Vulkan decoder using the SPS-derived profile, and decodes 30 frames
// (one GOP-ish window) including any B-frames the encoder emitted. Failures
// here flag missing pieces in the IDR-P-B reference picture path.
TEST(OaVideoStream, DecodesShibuyaH265FirstFrame)
{
	if (not DatasetAvailable(kShibuyaH265)) {
		GTEST_SKIP() << "Shibuya H.265 MP4 dataset not present";
	}
	auto streamResult = OaVideoStream::OpenFile(kShibuyaH265);
	ASSERT_TRUE(streamResult.IsOk());

	if (!OaComputeEngine::GetGlobal()) {
		GTEST_SKIP() << "No Vulkan compute engine available";
	}
	auto& rt = *OaComputeEngine::GetGlobal();
	if (not OaVideoDecoder::IsCodecSupported(rt, OaVideoCodec::H265)) {
		GTEST_SKIP() << "Vulkan Video H.265 decode not supported";
	}

	auto profile = streamResult->GetVideoProfile();
	profile.MaxDpbSlots = 16;
	auto decoderResult = OaVideoDecoder::Create(rt, profile);
	ASSERT_TRUE(decoderResult.IsOk()) << decoderResult.GetStatus().ToString();
	auto decoder = OaStdMove(*decoderResult);

	OaVideoPacket pkt{};
	ASSERT_TRUE(streamResult->ReadNextPacket(pkt).IsOk());
	ASSERT_GT(pkt.Data.Size(), 0U);

	OaVideoFrame frame{};
	ASSERT_TRUE(decoder.DecodeFrame(
		OaSpan<const OaU8>(pkt.Data.Data(), pkt.Data.Size()),
		frame).IsOk());
	EXPECT_NE(frame.ImageView, VK_NULL_HANDLE);
	EXPECT_EQ(OaVideoDecoderInternal::GetCachedH265VpsCount(decoder), 1u);
	EXPECT_EQ(OaVideoDecoderInternal::GetCachedH265SpsCount(decoder), 1u);
	EXPECT_EQ(OaVideoDecoderInternal::GetCachedH265PpsCount(decoder), 1u);
	decoder.Destroy();
}


TEST(OaVideoStream, DecodesEntireShibuyaH265)
{
	if (not DatasetAvailable(kShibuyaH265)) {
		GTEST_SKIP() << "Shibuya H.265 MP4 dataset not present";
	}
	auto streamResult = OaVideoStream::OpenFile(kShibuyaH265);
	ASSERT_TRUE(streamResult.IsOk());

	if (!OaComputeEngine::GetGlobal()) {
		GTEST_SKIP() << "No Vulkan compute engine available";
	}
	auto& rt = *OaComputeEngine::GetGlobal();
	if (not OaVideoDecoder::IsCodecSupported(rt, OaVideoCodec::H265)) {
		GTEST_SKIP() << "Vulkan Video H.265 decode not supported";
	}

	auto profile = streamResult->GetVideoProfile();
	profile.MaxDpbSlots = 16;
	auto decoderResult = OaVideoDecoder::Create(rt, profile);
	ASSERT_TRUE(decoderResult.IsOk()) << decoderResult.GetStatus().ToString();
	auto decoder = OaStdMove(*decoderResult);

	OaU32 decoded = 0;
	constexpr OaU32 frameCount = 1607;
	for (OaU32 i = 0; i < frameCount; ++i) {
		OaVideoPacket pkt{};
		ASSERT_TRUE(streamResult->ReadNextPacket(pkt).IsOk()) << "packet " << i;
		OaVideoFrame frame{};
		const OaStatus decodeStatus = decoder.DecodeFrame(
			OaSpan<const OaU8>(pkt.Data.Data(), pkt.Data.Size()),
			frame);
		ASSERT_TRUE(decodeStatus.IsOk())
			<< "frame " << i << ": " << decodeStatus.ToString();
		if (frame.ImageView != VK_NULL_HANDLE) {
			++decoded;
		}
	}
	EXPECT_EQ(decoded, frameCount);
	decoder.Destroy();
}


TEST(OaVideoStream, DecodesShibuyaGopH264)
{
	if (not DatasetAvailable(kShibuyaH264)) {
		GTEST_SKIP() << "Shibuya MP4 dataset not present";
	}
	auto streamResult = OaVideoStream::OpenFile(kShibuyaH264);
	ASSERT_TRUE(streamResult.IsOk());

	if (!OaComputeEngine::GetGlobal()) {
		GTEST_SKIP() << "No Vulkan compute engine available";
	}
	auto& rt = *OaComputeEngine::GetGlobal();
	if (not OaVideoDecoder::IsCodecSupported(rt, OaVideoCodec::H264)) {
		GTEST_SKIP() << "Vulkan Video H.264 decode not supported";
	}

	auto profile = streamResult->GetVideoProfile();
	profile.MaxDpbSlots = 16;
	auto decoderResult = OaVideoDecoder::Create(rt, profile);
	ASSERT_TRUE(decoderResult.IsOk()) << decoderResult.GetStatus().ToString();
	auto decoder = OaStdMove(*decoderResult);
	EXPECT_EQ(OaVideoDecoderInternal::GetBitstreamRingSize(decoder), 4U);

	OaU32 decoded = 0;
	OaU32 keyframes = 0;
	for (int i = 0; i < 30; ++i) {
		OaVideoPacket pkt{};
		auto packetStatus = streamResult->ReadNextPacket(pkt);
		ASSERT_TRUE(packetStatus.IsOk()) << "packet " << i;

		OaVideoFrame frame{};
		auto decodeStatus = decoder.DecodeFrame(
			OaSpan<const OaU8>(pkt.Data.Data(), pkt.Data.Size()),
			frame);
		ASSERT_TRUE(decodeStatus.IsOk())
			<< "frame " << i << " (keyframe=" << pkt.IsKeyframe << "): "
			<< decodeStatus.ToString();
		++decoded;
		if (pkt.IsKeyframe) { ++keyframes; }
	}
	EXPECT_EQ(decoded, 30U);
	EXPECT_GE(keyframes, 1U);   // at least the bootstrap IDR
}

TEST(OaVideoStream, DecodesShibuyaVp9FirstFrame)
{
	if (not DatasetAvailable(kShibuyaVp9)) {
		GTEST_SKIP() << "Shibuya VP9 MP4 dataset not present";
	}
	auto streamResult = OaVideoStream::OpenFile(kShibuyaVp9);
	ASSERT_TRUE(streamResult.IsOk());

	if (!OaComputeEngine::GetGlobal()) {
		GTEST_SKIP() << "No Vulkan compute engine available";
	}
	auto& rt = *OaComputeEngine::GetGlobal();
	if (not OaVideoDecoder::IsCodecSupported(rt, OaVideoCodec::VP9)) {
		GTEST_SKIP() << "Vulkan Video VP9 decode not supported";
	}

	auto profile = streamResult->GetVideoProfile();
	profile.MaxDpbSlots = 9;
	auto decoderResult = OaVideoDecoder::Create(rt, profile);
	ASSERT_TRUE(decoderResult.IsOk()) << decoderResult.GetStatus().ToString();
	auto decoder = OaStdMove(*decoderResult);

	OaVideoPacket pkt{};
	ASSERT_TRUE(streamResult->ReadNextPacket(pkt).IsOk());
	ASSERT_GT(pkt.Data.Size(), 0U);

	OaVideoFrame frame{};
	ASSERT_TRUE(decoder.DecodeFrame(
		OaSpan<const OaU8>(pkt.Data.Data(), pkt.Data.Size()),
		frame).IsOk());
	EXPECT_NE(frame.ImageView, VK_NULL_HANDLE);
	decoder.Destroy();
}


TEST(OaVideoStream, LongPlaybackLoopResetAndBackStep)
{
	if (not DatasetAvailable(kShibuyaH264)) {
		GTEST_SKIP() << "Shibuya MP4 dataset not present";
	}
	auto* engine = OaComputeEngine::GetGlobal();
	if (engine == nullptr
		or not OaVideoDecoder::IsCodecSupported(*engine, OaVideoCodec::H264)) {
		GTEST_SKIP() << "Vulkan Video H.264 decode not supported";
	}

	OaItVideoConfig cfg;
	cfg.Path = kShibuyaH264;
	cfg.Loop = true;
	cfg.StartPlaying = false;
	cfg.ReorderDepth = 4;
	cfg.PreferHardwareYCbCr = false;
	auto itResult = OaItVideo::Create(*engine, cfg);
	ASSERT_TRUE(itResult.IsOk()) << itResult.GetStatus().ToString();
	OaItVideo it = OaStdMove(*itResult);

	OaU64 priorPts = it.CurrentFrame().PresentationTimestamp;
	for (OaU32 i = 0; i < 180; ++i) {
		ASSERT_TRUE(it.StepForward().IsOk()) << "step " << i;
		const OaU64 pts = it.CurrentFrame().PresentationTimestamp;
		EXPECT_GE(pts, priorPts);
		priorPts = pts;
	}

	const OaI64 beforeScrub = it.Index();
	ASSERT_TRUE(it.StepFrames(5).IsOk());
	EXPECT_EQ(it.Index(), beforeScrub + 5);
	ASSERT_TRUE(it.StepFrames(-5).IsOk());
	EXPECT_EQ(it.Index(), beforeScrub);
	ASSERT_TRUE(it.StepBackward().IsOk());
	EXPECT_EQ(it.Index(), beforeScrub - 1);
	ASSERT_TRUE(it.StepForward().IsOk());
	EXPECT_EQ(it.Index(), beforeScrub);
	for (OaU32 i = 0; i < 12; ++i) {
		ASSERT_TRUE(it.StepFrames(-5).IsOk()) << "repeat backward scrub " << i;
		EXPECT_EQ(it.Index(), beforeScrub - 5);
		ASSERT_TRUE(it.StepFrames(5).IsOk()) << "repeat forward scrub " << i;
		EXPECT_EQ(it.Index(), beforeScrub);
	}

	it.Reset();
	ASSERT_TRUE(it.StepForward().IsOk());
	EXPECT_EQ(it.Index(), 1);
	const OaUsize frameCount = it.FrameCount();
	for (OaUsize i = 1; i < frameCount; ++i) {
		ASSERT_TRUE(it.StepForward().IsOk()) << "loop step " << i;
	}
	EXPECT_EQ(it.Index(), static_cast<OaI64>(frameCount));
	ASSERT_TRUE(it.StepForward().IsOk());
	EXPECT_EQ(it.Index(), 1);
	EXPECT_FALSE(it.IsDone());
	it.Destroy();
}

TEST(OaVideoStream, FirstFrameMatchesFfmpegReference)
{
	if (not DatasetAvailable(kShibuyaH264)) {
		GTEST_SKIP() << "Shibuya MP4 dataset not present";
	}
	if (std::system("command -v ffmpeg >/dev/null 2>&1") != 0) {
		GTEST_SKIP() << "ffmpeg is not installed";
	}
	auto* engine = OaComputeEngine::GetGlobal();
	if (engine == nullptr
		or not OaVideoDecoder::IsCodecSupported(*engine, OaVideoCodec::H264)) {
		GTEST_SKIP() << "Vulkan Video H.264 decode not supported";
	}

	auto streamResult = OaVideoStream::OpenFile(kShibuyaH264);
	ASSERT_TRUE(streamResult.IsOk());
	auto profile = streamResult->GetVideoProfile();
	profile.MaxDpbSlots = 16;
	auto decoderResult = OaVideoDecoder::Create(*engine, profile);
	ASSERT_TRUE(decoderResult.IsOk()) << decoderResult.GetStatus().ToString();
	auto decoder = OaStdMove(*decoderResult);

	OaVideoPacket packet;
	ASSERT_TRUE(streamResult->ReadNextPacket(packet).IsOk());
	OaVideoConversionOptions options;
	options.PreferHardwareYCbCr = false;
	options.Filter = OaFilter::Nearest;
	OaVideoFrame frame;
	ASSERT_TRUE(decoder.DecodeFrameWithConversion(
		OaSpan<const OaU8>(packet.Data.Data(), packet.Data.Size()),
		options,
		frame).IsOk());
	auto rgba = OaFnVideo::ReadbackRgba(decoder, frame);
	ASSERT_TRUE(rgba.IsOk()) << rgba.GetStatus().ToString();

	const char* refPath = "/tmp/oa_shibuya_frame0_rgba.bin";
	const OaString command = OaString(
		"ffmpeg -v error -y -i \"") + kShibuyaH264
		+ "\" -frames:v 1 -f rawvideo -pix_fmt rgba " + refPath;
	ASSERT_EQ(std::system(command.CStr()), 0);
	auto reference = OaFileIo::ReadBinary(OaPath(refPath));
	std::remove(refPath);
	ASSERT_TRUE(reference.IsOk());
	ASSERT_EQ(reference->Size(), rgba->Size());

	OaU64 absoluteError = 0;
	OaU8 maxError = 0;
	for (OaUsize i = 0; i < rgba->Size(); ++i) {
		const OaU8 error = static_cast<OaU8>(std::abs(
			static_cast<int>((*rgba)[i]) - static_cast<int>((*reference)[i])));
		absoluteError += error;
		if (error > maxError) {
			maxError = error;
		}
	}
	const OaF64 meanAbsoluteError =
		static_cast<OaF64>(absoluteError) / static_cast<OaF64>(rgba->Size());
	EXPECT_LT(meanAbsoluteError, 3.0);
	EXPECT_LT(maxError, 32U);
}

TEST(OaVideoStream, FirstFrameNv12MatchesFfmpegReference)
{
	if (not DatasetAvailable(kShibuyaH264)) {
		GTEST_SKIP() << "Shibuya MP4 dataset not present";
	}
	if (std::system("command -v ffmpeg >/dev/null 2>&1") != 0) {
		GTEST_SKIP() << "ffmpeg is not installed";
	}
	auto* engine = OaComputeEngine::GetGlobal();
	if (engine == nullptr
		or not OaVideoDecoder::IsCodecSupported(*engine, OaVideoCodec::H264)) {
		GTEST_SKIP() << "Vulkan Video H.264 decode not supported";
	}

	auto streamResult = OaVideoStream::OpenFile(kShibuyaH264);
	ASSERT_TRUE(streamResult.IsOk());
	auto profile = streamResult->GetVideoProfile();
	profile.MaxDpbSlots = 16;
	auto decoderResult = OaVideoDecoder::Create(*engine, profile);
	ASSERT_TRUE(decoderResult.IsOk()) << decoderResult.GetStatus().ToString();
	auto decoder = OaStdMove(*decoderResult);

	OaVideoPacket packet;
	ASSERT_TRUE(streamResult->ReadNextPacket(packet).IsOk());
	OaVideoFrame frame;
	ASSERT_TRUE(decoder.DecodeFrame(
		OaSpan<const OaU8>(packet.Data.Data(), packet.Data.Size()),
		frame).IsOk());
	auto nv12 = OaFnVideo::ReadbackNv12(decoder, frame);
	ASSERT_TRUE(nv12.IsOk()) << nv12.GetStatus().ToString();

	const char* refPath = "/tmp/oa_shibuya_frame0_nv12.bin";
	const OaString command = OaString(
		"ffmpeg -v error -y -i \"") + kShibuyaH264
		+ "\" -frames:v 1 -f rawvideo -pix_fmt nv12 " + refPath;
	ASSERT_EQ(std::system(command.CStr()), 0);
	auto reference = OaFileIo::ReadBinary(OaPath(refPath));
	std::remove(refPath);
	ASSERT_TRUE(reference.IsOk());
	ASSERT_EQ(reference->Size(), nv12->Size());

	const OaUsize lumaBytes =
		static_cast<OaUsize>(profile.Width) * static_cast<OaUsize>(profile.Height);
	OaU64 lumaError = 0;
	OaU64 chromaError = 0;
	OaU64 decodedLumaSum = 0;
	OaU64 referenceLumaSum = 0;
	for (OaUsize i = 0; i < nv12->Size(); ++i) {
		const OaU64 error = static_cast<OaU64>(std::abs(
			static_cast<int>((*nv12)[i]) - static_cast<int>((*reference)[i])));
		if (i < lumaBytes) {
			lumaError += error;
			decodedLumaSum += (*nv12)[i];
			referenceLumaSum += (*reference)[i];
		} else {
			chromaError += error;
		}
	}
	const OaF64 lumaMae =
		static_cast<OaF64>(lumaError) / static_cast<OaF64>(lumaBytes);
	const OaF64 chromaMae =
		static_cast<OaF64>(chromaError)
		/ static_cast<OaF64>(nv12->Size() - lumaBytes);
	OaF64 bestShiftedLumaMae = lumaMae;
	OaI32 bestRowShift = 0;
	for (OaI32 rowShift = -16; rowShift <= 16; ++rowShift) {
		OaU64 shiftedError = 0;
		OaU64 shiftedPixels = 0;
		for (OaI32 y = 0; y < static_cast<OaI32>(profile.Height); ++y) {
			const OaI32 referenceY = y + rowShift;
			if (referenceY < 0 || referenceY >= static_cast<OaI32>(profile.Height)) {
				continue;
			}
			for (OaU32 x = 0; x < profile.Width; ++x) {
				const OaUsize decodedIndex =
					static_cast<OaUsize>(y) * profile.Width + x;
				const OaUsize referenceIndex =
					static_cast<OaUsize>(referenceY) * profile.Width + x;
				shiftedError += static_cast<OaU64>(std::abs(
					static_cast<int>((*nv12)[decodedIndex])
					- static_cast<int>((*reference)[referenceIndex])));
				++shiftedPixels;
			}
		}
		const OaF64 shiftedMae =
			static_cast<OaF64>(shiftedError) / static_cast<OaF64>(shiftedPixels);
		if (shiftedMae < bestShiftedLumaMae) {
			bestShiftedLumaMae = shiftedMae;
			bestRowShift = rowShift;
		}
	}
	EXPECT_LT(lumaMae, 3.0)
		<< "decodedMean="
		<< static_cast<OaF64>(decodedLumaSum) / static_cast<OaF64>(lumaBytes)
		<< " referenceMean="
		<< static_cast<OaF64>(referenceLumaSum) / static_cast<OaF64>(lumaBytes)
		<< " bestRowShift=" << bestRowShift
		<< " shiftedMae=" << bestShiftedLumaMae;
	EXPECT_LT(chromaMae, 3.0);
	decoder.Destroy();
}

TEST(OaVideoStream, FirstTwelveDisplayFramesMatchFfmpegReference)
{
	if (not DatasetAvailable(kShibuyaH264)) {
		GTEST_SKIP() << "Shibuya MP4 dataset not present";
	}
	if (std::system("command -v ffmpeg >/dev/null 2>&1") != 0) {
		GTEST_SKIP() << "ffmpeg is not installed";
	}
	auto* engine = OaComputeEngine::GetGlobal();
	if (engine == nullptr
		or not OaVideoDecoder::IsCodecSupported(*engine, OaVideoCodec::H264)) {
		GTEST_SKIP() << "Vulkan Video H.264 decode not supported";
	}

	constexpr OaUsize frameCount = 12;
	constexpr OaUsize width = 1920;
	constexpr OaUsize height = 1080;
	constexpr OaUsize frameBytes = width * height * 4;
	const char* refPath = "/tmp/oa_shibuya_first12_rgba.bin";
	const OaString command = OaString(
		"ffmpeg -v error -y -i \"") + kShibuyaH264
		+ "\" -frames:v 12 -f rawvideo -pix_fmt rgba " + refPath;
	ASSERT_EQ(std::system(command.CStr()), 0);
	auto reference = OaFileIo::ReadBinary(OaPath(refPath));
	std::remove(refPath);
	ASSERT_TRUE(reference.IsOk());
	ASSERT_EQ(reference->Size(), frameBytes * frameCount);

	OaItVideoConfig cfg;
	cfg.Path = kShibuyaH264;
	cfg.StartPlaying = false;
	cfg.Loop = false;
	cfg.PreferHardwareYCbCr = false;
	cfg.Filter = OaFilter::Nearest;
	auto itResult = OaItVideo::Create(*engine, cfg);
	ASSERT_TRUE(itResult.IsOk()) << itResult.GetStatus().ToString();
	OaItVideo it = OaStdMove(*itResult);

	for (OaUsize frameIndex = 0; frameIndex < frameCount; ++frameIndex) {
		if (frameIndex > 0) {
			ASSERT_TRUE(it.StepForward().IsOk()) << "frame " << frameIndex;
		}
		auto rgba = it.ReadbackCurrentRgba();
		ASSERT_TRUE(rgba.IsOk()) << "frame " << frameIndex << ": "
			<< rgba.GetStatus().ToString();
		ASSERT_EQ(rgba->Size(), frameBytes);

		OaU64 absoluteError = 0;
		OaU8 maxError = 0;
		const OaU8* expected = reference->Data() + frameIndex * frameBytes;
		for (OaUsize i = 0; i < frameBytes; ++i) {
			const OaU8 error = static_cast<OaU8>(std::abs(
				static_cast<int>((*rgba)[i]) - static_cast<int>(expected[i])));
			absoluteError += error;
			maxError = std::max(maxError, error);
		}
		const OaF64 meanAbsoluteError =
			static_cast<OaF64>(absoluteError) / static_cast<OaF64>(frameBytes);
		EXPECT_LT(meanAbsoluteError, 3.0)
			<< "display frame " << frameIndex
			<< " PTS=" << it.CurrentFrame().PresentationTimestamp
			<< " maxError=" << static_cast<OaU32>(maxError);
	}
	it.Destroy();
}

TEST(OaVideoStream, SustainedH264DisplayFramesMatchFfmpegReference)
{
	if (not DatasetAvailable(kShibuyaH264)) {
		GTEST_SKIP() << "Shibuya MP4 dataset not present";
	}
	if (std::system("command -v ffmpeg >/dev/null 2>&1") != 0) {
		GTEST_SKIP() << "ffmpeg is not installed";
	}
	auto* engine = OaComputeEngine::GetGlobal();
	if (engine == nullptr
		or not OaVideoDecoder::IsCodecSupported(*engine, OaVideoCodec::H264)) {
		GTEST_SKIP() << "Vulkan Video H.264 decode not supported";
	}

	constexpr OaUsize width = 1920;
	constexpr OaUsize height = 1080;
	constexpr OaUsize frameBytes = width * height * 4;
	constexpr OaU32 sampleFrames[] = {
		0, 1, 2, 3, 4, 5, 11, 12, 13, 29, 30, 31,
		59, 60, 61, 89, 90, 91, 119, 120, 121, 179, 239, 299,
	};
	constexpr OaUsize sampleCount = sizeof(sampleFrames) / sizeof(sampleFrames[0]);
	const char* refPath = "/tmp/oa_shibuya_h264_sustained_rgba.bin";
	OaString select = "select='";
	for (OaUsize i = 0; i < sampleCount; ++i) {
		if (i > 0) {
			select += "+";
		}
		select += "eq(n\\,";
		select += std::to_string(sampleFrames[i]);
		select += ")";
	}
	select += "'";
	const OaString command = OaString(
		"ffmpeg -v error -y -i \"") + kShibuyaH264
		+ "\" -vf \"" + select
		+ "\" -vsync 0 -f rawvideo -pix_fmt rgba " + refPath;
	ASSERT_EQ(std::system(command.CStr()), 0);
	auto reference = OaFileIo::ReadBinary(OaPath(refPath));
	std::remove(refPath);
	ASSERT_TRUE(reference.IsOk());
	ASSERT_EQ(reference->Size(), frameBytes * sampleCount);

	OaItVideoConfig cfg;
	cfg.Path = kShibuyaH264;
	cfg.StartPlaying = false;
	cfg.Loop = false;
	cfg.PreferHardwareYCbCr = false;
	cfg.Filter = OaFilter::Nearest;
	auto itResult = OaItVideo::Create(*engine, cfg);
	ASSERT_TRUE(itResult.IsOk()) << itResult.GetStatus().ToString();
	OaItVideo it = OaStdMove(*itResult);

	OaU64 priorPts = it.CurrentFrame().PresentationTimestamp;
	OaUsize sampleIndex = 0;
	for (OaU32 frameIndex = 0; frameIndex <= sampleFrames[sampleCount - 1]; ++frameIndex) {
		if (frameIndex > 0) {
			ASSERT_TRUE(it.StepForward().IsOk()) << "frame " << frameIndex;
			EXPECT_GT(it.CurrentFrame().PresentationTimestamp, priorPts)
				<< "display PTS repeated/regressed at frame " << frameIndex;
			priorPts = it.CurrentFrame().PresentationTimestamp;
		}
		if (sampleIndex >= sampleCount or frameIndex != sampleFrames[sampleIndex]) {
			continue;
		}

		auto rgba = it.ReadbackCurrentRgba();
		ASSERT_TRUE(rgba.IsOk()) << "frame " << frameIndex << ": "
			<< rgba.GetStatus().ToString();
		ASSERT_EQ(rgba->Size(), frameBytes);

		OaU64 absoluteError = 0;
		OaU8 maxError = 0;
		const OaU8* expected = reference->Data() + sampleIndex * frameBytes;
		for (OaUsize i = 0; i < frameBytes; ++i) {
			const OaU8 error = static_cast<OaU8>(std::abs(
				static_cast<int>((*rgba)[i]) - static_cast<int>(expected[i])));
			absoluteError += error;
			maxError = std::max(maxError, error);
		}
		const OaF64 meanAbsoluteError =
			static_cast<OaF64>(absoluteError) / static_cast<OaF64>(frameBytes);
		EXPECT_LT(meanAbsoluteError, 3.0)
			<< "display frame " << frameIndex
			<< " PTS=" << it.CurrentFrame().PresentationTimestamp
			<< " maxError=" << static_cast<OaU32>(maxError);
		++sampleIndex;
	}
	EXPECT_EQ(sampleIndex, sampleCount);
	it.Destroy();
}

void ExpectCodecIteratorMatchesFirstTwelveFfmpegFrames(
	const char* InName,
	const char* InPath,
	OaVideoCodec InCodec)
{
	auto* engine = OaComputeEngine::GetGlobal();
	if (engine == nullptr) {
		GTEST_SKIP() << "No Vulkan compute engine available";
	}
	if (not DatasetAvailable(InPath)
		or not OaVideoDecoder::IsCodecSupported(*engine, InCodec)) {
		GTEST_SKIP() << InName << " fixture or Vulkan Video support unavailable";
	}

	auto streamResult = OaVideoStream::OpenFile(InPath);
	ASSERT_TRUE(streamResult.IsOk()) << InName;
	const OaUsize frameBytes =
		static_cast<OaUsize>(streamResult->GetInfo().Width)
		* streamResult->GetInfo().Height * 4U;
	const OaString refPath =
		OaString("/tmp/oa_shibuya_") + InName + "_first12_rgba.bin";
	const OaString command = OaString(
		"ffmpeg -v error -y -i \"") + InPath
		+ "\" -vf \"select='between(n\\,0\\,11)'\""
		+ " -vsync 0 -f rawvideo -pix_fmt rgba " + refPath;
	ASSERT_EQ(std::system(command.CStr()), 0) << InName;
	auto reference = OaFileIo::ReadBinary(OaPath(refPath));
	std::remove(refPath.CStr());
	ASSERT_TRUE(reference.IsOk()) << InName;
	ASSERT_EQ(reference->Size(), frameBytes * 12U) << InName;

	OaItVideoConfig cfg;
	cfg.Path = InPath;
	cfg.StartPlaying = false;
	cfg.Loop = false;
	cfg.PreferHardwareYCbCr = false;
	cfg.Filter = OaFilter::Nearest;
	auto itResult = OaItVideo::Create(*engine, cfg);
	ASSERT_TRUE(itResult.IsOk()) << InName << ": "
		<< itResult.GetStatus().ToString();
	OaItVideo it = OaStdMove(*itResult);

	OaU64 priorPts = it.CurrentFrame().PresentationTimestamp;
	for (OaU32 frameIndex = 0; frameIndex < 12U; ++frameIndex) {
		if (frameIndex > 0) {
			ASSERT_TRUE(it.StepForward().IsOk()) << InName
				<< " frame " << frameIndex;
			EXPECT_GT(it.CurrentFrame().PresentationTimestamp, priorPts)
				<< InName << " frame " << frameIndex;
			priorPts = it.CurrentFrame().PresentationTimestamp;
		}
		auto rgba = it.ReadbackCurrentRgba();
		ASSERT_TRUE(rgba.IsOk()) << InName << " frame " << frameIndex;
		ASSERT_EQ(rgba->Size(), frameBytes) << InName;

		OaU64 absoluteError = 0;
		const OaU8* expected = reference->Data() + frameIndex * frameBytes;
		for (OaUsize i = 0; i < frameBytes; ++i) {
			absoluteError += static_cast<OaU64>(std::abs(
				static_cast<int>((*rgba)[i])
				- static_cast<int>(expected[i])));
		}
		const OaF64 meanAbsoluteError =
			static_cast<OaF64>(absoluteError)
			/ static_cast<OaF64>(frameBytes);
		EXPECT_LT(meanAbsoluteError, 3.0)
			<< InName << " frame " << frameIndex;
	}
	it.Destroy();
}

TEST(OaVideoStream, H264IteratorMatchesFirstTwelveFfmpegFrames)
{
	ExpectCodecIteratorMatchesFirstTwelveFfmpegFrames(
		"h264", kShibuyaH264, OaVideoCodec::H264);
}

TEST(OaVideoStream, H265IteratorMatchesFirstTwelveFfmpegFrames)
{
	ExpectCodecIteratorMatchesFirstTwelveFfmpegFrames(
		"h265", kShibuyaH265, OaVideoCodec::H265);
}

TEST(OaVideoStream, H265IteratorMatchesFfmpegAcrossEntireVideo)
{
	auto* engine = OaComputeEngine::GetGlobal();
	if (engine == nullptr) {
		GTEST_SKIP() << "No Vulkan compute engine available";
	}
	if (not DatasetAvailable(kShibuyaH265) ||
		not OaVideoDecoder::IsCodecSupported(*engine, OaVideoCodec::H265)) {
		GTEST_SKIP() << "H.265 fixture or Vulkan Video support unavailable";
	}

	constexpr OaU32 sampleFrames[] = {
		0u, 255u, 256u, 257u, 470u, 474u, 480u, 720u, 899u,
		1200u, 1500u, 1606u};
	constexpr OaUsize sampleCount =
		sizeof(sampleFrames) / sizeof(sampleFrames[0]);
	const OaString refPath = "/tmp/oa_shibuya_h265_sustained_rgba.bin";
	const OaString command = OaString(
		"ffmpeg -v error -y -i \"") + kShibuyaH265
		+ "\" -vf \"select='"
		+ "eq(n\\,0)+eq(n\\,255)+eq(n\\,256)+eq(n\\,257)+"
		+ "eq(n\\,470)+eq(n\\,474)+eq(n\\,480)+eq(n\\,720)+eq(n\\,899)+"
		+ "eq(n\\,1200)+eq(n\\,1500)+eq(n\\,1606)'\""
		+ " -vsync 0 -f rawvideo -pix_fmt rgba " + refPath;
	ASSERT_EQ(std::system(command.CStr()), 0);

	auto streamResult = OaVideoStream::OpenFile(kShibuyaH265);
	ASSERT_TRUE(streamResult.IsOk());
	const OaUsize frameBytes =
		static_cast<OaUsize>(streamResult->GetInfo().Width) *
		streamResult->GetInfo().Height * 4u;
	auto reference = OaFileIo::ReadBinary(OaPath(refPath));
	std::remove(refPath.CStr());
	ASSERT_TRUE(reference.IsOk());
	ASSERT_EQ(reference->Size(), frameBytes * sampleCount);

	OaItVideoConfig cfg;
	cfg.Path = kShibuyaH265;
	cfg.StartPlaying = false;
	cfg.Loop = false;
	cfg.PreferHardwareYCbCr = false;
	cfg.Filter = OaFilter::Nearest;
	auto itResult = OaItVideo::Create(*engine, cfg);
	ASSERT_TRUE(itResult.IsOk()) << itResult.GetStatus().ToString();
	OaItVideo it = OaStdMove(*itResult);

	OaUsize sampleIndex = 0;
	for (OaU32 frameIndex = 0; frameIndex <= sampleFrames[sampleCount - 1u]; ++frameIndex) {
		if (frameIndex > 0) {
			const OaStatus stepStatus = it.StepForward();
			ASSERT_TRUE(stepStatus.IsOk())
				<< "display frame " << frameIndex << ": "
				<< stepStatus.ToString();
		}
		if (frameIndex != sampleFrames[sampleIndex]) {
			continue;
		}
		auto rgba = it.ReadbackCurrentRgba();
		ASSERT_TRUE(rgba.IsOk()) << "display frame " << frameIndex;
		ASSERT_EQ(rgba->Size(), frameBytes);

		OaU64 absoluteError = 0;
		const OaU8* expected =
			reference->Data() + sampleIndex * frameBytes;
		for (OaUsize i = 0; i < frameBytes; ++i) {
			absoluteError += static_cast<OaU64>(std::abs(
				static_cast<int>((*rgba)[i]) -
				static_cast<int>(expected[i])));
		}
		const OaF64 meanAbsoluteError =
			static_cast<OaF64>(absoluteError) /
			static_cast<OaF64>(frameBytes);
		EXPECT_LT(meanAbsoluteError, 3.0)
			<< "display frame " << frameIndex;
		++sampleIndex;
		if (sampleIndex == sampleCount) {
			break;
		}
	}
	EXPECT_EQ(sampleIndex, sampleCount);
	it.Destroy();
}

TEST(OaVideoStream, Av1IteratorMatchesFirstTwelveFfmpegFrames)
{
	ExpectCodecIteratorMatchesFirstTwelveFfmpegFrames(
		"av1", kShibuyaAv1, OaVideoCodec::AV1);
}

TEST(OaVideoStream, Vp9IteratorMatchesFirstTwelveFfmpegFrames)
{
	ExpectCodecIteratorMatchesFirstTwelveFfmpegFrames(
		"vp9", kShibuyaVp9, OaVideoCodec::VP9);
}

TEST(OaVideoStream, SustainedH264StepLatencyStaysBelowFrameBudget)
{
	if (not DatasetAvailable(kShibuyaH264)) {
		GTEST_SKIP() << "Shibuya MP4 dataset not present";
	}
	auto* engine = OaComputeEngine::GetGlobal();
	if (engine == nullptr
		or not OaVideoDecoder::IsCodecSupported(*engine, OaVideoCodec::H264)) {
		GTEST_SKIP() << "Vulkan Video H.264 decode not supported";
	}

	OaItVideoConfig cfg;
	cfg.Path = kShibuyaH264;
	cfg.StartPlaying = false;
	cfg.Loop = false;
	cfg.PreferHardwareYCbCr = false;
	auto itResult = OaItVideo::Create(*engine, cfg);
	ASSERT_TRUE(itResult.IsOk()) << itResult.GetStatus().ToString();
	OaItVideo it = OaStdMove(*itResult);

	for (OaU32 i = 0; i < 12U; ++i) {
		ASSERT_TRUE(it.StepForward().IsOk());
	}
	OaF64 maxStepMs = 0.0;
	OaF64 totalStepMs = 0.0;
	constexpr OaU32 sampleCount = 180;
	for (OaU32 i = 0; i < sampleCount; ++i) {
		const auto start = std::chrono::steady_clock::now();
		ASSERT_TRUE(it.StepForward().IsOk()) << "step " << i;
		const auto end = std::chrono::steady_clock::now();
		const OaF64 stepMs =
			std::chrono::duration<OaF64, std::milli>(end - start).count();
		maxStepMs = std::max(maxStepMs, stepMs);
		totalStepMs += stepMs;
	}
	EXPECT_LT(totalStepMs / sampleCount, it.FrameIntervalMs() * 0.5)
		<< "average decode/convert step exceeds half of the frame budget";
	EXPECT_LT(maxStepMs, it.FrameIntervalMs())
		<< "a decode/convert step exceeded the full frame budget";
	it.Destroy();
}


TEST(OaFnVideoNal, ParseAndEmitRoundTrip)
{
	// 4-byte start code + tiny SPS-like payload, twice.
	const OaU8 kSps[] = { 0x67, 0x42, 0xC0, 0x1E };
	const OaU8 kPps[] = { 0x68, 0xCE, 0x38, 0x80 };

	OaVec<OaU8> stream;
	const OaU8 startCode[4] = { 0, 0, 0, 1 };
	for (auto byte : startCode) { stream.PushBack(byte); }
	for (auto byte : kSps)      { stream.PushBack(byte); }
	for (auto byte : startCode) { stream.PushBack(byte); }
	for (auto byte : kPps)      { stream.PushBack(byte); }

	auto units = OaFnVideo::ParseNalAnnexB(OaSpan<const OaU8>(stream.Data(), stream.Size()));
	ASSERT_EQ(units.Size(), 2U);
	EXPECT_EQ(units[0].Type, 7U);  // SPS
	EXPECT_EQ(units[1].Type, 8U);  // PPS

	auto roundTrip = OaFnVideo::EmitNalAnnexB(OaSpan<const OaNalUnit>(units.Data(), units.Size()));
	ASSERT_EQ(roundTrip.Size(), stream.Size());
	for (OaUsize i = 0; i < stream.Size(); ++i) {
		EXPECT_EQ(roundTrip[i], stream[i]) << "byte " << i << " differs";
	}

	auto extractedSps = OaFnVideo::ExtractSps(OaSpan<const OaU8>(stream.Data(), stream.Size()));
	auto extractedPps = OaFnVideo::ExtractPps(OaSpan<const OaU8>(stream.Data(), stream.Size()));
	ASSERT_EQ(extractedSps.Size(), sizeof(kSps));
	ASSERT_EQ(extractedPps.Size(), sizeof(kPps));
}
