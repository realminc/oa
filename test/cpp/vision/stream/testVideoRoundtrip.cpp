// OA Vision — Video Roundtrip test (encode → mux → demux → readback)
// End-to-end proof that oa::VideoEncoder + oa::VideoMuxer produce a valid MP4
// that oa::VideoDemuxer can open and read back the same packets from.

#include "../../oaTest.h"
#include "../videoTestSupport.h"

#include <oa/runtime/engine.h>
#include <oa/runtime/engine/submissionAccess.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/stream.h>
#include <oa/ml/fnMatrix.h>
#include <oa/vision/cameraCapture.h>
#include <oa/vision/screenCapture.h>
#include <oa/vision/videoEncoder.h>
#include <oa/vision/videoMuxer.h>
#include <oa/vision/videoRecorder.h>
#include <oa/runtime/externalMemory.h>
#include <oa/vision/videoDemuxer.h>
#include <oa/vision/fnVideo.h>
#include <oa/ui/image.h>

#include <oa/runtime/textureAccess.h>
#include <oa/vision/video/encoder/videoEncoderInternal.h>

#include <cstdio>
#include <cmath>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

namespace {

// Synthetic gradient + moving rectangle, identical to TutorialEncodeBasic's
// frame producer. Doesn't depend on any dataset.
void paintFrame(std::vector<oa::U8>& inOut, oa::I32 inW, oa::I32 inH, oa::U32 inFrame, oa::U32 inTotal) {
	const oa::F32 t = static_cast<oa::F32>(inFrame) / static_cast<oa::F32>(inTotal);
	const oa::I32 rectW = inW / 6;
	const oa::I32 rectH = inH / 6;
	const oa::I32 rectX = static_cast<oa::I32>(t * static_cast<oa::F32>(inW - rectW));
	const oa::I32 rectY = (inH - rectH) / 2;
	for (oa::I32 y = 0; y < inH; ++y) {
		for (oa::I32 x = 0; x < inW; ++x) {
			oa::U8 r = static_cast<oa::U8>((x * 255) / inW);
			oa::U8 g = static_cast<oa::U8>((y * 255) / inH);
			oa::U8 b = static_cast<oa::U8>(((inFrame * 4U) & 0xFFU));
			if (x >= rectX && x < rectX + rectW
				&& y >= rectY && y < rectY + rectH) {
				r = 240; g = 80; b = 80;
			}
			const oa::I64 i = (static_cast<oa::I64>(y) * inW + x) * 4;
			inOut[static_cast<oa::Usize>(i + 0)] = r;
			inOut[static_cast<oa::Usize>(i + 1)] = g;
			inOut[static_cast<oa::Usize>(i + 2)] = b;
			inOut[static_cast<oa::Usize>(i + 3)] = 255U;
		}
	}
}

std::string testOutputPath(const char* inStem) {
#if defined(_WIN32)
	const auto processId = static_cast<unsigned long>(_getpid());
#else
	const auto processId = static_cast<unsigned long>(getpid());
#endif
	const auto filename = std::string(inStem) + "-"
		+ std::to_string(processId) + ".mp4";
	return (std::filesystem::temp_directory_path() / filename).string();
}

} // namespace

TEST(CaptureLifecycle, DefaultCloseIsIdempotent) {
	oa::CameraCapture camera;
	oa::ScreenCapture screen;
	EXPECT_TRUE(camera.close().isOk());
	EXPECT_TRUE(camera.close().isOk());
	EXPECT_TRUE(screen.close().isOk());
	EXPECT_TRUE(screen.close().isOk());
	oa::VideoRecorder recorder;
	EXPECT_TRUE(recorder.close().isOk());
	EXPECT_TRUE(recorder.close().isOk());
	oa::VideoMuxer muxer;
	EXPECT_TRUE(muxer.close().isOk());
	EXPECT_TRUE(muxer.close().isOk());
}

TEST(CaptureLifecycle, AbandonedCameraRetiresAtEngineClose) {
	const char* enabled = std::getenv("OA_TEST_CAMERA_CAPTURE");
	if (enabled == nullptr or enabled[0] != '1') {
		GTEST_SKIP() << "set OA_TEST_CAMERA_CAPTURE=1 for the physical-camera lifecycle gate";
	}
	auto config = testEngineConfig(oa::Precision::FP32);
	config.selectForThread = false;
	config.preloadEmbeddedPipelines = false;
	config.enablePipelineCache = false;
	auto engineResult = oa::Engine::create(config);
	ASSERT_TRUE(engineResult.isOk()) << engineResult.getStatus().toString();
	auto engine = oa::move(*engineResult);
	oa::String skipReason;
	{
		oa::CameraCapture capture;
		oa::CameraCaptureConfig captureConfig;
		captureConfig.width = 640;
		captureConfig.height = 480;
		if (const char* device = std::getenv("OA_TEST_CAMERA_DEVICE")) {
			captureConfig.devicePath = device;
		}
		auto captureResult = oa::CameraCapture::open(*engine, captureConfig);
		if (not captureResult.isOk()) {
			skipReason = captureResult.getStatus().toString();
		} else {
			capture = oa::move(*captureResult);
			oa::VideoFrame frame;
			const auto deadline = std::chrono::steady_clock::now()
				+ std::chrono::seconds(5);
			while (not capture.pollFrame(frame)
				and std::chrono::steady_clock::now() < deadline) {
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
			}
			if (frame.width == 0U or frame.height == 0U) {
				skipReason = "camera produced no frame within five seconds";
			} else {
				auto* stream = oa::EngineSubmissionAccess::acquireStream(*engine);
				ASSERT_NE(stream, nullptr);
				ASSERT_TRUE(stream->begin(oa::EngineDeviceAccess::get(*engine)).isOk());
				stream->recordBufferBarrier();
				ASSERT_TRUE(stream->submit(*engine).isOk());
				const auto consumed = stream->completion(oa::EngineDeviceAccess::get(*engine));
				ASSERT_TRUE(consumed.isValid());
				capture.release(frame, consumed);
			}
		}
		// No Close: producer state and the exact frame-consumer token move
		// to engine-owned retirement without waiting in the capture destructor.
	}
	ASSERT_TRUE(engine->close().isOk());
	if (not skipReason.empty()) GTEST_SKIP() << skipReason.cStr();
}

TEST(CaptureLifecycle, AbandonedScreenCaptureRetiresAtEngineClose) {
	const char* enabled = std::getenv("OA_TEST_SCREEN_CAPTURE");
	if (enabled == nullptr or enabled[0] != '1') {
		GTEST_SKIP() << "set OA_TEST_SCREEN_CAPTURE=1 for the interactive portal lifecycle gate";
	}
	auto config = testEngineConfig(oa::Precision::FP32);
	config.selectForThread = false;
	config.preloadEmbeddedPipelines = false;
	config.enablePipelineCache = false;
	auto engineResult = oa::Engine::create(config);
	ASSERT_TRUE(engineResult.isOk()) << engineResult.getStatus().toString();
	auto engine = oa::move(*engineResult);
	oa::String skipReason;
	{
		auto captureResult = oa::ScreenCapture::open(*engine);
		if (not captureResult.isOk()) {
			skipReason = captureResult.getStatus().toString();
		} else {
			auto capture = oa::move(*captureResult);
			oa::VideoFrame frame;
			const auto deadline = std::chrono::steady_clock::now()
				+ std::chrono::seconds(5);
			while (not capture.poll(frame)
				and std::chrono::steady_clock::now() < deadline) {
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
			}
			if (frame.width == 0U or frame.height == 0U) {
				skipReason = "screen capture produced no frame within five seconds";
			} else {
				auto* stream = oa::EngineSubmissionAccess::acquireStream(*engine);
				ASSERT_NE(stream, nullptr);
				ASSERT_TRUE(stream->begin(oa::EngineDeviceAccess::get(*engine)).isOk());
				stream->recordBufferBarrier();
				ASSERT_TRUE(stream->submit(*engine).isOk());
				const auto consumed = stream->completion(oa::EngineDeviceAccess::get(*engine));
				ASSERT_TRUE(consumed.isValid());
				capture.release(frame, consumed);
			}
			// No Close: the callback loop is asked to stop without a join;
			// engine close joins it and completes exact consumer dependencies.
		}
	}
	ASSERT_TRUE(engine->close().isOk());
	if (not skipReason.empty()) GTEST_SKIP() << skipReason.cStr();
}

TEST(VideoRoundtrip, DmaBufImporterRejectsIncompleteDescriptions) {
	auto* engine = testEnginePtr();
	if (engine == nullptr) GTEST_SKIP() << "No vulkan engine available";
	oa::DmaBufImageDesc description;
	auto result = oa::ImportedDmaBufImage::import(*engine, description);
	ASSERT_FALSE(result.isOk());
	EXPECT_EQ(result.getStatus().getCode(), oa::StatusCode::InvalidArgument);
}


TEST(VideoRoundtrip, AbandonedSubmittedEncoderRetiresAtEngineClose) {
	auto config = testEngineConfig(oa::Precision::FP32);
	config.selectForThread = false;
	config.preloadEmbeddedPipelines = false;
	config.enablePipelineCache = false;
	auto engineResult = oa::Engine::create(config);
	ASSERT_TRUE(engineResult.isOk()) << engineResult.getStatus().toString();
	auto engine = oa::move(*engineResult);
	if (not testVideoEncodeSupported(*engine, oa::VideoCodec::H264)) {
		ASSERT_TRUE(engine->close().isOk());
		GTEST_SKIP() << "H.264 vulkan Video encode is not supported on selected device";
	}

	constexpr oa::U32 width = 320U;
	constexpr oa::U32 height = 192U;
	std::vector<oa::U8> rgba(static_cast<oa::Usize>(width) * height * 4U, 127U);
	auto textureResult = oa::FnTexture::fromPixels(
		*engine, oa::Span<const oa::U8>(rgba.data(), rgba.size()), width, height);
	ASSERT_TRUE(textureResult.isOk()) << textureResult.getStatus().toString();
	auto texture = oa::move(*textureResult);
	{
		oa::VideoEncodeProfile profile;
		profile.width = width;
		profile.height = height;
		profile.asyncDepth = 2U;
		auto encoderResult = oa::VideoEncoder::create(*engine, profile);
		ASSERT_TRUE(encoderResult.isOk()) << encoderResult.getStatus().toString();
		auto encoder = oa::move(*encoderResult);
		oa::Vector<oa::EncodedVideoPacket> ready;
		ASSERT_TRUE(oa::VideoEncoderAccess::submitRgba(
			encoder, *oa::TextureAccess::buffer(texture),
			width, height, 0U, ready).isOk());
		// No Close: the pending video fence, conversion ticket, and
		// complete encoder session must transfer to engine retirement.
	}

	ASSERT_TRUE(engine->close().isOk());
}


TEST(VideoRoundtrip, AbandonedRecorderRetiresSubmittedEncoderAtEngineClose) {
	auto config = testEngineConfig(oa::Precision::FP32);
	config.selectForThread = false;
	config.preloadEmbeddedPipelines = false;
	config.enablePipelineCache = false;
	auto engineResult = oa::Engine::create(config);
	ASSERT_TRUE(engineResult.isOk()) << engineResult.getStatus().toString();
	auto engine = oa::move(*engineResult);
	if (not testVideoEncodeSupported(*engine, oa::VideoCodec::H264)) {
		ASSERT_TRUE(engine->close().isOk());
		GTEST_SKIP() << "H.264 vulkan Video encode is not supported on selected device";
	}

	constexpr oa::U32 width = 320U;
	constexpr oa::U32 height = 192U;
	const std::string pathStorage = testOutputPath("oa_abandoned_recorder");
	const char* path = pathStorage.c_str();
	std::vector<oa::U8> rgba(static_cast<oa::Usize>(width) * height * 4U, 63U);
	auto textureResult = oa::FnTexture::fromPixels(
		*engine, oa::Span<const oa::U8>(rgba.data(), rgba.size()), width, height);
	ASSERT_TRUE(textureResult.isOk()) << textureResult.getStatus().toString();
	auto texture = oa::move(*textureResult);
	{
		oa::VideoRecorderConfig recorderConfig;
		recorderConfig.outputPath = path;
		recorderConfig.encode.width = width;
		recorderConfig.encode.height = height;
		recorderConfig.encode.asyncDepth = 2U;
		auto recorderResult = oa::VideoRecorder::create(*engine, recorderConfig);
		ASSERT_TRUE(recorderResult.isOk()) << recorderResult.getStatus().toString();
		auto recorder = oa::move(*recorderResult);
		ASSERT_TRUE(recorder.writeRgba(
			*oa::TextureAccess::buffer(texture), width, height, 0U).isOk());
		// No finalize/Close: close the host file, but leave submitted GPU work
		// to the encoder's engine-owned retirement path.
	}

	std::remove(path);
	ASSERT_TRUE(engine->close().isOk());
}


TEST(VideoRoundtrip, TranscoderUsesGpuDecodeConvertEncodePath) {
	auto* engine = testEnginePtr();
	if (engine == nullptr) GTEST_SKIP() << "No vulkan engine available";
	if (not testVideoDecodeSupported(*engine, oa::VideoCodec::H264)
		or not testVideoEncodeSupported(*engine, oa::VideoCodec::H264)) {
		GTEST_SKIP() << "H.264 decode+encode are not both supported";
	}
	const oa::String sourcePath = oa::Paths::data(
		"video/shibuya_crossing_1080p30_h264.mp4").string();
	auto streamResult = oa::VideoDemuxer::open(sourcePath);
	if (not streamResult.isOk()) {
		GTEST_SKIP() << "Shibuya H.264 source is unavailable";
	}
	oa::VideoDemuxer stream = oa::move(*streamResult);
	oa::VideoPacket packet{};
	ASSERT_TRUE(stream.readNextPacket(packet).isOk());

	oa::VideoProfile decode = stream.getVideoProfile();
	decode.maxDpbSlots = 16U;
	oa::VideoEncodeProfile encode;
	encode.codec = oa::VideoCodec::H264;
	encode.width = decode.width;
	encode.height = decode.height;
	encode.frameRate = stream.getInfo().frameRate;
	encode.asyncDepth = 1U;
	auto transcoderResult = oa::VideoTranscoder::create(*engine, decode, encode);
	ASSERT_TRUE(transcoderResult.isOk())
		<< transcoderResult.getStatus().toString();
	oa::VideoTranscoder transcoder = oa::move(*transcoderResult);
	oa::EncodedVideoPacket output{};
	ASSERT_TRUE(transcoder.transcodeFrame(
		oa::Span<const oa::U8>(packet.data.data(), packet.data.size()), output).isOk());
	EXPECT_GT(output.frameSize, 4U);
	EXPECT_EQ(output.frameSize, output.bitstream.size());
	EXPECT_TRUE(output.isKeyframe);
	EXPECT_EQ(output.bitstream[0], 0U);
	EXPECT_EQ(output.bitstream[1], 0U);
	EXPECT_TRUE(transcoder.close().isOk());
	EXPECT_TRUE(transcoder.close().isOk());
}


TEST(VideoRoundtrip, RecorderComposesEncodeAndMux)
{
	auto* engine = testEnginePtr();
	if (engine == nullptr) GTEST_SKIP() << "No vulkan engine available";
	if (not testVideoEncodeSupported(*engine, oa::VideoCodec::H264)) {
		GTEST_SKIP() << "VK_KHR_video_encode_h264 not supported";
	}

	constexpr oa::I32 W = 320;
	constexpr oa::I32 H = 192;
	constexpr oa::U32 kFrames = 6;
	const std::string pathStorage = testOutputPath("oa_recorder_roundtrip");
	const char* path = pathStorage.c_str();

	oa::VideoRecorderConfig cfg;
	cfg.outputPath = path;
	cfg.encode.codec = oa::VideoCodec::H264;
	cfg.encode.width = W;
	cfg.encode.height = H;
	cfg.encode.frameRate = 30;
	cfg.encode.gopSize = 3;
	cfg.audioEnabled = true;
	cfg.audio.sampleRate = 48'000U;
	cfg.audio.channelCount = 2U;
	auto recorderResult = oa::VideoRecorder::create(*engine, cfg);
	ASSERT_TRUE(recorderResult.isOk()) << recorderResult.getStatus().toString();
	oa::VideoRecorder recorder = oa::move(*recorderResult);

	std::vector<oa::U8> rgba(static_cast<oa::Usize>(W) * H * 4U);
	constexpr oa::U32 kAudioFramesPerVideoFrame = 48'000U / 30U;
	std::vector<oa::F32> audio(kAudioFramesPerVideoFrame * 2U);
	for (oa::U32 frame = 0; frame < kFrames; ++frame) {
		for (oa::U32 i = 0U; i < kAudioFramesPerVideoFrame; ++i) {
			const oa::U64 absoluteFrame = static_cast<oa::U64>(frame) * kAudioFramesPerVideoFrame + i;
			const oa::F32 sample = 0.2F * std::sin(
				2.0F * 3.14159265358979323846F * 440.0F
				* static_cast<oa::F32>(absoluteFrame) / 48'000.0F);
			audio[i * 2U] = sample;
			audio[i * 2U + 1U] = sample;
		}
		const oa::U64 pts = static_cast<oa::U64>(frame) * 1'000'000ULL / 30ULL;
		// The first chunk intentionally arrives before the first video frame;
		// the recorder must establish one common monotonic epoch and retain it.
		ASSERT_TRUE(recorder.writeAudio(
			oa::Span<const oa::F32>(audio.data(), audio.size()), 48'000U, 2U, pts).isOk());
		paintFrame(rgba, W, H, frame, kFrames);
		auto textureResult = oa::FnTexture::fromPixels(
			*engine, oa::Span<const oa::U8>(rgba.data(), rgba.size()), W, H);
		ASSERT_TRUE(textureResult.isOk());
		oa::Texture texture = oa::move(*textureResult);
		auto status = recorder.writeRgba(
			*oa::TextureAccess::buffer(texture), W, H, pts);
		ASSERT_TRUE(status.isOk()) << status.toString();
	}
	EXPECT_EQ(recorder.getFrameCount(), kFrames);
	ASSERT_TRUE(recorder.finalize().isOk());
	ASSERT_TRUE(recorder.close().isOk());

	auto streamResult = oa::VideoDemuxer::open(path);
	ASSERT_TRUE(streamResult.isOk()) << streamResult.getStatus().toString();
	oa::VideoDemuxer stream = oa::move(*streamResult);
	EXPECT_EQ(stream.getInfo().codec, oa::VideoCodec::H264);
	EXPECT_EQ(stream.getInfo().width, static_cast<oa::U32>(W));
	EXPECT_EQ(stream.getInfo().height, static_cast<oa::U32>(H));
	oa::U32 packets = 0;
	for (; packets < kFrames; ++packets) {
		oa::VideoPacket packet = {};
		ASSERT_TRUE(stream.readNextPacket(packet).isOk());
		ASSERT_GT(packet.data.size(), 4U);
	}
	EXPECT_EQ(packets, kFrames);
	if (std::system("command -v ffmpeg >/dev/null 2>&1") == 0) {
		const std::string command = std::string("ffmpeg -xerror -v error -y -i \"")
			+ path
			+ "\" -map 0:v:0 -f null /dev/null -map 0:a:0 -f null /dev/null >/dev/null 2>&1";
		EXPECT_EQ(std::system(command.c_str()), 0)
			<< "Independent FFmpeg decoder rejected OA's A/V MP4";
	}
	if (std::system("command -v ffprobe >/dev/null 2>&1") == 0) {
		const std::string command = std::string(
			"ffprobe -v error -select_streams a:0 -show_entries stream=codec_name,sample_rate,channels ")
			+ "-of csv=p=0 \"" + path + "\" | grep -qx 'pcm_s16le,48000,2'";
		EXPECT_EQ(std::system(command.c_str()), 0)
			<< "OA recording does not expose the expected native PcmS16 stereo track";
		const std::string syncCommand = std::string(
			"ffprobe -v error -show_entries stream=index,codec_type,start_time -of csv=p=0 \"")
			+ path
			+ "\" | awk -F, '$2==\"video\"{v=$3} $2==\"audio\"{a=$3} "
				"END{d=v-a; if(d<0)d=-d; exit !(v!=\"\" && a!=\"\" && d<0.03)}'";
		EXPECT_EQ(std::system(syncCommand.c_str()), 0)
			<< "Audio and video track starts differ by 30 ms or more";
	}
	std::remove(path);
}


TEST(VideoRoundtrip, AdvertisedRateControlModesEncode)
{
	auto* engine = testEnginePtr();
	if (engine == nullptr) GTEST_SKIP() << "No vulkan engine available";
	auto capsResult = oa::VideoEncoder::queryEncodeCapabilities(*engine, oa::VideoCodec::H264);
	if (not capsResult.isOk()) GTEST_SKIP() << capsResult.getStatus().toString();

	const struct {
		oa::VideoRateControl rateControl;
		VkVideoEncodeRateControlModeFlagBitsKHR vulkan;
		const char* name;
	} modes[] = {
		{oa::VideoRateControl::Cbr, VK_VIDEO_ENCODE_RATE_CONTROL_MODE_CBR_BIT_KHR, "CBR"},
		{oa::VideoRateControl::Vbr, VK_VIDEO_ENCODE_RATE_CONTROL_MODE_VBR_BIT_KHR, "VBR"},
	};

	constexpr oa::U32 width = 320U;
	constexpr oa::U32 height = 192U;
	std::vector<oa::U8> rgba(static_cast<oa::Usize>(width) * height * 4U);
	oa::U32 testedModes = 0U;
	for (const auto& mode : modes) {
		if (((*capsResult).rateControlModes & mode.vulkan) == 0U) continue;
		++testedModes;
		oa::VideoEncodeProfile profile;
		profile.codec = oa::VideoCodec::H264;
		profile.width = width;
		profile.height = height;
		profile.frameRate = 30U;
		profile.gopSize = 3U;
		profile.rateControl = mode.rateControl;
		profile.bitrate = 1'000'000U;
		profile.maxBitrate = mode.rateControl == oa::VideoRateControl::Vbr ? 1'500'000U : 0U;
		auto encoderResult = oa::VideoEncoder::create(*engine, profile);
		ASSERT_TRUE(encoderResult.isOk()) << mode.name << ": "
			<< encoderResult.getStatus().toString();
		oa::VideoEncoder encoder = oa::move(*encoderResult);
		for (oa::U32 frameIndex = 0; frameIndex < 3U; ++frameIndex) {
			paintFrame(rgba, width, height, frameIndex, 3U);
			auto textureResult = oa::FnTexture::fromPixels(
				*engine, oa::Span<const oa::U8>(rgba.data(), rgba.size()), width, height);
			ASSERT_TRUE(textureResult.isOk());
			oa::Texture texture = oa::move(*textureResult);
			ASSERT_TRUE(oa::VideoEncoderAccess::uploadInputRgba(
				encoder, *oa::TextureAccess::buffer(texture), width, height).isOk());
			oa::EncodedVideoPacket encoded;
			auto status = oa::VideoEncoderAccess::encodeFrame(
				encoder, VK_NULL_HANDLE,
				frameIndex * (1'000'000ULL / 30ULL), encoded);
			ASSERT_TRUE(status.isOk()) << mode.name << " frame " << frameIndex
				<< ": " << status.toString();
			EXPECT_GT(encoded.frameSize, 0U);
		}
	}
	if (testedModes == 0U) {
		GTEST_SKIP() << "Device advertises neither CBR nor VBR vulkan Video encode";
	}
}


TEST(VideoRoundtrip, EncodeMuxDemuxH264)
{
	auto* engine = testEnginePtr();
	if (engine == nullptr) {
		GTEST_SKIP() << "No vulkan engine available";
	}
	if (not testVideoEncodeSupported(*engine, oa::VideoCodec::H264)) {
		GTEST_SKIP() << "VK_KHR_video_encode_h264 not supported";
	}

	const oa::I32 W = 640;
	const oa::I32 H = 360;
	const oa::U32 kFrames = 10;

	// 1. Bring up encoder.
	oa::VideoEncodeProfile prof;
	prof.codec      = oa::VideoCodec::H264;
	prof.width      = static_cast<oa::U32>(W);
	prof.height     = static_cast<oa::U32>(H);
	prof.bitrate    = 4'000'000U;
	prof.frameRate  = 30U;
	prof.gopSize    = 5U;
	prof.maxBFrames = 0U;
	auto encR = oa::VideoEncoder::create(*engine, prof);
	ASSERT_TRUE(encR.isOk()) << encR.getStatus().toString();
	oa::VideoEncoder encoder = oa::move(*encR);

	// 2. Bring up muxer.
	const std::string outPathStorage = testOutputPath("oa_roundtrip");
	const char* outPath = outPathStorage.c_str();
	oa::VideoMuxerConfig muxInfo;
	muxInfo.outputPath = outPath;
	muxInfo.codec     = oa::VideoCodec::H264;
	muxInfo.width     = prof.width;
	muxInfo.height    = prof.height;
	muxInfo.frameRate = prof.frameRate;
	auto muxR = oa::VideoMuxer::create(muxInfo);
	ASSERT_TRUE(muxR.isOk()) << muxR.getStatus().toString();
	oa::VideoMuxer muxer = oa::move(*muxR);

	// 3. Encode N frames + push to muxer.
	std::vector<oa::U8> rgba(static_cast<oa::Usize>(W) * H * 4U);
	oa::U32 muxedPackets = 0;
	oa::U32 keyframes = 0;
	bool gotConfig = false;
	for (oa::U32 f = 0; f < kFrames; ++f) {
		paintFrame(rgba, W, H, f, kFrames);
		auto texR = oa::FnTexture::fromPixels(
			*engine,
			oa::Span<const oa::U8>(rgba.data(), rgba.size()), W, H);
		ASSERT_TRUE(texR.isOk());
		oa::Texture tex = oa::move(*texR);

		oa::EncodedVideoPacket eframe;
		const oa::U64 pts = (static_cast<oa::U64>(f) * 1'000'000ULL) / prof.frameRate;
		auto encStatus = encoder.encode(tex, eframe, pts);
		ASSERT_TRUE(encStatus.isOk())
			<< "frame " << f << ": " << encStatus.toString();
		ASSERT_GT(eframe.bitstream.size(), 0U);

		// Lift SPS+PPS from the first IDR for the avcC box.
		if (!gotConfig && eframe.isKeyframe) {
			auto sps = oa::FnVideo::extractSps(
				oa::Span<const oa::U8>(eframe.bitstream.data(), eframe.bitstream.size()));
			auto pps = oa::FnVideo::extractPps(
				oa::Span<const oa::U8>(eframe.bitstream.data(), eframe.bitstream.size()));
			ASSERT_GT(sps.size(), 0U);
			ASSERT_GT(pps.size(), 0U);
			muxer.setCodecConfig(sps, pps);
			gotConfig = true;
		}

		auto writeStatus = muxer.writePacket(eframe);
		ASSERT_TRUE(writeStatus.isOk()) << writeStatus.toString();
		++muxedPackets;
		if (eframe.isKeyframe) { ++keyframes; }
	}
	EXPECT_EQ(muxedPackets, kFrames);
	EXPECT_EQ(keyframes, 2U);  // GOP=5 across ten frames: IDR at 0 and 5.

	// 4. finalize MP4 + close (dtors handle the rest).
	auto finalStatus = muxer.finalize();
	ASSERT_TRUE(finalStatus.isOk()) << finalStatus.toString();
	ASSERT_TRUE(muxer.close().isOk());

	// 5. Reopen and demux.
	auto streamR = oa::VideoDemuxer::open(outPath);
	ASSERT_TRUE(streamR.isOk()) << streamR.getStatus().toString();
	oa::VideoDemuxer stream = oa::move(*streamR);
	const auto& info = stream.getInfo();
	EXPECT_EQ(info.codec,  oa::VideoCodec::H264);
	EXPECT_EQ(info.width,  prof.width);
	EXPECT_EQ(info.height, prof.height);

	// avcC must round-trip — demuxer should parse the box the muxer wrote.
	const auto& avc = stream.getAvcConfig();
	EXPECT_TRUE(avc.valid);
	EXPECT_GE(avc.lengthSize, 1U);
	EXPECT_LE(avc.lengthSize, 4U);

	// Re-read every packet. Demuxer must produce Annex-B output.
	oa::U32 demuxedPackets = 0;
	for (oa::U32 i = 0; i < kFrames; ++i) {
		oa::VideoPacket pkt{};
		auto rs = stream.readNextPacket(pkt);
		ASSERT_TRUE(rs.isOk()) << "packet " << i << ": " << rs.toString();
		ASSERT_GE(pkt.data.size(), 4U);
		EXPECT_EQ(pkt.data[0], 0U);
		EXPECT_EQ(pkt.data[1], 0U);
		EXPECT_EQ(pkt.data[2], 0U);
		EXPECT_EQ(pkt.data[3], 1U);
		++demuxedPackets;
	}
	EXPECT_EQ(demuxedPackets, muxedPackets);

	std::remove(outPath);
}


TEST(VideoRoundtrip, DeferredMatrixTextureCompletesBeforeEncodeSnapshot)
{
	auto* engine = testEnginePtr();
	if (engine == nullptr) {
		GTEST_SKIP() << "No vulkan engine available";
	}
	if (not testVideoEncodeSupported(*engine, oa::VideoCodec::H264)
		or not testVideoDecodeSupported(*engine, oa::VideoCodec::H264)) {
		GTEST_SKIP() << "H.264 encode+decode not both supported";
	}

	constexpr oa::U32 width = 320U;
	constexpr oa::U32 height = 192U;
	auto& context = oa::ExecutionSession::getActive();
	auto white = oa::FnMatrix::ones(
		oa::MatrixShape{1, 4, height, width}, oa::ScalarType::Float32);
	oa::Image whiteImage(
		oa::move(white), oa::ImageLayout::Nchw, oa::ImageFormat::Rgba);
	auto textureResult = oa::FnTexture::fromImage(*engine, whiteImage);
	ASSERT_TRUE(textureResult.isOk()) << textureResult.getStatus().toString();
	auto texture = oa::move(*textureResult);
	ASSERT_GT(context.nodeCount(), 0U);

	// Preserve a deterministic stale snapshot. FromMatrix has only recorded its
	// white producer, so encode(...) must execute that graph before the
	// encoder is allowed to read this deliberately black packed buffer.
	oa::Vector<oa::U8> black(static_cast<oa::Usize>(width) * height * 4U, 0U);
	auto poisonStatus = oa::EngineResourceAccess::uploadBuffer(*engine,
		*oa::TextureAccess::buffer(texture), 0U, black.data(), black.size());
	ASSERT_TRUE(poisonStatus.isOk()) << poisonStatus.toString();

	oa::VideoEncodeProfile encodeProfile;
	encodeProfile.codec = oa::VideoCodec::H264;
	encodeProfile.width = width;
	encodeProfile.height = height;
	encodeProfile.gopSize = 1U;
	encodeProfile.asyncDepth = 1U;
	auto encoderResult = oa::VideoEncoder::create(*engine, encodeProfile);
	ASSERT_TRUE(encoderResult.isOk()) << encoderResult.getStatus().toString();
	auto encoder = oa::move(*encoderResult);

	oa::EncodedVideoPacket encoded;
	auto encodeStatus = encoder.encode(texture, encoded, 0U);
	ASSERT_TRUE(encodeStatus.isOk()) << encodeStatus.toString();
	ASSERT_GT(encoded.bitstream.size(), 0U);

	oa::VideoProfile decodeProfile = {
		oa::VideoCodec::H264, width, height, 8U};
	auto decoderResult = oa::VideoDecoder::create(*engine, decodeProfile);
	ASSERT_TRUE(decoderResult.isOk()) << decoderResult.getStatus().toString();
	auto decoder = oa::move(*decoderResult);
	auto frameResult = decoder.decode(
		oa::Span<const oa::U8>(encoded.bitstream.data(), encoded.bitstream.size()),
		0U);
	ASSERT_TRUE(frameResult.isOk()) << frameResult.getStatus().toString();
	auto lumaResult = decoder.readbackLuma(*frameResult);
	ASSERT_TRUE(lumaResult.isOk()) << lumaResult.getStatus().toString();
	const oa::Usize lumaSize = static_cast<oa::Usize>(width) * height;
	ASSERT_GE(lumaResult->size(), lumaSize);
	oa::U64 lumaSum = 0U;
	for (oa::Usize pixelIndex = 0U; pixelIndex < lumaSize; ++pixelIndex) {
		lumaSum += (*lumaResult)[pixelIndex];
	}
	const oa::F64 meanLuma = static_cast<oa::F64>(lumaSum)
		/ static_cast<oa::F64>(lumaSize);
	EXPECT_GT(meanLuma, 220.0)
		<< "encoder captured the black poison instead of the deferred white producer";

	ASSERT_TRUE(decoder.close().isOk());
	ASSERT_TRUE(encoder.close().isOk());
}


// Bonus coverage: pipe the demuxed output through oa::VideoDecoder to prove
// the whole encode → mux → demux → decode chain produces a valid bitstream.
TEST(VideoRoundtrip, EncodeMuxDemuxDecodeH264)
{
	auto* engine = testEnginePtr();
	if (engine == nullptr) {
		GTEST_SKIP() << "No vulkan engine available";
	}
	if (not testVideoEncodeSupported(*engine, oa::VideoCodec::H264)
	    || not testVideoDecodeSupported(*engine, oa::VideoCodec::H264)) {
		GTEST_SKIP() << "H.264 encode+decode not both supported";
	}

	const oa::I32 W = 640;
	const oa::I32 H = 360;
	const oa::U32 kFrames = 6;
	const std::string outPathStorage = testOutputPath("oa_roundtrip_dec");
	const char* outPath = outPathStorage.c_str();

	{
		oa::VideoEncodeProfile prof;
		prof.codec      = oa::VideoCodec::H264;
		prof.width      = static_cast<oa::U32>(W);
		prof.height     = static_cast<oa::U32>(H);
		prof.bitrate    = 4'000'000U;
		prof.frameRate  = 30U;
		prof.gopSize    = 3U;
		auto encR = oa::VideoEncoder::create(*engine, prof);
		ASSERT_TRUE(encR.isOk());
		oa::VideoEncoder encoder = oa::move(*encR);

		oa::VideoMuxerConfig muxInfo;
		muxInfo.outputPath = outPath;
		muxInfo.codec     = oa::VideoCodec::H264;
		muxInfo.width     = prof.width;
		muxInfo.height    = prof.height;
		muxInfo.frameRate = prof.frameRate;
		auto muxR = oa::VideoMuxer::create(muxInfo);
		ASSERT_TRUE(muxR.isOk());
		oa::VideoMuxer muxer = oa::move(*muxR);

		std::vector<oa::U8> rgba(static_cast<oa::Usize>(W) * H * 4U);
		bool gotConfig = false;
		for (oa::U32 f = 0; f < kFrames; ++f) {
			paintFrame(rgba, W, H, f, kFrames);
			auto texR = oa::FnTexture::fromPixels(
				*engine,
				oa::Span<const oa::U8>(rgba.data(), rgba.size()), W, H);
			ASSERT_TRUE(texR.isOk());
			oa::Texture tex = oa::move(*texR);

			oa::EncodedVideoPacket eframe;
			auto encodeStatus = encoder.encode(tex, eframe, f);
			ASSERT_TRUE(encodeStatus.isOk())
				<< "frame " << f << ": " << encodeStatus.getMessage().cStr();

			if (!gotConfig && eframe.isKeyframe) {
				auto sps = oa::FnVideo::extractSps(
					oa::Span<const oa::U8>(eframe.bitstream.data(), eframe.bitstream.size()));
				auto pps = oa::FnVideo::extractPps(
					oa::Span<const oa::U8>(eframe.bitstream.data(), eframe.bitstream.size()));
				muxer.setCodecConfig(sps, pps);
				gotConfig = true;
			}
			ASSERT_TRUE(muxer.writePacket(eframe).isOk());
		}
		ASSERT_TRUE(muxer.finalize().isOk());
		ASSERT_TRUE(muxer.close().isOk());
	}

	// Re-open + decode the roundtripped MP4.
	auto streamR = oa::VideoDemuxer::open(outPath);
	ASSERT_TRUE(streamR.isOk());
	oa::VideoDemuxer stream = oa::move(*streamR);

	auto profile = stream.getVideoProfile();
	profile.maxDpbSlots = 8;
	auto decR = oa::VideoDecoder::create(*engine, profile);
	ASSERT_TRUE(decR.isOk()) << decR.getStatus().toString();
	oa::VideoDecoder decoder = oa::move(*decR);

	// Re-record the decoder-owned RGBA images directly. This exercises the
	// image-backed oa::VideoFrame -> sampled image -> NV12 -> encode contract
	// without a host pixel readback or a buffer staging API at the call site.
	const std::string imageOutPathStorage = testOutputPath("oa_roundtrip_image");
	const char* imageOutPath = imageOutPathStorage.c_str();
	oa::VideoRecorderConfig imageRecorderConfig;
	imageRecorderConfig.outputPath = imageOutPath;
	imageRecorderConfig.encode.codec = oa::VideoCodec::H264;
	imageRecorderConfig.encode.width = static_cast<oa::U32>(W);
	imageRecorderConfig.encode.height = static_cast<oa::U32>(H);
	imageRecorderConfig.encode.frameRate = 30U;
	imageRecorderConfig.encode.gopSize = 3U;
	auto imageRecorderResult = oa::VideoRecorder::create(*engine, imageRecorderConfig);
	ASSERT_TRUE(imageRecorderResult.isOk()) << imageRecorderResult.getStatus().toString();
	oa::VideoRecorder imageRecorder = oa::move(*imageRecorderResult);

	oa::U32 decoded = 0;
	for (oa::U32 i = 0; i < kFrames; ++i) {
		oa::VideoPacket pkt{};
		ASSERT_TRUE(stream.readNextPacket(pkt).isOk());

		auto frameResult = decoder.decode(
			oa::Span<const oa::U8>(pkt.data.data(), pkt.data.size()),
			pkt.presentationTimestamp);
		ASSERT_TRUE(frameResult.isOk()) << "decode packet " << i << ": "
			<< frameResult.getStatus().toString();
		oa::VideoFrame fr = *frameResult;
		auto rgbResult = decoder.convert(fr);
		ASSERT_TRUE(rgbResult.isOk()) << "convert packet " << i << ": "
			<< rgbResult.getStatus().toString();
		oa::VideoFrame rgb = *rgbResult;
		oa::Texture renderTarget = oa::TextureAccess::fromBorrowedImage(
			*engine,
			rgb.image,
			rgb.imageView,
			rgb.format,
			rgb.layout,
			static_cast<oa::I32>(rgb.width),
			static_cast<oa::I32>(rgb.height));
		const oa::U64 pts = static_cast<oa::U64>(i) * 1'000'000ULL / 30ULL;
		auto renderFrame = oa::FnVideo::fromTexture(
			renderTarget, pts, rgb.ready);
		ASSERT_TRUE(renderFrame.isOk()) << renderFrame.getStatus().toString();
		oa::Event inputConsumed;
		auto recordStatus = imageRecorder.writeAsync(*renderFrame, inputConsumed);
		ASSERT_TRUE(recordStatus.isOk()) << "record image " << i << ": "
			<< recordStatus.toString();
		ASSERT_TRUE(inputConsumed.isValid());
		ASSERT_TRUE(inputConsumed.wait().isOk());
		++decoded;
	}
	EXPECT_EQ(decoded, kFrames);
	ASSERT_TRUE(imageRecorder.finalize().isOk());
	ASSERT_TRUE(imageRecorder.close().isOk());
	if (std::system("command -v ffmpeg >/dev/null 2>&1") == 0) {
		const std::string command = std::string("ffmpeg -xerror -v error -y -i \"")
			+ imageOutPath + "\" -f rawvideo -pix_fmt rgba /dev/null >/dev/null 2>&1";
		EXPECT_EQ(std::system(command.c_str()), 0)
			<< "Independent FFmpeg decoder rejected the image-backed recording";
	}

	std::remove(outPath);
	std::remove(imageOutPath);
}


// Native HEVC proof: vulkan encode -> hvc1/hvcC MP4 -> OA demux/decode,
// followed by an independent FFmpeg decode of the same file.
TEST(VideoRoundtrip, EncodeMuxDemuxDecodeH265)
{
	auto* engine = testEnginePtr();
	if (engine == nullptr) GTEST_SKIP() << "No vulkan engine available";
	if (not testVideoEncodeSupported(*engine, oa::VideoCodec::H265)
		|| not testVideoDecodeSupported(*engine, oa::VideoCodec::H265)) {
		GTEST_SKIP() << "H.265 encode+decode not both supported";
	}

	constexpr oa::I32 width = 320;
	constexpr oa::I32 height = 192;
	constexpr oa::U32 frameCount = 6U;
	constexpr oa::U32 frameRate = 30U;
	const std::string pathStorage = testOutputPath("oa_roundtrip_h265");
	const char* path = pathStorage.c_str();

	oa::VideoRecorderConfig config;
	config.outputPath = path;
	config.encode.codec = oa::VideoCodec::H265;
	config.encode.width = static_cast<oa::U32>(width);
	config.encode.height = static_cast<oa::U32>(height);
	config.encode.frameRate = frameRate;
	config.encode.gopSize = 3U;
	config.encode.rateControl = oa::VideoRateControl::ConstantQp;
	auto recorderResult = oa::VideoRecorder::create(*engine, config);
	ASSERT_TRUE(recorderResult.isOk()) << recorderResult.getStatus().toString();
	oa::VideoRecorder recorder = oa::move(*recorderResult);

	std::vector<oa::U8> rgba(static_cast<oa::Usize>(width) * height * 4U);
	for (oa::U32 frameIndex = 0; frameIndex < frameCount; ++frameIndex) {
		paintFrame(rgba, width, height, frameIndex, frameCount);
		auto textureResult = oa::FnTexture::fromPixels(
			*engine, oa::Span<const oa::U8>(rgba.data(), rgba.size()), width, height);
		ASSERT_TRUE(textureResult.isOk()) << textureResult.getStatus().toString();
		oa::Texture texture = oa::move(*textureResult);
		const oa::U64 pts = static_cast<oa::U64>(frameIndex) * 1'000'000ULL / frameRate;
		auto status = recorder.writeRgba(
			*oa::TextureAccess::buffer(texture), width, height, pts);
		ASSERT_TRUE(status.isOk()) << "encode frame " << frameIndex << ": "
			<< status.toString();
	}
	ASSERT_TRUE(recorder.finalize().isOk());
	ASSERT_TRUE(recorder.close().isOk());

	auto streamResult = oa::VideoDemuxer::open(path);
	ASSERT_TRUE(streamResult.isOk()) << streamResult.getStatus().toString();
	oa::VideoDemuxer stream = oa::move(*streamResult);
	EXPECT_EQ(stream.getInfo().codec, oa::VideoCodec::H265);
	EXPECT_EQ(stream.getInfo().width, static_cast<oa::U32>(width));
	EXPECT_EQ(stream.getInfo().height, static_cast<oa::U32>(height));
	EXPECT_TRUE(stream.getHvcConfig().valid);

	auto profile = stream.getVideoProfile();
	profile.maxDpbSlots = 8U;
	auto decoderResult = oa::VideoDecoder::create(*engine, profile);
	ASSERT_TRUE(decoderResult.isOk()) << decoderResult.getStatus().toString();
	oa::VideoDecoder decoder = oa::move(*decoderResult);
	for (oa::U32 frameIndex = 0; frameIndex < frameCount; ++frameIndex) {
		oa::VideoPacket packet = {};
		ASSERT_TRUE(stream.readNextPacket(packet).isOk()) << "demux frame " << frameIndex;
		ASSERT_GT(packet.data.size(), 4U);
		auto frameResult = decoder.decode(
			oa::Span<const oa::U8>(packet.data.data(), packet.data.size()),
			packet.presentationTimestamp);
		ASSERT_TRUE(frameResult.isOk()) << "decode frame " << frameIndex << ": "
			<< frameResult.getStatus().toString();
	}

	if (std::system("command -v ffmpeg >/dev/null 2>&1") == 0) {
		const std::string command = std::string("ffmpeg -xerror -v error -y -i \"")
			+ path + "\" -f rawvideo -pix_fmt rgba /dev/null >/dev/null 2>&1";
		EXPECT_EQ(std::system(command.c_str()), 0)
			<< "Independent FFmpeg decoder rejected OA's H.265 MP4";
	}
	std::remove(path);
}
