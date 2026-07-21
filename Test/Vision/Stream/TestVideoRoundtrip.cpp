// OA Vision — Video Roundtrip Test (encode → mux → demux → readback)
// End-to-end proof that OaVideoEncoder + OaVideoMuxer produce a valid MP4
// that OaVideoStream can open and read back the same packets from.

#include "../../OaTest.h"

#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Stream.h>
#include <Oa/Ml/FnMatrix.h>
#include <Oa/Vision/CameraCapture.h>
#include <Oa/Vision/ScreenCapture.h>
#include <Oa/Vision/VideoEncoder.h>
#include <Oa/Vision/VideoMuxer.h>
#include <Oa/Vision/VideoRecorder.h>
#include <Oa/Runtime/ExternalMemory.h>
#include <Oa/Vision/VideoStream.h>
#include <Oa/Vision/FnVideo.h>
#include <Oa/Ui/Image.h>

#include <cstdio>
#include <cmath>
#include <string>
#include <thread>
#include <vector>

namespace {

// Synthetic gradient + moving rectangle, identical to TutorialEncodeBasic's
// frame producer. Doesn't depend on any dataset.
void PaintFrame(std::vector<OaU8>& InOut, OaI32 InW, OaI32 InH, OaU32 InFrame, OaU32 InTotal) {
	const OaF32 t = static_cast<OaF32>(InFrame) / static_cast<OaF32>(InTotal);
	const OaI32 rectW = InW / 6;
	const OaI32 rectH = InH / 6;
	const OaI32 rectX = static_cast<OaI32>(t * static_cast<OaF32>(InW - rectW));
	const OaI32 rectY = (InH - rectH) / 2;
	for (OaI32 y = 0; y < InH; ++y) {
		for (OaI32 x = 0; x < InW; ++x) {
			OaU8 r = static_cast<OaU8>((x * 255) / InW);
			OaU8 g = static_cast<OaU8>((y * 255) / InH);
			OaU8 b = static_cast<OaU8>(((InFrame * 4U) & 0xFFU));
			if (x >= rectX && x < rectX + rectW
				&& y >= rectY && y < rectY + rectH) {
				r = 240; g = 80; b = 80;
			}
			const OaI64 i = (static_cast<OaI64>(y) * InW + x) * 4;
			InOut[static_cast<OaUsize>(i + 0)] = r;
			InOut[static_cast<OaUsize>(i + 1)] = g;
			InOut[static_cast<OaUsize>(i + 2)] = b;
			InOut[static_cast<OaUsize>(i + 3)] = 255U;
		}
	}
}

} // namespace

TEST(OaCaptureLifecycle, DefaultCloseIsIdempotent) {
	OaCameraCapture camera;
	OaScreenCapture screen;
	EXPECT_TRUE(camera.Close().IsOk());
	EXPECT_TRUE(camera.Close().IsOk());
	EXPECT_TRUE(screen.Close().IsOk());
	EXPECT_TRUE(screen.Close().IsOk());
}

TEST(OaCaptureLifecycle, AbandonedCameraRetiresAtEngineClose) {
	const char* enabled = std::getenv("OA_TEST_CAMERA_CAPTURE");
	if (enabled == nullptr or enabled[0] != '1') {
		GTEST_SKIP() << "Set OA_TEST_CAMERA_CAPTURE=1 for the physical-camera lifecycle gate";
	}
	auto config = OaTestEngineConfig(OaPrecision::FP32);
	config.RegisterAsGlobal = false;
	config.PreloadEmbeddedPipelines = false;
	config.EnablePipelineCache = false;
	auto engineResult = OaEngine::Create(config);
	ASSERT_TRUE(engineResult.IsOk()) << engineResult.GetStatus().ToString();
	auto engine = OaStdMove(*engineResult);
	OaString skipReason;
	{
		OaCameraCapture capture;
		OaCameraCaptureConfig captureConfig;
		captureConfig.Width = 640;
		captureConfig.Height = 480;
		if (const char* device = std::getenv("OA_TEST_CAMERA_DEVICE")) {
			captureConfig.DevicePath = device;
		}
		const auto initStatus = capture.Init(*engine, captureConfig);
		if (not initStatus.IsOk()) {
			skipReason = initStatus.ToString();
		} else {
			OaVideoFrame frame;
			const auto deadline = std::chrono::steady_clock::now()
				+ std::chrono::seconds(5);
			while (not capture.PollFrame(frame)
				and std::chrono::steady_clock::now() < deadline) {
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
			}
			if (frame.Width == 0U or frame.Height == 0U) {
				skipReason = "camera produced no frame within five seconds";
			} else {
				auto* stream = engine->AcquireStream();
				ASSERT_NE(stream, nullptr);
				ASSERT_TRUE(stream->Begin(engine->Device).IsOk());
				stream->RecordBufferBarrier();
				ASSERT_TRUE(stream->Submit(*engine).IsOk());
				const auto consumed = stream->Completion(engine->Device);
				ASSERT_TRUE(consumed.IsValid());
				capture.Release(frame, consumed);
			}
		}
		// No Close/Destroy: producer state and the exact frame-consumer token move
		// to engine-owned retirement without waiting in the capture destructor.
	}
	ASSERT_TRUE(engine->Close().IsOk());
	if (not skipReason.empty()) GTEST_SKIP() << skipReason.c_str();
}

TEST(OaCaptureLifecycle, AbandonedScreenCaptureRetiresAtEngineClose) {
	const char* enabled = std::getenv("OA_TEST_SCREEN_CAPTURE");
	if (enabled == nullptr or enabled[0] != '1') {
		GTEST_SKIP() << "Set OA_TEST_SCREEN_CAPTURE=1 for the interactive portal lifecycle gate";
	}
	auto config = OaTestEngineConfig(OaPrecision::FP32);
	config.RegisterAsGlobal = false;
	config.PreloadEmbeddedPipelines = false;
	config.EnablePipelineCache = false;
	auto engineResult = OaEngine::Create(config);
	ASSERT_TRUE(engineResult.IsOk()) << engineResult.GetStatus().ToString();
	auto engine = OaStdMove(*engineResult);
	OaString skipReason;
	{
		auto captureResult = OaScreenCapture::Open(*engine);
		if (not captureResult.IsOk()) {
			skipReason = captureResult.GetStatus().ToString();
		} else {
			auto capture = OaStdMove(*captureResult);
			OaVideoFrame frame;
			const auto deadline = std::chrono::steady_clock::now()
				+ std::chrono::seconds(5);
			while (not capture.Poll(frame)
				and std::chrono::steady_clock::now() < deadline) {
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
			}
			if (frame.Width == 0U or frame.Height == 0U) {
				skipReason = "screen capture produced no frame within five seconds";
			} else {
				auto* stream = engine->AcquireStream();
				ASSERT_NE(stream, nullptr);
				ASSERT_TRUE(stream->Begin(engine->Device).IsOk());
				stream->RecordBufferBarrier();
				ASSERT_TRUE(stream->Submit(*engine).IsOk());
				const auto consumed = stream->Completion(engine->Device);
				ASSERT_TRUE(consumed.IsValid());
				capture.Release(frame, consumed);
			}
			// No Close/Destroy: the callback loop is asked to stop without a join;
			// engine close joins it and completes exact consumer dependencies.
		}
	}
	ASSERT_TRUE(engine->Close().IsOk());
	if (not skipReason.empty()) GTEST_SKIP() << skipReason.c_str();
}

TEST(OaVideoRoundtrip, DmaBufImporterRejectsIncompleteDescriptions) {
	auto* engine = OaEngine::GetGlobal();
	if (engine == nullptr) GTEST_SKIP() << "No Vulkan engine available";
	OaDmaBufImageDesc description;
	auto result = OaImportedDmaBufImage::Import(*engine, description);
	ASSERT_FALSE(result.IsOk());
	EXPECT_EQ(result.GetStatus().GetCode(), OaStatusCode::InvalidArgument);
}


TEST(OaVideoRoundtrip, AbandonedSubmittedEncoderRetiresAtEngineClose) {
	auto config = OaTestEngineConfig(OaPrecision::FP32);
	config.RegisterAsGlobal = false;
	config.PreloadEmbeddedPipelines = false;
	config.EnablePipelineCache = false;
	auto engineResult = OaEngine::Create(config);
	ASSERT_TRUE(engineResult.IsOk()) << engineResult.GetStatus().ToString();
	auto engine = OaStdMove(*engineResult);
	if (not OaVideoEncoder::IsCodecSupported(*engine, OaVideoCodec::H264)) {
		ASSERT_TRUE(engine->Close().IsOk());
		GTEST_SKIP() << "H.264 Vulkan Video encode is not supported on selected device";
	}

	constexpr OaU32 width = 320U;
	constexpr OaU32 height = 192U;
	std::vector<OaU8> rgba(static_cast<OaUsize>(width) * height * 4U, 127U);
	auto textureResult = OaTexture::FromPixels(
		*engine, OaSpan<const OaU8>(rgba.data(), rgba.size()), width, height);
	ASSERT_TRUE(textureResult.IsOk()) << textureResult.GetStatus().ToString();
	auto texture = OaStdMove(*textureResult);
	{
		OaVideoEncodeProfile profile;
		profile.Width = width;
		profile.Height = height;
		profile.AsyncDepth = 2U;
		auto encoderResult = OaVideoEncoder::Create(*engine, profile);
		ASSERT_TRUE(encoderResult.IsOk()) << encoderResult.GetStatus().ToString();
		auto encoder = OaStdMove(*encoderResult);
		OaVec<OaEncodedFrame> ready;
		ASSERT_TRUE(encoder.SubmitRgba(
			texture.DeviceBuf, width, height, 0U, ready).IsOk());
		// No Close/Destroy: the pending video fence, conversion ticket, and
		// complete encoder session must transfer to engine retirement.
	}

	texture.Destroy(*engine);
	ASSERT_TRUE(engine->Close().IsOk());
}


TEST(OaVideoRoundtrip, AbandonedRecorderRetiresSubmittedEncoderAtEngineClose) {
	auto config = OaTestEngineConfig(OaPrecision::FP32);
	config.RegisterAsGlobal = false;
	config.PreloadEmbeddedPipelines = false;
	config.EnablePipelineCache = false;
	auto engineResult = OaEngine::Create(config);
	ASSERT_TRUE(engineResult.IsOk()) << engineResult.GetStatus().ToString();
	auto engine = OaStdMove(*engineResult);
	if (not OaVideoEncoder::IsCodecSupported(*engine, OaVideoCodec::H264)) {
		ASSERT_TRUE(engine->Close().IsOk());
		GTEST_SKIP() << "H.264 Vulkan Video encode is not supported on selected device";
	}

	constexpr OaU32 width = 320U;
	constexpr OaU32 height = 192U;
	const char* path = "/tmp/oa_abandoned_recorder.mp4";
	std::vector<OaU8> rgba(static_cast<OaUsize>(width) * height * 4U, 63U);
	auto textureResult = OaTexture::FromPixels(
		*engine, OaSpan<const OaU8>(rgba.data(), rgba.size()), width, height);
	ASSERT_TRUE(textureResult.IsOk()) << textureResult.GetStatus().ToString();
	auto texture = OaStdMove(*textureResult);
	{
		OaVideoRecorderConfig recorderConfig;
		recorderConfig.OutputPath = path;
		recorderConfig.Encode.Width = width;
		recorderConfig.Encode.Height = height;
		recorderConfig.Encode.AsyncDepth = 2U;
		auto recorderResult = OaVideoRecorder::Create(*engine, recorderConfig);
		ASSERT_TRUE(recorderResult.IsOk()) << recorderResult.GetStatus().ToString();
		auto recorder = OaStdMove(*recorderResult);
		ASSERT_TRUE(recorder.WriteRgba(
			texture.DeviceBuf, width, height, 0U).IsOk());
		// No Finalize/Destroy: close the host file, but leave submitted GPU work
		// to the encoder's engine-owned retirement path.
	}

	texture.Destroy(*engine);
	std::remove(path);
	ASSERT_TRUE(engine->Close().IsOk());
}


TEST(OaVideoRoundtrip, TranscoderUsesGpuDecodeConvertEncodePath) {
	auto* engine = OaEngine::GetGlobal();
	if (engine == nullptr) GTEST_SKIP() << "No Vulkan engine available";
	if (not OaVideoDecoder::IsCodecSupported(*engine, OaVideoCodec::H264)
		or not OaVideoEncoder::IsCodecSupported(*engine, OaVideoCodec::H264)) {
		GTEST_SKIP() << "H.264 decode+encode are not both supported";
	}
#if defined(_WIN32)
	const char* sourcePath =
		"../dataset/video/shibuya_crossing_1080p30_h264.mp4";
#else
	const char* sourcePath =
		"../dataset/video/shibuya_crossing_1080p30_h264.mp4";
#endif
	auto streamResult = OaVideoStream::OpenFile(sourcePath);
	if (not streamResult.IsOk()) {
		GTEST_SKIP() << "Shibuya H.264 source is unavailable";
	}
	OaVideoStream stream = OaStdMove(*streamResult);
	OaVideoPacket packet{};
	ASSERT_TRUE(stream.ReadNextPacket(packet).IsOk());

	OaVideoProfile decode = stream.GetVideoProfile();
	decode.MaxDpbSlots = 16U;
	OaVideoEncodeProfile encode;
	encode.Codec = OaVideoCodec::H264;
	encode.Width = decode.Width;
	encode.Height = decode.Height;
	encode.FrameRate = stream.GetInfo().FrameRate;
	encode.AsyncDepth = 1U;
	auto transcoderResult = OaVideoTranscoder::Create(*engine, decode, encode);
	ASSERT_TRUE(transcoderResult.IsOk())
		<< transcoderResult.GetStatus().ToString();
	OaVideoTranscoder transcoder = OaStdMove(*transcoderResult);
	OaEncodedFrame output{};
	ASSERT_TRUE(transcoder.TranscodeFrame(
		OaSpan<const OaU8>(packet.Data.Data(), packet.Data.Size()), output).IsOk());
	EXPECT_GT(output.FrameSize, 4U);
	EXPECT_EQ(output.FrameSize, output.Bitstream.Size());
	EXPECT_TRUE(output.IsKeyframe);
	EXPECT_EQ(output.Bitstream[0], 0U);
	EXPECT_EQ(output.Bitstream[1], 0U);
	transcoder.Destroy();
}


TEST(OaVideoRoundtrip, RecorderComposesEncodeAndMux)
{
	auto* engine = OaEngine::GetGlobal();
	if (engine == nullptr) GTEST_SKIP() << "No Vulkan engine available";
	if (not OaVideoEncoder::IsCodecSupported(*engine, OaVideoCodec::H264)) {
		GTEST_SKIP() << "VK_KHR_video_encode_h264 not supported";
	}

	constexpr OaI32 W = 320;
	constexpr OaI32 H = 192;
	constexpr OaU32 kFrames = 6;
	const char* path = "/tmp/oa_recorder_roundtrip.mp4";

	OaVideoRecorderConfig cfg;
	cfg.OutputPath = path;
	cfg.Encode.Codec = OaVideoCodec::H264;
	cfg.Encode.Width = W;
	cfg.Encode.Height = H;
	cfg.Encode.FrameRate = 30;
	cfg.Encode.GopSize = 3;
	cfg.AudioEnabled = true;
	cfg.Audio.SampleRate = 48'000U;
	cfg.Audio.ChannelCount = 2U;
	auto recorderResult = OaVideoRecorder::Create(*engine, cfg);
	ASSERT_TRUE(recorderResult.IsOk()) << recorderResult.GetStatus().ToString();
	OaVideoRecorder recorder = OaStdMove(*recorderResult);

	std::vector<OaU8> rgba(static_cast<OaUsize>(W) * H * 4U);
	constexpr OaU32 kAudioFramesPerVideoFrame = 48'000U / 30U;
	std::vector<OaF32> audio(kAudioFramesPerVideoFrame * 2U);
	for (OaU32 frame = 0; frame < kFrames; ++frame) {
		for (OaU32 i = 0U; i < kAudioFramesPerVideoFrame; ++i) {
			const OaU64 absoluteFrame = static_cast<OaU64>(frame) * kAudioFramesPerVideoFrame + i;
			const OaF32 sample = 0.2F * std::sin(
				2.0F * 3.14159265358979323846F * 440.0F
				* static_cast<OaF32>(absoluteFrame) / 48'000.0F);
			audio[i * 2U] = sample;
			audio[i * 2U + 1U] = sample;
		}
		const OaU64 pts = static_cast<OaU64>(frame) * 1'000'000ULL / 30ULL;
		// The first chunk intentionally arrives before the first video frame;
		// the recorder must establish one common monotonic epoch and retain it.
		ASSERT_TRUE(recorder.WriteAudio(
			OaSpan<const OaF32>(audio.data(), audio.size()), 48'000U, 2U, pts).IsOk());
		PaintFrame(rgba, W, H, frame, kFrames);
		auto textureResult = OaTexture::FromPixels(
			*engine, OaSpan<const OaU8>(rgba.data(), rgba.size()), W, H);
		ASSERT_TRUE(textureResult.IsOk());
		OaTexture texture = OaStdMove(*textureResult);
		auto status = recorder.WriteRgba(texture.DeviceBuf, W, H, pts);
		texture.Destroy(*engine);
		ASSERT_TRUE(status.IsOk()) << status.ToString();
	}
	EXPECT_EQ(recorder.GetFrameCount(), kFrames);
	ASSERT_TRUE(recorder.Finalize().IsOk());

	auto streamResult = OaVideoStream::OpenFile(path);
	ASSERT_TRUE(streamResult.IsOk()) << streamResult.GetStatus().ToString();
	OaVideoStream stream = OaStdMove(*streamResult);
	EXPECT_EQ(stream.GetInfo().Codec, OaVideoCodec::H264);
	EXPECT_EQ(stream.GetInfo().Width, static_cast<OaU32>(W));
	EXPECT_EQ(stream.GetInfo().Height, static_cast<OaU32>(H));
	OaU32 packets = 0;
	for (; packets < kFrames; ++packets) {
		OaVideoPacket packet = {};
		ASSERT_TRUE(stream.ReadNextPacket(packet).IsOk());
		ASSERT_GT(packet.Data.Size(), 4U);
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


TEST(OaVideoRoundtrip, AdvertisedRateControlModesEncode)
{
	auto* engine = OaEngine::GetGlobal();
	if (engine == nullptr) GTEST_SKIP() << "No Vulkan engine available";
	auto capsResult = OaVideoEncoder::QueryEncodeCapabilities(*engine, OaVideoCodec::H264);
	if (not capsResult.IsOk()) GTEST_SKIP() << capsResult.GetStatus().ToString();

	const struct {
		OaVideoRateControl Public;
		VkVideoEncodeRateControlModeFlagBitsKHR Vulkan;
		const char* Name;
	} modes[] = {
		{OaVideoRateControl::Cbr, VK_VIDEO_ENCODE_RATE_CONTROL_MODE_CBR_BIT_KHR, "CBR"},
		{OaVideoRateControl::Vbr, VK_VIDEO_ENCODE_RATE_CONTROL_MODE_VBR_BIT_KHR, "VBR"},
	};

	constexpr OaU32 width = 320U;
	constexpr OaU32 height = 192U;
	std::vector<OaU8> rgba(static_cast<OaUsize>(width) * height * 4U);
	OaU32 testedModes = 0U;
	for (const auto& mode : modes) {
		if (((*capsResult).RateControlModes & mode.Vulkan) == 0U) continue;
		++testedModes;
		OaVideoEncodeProfile profile;
		profile.Codec = OaVideoCodec::H264;
		profile.Width = width;
		profile.Height = height;
		profile.FrameRate = 30U;
		profile.GopSize = 3U;
		profile.RateControl = mode.Public;
		profile.Bitrate = 1'000'000U;
		profile.MaxBitrate = mode.Public == OaVideoRateControl::Vbr ? 1'500'000U : 0U;
		auto encoderResult = OaVideoEncoder::Create(*engine, profile);
		ASSERT_TRUE(encoderResult.IsOk()) << mode.Name << ": "
			<< encoderResult.GetStatus().ToString();
		OaVideoEncoder encoder = OaStdMove(*encoderResult);
		for (OaU32 frameIndex = 0; frameIndex < 3U; ++frameIndex) {
			PaintFrame(rgba, width, height, frameIndex, 3U);
			auto textureResult = OaTexture::FromPixels(
				*engine, OaSpan<const OaU8>(rgba.data(), rgba.size()), width, height);
			ASSERT_TRUE(textureResult.IsOk());
			OaTexture texture = OaStdMove(*textureResult);
			ASSERT_TRUE(encoder.UploadInputRgba(texture.DeviceBuf, width, height).IsOk());
			OaEncodedFrame encoded;
			auto status = encoder.EncodeFrame(
				VK_NULL_HANDLE, frameIndex * (1'000'000ULL / 30ULL), encoded);
			texture.Destroy(*engine);
			ASSERT_TRUE(status.IsOk()) << mode.Name << " frame " << frameIndex
				<< ": " << status.ToString();
			EXPECT_GT(encoded.FrameSize, 0U);
		}
	}
	if (testedModes == 0U) {
		GTEST_SKIP() << "Device advertises neither CBR nor VBR Vulkan Video encode";
	}
}


TEST(OaVideoRoundtrip, EncodeMuxDemuxH264)
{
	auto* engine = OaEngine::GetGlobal();
	if (engine == nullptr) {
		GTEST_SKIP() << "No Vulkan engine available";
	}
	if (not OaVideoEncoder::IsCodecSupported(*engine, OaVideoCodec::H264)) {
		GTEST_SKIP() << "VK_KHR_video_encode_h264 not supported";
	}

	const OaI32 W = 640;
	const OaI32 H = 360;
	const OaU32 kFrames = 10;

	// 1. Bring up encoder.
	OaVideoEncodeProfile prof;
	prof.Codec      = OaVideoCodec::H264;
	prof.Width      = static_cast<OaU32>(W);
	prof.Height     = static_cast<OaU32>(H);
	prof.Bitrate    = 4'000'000U;
	prof.FrameRate  = 30U;
	prof.GopSize    = 5U;
	prof.MaxBFrames = 0U;
	auto encR = OaVideoEncoder::Create(*engine, prof);
	ASSERT_TRUE(encR.IsOk()) << encR.GetStatus().ToString();
	OaVideoEncoder encoder = OaStdMove(*encR);

	// 2. Bring up muxer.
	const char* outPath = "/tmp/oa_roundtrip.mp4";
	OaVideoMuxer::CreateInfo muxInfo;
	muxInfo.Codec     = OaVideoCodec::H264;
	muxInfo.Width     = prof.Width;
	muxInfo.Height    = prof.Height;
	muxInfo.FrameRate = prof.FrameRate;
	auto muxR = OaVideoMuxer::Create(outPath, muxInfo);
	ASSERT_TRUE(muxR.IsOk()) << muxR.GetStatus().ToString();
	OaVideoMuxer muxer = OaStdMove(*muxR);

	// 3. Encode N frames + push to muxer.
	auto& ctx = OaContext::GetDefault();
	std::vector<OaU8> rgba(static_cast<OaUsize>(W) * H * 4U);
	OaU32 muxedPackets = 0;
	OaU32 keyframes = 0;
	bool gotConfig = false;
	for (OaU32 f = 0; f < kFrames; ++f) {
		PaintFrame(rgba, W, H, f, kFrames);
		auto texR = OaTexture::FromPixels(
			*engine,
			OaSpan<const OaU8>(rgba.data(), rgba.size()), W, H);
		ASSERT_TRUE(texR.IsOk());
		OaTexture tex = OaStdMove(*texR);

		OaEncodedFrame eframe;
		const OaU64 pts = (static_cast<OaU64>(f) * 1'000'000ULL) / prof.FrameRate;
		auto encStatus = OaFnVideo::Encode(ctx, encoder, tex, eframe, pts);
		tex.Destroy(*engine);
		ASSERT_TRUE(encStatus.IsOk())
			<< "frame " << f << ": " << encStatus.ToString();
		ASSERT_GT(eframe.Bitstream.Size(), 0U);

		// Lift SPS+PPS from the first IDR for the avcC box.
		if (!gotConfig && eframe.IsKeyframe) {
			auto sps = OaFnVideo::ExtractSps(
				OaSpan<const OaU8>(eframe.Bitstream.Data(), eframe.Bitstream.Size()));
			auto pps = OaFnVideo::ExtractPps(
				OaSpan<const OaU8>(eframe.Bitstream.Data(), eframe.Bitstream.Size()));
			ASSERT_GT(sps.Size(), 0U);
			ASSERT_GT(pps.Size(), 0U);
			muxer.SetCodecConfig(sps, pps);
			gotConfig = true;
		}

		auto writeStatus = muxer.WritePacket(eframe);
		ASSERT_TRUE(writeStatus.IsOk()) << writeStatus.ToString();
		++muxedPackets;
		if (eframe.IsKeyframe) { ++keyframes; }
	}
	EXPECT_EQ(muxedPackets, kFrames);
	EXPECT_EQ(keyframes, 2U);  // GOP=5 across ten frames: IDR at 0 and 5.

	// 4. Finalize MP4 + close (dtors handle the rest).
	auto finalStatus = muxer.Finalize();
	ASSERT_TRUE(finalStatus.IsOk()) << finalStatus.ToString();

	// 5. Reopen and demux.
	auto streamR = OaVideoStream::OpenFile(outPath);
	ASSERT_TRUE(streamR.IsOk()) << streamR.GetStatus().ToString();
	OaVideoStream stream = OaStdMove(*streamR);
	const auto& info = stream.GetInfo();
	EXPECT_EQ(info.Codec,  OaVideoCodec::H264);
	EXPECT_EQ(info.Width,  prof.Width);
	EXPECT_EQ(info.Height, prof.Height);

	// avcC must round-trip — demuxer should parse the box the muxer wrote.
	const auto& avc = stream.GetAvcConfig();
	EXPECT_TRUE(avc.Valid);
	EXPECT_GE(avc.LengthSize, 1U);
	EXPECT_LE(avc.LengthSize, 4U);

	// Re-read every packet. Demuxer must produce Annex-B output.
	OaU32 demuxedPackets = 0;
	for (OaU32 i = 0; i < kFrames; ++i) {
		OaVideoPacket pkt{};
		auto rs = stream.ReadNextPacket(pkt);
		ASSERT_TRUE(rs.IsOk()) << "packet " << i << ": " << rs.ToString();
		ASSERT_GE(pkt.Data.Size(), 4U);
		EXPECT_EQ(pkt.Data[0], 0U);
		EXPECT_EQ(pkt.Data[1], 0U);
		EXPECT_EQ(pkt.Data[2], 0U);
		EXPECT_EQ(pkt.Data[3], 1U);
		++demuxedPackets;
	}
	EXPECT_EQ(demuxedPackets, muxedPackets);

	std::remove(outPath);
}


TEST(OaVideoRoundtrip, DeferredMatrixTextureCompletesBeforeEncodeSnapshot)
{
	auto* engine = OaEngine::GetGlobal();
	if (engine == nullptr) {
		GTEST_SKIP() << "No Vulkan engine available";
	}
	if (not OaVideoEncoder::IsCodecSupported(*engine, OaVideoCodec::H264)
		or not OaVideoDecoder::IsCodecSupported(*engine, OaVideoCodec::H264)) {
		GTEST_SKIP() << "H.264 encode+decode not both supported";
	}

	constexpr OaU32 width = 320U;
	constexpr OaU32 height = 192U;
	auto& context = OaContext::GetDefault();
	auto white = OaFnMatrix::Ones(
		OaMatrixShape{1, 4, height, width}, OaScalarType::Float32);
	auto textureResult = OaTexture::FromMatrix(context, white);
	ASSERT_TRUE(textureResult.IsOk()) << textureResult.GetStatus().ToString();
	auto texture = OaStdMove(*textureResult);
	ASSERT_GT(context.NodeCount(), 0U);

	// Preserve a deterministic stale snapshot. FromMatrix has only recorded its
	// white producer, so Encode(context, ...) must execute that graph before the
	// encoder is allowed to read this deliberately black packed buffer.
	OaVec<OaU8> black(static_cast<OaUsize>(width) * height * 4U, 0U);
	auto poisonStatus = engine->UploadBuffer(
		texture.DeviceBuf, 0U, black.Data(), black.Size());
	ASSERT_TRUE(poisonStatus.IsOk()) << poisonStatus.ToString();

	OaVideoEncodeProfile encodeProfile;
	encodeProfile.Codec = OaVideoCodec::H264;
	encodeProfile.Width = width;
	encodeProfile.Height = height;
	encodeProfile.GopSize = 1U;
	encodeProfile.AsyncDepth = 1U;
	auto encoderResult = OaVideoEncoder::Create(*engine, encodeProfile);
	ASSERT_TRUE(encoderResult.IsOk()) << encoderResult.GetStatus().ToString();
	auto encoder = OaStdMove(*encoderResult);

	OaEncodedFrame encoded;
	auto encodeStatus = OaFnVideo::Encode(
		context, encoder, texture, encoded, 0U);
	ASSERT_TRUE(encodeStatus.IsOk()) << encodeStatus.ToString();
	ASSERT_GT(encoded.Bitstream.Size(), 0U);

	OaVideoProfile decodeProfile = {
		OaVideoCodec::H264, width, height, 8U};
	auto decoderResult = OaVideoDecoder::Create(*engine, decodeProfile);
	ASSERT_TRUE(decoderResult.IsOk()) << decoderResult.GetStatus().ToString();
	auto decoder = OaStdMove(*decoderResult);
	auto frameResult = OaFnVideo::Decode(
		context,
		decoder,
		OaSpan<const OaU8>(encoded.Bitstream.Data(), encoded.Bitstream.Size()),
		0U);
	ASSERT_TRUE(frameResult.IsOk()) << frameResult.GetStatus().ToString();
	auto lumaResult = OaFnVideo::ReadbackLuma(decoder, *frameResult);
	ASSERT_TRUE(lumaResult.IsOk()) << lumaResult.GetStatus().ToString();
	const OaUsize lumaSize = static_cast<OaUsize>(width) * height;
	ASSERT_GE(lumaResult->Size(), lumaSize);
	OaU64 lumaSum = 0U;
	for (OaUsize pixelIndex = 0U; pixelIndex < lumaSize; ++pixelIndex) {
		lumaSum += (*lumaResult)[pixelIndex];
	}
	const OaF64 meanLuma = static_cast<OaF64>(lumaSum)
		/ static_cast<OaF64>(lumaSize);
	EXPECT_GT(meanLuma, 220.0)
		<< "encoder captured the black poison instead of the deferred white producer";

	ASSERT_TRUE(decoder.Close().IsOk());
	ASSERT_TRUE(encoder.Close().IsOk());
	texture.Destroy(*engine);
}


// Bonus coverage: pipe the demuxed output through OaVideoDecoder to prove
// the whole encode → mux → demux → decode chain produces a valid bitstream.
TEST(OaVideoRoundtrip, EncodeMuxDemuxDecodeH264)
{
	auto* engine = OaEngine::GetGlobal();
	if (engine == nullptr) {
		GTEST_SKIP() << "No Vulkan engine available";
	}
	if (not OaVideoEncoder::IsCodecSupported(*engine, OaVideoCodec::H264)
	    || not OaVideoDecoder::IsCodecSupported(*engine, OaVideoCodec::H264)) {
		GTEST_SKIP() << "H.264 encode+decode not both supported";
	}

	const OaI32 W = 640;
	const OaI32 H = 360;
	const OaU32 kFrames = 6;
	const char* outPath = "/tmp/oa_roundtrip_dec.mp4";

	{
		OaVideoEncodeProfile prof;
		prof.Codec      = OaVideoCodec::H264;
		prof.Width      = static_cast<OaU32>(W);
		prof.Height     = static_cast<OaU32>(H);
		prof.Bitrate    = 4'000'000U;
		prof.FrameRate  = 30U;
		prof.GopSize    = 3U;
		auto encR = OaVideoEncoder::Create(*engine, prof);
		ASSERT_TRUE(encR.IsOk());
		OaVideoEncoder encoder = OaStdMove(*encR);

		OaVideoMuxer::CreateInfo muxInfo;
		muxInfo.Codec     = OaVideoCodec::H264;
		muxInfo.Width     = prof.Width;
		muxInfo.Height    = prof.Height;
		muxInfo.FrameRate = prof.FrameRate;
		auto muxR = OaVideoMuxer::Create(outPath, muxInfo);
		ASSERT_TRUE(muxR.IsOk());
		OaVideoMuxer muxer = OaStdMove(*muxR);

		auto& ctx = OaContext::GetDefault();
		std::vector<OaU8> rgba(static_cast<OaUsize>(W) * H * 4U);
		bool gotConfig = false;
		for (OaU32 f = 0; f < kFrames; ++f) {
			PaintFrame(rgba, W, H, f, kFrames);
			auto texR = OaTexture::FromPixels(
				*engine,
				OaSpan<const OaU8>(rgba.data(), rgba.size()), W, H);
			ASSERT_TRUE(texR.IsOk());
			OaTexture tex = OaStdMove(*texR);

			OaEncodedFrame eframe;
			auto encodeStatus = OaFnVideo::Encode(ctx, encoder, tex, eframe, f);
			ASSERT_TRUE(encodeStatus.IsOk())
				<< "frame " << f << ": " << encodeStatus.GetMessage().CStr();
			tex.Destroy(*engine);

			if (!gotConfig && eframe.IsKeyframe) {
				auto sps = OaFnVideo::ExtractSps(
					OaSpan<const OaU8>(eframe.Bitstream.Data(), eframe.Bitstream.Size()));
				auto pps = OaFnVideo::ExtractPps(
					OaSpan<const OaU8>(eframe.Bitstream.Data(), eframe.Bitstream.Size()));
				muxer.SetCodecConfig(sps, pps);
				gotConfig = true;
			}
			ASSERT_TRUE(muxer.WritePacket(eframe).IsOk());
		}
		ASSERT_TRUE(muxer.Finalize().IsOk());
	}

	// Re-open + decode the roundtripped MP4.
	auto streamR = OaVideoStream::OpenFile(outPath);
	ASSERT_TRUE(streamR.IsOk());
	OaVideoStream stream = OaStdMove(*streamR);

	auto profile = stream.GetVideoProfile();
	profile.MaxDpbSlots = 8;
	auto decR = OaVideoDecoder::Create(*engine, profile);
	ASSERT_TRUE(decR.IsOk()) << decR.GetStatus().ToString();
	OaVideoDecoder decoder = OaStdMove(*decR);
	auto& decodeContext = OaContext::GetDefault();

	// Re-record the decoder-owned RGBA images directly. This exercises the
	// image-backed OaVideoFrame -> sampled image -> NV12 -> encode contract
	// without a host pixel readback or a buffer staging API at the call site.
	const char* imageOutPath = "/tmp/oa_roundtrip_image.mp4";
	OaVideoRecorderConfig imageRecorderConfig;
	imageRecorderConfig.OutputPath = imageOutPath;
	imageRecorderConfig.Encode.Codec = OaVideoCodec::H264;
	imageRecorderConfig.Encode.Width = static_cast<OaU32>(W);
	imageRecorderConfig.Encode.Height = static_cast<OaU32>(H);
	imageRecorderConfig.Encode.FrameRate = 30U;
	imageRecorderConfig.Encode.GopSize = 3U;
	auto imageRecorderResult = OaVideoRecorder::Create(*engine, imageRecorderConfig);
	ASSERT_TRUE(imageRecorderResult.IsOk()) << imageRecorderResult.GetStatus().ToString();
	OaVideoRecorder imageRecorder = OaStdMove(*imageRecorderResult);

	OaU32 decoded = 0;
	for (OaU32 i = 0; i < kFrames; ++i) {
		OaVideoPacket pkt{};
		ASSERT_TRUE(stream.ReadNextPacket(pkt).IsOk());

		auto frameResult = OaFnVideo::Decode(
			decodeContext,
			decoder,
			OaSpan<const OaU8>(pkt.Data.Data(), pkt.Data.Size()),
			pkt.PresentationTimestamp);
		ASSERT_TRUE(frameResult.IsOk()) << "decode packet " << i << ": "
			<< frameResult.GetStatus().ToString();
		OaVideoFrame fr = *frameResult;
		auto rgbResult = OaFnVideo::Convert(decoder, fr);
		ASSERT_TRUE(rgbResult.IsOk()) << "convert packet " << i << ": "
			<< rgbResult.GetStatus().ToString();
		OaVideoFrame rgb = *rgbResult;
		OaTexture renderTarget;
		renderTarget.Image = rgb.Image;
		renderTarget.View = rgb.ImageView;
		renderTarget.Format = static_cast<OaI32>(rgb.Format);
		renderTarget.Layout = static_cast<OaI32>(rgb.Layout);
		renderTarget.Width = static_cast<OaI32>(rgb.Width);
		renderTarget.Height = static_cast<OaI32>(rgb.Height);
		const OaU64 pts = static_cast<OaU64>(i) * 1'000'000ULL / 30ULL;
		auto renderFrame = OaFnVideo::FromTexture(
			renderTarget, pts, rgb.Ready);
		ASSERT_TRUE(renderFrame.IsOk()) << renderFrame.GetStatus().ToString();
		OaCompletionToken inputConsumed;
		auto recordStatus = imageRecorder.WriteAsync(*renderFrame, inputConsumed);
		ASSERT_TRUE(recordStatus.IsOk()) << "record image " << i << ": "
			<< recordStatus.ToString();
		ASSERT_TRUE(inputConsumed.IsValid());
		ASSERT_TRUE(inputConsumed.Wait().IsOk());
		++decoded;
	}
	EXPECT_EQ(decoded, kFrames);
	ASSERT_TRUE(imageRecorder.Finalize().IsOk());
	if (std::system("command -v ffmpeg >/dev/null 2>&1") == 0) {
		const std::string command = std::string("ffmpeg -xerror -v error -y -i \"")
			+ imageOutPath + "\" -f rawvideo -pix_fmt rgba /dev/null >/dev/null 2>&1";
		EXPECT_EQ(std::system(command.c_str()), 0)
			<< "Independent FFmpeg decoder rejected the image-backed recording";
	}

	std::remove(outPath);
	std::remove(imageOutPath);
}


// Native HEVC proof: Vulkan encode -> hvc1/hvcC MP4 -> OA demux/decode,
// followed by an independent FFmpeg decode of the same file.
TEST(OaVideoRoundtrip, EncodeMuxDemuxDecodeH265)
{
	auto* engine = OaEngine::GetGlobal();
	if (engine == nullptr) GTEST_SKIP() << "No Vulkan engine available";
	if (not OaVideoEncoder::IsCodecSupported(*engine, OaVideoCodec::H265)
		|| not OaVideoDecoder::IsCodecSupported(*engine, OaVideoCodec::H265)) {
		GTEST_SKIP() << "H.265 encode+decode not both supported";
	}

	constexpr OaI32 width = 320;
	constexpr OaI32 height = 192;
	constexpr OaU32 frameCount = 6U;
	constexpr OaU32 frameRate = 30U;
	const char* path = "/tmp/oa_roundtrip_h265.mp4";

	OaVideoRecorderConfig config;
	config.OutputPath = path;
	config.Encode.Codec = OaVideoCodec::H265;
	config.Encode.Width = static_cast<OaU32>(width);
	config.Encode.Height = static_cast<OaU32>(height);
	config.Encode.FrameRate = frameRate;
	config.Encode.GopSize = 3U;
	config.Encode.RateControl = OaVideoRateControl::ConstantQp;
	auto recorderResult = OaVideoRecorder::Create(*engine, config);
	ASSERT_TRUE(recorderResult.IsOk()) << recorderResult.GetStatus().ToString();
	OaVideoRecorder recorder = OaStdMove(*recorderResult);

	std::vector<OaU8> rgba(static_cast<OaUsize>(width) * height * 4U);
	for (OaU32 frameIndex = 0; frameIndex < frameCount; ++frameIndex) {
		PaintFrame(rgba, width, height, frameIndex, frameCount);
		auto textureResult = OaTexture::FromPixels(
			*engine, OaSpan<const OaU8>(rgba.data(), rgba.size()), width, height);
		ASSERT_TRUE(textureResult.IsOk()) << textureResult.GetStatus().ToString();
		OaTexture texture = OaStdMove(*textureResult);
		const OaU64 pts = static_cast<OaU64>(frameIndex) * 1'000'000ULL / frameRate;
		auto status = recorder.WriteRgba(texture.DeviceBuf, width, height, pts);
		texture.Destroy(*engine);
		ASSERT_TRUE(status.IsOk()) << "encode frame " << frameIndex << ": "
			<< status.ToString();
	}
	ASSERT_TRUE(recorder.Finalize().IsOk());

	auto streamResult = OaVideoStream::OpenFile(path);
	ASSERT_TRUE(streamResult.IsOk()) << streamResult.GetStatus().ToString();
	OaVideoStream stream = OaStdMove(*streamResult);
	EXPECT_EQ(stream.GetInfo().Codec, OaVideoCodec::H265);
	EXPECT_EQ(stream.GetInfo().Width, static_cast<OaU32>(width));
	EXPECT_EQ(stream.GetInfo().Height, static_cast<OaU32>(height));
	EXPECT_TRUE(stream.GetHvcConfig().Valid);

	auto profile = stream.GetVideoProfile();
	profile.MaxDpbSlots = 8U;
	auto decoderResult = OaVideoDecoder::Create(*engine, profile);
	ASSERT_TRUE(decoderResult.IsOk()) << decoderResult.GetStatus().ToString();
	OaVideoDecoder decoder = OaStdMove(*decoderResult);
	auto& decodeContext = OaContext::GetDefault();
	for (OaU32 frameIndex = 0; frameIndex < frameCount; ++frameIndex) {
		OaVideoPacket packet = {};
		ASSERT_TRUE(stream.ReadNextPacket(packet).IsOk()) << "demux frame " << frameIndex;
		ASSERT_GT(packet.Data.Size(), 4U);
		auto frameResult = OaFnVideo::Decode(
			decodeContext,
			decoder,
			OaSpan<const OaU8>(packet.Data.Data(), packet.Data.Size()),
			packet.PresentationTimestamp);
		ASSERT_TRUE(frameResult.IsOk()) << "decode frame " << frameIndex << ": "
			<< frameResult.GetStatus().ToString();
	}

	if (std::system("command -v ffmpeg >/dev/null 2>&1") == 0) {
		const std::string command = std::string("ffmpeg -xerror -v error -y -i \"")
			+ path + "\" -f rawvideo -pix_fmt rgba /dev/null >/dev/null 2>&1";
		EXPECT_EQ(std::system(command.c_str()), 0)
			<< "Independent FFmpeg decoder rejected OA's H.265 MP4";
	}
	std::remove(path);
}
