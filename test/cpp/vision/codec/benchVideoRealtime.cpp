// OA local real-time video benchmark.
//
// This target is deliberately not registered with CTest. Correctness remains
// mandatory, but performance evidence comes from seven fresh processes through
// tools/diagnostics/oaBench.py. The timed region performs no pixel readback.

#include "../../oaTest.h"
#include "../videoTestSupport.h"

#include <oa/core/filesystem.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/timer.h>
#include <oa/vision/videoDemuxer.h>
#include <oa/vision/videoPlayer.h>
#include <oa/vision/videoEncoder.h>
#include <oa/vision/video/decoder/videoDecoderInternal.h>
#include <oa/vision/video/encoder/videoEncoderInternal.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

namespace {

using Clock = oa::SteadyClock;

struct Distribution {
	double p50 = 0.0;
	double p95 = 0.0;
	double p99 = 0.0;
	double maximum = 0.0;
};

struct PendingFrame {
	oa::Event event;
	oa::SteadyTimePoint submitted;
	bool observed = false;
};

struct BenchmarkFixture {
	const char* relativePath = nullptr;
	const char* codecName = nullptr;
	oa::VideoCodec decodeCodec = oa::VideoCodec::H264;
	oa::U32 width = 0U;
	oa::U32 height = 0U;
	bool valid = false;
};

struct DecodePresentMeasurement {
	Distribution submit;
	Distribution ready;
	double throughputFps = 0.0;
	oa::U64 hardwareYcbcrConversions = 0U;
};

struct ColorConvertMeasurement {
	Distribution gpu;
	oa::U64 hardwareYcbcrConversions = 0U;
};

oa::U32 envU32(const char* inName, oa::U32 inDefault, oa::U32 inMinimum, oa::U32 inMaximum) {
	const char* value = ::getenv(inName);
	if (value == nullptr or value[0] == '\0') return inDefault;
	char* end = nullptr;
	const unsigned long parsed = ::strtoul(value, &end, 10);
	if (end == value or *end != '\0' or parsed < inMinimum or parsed > inMaximum) {
		return inDefault;
	}
	return static_cast<oa::U32>(parsed);
}

double elapsedMs(oa::SteadyTimePoint inBegin, oa::SteadyTimePoint inEnd) {
	return (inEnd - inBegin).toMilliseconds();
}

double percentile(const oa::Vector<double>& inSorted, double inPercentile) {
	if (inSorted.empty()) return 0.0;
	const double rank = oa::ceil(inPercentile * static_cast<double>(inSorted.size()));
	const oa::Usize index = static_cast<oa::Usize>(oa::max(1.0, rank)) - 1U;
	return inSorted[oa::min(index, inSorted.size() - 1U)];
}

Distribution distribution(oa::Vector<double> inSamples) {
	oa::sort(inSamples.begin(), inSamples.end());
	Distribution result;
	result.p50 = percentile(inSamples, 0.50);
	result.p95 = percentile(inSamples, 0.95);
	result.p99 = percentile(inSamples, 0.99);
	result.maximum = inSamples.empty() ? 0.0 : inSamples.back();
	return result;
}

BenchmarkFixture benchmarkFixture() {
	const char* variant = ::getenv("OA_VIDEO_BENCH_VARIANT");
	if (variant == nullptr or variant[0] == '\0'
		or ::strcmp(variant, "1080p60") == 0
		or ::strcmp(variant, "1080p60-h264") == 0) {
		return {
			.relativePath =
				"video/clip/ready_set_go_1080p_60fps_h264_high_8bit_420.mp4",
			.codecName = "h264",
			.decodeCodec = oa::VideoCodec::H264,
			.width = 1920U,
			.height = 1080U,
			.valid = true,
		};
	}
	if (::strcmp(variant, "1080p60-av1") == 0) {
		return {
			.relativePath =
				"video/clip/ready_set_go_1080p_60fps_av1_main_8bit_420.mp4",
			.codecName = "av1",
			.decodeCodec = oa::VideoCodec::AV1,
			.width = 1920U,
			.height = 1080U,
			.valid = true,
		};
	}
	if (::strcmp(variant, "1080p60-h265") == 0) {
		return {
			.relativePath =
				"video/clip/ready_set_go_1080p_60fps_h265_main_8bit_420.mp4",
			.codecName = "h265",
			.decodeCodec = oa::VideoCodec::H265,
			.width = 1920U,
			.height = 1080U,
			.valid = true,
		};
	}
	if (::strcmp(variant, "1080p60-vp9") == 0) {
		return {
			.relativePath =
				"video/clip/ready_set_go_1080p_60fps_vp9_profile0_8bit_420.mp4",
			.codecName = "vp9",
			.decodeCodec = oa::VideoCodec::VP9,
			.width = 1920U,
			.height = 1080U,
			.valid = true,
		};
	}
	if (::strcmp(variant, "2160p60-av1") == 0) {
		return {
			.relativePath =
				"video/clip/ready_set_go_2160p_60fps_av1_main_8bit_420.mp4",
			.codecName = "av1",
			.decodeCodec = oa::VideoCodec::AV1,
			.width = 3840U,
			.height = 2160U,
			.valid = true,
		};
	}
	if (::strcmp(variant, "2160p60-h264") == 0) {
		return {
			.relativePath =
				"video/clip/ready_set_go_2160p_60fps_h264_high_8bit_420.mp4",
			.codecName = "h264",
			.decodeCodec = oa::VideoCodec::H264,
			.width = 3840U,
			.height = 2160U,
			.valid = true,
		};
	}
	if (::strcmp(variant, "uhd60") == 0
		or ::strcmp(variant, "2160p60-h265") == 0) {
		return {
			.relativePath =
				"video/clip/ready_set_go_2160p_60fps_h265_main_8bit_420.mp4",
			.codecName = "h265",
			.decodeCodec = oa::VideoCodec::H265,
			.width = 3840U,
			.height = 2160U,
			.valid = true,
		};
	}
	if (::strcmp(variant, "2160p60-vp9") == 0) {
		return {
			.relativePath =
				"video/clip/ready_set_go_2160p_60fps_vp9_profile0_8bit_420.mp4",
			.codecName = "vp9",
			.decodeCodec = oa::VideoCodec::VP9,
			.width = 3840U,
			.height = 2160U,
			.valid = true,
		};
	}
	return {};
}

oa::String benchmarkAsset(const BenchmarkFixture& inFixture) {
	if (const char* path = ::getenv("OA_VIDEO_BENCH_ASSET")) {
		if (path[0] != '\0') return oa::String(path);
	}
	return oa::Paths::asset(inFixture.relativePath).string();
}

oa::U64 assetBytes(const oa::String& inAsset) {
	auto bytes = oa::Filesystem::getFileSize(oa::Path(inAsset));
	return bytes.isOk() ? static_cast<oa::U64>(*bytes) : 0U;
}

bool ffmpegAvailable() {
	return ::system("ffmpeg -version >/dev/null 2>&1") == 0;
}

oa::Result<oa::VideoPlayer> openPlayer(
	oa::Engine& inEngine,
	const oa::String& inAsset,
	oa::U32 inPresentationCacheFrames = 0U,
	const char* inYcbcrMode = nullptr)
{
	oa::VideoPlayerConfig config;
	config.uri = inAsset;
	config.loop = false;
	config.startPlaying = false;
	config.audio = false;
	// The low-latency corpus contains no B frames or alternate-reference delay.
	// Retaining a
	// four-frame reorder window would benchmark movie-playback policy rather
	// than the admitted real-time path.
	config.reorderDepth = 0U;
	config.presentationCacheFrames = inPresentationCacheFrames;
	config.presentationCacheBytes = inPresentationCacheFrames > 0U
		? 256ULL * 1024ULL * 1024ULL : 0U;
	const char* conversion = inYcbcrMode != nullptr
		? inYcbcrMode
		: ::getenv("OA_VIDEO_BENCH_YCBCR");
	config.preferHardwareYCbCr = conversion == nullptr
		or ::strcmp(conversion, "compute") != 0;
	config.filter = oa::Filter::Nearest;
	return oa::VideoPlayer::open(inEngine, config);
}

const char* requestedYcbcrMode() {
	const char* conversion = ::getenv("OA_VIDEO_BENCH_YCBCR");
	return conversion != nullptr and ::strcmp(conversion, "compute") == 0
		? "compute"
		: "hardware";
}

oa::Status advance(oa::VideoPlayer& inPlayer) {
	OA_RETURN_IF_ERROR(inPlayer.next());
	if (inPlayer.isDone()) {
		return oa::Status::error(
			oa::StatusCode::OutOfRange,
			"video benchmark source ended before the requested frame count");
	}
	const oa::VideoFrame& frame = inPlayer.currentFrame();
	if (frame.resource != oa::VideoFrameResource::Image
		or frame.image == VK_NULL_HANDLE
		or frame.imageView == VK_NULL_HANDLE
		or not frame.isRgb) {
		return oa::Status::error(
			oa::StatusCode::Internal,
			"video benchmark expected an image-backed RGBA frame");
	}
	return oa::Status::ok();
}

void observeCompletions(
	oa::Vector<PendingFrame>& inOutPending,
	oa::Vector<double>& outCompletionMs)
{
	const auto now = Clock::now();
	for (PendingFrame& pending : inOutPending) {
		if (pending.observed or not pending.event.isComplete()) continue;
		pending.observed = true;
		outCompletionMs.pushBack(elapsedMs(pending.submitted, now));
	}
}

oa::Status finishCompletions(
	oa::Vector<PendingFrame>& inOutPending,
	oa::Vector<double>& outCompletionMs)
{
	for (PendingFrame& pending : inOutPending) {
		if (pending.observed) continue;
		OA_RETURN_IF_ERROR(pending.event.wait());
		pending.observed = true;
		outCompletionMs.pushBack(elapsedMs(pending.submitted, Clock::now()));
	}
	return oa::Status::ok();
}

oa::Status warmDecode(oa::VideoPlayer& inPlayer, oa::U32 inFrames) {
	for (oa::U32 frame = 0U; frame < inFrames; ++frame) {
		OA_RETURN_IF_ERROR(advance(inPlayer));
		OA_RETURN_IF_ERROR(inPlayer.currentFrame().ready.wait());
	}
	return oa::Status::ok();
}

oa::Status requireFixture60(
	const oa::VideoPlayer& inPlayer,
	const BenchmarkFixture& inFixture);

oa::Result<DecodePresentMeasurement> measureDecodePresent(
	oa::VideoPlayer& inPlayer,
	oa::U32 inWarmup,
	oa::U32 inFrames)
{
	OA_RETURN_IF_ERROR(warmDecode(inPlayer, inWarmup));
	oa::Vector<double> submitMs;
	oa::Vector<double> completionMs;
	oa::Vector<PendingFrame> pending;
	submitMs.reserve(inFrames);
	completionMs.reserve(inFrames);
	pending.reserve(inFrames);
	const auto pipelineBegin = Clock::now();
	for (oa::U32 frame = 0U; frame < inFrames; ++frame) {
		const auto begin = Clock::now();
		OA_RETURN_IF_ERROR(advance(inPlayer));
		const auto submitted = Clock::now();
		submitMs.pushBack(elapsedMs(begin, submitted));
		pending.pushBack(PendingFrame{
			.event = inPlayer.currentFrame().ready,
			.submitted = begin,
		});
		observeCompletions(pending, completionMs);
	}
	OA_RETURN_IF_ERROR(finishCompletions(pending, completionMs));
	const auto pipelineEnd = Clock::now();
	if (completionMs.size() != inFrames) {
		return oa::Status::error(
			oa::StatusCode::Internal,
			"video benchmark did not observe every frame completion");
	}
	const double wallMs = elapsedMs(pipelineBegin, pipelineEnd);
	return DecodePresentMeasurement{
		.submit = distribution(oa::move(submitMs)),
		.ready = distribution(oa::move(completionMs)),
		.throughputFps = static_cast<double>(inFrames) * 1000.0 / wallMs,
		.hardwareYcbcrConversions =
			inPlayer.getPlaybackStats().hardwareYcbcrConversions,
	};
}

oa::Result<DecodePresentMeasurement> runDecodePresent(
	oa::Engine& inEngine,
	const BenchmarkFixture& inFixture,
	const oa::String& inAsset,
	const char* inYcbcrMode,
	oa::U32 inWarmup,
	oa::U32 inFrames,
	oa::U32 inPrimeFrames = 0U)
{
	auto playerResult = openPlayer(inEngine, inAsset, 0U, inYcbcrMode);
	if (not playerResult.isOk()) return playerResult.getStatus();
	oa::VideoPlayer player = oa::move(*playerResult);
	const oa::Status fixtureStatus = requireFixture60(player, inFixture);
	if (not fixtureStatus.isOk()) {
		(void) player.close();
		return fixtureStatus;
	}
	if (inPrimeFrames > 0U) {
		const oa::Status primeStatus = warmDecode(player, inPrimeFrames);
		if (not primeStatus.isOk()) {
			(void) player.close();
			return primeStatus;
		}
		const oa::Status rewindStatus = player.seekFrame(0U);
		if (not rewindStatus.isOk()) {
			(void) player.close();
			return rewindStatus;
		}
	}
	auto measurement = measureDecodePresent(player, inWarmup, inFrames);
	const oa::Status closeStatus = player.close();
	if (not measurement.isOk()) return measurement.getStatus();
	if (not closeStatus.isOk()) return closeStatus;
	return measurement;
}

oa::Result<oa::VideoFrame> decodeFirstVisibleFrame(
	oa::VideoDemuxer& inDemuxer,
	oa::VideoDecoder& inDecoder)
{
	for (oa::U32 packetIndex = 0U; packetIndex < 64U; ++packetIndex) {
		oa::VideoPacket packet;
		OA_RETURN_IF_ERROR(inDemuxer.readNextPacket(packet));
		auto frame = inDecoder.decode(
			oa::Span<const oa::U8>(packet.data.data(), packet.data.size()),
			packet.presentationTimestamp);
		if (not frame.isOk()) return frame.getStatus();
		if (frame->imageView != VK_NULL_HANDLE and frame->shown) {
			OA_RETURN_IF_ERROR(frame->ready.wait());
			return frame;
		}
	}
	return oa::Status::error(
		oa::StatusCode::DataLoss,
		"video color-conversion benchmark found no visible frame");
}

oa::Result<ColorConvertMeasurement> measureColorConvertGpu(
	oa::Engine& inEngine,
	oa::VideoDecoder& inDecoder,
	const oa::VideoFrame& inYcbcrFrame,
	const oa::VideoFrame& inRgbTarget,
	bool inPreferHardware,
	oa::U32 inWarmup,
	oa::U32 inSamples)
{
	oa::VideoConversionOptions options;
	options.preferHardwareYCbCr = inPreferHardware;
	options.filter = oa::Filter::Nearest;
	const oa::U64 hardwareBefore =
		oa::VideoDecoderInternal::getHardwareYcbcrDispatchCount(inDecoder);

	for (oa::U32 sample = 0U; sample < inWarmup; ++sample) {
		auto completion = inDecoder.convertIntoAsync(
			inYcbcrFrame, options, inRgbTarget);
		if (not completion.isOk()) return completion.getStatus();
		OA_RETURN_IF_ERROR(completion->wait());
		OA_RETURN_IF_ERROR(inDecoder.waitForCompletion());
	}

	oa::Timer timer(oa::TimerDomain::Device, "video.color_convert");
	OA_RETURN_IF_ERROR(timer.init(inEngine));
	oa::Vector<double> gpuMs;
	gpuMs.reserve(inSamples);
	for (oa::U32 sample = 0U; sample < inSamples; ++sample) {
		auto completion = oa::VideoDecoderInternal::convertIntoAsyncProfiled(
			inDecoder, inYcbcrFrame, options, inRgbTarget, timer);
		if (not completion.isOk()) return completion.getStatus();
		OA_RETURN_IF_ERROR(completion->wait());
		auto elapsed = timer.commit(inEngine);
		if (not elapsed.isOk()) return elapsed.getStatus();
		if (*elapsed <= 0.0) {
			return oa::Status::error(
				oa::StatusCode::Internal,
				"video color-conversion benchmark returned a non-positive GPU timestamp");
		}
		gpuMs.pushBack(*elapsed);
		OA_RETURN_IF_ERROR(inDecoder.waitForCompletion());
	}
	const oa::U64 hardwareAfter =
		oa::VideoDecoderInternal::getHardwareYcbcrDispatchCount(inDecoder);
	return ColorConvertMeasurement{
		.gpu = distribution(oa::move(gpuMs)),
		.hardwareYcbcrConversions = hardwareAfter - hardwareBefore,
	};
}

oa::Status requireFixture60(
	const oa::VideoPlayer& inPlayer,
	const BenchmarkFixture& inFixture)
{
	if (not inFixture.valid) {
		return oa::Status::error(
			oa::StatusCode::InvalidArgument,
			"OA_VIDEO_BENCH_VARIANT is not a known 60 fps video fixture");
	}
	if (inPlayer.width() != inFixture.width or inPlayer.height() != inFixture.height
		or inPlayer.frameRate() != 60U) {
		return oa::Status::error(
			oa::StatusCode::InvalidArgument,
			"video real-time source does not match its selected fixture contract");
	}
	return oa::Status::ok();
}

oa::Status requireVideoCapabilities(
	oa::Engine& inEngine,
	const BenchmarkFixture& inFixture,
	const oa::String& inAsset,
	bool inRequireEncode)
{
	auto demuxerResult = oa::VideoDemuxer::open(inAsset);
	if (not demuxerResult.isOk()) return demuxerResult.getStatus();
	const oa::VideoProfile exactProfile = demuxerResult->getVideoProfile();
	if (exactProfile.codec != inFixture.decodeCodec
		or exactProfile.width != inFixture.width
		or exactProfile.height != inFixture.height) {
		return oa::Status::error(
			oa::StatusCode::InvalidArgument,
			"selected video fixture does not match its declared codec and extent");
	}
	auto decodeResult = oa::VideoDecoder::queryDecodeCapabilities(
		inEngine,
		exactProfile);
	if (not decodeResult.isOk()) return decodeResult.getStatus();
	const oa::VideoDecodeCapabilities& decode = *decodeResult;
	if (not decode.supported
		or decode.maxWidth < inFixture.width
		or decode.maxHeight < inFixture.height) {
		return oa::Status::error(
			oa::StatusCode::Unavailable,
			"selected device does not admit the decode fixture extent");
	}

	oa::U32 encodeMaxWidth = 0U;
	oa::U32 encodeMaxHeight = 0U;
	oa::U32 encodeMaxLevel = 0U;
	if (inRequireEncode) {
		auto encodeResult = oa::VideoEncoder::queryEncodeCapabilities(
			inEngine,
			oa::VideoCodec::H264);
		if (not encodeResult.isOk()) return encodeResult.getStatus();
		const oa::VideoEncodeCapabilities& encode = *encodeResult;
		encodeMaxWidth = encode.maxWidth;
		encodeMaxHeight = encode.maxHeight;
		encodeMaxLevel = encode.maxLevel;
		if (not encode.supported
			or encode.maxWidth < inFixture.width
			or encode.maxHeight < inFixture.height) {
			return oa::Status::error(
				oa::StatusCode::Unavailable,
				"selected device does not admit the H.264 encode fixture extent");
		}
	}

	OA_RETURN_IF_ERROR(oa::print(
		"OABENCH video.realtime.capabilities decode_codec={} requested={}x{} "
		"decode_max={}x{} decode_max_level={} "
		"encode_required={} encode_max={}x{} encode_max_level={}",
		inFixture.codecName,
		inFixture.width,
		inFixture.height,
		decode.maxWidth,
		decode.maxHeight,
		decode.maxLevel,
		inRequireEncode ? 1U : 0U,
		encodeMaxWidth,
		encodeMaxHeight,
		encodeMaxLevel));
	return oa::Status::ok();
}

oa::Result<oa::VideoEncoder> createH264Encoder(
	oa::Engine& inEngine,
	oa::U32 inWidth,
	oa::U32 inHeight,
	oa::U32 inFrameRate,
	oa::U32 inAsyncDepth)
{
	oa::VideoEncodeProfile profile;
	profile.codec = oa::VideoCodec::H264;
	profile.width = inWidth;
	profile.height = inHeight;
	profile.frameRate = inFrameRate;
	profile.gopSize = inFrameRate;
	profile.maxBFrames = 0U;
	profile.rateControl = oa::VideoRateControl::ConstantQp;
	profile.constantQp = 26U;
	profile.asyncDepth = inAsyncDepth;
	return oa::VideoEncoder::create(inEngine, profile);
}

oa::Status submitCurrentFrame(
	oa::VideoPlayer& inPlayer,
	oa::VideoEncoder& inEncoder,
	oa::U64 inPts,
	oa::Vector<oa::EncodedVideoPacket>& outReady)
{
	const oa::VideoFrame& frame = inPlayer.currentFrame();
	oa::Event consumed;
	OA_RETURN_IF_ERROR(oa::VideoEncoderAccess::submitRgbaImage(
		inEncoder,
		frame.image,
		frame.imageView,
		frame.format,
		frame.layout,
		frame.width,
		frame.height,
		inPts,
		outReady,
		frame.colorSpace,
		frame.fullRange,
		frame.arrayLayer,
		frame.ready,
		frame.externalQueueFamilyIndex,
		&consumed));
	if (not consumed.isValid()) {
		return oa::Status::error(
			oa::StatusCode::Internal,
			"image-backed encoder submit did not return an input-consumed event");
	}
	inPlayer.markCurrentFrameConsumed(consumed);
	return oa::Status::ok();
}

oa::Status warmPipeline(
	oa::Engine& inEngine,
	oa::VideoPlayer& inPlayer,
	oa::U32 inFrames,
	oa::U32 inFrameRate,
	oa::U32 inAsyncDepth)
{
	auto encoderResult = createH264Encoder(
		inEngine, inPlayer.width(), inPlayer.height(), inFrameRate, inAsyncDepth);
	if (not encoderResult.isOk()) return encoderResult.getStatus();
	oa::VideoEncoder encoder = oa::move(*encoderResult);
	oa::U32 outputCount = 0U;
	for (oa::U32 frame = 0U; frame < inFrames; ++frame) {
		OA_RETURN_IF_ERROR(advance(inPlayer));
		oa::Vector<oa::EncodedVideoPacket> ready;
		OA_RETURN_IF_ERROR(submitCurrentFrame(
			inPlayer, encoder,
			static_cast<oa::U64>(frame) * 1'000'000ULL / inFrameRate,
			ready));
		outputCount += static_cast<oa::U32>(ready.size());
	}
	oa::Vector<oa::EncodedVideoPacket> drained;
	OA_RETURN_IF_ERROR(encoder.flush(drained));
	outputCount += static_cast<oa::U32>(drained.size());
	if (outputCount != inFrames) {
		return oa::Status::error(
			oa::StatusCode::DataLoss,
			"warm video pipeline did not produce one packet per submitted frame");
	}
	return encoder.close();
}

oa::Path oraclePath() {
#if defined(_WIN32)
	const oa::U64 processId = static_cast<oa::U64>(_getpid());
#else
	const oa::U64 processId = static_cast<oa::U64>(getpid());
#endif
	return oa::Paths::temp()
		/ oa::Path(oa::format("oa-video-realtime-{}.h264", processId));
}

bool validateWithFfmpeg(
	const oa::Vector<oa::EncodedVideoPacket>& inPackets,
	oa::U32 inExpectedFrames)
{
	const oa::Path path = oraclePath();
	FILE* file = ::fopen(path.cStr(), "wb");
	if (file == nullptr) return false;
	bool writeOk = true;
	for (const oa::EncodedVideoPacket& packet : inPackets) {
		if (packet.frameSize != packet.bitstream.size() or packet.frameSize <= 4U) {
			writeOk = false;
			break;
		}
		writeOk = ::fwrite(
			packet.bitstream.data(), 1U, packet.bitstream.size(), file)
			== packet.bitstream.size();
		if (not writeOk) break;
	}
	writeOk = ::fclose(file) == 0 and writeOk;
	if (not writeOk) {
		(void) oa::Filesystem::removeFile(path);
		return false;
	}
	const oa::String command = oa::format(
		"ffmpeg -v error -f h264 -i \"{}\" -frames:v {} -f null - >/dev/null 2>&1",
		path.string(),
		inExpectedFrames);
	const bool decoded = ::system(command.cStr()) == 0;
	(void) oa::Filesystem::removeFile(path);
	return decoded;
}

} // namespace

class BenchVideoRealtime : public ::testing::Test {};

TEST_VK(BenchVideoRealtime, DecodePresentWall) {
	if (not vkTestEngineOk()) GTEST_SKIP() << "no vulkan device";
	const BenchmarkFixture fixture = benchmarkFixture();
	ASSERT_TRUE(fixture.valid) << "OA_VIDEO_BENCH_VARIANT is not a known 60 fps video fixture";
	const oa::String asset = benchmarkAsset(fixture);
	if (not oa::Filesystem::exists(oa::Path(asset))) {
		GTEST_SKIP() << "video benchmark asset is unavailable: " << asset;
	}
	const oa::U32 warmup = envU32("OA_VIDEO_BENCH_WARMUP", 8U, 0U, 120U);
	const oa::U32 frames = envU32("OA_VIDEO_BENCH_FRAMES", 288U, 8U, 600U);
	const oa::Status capabilities =
		requireVideoCapabilities(testEngine(), fixture, asset, false);
	ASSERT_TRUE(capabilities.isOk()) << capabilities.toString();
	auto measured = runDecodePresent(
		testEngine(), fixture, asset, requestedYcbcrMode(), warmup, frames);
	ASSERT_TRUE(measured.isOk()) << measured.getStatus().toString();
	const DecodePresentMeasurement& measurement = *measured;
	if (::strcmp(requestedYcbcrMode(), "compute") == 0) {
		ASSERT_EQ(measurement.hardwareYcbcrConversions, 0U);
	}
	ASSERT_TRUE(oa::print(
		"OABENCH video.realtime.decode_present "
		"width={} height={} source_fps={} frames={} asset_bytes={} "
		"submit_p50_ms={:.6f} submit_p95_ms={:.6f} submit_p99_ms={:.6f} submit_max_ms={:.6f} "
		"ready_p50_ms={:.6f} ready_p95_ms={:.6f} ready_p99_ms={:.6f} ready_max_ms={:.6f} "
		"throughput_fps={:.3f} requested_ycbcr={} hardware_ycbcr_conversions={} "
		"pixel_readback_bytes=0",
		fixture.width, fixture.height, 60U, frames,
		assetBytes(asset),
		measurement.submit.p50,
		measurement.submit.p95,
		measurement.submit.p99,
		measurement.submit.maximum,
		measurement.ready.p50,
		measurement.ready.p95,
		measurement.ready.p99,
		measurement.ready.maximum,
		measurement.throughputFps,
		requestedYcbcrMode(), measurement.hardwareYcbcrConversions).isOk());
}

TEST_VK(BenchVideoRealtime, DecodePresentYcbcrPair) {
	if (not vkTestEngineOk()) GTEST_SKIP() << "no vulkan device";
	const BenchmarkFixture fixture = benchmarkFixture();
	ASSERT_TRUE(fixture.valid) << "OA_VIDEO_BENCH_VARIANT is not a known 60 fps video fixture";
	const oa::String asset = benchmarkAsset(fixture);
	if (not oa::Filesystem::exists(oa::Path(asset))) {
		GTEST_SKIP() << "video benchmark asset is unavailable: " << asset;
	}
	const oa::Status capabilities =
		requireVideoCapabilities(testEngine(), fixture, asset, false);
	ASSERT_TRUE(capabilities.isOk()) << capabilities.toString();
	const oa::U32 warmup = envU32("OA_VIDEO_BENCH_WARMUP", 8U, 0U, 120U);
	const oa::U32 frames = envU32("OA_VIDEO_BENCH_FRAMES", 288U, 8U, 600U);
	const oa::U32 primeFrames = envU32(
		"OA_VIDEO_BENCH_PRIME_FRAMES", 120U, 0U, 240U);
	const char* orderEnv = ::getenv("OA_VIDEO_BENCH_PAIR_ORDER");
	const bool computeFirst = orderEnv != nullptr
		and ::strcmp(orderEnv, "compute-first") == 0;
	const char* firstMode = computeFirst ? "compute" : "hardware";
	const char* secondMode = computeFirst ? "hardware" : "compute";
	auto first = runDecodePresent(
		testEngine(), fixture, asset, firstMode, warmup, frames, primeFrames);
	ASSERT_TRUE(first.isOk()) << first.getStatus().toString();
	auto second = runDecodePresent(
		testEngine(), fixture, asset, secondMode, warmup, frames, primeFrames);
	ASSERT_TRUE(second.isOk()) << second.getStatus().toString();
	const DecodePresentMeasurement& hardware = computeFirst ? *second : *first;
	const DecodePresentMeasurement& compute = computeFirst ? *first : *second;
	ASSERT_GT(hardware.hardwareYcbcrConversions, 0U);
	ASSERT_EQ(compute.hardwareYcbcrConversions, 0U);
	const double hardwareOverCompute =
		hardware.throughputFps / compute.throughputFps;
	ASSERT_TRUE(oa::print(
		"OABENCH video.realtime.decode_present_ycbcr_pair "
		"width={} height={} source_fps=60 prime_frames_per_path={} "
		"frames_per_path={} asset_bytes={} "
		"order={} hardware_fps={:.3f} compute_fps={:.3f} "
		"hardware_over_compute={:.6f} hardware_ycbcr_conversions={} "
		"compute_ycbcr_conversions={} pixel_readback_bytes=0",
		fixture.width,
		fixture.height,
		primeFrames,
		frames,
		assetBytes(asset),
		computeFirst ? "compute-first" : "hardware-first",
		hardware.throughputFps,
		compute.throughputFps,
		hardwareOverCompute,
		hardware.hardwareYcbcrConversions,
		compute.hardwareYcbcrConversions).isOk());
}

TEST_VK(BenchVideoRealtime, ColorConvertGpuPair) {
	if (not vkTestEngineOk()) GTEST_SKIP() << "no vulkan device";
	const BenchmarkFixture fixture = benchmarkFixture();
	ASSERT_TRUE(fixture.valid)
		<< "OA_VIDEO_BENCH_VARIANT is not a known 60 fps video fixture";
	const oa::String asset = benchmarkAsset(fixture);
	if (not oa::Filesystem::exists(oa::Path(asset))) {
		GTEST_SKIP() << "video benchmark asset is unavailable: " << asset;
	}
	ASSERT_TRUE(requireVideoCapabilities(testEngine(), fixture, asset, false).isOk());

	auto demuxerResult = oa::VideoDemuxer::open(asset);
	ASSERT_TRUE(demuxerResult.isOk()) << demuxerResult.getStatus().toString();
	oa::VideoDemuxer demuxer = oa::move(*demuxerResult);
	auto decoderResult = oa::VideoDecoder::create(
		testEngine(), demuxer.getVideoProfile());
	ASSERT_TRUE(decoderResult.isOk()) << decoderResult.getStatus().toString();
	oa::VideoDecoder decoder = oa::move(*decoderResult);
	auto sourceResult = decodeFirstVisibleFrame(demuxer, decoder);
	ASSERT_TRUE(sourceResult.isOk()) << sourceResult.getStatus().toString();
	const oa::VideoFrame source = *sourceResult;

	auto hardwareTargetResult = decoder.allocateRgbaFrame(
		fixture.width, fixture.height);
	ASSERT_TRUE(hardwareTargetResult.isOk())
		<< hardwareTargetResult.getStatus().toString();
	oa::VideoFrame hardwareTarget = *hardwareTargetResult;
	auto computeTargetResult = decoder.allocateRgbaFrame(
		fixture.width, fixture.height);
	ASSERT_TRUE(computeTargetResult.isOk())
		<< computeTargetResult.getStatus().toString();
	oa::VideoFrame computeTarget = *computeTargetResult;

	const oa::U32 warmup = envU32(
		"OA_VIDEO_CONVERT_BENCH_WARMUP", 120U, 1U, 240U);
	const oa::U32 samples = envU32(
		"OA_VIDEO_CONVERT_BENCH_SAMPLES", 256U, 7U, 1200U);
	const char* order = ::getenv("OA_VIDEO_BENCH_PAIR_ORDER");
	const bool computeFirst = order != nullptr
		and ::strcmp(order, "compute-first") == 0;

	oa::Result<ColorConvertMeasurement> first = computeFirst
		? measureColorConvertGpu(
			testEngine(), decoder, source, computeTarget, false, warmup, samples)
		: measureColorConvertGpu(
			testEngine(), decoder, source, hardwareTarget, true, warmup, samples);
	ASSERT_TRUE(first.isOk()) << first.getStatus().toString();
	oa::Result<ColorConvertMeasurement> second = computeFirst
		? measureColorConvertGpu(
			testEngine(), decoder, source, hardwareTarget, true, warmup, samples)
		: measureColorConvertGpu(
			testEngine(), decoder, source, computeTarget, false, warmup, samples);
	ASSERT_TRUE(second.isOk()) << second.getStatus().toString();
	const ColorConvertMeasurement& hardware = computeFirst ? *second : *first;
	const ColorConvertMeasurement& compute = computeFirst ? *first : *second;
	ASSERT_EQ(hardware.hardwareYcbcrConversions, warmup + samples);
	ASSERT_EQ(compute.hardwareYcbcrConversions, 0U);

	auto hardwareRgba = decoder.readbackRgba(hardwareTarget);
	ASSERT_TRUE(hardwareRgba.isOk()) << hardwareRgba.getStatus().toString();
	auto computeRgba = decoder.readbackRgba(computeTarget);
	ASSERT_TRUE(computeRgba.isOk()) << computeRgba.getStatus().toString();
	ASSERT_EQ(hardwareRgba->size(), computeRgba->size());
	oa::U64 absoluteError = 0U;
	oa::U32 maximumError = 0U;
	for (oa::Usize pixel = 0U; pixel < hardwareRgba->size(); pixel += 4U) {
		ASSERT_EQ((*hardwareRgba)[pixel + 3U], 255U);
		ASSERT_EQ((*computeRgba)[pixel + 3U], 255U);
		for (oa::Usize channel = 0U; channel < 3U; ++channel) {
			const oa::U8 hardwareValue = (*hardwareRgba)[pixel + channel];
			const oa::U8 computeValue = (*computeRgba)[pixel + channel];
			const oa::U32 error = hardwareValue > computeValue
				? hardwareValue - computeValue
				: computeValue - hardwareValue;
			absoluteError += error;
			maximumError = oa::max(maximumError, error);
		}
	}
	const double rgbMae = static_cast<double>(absoluteError)
		/ static_cast<double>(fixture.width * fixture.height * 3U);
	ASSERT_LT(rgbMae, 6.0);
	ASSERT_LT(maximumError, 96U);

	const double hardwareOverCompute =
		compute.gpu.p50 / hardware.gpu.p50;
	ASSERT_TRUE(oa::print(
		"OABENCH video.realtime.color_convert_gpu "
		"width={} height={} samples_per_path={} warmup_per_path={} order={} "
		"hardware_gpu_p50_us={:.3f} hardware_gpu_p95_us={:.3f} "
		"hardware_gpu_p99_us={:.3f} hardware_gpu_max_us={:.3f} "
		"compute_gpu_p50_us={:.3f} compute_gpu_p95_us={:.3f} "
		"compute_gpu_p99_us={:.3f} compute_gpu_max_us={:.3f} "
		"hardware_over_compute={:.6f} rgb_mae={:.6f} rgb_max_error={} "
		"hardware_ycbcr_conversions={} compute_ycbcr_conversions={} "
		"timestamp_scope=shader_dispatch pixel_readback_bytes_timed=0",
		fixture.width,
		fixture.height,
		samples,
		warmup,
		computeFirst ? "compute-first" : "hardware-first",
		hardware.gpu.p50 * 1000.0,
		hardware.gpu.p95 * 1000.0,
		hardware.gpu.p99 * 1000.0,
		hardware.gpu.maximum * 1000.0,
		compute.gpu.p50 * 1000.0,
		compute.gpu.p95 * 1000.0,
		compute.gpu.p99 * 1000.0,
		compute.gpu.maximum * 1000.0,
		hardwareOverCompute,
		rgbMae,
		maximumError,
		hardware.hardwareYcbcrConversions,
		compute.hardwareYcbcrConversions).isOk());
	ASSERT_TRUE(decoder.close().isOk());
}

TEST_VK(BenchVideoRealtime, PacedDecodePresent60) {
	if (not vkTestEngineOk()) GTEST_SKIP() << "no vulkan device";
	const BenchmarkFixture fixture = benchmarkFixture();
	ASSERT_TRUE(fixture.valid) << "OA_VIDEO_BENCH_VARIANT is not a known 60 fps video fixture";
	const oa::String asset = benchmarkAsset(fixture);
	if (not oa::Filesystem::exists(oa::Path(asset))) {
		GTEST_SKIP() << "video benchmark asset is unavailable: " << asset;
	}
	ASSERT_TRUE(requireVideoCapabilities(testEngine(), fixture, asset, false).isOk());
	const oa::U32 decodeAhead = envU32(
		"OA_VIDEO_BENCH_DECODE_AHEAD", 3U, 2U, 8U);
	auto playerResult = openPlayer(testEngine(), asset, decodeAhead + 1U);
	ASSERT_TRUE(playerResult.isOk()) << playerResult.getStatus().toString();
	oa::VideoPlayer player = oa::move(*playerResult);
	ASSERT_TRUE(requireFixture60(player, fixture).isOk());

	const oa::U32 warmup = envU32("OA_VIDEO_BENCH_WARMUP", 8U, 0U, 120U);
	const oa::U32 frames = envU32("OA_VIDEO_BENCH_FRAMES", 240U, 8U, 292U);
	ASSERT_TRUE(warmDecode(player, warmup).isOk());

	constexpr double frameBudgetMs = 1000.0 / 60.0;
	constexpr oa::I64 framePeriodNs = 1'000'000'000LL / 60LL;
	oa::Vector<double> queueResidenceMs;
	oa::Vector<double> latenessMs;
	oa::Vector<double> scheduleLatenessMs;
	queueResidenceMs.reserve(frames);
	latenessMs.reserve(frames);
	scheduleLatenessMs.reserve(frames);
	oa::U32 deadlineMisses = 0U;
	oa::U32 hostScheduleMisses = 0U;
	oa::Mutex pendingMutex;
	oa::Condition pendingChanged;
	oa::Array<PendingFrame, 8> pending;
	oa::U32 pendingBegin = 0U;
	oa::U32 pendingCount = 0U;
	bool stopProducer = false;
	bool producerDone = false;
	oa::Status producerStatus = oa::Status::ok();

	auto producerResult = oa::Thread::create([&] {
		for (oa::U32 frame = 0U; frame < frames; ++frame) {
			{
				oa::UniqueLock<oa::Mutex> lock(pendingMutex);
				pendingChanged.wait(lock, [&] {
					return stopProducer or pendingCount < decodeAhead;
				});
				if (stopProducer) break;
			}
			const auto submitted = Clock::now();
			if (frame > 0U) {
				const oa::Status status = advance(player);
				if (not status.isOk()) {
					oa::ScopedLock<oa::Mutex> lock(pendingMutex);
					producerStatus = status;
					producerDone = true;
					pendingChanged.notifyAll();
					return;
				}
			}
			{
				oa::ScopedLock<oa::Mutex> lock(pendingMutex);
				pending[(pendingBegin + pendingCount) % pending.size()] = PendingFrame{
					.event = player.currentFrame().ready,
					.submitted = submitted,
				};
				++pendingCount;
			}
			pendingChanged.notifyAll();
		}
		{
			oa::ScopedLock<oa::Mutex> lock(pendingMutex);
			producerDone = true;
		}
		pendingChanged.notifyAll();
	});
	ASSERT_TRUE(producerResult.isOk()) << producerResult.getStatus().toString();
	oa::Thread producer = oa::move(*producerResult);
	{
		oa::UniqueLock<oa::Mutex> lock(pendingMutex);
		pendingChanged.wait(lock, [&] {
			return pendingCount >= decodeAhead or producerDone;
		});
	}
	const oa::SteadyTimePoint scheduleBegin =
		Clock::now() + oa::Duration::fromMilliseconds(5);
	for (oa::U32 frame = 0U; frame < frames; ++frame) {
		const oa::SteadyTimePoint presentation = scheduleBegin
			+ oa::Duration::fromNanoseconds(framePeriodNs * static_cast<oa::I64>(frame));
		const oa::SteadyTimePoint beforeSleep = Clock::now();
		if (beforeSleep < presentation) {
			oa::Thread::sleepFor(presentation - beforeSleep);
		}
		const oa::SteadyTimePoint observed = Clock::now();
		const double scheduleLate = observed > presentation
			? elapsedMs(presentation, observed) : 0.0;
		scheduleLatenessMs.pushBack(scheduleLate);
		hostScheduleMisses += scheduleLate > 1.0 ? 1U : 0U;
		PendingFrame presented;
		bool queueEmptyAtDeadline = false;
		{
			oa::UniqueLock<oa::Mutex> lock(pendingMutex);
			queueEmptyAtDeadline = pendingCount == 0U;
			if (queueEmptyAtDeadline) {
				pendingChanged.wait(lock, [&] {
					return pendingCount != 0U or producerDone;
				});
			}
			if (pendingCount == 0U) {
				producerStatus = oa::Status::error(
					oa::StatusCode::DataLoss,
					"paced video producer ended before all frames were presented");
				break;
			}
			presented = oa::move(pending[pendingBegin]);
			pendingBegin = (pendingBegin + 1U) % pending.size();
			--pendingCount;
		}
		pendingChanged.notifyAll();
		const bool readyAtObservation = presented.event.isComplete();
		auto completed = observed;
		if (not readyAtObservation) {
			const oa::Status readyStatus = presented.event.wait();
			if (not readyStatus.isOk()) {
				oa::ScopedLock<oa::Mutex> lock(pendingMutex);
				producerStatus = readyStatus;
				break;
			}
			completed = Clock::now();
		}
		deadlineMisses += queueEmptyAtDeadline or not readyAtObservation ? 1U : 0U;
		queueResidenceMs.pushBack(elapsedMs(
			presented.submitted, completed));
		const double late = readyAtObservation and not queueEmptyAtDeadline
			? 0.0 : elapsedMs(presentation, completed);
		if (::getenv("OA_VIDEO_BENCH_TRACE") != nullptr) {
			ASSERT_TRUE(oa::print(
				"OATRACE video.realtime.paced frame={} ready={} "
				"queue_residence_ms={:.6f} lateness_ms={:.6f}",
				frame, readyAtObservation ? 1U : 0U,
				queueResidenceMs.back(), late).isOk());
		}
		latenessMs.pushBack(late);
	}
	const auto scheduleEnd = Clock::now();
	{
		oa::ScopedLock<oa::Mutex> lock(pendingMutex);
		stopProducer = true;
	}
	pendingChanged.notifyAll();
	ASSERT_TRUE(producer.join().isOk());
	ASSERT_TRUE(producerStatus.isOk()) << producerStatus.toString();
	ASSERT_EQ(queueResidenceMs.size(), frames);

	const Distribution queueResidence = distribution(oa::move(queueResidenceMs));
	const Distribution lateness = distribution(oa::move(latenessMs));
	const Distribution scheduleLateness = distribution(oa::move(scheduleLatenessMs));
	const double wallMs = elapsedMs(scheduleBegin, scheduleEnd);
	const double throughput = static_cast<double>(frames) * 1000.0 / wallMs;
	const double missPercent = static_cast<double>(deadlineMisses) * 100.0
		/ static_cast<double>(frames);
	ASSERT_TRUE(oa::print(
		"OABENCH video.realtime.paced_decode_present "
		"width={} height={} source_fps={} frames={} asset_bytes={} "
		"frame_budget_ms={:.6f} decode_ahead_frames={} decode_lead_ms={:.6f} "
		"queue_residence_p50_ms={:.6f} queue_residence_p95_ms={:.6f} "
		"queue_residence_p99_ms={:.6f} queue_residence_max_ms={:.6f} "
		"lateness_p50_ms={:.6f} lateness_p95_ms={:.6f} lateness_p99_ms={:.6f} "
		"lateness_max_ms={:.6f} schedule_lateness_p99_ms={:.6f} host_schedule_misses={} "
		"deadline_misses={} deadline_miss_percent={:.6f} "
		"throughput_fps={:.3f} pixel_readback_bytes=0",
		player.width(), player.height(), player.frameRate(), frames,
		assetBytes(asset), frameBudgetMs, decodeAhead,
		frameBudgetMs * static_cast<double>(decodeAhead),
		queueResidence.p50, queueResidence.p95,
		queueResidence.p99, queueResidence.maximum,
		lateness.p50, lateness.p95, lateness.p99, lateness.maximum,
		scheduleLateness.p99, hostScheduleMisses,
		deadlineMisses, missPercent, throughput).isOk());
	ASSERT_TRUE(player.close().isOk());
}

TEST_VK(BenchVideoRealtime, DirectImageDecodeConvertEncode) {
	if (not vkTestEngineOk()) GTEST_SKIP() << "no vulkan device";
	if (not ffmpegAvailable()) GTEST_SKIP() << "ffmpeg is required for the bitstream oracle";
	oa::Engine& engine = testEngine();
	if (not testVideoEncodeSupported(engine, oa::VideoCodec::H264)) {
		GTEST_SKIP() << "H.264 vulkan Video encode is unavailable";
	}
	const BenchmarkFixture fixture = benchmarkFixture();
	ASSERT_TRUE(fixture.valid) << "OA_VIDEO_BENCH_VARIANT is not a known 60 fps video fixture";
	const oa::String asset = benchmarkAsset(fixture);
	if (not oa::Filesystem::exists(oa::Path(asset))) {
		GTEST_SKIP() << "video benchmark asset is unavailable: " << asset;
	}
	ASSERT_TRUE(requireVideoCapabilities(engine, fixture, asset, true).isOk());
	auto playerResult = openPlayer(engine, asset);
	ASSERT_TRUE(playerResult.isOk()) << playerResult.getStatus().toString();
	oa::VideoPlayer player = oa::move(*playerResult);
	ASSERT_TRUE(requireFixture60(player, fixture).isOk());

	const oa::U32 warmup = envU32("OA_VIDEO_BENCH_WARMUP", 8U, 0U, 120U);
	const oa::U32 frames = envU32("OA_VIDEO_BENCH_FRAMES", 240U, 8U, 600U);
	const oa::U32 targetFps = envU32("OA_VIDEO_BENCH_TARGET_FPS", 60U, 1U, 240U);
	const oa::U32 asyncDepth = envU32("OA_VIDEO_BENCH_ASYNC_DEPTH", 3U, 1U, 16U);
	const oa::Status warmStatus = warmPipeline(
		engine, player, warmup, targetFps, asyncDepth);
	ASSERT_TRUE(warmStatus.isOk()) << warmStatus.toString();

	auto encoderResult = createH264Encoder(
		engine, player.width(), player.height(), targetFps, asyncDepth);
	ASSERT_TRUE(encoderResult.isOk()) << encoderResult.getStatus().toString();
	oa::VideoEncoder encoder = oa::move(*encoderResult);

	oa::Vector<double> decodeSubmitMs;
	oa::Vector<double> encodeSubmitMs;
	oa::Vector<double> bitstreamLatencyMs;
	oa::Vector<oa::SteadyTimePoint> frameBegins(frames);
	oa::Vector<oa::EncodedVideoPacket> packets;
	decodeSubmitMs.reserve(frames);
	encodeSubmitMs.reserve(frames);
	bitstreamLatencyMs.reserve(frames);
	packets.reserve(frames);
	const oa::U64 frameDurationUs = 1'000'000ULL / targetFps;
	const auto pipelineBegin = Clock::now();

	auto retainPackets = [&](oa::Vector<oa::EncodedVideoPacket>& inReady) {
		const auto visible = Clock::now();
		for (oa::EncodedVideoPacket& packet : inReady) {
			const oa::U64 index64 = frameDurationUs > 0U
				? packet.presentationTimestamp / frameDurationUs : 0U;
			ASSERT_LT(index64, frames);
			bitstreamLatencyMs.pushBack(elapsedMs(
				frameBegins[static_cast<oa::Usize>(index64)], visible));
			packets.pushBack(oa::move(packet));
		}
	};

	for (oa::U32 frame = 0U; frame < frames; ++frame) {
		const auto frameBegin = Clock::now();
		frameBegins[frame] = frameBegin;
		const oa::Status advanceStatus = advance(player);
		const auto decodeSubmitted = Clock::now();
		ASSERT_TRUE(advanceStatus.isOk()) << advanceStatus.toString();
		decodeSubmitMs.pushBack(elapsedMs(frameBegin, decodeSubmitted));

		oa::Vector<oa::EncodedVideoPacket> ready;
		const oa::Status submitStatus = submitCurrentFrame(
			player, encoder,
			static_cast<oa::U64>(frame) * frameDurationUs,
			ready);
		const auto encodeSubmitted = Clock::now();
		ASSERT_TRUE(submitStatus.isOk()) << submitStatus.toString();
		encodeSubmitMs.pushBack(elapsedMs(decodeSubmitted, encodeSubmitted));
		retainPackets(ready);
	}

	oa::Vector<oa::EncodedVideoPacket> drained;
	const oa::Status flushStatus = encoder.flush(drained);
	ASSERT_TRUE(flushStatus.isOk()) << flushStatus.toString();
	retainPackets(drained);
	const auto pipelineEnd = Clock::now();
	ASSERT_EQ(packets.size(), frames);
	ASSERT_EQ(bitstreamLatencyMs.size(), frames);
	ASSERT_TRUE(validateWithFfmpeg(packets, frames));

	const Distribution decodeSubmit = distribution(oa::move(decodeSubmitMs));
	const Distribution encodeSubmit = distribution(oa::move(encodeSubmitMs));
	const Distribution bitstream = distribution(oa::move(bitstreamLatencyMs));
	const double wallMs = elapsedMs(pipelineBegin, pipelineEnd);
	const double throughput = static_cast<double>(frames) * 1000.0 / wallMs;
	const oa::U64 feedbackRecoveries =
		oa::VideoEncoderAccess::zeroFeedbackRecoveryCount(encoder);
	ASSERT_TRUE(oa::print(
		"OABENCH video.realtime.direct_image_encode "
		"width={} height={} source_fps={} target_fps={} frames={} async_depth={} asset_bytes={} "
		"decode_submit_p50_ms={:.6f} decode_submit_p95_ms={:.6f} "
		"decode_submit_p99_ms={:.6f} decode_submit_max_ms={:.6f} "
		"encode_submit_p50_ms={:.6f} encode_submit_p95_ms={:.6f} "
		"encode_submit_p99_ms={:.6f} encode_submit_max_ms={:.6f} "
		"bitstream_p50_ms={:.6f} bitstream_p95_ms={:.6f} "
		"bitstream_p99_ms={:.6f} bitstream_max_ms={:.6f} "
		"throughput_fps={:.3f} pixel_cpu_copy_bytes=0 feedback_recoveries={} oracle=ffmpeg",
		player.width(), player.height(), player.frameRate(), targetFps, frames, asyncDepth,
		assetBytes(asset),
		decodeSubmit.p50, decodeSubmit.p95, decodeSubmit.p99, decodeSubmit.maximum,
		encodeSubmit.p50, encodeSubmit.p95, encodeSubmit.p99, encodeSubmit.maximum,
		bitstream.p50, bitstream.p95, bitstream.p99, bitstream.maximum,
		throughput, feedbackRecoveries).isOk());
	ASSERT_TRUE(encoder.close().isOk());
	ASSERT_TRUE(player.close().isOk());
}
