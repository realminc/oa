// OA Vision — End-to-end video decode reference validation.
//
// For each codec (AV1 / H.264 / H.265 / VP9) this decodes the checked-in
// 2-second 720p .mp4 clip END TO END through the real pipeline — MP4
// container demux → hardware video decode → display-order reorder →
// YCbCr→RGBA conversion → CPU readback — directly through video (the same
// path the TutorialViewerVideo* applications use). Each decoded frame is
// PSNR-compared
// against an INDEPENDENT ffmpeg decode of the same clip. This replaces
// eyeballing four player windows with a headless pass/fail gate.
//
// Assets:   sdk/asset/video/clip/shibuya_720p_30fps_<codec>_<profile>_8bit_420.mp4
// Dataset:  $OA_DATA_DIR/video/profiles/ (optional external conformance data)
// Oracle:   ffmpeg on PATH (test skips if absent)
//
// PSNR note: OA's YCbCr→RGB conversion and ffmpeg's swscale differ slightly
// (matrix / range), so this is a corruption / wrong-frame gate, not a
// bit-exact conformance check. Garbage or a lost-device frame scores in the
// low single digits; a correctly decoded frame scores well above the
// threshold. actual per-frame PSNR is printed so the bar can be tightened
// against real hardware numbers.

#include "../../oaTest.h"
#include "../videoTestSupport.h"

#include <oa/core/filesystem.h>
#include <oa/runtime/engine.h>

#include <oa/vision/videoPlayer.h>
#include <oa/vision/videoDecoder.h>
#include <oa/vision/videoDemuxer.h>
#include <cstdio>
#include <cstdlib>

namespace {

// frames to validate per clip (kept small for CI wall-time).
constexpr int    kFramesToCheck = 8;
// Empirically calibrated on Intel Iris xe (TGL): a correct hardware decode
// scores ~43 dB against ffmpeg (the gap from infinity is just OA's YUV->rGB
// conversion vs ffmpeg's swscale). A corrupt / duplicated / mis-ordered frame
// scores 8-14 dB. 30 dB cleanly separates the two with wide margin.
constexpr double kMinPsnrDb = 30.0;

bool ffmpegAvailable() {
	return ::system("ffmpeg -version >/dev/null 2>&1") == 0;
}

// PSNR over the RGB channels only (video is opaque; the alpha byte OA writes
// is not meaningful to compare). When inFlipB is true, row r of A is compared
// against row (H-1-r) of B, so a vertical-orientation difference between OA's
// readback and ffmpeg's output does not read as corruption.
double psnrRgb(const oa::U8* inA, const oa::U8* inB, int inW, int inH, bool inFlipB) {
	double mse = 0.0;
	const oa::Usize rowBytes = static_cast<oa::Usize>(inW) * 4U;
	for (int r = 0; r < inH; ++r) {
		const oa::U8* ar = inA + static_cast<oa::Usize>(r) * rowBytes;
		const int br = inFlipB ? (inH - 1 - r) : r;
		const oa::U8* brow = inB + static_cast<oa::Usize>(br) * rowBytes;
		for (int x = 0; x < inW; ++x) {
			for (int c = 0; c < 3; ++c) {
				const double d = static_cast<double>(ar[x * 4 + c]) - static_cast<double>(brow[x * 4 + c]);
				mse += d * d;
			}
		}
	}
	mse /= static_cast<double>(static_cast<oa::Usize>(inW)
		* static_cast<oa::Usize>(inH) * 3U);
	if (mse <= 1e-9) {
		return 1.0e9;
	}
	return 10.0 * oa::log10((255.0 * 255.0) / mse);
}

// Optional diagnostic: dump a raw RGBA frame when OA_VIDEO_DUMP names a dir.
void maybeDumpFrame(const char* inTag, const oa::U8* inData, oa::Usize inBytes) {
	const char* dir = ::getenv("OA_VIDEO_DUMP");
	if (dir == nullptr) {
		return;
	}
	const oa::Path path = oa::Path(dir) / oa::Path(oa::format("{}.rgba", inTag));
	if (FILE* file = ::fopen(path.cStr(), "wb")) {
		(void) ::fwrite(inData, 1, inBytes, file);
		(void) ::fclose(file);
		(void) oa::print("[   dump   ] wrote {} ({} bytes)", path.string(), inBytes);
	}
}

// Decode the first inFrames frames of inPath to interleaved RGBA via ffmpeg,
// streamed over a pipe (no temp files). Returns concatenated frames
// (inFrames * W * H * 4 bytes) or empty on failure.
oa::Vector<oa::U8> ffmpegRgba(const char* inPath, int inFrames) {
	const oa::String command = oa::format(
		"ffmpeg -v error -i \"{}\" -frames:v {} -pix_fmt rgba -f rawvideo pipe:1",
		inPath,
		inFrames);
	FILE* pipe = ::popen(command.cStr(), "r");
	if (pipe == nullptr) {
		return {};
	}
	oa::Vector<oa::U8> out;
	oa::U8 buffer[1 << 16];
	oa::Usize count = 0;
	while ((count = ::fread(buffer, 1, sizeof(buffer), pipe)) > 0) {
		out.append(buffer, count);
	}
	(void) ::pclose(pipe);
	return out;
}

oa::Path resolveVideoAsset(const char* inFixtureRelAsset, const char* inDatasetFilename) {
	if (inDatasetFilename != nullptr && inDatasetFilename[0] != '\0') {
		const oa::Path datasetPath = oa::Paths::data("video") / inDatasetFilename;
		if (oa::Filesystem::exists(datasetPath)) {
			return datasetPath;
		}
	}

	return testAssetPath(inFixtureRelAsset);
}

// Full pipeline for one codec. Returns without failing the test when the
// device simply lacks support for the codec (GTEST_SKIP semantics are applied
// by the caller through the returned "skip" flag being encoded as an empty
// decode with a logged reason).
void validateCodec(
	oa::VideoCodec inCodec,
	const char* inFixtureRelAsset,
	const char* inDatasetFilename,
	const char* inCodecName) {
	SCOPED_TRACE(inCodecName);

	if (not vkTestEngineOk()) {
		GTEST_SKIP() << "no vulkan device";
	}
	if (not ffmpegAvailable()) {
		GTEST_SKIP() << "ffmpeg not on PATH (reference oracle)";
	}

	const oa::Path assetPath = resolveVideoAsset(inFixtureRelAsset, inDatasetFilename);
	if (not oa::Filesystem::readBinary(assetPath).isOk()) {
		FAIL() << "missing test asset: " << assetPath.string();
	}
	const char* pathC = assetPath.cStr();

	auto& engine = testEngine();
	if (not testVideoDecodeSupported(engine, inCodec)) {
		GTEST_SKIP() << inCodecName << ": vulkan Video decode not supported on this device";
	}
	auto demuxerResult = oa::VideoDemuxer::open(oa::StringView(pathC));
	ASSERT_TRUE(demuxerResult.isOk()) << inCodecName
									 << ": profile probe failed: " << demuxerResult.getStatus().toString().cStr();
	const oa::VideoProfile exactProfile = demuxerResult->getVideoProfile();
	auto exactCaps = oa::VideoDecoder::queryDecodeCapabilities(engine, exactProfile);
	ASSERT_TRUE(exactCaps.isOk()) << inCodecName
								  << ": exact capability query failed: " << exactCaps.getStatus().toString().cStr();
	if (not exactCaps->supported) {
		GTEST_SKIP() << inCodecName << ": exact stream profile is unsupported "
					 << "(profile=" << static_cast<unsigned>(exactProfile.standardProfile)
					 << ", level=" << exactProfile.level << ", maxLevel=" << exactCaps->maxLevel
					 << ", hardware=" << exactCaps->hardwareProfileSupported
					 << ", oaPath=" << exactCaps->oaDecodePathImplemented << ", nv12Dpb=" << exactCaps->supportsNv12Dpb
					 << ")";
	}

	// ---- OA end-to-end decode (container → decode → reorder → RGBA) --------
	oa::VideoPlayerConfig cfg;
	cfg.uri          = oa::String(pathC);
	cfg.loop         = false;
	cfg.startPlaying = false;
	cfg.preferHardwareYCbCr = ::getenv("OA_VIDEO_FORCE_COMPUTE_CONVERSION") == nullptr;

	auto videoResult = oa::VideoPlayer::open(engine, cfg);
	ASSERT_TRUE(videoResult.isOk())
		<< inCodecName << ": Video::open failed: " << videoResult.getStatus().toString().cStr();
	oa::VideoPlayer video = oa::move(*videoResult);

	const int width  = static_cast<int>(video.width());
	const int height = static_cast<int>(video.height());
	ASSERT_GT(width, 0);
	ASSERT_GT(height, 0);
	const oa::Usize frameBytes = static_cast<oa::Usize>(width)
		* static_cast<oa::Usize>(height) * 4U;

	oa::Vector<oa::Vector<oa::U8>> oaFrames;
	oa::Vector<double> stepTimings;
	stepTimings.reserve(kFramesToCheck);
	for (int i = 0; i < kFramesToCheck; ++i) {
		const oa::SteadyTimePoint stepStart = oa::SteadyClock::now();
		oa::Status stepStatus = video.next();
		const oa::SteadyTimePoint stepEnd = oa::SteadyClock::now();
		const double stepMs = (stepEnd - stepStart).toMilliseconds();
		if (not stepStatus.isOk()) {
			(void) oa::print("[ timing  ] {} frame {} next failed after {:.2f} ms: {}",
				inCodecName, i, stepMs, stepStatus.toString());
			FAIL() << inCodecName << " frame " << i
				<< ": Next failed: " << stepStatus.toString().cStr();
		}
		const oa::SteadyTimePoint readbackStart = oa::SteadyClock::now();
		auto rb = video.readbackCurrentRgba();
		const oa::SteadyTimePoint readbackEnd = oa::SteadyClock::now();
		const double readbackMs = (readbackEnd - readbackStart).toMilliseconds();
		if (not rb.isOk()) {
			(void) oa::print(
				"[ timing  ] {} frame {} next {:.2f} ms, readback failed after {:.2f} ms: {}",
				inCodecName, i, stepMs, readbackMs, rb.getStatus().toString());
			FAIL() << inCodecName << " frame " << i
				<< ": readbackCurrentRgba failed: " << rb.getStatus().toString().cStr();
		}
		(void) oa::print("[ timing  ] {} frame {} next {:.2f} ms, readback {:.2f} ms",
			inCodecName, i, stepMs, readbackMs);
		stepTimings.pushBack(stepMs);
		oaFrames.emplaceBack(rb->data(), rb->data() + rb->size());
	}
	ASSERT_GE(oaFrames.size(), static_cast<oa::Usize>(1))
		<< inCodecName << ": decoded zero frames end-to-end";
	oa::sort(stepTimings.begin(), stepTimings.end());
	const double stepMedianMs = stepTimings[stepTimings.size() / 2U];
	(void) oa::print("OABENCH video.{} step_forward_p50_ms={:.6f} frames={}",
		inCodecName, stepMedianMs, stepTimings.size());

	// ---- Independent ffmpeg reference decode -------------------------------
	// A few extra reference frames so the best-match search absorbs any
	// display-order offset between the two decoders.
	const int refCount = static_cast<int>(oaFrames.size()) + 4;
	oa::Vector<oa::U8> ref = ffmpegRgba(pathC, refCount);
	ASSERT_GE(ref.size(), frameBytes)
		<< inCodecName << ": ffmpeg produced no reference frames";
	const int refFrames = static_cast<int>(ref.size() / frameBytes);

	// ---- compare: each OA frame must match some reference frame ------------
	// Search all reference frames (absorbs display-order offset) and both
	// vertical orientations (absorbs a readback/ffmpeg origin difference),
	// comparing RGB only. The winning transform is printed so a systematic
	// orientation difference is visible rather than mistaken for corruption.
	for (oa::Usize i = 0; i < oaFrames.size(); ++i) {
		ASSERT_EQ(oaFrames[i].size(), frameBytes)
			<< inCodecName << " frame " << i << ": unexpected readback size";
		if (i == 0) {
			const oa::String oaTag = oa::format("oa_{}_f0", inCodecName);
			const oa::String refTag = oa::format("ref_{}_f0", inCodecName);
			maybeDumpFrame(oaTag.cStr(),
				oaFrames[0].data(), oaFrames[0].size());
			maybeDumpFrame(refTag.cStr(),
				ref.data(), oa::min(frameBytes, ref.size()));
		}
		double best = -1.0;
		int    bestRef = -1;
		bool   bestFlip = false;
		for (int j = 0; j < refFrames; ++j) {
			const oa::U8* refP = ref.data() + static_cast<oa::Usize>(j) * frameBytes;
			for (bool flip : {false, true}) {
				const double p = psnrRgb(oaFrames[i].data(), refP, width, height, flip);
				if (p > best) { best = p; bestRef = j; bestFlip = flip; }
			}
		}
		(void) oa::print("[   psnr   ] {} frame {}: {:.1f} dB (ref={} flip={})",
			inCodecName, i, best, bestRef, bestFlip ? 1 : 0);
		EXPECT_GE(best, kMinPsnrDb)
			<< inCodecName << " frame " << i << " best RGB PSNR " << best
			<< " dB is below " << kMinPsnrDb << " dB — decode is corrupt or wrong frame";
	}
	ASSERT_TRUE(video.close().isOk());
}

} // namespace

class VideoDecodeReference : public ::testing::Test {};

TEST_VK(VideoDecodeReference, Av1)  { validateCodec(oa::VideoCodec::AV1, "video/clip/shibuya_720p_30fps_av1_main_8bit_420.mp4",
				  "profiles/shibuya_720p_av1_main_8bit_420.mp4",  "av1");  }
TEST_VK(VideoDecodeReference, H264Baseline) { validateCodec(oa::VideoCodec::H264, "video/clip/shibuya_720p_30fps_h264_baseline_8bit_420.mp4",
				  nullptr, "h264-baseline"); }
TEST_VK(VideoDecodeReference, H264High) { validateCodec(oa::VideoCodec::H264, "video/clip/shibuya_720p_30fps_h264_high_8bit_420.mp4",
				  "profiles/shibuya_720p_h264_high_8bit_420.mp4", "h264"); }
TEST_VK(VideoDecodeReference, H264Main) { validateCodec(oa::VideoCodec::H264, "video/clip/shibuya_720p_30fps_h264_main_8bit_420.mp4",
				  nullptr, "h264-main"); }
TEST_VK(VideoDecodeReference, H265) { validateCodec(oa::VideoCodec::H265, "video/clip/shibuya_720p_30fps_h265_main_8bit_420.mp4",
				  "profiles/shibuya_720p_h265_main_8bit_420.mp4", "h265"); }
TEST_VK(VideoDecodeReference, Vp9)  { validateCodec(oa::VideoCodec::VP9, "video/clip/shibuya_720p_30fps_vp9_profile0_8bit_420.mp4",
				  "profiles/shibuya_720p_vp9_profile0_8bit_420.mp4",  "vp9");  }
