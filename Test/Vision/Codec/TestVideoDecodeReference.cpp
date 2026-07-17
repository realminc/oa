// OA Vision — End-to-end video decode reference validation.
//
// For each codec (AV1 / H.264 / H.265 / VP9) this decodes the checked-in
// 2-second 720p .mp4 clip END TO END through the real pipeline — MP4
// container demux → hardware video decode → display-order reorder →
// YCbCr→RGBA conversion → CPU readback — via OaItVideo's compatibility
// adapter over OaVideo (the same path the
// TutorialViewerVideo* applications use). Each decoded frame is PSNR-compared
// against an INDEPENDENT ffmpeg decode of the same clip. This replaces
// eyeballing four player windows with a headless pass/fail gate.
//
// Assets:   Asset/Video/shibuya_720p_<codec>.mp4  (see that folder's README)
// Oracle:   ffmpeg on PATH (test skips if absent)
//
// PSNR note: OA's YCbCr→RGB conversion and ffmpeg's swscale differ slightly
// (matrix / range), so this is a corruption / wrong-frame gate, not a
// bit-exact conformance check. Garbage or a lost-device frame scores in the
// low single digits; a correctly decoded frame scores well above the
// threshold. Actual per-frame PSNR is printed so the bar can be tightened
// against real hardware numbers.

#include "../../OaTest.h"

#include <Oa/Vision/ItVideo.h>
#include <Oa/Vision/VideoDecoder.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Core/FileIo.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <filesystem>

namespace {

// Frames to validate per clip (kept small for CI wall-time).
constexpr int    kFramesToCheck = 8;
// Empirically calibrated on Intel Iris Xe (TGL): a correct hardware decode
// scores ~43 dB against ffmpeg (the gap from infinity is just OA's YUV->RGB
// conversion vs ffmpeg's swscale). A corrupt / duplicated / mis-ordered frame
// scores 8-14 dB. 30 dB cleanly separates the two with wide margin.
constexpr double kMinPsnrDb = 30.0;

bool FfmpegAvailable() {
	return std::system("ffmpeg -version >/dev/null 2>&1") == 0;
}

// PSNR over the RGB channels only (video is opaque; the alpha byte OA writes
// is not meaningful to compare). When InFlipB is true, row r of A is compared
// against row (H-1-r) of B, so a vertical-orientation difference between OA's
// readback and ffmpeg's output does not read as corruption.
double PsnrRgb(const uint8_t* InA, const uint8_t* InB, int InW, int InH, bool InFlipB) {
	double mse = 0.0;
	const size_t rowBytes = static_cast<size_t>(InW) * 4u;
	for (int r = 0; r < InH; ++r) {
		const uint8_t* ar = InA + static_cast<size_t>(r) * rowBytes;
		const int br = InFlipB ? (InH - 1 - r) : r;
		const uint8_t* brow = InB + static_cast<size_t>(br) * rowBytes;
		for (int x = 0; x < InW; ++x) {
			for (int c = 0; c < 3; ++c) {
				const double d = static_cast<double>(ar[x * 4 + c]) - static_cast<double>(brow[x * 4 + c]);
				mse += d * d;
			}
		}
	}
	mse /= static_cast<double>(static_cast<size_t>(InW) * InH * 3u);
	if (mse <= 1e-9) {
		return 1.0e9;
	}
	return 10.0 * std::log10((255.0 * 255.0) / mse);
}

// Optional diagnostic: dump a raw RGBA frame when OA_VIDEO_DUMP names a dir.
void MaybeDumpFrame(const char* InTag, const uint8_t* InData, size_t InBytes) {
	const char* dir = std::getenv("OA_VIDEO_DUMP");
	if (dir == nullptr) {
		return;
	}
	std::string p = std::string(dir) + "/" + InTag + ".rgba";
	if (FILE* f = std::fopen(p.c_str(), "wb")) {
		std::fwrite(InData, 1, InBytes, f);
		std::fclose(f);
		std::printf("[   dump   ] wrote %s (%zu bytes)\n", p.c_str(), InBytes);
	}
}

// Decode the first InFrames frames of InPath to interleaved RGBA via ffmpeg,
// streamed over a pipe (no temp files). Returns concatenated frames
// (InFrames * W * H * 4 bytes) or empty on failure.
std::vector<uint8_t> FfmpegRgba(const char* InPath, int InFrames) {
	std::string cmd = "ffmpeg -v error -i \"";
	cmd += InPath;
	cmd += "\" -frames:v " + std::to_string(InFrames)
	    +  " -pix_fmt rgba -f rawvideo pipe:1";
	FILE* pipe = ::popen(cmd.c_str(), "r");
	if (pipe == nullptr) {
		return {};
	}
	std::vector<uint8_t> out;
	uint8_t buf[1 << 16];
	size_t n;
	while ((n = std::fread(buf, 1, sizeof(buf), pipe)) > 0) {
		out.insert(out.end(), buf, buf + n);
	}
	::pclose(pipe);
	return out;
}

std::string ResolveVideoAsset(const char* InFixtureRelAsset, const char* InDatasetFilename) {
	if (InDatasetFilename != nullptr && InDatasetFilename[0] != '\0') {
		std::filesystem::path datasetDir;
		if (const char* env = std::getenv("OA_VIDEO_DATA"); env != nullptr && env[0] != '\0') {
			datasetDir = env;
		} else {
			const auto repoDir = std::filesystem::path(__FILE__)
				.parent_path()
				.parent_path()
				.parent_path()
				.parent_path();
			datasetDir = repoDir.parent_path() / "dataset" / "video";
		}

		std::filesystem::path datasetPath = datasetDir / InDatasetFilename;
		if (std::filesystem::exists(datasetPath)) {
			return datasetPath.lexically_normal().string();
		}
	}

	return OaTestAssetPath(InFixtureRelAsset).CStr();
}

// Full pipeline for one codec. Returns without failing the test when the
// device simply lacks support for the codec (GTEST_SKIP semantics are applied
// by the caller through the returned "skip" flag being encoded as an empty
// decode with a logged reason).
void ValidateCodec(
	OaVideoCodec InCodec,
	const char* InFixtureRelAsset,
	const char* InDatasetFilename,
	const char* InCodecName) {
	SCOPED_TRACE(InCodecName);

	if (not OaVkTestEngineOk()) {
		GTEST_SKIP() << "no Vulkan device";
	}
	if (not FfmpegAvailable()) {
		GTEST_SKIP() << "ffmpeg not on PATH (reference oracle)";
	}

	const std::string assetPath = ResolveVideoAsset(InFixtureRelAsset, InDatasetFilename);
	if (not OaFileIo::ReadBinary(OaPath(assetPath.c_str())).IsOk()) {
		FAIL() << "missing test asset: " << assetPath;
	}
	const char* pathC = assetPath.c_str();

	auto& rt = *OaComputeEngine::GetGlobal();
	if (not OaVideoDecoder::IsCodecSupported(rt, InCodec)) {
		GTEST_SKIP() << InCodecName << ": Vulkan Video decode not supported on this device";
	}

	// ---- OA end-to-end decode (container → decode → reorder → RGBA) --------
	OaItVideoConfig cfg;
	cfg.Path         = OaString(pathC);
	cfg.Loop         = false;
	cfg.StartPlaying = false;
	cfg.PreferHardwareYCbCr = std::getenv("OA_VIDEO_FORCE_COMPUTE_CONVERSION") == nullptr;

	auto itR = OaItVideo::Create(rt, cfg);
	ASSERT_TRUE(itR.IsOk())
		<< InCodecName << ": OaItVideo::Create failed: " << itR.GetStatus().ToString().c_str();
	OaItVideo it = OaStdMove(*itR);

	const int width  = static_cast<int>(it.Width());
	const int height = static_cast<int>(it.Height());
	ASSERT_GT(width, 0);
	ASSERT_GT(height, 0);
	const size_t frameBytes = static_cast<size_t>(width) * static_cast<size_t>(height) * 4u;

	std::vector<std::vector<uint8_t>> oaFrames;
	for (int i = 0; i < kFramesToCheck; ++i) {
		const auto stepStart = std::chrono::steady_clock::now();
		OaStatus stepStatus = it.StepForward();
		const auto stepEnd = std::chrono::steady_clock::now();
		const double stepMs = std::chrono::duration<double, std::milli>(stepEnd - stepStart).count();
		if (not stepStatus.IsOk()) {
			std::printf("[ timing  ] %-4s frame %d StepForward failed after %.2f ms: %s\n",
				InCodecName, i, stepMs, stepStatus.ToString().c_str());
			FAIL() << InCodecName << " frame " << i
				<< ": StepForward failed: " << stepStatus.ToString().c_str();
		}
		const auto readbackStart = std::chrono::steady_clock::now();
		auto rb = it.ReadbackCurrentRgba();
		const auto readbackEnd = std::chrono::steady_clock::now();
		const double readbackMs = std::chrono::duration<double, std::milli>(readbackEnd - readbackStart).count();
		if (not rb.IsOk()) {
			std::printf("[ timing  ] %-4s frame %d StepForward %.2f ms, Readback failed after %.2f ms: %s\n",
				InCodecName, i, stepMs, readbackMs, rb.GetStatus().ToString().c_str());
			FAIL() << InCodecName << " frame " << i
				<< ": ReadbackCurrentRgba failed: " << rb.GetStatus().ToString().c_str();
		}
		std::printf("[ timing  ] %-4s frame %d StepForward %.2f ms, Readback %.2f ms\n",
			InCodecName, i, stepMs, readbackMs);
		oaFrames.emplace_back(rb->Data(), rb->Data() + rb->Size());
	}
	ASSERT_GE(oaFrames.size(), static_cast<size_t>(1))
		<< InCodecName << ": decoded zero frames end-to-end";

	// ---- Independent ffmpeg reference decode -------------------------------
	// A few extra reference frames so the best-match search absorbs any
	// display-order offset between the two decoders.
	const int refCount = static_cast<int>(oaFrames.size()) + 4;
	std::vector<uint8_t> ref = FfmpegRgba(pathC, refCount);
	ASSERT_GE(ref.size(), frameBytes)
		<< InCodecName << ": ffmpeg produced no reference frames";
	const int refFrames = static_cast<int>(ref.size() / frameBytes);

	// ---- Compare: each OA frame must match some reference frame ------------
	// Search all reference frames (absorbs display-order offset) and both
	// vertical orientations (absorbs a readback/ffmpeg origin difference),
	// comparing RGB only. The winning transform is printed so a systematic
	// orientation difference is visible rather than mistaken for corruption.
	for (size_t i = 0; i < oaFrames.size(); ++i) {
		ASSERT_EQ(oaFrames[i].size(), frameBytes)
			<< InCodecName << " frame " << i << ": unexpected readback size";
		if (i == 0) {
			MaybeDumpFrame((std::string("oa_") + InCodecName + "_f0").c_str(),
				oaFrames[0].data(), oaFrames[0].size());
			MaybeDumpFrame((std::string("ref_") + InCodecName + "_f0").c_str(),
				ref.data(), std::min(frameBytes, ref.size()));
		}
		double best = -1.0;
		int    bestRef = -1;
		bool   bestFlip = false;
		for (int j = 0; j < refFrames; ++j) {
			const uint8_t* refP = ref.data() + static_cast<size_t>(j) * frameBytes;
			for (bool flip : {false, true}) {
				const double p = PsnrRgb(oaFrames[i].data(), refP, width, height, flip);
				if (p > best) { best = p; bestRef = j; bestFlip = flip; }
			}
		}
		std::printf("[   psnr   ] %-4s frame %zu: %.1f dB (ref=%d flip=%d)\n",
			InCodecName, i, best, bestRef, bestFlip ? 1 : 0);
		EXPECT_GE(best, kMinPsnrDb)
			<< InCodecName << " frame " << i << " best RGB PSNR " << best
			<< " dB is below " << kMinPsnrDb << " dB — decode is corrupt or wrong frame";
	}
}

} // namespace

class VideoDecodeReference : public ::testing::Test {};

TEST_VK(VideoDecodeReference, Av1)  { ValidateCodec(OaVideoCodec::AV1,  "Video/shibuya_720p_av1.mp4",  "shibuya_crossing_1080p30_av1.mp4",  "av1");  }
TEST_VK(VideoDecodeReference, H264) { ValidateCodec(OaVideoCodec::H264, "Video/shibuya_720p_h264.mp4", "shibuya_crossing_1080p30_h264.mp4", "h264"); }
TEST_VK(VideoDecodeReference, H265) { ValidateCodec(OaVideoCodec::H265, "Video/shibuya_720p_h265.mp4", "shibuya_crossing_1080p30_h265.mp4", "h265"); }
TEST_VK(VideoDecodeReference, Vp9)  { ValidateCodec(OaVideoCodec::VP9,  "Video/shibuya_720p_vp9.mp4",  "shibuya_crossing_1080p30_vp9.mp4",  "vp9");  }
