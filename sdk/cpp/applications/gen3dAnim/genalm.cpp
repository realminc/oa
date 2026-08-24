// genalm — generate canonical motion from one trained oa::AlmAg bundle.
//
// A frozen text feature may come from an arbitrary prompt or an aligned dataset
// caption. The bundle owns tokenizer/prior architecture and exact text-encoder
// identity; the CLI owns only generation policy and output selection.
//
// usage:
//   genalm --model var/model/dev/Alm/Alm.oam \
//          --prompt "a person walks forward" --dataset /path/to/Cmp

#include <oa/runtime/app.h>
#include <oa/runtime/executionSession.h>
#include <oa/core/cli.h>
#include <oa/core/log.h>
#include <oa/core/filesystem.h>
#include <oa/core/paths.h>
#include <oa/core/time.h>
#include <oa/ml.h>
#include <oa/ml/modelFile.h>
#include <data/dsHumanMl3d.h>
#include <anim/usd.h>
#include <rig/skeleton.h>
#include <rig/skeletonUsd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>

#include <ml/nn/alm/almAg.h>

// ── Config ──────────────────────────────────────────────────────────────────

struct GenAlmConfig {
	oa::String model    = "var/model/dev/Alm/Alm.oam";
	oa::String dataset  = oa::Paths::data("humanMl3d/Cmp").string();
	oa::String split    = "train";
	oa::String outDir   = oa::Paths::var("alm").string();
	oa::String name     = "Alm";
	oa::String precisionStr = "fp32";

	[[nodiscard]] oa::Precision precision() const {
		if (precisionStr == "fp32") return oa::Precision::FP32;
		if (precisionStr == "bf16") return oa::Precision::BF16;
		if (precisionStr == "fp64") return oa::Precision::FP64;
		return oa::Precision::FP32;
	}

	// generation
	oa::I32 genCount       = 3;
	oa::F32 genTemperature = 1.0F;
	oa::I32 genMaxLen      = 64;
	oa::I32 seed           = 42;
	oa::I32 conditioningClip = 0;    // index in the selected split
	oa::I32 captionIndex = 0;        // caption/feature row for that clip
	oa::String prompt;
	oa::String textFeature;          // raw textFeatureDim float32 values
};

class GenAlmCli : public oa::Cli<GenAlmConfig> {
public:
	GenAlmCli()
		: oa::Cli<GenAlmConfig>(
			"genalm", "generate motion from one trained oa::AlmAg .oam bundle") {
		addOption("--model",     cfg_.model,   "oa::AlmAg bundle path");
		addOption("--dataset",  cfg_.dataset, "CMP dataset (for denormalization)");
		addOption("--split",    cfg_.split,   "Dataset split");
		addOption("--out-dir",  cfg_.outDir,  "output directory for .usda files");
		addOption("--name",     cfg_.name,    "Model name (output prefix)");

		addOption("--gen-count", cfg_.genCount,      "Number of generated clips");
		addOption("--gen-temp",  cfg_.genTemperature, "generation temperature");
		addOption("--gen-len",   cfg_.genMaxLen,     "Max generated token length");
		addOption("--seed",      cfg_.seed,          "RNG seed");
		addOption("--conditioning-clip", cfg_.conditioningClip,
			"clip index whose precomputed CLIP caption feature conditions generation");
		addOption("--caption-index", cfg_.captionIndex,
			"caption/CLIP feature row within --conditioning-clip");
		addOption("--prompt", cfg_.prompt,
			"Literal text prompt for the bundle's native GPU text encoder");
		addOption("--text-feature", cfg_.textFeature,
			"Reference-only raw float32 text feature override");
		addOption("--precision", cfg_.precisionStr,  "fp32 | bf16 | fp64");
	}

	void loadYaml(const oa::Yaml::Node& inYaml) override {
		cfg_.name = oa::Yaml::get<oa::String>(inYaml, "name", cfg_.name);

		const oa::Yaml::Node g = inYaml["generation"];
		cfg_.model   = oa::Yaml::get<oa::String>(g, "model",     cfg_.model);
		cfg_.dataset = oa::Yaml::get<oa::String>(g, "dataset",  cfg_.dataset);
		cfg_.outDir  = oa::Yaml::get<oa::String>(g, "out_dir",  cfg_.outDir);
		cfg_.genCount      = oa::Yaml::get<oa::I32>(g, "count",      cfg_.genCount);
		cfg_.genTemperature = oa::Yaml::get<oa::F32>(g, "temperature", cfg_.genTemperature);
		cfg_.genMaxLen     = oa::Yaml::get<oa::I32>(g, "max_len",   cfg_.genMaxLen);
		cfg_.seed          = oa::Yaml::get<oa::I32>(g, "seed",      cfg_.seed);
		cfg_.conditioningClip = oa::Yaml::get<oa::I32>(g, "conditioning_clip", cfg_.conditioningClip);
		cfg_.captionIndex = oa::Yaml::get<oa::I32>(g, "caption_index", cfg_.captionIndex);
		cfg_.prompt        = oa::Yaml::get<oa::String>(g, "prompt", cfg_.prompt);
		cfg_.textFeature   = oa::Yaml::get<oa::String>(g, "text_feature", cfg_.textFeature);
		cfg_.precisionStr  = oa::Yaml::get<oa::String>(g, "precision", cfg_.precisionStr);
	}
};

// ── Helpers ──────────────────────────────────────────────────────────────────

namespace {

oa::Result<std::vector<oa::F32>> loadRawF32(const oa::String& inPath, oa::I32 inCount) {
	std::ifstream file(inPath.cStr(), std::ios::binary | std::ios::ate);
	if (not file) return oa::Status::error("cannot open prompt feature: " + inPath);
	const auto bytes = file.tellg();
	const std::streamoff expected = static_cast<std::streamoff>(inCount)
		* static_cast<std::streamoff>(sizeof(oa::F32));
	if (bytes != expected) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"prompt feature byte count does not match the oa::AlmAg text encoder");
	}
	file.seekg(0);
	std::vector<oa::F32> values(static_cast<size_t>(inCount));
	file.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(expected));
	if (not file) return oa::Status::error("failed to read prompt feature: " + inPath);
	return values;
}

// Copy a matrix to host FP32. Safe for BF16/FP16 storage models.
oa::Vec<oa::F32> hostFloatData(const oa::Matrix& inMatrix) {
	auto& ctx = oa::ExecutionSession::getActive();
	if (inMatrix.getDtype() == oa::ScalarType::Float32) {
		(void)ctx.submitAndWait();
		const oa::F32* p = inMatrix.dataAs<const oa::F32>();
		return oa::Vec<oa::F32>(p, p + inMatrix.numElements());
	}
	oa::Matrix f32 = oa::FnMatrix::empty(inMatrix.getShape(), oa::ScalarType::Float32);
	oa::FnMatrix::castInto(inMatrix, f32);
	(void)ctx.submitAndWait();
	const oa::F32* p = f32.dataAs<const oa::F32>();
	return oa::Vec<oa::F32>(p, p + f32.numElements());
}

} // namespace

// ── App ─────────────────────────────────────────────────────────────────────

struct GenAlmApp : oa::ComputeApp {
	GenAlmCli cli;

	int setup(int argc, char** argv) override {
		if (not cli.parse(argc, argv)) { isRunning = false; }
		const auto& c = cli.getConfig();
		engineConfig_.precision = c.precision();
		return 0;
	}

	oa::Status init() override {
		return oa::Status::ok();
	}

	oa::Status tick() override {
		const auto& c = cli.getConfig();
		auto& ctx = oa::ExecutionSession::getActive();

		if (c.model.empty()) {
			OaLogError(oa::LogComponent::Ml, "genalm: --model is required");
			isRunning = false; return oa::Status::ok();
		}
		auto loadedAlm = oa::AlmAg::loadBundle(engine(), c.model);
		if (not loadedAlm.isOk()) {
			OaLogError(oa::LogComponent::Ml, "genalm: oa::AlmAg load failed: %s",
				loadedAlm.getStatus().getMessage().cStr());
			isRunning = false; return oa::Status::ok();
		}
		auto alm = oa::move(loadedAlm).getValue();
		const auto& almCfg = alm->config();
		const bool conditioned = almCfg.prior.textFeatureDim > 0;
		if (c.genMaxLen <= 0 or c.genMaxLen + (conditioned ? 1 : 0) > almCfg.prior.maxSeqLen) {
			OaLogError(oa::LogComponent::Ml,
				"genalm: generation length %d does not fit bundle maxSeqLen %d",
				c.genMaxLen, almCfg.prior.maxSeqLen);
			isRunning = false; return oa::Status::ok();
		}

		// ── load dataset (for denormalization Mean/std) ──
		if (c.conditioningClip < 0 or c.captionIndex < 0) {
			OaLogError(oa::LogComponent::Ml, "genalm: conditioning clip/caption indices cannot be negative");
			isRunning = false; return oa::Status::ok();
		}
		const bool promptMode = not c.prompt.empty();
		const bool featureOverride = not c.textFeature.empty();
		if (promptMode and featureOverride) {
			OaLogError(oa::LogComponent::Ml,
				"genalm: --prompt and --text-feature are mutually exclusive");
			isRunning = false; return oa::Status::ok();
		}
		if ((promptMode or featureOverride) and not conditioned) {
			OaLogError(oa::LogComponent::Ml,
				"genalm: text cannot condition an unconditional oa::AlmAg bundle");
			isRunning = false; return oa::Status::ok();
		}
		if (promptMode and not alm->hasNativeTextEncoder()) {
			OaLogError(oa::LogComponent::Ml,
				"genalm: this oa::AlmAg bundle has no native text encoder");
			isRunning = false; return oa::Status::ok();
		}
		oa::DsCombatMotionProcessed ds(c.dataset, c.split,
			conditioned and not promptMode and not featureOverride ? c.conditioningClip + 1 : 1);
		if (not ds.ok()) {
			OaLogError(oa::LogComponent::Ml, "genalm: failed to load CMP from %s", c.dataset.cStr());
			isRunning = false; return oa::Status::ok();
		}
		const oa::I32 featDim   = ds.featDim();
		const oa::I32 textFeatureDim = almCfg.prior.textFeatureDim;
		const oa::F32* textFeatureData = nullptr;
		std::vector<oa::F32> promptFeature;
		oa::String conditioningCaption;
		if (featureOverride) {
			auto loadedFeature = loadRawF32(c.textFeature, textFeatureDim);
			if (not loadedFeature.isOk()) {
				OaLogError(oa::LogComponent::Ml, "genalm: %s",
					loadedFeature.getStatus().getMessage().cStr());
				isRunning = false; return oa::Status::ok();
			}
			promptFeature = oa::move(loadedFeature).getValue();
			textFeatureData = promptFeature.data();
			conditioningCaption = "external feature override";
		} else if (promptMode) {
			conditioningCaption = c.prompt;
		} else if (conditioned) {
			if (c.conditioningClip >= ds.numClips()) {
				OaLogError(oa::LogComponent::Ml,
					"genalm: conditioning clip %d is outside split '%s' (%lld clips loaded)",
					c.conditioningClip, c.split.cStr(), static_cast<long long>(ds.numClips()));
				isRunning = false; return oa::Status::ok();
			}
			const oa::I32 featureCount = ds.clipTextFeatureCount(c.conditioningClip);
			if (textFeatureDim <= 0 or ds.textFeatureFormat() != "oa_clip_text_v1" or
				ds.textFeatureModel() != almCfg.textEncoder or c.captionIndex >= featureCount or
				c.captionIndex >= static_cast<oa::I32>(ds.clipCaptions(c.conditioningClip).size())) {
				OaLogError(oa::LogComponent::Ml,
					"genalm: missing CLIP feature row %d for clip %s; run trainalm once to native-bake the caption cache",
					c.captionIndex, ds.clipId(c.conditioningClip).cStr());
				isRunning = false; return oa::Status::ok();
			}
			textFeatureData = ds.clipTextFeatureData(c.conditioningClip)
				+ static_cast<size_t>(c.captionIndex) * textFeatureDim;
			conditioningCaption = ds.clipCaptions(c.conditioningClip)[c.captionIndex].text;
		}

		oa::FnMatrix::setRngSeed(static_cast<oa::U64>(c.seed));
		std::printf("Loaded oa::AlmAg: %s\n", c.model.cStr());

		// ── generate ──
		(void)oa::Filesystem::createDirectories(oa::Path(c.outDir));
		ctx.clear();
		oa::Matrix textFeature;
		if (promptMode) {
			auto encodedPrompt = alm->encodePrompt(c.prompt);
			if (encodedPrompt.isError()) {
				OaLogError(oa::LogComponent::Ml, "genalm: prompt encoding failed: %s",
					encodedPrompt.getStatus().getMessage().cStr());
				isRunning = false; return oa::Status::ok();
			}
			textFeature = oa::move(encodedPrompt.getValue());
			auto hostFeature = hostFloatData(textFeature);
			promptFeature.assign(hostFeature.data(), hostFeature.data() + hostFeature.size());
			textFeatureData = promptFeature.data();
		} else if (conditioned) {
			textFeature = oa::FnMatrix::fromBytes(
				oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(textFeatureData),
					static_cast<size_t>(textFeatureDim) * sizeof(oa::F32)),
				oa::MatrixShape{1, textFeatureDim}, oa::ScalarType::Float32);
		}

		std::printf("\ngenalm — generating %d clips (maxLen=%d, seed=%d)\n",
			c.genCount, c.genMaxLen, c.seed);
		if (conditioned) {
			std::printf("Conditioning: %s-%d · %s · \"%s\"\n",
				almCfg.textEncoder.cStr(), textFeatureDim,
				promptMode ? "native prompt" : (featureOverride ? "feature override" : "dataset caption"),
				conditioningCaption.cStr());
		}

		oa::Stopwatch totalTimer;
		totalTimer.start();

		for (oa::I32 g = 0; g < c.genCount; ++g) {
			const float temp = c.genTemperature;

			oa::Stopwatch stepTimer;
			stepTimer.start();

			auto generated = conditioned
				? alm->prior().generateConditioned(textFeature, temp, 0, 0.9F, c.genMaxLen)
				: alm->prior().generate(1, temp, 0, 0.9F, c.genMaxLen);
			auto motion = alm->prior().decodeToMotion(generated, alm->tokenizer());
			(void)ctx.submitAndWait();

			const double genMs = stepTimer.elapsedMs();

			const oa::I32 frames = static_cast<oa::I32>(motion.size(0));
			if (frames <= 0) {
				std::printf("  [gen %d] T=%.2f: empty motion\n", g, temp);
				continue;
			}

			const double tokSps = static_cast<double>(generated.numElements()) / (genMs * 0.001);
			const double frameSps = static_cast<double>(frames) / (genMs * 0.001);

			std::printf("  [gen %d] T=%.2f: %d frames × %lld dims | %.1f ms | %.0f tok/s | %.0f fps\n",
				g, temp, frames, static_cast<long long>(motion.size(1)),
				genMs, tokSps, frameSps);

			// denormalize and recover world joints.
			auto motionHost = hostFloatData(motion);
			std::vector<float> featHost(motionHost.data(), motionHost.data() + motionHost.size());
			ds.denormalize(featHost.data(), frames);
			auto worldJoints = oa::humanMl3dRecoverWorldJoints(
				oa::Span<const oa::F32>(featHost.data(), featHost.size()), frames, featDim);

			auto skelClip = oa::usdClipFromWorldJoints(
				oa::skHumanMl3d(),
				oa::Span<const oa::F32>(worldJoints.data(), worldJoints.size()),
				frames, 20.0F, 1, 100.0F);

			char pathBuf[256];
			std::snprintf(pathBuf, sizeof(pathBuf), "%s/%s_gen_%d_T%.1f.usda",
				c.outDir.cStr(), c.name.cStr(), g, temp);
			oa::Path usdPath(pathBuf);
			auto usdSt = oa::Usd::writeUsda(usdPath, skelClip, "humanml3d");
			std::printf("         saved %s (%s)\n",
				usdPath.cStr(), usdSt.isOk() ? "ok" : usdSt.toString().cStr());

			// Keep generation provenance beside the preview. This is intentionally a
			// transparent line-based manifest rather than hidden checkpoint state.
			std::ostringstream metadata;
			metadata << "format=oa_alm_generation_v1\n"
				<< "model=" << c.name.cStr() << "\n"
				<< "dataset=" << c.dataset.cStr() << "\n"
				<< "split=" << c.split.cStr() << "\n"
				<< "bundle=" << c.model.cStr() << "\n"
				<< "seed=" << c.seed << "\n"
				<< "temperature=" << temp << "\n"
				<< "max_motion_tokens=" << c.genMaxLen << "\n"
				<< "generated_tokens=" << generated.numElements() << "\n"
				<< "frames=" << frames << "\n"
				<< "feature_dim=" << featDim << "\n"
				<< "position_inverse=humanml3d_recover_from_ric\n"
				<< "usd=" << usdPath.cStr() << "\n";
			if (conditioned) {
				metadata << "text_feature_model=" << almCfg.textEncoder.cStr() << "\n"
					<< "text_feature_dim=" << textFeatureDim << "\n"
					<< "text_feature_hash=" << oa::modelFileHash(
						reinterpret_cast<const oa::U8*>(textFeatureData),
						static_cast<oa::Usize>(textFeatureDim) * sizeof(oa::F32)) << "\n"
					<< "prompt=" << conditioningCaption.cStr() << "\n";
				if (featureOverride) {
					metadata << "text_feature_file=" << c.textFeature.cStr() << "\n";
				} else if (promptMode) {
					metadata << "text_encoder=native_oa_vulkan\n";
				} else {
					metadata << "conditioning_clip_index=" << c.conditioningClip << "\n"
						<< "conditioning_clip_id=" << ds.clipId(c.conditioningClip).cStr() << "\n"
						<< "caption_index=" << c.captionIndex << "\n";
				}
			}
			const oa::Path metadataPath(oa::String(pathBuf) + ".meta.txt");
			const auto metadataSt = oa::Filesystem::writeText(
				metadataPath, oa::String(metadata.str().c_str()));
			std::printf("         metadata %s (%s)\n", metadataPath.cStr(),
				metadataSt.isOk() ? "ok" : metadataSt.toString().cStr());
			std::fflush(stdout);
		}

		const double totalSec = totalTimer.elapsedSec();
		std::printf("\nGeneration complete: %d clips in %.2f s\n", c.genCount, totalSec);
		std::fflush(stdout);

		isRunning = false;
		return oa::Status::ok();
	}
};

int main(int argc, char** argv) {
	GenAlmApp app;
	return app.main(argc, argv);
}
