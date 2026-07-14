// genalm — generate canonical motion from one trained OaAlmAg bundle.
//
// A frozen text feature may come from an arbitrary prompt or an aligned dataset
// caption. The bundle owns tokenizer/prior architecture and exact text-encoder
// identity; the CLI owns only generation policy and output selection.
//
// Usage:
//   genalm --model var/model/dev/Alm/Alm.oam \
//          --prompt "a person walks forward" --dataset /path/to/Cmp

#include <Oa/Runtime/App.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Core/Cli.h>
#include <Oa/Core/Log.h>
#include <Oa/Core/FileIo.h>
#include <Oa/Core/Time.h>
#include <Oa/Ml.h>
#include <Oa/Ml/Oam.h>
#include <Oa/Data/DsHumanMl3d.h>
#include <Anim/Usd.h>
#include <Rig/Skeleton.h>
#include <Rig/SkeletonUsd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>

#include <Ml/Nn/Alm/AlmAg.h>

// ── Config ──────────────────────────────────────────────────────────────────

struct GenAlmConfig {
	OaString Model    = "var/model/dev/Alm/Alm.oam";
	OaString Dataset  = "../dataset/gen/3d/anim/ds/Cmp";
	OaString Split    = "train";
	OaString OutDir   = OaFileIo::GetVarDir("alm").String();
	OaString Name     = "Alm";
	OaString PrecisionStr = "fp32";

	[[nodiscard]] OaPrecision Precision() const {
		if (PrecisionStr == "fp32") return OaPrecision::FP32;
		if (PrecisionStr == "bf16") return OaPrecision::BF16;
		if (PrecisionStr == "tf32") return OaPrecision::TF32;
		if (PrecisionStr == "fp16") return OaPrecision::FP16;
		return OaPrecision::FP32;
	}

	// Generation
	OaI32 GenCount       = 3;
	OaF32 GenTemperature = 1.0F;
	OaI32 GenMaxLen      = 64;
	OaI32 Seed           = 42;
	OaI32 ConditioningClip = 0;    // index in the selected split
	OaI32 CaptionIndex = 0;        // caption/feature row for that clip
	OaString Prompt;
	OaString TextFeature;          // raw TextFeatureDim float32 values
};

class GenAlmCli : public OaCli<GenAlmConfig> {
public:
	GenAlmCli()
		: OaCli("genalm", "Generate motion from one trained OaAlmAg .oam bundle") {
		AddOption("--model",     Cfg_.Model,   "OaAlmAg bundle path");
		AddOption("--dataset",  Cfg_.Dataset, "CMP dataset (for denormalization)");
		AddOption("--split",    Cfg_.Split,   "Dataset split");
		AddOption("--out-dir",  Cfg_.OutDir,  "Output directory for .usda files");
		AddOption("--name",     Cfg_.Name,    "Model name (output prefix)");

		AddOption("--gen-count", Cfg_.GenCount,      "Number of generated clips");
		AddOption("--gen-temp",  Cfg_.GenTemperature, "Generation temperature");
		AddOption("--gen-len",   Cfg_.GenMaxLen,     "Max generated token length");
		AddOption("--seed",      Cfg_.Seed,          "RNG seed");
		AddOption("--conditioning-clip", Cfg_.ConditioningClip,
			"Clip index whose precomputed CLIP caption feature conditions generation");
		AddOption("--caption-index", Cfg_.CaptionIndex,
			"Caption/CLIP feature row within --conditioning-clip");
		AddOption("--prompt", Cfg_.Prompt,
			"Literal text prompt for the bundle's native GPU text encoder");
		AddOption("--text-feature", Cfg_.TextFeature,
			"Reference-only raw float32 text feature override");
		AddOption("--precision", Cfg_.PrecisionStr,  "fp32 | bf16 | tf32 | fp16");
	}

	void LoadYaml(const OaYaml::Node& InYaml) override {
		Cfg_.Name = OaYaml::Get<OaString>(InYaml, "name", Cfg_.Name);

		const OaYaml::Node g = InYaml["generation"];
		Cfg_.Model   = OaYaml::Get<OaString>(g, "model",     Cfg_.Model);
		Cfg_.Dataset = OaYaml::Get<OaString>(g, "dataset",  Cfg_.Dataset);
		Cfg_.OutDir  = OaYaml::Get<OaString>(g, "out_dir",  Cfg_.OutDir);
		Cfg_.GenCount      = OaYaml::Get<OaI32>(g, "count",      Cfg_.GenCount);
		Cfg_.GenTemperature = OaYaml::Get<OaF32>(g, "temperature", Cfg_.GenTemperature);
		Cfg_.GenMaxLen     = OaYaml::Get<OaI32>(g, "max_len",   Cfg_.GenMaxLen);
		Cfg_.Seed          = OaYaml::Get<OaI32>(g, "seed",      Cfg_.Seed);
		Cfg_.ConditioningClip = OaYaml::Get<OaI32>(g, "conditioning_clip", Cfg_.ConditioningClip);
		Cfg_.CaptionIndex = OaYaml::Get<OaI32>(g, "caption_index", Cfg_.CaptionIndex);
		Cfg_.Prompt        = OaYaml::Get<OaString>(g, "prompt", Cfg_.Prompt);
		Cfg_.TextFeature   = OaYaml::Get<OaString>(g, "text_feature", Cfg_.TextFeature);
		Cfg_.PrecisionStr  = OaYaml::Get<OaString>(g, "precision", Cfg_.PrecisionStr);
	}
};

// ── Helpers ──────────────────────────────────────────────────────────────────

namespace {

OaResult<std::vector<OaF32>> LoadRawF32(const OaString& InPath, OaI32 InCount) {
	std::ifstream file(InPath.CStr(), std::ios::binary | std::ios::ate);
	if (not file) return OaStatus::Error("cannot open prompt feature: " + InPath);
	const auto bytes = file.tellg();
	const std::streamoff expected = static_cast<std::streamoff>(InCount)
		* static_cast<std::streamoff>(sizeof(OaF32));
	if (bytes != expected) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"prompt feature byte count does not match the OaAlmAg text encoder");
	}
	file.seekg(0);
	std::vector<OaF32> values(static_cast<size_t>(InCount));
	file.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(expected));
	if (not file) return OaStatus::Error("failed to read prompt feature: " + InPath);
	return values;
}

// Copy a matrix to host FP32. Safe for BF16/FP16 storage models.
OaVec<OaF32> HostFloatData(const OaMatrix& InMatrix) {
	auto& ctx = OaContext::GetDefault();
	if (InMatrix.GetDtype() == OaScalarType::Float32) {
		(void)ctx.Execute(); (void)ctx.Sync();
		const OaF32* p = InMatrix.DataAs<const OaF32>();
		return OaVec<OaF32>(p, p + InMatrix.NumElements());
	}
	OaMatrix f32 = OaFnMatrix::Empty(InMatrix.GetShape(), OaScalarType::Float32);
	OaFnMatrix::CastInto(InMatrix, f32);
	(void)ctx.Execute(); (void)ctx.Sync();
	const OaF32* p = f32.DataAs<const OaF32>();
	return OaVec<OaF32>(p, p + f32.NumElements());
}

} // namespace

// ── App ─────────────────────────────────────────────────────────────────────

struct GenAlmApp : OaComputeApp {
	GenAlmCli Cli;

	int Setup(int argc, char** argv) override {
		if (not Cli.Parse(argc, argv)) { IsRunning = false; }
		const auto& c = Cli.GetConfig();
		EngineConfig_.Precision = c.Precision();
		return 0;
	}

	OaStatus Init() override {
		return OaStatus::Ok();
	}

	OaStatus Tick() override {
		const auto& c = Cli.GetConfig();
		auto& ctx = OaContext::GetDefault();

		if (c.Model.Empty()) {
			OA_LOG_ERROR(OaLogComponent::ML, "genalm: --model is required");
			IsRunning = false; return OaStatus::Ok();
		}
		auto loadedAlm = OaAlmAg::LoadBundle(c.Model);
		if (not loadedAlm.IsOk()) {
			OA_LOG_ERROR(OaLogComponent::ML, "genalm: OaAlmAg load failed: %s",
				loadedAlm.GetStatus().GetMessage().CStr());
			IsRunning = false; return OaStatus::Ok();
		}
		auto alm = std::move(loadedAlm).GetValue();
		const auto& almCfg = alm->Config();
		const bool conditioned = almCfg.Prior.TextFeatureDim > 0;
		if (c.GenMaxLen <= 0 or c.GenMaxLen + (conditioned ? 1 : 0) > almCfg.Prior.MaxSeqLen) {
			OA_LOG_ERROR(OaLogComponent::ML,
				"genalm: generation length %d does not fit bundle MaxSeqLen %d",
				c.GenMaxLen, almCfg.Prior.MaxSeqLen);
			IsRunning = false; return OaStatus::Ok();
		}

		// ── Load dataset (for denormalization Mean/Std) ──
		if (c.ConditioningClip < 0 or c.CaptionIndex < 0) {
			OA_LOG_ERROR(OaLogComponent::ML, "genalm: conditioning clip/caption indices cannot be negative");
			IsRunning = false; return OaStatus::Ok();
		}
		const bool promptMode = not c.Prompt.Empty();
		const bool featureOverride = not c.TextFeature.Empty();
		if (promptMode and featureOverride) {
			OA_LOG_ERROR(OaLogComponent::ML,
				"genalm: --prompt and --text-feature are mutually exclusive");
			IsRunning = false; return OaStatus::Ok();
		}
		if ((promptMode or featureOverride) and not conditioned) {
			OA_LOG_ERROR(OaLogComponent::ML,
				"genalm: text cannot condition an unconditional OaAlmAg bundle");
			IsRunning = false; return OaStatus::Ok();
		}
		if (promptMode and not alm->HasNativeTextEncoder()) {
			OA_LOG_ERROR(OaLogComponent::ML,
				"genalm: this OaAlmAg bundle has no native text encoder");
			IsRunning = false; return OaStatus::Ok();
		}
		OaDsCmp ds(c.Dataset, c.Split,
			conditioned and not promptMode and not featureOverride ? c.ConditioningClip + 1 : 1);
		if (not ds.Ok()) {
			OA_LOG_ERROR(OaLogComponent::ML, "genalm: failed to load CMP from %s", c.Dataset.CStr());
			IsRunning = false; return OaStatus::Ok();
		}
		const OaI32 featDim   = ds.FeatDim();
		const OaI32 textFeatureDim = almCfg.Prior.TextFeatureDim;
		const OaF32* textFeatureData = nullptr;
		std::vector<OaF32> promptFeature;
		OaString conditioningCaption;
		if (featureOverride) {
			auto loadedFeature = LoadRawF32(c.TextFeature, textFeatureDim);
			if (not loadedFeature.IsOk()) {
				OA_LOG_ERROR(OaLogComponent::ML, "genalm: %s",
					loadedFeature.GetStatus().GetMessage().CStr());
				IsRunning = false; return OaStatus::Ok();
			}
			promptFeature = std::move(loadedFeature).GetValue();
			textFeatureData = promptFeature.data();
			conditioningCaption = "external feature override";
		} else if (promptMode) {
			conditioningCaption = c.Prompt;
		} else if (conditioned) {
			if (c.ConditioningClip >= ds.NumClips()) {
				OA_LOG_ERROR(OaLogComponent::ML,
					"genalm: conditioning clip %d is outside split '%s' (%lld clips loaded)",
					c.ConditioningClip, c.Split.CStr(), static_cast<long long>(ds.NumClips()));
				IsRunning = false; return OaStatus::Ok();
			}
			const OaI32 featureCount = ds.ClipTextFeatureCount(c.ConditioningClip);
			if (textFeatureDim <= 0 or ds.TextFeatureFormat() != "oa_clip_text_v1" or
				ds.TextFeatureModel() != almCfg.TextEncoder or c.CaptionIndex >= featureCount or
				c.CaptionIndex >= static_cast<OaI32>(ds.ClipCaptions(c.ConditioningClip).Size())) {
				OA_LOG_ERROR(OaLogComponent::ML,
					"genalm: missing CLIP feature row %d for clip %s; run trainalm once to native-bake the caption cache",
					c.CaptionIndex, ds.ClipId(c.ConditioningClip).CStr());
				IsRunning = false; return OaStatus::Ok();
			}
			textFeatureData = ds.ClipTextFeatureData(c.ConditioningClip)
				+ static_cast<size_t>(c.CaptionIndex) * textFeatureDim;
			conditioningCaption = ds.ClipCaptions(c.ConditioningClip)[c.CaptionIndex].Text;
		}

		OaFnMatrix::SetRngSeed(static_cast<OaU64>(c.Seed));
		std::printf("Loaded OaAlmAg: %s\n", c.Model.CStr());

		// ── Generate ──
		(void)OaFileIo::CreateDirectories(OaPath(c.OutDir));
		ctx.Clear();
		OaMatrix textFeature;
		if (promptMode) {
			auto encodedPrompt = alm->EncodePrompt(c.Prompt);
			if (encodedPrompt.IsError()) {
				OA_LOG_ERROR(OaLogComponent::ML, "genalm: prompt encoding failed: %s",
					encodedPrompt.GetStatus().GetMessage().CStr());
				IsRunning = false; return OaStatus::Ok();
			}
			textFeature = OaStdMove(encodedPrompt.GetValue());
			auto hostFeature = HostFloatData(textFeature);
			promptFeature.assign(hostFeature.Data(), hostFeature.Data() + hostFeature.Size());
			textFeatureData = promptFeature.data();
		} else if (conditioned) {
			textFeature = OaFnMatrix::FromBytes(
				OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(textFeatureData),
					static_cast<size_t>(textFeatureDim) * sizeof(OaF32)),
				OaMatrixShape{1, textFeatureDim}, OaScalarType::Float32);
		}

		std::printf("\ngenalm — generating %d clips (maxLen=%d, seed=%d)\n",
			c.GenCount, c.GenMaxLen, c.Seed);
		if (conditioned) {
			std::printf("Conditioning: %s-%d · %s · \"%s\"\n",
				almCfg.TextEncoder.CStr(), textFeatureDim,
				promptMode ? "native prompt" : (featureOverride ? "feature override" : "dataset caption"),
				conditioningCaption.CStr());
		}

		OaStopwatch totalTimer;
		totalTimer.Start();

		for (OaI32 g = 0; g < c.GenCount; ++g) {
			const float temp = c.GenTemperature;

			OaStopwatch stepTimer;
			stepTimer.Start();

			auto generated = conditioned
				? alm->Prior().GenerateConditioned(textFeature, temp, 0, 0.9F, c.GenMaxLen)
				: alm->Prior().Generate(1, temp, 0, 0.9F, c.GenMaxLen);
			auto motion = alm->Prior().DecodeToMotion(generated, alm->Tokenizer());
			(void)ctx.Execute(); (void)ctx.Sync();

			const double genMs = stepTimer.ElapsedMs();

			const OaI32 frames = static_cast<OaI32>(motion.Size(0));
			if (frames <= 0) {
				std::printf("  [gen %d] T=%.2f: empty motion\n", g, temp);
				continue;
			}

			const double tokSps = static_cast<double>(generated.NumElements()) / (genMs * 0.001);
			const double frameSps = static_cast<double>(frames) / (genMs * 0.001);

			std::printf("  [gen %d] T=%.2f: %d frames × %lld dims | %.1f ms | %.0f tok/s | %.0f fps\n",
				g, temp, frames, static_cast<long long>(motion.Size(1)),
				genMs, tokSps, frameSps);

			// Denormalize and recover world joints.
			auto motionHost = HostFloatData(motion);
			std::vector<float> featHost(motionHost.Data(), motionHost.Data() + motionHost.Size());
			ds.Denormalize(featHost.data(), frames);
			auto worldJoints = OaHumanMl3dRecoverWorldJoints(
				OaSpan<const OaF32>(featHost.data(), featHost.size()), frames, featDim);

			auto skelClip = OaUsdClipFromWorldJoints(
				OaSkHumanMl3d(),
				OaSpan<const OaF32>(worldJoints.Data(), worldJoints.Size()),
				frames, 20.0F, 1, 100.0F);

			char pathBuf[256];
			std::snprintf(pathBuf, sizeof(pathBuf), "%s/%s_gen_%d_T%.1f.usda",
				c.OutDir.CStr(), c.Name.CStr(), g, temp);
			OaPath usdPath(pathBuf);
			auto usdSt = OaUsd::WriteUsda(usdPath, skelClip, "humanml3d");
			std::printf("         saved %s (%s)\n",
				usdPath.CStr(), usdSt.IsOk() ? "ok" : usdSt.ToString().CStr());

			// Keep generation provenance beside the preview. This is intentionally a
			// transparent line-based manifest rather than hidden checkpoint state.
			std::ostringstream metadata;
			metadata << "format=oa_alm_generation_v1\n"
				<< "model=" << c.Name.CStr() << "\n"
				<< "dataset=" << c.Dataset.CStr() << "\n"
				<< "split=" << c.Split.CStr() << "\n"
				<< "bundle=" << c.Model.CStr() << "\n"
				<< "seed=" << c.Seed << "\n"
				<< "temperature=" << temp << "\n"
				<< "max_motion_tokens=" << c.GenMaxLen << "\n"
				<< "generated_tokens=" << generated.NumElements() << "\n"
				<< "frames=" << frames << "\n"
				<< "feature_dim=" << featDim << "\n"
				<< "position_inverse=humanml3d_recover_from_ric\n"
				<< "usd=" << usdPath.CStr() << "\n";
			if (conditioned) {
				metadata << "text_feature_model=" << almCfg.TextEncoder.CStr() << "\n"
					<< "text_feature_dim=" << textFeatureDim << "\n"
					<< "text_feature_hash=" << OamHash(
						reinterpret_cast<const OaU8*>(textFeatureData),
						static_cast<OaUsize>(textFeatureDim) * sizeof(OaF32)) << "\n"
					<< "prompt=" << conditioningCaption.CStr() << "\n";
				if (featureOverride) {
					metadata << "text_feature_file=" << c.TextFeature.CStr() << "\n";
				} else if (promptMode) {
					metadata << "text_encoder=native_oa_vulkan\n";
				} else {
					metadata << "conditioning_clip_index=" << c.ConditioningClip << "\n"
						<< "conditioning_clip_id=" << ds.ClipId(c.ConditioningClip).CStr() << "\n"
						<< "caption_index=" << c.CaptionIndex << "\n";
				}
			}
			const OaPath metadataPath(OaString(pathBuf) + ".meta.txt");
			const auto metadataSt = OaFileIo::WriteText(
				metadataPath, OaString(metadata.str().c_str()));
			std::printf("         metadata %s (%s)\n", metadataPath.CStr(),
				metadataSt.IsOk() ? "ok" : metadataSt.ToString().CStr());
			std::fflush(stdout);
		}

		const double totalSec = totalTimer.ElapsedSec();
		std::printf("\nGeneration complete: %d clips in %.2f s\n", c.GenCount, totalSec);
		std::fflush(stdout);

		IsRunning = false;
		return OaStatus::Ok();
	}
};

int main(int argc, char** argv) {
	GenAlmApp app;
	return app.Main(argc, argv);
}
