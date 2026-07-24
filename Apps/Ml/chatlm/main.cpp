// Unified Model Chat (Vulkan Compute Inference)
//
// Architecture auto-detected from .oam checkpoint.
// Supports: llm (OaLlm), gpt2 (OaGpt2; .oam: oagpt_v1)
//
// Usage: ./chat -m var/model/dev/OaLlm/OaLlm.oam
//        ./chat -m model.oam -p "Once upon a time" -t 0.5

#include <Oa/Runtime/App.h>  // OaComputeApp
#include <Oa/Ml/Adapter.h>
#include <Oa/Ml/AdapterRegistry.h>
#include <Ml/Extensions/Pack/RealmDefault/RealmDefaultPackExtension.h>
#include <Ml/Config.h>
#include <Ml/SpirvRegistry.h>
#include <Oa/Ml/Oam.h>
#include <Oa/Core/Log.h>
#include <Oa/Core/Filesystem.h>

#include <chrono>
#include <csignal>
#include <cstdio>
#include <iostream>
#include <string>

static volatile sig_atomic_t GInterrupted = 0;
static void HandleSigint(int) { GInterrupted = 1; }

static void LogGenerationMetrics(
	OaBool InVerbose,
	OaI32 InPromptBytes,
	size_t InOutBytes,
	double InSeconds
) {
	if (!InVerbose) return;
	OaI32 nGen = static_cast<OaI32>(InOutBytes) - InPromptBytes;
	if (nGen < 0) nGen = 0;
	const double tokPerSec = (InSeconds > 1e-9) ? (static_cast<double>(nGen) / InSeconds) : 0.0;
	const double ms = InSeconds * 1000.0;
	fprintf(stderr, "  gen: %d tok in %.2f ms (%.0f tok/s)\n", nGen, ms, tokPerSec);
	fflush(stderr);
}

static OaString ResolveArchName(const OaString& InOamArch) {
	if (InOamArch == "oallm_v6" || InOamArch == "oallm_v5" || InOamArch == "oallm_v4" || InOamArch == "llm") {
		return "llm";
	}
	if (InOamArch == "oagpt_v1" || InOamArch == "gpt" || InOamArch == "gpt2") { 
		return "gpt2";
	}
	return InOamArch;
}

struct ChatApp : OaComputeApp {
	OaChatCli Cli;
	OaUniquePtr<OaAdapter> Adapter;
	OaString ArchName;
	OamModel Oam;
	bool SinglePrompt = false;

	int Setup(int argc, char** argv) override {
		if (!Cli.Parse(argc, argv)) {
			IsRunning = false;
			return 0;
		}
		auto& cfg = Cli.GetConfig();

		if (cfg.ModelPath.empty()) {
			OA_LOG_ERROR(OaLogComponent::ML, "--model is required. Use --help for usage.");
			return 1;
		}

		auto oamResult = OamModel::Load(cfg.ModelPath);
		if (!oamResult) {
			OA_LOG_ERROR(OaLogComponent::ML, "Failed to load .oam: %s", cfg.ModelPath.c_str());
			return 1;
		}
		Oam = std::move(*oamResult);

		ArchName = ResolveArchName(OaString(Oam.Config.Architecture));
		OA_LOG_INFO(OaLogComponent::ML, "Architecture: %s (from .oam: %s)",
			ArchName.c_str(), Oam.Config.Architecture);

		Adapter = OaAdapterRegistry::Get().Create(ArchName);
		if (!Adapter) {
			OA_LOG_ERROR(OaLogComponent::ML, "Unknown architecture: %s", ArchName.c_str());
			return 1;
		}

		auto devicePref = OaDevicePreference::Discrete;
		auto precision = cfg.Precision();

		EngineConfig_ = {
			.DevicePref = devicePref,
			.Precision = precision,
			.EnableValidation = cfg.Validate,
			.AppName = "chat",
			.MeshVulkanIndices = {}
		};

		SinglePrompt = !cfg.Prompt.empty();
		return 0;
	}

	OaStatus Init() override {
		auto& cfg = Cli.GetConfig();
		MlSpvInit();
		MlAddShaderSearchPaths(Rt);

		OA_RETURN_IF_ERROR(Adapter->InitFromOam(Rt, cfg.ModelPath, Oam));
		OA_RETURN_IF_ERROR(Adapter->Load(Rt, cfg.ModelPath));

		OA_LOG_INFO(OaLogComponent::ML, "Model loaded: %s (%.2fM params)",
			cfg.ModelPath.c_str(), static_cast<OaF64>(Adapter->NumParams()) / 1e6);

		if (SinglePrompt) return OaStatus::Ok();

		signal(SIGINT, HandleSigint);
		fprintf(stderr, "\n");
		fprintf(stderr, "  OA Chat [arch=%s] (Vulkan Compute)\n", ArchName.c_str());
		fprintf(stderr, "  Model: %s (%.2fM params)\n",
			cfg.ModelPath.c_str(), static_cast<OaF64>(Adapter->NumParams()) / 1e6);
		fprintf(stderr, "  Temperature: %.1f | Max tokens: %d | Seq: %d | Tok/s metrics: %s\n",
			cfg.Temperature, cfg.MaxTokens, cfg.SeqLen, cfg.Verbose ? "on" : "off");
		fprintf(stderr, "  Ctrl+C or Ctrl+D to exit.\n");
		fprintf(stderr, "\n");
		return OaStatus::Ok();
	}

	OaStatus Tick() override {
		auto& cfg = Cli.GetConfig();

		if (SinglePrompt) {
			OaVec<OaU8> prompt(cfg.Prompt.begin(), cfg.Prompt.end());
			const OaI32 promptLen = static_cast<OaI32>(prompt.Size());
			fprintf(stdout, "%s", cfg.Prompt.c_str());
			fflush(stdout);
			const auto t0 = std::chrono::steady_clock::now();
			OaVec<OaU8> out = Adapter->Generate(
				Rt, OaSpan<const OaU8>(prompt.Data(), prompt.Size()), cfg.MaxTokens, cfg.Temperature);
			const auto t1 = std::chrono::steady_clock::now();
			fflush(stdout);
			const double sec = std::chrono::duration<OaF64>(t1 - t0).count();
			LogGenerationMetrics(cfg.Verbose, promptLen, out.Size(), sec);
			fprintf(stdout, "\n");
			IsRunning = false;
			return OaStatus::Ok();
		}

		if (GInterrupted) {
			IsRunning = false;
			return OaStatus::Ok();
		}

		fprintf(stderr, "> ");
		fflush(stderr);
		std::string line;
		if (!std::getline(std::cin, line) || GInterrupted) {
			IsRunning = false;
			return OaStatus::Ok();
		}
		if (line.empty()) return OaStatus::Ok();

		OaVec<OaU8> prompt(line.begin(), line.end());
		const OaI32 promptLen = static_cast<OaI32>(prompt.Size());
		fprintf(stdout, "\n< %s", line.c_str());
		fflush(stdout);
		const auto t0 = std::chrono::steady_clock::now();
		OaVec<OaU8> out = Adapter->Generate(
			Rt, OaSpan<const OaU8>(prompt.Data(), prompt.Size()), cfg.MaxTokens, cfg.Temperature);
		const auto t1 = std::chrono::steady_clock::now();
		fflush(stdout);
		const double sec = std::chrono::duration<OaF64>(t1 - t0).count();
		LogGenerationMetrics(cfg.Verbose, promptLen, out.Size(), sec);
		fprintf(stdout, "\n\n");
		fflush(stdout);

		return OaStatus::Ok();
	}

	void Shutdown() override {
		Adapter->Destroy();
		fprintf(stderr, "\nBye.\n");
	}
};

int main(int argc, char** argv) {
	OaRealmDefaultPackExtension::Get().RegisterAdapters(OaAdapterRegistry::Get());
	ChatApp app;
	app.AddExtension(&OaRealmDefaultPackExtension::Get());
	return app.Main(argc, argv);
}
