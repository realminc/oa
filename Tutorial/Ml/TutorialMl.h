// ═══════════════════════════════════════════════════════════════════════════
// TutorialMl.h — Unified helpers for OA ML tutorials
//
// Reduces boilerplate across tutorials by providing:
//   - Common config struct (batch, epochs, lr, device_index, etc.)
//   - YAML config loading from var/config/Base.yaml or custom path
//   - Standard progress bar / metrics / summary setup
//   - Header printing helpers
//   - Training loop wrapper
//
// Usage:
//   #include "TutorialMl.h"
//   TutorialMlConfig cfg = TutorialLoadConfig("var/config/Alm.yaml");
//   auto loop = TutorialMakeTrainingLoop(cfg, optimizer, kSteps, kBatch);
// ═══════════════════════════════════════════════════════════════════════════

#pragma once

#include <Oa/Ml/Callbacks.h>
#include <Oa/Ml/ItTraining.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Ml/Metric.h>
#include <Oa/Core/EnvFlag.h>
#include <Oa/Core/Yaml.h>
#include <Oa/Core/Time.h>

#include <cstdio>
#include <cstdlib>
#include <string>

// ─── TutorialMlConfig ─────────────────────────────────────────────────────

struct TutorialMlConfig {
	// Device
	OaI32 DeviceIndex = -1;  // -1 = auto (env var / default), 0+ = force index

	// Training
	OaI32 Epochs = 5;
	OaI32 Steps = 0;           // 0 = auto (epochs * steps_per_epoch)
	OaI32 BatchSize = 64;
	OaF32 Lr = 0.001f;

	// Logging
	OaBool PrintHeader = true;
	OaBool PrintSummary = true;

	// Paths
	OaString ConfigPath;
	OaString DataPath;
};

// ─── Device Index Pre-Parse ─────────────────────────────────────────────────
//
// Call from main() BEFORE testing::InitGoogleTest() to extract --device-index
// from argv. Returns the parsed index (-1 if not found) and removes the arg
// from argv so GTest doesn't see it.

inline OaI32 TutorialPreParseDeviceIndex(int& InOutArgc, char** InOutArgv) {
	OaI32 idx = -1;
	for (int i = 1; i < InOutArgc; ++i) {
		OaString arg(InOutArgv[i]);
		if ((arg == "--device-index" || arg == "-d") && i + 1 < InOutArgc) {
			idx = static_cast<OaI32>(std::strtol(InOutArgv[i + 1], nullptr, 10));
			// Remove both tokens by shifting
			for (int j = i; j + 2 < InOutArgc; ++j) {
				InOutArgv[j] = InOutArgv[j + 2];
			}
			InOutArgc -= 2;
			break;
		}
		if (arg.substr(0, 15) == "--device-index=") {
			idx = static_cast<OaI32>(std::strtol(arg.c_str() + 15, nullptr, 10));
			// Remove token by shifting
			for (int j = i; j + 1 < InOutArgc; ++j) {
				InOutArgv[j] = InOutArgv[j + 1];
			}
			InOutArgc -= 1;
			break;
		}
	}
	return idx;
}

// ─── Config Loading ───────────────────────────────────────────────────────

inline TutorialMlConfig TutorialLoadConfig(const OaString& InPath) {
	TutorialMlConfig cfg;
	if (InPath.empty()) return cfg;
	try {
		OaYaml::Node yaml = OaYaml::LoadFile(InPath);
		if (auto t = yaml["training"]) {
			cfg.BatchSize = OaYaml::Get<int>(t, "batch_size", cfg.BatchSize);
			cfg.Steps = OaYaml::Get<int>(t, "steps", cfg.Steps);
			cfg.Lr = OaYaml::Get<float>(t, "lr", cfg.Lr);
			cfg.DataPath = OaYaml::Get<OaString>(t, "data", cfg.DataPath);
		}
		if (auto e = yaml["engine"]) {
			cfg.DeviceIndex = OaYaml::Get<int>(e, "vulkan_index", cfg.DeviceIndex);
		}
	} catch (const OaYaml::Exception&) {
		// Config optional — use defaults
	}
	return cfg;
}

// Merge Base.yaml defaults then model config
inline TutorialMlConfig TutorialLoadConfigWithBase(const OaString& InModelConfigPath) {
	// Start with hardcoded defaults
	TutorialMlConfig cfg;
	// Load Base.yaml if present
	if (!InModelConfigPath.empty()) {
		cfg = TutorialLoadConfig("var/config/Base.yaml");
		cfg.ConfigPath = InModelConfigPath;
		// Model config overrides base
		TutorialMlConfig modelCfg = TutorialLoadConfig(InModelConfigPath);
		if (modelCfg.DeviceIndex >= 0) cfg.DeviceIndex = modelCfg.DeviceIndex;
		if (modelCfg.BatchSize > 0) cfg.BatchSize = modelCfg.BatchSize;
		if (modelCfg.Steps > 0) cfg.Steps = modelCfg.Steps;
		if (modelCfg.Lr > 0.0f) cfg.Lr = modelCfg.Lr;
		if (!modelCfg.DataPath.empty()) cfg.DataPath = modelCfg.DataPath;
	}
	return cfg;
}

// ─── Standard Callback Setup ────────────────────────────────────────────────

struct TutorialTrainingLoop {
	OaMetricLoss LossMetric;
	OaMetricAccuracy AccuracyMetric;
	OaCbProgressBar ProgressBar;
	OaCbSummary Summary;
	OaItTraining Loop;

	// Convenience: wraps the common callback+metric wiring.
	// Metrics, ProgressBar, and Summary are automatically registered with
	// the training loop, so callers do not need to pass .Callbacks that point
	// back into the not-yet-constructed TutorialTrainingLoop object.
	TutorialTrainingLoop(OaOptimizer& InOpt, const OaItTrainingConfig& InCfg)
		: LossMetric()
		, Loop(InOpt, InCfg)
	{
		Loop.AddMetric(&LossMetric);
		ProgressBar.AddMetric(&LossMetric);
		Loop.AddCallback(&ProgressBar);
		Loop.AddCallback(&Summary);
	}

	void AddAccuracyMetric() {
		Loop.AddMetric(&AccuracyMetric);
		ProgressBar.AddMetric(&AccuracyMetric);
	}
};

// ─── Header Printing ────────────────────────────────────────────────────────

inline void TutorialPrintBanner(const char* InTitle, const char* InSubtitle = nullptr) {
	std::printf("\n╔══════════════════════════════════════════════════════════════════╗\n");
	std::printf("║  %-64s║\n", InTitle);
	std::printf("╚══════════════════════════════════════════════════════════════════╝\n");
	if (InSubtitle) {
		std::printf("%s\n", InSubtitle);
	}
}

inline void TutorialPrintTrainingHeader(OaI32 InEpochs, OaI32 InStepsPerEpoch, OaI32 InBatchSize) {
	if (InEpochs > 0) {
		std::printf("Training: %d epochs × %d steps/epoch · batch=%d\n",
			InEpochs, InStepsPerEpoch, InBatchSize);
	} else {
		std::printf("Training: %d steps · batch=%d\n",
			InStepsPerEpoch * InEpochs, InBatchSize);
	}
}

// ─── Env Var Helpers ────────────────────────────────────────────────────────

// Apply device index from config to the engine. Call before engine creation.
inline void TutorialApplyDeviceIndex(const TutorialMlConfig& InCfg) {
	if (InCfg.DeviceIndex >= 0) {
		OaString idxStr = OaString(std::to_string(InCfg.DeviceIndex).c_str());
#if defined(_WIN32)
		_putenv_s("OA_DEVICE", idxStr.c_str());
#else
		::setenv("OA_DEVICE", idxStr.c_str(), 1);
#endif
	}
}

// Override config from env var (env wins over YAML)
inline void TutorialApplyEnvOverrides(TutorialMlConfig& InOutCfg) {
	if (const char* d = std::getenv("OA_DEVICE"); d && *d) {
		char* end = nullptr;
		unsigned long v = std::strtoul(d, &end, 10);
		if (end != d && *end == '\0') {
			InOutCfg.DeviceIndex = static_cast<OaI32>(v);
		}
	}
	if (const char* v = std::getenv("OA_TUTORIAL_BATCH_SIZE"); v && *v) {
		InOutCfg.BatchSize = static_cast<OaI32>(std::strtol(v, nullptr, 10));
	}
	if (const char* v = std::getenv("OA_TUTORIAL_STEPS"); v && *v) {
		InOutCfg.Steps = static_cast<OaI32>(std::strtol(v, nullptr, 10));
	}
	if (const char* v = std::getenv("OA_TUTORIAL_LR"); v && *v) {
		InOutCfg.Lr = static_cast<OaF32>(std::strtod(v, nullptr));
	}
}
