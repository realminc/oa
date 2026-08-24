// ═══════════════════════════════════════════════════════════════════════════
// TutorialMl.h — Unified helpers for OA ML tutorials
//
// Reduces boilerplate across tutorials by providing:
//   - Common config struct (batch, epochs, lr, device_index, etc.)
//   - YAML config loading from var/config/base.yaml or custom path
//   - Standard progress bar / metrics / summary setup
//   - header printing helpers
//   - training loop wrapper
//
// usage:
//   #include "tutorialMl.h"
//   TutorialMlConfig cfg = tutorialLoadConfig("var/config/Alm.yaml");
//   auto loop = tutorialMakeTrainingLoop(cfg, optimizer, kSteps, kBatch);
// ═══════════════════════════════════════════════════════════════════════════

#pragma once

#include <oa/ml/callbacks.h>
#include <oa/ml/itTraining.h>
#include <oa/runtime/engine.h>
#include <oa/ml/metric.h>
#include <oa/core/envFlag.h>
#include <oa/core/yaml.h>
#include <oa/core/time.h>

#include <cstdio>
#include <cstdlib>
#include <string>

// ─── TutorialMlConfig ─────────────────────────────────────────────────────

struct TutorialMlConfig {
	// Device
	oa::I32 deviceIndex = -1;  // -1 = auto (env var / default), 0+ = force index

	// training
	oa::I32 epochs = 5;
	oa::I32 steps = 0;           // 0 = auto (epochs * steps_per_epoch)
	oa::I32 batchSize = 64;
	oa::F32 lr = 0.001f;

	// Logging
	oa::Bool printHeader = true;
	oa::Bool printSummary = true;

	// Paths
	oa::String configPath;
	oa::String dataPath;
};

// Complete the eager work recorded through the public operation surface.
// oa::ExecutionSession is an implementation-owned recorder; tutorials synchronize through
// the engine and the exact event returned by submission.
[[nodiscard]] inline oa::Status tutorialSubmitAndWait(oa::Engine& inEngine) {
	auto submitted = inEngine.submit();
	if (not submitted.isOk()) {
		return submitted.getStatus();
	}
	return inEngine.wait(submitted.getValue());
}

// ─── Device index Pre-parse ─────────────────────────────────────────────────
//
// call from main() BEFORE testing::InitGoogleTest() to extract --device-index
// from argv. Returns the parsed index (-1 if not found) and removes the arg
// from argv so GTest doesn't see it.

inline oa::I32 tutorialPreParseDeviceIndex(int& inOutArgc, char** inOutArgv) {
	oa::I32 idx = -1;
	for (int i = 1; i < inOutArgc; ++i) {
		oa::String arg(inOutArgv[i]);
		if ((arg == "--device-index" || arg == "-d") && i + 1 < inOutArgc) {
			idx = static_cast<oa::I32>(std::strtol(inOutArgv[i + 1], nullptr, 10));
			// remove both tokens by shifting
			for (int j = i; j + 2 < inOutArgc; ++j) {
				inOutArgv[j] = inOutArgv[j + 2];
			}
			inOutArgc -= 2;
			break;
		}
		if (arg.substr(0, 15) == "--device-index=") {
			idx = static_cast<oa::I32>(std::strtol(arg.cStr() + 15, nullptr, 10));
			// remove token by shifting
			for (int j = i; j + 1 < inOutArgc; ++j) {
				inOutArgv[j] = inOutArgv[j + 1];
			}
			inOutArgc -= 1;
			break;
		}
	}
	return idx;
}

// ─── Config Loading ───────────────────────────────────────────────────────

inline TutorialMlConfig tutorialLoadConfig(const oa::String& inPath) {
	TutorialMlConfig cfg;
	if (inPath.empty()) return cfg;
	try {
		oa::Yaml::Node yaml = oa::Yaml::loadFile(inPath);
		if (auto t = yaml["training"]) {
			cfg.batchSize = oa::Yaml::get<int>(t, "batch_size", cfg.batchSize);
			cfg.steps = oa::Yaml::get<int>(t, "steps", cfg.steps);
			cfg.lr = oa::Yaml::get<float>(t, "lr", cfg.lr);
			cfg.dataPath = oa::Yaml::get<oa::String>(t, "data", cfg.dataPath);
		}
		if (auto e = yaml["engine"]) {
			cfg.deviceIndex = oa::Yaml::get<int>(e, "vulkan_index", cfg.deviceIndex);
		}
	} catch (const oa::Yaml::Exception&) {
		// Config optional — use defaults
	}
	return cfg;
}

// Merge base.yaml defaults then model config
inline TutorialMlConfig tutorialLoadConfigWithBase(const oa::String& inModelConfigPath) {
	// Start with hardcoded defaults
	TutorialMlConfig cfg;
	// load base.yaml if present
	if (!inModelConfigPath.empty()) {
		cfg = tutorialLoadConfig("var/config/base.yaml");
		cfg.configPath = inModelConfigPath;
		// Model config overrides base
		TutorialMlConfig modelCfg = tutorialLoadConfig(inModelConfigPath);
		if (modelCfg.deviceIndex >= 0) cfg.deviceIndex = modelCfg.deviceIndex;
		if (modelCfg.batchSize > 0) cfg.batchSize = modelCfg.batchSize;
		if (modelCfg.steps > 0) cfg.steps = modelCfg.steps;
		if (modelCfg.lr > 0.0f) cfg.lr = modelCfg.lr;
		if (!modelCfg.dataPath.empty()) cfg.dataPath = modelCfg.dataPath;
	}
	return cfg;
}

// ─── Standard Callback Setup ────────────────────────────────────────────────

struct TutorialTrainingLoop {
	oa::MetricLoss lossMetric;
	oa::MetricAccuracy accuracyMetric;
	oa::CbProgressBar progressBar;
	oa::CbSummary summary;
	oa::ItTraining loop;

	// Convenience: wraps the common callback+metric wiring.
	// metrics, progressBar, and summary are automatically registered with
	// the training loop, so callers do not need to pass .callbacks that point
	// back into the not-yet-constructed TutorialTrainingLoop object.
	TutorialTrainingLoop(
		oa::Engine& inEngine,
		oa::Optimizer& inOpt,
		const oa::ItTrainingConfig& inCfg)
		: lossMetric()
		, loop(inEngine, inOpt, inCfg)
	{
		loop.addMetric(&lossMetric);
		progressBar.addMetric(&lossMetric);
		loop.addCallback(&progressBar);
		loop.addCallback(&summary);
	}

	void addAccuracyMetric() {
		loop.addMetric(&accuracyMetric);
		progressBar.addMetric(&accuracyMetric);
	}
};

// ─── header Printing ────────────────────────────────────────────────────────

inline void tutorialPrintBanner(const char* inTitle, const char* inSubtitle = nullptr) {
	std::printf("\n╔══════════════════════════════════════════════════════════════════╗\n");
	std::printf("║  %-64s║\n", inTitle);
	std::printf("╚══════════════════════════════════════════════════════════════════╝\n");
	if (inSubtitle) {
		std::printf("%s\n", inSubtitle);
	}
}

inline void tutorialPrintTrainingHeader(oa::I32 inEpochs, oa::I32 inStepsPerEpoch, oa::I32 inBatchSize) {
	if (inEpochs > 0) {
		std::printf("training: %d epochs × %d steps/epoch · batch=%d\n",
			inEpochs, inStepsPerEpoch, inBatchSize);
	} else {
		std::printf("training: %d steps · batch=%d\n",
			inStepsPerEpoch * inEpochs, inBatchSize);
	}
}

// ─── env Var Helpers ────────────────────────────────────────────────────────

// apply device index from config to the engine. call before engine creation.
inline void tutorialApplyDeviceIndex(const TutorialMlConfig& inCfg) {
	if (inCfg.deviceIndex >= 0) {
		oa::String idxStr = oa::String(std::to_string(inCfg.deviceIndex).c_str());
#if defined(_WIN32)
		_putenv_s("OA_DEVICE", idxStr.cStr());
#else
		::setenv("OA_DEVICE", idxStr.cStr(), 1);
#endif
	}
}

// Override config from env var (env wins over YAML)
inline void tutorialApplyEnvOverrides(TutorialMlConfig& inOutCfg) {
	if (const char* d = std::getenv("OA_DEVICE"); d && *d) {
		char* end = nullptr;
		unsigned long v = std::strtoul(d, &end, 10);
		if (end != d && *end == '\0') {
			inOutCfg.deviceIndex = static_cast<oa::I32>(v);
		}
	}
	if (const char* v = std::getenv("OA_TUTORIAL_BATCH_SIZE"); v && *v) {
		inOutCfg.batchSize = static_cast<oa::I32>(std::strtol(v, nullptr, 10));
	}
	if (const char* v = std::getenv("OA_TUTORIAL_STEPS"); v && *v) {
		inOutCfg.steps = static_cast<oa::I32>(std::strtol(v, nullptr, 10));
	}
	if (const char* v = std::getenv("OA_TUTORIAL_LR"); v && *v) {
		inOutCfg.lr = static_cast<oa::F32>(std::strtod(v, nullptr));
	}
}
