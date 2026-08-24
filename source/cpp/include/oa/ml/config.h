// OA ML - Configuration Structs
//
// ML-specific config: optimizer, training, VRAM budget / auto-tune, YAML loaders.
// Separate from oa/core/config.h (infrastructure: checkpoint, log, YAML helpers).
//
// These are building blocks — consumer code composes them:
//
//   struct MyTrainConfig {
//       oa::OptimizerConfig optimizer;
//       oa::TrainBaseConfig training;
//       oa::VramBudgetConfig vram;  // optional auto batch/seq from VRAM
//       oa::LogConfig log;          // from core/config.h
//       oa::String dataPath;        // app-specific
//   };

#pragma once

#include <oa/core/config.h>
#include <oa/core/device.h>

namespace oa {

class Engine;

// OPTIMIZER CONFIG — AdamW, SGD, schedulers

class OptimizerConfig {
public:
	oa::F32 lr = 3e-4f;                   // Learning rate
	oa::F32 lrMin = 1e-5f;                // minimum LR (for schedulers)
	oa::F32 beta1 = 0.9f;                 // Adam beta1
	oa::F32 beta2 = 0.999f;               // Adam beta2
	oa::F32 epsilon = 1e-8f;              // Adam epsilon
	oa::F32 weightDecay = 0.01f;          // AdamW weight decay
	oa::F32 maxGradNorm = 1.0f;           // Gradient clipping
	oa::String scheduler = "cosine";       // cosine | constant | linear | cosine_restarts
	oa::I64 warmupSteps = 1000;           // LR warmup
	oa::I64 cycleSteps = 0;              // For cosine_restarts
	oa::F32 cycleMultiplier = 1.0f;
};

// TRAINING PHASE CONFIG — Multi-phase training schedule

struct TrainPhaseConfig {
	oa::String id;                         // phase identifier for logging/checkpoints
	oa::I64 steps = 0;                     // steps in this phase
	oa::Optional<oa::F32> lr;                     // Override learning rate
	oa::Optional<oa::F32> lrMin;                  // Override minimum LR
	oa::Optional<oa::I64> warmupSteps;            // Override warmup steps
	oa::Optional<oa::I32> checkpointInterval;     // Override checkpoint interval
};

// BASE TRAINING CONFIG — Shared across LLM, RL, GAN, etc.

class TrainBaseConfig {
public:
	oa::I32 batchSize = 32;
	oa::I64 totalSteps = 100000;
	oa::I32 seed = 42;
	oa::Precision precision = oa::Precision::FP32;
	oa::String device = "vulkan";          // vulkan | cpu

	// Evaluation and checkpoint policies
	oa::I32 evalInterval = 1000;          // Evaluate every N steps
	oa::I32 checkpointInterval = 5000;    // save every N steps

	// Gradient accumulation
	oa::I32 gradAccumSteps = 1;

	// Data augmentation (common)
	oa::Bool augmentationEnabled = false;
	oa::F32 noiseStd = 0.0f;

	// Multi-phase training (optional)
	oa::Vec<TrainPhaseConfig> phases;

	[[nodiscard]] bool usesPhases() const { return !phases.empty(); }
};

// YAML LOADERS — ML config sections

inline void loadOptimizerYaml(
	const oa::Yaml::Node& inYaml,
	OptimizerConfig& outConfig
) {
	if (auto o = inYaml["optimizer"]) {
		outConfig.lr = oa::Yaml::get<float>(o, "lr", outConfig.lr);
		outConfig.lrMin = oa::Yaml::get<float>(o, "lr_min", outConfig.lrMin);
		outConfig.beta1 = oa::Yaml::get<float>(o, "beta1", outConfig.beta1);
		outConfig.beta2 = oa::Yaml::get<float>(o, "beta2", outConfig.beta2);
		outConfig.epsilon = oa::Yaml::get<float>(o, "epsilon", outConfig.epsilon);
		outConfig.weightDecay = oa::Yaml::get<float>(o, "weight_decay", outConfig.weightDecay);
		outConfig.maxGradNorm = oa::Yaml::get<float>(o, "max_grad_norm", outConfig.maxGradNorm);
		outConfig.scheduler = oa::Yaml::get<oa::String>(o, "scheduler", outConfig.scheduler);
		outConfig.warmupSteps = oa::Yaml::get<int64_t>(o, "warmup_steps", outConfig.warmupSteps);
		outConfig.cycleSteps = oa::Yaml::get<int64_t>(o, "cycle_steps", outConfig.cycleSteps);
		outConfig.cycleMultiplier = oa::Yaml::get<float>(o, "cycle_multiplier", outConfig.cycleMultiplier);
	}
}

inline void loadTrainBaseYaml(
	const oa::Yaml::Node& inYaml,
	TrainBaseConfig& outConfig
) {
	if (auto t = inYaml["training"]) {
		outConfig.batchSize = oa::Yaml::get<int>(t, "batch_size", outConfig.batchSize);
		outConfig.totalSteps = oa::Yaml::get<int64_t>(t, "total_steps", outConfig.totalSteps);
		outConfig.seed = oa::Yaml::get<int>(t, "seed", outConfig.seed);
		outConfig.evalInterval = oa::Yaml::get<int>(t, "eval_interval", outConfig.evalInterval);
		outConfig.checkpointInterval = oa::Yaml::get<int>(t, "checkpoint_interval", outConfig.checkpointInterval);
		outConfig.gradAccumSteps = oa::Yaml::get<int>(t, "grad_accum_steps", outConfig.gradAccumSteps);

		auto precStr = oa::Yaml::get<oa::String>(t, "precision", oa::String("fp32"));
		if (precStr == "fp32") outConfig.precision = oa::Precision::FP32;
		else if (precStr == "bf16") outConfig.precision = oa::Precision::BF16;
		else if (precStr == "fp64") outConfig.precision = oa::Precision::FP64;
		else outConfig.precision = oa::Precision::FP32;

		outConfig.device = oa::Yaml::get<oa::String>(t, "device", outConfig.device);
	}
}

// VRAM BUDGET — auto-tune batch_size * seq_len from free VRAM (implementation in budget.cpp)

class VramBudgetConfig {
public:
	oa::F32 safetyMarginPercent = 0.10f;
	oa::I32 minBatchSize = 1;
	oa::I32 maxBatchSize = 4096;
	oa::I32 minSeqLen = 64;
	oa::I32 maxSeqLen = 8192;
	oa::I32 preferredBatchSize = 0;
	oa::I32 preferredSeqLen = 0;
	oa::I32 optimizerStatesPerParam = 4;
	oa::I32 bytesPerParam = 4;
};

class VramBudgetResult {
public:
	oa::I32 batchSize = 0;
	oa::I32 seqLen = 0;
	oa::Usize modelBytes = 0;
	oa::Usize activationBytes = 0;
	oa::Usize totalBytes = 0;
	oa::Usize availableBytes = 0;
	oa::F32 utilizationPercent = 0.0f;
	oa::Bool fitsInVRAM = false;
};

[[nodiscard]] VramBudgetResult computeVramBudget(
	const Engine& inEngine,
	Usize inModelParams,
	Usize inActivationBytesPerToken,
	VramBudgetConfig inConfig = {}
);

void printVramBudget(const VramBudgetResult& inResult);

} // namespace oa
