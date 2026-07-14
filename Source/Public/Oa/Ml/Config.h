// OA ML - Configuration Structs
//
// ML-specific config: optimizer, training, VRAM budget / auto-tune, YAML loaders.
// Separate from oa/core/config.h (infrastructure: checkpoint, log, YAML helpers).
//
// These are building blocks — consumer code composes them:
//
//   struct MyTrainConfig {
//       OaOptimizerConfig Optimizer;
//       OaTrainBaseConfig Training;
//       OaVRAMBudgetConfig Vram;  // optional auto batch/seq from VRAM
//       OaLogConfig Log;          // from core/config.h
//       OaString DataPath;        // app-specific
//   };

#pragma once

#include <Oa/Core/Config.h>
#include <Oa/Core/Device.h>

// OPTIMIZER CONFIG — AdamW, SGD, schedulers

class OaOptimizerConfig {
public:
	OaF32 Lr = 3e-4f;                   // Learning rate
	OaF32 LrMin = 1e-5f;                // Minimum LR (for schedulers)
	OaF32 Beta1 = 0.9f;                 // Adam beta1
	OaF32 Beta2 = 0.999f;               // Adam beta2
	OaF32 Epsilon = 1e-8f;              // Adam epsilon
	OaF32 WeightDecay = 0.01f;          // AdamW weight decay
	OaF32 MaxGradNorm = 1.0f;           // Gradient clipping
	OaString Scheduler = "cosine";       // cosine | constant | linear | cosine_restarts
	OaI64 WarmupSteps = 1000;           // LR warmup
	OaI64 CycleSteps = 0;              // For cosine_restarts
	OaF32 CycleMultiplier = 1.0f;
};

// TRAINING PHASE CONFIG — Multi-phase training schedule

struct OaTrainPhaseConfig {
	OaString Id;                         // Phase identifier for logging/checkpoints
	OaI64 Steps = 0;                     // Steps in this phase
	OaOpt<OaF32> Lr;                     // Override learning rate
	OaOpt<OaF32> LrMin;                  // Override minimum LR
	OaOpt<OaI64> WarmupSteps;            // Override warmup steps
	OaOpt<OaI32> CheckpointInterval;     // Override checkpoint interval
};

// BASE TRAINING CONFIG — Shared across LLM, RL, GAN, etc.

class OaTrainBaseConfig {
public:
	OaI32 BatchSize = 32;
	OaI64 TotalSteps = 100000;
	OaI32 Seed = 42;
	OaPrecision Precision = OaPrecision::BF16;
	OaString Device = "vulkan";          // vulkan | cpu

	// Evaluation and checkpoint policies
	OaI32 EvalInterval = 1000;          // Evaluate every N steps
	OaI32 CheckpointInterval = 5000;    // Save every N steps

	// Gradient accumulation
	OaI32 GradAccumSteps = 1;

	// Data augmentation (common)
	OaBool AugmentationEnabled = false;
	OaF32 NoiseStd = 0.0f;

	// Multi-phase training (optional)
	OaVec<OaTrainPhaseConfig> Phases;

	[[nodiscard]] bool UsesPhases() const { return !Phases.Empty(); }
};

// YAML LOADERS — ML config sections

inline void OaLoadOptimizerYaml(const OaYaml::Node& InYaml, OaOptimizerConfig& OutCfg) {
	if (auto o = InYaml["optimizer"]) {
		OutCfg.Lr = OaYaml::Get<float>(o, "lr", OutCfg.Lr);
		OutCfg.LrMin = OaYaml::Get<float>(o, "lr_min", OutCfg.LrMin);
		OutCfg.Beta1 = OaYaml::Get<float>(o, "beta1", OutCfg.Beta1);
		OutCfg.Beta2 = OaYaml::Get<float>(o, "beta2", OutCfg.Beta2);
		OutCfg.Epsilon = OaYaml::Get<float>(o, "epsilon", OutCfg.Epsilon);
		OutCfg.WeightDecay = OaYaml::Get<float>(o, "weight_decay", OutCfg.WeightDecay);
		OutCfg.MaxGradNorm = OaYaml::Get<float>(o, "max_grad_norm", OutCfg.MaxGradNorm);
		OutCfg.Scheduler = OaYaml::Get<OaString>(o, "scheduler", OutCfg.Scheduler);
		OutCfg.WarmupSteps = OaYaml::Get<int64_t>(o, "warmup_steps", OutCfg.WarmupSteps);
		OutCfg.CycleSteps = OaYaml::Get<int64_t>(o, "cycle_steps", OutCfg.CycleSteps);
		OutCfg.CycleMultiplier = OaYaml::Get<float>(o, "cycle_multiplier", OutCfg.CycleMultiplier);
	}
}

inline void OaLoadTrainBaseYaml(const OaYaml::Node& InYaml, OaTrainBaseConfig& OutCfg) {
	if (auto t = InYaml["training"]) {
		OutCfg.BatchSize = OaYaml::Get<int>(t, "batch_size", OutCfg.BatchSize);
		OutCfg.TotalSteps = OaYaml::Get<int64_t>(t, "total_steps", OutCfg.TotalSteps);
		OutCfg.Seed = OaYaml::Get<int>(t, "seed", OutCfg.Seed);
		OutCfg.EvalInterval = OaYaml::Get<int>(t, "eval_interval", OutCfg.EvalInterval);
		OutCfg.CheckpointInterval = OaYaml::Get<int>(t, "checkpoint_interval", OutCfg.CheckpointInterval);
		OutCfg.GradAccumSteps = OaYaml::Get<int>(t, "grad_accum_steps", OutCfg.GradAccumSteps);

		auto precStr = OaYaml::Get<OaString>(t, "precision", OaString("bf16"));
		if (precStr == "fp32") OutCfg.Precision = OaPrecision::FP32;
		else if (precStr == "tf32") OutCfg.Precision = OaPrecision::TF32;
		else if (precStr == "fp16") OutCfg.Precision = OaPrecision::FP16;
		else OutCfg.Precision = OaPrecision::BF16;

		OutCfg.Device = OaYaml::Get<OaString>(t, "device", OutCfg.Device);
	}
}

// VRAM BUDGET — auto-tune batch_size * seq_len from free VRAM (implementation in budget.cpp)

class OaVRAMBudgetConfig {
public:
	OaF32 SafetyMarginPercent = 0.10f;
	OaI32 MinBatchSize = 1;
	OaI32 MaxBatchSize = 4096;
	OaI32 MinSeqLen = 64;
	OaI32 MaxSeqLen = 8192;
	OaI32 PreferredBatchSize = 0;
	OaI32 PreferredSeqLen = 0;
	OaI32 OptimizerStatesPerParam = 4;
	OaI32 BytesPerParam = 4;
};

class OaVRAMBudgetResult {
public:
	OaI32 BatchSize = 0;
	OaI32 SeqLen = 0;
	OaUsize ModelBytes = 0;
	OaUsize ActivationBytes = 0;
	OaUsize TotalBytes = 0;
	OaUsize AvailableBytes = 0;
	OaF32 UtilizationPercent = 0.0f;
	OaBool FitsInVRAM = false;
};

[[nodiscard]] OaVRAMBudgetResult OaComputeVRAMBudget(
	OaUsize InModelParams,
	OaUsize InActivationBytesPerToken,
	OaVRAMBudgetConfig InConfig = {},
	OaDevice InDevice = OaDevice{OaDeviceType::VkDiscrete, 0}
);

void OaPrintVRAMBudget(const OaVRAMBudgetResult& InResult);
