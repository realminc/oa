// ML Training & Chat Configuration (Unified, Architecture-Agnostic)
// 4-way precedence: struct defaults < Base.yaml < Model YAML < CLI args
// Built on OaCli<T> from oa/core/cli.h
//
// Base.yaml (var/config/Base.yaml) provides shared defaults (dataset, training params, engine).
// Model configs override Base.yaml for model-specific values.
// CLI flags have final precedence.
//
// YAML `architecture:` field selects the model class (simple_llm, rem1, gptoss, granite4).

#pragma once

#include <limits>

#include <fmt/format.h>

#include <Oa/Core/Cli.h>
#include <Oa/Core/Time.h>
#include <Oa/Ml/Config.h>

// One segment of a multi-phase YAML `training_phases:` schedule. Unset optional fields inherit the base config.
// Epochs is the preferred length unit (steps/epoch derive from corpus size); Steps is the legacy fallback.
struct OaConfigTrainPhase {
	OaString Id;
	OaI32 Epochs = 0;
	OaI32 Steps = 0;
	OaString DataPath;
	OaString ValidationDataPath;
	OaOpt<OaF32> Lr;
	OaOpt<OaF32> MinLr;
	OaOpt<OaI32> WarmupSteps;
	OaOpt<OaI32> EvalInterval;
	OaOpt<OaI32> ValBatches;
	OaOpt<OaI32> EarlyStopPatience;
	OaOpt<OaF32> EarlyStopMinDelta;
};

// Training config — arch-agnostic superset
struct OaConfigTrain {
	OaString ConfigPath;
	OaString Arch = "llm";
	OaString DataPath;
	OaString ModelName = "OaLlm";
	OaString Context;
	OaString ResumePath;
	OaString PrecisionStr = "fp32";
	OaBool Validate = false;

	// Vulkan / engine topology (see oa OaEngineConfig)
	// topology: single | multi | mesh | all | server (server = multi + future cluster hooks)
	// device: discrete | integrated | cpu | by_index — preference for auto-pick / mesh scoring
	// vulkan_index >= 0: single topology — force VkPhysicalDevice by loader enumeration index (overrides device)
	// multi + mesh_indices: explicit node order (primary = first); vulkan_index alone does not set mesh order
	OaString EngineTopology = "single";
	OaString EngineDevice = "discrete";
	OaI32 VulkanIndex = -1;
	OaI32 MaxMeshDevices = 8;
	OaString MeshIndices;
	// When true with multi-device engine: each step runs a tiny `scale` dispatch on the mesh auxiliary
	// VkDevice (presentation / profiling — LM math still uses primary only). YAML: engine.mesh_aux_demo
	OaBool MeshAuxDemo = false;

	// Training
	OaI32 BatchSize = 4;
	OaI32 SeqLen = 128;
	OaI32 Steps = 1000;
	OaString TimeBudget;
	OaF64 TimeBudgetSec = 0.0;
	OaI32 SampleInterval = 0;
	OaI32 SampleTokens = 128;
	OaF32 SampleTemperature = 0.8f;

	// Validation / early stopping (optional). Val bytes: .oad ValSpan(), or training.validation_data file.
	OaString ValidationDataPath;
	OaI32 EvalInterval = 0;  // Step-only runs; epoch training validates each epoch
	OaI32 ValBatches = 8;
	OaI32 EarlyStopPatience = 0;
	OaF32 EarlyStopMinDelta = 0.0f;

	// Loss function (YAML training.loss). Names the loss metric in logs too.
	OaString LossName = "cross_entropy";

	// Optimizer
	OaString OptimizerType = "adamw";  // adamw | adam | sgd | muon (muon: official split — Muon on 2D body, AdamW on embed/head/1D)
	OaF32 Lr = 3e-4f;
	OaF32 MinLr = 3e-5f;
	OaI32 WarmupSteps = 50;
	OaF32 WeightDecay = 0.01f;
	OaF32 Beta1 = 0.9f;       // Adam/AdamW
	OaF32 Beta2 = 0.95f;      // Adam/AdamW (0.95 preferred for LLMs)
	OaF32 Eps = 1e-8f;        // Adam/AdamW epsilon
	OaF32 Momentum = 0.0f;    // SGD momentum

	// Muon (official split: Muon on 2D body, AdamW on embed/head/1D)
	OaF32 MuonLr = 2e-2f;
	OaF32 MuonBeta = 0.95f;
	OaF32 MuonWeightDecay = 0.1f;
	OaF32 MuonEps = 1e-7f;
	OaI32 MuonNs5Iters = 5;

	// Epoch-based training: epochs × steps_per_epoch overrides raw `steps` when both are set.
	// steps_per_epoch = 0 → auto-compute from dataset size / batch_size.
	OaI32 Epochs = 0;
	OaI32 StepsPerEpoch = 0;

	// When non-empty, `Steps` is the sum of phase steps (YAML); CLI `--steps` can cap total after parse.
	// Per-phase LR schedule uses phase-local step counts; TrainStep still receives the global step index.
	OaVec<OaConfigTrainPhase> TrainingPhases;

	// Callback config (YAML training.callbacks). Each entry accepts either a
	// bool toggle or a map with per-callback params:
	//   callbacks:
	//     progress_bar: true
	//     summary: true
	//     metrics: [loss]                     # or bool — true = default set
	//     early_stop: { patience: 5, min_delta: 1e-4 }   # or bool
	//     checkpoint: { restore_best: true, max_keep: 5 } # or bool
	//     phase: true
	// early_stop patience/min_delta share storage with the top-level
	// EarlyStopPatience/EarlyStopMinDelta fields (legacy keys still work).
	struct {
		OaBool ProgressBar = true;
		OaBool Summary = true;
		OaBool Metrics = true;
		OaVec<OaString> MetricNames;          // empty = default {loss}
		OaBool EarlyStop = true;
		OaBool Phase = true;
		OaBool Checkpoint = true;
		OaBool CheckpointRestoreBest = true;  // reload best weights at train end
		OaI32  CheckpointMaxKeep = 5;         // incremental checkpoint rotation
		OaI64  CheckpointSaveEvery = 0;       // extra mid-epoch saves every N steps; 0 = epoch-end only
	} Callbacks;

	[[nodiscard]] bool UsesTrainingPhases() const { return !TrainingPhases.Empty(); }
	[[nodiscard]] bool HasTimeBudget() const { return TimeBudgetSec > 0.0; }
	[[nodiscard]] bool UsesEpochs() const { return Epochs > 0; }

	void ResolveTimeBudget() {
		if (!TimeBudget.empty()) {
			TimeBudgetSec = OaParseDuration(TimeBudget);
			if (TimeBudgetSec > 0.0 && Steps == 1000) {
				Steps = std::numeric_limits<OaI32>::max();
			}
		}
	}

	[[nodiscard]] OaPrecision Precision() const {
		if (PrecisionStr == "fp32") return OaPrecision::FP32;
		if (PrecisionStr == "tf32") return OaPrecision::TF32;
		if (PrecisionStr == "bf16") return OaPrecision::BF16;
		if (PrecisionStr == "fp16") return OaPrecision::FP16;
		return OaPrecision::FP32;
	}

};

// Effective config for the current phase: base (YAML + CLI) overlaid with the phase. `Steps` is the global total.
[[nodiscard]] inline OaConfigTrain OaMergeTrainingPhase(
	const OaConfigTrain& InBase,
	const OaConfigTrainPhase& InPhase,
	OaI32 InTotalSteps) {
	OaConfigTrain c = InBase;
	c.Steps = InTotalSteps;
	if (!InPhase.DataPath.empty()) {
		c.DataPath = InPhase.DataPath;
	}
	if (!InPhase.ValidationDataPath.empty()) {
		c.ValidationDataPath = InPhase.ValidationDataPath;
	}
	if (InPhase.Lr) {
		c.Lr = *InPhase.Lr;
	}
	if (InPhase.MinLr) {
		c.MinLr = *InPhase.MinLr;
	}
	if (InPhase.WarmupSteps) {
		c.WarmupSteps = *InPhase.WarmupSteps;
	}
	if (InPhase.EvalInterval) {
		c.EvalInterval = *InPhase.EvalInterval;
	}
	if (InPhase.ValBatches) {
		c.ValBatches = *InPhase.ValBatches;
	}
	if (InPhase.EarlyStopPatience) {
		c.EarlyStopPatience = *InPhase.EarlyStopPatience;
	}
	if (InPhase.EarlyStopMinDelta) {
		c.EarlyStopMinDelta = *InPhase.EarlyStopMinDelta;
	}
	return c;
}

// OaTrainCli — unified training CLI for all architectures
class OaTrainCli : public OaCli<OaConfigTrain> {
public:
	OaTrainCli() : OaCli("train", "OA Model Trainer (Vulkan Compute)") {
		AddOption("--data,-d", Cfg_.DataPath, "Training data file");
		AddOption("--name", Cfg_.ModelName, "Model name for checkpoints");
		AddOption("--context", Cfg_.Context, "Checkpoint context");
		AddOption("--resume", Cfg_.ResumePath, "Resume from .oam checkpoint");
		AddOption("--precision", Cfg_.PrecisionStr, "fp32 | bf16 | tf32 | fp16");
		AddFlag("--validate", Cfg_.Validate, "Enable Vulkan validation layers");

		AddOption("--engine-topology", Cfg_.EngineTopology,
			"single | multi | mesh | all | server — multi/mesh/all enable OaDeviceMesh");
		AddOption("--engine-device", Cfg_.EngineDevice,
			"discrete | integrated | cpu | by_index — device preference (by_index needs --vulkan-index)");
		AddOption("--vulkan-index", Cfg_.VulkanIndex,
			"If >= 0 with single topology: force that VkPhysicalDevice index (overrides --engine-device). "
			"With by_index only: same. Multi: use --mesh-indices (vulkan_index alone is ignored without mesh_indices).");
		AddOption("--max-mesh-devices", Cfg_.MaxMeshDevices,
			"Cap mesh node count when topology is multi (default 8)");
		AddOption("--mesh-indices", Cfg_.MeshIndices,
			"Optional comma list e.g. 0,1 — explicit mesh order (primary = first); caps by max-mesh-devices");
		AddFlag("--mesh-aux-demo", Cfg_.MeshAuxDemo,
			"Multi-device only: run a small auxiliary-GPU `scale` kernel each step (demo / Nsight — not LM compute)");

		AddOption("--batch-size", Cfg_.BatchSize, "Batch size");
		AddOption("--seq-len", Cfg_.SeqLen, "Sequence length");
		AddOption("--steps", Cfg_.Steps, "Training steps (overrides epochs×steps_per_epoch)");
		AddOption("--epochs", Cfg_.Epochs, "Number of epochs (auto-computes steps if steps_per_epoch not set)");
		AddOption("--steps-per-epoch", Cfg_.StepsPerEpoch, "Steps per epoch (0 = auto from dataset size)");
		AddOption("--time", Cfg_.TimeBudget, "Time budget (e.g. 20m, 2h, 1d). Trains until time or steps, whichever first.");
		AddOption("--lr", Cfg_.Lr, "Learning rate");
		AddOption("--min-lr", Cfg_.MinLr, "Min learning rate (cosine)");
		AddOption("--warmup", Cfg_.WarmupSteps, "Warmup steps");
		AddOption("--weight-decay", Cfg_.WeightDecay, "Weight decay");
		AddOption("--sample-interval", Cfg_.SampleInterval, "Generate sample text every N steps (0 = off)");
		AddOption("--sample-tokens", Cfg_.SampleTokens, "Tokens to generate per sample");
		AddOption("--sample-temperature", Cfg_.SampleTemperature, "Temperature for sample generation");
		AddOption("--validation-data", Cfg_.ValidationDataPath,
			"Optional UTF-8 corpus for val loss (use with --eval-interval / early stopping)");
		AddOption("--eval-interval", Cfg_.EvalInterval,
			"Run validation every N steps (0 = off). Requires .oad val split or --validation-data");
		AddOption("--val-batches", Cfg_.ValBatches, "Batches to average for each validation pass");
		AddOption("--early-stop-patience", Cfg_.EarlyStopPatience,
			"Stop if val loss does not improve for this many evals (0 = off; requires validation)");
		AddOption("--early-stop-min-delta", Cfg_.EarlyStopMinDelta,
			"Minimum val loss decrease to count as improvement");

		AddOption("--loss", Cfg_.LossName, "Loss function (cross_entropy); also names the loss metric in logs");

		// Optimizer settings
		AddOption("--optimizer", Cfg_.OptimizerType,
			"Optimizer type: adamw | adam | sgd | muon (muon = official Muon+AdamW split)");
		AddOption("--beta1", Cfg_.Beta1, "Adam/AdamW beta1 (default 0.9)");
		AddOption("--beta2", Cfg_.Beta2, "Adam/AdamW beta2 (default 0.95 for LLMs)");
		AddOption("--eps", Cfg_.Eps, "Adam/AdamW epsilon (default 1e-8)");
		AddOption("--momentum", Cfg_.Momentum, "SGD momentum (default 0.0)");
		AddOption("--muon-lr", Cfg_.MuonLr, "Muon learning rate (default 0.02; AdamW uses --lr)");
		AddOption("--muon-beta", Cfg_.MuonBeta, "Muon momentum beta (default 0.95)");
		AddOption("--muon-weight-decay", Cfg_.MuonWeightDecay, "Muon weight decay (default 0.1)");
		AddOption("--muon-eps", Cfg_.MuonEps, "Muon NS5 normalization epsilon (default 1e-7)");
		AddOption("--muon-ns5-iters", Cfg_.MuonNs5Iters, "Muon Newton-Schulz5 iterations (default 5)");

		SetEpilog(
			"Architecture is selected by the `architecture:` field in the YAML config.\n"
			"Model-specific parameters are in the YAML `model:` section.\n"
			"CLI flags override YAML values (4-way precedence: defaults < Base.yaml < Model YAML < CLI).\n"
			"\n"
			"Base config (var/config/Base.yaml) provides shared defaults (dataset path, training params, engine settings).\n"
			"Model configs override Base.yaml for model-specific values.\n"
			"\n"
			"Examples:\n"
			"  ./trainalm --config var/config/Alm.yaml\n"
			"  ./modelctl inspect model.safetensors\n"
			"\n"
			"Multi-phase schedule: YAML key `training.phases:` (sequence inside `training:`; legacy `training_phases:` also accepted).\n"
			"Each item: id, epochs, and optional overrides (data, lr, min_lr, warmup_steps,\n"
			"eval_interval, val_batches, early_stop_patience, early_stop_min_delta, validation_data).\n"
			"Unset phase fields inherit the parent `training:` values (optimizer fields included — no separate `optimizer:` needed).\n"
			"Total steps = sum of phase steps (overrides training.steps). CLI --steps caps the global total.\n"
			"Weights stay in GPU memory between phases (no reload); phase checkpoint written at each phase end\n"
			"under checkpoint_{context}/ as {model}_{context}_phase{NN}_{id}_end_step{S}.oam.\n"
			"Resume: .oam Progress section stores phase index + global step for fast-forward.\n"
		);
	}

	// Override Parse to support Base.yaml + Model config merge
	bool Parse(int InArgc, char** InArgv) {
		// Extract --config path first (same as base class)
		for (int i = 1; i < InArgc; ++i) {
			OaString arg(InArgv[i]);
			if ((arg == "--config" || arg == "-c") && i + 1 < InArgc) {
				ConfigPath_ = InArgv[i + 1];
				break;
			}
			if (arg.substr(0, 9) == "--config=") {
				ConfigPath_ = arg.substr(9);
				break;
			}
			if (arg.substr(0, 3) == "-c=") {
				ConfigPath_ = arg.substr(3);
				break;
			}
		}

		// Step 3a: Load Base.yaml first (shared defaults)
		OaYaml::Node baseYaml;
		bool hasBase = false;
		try {
			OaString basePath = "var/config/Base.yaml";
			baseYaml = OaYaml::LoadFile(basePath);
			LoadYaml(baseYaml);
			hasBase = true;
		} catch (const OaYaml::Exception&) {
			// Base.yaml optional - continue with hardcoded defaults
		}

		// Step 3b: Load model-specific config (overrides Base.yaml)
		if (!ConfigPath_.empty()) {
			try {
				OaYaml::Node modelYaml = OaYaml::LoadFile(ConfigPath_);
				LoadYaml(modelYaml);
			} catch (const OaYaml::Exception& e) {
				fmt::print(stderr, "[OA CONFIG] YAML load failed: {} (using defaults)\n", e.what());
				if (!hasBase) {
					return false;  // No base and no model config
				}
			}
		}

		// Step 4: CLI11 parse -> only modifies Cfg_ fields where user explicitly provided CLI args
		try {
			App_.parse(InArgc, InArgv);
			ApplyCliOverrides();
			return true;
		} catch (const CLI::ParseError& e) {
			App_.exit(e);
			return false;
		}
	}

protected:
	// One training.callbacks entry: scalar bool toggle, or a map with params
	// (which implies enabled unless `enabled: false` is given).
	static bool ParseCallbackToggle(const OaYaml::Node& InNode, bool InCurrent) {
		if (!InNode) return InCurrent;
		if (InNode.IsScalar()) {
			try { return InNode.as<bool>(); } catch (...) { return InCurrent; }
		}
		if (InNode.IsMap()) return OaYaml::Get<bool>(InNode, "enabled", true);
		return InCurrent;
	}

	void LoadCallbacksYaml(const OaYaml::Node& InCb) {
		if (!InCb) return;
		auto& cbs = Cfg_.Callbacks;
		cbs.ProgressBar = ParseCallbackToggle(InCb["progress_bar"], cbs.ProgressBar);
		cbs.Summary = ParseCallbackToggle(InCb["summary"], cbs.Summary);
		cbs.Phase = ParseCallbackToggle(InCb["phase"], cbs.Phase);

		// metrics: bool (true = default set) or a list of metric names.
		if (auto m = InCb["metrics"]) {
			if (m.IsSequence()) {
				cbs.Metrics = true;
				cbs.MetricNames.Clear();
				for (const auto& item : m) {
					try { cbs.MetricNames.PushBack(OaString(item.as<std::string>())); } catch (...) {}
				}
			} else {
				cbs.Metrics = ParseCallbackToggle(m, cbs.Metrics);
			}
		}

		// early_stop: bool or { patience, min_delta }.
		if (auto es = InCb["early_stop"]) {
			cbs.EarlyStop = ParseCallbackToggle(es, cbs.EarlyStop);
			if (es.IsMap()) {
				Cfg_.EarlyStopPatience = OaYaml::Get<int>(es, "patience", Cfg_.EarlyStopPatience);
				Cfg_.EarlyStopMinDelta = OaYaml::Get<float>(es, "min_delta", Cfg_.EarlyStopMinDelta);
			}
		}

		// checkpoint: bool or { restore_best, max_keep, save_every }.
		if (auto ck = InCb["checkpoint"]) {
			cbs.Checkpoint = ParseCallbackToggle(ck, cbs.Checkpoint);
			if (ck.IsMap()) {
				cbs.CheckpointRestoreBest = OaYaml::Get<bool>(ck, "restore_best", cbs.CheckpointRestoreBest);
				cbs.CheckpointMaxKeep = OaYaml::Get<int>(ck, "max_keep", cbs.CheckpointMaxKeep);
				cbs.CheckpointSaveEvery = OaYaml::Get<int64_t>(ck, "save_every", cbs.CheckpointSaveEvery);
			}
		}
	}

	void LoadYaml(const OaYaml::Node& InYaml) override {
		Cfg_.Arch = OaYaml::Get<OaString>(InYaml, "architecture", Cfg_.Arch);
		Cfg_.ModelName = OaYaml::Get<OaString>(InYaml, "name", Cfg_.ModelName);
		Cfg_.Context = OaYaml::Get<OaString>(InYaml, "context", Cfg_.Context);
		Cfg_.ResumePath = OaYaml::Get<OaString>(InYaml, "resume", Cfg_.ResumePath);
		// Legacy: top-level `optimizer:` section (lowest priority; `training:` overrides when both present).
		if (auto o = InYaml["optimizer"]) {
			Cfg_.OptimizerType = OaYaml::Get<OaString>(o, "type", Cfg_.OptimizerType);
			Cfg_.Lr = OaYaml::Get<float>(o, "lr", Cfg_.Lr);
			Cfg_.MinLr = OaYaml::Get<float>(o, "min_lr", Cfg_.MinLr);
			Cfg_.WarmupSteps = OaYaml::Get<int>(o, "warmup_steps", Cfg_.WarmupSteps);
			Cfg_.WeightDecay = OaYaml::Get<float>(o, "weight_decay", Cfg_.WeightDecay);
			Cfg_.Beta1 = OaYaml::Get<float>(o, "beta1", Cfg_.Beta1);
			Cfg_.Beta2 = OaYaml::Get<float>(o, "beta2", Cfg_.Beta2);
			Cfg_.Eps = OaYaml::Get<float>(o, "eps", Cfg_.Eps);
			Cfg_.Momentum = OaYaml::Get<float>(o, "momentum", Cfg_.Momentum);
			Cfg_.MuonLr = OaYaml::Get<float>(o, "muon_lr", Cfg_.MuonLr);
			Cfg_.MuonBeta = OaYaml::Get<float>(o, "muon_beta", Cfg_.MuonBeta);
			Cfg_.MuonWeightDecay = OaYaml::Get<float>(o, "muon_weight_decay", Cfg_.MuonWeightDecay);
			Cfg_.MuonEps = OaYaml::Get<float>(o, "muon_eps", Cfg_.MuonEps);
			Cfg_.MuonNs5Iters = OaYaml::Get<int>(o, "muon_ns5_iters", Cfg_.MuonNs5Iters);
		}
		if (auto t = InYaml["training"]) {
			Cfg_.BatchSize = OaYaml::Get<int>(t, "batch_size", Cfg_.BatchSize);
			Cfg_.SeqLen = OaYaml::Get<int>(t, "seq_len", Cfg_.SeqLen);
			Cfg_.Steps = OaYaml::Get<int>(t, "steps", Cfg_.Steps);
			Cfg_.TimeBudget = OaYaml::Get<OaString>(t, "time", Cfg_.TimeBudget);
			Cfg_.SampleInterval = OaYaml::Get<int>(t, "sample_interval", Cfg_.SampleInterval);
			Cfg_.SampleTokens = OaYaml::Get<int>(t, "sample_tokens", Cfg_.SampleTokens);
			Cfg_.SampleTemperature = OaYaml::Get<float>(t, "sample_temperature", Cfg_.SampleTemperature);
			Cfg_.DataPath = OaYaml::Get<OaString>(t, "data", Cfg_.DataPath);
			Cfg_.PrecisionStr = OaYaml::Get<OaString>(t, "precision", Cfg_.PrecisionStr);
			Cfg_.Validate = OaYaml::Get<bool>(t, "validate", Cfg_.Validate);
			Cfg_.Epochs = OaYaml::Get<int>(t, "epochs", Cfg_.Epochs);
			Cfg_.StepsPerEpoch = OaYaml::Get<int>(t, "steps_per_epoch", Cfg_.StepsPerEpoch);
			Cfg_.ValidationDataPath = OaYaml::Get<OaString>(t, "validation_data", Cfg_.ValidationDataPath);
			Cfg_.EvalInterval = OaYaml::Get<int>(t, "eval_interval", Cfg_.EvalInterval);
			Cfg_.ValBatches = OaYaml::Get<int>(t, "val_batches", Cfg_.ValBatches);
			Cfg_.EarlyStopPatience = OaYaml::Get<int>(t, "early_stop_patience", Cfg_.EarlyStopPatience);
			Cfg_.EarlyStopMinDelta = OaYaml::Get<float>(t, "early_stop_min_delta", Cfg_.EarlyStopMinDelta);
			Cfg_.LossName = OaYaml::Get<OaString>(t, "loss", Cfg_.LossName);
			// Optimizer fields inside training: (new layout; overrides top-level `optimizer:` when both present).
			Cfg_.OptimizerType = OaYaml::Get<OaString>(t, "optimizer", Cfg_.OptimizerType);
			Cfg_.Lr = OaYaml::Get<float>(t, "lr", Cfg_.Lr);
			Cfg_.MinLr = OaYaml::Get<float>(t, "min_lr", Cfg_.MinLr);
			Cfg_.WarmupSteps = OaYaml::Get<int>(t, "warmup_steps", Cfg_.WarmupSteps);
			Cfg_.WeightDecay = OaYaml::Get<float>(t, "weight_decay", Cfg_.WeightDecay);
			Cfg_.Beta1 = OaYaml::Get<float>(t, "beta1", Cfg_.Beta1);
			Cfg_.Beta2 = OaYaml::Get<float>(t, "beta2", Cfg_.Beta2);
			Cfg_.Eps = OaYaml::Get<float>(t, "eps", Cfg_.Eps);
			Cfg_.Momentum = OaYaml::Get<float>(t, "momentum", Cfg_.Momentum);
			Cfg_.MuonLr = OaYaml::Get<float>(t, "muon_lr", Cfg_.MuonLr);
			Cfg_.MuonBeta = OaYaml::Get<float>(t, "muon_beta", Cfg_.MuonBeta);
			Cfg_.MuonWeightDecay = OaYaml::Get<float>(t, "muon_weight_decay", Cfg_.MuonWeightDecay);
			Cfg_.MuonEps = OaYaml::Get<float>(t, "muon_eps", Cfg_.MuonEps);
			Cfg_.MuonNs5Iters = OaYaml::Get<int>(t, "muon_ns5_iters", Cfg_.MuonNs5Iters);
		}
		if (auto e = InYaml["engine"]) {
			Cfg_.EngineTopology = OaYaml::Get<OaString>(e, "topology", Cfg_.EngineTopology);
			Cfg_.EngineDevice = OaYaml::Get<OaString>(e, "device", Cfg_.EngineDevice);
			Cfg_.VulkanIndex = OaYaml::Get<int>(e, "vulkan_index", Cfg_.VulkanIndex);
			Cfg_.MaxMeshDevices = OaYaml::Get<int>(e, "max_devices", Cfg_.MaxMeshDevices);
			Cfg_.MeshIndices = OaYaml::Get<OaString>(e, "mesh_indices", Cfg_.MeshIndices);
			Cfg_.MeshAuxDemo = OaYaml::Get<bool>(e, "mesh_aux_demo", Cfg_.MeshAuxDemo);
			Cfg_.PrecisionStr = OaYaml::Get<OaString>(e, "precision", Cfg_.PrecisionStr);
		}
		// Callback config: training.callbacks (top-level callbacks: also accepted).
		// Entries are bool toggles or maps with per-callback params — see Callbacks docs.
		LoadCallbacksYaml(InYaml["callbacks"]);
		if (auto t = InYaml["training"]) {
			LoadCallbacksYaml(t["callbacks"]);
		}

		// Phases: prefer training.phases: (new layout), fall back to top-level training_phases: (legacy).
		auto phaseSeq = [&]() -> OaYaml::Node {
			if (auto t = InYaml["training"]) {
				if (auto p = t["phases"]; p && p.IsSequence()) return p;
			}
			if (auto p = InYaml["training_phases"]; p && p.IsSequence()) return p;
			return {};
		}();
		if (auto seq = phaseSeq; seq && seq.IsSequence()) {
			Cfg_.TrainingPhases.Clear();
			for (size_t i = 0; i < seq.size(); ++i) {
				const auto& n = seq[i];
				if (!n || !n.IsMap()) {
					continue;
				}
				OaConfigTrainPhase p;
				p.Id = OaYaml::Get<OaString>(n, "id", OaString("phase_") + OaToString(static_cast<OaU32>(i)));
				p.Epochs = OaYaml::Get<int>(n, "epochs", 0);
				p.Steps = OaYaml::Get<int>(n, "steps", 0);
				p.DataPath = OaYaml::Get<OaString>(n, "data", OaString{});
				p.ValidationDataPath = OaYaml::Get<OaString>(n, "validation_data", OaString{});
				if (n["lr"]) {
					p.Lr = n["lr"].as<OaF32>();
				}
				if (n["min_lr"]) {
					p.MinLr = n["min_lr"].as<OaF32>();
				}
				if (n["warmup_steps"]) {
					p.WarmupSteps = n["warmup_steps"].as<OaI32>();
				}
				if (n["eval_interval"]) {
					p.EvalInterval = n["eval_interval"].as<OaI32>();
				}
				if (n["val_batches"]) {
					p.ValBatches = n["val_batches"].as<OaI32>();
				}
				if (n["early_stop_patience"]) {
					p.EarlyStopPatience = n["early_stop_patience"].as<OaI32>();
				}
				if (n["early_stop_min_delta"]) {
					p.EarlyStopMinDelta = n["early_stop_min_delta"].as<OaF32>();
				}
				Cfg_.TrainingPhases.PushBack(std::move(p));
			}
			if (!Cfg_.TrainingPhases.Empty()) {
				// Steps total only when every phase gives raw steps; epoch-based
				// phases resolve to steps later (corpus-derived steps/epoch).
				OaI32 sum = 0;
				bool allSteps = true;
				for (const auto& ph : Cfg_.TrainingPhases) {
					sum += ph.Steps;
					if (ph.Epochs > 0 || ph.Steps <= 0) allSteps = false;
				}
				if (allSteps) Cfg_.Steps = sum;
				if (Cfg_.DataPath.empty() && !Cfg_.TrainingPhases[0].DataPath.empty()) {
					Cfg_.DataPath = Cfg_.TrainingPhases[0].DataPath;
				}
				if (Cfg_.ValidationDataPath.empty() && !Cfg_.TrainingPhases[0].ValidationDataPath.empty()) {
					Cfg_.ValidationDataPath = Cfg_.TrainingPhases[0].ValidationDataPath;
				}
			}
		}
	}
};

// Chat config — model config comes from .oam
struct OaConfigChat {
	OaString ModelPath;
	OaString Prompt;
	OaString PrecisionStr = "fp32";
	OaI32 MaxTokens = 256;
	OaF32 Temperature = 0.8f;
	OaI32 SeqLen = 128;
	OaBool Validate = false;
	// Log tok/s and wall time after each generation (stderr). CLI: --gen-metrics / --no-gen-metrics.
	// YAML: verbose. (Global -v,--verbose is OA log level, separate from this.)
	OaBool Verbose = true;

	[[nodiscard]] OaPrecision Precision() const {
		if (PrecisionStr == "fp32") return OaPrecision::FP32;
		if (PrecisionStr == "tf32") return OaPrecision::TF32;
		if (PrecisionStr == "bf16") return OaPrecision::BF16;
		if (PrecisionStr == "fp16") return OaPrecision::FP16;
		return OaPrecision::FP32;
	}
};

// OaChatCli — unified chat CLI for all architectures
class OaChatCli : public OaCli<OaConfigChat> {
public:
	OaChatCli() : OaCli("chat", "OA Model Chat (Vulkan Compute)") {
		AddOption("--model,-m", Cfg_.ModelPath, "Path to .oam checkpoint");
		AddOption("--prompt,-p", Cfg_.Prompt, "Input prompt (omit for interactive)");
		AddOption("--max-tokens,-n", Cfg_.MaxTokens, "Max tokens to generate");
		AddOption("--temperature,-t", Cfg_.Temperature, "Sampling temperature");
		AddOption("--seq-len", Cfg_.SeqLen, "Context window for generation");
		AddOption("--precision", Cfg_.PrecisionStr, "fp32 | bf16 | tf32 | fp16");
		AddFlag("--validate", Cfg_.Validate, "Enable Vulkan validation layers");
		AddFlag("--gen-metrics,--verbose-gen,--no-gen-metrics{false}", Cfg_.Verbose,
			"Log generation tok/s after each reply (default: on; YAML: verbose. Global -v is log level only)");

		SetEpilog(
			"Architecture is auto-detected from the .oam checkpoint.\n"
			"\n"
			"Examples:\n"
			"  ./chat -m var/model/dev/OaLlm/OaLlm.oam\n"
			"  ./chat -m model.oam -p \"Once upon a time\" -t 0.5\n"
		);
	}

protected:
	void LoadYaml(const OaYaml::Node& InYaml) override {
		Cfg_.ModelPath = OaYaml::Get<OaString>(InYaml, "model", Cfg_.ModelPath);
		Cfg_.PrecisionStr = OaYaml::Get<OaString>(InYaml, "precision", Cfg_.PrecisionStr);
		Cfg_.MaxTokens = OaYaml::Get<int>(InYaml, "max_tokens", Cfg_.MaxTokens);
		Cfg_.Temperature = OaYaml::Get<float>(InYaml, "temperature", Cfg_.Temperature);
		Cfg_.SeqLen = OaYaml::Get<int>(InYaml, "seq_len", Cfg_.SeqLen);
		Cfg_.Verbose = OaYaml::Get<bool>(InYaml, "verbose", Cfg_.Verbose);
	}
};
