// ML training & Chat configuration (Unified, architecture-Agnostic)
// 4-way precedence: struct defaults < base.yaml < Model YAML < CLI args
// Built on oa::Cli<T> from oa/core/cli.h
//
// base.yaml (var/config/base.yaml) provides shared defaults (dataset, training params, engine).
// Model configs override base.yaml for model-specific values.
// CLI flags have final precedence.
//
// YAML `architecture:` field selects the model class (simple_llm, rem1, gptoss, granite4).

#pragma once

#include <limits>

#include <oa/core/cli.h>
#include <oa/core/std/print.h>
#include <oa/core/time.h>
#include <oa/ml/config.h>

namespace oa {

// One segment of a multi-phase YAML `training_phases:` schedule. Unset optional fields inherit the base config.
// epochs is the preferred length unit (steps/epoch derive from corpus size); steps is the legacy fallback.
struct ConfigTrainPhase {
	oa::String id;
	oa::I32 epochs = 0;
	oa::I32 steps = 0;
	oa::String dataPath;
	oa::String validationDataPath;
	oa::Optional<oa::F32> lr;
	oa::Optional<oa::F32> minLr;
	oa::Optional<oa::I32> warmupSteps;
	oa::Optional<oa::I32> evalInterval;
	oa::Optional<oa::I32> valBatches;
	oa::Optional<oa::I32> earlyStopPatience;
	oa::Optional<oa::F32> earlyStopMinDelta;
};

// training config — arch-agnostic superset
struct ConfigTrain {
	oa::String configPath;
	oa::String arch = "llm";
	oa::String dataPath;
	oa::String modelName = "OaLlm";
	oa::String context;
	oa::String resumePath;
	oa::String precisionStr = "fp32";
	oa::Bool validate = false;

	// vulkan / engine topology (see oa oa::EngineConfig)
	// topology: single | multi | mesh | all. Multi-device modes expose experimental
	// mesh discovery only; they are not a supported distributed NLP path.
	// device: discrete | integrated | cpu | by_index — preference for auto-pick / mesh scoring
	// vulkan_index >= 0: single topology — force VkPhysicalDevice by loader enumeration index (overrides device)
	// multi + mesh_indices: explicit node order (primary = first); vulkan_index alone does not set mesh order
	oa::String engineTopology = "single";
	oa::String engineDevice = "discrete";
	oa::I32 vulkanIndex = -1;
	oa::I32 maxMeshDevices = 8;
	oa::String meshIndices;
	// When true with multi-device engine: each step runs a tiny `scale` dispatch on the mesh auxiliary
	// vkDevice (presentation / profiling — LM math still uses primary only). YAML: engine.mesh_aux_demo
	oa::Bool meshAuxDemo = false;

	// training
	oa::I32 batchSize = 4;
	oa::I32 seqLen = 128;
	oa::I32 steps = 1000;
	oa::String timeBudget;
	oa::F64 timeBudgetSec = 0.0;
	oa::I32 sampleInterval = 0;
	oa::I32 sampleTokens = 128;
	oa::F32 sampleTemperature = 0.8f;

	// Validation / early stopping (optional). Val bytes: .oad valSpan(), or training.validation_data file.
	oa::String validationDataPath;
	oa::I32 evalInterval = 0;  // step-only runs; epoch training validates each epoch
	oa::I32 valBatches = 8;
	oa::I32 earlyStopPatience = 0;
	oa::F32 earlyStopMinDelta = 0.0f;

	// loss function (YAML training.loss). Names the loss metric in logs too.
	oa::String lossName = "cross_entropy";

	// Optimizer
	oa::String optimizerType = "adamw";  // adamw | adam | sgd | muon
	oa::F32 lr = 3e-4f;
	oa::F32 minLr = 3e-5f;
	oa::I32 warmupSteps = 50;
	oa::F32 weightDecay = 0.01f;
	oa::F32 beta1 = 0.9f;       // Adam/AdamW
	oa::F32 beta2 = 0.95f;      // Adam/AdamW (0.95 preferred for LLMs)
	oa::F32 eps = 1e-8f;        // Adam/AdamW epsilon
	oa::F32 momentum = 0.0f;    // SGD momentum

	// Pure Muon parameters. Callers own any explicit Muon/AdamW parameter split.
	oa::F32 muonLr = 2e-2f;
	oa::F32 muonBeta = 0.95f;
	oa::F32 muonWeightDecay = 0.1f;
	oa::F32 muonEps = 1e-7f;
	oa::I32 muonNs5Iters = 5;

	// epoch-based training: epochs × steps_per_epoch overrides raw `steps` when both are set.
	// steps_per_epoch = 0 → auto-compute from dataset size / batch_size.
	oa::I32 epochs = 0;
	oa::I32 stepsPerEpoch = 0;

	// When non-empty, `steps` is the sum of phase steps (YAML); CLI `--steps` can cap total after parse.
	// Per-phase LR schedule uses phase-local step counts; TrainStep still receives the global step index.
	oa::Vector<ConfigTrainPhase> trainingPhases;

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
	// earlyStopPatience/earlyStopMinDelta fields (legacy keys still work).
	struct {
		oa::Bool progressBar = true;
		oa::Bool summary = true;
		oa::Bool metrics = true;
		oa::Vector<oa::String> metricNames;          // empty = default {loss}
		oa::Bool earlyStop = true;
		oa::Bool phase = true;
		oa::Bool checkpoint = true;
		oa::Bool checkpointRestoreBest = true;  // reload best weights at train end
		oa::I32  checkpointMaxKeep = 5;         // incremental checkpoint rotation
		oa::I64  checkpointSaveEvery = 0;       // extra mid-epoch saves every N steps; 0 = epoch-end only
	} callbacks;

	[[nodiscard]] bool usesTrainingPhases() const { return !trainingPhases.empty(); }
	[[nodiscard]] bool hasTimeBudget() const { return timeBudgetSec > 0.0; }
	[[nodiscard]] bool usesEpochs() const { return epochs > 0; }

	void resolveTimeBudget() {
		if (!timeBudget.empty()) {
			timeBudgetSec = oa::parseDuration(timeBudget);
			if (timeBudgetSec > 0.0 && steps == 1000) {
				steps = std::numeric_limits<oa::I32>::max();
			}
		}
	}

	[[nodiscard]] oa::Precision precision() const {
		if (precisionStr == "fp32") return oa::Precision::FP32;
		if (precisionStr == "bf16") return oa::Precision::BF16;
		if (precisionStr == "fp64") return oa::Precision::FP64;
		return oa::Precision::FP32;
	}

};

// Effective config for the current phase: base (YAML + CLI) overlaid with the phase. `steps` is the global total.
[[nodiscard]] inline ConfigTrain mergeTrainingPhase(
	const ConfigTrain& inBase,
	const ConfigTrainPhase& inPhase,
	oa::I32 inTotalSteps) {
	ConfigTrain c = inBase;
	c.steps = inTotalSteps;
	if (!inPhase.dataPath.empty()) {
		c.dataPath = inPhase.dataPath;
	}
	if (!inPhase.validationDataPath.empty()) {
		c.validationDataPath = inPhase.validationDataPath;
	}
	if (inPhase.lr) {
		c.lr = *inPhase.lr;
	}
	if (inPhase.minLr) {
		c.minLr = *inPhase.minLr;
	}
	if (inPhase.warmupSteps) {
		c.warmupSteps = *inPhase.warmupSteps;
	}
	if (inPhase.evalInterval) {
		c.evalInterval = *inPhase.evalInterval;
	}
	if (inPhase.valBatches) {
		c.valBatches = *inPhase.valBatches;
	}
	if (inPhase.earlyStopPatience) {
		c.earlyStopPatience = *inPhase.earlyStopPatience;
	}
	if (inPhase.earlyStopMinDelta) {
		c.earlyStopMinDelta = *inPhase.earlyStopMinDelta;
	}
	return c;
}

// TrainCli — unified training CLI for all architectures
class TrainCli : public oa::Cli<ConfigTrain> {
public:
	TrainCli() : oa::Cli("train", "OA Model trainer (vulkan Compute)") {
		addOption("--data,-d", cfg_.dataPath, "training data file");
		addOption("--name", cfg_.modelName, "Model name for checkpoints");
		addOption("--context", cfg_.context, "checkpoint context");
		addOption("--resume", cfg_.resumePath, "Resume from .oam checkpoint");
		addOption("--precision", cfg_.precisionStr, "fp32 | bf16 | fp64");
		addFlag("--validate", cfg_.validate, "Enable vulkan validation layers");

		addOption("--engine-topology", cfg_.engineTopology,
			"single | multi | mesh | all — multi-device mesh modes are experimental");
		addOption("--engine-device", cfg_.engineDevice,
			"discrete | integrated | cpu | by_index — device preference (by_index needs --vulkan-index)");
		addOption("--vulkan-index", cfg_.vulkanIndex,
			"If >= 0 with single topology: force that VkPhysicalDevice index (overrides --engine-device). "
			"With by_index only: same. Multi: use --mesh-indices (vulkan_index alone is ignored without mesh_indices).");
		addOption("--max-mesh-devices", cfg_.maxMeshDevices,
			"Cap mesh node count when topology is multi (default 8)");
		addOption("--mesh-indices", cfg_.meshIndices,
			"Optional comma list e.g. 0,1 — explicit mesh order (primary = first); caps by max-mesh-devices");
		addFlag("--mesh-aux-demo", cfg_.meshAuxDemo,
			"Multi-device only: run a small auxiliary-GPU `scale` kernel each step (demo / Nsight — not LM compute)");

		addOption("--batch-size", cfg_.batchSize, "Batch size");
		addOption("--seq-len", cfg_.seqLen, "sequence length");
		addOption("--steps", cfg_.steps, "training steps (overrides epochs×steps_per_epoch)");
		addOption("--epochs", cfg_.epochs, "Number of epochs (auto-computes steps if steps_per_epoch not set)");
		addOption("--steps-per-epoch", cfg_.stepsPerEpoch, "steps per epoch (0 = auto from dataset size)");
		addOption("--time", cfg_.timeBudget, "time budget (e.g. 20m, 2h, 1d). Trains until time or steps, whichever first.");
		addOption("--lr", cfg_.lr, "Learning rate");
		addOption("--min-lr", cfg_.minLr, "Min learning rate (cosine)");
		addOption("--warmup", cfg_.warmupSteps, "Warmup steps");
		addOption("--weight-decay", cfg_.weightDecay, "weight decay");
		addOption("--sample-interval", cfg_.sampleInterval, "generate sample text every N steps (0 = off)");
		addOption("--sample-tokens", cfg_.sampleTokens, "tokens to generate per sample");
		addOption("--sample-temperature", cfg_.sampleTemperature, "temperature for sample generation");
		addOption("--validation-data", cfg_.validationDataPath,
			"Optional UTF-8 corpus for val loss (use with --eval-interval / early stopping)");
		addOption("--eval-interval", cfg_.evalInterval,
			"run validation every N steps (0 = off). Requires .oad val split or --validation-data");
		addOption("--val-batches", cfg_.valBatches, "batches to average for each validation pass");
		addOption("--early-stop-patience", cfg_.earlyStopPatience,
			"Stop if val loss does not improve for this many evals (0 = off; requires validation)");
		addOption("--early-stop-min-delta", cfg_.earlyStopMinDelta,
			"minimum val loss decrease to count as improvement");

		addOption("--loss", cfg_.lossName, "loss function (cross_entropy); also names the loss metric in logs");

		// Optimizer settings
		addOption("--optimizer", cfg_.optimizerType,
			"Optimizer type: adamw | adam | sgd | muon (no implicit parameter split)");
		addOption("--beta1", cfg_.beta1, "Adam/AdamW beta1 (default 0.9)");
		addOption("--beta2", cfg_.beta2, "Adam/AdamW beta2 (default 0.95 for LLMs)");
		addOption("--eps", cfg_.eps, "Adam/AdamW epsilon (default 1e-8)");
		addOption("--momentum", cfg_.momentum, "SGD momentum (default 0.0)");
		addOption("--muon-lr", cfg_.muonLr, "Muon learning rate (default 0.02; AdamW uses --lr)");
		addOption("--muon-beta", cfg_.muonBeta, "Muon momentum beta (default 0.95)");
		addOption("--muon-weight-decay", cfg_.muonWeightDecay, "Muon weight decay (default 0.1)");
		addOption("--muon-eps", cfg_.muonEps, "Muon NS5 normalization epsilon (default 1e-7)");
		addOption("--muon-ns5-iters", cfg_.muonNs5Iters, "Muon Newton-Schulz5 iterations (default 5)");

		setEpilog(
			"architecture is selected by the `architecture:` field in the YAML config.\n"
			"Model-specific parameters are in the YAML `model:` section.\n"
			"CLI flags override YAML values (4-way precedence: defaults < base.yaml < Model YAML < CLI).\n"
			"\n"
			"base config (var/config/base.yaml) provides shared defaults (dataset path, training params, engine settings).\n"
			"Model configs override base.yaml for model-specific values.\n"
			"\n"
			"Examples:\n"
			"  ./trainalm --config var/config/Alm.yaml\n"
			"  ./modelctl inspect model.safetensors\n"
			"\n"
			"Multi-phase schedule: YAML key `training.phases:` (sequence inside `training:`; legacy `training_phases:` also accepted).\n"
			"Each item: id, epochs, and optional overrides (data, lr, min_lr, warmup_steps,\n"
			"eval_interval, val_batches, early_stop_patience, early_stop_min_delta, validation_data).\n"
			"Unset phase fields inherit the parent `training:` values (optimizer fields included — no separate `optimizer:` needed).\n"
			"total steps = sum of phase steps (overrides training.steps). CLI --steps caps the global total.\n"
			"Weights stay in GPU memory between phases (no reload); phase checkpoint written at each phase end\n"
			"under checkpoint_{context}/ as {model}_{context}_phase{NN}_{id}_end_step{S}.oam.\n"
			"Resume: .oam Progress section stores phase index + global step for fast-forward.\n"
		);
	}

	// Override parse to support base.yaml + Model config merge
	bool parse(int inArgc, char** inArgv) {
		scanConfigPath(inArgc, inArgv);

		// step 3a: load base.yaml first (shared defaults)
		oa::Yaml::Node baseYaml;
		bool hasBase = false;
		try {
			oa::String basePath = "var/config/base.yaml";
			baseYaml = oa::Yaml::loadFile(basePath);
			loadYaml(baseYaml);
			hasBase = true;
		} catch (const oa::Yaml::Exception&) {
			// base.yaml optional - continue with hardcoded defaults
		}

		// step 3b: load model-specific config (overrides base.yaml)
		if (!configPath_.empty()) {
			try {
				oa::Yaml::Node modelYaml = oa::Yaml::loadFile(configPath_);
				loadYaml(modelYaml);
			} catch (const oa::Yaml::Exception& e) {
				(void)oa::print(oa::PrintStream::Error,
					"[OA CONFIG] YAML load failed: {} (using defaults)", e.what());
				if (!hasBase) {
					return false;  // No base and no model config
				}
			}
		}

		// Explicit command-line values have final precedence.
		if (not parseArguments(inArgc, inArgv)) return false;
		applyCliOverrides();
		return true;
	}

protected:
	// One training.callbacks entry: scalar bool toggle, or a map with params
	// (which implies enabled unless `enabled: false` is given).
	static bool parseCallbackToggle(const oa::Yaml::Node& inNode, bool inCurrent) {
		if (!inNode) return inCurrent;
		if (inNode.isScalar()) {
			try { return inNode.as<bool>(); } catch (...) { return inCurrent; }
		}
		if (inNode.isMap()) return oa::Yaml::get<bool>(inNode, "enabled", true);
		return inCurrent;
	}

	void loadCallbacksYaml(const oa::Yaml::Node& inCb) {
		if (!inCb) return;
		auto& cbs = cfg_.callbacks;
		cbs.progressBar = parseCallbackToggle(inCb["progress_bar"], cbs.progressBar);
		cbs.summary = parseCallbackToggle(inCb["summary"], cbs.summary);
		cbs.phase = parseCallbackToggle(inCb["phase"], cbs.phase);

		// metrics: bool (true = default set) or a list of metric names.
		if (auto m = inCb["metrics"]) {
			if (m.isSequence()) {
				cbs.metrics = true;
				cbs.metricNames.clear();
				for (const auto& item : m) {
					try {
						const std::string value = item.as<std::string>();
						cbs.metricNames.pushBack(oa::String(value.data(), value.size()));
					} catch (...) {}
				}
			} else {
				cbs.metrics = parseCallbackToggle(m, cbs.metrics);
			}
		}

		// early_stop: bool or { patience, min_delta }.
		if (auto es = inCb["early_stop"]) {
			cbs.earlyStop = parseCallbackToggle(es, cbs.earlyStop);
			if (es.isMap()) {
				cfg_.earlyStopPatience = oa::Yaml::get<int>(es, "patience", cfg_.earlyStopPatience);
				cfg_.earlyStopMinDelta = oa::Yaml::get<float>(es, "min_delta", cfg_.earlyStopMinDelta);
			}
		}

		// checkpoint: bool or { restore_best, max_keep, save_every }.
		if (auto ck = inCb["checkpoint"]) {
			cbs.checkpoint = parseCallbackToggle(ck, cbs.checkpoint);
			if (ck.isMap()) {
				cbs.checkpointRestoreBest = oa::Yaml::get<bool>(ck, "restore_best", cbs.checkpointRestoreBest);
				cbs.checkpointMaxKeep = oa::Yaml::get<int>(ck, "max_keep", cbs.checkpointMaxKeep);
				cbs.checkpointSaveEvery = oa::Yaml::get<int64_t>(ck, "save_every", cbs.checkpointSaveEvery);
			}
		}
	}

	void loadYaml(const oa::Yaml::Node& inYaml) override {
		cfg_.arch = oa::Yaml::get<oa::String>(inYaml, "architecture", cfg_.arch);
		cfg_.modelName = oa::Yaml::get<oa::String>(inYaml, "name", cfg_.modelName);
		cfg_.context = oa::Yaml::get<oa::String>(inYaml, "context", cfg_.context);
		cfg_.resumePath = oa::Yaml::get<oa::String>(inYaml, "resume", cfg_.resumePath);
		// Legacy: top-level `optimizer:` section (lowest priority; `training:` overrides when both present).
		if (auto o = inYaml["optimizer"]) {
			cfg_.optimizerType = oa::Yaml::get<oa::String>(o, "type", cfg_.optimizerType);
			cfg_.lr = oa::Yaml::get<float>(o, "lr", cfg_.lr);
			cfg_.minLr = oa::Yaml::get<float>(o, "min_lr", cfg_.minLr);
			cfg_.warmupSteps = oa::Yaml::get<int>(o, "warmup_steps", cfg_.warmupSteps);
			cfg_.weightDecay = oa::Yaml::get<float>(o, "weight_decay", cfg_.weightDecay);
			cfg_.beta1 = oa::Yaml::get<float>(o, "beta1", cfg_.beta1);
			cfg_.beta2 = oa::Yaml::get<float>(o, "beta2", cfg_.beta2);
			cfg_.eps = oa::Yaml::get<float>(o, "eps", cfg_.eps);
			cfg_.momentum = oa::Yaml::get<float>(o, "momentum", cfg_.momentum);
			cfg_.muonLr = oa::Yaml::get<float>(o, "muon_lr", cfg_.muonLr);
			cfg_.muonBeta = oa::Yaml::get<float>(o, "muon_beta", cfg_.muonBeta);
			cfg_.muonWeightDecay = oa::Yaml::get<float>(o, "muon_weight_decay", cfg_.muonWeightDecay);
			cfg_.muonEps = oa::Yaml::get<float>(o, "muon_eps", cfg_.muonEps);
			cfg_.muonNs5Iters = oa::Yaml::get<int>(o, "muon_ns5_iters", cfg_.muonNs5Iters);
		}
		if (auto t = inYaml["training"]) {
			cfg_.batchSize = oa::Yaml::get<int>(t, "batch_size", cfg_.batchSize);
			cfg_.seqLen = oa::Yaml::get<int>(t, "seq_len", cfg_.seqLen);
			cfg_.steps = oa::Yaml::get<int>(t, "steps", cfg_.steps);
			cfg_.timeBudget = oa::Yaml::get<oa::String>(t, "time", cfg_.timeBudget);
			cfg_.sampleInterval = oa::Yaml::get<int>(t, "sample_interval", cfg_.sampleInterval);
			cfg_.sampleTokens = oa::Yaml::get<int>(t, "sample_tokens", cfg_.sampleTokens);
			cfg_.sampleTemperature = oa::Yaml::get<float>(t, "sample_temperature", cfg_.sampleTemperature);
			cfg_.dataPath = oa::Yaml::get<oa::String>(t, "data", cfg_.dataPath);
			cfg_.precisionStr = oa::Yaml::get<oa::String>(t, "precision", cfg_.precisionStr);
			cfg_.validate = oa::Yaml::get<bool>(t, "validate", cfg_.validate);
			cfg_.epochs = oa::Yaml::get<int>(t, "epochs", cfg_.epochs);
			cfg_.stepsPerEpoch = oa::Yaml::get<int>(t, "steps_per_epoch", cfg_.stepsPerEpoch);
			cfg_.validationDataPath = oa::Yaml::get<oa::String>(t, "validation_data", cfg_.validationDataPath);
			cfg_.evalInterval = oa::Yaml::get<int>(t, "eval_interval", cfg_.evalInterval);
			cfg_.valBatches = oa::Yaml::get<int>(t, "val_batches", cfg_.valBatches);
			cfg_.earlyStopPatience = oa::Yaml::get<int>(t, "early_stop_patience", cfg_.earlyStopPatience);
			cfg_.earlyStopMinDelta = oa::Yaml::get<float>(t, "early_stop_min_delta", cfg_.earlyStopMinDelta);
			cfg_.lossName = oa::Yaml::get<oa::String>(t, "loss", cfg_.lossName);
			// Optimizer fields inside training: (new layout; overrides top-level `optimizer:` when both present).
			cfg_.optimizerType = oa::Yaml::get<oa::String>(t, "optimizer", cfg_.optimizerType);
			cfg_.lr = oa::Yaml::get<float>(t, "lr", cfg_.lr);
			cfg_.minLr = oa::Yaml::get<float>(t, "min_lr", cfg_.minLr);
			cfg_.warmupSteps = oa::Yaml::get<int>(t, "warmup_steps", cfg_.warmupSteps);
			cfg_.weightDecay = oa::Yaml::get<float>(t, "weight_decay", cfg_.weightDecay);
			cfg_.beta1 = oa::Yaml::get<float>(t, "beta1", cfg_.beta1);
			cfg_.beta2 = oa::Yaml::get<float>(t, "beta2", cfg_.beta2);
			cfg_.eps = oa::Yaml::get<float>(t, "eps", cfg_.eps);
			cfg_.momentum = oa::Yaml::get<float>(t, "momentum", cfg_.momentum);
			cfg_.muonLr = oa::Yaml::get<float>(t, "muon_lr", cfg_.muonLr);
			cfg_.muonBeta = oa::Yaml::get<float>(t, "muon_beta", cfg_.muonBeta);
			cfg_.muonWeightDecay = oa::Yaml::get<float>(t, "muon_weight_decay", cfg_.muonWeightDecay);
			cfg_.muonEps = oa::Yaml::get<float>(t, "muon_eps", cfg_.muonEps);
			cfg_.muonNs5Iters = oa::Yaml::get<int>(t, "muon_ns5_iters", cfg_.muonNs5Iters);
		}
		if (auto e = inYaml["engine"]) {
			cfg_.engineTopology = oa::Yaml::get<oa::String>(e, "topology", cfg_.engineTopology);
			cfg_.engineDevice = oa::Yaml::get<oa::String>(e, "device", cfg_.engineDevice);
			cfg_.vulkanIndex = oa::Yaml::get<int>(e, "vulkan_index", cfg_.vulkanIndex);
			cfg_.maxMeshDevices = oa::Yaml::get<int>(e, "max_devices", cfg_.maxMeshDevices);
			cfg_.meshIndices = oa::Yaml::get<oa::String>(e, "mesh_indices", cfg_.meshIndices);
			cfg_.meshAuxDemo = oa::Yaml::get<bool>(e, "mesh_aux_demo", cfg_.meshAuxDemo);
			cfg_.precisionStr = oa::Yaml::get<oa::String>(e, "precision", cfg_.precisionStr);
		}
		// Callback config: training.callbacks (top-level callbacks: also accepted).
		// Entries are bool toggles or maps with per-callback params — see callbacks docs.
		loadCallbacksYaml(inYaml["callbacks"]);
		if (auto t = inYaml["training"]) {
			loadCallbacksYaml(t["callbacks"]);
		}

		// phases: prefer training.phases: (new layout), fall back to top-level training_phases: (legacy).
		auto phaseSeq = [&]() -> oa::Yaml::Node {
			if (auto t = inYaml["training"]) {
				if (auto p = t["phases"]; p && p.isSequence()) return p;
			}
			if (auto p = inYaml["training_phases"]; p && p.isSequence()) return p;
			return {};
		}();
		if (auto seq = phaseSeq; seq && seq.isSequence()) {
			cfg_.trainingPhases.clear();
			for (size_t i = 0; i < seq.size(); ++i) {
				const auto& n = seq[i];
				if (!n || !n.isMap()) {
					continue;
				}
				ConfigTrainPhase p;
				p.id = oa::Yaml::get<oa::String>(n, "id", oa::String("phase_") + oa::toString(static_cast<oa::U32>(i)));
				p.epochs = oa::Yaml::get<int>(n, "epochs", 0);
				p.steps = oa::Yaml::get<int>(n, "steps", 0);
				p.dataPath = oa::Yaml::get<oa::String>(n, "data", oa::String{});
				p.validationDataPath = oa::Yaml::get<oa::String>(n, "validation_data", oa::String{});
				if (n["lr"]) {
					p.lr = n["lr"].as<oa::F32>();
				}
				if (n["min_lr"]) {
					p.minLr = n["min_lr"].as<oa::F32>();
				}
				if (n["warmup_steps"]) {
					p.warmupSteps = n["warmup_steps"].as<oa::I32>();
				}
				if (n["eval_interval"]) {
					p.evalInterval = n["eval_interval"].as<oa::I32>();
				}
				if (n["val_batches"]) {
					p.valBatches = n["val_batches"].as<oa::I32>();
				}
				if (n["early_stop_patience"]) {
					p.earlyStopPatience = n["early_stop_patience"].as<oa::I32>();
				}
				if (n["early_stop_min_delta"]) {
					p.earlyStopMinDelta = n["early_stop_min_delta"].as<oa::F32>();
				}
				cfg_.trainingPhases.pushBack(std::move(p));
			}
			if (!cfg_.trainingPhases.empty()) {
				// steps total only when every phase gives raw steps; epoch-based
				// phases resolve to steps later (corpus-derived steps/epoch).
				oa::I32 sum = 0;
				bool allSteps = true;
				for (const auto& ph : cfg_.trainingPhases) {
					sum += ph.steps;
					if (ph.epochs > 0 || ph.steps <= 0) allSteps = false;
				}
				if (allSteps) cfg_.steps = sum;
				if (cfg_.dataPath.empty() && !cfg_.trainingPhases[0].dataPath.empty()) {
					cfg_.dataPath = cfg_.trainingPhases[0].dataPath;
				}
				if (cfg_.validationDataPath.empty() && !cfg_.trainingPhases[0].validationDataPath.empty()) {
					cfg_.validationDataPath = cfg_.trainingPhases[0].validationDataPath;
				}
			}
		}
	}
};

// Chat config — model config comes from .oam
struct ConfigChat {
	oa::String modelPath;
	oa::String prompt;
	oa::String precisionStr = "fp32";
	oa::I32 maxTokens = 256;
	oa::F32 temperature = 0.8f;
	oa::I32 seqLen = 128;
	oa::Bool validate = false;
	// log tok/s and wall time after each generation (stderr). CLI: --gen-metrics / --no-gen-metrics.
	// YAML: verbose. (Global -v,--verbose is OA log level, separate from this.)
	oa::Bool verbose = true;

	[[nodiscard]] oa::Precision precision() const {
		if (precisionStr == "fp32") return oa::Precision::FP32;
		if (precisionStr == "bf16") return oa::Precision::BF16;
		if (precisionStr == "fp64") return oa::Precision::FP64;
		return oa::Precision::FP32;
	}
};

// ChatCli — unified chat CLI for all architectures
class ChatCli : public oa::Cli<ConfigChat> {
public:
	ChatCli() : oa::Cli("chat", "OA Model chat (vulkan Compute)") {
		addOption("--model,-m", cfg_.modelPath, "Path to .oam checkpoint");
		addOption("--prompt,-p", cfg_.prompt, "input prompt (omit for interactive)");
		addOption("--max-tokens,-n", cfg_.maxTokens, "Max tokens to generate");
		addOption("--temperature,-t", cfg_.temperature, "Sampling temperature");
		addOption("--seq-len", cfg_.seqLen, "context window for generation");
		addOption("--precision", cfg_.precisionStr, "fp32 | bf16 | fp64");
		addFlag("--validate", cfg_.validate, "Enable vulkan validation layers");
		addFlag("--gen-metrics,--verbose-gen,--no-gen-metrics{false}", cfg_.verbose,
			"log generation tok/s after each reply (default: on; YAML: verbose. Global -v is log level only)");

		setEpilog(
			"architecture is auto-detected from the .oam checkpoint.\n"
			"\n"
			"Examples:\n"
			"  ./chat -m var/model/dev/OaLlm/OaLlm.oam\n"
			"  ./chat -m model.oam -p \"Once upon a time\" -t 0.5\n"
		);
	}

protected:
	void loadYaml(const oa::Yaml::Node& inYaml) override {
		cfg_.modelPath = oa::Yaml::get<oa::String>(inYaml, "model", cfg_.modelPath);
		cfg_.precisionStr = oa::Yaml::get<oa::String>(inYaml, "precision", cfg_.precisionStr);
		cfg_.maxTokens = oa::Yaml::get<int>(inYaml, "max_tokens", cfg_.maxTokens);
		cfg_.temperature = oa::Yaml::get<float>(inYaml, "temperature", cfg_.temperature);
		cfg_.seqLen = oa::Yaml::get<int>(inYaml, "seq_len", cfg_.seqLen);
		cfg_.verbose = oa::Yaml::get<bool>(inYaml, "verbose", cfg_.verbose);
	}
};

} // namespace oa
