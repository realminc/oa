// ═══════════════════════════════════════════════════════════════════════════
// Training.h — Zero-boilerplate ML training framework
//
// Combines OaComputeEngine ownership + OaItTraining into a single "just train" API.
// No engine setup, no context management, no tick loops, no app boilerplate.
//
// **Usage**:
//   OaMlTraining train(model, optimizer, config);
//   while (train.Step()) {
//       OaGradientTape tape;
//       auto loss = Loss(model->Forward(batch), target);
//       tape.Backward(loss);
//       train.Next(loss);
//   }
//   train.Finish();
//
// Or even simpler with the macro:
//   OA_ML_TRAIN(model, opt, steps, batch) {
//       OaGradientTape tape;
//       auto loss = Loss(model->Forward(GetBatch()), target);
//       tape.Backward(loss);
//       OA_ML_NEXT(loss);
//   }
//
// **Design**: This is the "language/framework" layer - you write training logic,
// we handle runtime/context/metrics/progress/checkpoints automatically.
//
// **Location**: Source/Public/Oa/Ml/Training.h (public API, not tutorial helper)
// ═══════════════════════════════════════════════════════════════════════════

#pragma once

#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Ml/ItTraining.h>
#include <Oa/Ml/TrainingProgram.h>
#include <Oa/Ml/TrainingSession.h>
#include <Oa/Ml/Callbacks.h>
#include <Oa/Ml/Metric.h>
#include <Oa/Ml/Optim.h>
#include <Oa/Ml/Checkpoint.h>
#include <Oa/Core/Types.h>

// ═════════════════════════════════════════════════════════════════════════════
// OaMlTrainingConfig — Simple training configuration
// ═════════════════════════════════════════════════════════════════════════════

struct OaMlTrainingConfig {
	// Training schedule
	OaI32 TotalSteps = 1000;
	OaI32 BatchSize = 64;
	OaI32 SequenceLength = 0;
	OaString SequenceUnit = "token";
	OaI32 StepsPerEpoch = 0;  // 0 = no epoch boundaries
	
	// Metrics & logging
	OaString TimerName = "train_step";
	
	// Checkpointing
	OaBool EnableCheckpoints = false;
	OaString CheckpointDir = "";
	OaString ModelName = "model";
	OaI32 CheckpointSaveEvery = 0;   // 0 = epoch-end; >0 also saves every N completed steps
	OaI32 CheckpointMaxKeep = 5;
	OaBool CheckpointRestoreBest = true;
	
	// Runtime
	OaPrecision Precision = OaPrecision::FP32;
	OaI32 DeviceIndex = -1;  // -1 = auto
	OaBool EnableValidation = false;
};

// ═════════════════════════════════════════════════════════════════════════════
// OaMlTraining — Complete training wrapper
// ═════════════════════════════════════════════════════════════════════════════

class OaMlTraining {
public:
	// ─── Construction ────────────────────────────────────────────────────────

	// Construct with model, optimizer, and config
	// Model must be OaModule* or OaSharedPtr<OaModule>
	template<typename ModelT>
	OaMlTraining(
		ModelT& InModel,
		OaOptimizer& InOptimizer,
		const OaMlTrainingConfig& InConfig = {}
	)
		: Config_(InConfig)
		, Optimizer_(&InOptimizer)
		, Engine_(nullptr)
		, Loop_(nullptr)
		, LossMetric_(nullptr)
		, ProgressBar_(nullptr)
		, Summary_(nullptr)
		, CheckpointCb_(nullptr)
	{
		if constexpr (std::is_pointer_v<ModelT>) {
			Model_ = InModel;
		} else {
			Model_ = InModel.get();
		}
		// Check if we're running inside a TEST() fixture that already has a global engine
		OaComputeEngine* globalEngine = OaComputeEngine::GetGlobal();
		if (globalEngine != nullptr) {
			// Use existing global engine from TEST() fixture
			Engine_ = globalEngine;
		} else {
			OaEngineConfig engineConfig;
			engineConfig.AppName = "OaMlTraining";
			engineConfig.Precision = Config_.Precision;
			engineConfig.EnableValidation = Config_.EnableValidation;
			if (Config_.DeviceIndex >= 0) {
				engineConfig.DevicePref = OaDevicePreference::ByIndex;
				engineConfig.DeviceIndex = static_cast<OaU32>(Config_.DeviceIndex);
			}
			auto engine = OaComputeEngine::Create(engineConfig);
			if (!engine.IsOk()) {
				OA_LOG_ERROR(OaLogComponent::Core,
					"OaMlTraining: engine creation failed: %s",
					engine.GetStatus().GetMessage().c_str());
				Valid_ = false;
				return;
			}
			OwnedEngine_ = std::move(*engine);
			Engine_ = OwnedEngine_.get();
		}

		// Setup metrics
		LossMetric_ = new OaMetricLoss();
		
		ProgressBar_ = new OaCbProgressBar();
		ProgressBar_->AddMetric(LossMetric_);
		
		Summary_ = new OaCbSummary();
		
		// Setup training iterator
		OaItTrainingConfig itCfg;
		itCfg.TotalSteps = Config_.TotalSteps;
		itCfg.StepsPerEpoch = Config_.StepsPerEpoch;
		itCfg.BatchSize = Config_.BatchSize;
		itCfg.SequenceLength = Config_.SequenceLength;
		itCfg.SequenceUnit = Config_.SequenceUnit;
		itCfg.TimerName = Config_.TimerName.c_str();
		itCfg.Metrics = {LossMetric_};
		
		Loop_ = new OaItTraining(*Optimizer_, itCfg);
		Loop_->AddCallback(ProgressBar_);
		
		// Setup checkpoint manager if enabled
		if (Config_.EnableCheckpoints && !Config_.CheckpointDir.empty()) {
			OaCheckpointManagerConfig ckptCfg;
			ckptCfg.Dir = Config_.CheckpointDir;
			ckptCfg.ModelName = Config_.ModelName;
			ckptCfg.MaxKeep = Config_.CheckpointMaxKeep;
			CheckpointMgr_ = new OaCheckpointManager(ckptCfg);
			CheckpointCb_ = new OaCbCheckpoint(*CheckpointMgr_, *Model_, *Optimizer_,
				Config_.CheckpointSaveEvery, nullptr, Config_.CheckpointRestoreBest);
			Loop_->AddCallback(CheckpointCb_);
		}
		Loop_->AddCallback(Summary_);
		
		Valid_ = true;
	}

	~OaMlTraining() {
		delete Loop_;
		delete CheckpointCb_;
		delete Summary_;
		delete ProgressBar_;
		delete LossMetric_;
		delete CheckpointMgr_;
		OwnedEngine_.reset();
	}

	// Non-copyable, non-movable
	OaMlTraining(const OaMlTraining&) = delete;
	OaMlTraining& operator=(const OaMlTraining&) = delete;
	OaMlTraining(OaMlTraining&&) = delete;
	OaMlTraining& operator=(OaMlTraining&&) = delete;

	// ─── Training Loop ───────────────────────────────────────────────────────

	// Check if training should continue
	[[nodiscard]] bool Step() {
		if (!Valid_) return false;
		return Loop_->Session() != nullptr
			? Loop_->Session()->TryBeginStep()
			: !Loop_->IsDone();
	}

	// Record loss and advance (call after backward)
	void Next(const OaMatrix& InLoss) {
		Loop_->Next(InLoss);
	}

	// Finish training (drains in-flight work)
	[[nodiscard]] OaStatus Finish() {
		if (!Valid_) return OaStatus::Error("Training not initialized");
		return Loop_->Finish();
	}

	// ─── Accessors ───────────────────────────────────────────────────────────

	[[nodiscard]] bool IsValid() const noexcept { return Valid_; }
	[[nodiscard]] OaI64 CurrentStep() const { return Loop_->Index(); }
	[[nodiscard]] OaF32 CurrentLoss() const { return Loop_->LiveLoss(); }
	[[nodiscard]] OaComputeEngine& GetEngine() const { return *Engine_; }
	[[nodiscard]] OaContext& GetContext() const { return Engine_->GetContext(); }
	[[nodiscard]] OaItTraining& GetLoop() { return *Loop_; }

	// Add custom metric
	void AddMetric(OaMetric* InMetric) {
		Loop_->AddMetric(InMetric);
		ProgressBar_->AddMetric(InMetric);
	}

private:
	OaMlTrainingConfig Config_;
	OaOptimizer* Optimizer_;
	OaModule* Model_ = nullptr;
	OaComputeEngine* Engine_;
	OaUniquePtr<OaComputeEngine> OwnedEngine_;
	OaItTraining* Loop_;
	
	// Metrics
	OaMetricLoss* LossMetric_;
	
	// Callbacks
	OaCbProgressBar* ProgressBar_;
	OaCbSummary* Summary_;
	OaCbCheckpoint* CheckpointCb_;
	
	// Checkpointing
	OaCheckpointManager* CheckpointMgr_ = nullptr;

	bool Valid_ = false;
};

// ═════════════════════════════════════════════════════════════════════════════
// Convenience Macros
// ═════════════════════════════════════════════════════════════════════════════

// Simple training loop macro
// Usage:
//   OA_ML_TRAIN(model, optimizer, 1000, 64) {
//       OaGradientTape tape;
//       auto loss = Loss(model->Forward(batch), target);
//       tape.Backward(loss);
//       OA_ML_NEXT(loss);
//   }
#define OA_ML_TRAIN(model, optimizer, steps, batch) \
	OaMlTraining __ml_train(model, optimizer, OaMlTrainingConfig{ \
		.TotalSteps = steps, \
		.BatchSize = batch \
	}); \
	while (__ml_train.Step())

// Training loop with config
#define OA_ML_TRAIN_CFG(model, optimizer, config) \
	OaMlTraining __ml_train(model, optimizer, config); \
	while (__ml_train.Step())

// Record loss (call after backward)
#define OA_ML_NEXT(loss) \
	__ml_train.Next(loss)

// Finish training
#define OA_ML_FINISH() \
	(void)__ml_train.Finish()

// Access current step
#define OA_ML_STEP() \
	__ml_train.CurrentStep()

// Access context
#define OA_ML_CONTEXT() \
	__ml_train.GetContext()
