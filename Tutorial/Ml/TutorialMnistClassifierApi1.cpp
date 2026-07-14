// ═══════════════════════════════════════════════════════════════════════════
// OA Tutorial — Fashion-MNIST Image Classification
// Module API: OaModule + OaLinear + OaAdamW + OaFnLoss + OaContext
// ═══════════════════════════════════════════════════════════════════════════
//
// Parallel to TensorFlow Keras (https://www.tensorflow.org/tutorials/keras/classification):
//   tf.keras.Sequential([Dense(128, 'relu'), Dense(10)])
//     ↔  class OaMnistClassifier : public OaModule { OaLinear fc1, fc2; ... };
//   model.compile('adam', loss=SparseCategoricalCrossentropy)
//     ↔  OaAdamW opt(model.Parameters(), lr);   OaFnLoss::CrossEntropy(...)
//   model.fit(x, y, epochs=N)
//     ↔  for (...) { forward; loss; backward; opt.Step(); }
//
// Data: ../oapy/dataset/FashionMNIST/raw (override via OA_MNIST_DATA).
// ═══════════════════════════════════════════════════════════════════════════

#include "../../Test/OaTest.h"
#include "TutorialMl.h"
#include <Oa/Core/EnvFlag.h>
#include <Oa/Ml.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/GpuTimer.h>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <random>
#include <cstring>

// ─── Fashion-MNIST Dataset ─────────────────────────────────────────────────

static const char* kClasses[] = {
	"T-shirt/top", "Trouser",  "Pullover", "Dress",   "Coat",
	"Sandal",      "Shirt",    "Sneaker",  "Bag",     "Ankle boot"
};
static constexpr OaI32 kNumClasses = 10;

// ─── Model: FC1(784→128, ReLU) → FC2(128→10) ──────────────────────────────
// Matches TF Keras Sequential([Flatten, Dense(128, relu), Dense(10)]).
// Backward is hand-wired into the same OaContext as the forward pass
// (Phase 3 implicit-autograd will eliminate this method).

class OaMnistClassifier : public OaModule {
public:
	OaMnistClassifier() {
		auto wd = OaFnMatrix::GetWeightDtype();
		Fc1_ = OaMakeSharedPtr<OaLinear>(784, 128);
		Fc1_->SetActivation(OaActivation::Relu);
		Fc1_->Parameters()[0].Data = OaFnMatrix::RandKaimingUniform(OaMatrixShape{128, 784}, wd);
		Fc2_ = OaMakeSharedPtr<OaLinear>(128, kNumClasses);
		Fc2_->Parameters()[0].Data = OaFnMatrix::RandGlorotUniform(OaMatrixShape{kNumClasses, 128}, wd);
		RegisterModule("fc1", Fc1_);
		RegisterModule("fc2", Fc2_);
	}

	OaMatrix Forward(const OaMatrix& InX) override {
		XNorm_  = OaFnMatrix::Scale(InX, 1.0F / 255.0F);
		H1_     = Fc1_->Forward(XNorm_);   // GEMM + bias + ReLU fused
		Logits_ = Fc2_->Forward(H1_);
		return Logits_;
	}

	void Backward(const OaMatrix& InDLogits) {
		auto& fc1P = Fc1_->Parameters();
		auto& fc2P = Fc2_->Parameters();
		auto gbw2 = OaFnMatrix::LinearWeightBiasBwd(H1_, InDLogits);
		auto dZ1  = OaFnMatrix::LinearDataReluBwd(InDLogits, fc2P[0].Data, H1_);
		auto gbw1 = OaFnMatrix::LinearWeightBiasBwd(XNorm_, dZ1);
		fc1P[0].Grad() = gbw1.GradWeight;
		fc1P[1].Grad() = gbw1.GradBias;
		fc2P[0].Grad() = gbw2.GradWeight;
		fc2P[1].Grad() = gbw2.GradBias;
		}

private:
	OaSharedPtr<OaLinear> Fc1_, Fc2_;
	OaMatrix XNorm_, H1_, Logits_;
};

// ─── Inference ─────────────────────────────────────────────────────────────

struct Prediction { OaI32 ClassIdx; OaF32 Confidence; };

static OaVec<Prediction> Predict(OaMnistClassifier& InModel, const OaMatrix& InX) {
	auto probs = OaFnMatrix::Softmax(InModel.Forward(InX), -1);
	OaI32 batch = static_cast<OaI32>(probs.Size(0));
	OaI32 nCls  = static_cast<OaI32>(probs.Size(1));
	OaVec<OaF32> host(batch * nCls);
	(void)OaFnMatrix::CopyToHost(probs, host.Data(), host.Size() * sizeof(OaF32));

	OaVec<Prediction> out(batch);
	for (OaI32 i = 0; i < batch; ++i) {
		OaI32 best = 0;
		OaF32 bestV = host[i * nCls];
		for (OaI32 j = 1; j < nCls; ++j) {
			OaF32 v = host[i * nCls + j];
			if (v > bestV) { bestV = v; best = j; }
		}
		out[i] = { best, bestV * 100.0F };
	}
	return out;
}

static OaF32 EvalAccuracy(OaMnistClassifier& InModel, OaDsMnist& InLoader, OaI32 InBatch = 100) {
	OaI32 correct = 0, total = 0;
	OaMatrix x, y;
	while (InLoader.NextBatch(x, y)) {
		auto preds = Predict(InModel, x);
		const OaU8* labels = y.DataAs<const OaU8>();
		for (OaI32 i = 0; i < InBatch; ++i) {
			if (preds[i].ClassIdx == OaI32(labels[i])) ++correct;
		}
		total += InBatch;
	}
	InLoader.Reset(false);  // Reset without reshuffling for next eval
	return 100.0F * OaF32(correct) / OaF32(total);
}

// ─── Tutorial ──────────────────────────────────────────────────────────────

TEST(TutorialMnist, FashionMnistClassification) {
	const char* dataDir = std::getenv("OA_MNIST_DATA");
	if (!dataDir) dataDir = "../oapy/dataset/FashionMNIST/raw";

	OaDsMnist trainLoader(OaString(dataDir), "train", 64, /*shuffle=*/true);
	OaDsMnist testLoader(OaString(dataDir), "t10k", 100, /*shuffle=*/false);

	if (trainLoader.NumSamples() == 0 || testLoader.NumSamples() == 0) {
		printf("Fashion-MNIST not found at: %s (set OA_MNIST_DATA).\n", dataDir);
		GTEST_SKIP() << "Dataset not found";
	}

	printf("\n╔══════════════════════════════════════════════════════════════════╗\n");
	printf("║  OA Tutorial — Fashion-MNIST Classification (Module API)        ║\n");
	printf("╚══════════════════════════════════════════════════════════════════╝\n\n");
	printf("Dataset: %d train / %d test, 28×28 grayscale, %d classes\n\n",
		trainLoader.NumSamples(), testLoader.NumSamples(), kNumClasses);

	// ── Model + Optimizer ──
	auto  model  = OaMakeSharedPtr<OaMnistClassifier>();
	auto  params = model->AllParameterPtrs();
	auto  opt    = OaMakeUniquePtr<OaAdamW>(params, 0.001F);
	auto& ctx    = OaContext::GetDefault();
	auto& rt     = *OaComputeEngine::GetGlobal();

	printf("Model: 784 → Linear(128) + ReLU → Linear(%d)\n", kNumClasses);
	printf("Params: %lld    Optimizer: AdamW(lr=0.001)    Loss: CrossEntropy\n\n",
		static_cast<long long>(model->NumParameters()));

	// ── Training Loop ──
	const OaI32 kEpochs = 5;
	const OaI32 kBatch  = 64;
	const OaI32 kSteps  = kEpochs * (trainLoader.NumSamples() / kBatch);

	// Keras-flavored iterator: callbacks declared above the iterator (lifetime
	// is stack scope) and registered via the Callbacks field. The while-loop
	// body looks like an inference call followed by `iter.Next(loss)` —
	// OaItTraining completes and measures every optimizer step before callbacks.
	TutorialTrainingLoop training(*opt, OaItTrainingConfig{
		.TotalSteps     = kSteps,
		.StepsPerEpoch  = trainLoader.NumSamples() / kBatch,
		.EpochSteps     = {},
		.BatchSize      = kBatch,
		.TimerName      = "mnist_api1_step",
		.Callbacks      = {},
	});
	training.AddAccuracyMetric();

	printf("Training: %d epochs × %d steps/epoch · batch=%d\n",
		kEpochs, trainLoader.NumSamples() / kBatch, kBatch);

	// Ring-buffered batches so the GPU can race ahead of CPU sampling.
	OaVec<OaMatrix> xRing(static_cast<OaI64>(ctx.MaxAsyncSubmissions()));
	OaVec<OaMatrix> yRing(static_cast<OaI64>(ctx.MaxAsyncSubmissions()));
	OaF32 initialLoss = 0;
	OaF32 lastLoss    = 0;

	while (not training.Loop.IsDone()) {
		const OaI64 step = training.Loop.Index();  // 1-based current step
		OaMatrix& batchX = xRing[(step - 1) % xRing.Size()];
		OaMatrix& batchY = yRing[(step - 1) % yRing.Size()];
		
		if (not trainLoader.NextBatch(batchX, batchY)) {
			trainLoader.Reset();  // new epoch
			trainLoader.NextBatch(batchX, batchY);
		}

		auto logits     = model->Forward(batchX);
		auto loss       = OaFnLoss::CrossEntropy(logits, batchY);
		auto gradLogits = OaFnLoss::CrossEntropyBwd(logits, batchY);
		model->Backward(gradLogits);
		training.Loop.Next(loss);

		if (step == 1) initialLoss = training.Loop.LiveLoss();
	}
	ASSERT_TRUE(training.Loop.Finish().IsOk()) << "Finish failed";
	lastLoss = training.Loop.LastLoss();

	// ── Evaluate ──
	OaF32 testAcc = EvalAccuracy(*model, testLoader);
	printf("Test accuracy: %.2f%% (over %d samples)\n\n", testAcc, testLoader.NumSamples());

	printf("Predictions on the first 10 test samples:\n");
	printf("  # | Actual              | Predicted           | Conf   \n");
	printf("  ──┼─────────────────────┼─────────────────────┼────────\n");
	
	OaMatrix x10, y10;
	testLoader.NextBatch(x10, y10);
	auto preds = Predict(*model, x10);
	const OaU8* labels = y10.DataAs<const OaU8>();
	for (OaI32 i = 0; i < 10; ++i) {
		OaI32 actual = OaI32(labels[i]);
		OaI32 pred   = preds[i].ClassIdx;
		printf("  %d | %-19s | %-19s | %5.1f%% %s\n",
			i, kClasses[actual], kClasses[pred], preds[i].Confidence,
			actual == pred ? "✓" : "✗");
	}
	testLoader.Reset(false);  // Reset for next eval
	printf("\n");

	ASSERT_GT(initialLoss, 0.0F) << "Initial loss must be non-zero";
	EXPECT_LT(lastLoss, initialLoss) << "Loss must decrease during training";
	EXPECT_GT(testAcc, 70.0F)        << "Test accuracy should exceed 70% after training";

	// ── Checkpoint round-trip via OaCheckpointManager ────────────────────────
	// Demonstrates the canonical save/load path: manager owns directory layout,
	// naming, rotation, best-metric tracking; the OaModule + OaOptimizer overload
	// writes weights + AdamW state in one .oam.
	OaCheckpointManager mgr({
		.ModelName     = "MnistClassifierApi1",
		.Context       = "tutorial",
		.MaxKeep       = 3,
		.MetricName    = "loss",
		.LowerIsBetter = true,
	});
	auto saveStatus = mgr.MaybeSave(*model, *opt, /*step=*/kSteps, /*metric=*/lastLoss);
	ASSERT_TRUE(saveStatus.IsOk()) << "MaybeSave failed: " << saveStatus.GetMessage();

	auto reloaded     = OaMakeSharedPtr<OaMnistClassifier>();
	auto reloadParams = reloaded->AllParameterPtrs();
	auto reloadedOpt  = OaMakeUniquePtr<OaAdamW>(reloadParams, 0.001F);
	auto loadStatus   = mgr.LoadBestInto(*reloaded, *reloadedOpt);
	ASSERT_TRUE(loadStatus.IsOk()) << "LoadBestInto failed: " << loadStatus.GetMessage();

	OaF32 reloadedAcc = EvalAccuracy(*reloaded, testLoader);
	printf("Checkpoint master: %s\n", mgr.GetMasterPath().c_str());
	printf("Reload accuracy: %.2f%% (was %.2f%%)    Optimizer step: %llu (was %llu)\n\n",
		reloadedAcc, testAcc,
		static_cast<unsigned long long>(reloadedOpt->GetStep()),
		static_cast<unsigned long long>(opt->GetStep()));
	EXPECT_NEAR(reloadedAcc, testAcc, 0.5F)            << "Reload accuracy must match within 0.5%";
	EXPECT_EQ(reloadedOpt->GetStep(), opt->GetStep()) << "Optimizer step count must round-trip";
}
