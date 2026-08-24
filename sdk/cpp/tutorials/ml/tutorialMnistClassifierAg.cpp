// ═══════════════════════════════════════════════════════════════════════════
// OA Tutorial — Fashion-MNIST Image classification (Implicit autograd)
// Module API: oa::Module + oa::Linear + oa::AdamW + oa::FnLoss + oa::GradientTape
// ═══════════════════════════════════════════════════════════════════════════
//
// Sibling of TutorialMnistClassifierApi1.cpp (manual backward — the speed-of-
// light reference). This variant uses *implicit* autograd: oa::GradientTape
// attaches a tape on each forward op; tape.backward(loss) walks the tape and
// records the matching *Bwd kernels into the same engine-owned recording. No hand-written
// backward method on the model.
//
// compare the two side by side to see what the autograd machinery costs.
//
// Data: $OA_DATA_DIR/fashionMnist (fetch with tools/data/manage.py).
// ═══════════════════════════════════════════════════════════════════════════

#include "oaTest.h"
#include "tutorialMl.h"
#include <data/dsMnist.h>
#include <oa/core/envFlag.h>
#include <oa/core/paths.h>
#include <oa/ml.h>
#include <oa/ml/autograd.h>
#include <oa/runtime/engine.h>
#include <algorithm>
#include <chrono>
#include <random>

// ─── Fashion-MNIST Dataset ─────────────────────────────────────────────────

static const char* kClasses[] = {
	"T-shirt/top", "Trouser",  "Pullover", "Dress",   "Coat",
	"Sandal",      "Shirt",    "Sneaker",  "Bag",     "Ankle boot"
};
static constexpr oa::I32 kNumClasses = 10;

// ─── Model: FC1(784→128, ReLU) → FC2(128→10) ──────────────────────────────
// No manual backward — oa::GradientTape walks the tape that oa::Linear/scale/
// CrossEntropy attached during the forward pass.

class MnistClassifier : public oa::Module {
public:
	MnistClassifier() {
		auto wd = oa::FnMatrix::weightDtype();
		fc1_ = oa::makeShared<oa::Linear>(784, 128);
		fc1_->setActivation(oa::Activation::Relu);
		fc1_->parameters()[0].data = oa::FnMatrix::randKaimingUniform(oa::MatrixShape{128, 784}, wd);
		// Re-enable grad tracking on the freshly-assigned Data.
		fc1_->parameters()[0].data.setRequiresGrad(true);
		fc2_ = oa::makeShared<oa::Linear>(128, kNumClasses);
		fc2_->parameters()[0].data = oa::FnMatrix::randGlorotUniform(oa::MatrixShape{kNumClasses, 128}, wd);
		fc2_->parameters()[0].data.setRequiresGrad(true);
		registerModule("fc1", fc1_);
		registerModule("fc2", fc2_);
	}

	oa::Matrix forward(const oa::Matrix& inX) override {
		auto x = oa::FnMatrix::scale(inX, 1.0F / 255.0F);  // tape: GradScale
		auto h = fc1_->forward(x);                       // tape: GradLinearRelu
		return fc2_->forward(h);                         // tape: GradLinear
	}

private:
	oa::SharedPtr<oa::Linear> fc1_, fc2_;
};

// ─── Inference ─────────────────────────────────────────────────────────────

struct Prediction { oa::I32 classIdx; oa::F32 confidence; };

static oa::Vec<Prediction> predict(MnistClassifier& inModel, const oa::Matrix& inX) {
	auto probs = oa::FnMatrix::softmax(inModel.forward(inX), -1);
	oa::I32 batch = static_cast<oa::I32>(probs.size(0));
	oa::I32 nCls  = static_cast<oa::I32>(probs.size(1));
	oa::Vec<oa::F32> host(batch * nCls);
	(void)oa::FnMatrix::copyToHost(probs, host.data(), host.size() * sizeof(oa::F32));

	oa::Vec<Prediction> out(batch);
	for (oa::I32 i = 0; i < batch; ++i) {
		oa::I32 best = 0;
		oa::F32 bestV = host[i * nCls];
		for (oa::I32 j = 1; j < nCls; ++j) {
			oa::F32 v = host[i * nCls + j];
			if (v > bestV) { bestV = v; best = j; }
		}
		out[i] = { best, bestV * 100.0F };
	}
	return out;
}

static oa::F32 evalAccuracy(MnistClassifier& inModel, oa::DsMnist& inLoader) {
	oa::I32 correct = 0, total = 0;
	oa::Matrix x, y;
	while (inLoader.nextBatch(x, y)) {
		auto preds = predict(inModel, x);
		const oa::U8* labels = y.dataAs<const oa::U8>();
		oa::I32 batch = static_cast<oa::I32>(preds.size());
		for (oa::I32 i = 0; i < batch; ++i) {
			if (preds[i].classIdx == oa::I32(labels[i])) ++correct;
		}
		total += batch;
	}
	inLoader.reset(false);
	return 100.0F * oa::F32(correct) / oa::F32(total);
}

// ─── Tutorial ──────────────────────────────────────────────────────────────

TEST(TutorialMnistClassifierAg, FashionMnistClassification) {
	const oa::String dataDir = oa::Paths::data("fashionMnist").string();

	const oa::I32 kBatch = 64;
	oa::DsMnist trainLoader(dataDir, "train", kBatch, /*shuffle=*/true);
	oa::DsMnist testLoader(dataDir, "t10k", 100, /*shuffle=*/false);

	if (trainLoader.numSamples() == 0 || testLoader.numSamples() == 0) {
		printf("Fashion-MNIST not found at: %s (run tools/data/manage.py fetch fashionMnist).\n",
			dataDir.cStr());
		GTEST_SKIP() << "Dataset not found";
	}

	printf("\n╔══════════════════════════════════════════════════════════════════╗\n");
	printf("║  OA Tutorial — Fashion-MNIST classification (Implicit autograd) ║\n");
	printf("╚══════════════════════════════════════════════════════════════════╝\n\n");
	printf("Dataset: %d train / %d test, 28×28 grayscale, %d classes\n\n",
		trainLoader.numSamples(), testLoader.numSamples(), kNumClasses);

	// ── Model + Optimizer ──
	auto  model  = oa::makeShared<MnistClassifier>();
	auto  params = model->allParameterPtrs();
	auto  opt    = oa::makeUnique<oa::AdamW>(params, 0.001F);

	printf("Model: 784 → Linear(128) + ReLU → Linear(%d)\n", kNumClasses);
	printf("params: %lld    Optimizer: AdamW(lr=0.001)    loss: CrossEntropy\n\n",
		static_cast<long long>(model->numParameters()));

	// ── training Loop ──
	const oa::I32 kEpochs = 5;
	const oa::I32 kSteps  = kEpochs * (trainLoader.numSamples() / kBatch);

	// keras-flavored iterator: callbacks declared above the iterator (lifetime
	// is stack scope) and registered via the callbacks field. The while-loop
	// body looks like an inference call followed by `iter.next(loss)` —
	// oa::ItTraining completes and measures every optimizer step before callbacks.
	TutorialTrainingLoop training(testEngine(), *opt, oa::ItTrainingConfig{
		.totalSteps     = kSteps,
		.stepsPerEpoch  = trainLoader.numSamples() / kBatch,
		.epochSteps     = {},
		.batchSize      = kBatch,
		.timerName      = "mnist_autograd_step",
		.callbacks      = {},
	});

	printf("training: %d epochs × %d steps/epoch · batch=%d\n",
		kEpochs, trainLoader.numSamples() / kBatch, kBatch);

	// next() completes each exact step before the following sampler refill.
	oa::Matrix batchX;
	oa::Matrix batchY;
	oa::F32 initialLoss = 0;
	oa::F32 lastLoss    = 0;

	while (not training.loop.isDone()) {
		const oa::I64 step = training.loop.index();  // 1-based current step
		if (!trainLoader.nextBatch(batchX, batchY)) {
			trainLoader.reset();
			trainLoader.nextBatch(batchX, batchY);
		}

		opt->zeroGrad();             // implicit-autograd accumulates, so clear each step
		oa::GradientTape tape;
		auto logits = model->forward(batchX);
		auto loss   = oa::FnLoss::crossEntropy(logits, batchY);
		tape.backward(loss);
		training.loop.next(loss);

		if (step == 1) initialLoss = training.loop.lastLoss();
	}
	ASSERT_TRUE(training.loop.finish().isOk()) << "finish failed";
	lastLoss = training.loop.lastLoss();

	// ── Evaluate ──
	oa::F32 testAcc = evalAccuracy(*model, testLoader);
	printf("Test accuracy: %.2f%% (over %d samples)\n\n", testAcc, testLoader.numSamples());

	printf("Predictions on the first 10 test samples:\n");
	printf("  # | actual              | predicted           | Conf   \n");
	printf("  ──┼─────────────────────┼─────────────────────┼────────\n");
	oa::Matrix x10, y10;
	testLoader.nextBatch(x10, y10);
	auto preds  = predict(*model, x10);
	const oa::U8* labels = y10.dataAs<const oa::U8>();
	for (oa::I32 i = 0; i < 10; ++i) {
		oa::I32 actual = oa::I32(labels[i]);
		oa::I32 pred   = preds[i].classIdx;
		printf("  %d | %-19s | %-19s | %5.1f%% %s\n",
			i, kClasses[actual], kClasses[pred], preds[i].confidence,
			actual == pred ? "✓" : "✗");
	}
	testLoader.reset(false);
	printf("\n");

	ASSERT_GT(initialLoss, 0.0F) << "Initial loss must be non-zero";
	EXPECT_LT(lastLoss, initialLoss) << "loss must decrease during training";
	EXPECT_GT(testAcc, 70.0F)        << "Test accuracy should exceed 70% after training";

	// ── checkpoint round-trip via oa::CheckpointManager ────────────────────────
	// Demonstrates the canonical save/load path: manager owns directory layout,
	// naming, rotation, best-metric tracking; the oa::Module + oa::Optimizer overload
	// writes weights + AdamW state in one .oam.
	oa::CheckpointManager mgr(testEngine(), {
		.modelName     = "MnistClassifierAg",
		.context       = "tutorial",
		.maxKeep       = 3,
		.metricName    = "loss",
		.lowerIsBetter = true,
	});
	auto saveStatus = mgr.maybeSave(*model, *opt, /*step=*/kSteps, /*metric=*/lastLoss);
	ASSERT_TRUE(saveStatus.isOk()) << "maybeSave failed: " << saveStatus.getMessage();

	auto reloaded     = oa::makeShared<MnistClassifier>();
	auto reloadParams = reloaded->allParameterPtrs();
	auto reloadedOpt  = oa::makeUnique<oa::AdamW>(reloadParams, 0.001F);
	auto loadStatus   = mgr.loadBestInto(*reloaded, *reloadedOpt);
	ASSERT_TRUE(loadStatus.isOk()) << "loadBestInto failed: " << loadStatus.getMessage();

	oa::F32 reloadedAcc = evalAccuracy(*reloaded, testLoader);
	printf("checkpoint master: %s\n", mgr.masterPath().cStr());
	printf("Reload accuracy: %.2f%% (was %.2f%%)    Optimizer step: %llu (was %llu)\n\n",
		reloadedAcc, testAcc,
		static_cast<unsigned long long>(reloadedOpt->getStep()),
		static_cast<unsigned long long>(opt->getStep()));
	EXPECT_NEAR(reloadedAcc, testAcc, 0.5F)            << "Reload accuracy must match within 0.5%";
	EXPECT_EQ(reloadedOpt->getStep(), opt->getStep()) << "Optimizer step count must round-trip";
}
