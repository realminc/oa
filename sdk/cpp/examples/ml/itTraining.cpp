// OA example — one canonical training lifecycle composed from ItTraining,
// metrics, callbacks, and optional checkpointing.

#include "oaTest.h"

#include <oa/ml.h>
#include <oa/ml/autograd.h>

class SimpleModel : public oa::Module {
public:
	SimpleModel() {
		fc1_ = oa::makeShared<oa::Linear>(10, 32);
		fc1_->setActivation(oa::Activation::Relu);
		fc2_ = oa::makeShared<oa::Linear>(32, 2);
		registerModule("fc1", fc1_);
		registerModule("fc2", fc2_);
	}

	oa::Matrix forward(const oa::Matrix& inX) override {
		return fc2_->forward(fc1_->forward(inX));
	}

private:
	oa::SharedPtr<oa::Linear> fc1_;
	oa::SharedPtr<oa::Linear> fc2_;
};

TEST(ItTrainingExample, ComposesStandardCallbacks) {
	auto model = oa::makeShared<SimpleModel>();
	auto parameters = model->allParameterPtrs();
	oa::AdamW optimizer(parameters, 0.001F);
	const oa::Matrix x = oa::FnMatrix::randN(oa::MatrixShape{64, 10});
	const oa::Matrix y = oa::FnMatrix::zeros(
		oa::MatrixShape{64}, oa::ScalarType::Int32);

	oa::MetricLoss lossMetric;
	oa::CbProgressBar progress;
	progress.addMetric(&lossMetric);
	oa::CbSummary summary;
	oa::ItTraining training(testEngine(), optimizer, oa::ItTrainingConfig{
		.totalSteps = 100,
		.batchSize = 64,
		.metrics = {&lossMetric},
		.callbacks = {&progress, &summary},
	});

	while (not training.isDone()) {
		optimizer.zeroGrad();
		oa::GradientTape tape;
		const oa::Matrix logits = model->forward(x);
		const oa::Matrix loss = oa::FnLoss::crossEntropy(logits, y);
		tape.backward(loss);
		training.next(loss);
	}

	ASSERT_TRUE(training.finish().isOk());
	EXPECT_EQ(training.stepCount(), 100);
	EXPECT_TRUE(training.lastLoss() >= 0.0F);
}

TEST(ItTrainingExample, CheckpointingIsAnExplicitCallback) {
	auto model = oa::makeShared<SimpleModel>();
	auto parameters = model->allParameterPtrs();
	oa::AdamW optimizer(parameters, 0.001F);
	const oa::Matrix x = oa::FnMatrix::randN(oa::MatrixShape{64, 10});
	const oa::Matrix y = oa::FnMatrix::zeros(
		oa::MatrixShape{64}, oa::ScalarType::Int32);

	oa::CheckpointManager checkpoints(testEngine(), oa::CheckpointManagerConfig{
		.dir = "/tmp/oa-it-training-example",
		.modelName = "simple_model",
		.maxKeep = 2,
	});
	oa::CbCheckpoint checkpoint(
		checkpoints, *model, optimizer, 25, nullptr, false);
	oa::ItTraining training(testEngine(), optimizer, oa::ItTrainingConfig{
		.totalSteps = 50,
		.batchSize = 64,
		.callbacks = {&checkpoint},
	});

	while (not training.isDone()) {
		optimizer.zeroGrad();
		oa::GradientTape tape;
		const oa::Matrix logits = model->forward(x);
		const oa::Matrix loss = oa::FnLoss::crossEntropy(logits, y);
		tape.backward(loss);
		training.next(loss);
	}

	ASSERT_TRUE(training.finish().isOk());
	EXPECT_EQ(training.stepCount(), 50);
}
