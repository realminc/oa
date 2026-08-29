#pragma once

#include "../tutorialMl.h"
#include "oaTest.h"

#include <data/dsHumanMl3d.h>
#include <oa/core/filesystem.h>
#include <oa/core/paths.h>
#include <oa/ml.h>
#include <oa/ml/autograd.h>
#include <oa/runtime/engine.h>

#include <anim/usd.h>
#include <rig/skeleton.h>
#include <rig/skeletonUsd.h>

#include <stdlib.h>

namespace tutorialMotionFlow {

constexpr oa::I32 kSequence = 64;
constexpr oa::I32 kBatch = 4;
constexpr oa::I32 kModelDim = 128;
constexpr oa::F32 kDefaultLearningRate = 2.0e-4F;

inline oa::I32 trainSteps() {
	const char* value = ::getenv("OA_GENERATIVE_MOTION_STEPS");
	if (!value) return 500;
	return oa::max<oa::I32>(1,
		static_cast<oa::I32>(::strtol(value, nullptr, 10)));
}

inline oa::F32 learningRateFromEnvironment() {
	const char* value = ::getenv("OA_GENERATIVE_MOTION_LR");
	if (!value) return kDefaultLearningRate;
	const oa::F32 parsed = ::strtof(value, nullptr);
	return oa::isFinite(parsed) && parsed > 0.0F
		? parsed : kDefaultLearningRate;
}

class Model final : public oa::Module {
public:
	Model(oa::I32 inFeatureDim, oa::I32 inConditionDim, bool inMoe)
		: featureDim_(inFeatureDim), conditionDim_(inConditionDim) {
		denoiser_ = oa::makeShared<oa::FlowDenoiser>(oa::FlowDenoiserConfig{
			.inputDim = featureDim_,
			.conditionDim = conditionDim_,
			.backbone = {
				.dModel = kModelDim,
				.hiddenDim = inMoe ? kModelDim * 2 : kModelDim * 4,
				.sequenceLength = kSequence,
				.numLayers = 2,
				.numHeads = 4,
				.numExperts = inMoe ? 4 : 0,
				.expertsPerToken = inMoe ? 2 : 0,
			},
			.timeScale = 1.0F,
			.conditionDropoutP = 0.1F,
		});
		registerModule("denoiser", denoiser_);
	}

	oa::Matrix forward(const oa::Matrix& inMotion) override {
		return denoiser_->forward(inMotion);
	}

	oa::Matrix forwardFlow(
		const oa::Matrix& inMotion,
		const oa::Matrix& inTime,
		const oa::Matrix& inCondition,
		const oa::Matrix& inTokenMask) {
		return denoiser_->forwardConditioned(
			inMotion, inTime, inCondition, inTokenMask);
	}

	oa::Matrix forwardGuided(
		const oa::Matrix& inMotion,
		const oa::Matrix& inTime,
		const oa::Matrix& inCondition,
		oa::F32 inGuidanceScale,
		const oa::Matrix& inTokenMask) {
		return denoiser_->forwardGuided(
			inMotion, inTime, inCondition, inGuidanceScale, inTokenMask);
	}

private:
	oa::I32 featureDim_ = 0;
	oa::I32 conditionDim_ = 0;
	oa::SharedPtr<oa::FlowDenoiser> denoiser_;
};

struct BatchData {
	oa::Matrix motion;
	oa::Matrix mask;
	oa::Matrix condition;
	oa::Vector<oa::I32> lengths;
};

inline BatchData nextBatch(oa::DsCombatMotionProcessed& inDataset, oa::I32& inOutCursor) {
	const oa::I32 features = inDataset.featDim();
	const oa::I32 conditionDim = inDataset.textFeatureDim();
	oa::Vector<oa::F32> motion(static_cast<oa::I64>(kBatch) * kSequence * features);
	oa::Vector<oa::F32> mask(static_cast<oa::I64>(kBatch) * kSequence);
	oa::Vector<oa::F32> condition;
	if (conditionDim > 0) condition.resize(static_cast<oa::I64>(kBatch) * conditionDim);
	oa::fill(motion.begin(), motion.end(), 0.0F);
	oa::fill(mask.begin(), mask.end(), 0.0F);
	if (!condition.empty()) oa::fill(condition.begin(), condition.end(), 0.0F);

	for (oa::I32 batch = 0; batch < kBatch; ++batch) {
		const oa::I32 clip = inOutCursor++ % inDataset.numClips();
		const oa::I32 frames = oa::min(kSequence, inDataset.clipFrames(clip));
		if (frames > 0) {
			oa::memcpy(
				motion.data() + static_cast<oa::I64>(batch) * kSequence * features,
				inDataset.clipData(clip),
				static_cast<oa::Usize>(frames) * features * sizeof(oa::F32));
			oa::fill(mask.begin() + static_cast<oa::I64>(batch) * kSequence,
				mask.begin() + static_cast<oa::I64>(batch) * kSequence + frames, 1.0F);
		}
		if (conditionDim > 0 && inDataset.clipTextFeatureCount(clip) > 0) {
			oa::memcpy(condition.data() + static_cast<oa::I64>(batch) * conditionDim,
				inDataset.clipTextFeatureData(clip),
				static_cast<oa::Usize>(conditionDim) * sizeof(oa::F32));
		}
	}

	BatchData result;
	result.lengths.resize(kBatch);
	result.motion = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(motion.data()),
			motion.size() * sizeof(oa::F32)),
		oa::MatrixShape{kBatch, kSequence, features}, oa::ScalarType::Float32);
	result.mask = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(mask.data()),
			mask.size() * sizeof(oa::F32)),
		oa::MatrixShape{kBatch, kSequence, 1}, oa::ScalarType::Float32);
	if (conditionDim > 0) {
		result.condition = oa::FnMatrix::fromBytes(
			oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(condition.data()),
				condition.size() * sizeof(oa::F32)),
			oa::MatrixShape{kBatch, conditionDim}, oa::ScalarType::Float32);
	}
	// The cursor has advanced; recover the exact valid lengths represented by
	// this batch without reading the GPU mask back to the host.
	for (oa::I32 batch = 0; batch < kBatch; ++batch) {
		const oa::I32 clip = (inOutCursor - kBatch + batch) % inDataset.numClips();
		result.lengths[batch] = oa::min(kSequence, inDataset.clipFrames(clip));
	}
	return result;
}

inline oa::F32 validate(Model& inModel, oa::DsCombatMotionProcessed& inDataset) {
	oa::Module::ScopedEval eval(inModel);
	oa::I32 cursor = 0;
	oa::F64 total = 0.0;
	constexpr oa::I32 batches = 4;
	for (oa::I32 index = 0; index < batches; ++index) {
		auto batch = nextBatch(inDataset, cursor);
		auto time = oa::FnMatrix::full(oa::MatrixShape{kBatch}, 0.5F);
		auto noise = oa::FnMatrix::philoxNormal(
			oa::FnMatrix::empty(batch.motion.getShape()), 0.0F, 1.0F,
			12000U + static_cast<oa::U64>(index));
		auto flow = oa::FnFlow::linearMatch(batch.motion, noise, time);
		auto prediction = inModel.forwardFlow(
			flow.state, time, batch.condition, batch.mask);
		auto maskedLoss = oa::FnFlow::maskedMse(
			prediction, flow.velocity, batch.mask);
		const auto execute = tutorialSubmitAndWait(testEngine());
		OA_REQUIRE_MSG(execute.isOk(), execute.getMessage().cStr());
		total += maskedLoss.item();
	}
	return static_cast<oa::F32>(total / batches);
}

inline void validateGeometryAndExport(
	Model& inModel, oa::DsCombatMotionProcessed& inDataset, bool inMoe) {
	oa::Module::ScopedEval eval(inModel);
	oa::I32 cursor = 0;
	auto batch = nextBatch(inDataset, cursor);
	auto state = oa::FnMatrix::philoxNormal(
		oa::FnMatrix::empty(batch.motion.getShape()), 0.0F, 1.0F, 44001U);
	constexpr oa::I32 sampleSteps = 20;
	constexpr oa::F32 guidance = 2.0F;
	const oa::F32 delta = 1.0F / static_cast<oa::F32>(sampleSteps);
	for (oa::I32 step = sampleSteps; step > 0; --step) {
		auto time = oa::FnMatrix::full(
			oa::MatrixShape{kBatch, 1}, static_cast<oa::F32>(step) * delta);
		auto velocity = batch.condition.isEmpty()
			? inModel.forwardFlow(state, time, batch.condition, batch.mask)
			: inModel.forwardGuided(
				state, time, batch.condition, guidance, batch.mask);
		state = oa::FnFlow::eulerStep(state, velocity, -delta);
	}
	ASSERT_TRUE(tutorialSubmitAndWait(testEngine()).isOk());

	const oa::I32 features = inDataset.featDim();
	const oa::I32 frames = batch.lengths[0];
	ASSERT_GT(frames, 0);
	oa::Vector<oa::F32> generatedHost(
		static_cast<oa::Usize>(kBatch) * kSequence * features);
	oa::Vector<oa::F32> targetHost(
		static_cast<oa::Usize>(kBatch) * kSequence * features);
	ASSERT_TRUE(oa::FnMatrix::copyToHost(state, generatedHost.data(),
		generatedHost.size() * sizeof(oa::F32)).isOk());
	ASSERT_TRUE(oa::FnMatrix::copyToHost(batch.motion, targetHost.data(),
		targetHost.size() * sizeof(oa::F32)).isOk());

	oa::Vector<oa::F32> generated(static_cast<oa::Usize>(frames) * features);
	oa::Vector<oa::F32> target(static_cast<oa::Usize>(frames) * features);
	oa::memcpy(generated.data(), generatedHost.data(),
		generated.size() * sizeof(oa::F32));
	oa::memcpy(target.data(), targetHost.data(),
		target.size() * sizeof(oa::F32));
	inDataset.denormalize(generated.data(), frames);
	inDataset.denormalize(target.data(), frames);
	auto metrics = oa::humanMl3dEvaluateMotion(
		oa::Span<const oa::F32>(generated.data(), generated.size()),
		oa::Span<const oa::F32>(target.data(), target.size()),
		frames, features);
	ASSERT_TRUE(metrics.ok);
	oa::print("geometry MPJPE {:.3f} cm · velocity {:.3f} cm/frame · contact {:.2f}% · foot skate {:.3f} cm/frame",
		metrics.mpjpeCm, metrics.velocityErrorCmPerFrame,
		100.0 * metrics.contactAccuracy, metrics.footSkateCmPerFrame);

	auto world = oa::humanMl3dRecoverWorldJoints(
		oa::Span<const oa::F32>(generated.data(), generated.size()), frames, features);
	ASSERT_EQ(world.size(), static_cast<oa::Usize>(frames) * inDataset.numJoints() * 3);
	for (oa::F32 coordinate : world) ASSERT_TRUE(oa::isFinite(coordinate));
	auto clip = oa::usdClipFromWorldJoints(
		oa::skHumanMl3d(), oa::Span<const oa::F32>(world.data(), world.size()),
		frames, 20.0F, 1, 100.0F);
	ASSERT_TRUE(clip.isValid());
	const oa::Path directory = oa::Paths::var() / "generative";
	ASSERT_TRUE(oa::Filesystem::createDirectories(directory).isOk());
	const oa::Path path = directory /
		(inMoe ? "motion_flow_moe.usda" : "motion_flow_dense.usda");
	ASSERT_TRUE(oa::Usd::writeUsda(path, clip, "humanml3d").isOk());
	auto roundTrip = oa::Usd::readUsda(path);
	ASSERT_TRUE(roundTrip.isOk());
	EXPECT_EQ(roundTrip->frameCount, static_cast<oa::U32>(frames));
	EXPECT_EQ(roundTrip->jointCount(), inDataset.numJoints());
	oa::print("USD round-trip: {} · {} frames · {} joints",
		path.cStr(), frames, inDataset.numJoints());
}

inline void run(bool inMoe) {
	const char* dataDirectory = ::getenv("OA_MOTION_DATA");
	if (!dataDirectory) {
		static const oa::String defaultData =
			oa::Paths::data("humanMl3d/Cmp").string();
		dataDirectory = defaultData.cStr();
	}
	oa::DsCombatMotionProcessed train(dataDirectory, "train");
	oa::DsCombatMotionProcessed validation(dataDirectory, "val");
	if (!train.ok() || !validation.ok()) {
		GTEST_SKIP() << "CMP/HumanML3D data not found (set OA_MOTION_DATA)";
	}
	if (train.textFeatureDim() != validation.textFeatureDim()) {
		GTEST_SKIP() << "train/validation text-feature dimensions do not match";
	}

	const oa::I32 steps = trainSteps();
	const oa::F32 learningRate = learningRateFromEnvironment();
	oa::print("\nOA motion flow core — {} FFN",
		inMoe ? "dropless MoE" : "dense");
	oa::print("clips train={} val={} · features={} · text={} · B={} S={} · steps={} · lr={:.1e}",
		train.numClips(), validation.numClips(), train.featDim(),
		train.textFeatureDim(), kBatch, kSequence, steps, learningRate);
	oa::FnMatrix::setRngSeed(2026);
	auto& engine = testEngine();
	auto model = oa::makeShared<Model>(
		train.featDim(), train.textFeatureDim(), inMoe);
	auto parameters = model->allParameterPtrs();
	oa::AdamW optimizer(parameters, learningRate);
	// Parameter initialization is deferred GPU work. Resolve it before the
	// fixed validation baseline so short smoke runs cannot recycle an
	// initialization temporary into the first inference graph.
	ASSERT_TRUE(tutorialSubmitAndWait(engine).isOk());
	const oa::F32 initialValidationLoss = validate(*model, validation);
	// Validation uses fixed Philox streams but still consumes runtime RNG state;
	// restore the training stream so the dense/MoE comparison starts identically.
	oa::FnMatrix::setRngSeed(2026);
	TutorialTrainingLoop training(engine, optimizer, oa::ItTrainingConfig{
		.totalSteps = steps,
		.batchSize = kBatch,
		.sequenceLength = kSequence,
		.timerName = inMoe ? "motion_flow_moe" : "motion_flow_dense",
	});
	oa::I32 cursor = 0;
	oa::F32 initialLoss = 0.0F;
	while (!training.loop.isDone()) {
		auto batch = nextBatch(train, cursor);
		auto time = oa::FnMatrix::philoxUniform(
			oa::FnMatrix::empty(oa::MatrixShape{kBatch}), 0.0F, 1.0F, 0);
		auto noise = oa::FnMatrix::philoxNormal(
			oa::FnMatrix::empty(batch.motion.getShape()), 0.0F, 1.0F, 0);
		auto flow = oa::FnFlow::linearMatch(batch.motion, noise, time);
		optimizer.zeroGrad();
		oa::GradientTape tape;
		auto prediction = model->forwardFlow(
			flow.state, time, batch.condition, batch.mask);
		auto loss = oa::FnFlow::maskedMse(
			prediction, flow.velocity, batch.mask);
		tape.backward(loss);
		training.loop.next(loss);
		if (training.loop.lastLossStep() == 1) initialLoss = training.loop.lastLoss();
	}
	ASSERT_TRUE(training.loop.finish().isOk());
	const oa::F32 finalLoss = training.loop.lastLoss();
	const oa::F32 validationLoss = validate(*model, validation);
	oa::print("masked_flow_mse train {:.6f} -> {:.6f} · val {:.6f} -> {:.6f}",
		initialLoss, finalLoss, initialValidationLoss, validationLoss);
	const oa::String checkpoint = inMoe
		? "/tmp/oa_motion_flow_moe.oam"
		: "/tmp/oa_motion_flow_dense.oam";
	ASSERT_TRUE(model->save(engine, checkpoint, optimizer).isOk());
	EXPECT_GT(initialLoss, 0.0F);
	EXPECT_TRUE(oa::isFinite(finalLoss));
	EXPECT_TRUE(oa::isFinite(initialValidationLoss));
	EXPECT_TRUE(oa::isFinite(validationLoss));
	if (steps >= 20) EXPECT_LT(validationLoss, initialValidationLoss);
	EXPECT_GT(validationLoss, 0.0F);
	validateGeometryAndExport(*model, validation, inMoe);
}

} // namespace tutorialMotionFlow
