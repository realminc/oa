#pragma once

#include "../TutorialMl.h"
#include "../../../Test/OaTest.h"

#include <Oa/Data/DsHumanMl3d.h>
#include <Oa/Core/FileIo.h>
#include <Oa/Ml.h>
#include <Oa/Ml/Autograd.h>
#include <Oa/Runtime/Context.h>

#include <Anim/Usd.h>
#include <Rig/Skeleton.h>
#include <Rig/SkeletonUsd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace TutorialMotionFlow {

constexpr OaI32 Sequence = 64;
constexpr OaI32 Batch = 4;
constexpr OaI32 ModelDim = 128;
constexpr OaF32 DefaultLearningRate = 2.0e-4F;

inline OaI32 TrainSteps() {
	const char* value = std::getenv("OA_GENERATIVE_MOTION_STEPS");
	if (!value) return 500;
	return std::max<OaI32>(1,
		static_cast<OaI32>(std::strtol(value, nullptr, 10)));
}

inline OaF32 LearningRate() {
	const char* value = std::getenv("OA_GENERATIVE_MOTION_LR");
	if (!value) return DefaultLearningRate;
	const OaF32 parsed = std::strtof(value, nullptr);
	return std::isfinite(parsed) && parsed > 0.0F
		? parsed : DefaultLearningRate;
}

class Model final : public OaModule {
public:
	Model(OaI32 InFeatureDim, OaI32 InConditionDim, bool InMoe)
		: FeatureDim_(InFeatureDim), ConditionDim_(InConditionDim) {
		Denoiser_ = OaMakeSharedPtr<OaFlowDenoiser>(OaFlowDenoiserConfig{
			.InputDim = FeatureDim_,
			.ConditionDim = ConditionDim_,
			.Backbone = {
				.DModel = ModelDim,
				.HiddenDim = InMoe ? ModelDim * 2 : ModelDim * 4,
				.SequenceLength = Sequence,
				.NumLayers = 2,
				.NumHeads = 4,
				.NumExperts = InMoe ? 4 : 0,
				.ExpertsPerToken = InMoe ? 2 : 0,
			},
			.TimeScale = 1.0F,
			.ConditionDropoutP = 0.1F,
		});
		RegisterModule("denoiser", Denoiser_);
	}

	OaMatrix Forward(const OaMatrix& InMotion) override {
		return Denoiser_->Forward(InMotion);
	}

	OaMatrix ForwardFlow(
		const OaMatrix& InMotion,
		const OaMatrix& InTime,
		const OaMatrix& InCondition,
		const OaMatrix& InTokenMask) {
		return Denoiser_->ForwardConditioned(
			InMotion, InTime, InCondition, InTokenMask);
	}

	OaMatrix ForwardGuided(
		const OaMatrix& InMotion,
		const OaMatrix& InTime,
		const OaMatrix& InCondition,
		OaF32 InGuidanceScale,
		const OaMatrix& InTokenMask) {
		return Denoiser_->ForwardGuided(
			InMotion, InTime, InCondition, InGuidanceScale, InTokenMask);
	}

private:
	OaI32 FeatureDim_ = 0;
	OaI32 ConditionDim_ = 0;
	OaSharedPtr<OaFlowDenoiser> Denoiser_;
};

struct BatchData {
	OaMatrix Motion;
	OaMatrix Mask;
	OaMatrix Condition;
	OaVec<OaI32> Lengths;
};

inline BatchData NextBatch(OaDsCmp& InDataset, OaI32& InOutCursor) {
	const OaI32 features = InDataset.FeatDim();
	const OaI32 conditionDim = InDataset.TextFeatureDim();
	OaVec<OaF32> motion(static_cast<OaI64>(Batch) * Sequence * features);
	OaVec<OaF32> mask(static_cast<OaI64>(Batch) * Sequence);
	OaVec<OaF32> condition;
	if (conditionDim > 0) condition.Resize(static_cast<OaI64>(Batch) * conditionDim);
	std::fill(motion.begin(), motion.end(), 0.0F);
	std::fill(mask.begin(), mask.end(), 0.0F);
	if (!condition.Empty()) std::fill(condition.begin(), condition.end(), 0.0F);

	for (OaI32 batch = 0; batch < Batch; ++batch) {
		const OaI32 clip = InOutCursor++ % InDataset.NumClips();
		const OaI32 frames = std::min(Sequence, InDataset.ClipFrames(clip));
		if (frames > 0) {
			std::memcpy(
				motion.Data() + static_cast<OaI64>(batch) * Sequence * features,
				InDataset.ClipData(clip),
				static_cast<OaUsize>(frames) * features * sizeof(OaF32));
			std::fill(mask.begin() + static_cast<OaI64>(batch) * Sequence,
				mask.begin() + static_cast<OaI64>(batch) * Sequence + frames, 1.0F);
		}
		if (conditionDim > 0 && InDataset.ClipTextFeatureCount(clip) > 0) {
			std::memcpy(condition.Data() + static_cast<OaI64>(batch) * conditionDim,
				InDataset.ClipTextFeatureData(clip),
				static_cast<OaUsize>(conditionDim) * sizeof(OaF32));
		}
	}

	BatchData result;
	result.Lengths.Resize(Batch);
	result.Motion = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(motion.Data()),
			motion.Size() * sizeof(OaF32)),
		OaMatrixShape{Batch, Sequence, features}, OaScalarType::Float32);
	result.Mask = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(mask.Data()),
			mask.Size() * sizeof(OaF32)),
		OaMatrixShape{Batch, Sequence, 1}, OaScalarType::Float32);
	if (conditionDim > 0) {
		result.Condition = OaFnMatrix::FromBytes(
			OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(condition.Data()),
				condition.Size() * sizeof(OaF32)),
			OaMatrixShape{Batch, conditionDim}, OaScalarType::Float32);
	}
	// The cursor has advanced; recover the exact valid lengths represented by
	// this batch without reading the GPU mask back to the host.
	for (OaI32 batch = 0; batch < Batch; ++batch) {
		const OaI32 clip = (InOutCursor - Batch + batch) % InDataset.NumClips();
		result.Lengths[batch] = std::min(Sequence, InDataset.ClipFrames(clip));
	}
	return result;
}

inline OaF32 Validate(Model& InModel, OaDsCmp& InDataset) {
	OaContext::ScopedEval eval(OaContext::GetDefault());
	OaI32 cursor = 0;
	OaF64 total = 0.0;
	constexpr OaI32 batches = 4;
	for (OaI32 index = 0; index < batches; ++index) {
		auto batch = NextBatch(InDataset, cursor);
		auto time = OaFnMatrix::Full(OaMatrixShape{Batch}, 0.5F);
		auto noise = OaFnMatrix::PhiloxNormal(
			OaFnMatrix::Empty(batch.Motion.GetShape()), 0.0F, 1.0F,
			12000U + static_cast<OaU64>(index));
		auto flow = OaFnFlow::LinearMatch(batch.Motion, noise, time);
		auto prediction = InModel.ForwardFlow(
			flow.State, time, batch.Condition, batch.Mask);
		auto maskedLoss = OaFnFlow::MaskedMse(
			prediction, flow.Velocity, batch.Mask);
		auto& context = OaContext::GetDefault();
		const auto execute = context.Execute();
		if (!execute.IsOk()) throw std::runtime_error(execute.GetMessage().c_str());
		const auto sync = context.Sync();
		if (!sync.IsOk()) throw std::runtime_error(sync.GetMessage().c_str());
		total += maskedLoss.Item();
	}
	return static_cast<OaF32>(total / batches);
}

inline void ValidateGeometryAndExport(
	Model& InModel, OaDsCmp& InDataset, bool InMoe) {
	OaContext::ScopedEval eval(OaContext::GetDefault());
	OaI32 cursor = 0;
	auto batch = NextBatch(InDataset, cursor);
	auto state = OaFnMatrix::PhiloxNormal(
		OaFnMatrix::Empty(batch.Motion.GetShape()), 0.0F, 1.0F, 44001U);
	constexpr OaI32 sampleSteps = 20;
	constexpr OaF32 guidance = 2.0F;
	const OaF32 delta = 1.0F / static_cast<OaF32>(sampleSteps);
	for (OaI32 step = sampleSteps; step > 0; --step) {
		auto time = OaFnMatrix::Full(
			OaMatrixShape{Batch, 1}, static_cast<OaF32>(step) * delta);
		auto velocity = batch.Condition.IsEmpty()
			? InModel.ForwardFlow(state, time, batch.Condition, batch.Mask)
			: InModel.ForwardGuided(
				state, time, batch.Condition, guidance, batch.Mask);
		state = OaFnFlow::EulerStep(state, velocity, -delta);
	}
	auto& context = OaContext::GetDefault();
	ASSERT_TRUE(context.Execute().IsOk());
	ASSERT_TRUE(context.Sync().IsOk());

	const OaI32 features = InDataset.FeatDim();
	const OaI32 frames = batch.Lengths[0];
	ASSERT_GT(frames, 0);
	OaVec<OaF32> generatedHost(
		static_cast<OaUsize>(Batch) * Sequence * features);
	OaVec<OaF32> targetHost(
		static_cast<OaUsize>(Batch) * Sequence * features);
	ASSERT_TRUE(OaFnMatrix::CopyToHost(state, generatedHost.Data(),
		generatedHost.Size() * sizeof(OaF32)).IsOk());
	ASSERT_TRUE(OaFnMatrix::CopyToHost(batch.Motion, targetHost.Data(),
		targetHost.Size() * sizeof(OaF32)).IsOk());

	OaVec<OaF32> generated(static_cast<OaUsize>(frames) * features);
	OaVec<OaF32> target(static_cast<OaUsize>(frames) * features);
	std::memcpy(generated.Data(), generatedHost.Data(),
		generated.Size() * sizeof(OaF32));
	std::memcpy(target.Data(), targetHost.Data(),
		target.Size() * sizeof(OaF32));
	InDataset.Denormalize(generated.Data(), frames);
	InDataset.Denormalize(target.Data(), frames);
	auto metrics = OaHumanMl3dEvaluateMotion(
		OaSpan<const OaF32>(generated.Data(), generated.Size()),
		OaSpan<const OaF32>(target.Data(), target.Size()),
		frames, features);
	ASSERT_TRUE(metrics.Ok);
	std::printf(
		"geometry MPJPE %.3f cm · velocity %.3f cm/frame · contact %.2f%% · foot skate %.3f cm/frame\n",
		metrics.MpjpeCm, metrics.VelocityErrorCmPerFrame,
		100.0 * metrics.ContactAccuracy, metrics.FootSkateCmPerFrame);

	auto world = OaHumanMl3dRecoverWorldJoints(
		OaSpan<const OaF32>(generated.Data(), generated.Size()), frames, features);
	ASSERT_EQ(world.Size(), static_cast<OaUsize>(frames) * InDataset.NumJoints() * 3);
	for (OaF32 coordinate : world) ASSERT_TRUE(std::isfinite(coordinate));
	auto clip = OaUsdClipFromWorldJoints(
		OaSkHumanMl3d(), OaSpan<const OaF32>(world.Data(), world.Size()),
		frames, 20.0F, 1, 100.0F);
	ASSERT_TRUE(clip.IsValid());
	const OaPath directory = OaFileIo::GetVarDir() / "generative";
	ASSERT_TRUE(OaFileIo::CreateDirectories(directory).IsOk());
	const OaPath path = directory /
		(InMoe ? "motion_flow_moe.usda" : "motion_flow_dense.usda");
	ASSERT_TRUE(OaUsd::WriteUsda(path, clip, "humanml3d").IsOk());
	auto roundTrip = OaUsd::ReadUsda(path);
	ASSERT_TRUE(roundTrip.IsOk());
	EXPECT_EQ(roundTrip->FrameCount, static_cast<OaU32>(frames));
	EXPECT_EQ(roundTrip->JointCount(), InDataset.NumJoints());
	std::printf("USD round-trip: %s · %d frames · %d joints\n",
		path.CStr(), frames, InDataset.NumJoints());
}

inline void Run(bool InMoe) {
	const char* dataDirectory = std::getenv("OA_MOTION_DATA");
	if (!dataDirectory) dataDirectory = "../dataset/gen/3d/anim/ds/Cmp";
	OaDsCmp train(dataDirectory, "train");
	OaDsCmp validation(dataDirectory, "val");
	if (!train.Ok() || !validation.Ok()) {
		GTEST_SKIP() << "CMP/HumanML3D data not found (set OA_MOTION_DATA)";
	}
	if (train.TextFeatureDim() != validation.TextFeatureDim()) {
		GTEST_SKIP() << "train/validation text-feature dimensions do not match";
	}

	const OaI32 steps = TrainSteps();
	const OaF32 learningRate = LearningRate();
	std::printf("\nOA motion flow core — %s FFN\n",
		InMoe ? "dropless MoE" : "dense");
	std::printf("clips train=%d val=%d · features=%d · text=%d · B=%d S=%d · steps=%d · lr=%.1e\n",
		train.NumClips(), validation.NumClips(), train.FeatDim(),
		train.TextFeatureDim(), Batch, Sequence, steps, learningRate);
	OaFnMatrix::SetRngSeed(2026);
	auto model = OaMakeSharedPtr<Model>(
		train.FeatDim(), train.TextFeatureDim(), InMoe);
	auto parameters = model->AllParameterPtrs();
	OaAdamW optimizer(parameters, learningRate);
	// Parameter initialization is deferred GPU work. Resolve it before the
	// fixed validation baseline so short smoke runs cannot recycle an
	// initialization temporary into the first inference graph.
	ASSERT_TRUE(OaContext::GetDefault().Execute().IsOk());
	ASSERT_TRUE(OaContext::GetDefault().Sync().IsOk());
	const OaF32 initialValidationLoss = Validate(*model, validation);
	// Validation uses fixed Philox streams but still consumes runtime RNG state;
	// restore the training stream so the dense/MoE comparison starts identically.
	OaFnMatrix::SetRngSeed(2026);
	TutorialTrainingLoop training(optimizer, OaItTrainingConfig{
		.TotalSteps = steps,
		.BatchSize = Batch,
		.SequenceLength = Sequence,
		.TimerName = InMoe ? "motion_flow_moe" : "motion_flow_dense",
	});
	OaI32 cursor = 0;
	OaF32 initialLoss = 0.0F;
	while (!training.Loop.IsDone()) {
		auto batch = NextBatch(train, cursor);
		auto time = OaFnMatrix::PhiloxUniform(
			OaFnMatrix::Empty(OaMatrixShape{Batch}), 0.0F, 1.0F, 0);
		auto noise = OaFnMatrix::PhiloxNormal(
			OaFnMatrix::Empty(batch.Motion.GetShape()), 0.0F, 1.0F, 0);
		auto flow = OaFnFlow::LinearMatch(batch.Motion, noise, time);
		optimizer.ZeroGrad();
		OaGradientTape tape;
		auto prediction = model->ForwardFlow(
			flow.State, time, batch.Condition, batch.Mask);
		auto loss = OaFnFlow::MaskedMse(
			prediction, flow.Velocity, batch.Mask);
		tape.Backward(loss);
		training.Loop.Next(loss);
		if (training.Loop.LastLossStep() == 1) initialLoss = training.Loop.LastLoss();
	}
	ASSERT_TRUE(training.Loop.Finish().IsOk());
	const OaF32 finalLoss = training.Loop.LastLoss();
	const OaF32 validationLoss = Validate(*model, validation);
	std::printf("masked_flow_mse train %.6f -> %.6f · val %.6f -> %.6f\n",
		initialLoss, finalLoss, initialValidationLoss, validationLoss);
	const OaString checkpoint = InMoe
		? "/tmp/oa_motion_flow_moe.oam"
		: "/tmp/oa_motion_flow_dense.oam";
	ASSERT_TRUE(model->Save(checkpoint, optimizer).IsOk());
	EXPECT_GT(initialLoss, 0.0F);
	EXPECT_TRUE(std::isfinite(finalLoss));
	EXPECT_TRUE(std::isfinite(initialValidationLoss));
	EXPECT_TRUE(std::isfinite(validationLoss));
	if (steps >= 20) EXPECT_LT(validationLoss, initialValidationLoss);
	EXPECT_GT(validationLoss, 0.0F);
	ValidateGeometryAndExport(*model, validation, InMoe);
}

} // namespace TutorialMotionFlow
