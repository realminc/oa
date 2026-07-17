#pragma once

#include "../TutorialMl.h"
#include "../../../Test/OaTest.h"

#include <Oa/Core/FileIo.h>
#include <Oa/Data/DsMnist.h>
#include <Oa/Ml.h>
#include <Oa/Ml/Autograd.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Ui/Image.h>
#include <Oa/Vision/FnImage.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace TutorialFashionFlow {

constexpr OaI32 ImageSize = 28;
constexpr OaI32 Pixels = ImageSize * ImageSize;
constexpr OaI32 Sequence = 49;
constexpr OaI32 PatchDim = 16;
constexpr OaI32 ModelDim = 32;
constexpr OaI32 Classes = 10;
constexpr OaI32 Batch = 64;
constexpr OaI32 SampleSteps = 20;
constexpr OaF32 LearningRate = 1.0e-3F;

inline OaI32 TrainSteps() {
	const char* value = std::getenv("OA_GENERATIVE_STEPS");
	if (!value) return 1000;
	return std::max<OaI32>(1,
		static_cast<OaI32>(std::strtol(value, nullptr, 10)));
}

class Model final : public OaModule {
public:
	explicit Model(bool InMoe) : IsMoe_(InMoe) {
		ClassEmbedding_ = OaMakeSharedPtr<OaEmbedding>(Classes, ModelDim);
		OaFlowDenoiserConfig config{
			.InputDim = PatchDim,
			.ConditionDim = ModelDim,
			.Backbone = {
				.DModel = ModelDim,
				.HiddenDim = InMoe ? ModelDim * 2 : ModelDim * 4,
				.SequenceLength = Sequence,
				.NumLayers = 3,
				.NumHeads = 4,
				.NumExperts = InMoe ? 4 : 0,
				.ExpertsPerToken = InMoe ? 2 : 0,
			},
			.TimeScale = 1.0F,
			.ConditionDropoutP = 0.1F,
		};
		Denoiser_ = OaMakeSharedPtr<OaFlowDenoiser>(config);
		RegisterModule("class_embedding", ClassEmbedding_);
		RegisterModule("denoiser", Denoiser_);
	}

	OaMatrix Forward(const OaMatrix& InSample) override {
		return Denoiser_->Forward(InSample.Reshape(OaMatrixShape{
			InSample.Size(0), Sequence, PatchDim})).Reshape(
			OaMatrixShape{InSample.Size(0), Pixels});
	}

	OaMatrix ForwardFlow(
		const OaMatrix& InSample,
		const OaMatrix& InTime,
		const OaMatrix& InLabels) {
		const OaI64 batch = InSample.Size(0);
		auto condition = ClassEmbedding_->Forward(InLabels);
		return Denoiser_->ForwardConditioned(
			InSample.Reshape(OaMatrixShape{batch, Sequence, PatchDim}),
			InTime, condition).Reshape(OaMatrixShape{batch, Pixels});
	}

	OaMatrix ForwardGuided(
		const OaMatrix& InSample,
		const OaMatrix& InTime,
		const OaMatrix& InLabels,
		OaF32 InGuidanceScale) {
		const OaI64 batch = InSample.Size(0);
		auto condition = ClassEmbedding_->Forward(InLabels);
		return Denoiser_->ForwardGuided(
			InSample.Reshape(OaMatrixShape{batch, Sequence, PatchDim}),
			InTime, condition, InGuidanceScale)
			.Reshape(OaMatrixShape{batch, Pixels});
	}

	[[nodiscard]] bool IsMoe() const noexcept { return IsMoe_; }

private:
	bool IsMoe_ = false;
	OaSharedPtr<OaEmbedding> ClassEmbedding_;
	OaSharedPtr<OaFlowDenoiser> Denoiser_;
};

inline OaMatrix Sample(Model& InModel, OaU64 InSeed) {
	OaContext::ScopedEval eval(OaContext::GetDefault());
	OaVec<OaU8> labels(Classes);
	for (OaI32 index = 0; index < Classes; ++index) {
		labels[index] = static_cast<OaU8>(index);
	}
	auto labelMatrix = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(labels.Data(), labels.Size()),
		OaMatrixShape{Classes}, OaScalarType::UInt8);
	auto state = OaFnMatrix::PhiloxNormal(
		OaFnMatrix::Empty(OaMatrixShape{Classes, Pixels}),
		0.0F, 1.0F, InSeed);
	const OaF32 delta = 1.0F / static_cast<OaF32>(SampleSteps);
	for (OaI32 step = SampleSteps; step > 0; --step) {
		auto time = OaFnMatrix::Full(
			OaMatrixShape{Classes, 1}, static_cast<OaF32>(step) * delta);
		auto velocity = InModel.ForwardGuided(
			state, time, labelMatrix, 2.0F);
		state = OaFnFlow::EulerStep(state, velocity, -delta);
	}
	return state;
}

inline OaResult<OaTexture> MakeGrid(
	OaComputeEngine& InEngine, const OaMatrix& InGenerated) {
	auto mapped = OaFnMatrix::ClampMin(OaFnMatrix::ClampMax(
		(InGenerated * 0.5F) + 0.5F, 1.0F), 0.0F);
	// [class,H,W] -> [H,class,W] -> [1,1,H,class*W].
	auto grid = OaFnMatrix::Transpose(
		mapped.Reshape(OaMatrixShape{Classes, ImageSize, ImageSize}), 0, 1)
		.Contiguous().Reshape(OaMatrixShape{1, 1, ImageSize, Classes * ImageSize});
	return OaTexture::FromMatrix(InEngine, grid);
}

inline OaF32 Validate(Model& InModel, OaDsMnist& InValidation) {
	OaContext::ScopedEval eval(OaContext::GetDefault());
	OaF64 total = 0.0;
	OaI32 batches = 0;
	InValidation.Reset(false);
	OaMatrix images;
	OaMatrix labels;
	while (batches < 8 && InValidation.NextBatch(images, labels)) {
		auto clean = OaFnMatrix::Scale(images, 2.0F / 255.0F) - 1.0F;
		auto time = OaFnMatrix::Full(OaMatrixShape{Batch}, 0.5F);
		auto noise = OaFnMatrix::PhiloxNormal(
			OaFnMatrix::Empty(clean.GetShape()), 0.0F, 1.0F,
			9000U + static_cast<OaU64>(batches));
		auto flow = OaFnFlow::LinearMatch(clean, noise, time);
		auto prediction = InModel.ForwardFlow(flow.State, time, labels);
		auto loss = OaFnLoss::Mse(prediction, flow.Velocity);
		auto& context = OaContext::GetDefault();
		const auto execute = context.Execute();
		if (!execute.IsOk()) throw std::runtime_error(execute.GetMessage().c_str());
		const auto sync = context.Sync();
		if (!sync.IsOk()) throw std::runtime_error(sync.GetMessage().c_str());
		total += loss.Item();
		++batches;
	}
	return batches > 0 ? static_cast<OaF32>(total / batches) : 0.0F;
}

inline void Run(bool InMoe) {
	const char* dataDirectory = std::getenv("OA_MNIST_DATA");
	if (!dataDirectory) dataDirectory = "../oapy/dataset/FashionMNIST/raw";
	OaDsMnist train(dataDirectory, "train", Batch, true);
	OaDsMnist validation(dataDirectory, "t10k", Batch, false);
	if (train.NumSamples() == 0 || validation.NumSamples() == 0) {
		GTEST_SKIP() << "Fashion-MNIST not found (set OA_MNIST_DATA)";
	}

	const OaI32 steps = TrainSteps();
	std::printf("\nOA Fashion-MNIST flow — %s FFN\n",
		InMoe ? "dropless MoE" : "dense");
	std::printf("train=%d val=%d batch=%d steps=%d seed=2026\n",
		train.NumSamples(), validation.NumSamples(), Batch, steps);
	OaFnMatrix::SetRngSeed(2026);
	auto model = OaMakeSharedPtr<Model>(InMoe);
	auto parameters = model->AllParameterPtrs();
	OaAdamW optimizer(parameters, LearningRate);
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
		.TimerName = InMoe ? "fashion_flow_moe" : "fashion_flow_dense",
	});

	OaMatrix images;
	OaMatrix labels;
	OaF32 initialLoss = 0.0F;
	while (!training.Loop.IsDone()) {
		if (!train.NextBatch(images, labels)) {
			train.Reset();
			ASSERT_TRUE(train.NextBatch(images, labels));
		}
		auto clean = OaFnMatrix::Scale(images, 2.0F / 255.0F) - 1.0F;
		auto time = OaFnMatrix::PhiloxUniform(
			OaFnMatrix::Empty(OaMatrixShape{Batch}), 0.0F, 1.0F, 0);
		auto noise = OaFnMatrix::PhiloxNormal(
			OaFnMatrix::Empty(clean.GetShape()), 0.0F, 1.0F, 0);
		auto flow = OaFnFlow::LinearMatch(clean, noise, time);
		optimizer.ZeroGrad();
		OaGradientTape tape;
		auto prediction = model->ForwardFlow(flow.State, time, labels);
		auto loss = OaFnLoss::Mse(prediction, flow.Velocity);
		tape.Backward(loss);
		training.Loop.Next(loss);
		if (training.Loop.LastLossStep() == 1) initialLoss = training.Loop.LastLoss();
	}
	ASSERT_TRUE(training.Loop.Finish().IsOk());
	const OaF32 finalLoss = training.Loop.LastLoss();
	const OaF32 validationLoss = Validate(*model, validation);
	std::printf("flow_mse train %.6f -> %.6f · val %.6f -> %.6f\n",
		initialLoss, finalLoss, initialValidationLoss, validationLoss);

	auto& context = OaContext::GetDefault();
	auto& engine = *OaComputeEngine::GetGlobal();
	auto generated = Sample(*model, 2026);
	auto textureResult = MakeGrid(engine, generated);
	ASSERT_TRUE(textureResult.IsOk()) << textureResult.GetStatus().ToString().c_str();
	ASSERT_TRUE(context.Execute().IsOk());
	ASSERT_TRUE(context.Sync().IsOk());
	const OaPath directory = OaFileIo::GetVarDir() / "generative";
	ASSERT_TRUE(OaFileIo::CreateDirectories(directory).IsOk());
	const OaPath imagePath = directory /
		(InMoe ? "fashion_flow_moe.png" : "fashion_flow_dense.png");
	ASSERT_TRUE(OaFnImage::SaveFile(
		engine, *textureResult, imagePath.String()).IsOk());
	textureResult->Destroy(engine);

	const OaString checkpoint = InMoe
		? "/tmp/oa_fashion_flow_moe.oam"
		: "/tmp/oa_fashion_flow_dense.oam";
	ASSERT_TRUE(model->Save(checkpoint, optimizer).IsOk());
	auto reloaded = OaMakeSharedPtr<Model>(InMoe);
	auto reloadedParameters = reloaded->AllParameterPtrs();
	OaAdamW reloadedOptimizer(reloadedParameters, LearningRate);
	ASSERT_TRUE(reloaded->Load(checkpoint, reloadedOptimizer).IsOk());
	EXPECT_EQ(reloaded->NumParameters(), model->NumParameters());
	EXPECT_GT(initialLoss, 0.0F);
	EXPECT_TRUE(std::isfinite(finalLoss));
	EXPECT_TRUE(std::isfinite(initialValidationLoss));
	EXPECT_TRUE(std::isfinite(validationLoss));
	if (steps >= 20) EXPECT_LT(validationLoss, initialValidationLoss);
	EXPECT_GT(validationLoss, 0.0F);
}

} // namespace TutorialFashionFlow
