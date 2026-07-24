#pragma once

#include "../TutorialMl.h"
#include "../../../Test/OaTest.h"

#include <Oa/Core/Filesystem.h>
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
constexpr OaI32 PatchSize = 4;
constexpr OaI32 PatchesPerSide = ImageSize / PatchSize;
constexpr OaI32 Sequence = 49;
constexpr OaI32 PatchDim = PatchSize * PatchSize;
constexpr OaI32 ModelDim = 32;
constexpr OaI32 Classes = 10;
constexpr OaI32 Batch = 64;
constexpr OaI32 SampleSteps = 20;
constexpr OaF32 LearningRate = 1.0e-3F;

static_assert(ImageSize % PatchSize == 0);
static_assert(Sequence == PatchesPerSide * PatchesPerSide);

inline OaI32 TrainSteps() {
	const char* value = std::getenv("OA_GENERATIVE_STEPS");
	if (!value) return 1000;
	return std::max<OaI32>(1,
		static_cast<OaI32>(std::strtol(value, nullptr, 10)));
}

inline OaI32 GeneratedClass() {
	const char* value = std::getenv("OA_GENERATIVE_CLASS");
	if (!value) return 0;
	char* end = nullptr;
	const long parsed = std::strtol(value, &end, 10);
	if (end == value || *end != '\0' || parsed < 0 || parsed >= Classes) {
		throw std::invalid_argument(
			"OA_GENERATIVE_CLASS must be an integer in [0,9]");
	}
	return static_cast<OaI32>(parsed);
}

inline OaMatrix Patchify(const OaMatrix& InImages) {
	if (InImages.Rank() != 2 || InImages.Size(1) != Pixels) {
		throw std::invalid_argument(
			"Fashion flow patchify expects [B,784] images");
	}
	const OaI64 batch = InImages.Size(0);
	// [B,7,4,7,4] -> [B,7,7,4,4] without requiring a rank-5
	// permutation. Both rank-3 transposes are materialized GPU operations.
	auto columns = OaFnMatrix::Transpose(
		InImages.Reshape(OaMatrixShape{
			batch * PatchesPerSide, PatchSize, ImageSize}),
		1, 2);
	auto patches = OaFnMatrix::Transpose(
		columns.Reshape(OaMatrixShape{
			batch * Sequence, PatchSize, PatchSize}),
		1, 2);
	return patches.Reshape(OaMatrixShape{batch, Sequence, PatchDim});
}

inline OaMatrix Unpatchify(const OaMatrix& InPatches) {
	if (InPatches.Rank() != 3 || InPatches.Size(1) != Sequence
		|| InPatches.Size(2) != PatchDim) {
		throw std::invalid_argument(
			"Fashion flow unpatchify expects [B,49,16] patches");
	}
	const OaI64 batch = InPatches.Size(0);
	auto columns = OaFnMatrix::Transpose(
		InPatches.Reshape(OaMatrixShape{
			batch * Sequence, PatchSize, PatchSize}),
		1, 2);
	auto rows = OaFnMatrix::Transpose(
		columns.Reshape(OaMatrixShape{
			batch * PatchesPerSide, ImageSize, PatchSize}),
		1, 2);
	return rows.Reshape(OaMatrixShape{batch, Pixels});
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
		return Unpatchify(Denoiser_->Forward(Patchify(InSample)));
	}

	OaMatrix ForwardFlow(
		const OaMatrix& InSample,
		const OaMatrix& InTime,
		const OaMatrix& InLabels) {
		auto condition = ClassEmbedding_->Forward(InLabels);
		return Unpatchify(Denoiser_->ForwardConditioned(
			Patchify(InSample), InTime, condition));
	}

	OaMatrix ForwardGuided(
		const OaMatrix& InSample,
		const OaMatrix& InTime,
		const OaMatrix& InLabels,
		OaF32 InGuidanceScale) {
		auto condition = ClassEmbedding_->Forward(InLabels);
		return Unpatchify(Denoiser_->ForwardGuided(
			Patchify(InSample), InTime, condition, InGuidanceScale));
	}

	[[nodiscard]] bool IsMoe() const noexcept { return IsMoe_; }

private:
	bool IsMoe_ = false;
	OaSharedPtr<OaEmbedding> ClassEmbedding_;
	OaSharedPtr<OaFlowDenoiser> Denoiser_;
};

inline OaMatrix Sample(
	Model& InModel,
	OaU64 InSeed,
	OaI32 InClass) {
	OaModule::ScopedEval eval(InModel);
	const OaU8 label = static_cast<OaU8>(InClass);
	auto labelMatrix = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(&label, 1),
		OaMatrixShape{1}, OaScalarType::UInt8);
	auto state = OaFnMatrix::PhiloxNormal(
		OaFnMatrix::Empty(OaMatrixShape{1, Pixels}),
		0.0F, 1.0F, InSeed);
	const OaF32 delta = 1.0F / static_cast<OaF32>(SampleSteps);
	for (OaI32 step = SampleSteps; step > 0; --step) {
		auto time = OaFnMatrix::Full(
			OaMatrixShape{1, 1}, static_cast<OaF32>(step) * delta);
		auto velocity = InModel.ForwardGuided(
			state, time, labelMatrix, 2.0F);
		state = OaFnFlow::EulerStep(state, velocity, -delta);
	}
	return state;
}

inline OaResult<OaTexture> MakeImage(
	OaEngine& InEngine, const OaMatrix& InGenerated) {
	if (InGenerated.GetShape() != OaMatrixShape{1, Pixels}) {
		return OaStatus::InvalidArgument(
			"Fashion flow image expects one flattened 28x28 sample");
	}
	auto mapped = OaFnMatrix::ClampMin(OaFnMatrix::ClampMax(
		(InGenerated * 0.5F) + 0.5F, 1.0F), 0.0F);
	return OaTexture::FromMatrix(InEngine,
		mapped.Reshape(OaMatrixShape{1, 1, ImageSize, ImageSize}));
}

inline void ValidatePatchRoundTrip() {
	OaVec<OaF32> input(Pixels);
	for (OaI32 index = 0; index < Pixels; ++index) {
		input[static_cast<OaUsize>(index)] = static_cast<OaF32>(index);
	}
	auto matrix = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(
			reinterpret_cast<const OaU8*>(input.Data()),
			input.Size() * sizeof(OaF32)),
		OaMatrixShape{1, Pixels}, OaScalarType::Float32);
	auto roundTrip = Unpatchify(Patchify(matrix));
	OaVec<OaF32> output(Pixels);
	ASSERT_TRUE(OaFnMatrix::CopyToHost(
		roundTrip, output.Data(), output.Size() * sizeof(OaF32)).IsOk());
	EXPECT_EQ(output, input);
}

inline void ValidateImageLayout(OaEngine& InEngine) {
	OaVec<OaF32> input(Pixels);
	for (OaI32 y = 0; y < ImageSize; ++y) {
		for (OaI32 x = 0; x < ImageSize; ++x) {
			const bool high = ((y * 5 + x * 7) & 1) != 0;
			input[static_cast<OaUsize>(y * ImageSize + x)] =
				high ? 1.0F : -1.0F;
		}
	}
	auto matrix = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(
			reinterpret_cast<const OaU8*>(input.Data()),
			input.Size() * sizeof(OaF32)),
		OaMatrixShape{1, Pixels}, OaScalarType::Float32);
	auto texture = MakeImage(InEngine, matrix);
	ASSERT_TRUE(texture.IsOk()) << texture.GetStatus().ToString().c_str();
	auto& context = OaContext::GetDefault();
	ASSERT_TRUE(context.Execute().IsOk());
	ASSERT_TRUE(context.Sync().IsOk());
	EXPECT_EQ(texture->Width, ImageSize);
	EXPECT_EQ(texture->Height, ImageSize);

	OaVec<OaU32> pixels(Pixels);
	ASSERT_TRUE(InEngine.ReadbackBuffer(
		texture->DeviceBuf, 0U, pixels.Data(),
		pixels.Size() * sizeof(OaU32)).IsOk());
	OaU32 mismatches = 0U;
	OaI32 firstY = -1;
	OaI32 firstX = -1;
	OaU8 firstExpected = 0U;
	OaU8 firstActual = 0U;
	for (OaI32 y = 0; y < ImageSize; ++y) {
		for (OaI32 x = 0; x < ImageSize; ++x) {
			const OaI32 index = y * ImageSize + x;
			const bool high = ((y * 5 + x * 7) & 1) != 0;
			const OaU8 expected = high ? 255U : 0U;
			const OaU8 actual = static_cast<OaU8>(
				pixels[static_cast<OaUsize>(index)] & 0xFFU);
			if (actual == expected) continue;
			if (mismatches == 0U) {
				firstY = y;
				firstX = x;
				firstExpected = expected;
				firstActual = actual;
			}
			++mismatches;
		}
	}
	EXPECT_EQ(mismatches, 0U)
		<< "first image mismatch y=" << firstY << " x=" << firstX
		<< " expected=" << static_cast<unsigned>(firstExpected)
		<< " actual=" << static_cast<unsigned>(firstActual);
	texture->Destroy(InEngine);
}

inline void ValidateImagePixels(
	OaEngine& InEngine,
	const OaTexture& InTexture) {
	const OaI64 pixelCount = Pixels;
	OaVec<OaU32> pixels(pixelCount);
	ASSERT_TRUE(InEngine.ReadbackBuffer(
		InTexture.DeviceBuf, 0U, pixels.Data(),
		pixels.Size() * sizeof(OaU32)).IsOk());
	OaU8 minimum = 255U;
	OaU8 maximum = 0U;
	OaF64 sum = 0.0;
	OaF64 sumSquares = 0.0;
	for (OaU32 pixel : pixels) {
		const OaU8 value = static_cast<OaU8>(pixel & 0xFFU);
		minimum = std::min(minimum, value);
		maximum = std::max(maximum, value);
		sum += value;
		sumSquares += static_cast<OaF64>(value) * value;
	}
	const OaF64 mean = sum / static_cast<OaF64>(pixelCount);
	const OaF64 variance = std::max(
		0.0, sumSquares / static_cast<OaF64>(pixelCount) - mean * mean);
	const OaF64 standardDeviation = std::sqrt(variance);
	std::printf("image pixels min=%u max=%u mean=%.2f stddev=%.2f\n",
		static_cast<unsigned>(minimum), static_cast<unsigned>(maximum),
		mean, standardDeviation);
	EXPECT_LT(minimum, maximum);
	EXPECT_GT(standardDeviation, 1.0);
}

inline OaF32 Validate(Model& InModel, OaDsMnist& InValidation) {
	OaModule::ScopedEval eval(InModel);
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
	const OaI32 generatedClass = GeneratedClass();
	std::printf("\nOA Fashion-MNIST flow — %s FFN\n",
		InMoe ? "dropless MoE" : "dense");
	std::printf("train=%d val=%d batch=%d steps=%d seed=2026 class=%d\n",
		train.NumSamples(), validation.NumSamples(), Batch, steps,
		generatedClass);
	OaFnMatrix::SetRngSeed(2026);
	auto& engine = *OaEngine::GetGlobal();
	ValidatePatchRoundTrip();
	ValidateImageLayout(engine);
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
	auto generated = Sample(*model, 2026, generatedClass);
	auto textureResult = MakeImage(engine, generated);
	ASSERT_TRUE(textureResult.IsOk()) << textureResult.GetStatus().ToString().c_str();
	ASSERT_TRUE(context.Execute().IsOk());
	ASSERT_TRUE(context.Sync().IsOk());
	EXPECT_EQ(textureResult->Width, ImageSize);
	EXPECT_EQ(textureResult->Height, ImageSize);
	ValidateImagePixels(engine, *textureResult);
	const OaPath directory = OaPaths::Var() / "generative";
	ASSERT_TRUE(OaFilesystem::CreateDirectories(directory).IsOk());
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
