#pragma once

#include "../tutorialMl.h"
#include "oaTest.h"

#include <oa/core/filesystem.h>
#include <data/dsMnist.h>
#include <oa/ml.h>
#include <oa/ml/autograd.h>
#include <oa/runtime/engine.h>
#include <oa/ui/image.h>
#include <oa/vision/fnImage.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace tutorialFashionFlow {

constexpr oa::I32 kImageSize = 28;
constexpr oa::I32 kPixels = kImageSize * kImageSize;
constexpr oa::I32 kPatchSize = 4;
constexpr oa::I32 kPatchesPerSide = kImageSize / kPatchSize;
constexpr oa::I32 kSequence = 49;
constexpr oa::I32 kPatchDim = kPatchSize * kPatchSize;
constexpr oa::I32 kModelDim = 32;
constexpr oa::I32 kClasses = 10;
constexpr oa::I32 kBatch = 64;
constexpr oa::I32 kSampleSteps = 20;
constexpr oa::F32 kLearningRate = 1.0e-3F;

static_assert(kImageSize % kPatchSize == 0);
static_assert(kSequence == kPatchesPerSide * kPatchesPerSide);

inline oa::I32 trainSteps() {
	const char* value = std::getenv("OA_GENERATIVE_STEPS");
	if (!value) return 1000;
	return std::max<oa::I32>(1,
		static_cast<oa::I32>(std::strtol(value, nullptr, 10)));
}

inline oa::I32 generatedClassFromEnvironment() {
	const char* value = std::getenv("OA_GENERATIVE_CLASS");
	if (!value) return 0;
	char* end = nullptr;
	const long parsed = std::strtol(value, &end, 10);
	if (end == value || *end != '\0' || parsed < 0 || parsed >= kClasses) {
		throw std::invalid_argument(
			"OA_GENERATIVE_CLASS must be an integer in [0,9]");
	}
	return static_cast<oa::I32>(parsed);
}

inline oa::Matrix patchify(const oa::Matrix& inImages) {
	if (inImages.rank() != 2 || inImages.size(1) != kPixels) {
		throw std::invalid_argument(
			"Fashion flow patchify expects [B,784] images");
	}
	const oa::I64 batch = inImages.size(0);
	// [B,7,4,7,4] -> [B,7,7,4,4] without requiring a rank-5
	// permutation. Both rank-3 transposes are materialized GPU operations.
	auto columns = oa::FnMatrix::transpose(
		inImages.reshape(oa::MatrixShape{
			batch * kPatchesPerSide, kPatchSize, kImageSize}),
		1, 2);
	auto patches = oa::FnMatrix::transpose(
		columns.reshape(oa::MatrixShape{
			batch * kSequence, kPatchSize, kPatchSize}),
		1, 2);
	return patches.reshape(oa::MatrixShape{batch, kSequence, kPatchDim});
}

inline oa::Matrix unpatchify(const oa::Matrix& inPatches) {
	if (inPatches.rank() != 3 || inPatches.size(1) != kSequence
		|| inPatches.size(2) != kPatchDim) {
		throw std::invalid_argument(
			"Fashion flow unpatchify expects [B,49,16] patches");
	}
	const oa::I64 batch = inPatches.size(0);
	auto columns = oa::FnMatrix::transpose(
		inPatches.reshape(oa::MatrixShape{
			batch * kSequence, kPatchSize, kPatchSize}),
		1, 2);
	auto rows = oa::FnMatrix::transpose(
		columns.reshape(oa::MatrixShape{
			batch * kPatchesPerSide, kImageSize, kPatchSize}),
		1, 2);
	return rows.reshape(oa::MatrixShape{batch, kPixels});
}

class Model final : public oa::Module {
public:
	explicit Model(bool inMoe) : isMoe_(inMoe) {
		classEmbedding_ = oa::makeShared<oa::Embedding>(kClasses, kModelDim);
		oa::FlowDenoiserConfig config{
			.inputDim = kPatchDim,
			.conditionDim = kModelDim,
			.backbone = {
				.dModel = kModelDim,
				.hiddenDim = inMoe ? kModelDim * 2 : kModelDim * 4,
				.sequenceLength = kSequence,
				.numLayers = 3,
				.numHeads = 4,
				.numExperts = inMoe ? 4 : 0,
				.expertsPerToken = inMoe ? 2 : 0,
			},
			.timeScale = 1.0F,
			.conditionDropoutP = 0.1F,
		};
		denoiser_ = oa::makeShared<oa::FlowDenoiser>(config);
		registerModule("class_embedding", classEmbedding_);
		registerModule("denoiser", denoiser_);
	}

	oa::Matrix forward(const oa::Matrix& inSample) override {
		return unpatchify(denoiser_->forward(patchify(inSample)));
	}

	oa::Matrix forwardFlow(
		const oa::Matrix& inSample,
		const oa::Matrix& inTime,
		const oa::Matrix& inLabels) {
		auto condition = classEmbedding_->forward(inLabels);
		return unpatchify(denoiser_->forwardConditioned(
			patchify(inSample), inTime, condition));
	}

	oa::Matrix forwardGuided(
		const oa::Matrix& inSample,
		const oa::Matrix& inTime,
		const oa::Matrix& inLabels,
		oa::F32 inGuidanceScale) {
		auto condition = classEmbedding_->forward(inLabels);
		return unpatchify(denoiser_->forwardGuided(
			patchify(inSample), inTime, condition, inGuidanceScale));
	}

	[[nodiscard]] bool isMoe() const noexcept { return isMoe_; }

private:
	bool isMoe_ = false;
	oa::SharedPtr<oa::Embedding> classEmbedding_;
	oa::SharedPtr<oa::FlowDenoiser> denoiser_;
};

inline oa::Matrix sample(
	Model& inModel,
	oa::U64 inSeed,
	oa::I32 inClass) {
	oa::Module::ScopedEval eval(inModel);
	const oa::U8 label = static_cast<oa::U8>(inClass);
	auto labelMatrix = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(&label, 1),
		oa::MatrixShape{1}, oa::ScalarType::UInt8);
	auto state = oa::FnMatrix::philoxNormal(
		oa::FnMatrix::empty(oa::MatrixShape{1, kPixels}),
		0.0F, 1.0F, inSeed);
	const oa::F32 delta = 1.0F / static_cast<oa::F32>(kSampleSteps);
	for (oa::I32 step = kSampleSteps; step > 0; --step) {
		auto time = oa::FnMatrix::full(
			oa::MatrixShape{1, 1}, static_cast<oa::F32>(step) * delta);
		auto velocity = inModel.forwardGuided(
			state, time, labelMatrix, 2.0F);
		state = oa::FnFlow::eulerStep(state, velocity, -delta);
	}
	return state;
}

inline oa::Result<oa::Texture> makeImage(
	oa::Engine& inEngine, const oa::Matrix& inGenerated) {
	if (inGenerated.getShape() != oa::MatrixShape{1, kPixels}) {
		return oa::Status::invalidArgument(
			"Fashion flow image expects one flattened 28x28 sample");
	}
	auto mapped = oa::FnMatrix::clampMin(oa::FnMatrix::clampMax(
		(inGenerated * 0.5F) + 0.5F, 1.0F), 0.0F);
	oa::Image image(
		mapped.reshape(oa::MatrixShape{1, 1, kImageSize, kImageSize}),
		oa::ImageLayout::Nchw,
		oa::ImageFormat::Gray);
	return oa::FnTexture::fromImage(inEngine, image);
}

inline void validatePatchRoundTrip() {
	oa::Vec<oa::F32> input(kPixels);
	for (oa::I32 index = 0; index < kPixels; ++index) {
		input[static_cast<oa::Usize>(index)] = static_cast<oa::F32>(index);
	}
	auto matrix = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(
			reinterpret_cast<const oa::U8*>(input.data()),
			input.size() * sizeof(oa::F32)),
		oa::MatrixShape{1, kPixels}, oa::ScalarType::Float32);
	auto roundTrip = unpatchify(patchify(matrix));
	oa::Vec<oa::F32> output(kPixels);
	ASSERT_TRUE(oa::FnMatrix::copyToHost(
		roundTrip, output.data(), output.size() * sizeof(oa::F32)).isOk());
	EXPECT_EQ(output, input);
}

inline void validateImageLayout(oa::Engine& inEngine) {
	oa::Vec<oa::F32> input(kPixels);
	for (oa::I32 y = 0; y < kImageSize; ++y) {
		for (oa::I32 x = 0; x < kImageSize; ++x) {
			const bool high = ((y * 5 + x * 7) & 1) != 0;
			input[static_cast<oa::Usize>(y * kImageSize + x)] =
				high ? 1.0F : -1.0F;
		}
	}
	auto matrix = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(
			reinterpret_cast<const oa::U8*>(input.data()),
			input.size() * sizeof(oa::F32)),
		oa::MatrixShape{1, kPixels}, oa::ScalarType::Float32);
	auto texture = makeImage(inEngine, matrix);
	ASSERT_TRUE(texture.isOk()) << texture.getStatus().toString().cStr();
	ASSERT_TRUE(tutorialSubmitAndWait(inEngine).isOk());
	EXPECT_EQ(texture->width(), kImageSize);
	EXPECT_EQ(texture->height(), kImageSize);

	oa::Vec<oa::U32> pixels(kPixels);
	ASSERT_TRUE(oa::FnTexture::copyToHost(
		inEngine, *texture, pixels.data(),
		pixels.size() * sizeof(oa::U32)).isOk());
	oa::U32 mismatches = 0U;
	oa::I32 firstY = -1;
	oa::I32 firstX = -1;
	oa::U8 firstExpected = 0U;
	oa::U8 firstActual = 0U;
	for (oa::I32 y = 0; y < kImageSize; ++y) {
		for (oa::I32 x = 0; x < kImageSize; ++x) {
			const oa::I32 index = y * kImageSize + x;
			const bool high = ((y * 5 + x * 7) & 1) != 0;
			const oa::U8 expected = high ? 255U : 0U;
			const oa::U8 actual = static_cast<oa::U8>(
				pixels[static_cast<oa::Usize>(index)] & 0xFFU);
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
}

inline void validateImagePixels(
	oa::Engine& inEngine,
	const oa::Texture& inTexture) {
	const oa::I64 pixelCount = kPixels;
	oa::Vec<oa::U32> pixels(pixelCount);
	ASSERT_TRUE(oa::FnTexture::copyToHost(
		inEngine, inTexture, pixels.data(),
		pixels.size() * sizeof(oa::U32)).isOk());
	oa::U8 minimum = 255U;
	oa::U8 maximum = 0U;
	oa::F64 sum = 0.0;
	oa::F64 sumSquares = 0.0;
	for (oa::U32 pixel : pixels) {
		const oa::U8 value = static_cast<oa::U8>(pixel & 0xFFU);
		minimum = std::min(minimum, value);
		maximum = std::max(maximum, value);
		sum += value;
		sumSquares += static_cast<oa::F64>(value) * value;
	}
	const oa::F64 mean = sum / static_cast<oa::F64>(pixelCount);
	const oa::F64 variance = std::max(
		0.0, sumSquares / static_cast<oa::F64>(pixelCount) - mean * mean);
	const oa::F64 standardDeviation = std::sqrt(variance);
	std::printf("image pixels min=%u max=%u mean=%.2f stddev=%.2f\n",
		static_cast<unsigned>(minimum), static_cast<unsigned>(maximum),
		mean, standardDeviation);
	EXPECT_LT(minimum, maximum);
	EXPECT_GT(standardDeviation, 1.0);
}

inline oa::F32 validate(Model& inModel, oa::DsMnist& inValidation) {
	oa::Module::ScopedEval eval(inModel);
	oa::F64 total = 0.0;
	oa::I32 batches = 0;
	inValidation.reset(false);
	oa::Matrix images;
	oa::Matrix labels;
	while (batches < 8 && inValidation.nextBatch(images, labels)) {
		auto clean = oa::FnMatrix::scale(images, 2.0F / 255.0F) - 1.0F;
		auto time = oa::FnMatrix::full(oa::MatrixShape{kBatch}, 0.5F);
		auto noise = oa::FnMatrix::philoxNormal(
			oa::FnMatrix::empty(clean.getShape()), 0.0F, 1.0F,
			9000U + static_cast<oa::U64>(batches));
		auto flow = oa::FnFlow::linearMatch(clean, noise, time);
		auto prediction = inModel.forwardFlow(flow.state, time, labels);
		auto loss = oa::FnLoss::mse(prediction, flow.velocity);
		const auto execute = tutorialSubmitAndWait(testEngine());
		if (!execute.isOk()) throw std::runtime_error(execute.getMessage().cStr());
		total += loss.item();
		++batches;
	}
	return batches > 0 ? static_cast<oa::F32>(total / batches) : 0.0F;
}

inline void run(bool inMoe) {
	const char* dataDirectory = std::getenv("OA_MNIST_DATA");
	if (!dataDirectory) dataDirectory = "../oapy/dataset/FashionMNIST/raw";
	oa::DsMnist train(dataDirectory, "train", kBatch, true);
	oa::DsMnist validation(dataDirectory, "t10k", kBatch, false);
	if (train.numSamples() == 0 || validation.numSamples() == 0) {
		GTEST_SKIP() << "Fashion-MNIST not found (set OA_MNIST_DATA)";
	}

	const oa::I32 steps = trainSteps();
	const oa::I32 generatedClass = generatedClassFromEnvironment();
	std::printf("\nOA Fashion-MNIST flow — %s FFN\n",
		inMoe ? "dropless MoE" : "dense");
	std::printf("train=%d val=%d batch=%d steps=%d seed=2026 class=%d\n",
		train.numSamples(), validation.numSamples(), kBatch, steps,
		generatedClass);
	oa::FnMatrix::setRngSeed(2026);
	auto& engine = testEngine();
	validatePatchRoundTrip();
	validateImageLayout(engine);
	auto model = oa::makeShared<Model>(inMoe);
	auto parameters = model->allParameterPtrs();
	oa::AdamW optimizer(parameters, kLearningRate);
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
		.timerName = inMoe ? "fashion_flow_moe" : "fashion_flow_dense",
	});

	oa::Matrix images;
	oa::Matrix labels;
	oa::F32 initialLoss = 0.0F;
	while (!training.loop.isDone()) {
		if (!train.nextBatch(images, labels)) {
			train.reset();
			ASSERT_TRUE(train.nextBatch(images, labels));
		}
		auto clean = oa::FnMatrix::scale(images, 2.0F / 255.0F) - 1.0F;
		auto time = oa::FnMatrix::philoxUniform(
			oa::FnMatrix::empty(oa::MatrixShape{kBatch}), 0.0F, 1.0F, 0);
		auto noise = oa::FnMatrix::philoxNormal(
			oa::FnMatrix::empty(clean.getShape()), 0.0F, 1.0F, 0);
		auto flow = oa::FnFlow::linearMatch(clean, noise, time);
		optimizer.zeroGrad();
		oa::GradientTape tape;
		auto prediction = model->forwardFlow(flow.state, time, labels);
		auto loss = oa::FnLoss::mse(prediction, flow.velocity);
		tape.backward(loss);
		training.loop.next(loss);
		if (training.loop.lastLossStep() == 1) initialLoss = training.loop.lastLoss();
	}
	ASSERT_TRUE(training.loop.finish().isOk());
	const oa::F32 finalLoss = training.loop.lastLoss();
	const oa::F32 validationLoss = validate(*model, validation);
	std::printf("flow_mse train %.6f -> %.6f · val %.6f -> %.6f\n",
		initialLoss, finalLoss, initialValidationLoss, validationLoss);

	auto generated = sample(*model, 2026, generatedClass);
	auto textureResult = makeImage(engine, generated);
	ASSERT_TRUE(textureResult.isOk()) << textureResult.getStatus().toString().cStr();
	ASSERT_TRUE(tutorialSubmitAndWait(engine).isOk());
	EXPECT_EQ(textureResult->width(), kImageSize);
	EXPECT_EQ(textureResult->height(), kImageSize);
	validateImagePixels(engine, *textureResult);
	const oa::Path directory = oa::Paths::var() / "generative";
	ASSERT_TRUE(oa::Filesystem::createDirectories(directory).isOk());
	const oa::Path imagePath = directory /
		(inMoe ? "fashion_flow_moe.png" : "fashion_flow_dense.png");
	ASSERT_TRUE(oa::FnImage::saveTextureFile(
		engine, *textureResult, imagePath.string()).isOk());

	const oa::String checkpoint = inMoe
		? "/tmp/oa_fashion_flow_moe.oam"
		: "/tmp/oa_fashion_flow_dense.oam";
	ASSERT_TRUE(model->save(engine, checkpoint, optimizer).isOk());
	auto reloaded = oa::makeShared<Model>(inMoe);
	auto reloadedParameters = reloaded->allParameterPtrs();
	oa::AdamW reloadedOptimizer(reloadedParameters, kLearningRate);
	ASSERT_TRUE(reloaded->load(engine, checkpoint, reloadedOptimizer).isOk());
	EXPECT_EQ(reloaded->numParameters(), model->numParameters());
	EXPECT_GT(initialLoss, 0.0F);
	EXPECT_TRUE(std::isfinite(finalLoss));
	EXPECT_TRUE(std::isfinite(initialValidationLoss));
	EXPECT_TRUE(std::isfinite(validationLoss));
	if (steps >= 20) EXPECT_LT(validationLoss, initialValidationLoss);
	EXPECT_GT(validationLoss, 0.0F);
}

} // namespace tutorialFashionFlow
