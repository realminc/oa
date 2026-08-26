// OA Tutorial — Pixel-Space Mamba Diffusion on Fashion-MNIST
//
// Flow-matching generative model with a Mamba-3 backbone. No VAE, no U-Net:
// diffuse directly on 28×28 = 784 pixel vectors. The model predicts the
// velocity field of an optimal-transport flow, trained with MSE.
//
// architecture:
//   input: xt [B, 784] + GPU sinusoidal timestep embed [B, 32]
//   reshape tokens → [B, 49, 16]
//   4× Mamba-3 block (pre-LayerNorm + residual + gated output norm)
//   output: velocity [B, 784]
//
// Sampling: 20-step Euler ODE solve from pure noise.
// output: grid of generated images saved as PNG under var/mamba_diffusion/.
//
#include "oaTest.h"
#include "tutorialMl.h"
#include <oa/core/envFlag.h>
#include <oa/core/filesystem.h>
#include <oa/core/log.h>
#include <oa/core/paths.h>
#include <data/dsMnist.h>
#include <oa/ml.h>
#include <oa/ml/autograd.h>
#include <oa/runtime/engine.h>
#include <oa/ui/image.h>
#include <oa/vision/fnImage.h>

// ─── Hyperparameters ───────────────────────────────────────────────────────

static constexpr oa::I32 kImageSize  = 28;
static constexpr oa::I32 kPixels     = kImageSize * kImageSize;  // 784
static constexpr oa::I32 kNumPatches = 49;
static constexpr oa::I32 kPatchDim   = 16;   // input/output patch dim
static constexpr oa::I32 kLatentDim  = 32;   // Mamba block dim
static constexpr oa::I32 kTimeDim    = 32;   // must match latent dim for broadcast
static constexpr oa::I32 kNumClasses = 10;
static constexpr oa::I32 kNumBlocks  = 6;
static constexpr oa::I32 kDState   = 32;
static constexpr oa::I32 kExpand   = 2;
static constexpr oa::I32 kHeadDim    = 32;
static constexpr oa::I32 kNumSteps = 20;   // Euler sampling steps
static constexpr oa::I32 kRefineChannels = 2;  // ConvTranspose2d bottleneck channels
static constexpr oa::I32 kBatch      = 64;
static constexpr oa::I32 kTrainSteps = 10000;
static constexpr oa::F32 kLr         = 0.001F;

// ─── Model: Pixel-Space Mamba Diffusion ──────────────────────────────────────

class PixelMambaDiffusion : public oa::Module {
public:
	PixelMambaDiffusion() {
		auto wd = oa::FnMatrix::weightDtype();

		classEmbed_ = oa::makeShared<oa::Embedding>(kNumClasses, kTimeDim);
		registerModule("class_embed", classEmbed_);

		patchInProj_ = oa::makeShared<oa::Linear>(kPatchDim, kLatentDim, false);
		patchInProj_->parameters()[0].data = oa::FnMatrix::randGlorotUniform(oa::MatrixShape{kLatentDim, kPatchDim}, wd);
		patchInProj_->parameters()[0].data.setRequiresGrad(true);
		registerModule("patch_in_proj", patchInProj_);

		for (oa::I32 i = 0; i < kNumBlocks; ++i) {
			auto block = oa::makeShared<oa::Mamba3Module>(
				kLatentDim, kDState, kExpand, kHeadDim,
				/*nGroups*/ 1, /*RopeFraction*/ 0.5F, /*Mimo*/ false, /*mimoRank*/ 4,
				/*dtMin*/ 0.001F, /*dtMax*/ 0.1F, /*DtInitFloor*/ 1e-4F, /*aFloor*/ 1e-4F,
				/*OutprojNorm*/ true);
			const std::string indexText = std::to_string(i);
			oa::String name = oa::String("mamba")
				+ oa::String(indexText.data(), indexText.size());
			registerModule(name.cStr(), block);
			blocks_.pushBack(block);

			auto ffn = oa::makeShared<oa::Linear>(kLatentDim, kLatentDim);
			ffn->setActivation(oa::Activation::Gelu);
			ffn->parameters()[0].data = oa::FnMatrix::randGlorotUniform(oa::MatrixShape{kLatentDim, kLatentDim}, wd);
			ffn->parameters()[0].data.setRequiresGrad(true);
			oa::String ffnName = oa::String("ffn")
				+ oa::String(indexText.data(), indexText.size());
			registerModule(ffnName.cStr(), ffn);
			ffns_.pushBack(ffn);

			auto norm = oa::makeShared<oa::RmsNorm>(kLatentDim);
			oa::String normName = oa::String("norm")
				+ oa::String(indexText.data(), indexText.size());
			registerModule(normName.cStr(), norm);
			norms_.pushBack(norm);
		}

		outProj_ = oa::makeShared<oa::Linear>(kLatentDim, kPatchDim, false);
		outProj_->parameters()[0].data = oa::FnMatrix::randGlorotUniform(oa::MatrixShape{kPatchDim, kLatentDim}, wd);
		outProj_->parameters()[0].data.setRequiresGrad(true);
		registerModule("out_proj", outProj_);

		// residual two-stage transposed-conv refinement: 16-channel 7x7 -> 2-channel 14x14 -> 28x28.
		// Starts at zero so the model initially behaves like the original linear reshape.
		outConv1_ = oa::makeShared<oa::ConvTranspose2d>(kPatchDim, kRefineChannels, /*KernelSize=*/2, /*Stride=*/2, /*Padding=*/0);
		outConv1_->parameters()[0].data = oa::FnMatrix::zeros(oa::MatrixShape{kPatchDim, kRefineChannels, 2, 2}, wd);
		outConv1_->parameters()[0].data.setRequiresGrad(true);
		outConv1_->parameters()[1].data = oa::FnMatrix::zeros(oa::MatrixShape{kRefineChannels}, wd);
		outConv1_->parameters()[1].data.setRequiresGrad(true);
		registerModule("out_conv1", outConv1_);

		outConv2_ = oa::makeShared<oa::ConvTranspose2d>(kRefineChannels, 1, /*KernelSize=*/2, /*Stride=*/2, /*Padding=*/0);
		outConv2_->parameters()[0].data = oa::FnMatrix::zeros(oa::MatrixShape{kRefineChannels, 1, 2, 2}, wd);
		outConv2_->parameters()[0].data.setRequiresGrad(true);
		outConv2_->parameters()[1].data = oa::FnMatrix::zeros(oa::MatrixShape{1}, wd);
		outConv2_->parameters()[1].data.setRequiresGrad(true);
		registerModule("out_conv2", outConv2_);
	}

	// training interface: xt, timestep embedding, and class embedding are supplied separately.
	oa::Matrix forwardDiffusion(const oa::Matrix& inXt, const oa::Matrix& inTEmbed, const oa::Matrix& inClassEmbed) {
		const oa::I64 batch = inXt.size(0);
		auto tokens = inXt.reshape(oa::MatrixShape{batch, kNumPatches, kPatchDim});
		auto tokens2d = tokens.reshape(oa::MatrixShape{batch * kNumPatches, kPatchDim});
		auto latent2d = patchInProj_->forward(tokens2d);
		auto x = latent2d.reshape(oa::MatrixShape{batch, kNumPatches, kLatentDim});

		auto cond = inTEmbed + inClassEmbed;
		auto condBroadcast = cond.reshape(oa::MatrixShape{batch, 1, kLatentDim});
		x = x + condBroadcast;

		for (oa::I32 i = 0; i < kNumBlocks; ++i) {
			auto h = norms_[i]->forward(x);
			h = blocks_[i]->forward(h);  // [B, 49, 32]
			auto h2d = h.reshape(oa::MatrixShape{batch * kNumPatches, kLatentDim});
			auto ffn2d = ffns_[i]->forward(h2d);
			h = ffn2d.reshape(oa::MatrixShape{batch, kNumPatches, kLatentDim});
			x = x + h;
		}

		auto x2d = x.reshape(oa::MatrixShape{batch * kNumPatches, kLatentDim});
		auto patchPixels = outProj_->forward(x2d);          // [B*49, 16]
		auto base = patchPixels.reshape(oa::MatrixShape{batch, kPixels});

		// residual ConvTranspose2d refinement: 16x7x7 -> 2x14x14 -> 1x28x28.
		auto patches3d = patchPixels.reshape(oa::MatrixShape{batch, kNumPatches, kPatchDim}); // [B, 49, 16]
		auto patchesNchw = oa::FnMatrix::transpose(patches3d, 1, 2)
			.reshape(oa::MatrixShape{batch, kPatchDim, 7, 7});   // [B, 16, 7, 7]
		auto refine14 = outConv1_->forward(patchesNchw);   // [B, 2, 14, 14]
		auto refine28 = outConv2_->forward(refine14);      // [B, 1, 28, 28]
		return base + refine28.reshape(oa::MatrixShape{batch, kPixels});
	}

	// oa::Module interface: concatenated [xt, t_embed, class_embed] along feature dim.
	oa::Matrix forward(const oa::Matrix& inInput) override {
		oa::I64 sizes[3] = {kPixels, kTimeDim, kTimeDim};
		auto parts = oa::FnMatrix::split(inInput, oa::Span<oa::I64>(sizes, 3), 1);
		return forwardDiffusion(parts[0], parts[1], parts[2]);
	}

	[[nodiscard]] oa::SharedPtr<oa::Embedding> classEmbed() const { return classEmbed_; }

private:
	oa::SharedPtr<oa::Embedding> classEmbed_;
	oa::SharedPtr<oa::Linear> patchInProj_;
	oa::Vec<oa::SharedPtr<oa::Mamba3Module>> blocks_;
	oa::Vec<oa::SharedPtr<oa::Linear>> ffns_;
	oa::Vec<oa::SharedPtr<oa::RmsNorm>> norms_;
	oa::SharedPtr<oa::Linear> outProj_;
	oa::SharedPtr<oa::ConvTranspose2d> outConv1_;
	oa::SharedPtr<oa::ConvTranspose2d> outConv2_;
};

// ─── Image save Helper ───────────────────────────────────────────────────────

static oa::Status saveImage(oa::Engine& inRt,
                          const oa::Matrix& inPixels,
                          const oa::Path& inPath,
                          oa::I32 inW, oa::I32 inH) {
	// inPixels: [1, pixels] in [-1, 1] flow-matching space. map to [0, 1].
	oa::Vec<oa::F32> host(static_cast<oa::I64>(inW) * inH);
	oa::Matrix mapped = oa::FnMatrix::clampMin(oa::FnMatrix::clampMax((inPixels * 0.5F) + 0.5F, 1.0F), 0.0F);
	OA_RETURN_IF_ERROR(oa::FnMatrix::copyToHost(mapped, host.data(), static_cast<oa::I64>(host.size()) * sizeof(oa::F32)));

	oa::Vec<oa::U8> rgba(static_cast<oa::I64>(inW) * inH * 4);
	for (oa::I64 i = 0; i < static_cast<oa::I64>(inW) * inH; ++i) {
		oa::U8 g = static_cast<oa::U8>(host[i] * 255.0F);
		rgba[(static_cast<oa::I64>(i) * 4) + 0] = g;
		rgba[(static_cast<oa::I64>(i) * 4) + 1] = g;
		rgba[(static_cast<oa::I64>(i) * 4) + 2] = g;
		rgba[(static_cast<oa::I64>(i) * 4) + 3] = 255U;
	}

	auto r = oa::FnTexture::fromPixels(inRt,
		oa::Span<const oa::U8>(rgba.data(), rgba.size()), inW, inH);
	if (!r.isOk()) {
		return oa::Status::error("oa::FnTexture::fromPixels failed");
	}
	return oa::FnImage::saveTextureFile(inRt, *r, inPath.string().cStr());
}

// ─── Sampling ────────────────────────────────────────────────────────────────

static oa::Matrix sampleImages(PixelMambaDiffusion& inModel,
                             oa::FlowTimeEmbedding& inTimeEmbedding,
                             oa::Engine& inRt,
                             const oa::Vec<oa::U8>& inLabels,
                             oa::U64 inSeed) {
	(void)inRt;
	const oa::I32 count = static_cast<oa::I32>(inLabels.size());
	oa::FnMatrix::setRngSeed(inSeed);
	auto x = oa::FnMatrix::philoxNormal(
		oa::FnMatrix::empty(oa::MatrixShape{count, kPixels}), 0.0F, 1.0F, inSeed);
	const oa::F32 dt = 1.0F / static_cast<oa::F32>(kNumSteps);

	auto labelMat = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(inLabels.data(), static_cast<oa::I64>(inLabels.size())),
		oa::MatrixShape{count}, oa::ScalarType::UInt8);
	auto classEmbed = inModel.classEmbed()->forward(labelMat);

	for (oa::I32 step = kNumSteps; step >= 1; --step) {
		const oa::F32 tVal = static_cast<oa::F32>(step) * dt;
		auto time = oa::FnMatrix::full(oa::MatrixShape{count, 1}, tVal);
		auto tEmbed = inTimeEmbedding.forward(time);
		auto v = inModel.forwardDiffusion(x, tEmbed, classEmbed);
		x = oa::FnFlow::eulerStep(x, v, -dt);
	}
	return x;
}

// ─── Tutorial ────────────────────────────────────────────────────────────────

TEST(TutorialMambaDiffusionPixel, FashionMnistFlowMatching) {
	const oa::String dataDir = oa::Paths::data("fashionMnist").string();

	oa::DsMnist trainLoader(dataDir, "train", kBatch, /*inShuffle=*/true);
	if (trainLoader.numSamples() == 0) {
		printf("Fashion-MNIST not found at: %s (run tools/data/manage.py fetch fashionMnist).\n",
			dataDir.cStr());
		GTEST_SKIP() << "Dataset not found";
	}

	printf("\n╔══════════════════════════════════════════════════════════════════╗\n");
	printf("║  OA Tutorial — Pixel-Space Mamba Diffusion                       ║\n");
	printf("║  Flow Matching · Fashion-MNIST · Mamba-3 backbone                ║\n");
	printf("╚══════════════════════════════════════════════════════════════════╝\n\n");
	printf("Dataset: %d train images, 28×28 grayscale\n", trainLoader.numSamples());
	printf("Model: %d patches × %d dims, %d Mamba-3 blocks, d_state=%d\n\n",
		kNumPatches, kPatchDim, kNumBlocks, kDState);

	oa::FnMatrix::setRngSeed(2026);

	auto model  = oa::makeShared<PixelMambaDiffusion>();
	// Preserve the tutorial's original normalized-time frequency convention.
	oa::FlowTimeEmbedding timeEmbedding(kTimeDim, 10000.0F, 1.0F);
	auto params = model->allParameterPtrs();
	auto opt    = oa::makeUnique<oa::AdamW>(params, kLr);

	printf("params: %lld    Optimizer: AdamW(lr=%g)    loss: MSE\n\n",
		static_cast<long long>(model->numParameters()), static_cast<double>(kLr));

	TutorialTrainingLoop training(testEngine(), *opt, oa::ItTrainingConfig{
		.totalSteps     = kTrainSteps,
		.epochSteps     = {},
		.batchSize      = kBatch,
		.timerName      = "mamba_diffusion_step",
		.callbacks      = {},
	});

	auto& rt  = testEngine();

	oa::Matrix batchX;
	oa::Matrix batchY;
	oa::F32 initialLoss = 0.0F;
	oa::F32 lastLoss    = 0.0F;

	while (not training.loop.isDone()) {
		const oa::I64 step = training.loop.index();
		if (!trainLoader.nextBatch(batchX, batchY)) {
			trainLoader.reset();
			trainLoader.nextBatch(batchX, batchY);
		}

		// normalize images to [-1, 1]
		auto x0 = (oa::FnMatrix::scale(batchX, 2.0F / 255.0F)) - 1.0F;

		// Sample timestep and noise on-device, then construct the common
		// linear flow-matching objective without a tensor-sized host loop.
		auto time = oa::FnMatrix::philoxUniform(
			oa::FnMatrix::empty(oa::MatrixShape{kBatch, 1}), 0.0F, 1.0F, 0);
		auto tEmbed = timeEmbedding.forward(time);
		auto classEmbed = model->classEmbed()->forward(batchY);
		auto noise = oa::FnMatrix::philoxNormal(
			oa::FnMatrix::empty(x0.getShape()), 0.0F, 1.0F, 0);
		auto flow = oa::FnFlow::linearMatch(x0, noise, time);

		opt->zeroGrad();
		oa::GradientTape tape;
		auto pred = model->forwardDiffusion(flow.state, tEmbed, classEmbed);
		auto loss = oa::FnLoss::mse(pred, flow.velocity);
		tape.backward(loss);
		training.loop.next(loss);

		if (step == 1) {
			initialLoss = training.loop.lastLoss();
		}
	}
	ASSERT_TRUE(training.loop.finish().isOk());
	lastLoss = training.loop.lastLoss();

	printf("\n─── results ───\n");
	printf("Initial loss: %.4f -> Final loss: %.4f\n", initialLoss, lastLoss);

	// generate and save images — one per Fashion-MNIST class
	static const char* kClassNames[] = {
		"T-shirt", "Trouser", "Pullover", "Dress", "Coat",
		"Sandal", "Shirt", "Sneaker", "Bag", "Boot"
	};
	oa::Path outDir = oa::Paths::var() / "mamba_diffusion";
	ASSERT_TRUE(oa::Filesystem::createDirectories(outDir).isOk());
	printf("\nSampling %d class-conditional images (%d Euler steps) ...\n", kNumClasses, kNumSteps);
	oa::Vec<oa::U8> labels(kNumClasses);
	for (oa::I32 i = 0; i < kNumClasses; ++i) {
		labels[i] = static_cast<oa::U8>(i);
	}
	auto generated = sampleImages(*model, timeEmbedding, rt, labels, /*inSeed=*/2026);
	ASSERT_TRUE(tutorialSubmitAndWait(rt).isOk());

	for (oa::I32 i = 0; i < kNumClasses; ++i) {
		auto img = oa::FnMatrix::slice(generated, 0, i, i + 1).reshape(oa::MatrixShape{kPixels});
		oa::String filename = oa::String("class_")
			+ oa::toString(static_cast<oa::U32>(i))
			+ "_" + kClassNames[i] + ".png";
		oa::Path path = outDir / oa::StringView(filename);
		auto status = saveImage(rt, img, path, kImageSize, kImageSize);
		if (status.isOk()) {
			printf("  Saved %s\n", path.string().cStr());
		} else {
			printf("  Failed to save %s: %s\n", path.string().cStr(), status.getMessage().cStr());
		}
	}
	printf("\n");

	ASSERT_GT(initialLoss, 0.0F);
	EXPECT_LT(lastLoss, initialLoss) << "MSE must decrease during training";
	EXPECT_LT(lastLoss, 1.0F) << "Final loss should be below 1.0";

	// checkpoint round-trip
	const oa::String ckptPath = "/tmp/mamba_diffusion_pixel.oam";
	ASSERT_TRUE(model->save(rt, ckptPath, *opt).isOk());
	auto reloaded = oa::makeShared<PixelMambaDiffusion>();
	auto reloadParams = reloaded->allParameterPtrs();
	auto reloadedOpt = oa::makeUnique<oa::AdamW>(reloadParams, kLr);
	ASSERT_TRUE(reloaded->load(rt, ckptPath, *reloadedOpt).isOk());
	EXPECT_EQ(reloaded->numParameters(), model->numParameters());
}
