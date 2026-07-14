// OA Tutorial — Pixel-Space Mamba Diffusion on Fashion-MNIST
//
// Flow-matching generative model with a Mamba-3 backbone. No VAE, no U-Net:
// diffuse directly on 28×28 = 784 pixel vectors. The model predicts the
// velocity field of an optimal-transport flow, trained with MSE.
//
// Architecture:
//   Input: xt [B, 784] + sinusoidal timestep embed [B, 16]
//   Reshape tokens → [B, 49, 16]
//   4× Mamba-3 block (pre-LayerNorm + residual + gated output norm)
//   Output: velocity [B, 784]
//
// Sampling: 20-step Euler ODE solve from pure noise.
// Output: grid of generated images saved as PNG under var/mamba_diffusion/.
//
#include "../../Test/OaTest.h"
#include "TutorialMl.h"
#include <Oa/Core/EnvFlag.h>
#include <Oa/Core/FileIo.h>
#include <Oa/Core/Log.h>
#include <Oa/Data/DsMnist.h>
#include <Oa/Ml.h>
#include <Oa/Ml/Autograd.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Ui/Image.h>
#include <Oa/Vision/FnImage.h>

#include <cmath>
#include <random>

// ─── Hyperparameters ───────────────────────────────────────────────────────

static constexpr OaI32 kImageSize  = 28;
static constexpr OaI32 kPixels     = kImageSize * kImageSize;  // 784
static constexpr OaI32 kNumPatches = 49;
static constexpr OaI32 kPatchDim   = 16;   // input/output patch dim
static constexpr OaI32 kLatentDim  = 32;   // Mamba block dim
static constexpr OaI32 kTimeDim    = 32;   // must match latent dim for broadcast
static constexpr OaI32 kNumClasses = 10;
static constexpr OaI32 kNumBlocks  = 6;
static constexpr OaI32 kDState   = 32;
static constexpr OaI32 kExpand   = 2;
static constexpr OaI32 kHeadDim    = 32;
static constexpr OaI32 kNumSteps = 20;   // Euler sampling steps
static constexpr OaI32 kRefineChannels = 2;  // ConvTranspose2d bottleneck channels
static constexpr OaI32 kBatch      = 64;
static constexpr OaI32 kTrainSteps = 10000;
static constexpr OaF32 kLr         = 0.001F;

// ─── Timestep Embedding ──────────────────────────────────────────────────────

static OaMatrix TimestepEmbedding(const OaVec<OaF32>& InT, OaI32 InDim) {
	const OaI32 half = InDim / 2;
	const OaI32 batch = static_cast<OaI32>(InT.Size());
	OaVec<OaF32> emb(static_cast<OaI64>(batch) * InDim);
	const OaF32 logMax = std::log(10000.0F);
	for (OaI32 b = 0; b < batch; ++b) {
		const OaF32 t = InT[b];
		for (OaI32 i = 0; i < half; ++i) {
			const OaF32 freq = std::exp(-logMax * static_cast<OaF32>(i) / static_cast<OaF32>(half));
			emb[(static_cast<OaI64>(b) * InDim) + i] = std::sin(t * freq);
			emb[(static_cast<OaI64>(b) * InDim) + half + i] = std::cos(t * freq);
		}
	}
	return OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(emb.Data()), static_cast<OaI64>(emb.Size()) * sizeof(OaF32)),
		OaMatrixShape{batch, InDim});
}

// ─── Model: Pixel-Space Mamba Diffusion ──────────────────────────────────────

class OaPixelMambaDiffusion : public OaModule {
public:
	OaPixelMambaDiffusion() {
		auto wd = OaFnMatrix::GetWeightDtype();

		ClassEmbed_ = OaMakeSharedPtr<OaEmbedding>(kNumClasses, kTimeDim);
		RegisterModule("class_embed", ClassEmbed_);

		PatchInProj_ = OaMakeSharedPtr<OaLinear>(kPatchDim, kLatentDim, false);
		PatchInProj_->Parameters()[0].Data = OaFnMatrix::RandGlorotUniform(OaMatrixShape{kLatentDim, kPatchDim}, wd);
		PatchInProj_->Parameters()[0].Data.SetRequiresGrad(true);
		RegisterModule("patch_in_proj", PatchInProj_);

		for (OaI32 i = 0; i < kNumBlocks; ++i) {
			auto block = OaMakeSharedPtr<OaMamba3Module>(
				kLatentDim, kDState, kExpand, kHeadDim,
				/*NGroups*/ 1, /*RopeFraction*/ 0.5F, /*Mimo*/ false, /*MimoRank*/ 4,
				/*DtMin*/ 0.001F, /*DtMax*/ 0.1F, /*DtInitFloor*/ 1e-4F, /*AFloor*/ 1e-4F,
				/*OutprojNorm*/ true);
			OaString name = OaString("mamba") + std::to_string(i);
			RegisterModule(name.c_str(), block);
			Blocks_.PushBack(block);

			auto ffn = OaMakeSharedPtr<OaLinear>(kLatentDim, kLatentDim);
			ffn->SetActivation(OaActivation::Gelu);
			ffn->Parameters()[0].Data = OaFnMatrix::RandGlorotUniform(OaMatrixShape{kLatentDim, kLatentDim}, wd);
			ffn->Parameters()[0].Data.SetRequiresGrad(true);
			OaString ffnName = OaString("ffn") + std::to_string(i);
			RegisterModule(ffnName.c_str(), ffn);
			Ffns_.PushBack(ffn);

			auto norm = OaMakeSharedPtr<OaRmsNorm>(kLatentDim);
			OaString normName = OaString("norm") + std::to_string(i);
			RegisterModule(normName.c_str(), norm);
			Norms_.PushBack(norm);
		}

		OutProj_ = OaMakeSharedPtr<OaLinear>(kLatentDim, kPatchDim, false);
		OutProj_->Parameters()[0].Data = OaFnMatrix::RandGlorotUniform(OaMatrixShape{kPatchDim, kLatentDim}, wd);
		OutProj_->Parameters()[0].Data.SetRequiresGrad(true);
		RegisterModule("out_proj", OutProj_);

		// Residual two-stage transposed-conv refinement: 16-channel 7x7 -> 2-channel 14x14 -> 28x28.
		// Starts at zero so the model initially behaves like the original linear reshape.
		OutConv1_ = OaMakeSharedPtr<OaConvTranspose2d>(kPatchDim, kRefineChannels, /*KernelSize=*/2, /*Stride=*/2, /*Padding=*/0);
		OutConv1_->Parameters()[0].Data = OaFnMatrix::Zeros(OaMatrixShape{kPatchDim, kRefineChannels, 2, 2}, wd);
		OutConv1_->Parameters()[0].Data.SetRequiresGrad(true);
		OutConv1_->Parameters()[1].Data = OaFnMatrix::Zeros(OaMatrixShape{kRefineChannels}, wd);
		OutConv1_->Parameters()[1].Data.SetRequiresGrad(true);
		RegisterModule("out_conv1", OutConv1_);

		OutConv2_ = OaMakeSharedPtr<OaConvTranspose2d>(kRefineChannels, 1, /*KernelSize=*/2, /*Stride=*/2, /*Padding=*/0);
		OutConv2_->Parameters()[0].Data = OaFnMatrix::Zeros(OaMatrixShape{kRefineChannels, 1, 2, 2}, wd);
		OutConv2_->Parameters()[0].Data.SetRequiresGrad(true);
		OutConv2_->Parameters()[1].Data = OaFnMatrix::Zeros(OaMatrixShape{1}, wd);
		OutConv2_->Parameters()[1].Data.SetRequiresGrad(true);
		RegisterModule("out_conv2", OutConv2_);
	}

	// Training interface: xt, timestep embedding, and class embedding are supplied separately.
	OaMatrix ForwardDiffusion(const OaMatrix& InXt, const OaMatrix& InTEmbed, const OaMatrix& InClassEmbed) {
		const OaI64 batch = InXt.Size(0);
		auto tokens = InXt.Reshape(OaMatrixShape{batch, kNumPatches, kPatchDim});
		auto tokens2d = tokens.Reshape(OaMatrixShape{batch * kNumPatches, kPatchDim});
		auto latent2d = PatchInProj_->Forward(tokens2d);
		auto x = latent2d.Reshape(OaMatrixShape{batch, kNumPatches, kLatentDim});

		auto cond = InTEmbed + InClassEmbed;
		auto condBroadcast = cond.Reshape(OaMatrixShape{batch, 1, kLatentDim});
		x = x + condBroadcast;

		for (OaI32 i = 0; i < kNumBlocks; ++i) {
			auto h = Norms_[i]->Forward(x);
			h = Blocks_[i]->Forward(h);  // [B, 49, 32]
			auto h2d = h.Reshape(OaMatrixShape{batch * kNumPatches, kLatentDim});
			auto ffn2d = Ffns_[i]->Forward(h2d);
			h = ffn2d.Reshape(OaMatrixShape{batch, kNumPatches, kLatentDim});
			x = x + h;
		}

		auto x2d = x.Reshape(OaMatrixShape{batch * kNumPatches, kLatentDim});
		auto patchPixels = OutProj_->Forward(x2d);          // [B*49, 16]
		auto base = patchPixels.Reshape(OaMatrixShape{batch, kPixels});

		// Residual ConvTranspose2d refinement: 16x7x7 -> 2x14x14 -> 1x28x28.
		auto patches3d = patchPixels.Reshape(OaMatrixShape{batch, kNumPatches, kPatchDim}); // [B, 49, 16]
		auto patchesNchw = OaFnMatrix::Transpose(patches3d, 1, 2)
			.Reshape(OaMatrixShape{batch, kPatchDim, 7, 7});   // [B, 16, 7, 7]
		auto refine14 = OutConv1_->Forward(patchesNchw);   // [B, 2, 14, 14]
		auto refine28 = OutConv2_->Forward(refine14);      // [B, 1, 28, 28]
		return base + refine28.Reshape(OaMatrixShape{batch, kPixels});
	}

	// OaModule interface: concatenated [xt, t_embed, class_embed] along feature dim.
	OaMatrix Forward(const OaMatrix& InInput) override {
		OaI64 sizes[3] = {kPixels, kTimeDim, kTimeDim};
		auto parts = OaFnMatrix::Split(InInput, OaSpan<OaI64>(sizes, 3), 1);
		return ForwardDiffusion(parts[0], parts[1], parts[2]);
	}

	[[nodiscard]] OaSharedPtr<OaEmbedding> ClassEmbed() const { return ClassEmbed_; }

private:
	OaSharedPtr<OaEmbedding> ClassEmbed_;
	OaSharedPtr<OaLinear> PatchInProj_;
	OaVec<OaSharedPtr<OaMamba3Module>> Blocks_;
	OaVec<OaSharedPtr<OaLinear>> Ffns_;
	OaVec<OaSharedPtr<OaRmsNorm>> Norms_;
	OaSharedPtr<OaLinear> OutProj_;
	OaSharedPtr<OaConvTranspose2d> OutConv1_;
	OaSharedPtr<OaConvTranspose2d> OutConv2_;
};

// ─── Image Save Helper ───────────────────────────────────────────────────────

static OaStatus SaveImage(OaComputeEngine& InRt,
                          const OaMatrix& InPixels,
                          const OaPath& InPath,
                          OaI32 InW, OaI32 InH) {
	// InPixels: [1, pixels] in [-1, 1] flow-matching space. Map to [0, 1].
	OaVec<OaF32> host(static_cast<OaI64>(InW) * InH);
	OaMatrix mapped = OaFnMatrix::ClampMin(OaFnMatrix::ClampMax((InPixels * 0.5F) + 0.5F, 1.0F), 0.0F);
	OA_RETURN_IF_ERROR(OaFnMatrix::CopyToHost(mapped, host.Data(), static_cast<OaI64>(host.Size()) * sizeof(OaF32)));

	OaVec<OaU8> rgba(static_cast<OaI64>(InW) * InH * 4);
	for (OaI64 i = 0; i < static_cast<OaI64>(InW) * InH; ++i) {
		OaU8 g = static_cast<OaU8>(host[i] * 255.0F);
		rgba[(static_cast<OaI64>(i) * 4) + 0] = g;
		rgba[(static_cast<OaI64>(i) * 4) + 1] = g;
		rgba[(static_cast<OaI64>(i) * 4) + 2] = g;
		rgba[(static_cast<OaI64>(i) * 4) + 3] = 255U;
	}

	auto r = OaTexture::FromPixels(InRt,
		OaSpan<const OaU8>(rgba.Data(), rgba.Size()), InW, InH);
	if (!r.IsOk()) {
		return OaStatus::Error("OaTexture::FromPixels failed");
	}
	return OaFnImage::SaveFile(InRt, *r, InPath.String().c_str());
}

// ─── Sampling ────────────────────────────────────────────────────────────────

static OaMatrix SampleImages(OaPixelMambaDiffusion& InModel,
                             OaComputeEngine& InRt,
                             const OaVec<OaU8>& InLabels,
                             OaU64 InSeed) {
	(void)InRt;
	const OaI32 count = static_cast<OaI32>(InLabels.Size());
	OaFnMatrix::SetRngSeed(InSeed);
	auto x = OaFnMatrix::PhiloxNormal(OaFnMatrix::Zeros(OaMatrixShape{count, kPixels}), 0.0F, 1.0F, InSeed);
	const OaF32 dt = 1.0F / static_cast<OaF32>(kNumSteps);

	auto labelMat = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(InLabels.Data(), static_cast<OaI64>(InLabels.Size())),
		OaMatrixShape{count}, OaScalarType::UInt8);
	auto classEmbed = InModel.ClassEmbed()->Forward(labelMat);

	for (OaI32 step = kNumSteps; step >= 1; --step) {
		const OaF32 tVal = static_cast<OaF32>(step) * dt;
		OaVec<OaF32> tVals(static_cast<OaI64>(count), tVal);
		auto tEmbed = TimestepEmbedding(tVals, kTimeDim);
		auto v = InModel.ForwardDiffusion(x, tEmbed, classEmbed);
		x = x - (v * dt);
	}
	return x;
}

// ─── Tutorial ────────────────────────────────────────────────────────────────

TEST(TutorialMambaDiffusionPixel, FashionMnistFlowMatching) {
	const char* dataDir = std::getenv("OA_MNIST_DATA");
	if (dataDir == nullptr) {
		dataDir = "../oapy/dataset/FashionMNIST/raw";
	}

	OaDsMnist trainLoader(OaString(dataDir), "train", kBatch, /*InShuffle=*/true);
	if (trainLoader.NumSamples() == 0) {
		printf("Fashion-MNIST not found at: %s (set OA_MNIST_DATA).\n", dataDir);
		GTEST_SKIP() << "Dataset not found";
	}

	printf("\n╔══════════════════════════════════════════════════════════════════╗\n");
	printf("║  OA Tutorial — Pixel-Space Mamba Diffusion                       ║\n");
	printf("║  Flow Matching · Fashion-MNIST · Mamba-3 Backbone                ║\n");
	printf("╚══════════════════════════════════════════════════════════════════╝\n\n");
	printf("Dataset: %d train images, 28×28 grayscale\n", trainLoader.NumSamples());
	printf("Model: %d patches × %d dims, %d Mamba-3 blocks, d_state=%d\n\n",
		kNumPatches, kPatchDim, kNumBlocks, kDState);

	OaFnMatrix::SetRngSeed(2026);

	auto model  = OaMakeSharedPtr<OaPixelMambaDiffusion>();
	auto params = model->AllParameterPtrs();
	auto opt    = OaMakeUniquePtr<OaAdamW>(params, kLr);

	printf("Params: %lld    Optimizer: AdamW(lr=%g)    Loss: MSE\n\n",
		static_cast<long long>(model->NumParameters()), static_cast<double>(kLr));

	TutorialTrainingLoop training(*opt, OaItTrainingConfig{
		.TotalSteps     = kTrainSteps,
		.EpochSteps     = {},
		.BatchSize      = kBatch,
		.TimerName      = "mamba_diffusion_step",
		.Callbacks      = {},
	});

	auto& ctx = OaContext::GetDefault();
	auto& rt  = *OaComputeEngine::GetGlobal();

	OaVec<OaMatrix> xRing(static_cast<OaI64>(ctx.MaxAsyncSubmissions()));
	OaVec<OaMatrix> yRing(static_cast<OaI64>(ctx.MaxAsyncSubmissions()));
	OaF32 initialLoss = 0.0F;
	OaF32 lastLoss    = 0.0F;

	std::mt19937 rng(2026);
	std::uniform_real_distribution<OaF32> tDist(0.0F, 1.0F);

	while (not training.Loop.IsDone()) {
		const OaI64 step = training.Loop.Index();
		OaMatrix& batchX = xRing[(step - 1) % xRing.Size()];
		OaMatrix& batchY = yRing[(step - 1) % yRing.Size()];
		if (!trainLoader.NextBatch(batchX, batchY)) {
			trainLoader.Reset();
			trainLoader.NextBatch(batchX, batchY);
		}

		// Normalize images to [-1, 1]
		auto x0 = (OaFnMatrix::Scale(batchX, 2.0F / 255.0F)) - 1.0F;

		// Sample timestep and noise
		OaVec<OaF32> tVals(kBatch);
		for (OaI32 i = 0; i < kBatch; ++i) {
			tVals[i] = tDist(rng);
		}
		auto tEmbed = TimestepEmbedding(tVals, kTimeDim);
		auto classEmbed = model->ClassEmbed()->Forward(batchY);
		auto noise  = OaFnMatrix::PhiloxNormal(OaFnMatrix::Zeros(x0.GetShape()), 0.0F, 1.0F, /*InSeed=*/0);
		auto xt = x0 + ((noise - x0) * OaFnMatrix::FromBytes(
			OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(tVals.Data()), static_cast<OaI64>(tVals.Size()) * sizeof(OaF32)),
			OaMatrixShape{kBatch, 1}));
		auto target = noise - x0;

		opt->ZeroGrad();
		OaGradientTape tape;
		auto pred = model->ForwardDiffusion(xt, tEmbed, classEmbed);
		auto loss = OaFnLoss::Mse(pred, target);
		tape.Backward(loss);
		training.Loop.Next(loss);

		if (step == 1) {
			initialLoss = training.Loop.LiveLoss();
		}
	}
	ASSERT_TRUE(training.Loop.Finish().IsOk());
	lastLoss = training.Loop.LastLoss();

	printf("\n─── Results ───\n");
	printf("Initial loss: %.4f -> Final loss: %.4f\n", initialLoss, lastLoss);

	// Generate and save images — one per Fashion-MNIST class
	static const char* kClassNames[] = {
		"T-shirt", "Trouser", "Pullover", "Dress", "Coat",
		"Sandal", "Shirt", "Sneaker", "Bag", "Boot"
	};
	OaPath outDir = OaFileIo::GetVarDir() / "mamba_diffusion";
	ASSERT_TRUE(OaFileIo::CreateDirectories(outDir).IsOk());
	printf("\nSampling %d class-conditional images (%d Euler steps) ...\n", kNumClasses, kNumSteps);
	OaVec<OaU8> labels(kNumClasses);
	for (OaI32 i = 0; i < kNumClasses; ++i) {
		labels[i] = static_cast<OaU8>(i);
	}
	auto generated = SampleImages(*model, rt, labels, /*InSeed=*/2026);
	(void)ctx.Execute();
	(void)ctx.Sync();

	for (OaI32 i = 0; i < kNumClasses; ++i) {
		auto img = OaFnMatrix::Slice(generated, 0, i, i + 1).Reshape(OaMatrixShape{kPixels});
		OaPath path = outDir / OaPath("class_" + std::to_string(i) + "_" + kClassNames[i] + ".png");
		auto status = SaveImage(rt, img, path, kImageSize, kImageSize);
		if (status.IsOk()) {
			printf("  Saved %s\n", path.String().c_str());
		} else {
			printf("  Failed to save %s: %s\n", path.String().c_str(), status.GetMessage().c_str());
		}
	}
	printf("\n");

	ASSERT_GT(initialLoss, 0.0F);
	EXPECT_LT(lastLoss, initialLoss) << "MSE must decrease during training";
	EXPECT_LT(lastLoss, 1.0F) << "Final loss should be below 1.0";

	// Checkpoint round-trip
	const OaString ckptPath = "/tmp/mamba_diffusion_pixel.oam";
	ASSERT_TRUE(model->Save(ckptPath, *opt).IsOk());
	auto reloaded = OaMakeSharedPtr<OaPixelMambaDiffusion>();
	auto reloadParams = reloaded->AllParameterPtrs();
	auto reloadedOpt = OaMakeUniquePtr<OaAdamW>(reloadParams, kLr);
	ASSERT_TRUE(reloaded->Load(ckptPath, *reloadedOpt).IsOk());
	EXPECT_EQ(reloaded->NumParameters(), model->NumParameters());
}
