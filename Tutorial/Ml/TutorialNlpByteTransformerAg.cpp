// ═══════════════════════════════════════════════════════════════════════════
// OA Tutorial — Transformer Block (Autograd)
// Module API: OaEmbedding + OaTransformerBlock + OaLayerNorm + OaLinear
//              + OaAdamW + OaGradientTape
// ═══════════════════════════════════════════════════════════════════════════
//
// Pre-norm transformer block with causal self-attention, built from the
// library OaTransformerBlock module. Architecture:
//   TokEmbed + PosEmbed → OaTransformerBlock → LayerNorm → Linear head
//
// All layers use implicit autograd (OaGradientTape).  No manual backward.
//
// ═══════════════════════════════════════════════════════════════════════════

#include "../../Test/OaTest.h"
#include "TutorialMl.h"
#include "TutorialNlpCommon.h"
#include <Oa/Ml.h>
#include <Oa/Ml/Byte.h>
#include <Oa/Ml/Autograd.h>
#include <Oa/Core/EnvFlag.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/GpuTimer.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define NVTX_RANGE_PUSH(name) nvtxRangePushA(name)
#define NVTX_RANGE_POP() nvtxRangePop()
#else
#define NVTX_RANGE_PUSH(name) ((void)0)
#define NVTX_RANGE_POP() ((void)0)
#endif

static constexpr OaI32 kVocabSize = OA_BYTE_VOCAB_SIZE;  // 256 — byte vocab family
// kContextLen / kDModel / kHiddenDim come from TutorialNlpCommon.h (shared suite dims).

// ─── Transformer LM: Embed + OaTransformerBlock + Head ─────────────────────
//
// Thin wrapper around the library OaTransformerBlock module. Adds token +
// positional embeddings, a final LayerNorm (GPT-style ln_f), and a linear
// output head. The block-diagonal causal mask and attention machinery live
// inside OaTransformerBlock.

class OaByteTransformerLM : public OaModule {
public:
	OaByteTransformerLM() {
		TokEmbed_ = OaMakeSharedPtr<OaEmbedding>(kVocabSize, kDModel);
		RegisterModule("tok_embed", TokEmbed_);
		PosEmbed_ = OaMakeSharedPtr<OaEmbedding>(kContextLen, kDModel);
		RegisterModule("pos_embed", PosEmbed_);

		Block_ = OaMakeSharedPtr<OaTransformerBlock>(kDModel, kHiddenDim, kContextLen);
		RegisterModule("block", Block_);

		LnFinal_ = OaMakeSharedPtr<OaLayerNorm>(kDModel, 1e-5f);
		RegisterModule("ln_final", LnFinal_);

		Head_ = OaMakeSharedPtr<OaLinear>(kDModel, kVocabSize);
		Head_->Parameters()[0].Data = OaFnMatrix::Rand(OaMatrixShape{kVocabSize, kDModel},
			OaFnMatrix::GetWeightDtype());
		RegisterModule("head", Head_);

		for (auto& p : AllParameterPtrs()) {
			p->Data.SetRequiresGrad(true);
		}
	}

	OaMatrix Forward(const OaMatrix& InTokens) override {
		const OaI32 b = static_cast<OaI32>(InTokens.Size(0));
		const OaI32 n = b * kContextLen;

		auto tokEmb = OaFnMatrix::Reshape(TokEmbed_->Forward(InTokens), OaMatrixShape{n, kDModel});
		auto posEmb = PosEmbed_->Forward(PositionIds(b));
		auto x = OaFnMatrix::Add(tokEmb, posEmb);

		x = Block_->Forward(x);

		return Head_->Forward(LnFinal_->Forward(x));
	}

private:
	OaMatrix PositionIds(OaI32 InBatch) const {
		const OaI32 n = InBatch * kContextLen;
		OaVec<OaU8> ids(n);
		for (OaI32 i = 0; i < n; ++i) ids[i] = static_cast<OaU8>(i % kContextLen);
		return OaFnMatrix::FromBytes(OaSpan<const OaU8>(ids.Data(), ids.Size()),
			OaMatrixShape{n}, OaScalarType::UInt8);
	}

	OaSharedPtr<OaEmbedding>        TokEmbed_;
	OaSharedPtr<OaEmbedding>        PosEmbed_;
	OaSharedPtr<OaTransformerBlock> Block_;
	OaSharedPtr<OaLayerNorm>        LnFinal_;
	OaSharedPtr<OaLinear>           Head_;
};

// ─── MatMul autograd: the primitive that makes attention trainable ──────────
//
// Attention needs activation×activation matmuls (Q@Kᵀ, attn@V). This verifies
// OaGradMatMulNt / OaGradTranspose flow gradients into both operands — without
// these, the score/context matmuls are gradient sinks and attention never
// learns (the original failure mode of this tutorial).

TEST(TutorialNlpByteTransformerAg, MatMulBackpropFlowsToBothOperands) {
	auto& ctx = OaContext::GetDefault();

	// Two learnable [4, 3] activations; CE loss on rows of C = A @ Bᵀ.
	auto A = OaFnMatrix::RandN(OaMatrixShape{4, 3}, OaScalarType::Float32);
	auto B = OaFnMatrix::RandN(OaMatrixShape{4, 3}, OaScalarType::Float32);
	A.SetRequiresGrad(true);
	B.SetRequiresGrad(true);

	const OaU8 targetIds[4] = {0, 1, 2, 3};
	auto targets = OaFnMatrix::FromBytes(OaSpan<const OaU8>(targetIds, 4),
		OaMatrixShape{4}, OaScalarType::UInt8);

	OaGradientTape tape;
	auto C    = OaFnMatrix::MatMulNt(A, B);     // [4, 4] = A @ Bᵀ
	auto loss = OaFnLoss::CrossEntropy(C, targets);
	tape.Backward(loss);
	(void)ctx.Execute();
	(void)ctx.Sync();

	auto gradMag = [](const OaMatrix& g) {
		std::vector<float> h(static_cast<size_t>(g.NumElements()));
		(void)OaFnMatrix::CopyToHost(g, h.data(), h.size() * sizeof(float));
		float s = 0.0F;
		for (float v : h) s += std::abs(v);
		return s;
	};
	EXPECT_GT(gradMag(A.GradMatrix()), 0.0F) << "MatMul did not backprop into A";
	EXPECT_GT(gradMag(B.GradMatrix()), 0.0F) << "MatMul did not backprop into B (Bᵀ path)";
}

// ─── Tutorial ──────────────────────────────────────────────────────────────

TEST(TutorialNlpByteTransformerAg, TransformerByteNextToken) {
	printf("\n╔══════════════════════════════════════════════════════════════════╗\n");
	printf("║  OA Tutorial — Transformer Block (Byte Autograd)                ║\n");
	printf("╚══════════════════════════════════════════════════════════════════╝\n\n");
	printf("Vocab: 256 bytes · Context: %d · DModel: %d · Attention: causal self-attn (1 head)\n", kContextLen, kDModel);
	printf("Task: all-position next-byte (dense windows, matched to ByteRnnAg corpus)\n\n");

	OaFnMatrix::SetRngSeed(OaNlpSuiteRngSeed);
	auto  model  = OaMakeSharedPtr<OaByteTransformerLM>();
	auto  params = model->AllParameterPtrs();
	auto  opt    = OaMakeUniquePtr<OaAdamW>(params, 0.01F);
	auto& rt     = *OaComputeEngine::GetGlobal();

	printf("Model: Embed(%d→%d) + PosEmbed → OaTransformerBlock(%d, %d) → LN → Linear(%d→%d)\n",
		kVocabSize, kDModel, kDModel, kHiddenDim, kDModel, kVocabSize);
	printf("Params: %lld    Optimizer: AdamW(lr=0.01)\n\n",
		static_cast<long long>(model->NumParameters()));

	const OaI32 steps = static_cast<OaI32>(
		std::max<OaI64>(OaEnvFlag::GetInt("OA_TUTORIAL_STEPS", 300), 1));
	constexpr OaI32 kBatch = 64;
	NlpAllPositionSampler sampler(NlpCorpus(), kBatch);
	const OaBool useTrainingProgram = OaEnvFlag::IsSet("OA_TRAINING_PROGRAM");
	OaTrainingProgram program;

	TutorialTrainingLoop training(*opt, OaItTrainingConfig{
		.TotalSteps     = steps,
		.EpochSteps     = {},
		.BatchSize      = kBatch,
		.SequenceLength = kContextLen,
		.SequenceUnit   = "token",
		.SourceUnitsPerSample = kContextLen,
		.SourceUnit     = "byte",
		.TimerName      = "transformer_step",
		.Callbacks      = {},
		.Program        = useTrainingProgram ? &program : nullptr,
	});

	printf("Training: %d steps · batch=%d · sequence=%d tokens · execution=%s\n",
		steps, kBatch, kContextLen, useTrainingProgram ? "captured" : "eager");

	OaMatrix batchX;
	OaMatrix batchY;
	OaF32 initialLoss = 0.0F;
	OaF32 lastLoss = 0.0F;

	while (not training.Loop.IsDone()) {
		NVTX_RANGE_PUSH("TransformerStep");
		training.Loop.Step(
			[&] { sampler.NextBatch(batchX, batchY); },
			[&] {
				opt->ZeroGrad();
				OaGradientTape tape;
				NVTX_RANGE_PUSH("Forward");
				auto logits = model->Forward(batchX);
				NVTX_RANGE_POP();
				NVTX_RANGE_PUSH("Loss");
				auto loss = OaFnLoss::CrossEntropy(logits,
					batchY.Reshape(OaMatrixShape{batchY.Size(0) * batchY.Size(1)}));
				NVTX_RANGE_POP();
				NVTX_RANGE_PUSH("Backward");
				tape.Backward(loss);
				NVTX_RANGE_POP();
				training.Loop.RecordLoss(loss);
			});
		NVTX_RANGE_POP(); // TransformerStep

		if (training.Loop.Index() == 1) initialLoss = training.Loop.LastLoss();
	}
	ASSERT_TRUE(training.Loop.Finish().IsOk()) << "Finish failed";
	lastLoss = training.Loop.LastLoss();

	const OaF32 finalBatchAcc = NlpAccuracyAllPositions(*model, batchX, batchY, kVocabSize);
	(void)rt;

	printf("\nEvaluation:\n");
	printf("  Random-loss baseline ln(%d) = %.4f\n",
		kVocabSize, static_cast<double>(std::log(static_cast<float>(kVocabSize))));
	printf("  Bits/byte: %.4f\n", NlpBitsPerByte(lastLoss));
	printf("  Accuracy: %.1f%%\n", finalBatchAcc);

	// ── Generate ──
	OaString generated = NlpGenerateGreedy(*model, kNlpGenerationPrompt,
		kNlpGenerationBytes, kVocabSize);
	printf("\nGeneration:\n  Prompt: '%s'\n  Generated: '%s'\n\n",
		kNlpGenerationPrompt, generated.c_str());

	ASSERT_GT(initialLoss, 0.0F);
	EXPECT_LT(lastLoss, initialLoss) << "Loss must decrease during training";
	EXPECT_GT(finalBatchAcc, 30.0F)  << "Final batch accuracy should exceed 30%";

	// ── Save / Load round-trip ──
	const OaString ckptPath = "/tmp/transformer_byte_autograd.oam";
	auto saveStatus = model->Save(ckptPath, *opt);
	ASSERT_TRUE(saveStatus.IsOk()) << "Save failed: " << saveStatus.GetMessage();

	auto reloaded    = OaMakeSharedPtr<OaByteTransformerLM>();
	auto reloadParam = reloaded->AllParameterPtrs();
	auto reloadedOpt = OaMakeUniquePtr<OaAdamW>(reloadParam, 0.01F);
	auto loadStatus  = reloaded->Load(ckptPath, *reloadedOpt);
	ASSERT_TRUE(loadStatus.IsOk()) << "Load failed: " << loadStatus.GetMessage();

	OaF32 reloadedAcc = NlpAccuracyAllPositions(*reloaded, batchX, batchY, kVocabSize);
	printf("Reload accuracy: %.1f%% (was %.1f%%)    Optimizer step: %llu (was %llu)\n\n",
		reloadedAcc, finalBatchAcc,
		static_cast<unsigned long long>(reloadedOpt->GetStep()),
		static_cast<unsigned long long>(opt->GetStep()));
	EXPECT_NEAR(reloadedAcc, finalBatchAcc, 0.5F);
	EXPECT_EQ(reloadedOpt->GetStep(), opt->GetStep());
}
