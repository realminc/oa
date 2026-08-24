// ═══════════════════════════════════════════════════════════════════════════
// OA Tutorial — Transformer Block (autograd)
// Module API: oa::Embedding + oa::TransformerBlock + oa::LayerNorm + oa::Linear
//              + oa::AdamW + oa::GradientTape
// ═══════════════════════════════════════════════════════════════════════════
//
// Pre-norm transformer block with causal self-attention, built from the
// library oa::TransformerBlock module. architecture:
//   TokEmbed + PosEmbed → oa::TransformerBlock → LayerNorm → Linear head
//
// All layers use implicit autograd (oa::GradientTape).  No manual backward.
//
// ═══════════════════════════════════════════════════════════════════════════

#include "oaTest.h"
#include "tutorialMl.h"
#include "tutorialNlpCommon.h"
#include <oa/ml.h>
#include <oa/ml/byte.h>
#include <oa/ml/autograd.h>
#include <oa/core/envFlag.h>
#include <oa/runtime/engine.h>
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

static constexpr oa::I32 kVocabSize = oa::ByteVocabSize;  // 256 — byte vocab family
// kContextLen / kDModel / kHiddenDim come from TutorialNlpCommon.h (shared suite dims).

// ─── Transformer LM: Embed + oa::TransformerBlock + head ─────────────────────
//
// Thin wrapper around the library oa::TransformerBlock module. Adds token +
// positional embeddings, a final LayerNorm (GPT-style ln_f), and a linear
// output head. The block-diagonal causal mask and attention machinery live
// inside oa::TransformerBlock.

class ByteTransformerLM : public oa::Module {
public:
	ByteTransformerLM() {
		tokEmbed_ = oa::makeShared<oa::Embedding>(kVocabSize, kDModel);
		registerModule("tok_embed", tokEmbed_);
		posEmbed_ = oa::makeShared<oa::Embedding>(kContextLen, kDModel);
		registerModule("pos_embed", posEmbed_);

		block_ = oa::makeShared<oa::TransformerBlock>(kDModel, kHiddenDim, kContextLen);
		registerModule("block", block_);

		lnFinal_ = oa::makeShared<oa::LayerNorm>(kDModel, 1e-5f);
		registerModule("ln_final", lnFinal_);

		head_ = oa::makeShared<oa::Linear>(kDModel, kVocabSize);
		head_->parameters()[0].data = oa::FnMatrix::rand(oa::MatrixShape{kVocabSize, kDModel},
			oa::FnMatrix::weightDtype());
		registerModule("head", head_);

		for (auto& p : allParameterPtrs()) {
			p->data.setRequiresGrad(true);
		}
	}

	oa::Matrix forward(const oa::Matrix& inTokens) override {
		const oa::I32 b = static_cast<oa::I32>(inTokens.size(0));
		const oa::I32 n = b * kContextLen;

		auto tokEmb = oa::FnMatrix::reshape(tokEmbed_->forward(inTokens), oa::MatrixShape{n, kDModel});
		auto posEmb = posEmbed_->forward(positionIds(b));
		auto x = oa::FnMatrix::add(tokEmb, posEmb);

		x = block_->forward(x);

		return head_->forward(lnFinal_->forward(x));
	}

private:
	oa::Matrix positionIds(oa::I32 inBatch) const {
		const oa::I32 n = inBatch * kContextLen;
		oa::Vec<oa::U8> ids(n);
		for (oa::I32 i = 0; i < n; ++i) ids[i] = static_cast<oa::U8>(i % kContextLen);
		return oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(ids.data(), ids.size()),
			oa::MatrixShape{n}, oa::ScalarType::UInt8);
	}

	oa::SharedPtr<oa::Embedding>        tokEmbed_;
	oa::SharedPtr<oa::Embedding>        posEmbed_;
	oa::SharedPtr<oa::TransformerBlock> block_;
	oa::SharedPtr<oa::LayerNorm>        lnFinal_;
	oa::SharedPtr<oa::Linear>           head_;
};

// ─── MatMul autograd: the primitive that makes attention trainable ──────────
//
// Attention needs activation×activation matmuls (Q@Kᵀ, attn@V). This verifies
// GradMatMulNt / GradTranspose flow gradients into both operands — without
// these, the score/context matmuls are gradient sinks and attention never
// learns (the original failure mode of this tutorial).

TEST(TutorialNlpByteTransformerAg, MatMulBackpropFlowsToBothOperands) {
	// Two learnable [4, 3] activations; CE loss on rows of C = A @ Bᵀ.
	auto A = oa::FnMatrix::randN(oa::MatrixShape{4, 3}, oa::ScalarType::Float32);
	auto B = oa::FnMatrix::randN(oa::MatrixShape{4, 3}, oa::ScalarType::Float32);
	A.setRequiresGrad(true);
	B.setRequiresGrad(true);

	const oa::U8 targetIds[4] = {0, 1, 2, 3};
	auto targets = oa::FnMatrix::fromBytes(oa::Span<const oa::U8>(targetIds, 4),
		oa::MatrixShape{4}, oa::ScalarType::UInt8);

	oa::GradientTape tape;
	auto C    = oa::FnMatrix::matMulNt(A, B);     // [4, 4] = A @ Bᵀ
	auto loss = oa::FnLoss::crossEntropy(C, targets);
	tape.backward(loss);
	ASSERT_TRUE(tutorialSubmitAndWait(testEngine()).isOk());

	auto gradMag = [](const oa::Matrix& g) {
		std::vector<float> h(static_cast<size_t>(g.numElements()));
		(void)oa::FnMatrix::copyToHost(g, h.data(), h.size() * sizeof(float));
		float s = 0.0F;
		for (float v : h) s += std::abs(v);
		return s;
	};
	EXPECT_GT(gradMag(A.gradMatrix()), 0.0F) << "MatMul did not backprop into A";
	EXPECT_GT(gradMag(B.gradMatrix()), 0.0F) << "MatMul did not backprop into B (Bᵀ path)";
}

// ─── Tutorial ──────────────────────────────────────────────────────────────

TEST(TutorialNlpByteTransformerAg, TransformerByteNextToken) {
	printf("\n╔══════════════════════════════════════════════════════════════════╗\n");
	printf("║  OA Tutorial — Transformer Block (Byte autograd)                ║\n");
	printf("╚══════════════════════════════════════════════════════════════════╝\n\n");
	printf("Vocab: 256 bytes · context: %d · dModel: %d · Attention: causal self-attn (1 head)\n", kContextLen, kDModel);
	printf("Task: all-position next-byte (dense windows, matched to ByteRnnAg corpus)\n\n");

	oa::FnMatrix::setRngSeed(oa::NlpSuiteRngSeed);
	auto  model  = oa::makeShared<ByteTransformerLM>();
	auto  params = model->allParameterPtrs();
	auto  opt    = oa::makeUnique<oa::AdamW>(params, 0.01F);
	auto& rt     = testEngine();

	printf("Model: embed(%d→%d) + PosEmbed → oa::TransformerBlock(%d, %d) → LN → Linear(%d→%d)\n",
		kVocabSize, kDModel, kDModel, kHiddenDim, kDModel, kVocabSize);
	printf("params: %lld    Optimizer: AdamW(lr=0.01)\n\n",
		static_cast<long long>(model->numParameters()));

	const oa::I32 steps = static_cast<oa::I32>(
		std::max<oa::I64>(oa::EnvFlag::getInt("OA_TUTORIAL_STEPS", 300), 1));
	constexpr oa::I32 kBatch = 64;
	NlpAllPositionSampler sampler(nlpCorpus(), kBatch);
	const oa::Bool useTrainingProgram = oa::EnvFlag::isSet("OA_TRAINING_PROGRAM");
	oa::TrainingProgram program;

	TutorialTrainingLoop training(rt, *opt, oa::ItTrainingConfig{
		.totalSteps     = steps,
		.epochSteps     = {},
		.batchSize      = kBatch,
		.sequenceLength = kContextLen,
		.sequenceUnit   = "token",
		.sourceUnitsPerSample = kContextLen,
		.sourceUnit     = "byte",
		.timerName      = "transformer_step",
		.callbacks      = {},
		.program        = useTrainingProgram ? &program : nullptr,
	});

	printf("training: %d steps · batch=%d · sequence=%d tokens · execution=%s\n",
		steps, kBatch, kContextLen, useTrainingProgram ? "captured" : "eager");

	oa::Matrix batchX;
	oa::Matrix batchY;
	oa::F32 initialLoss = 0.0F;
	oa::F32 lastLoss = 0.0F;

	while (not training.loop.isDone()) {
		NVTX_RANGE_PUSH("TransformerStep");
		training.loop.step(
			[&] { sampler.nextBatch(batchX, batchY); },
			[&] {
				opt->zeroGrad();
				oa::GradientTape tape;
				NVTX_RANGE_PUSH("forward");
				auto logits = model->forward(batchX);
				NVTX_RANGE_POP();
				NVTX_RANGE_PUSH("loss");
				auto loss = oa::FnLoss::crossEntropy(logits,
					batchY.reshape(oa::MatrixShape{batchY.size(0) * batchY.size(1)}));
				NVTX_RANGE_POP();
				NVTX_RANGE_PUSH("backward");
				tape.backward(loss);
				NVTX_RANGE_POP();
				training.loop.recordLoss(loss);
			});
		NVTX_RANGE_POP(); // TransformerStep

		if (training.loop.index() == 1) initialLoss = training.loop.lastLoss();
	}
	ASSERT_TRUE(training.loop.finish().isOk()) << "finish failed";
	lastLoss = training.loop.lastLoss();

	const oa::F32 finalBatchAcc = nlpAccuracyAllPositions(*model, batchX, batchY, kVocabSize);
	(void)rt;

	printf("\nEvaluation:\n");
	printf("  Random-loss baseline ln(%d) = %.4f\n",
		kVocabSize, static_cast<double>(std::log(static_cast<float>(kVocabSize))));
	printf("  bits/byte: %.4f\n", nlpBitsPerByte(lastLoss));
	printf("  Accuracy: %.1f%%\n", finalBatchAcc);

	// ── generate ──
	oa::String generated = nlpGenerateGreedy(*model, kNlpGenerationPrompt,
		kNlpGenerationBytes, kVocabSize);
	printf("\nGeneration:\n  prompt: '%s'\n  generated: '%s'\n\n",
		kNlpGenerationPrompt, generated.cStr());

	ASSERT_GT(initialLoss, 0.0F);
	EXPECT_LT(lastLoss, initialLoss) << "loss must decrease during training";
	EXPECT_GT(finalBatchAcc, 30.0F)  << "Final batch accuracy should exceed 30%";

	// ── save / load round-trip ──
	const oa::String ckptPath = "/tmp/transformer_byte_autograd.oam";
	auto saveStatus = model->save(rt, ckptPath, *opt);
	ASSERT_TRUE(saveStatus.isOk()) << "save failed: " << saveStatus.getMessage();

	auto reloaded    = oa::makeShared<ByteTransformerLM>();
	auto reloadParam = reloaded->allParameterPtrs();
	auto reloadedOpt = oa::makeUnique<oa::AdamW>(reloadParam, 0.01F);
	auto loadStatus  = reloaded->load(rt, ckptPath, *reloadedOpt);
	ASSERT_TRUE(loadStatus.isOk()) << "load failed: " << loadStatus.getMessage();

	oa::F32 reloadedAcc = nlpAccuracyAllPositions(*reloaded, batchX, batchY, kVocabSize);
	printf("Reload accuracy: %.1f%% (was %.1f%%)    Optimizer step: %llu (was %llu)\n\n",
		reloadedAcc, finalBatchAcc,
		static_cast<unsigned long long>(reloadedOpt->getStep()),
		static_cast<unsigned long long>(opt->getStep()));
	EXPECT_NEAR(reloadedAcc, finalBatchAcc, 0.5F);
	EXPECT_EQ(reloadedOpt->getStep(), opt->getStep());
}
