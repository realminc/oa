// ═══════════════════════════════════════════════════════════════════════════
// OA Tutorial — Byte-level Recurrent RNN, all-position LM (Implicit autograd)
// Module API: oa::ByteEmbedding + oa::Rnn + oa::Linear + oa::AdamW + oa::GradientTape
// ═══════════════════════════════════════════════════════════════════════════
//
// The recurrent RNN member of the NLP fair-comparison suite (TutorialNlpCommon.h):
// every tutorial trains the same all-position next-token task on the same corpus
// with the same dims, so RNN / GRU / Transformer / Mamba-3 / empyrealm are directly
// comparable. oa::Rnn carries hidden state across timesteps (fused Add+Tanh via
// RnnCellPointwise) and we project *every* timestep to the vocab — not just the
// last — so this is a genuine language model, not a flatten-window classifier.
// ═══════════════════════════════════════════════════════════════════════════

#include "oaTest.h"
#include "tutorialMl.h"
#include "tutorialNlpCommon.h"
#include <oa/ml.h>
#include <oa/ml/byte.h>
#include <oa/ml/autograd.h>
#include <oa/runtime/engine.h>

static constexpr oa::I32 kVocabSize = oa::ByteVocabSize;  // 256 — byte vocab family

// ─── Model: ByteEmbedding → RNN (recurrent) → Linear, projected at every step ─

class ByteRnnLM : public oa::Module {
public:
	ByteRnnLM() {
		auto wd = oa::FnMatrix::weightDtype();

		embed_ = oa::makeShared<oa::ByteEmbedding>(kDModel);
		embed_->parameters()[0].data = oa::FnMatrix::randN(oa::MatrixShape{kVocabSize, kDModel}, wd);
		embed_->parameters()[0].data.setRequiresGrad(true);

		rnn_ = oa::makeShared<oa::Rnn>(kDModel, kHiddenDim, 1);
		for (auto& param : rnn_->parameters()) {
			param.data.setRequiresGrad(true);
		}

		head_ = oa::makeShared<oa::Linear>(kHiddenDim, kVocabSize);
		head_->parameters()[0].data = oa::FnMatrix::rand(oa::MatrixShape{kVocabSize, kHiddenDim}, wd);
		for (auto& param : head_->parameters()) {
			param.data.setRequiresGrad(true);
		}

		registerModule("embed", embed_);
		registerModule("rnn",   rnn_);
		registerModule("head",  head_);
	}

	oa::Matrix forward(const oa::Matrix& inTokens) override {
		const oa::I32 batch  = static_cast<oa::I32>(inTokens.size(0));
		const oa::I32 seqLen = static_cast<oa::I32>(inTokens.size(1));

		// Gather/ByteEmbedding flattens to [batch*seq, d_model]; restore the rank-3
		// shape oa::Rnn::forward expects (else it slices the wrong axis → O(seq^2)).
		auto embedded = embed_->forward(inTokens).reshape(oa::MatrixShape{batch, seqLen, kDModel});
		auto rnnOut   = rnn_->forward(embedded);                          // [B, S, H]
		auto flat     = rnnOut.reshape(oa::MatrixShape{static_cast<oa::I64>(batch) * seqLen, kHiddenDim});  // [B*S, H]
		return head_->forward(flat);                                      // [B*S, V]
	}

private:
	oa::SharedPtr<oa::ByteEmbedding> embed_;
	oa::SharedPtr<oa::Rnn>           rnn_;
	oa::SharedPtr<oa::Linear>        head_;
};

// ─── Tutorial ──────────────────────────────────────────────────────────────

TEST(TutorialNlpByteRnnAg, RecurrentRnnAllPositionLM) {
	oa::print("\n╔══════════════════════════════════════════════════════════════════╗");
	oa::print("║  OA Tutorial — Byte RNN · all-position LM (autograd)             ║");
	oa::print("╚══════════════════════════════════════════════════════════════════╝\n");
	oa::print("Vocab: {} bytes · context: {} · dModel: {} · Hidden: {}",
		kVocabSize, kContextLen, kDModel, kHiddenDim);
	oa::print("Task: dense next-byte at every position via a recurrent RNN\n");

	oa::FnMatrix::setRngSeed(oa::NlpSuiteRngSeed);
	auto  model  = oa::makeShared<ByteRnnLM>();
	auto  params = model->allParameterPtrs();
	auto  opt    = oa::makeUnique<oa::AdamW>(params, 0.01F);
	auto& rt     = testEngine();

	oa::print("Model: byteEmbed({}→{}) → RNN({}→{}, layers=1) → Linear({}→{})",
		kVocabSize, kDModel, kDModel, kHiddenDim, kHiddenDim, kVocabSize);
	oa::print("params: {}    Optimizer: AdamW(lr=0.01)\n",
		static_cast<long long>(model->numParameters()));

	NlpAllPositionSampler sampler(nlpCorpus(), kBatch);
	TutorialTrainingLoop training(rt, *opt, oa::ItTrainingConfig{
		.totalSteps     = kSteps,
		.epochSteps     = {},
		.batchSize      = kBatch,
		.sequenceLength = kContextLen,
		.sequenceUnit   = "token",
		.sourceUnitsPerSample = kContextLen,
		.sourceUnit     = "byte",
		.timerName      = "byte_rnn_allpos_step",
		.callbacks      = {},
	});
	oa::print("training: {} steps · batch={} · sequence={} tokens", kSteps, kBatch, kContextLen);

	oa::Matrix batchX;
	oa::Matrix batchY;
	oa::F32 initialLoss = 0.0F;

	while (not training.loop.isDone()) {
		sampler.nextBatch(batchX, batchY);
		opt->zeroGrad();
		oa::GradientTape tape;
		auto logits = model->forward(batchX);
		auto loss   = oa::FnLoss::crossEntropy(logits,
			batchY.reshape(oa::MatrixShape{batchY.size(0) * batchY.size(1)}));
		tape.backward(loss);
		training.loop.next(loss);
		if (training.loop.index() == 1) { initialLoss = training.loop.lastLoss(); }
	}
	ASSERT_TRUE(training.loop.finish().isOk()) << "finish failed";
	const oa::F32 lastLoss = training.loop.lastLoss();

	const oa::F32 finalAcc = nlpAccuracyAllPositions(*model, batchX, batchY, kVocabSize);
	(void)rt;
	oa::print("\nEvaluation:");
	oa::print("  Random-loss baseline ln({}) = {:.4f}",
		kVocabSize, oa::log(static_cast<double>(kVocabSize)));
	oa::print("  bits/byte: {:.4f}", nlpBitsPerByte(lastLoss));
	oa::print("  Accuracy: {:.1f}%", finalAcc);
	oa::print("\nGeneration:\n  prompt: '{}'\n  generated: '{}'\n", kNlpGenerationPrompt,
		nlpGenerateGreedy(*model, kNlpGenerationPrompt, kNlpGenerationBytes, kVocabSize).cStr());

	ASSERT_GT(initialLoss, 0.0F);
	EXPECT_LT(lastLoss, initialLoss) << "loss must decrease during training";
	EXPECT_GT(finalAcc, 50.0F)       << "All-position accuracy should exceed 50%";

	// ── save / load round-trip (model + optimizer) ──
	const oa::String ckptPath = "/tmp/byte_rnn_allpos.oam";
	ASSERT_TRUE(model->save(rt, ckptPath, *opt).isOk());
	auto reloaded    = oa::makeShared<ByteRnnLM>();
	auto reloadParam = reloaded->allParameterPtrs();
	auto reloadedOpt = oa::makeUnique<oa::AdamW>(reloadParam, 0.01F);
	ASSERT_TRUE(reloaded->load(rt, ckptPath, *reloadedOpt).isOk());
	const oa::F32 reloadedAcc = nlpAccuracyAllPositions(*reloaded, batchX, batchY, kVocabSize);
	EXPECT_NEAR(reloadedAcc, finalAcc, 0.5F)          << "Reload accuracy must match";
	EXPECT_EQ(reloadedOpt->getStep(), opt->getStep()) << "Optimizer step must round-trip";
}
