#include "oaTest.h"
#include "tutorialNlpBpeModels.h"

TEST(TutorialNlpBpeRnnAg, RnnAllPositionLM) {
	runNlpBpeTutorial<BpeRnnLM>("OA Tutorial — BPE RNN · all-position LM (autograd)",
		"BPE Embed → RNN(32→64) → Linear(64→320)", "bpe_rnn_step", "/tmp/bpe_rnn.oam", 0.01F);
}
