#include "../../Test/OaTest.h"
#include "TutorialNlpBpeModels.h"

TEST(TutorialNlpBpeRnnAg, RnnAllPositionLM) {
	RunNlpBpeTutorial<OaBpeRnnLM>("OA Tutorial — BPE RNN · all-position LM (Autograd)",
		"BPE Embed → RNN(32→64) → Linear(64→320)", "bpe_rnn_step", "/tmp/bpe_rnn.oam", 0.01F);
}
