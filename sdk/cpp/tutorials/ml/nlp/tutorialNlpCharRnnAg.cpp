#include "oaTest.h"
#include "tutorialNlpCharModels.h"

TEST(TutorialNlpCharRnnAg, RnnAllPositionLM) {
	runNlpCharTutorial<CharRnnLM>("OA Tutorial — Char RNN · all-position LM (autograd)",
		"Char Embed → RNN(32→64) → Linear(64→27)", "char_rnn_step", "/tmp/char_rnn.oam", 0.01F);
}
