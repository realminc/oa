#include "OaTest.h"
#include "TutorialNlpCharModels.h"

TEST(TutorialNlpCharRnnAg, RnnAllPositionLM) {
	RunNlpCharTutorial<OaCharRnnLM>("OA Tutorial — Char RNN · all-position LM (Autograd)",
		"Char Embed → RNN(32→64) → Linear(64→27)", "char_rnn_step", "/tmp/char_rnn.oam", 0.01F);
}
