#include "oaTest.h"
#include "tutorialNlpCharModels.h"

TEST(TutorialNlpCharMoeAg, MoeAllPositionLM) {
	runNlpCharTutorial<CharMoeLM>("OA Tutorial — Char MoE Transformer · all-position LM (autograd)",
		"Char + position Embed → Attention + moE(E=4,K=2,DFF=16) → LN → Linear(32→27)",
		"char_moe_step", "/tmp/char_moe.oam", 0.01F);
}
