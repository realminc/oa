#include "../../Test/OaTest.h"
#include "TutorialNlpCharModels.h"

TEST(TutorialNlpCharMoeAg, MoeAllPositionLM) {
	RunNlpCharTutorial<OaCharMoeLM>("OA Tutorial — Char MoE Transformer · all-position LM (Autograd)",
		"Char + position Embed → Attention + MoE(E=4,K=2,DFF=16) → LN → Linear(32→27)",
		"char_moe_step", "/tmp/char_moe.oam", 0.01F);
}
