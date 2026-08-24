#include "oaTest.h"
#include "tutorialNlpBpeModels.h"

TEST(TutorialNlpBpeMoeAg, MoeAllPositionLM) {
	runNlpBpeTutorial<BpeMoeLM>("OA Tutorial — BPE MoE Transformer · all-position LM (autograd)",
		"BPE + position Embed → Attention + moE(E=4,K=2,DFF=16) → LN → Linear(32→320)",
		"bpe_moe_step", "/tmp/bpe_moe.oam", 0.01F);
}
