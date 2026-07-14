#include "../../Test/OaTest.h"
#include "TutorialNlpBpeModels.h"

TEST(TutorialNlpBpeMoeAg, MoeAllPositionLM) {
	RunNlpBpeTutorial<OaBpeMoeLM>("OA Tutorial — BPE MoE Transformer · all-position LM (Autograd)",
		"BPE + position Embed → Attention + MoE(E=4,K=2,DFF=16) → LN → Linear(32→320)",
		"bpe_moe_step", "/tmp/bpe_moe.oam", 0.01F);
}
