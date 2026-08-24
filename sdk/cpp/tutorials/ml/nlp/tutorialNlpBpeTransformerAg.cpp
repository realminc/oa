#include "oaTest.h"
#include "tutorialNlpBpeModels.h"

TEST(TutorialNlpBpeTransformerAg, TransformerAllPositionLM) {
	runNlpBpeTutorial<BpeTransformerLM>("OA Tutorial — BPE Transformer · all-position LM (autograd)",
		"BPE + position Embed → transformer(32,64) → LN → Linear(32→320)",
		"bpe_transformer_step", "/tmp/bpe_transformer.oam", 0.01F);
}
