#include "../../Test/OaTest.h"
#include "TutorialNlpBpeModels.h"

TEST(TutorialNlpBpeTransformerAg, TransformerAllPositionLM) {
	RunNlpBpeTutorial<OaBpeTransformerLM>("OA Tutorial — BPE Transformer · all-position LM (Autograd)",
		"BPE + position Embed → Transformer(32,64) → LN → Linear(32→320)",
		"bpe_transformer_step", "/tmp/bpe_transformer.oam", 0.01F);
}
