#include "../../Test/OaTest.h"
#include "TutorialNlpCharModels.h"

TEST(TutorialNlpCharTransformerAg, TransformerAllPositionLM) {
	RunNlpCharTutorial<OaCharTransformerLM>("OA Tutorial — Char Transformer · all-position LM (Autograd)",
		"Char + position Embed → Transformer(32,64) → LN → Linear(32→27)",
		"char_transformer_step", "/tmp/char_transformer.oam", 0.01F);
}
