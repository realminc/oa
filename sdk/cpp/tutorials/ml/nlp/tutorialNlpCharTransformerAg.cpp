#include "oaTest.h"
#include "tutorialNlpCharModels.h"

TEST(TutorialNlpCharTransformerAg, TransformerAllPositionLM) {
	runNlpCharTutorial<CharTransformerLM>("OA Tutorial — Char Transformer · all-position LM (autograd)",
		"Char + position Embed → transformer(32,64) → LN → Linear(32→27)",
		"char_transformer_step", "/tmp/char_transformer.oam", 0.01F);
}
