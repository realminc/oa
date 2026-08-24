#include "oaTest.h"
#include "tutorialNlpCharModels.h"

TEST(TutorialNlpCharGruAg, GruAllPositionLM) {
	runNlpCharTutorial<CharGruLM>("OA Tutorial — Char GRU · all-position LM (autograd)",
		"Char Embed → GRU(32→64) → Linear(64→27)", "char_gru_step", "/tmp/char_gru.oam", 0.01F);
}
