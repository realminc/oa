#include "../../Test/OaTest.h"
#include "TutorialNlpCharModels.h"

TEST(TutorialNlpCharGruAg, GruAllPositionLM) {
	RunNlpCharTutorial<OaCharGruLM>("OA Tutorial — Char GRU · all-position LM (Autograd)",
		"Char Embed → GRU(32→64) → Linear(64→27)", "char_gru_step", "/tmp/char_gru.oam", 0.01F);
}
