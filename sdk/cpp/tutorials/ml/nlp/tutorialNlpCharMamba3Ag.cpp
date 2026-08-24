#include "oaTest.h"
#include "tutorialNlpCharModels.h"

TEST(TutorialNlpCharMamba3Ag, Mamba3AllPositionLM) {
	runNlpCharTutorial<CharMamba3LM>("OA Tutorial — Char Mamba-3 · all-position LM (autograd)",
		"Char Embed → Mamba-3(32,state=32,expand=2) + residual → Linear(32→27)",
		"char_mamba3_step", "/tmp/char_mamba3.oam", 0.003F);
}
