#include "OaTest.h"
#include "TutorialNlpCharModels.h"

TEST(TutorialNlpCharMamba3Ag, Mamba3AllPositionLM) {
	RunNlpCharTutorial<OaCharMamba3LM>("OA Tutorial — Char Mamba-3 · all-position LM (Autograd)",
		"Char Embed → Mamba-3(32,state=32,expand=2) + residual → Linear(32→27)",
		"char_mamba3_step", "/tmp/char_mamba3.oam", 0.003F);
}
