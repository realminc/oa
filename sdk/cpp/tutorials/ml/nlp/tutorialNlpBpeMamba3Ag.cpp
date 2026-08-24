#include "oaTest.h"
#include "tutorialNlpBpeModels.h"

TEST(TutorialNlpBpeMamba3Ag, Mamba3AllPositionLM) {
	runNlpBpeTutorial<BpeMamba3LM>("OA Tutorial — BPE Mamba-3 · all-position LM (autograd)",
		"BPE Embed → Mamba-3(32,state=32,expand=2) + residual → Linear(32→320)",
		"bpe_mamba3_step", "/tmp/bpe_mamba3.oam", 0.003F);
}
