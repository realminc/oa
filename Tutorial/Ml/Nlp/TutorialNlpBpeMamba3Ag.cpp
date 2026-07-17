#include "OaTest.h"
#include "TutorialNlpBpeModels.h"

TEST(TutorialNlpBpeMamba3Ag, Mamba3AllPositionLM) {
	RunNlpBpeTutorial<OaBpeMamba3LM>("OA Tutorial — BPE Mamba-3 · all-position LM (Autograd)",
		"BPE Embed → Mamba-3(32,state=32,expand=2) + residual → Linear(32→320)",
		"bpe_mamba3_step", "/tmp/bpe_mamba3.oam", 0.003F);
}
