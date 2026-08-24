#include "oaTest.h"
#include "tutorialNlpBpeModels.h"

TEST(TutorialNlpBpeGruAg, GruAllPositionLM) {
	runNlpBpeTutorial<BpeGruLM>("OA Tutorial — BPE GRU · all-position LM (autograd)",
		"BPE Embed → GRU(32→64) → Linear(64→320)", "bpe_gru_step", "/tmp/bpe_gru.oam", 0.01F);
}
