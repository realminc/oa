#include "../../Test/OaTest.h"
#include "TutorialNlpBpeModels.h"

TEST(TutorialNlpBpeGruAg, GruAllPositionLM) {
	RunNlpBpeTutorial<OaBpeGruLM>("OA Tutorial — BPE GRU · all-position LM (Autograd)",
		"BPE Embed → GRU(32→64) → Linear(64→320)", "bpe_gru_step", "/tmp/bpe_gru.oam", 0.01F);
}
