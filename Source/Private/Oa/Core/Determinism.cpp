// OA Determinism Mode — Runtime control for numeric behavior

#include <Oa/Core/Determinism.h>
#include <cstdlib>
#include <cstring>

static OaDeterminismMode gDeterminismMode = OaDeterminismMode::Stable;
static bool gModeSet = false;

OaDeterminismMode OaGetDeterminismMode() {
	if (!gModeSet) {
		const char* env = std::getenv("OA_DETERMINISM_MODE");
		if (env) {
			std::string s(env);
			if (s == "Fast" || s == "fast" || s == "0") {
				gDeterminismMode = OaDeterminismMode::Fast;
			} else if (s == "Stable" || s == "stable" || s == "1") {
				gDeterminismMode = OaDeterminismMode::Stable;
			} else if (s == "Deterministic" || s == "deterministic" || s == "2") {
				gDeterminismMode = OaDeterminismMode::Deterministic;
			}
		}
		gModeSet = true;
	}
	return gDeterminismMode;
}

void OaSetDeterminismMode(OaDeterminismMode InMode) {
	gDeterminismMode = InMode;
	gModeSet = true;
}

bool OaIsFastMode() {
	return OaGetDeterminismMode() == OaDeterminismMode::Fast;
}

bool OaIsStableMode() {
	return OaGetDeterminismMode() == OaDeterminismMode::Stable;
}

bool OaIsDeterministicMode() {
	return OaGetDeterminismMode() == OaDeterminismMode::Deterministic;
}
