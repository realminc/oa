// OA determinism Mode — Runtime control for numeric behavior

#include <oa/core/determinism.h>
#include <cstdlib>
#include <cstring>

static oa::DeterminismMode gDeterminismMode = oa::DeterminismMode::Stable;
static bool gModeSet = false;

oa::DeterminismMode oa::getDeterminismMode() {
	if (!gModeSet) {
		const char* env = std::getenv("OA_DETERMINISM_MODE");
		if (env) {
			std::string s(env);
			if (s == "Fast" || s == "fast" || s == "0") {
				gDeterminismMode = oa::DeterminismMode::Fast;
			} else if (s == "Stable" || s == "stable" || s == "1") {
				gDeterminismMode = oa::DeterminismMode::Stable;
			} else if (s == "Deterministic" || s == "deterministic" || s == "2") {
				gDeterminismMode = oa::DeterminismMode::Deterministic;
			}
		}
		gModeSet = true;
	}
	return gDeterminismMode;
}

void oa::setDeterminismMode(oa::DeterminismMode inMode) {
	gDeterminismMode = inMode;
	gModeSet = true;
}

bool oa::isFastMode() {
	return oa::getDeterminismMode() == oa::DeterminismMode::Fast;
}

bool oa::isStableMode() {
	return oa::getDeterminismMode() == oa::DeterminismMode::Stable;
}

bool oa::isDeterministicMode() {
	return oa::getDeterminismMode() == oa::DeterminismMode::Deterministic;
}
