// OA determinism Mode — Runtime control for numeric behavior

#include <oa/core/determinism.h>
#include <oa/core/std/stringView.h>

#include <stdlib.h>

static oa::DeterminismMode gDeterminismMode = oa::DeterminismMode::Stable;
static bool gModeSet = false;

oa::DeterminismMode oa::getDeterminismMode() {
	if (!gModeSet) {
		const char* env = ::getenv("OA_DETERMINISM_MODE");
		if (env) {
			const oa::StringView mode(env);
			if (mode == "Fast" || mode == "fast" || mode == "0") {
				gDeterminismMode = oa::DeterminismMode::Fast;
			} else if (mode == "Stable" || mode == "stable" || mode == "1") {
				gDeterminismMode = oa::DeterminismMode::Stable;
			} else if (mode == "Deterministic" || mode == "deterministic" || mode == "2") {
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
