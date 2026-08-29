#pragma once

#include "../ml/tutorialMl.h"
#include "tutorialVision.h"

#include <oa/ui/viewer.h>

#include <stdlib.h>

inline int runTutorialViewerVideo(
	int argc,
	char** argv,
	const oa::String& inDefaultPath,
	const char* inTitle,
	bool inLoop = true)
{
	oa::I32 deviceIdx = tutorialPreParseDeviceIndex(argc, argv);
	if (deviceIdx >= 0) {
		const oa::String idxStr = oa::toString(static_cast<oa::I64>(deviceIdx));
#if defined(_WIN32)
		_putenv_s("OA_DEVICE", idxStr.cStr());
#else
		::setenv("OA_DEVICE", idxStr.cStr(), 1);
#endif
	}

	oa::ViewerConfig config;
	config.mode = oa::ViewerMode::Video;
	config.path = argc > 1 ? argv[1] : inDefaultPath.cStr();
	config.title = inTitle;
	config.loop = inLoop;

	oa::Viewer viewer(config);
	oa::Status status = viewer.run();
	if (not status.isOk()) {
		oa::print(oa::PrintStream::Error, "{}: {}", inTitle, status.toString().cStr());
		return 1;
	}
	return 0;
}
