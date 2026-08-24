#pragma once

#include "../ml/tutorialMl.h"
#include "tutorialVision.h"

#include <oa/ui/viewer.h>

#include <cstdio>
#include <cstdlib>
#include <string>

inline int runTutorialViewerVideo(
	int argc,
	char** argv,
	const std::string& inDefaultPath,
	const char* inTitle,
	bool inLoop = true)
{
	oa::I32 deviceIdx = tutorialPreParseDeviceIndex(argc, argv);
	if (deviceIdx >= 0) {
		oa::String idxStr = oa::String(std::to_string(deviceIdx).c_str());
#if defined(_WIN32)
		_putenv_s("OA_DEVICE", idxStr.cStr());
#else
		::setenv("OA_DEVICE", idxStr.cStr(), 1);
#endif
	}

	oa::ViewerConfig config;
	config.mode = oa::ViewerMode::Video;
	config.path = argc > 1 ? argv[1] : inDefaultPath.c_str();
	config.title = inTitle;
	config.loop = inLoop;

	oa::Viewer viewer(config);
	oa::Status status = viewer.run();
	if (not status.isOk()) {
		std::fprintf(stderr, "%s: %s\n", inTitle, status.toString().cStr());
		return 1;
	}
	return 0;
}
