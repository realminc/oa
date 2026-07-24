#pragma once

#include "../Ml/TutorialMl.h"
#include "TutorialVision.h"

#include <Oa/Ui/Viewer.h>

#include <cstdio>
#include <cstdlib>
#include <string>

inline int RunTutorialViewerVideo(
	int argc,
	char** argv,
	const std::string& InDefaultPath,
	const char* InTitle,
	bool InLoop = true)
{
	OaI32 deviceIdx = TutorialPreParseDeviceIndex(argc, argv);
	if (deviceIdx >= 0) {
		OaString idxStr = OaString(std::to_string(deviceIdx).c_str());
#if defined(_WIN32)
		_putenv_s("OA_DEVICE", idxStr.c_str());
#else
		::setenv("OA_DEVICE", idxStr.c_str(), 1);
#endif
	}

	OaViewerConfig config;
	config.Mode = OaViewerMode::Video;
	config.Path = argc > 1 ? argv[1] : InDefaultPath.c_str();
	config.Title = InTitle;
	config.Loop = InLoop;

	OaViewer viewer(config);
	OaStatus status = viewer.Run();
	if (not status.IsOk()) {
		std::fprintf(stderr, "%s: %s\n", InTitle, status.ToString().c_str());
		return 1;
	}
	return 0;
}
