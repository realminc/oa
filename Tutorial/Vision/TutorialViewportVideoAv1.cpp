// OA Tutorial: AV1 video playback via OaVideoPlayer.

#include "../Ml/TutorialMl.h"
#include "TutorialVision.h"

#include <Oa/Ui/VideoPlayer.h>

#include <cstdlib>
#include <string>

namespace {

int RunVideoTutorial(
	int argc,
	char** argv,
	const char* InDefaultFilename,
	OaVideoCodec InCodec)
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

	const std::string defaultPath = TutorialVideoPath(InDefaultFilename);
	OaVideoPlayerConfig cfg;
	cfg.Path = (argc > 1) ? argv[1] : defaultPath.c_str();
	cfg.Codec = InCodec;

	OaVideoPlayer player(cfg);
	return player.Run().IsOk() ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
	return RunVideoTutorial(
		argc,
		argv,
		"shibuya_crossing_1080p30_av1.mp4",
		OaVideoCodec::AV1);
}
