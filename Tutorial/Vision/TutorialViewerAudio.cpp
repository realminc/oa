// OA Tutorial: TutorialViewerAudio — audio playback through OaViewer.

#include <Oa/Ui/Viewer.h>

int main(int argc, char** argv) {
	OaViewerConfig config;
	config.Mode = OaViewerMode::Audio;
	config.Path = argc > 1 ? argv[1] : "Asset/Audio/0_jackson_0.flac";
	config.Title = "OaViewer · Audio";
	config.Width = 960;
	config.Height = 360;
	OaViewer viewer(config);
	return viewer.Run().IsOk() ? 0 : 1;
}
