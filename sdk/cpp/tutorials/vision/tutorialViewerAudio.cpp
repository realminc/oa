// OA Tutorial: TutorialViewerAudio — audio playback through oa::Viewer.

#include <oa/ui/viewer.h>

int main(int argc, char** argv) {
	oa::ViewerConfig config;
	config.mode = oa::ViewerMode::Audio;
	config.path = argc > 1 ? argv[1] : "Asset/Audio/0_jackson_0.flac";
	if (argc > 2) {
		const oa::StringView view(argv[2]);
		if (view == "spectrum") config.audioView = oa::ViewerAudioView::Spectrum;
		else if (view == "mel") config.audioView = oa::ViewerAudioView::Mel;
	}
	config.title = "oa::Viewer · Audio";
	config.width = 960;
	config.height = 360;
	oa::Viewer viewer(config);
	return viewer.run().isOk() ? 0 : 1;
}
