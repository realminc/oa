// OA Tutorial: TutorialViewerAudio — audio playback through oa::Viewer.

#include <oa/ui/viewer.h>
#include <oa/core/paths.h>

int main(int argc, char** argv) {
	oa::ViewerConfig config;
	config.mode = oa::ViewerMode::Audio;
	config.path = argc > 1
		? oa::String(argv[1])
		: oa::Paths::asset("audio/oaNarration.flac").string();
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
