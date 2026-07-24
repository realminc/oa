#include "LunarLander3dViewerSource.h"

#include <Oa/Ui/Viewer.h>

int main() {
	OaLunarLander3dViewerSource source;
	OaViewer viewer({
		.Mode = OaViewerMode::Live,
		.LiveSource = &source,
		.Title = "OA · Lunar Lander 3D",
		.Width = 1280U,
		.Height = 720U,
		.ShowHelp = true,
		.ShowStats = false,
		.ShowTimeline = false,
		.Vsync = true,
		.PresentFilter = OaFilter::Linear,
	});
	return viewer.Run().IsOk() ? 0 : 1;
}
