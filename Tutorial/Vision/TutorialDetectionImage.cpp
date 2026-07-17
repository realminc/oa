// OA Tutorial: image inspection with GPU detection boxes and SDF labels.
//
// The boxes are display fixtures. Replace them with model/NMS output; the
// OaViewer and OaDetectionOverlay path remains unchanged.

#include <Oa/Ui/ImageViewer.h>

#include <cstdlib>

int main(int argc, char** argv) {
	OaViewerConfig config;
	config.Title = "OA Detection Image";
	config.Path = argc > 1 ? argv[1] : "Asset/Image/person_demo.jpg";
	config.Annotations.PushBack({
		.Detection = {
			.CenterX = 0.50F,
			.CenterY = 0.52F,
			.Width = 0.31F,
			.Height = 0.82F,
			.Confidence = 0.98F,
			.ClassId = 0,
			.ColorRgba = OaColor::Success().ToU32(),
			.TrackId = 1,
		},
		.Label = "person 98% / track 1",
	});
	config.Annotations.PushBack({
		.Detection = {
			.CenterX = 0.18F,
			.CenterY = 0.69F,
			.Width = 0.20F,
			.Height = 0.24F,
			.Confidence = 0.91F,
			.ClassId = 24,
			.ColorRgba = OaColor::Cyan().ToU32(),
			.TrackId = 2,
		},
		.Label = "backpack 91% / track 2",
	});

	OaImageViewer viewer(config);
	return viewer.Run().IsOk() ? EXIT_SUCCESS : EXIT_FAILURE;
}
