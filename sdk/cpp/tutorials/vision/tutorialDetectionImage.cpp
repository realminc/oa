// OA Tutorial: image inspection with GPU detection boxes and SDF labels.
//
// The boxes are display fixtures. Replace them with model/NMS output; the
// oa::Viewer and oa::DetectionOverlay path remains unchanged.

#include <oa/ui/viewer.h>
#include <oa/core/paths.h>

#include <stdlib.h>
int main(int argc, char** argv) {
	oa::ViewerConfig config;
	config.title = "OA detection Image";
	config.path = argc > 1
		? oa::String(argv[1])
		: oa::Paths::asset("image/coverMl.jpg").string();
	config.annotations.pushBack({
		.detection = {
			.centerX = 0.50F,
			.centerY = 0.52F,
			.width = 0.31F,
			.height = 0.82F,
			.confidence = 0.98F,
			.classId = 0,
			.colorRgba = oa::Color::success().toU32(),
			.trackId = 1,
		},
		.label = "person 98% / track 1",
	});
	config.annotations.pushBack({
		.detection = {
			.centerX = 0.18F,
			.centerY = 0.69F,
			.width = 0.20F,
			.height = 0.24F,
			.confidence = 0.91F,
			.classId = 24,
			.colorRgba = oa::Color::cyan().toU32(),
			.trackId = 2,
		},
		.label = "backpack 91% / track 2",
	});

	oa::Viewer viewer(config);
	return viewer.run().isOk() ? EXIT_SUCCESS : EXIT_FAILURE;
}
