// OA_DOC_BEGIN: vision-image
#include <oa/oa.h>

OA_MAIN_PREVIEW("ExampleVisionImage") {
	auto sourceResult = oa::FnImage::decodeFile(
		oa::Paths::asset("image/coverVision.jpg"), oa::ImageFormat::Rgb);
	if (not sourceResult.isOk()) return 1;
	auto source = oa::move(sourceResult).getValue();

	auto grayscale = oa::FnImage::grayscale(source);

	const oa::Path output = oa::Paths::var("example/vision/coverVisionGrayscale.jpg");
	if (not oa::Filesystem::createDirectories(output.parentPath()).isOk()) return 1;
	if (not oa::FnImage::saveFile(output, grayscale, 92U).isOk()) return 1;

	if (not grayscale.validate()) return 1;
	if (grayscale.width() != 1672 or grayscale.height() != 941) return 1;
	if (grayscale.format() != oa::ImageFormat::Gray) return 1;
	if (not oa::Filesystem::isFile(output)) return 1;

	oa::print("Saved grayscale image: {}", output.cStr());

	if (argc > 1 and oa::StringView(argv[1]) == "--preview") {
		oa::ViewerConfig previewConfig;
		previewConfig.mode = oa::ViewerMode::Image;
		previewConfig.title = "OA vision · grayscale cover";
		previewConfig.width = 1280U;
		previewConfig.height = 646U;
		if (not oa::Viewer::preview(engine, oa::Paths::var("example/vision/coverVisionGrayscale.jpg").string(),
			previewConfig).isOk()) {
			return 1;
		}
	}
	return 0;
}
// OA_DOC_END: vision-image
