// OA Vision tutorial — deterministic GPU data-augmentation views.

#include <oa/core/filesystem.h>
#include <oa/core/paths.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/texture.h>
#include <oa/ui/plot/plot.h>
#include <oa/vision/fnImage.h>

namespace {

oa::Image rgbImage(oa::Matrix inMatrix) {
	return oa::Image(
		oa::move(inMatrix),
		oa::ImageLayout::Nchw,
		oa::ImageFormat::Rgb);
}

} // namespace

int main(int argc, char** argv) {
	const oa::Path input = argc > 1
		? oa::Path(argv[1])
		: oa::Paths::asset("image/visionTestPattern320x180.jpg");
	const oa::Path output = argc > 2
		? oa::Path(argv[2])
		: oa::Path("/tmp/oa-vision-data-augmentation.png");

	oa::EngineConfig config;
	config.appName = "TutorialVisionDataAugmentation";
	config.presentationMode = oa::PresentationMode::Headless;
	auto created = oa::Engine::create(config);
	if (not created.isOk()) {
		oa::print(oa::PrintStream::Error, "Engine creation failed: {}",
			created.getStatus().toString().cStr());
		return 1;
	}
	auto engine = oa::move(created).getValue();

	auto decoded = oa::FnImage::decodeFile(input);
	if (not decoded.isOk()) {
		oa::print(oa::PrintStream::Error, "Image decode failed: {}",
			decoded.getStatus().toString().cStr());
		return 1;
	}

	// OA_DOC_BEGIN: vision-data-augmentation
	oa::Vector<oa::Image> views;
	views.reserve(6);
	views.pushBack(oa::move(decoded).getValue());
	const oa::Matrix& source = views.front().asMatrix();

	views.pushBack(rgbImage(oa::FnImage::flip(source, true, false)));
	auto crop = oa::FnImage::centerCrop(source, 280, 150);
	views.pushBack(rgbImage(oa::FnImage::resize(crop, 320, 180)));
	views.pushBack(rgbImage(
		oa::FnImage::brightnessContrast(source, 0.08F, 1.15F)));
	auto noisy = oa::FnImage::gaussianNoise(source, 0.0F, 0.035F, 2026U);
	views.pushBack(rgbImage(oa::FnImage::clamp(noisy, 0.0F, 1.0F)));
	views.pushBack(rgbImage(oa::FnImage::solarize(source, 0.55F, 1.0F)));
	// OA_DOC_END: vision-data-augmentation

	const char* titles[] = {
		"source", "horizontal flip", "crop + resize",
		"brightness + contrast", "seeded Gaussian noise", "solarize",
	};
	oa::plot::Figure figure({
		.title = "OA GPU data augmentation",
		.rows = 2,
		.cols = 3,
		.width = 1200,
		.height = 760,
		.hSpacing = 18,
		.vSpacing = 20,
		.padding = 20,
		.theme = oa::plot::Theme::Dark,
	});

	oa::Vector<oa::Texture> textures;
	textures.reserve(views.size());
	for (oa::Usize index = 0; index < views.size(); ++index) {
		auto texture = oa::FnTexture::fromImage(*engine, views[index]);
		if (not texture.isOk()) {
			oa::print(oa::PrintStream::Error, "texture conversion failed: {}",
				texture.getStatus().toString().cStr());
			return 1;
		}
		textures.pushBack(oa::move(texture).getValue());
		auto& axes = figure.ax(
			static_cast<oa::I32>(index / 3),
			static_cast<oa::I32>(index % 3));
		axes.imshow(textures.back());
		axes.title(titles[index]);
		axes.grid(false);
		axes.legend(false);
	}

	const oa::Path outputDirectory = output.parentPath();
	if (not outputDirectory.empty()) {
		if (auto status = oa::Filesystem::createDirectories(outputDirectory);
			not status.isOk()) {
			oa::print(oa::PrintStream::Error, "output directory creation failed: {}",
				status.toString().cStr());
			return 1;
		}
	}
	if (auto status = figure.saveTo(*engine, output.cStr()); not status.isOk()) {
		oa::print(oa::PrintStream::Error, "Figure save failed: {}",
			status.toString().cStr());
		return 1;
	}

	oa::print("saved 6 GPU augmentation views to {}", output.cStr());
	return 0;
}
