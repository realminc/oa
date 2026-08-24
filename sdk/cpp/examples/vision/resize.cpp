// OA_DOC_BEGIN: vision-resize
#include <oa/core/fnMatrix.h>
#include <oa/core/image.h>
#include <oa/runtime/engine.h>
#include <oa/vision/fnImage.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <utility>

int main() {
	oa::EngineConfig config;
	config.appName = "ExampleVisionResize";
	config.presentationMode = oa::PresentationMode::None;

	auto created = oa::Engine::create(config);
	if (not created.isOk()) return 1;
	auto engine = std::move(created).getValue();

	oa::Image image(
		oa::FnMatrix::full({1, 3, 2, 2}, 0.25F),
		oa::ImageLayout::Nchw,
		oa::ImageFormat::Rgb);
	auto resized = oa::FnImage::resize(image, 4, 3);

	auto submitted = engine->submit();
	if (not submitted.isOk()) return 1;
	if (not engine->wait(submitted.getValue()).isOk()) return 1;
	if (not resized.validate() || resized.width() != 4 || resized.height() != 3) {
		return 1;
	}

	std::array<oa::F32, 36> values{};
	if (not oa::FnMatrix::copyToHost(
		resized.asMatrix(), values.data(), sizeof(values)).isOk()) return 1;
	for (const oa::F32 value : values) {
		if (std::abs(value - 0.25F) > 1.0e-6F) return 1;
	}

	std::puts("RGB NCHW image resized from 2x2 to 4x3");
	return 0;
}
// OA_DOC_END: vision-resize
