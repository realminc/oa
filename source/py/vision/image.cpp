// OA Python bindings — semantic images and GPU image operations.
#include "../binding.h"

#include <oa/core/image.h>
#include <oa/vision/fnImage.h>

namespace {

oa::Image* imagePtr(oa::Image&& inImage) {
	return new oa::Image(std::move(inImage));
}

} // namespace

void bindVisionImage(nb::module_& inModule, nb::module_& m) {
	inModule.def(
		"imageFormatChannels", &oa::imageFormatChannels, nb::arg("format"));

	nb::class_<oa::Image>(inModule, "Image")
		.def(nb::init<>())
		.def("__init__", [](oa::Image* self, const oa::Matrix& data,
				oa::ImageLayout layout, oa::ImageFormat format) {
			new (self) oa::Image(data, layout, format);
		}, nb::arg("data"), nb::arg("layout"), nb::arg("format"))
		.def("asMatrix",
			static_cast<oa::Matrix& (oa::Image::*)()>(&oa::Image::asMatrix),
			nb::rv_policy::reference_internal)
		.def("width", &oa::Image::width)
		.def("height", &oa::Image::height)
		.def("channels", &oa::Image::channels)
		.def("batchSize", &oa::Image::batchSize)
		.def("layout", &oa::Image::layout)
		.def("format", &oa::Image::format)
		.def("dtype", &oa::Image::getDtype)
		.def("isEmpty", &oa::Image::isEmpty)
		.def("validate", &oa::Image::validate);

	nb::class_<oa::ImageBatch>(inModule, "ImageBatch")
		.def(nb::init<>())
		.def("__init__", [](oa::ImageBatch* self, const oa::Matrix& data,
				oa::ImageLayout layout, oa::ImageFormat format) {
			new (self) oa::ImageBatch(data, layout, format);
		}, nb::arg("data"), nb::arg("layout"), nb::arg("format"))
		.def("asMatrix",
			static_cast<oa::Matrix& (oa::ImageBatch::*)()>(&oa::ImageBatch::asMatrix),
			nb::rv_policy::reference_internal)
		.def("batchSize", &oa::ImageBatch::batchSize)
		.def("width", &oa::ImageBatch::width)
		.def("height", &oa::ImageBatch::height)
		.def("channels", &oa::ImageBatch::channels)
		.def("layout", &oa::ImageBatch::layout)
		.def("format", &oa::ImageBatch::format)
		.def("dtype", &oa::ImageBatch::getDtype)
		.def("isEmpty", &oa::ImageBatch::isEmpty)
		.def("validate", &oa::ImageBatch::validate);

	// Tensor-native oa::FnImage registrations are generated from the same
	// schemas as their C++ declarations, contracts, docs, and provenance.
#include "fnImageOps.gen.inl"

	// Semantic-image overloads preserve layout and pixel-format metadata.
	m.def("resize", [](const oa::Image& image, oa::U32 width, oa::U32 height) {
		return imagePtr(oa::FnImage::resize(image, width, height));
	}, nb::arg("image"), nb::arg("width"), nb::arg("height"),
		nb::rv_policy::take_ownership);

	m.def("normalize",
		[](const oa::Image& image, const oa::NormalizationParams& params) {
			return imagePtr(oa::FnImage::normalize(image, params));
		},
		nb::arg("image"), nb::arg("params"), nb::rv_policy::take_ownership);

	m.def("brightnessContrast",
		[](const oa::Image& image, oa::F32 brightness, oa::F32 contrast) {
			return imagePtr(
				oa::FnImage::brightnessContrast(image, brightness, contrast));
		},
		nb::arg("image"), nb::arg("brightness") = 0.0F,
		nb::arg("contrast") = 1.0F, nb::rv_policy::take_ownership);

	m.def("grayscale", [](const oa::Image& image) {
		return imagePtr(oa::FnImage::grayscale(image));
	}, nb::arg("image"), nb::rv_policy::take_ownership,
		"Convert a semantic NCHW RGB/RGBA image to Rec.709 grayscale.");

}
