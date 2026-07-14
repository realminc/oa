// OA Python bindings — semantic images and GPU image operations.
#include "../Binding.h"

#include <Oa/Core/Image.h>
#include <Oa/Vision/FnImage.h>

namespace {

OaImage* image_ptr(OaImage&& image) {
    return new OaImage(std::move(image));
}

} // namespace

void BindVisionImage(nb::module_& m) {
    m.def("OaImageFormatChannels", &OaImageFormatChannels, nb::arg("format"));

    nb::class_<OaImage>(m, "OaImage")
        .def(nb::init<>())
        .def("__init__", [](OaImage* self, const OaMatrix& data, OaImageLayout layout, OaImageFormat format) {
            new (self) OaImage(data, layout, format);
        }, nb::arg("data"), nb::arg("layout"), nb::arg("format"))
        .def("AsMatrix", static_cast<OaMatrix& (OaImage::*)()>(&OaImage::AsMatrix), nb::rv_policy::reference_internal)
        .def("Width", &OaImage::Width)
        .def("Height", &OaImage::Height)
        .def("Channels", &OaImage::Channels)
        .def("BatchSize", &OaImage::BatchSize)
        .def("Layout", &OaImage::Layout)
        .def("Format", &OaImage::Format)
        .def("Dtype", &OaImage::GetDtype)
        .def("IsEmpty", &OaImage::IsEmpty)
        .def("Validate", &OaImage::Validate);

    nb::class_<OaImageBatch>(m, "OaImageBatch")
        .def(nb::init<>())
        .def("__init__", [](OaImageBatch* self, const OaMatrix& data,
                            OaImageLayout layout, OaImageFormat format) {
            new (self) OaImageBatch(data, layout, format);
        }, nb::arg("data"), nb::arg("layout"), nb::arg("format"))
        .def("AsMatrix", static_cast<OaMatrix& (OaImageBatch::*)()>(&OaImageBatch::AsMatrix),
             nb::rv_policy::reference_internal)
        .def("BatchSize", &OaImageBatch::BatchSize)
        .def("Width", &OaImageBatch::Width)
        .def("Height", &OaImageBatch::Height)
        .def("Channels", &OaImageBatch::Channels)
        .def("Layout", &OaImageBatch::Layout)
        .def("Format", &OaImageBatch::Format)
        .def("Dtype", &OaImageBatch::GetDtype)
        .def("IsEmpty", &OaImageBatch::IsEmpty)
        .def("Validate", &OaImageBatch::Validate);

    m.def("Resize", [](const OaMatrix& image, OaU32 width, OaU32 height,
                        OaInterpolationMode mode) {
        return matrix_ptr(OaFnImage::Resize(
            PythonComputeEngine(), image, width, height, mode));
    }, nb::arg("image"), nb::arg("width"), nb::arg("height"),
       nb::arg("interpolation") = OaInterpolationMode::Bilinear,
       nb::rv_policy::take_ownership);

    m.def("Resize", [](const OaImage& image, OaU32 width, OaU32 height) {
        return image_ptr(OaFnImage::Resize(image, width, height));
    }, nb::arg("image"), nb::arg("width"), nb::arg("height"),
       nb::rv_policy::take_ownership);

    m.def("Normalize", [](const OaMatrix& image, const OaNormalizationParams& params) {
        return matrix_ptr(OaFnImage::Normalize(image, params));
    }, nb::arg("image"), nb::arg("params"), nb::rv_policy::take_ownership);

    m.def("Normalize", [](const OaImage& image, const OaNormalizationParams& params) {
        return image_ptr(OaFnImage::Normalize(image, params));
    }, nb::arg("image"), nb::arg("params"), nb::rv_policy::take_ownership);

    m.def("GaussianBlur", [](const OaMatrix& image, OaF32 sigma, OaU32 kernel_size) {
        return matrix_ptr(OaFnImage::GaussianBlur(image, sigma, kernel_size));
    }, nb::arg("image"), nb::arg("sigma"), nb::arg("kernel_size") = 0,
       nb::rv_policy::take_ownership);

    m.def("ThresholdBinary", [](const OaMatrix& image, OaF32 threshold, OaF32 max_value) {
        return matrix_ptr(OaFnImage::ThresholdBinary(image, threshold, max_value));
    }, nb::arg("image"), nb::arg("threshold"), nb::arg("max_value") = 1.0F,
       nb::rv_policy::take_ownership);
    m.def("ThresholdBinaryInv", [](const OaMatrix& image, OaF32 threshold, OaF32 max_value) {
        return matrix_ptr(OaFnImage::ThresholdBinaryInv(image, threshold, max_value));
    }, nb::arg("image"), nb::arg("threshold"), nb::arg("max_value") = 1.0F,
       nb::rv_policy::take_ownership);
    m.def("ThresholdTruncate", [](const OaMatrix& image, OaF32 threshold) {
        return matrix_ptr(OaFnImage::ThresholdTruncate(image, threshold));
    }, nb::arg("image"), nb::arg("threshold"), nb::rv_policy::take_ownership);
    m.def("ThresholdToZero", [](const OaMatrix& image, OaF32 threshold) {
        return matrix_ptr(OaFnImage::ThresholdToZero(image, threshold));
    }, nb::arg("image"), nb::arg("threshold"), nb::rv_policy::take_ownership);
    m.def("ThresholdToZeroInv", [](const OaMatrix& image, OaF32 threshold) {
        return matrix_ptr(OaFnImage::ThresholdToZeroInv(image, threshold));
    }, nb::arg("image"), nb::arg("threshold"), nb::rv_policy::take_ownership);
    m.def("InRange", [](const OaMatrix& image, OaF32 low, OaF32 high, OaF32 true_value) {
        return matrix_ptr(OaFnImage::InRange(image, low, high, true_value));
    }, nb::arg("image"), nb::arg("low"), nb::arg("high"),
       nb::arg("true_value") = 1.0F, nb::rv_policy::take_ownership);
    m.def("Clamp", [](const OaMatrix& image, OaF32 low, OaF32 high) {
        return matrix_ptr(OaFnImage::Clamp(image, low, high));
    }, nb::arg("image"), nb::arg("low"), nb::arg("high"),
       nb::rv_policy::take_ownership);
    m.def("Invert", [](const OaMatrix& image, OaF32 max_value) {
        return matrix_ptr(OaFnImage::Invert(image, max_value));
    }, nb::arg("image"), nb::arg("max_value") = 1.0F,
       nb::rv_policy::take_ownership);
    m.def("BrightnessContrast", [](const OaMatrix& image, OaF32 brightness, OaF32 contrast) {
        return matrix_ptr(OaFnImage::BrightnessContrast(image, brightness, contrast));
    }, nb::arg("image"), nb::arg("brightness") = 0.0F,
       nb::arg("contrast") = 1.0F, nb::rv_policy::take_ownership);
    m.def("GammaContrast", [](const OaMatrix& image, OaF32 gamma, OaF32 gain) {
        return matrix_ptr(OaFnImage::GammaContrast(image, gamma, gain));
    }, nb::arg("image"), nb::arg("gamma"), nb::arg("gain") = 1.0F,
       nb::rv_policy::take_ownership);
    m.def("Solarize", [](const OaMatrix& image, OaF32 threshold, OaF32 max_value) {
        return matrix_ptr(OaFnImage::Solarize(image, threshold, max_value));
    }, nb::arg("image"), nb::arg("threshold") = 0.5F,
       nb::arg("max_value") = 1.0F, nb::rv_policy::take_ownership);
    m.def("Posterize", [](const OaMatrix& image, OaU32 levels, OaF32 low, OaF32 high) {
        return matrix_ptr(OaFnImage::Posterize(image, levels, low, high));
    }, nb::arg("image"), nb::arg("levels"), nb::arg("low") = 0.0F,
       nb::arg("high") = 1.0F, nb::rv_policy::take_ownership);
    m.def("Grayscale", [](const OaMatrix& image) {
        return matrix_ptr(OaFnImage::Grayscale(image));
    }, nb::arg("image"), nb::rv_policy::take_ownership);
    m.def("ChannelReorder", [](const OaMatrix& image, OaU32 channel0,
                                 OaU32 channel1, OaU32 channel2, OaU32 channel3) {
        return matrix_ptr(OaFnImage::ChannelReorder(
            image, channel0, channel1, channel2, channel3));
    }, nb::arg("image"), nb::arg("channel0"), nb::arg("channel1") = 1,
       nb::arg("channel2") = 2, nb::arg("channel3") = 3,
       nb::rv_policy::take_ownership);
    m.def("AlphaBlend", [](const OaMatrix& a, const OaMatrix& b, OaF32 alpha) {
        return matrix_ptr(OaFnImage::AlphaBlend(a, b, alpha));
    }, nb::arg("a"), nb::arg("b"), nb::arg("alpha"),
       nb::rv_policy::take_ownership);
    m.def("Composite", [](const OaMatrix& a, const OaMatrix& b, const OaMatrix& mask) {
        return matrix_ptr(OaFnImage::Composite(a, b, mask));
    }, nb::arg("a"), nb::arg("b"), nb::arg("mask"),
       nb::rv_policy::take_ownership);
    m.def("Erase", [](const OaMatrix& image, OaU32 x, OaU32 y,
                        OaU32 width, OaU32 height, OaF32 value) {
        return matrix_ptr(OaFnImage::Erase(image, x, y, width, height, value));
    }, nb::arg("image"), nb::arg("x"), nb::arg("y"), nb::arg("width"),
       nb::arg("height"), nb::arg("value") = 0.0F,
       nb::rv_policy::take_ownership);
    m.def("ColorTwist", [](const OaMatrix& image, const OaMatrix& transform) {
        return matrix_ptr(OaFnImage::ColorTwist(image, transform));
    }, nb::arg("image"), nb::arg("transform"), nb::rv_policy::take_ownership);
    m.def("GaussianNoise", [](const OaMatrix& image, OaF32 mean,
                                OaF32 stddev, OaU64 seed) {
        return matrix_ptr(OaFnImage::GaussianNoise(image, mean, stddev, seed));
    }, nb::arg("image"), nb::arg("mean") = 0.0F, nb::arg("stddev") = 0.01F,
       nb::arg("seed") = 0, nb::rv_policy::take_ownership);
    m.def("SaltPepperNoise", [](const OaMatrix& image, OaF32 probability,
                                  OaF32 salt_value, OaF32 pepper_value, OaU64 seed) {
        return matrix_ptr(OaFnImage::SaltPepperNoise(
            image, probability, salt_value, pepper_value, seed));
    }, nb::arg("image"), nb::arg("probability") = 0.01F,
       nb::arg("salt_value") = 1.0F, nb::arg("pepper_value") = 0.0F,
       nb::arg("seed") = 0, nb::rv_policy::take_ownership);

    m.def("Convolve2d", [](const OaMatrix& image, const OaMatrix& kernel,
                            OaBorderMode border, OaF32 border_value) {
        return matrix_ptr(OaFnImage::Convolve2d(image, kernel, border, border_value));
    }, nb::arg("image"), nb::arg("kernel"),
       nb::arg("border") = OaBorderMode::Reflect101,
       nb::arg("border_value") = 0.0F, nb::rv_policy::take_ownership);

    m.def("SeparableConvolve2d", [](const OaMatrix& image,
                                     const OaMatrix& kernel_x,
                                     const OaMatrix& kernel_y,
                                     OaBorderMode border, OaF32 border_value) {
        return matrix_ptr(OaFnImage::SeparableConvolve2d(
            image, kernel_x, kernel_y, border, border_value));
    }, nb::arg("image"), nb::arg("kernel_x"), nb::arg("kernel_y"),
       nb::arg("border") = OaBorderMode::Reflect101,
       nb::arg("border_value") = 0.0F, nb::rv_policy::take_ownership);

    m.def("AverageBlur", [](const OaMatrix& image, OaU32 kernel_width,
                             OaU32 kernel_height, OaBorderMode border) {
        return matrix_ptr(OaFnImage::AverageBlur(
            image, kernel_width, kernel_height, border));
    }, nb::arg("image"), nb::arg("kernel_width") = 3,
       nb::arg("kernel_height") = 3,
       nb::arg("border") = OaBorderMode::Reflect101,
       nb::rv_policy::take_ownership);

    m.def("Sobel", [](const OaMatrix& image, OaU32 dx, OaU32 dy,
                       OaBorderMode border) {
        return matrix_ptr(OaFnImage::Sobel(image, dx, dy, border));
    }, nb::arg("image"), nb::arg("dx") = 1, nb::arg("dy") = 0,
       nb::arg("border") = OaBorderMode::Reflect101,
       nb::rv_policy::take_ownership);

    m.def("Scharr", [](const OaMatrix& image, OaU32 dx, OaU32 dy,
                        OaBorderMode border) {
        return matrix_ptr(OaFnImage::Scharr(image, dx, dy, border));
    }, nb::arg("image"), nb::arg("dx") = 1, nb::arg("dy") = 0,
       nb::arg("border") = OaBorderMode::Reflect101,
       nb::rv_policy::take_ownership);

    m.def("Laplacian", [](const OaMatrix& image, OaBorderMode border) {
        return matrix_ptr(OaFnImage::Laplacian(image, border));
    }, nb::arg("image"), nb::arg("border") = OaBorderMode::Reflect101,
       nb::rv_policy::take_ownership);

    m.def("Erode", [](const OaMatrix& image, OaU32 kernel_width,
                       OaU32 kernel_height, OaBorderMode border, OaF32 border_value) {
        return matrix_ptr(OaFnImage::Erode(
            image, kernel_width, kernel_height, border, border_value));
    }, nb::arg("image"), nb::arg("kernel_width") = 3,
       nb::arg("kernel_height") = 3,
       nb::arg("border") = OaBorderMode::Reflect101,
       nb::arg("border_value") = 0.0F, nb::rv_policy::take_ownership);

    m.def("Dilate", [](const OaMatrix& image, OaU32 kernel_width,
                        OaU32 kernel_height, OaBorderMode border, OaF32 border_value) {
        return matrix_ptr(OaFnImage::Dilate(
            image, kernel_width, kernel_height, border, border_value));
    }, nb::arg("image"), nb::arg("kernel_width") = 3,
       nb::arg("kernel_height") = 3,
       nb::arg("border") = OaBorderMode::Reflect101,
       nb::arg("border_value") = 0.0F, nb::rv_policy::take_ownership);

    m.def("MorphologyOpen", [](const OaMatrix& image, OaU32 kernel_width,
                                OaU32 kernel_height, OaBorderMode border,
                                OaF32 border_value) {
        return matrix_ptr(OaFnImage::MorphologyOpen(
            image, kernel_width, kernel_height, border, border_value));
    }, nb::arg("image"), nb::arg("kernel_width") = 3,
       nb::arg("kernel_height") = 3,
       nb::arg("border") = OaBorderMode::Reflect101,
       nb::arg("border_value") = 0.0F, nb::rv_policy::take_ownership);

    m.def("MorphologyClose", [](const OaMatrix& image, OaU32 kernel_width,
                                 OaU32 kernel_height, OaBorderMode border,
                                 OaF32 border_value) {
        return matrix_ptr(OaFnImage::MorphologyClose(
            image, kernel_width, kernel_height, border, border_value));
    }, nb::arg("image"), nb::arg("kernel_width") = 3,
       nb::arg("kernel_height") = 3,
       nb::arg("border") = OaBorderMode::Reflect101,
       nb::arg("border_value") = 0.0F, nb::rv_policy::take_ownership);

    m.def("MorphologyGradient", [](const OaMatrix& image, OaU32 kernel_width,
                                    OaU32 kernel_height, OaBorderMode border,
                                    OaF32 border_value) {
        return matrix_ptr(OaFnImage::MorphologyGradient(
            image, kernel_width, kernel_height, border, border_value));
    }, nb::arg("image"), nb::arg("kernel_width") = 3,
       nb::arg("kernel_height") = 3,
       nb::arg("border") = OaBorderMode::Reflect101,
       nb::arg("border_value") = 0.0F, nb::rv_policy::take_ownership);

    m.def("Sharpen", [](const OaMatrix& image, OaF32 amount, OaBorderMode border) {
        return matrix_ptr(OaFnImage::Sharpen(image, amount, border));
    }, nb::arg("image"), nb::arg("amount") = 1.0F,
       nb::arg("border") = OaBorderMode::Reflect101,
       nb::rv_policy::take_ownership);
    m.def("MedianBlur", [](const OaMatrix& image, OaU32 kernel_size, OaBorderMode border) {
        return matrix_ptr(OaFnImage::MedianBlur(image, kernel_size, border));
    }, nb::arg("image"), nb::arg("kernel_size") = 3,
       nb::arg("border") = OaBorderMode::Reflect101,
       nb::rv_policy::take_ownership);
    m.def("BilateralFilter", [](const OaMatrix& image, OaU32 kernel_size,
                                  OaF32 sigma_color, OaF32 sigma_space,
                                  OaBorderMode border) {
        return matrix_ptr(OaFnImage::BilateralFilter(
            image, kernel_size, sigma_color, sigma_space, border));
    }, nb::arg("image"), nb::arg("kernel_size") = 5,
       nb::arg("sigma_color") = 0.1F, nb::arg("sigma_space") = 1.0F,
       nb::arg("border") = OaBorderMode::Reflect101,
       nb::rv_policy::take_ownership);
    m.def("UnsharpMask", [](const OaMatrix& image, OaF32 sigma,
                              OaF32 amount, OaU32 kernel_size) {
        return matrix_ptr(OaFnImage::UnsharpMask(image, sigma, amount, kernel_size));
    }, nb::arg("image"), nb::arg("sigma") = 1.0F,
       nb::arg("amount") = 1.0F, nb::arg("kernel_size") = 0,
       nb::rv_policy::take_ownership);
    m.def("MorphologyTopHat", [](const OaMatrix& image, OaU32 kernel_width,
                                   OaU32 kernel_height, OaBorderMode border,
                                   OaF32 border_value) {
        return matrix_ptr(OaFnImage::MorphologyTopHat(
            image, kernel_width, kernel_height, border, border_value));
    }, nb::arg("image"), nb::arg("kernel_width") = 3,
       nb::arg("kernel_height") = 3, nb::arg("border") = OaBorderMode::Reflect101,
       nb::arg("border_value") = 0.0F, nb::rv_policy::take_ownership);
    m.def("MorphologyBlackHat", [](const OaMatrix& image, OaU32 kernel_width,
                                     OaU32 kernel_height, OaBorderMode border,
                                     OaF32 border_value) {
        return matrix_ptr(OaFnImage::MorphologyBlackHat(
            image, kernel_width, kernel_height, border, border_value));
    }, nb::arg("image"), nb::arg("kernel_width") = 3,
       nb::arg("kernel_height") = 3, nb::arg("border") = OaBorderMode::Reflect101,
       nb::arg("border_value") = 0.0F, nb::rv_policy::take_ownership);
    m.def("AdaptiveThresholdMean", [](const OaMatrix& image, OaU32 kernel_size,
                                        OaF32 c, OaF32 max_value, OaBorderMode border) {
        return matrix_ptr(OaFnImage::AdaptiveThresholdMean(
            image, kernel_size, c, max_value, border));
    }, nb::arg("image"), nb::arg("kernel_size") = 11, nb::arg("c") = 0.0F,
       nb::arg("max_value") = 1.0F, nb::arg("border") = OaBorderMode::Reflect101,
       nb::rv_policy::take_ownership);
    m.def("AdaptiveThresholdGaussian", [](const OaMatrix& image, OaU32 kernel_size,
                                            OaF32 c, OaF32 max_value, OaF32 sigma,
                                            OaBorderMode border) {
        return matrix_ptr(OaFnImage::AdaptiveThresholdGaussian(
            image, kernel_size, c, max_value, sigma, border));
    }, nb::arg("image"), nb::arg("kernel_size") = 11, nb::arg("c") = 0.0F,
       nb::arg("max_value") = 1.0F, nb::arg("sigma") = 0.0F,
       nb::arg("border") = OaBorderMode::Reflect101,
       nb::rv_policy::take_ownership);

    m.def("Crop", [](const OaMatrix& image, OaU32 x, OaU32 y,
                      OaU32 width, OaU32 height) {
        return matrix_ptr(OaFnImage::Crop(image, x, y, width, height));
    }, nb::arg("image"), nb::arg("x"), nb::arg("y"),
       nb::arg("width"), nb::arg("height"), nb::rv_policy::take_ownership);

    m.def("Flip", [](const OaMatrix& image, bool horizontal, bool vertical) {
        return matrix_ptr(OaFnImage::Flip(image, horizontal, vertical));
    }, nb::arg("image"), nb::arg("horizontal") = false,
       nb::arg("vertical") = false, nb::rv_policy::take_ownership);

    m.def("Rotate", [](const OaMatrix& image, OaU32 degrees) {
        return matrix_ptr(OaFnImage::Rotate(image, degrees));
    }, nb::arg("image"), nb::arg("degrees"), nb::rv_policy::take_ownership);

    m.def("Pad", [](const OaMatrix& image, OaU32 left, OaU32 right,
                      OaU32 top, OaU32 bottom, OaBorderMode border, OaF32 border_value) {
        return matrix_ptr(OaFnImage::Pad(
            image, left, right, top, bottom, border, border_value));
    }, nb::arg("image"), nb::arg("left"), nb::arg("right"),
       nb::arg("top"), nb::arg("bottom"),
       nb::arg("border") = OaBorderMode::Constant,
       nb::arg("border_value") = 0.0F, nb::rv_policy::take_ownership);
    m.def("CenterCrop", [](const OaMatrix& image, OaU32 width, OaU32 height) {
        return matrix_ptr(OaFnImage::CenterCrop(image, width, height));
    }, nb::arg("image"), nb::arg("width"), nb::arg("height"),
       nb::rv_policy::take_ownership);
    m.def("Remap", [](const OaMatrix& image, const OaMatrix& map,
                        OaInterpolationMode interpolation, OaBorderMode border,
                        OaF32 border_value) {
        return matrix_ptr(OaFnImage::Remap(
            image, map, interpolation, border, border_value));
    }, nb::arg("image"), nb::arg("map"),
       nb::arg("interpolation") = OaInterpolationMode::Bilinear,
       nb::arg("border") = OaBorderMode::Constant,
       nb::arg("border_value") = 0.0F, nb::rv_policy::take_ownership);
    m.def("WarpAffine", [](const OaMatrix& image, const OaMatrix& transform,
                             OaU32 width, OaU32 height,
                             OaInterpolationMode interpolation, OaBorderMode border,
                             OaF32 border_value) {
        return matrix_ptr(OaFnImage::WarpAffine(
            image, transform, width, height, interpolation, border, border_value));
    }, nb::arg("image"), nb::arg("transform"), nb::arg("width"), nb::arg("height"),
       nb::arg("interpolation") = OaInterpolationMode::Bilinear,
       nb::arg("border") = OaBorderMode::Constant,
       nb::arg("border_value") = 0.0F, nb::rv_policy::take_ownership);
    m.def("WarpPerspective", [](const OaMatrix& image, const OaMatrix& transform,
                                  OaU32 width, OaU32 height,
                                  OaInterpolationMode interpolation, OaBorderMode border,
                                  OaF32 border_value) {
        return matrix_ptr(OaFnImage::WarpPerspective(
            image, transform, width, height, interpolation, border, border_value));
    }, nb::arg("image"), nb::arg("transform"), nb::arg("width"), nb::arg("height"),
       nb::arg("interpolation") = OaInterpolationMode::Bilinear,
       nb::arg("border") = OaBorderMode::Constant,
       nb::arg("border_value") = 0.0F, nb::rv_policy::take_ownership);

    m.def("ConvertColor", [](const OaImage& image, OaImageFormat destination) {
        return image_ptr(OaFnImage::ConvertColor(image, destination));
    }, nb::arg("image"), nb::arg("destination"), nb::rv_policy::take_ownership);

    m.def("ResizeNormalize", [](const OaImage& image, OaU32 width, OaU32 height,
                                const OaNormalizationParams& params) {
        return image_ptr(OaFnImage::ResizeNormalize(image, width, height, params));
    }, nb::arg("image"), nb::arg("width"), nb::arg("height"), nb::arg("params"),
       nb::rv_policy::take_ownership);
}
