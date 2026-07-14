// OA Python bindings — Vision enums, metadata, and configuration records.
#include "../Binding.h"

#include <Oa/Core/Image.h>
#include <Oa/Core/Type.h>
#include <Oa/Vision/CameraCapture.h>
#include <Oa/Vision/FnImage.h>
#include <Oa/Vision/ScreenCapture.h>
#include <Oa/Vision/Video.h>
#include <Oa/Vision/VideoEncoder.h>
#include <Oa/Vision/VideoRecorder.h>

void BindVisionType(nb::module_& m) {
    nb::enum_<OaImageLayout>(m, "OaImageLayout")
        .value("Nchw", OaImageLayout::Nchw)
        .value("Nhwc", OaImageLayout::Nhwc)
        .value("Chw", OaImageLayout::Chw)
        .value("Hwc", OaImageLayout::Hwc)
        .value("Hw", OaImageLayout::Hw);

    nb::enum_<OaImageFormat>(m, "OaImageFormat")
        .value("Gray", OaImageFormat::Gray)
        .value("GrayAlpha", OaImageFormat::GrayAlpha)
        .value("Rgb", OaImageFormat::Rgb)
        .value("Rgba", OaImageFormat::Rgba)
        .value("Bgr", OaImageFormat::Bgr)
        .value("Bgra", OaImageFormat::Bgra);

    // Tensor processing formats from OaFnImage. These are intentionally kept
    // separate from OaImageFormat, which describes semantic channel meaning.
    nb::enum_<OaPixelFormat>(m, "OaPixelFormat")
        .value("NV12", OaPixelFormat::NV12)
        .value("RGB8", OaPixelFormat::RGB8)
        .value("RGBA8", OaPixelFormat::RGBA8)
        .value("BF16", OaPixelFormat::BF16)
        .value("F32", OaPixelFormat::F32);

    nb::enum_<OaInterpolationMode>(m, "OaInterpolationMode")
        .value("Nearest", OaInterpolationMode::Nearest)
        .value("Bilinear", OaInterpolationMode::Bilinear)
        .value("Bicubic", OaInterpolationMode::Bicubic);

    nb::enum_<OaBorderMode>(m, "OaBorderMode")
        .value("Constant", OaBorderMode::Constant)
        .value("Replicate", OaBorderMode::Replicate)
        .value("Reflect", OaBorderMode::Reflect)
        .value("Reflect101", OaBorderMode::Reflect101)
        .value("Wrap", OaBorderMode::Wrap);

    nb::enum_<OaFilter>(m, "OaFilter")
        .value("Nearest", OaFilter::Nearest)
        .value("Linear", OaFilter::Linear);

    nb::enum_<OaVideoCodec>(m, "OaVideoCodec")
        .value("H264", OaVideoCodec::H264)
        .value("H265", OaVideoCodec::H265)
        .value("AV1", OaVideoCodec::AV1)
        .value("VP9", OaVideoCodec::VP9);

    nb::enum_<OaYCbCrModel>(m, "OaYCbCrModel")
        .value("Auto", OaYCbCrModel::Auto)
        .value("BT709", OaYCbCrModel::BT709)
        .value("BT2020", OaYCbCrModel::BT2020);

    nb::enum_<OaContainerKind>(m, "OaContainerKind")
        .value("Unknown", OaContainerKind::Unknown)
        .value("Mp4", OaContainerKind::Mp4)
        .value("WebM", OaContainerKind::WebM)
        .value("MpegTs", OaContainerKind::MpegTs)
        .value("Matroska", OaContainerKind::Matroska);

    nb::enum_<OaVideoRateControl>(m, "OaVideoRateControl")
        .value("ConstantQp", OaVideoRateControl::ConstantQp)
        .value("Cbr", OaVideoRateControl::Cbr)
        .value("Vbr", OaVideoRateControl::Vbr);

    nb::enum_<OaScreenCaptureTarget>(m, "OaScreenCaptureTarget")
        .value("MonitorOrWindow", OaScreenCaptureTarget::MonitorOrWindow)
        .value("Monitor", OaScreenCaptureTarget::Monitor)
        .value("Window", OaScreenCaptureTarget::Window);

    nb::enum_<OaScreenCaptureCursor>(m, "OaScreenCaptureCursor")
        .value("Hidden", OaScreenCaptureCursor::Hidden)
        .value("Embedded", OaScreenCaptureCursor::Embedded);

    nb::class_<OaNormalizationParams>(m, "OaNormalizationParams")
        .def(nb::init<>())
        .def_prop_rw("Mean",
            [](const OaNormalizationParams& p) {
                return std::vector<OaF32>{p.Mean[0], p.Mean[1], p.Mean[2]};
            },
            [](OaNormalizationParams& p, const std::vector<OaF32>& values) {
                if (values.size() != 3) {
                    throw std::runtime_error("Mean must contain exactly 3 values");
                }
                for (size_t i = 0; i < 3; ++i) p.Mean[i] = values[i];
            })
        .def_prop_rw("Std",
            [](const OaNormalizationParams& p) {
                return std::vector<OaF32>{p.Std[0], p.Std[1], p.Std[2]};
            },
            [](OaNormalizationParams& p, const std::vector<OaF32>& values) {
                if (values.size() != 3) {
                    throw std::runtime_error("Std must contain exactly 3 values");
                }
                for (size_t i = 0; i < 3; ++i) p.Std[i] = values[i];
            });

    nb::class_<OaVideoStreamOptions>(m, "OaVideoStreamOptions")
        .def(nb::init<>())
        .def_rw("Reconnect", &OaVideoStreamOptions::Reconnect)
        .def_rw("MaxReconnectAttempts", &OaVideoStreamOptions::MaxReconnectAttempts)
        .def_rw("ReconnectBackoffMs", &OaVideoStreamOptions::ReconnectBackoffMs)
        .def_rw("ReadTimeoutMs", &OaVideoStreamOptions::ReadTimeoutMs)
        .def_rw("JitterBufferMs", &OaVideoStreamOptions::JitterBufferMs)
        .def_rw("ReorderQueuePackets", &OaVideoStreamOptions::ReorderQueuePackets)
        .def_rw("MaxTimestampDiscontinuityMs", &OaVideoStreamOptions::MaxTimestampDiscontinuityMs)
        .def_prop_rw("RtspTransport",
            [](const OaVideoStreamOptions& v) { return v.RtspTransport.StdStr(); },
            [](OaVideoStreamOptions& v, const std::string& s) {
                v.RtspTransport = OaString(s);
            });

    nb::class_<OaVideoConfig>(m, "OaVideoConfig")
        .def(nb::init<>())
        .def_prop_rw("Uri",
            [](const OaVideoConfig& v) { return v.Uri.StdStr(); },
            [](OaVideoConfig& v, const std::string& s) { v.Uri = OaString(s); })
        .def_prop_rw("Path",
            [](const OaVideoConfig& v) { return v.Path.StdStr(); },
            [](OaVideoConfig& v, const std::string& s) { v.Path = OaString(s); })
        .def_rw("MaxDpbSlots", &OaVideoConfig::MaxDpbSlots)
        .def_rw("Loop", &OaVideoConfig::Loop)
        .def_rw("PreferHardwareYCbCr", &OaVideoConfig::PreferHardwareYCbCr)
        .def_rw("FrameRateOverride", &OaVideoConfig::FrameRateOverride)
        .def_rw("StartPlaying", &OaVideoConfig::StartPlaying)
        .def_rw("Audio", &OaVideoConfig::Audio)
        .def_rw("StreamOptions", &OaVideoConfig::StreamOptions)
        .def_rw("ReorderDepth", &OaVideoConfig::ReorderDepth)
        .def_rw("Filter", &OaVideoConfig::Filter);

    nb::class_<OaVideoEncodeProfile>(m, "OaVideoEncodeProfile")
        .def(nb::init<>())
        .def_rw("Codec", &OaVideoEncodeProfile::Codec)
        .def_rw("Width", &OaVideoEncodeProfile::Width)
        .def_rw("Height", &OaVideoEncodeProfile::Height)
        .def_rw("RateControl", &OaVideoEncodeProfile::RateControl)
        .def_rw("Bitrate", &OaVideoEncodeProfile::Bitrate)
        .def_rw("MaxBitrate", &OaVideoEncodeProfile::MaxBitrate)
        .def_rw("ConstantQp", &OaVideoEncodeProfile::ConstantQp)
        .def_rw("FrameRate", &OaVideoEncodeProfile::FrameRate)
        .def_rw("GopSize", &OaVideoEncodeProfile::GopSize)
        .def_rw("MaxBFrames", &OaVideoEncodeProfile::MaxBFrames)
        .def_rw("MaxDpbSlots", &OaVideoEncodeProfile::MaxDpbSlots)
        .def_rw("QualityLevel", &OaVideoEncodeProfile::QualityLevel)
        .def_rw("AsyncDepth", &OaVideoEncodeProfile::AsyncDepth);

    nb::class_<OaVideoRecorderConfig>(m, "OaVideoRecorderConfig")
        .def(nb::init<>())
        .def_prop_rw("OutputPath",
            [](const OaVideoRecorderConfig& v) { return v.OutputPath.StdStr(); },
            [](OaVideoRecorderConfig& v, const std::string& s) {
                v.OutputPath = OaString(s);
            })
        .def_rw("Encode", &OaVideoRecorderConfig::Encode)
        .def_rw("ColorSpace", &OaVideoRecorderConfig::ColorSpace)
        .def_rw("FullRange", &OaVideoRecorderConfig::FullRange)
        .def_rw("AudioEnabled", &OaVideoRecorderConfig::AudioEnabled);

    nb::class_<OaScreenCaptureConfig>(m, "OaScreenCaptureConfig")
        .def(nb::init<>())
        .def_rw("Target", &OaScreenCaptureConfig::Target)
        .def_rw("Cursor", &OaScreenCaptureConfig::Cursor)
        .def_rw("PreferredWidth", &OaScreenCaptureConfig::PreferredWidth)
        .def_rw("PreferredHeight", &OaScreenCaptureConfig::PreferredHeight)
        .def_rw("PreferredFps", &OaScreenCaptureConfig::PreferredFps)
        .def_rw("RingFrames", &OaScreenCaptureConfig::RingFrames);

    nb::class_<OaCameraCaptureConfig>(m, "OaCameraCaptureConfig")
        .def(nb::init<>())
        .def_rw("DeviceIndex", &OaCameraCaptureConfig::DeviceIndex)
        .def_rw("Width", &OaCameraCaptureConfig::Width)
        .def_rw("Height", &OaCameraCaptureConfig::Height)
        .def_rw("Fps", &OaCameraCaptureConfig::Fps)
        .def_rw("RingFrames", &OaCameraCaptureConfig::RingFrames)
        .def_prop_rw("DevicePath",
            [](const OaCameraCaptureConfig& v) { return v.DevicePath.StdStr(); },
            [](OaCameraCaptureConfig& v, const std::string& s) {
                v.DevicePath = OaString(s);
            })
        .def_rw("PreferDmaBuf", &OaCameraCaptureConfig::PreferDmaBuf)
        .def_rw("ReconnectAttempts", &OaCameraCaptureConfig::ReconnectAttempts)
        .def_rw("ReconnectBackoffMs", &OaCameraCaptureConfig::ReconnectBackoffMs);
}
