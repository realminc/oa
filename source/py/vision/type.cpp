// OA Python bindings — Vision enums, metadata, and configuration records.
#include "../binding.h"

#include <oa/core/image.h>
#include <oa/core/type.h>
#include <oa/vision/cameraCapture.h>
#include <oa/vision/fnImage.h>
#include <oa/vision/screenCapture.h>
#include <oa/vision/videoPlayer.h>
#include <oa/vision/videoEncoder.h>
#include <oa/vision/videoRecorder.h>

void bindVisionType(nb::module_& m) {
    nb::enum_<oa::ImageLayout>(m, "ImageLayout")
        .value("Nchw", oa::ImageLayout::Nchw)
        .value("Nhwc", oa::ImageLayout::Nhwc)
        .value("Chw", oa::ImageLayout::Chw)
        .value("Hwc", oa::ImageLayout::Hwc)
        .value("Hw", oa::ImageLayout::Hw);

    nb::enum_<oa::ImageFormat>(m, "ImageFormat")
        .value("Gray", oa::ImageFormat::Gray)
        .value("GrayAlpha", oa::ImageFormat::GrayAlpha)
        .value("Rgb", oa::ImageFormat::Rgb)
        .value("Rgba", oa::ImageFormat::Rgba)
        .value("Bgr", oa::ImageFormat::Bgr)
        .value("Bgra", oa::ImageFormat::Bgra);

    // Tensor processing formats from oa::FnImage. These are intentionally kept
    // separate from oa::ImageFormat, which describes semantic channel meaning.
    nb::enum_<oa::PixelFormat>(m, "PixelFormat")
        .value("NV12", oa::PixelFormat::NV12)
        .value("RGB8", oa::PixelFormat::RGB8)
        .value("RGBA8", oa::PixelFormat::RGBA8)
        .value("BF16", oa::PixelFormat::BF16)
        .value("F32", oa::PixelFormat::F32);

    nb::enum_<oa::InterpolationMode>(m, "InterpolationMode")
        .value("Nearest", oa::InterpolationMode::Nearest)
        .value("Bilinear", oa::InterpolationMode::Bilinear)
        .value("Bicubic", oa::InterpolationMode::Bicubic);

    nb::enum_<oa::BorderMode>(m, "BorderMode")
        .value("Constant", oa::BorderMode::Constant)
        .value("Replicate", oa::BorderMode::Replicate)
        .value("Reflect", oa::BorderMode::Reflect)
        .value("Reflect101", oa::BorderMode::Reflect101)
        .value("Wrap", oa::BorderMode::Wrap);

    nb::enum_<oa::Filter>(m, "Filter")
        .value("Nearest", oa::Filter::Nearest)
        .value("Linear", oa::Filter::Linear);

    nb::enum_<oa::VideoCodec>(m, "VideoCodec")
        .value("H264", oa::VideoCodec::H264)
        .value("H265", oa::VideoCodec::H265)
        .value("AV1", oa::VideoCodec::AV1)
        .value("VP9", oa::VideoCodec::VP9);

    nb::enum_<oa::YCbCrModel>(m, "YCbCrModel")
        .value("Auto", oa::YCbCrModel::Auto)
        .value("BT709", oa::YCbCrModel::BT709)
        .value("BT2020", oa::YCbCrModel::BT2020);

    nb::enum_<oa::VideoContainerKind>(m, "VideoContainerKind")
        .value("Unknown", oa::VideoContainerKind::Unknown)
        .value("Mp4", oa::VideoContainerKind::Mp4)
        .value("WebM", oa::VideoContainerKind::WebM)
        .value("MpegTs", oa::VideoContainerKind::MpegTs)
        .value("Matroska", oa::VideoContainerKind::Matroska);

    nb::enum_<oa::VideoRateControl>(m, "VideoRateControl")
        .value("ConstantQp", oa::VideoRateControl::ConstantQp)
        .value("Cbr", oa::VideoRateControl::Cbr)
        .value("Vbr", oa::VideoRateControl::Vbr);

    nb::enum_<oa::ScreenCaptureTarget>(m, "ScreenCaptureTarget")
        .value("MonitorOrWindow", oa::ScreenCaptureTarget::MonitorOrWindow)
        .value("Monitor", oa::ScreenCaptureTarget::Monitor)
        .value("Window", oa::ScreenCaptureTarget::Window);

    nb::enum_<oa::ScreenCaptureCursor>(m, "ScreenCaptureCursor")
        .value("Hidden", oa::ScreenCaptureCursor::Hidden)
        .value("Embedded", oa::ScreenCaptureCursor::Embedded);

    nb::class_<oa::NormalizationParams>(m, "NormalizationParams")
        .def(nb::init<>())
        .def_prop_rw("mean",
            [](const oa::NormalizationParams& p) {
                return std::vector<oa::F32>{p.mean[0], p.mean[1], p.mean[2]};
            },
            [](oa::NormalizationParams& p, const std::vector<oa::F32>& values) {
                if (values.size() != 3) {
                    throw std::runtime_error("Mean must contain exactly 3 values");
                }
                for (size_t i = 0; i < 3; ++i) p.mean[i] = values[i];
            })
        .def_prop_rw("std",
            [](const oa::NormalizationParams& p) {
                return std::vector<oa::F32>{p.std[0], p.std[1], p.std[2]};
            },
            [](oa::NormalizationParams& p, const std::vector<oa::F32>& values) {
                if (values.size() != 3) {
                    throw std::runtime_error("Std must contain exactly 3 values");
                }
                for (size_t i = 0; i < 3; ++i) p.std[i] = values[i];
            });

    nb::class_<oa::VideoDemuxerConfig>(m, "VideoDemuxerConfig")
        .def(nb::init<>())
        .def_rw("reconnect", &oa::VideoDemuxerConfig::reconnect)
        .def_rw("maxReconnectAttempts", &oa::VideoDemuxerConfig::maxReconnectAttempts)
        .def_rw("reconnectBackoffMs", &oa::VideoDemuxerConfig::reconnectBackoffMs)
        .def_rw("readTimeoutMs", &oa::VideoDemuxerConfig::readTimeoutMs)
        .def_rw("jitterBufferMs", &oa::VideoDemuxerConfig::jitterBufferMs)
        .def_rw("reorderQueuePackets", &oa::VideoDemuxerConfig::reorderQueuePackets)
        .def_rw("maxTimestampDiscontinuityMs", &oa::VideoDemuxerConfig::maxTimestampDiscontinuityMs)
        .def_prop_rw("rtspTransport",
            [](const oa::VideoDemuxerConfig& v) { return v.rtspTransport.stdStr(); },
            [](oa::VideoDemuxerConfig& v, const std::string& s) {
                v.rtspTransport = oa::String(s);
            });

    nb::class_<oa::VideoPlayerConfig>(m, "VideoPlayerConfig")
        .def(nb::init<>())
        .def_prop_rw("uri",
            [](const oa::VideoPlayerConfig& v) { return v.uri.stdStr(); },
            [](oa::VideoPlayerConfig& v, const std::string& s) { v.uri = oa::String(s); })
        .def_rw("maxDpbSlots", &oa::VideoPlayerConfig::maxDpbSlots)
        .def_rw("loop", &oa::VideoPlayerConfig::loop)
        .def_rw("preferHardwareYCbCr", &oa::VideoPlayerConfig::preferHardwareYCbCr)
        .def_rw("frameRateOverride", &oa::VideoPlayerConfig::frameRateOverride)
        .def_rw("startPlaying", &oa::VideoPlayerConfig::startPlaying)
        .def_rw("audio", &oa::VideoPlayerConfig::audio)
        .def_rw("demuxerConfig", &oa::VideoPlayerConfig::demuxerConfig)
        .def_rw("reorderDepth", &oa::VideoPlayerConfig::reorderDepth)
        .def_rw("filter", &oa::VideoPlayerConfig::filter);

    nb::class_<oa::VideoEncodeProfile>(m, "VideoEncodeProfile")
        .def(nb::init<>())
        .def_rw("codec", &oa::VideoEncodeProfile::codec)
        .def_rw("width", &oa::VideoEncodeProfile::width)
        .def_rw("height", &oa::VideoEncodeProfile::height)
        .def_rw("rateControl", &oa::VideoEncodeProfile::rateControl)
        .def_rw("bitrate", &oa::VideoEncodeProfile::bitrate)
        .def_rw("maxBitrate", &oa::VideoEncodeProfile::maxBitrate)
        .def_rw("constantQp", &oa::VideoEncodeProfile::constantQp)
        .def_rw("frameRate", &oa::VideoEncodeProfile::frameRate)
        .def_rw("gopSize", &oa::VideoEncodeProfile::gopSize)
        .def_rw("maxBFrames", &oa::VideoEncodeProfile::maxBFrames)
        .def_rw("maxDpbSlots", &oa::VideoEncodeProfile::maxDpbSlots)
        .def_rw("qualityLevel", &oa::VideoEncodeProfile::qualityLevel)
        .def_rw("asyncDepth", &oa::VideoEncodeProfile::asyncDepth);

    nb::class_<oa::VideoRecorderConfig>(m, "VideoRecorderConfig")
        .def(nb::init<>())
        .def_prop_rw("outputPath",
            [](const oa::VideoRecorderConfig& v) { return v.outputPath.stdStr(); },
            [](oa::VideoRecorderConfig& v, const std::string& s) {
                v.outputPath = oa::String(s);
            })
        .def_rw("encode", &oa::VideoRecorderConfig::encode)
        .def_rw("colorSpace", &oa::VideoRecorderConfig::colorSpace)
        .def_rw("fullRange", &oa::VideoRecorderConfig::fullRange)
        .def_rw("audioEnabled", &oa::VideoRecorderConfig::audioEnabled);

    nb::class_<oa::ScreenCaptureConfig>(m, "ScreenCaptureConfig")
        .def(nb::init<>())
        .def_rw("target", &oa::ScreenCaptureConfig::target)
        .def_rw("cursor", &oa::ScreenCaptureConfig::cursor)
        .def_rw("preferredWidth", &oa::ScreenCaptureConfig::preferredWidth)
        .def_rw("preferredHeight", &oa::ScreenCaptureConfig::preferredHeight)
        .def_rw("preferredFps", &oa::ScreenCaptureConfig::preferredFps)
        .def_rw("ringFrames", &oa::ScreenCaptureConfig::ringFrames);

    nb::class_<oa::CameraCaptureConfig>(m, "CameraCaptureConfig")
        .def(nb::init<>())
        .def_rw("deviceIndex", &oa::CameraCaptureConfig::deviceIndex)
        .def_rw("width", &oa::CameraCaptureConfig::width)
        .def_rw("height", &oa::CameraCaptureConfig::height)
        .def_rw("fps", &oa::CameraCaptureConfig::fps)
        .def_rw("ringFrames", &oa::CameraCaptureConfig::ringFrames)
        .def_prop_rw("devicePath",
            [](const oa::CameraCaptureConfig& v) { return v.devicePath.stdStr(); },
            [](oa::CameraCaptureConfig& v, const std::string& s) {
                v.devicePath = oa::String(s);
            })
        .def_rw("preferDmaBuf", &oa::CameraCaptureConfig::preferDmaBuf)
        .def_rw("reconnectAttempts", &oa::CameraCaptureConfig::reconnectAttempts)
        .def_rw("reconnectBackoffMs", &oa::CameraCaptureConfig::reconnectBackoffMs);
}
