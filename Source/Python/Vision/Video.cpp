// OA Python bindings — high-level video, capture, and recording.
#include "../Binding.h"

#include <Oa/Vision/CameraCapture.h>
#include <Oa/Vision/ScreenCapture.h>
#include <Oa/Vision/Video.h>
#include <Oa/Vision/VideoEncoder.h>
#include <Oa/Vision/VideoRecorder.h>

namespace {

nb::bytes bytes_from(const OaVec<OaU8>& data) {
    return nb::bytes(reinterpret_cast<const char*>(data.data()), data.size());
}

OaVideo* open_video(const OaVideoConfig& config) {
    auto result = OaVideo::Open(PythonEngine(), config);
    if (result.IsError()) {
        throw std::runtime_error(result.GetStatus().ToString().c_str());
    }
    return new OaVideo(std::move(result).GetValue());
}

} // namespace

void BindVisionVideo(nb::module_& m) {
    nb::enum_<OaVideoFrameResource>(m, "OaVideoFrameResource")
        .value("Image", OaVideoFrameResource::Image)
        .value("Buffer", OaVideoFrameResource::Buffer);

    nb::class_<OaVideoFrame>(m, "OaVideoFrame")
        .def(nb::init<>())
        .def_ro("Resource", &OaVideoFrame::Resource)
        .def_ro("Width", &OaVideoFrame::Width)
        .def_ro("Height", &OaVideoFrame::Height)
        .def_ro("PresentationTimestamp", &OaVideoFrame::PresentationTimestamp)
        .def_ro("Duration", &OaVideoFrame::Duration)
        .def_ro("IsRgb", &OaVideoFrame::IsRgb)
        .def_ro("ColorSpace", &OaVideoFrame::ColorSpace)
        .def_ro("FullRange", &OaVideoFrame::FullRange)
        .def_ro("ArrayLayer", &OaVideoFrame::ArrayLayer)
        .def_ro("Shown", &OaVideoFrame::Shown);

    nb::class_<OaVideoDecodeCapabilities>(m, "OaVideoDecodeCapabilities")
        .def_ro("Supported", &OaVideoDecodeCapabilities::Supported)
        .def_ro("SupportsDpbAndOutputCoincide", &OaVideoDecodeCapabilities::SupportsDpbAndOutputCoincide)
        .def_ro("SupportsDpbAndOutputDistinct", &OaVideoDecodeCapabilities::SupportsDpbAndOutputDistinct)
        .def_ro("SupportsNv12Dpb", &OaVideoDecodeCapabilities::SupportsNv12Dpb)
        .def_ro("MaxWidth", &OaVideoDecodeCapabilities::MaxWidth)
        .def_ro("MaxHeight", &OaVideoDecodeCapabilities::MaxHeight)
        .def_ro("MinWidth", &OaVideoDecodeCapabilities::MinWidth)
        .def_ro("MinHeight", &OaVideoDecodeCapabilities::MinHeight)
        .def_ro("MaxDpbSlots", &OaVideoDecodeCapabilities::MaxDpbSlots)
        .def_ro("MaxActiveReferencePictures", &OaVideoDecodeCapabilities::MaxActiveReferencePictures);

    nb::class_<OaVideoEncodeCapabilities>(m, "OaVideoEncodeCapabilities")
        .def_ro("Supported", &OaVideoEncodeCapabilities::Supported)
        .def_ro("MaxWidth", &OaVideoEncodeCapabilities::MaxWidth)
        .def_ro("MaxHeight", &OaVideoEncodeCapabilities::MaxHeight)
        .def_ro("MinWidth", &OaVideoEncodeCapabilities::MinWidth)
        .def_ro("MinHeight", &OaVideoEncodeCapabilities::MinHeight)
        .def_ro("MaxDpbSlots", &OaVideoEncodeCapabilities::MaxDpbSlots)
        .def_ro("MaxActiveReferencePictures", &OaVideoEncodeCapabilities::MaxActiveReferencePictures)
        .def_ro("MaxBitrate", &OaVideoEncodeCapabilities::MaxBitrate)
        .def_ro("MaxQualityLevels", &OaVideoEncodeCapabilities::MaxQualityLevels)
        .def_ro("MaxH264SliceCount", &OaVideoEncodeCapabilities::MaxH264SliceCount)
        .def_ro("MaxH265SliceSegmentCount", &OaVideoEncodeCapabilities::MaxH265SliceSegmentCount)
        .def_ro("MinH265Qp", &OaVideoEncodeCapabilities::MinH265Qp)
        .def_ro("MaxH265Qp", &OaVideoEncodeCapabilities::MaxH265Qp);

    m.def("QueryDecodeCapabilities", [](OaVideoCodec codec) {
        auto result = OaVideoDecoder::QueryDecodeCapabilities(PythonEngine(), codec);
        if (result.IsError()) {
            throw std::runtime_error(result.GetStatus().ToString().c_str());
        }
        return std::move(result).GetValue();
    }, nb::arg("Codec"));

    m.def("QueryEncodeCapabilities", [](OaVideoCodec codec) {
        auto result = OaVideoEncoder::QueryEncodeCapabilities(PythonEngine(), codec);
        if (result.IsError()) {
            throw std::runtime_error(result.GetStatus().ToString().c_str());
        }
        return std::move(result).GetValue();
    }, nb::arg("Codec"));

    nb::class_<OaVideo>(m, "OaVideo")
        .def_static("Open", &open_video, nb::arg("Config"), nb::rv_policy::take_ownership)
        .def("Next", [](OaVideo& video) { throw_if_error(video.Next()); })
        .def("Reset", &OaVideo::Reset)
        .def("Play", &OaVideo::Play)
        .def("Pause", &OaVideo::Pause)
        .def("TogglePlay", &OaVideo::TogglePlay)
        .def("IsPlaying", &OaVideo::IsPlaying)
        .def("IsDone", &OaVideo::IsDone)
        .def("IsEos", &OaVideo::IsEos)
        .def("HasAudio", &OaVideo::HasAudio)
        .def("StepForward", [](OaVideo& video) { throw_if_error(video.StepForward()); })
        .def("StepBackward", [](OaVideo& video) { throw_if_error(video.StepBackward()); })
        .def("StepFrames", [](OaVideo& video, OaI32 count) {
            throw_if_error(video.StepFrames(count));
        }, nb::arg("Count"))
        .def("Seek", [](OaVideo& video, OaU64 timestamp) {
            throw_if_error(video.Seek(timestamp));
        }, nb::arg("Timestamp"))
        .def("Flush", [](OaVideo& video) { throw_if_error(video.Flush()); })
        .def("Tick", &OaVideo::Tick, nb::arg("DeltaMs"))
        .def("CurrentFrame", &OaVideo::CurrentFrame, nb::rv_policy::reference_internal)
        .def("CurrentFrameToMatrix", [](OaVideo& video, bool normalize_imagenet) {
            auto result = video.CurrentFrameToMatrix(normalize_imagenet);
            if (result.IsError()) {
                throw std::runtime_error(result.GetStatus().ToString().c_str());
            }
            return matrix_ptr(OaStdMove(result).GetValue());
        }, nb::arg("NormalizeImagenet") = true, nb::rv_policy::take_ownership)
        .def("CurrentFrameToImage", [](OaVideo& video, bool normalize_imagenet) {
            auto result = video.CurrentFrameToImage(normalize_imagenet);
            if (result.IsError()) {
                throw std::runtime_error(result.GetStatus().ToString().c_str());
            }
            return new OaImage(OaStdMove(result).GetValue());
        }, nb::arg("NormalizeImagenet") = true, nb::rv_policy::take_ownership)
        .def("ReadbackCurrentRgba", [](OaVideo& video) {
            auto result = video.ReadbackCurrentRgba();
            if (result.IsError()) {
                throw std::runtime_error(result.GetStatus().ToString().c_str());
            }
            return bytes_from(result.GetValue());
        })
        .def("Index", &OaVideo::Index)
        .def("Width", &OaVideo::Width)
        .def("Height", &OaVideo::Height)
        .def("FrameRate", &OaVideo::FrameRate)
        .def("FrameCount", &OaVideo::FrameCount)
        .def("FrameIntervalMs", &OaVideo::FrameIntervalMs)
        .def("Close", [](OaVideo& video) { throw_if_error(video.Close()); })
        .def("Destroy", &OaVideo::Destroy);

    nb::class_<OaVideoRecorder>(m, "OaVideoRecorder")
        .def_static("Create", [](const OaVideoRecorderConfig& config) {
            auto result = OaVideoRecorder::Create(PythonEngine(), config);
            if (result.IsError()) {
                throw std::runtime_error(result.GetStatus().ToString().c_str());
            }
            return new OaVideoRecorder(std::move(result).GetValue());
        }, nb::arg("Config"), nb::rv_policy::take_ownership)
        .def("Write", [](OaVideoRecorder& recorder, const OaVideoFrame& frame) {
            throw_if_error(recorder.Write(frame));
        }, nb::arg("Frame"))
        .def("Finalize", [](OaVideoRecorder& recorder) {
            throw_if_error(recorder.Finalize());
        })
        .def("Destroy", &OaVideoRecorder::Destroy)
        .def("IsOpen", &OaVideoRecorder::IsOpen)
        .def("FrameCount", &OaVideoRecorder::GetFrameCount)
        .def("Config", &OaVideoRecorder::GetConfig, nb::rv_policy::reference_internal);

    nb::class_<OaScreenCapture>(m, "OaScreenCapture")
        .def_static("IsSupported", &OaScreenCapture::IsSupported)
        .def_static("Open", [](const OaScreenCaptureConfig& config) {
            auto result = OaScreenCapture::Open(PythonEngine(), config);
            if (result.IsError()) {
                throw std::runtime_error(result.GetStatus().ToString().c_str());
            }
            return new OaScreenCapture(std::move(result).GetValue());
        }, nb::arg("Config") = OaScreenCaptureConfig(), nb::rv_policy::take_ownership)
        .def("Poll", [](OaScreenCapture& capture) -> nb::object {
            OaVideoFrame frame;
            if (!capture.Poll(frame)) return nb::none();
            return nb::cast(new OaVideoFrame(frame), nb::rv_policy::take_ownership);
        })
        .def("Release", [](OaScreenCapture& capture, const OaVideoFrame& frame) {
            capture.Release(frame);
        }, nb::arg("Frame"))
        .def("Close", [](OaScreenCapture& capture) {
            throw_if_error(capture.Close());
        })
        .def("Destroy", &OaScreenCapture::Destroy)
        .def("IsStreaming", &OaScreenCapture::IsStreaming)
        .def("Width", &OaScreenCapture::Width)
        .def("Height", &OaScreenCapture::Height);

    nb::class_<OaCameraCapture>(m, "OaCameraCapture")
        .def(nb::init<>())
        .def("Init", [](OaCameraCapture& capture, const OaCameraCaptureConfig& config) {
            throw_if_error(capture.Init(PythonEngine(), config));
        }, nb::arg("Config") = OaCameraCaptureConfig())
        .def("Poll", [](OaCameraCapture& capture) -> nb::object {
            OaVideoFrame frame;
            if (!capture.PollFrame(frame)) return nb::none();
            return nb::cast(new OaVideoFrame(frame), nb::rv_policy::take_ownership);
        })
        .def("Release", [](OaCameraCapture& capture, const OaVideoFrame& frame) {
            capture.Release(frame);
        }, nb::arg("Frame"))
        .def("Close", [](OaCameraCapture& capture) {
            throw_if_error(capture.Close());
        })
        .def("Destroy", &OaCameraCapture::Destroy)
        .def("Width", &OaCameraCapture::Width)
        .def("Height", &OaCameraCapture::Height)
        .def("Fps", &OaCameraCapture::Fps)
        .def("IsStreaming", &OaCameraCapture::IsStreaming)
        .def("UsesDmaBuf", &OaCameraCapture::UsesDmaBuf)
        .def("FormatGeneration", &OaCameraCapture::FormatGeneration)
        .def("ReconnectCount", &OaCameraCapture::ReconnectCount);
}
