// OA Python bindings — high-level video, capture, and recording.
#include "../binding.h"

#include <oa/vision/cameraCapture.h>
#include <oa/vision/screenCapture.h>
#include <oa/vision/videoPlayer.h>
#include <oa/vision/videoEncoder.h>
#include <oa/vision/videoRecorder.h>

namespace {

nb::bytes bytesFrom(const oa::Vector<oa::U8>& data) {
    return nb::bytes(reinterpret_cast<const char*>(data.data()), data.size());
}

oa::VideoPlayer* openVideoPlayer(const oa::VideoPlayerConfig& config) {
    auto result = oa::VideoPlayer::open(pythonEngine(), config);
    if (result.isError()) {
        throw std::runtime_error(result.getStatus().toString().cStr());
    }
    return new oa::VideoPlayer(std::move(result).getValue());
}

} // namespace

void bindVisionVideo(nb::module_& m) {
    nb::enum_<oa::VideoFrameResource>(m, "VideoFrameResource")
        .value("Image", oa::VideoFrameResource::Image)
        .value("Buffer", oa::VideoFrameResource::Buffer);

    nb::class_<oa::VideoFrame>(m, "VideoFrame")
        .def(nb::init<>())
        .def_ro("resource", &oa::VideoFrame::resource)
        .def_ro("width", &oa::VideoFrame::width)
        .def_ro("height", &oa::VideoFrame::height)
        .def_ro("presentationTimestamp", &oa::VideoFrame::presentationTimestamp)
        .def_ro("duration", &oa::VideoFrame::duration)
        .def_ro("isRgb", &oa::VideoFrame::isRgb)
        .def_ro("colorSpace", &oa::VideoFrame::colorSpace)
        .def_ro("fullRange", &oa::VideoFrame::fullRange)
        .def_ro("arrayLayer", &oa::VideoFrame::arrayLayer)
        .def_ro("shown", &oa::VideoFrame::shown);

    nb::class_<oa::VideoDecodeCapabilities>(m, "VideoDecodeCapabilities")
        .def_ro("supported", &oa::VideoDecodeCapabilities::supported)
        .def_ro("supportsDpbAndOutputCoincide", &oa::VideoDecodeCapabilities::supportsDpbAndOutputCoincide)
        .def_ro("supportsDpbAndOutputDistinct", &oa::VideoDecodeCapabilities::supportsDpbAndOutputDistinct)
        .def_ro("supportsNv12Dpb", &oa::VideoDecodeCapabilities::supportsNv12Dpb)
        .def_ro("maxWidth", &oa::VideoDecodeCapabilities::maxWidth)
        .def_ro("maxHeight", &oa::VideoDecodeCapabilities::maxHeight)
        .def_ro("minWidth", &oa::VideoDecodeCapabilities::minWidth)
        .def_ro("minHeight", &oa::VideoDecodeCapabilities::minHeight)
        .def_ro("maxDpbSlots", &oa::VideoDecodeCapabilities::maxDpbSlots)
        .def_ro("maxActiveReferencePictures", &oa::VideoDecodeCapabilities::maxActiveReferencePictures);

    nb::class_<oa::VideoEncodeCapabilities>(m, "VideoEncodeCapabilities")
        .def_ro("supported", &oa::VideoEncodeCapabilities::supported)
        .def_ro("maxWidth", &oa::VideoEncodeCapabilities::maxWidth)
        .def_ro("maxHeight", &oa::VideoEncodeCapabilities::maxHeight)
        .def_ro("minWidth", &oa::VideoEncodeCapabilities::minWidth)
        .def_ro("minHeight", &oa::VideoEncodeCapabilities::minHeight)
        .def_ro("maxDpbSlots", &oa::VideoEncodeCapabilities::maxDpbSlots)
        .def_ro("maxActiveReferencePictures", &oa::VideoEncodeCapabilities::maxActiveReferencePictures)
        .def_ro("maxBitrate", &oa::VideoEncodeCapabilities::maxBitrate)
        .def_ro("maxQualityLevels", &oa::VideoEncodeCapabilities::maxQualityLevels)
        .def_ro("maxH264SliceCount", &oa::VideoEncodeCapabilities::maxH264SliceCount)
        .def_ro("maxH265SliceSegmentCount", &oa::VideoEncodeCapabilities::maxH265SliceSegmentCount)
        .def_ro("minH265Qp", &oa::VideoEncodeCapabilities::minH265Qp)
        .def_ro("maxH265Qp", &oa::VideoEncodeCapabilities::maxH265Qp);

    m.def("queryDecodeCapabilities", [](oa::VideoCodec codec) {
        auto result = oa::VideoDecoder::queryDecodeCapabilities(pythonEngine(), codec);
        if (result.isError()) {
            throw std::runtime_error(result.getStatus().toString().cStr());
        }
        return std::move(result).getValue();
    }, nb::arg("codec"));

    m.def("queryEncodeCapabilities", [](oa::VideoCodec codec) {
        auto result = oa::VideoEncoder::queryEncodeCapabilities(pythonEngine(), codec);
        if (result.isError()) {
            throw std::runtime_error(result.getStatus().toString().cStr());
        }
        return std::move(result).getValue();
    }, nb::arg("codec"));

    nb::class_<oa::VideoPlayer>(m, "VideoPlayer")
        .def_static("open", &openVideoPlayer, nb::arg("config"), nb::rv_policy::take_ownership)
        .def("next", [](oa::VideoPlayer& video) { throwIfError(video.next()); })
        .def("reset", &oa::VideoPlayer::reset)
        .def("play", &oa::VideoPlayer::play)
        .def("pause", &oa::VideoPlayer::pause)
        .def("togglePlay", &oa::VideoPlayer::togglePlay)
        .def("isPlaying", &oa::VideoPlayer::isPlaying)
        .def("isDone", &oa::VideoPlayer::isDone)
        .def("isEos", &oa::VideoPlayer::isEos)
        .def("hasAudio", &oa::VideoPlayer::hasAudio)
        .def("stepBackward", [](oa::VideoPlayer& video) { throwIfError(video.stepBackward()); })
        .def("stepFrames", [](oa::VideoPlayer& video, oa::I32 count) {
            throwIfError(video.stepFrames(count));
        }, nb::arg("count"))
        .def("seekFrame", [](oa::VideoPlayer& video, oa::Usize frameIndex) {
            throwIfError(video.seekFrame(frameIndex));
        }, nb::arg("frameIndex"))
        .def("seek", [](oa::VideoPlayer& video, oa::U64 timestamp) {
            throwIfError(video.seek(timestamp));
        }, nb::arg("timestamp"))
        .def("flush", [](oa::VideoPlayer& video) { throwIfError(video.flush()); })
        .def("tick", &oa::VideoPlayer::tick, nb::arg("deltaMs"))
        .def("currentFrame", &oa::VideoPlayer::currentFrame, nb::rv_policy::reference_internal)
        .def("currentFrameToMatrix", [](oa::VideoPlayer& video, bool normalizeImagenet) {
            auto result = video.currentFrameToMatrix(normalizeImagenet);
            if (result.isError()) {
                throw std::runtime_error(result.getStatus().toString().cStr());
            }
            return matrixPtr(oa::move(result).getValue());
        }, nb::arg("normalizeImagenet") = true, nb::rv_policy::take_ownership)
        .def("currentFrameToImage", [](oa::VideoPlayer& video, bool normalizeImagenet) {
            auto result = video.currentFrameToImage(normalizeImagenet);
            if (result.isError()) {
                throw std::runtime_error(result.getStatus().toString().cStr());
            }
            return new oa::Image(oa::move(result).getValue());
        }, nb::arg("normalizeImagenet") = true, nb::rv_policy::take_ownership)
        .def("readbackCurrentRgba", [](oa::VideoPlayer& video) {
            auto result = video.readbackCurrentRgba();
            if (result.isError()) {
                throw std::runtime_error(result.getStatus().toString().cStr());
            }
            return bytesFrom(result.getValue());
        })
        .def("currentFrameIndex", &oa::VideoPlayer::currentFrameIndex)
        .def("index", &oa::VideoPlayer::index)
        .def("width", &oa::VideoPlayer::width)
        .def("height", &oa::VideoPlayer::height)
        .def("frameRate", &oa::VideoPlayer::frameRate)
		.def("frameCount", &oa::VideoPlayer::frameCount)
		.def("frameIntervalMs", &oa::VideoPlayer::frameIntervalMs)
		.def("close", [](oa::VideoPlayer& video) { throwIfError(video.close()); });

    nb::class_<oa::VideoRecorder>(m, "VideoRecorder")
        .def_static("create", [](const oa::VideoRecorderConfig& config) {
            auto result = oa::VideoRecorder::create(pythonEngine(), config);
            if (result.isError()) {
                throw std::runtime_error(result.getStatus().toString().cStr());
            }
            return new oa::VideoRecorder(std::move(result).getValue());
        }, nb::arg("config"), nb::rv_policy::take_ownership)
        .def("write", [](oa::VideoRecorder& recorder, const oa::VideoFrame& frame) {
            throwIfError(recorder.write(frame));
        }, nb::arg("frame"))
		.def("finalize", [](oa::VideoRecorder& recorder) {
			throwIfError(recorder.finalize());
		})
		.def("close", [](oa::VideoRecorder& recorder) {
			throwIfError(recorder.close());
		})
        .def("isOpen", &oa::VideoRecorder::isOpen)
        .def("frameCount", &oa::VideoRecorder::getFrameCount)
        .def("config", &oa::VideoRecorder::getConfig, nb::rv_policy::reference_internal);

    nb::class_<oa::ScreenCapture>(m, "ScreenCapture")
        .def_static("isSupported", &oa::ScreenCapture::isSupported)
        .def_static("open", [](const oa::ScreenCaptureConfig& config) {
            auto result = oa::ScreenCapture::open(pythonEngine(), config);
            if (result.isError()) {
                throw std::runtime_error(result.getStatus().toString().cStr());
            }
            return new oa::ScreenCapture(std::move(result).getValue());
        }, nb::arg("config") = oa::ScreenCaptureConfig(), nb::rv_policy::take_ownership)
        .def("poll", [](oa::ScreenCapture& capture) -> nb::object {
            oa::VideoFrame frame;
            if (!capture.poll(frame)) return nb::none();
            return nb::cast(new oa::VideoFrame(frame), nb::rv_policy::take_ownership);
        })
        .def("release", [](oa::ScreenCapture& capture, const oa::VideoFrame& frame) {
            capture.release(frame);
        }, nb::arg("frame"))
        .def("close", [](oa::ScreenCapture& capture) {
            throwIfError(capture.close());
        })
        .def("isStreaming", &oa::ScreenCapture::isStreaming)
        .def("width", &oa::ScreenCapture::width)
        .def("height", &oa::ScreenCapture::height);

    nb::class_<oa::CameraCapture>(m, "CameraCapture")
        .def_static("open", [](const oa::CameraCaptureConfig& config) {
            auto result = oa::CameraCapture::open(pythonEngine(), config);
            if (result.isError()) {
                throw std::runtime_error(result.getStatus().toString().cStr());
            }
            return new oa::CameraCapture(std::move(result).getValue());
        }, nb::arg("config") = oa::CameraCaptureConfig(), nb::rv_policy::take_ownership)
        .def("poll", [](oa::CameraCapture& capture) -> nb::object {
            oa::VideoFrame frame;
            if (!capture.pollFrame(frame)) return nb::none();
            return nb::cast(new oa::VideoFrame(frame), nb::rv_policy::take_ownership);
        })
        .def("release", [](oa::CameraCapture& capture, const oa::VideoFrame& frame) {
            capture.release(frame);
        }, nb::arg("frame"))
        .def("close", [](oa::CameraCapture& capture) {
            throwIfError(capture.close());
        })
        .def("width", &oa::CameraCapture::width)
        .def("height", &oa::CameraCapture::height)
        .def("fps", &oa::CameraCapture::fps)
        .def("isStreaming", &oa::CameraCapture::isStreaming)
        .def("usesDmaBuf", &oa::CameraCapture::usesDmaBuf)
        .def("formatGeneration", &oa::CameraCapture::formatGeneration)
        .def("reconnectCount", &oa::CameraCapture::reconnectCount);
}
