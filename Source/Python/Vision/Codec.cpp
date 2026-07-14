// OA Python bindings — still-image codec and compressed video stream boundary.
#include "../Binding.h"


#include <Oa/Vision/JpegDecoder.h>
#include <Oa/Vision/VideoStream.h>

namespace {

nb::bytes bytes_from(const OaVec<OaU8>& data) {
    return nb::bytes(reinterpret_cast<const char*>(data.data()), data.size());
}

OaVideoStream* open_stream(const std::string& uri, const OaVideoStreamOptions& options) {
    auto result = OaVideoStream::Open(OaStringView(uri), options);
    if (result.IsError()) {
        throw std::runtime_error(result.GetStatus().ToString().c_str());
    }
    return new OaVideoStream(std::move(result).GetValue());
}

} // namespace

void BindVisionCodec(nb::module_& m) {
    nb::class_<OaJpegDecodeResult>(m, "OaJpegDecodeResult")
        .def_prop_ro("Pixels", [](const OaJpegDecodeResult& result) {
            return bytes_from(result.Pixels);
        })
        .def_ro("Width", &OaJpegDecodeResult::Width)
        .def_ro("Height", &OaJpegDecodeResult::Height)
        .def_ro("Channels", &OaJpegDecodeResult::Channels);

    nb::class_<OaJpegDecoder>(m, "OaJpegDecoder")
        .def_static("Decode", [](nb::bytes data) {
            const auto* ptr = reinterpret_cast<const OaU8*>(data.data());
            return OaJpegDecoder::Decode(OaSpan<const OaU8>(ptr, data.size()));
        }, nb::arg("data"))
        .def_static("DecodeFile", [](const std::string& path) {
            return OaJpegDecoder::DecodeFile(OaStringView(path));
        }, nb::arg("path"))
        .def_static("DecodeToGpu", [](nb::bytes data, OaU32 width, OaU32 height,
                                      bool normalize_imagenet) {
            const auto* ptr = reinterpret_cast<const OaU8*>(data.data());
            return matrix_ptr(OaJpegDecoder::DecodeToGpu(
                PythonComputeEngine(), OaSpan<const OaU8>(ptr, data.size()),
                width, height, normalize_imagenet));
        }, nb::arg("data"), nb::arg("width") = 0, nb::arg("height") = 0,
           nb::arg("normalize_imagenet") = false, nb::rv_policy::take_ownership)
        .def_static("DecodeFileToGpu", [](const std::string& path, OaU32 width,
                                          OaU32 height, bool normalize_imagenet) {
            return matrix_ptr(OaJpegDecoder::DecodeFileToGpu(
                PythonComputeEngine(), OaStringView(path), width, height,
                normalize_imagenet));
        }, nb::arg("path"), nb::arg("width") = 0, nb::arg("height") = 0,
           nb::arg("normalize_imagenet") = false, nb::rv_policy::take_ownership);

    nb::class_<OaContainerInfo>(m, "OaContainerInfo")
        .def(nb::init<>())
        .def_rw("Kind", &OaContainerInfo::Kind)
        .def_rw("Codec", &OaContainerInfo::Codec)
        .def_rw("Width", &OaContainerInfo::Width)
        .def_rw("Height", &OaContainerInfo::Height)
        .def_rw("FrameRate", &OaContainerInfo::FrameRate)
        .def_rw("Duration", &OaContainerInfo::Duration)
        .def_rw("TimebaseNum", &OaContainerInfo::TimebaseNum)
        .def_rw("TimebaseDen", &OaContainerInfo::TimebaseDen)
        .def_rw("TrackCount", &OaContainerInfo::TrackCount);

    nb::class_<OaVideoStreamStats>(m, "OaVideoStreamStats")
        .def_ro("ReconnectCount", &OaVideoStreamStats::ReconnectCount)
        .def_ro("TimestampDiscontinuities", &OaVideoStreamStats::TimestampDiscontinuities)
        .def_ro("FormatGeneration", &OaVideoStreamStats::FormatGeneration);

    nb::class_<OaVideoPacket>(m, "OaVideoPacket")
        .def_prop_ro("Data", [](const OaVideoPacket& packet) {
            return bytes_from(packet.Data);
        })
        .def_ro("PresentationTimestamp", &OaVideoPacket::PresentationTimestamp)
        .def_ro("DecodeTimestamp", &OaVideoPacket::DecodeTimestamp)
        .def_ro("IsKeyframe", &OaVideoPacket::IsKeyframe)
        .def_ro("TrackIndex", &OaVideoPacket::TrackIndex);

    nb::class_<OaVideoStream>(m, "OaVideoStream")
        .def_static("Open", &open_stream, nb::arg("uri"),
                    nb::arg("options") = OaVideoStreamOptions(),
                    nb::rv_policy::take_ownership)
        .def_static("Probe", [](const std::string& path) {
            auto result = OaVideoStream::Probe(path.c_str());
            if (result.IsError()) {
                throw std::runtime_error(result.GetStatus().ToString().c_str());
            }
            return std::move(result).GetValue();
        }, nb::arg("path"))
        .def("ReadNextPacket", [](OaVideoStream& stream) {
            auto* packet = new OaVideoPacket();
            auto status = stream.ReadNextPacket(*packet);
            if (status.IsError()) {
                delete packet;
                throw std::runtime_error(status.ToString().c_str());
            }
            return packet;
        }, nb::rv_policy::take_ownership)
        .def("Seek", [](OaVideoStream& stream, OaU64 timestamp) {
            throw_if_error(stream.Seek(timestamp));
        }, nb::arg("timestamp"))
        .def("Info", &OaVideoStream::GetInfo, nb::rv_policy::reference_internal)
        .def("Stats", &OaVideoStream::GetStats, nb::rv_policy::reference_internal)
        .def("IsEos", &OaVideoStream::IsEos)
        .def("IsLive", &OaVideoStream::IsLive)
        .def("IsSeekable", &OaVideoStream::IsSeekable)
        .def("FormatGeneration", &OaVideoStream::FormatGeneration)
        .def("Destroy", &OaVideoStream::Destroy);
}
