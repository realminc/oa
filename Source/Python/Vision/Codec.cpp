// OA Python bindings — still-image codec and compressed video stream boundary.
#include "../Binding.h"


#include <Oa/Vision/ImageDecoder.h>
#include <Oa/Vision/ImageEncoder.h>
#include <Oa/Vision/VideoStream.h>

namespace {

nb::bytes bytes_from(const OaVec<OaU8>& data) {
	return nb::bytes(reinterpret_cast<const char*>(data.data()), data.size());
}

OaImage* image_from_result(OaResult<OaImage>&& result) {
	if (result.IsError()) {
		throw std::runtime_error(result.GetStatus().ToString().CStr());
	}
	return new OaImage(OaStdMove(result).GetValue());
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
	nb::enum_<OaImageCodec>(m, "OaImageCodec")
		.value("Auto", OaImageCodec::Auto)
		.value("Jpeg", OaImageCodec::Jpeg)
		.value("Png", OaImageCodec::Png)
		.value("Webp", OaImageCodec::Webp)
		.value("Bmp", OaImageCodec::Bmp)
		.value("Tga", OaImageCodec::Tga);

	nb::class_<OaImageDecoder>(m, "OaImageDecoder")
		.def_static("LoadFile", [](nb::handle path, OaImageFormat format) {
			// Decoding uploads the semantic image into device storage. Preserve
			// Python's lazy-runtime contract at the codec boundary.
			(void)PythonEngine();
			return image_from_result(OaImageDecoder::LoadFile(
				path_from_python(path), format));
		}, nb::arg("Path"), nb::arg("Format") = OaImageFormat::Rgb,
			nb::rv_policy::take_ownership)
		.def_static("LoadMemory", [](nb::bytes data, OaImageFormat format) {
			(void)PythonEngine();
			const auto* bytes =
				reinterpret_cast<const OaU8*>(data.data());
			return image_from_result(OaImageDecoder::LoadMemory(
				OaSpan<const OaU8>(bytes, data.size()), format));
		}, nb::arg("Data"), nb::arg("Format") = OaImageFormat::Rgb,
			nb::rv_policy::take_ownership)
		.def_static("Supports", &OaImageDecoder::Supports, nb::arg("Codec"));

	nb::class_<OaImageEncoder>(m, "OaImageEncoder")
		.def_static("Encode", [](const OaImage& image, OaImageCodec codec,
			OaU32 quality) {
			auto result = OaImageEncoder::Encode(image, codec, quality);
			if (result.IsError()) {
				throw std::runtime_error(
					result.GetStatus().ToString().CStr());
			}
			return bytes_from(*result);
		}, nb::arg("Image"), nb::arg("Codec"), nb::arg("Quality") = 90U)
		.def_static("SaveFile", [](nb::handle path, const OaImage& image,
			OaU32 quality) {
			throw_if_error(OaImageEncoder::SaveFile(
				path_from_python(path), image, quality));
		}, nb::arg("Path"), nb::arg("Image"), nb::arg("Quality") = 90U)
		.def_static("Supports", &OaImageEncoder::Supports, nb::arg("Codec"));

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
        .def_static("Open", &open_stream, nb::arg("Uri"),
                    nb::arg("Options") = OaVideoStreamOptions(),
                    nb::rv_policy::take_ownership)
        .def_static("Probe", [](nb::handle path_value) {
            const OaPath path = path_from_python(path_value);
            auto result = OaVideoStream::Probe(path.CStr());
            if (result.IsError()) {
                throw std::runtime_error(result.GetStatus().ToString().c_str());
            }
            return std::move(result).GetValue();
        }, nb::arg("Path"))
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
        }, nb::arg("Timestamp"))
        .def("Info", &OaVideoStream::GetInfo, nb::rv_policy::reference_internal)
        .def("Stats", &OaVideoStream::GetStats, nb::rv_policy::reference_internal)
        .def("IsEos", &OaVideoStream::IsEos)
        .def("IsLive", &OaVideoStream::IsLive)
        .def("IsSeekable", &OaVideoStream::IsSeekable)
        .def("FormatGeneration", &OaVideoStream::FormatGeneration)
        .def("Destroy", &OaVideoStream::Destroy);
}
