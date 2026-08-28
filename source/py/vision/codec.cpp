// OA Python bindings — still-image codec and compressed video stream boundary.
#include "../binding.h"


#include <oa/vision/fnImage.h>
#include <oa/vision/videoDemuxer.h>

namespace {

nb::bytes bytesFrom(const oa::Vector<oa::U8>& data) {
	return nb::bytes(reinterpret_cast<const char*>(data.data()), data.size());
}

oa::Image* imageFromResult(oa::Result<oa::Image>&& result) {
	if (result.isError()) {
		throw std::runtime_error(result.getStatus().toString().cStr());
	}
	return new oa::Image(oa::move(result).getValue());
}

oa::VideoDemuxer* openDemuxer(const std::string& uri, const oa::VideoDemuxerConfig& config) {
	auto result = oa::VideoDemuxer::open(oa::StringView(
		uri.data(), static_cast<oa::Usize>(uri.size())), config);
	if (result.isError()) {
		throw std::runtime_error(result.getStatus().toString().cStr());
	}
	return new oa::VideoDemuxer(std::move(result).getValue());
}

} // namespace

void bindVisionCodec(nb::module_& m, nb::module_& inFnImage) {
	nb::enum_<oa::ImageCodec>(m, "ImageCodec")
		.value("Auto", oa::ImageCodec::Auto)
		.value("Jpeg", oa::ImageCodec::Jpeg)
		.value("Png", oa::ImageCodec::Png)
		.value("Webp", oa::ImageCodec::Webp)
		.value("Bmp", oa::ImageCodec::Bmp)
		.value("Tga", oa::ImageCodec::Tga);

	inFnImage.def("decodeFile", [](nb::handle path, oa::ImageFormat format) {
			// Decoding uploads the semantic image into device storage. Preserve
			// Python's lazy-runtime contract at the codec boundary.
			(void)pythonEngine();
			return imageFromResult(oa::FnImage::decodeFile(
				pathFromPython(path), format));
		}, nb::arg("path"), nb::arg("format") = oa::ImageFormat::Rgb,
			nb::rv_policy::take_ownership);
	inFnImage.def("decodeMemory", [](nb::bytes data, oa::ImageFormat format) {
			(void)pythonEngine();
			const auto* bytes =
				reinterpret_cast<const oa::U8*>(data.data());
			return imageFromResult(oa::FnImage::decodeMemory(
				oa::Span<const oa::U8>(bytes, data.size()), format));
		}, nb::arg("data"), nb::arg("format") = oa::ImageFormat::Rgb,
			nb::rv_policy::take_ownership);

	inFnImage.def("encode", [](const oa::Image& image, oa::ImageCodec codec,
			oa::U32 quality) {
			auto result = oa::FnImage::encode(image, codec, quality);
			if (result.isError()) {
				throw std::runtime_error(
					result.getStatus().toString().cStr());
			}
			return bytesFrom(*result);
		}, nb::arg("image"), nb::arg("codec"), nb::arg("quality") = 90U);
	inFnImage.def("saveFile", [](nb::handle path, const oa::Image& image,
			oa::U32 quality) {
			throwIfError(oa::FnImage::saveFile(
				pathFromPython(path), image, quality));
		}, nb::arg("path"), nb::arg("image"), nb::arg("quality") = 90U);
	inFnImage.def("canDecode", &oa::FnImage::canDecode, nb::arg("codec"));
	inFnImage.def("canEncode", &oa::FnImage::canEncode, nb::arg("codec"));

    nb::class_<oa::VideoContainerInfo>(m, "VideoContainerInfo")
        .def(nb::init<>())
        .def_rw("kind", &oa::VideoContainerInfo::kind)
        .def_rw("codec", &oa::VideoContainerInfo::codec)
        .def_rw("width", &oa::VideoContainerInfo::width)
        .def_rw("height", &oa::VideoContainerInfo::height)
        .def_rw("frameRate", &oa::VideoContainerInfo::frameRate)
        .def_rw("duration", &oa::VideoContainerInfo::duration)
        .def_rw("timebaseNum", &oa::VideoContainerInfo::timebaseNum)
        .def_rw("timebaseDen", &oa::VideoContainerInfo::timebaseDen)
        .def_rw("trackCount", &oa::VideoContainerInfo::trackCount);

    nb::class_<oa::VideoDemuxerStats>(m, "VideoDemuxerStats")
        .def_ro("reconnectCount", &oa::VideoDemuxerStats::reconnectCount)
        .def_ro("timestampDiscontinuities", &oa::VideoDemuxerStats::timestampDiscontinuities)
        .def_ro("formatGeneration", &oa::VideoDemuxerStats::formatGeneration);

    nb::class_<oa::VideoPacket>(m, "VideoPacket")
        .def_prop_ro("data", [](const oa::VideoPacket& packet) {
            return bytesFrom(packet.data);
        })
        .def_ro("presentationTimestamp", &oa::VideoPacket::presentationTimestamp)
        .def_ro("decodeTimestamp", &oa::VideoPacket::decodeTimestamp)
        .def_ro("isKeyframe", &oa::VideoPacket::isKeyframe)
        .def_ro("trackIndex", &oa::VideoPacket::trackIndex);

    nb::class_<oa::VideoDemuxer>(m, "VideoDemuxer")
        .def_static("open", &openDemuxer, nb::arg("uri"),
                    nb::arg("config") = oa::VideoDemuxerConfig(),
                    nb::rv_policy::take_ownership)
        .def_static("probe", [](nb::handle pathValue) {
            const oa::Path path = pathFromPython(pathValue);
            auto result = oa::VideoDemuxer::probe(path.cStr());
            if (result.isError()) {
                throw std::runtime_error(result.getStatus().toString().cStr());
            }
            return std::move(result).getValue();
        }, nb::arg("path"))
        .def("readNextPacket", [](oa::VideoDemuxer& stream) {
            auto* packet = new oa::VideoPacket();
            auto status = stream.readNextPacket(*packet);
            if (status.isError()) {
                delete packet;
                throw std::runtime_error(status.toString().cStr());
            }
            return packet;
        }, nb::rv_policy::take_ownership)
        .def("seek", [](oa::VideoDemuxer& stream, oa::U64 timestamp) {
            throwIfError(stream.seek(timestamp));
        }, nb::arg("timestamp"))
        .def("info", &oa::VideoDemuxer::getInfo, nb::rv_policy::reference_internal)
        .def("stats", &oa::VideoDemuxer::getStats, nb::rv_policy::reference_internal)
        .def("isEos", &oa::VideoDemuxer::isEos)
		.def("isLive", &oa::VideoDemuxer::isLive)
		.def("isSeekable", &oa::VideoDemuxer::isSeekable)
		.def("formatGeneration", &oa::VideoDemuxer::formatGeneration)
		.def("close", [](oa::VideoDemuxer& stream) {
			throwIfError(stream.close());
		});
}
