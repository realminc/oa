// OA Python bindings — audio decoder and encoder boundaries.
#include "../Binding.h"

#include <Oa/Audio/AudioDecoder.h>
#include <Oa/Audio/AudioEncoder.h>

#include <string>

void BindAudioCodec(nb::module_& m) {
    // ═════════════════════════════════════════════════════════════════════════
    // OaAudioDecodeResult — Buffer is [Channels, Samples] F32 on GPU.
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<OaAudioDecodeResult>(m, "OaAudioDecodeResult")
        .def_prop_ro("Buffer",
            [](OaAudioDecodeResult& self) -> OaMatrix& { return self.Buffer; },
            nb::rv_policy::reference_internal,
            "Decoded planar [Channels, Samples] F32 GPU buffer.")
        .def_ro("SampleRate", &OaAudioDecodeResult::SampleRate)
        .def_ro("ChannelCount", &OaAudioDecodeResult::ChannelCount)
        .def_ro("SampleCount", &OaAudioDecodeResult::SampleCount)
        .def("IsValid", &OaAudioDecodeResult::IsValid)
        .def("DurationSeconds", &OaAudioDecodeResult::DurationSeconds)
        .def("Meta", &OaAudioDecodeResult::Meta,
             "Metadata POD for the OaFnAudio ops (layout inferred from channels).");

    // ═════════════════════════════════════════════════════════════════════════
    // OaAudioDecoder — CPU codec boundary (miniaudio WAV/FLAC/MP3). Static.
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<OaAudioDecoder>(m, "OaAudioDecoder")
        .def_static("LoadFile", [](const std::string& path) {
            auto result = OaAudioDecoder::LoadFile(path.c_str());
            throw_if_error(result.GetStatus());
            return new OaAudioDecodeResult(std::move(result).GetValue());
        }, nb::arg("path"), nb::rv_policy::take_ownership,
           "Decode a WAV/FLAC/MP3 file to a planar [Channels, Samples] F32 GPU buffer.")
        .def_static("LoadMemory", [](nb::bytes data) {
            // Accept a Python `bytes` directly (symmetric with EncodeWavF32, which
            // returns bytes). nanobind's std::vector<uint8_t> caster rejects bytes,
            // so bind nb::bytes and read its raw buffer.
            auto result = OaAudioDecoder::LoadMemory(OaSpan<const OaU8>(
                reinterpret_cast<const OaU8*>(data.data()), data.size()));
            throw_if_error(result.GetStatus());
            return new OaAudioDecodeResult(std::move(result).GetValue());
        }, nb::arg("data"), nb::rv_policy::take_ownership,
           "Decode audio from an in-memory byte buffer (bytes; WAV/FLAC/MP3).");

    // ═════════════════════════════════════════════════════════════════════════
    // OaAudioEncoder — synchronous WAV-F32 sink. Static.
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<OaAudioEncoder>(m, "OaAudioEncoder")
        .def_static("EncodeWavF32", [](const OaMatrix& buffer, OaU32 sample_rate) {
            // Executes pending GPU work, reads planar F32, interleaves, encodes.
            auto result = OaAudioEncoder::EncodeWavF32(buffer, sample_rate);
            throw_if_error(result.GetStatus());
            auto& blob = result.GetValue();
            return nb::bytes(blob.Data(), blob.Size());
        }, nb::arg("buffer"), nb::arg("sample_rate"),
           "Encode a [Channels, Samples] F32 GPU buffer to WAV-F32 bytes (synchronous).")
        .def_static("SaveWavF32", [](const std::string& path, const OaMatrix& buffer,
                                     OaU32 sample_rate) {
            throw_if_error(OaAudioEncoder::SaveWavF32(path.c_str(), buffer, sample_rate));
        }, nb::arg("path"), nb::arg("buffer"), nb::arg("sample_rate"),
           "Encode + write a lossless WAV-F32 file (synchronous file/codec boundary).");
}
