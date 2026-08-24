// OA Python bindings — stateless audio codec boundaries.
#include "../binding.h"

#include <oa/audio/fnAudio.h>

#include <string>

void bindAudioCodec(nb::module_& inFnAudio) {
    inFnAudio.def("decodeFile", [](nb::handle path) {
            // Decoding uploads the planar samples into an oa::Matrix. Ensure the
            // process engine exists before the C++ codec reaches that upload.
            (void)pythonEngine();
            auto result = oa::FnAudio::decodeFile(pathFromPython(path));
            throwIfError(result.getStatus());
            return new oa::Audio(std::move(result).getValue());
        }, nb::arg("path"), nb::rv_policy::take_ownership,
           "Decode a WAV/FLAC/MP3 file to a planar [Channels, Samples] F32 GPU buffer.");
    inFnAudio.def("decodeMemory", [](nb::bytes data) {
            (void)pythonEngine();
            // Accept a Python `bytes` directly (symmetric with EncodeWavF32, which
            // returns bytes). nanobind's std::vector<uint8_t> caster rejects bytes,
            // so bind nb::bytes and read its raw buffer.
            auto result = oa::FnAudio::decodeMemory(oa::Span<const oa::U8>(
                reinterpret_cast<const oa::U8*>(data.data()), data.size()));
            throwIfError(result.getStatus());
            return new oa::Audio(std::move(result).getValue());
        }, nb::arg("data"), nb::rv_policy::take_ownership,
           "Decode audio from an in-memory byte buffer (bytes; WAV/FLAC/MP3).");

    inFnAudio.def("encodeWavF32", [](const oa::Audio& audio) {
            // Executes pending GPU work, reads planar F32, interleaves, encodes.
            auto result = oa::FnAudio::encodeWavF32(audio);
            throwIfError(result.getStatus());
            auto& blob = result.getValue();
            return nb::bytes(blob.data(), blob.size());
        }, nb::arg("audio"),
           "Encode an oa::Audio value to WAV-F32 bytes (synchronous).");
    inFnAudio.def("saveWavF32", [](nb::handle path, const oa::Audio& audio) {
            throwIfError(oa::FnAudio::saveWavF32(
                pathFromPython(path), audio));
        }, nb::arg("path"), nb::arg("audio"),
           "Encode + write a lossless WAV-F32 file (synchronous file/codec boundary).");
}
