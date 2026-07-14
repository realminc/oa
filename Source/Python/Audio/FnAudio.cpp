// OA Python bindings — GPU audio transform and signal operations.
#include "../Binding.h"

#include <Oa/Audio/FnAudio.h>

void BindAudioFn(nb::module_& m) {
    // ═════════════════════════════════════════════════════════════════════════
    // OaFnAudio ops — flat free functions (auto-context, take_ownership).
    // Grouped: Transform (Stft/Mel/Mfcc), Signal (the composed DSP ops), generic.
    // ═════════════════════════════════════════════════════════════════════════

    m.def("Stft", [](const OaMatrix& buffer, const OaStftConfig& cfg) {
        return matrix_ptr(OaFnAudio::Stft(buffer, cfg));
    }, nb::arg("buffer"), nb::arg("config") = OaStftConfig(), nb::rv_policy::take_ownership,
      "STFT magnitude → [Channels, Frames, FftSize/2+1] F32.");

    m.def("MelSpectrogram", [](const OaMatrix& buffer, OaU32 sample_rate,
                               const OaMelConfig& cfg) {
        return matrix_ptr(OaFnAudio::MelSpectrogram(buffer, sample_rate, cfg));
    }, nb::arg("buffer"), nb::arg("sample_rate"), nb::arg("config") = OaMelConfig(),
      nb::rv_policy::take_ownership,
      "Mel spectrogram → [Channels, NumMels, Frames] F32 (Whisper/CLAP layout).");

    m.def("Mfcc", [](const OaMatrix& buffer, OaU32 sample_rate, const OaMfccConfig& cfg) {
        return matrix_ptr(OaFnAudio::Mfcc(buffer, sample_rate, cfg));
    }, nb::arg("buffer"), nb::arg("sample_rate"), nb::arg("config") = OaMfccConfig(),
      nb::rv_policy::take_ownership,
      "MFCC → [Channels, NumCoeffs, Frames] F32 (orthonormal DCT-II of log-mel).");

    m.def("Normalize", [](const OaMatrix& buffer, OaF32 target_db, OaU8 mode) {
        return matrix_ptr(OaFnAudio::Normalize(buffer, target_db, mode));
    }, nb::arg("buffer"), nb::arg("target_db") = -3.0F, nb::arg("mode") = 0,
      nb::rv_policy::take_ownership,
      "Peak (mode 0) or RMS (mode 1) normalization to target_db.");

    m.def("Resample", [](const OaMatrix& buffer, OaU32 in_rate, OaU32 out_rate,
                         OaU32 filter_half_width) {
        return matrix_ptr(OaFnAudio::Resample(buffer, in_rate, out_rate, filter_half_width));
    }, nb::arg("buffer"), nb::arg("in_rate") = 48000, nb::arg("out_rate") = 16000,
      nb::arg("filter_half_width") = 64, nb::rv_policy::take_ownership,
      "Windowed-sinc resample → [Channels, Samples·out/in] F32.");

    m.def("Gain", [](const OaMatrix& buffer, OaF32 gain_db) {
        return matrix_ptr(OaFnAudio::Gain(buffer, gain_db));
    }, nb::arg("buffer"), nb::arg("gain_db"), nb::rv_policy::take_ownership,
      "Scalar gain in dB.");

    m.def("Clip", [](const OaMatrix& buffer, OaF32 min, OaF32 max) {
        return matrix_ptr(OaFnAudio::Clip(buffer, min, max));
    }, nb::arg("buffer"), nb::arg("min") = -1.0F, nb::arg("max") = 1.0F,
      nb::rv_policy::take_ownership, "Hard clip to [min, max].");

    m.def("AmplitudeToDb", [](const OaMatrix& buffer, OaF32 floor_db) {
        return matrix_ptr(OaFnAudio::AmplitudeToDb(buffer, floor_db));
    }, nb::arg("buffer"), nb::arg("floor_db") = -100.0F, nb::rv_policy::take_ownership,
      "20·log10(max(|x|, floor)) — amplitude to dB with a silence floor.");

    m.def("PreEmphasis", [](const OaMatrix& buffer, OaF32 alpha) {
        return matrix_ptr(OaFnAudio::PreEmphasis(buffer, alpha));
    }, nb::arg("buffer"), nb::arg("alpha") = 0.97F, nb::rv_policy::take_ownership,
      "Pre-emphasis filter y[n] = x[n] − α·x[n−1] (y[0] = x[0]).");

    m.def("ToMono", [](const OaMatrix& buffer) {
        return matrix_ptr(OaFnAudio::ToMono(buffer));
    }, nb::arg("buffer"), nb::rv_policy::take_ownership,
      "Average channels → [1, Samples].");

    m.def("Fade", [](const OaMatrix& buffer, OaU64 fade_in_samples, OaU64 fade_out_samples) {
        return matrix_ptr(OaFnAudio::Fade(buffer, fade_in_samples, fade_out_samples));
    }, nb::arg("buffer"), nb::arg("fade_in_samples"), nb::arg("fade_out_samples"),
      nb::rv_policy::take_ownership, "Linear fade-in/out envelope (sample counts per edge).");

    m.def("Mix", [](const OaMatrix& a, const OaMatrix& b, OaF32 gain_a, OaF32 gain_b) {
        return matrix_ptr(OaFnAudio::Mix(a, b, gain_a, gain_b));
    }, nb::arg("a"), nb::arg("b"), nb::arg("gain_a") = 1.0F, nb::arg("gain_b") = 1.0F,
      nb::rv_policy::take_ownership, "Weighted sum gain_a·a + gain_b·b.");

    m.def("ToMatrix", [](const OaMatrix& buffer) {
        auto result = OaFnAudio::ToMatrix(buffer);
        throw_if_error(result.GetStatus());
        return matrix_ptr(std::move(result).GetValue());
    }, nb::arg("buffer"), nb::rv_policy::take_ownership,
      "Zero-copy reshape of raw waveform to [Channels, 1, SampleCount].");
}
