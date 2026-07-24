// OA Python bindings — GPU audio transform and signal operations.
#include "../Binding.h"

#include <Oa/Audio/FnAudio.h>

namespace {

OaAudio* audio_ptr(OaAudio InAudio) {
    return new OaAudio(OaStdMove(InAudio));
}

} // namespace

void BindAudioFn(nb::module_& m) {
    // ═════════════════════════════════════════════════════════════════════════
    // OaFnAudio ops — flat free functions (auto-context, take_ownership).
    // Grouped: Transform (Stft/Mel/Mfcc), Signal (the composed DSP ops), generic.
    // ═════════════════════════════════════════════════════════════════════════

    m.def("Stft", [](const OaAudio& audio, const OaStftConfig& cfg) {
        return matrix_ptr(OaFnAudio::Stft(audio, cfg));
    }, nb::arg("Audio"), nb::arg("Config") = OaStftConfig(), nb::rv_policy::take_ownership,
      "STFT magnitude → [Channels, Frames, FftSize/2+1] F32.");

    m.def("MelSpectrogram", [](const OaAudio& audio, const OaMelConfig& cfg) {
        return matrix_ptr(OaFnAudio::MelSpectrogram(audio, cfg));
    }, nb::arg("Audio"), nb::arg("Config") = OaMelConfig(),
      nb::rv_policy::take_ownership,
      "Mel spectrogram → [Channels, NumMels, Frames] F32 (Whisper/CLAP layout).");

    m.def("Mfcc", [](const OaAudio& audio, const OaMfccConfig& cfg) {
        return matrix_ptr(OaFnAudio::Mfcc(audio, cfg));
    }, nb::arg("Audio"), nb::arg("Config") = OaMfccConfig(),
      nb::rv_policy::take_ownership,
      "MFCC → [Channels, NumCoeffs, Frames] F32 (orthonormal DCT-II of log-mel).");

    m.def("Normalize", [](const OaAudio& audio, OaF32 target_db, OaU8 mode) {
        return audio_ptr(OaFnAudio::Normalize(audio, target_db, mode));
    }, nb::arg("Audio"), nb::arg("TargetDb") = -3.0F, nb::arg("Mode") = 0,
      nb::rv_policy::take_ownership,
      "Peak (mode 0) or RMS (mode 1) normalization to target_db.");

    m.def("Resample", [](const OaAudio& audio, OaU32 out_rate,
                         OaU32 filter_half_width) {
        return audio_ptr(OaFnAudio::Resample(audio, out_rate, filter_half_width));
    }, nb::arg("Audio"), nb::arg("OutRate") = 16000,
      nb::arg("FilterHalfWidth") = 64, nb::rv_policy::take_ownership,
      "Windowed-sinc resample → [Channels, Samples·out/in] F32.");

    m.def("Gain", [](const OaAudio& audio, OaF32 gain_db) {
        return audio_ptr(OaFnAudio::Gain(audio, gain_db));
    }, nb::arg("Audio"), nb::arg("GainDb"), nb::rv_policy::take_ownership,
      "Scalar gain in dB.");

    m.def("Clip", [](const OaAudio& audio, OaF32 min, OaF32 max) {
        return audio_ptr(OaFnAudio::Clip(audio, min, max));
    }, nb::arg("Audio"), nb::arg("Min") = -1.0F, nb::arg("Max") = 1.0F,
      nb::rv_policy::take_ownership, "Hard clip to [min, max].");

    m.def("AmplitudeToDb", [](const OaAudio& audio, OaF32 floor_db) {
        return matrix_ptr(OaFnAudio::AmplitudeToDb(audio, floor_db));
    }, nb::arg("Audio"), nb::arg("FloorDb") = -100.0F, nb::rv_policy::take_ownership,
      "20·log10(max(|x|, floor)) — amplitude to dB with a silence floor.");

    m.def("PreEmphasis", [](const OaAudio& audio, OaF32 alpha) {
        return audio_ptr(OaFnAudio::PreEmphasis(audio, alpha));
    }, nb::arg("Audio"), nb::arg("Alpha") = 0.97F, nb::rv_policy::take_ownership,
      "Pre-emphasis filter y[n] = x[n] − α·x[n−1] (y[0] = x[0]).");

    m.def("ToMono", [](const OaAudio& audio) {
        return audio_ptr(OaFnAudio::ToMono(audio));
    }, nb::arg("Audio"), nb::rv_policy::take_ownership,
      "Average channels → [1, Samples].");

    m.def("Fade", [](const OaAudio& audio, OaU64 fade_in_samples, OaU64 fade_out_samples) {
        return audio_ptr(OaFnAudio::Fade(audio, fade_in_samples, fade_out_samples));
    }, nb::arg("Audio"), nb::arg("FadeInSamples"), nb::arg("FadeOutSamples"),
      nb::rv_policy::take_ownership, "Linear fade-in/out envelope (sample counts per edge).");

    m.def("Mix", [](const OaAudio& a, const OaAudio& b, OaF32 gain_a, OaF32 gain_b) {
        return audio_ptr(OaFnAudio::Mix(a, b, gain_a, gain_b));
    }, nb::arg("A"), nb::arg("B"), nb::arg("GainA") = 1.0F, nb::arg("GainB") = 1.0F,
      nb::rv_policy::take_ownership, "Weighted sum gain_a·a + gain_b·b.");

}
