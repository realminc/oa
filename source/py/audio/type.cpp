// OA Python bindings — semantic audio and DSP configuration types.
#include "../binding.h"

#include <oa/audio/type.h>
#include <oa/audio/fnAudio.h>

void bindAudioType(nb::module_& m) {
    // ═════════════════════════════════════════════════════════════════════════
    // oa::AudioChannelLayout + layout <-> channel-count helpers
    // ═════════════════════════════════════════════════════════════════════════

    nb::enum_<oa::AudioChannelLayout>(m, "AudioChannelLayout")
        .value("Mono", oa::AudioChannelLayout::Mono)
        .value("Stereo", oa::AudioChannelLayout::Stereo)
        .value("Stereo21", oa::AudioChannelLayout::Stereo21)
        .value("Surround51", oa::AudioChannelLayout::Surround51)
        .value("Surround71", oa::AudioChannelLayout::Surround71)
        .value("Unknown", oa::AudioChannelLayout::Unknown);

    m.def("channelsForLayout", [](oa::AudioChannelLayout layout) {
        return oa::channelsForLayout(layout);
    }, nb::arg("layout"), "Expected channel count for a layout (0 = unknown).");
    m.def("layoutForChannels", [](oa::U32 channels) {
        return oa::layoutForChannels(channels);
    }, nb::arg("channels"), "Best-effort layout for a raw channel count.");

    // ═════════════════════════════════════════════════════════════════════════
    // oa::Audio
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<oa::Audio>(m, "Audio")
        .def(nb::init<>())
        .def(nb::init<oa::Matrix, oa::U32, oa::AudioChannelLayout>(),
             nb::arg("matrix"), nb::arg("sampleRate"), nb::arg("layout"))
        .def_prop_ro("matrix",
            [](oa::Audio& self) -> oa::Matrix& { return self.asMatrix(); },
            nb::rv_policy::reference_internal)
        .def_prop_ro("sampleRate", &oa::Audio::sampleRate)
        .def_prop_ro("channelCount", &oa::Audio::channels)
        .def_prop_ro("sampleCount", &oa::Audio::samples)
        .def_prop_ro("layout", &oa::Audio::layout)
        .def("isValid", &oa::Audio::validate)
        .def("durationSeconds", &oa::Audio::durationSeconds);

    // ═════════════════════════════════════════════════════════════════════════
    // DSP configuration structs (plain-data; construct then set fields)
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<oa::StftConfig>(m, "StftConfig")
        .def(nb::init<>())
        .def_rw("fftSize", &oa::StftConfig::fftSize)
        .def_rw("hopSize", &oa::StftConfig::hopSize)
        .def_rw("winSize", &oa::StftConfig::winSize)
        .def_rw("window", &oa::StftConfig::window, "0=Hann, 1=Hamming, 2=Blackman, 3=Rect")
        .def_rw("center", &oa::StftConfig::center);

    // oa::MelConfig must register before oa::MfccConfig (nested Mel member).
    nb::class_<oa::MelConfig>(m, "MelConfig")
        .def(nb::init<>())
        .def_rw("fftSize", &oa::MelConfig::fftSize)
        .def_rw("hopSize", &oa::MelConfig::hopSize)
        .def_rw("numMels", &oa::MelConfig::numMels)
        .def_rw("fMin", &oa::MelConfig::fMin)
        .def_rw("fMax", &oa::MelConfig::fMax, "Highest frequency in Hz (0 = SampleRate/2).")
        .def_rw("logScale", &oa::MelConfig::logScale)
        .def_rw("normalize", &oa::MelConfig::normalize, "Per-channel instance normalization.");

    nb::class_<oa::MfccConfig>(m, "MfccConfig")
        .def(nb::init<>())
        .def_rw("numCoeffs", &oa::MfccConfig::numCoeffs)
        .def_rw("mel", &oa::MfccConfig::mel);

    nb::class_<oa::ResampleConfig>(m, "ResampleConfig")
        .def(nb::init<>())
        .def_rw("outRate", &oa::ResampleConfig::outRate)
        .def_rw("filterHalfWidth", &oa::ResampleConfig::filterHalfWidth);

    nb::class_<oa::NormalizeAudioConfig>(m, "NormalizeAudioConfig")
        .def(nb::init<>())
        .def_rw("mode", &oa::NormalizeAudioConfig::mode, "0=peak (max abs), 1=RMS")
        .def_rw("targetDb", &oa::NormalizeAudioConfig::targetDb);

    nb::class_<oa::BiquadCoefficients>(m, "BiquadCoefficients")
        .def(nb::init<>())
        .def_rw("b0", &oa::BiquadCoefficients::b0)
        .def_rw("b1", &oa::BiquadCoefficients::b1)
        .def_rw("b2", &oa::BiquadCoefficients::b2)
        .def_rw("a1", &oa::BiquadCoefficients::a1)
        .def_rw("a2", &oa::BiquadCoefficients::a2);
}
