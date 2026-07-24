// OA Python bindings — semantic audio and DSP configuration types.
#include "../Binding.h"

#include <Oa/Audio/Type.h>
#include <Oa/Audio/FnAudio.h>

void BindAudioType(nb::module_& m) {
    // ═════════════════════════════════════════════════════════════════════════
    // OaChannelLayout + layout <-> channel-count helpers
    // ═════════════════════════════════════════════════════════════════════════

    nb::enum_<OaChannelLayout>(m, "OaChannelLayout")
        .value("Mono", OaChannelLayout::Mono)
        .value("Stereo", OaChannelLayout::Stereo)
        .value("Surround5_1", OaChannelLayout::Surround5_1)
        .value("Surround7_1", OaChannelLayout::Surround7_1)
        .value("Unknown", OaChannelLayout::Unknown);

    m.def("OaChannelsForLayout", [](OaChannelLayout layout) {
        return OaChannelsForLayout(layout);
    }, nb::arg("Layout"), "Expected channel count for a layout (0 = unknown).");
    m.def("OaLayoutForChannels", [](OaU32 channels) {
        return OaLayoutForChannels(channels);
    }, nb::arg("Channels"), "Best-effort layout for a raw channel count.");

    // ═════════════════════════════════════════════════════════════════════════
    // OaAudio
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<OaAudio>(m, "OaAudio")
        .def(nb::init<>())
        .def(nb::init<OaMatrix, OaU32, OaChannelLayout>(),
             nb::arg("Matrix"), nb::arg("SampleRate"), nb::arg("Layout"))
        .def_prop_ro("Matrix",
            [](OaAudio& self) -> OaMatrix& { return self.AsMatrix(); },
            nb::rv_policy::reference_internal)
        .def_prop_ro("SampleRate", &OaAudio::SampleRate)
        .def_prop_ro("ChannelCount", &OaAudio::Channels)
        .def_prop_ro("SampleCount", &OaAudio::Samples)
        .def_prop_ro("Layout", &OaAudio::Layout)
        .def("IsValid", &OaAudio::Validate)
        .def("DurationSeconds", &OaAudio::DurationSeconds);

    // ═════════════════════════════════════════════════════════════════════════
    // DSP configuration structs (plain-data; construct then set fields)
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<OaStftConfig>(m, "OaStftConfig")
        .def(nb::init<>())
        .def_rw("FftSize", &OaStftConfig::FftSize)
        .def_rw("HopSize", &OaStftConfig::HopSize)
        .def_rw("WinSize", &OaStftConfig::WinSize)
        .def_rw("Window", &OaStftConfig::Window, "0=Hann, 1=Hamming, 2=Blackman, 3=Rect")
        .def_rw("Center", &OaStftConfig::Center);

    // OaMelConfig must register before OaMfccConfig (nested Mel member).
    nb::class_<OaMelConfig>(m, "OaMelConfig")
        .def(nb::init<>())
        .def_rw("FftSize", &OaMelConfig::FftSize)
        .def_rw("HopSize", &OaMelConfig::HopSize)
        .def_rw("NumMels", &OaMelConfig::NumMels)
        .def_rw("FMin", &OaMelConfig::FMin)
        .def_rw("FMax", &OaMelConfig::FMax, "Highest frequency in Hz (0 = SampleRate/2).")
        .def_rw("LogScale", &OaMelConfig::LogScale)
        .def_rw("Normalize", &OaMelConfig::Normalize, "Per-channel instance normalization.");

    nb::class_<OaMfccConfig>(m, "OaMfccConfig")
        .def(nb::init<>())
        .def_rw("NumCoeffs", &OaMfccConfig::NumCoeffs)
        .def_rw("Mel", &OaMfccConfig::Mel);

    nb::class_<OaResampleConfig>(m, "OaResampleConfig")
        .def(nb::init<>())
        .def_rw("OutRate", &OaResampleConfig::OutRate)
        .def_rw("FilterHalfWidth", &OaResampleConfig::FilterHalfWidth);

    nb::class_<OaNormalizeAudioConfig>(m, "OaNormalizeAudioConfig")
        .def(nb::init<>())
        .def_rw("Mode", &OaNormalizeAudioConfig::Mode, "0=peak (max abs), 1=RMS")
        .def_rw("TargetDb", &OaNormalizeAudioConfig::TargetDb);
}
