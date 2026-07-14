// OA Python bindings — iterator training, callbacks, timing, and metrics.
#include "../Binding.h"

#include <Oa/Ml/ItTraining.h>
#include <Oa/Ml/Callbacks.h>
#include <Oa/Ml/Metric.h>

#include <string>

void BindMlTraining(nb::module_& m) {
    // ═════════════════════════════════════════════════════════════════════════
    // OaItTrainingConfig
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<OaItTrainingConfig>(m, "OaItTrainingConfig")
        .def(nb::init<>())
        .def_rw("TotalSteps", &OaItTrainingConfig::TotalSteps)
        .def_rw("StepsPerEpoch", &OaItTrainingConfig::StepsPerEpoch)
		.def_rw("BatchSize", &OaItTrainingConfig::BatchSize)
		.def_rw("SequenceLength", &OaItTrainingConfig::SequenceLength)
		.def_rw("SequenceUnit", &OaItTrainingConfig::SequenceUnit);

    // Exact per-step GPU timing statistics.
    nb::class_<OaGpuTimingStats>(m, "OaGpuTimingStats")
        .def_ro("Count", &OaGpuTimingStats::Count)
        .def_ro("MeanMs", &OaGpuTimingStats::MeanMs)
        .def_ro("MinMs", &OaGpuTimingStats::MinMs)
        .def_ro("MedianMs", &OaGpuTimingStats::MedianMs)
        .def_ro("P95Ms", &OaGpuTimingStats::P95Ms)
        .def_ro("LastMs", &OaGpuTimingStats::LastMs);

    // ═════════════════════════════════════════════════════════════════════════
    // OaCallback (base class)
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<OaCallback>(m, "OaCallback");

    // ═════════════════════════════════════════════════════════════════════════
    // OaCbTraining (base callback class)
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<OaCbTraining, OaCallback>(m, "OaCbTraining")
        .def("OnTrainBegin", &OaCbTraining::OnTrainBegin)
        .def("OnEpochBegin", &OaCbTraining::OnEpochBegin)
        .def("OnStepEnd", &OaCbTraining::OnStepEnd)
        .def("OnEpochEnd", &OaCbTraining::OnEpochEnd)
        .def("OnTrainEnd", &OaCbTraining::OnTrainEnd);

    // ═════════════════════════════════════════════════════════════════════════
    // OaItTraining
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<OaItTraining>(m, "OaItTraining")
        .def("__init__", [](OaItTraining* self, OaOptimizer& opt, const OaItTrainingConfig& cfg) {
            new (self) OaItTraining(opt, cfg);
        }, nb::arg("optimizer"), nb::arg("config") = OaItTrainingConfig())
        .def("IsDone", &OaItTraining::IsDone)
        .def("Next", nb::overload_cast<>(&OaItTraining::Next))
        .def("Next", nb::overload_cast<const OaMatrix&>(&OaItTraining::Next), nb::arg("loss"))
        .def("Reset", &OaItTraining::Reset)
        .def("Index", &OaItTraining::Index)
        .def("RecordLoss", &OaItTraining::RecordLoss, nb::arg("loss"))
        .def("RecordAccuracy", &OaItTraining::RecordAccuracy, nb::arg("accuracy"))
        .def("Finish", [](OaItTraining& self) {
            auto status = self.Finish();
            if (!status.IsOk()) {
                throw std::runtime_error(status.ToString().c_str());
            }
        })
        .def("AddCallback", &OaItTraining::AddCallback, nb::arg("callback"), nb::keep_alive<1, 2>())
		.def("AddMetric", &OaItTraining::AddMetric, nb::arg("metric"), nb::keep_alive<1, 2>())
        .def("StepCount", &OaItTraining::StepCount)
        .def("TotalSteps", &OaItTraining::TotalSteps)
        .def("Epoch", &OaItTraining::Epoch)
        .def("StepInEpoch", &OaItTraining::StepInEpoch)
        .def("TotalEpochs", &OaItTraining::TotalEpochs)
        .def("LiveLoss", &OaItTraining::LiveLoss)
        .def("LastLoss", &OaItTraining::LastLoss)
        .def("LiveAccuracy", &OaItTraining::LiveAccuracy)
        .def("ElapsedSeconds", &OaItTraining::ElapsedSeconds)
        .def("LastGpuMs", &OaItTraining::LastGpuMs)
        .def("GpuTimingStats", &OaItTraining::GpuTimingStats,
             "Exact GPU per-step timing (mean/min/median/p95/last ms)")
		.def("WallMsPerStep", &OaItTraining::WallMsPerStep)
		.def("WallSamplesPerSecond", &OaItTraining::WallSamplesPerSecond)
		.def("GpuSamplesPerSecond", &OaItTraining::GpuSamplesPerSecond)
		.def("WallUnitsPerSecond", &OaItTraining::WallUnitsPerSecond)
		.def("GpuUnitsPerSecond", &OaItTraining::GpuUnitsPerSecond);

    // ═════════════════════════════════════════════════════════════════════════
    // Callback implementations
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<OaCbProgressBar, OaCbTraining>(m, "OaCbProgressBar")
        .def(nb::init<>())
        .def("AddMetric", &OaCbProgressBar::AddMetric, nb::arg("metric"), nb::keep_alive<1, 2>());

    nb::class_<OaCbSummary, OaCbTraining>(m, "OaCbSummary")
        .def(nb::init<>());

    // ═════════════════════════════════════════════════════════════════════════
    // Metrics
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<OaMetric>(m, "OaMetric")
        .def("Name", &OaMetric::Name)
        .def("Result", &OaMetric::Result)
        .def("Reset", &OaMetric::Reset);

    nb::class_<OaMetricLoss, OaMetric>(m, "OaMetricLoss")
        .def(nb::init<>())
        // OaMetricLoss(OaString); OaString(std::string) is explicit, so build it
        // by hand rather than nb::init<const std::string&> (which needs an
        // implicit conversion).
        .def("__init__", [](OaMetricLoss* self, const std::string& name) {
            new (self) OaMetricLoss(OaString(name.c_str()));
        }, nb::arg("name"));

    nb::class_<OaMetricAccuracy, OaMetric>(m, "OaMetricAccuracy")
        .def(nb::init<>());

}
