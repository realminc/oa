// OA Python bindings — iterator training, callbacks, timing, and metrics.
#include "../Binding.h"

#include <Oa/Ml/ItTraining.h>
#include <Oa/Ml/Callbacks.h>
#include <Oa/Ml/Metric.h>
#include <Oa/Ml/TrainingSession.h>

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
		.def_prop_rw("SequenceUnit",
			[](const OaItTrainingConfig& self) {
				return std::string(self.SequenceUnit.c_str());
			},
			[](OaItTrainingConfig& self, const std::string& value) {
				self.SequenceUnit = OaString(value.c_str());
			})
		.def_rw("SourceUnitsPerSample",
			&OaItTrainingConfig::SourceUnitsPerSample)
		.def_prop_rw("SourceUnit",
			[](const OaItTrainingConfig& self) {
				return std::string(self.SourceUnit.c_str());
			},
			[](OaItTrainingConfig& self, const std::string& value) {
				self.SourceUnit = OaString(value.c_str());
			})
		.def_prop_rw("TimerName",
			[](const OaItTrainingConfig& self) {
				return std::string(self.TimerName.c_str());
			},
			[](OaItTrainingConfig& self, const std::string& value) {
				self.TimerName = OaString(value.c_str());
			})
		.def_rw("EnableGpuTiming", &OaItTrainingConfig::EnableGpuTiming);

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
        }, nb::arg("Optimizer"), nb::arg("Config") = OaItTrainingConfig())
        .def("IsDone", &OaItTraining::IsDone)
        .def("Next", nb::overload_cast<>(&OaItTraining::Next))
        .def("Next", nb::overload_cast<const OaMatrix&>(&OaItTraining::Next), nb::arg("Loss"))
        .def("Reset", &OaItTraining::Reset)
        .def("Index", &OaItTraining::Index)
        .def("RecordLoss", &OaItTraining::RecordLoss, nb::arg("Loss"))
        .def("RecordAccuracy", &OaItTraining::RecordAccuracy, nb::arg("Accuracy"))
        .def("RecordSourceUnits", &OaItTraining::RecordSourceUnits,
			nb::arg("Units"))
        .def("Finish", [](OaItTraining& self) {
            auto status = self.Finish();
            if (!status.IsOk()) {
                throw std::runtime_error(status.ToString().c_str());
            }
        })
        .def("AddCallback", &OaItTraining::AddCallback, nb::arg("Callback"), nb::keep_alive<1, 2>())
		.def("AddMetric", &OaItTraining::AddMetric, nb::arg("Metric"), nb::keep_alive<1, 2>())
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
		.def("GpuUnitsPerSecond", &OaItTraining::GpuUnitsPerSecond)
		.def("WallSourceUnitsPerSecond",
			&OaItTraining::WallSourceUnitsPerSecond)
		.def("GpuSourceUnitsPerSecond",
			&OaItTraining::GpuSourceUnitsPerSecond)
		.def("TotalSourceUnits", &OaItTraining::TotalSourceUnits);

	// Typed, bounded live-training control. The session is independent of any
	// GUI or transport; Python and native viewers consume the same snapshots.
	nb::enum_<OaTrainingState>(m, "OaTrainingState")
		.value("Running", OaTrainingState::Running)
		.value("Paused", OaTrainingState::Paused)
		.value("Stopping", OaTrainingState::Stopping)
		.value("Completed", OaTrainingState::Completed)
		.value("Failed", OaTrainingState::Failed);

	nb::enum_<OaTrainingCommandDisposition>(m, "OaTrainingCommandDisposition")
		.value("Applied", OaTrainingCommandDisposition::Applied)
		.value("Rejected", OaTrainingCommandDisposition::Rejected);

	nb::class_<OaTrainingMetricSample>(m, "OaTrainingMetricSample")
		.def_prop_ro("Name", [](const OaTrainingMetricSample& metric) {
			return std::string(metric.Name.c_str());
		})
		.def_ro("Value", &OaTrainingMetricSample::Value)
		.def_ro("Step", &OaTrainingMetricSample::Step);

	nb::class_<OaTrainingSnapshot>(m, "OaTrainingSnapshot")
		.def_ro("Revision", &OaTrainingSnapshot::Revision)
		.def_ro("State", &OaTrainingSnapshot::State)
		.def_ro("Step", &OaTrainingSnapshot::Step)
		.def_ro("Epoch", &OaTrainingSnapshot::Epoch)
		.def_ro("LearningRate", &OaTrainingSnapshot::LearningRate)
		.def_ro("Loss", &OaTrainingSnapshot::Loss)
		.def_ro("GpuMs", &OaTrainingSnapshot::GpuMs)
		.def_ro("WallMs", &OaTrainingSnapshot::WallMs)
		.def_prop_ro("Metrics", [](const OaTrainingSnapshot& snapshot) {
			std::vector<OaTrainingMetricSample> metrics;
			metrics.reserve(static_cast<size_t>(snapshot.Metrics.Size()));
			for (const auto& metric : snapshot.Metrics) metrics.push_back(metric);
			return metrics;
		});

	nb::class_<OaTrainingCommandResult>(m, "OaTrainingCommandResult")
		.def_ro("Sequence", &OaTrainingCommandResult::Sequence)
		.def_ro("Revision", &OaTrainingCommandResult::Revision)
		.def_ro("Disposition", &OaTrainingCommandResult::Disposition)
		.def_ro("State", &OaTrainingCommandResult::State)
		.def_prop_ro("Success", [](const OaTrainingCommandResult& result) {
			return result.Status.IsOk();
		})
		.def_prop_ro("Status", [](const OaTrainingCommandResult& result) {
			return std::string(result.Status.ToString().c_str());
		});

	nb::class_<OaTrainingSession>(m, "OaTrainingSession")
		.def("__init__", [](OaTrainingSession* self, OaItTraining& training,
			OaU32 commandCapacity, OaU32 resultCapacity, OaU32 snapshotCapacity) {
			new (self) OaTrainingSession(training, OaTrainingSessionConfig{
				.CommandCapacity = commandCapacity,
				.ResultCapacity = resultCapacity,
				.SnapshotCapacity = snapshotCapacity,
			});
		}, nb::arg("Training"), nb::arg("CommandCapacity") = 64,
			nb::arg("ResultCapacity") = 128,
			nb::arg("SnapshotCapacity") = 256, nb::keep_alive<1, 2>())
		.def("Pause", [](OaTrainingSession& self, OaU64 revision) {
			auto result = self.Pause(revision);
			throw_if_error(result.GetStatus());
			return *result;
		}, nb::arg("ExpectedRevision") = 0)
		.def("Resume", [](OaTrainingSession& self, OaU64 revision) {
			auto result = self.Resume(revision);
			throw_if_error(result.GetStatus());
			return *result;
		}, nb::arg("ExpectedRevision") = 0)
		.def("Stop", [](OaTrainingSession& self, OaU64 revision) {
			auto result = self.Stop(revision);
			throw_if_error(result.GetStatus());
			return *result;
		}, nb::arg("ExpectedRevision") = 0)
		.def("Checkpoint", [](OaTrainingSession& self, OaU64 revision) {
			auto result = self.Checkpoint(revision);
			throw_if_error(result.GetStatus());
			return *result;
		}, nb::arg("ExpectedRevision") = 0)
		.def("Evaluate", [](OaTrainingSession& self, OaU64 revision) {
			auto result = self.Evaluate(revision);
			throw_if_error(result.GetStatus());
			return *result;
		}, nb::arg("ExpectedRevision") = 0)
		.def("SetFloat", [](OaTrainingSession& self, const std::string& name,
			double value, OaU64 revision) {
			auto result = self.SetParameter(OaString(name.c_str()),
				OaTrainingValue::FromFloat(value), revision);
			throw_if_error(result.GetStatus());
			return *result;
		}, nb::arg("Name"), nb::arg("Value"), nb::arg("ExpectedRevision") = 0)
		.def("SetInteger", [](OaTrainingSession& self, const std::string& name,
			OaI64 value, OaU64 revision) {
			auto result = self.SetParameter(OaString(name.c_str()),
				OaTrainingValue::FromInteger(value), revision);
			throw_if_error(result.GetStatus());
			return *result;
		}, nb::arg("Name"), nb::arg("Value"), nb::arg("ExpectedRevision") = 0)
		.def("SetBool", [](OaTrainingSession& self, const std::string& name,
			bool value, OaU64 revision) {
			auto result = self.SetParameter(OaString(name.c_str()),
				OaTrainingValue::FromBool(value), revision);
			throw_if_error(result.GetStatus());
			return *result;
		}, nb::arg("Name"), nb::arg("Value"), nb::arg("ExpectedRevision") = 0)
		.def("SetString", [](OaTrainingSession& self, const std::string& name,
			const std::string& value, OaU64 revision) {
			auto result = self.SetParameter(OaString(name.c_str()),
				OaTrainingValue::FromString(OaString(value.c_str())), revision);
			throw_if_error(result.GetStatus());
			return *result;
		}, nb::arg("Name"), nb::arg("Value"), nb::arg("ExpectedRevision") = 0)
		.def("TryBeginStep", &OaTrainingSession::TryBeginStep)
		.def("Poll", [](OaTrainingSession& self) { throw_if_error(self.Poll()); })
		.def("PublishMetric", [](OaTrainingSession& self,
			const std::string& name, double value) {
			self.PublishMetric(OaString(name.c_str()), value);
		})
		.def("State", &OaTrainingSession::State)
		.def("Revision", &OaTrainingSession::Revision)
		.def("LatestSnapshot", [](const OaTrainingSession& self) -> nb::object {
			auto snapshot = self.LatestSnapshot();
			return snapshot.HasValue() ? nb::cast(*snapshot) : nb::none();
		})
		.def("TakeResults", [](OaTrainingSession& self) {
			auto source = self.TakeResults();
			std::vector<OaTrainingCommandResult> results;
			results.reserve(static_cast<size_t>(source.Size()));
			for (auto& result : source) results.push_back(OaStdMove(result));
			return results;
		});

    // ═════════════════════════════════════════════════════════════════════════
    // Callback implementations
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<OaCbProgressBar, OaCbTraining>(m, "OaCbProgressBar")
        .def(nb::init<>())
        .def("AddMetric", &OaCbProgressBar::AddMetric, nb::arg("Metric"), nb::keep_alive<1, 2>());

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
        }, nb::arg("Name"));

    nb::class_<OaMetricAccuracy, OaMetric>(m, "OaMetricAccuracy")
        .def(nb::init<>());

}
