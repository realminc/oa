// OA Python bindings — iterator training, callbacks, timing, and metrics.
#include "../binding.h"

#include <oa/ml/itTraining.h>
#include <oa/ml/callbacks.h>
#include <oa/ml/metric.h>
#include <oa/ml/trainingSession.h>

#include <string>

void bindTraining(nb::module_& m) {
    // ═════════════════════════════════════════════════════════════════════════
    // oa::ItTrainingConfig
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<oa::ItTrainingConfig>(m, "ItTrainingConfig")
        .def(nb::init<>())
        .def_rw("totalSteps", &oa::ItTrainingConfig::totalSteps)
        .def_rw("stepsPerEpoch", &oa::ItTrainingConfig::stepsPerEpoch)
		.def_rw("batchSize", &oa::ItTrainingConfig::batchSize)
		.def_rw("sequenceLength", &oa::ItTrainingConfig::sequenceLength)
		.def_prop_rw("sequenceUnit",
			[](const oa::ItTrainingConfig& self) {
				return std::string(self.sequenceUnit.cStr());
			},
			[](oa::ItTrainingConfig& self, const std::string& value) {
				self.sequenceUnit = oa::String(value.c_str());
			})
		.def_rw("sourceUnitsPerSample",
			&oa::ItTrainingConfig::sourceUnitsPerSample)
		.def_prop_rw("sourceUnit",
			[](const oa::ItTrainingConfig& self) {
				return std::string(self.sourceUnit.cStr());
			},
			[](oa::ItTrainingConfig& self, const std::string& value) {
				self.sourceUnit = oa::String(value.c_str());
			})
		.def_prop_rw("timerName",
			[](const oa::ItTrainingConfig& self) {
				return std::string(self.timerName.cStr());
			},
			[](oa::ItTrainingConfig& self, const std::string& value) {
				self.timerName = oa::String(value.c_str());
			})
		.def_rw("enableGpuTiming", &oa::ItTrainingConfig::enableGpuTiming);

    // Exact per-step GPU timing statistics.
    nb::class_<oa::GpuTimingStats>(m, "GpuTimingStats")
        .def_ro("count", &oa::GpuTimingStats::count)
        .def_ro("meanMs", &oa::GpuTimingStats::meanMs)
        .def_ro("minMs", &oa::GpuTimingStats::minMs)
        .def_ro("medianMs", &oa::GpuTimingStats::medianMs)
        .def_ro("p95Ms", &oa::GpuTimingStats::p95Ms)
        .def_ro("lastMs", &oa::GpuTimingStats::lastMs);

    // ═════════════════════════════════════════════════════════════════════════
    // oa::Callback (base class)
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<oa::Callback>(m, "Callback");

    // ═════════════════════════════════════════════════════════════════════════
    // oa::CbTraining (base callback class)
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<oa::CbTraining, oa::Callback>(m, "CbTraining")
        .def("onTrainBegin", &oa::CbTraining::onTrainBegin)
        .def("onEpochBegin", &oa::CbTraining::onEpochBegin)
        .def("onStepEnd", &oa::CbTraining::onStepEnd)
        .def("onEpochEnd", &oa::CbTraining::onEpochEnd)
        .def("onTrainEnd", &oa::CbTraining::onTrainEnd);

    // ═════════════════════════════════════════════════════════════════════════
    // oa::ItTraining
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<oa::ItTraining>(m, "ItTraining")
        .def("__init__", [](oa::ItTraining* self, oa::Optimizer& opt, const oa::ItTrainingConfig& cfg) {
            new (self) oa::ItTraining(pythonEngine(), opt, cfg);
        }, nb::arg("optimizer"), nb::arg("config") = oa::ItTrainingConfig())
        .def("isDone", &oa::ItTraining::isDone)
        .def("next", nb::overload_cast<>(&oa::ItTraining::next))
        .def("next", nb::overload_cast<const oa::Matrix&>(&oa::ItTraining::next), nb::arg("loss"))
        .def("reset", &oa::ItTraining::reset)
        .def("index", &oa::ItTraining::index)
        .def("recordLoss", &oa::ItTraining::recordLoss, nb::arg("loss"))
        .def("recordAccuracy", &oa::ItTraining::recordAccuracy, nb::arg("accuracy"))
        .def("recordSourceUnits", &oa::ItTraining::recordSourceUnits,
			nb::arg("units"))
        .def("finish", [](oa::ItTraining& self) {
            auto status = self.finish();
            if (!status.isOk()) {
                throw std::runtime_error(status.toString().cStr());
            }
        })
        .def("addCallback", &oa::ItTraining::addCallback, nb::arg("callback"), nb::keep_alive<1, 2>())
		.def("addMetric", &oa::ItTraining::addMetric, nb::arg("metric"), nb::keep_alive<1, 2>())
        .def("stepCount", &oa::ItTraining::stepCount)
        .def("totalSteps", &oa::ItTraining::totalSteps)
        .def("epoch", &oa::ItTraining::epoch)
        .def("stepInEpoch", &oa::ItTraining::stepInEpoch)
        .def("totalEpochs", &oa::ItTraining::totalEpochs)
        .def("lastLoss", &oa::ItTraining::lastLoss)
        .def("liveAccuracy", &oa::ItTraining::liveAccuracy)
        .def("elapsedSeconds", &oa::ItTraining::elapsedSeconds)
        .def("lastGpuMs", &oa::ItTraining::lastGpuMs)
        .def("gpuTimingStats", &oa::ItTraining::gpuTimingStats,
             "Exact GPU per-step timing (mean/min/median/p95/last ms)")
		.def("wallMsPerStep", &oa::ItTraining::wallMsPerStep)
		.def("wallSamplesPerSecond", &oa::ItTraining::wallSamplesPerSecond)
		.def("gpuSamplesPerSecond", &oa::ItTraining::gpuSamplesPerSecond)
		.def("wallUnitsPerSecond", &oa::ItTraining::wallUnitsPerSecond)
		.def("gpuUnitsPerSecond", &oa::ItTraining::gpuUnitsPerSecond)
		.def("wallSourceUnitsPerSecond",
			&oa::ItTraining::wallSourceUnitsPerSecond)
		.def("gpuSourceUnitsPerSecond",
			&oa::ItTraining::gpuSourceUnitsPerSecond)
		.def("totalSourceUnits", &oa::ItTraining::totalSourceUnits);

	// Typed, bounded live-training control. The session is independent of any
	// GUI or transport; Python and native viewers consume the same snapshots.
	nb::enum_<oa::TrainingState>(m, "TrainingState")
		.value("Running", oa::TrainingState::Running)
		.value("Paused", oa::TrainingState::Paused)
		.value("Stopping", oa::TrainingState::Stopping)
		.value("Completed", oa::TrainingState::Completed)
		.value("Failed", oa::TrainingState::Failed);

	nb::enum_<oa::TrainingCommandDisposition>(m, "TrainingCommandDisposition")
		.value("Applied", oa::TrainingCommandDisposition::Applied)
		.value("Rejected", oa::TrainingCommandDisposition::Rejected);

	nb::class_<oa::TrainingMetricSample>(m, "TrainingMetricSample")
		.def_prop_ro("name", [](const oa::TrainingMetricSample& metric) {
			return std::string(metric.name.cStr());
		})
		.def_ro("value", &oa::TrainingMetricSample::value)
		.def_ro("step", &oa::TrainingMetricSample::step);

	nb::class_<oa::TrainingSnapshot>(m, "TrainingSnapshot")
		.def_ro("revision", &oa::TrainingSnapshot::revision)
		.def_ro("state", &oa::TrainingSnapshot::state)
		.def_ro("step", &oa::TrainingSnapshot::step)
		.def_ro("epoch", &oa::TrainingSnapshot::epoch)
		.def_ro("learningRate", &oa::TrainingSnapshot::learningRate)
		.def_ro("loss", &oa::TrainingSnapshot::loss)
		.def_ro("gpuMs", &oa::TrainingSnapshot::gpuMs)
		.def_ro("wallMs", &oa::TrainingSnapshot::wallMs)
		.def_prop_ro("metrics", [](const oa::TrainingSnapshot& snapshot) {
			std::vector<oa::TrainingMetricSample> metrics;
			metrics.reserve(static_cast<size_t>(snapshot.metrics.size()));
			for (const auto& metric : snapshot.metrics) metrics.push_back(metric);
			return metrics;
		});

	nb::class_<oa::TrainingCommandResult>(m, "TrainingCommandResult")
		.def_ro("sequence", &oa::TrainingCommandResult::sequence)
		.def_ro("revision", &oa::TrainingCommandResult::revision)
		.def_ro("disposition", &oa::TrainingCommandResult::disposition)
		.def_ro("state", &oa::TrainingCommandResult::state)
		.def_prop_ro("success", [](const oa::TrainingCommandResult& result) {
			return result.status.isOk();
		})
		.def_prop_ro("status", [](const oa::TrainingCommandResult& result) {
			return std::string(result.status.toString().cStr());
		});

	nb::class_<oa::TrainingSession>(m, "TrainingSession")
		.def("__init__", [](oa::TrainingSession* self, oa::ItTraining& training,
			oa::U32 commandCapacity, oa::U32 resultCapacity, oa::U32 snapshotCapacity) {
			new (self) oa::TrainingSession(training, oa::TrainingSessionConfig{
				.commandCapacity = commandCapacity,
				.resultCapacity = resultCapacity,
				.snapshotCapacity = snapshotCapacity,
			});
		}, nb::arg("training"), nb::arg("commandCapacity") = 64,
			nb::arg("resultCapacity") = 128,
			nb::arg("snapshotCapacity") = 256, nb::keep_alive<1, 2>())
		.def("pause", [](oa::TrainingSession& self, oa::U64 revision) {
			auto result = self.pause(revision);
			throwIfError(result.getStatus());
			return *result;
		}, nb::arg("expectedRevision") = 0)
		.def("resume", [](oa::TrainingSession& self, oa::U64 revision) {
			auto result = self.resume(revision);
			throwIfError(result.getStatus());
			return *result;
		}, nb::arg("expectedRevision") = 0)
		.def("stop", [](oa::TrainingSession& self, oa::U64 revision) {
			auto result = self.stop(revision);
			throwIfError(result.getStatus());
			return *result;
		}, nb::arg("expectedRevision") = 0)
		.def("checkpoint", [](oa::TrainingSession& self, oa::U64 revision) {
			auto result = self.checkpoint(revision);
			throwIfError(result.getStatus());
			return *result;
		}, nb::arg("expectedRevision") = 0)
		.def("evaluate", [](oa::TrainingSession& self, oa::U64 revision) {
			auto result = self.evaluate(revision);
			throwIfError(result.getStatus());
			return *result;
		}, nb::arg("expectedRevision") = 0)
		.def("setFloat", [](oa::TrainingSession& self, const std::string& name,
			double value, oa::U64 revision) {
			auto result = self.setParameter(oa::String(name.c_str()),
				oa::TrainingValue::fromFloat(value), revision);
			throwIfError(result.getStatus());
			return *result;
		}, nb::arg("name"), nb::arg("value"), nb::arg("expectedRevision") = 0)
		.def("setInteger", [](oa::TrainingSession& self, const std::string& name,
			oa::I64 value, oa::U64 revision) {
			auto result = self.setParameter(oa::String(name.c_str()),
				oa::TrainingValue::fromInteger(value), revision);
			throwIfError(result.getStatus());
			return *result;
		}, nb::arg("name"), nb::arg("value"), nb::arg("expectedRevision") = 0)
		.def("setBool", [](oa::TrainingSession& self, const std::string& name,
			bool value, oa::U64 revision) {
			auto result = self.setParameter(oa::String(name.c_str()),
				oa::TrainingValue::fromBool(value), revision);
			throwIfError(result.getStatus());
			return *result;
		}, nb::arg("name"), nb::arg("value"), nb::arg("expectedRevision") = 0)
		.def("setString", [](oa::TrainingSession& self, const std::string& name,
			const std::string& value, oa::U64 revision) {
			auto result = self.setParameter(oa::String(name.c_str()),
				oa::TrainingValue::fromString(oa::String(value.c_str())), revision);
			throwIfError(result.getStatus());
			return *result;
		}, nb::arg("name"), nb::arg("value"), nb::arg("expectedRevision") = 0)
		.def("tryBeginStep", &oa::TrainingSession::tryBeginStep)
		.def("poll", [](oa::TrainingSession& self) { throwIfError(self.poll()); })
		.def("publishMetric", [](oa::TrainingSession& self,
			const std::string& name, double value) {
			self.publishMetric(oa::String(name.c_str()), value);
		})
		.def("state", &oa::TrainingSession::state)
		.def("revision", &oa::TrainingSession::revision)
		.def("latestSnapshot", [](const oa::TrainingSession& self) -> nb::object {
			auto snapshot = self.latestSnapshot();
			return snapshot.hasValue() ? nb::cast(*snapshot) : nb::none();
		})
		.def("takeResults", [](oa::TrainingSession& self) {
			auto source = self.takeResults();
			std::vector<oa::TrainingCommandResult> results;
			results.reserve(static_cast<size_t>(source.size()));
			for (auto& result : source) results.push_back(oa::move(result));
			return results;
		});

    // ═════════════════════════════════════════════════════════════════════════
    // Callback implementations
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<oa::CbProgressBar, oa::CbTraining>(m, "CbProgressBar")
        .def(nb::init<>())
        .def("addMetric", &oa::CbProgressBar::addMetric, nb::arg("metric"), nb::keep_alive<1, 2>());

    nb::class_<oa::CbSummary, oa::CbTraining>(m, "CbSummary")
        .def(nb::init<>());

    // ═════════════════════════════════════════════════════════════════════════
    // Metrics
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<oa::Metric>(m, "Metric")
        .def("name", &oa::Metric::name)
        .def("result", &oa::Metric::result)
        .def("reset", &oa::Metric::reset);

    nb::class_<oa::MetricLoss, oa::Metric>(m, "MetricLoss")
        .def(nb::init<>())
		// Bind the hosted Python string explicitly at the OA text boundary.
		.def("__init__", [](oa::MetricLoss* self, const std::string& name) {
			new (self) oa::MetricLoss(oa::String(name.data(), name.size()));
        }, nb::arg("name"));

    nb::class_<oa::MetricAccuracy, oa::Metric>(m, "MetricAccuracy")
        .def(nb::init<>());

}
