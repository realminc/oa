// OA Python bindings — GPU detection postprocess and evaluation.
#include "../binding.h"

#include <oa/vision/fnDetection.h>

void bindVisionDetection(nb::module_& inModule, nb::module_& m) {
	nb::class_<oa::NmsConfig>(inModule, "NmsConfig")
		.def(nb::init<>())
		.def_rw("iouThreshold", &oa::NmsConfig::iouThreshold)
		.def_rw("scoreThreshold", &oa::NmsConfig::scoreThreshold)
		.def_rw("maxDetections", &oa::NmsConfig::maxDetections)
		.def_rw("classAgnostic", &oa::NmsConfig::classAgnostic);

	nb::class_<oa::NmsResult>(inModule, "NmsResult")
		.def_prop_ro("indices", [](oa::NmsResult& result) -> oa::Matrix& {
			return result.indices;
		}, nb::rv_policy::reference_internal)
		.def_prop_ro("count", [](oa::NmsResult& result) -> oa::Matrix& {
			return result.count;
		}, nb::rv_policy::reference_internal)
		.def("isValid", &oa::NmsResult::isValid);

	nb::class_<oa::DetectionMetricsResult>(inModule, "DetectionMetricsResult")
		.def_prop_ro("counts", [](oa::DetectionMetricsResult& result) -> oa::Matrix& {
			return result.counts;
		}, nb::rv_policy::reference_internal)
		.def_prop_ro("perClass", [](oa::DetectionMetricsResult& result) -> oa::Matrix& {
			return result.perClass;
		}, nb::rv_policy::reference_internal)
		.def_prop_ro("meanAveragePrecisionByThreshold",
			[](oa::DetectionMetricsResult& result) -> oa::Matrix& {
			return result.meanAveragePrecisionByThreshold;
		}, nb::rv_policy::reference_internal)
		.def_prop_ro("meanAveragePrecision", [](oa::DetectionMetricsResult& result) -> oa::Matrix& {
			return result.meanAveragePrecision;
		}, nb::rv_policy::reference_internal)
		.def("isValid", &oa::DetectionMetricsResult::isValid);

	nb::class_<oa::SegmentationMetricsResult>(
		inModule, "SegmentationMetricsResult")
		.def_prop_ro("confusion", [](oa::SegmentationMetricsResult& result) -> oa::Matrix& {
			return result.confusion;
		}, nb::rv_policy::reference_internal)
		.def_prop_ro("perClass", [](oa::SegmentationMetricsResult& result) -> oa::Matrix& {
			return result.perClass;
		}, nb::rv_policy::reference_internal)
		.def_prop_ro("meanIou", [](oa::SegmentationMetricsResult& result) -> oa::Matrix& {
			return result.meanIou;
		}, nb::rv_policy::reference_internal)
		.def_prop_ro("pixelAccuracy", [](oa::SegmentationMetricsResult& result) -> oa::Matrix& {
			return result.pixelAccuracy;
		}, nb::rv_policy::reference_internal)
		.def("isValid", &oa::SegmentationMetricsResult::isValid);

	#include "fnDetectionOps.gen.inl"
}
