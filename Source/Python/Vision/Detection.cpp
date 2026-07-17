// OA Python bindings — GPU detection postprocess and evaluation.
#include "../Binding.h"

#include <Oa/Vision/FnDetection.h>

void BindVisionDetection(nb::module_& m) {
	nb::class_<OaNmsConfig>(m, "OaNmsConfig")
		.def(nb::init<>())
		.def_rw("IouThreshold", &OaNmsConfig::IouThreshold)
		.def_rw("ScoreThreshold", &OaNmsConfig::ScoreThreshold)
		.def_rw("MaxDetections", &OaNmsConfig::MaxDetections)
		.def_rw("ClassAgnostic", &OaNmsConfig::ClassAgnostic);

	nb::class_<OaNmsResult>(m, "OaNmsResult")
		.def_prop_ro("Indices", [](OaNmsResult& result) -> OaMatrix& {
			return result.Indices;
		}, nb::rv_policy::reference_internal)
		.def_prop_ro("Count", [](OaNmsResult& result) -> OaMatrix& {
			return result.Count;
		}, nb::rv_policy::reference_internal)
		.def("IsValid", &OaNmsResult::IsValid);

	nb::class_<OaDetectionMetricsResult>(m, "OaDetectionMetricsResult")
		.def_prop_ro("Counts", [](OaDetectionMetricsResult& result) -> OaMatrix& {
			return result.Counts;
		}, nb::rv_policy::reference_internal)
		.def_prop_ro("PerClass", [](OaDetectionMetricsResult& result) -> OaMatrix& {
			return result.PerClass;
		}, nb::rv_policy::reference_internal)
		.def_prop_ro("MeanAveragePrecisionByThreshold",
			[](OaDetectionMetricsResult& result) -> OaMatrix& {
			return result.MeanAveragePrecisionByThreshold;
		}, nb::rv_policy::reference_internal)
		.def_prop_ro("MeanAveragePrecision", [](OaDetectionMetricsResult& result) -> OaMatrix& {
			return result.MeanAveragePrecision;
		}, nb::rv_policy::reference_internal)
		.def("IsValid", &OaDetectionMetricsResult::IsValid);

	nb::class_<OaSegmentationMetricsResult>(m, "OaSegmentationMetricsResult")
		.def_prop_ro("Confusion", [](OaSegmentationMetricsResult& result) -> OaMatrix& {
			return result.Confusion;
		}, nb::rv_policy::reference_internal)
		.def_prop_ro("PerClass", [](OaSegmentationMetricsResult& result) -> OaMatrix& {
			return result.PerClass;
		}, nb::rv_policy::reference_internal)
		.def_prop_ro("MeanIou", [](OaSegmentationMetricsResult& result) -> OaMatrix& {
			return result.MeanIou;
		}, nb::rv_policy::reference_internal)
		.def_prop_ro("PixelAccuracy", [](OaSegmentationMetricsResult& result) -> OaMatrix& {
			return result.PixelAccuracy;
		}, nb::rv_policy::reference_internal)
		.def("IsValid", &OaSegmentationMetricsResult::IsValid);

	m.def("BoxIou", [](const OaMatrix& a, const OaMatrix& b) {
		return matrix_ptr(OaFnDetection::BoxIou(a, b));
	}, nb::arg("a"), nb::arg("b"), nb::rv_policy::take_ownership);

	m.def("Nms", [](const OaMatrix& boxes, const OaMatrix& scores,
		const OaMatrix& classes, const OaNmsConfig& config) {
		return new OaNmsResult(OaFnDetection::Nms(
			boxes, scores, classes, config));
	}, nb::arg("boxes"), nb::arg("scores"), nb::arg("classes"),
		nb::arg("config") = OaNmsConfig(), nb::rv_policy::take_ownership);

	m.def("ConfusionMatrix", [](const OaMatrix& predicted,
		const OaMatrix& target, OaI32 classCount) {
		return matrix_ptr(OaFnDetection::ConfusionMatrix(
			predicted, target, classCount));
	}, nb::arg("predicted"), nb::arg("target"), nb::arg("class_count"),
		nb::rv_policy::take_ownership);

	m.def("BinaryMaskCounts", [](const OaMatrix& predicted,
		const OaMatrix& target) {
		return matrix_ptr(OaFnDetection::BinaryMaskCounts(predicted, target));
	}, nb::arg("predicted"), nb::arg("target"),
		nb::rv_policy::take_ownership);

	m.def("EvaluateDetections", [](const OaMatrix& predictedBoxes,
		const OaMatrix& predictedScores, const OaMatrix& predictedClasses,
		const OaMatrix& predictedImageIds, const OaMatrix& targetBoxes,
		const OaMatrix& targetClasses, const OaMatrix& targetImageIds,
		const OaMatrix& iouThresholds, OaI32 classCount,
		OaF32 scoreThreshold) {
		return new OaDetectionMetricsResult(OaFnDetection::Evaluate(
			predictedBoxes, predictedScores, predictedClasses,
			predictedImageIds, targetBoxes, targetClasses, targetImageIds,
			iouThresholds, classCount, scoreThreshold));
	}, nb::arg("predicted_boxes"), nb::arg("predicted_scores"),
		nb::arg("predicted_classes"), nb::arg("predicted_image_ids"),
		nb::arg("target_boxes"), nb::arg("target_classes"),
		nb::arg("target_image_ids"), nb::arg("iou_thresholds"),
		nb::arg("class_count"), nb::arg("score_threshold") = 0.0F,
		nb::rv_policy::take_ownership);

	m.def("EvaluateSegmentation", [](const OaMatrix& predicted,
		const OaMatrix& target, OaI32 classCount) {
		return new OaSegmentationMetricsResult(
			OaFnDetection::EvaluateSegmentation(predicted, target, classCount));
	}, nb::arg("predicted"), nb::arg("target"), nb::arg("class_count"),
		nb::rv_policy::take_ownership);
}
