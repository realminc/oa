// OA Python bindings — stateless machine-learning metrics.
#include "../binding.h"

#include <oa/ml/metric.h>

void bindMlMetric(nb::module_& inFnMetric) {
	inFnMetric.def("accuracy", &oa::FnMetric::accuracy,
		nb::arg("predictions"), nb::arg("labels"));
	inFnMetric.def("scalarLoss", &oa::FnMetric::scalarLoss,
		nb::arg("loss"));
}
