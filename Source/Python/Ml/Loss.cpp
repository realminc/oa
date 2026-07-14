// OA Python bindings — loss functions.
#include "../Binding.h"

#include <Oa/Ml/FnLoss.h>

void BindMlLoss(nb::module_& m) {
    // ═════════════════════════════════════════════════════════════════════════
    // OaFnLoss
    // ═════════════════════════════════════════════════════════════════════════

    m.def("CrossEntropy", [](const OaMatrix& logits, const OaMatrix& targets) {
        return matrix_ptr(OaFnLoss::CrossEntropy(logits, targets));
    }, nb::arg("logits"), nb::arg("targets"), nb::rv_policy::take_ownership,
      "Cross-entropy loss for classification");

    m.def("CrossEntropyBwd", [](const OaMatrix& logits, const OaMatrix& targets) {
        return matrix_ptr(OaFnLoss::CrossEntropyBwd(logits, targets));
    }, nb::arg("logits"), nb::arg("targets"), nb::rv_policy::take_ownership,
      "Cross-entropy backward: gradient w.r.t. logits");

    m.def("Mse", [](const OaMatrix& a, const OaMatrix& b) {
        return matrix_ptr(OaFnLoss::Mse(a, b));
    }, nb::arg("a"), nb::arg("b"), nb::rv_policy::take_ownership);

    m.def("MseBwd", [](const OaMatrix& a, const OaMatrix& b) {
        return matrix_ptr(OaFnLoss::MseBwd(a, b));
    }, nb::arg("a"), nb::arg("b"), nb::rv_policy::take_ownership);

    m.def("L1", [](const OaMatrix& a, const OaMatrix& b) {
        return matrix_ptr(OaFnLoss::L1(a, b));
    }, nb::arg("a"), nb::arg("b"), nb::rv_policy::take_ownership);

    m.def("L1Bwd", [](const OaMatrix& a, const OaMatrix& b) {
        return matrix_ptr(OaFnLoss::L1Bwd(a, b));
    }, nb::arg("a"), nb::arg("b"), nb::rv_policy::take_ownership);

    m.def("Bce", [](const OaMatrix& a, const OaMatrix& b) {
        return matrix_ptr(OaFnLoss::Bce(a, b));
    }, nb::arg("a"), nb::arg("b"), nb::rv_policy::take_ownership);

    m.def("BceBwd", [](const OaMatrix& a, const OaMatrix& b) {
        return matrix_ptr(OaFnLoss::BceBwd(a, b));
    }, nb::arg("a"), nb::arg("b"), nb::rv_policy::take_ownership);

    m.def("SmoothL1", [](const OaMatrix& a, const OaMatrix& b) {
        return matrix_ptr(OaFnLoss::SmoothL1(a, b));
    }, nb::arg("a"), nb::arg("b"), nb::rv_policy::take_ownership);

    m.def("SmoothL1Bwd", [](const OaMatrix& a, const OaMatrix& b) {
        return matrix_ptr(OaFnLoss::SmoothL1Bwd(a, b));
    }, nb::arg("a"), nb::arg("b"), nb::rv_policy::take_ownership);
}
