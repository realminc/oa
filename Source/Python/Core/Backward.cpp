// OA Python bindings — explicit backward operations for manual training loops.
#include "../Binding.h"

#include <Oa/Ml/FnMatrix.h>

void BindCoreBackward(nb::module_& m) {
    // ═════════════════════════════════════════════════════════════════════════
    // Manual backward ops (for hand-wired backward tutorials)
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<OaFnMatrix::OaLinearWeightBiasBwdResult>(m, "OaLinearWeightBiasBwdResult")
        .def_ro("GradWeight", &OaFnMatrix::OaLinearWeightBiasBwdResult::GradWeight)
        .def_ro("GradBias", &OaFnMatrix::OaLinearWeightBiasBwdResult::GradBias);

    m.def("LinearWeightBiasBwd", [](const OaMatrix& input, const OaMatrix& grad_output) {
        auto result = OaFnMatrix::LinearWeightBiasBwd(input, grad_output);
        return result;
    }, nb::arg("input"), nb::arg("grad_output"));

    m.def("LinearDataBwd", [](const OaMatrix& grad_output, const OaMatrix& weight) {
        return matrix_ptr(OaFnMatrix::LinearDataBwd(grad_output, weight));
    }, nb::arg("grad_output"), nb::arg("weight"), nb::rv_policy::take_ownership);

    m.def("TanhBwd", [](const OaMatrix& forward_output, const OaMatrix& grad_output) {
        return matrix_ptr(OaFnMatrix::TanhBwd(forward_output, grad_output));
    }, nb::arg("forward_output"), nb::arg("grad_output"), nb::rv_policy::take_ownership);

    m.def("ReluBwd", [](const OaMatrix& forward_output, const OaMatrix& grad_output) {
        return matrix_ptr(OaFnMatrix::ReluBwd(forward_output, grad_output));
    }, nb::arg("forward_output"), nb::arg("grad_output"), nb::rv_policy::take_ownership);

    m.def("GeluBwd", [](const OaMatrix& input, const OaMatrix& grad_output) {
        return matrix_ptr(OaFnMatrix::GeluBwd(input, grad_output));
    }, nb::arg("input"), nb::arg("grad_output"), nb::rv_policy::take_ownership);

    m.def("SiluBwd", [](const OaMatrix& input, const OaMatrix& grad_output) {
        return matrix_ptr(OaFnMatrix::SiluBwd(input, grad_output));
    }, nb::arg("input"), nb::arg("grad_output"), nb::rv_policy::take_ownership);

    m.def("SoftmaxBwd", [](const OaMatrix& forward_output, const OaMatrix& grad_output) {
        return matrix_ptr(OaFnMatrix::SoftmaxBwd(forward_output, grad_output));
    }, nb::arg("forward_output"), nb::arg("grad_output"), nb::rv_policy::take_ownership);

    m.def("GatherBwd", [](const OaMatrix& indices, const OaMatrix& grad_output, OaI32 vocab_size, OaI32 embed_dim) {
        return matrix_ptr(OaFnMatrix::GatherBwd(indices, grad_output, vocab_size, embed_dim));
    }, nb::arg("indices"), nb::arg("grad_output"), nb::arg("vocab_size"), nb::arg("embed_dim"),
      nb::rv_policy::take_ownership);
}
