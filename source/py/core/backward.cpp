// OA Python bindings — explicit backward operations for manual training loops.
#include "../binding.h"

#include <oa/ml/fnMatrix.h>

void bindCoreBackward(nb::module_& m) {
    // ═════════════════════════════════════════════════════════════════════════
    // Manual backward ops (for hand-wired backward tutorials)
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<oa::LinearWeightBiasBwdResult>(m, "LinearWeightBiasBwdResult")
        .def_ro("gradWeight", &oa::LinearWeightBiasBwdResult::gradWeight)
        .def_ro("gradBias", &oa::LinearWeightBiasBwdResult::gradBias);

    m.def("linearWeightBiasBwd", [](const oa::Matrix& input, const oa::Matrix& gradOutput) {
        auto result = oa::FnMatrix::linearWeightBiasBwd(input, gradOutput);
        return result;
    }, nb::arg("input"), nb::arg("gradOutput"));

    m.def("linearDataBwd", [](const oa::Matrix& gradOutput, const oa::Matrix& weight) {
        return matrixPtr(oa::FnMatrix::linearDataBwd(gradOutput, weight));
    }, nb::arg("gradOutput"), nb::arg("weight"), nb::rv_policy::take_ownership);

    m.def("gatherBwd", [](const oa::Matrix& indices, const oa::Matrix& gradOutput, oa::I32 vocabSize, oa::I32 embedDim) {
        return matrixPtr(oa::FnMatrix::gatherBwd(indices, gradOutput, vocabSize, embedDim));
    }, nb::arg("indices"), nb::arg("gradOutput"), nb::arg("vocabSize"), nb::arg("embedDim"),
      nb::rv_policy::take_ownership);
}
