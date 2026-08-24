// OA Python bindings — oa::FnMatrix factories and RNG policy.
#include "../binding.h"

#include <oa/core/fnMatrix.h>

void bindCoreFactory(nb::module_& m) {
    // ═════════════════════════════════════════════════════════════════════════

    m.def("empty", [](const std::vector<oa::I64>& dims, oa::ScalarType dtype) {
        return matrixPtr(oa::FnMatrix::empty(shapeFromVector(dims), dtype));
    }, nb::arg("shape"), nb::arg("dtype") = oa::ScalarType::Float32, nb::rv_policy::take_ownership);

    m.def("zeros", [](const std::vector<oa::I64>& dims, oa::ScalarType dtype) {
        return matrixPtr(oa::FnMatrix::zeros(shapeFromVector(dims), dtype));
    }, nb::arg("shape"), nb::arg("dtype") = oa::ScalarType::Float32, nb::rv_policy::take_ownership);

    m.def("ones", [](const std::vector<oa::I64>& dims, oa::ScalarType dtype) {
        return matrixPtr(oa::FnMatrix::ones(shapeFromVector(dims), dtype));
    }, nb::arg("shape"), nb::arg("dtype") = oa::ScalarType::Float32, nb::rv_policy::take_ownership);

    m.def("full", [](const std::vector<oa::I64>& dims, oa::F64 value, oa::ScalarType dtype) {
        return matrixPtr(oa::FnMatrix::full(shapeFromVector(dims), value, dtype));
    }, nb::arg("shape"), nb::arg("value"), nb::arg("dtype") = oa::ScalarType::Float32, nb::rv_policy::take_ownership);

    m.def("rand", [](const std::vector<oa::I64>& dims, oa::ScalarType dtype) {
        return matrixPtr(oa::FnMatrix::rand(shapeFromVector(dims), dtype));
    }, nb::arg("shape"), nb::arg("dtype") = oa::ScalarType::Float32, nb::rv_policy::take_ownership);

    m.def("randN", [](const std::vector<oa::I64>& dims, oa::ScalarType dtype) {
        return matrixPtr(oa::FnMatrix::randN(shapeFromVector(dims), dtype));
    }, nb::arg("shape"), nb::arg("dtype") = oa::ScalarType::Float32, nb::rv_policy::take_ownership);

    m.def("randXavier", [](const std::vector<oa::I64>& dims, oa::ScalarType dtype) {
        return matrixPtr(oa::FnMatrix::randXavier(shapeFromVector(dims), dtype));
    }, nb::arg("shape"), nb::arg("dtype") = oa::ScalarType::Float32, nb::rv_policy::take_ownership);

    m.def("randKaimingUniform", [](const std::vector<oa::I64>& dims, oa::ScalarType dtype) {
        return matrixPtr(oa::FnMatrix::randKaimingUniform(shapeFromVector(dims), dtype));
    }, nb::arg("shape"), nb::arg("dtype") = oa::ScalarType::Float32, nb::rv_policy::take_ownership);

    m.def("randGlorotUniform", [](const std::vector<oa::I64>& dims, oa::ScalarType dtype) {
        return matrixPtr(oa::FnMatrix::randGlorotUniform(shapeFromVector(dims), dtype));
    }, nb::arg("shape"), nb::arg("dtype") = oa::ScalarType::Float32, nb::rv_policy::take_ownership);

    m.def("zeros", [](oa::I64 d0, oa::I64 d1, oa::ScalarType dtype) {
        return matrixPtr(oa::FnMatrix::zeros(oa::MatrixShape{d0, d1}, dtype));
    }, nb::arg("d0"), nb::arg("d1"), nb::arg("dtype") = oa::ScalarType::Float32, nb::rv_policy::take_ownership);

    m.def("ones", [](oa::I64 d0, oa::I64 d1, oa::ScalarType dtype) {
        return matrixPtr(oa::FnMatrix::ones(oa::MatrixShape{d0, d1}, dtype));
    }, nb::arg("d0"), nb::arg("d1"), nb::arg("dtype") = oa::ScalarType::Float32, nb::rv_policy::take_ownership);

    m.def("full", [](oa::I64 d0, oa::I64 d1, oa::F32 value, oa::ScalarType dtype) {
        return matrixPtr(oa::FnMatrix::full(oa::MatrixShape{d0, d1}, static_cast<oa::F64>(value), dtype));
    }, nb::arg("d0"), nb::arg("d1"), nb::arg("value"), nb::arg("dtype") = oa::ScalarType::Float32, nb::rv_policy::take_ownership);

    m.def("rand", [](oa::I64 d0, oa::I64 d1, oa::ScalarType dtype) {
        return matrixPtr(oa::FnMatrix::rand(oa::MatrixShape{d0, d1}, dtype));
    }, nb::arg("d0"), nb::arg("d1"), nb::arg("dtype") = oa::ScalarType::Float32, nb::rv_policy::take_ownership);

    m.def("weightDtype", []() {
		(void)pythonEngine();
        return oa::FnMatrix::weightDtype();
    }, "Get the current weight dtype (Float32, BFloat16, or Float16)");

    m.def("randN", [](oa::I64 d0, oa::I64 d1, oa::ScalarType dtype) {
        return matrixPtr(oa::FnMatrix::randN(oa::MatrixShape{d0, d1}, dtype));
    }, nb::arg("d0"), nb::arg("d1"), nb::arg("dtype") = oa::ScalarType::Float32, nb::rv_policy::take_ownership);

    m.def("randXavier", [](oa::I64 d0, oa::I64 d1, oa::ScalarType dtype) {
        return matrixPtr(oa::FnMatrix::randXavier(oa::MatrixShape{d0, d1}, dtype));
    }, nb::arg("d0"), nb::arg("d1"), nb::arg("dtype") = oa::ScalarType::Float32, nb::rv_policy::take_ownership);

    m.def("randKaimingUniform", [](oa::I64 d0, oa::I64 d1, oa::ScalarType dtype) {
        return matrixPtr(oa::FnMatrix::randKaimingUniform(oa::MatrixShape{d0, d1}, dtype));
    }, nb::arg("d0"), nb::arg("d1"), nb::arg("dtype") = oa::ScalarType::Float32, nb::rv_policy::take_ownership);

    m.def("randGlorotUniform", [](oa::I64 d0, oa::I64 d1, oa::ScalarType dtype) {
        return matrixPtr(oa::FnMatrix::randGlorotUniform(oa::MatrixShape{d0, d1}, dtype));
    }, nb::arg("d0"), nb::arg("d1"), nb::arg("dtype") = oa::ScalarType::Float32, nb::rv_policy::take_ownership);

    m.def("setRngSeed", &oa::FnMatrix::setRngSeed, nb::arg("seed"));

}
