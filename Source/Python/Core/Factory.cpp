// OA Python bindings — OaFnMatrix factories and RNG policy.
#include "../Binding.h"

#include <Oa/Core/FnMatrix.h>

void BindCoreFactory(nb::module_& m) {
    // ═════════════════════════════════════════════════════════════════════════

    m.def("Empty", [](const std::vector<OaI64>& dims, OaScalarType dtype) {
        return matrix_ptr(OaFnMatrix::Empty(shape_from_vector(dims), dtype));
    }, nb::arg("Shape"), nb::arg("Dtype") = OaScalarType::Float32, nb::rv_policy::take_ownership);

    m.def("Zeros", [](const std::vector<OaI64>& dims, OaScalarType dtype) {
        return matrix_ptr(OaFnMatrix::Zeros(shape_from_vector(dims), dtype));
    }, nb::arg("Shape"), nb::arg("Dtype") = OaScalarType::Float32, nb::rv_policy::take_ownership);

    m.def("Ones", [](const std::vector<OaI64>& dims, OaScalarType dtype) {
        return matrix_ptr(OaFnMatrix::Ones(shape_from_vector(dims), dtype));
    }, nb::arg("Shape"), nb::arg("Dtype") = OaScalarType::Float32, nb::rv_policy::take_ownership);

    m.def("Full", [](const std::vector<OaI64>& dims, OaF64 value, OaScalarType dtype) {
        return matrix_ptr(OaFnMatrix::Full(shape_from_vector(dims), value, dtype));
    }, nb::arg("Shape"), nb::arg("Value"), nb::arg("Dtype") = OaScalarType::Float32, nb::rv_policy::take_ownership);

    m.def("Rand", [](const std::vector<OaI64>& dims, OaScalarType dtype) {
        return matrix_ptr(OaFnMatrix::Rand(shape_from_vector(dims), dtype));
    }, nb::arg("Shape"), nb::arg("Dtype") = OaScalarType::Float32, nb::rv_policy::take_ownership);

    m.def("RandN", [](const std::vector<OaI64>& dims, OaScalarType dtype) {
        return matrix_ptr(OaFnMatrix::RandN(shape_from_vector(dims), dtype));
    }, nb::arg("Shape"), nb::arg("Dtype") = OaScalarType::Float32, nb::rv_policy::take_ownership);

    m.def("RandXavier", [](const std::vector<OaI64>& dims, OaScalarType dtype) {
        return matrix_ptr(OaFnMatrix::RandXavier(shape_from_vector(dims), dtype));
    }, nb::arg("Shape"), nb::arg("Dtype") = OaScalarType::Float32, nb::rv_policy::take_ownership);

    m.def("RandKaimingUniform", [](const std::vector<OaI64>& dims, OaScalarType dtype) {
        return matrix_ptr(OaFnMatrix::RandKaimingUniform(shape_from_vector(dims), dtype));
    }, nb::arg("Shape"), nb::arg("Dtype") = OaScalarType::Float32, nb::rv_policy::take_ownership);

    m.def("RandGlorotUniform", [](const std::vector<OaI64>& dims, OaScalarType dtype) {
        return matrix_ptr(OaFnMatrix::RandGlorotUniform(shape_from_vector(dims), dtype));
    }, nb::arg("Shape"), nb::arg("Dtype") = OaScalarType::Float32, nb::rv_policy::take_ownership);

    m.def("Zeros", [](OaI64 d0, OaI64 d1, OaScalarType dtype) {
        return matrix_ptr(OaFnMatrix::Zeros(OaMatrixShape{d0, d1}, dtype));
    }, nb::arg("D0"), nb::arg("D1"), nb::arg("Dtype") = OaScalarType::Float32, nb::rv_policy::take_ownership);

    m.def("Ones", [](OaI64 d0, OaI64 d1, OaScalarType dtype) {
        return matrix_ptr(OaFnMatrix::Ones(OaMatrixShape{d0, d1}, dtype));
    }, nb::arg("D0"), nb::arg("D1"), nb::arg("Dtype") = OaScalarType::Float32, nb::rv_policy::take_ownership);

    m.def("Full", [](OaI64 d0, OaI64 d1, OaF32 value, OaScalarType dtype) {
        return matrix_ptr(OaFnMatrix::Full(OaMatrixShape{d0, d1}, static_cast<OaF64>(value), dtype));
    }, nb::arg("D0"), nb::arg("D1"), nb::arg("Value"), nb::arg("Dtype") = OaScalarType::Float32, nb::rv_policy::take_ownership);

    m.def("Rand", [](OaI64 d0, OaI64 d1, OaScalarType dtype) {
        return matrix_ptr(OaFnMatrix::Rand(OaMatrixShape{d0, d1}, dtype));
    }, nb::arg("D0"), nb::arg("D1"), nb::arg("Dtype") = OaScalarType::Float32, nb::rv_policy::take_ownership);

    m.def("GetWeightDtype", []() {
        return OaFnMatrix::GetWeightDtype();
    }, "Get the current weight dtype (Float32, BFloat16, or Float16)");

    m.def("SetWeightDtype", [](OaScalarType dtype) {
        OaFnMatrix::SetWeightDtype(dtype);
    }, nb::arg("Dtype"), "Set the weight dtype for new weight tensors");

    m.def("RandN", [](OaI64 d0, OaI64 d1, OaScalarType dtype) {
        return matrix_ptr(OaFnMatrix::RandN(OaMatrixShape{d0, d1}, dtype));
    }, nb::arg("D0"), nb::arg("D1"), nb::arg("Dtype") = OaScalarType::Float32, nb::rv_policy::take_ownership);

    m.def("RandXavier", [](OaI64 d0, OaI64 d1, OaScalarType dtype) {
        return matrix_ptr(OaFnMatrix::RandXavier(OaMatrixShape{d0, d1}, dtype));
    }, nb::arg("D0"), nb::arg("D1"), nb::arg("Dtype") = OaScalarType::Float32, nb::rv_policy::take_ownership);

    m.def("RandKaimingUniform", [](OaI64 d0, OaI64 d1, OaScalarType dtype) {
        return matrix_ptr(OaFnMatrix::RandKaimingUniform(OaMatrixShape{d0, d1}, dtype));
    }, nb::arg("D0"), nb::arg("D1"), nb::arg("Dtype") = OaScalarType::Float32, nb::rv_policy::take_ownership);

    m.def("RandGlorotUniform", [](OaI64 d0, OaI64 d1, OaScalarType dtype) {
        return matrix_ptr(OaFnMatrix::RandGlorotUniform(OaMatrixShape{d0, d1}, dtype));
    }, nb::arg("D0"), nb::arg("D1"), nb::arg("Dtype") = OaScalarType::Float32, nb::rv_policy::take_ownership);

    m.def("SetRngSeed", &OaFnMatrix::SetRngSeed, nb::arg("Seed"));

}
