// OA Python bindings — Core types and OaMatrix.
#include "../Binding.h"

void BindCoreType(nb::module_& m) {
    // ═════════════════════════════════════════════════════════════════════════
    // OaMatrixShape
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<OaMatrixShape>(m, "OaMatrixShape")
        .def(nb::init<>())
        // OaMatrixShape([d0, d1, ...]) from a Python list. nanobind wants a
        // placement-new __init__ lambda, not an nb::init factory-lambda.
        .def("__init__", [](OaMatrixShape* self, const std::vector<OaI64>& dims) {
            new (self) OaMatrixShape(shape_from_vector(dims));
        }, nb::arg("Dims"))
        .def("__getitem__", [](const OaMatrixShape& self, OaI32 idx) { return self[idx]; })
        .def("__setitem__", [](OaMatrixShape& self, OaI32 idx, OaI64 val) { self[idx] = val; })
        .def("NumElements", &OaMatrixShape::NumElements)
        .def("Dims", [](const OaMatrixShape& self) { return shape_to_vector(self); })
        .def_prop_ro("Rank", [](const OaMatrixShape& self) { return self.Rank; });

    // Construct from Python with a list of any rank: oa.core.OaMatrixShape([d0, d1, ...]).
    // The old OaShape1D..4D module functions were removed with the C++ wrappers.

    // ═════════════════════════════════════════════════════════════════════════
    // OaScalarType enum
    // ═════════════════════════════════════════════════════════════════════════

    nb::enum_<OaScalarType>(m, "OaScalarType")
        .value("Float32", OaScalarType::Float32)
        .value("Float16", OaScalarType::Float16)
        .value("BFloat16", OaScalarType::BFloat16)
        .value("Float64", OaScalarType::Float64)
        .value("Int32",   OaScalarType::Int32)
        .value("Int16",   OaScalarType::Int16)
        .value("Int64",   OaScalarType::Int64)
        .value("Int8",    OaScalarType::Int8)
        .value("UInt8",   OaScalarType::UInt8)
        .value("UInt16",  OaScalarType::UInt16)
        .value("UInt32",  OaScalarType::UInt32)
        .value("UInt64",  OaScalarType::UInt64)
        .value("Bool",    OaScalarType::Bool)
        .value("Complex64", OaScalarType::Complex64)
        .value("Complex128", OaScalarType::Complex128)
        .value("Q4_0", OaScalarType::Q4_0)
        .value("Q4_1", OaScalarType::Q4_1)
        .value("Q5_0", OaScalarType::Q5_0)
        .value("Q5_1", OaScalarType::Q5_1)
        .value("Q8_0", OaScalarType::Q8_0)
        .value("Q8_1", OaScalarType::Q8_1)
        .value("Q2_K", OaScalarType::Q2_K)
        .value("Q3_K", OaScalarType::Q3_K)
        .value("Q4_K", OaScalarType::Q4_K)
        .value("Q5_K", OaScalarType::Q5_K)
        .value("Q6_K", OaScalarType::Q6_K)
        .value("Q8_K", OaScalarType::Q8_K);

    // ═════════════════════════════════════════════════════════════════════════
    // OaMatMulPrecision enum
    // ═════════════════════════════════════════════════════════════════════════

    nb::enum_<OaMatMulPrecision>(m, "OaMatMulPrecision")
        .value("Auto", OaMatMulPrecision::Auto)
        .value("Fp32", OaMatMulPrecision::Fp32)
        .value("Bf16", OaMatMulPrecision::Bf16);

    // ═════════════════════════════════════════════════════════════════════════
    // OaMatrix
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<OaMatrix>(m, "OaMatrix")
        .def("Rank", &OaMatrix::Rank, "Get tensor rank")
        .def("NumElements", &OaMatrix::NumElements, "Get total number of elements")
        .def("ByteSize", &OaMatrix::ByteSize, "Get total byte size")
        .def("Size", &OaMatrix::Size, "Get size of dimension")
        .def("Shape", [](const OaMatrix& self) { return shape_to_vector(self.GetShape()); })
        .def("Dtype", &OaMatrix::GetDtype)
        .def("IsEmpty", &OaMatrix::IsEmpty)
        .def("Clone", [](const OaMatrix& self) { return matrix_ptr(self.Clone()); }, nb::rv_policy::take_ownership)
        .def("Reshape", [](const OaMatrix& self, const std::vector<OaI64>& dims) {
            return matrix_ptr(self.Reshape(shape_from_vector(dims)));
        }, nb::arg("Shape"), nb::rv_policy::take_ownership)
        .def("Flatten", [](const OaMatrix& self) { return matrix_ptr(self.Flatten()); }, nb::rv_policy::take_ownership)
        .def("Unsqueeze", [](const OaMatrix& self, OaI32 dim) {
            return matrix_ptr(self.Unsqueeze(dim));
        }, nb::arg("Dim"), nb::rv_policy::take_ownership)
        .def("Squeeze", [](const OaMatrix& self, OaI32 dim) {
            return matrix_ptr(self.Squeeze(dim));
        }, nb::arg("Dim"), nb::rv_policy::take_ownership)
        .def("Transpose", [](const OaMatrix& self, OaI32 dim0, OaI32 dim1) {
            return matrix_ptr(self.Transpose(dim0, dim1));
        }, nb::arg("Dim0"), nb::arg("Dim1"), nb::rv_policy::take_ownership)
        .def("Contiguous", [](const OaMatrix& self) { return matrix_ptr(self.Contiguous()); }, nb::rv_policy::take_ownership)
        .def("Item", &OaMatrix::Item)
        .def("At", &OaMatrix::At, nb::arg("Index"))
        .def("Set", &OaMatrix::Set, nb::arg("Index"), nb::arg("Value"))
        .def("Zero", &OaMatrix::Zero)
        .def("RequiresGrad", &OaMatrix::RequiresGrad)
        .def("SetRequiresGrad", &OaMatrix::SetRequiresGrad)
        .def("GradMatrix", &OaMatrix::GradMatrix, "Get persistent gradient accumulator")
        .def("MutGradMatrix", &OaMatrix::MutGradMatrix, nb::rv_policy::reference_internal,
             "Get mutable persistent gradient accumulator")
        .def("AccumulateGrad", &OaMatrix::AccumulateGrad, nb::arg("Contribution"),
             "Accumulate gradient: grad += contribution")
        .def("ZeroGrad", &OaMatrix::ZeroGrad, "Zero gradient: grad = 0")
        .def("IsLeaf", &OaMatrix::IsLeaf, "Check if tensor is a leaf (no grad_fn)")
        .def("HasGradFn", [](const OaMatrix& self) { return self.GetGradFn() != nullptr; },
             "Check if tensor has a gradient function attached")
        .def("__add__", [](const OaMatrix& a, const OaMatrix& b) { return matrix_ptr(a + b); },
             nb::rv_policy::take_ownership)
        .def("__add__", [](const OaMatrix& a, OaF32 b) { return matrix_ptr(a + b); },
             nb::rv_policy::take_ownership)
        .def("__sub__", [](const OaMatrix& a, const OaMatrix& b) { return matrix_ptr(a - b); },
             nb::rv_policy::take_ownership)
        .def("__sub__", [](const OaMatrix& a, OaF32 b) { return matrix_ptr(a - b); },
             nb::rv_policy::take_ownership)
        .def("__mul__", [](const OaMatrix& a, const OaMatrix& b) { return matrix_ptr(a * b); },
             nb::rv_policy::take_ownership)
        .def("__mul__", [](const OaMatrix& a, OaF32 b) { return matrix_ptr(a * b); },
             nb::rv_policy::take_ownership)
        .def("__truediv__", [](const OaMatrix& a, const OaMatrix& b) { return matrix_ptr(a / b); },
             nb::rv_policy::take_ownership)
        .def("__truediv__", [](const OaMatrix& a, OaF32 b) { return matrix_ptr(a / b); },
             nb::rv_policy::take_ownership)
        .def("__neg__", [](const OaMatrix& a) { return matrix_ptr(-a); },
             nb::rv_policy::take_ownership)
        .def("__iadd__", [](OaMatrix& a, const OaMatrix& b) -> OaMatrix& {
            a += b;
            return a;
        }, nb::rv_policy::reference)
        .def("__iadd__", [](OaMatrix& a, OaF32 b) -> OaMatrix& {
            a += b;
            return a;
        }, nb::rv_policy::reference)
        .def("__isub__", [](OaMatrix& a, const OaMatrix& b) -> OaMatrix& {
            a -= b;
            return a;
        }, nb::rv_policy::reference)
        .def("__isub__", [](OaMatrix& a, OaF32 b) -> OaMatrix& {
            a -= b;
            return a;
        }, nb::rv_policy::reference)
        .def("__imul__", [](OaMatrix& a, const OaMatrix& b) -> OaMatrix& {
            a *= b;
            return a;
        }, nb::rv_policy::reference)
        .def("__imul__", [](OaMatrix& a, OaF32 b) -> OaMatrix& {
            a *= b;
            return a;
        }, nb::rv_policy::reference)
        .def("__itruediv__", [](OaMatrix& a, const OaMatrix& b) -> OaMatrix& {
            a /= b;
            return a;
        }, nb::rv_policy::reference)
        .def("__itruediv__", [](OaMatrix& a, OaF32 b) -> OaMatrix& {
            a /= b;
            return a;
        }, nb::rv_policy::reference);
    // ═════════════════════════════════════════════════════════════════════════
    // OaFnMatrix factory functions (2D helpers for Python)
}
