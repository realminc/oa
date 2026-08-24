// OA Python bindings — Core types and oa::Matrix.
#include "../binding.h"

void bindCoreType(nb::module_& m) {
    // ═════════════════════════════════════════════════════════════════════════
    // oa::MatrixShape
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<oa::MatrixShape>(m, "MatrixShape")
        .def(nb::init<>())
        // oa::MatrixShape([d0, d1, ...]) from a Python list. nanobind wants a
        // placement-new __init__ lambda, not an nb::init factory-lambda.
        .def("__init__", [](oa::MatrixShape* self, const std::vector<oa::I64>& dims) {
            new (self) oa::MatrixShape(shapeFromVector(dims));
        }, nb::arg("dims"))
        .def("__getitem__", [](const oa::MatrixShape& self, oa::I32 idx) { return self[idx]; })
        .def("__setitem__", [](oa::MatrixShape& self, oa::I32 idx, oa::I64 val) { self[idx] = val; })
        .def("numElements", &oa::MatrixShape::numElements)
        .def("dims", [](const oa::MatrixShape& self) { return shapeToVector(self); })
        .def_prop_ro("rank", [](const oa::MatrixShape& self) { return self.rank; });

    // Construct from Python with a list of any rank: oa.MatrixShape([d0, d1, ...]).
    // The old oa::Shape1D..4D module functions were removed with the C++ wrappers.

    // ═════════════════════════════════════════════════════════════════════════
    // oa::ScalarType enum
    // ═════════════════════════════════════════════════════════════════════════

    nb::enum_<oa::ScalarType>(m, "ScalarType")
        .value("Float32", oa::ScalarType::Float32)
        .value("Float16", oa::ScalarType::Float16)
        .value("BFloat16", oa::ScalarType::BFloat16)
        .value("Float64", oa::ScalarType::Float64)
        .value("Int32",   oa::ScalarType::Int32)
        .value("Int16",   oa::ScalarType::Int16)
        .value("Int64",   oa::ScalarType::Int64)
        .value("Int8",    oa::ScalarType::Int8)
        .value("UInt8",   oa::ScalarType::UInt8)
        .value("UInt16",  oa::ScalarType::UInt16)
        .value("UInt32",  oa::ScalarType::UInt32)
        .value("UInt64",  oa::ScalarType::UInt64)
        .value("Bool",    oa::ScalarType::Bool)
        .value("Complex64", oa::ScalarType::Complex64)
        .value("Complex128", oa::ScalarType::Complex128);

    // ═════════════════════════════════════════════════════════════════════════
    // oa::MatMulPrecision enum
    // ═════════════════════════════════════════════════════════════════════════

    nb::enum_<oa::MatMulPrecision>(m, "MatMulPrecision")
        .value("Auto", oa::MatMulPrecision::Auto)
        .value("Fp32", oa::MatMulPrecision::Fp32)
        .value("Bf16", oa::MatMulPrecision::Bf16);

    // ═════════════════════════════════════════════════════════════════════════
    // oa::Matrix
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<oa::Matrix>(m, "Matrix")
        .def("rank", &oa::Matrix::rank, "Get tensor rank")
        .def("numElements", &oa::Matrix::numElements, "Get total number of elements")
        .def("byteSize", &oa::Matrix::byteSize, "Get total byte size")
        .def("size", &oa::Matrix::size, "Get size of dimension")
        .def("shape", [](const oa::Matrix& self) { return shapeToVector(self.getShape()); })
        .def("dtype", &oa::Matrix::getDtype)
        .def("isEmpty", &oa::Matrix::isEmpty)
        .def("clone", [](const oa::Matrix& self) { return matrixPtr(self.clone()); }, nb::rv_policy::take_ownership)
        .def("reshape", [](const oa::Matrix& self, const std::vector<oa::I64>& dims) {
            return matrixPtr(self.reshape(shapeFromVector(dims)));
        }, nb::arg("shape"), nb::rv_policy::take_ownership)
        .def("flatten", [](const oa::Matrix& self) { return matrixPtr(self.flatten()); }, nb::rv_policy::take_ownership)
        .def("unsqueeze", [](const oa::Matrix& self, oa::I32 dim) {
            return matrixPtr(self.unsqueeze(dim));
        }, nb::arg("dim"), nb::rv_policy::take_ownership)
        .def("squeeze", [](const oa::Matrix& self, oa::I32 dim) {
            return matrixPtr(self.squeeze(dim));
        }, nb::arg("dim"), nb::rv_policy::take_ownership)
        .def("transpose", [](const oa::Matrix& self, oa::I32 dim0, oa::I32 dim1) {
            return matrixPtr(self.transpose(dim0, dim1));
        }, nb::arg("dim0"), nb::arg("dim1"), nb::rv_policy::take_ownership)
        .def("contiguous", [](const oa::Matrix& self) { return matrixPtr(self.contiguous()); }, nb::rv_policy::take_ownership)
        .def("item", &oa::Matrix::item)
        .def("at", &oa::Matrix::at, nb::arg("index"))
        .def("set", &oa::Matrix::set, nb::arg("index"), nb::arg("value"))
        .def("zero", &oa::Matrix::zero)
        .def("requiresGrad", &oa::Matrix::requiresGrad)
        .def("setRequiresGrad", &oa::Matrix::setRequiresGrad)
        .def("gradMatrix", &oa::Matrix::gradMatrix, "Get persistent gradient accumulator")
        .def("mutGradMatrix", &oa::Matrix::mutGradMatrix, nb::rv_policy::reference_internal,
             "Get mutable persistent gradient accumulator")
        .def("accumulateGrad", &oa::Matrix::accumulateGrad, nb::arg("contribution"),
             "Accumulate gradient: grad += contribution")
        .def("zeroGrad", &oa::Matrix::zeroGrad, "Zero gradient: grad = 0")
        .def("isLeaf", &oa::Matrix::isLeaf, "Check if tensor is a leaf (no grad_fn)")
        .def("hasGradFn", [](const oa::Matrix& self) { return self.getGradFn() != nullptr; },
             "Check if tensor has a gradient function attached")
        .def("__add__", [](const oa::Matrix& a, const oa::Matrix& b) { return matrixPtr(a + b); },
             nb::rv_policy::take_ownership)
        .def("__add__", [](const oa::Matrix& a, oa::F32 b) { return matrixPtr(a + b); },
             nb::rv_policy::take_ownership)
        .def("__sub__", [](const oa::Matrix& a, const oa::Matrix& b) { return matrixPtr(a - b); },
             nb::rv_policy::take_ownership)
        .def("__sub__", [](const oa::Matrix& a, oa::F32 b) { return matrixPtr(a - b); },
             nb::rv_policy::take_ownership)
        .def("__mul__", [](const oa::Matrix& a, const oa::Matrix& b) { return matrixPtr(a * b); },
             nb::rv_policy::take_ownership)
        .def("__mul__", [](const oa::Matrix& a, oa::F32 b) { return matrixPtr(a * b); },
             nb::rv_policy::take_ownership)
        .def("__truediv__", [](const oa::Matrix& a, const oa::Matrix& b) { return matrixPtr(a / b); },
             nb::rv_policy::take_ownership)
        .def("__truediv__", [](const oa::Matrix& a, oa::F32 b) { return matrixPtr(a / b); },
             nb::rv_policy::take_ownership)
        .def("__neg__", [](const oa::Matrix& a) { return matrixPtr(-a); },
             nb::rv_policy::take_ownership)
        .def("__iadd__", [](oa::Matrix& a, const oa::Matrix& b) -> oa::Matrix& {
            a += b;
            return a;
        }, nb::rv_policy::reference)
        .def("__iadd__", [](oa::Matrix& a, oa::F32 b) -> oa::Matrix& {
            a += b;
            return a;
        }, nb::rv_policy::reference)
        .def("__isub__", [](oa::Matrix& a, const oa::Matrix& b) -> oa::Matrix& {
            a -= b;
            return a;
        }, nb::rv_policy::reference)
        .def("__isub__", [](oa::Matrix& a, oa::F32 b) -> oa::Matrix& {
            a -= b;
            return a;
        }, nb::rv_policy::reference)
        .def("__imul__", [](oa::Matrix& a, const oa::Matrix& b) -> oa::Matrix& {
            a *= b;
            return a;
        }, nb::rv_policy::reference)
        .def("__imul__", [](oa::Matrix& a, oa::F32 b) -> oa::Matrix& {
            a *= b;
            return a;
        }, nb::rv_policy::reference)
        .def("__itruediv__", [](oa::Matrix& a, const oa::Matrix& b) -> oa::Matrix& {
            a /= b;
            return a;
        }, nb::rv_policy::reference)
        .def("__itruediv__", [](oa::Matrix& a, oa::F32 b) -> oa::Matrix& {
            a /= b;
            return a;
        }, nb::rv_policy::reference);
    // ═════════════════════════════════════════════════════════════════════════
    // oa::FnMatrix factory functions (2D helpers for Python)
}
