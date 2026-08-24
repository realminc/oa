// OA Python bindings — functional matrix operations and host transfer.
#include "../binding.h"

#include <oa/core/fnMatrix.h>
#include <oa/ml/fnMatrix.h>

void bindCoreFnMatrix(nb::module_& m) {
    // ═════════════════════════════════════════════════════════════════════════
    // oa::FnMatrix core ops
    // ═════════════════════════════════════════════════════════════════════════

    // Schema-v2 operations. Signatures, argument names/defaults and docs are
    // emitted from tools/gen/fn/schema alongside their C++ contracts.
#include "fnMatrixOps.gen.inl"

    m.def("argmax", &oa::FnMatrix::argmax, nb::arg("a"), nb::arg("dim") = -1);

    m.def("scalar", &oa::FnMatrix::scalar, nb::arg("mat"));

    // ═════════════════════════════════════════════════════════════════════════
    // Host readback
    // ═════════════════════════════════════════════════════════════════════════

    m.def("copyToHost", [](const oa::Matrix& mat) -> nb::list {
        const auto copyTyped = [&mat]<typename T>() -> nb::list {
            std::vector<T> host(static_cast<size_t>(mat.numElements()));
            throwIfError(oa::FnMatrix::copyToHost(mat, host.data(), host.size() * sizeof(T)));
            nb::list result;
            for (const T value : host) {
                result.append(value);
            }
            return result;
        };

        switch (mat.getDtype()) {
            case oa::ScalarType::Float32: return copyTyped.template operator()<oa::F32>();
            case oa::ScalarType::Float64: return copyTyped.template operator()<oa::F64>();
            case oa::ScalarType::Int8:    return copyTyped.template operator()<oa::I8>();
            case oa::ScalarType::Int16:   return copyTyped.template operator()<oa::I16>();
            case oa::ScalarType::Int32:   return copyTyped.template operator()<oa::I32>();
            case oa::ScalarType::Int64:   return copyTyped.template operator()<oa::I64>();
            case oa::ScalarType::UInt8:
            case oa::ScalarType::Bool:    return copyTyped.template operator()<oa::U8>();
            case oa::ScalarType::UInt16:  return copyTyped.template operator()<oa::U16>();
            case oa::ScalarType::UInt32:  return copyTyped.template operator()<oa::U32>();
            case oa::ScalarType::UInt64:  return copyTyped.template operator()<oa::U64>();
            default:
                throw nb::type_error(
                    "CopyToHost does not yet decode this storage dtype; cast the matrix to "
                    "Float32 or an integer dtype before readback");
        }
    }, nb::arg("mat"), "Copy a device matrix to a Python list without reinterpreting its dtype");

    m.def("copyToHost2D", [](const oa::Matrix& mat, oa::I64 rows, oa::I64 cols) -> std::vector<std::vector<float>> {
        std::vector<float> flat(static_cast<size_t>(mat.numElements()));
        throwIfError(oa::FnMatrix::copyToHost(mat, flat.data(), flat.size() * sizeof(float)));
        std::vector<std::vector<float>> result;
        result.reserve(static_cast<size_t>(rows));
        for (oa::I64 r = 0; r < rows; ++r) {
            std::vector<float> row;
            row.reserve(static_cast<size_t>(cols));
            for (oa::I64 c = 0; c < cols; ++c) {
                row.push_back(flat[static_cast<size_t>(r * cols + c)]);
            }
            result.push_back(std::move(row));
        }
        return result;
    }, nb::arg("mat"), nb::arg("rows"), nb::arg("cols"), "Copy device matrix to host as a 2D list of float32 values");

    // ═════════════════════════════════════════════════════════════════════════
    // Remaining compatibility bindings not yet owned by a schema
    // ═════════════════════════════════════════════════════════════════════════

    m.def("reshape", [](const oa::Matrix& a, const std::vector<oa::I64>& dims) {
        return matrixPtr(oa::FnMatrix::reshape(a, shapeFromVector(dims)));
    }, nb::arg("a"), nb::arg("shape"), nb::rv_policy::take_ownership);

    m.def("reshape", [](const oa::Matrix& a, oa::I64 d0, oa::I64 d1) {
        return matrixPtr(oa::FnMatrix::reshape(a, oa::MatrixShape{d0, d1}));
    }, nb::arg("a"), nb::arg("d0"), nb::arg("d1"), nb::rv_policy::take_ownership);

    m.def("reshape", [](const oa::Matrix& a, oa::I64 d0) {
        return matrixPtr(oa::FnMatrix::reshape(a, oa::MatrixShape{d0}));
    }, nb::arg("a"), nb::arg("d0"), nb::rv_policy::take_ownership);

    m.def("causalMask", [](oa::I64 seqLen) {
        return matrixPtr(oa::FnMatrix::causalMask(seqLen));
    }, nb::arg("seqLen"), nb::rv_policy::take_ownership);

    m.def("fromBytes", [](const std::vector<uint8_t>& data, const std::vector<oa::I64>& dims, oa::ScalarType dtype) {
        (void)pythonEngine();
        return matrixPtr(oa::FnMatrix::fromBytes(
            oa::Span<const oa::U8>(data.data(), data.size()),
            shapeFromVector(dims), dtype));
    }, nb::arg("data"), nb::arg("shape"), nb::arg("dtype"),
      nb::rv_policy::take_ownership, "Create matrix from host byte data");

    m.def("fromBytes", [](const std::vector<uint8_t>& data, oa::I64 d0, oa::I64 d1, oa::ScalarType dtype) {
        (void)pythonEngine();
        return matrixPtr(oa::FnMatrix::fromBytes(
            oa::Span<const oa::U8>(data.data(), data.size()),
            oa::MatrixShape{d0, d1}, dtype));
    }, nb::arg("data"), nb::arg("d0"), nb::arg("d1"), nb::arg("dtype"),
      nb::rv_policy::take_ownership, "Create 2D matrix from host byte data");

    m.def("fromBytes", [](const std::vector<uint8_t>& data, oa::I64 d0, oa::ScalarType dtype) {
        (void)pythonEngine();
        auto mat = oa::FnMatrix::fromBytes(
            oa::Span<const oa::U8>(data.data(), data.size()),
            oa::MatrixShape{d0}, dtype);
        return matrixPtr(std::move(mat));
    }, nb::arg("data"), nb::arg("d0"), nb::arg("dtype"),
      nb::rv_policy::take_ownership, "Create 1D matrix from host byte data");

    m.def("fromInt32", [](const std::vector<oa::I32>& data, const std::vector<oa::I64>& dims, oa::ScalarType dtype) {
        (void)pythonEngine();
        return matrixPtr(oa::FnMatrix::fromInt32(
            oa::Span<const oa::I32>(data.data(), data.size()),
            shapeFromVector(dims), dtype));
    }, nb::arg("data"), nb::arg("shape"), nb::arg("dtype") = oa::ScalarType::Int32,
      nb::rv_policy::take_ownership, "Create matrix from host int32 data");

    // FromFloats — upload host float data as a Float32 matrix. Fills the gap left by
    // FromBytes (raw byte reinterpret) and FromInt32 (int upload): there was no
    // first-class way to get arbitrary float *values* onto the device — only the
    // Full constant. Integer inputs (Scale/matmul on a UInt8 matrix) silently
    // produce garbage, so any float feature tensor (e.g. normalized image pixels)
    // must come in as Float32 via this path.
    m.def("fromFloats", [](const std::vector<float>& data, const std::vector<oa::I64>& dims) {
        (void)pythonEngine();
        return matrixPtr(oa::FnMatrix::fromBytes(
            oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(data.data()),
                               data.size() * sizeof(float)),
            shapeFromVector(dims), oa::ScalarType::Float32));
    }, nb::arg("data"), nb::arg("shape"), nb::rv_policy::take_ownership,
      "Create a Float32 matrix from host float data");

    m.def("fromFloats", [](const std::vector<float>& data, oa::I64 d0, oa::I64 d1) {
        (void)pythonEngine();
        return matrixPtr(oa::FnMatrix::fromBytes(
            oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(data.data()),
                               data.size() * sizeof(float)),
            oa::MatrixShape{d0, d1}, oa::ScalarType::Float32));
    }, nb::arg("data"), nb::arg("d0"), nb::arg("d1"), nb::rv_policy::take_ownership,
      "Create a 2D Float32 matrix from host float data");

    m.def("fromFloats", [](const std::vector<float>& data, oa::I64 d0) {
        (void)pythonEngine();
        return matrixPtr(oa::FnMatrix::fromBytes(
            oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(data.data()),
                               data.size() * sizeof(float)),
            oa::MatrixShape{d0}, oa::ScalarType::Float32));
    }, nb::arg("data"), nb::arg("d0"), nb::rv_policy::take_ownership,
      "Create a 1D Float32 matrix from host float data");

}
