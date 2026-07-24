// OA Python bindings — functional matrix operations and host transfer.
#include "../Binding.h"

#include <Oa/Core/FnMatrix.h>
#include <Oa/Ml/FnMatrix.h>

void BindCoreFnMatrix(nb::module_& m) {
    // ═════════════════════════════════════════════════════════════════════════
    // OaFnMatrix core ops
    // ═════════════════════════════════════════════════════════════════════════

    // Schema-v2 operations. Signatures, argument names/defaults and docs are
    // emitted from Tools/FnAutogen/Schema alongside their C++ contracts.
#include "FnMatrixOps.gen.inl"

    m.def("Sub", [](const OaMatrix& a, const OaMatrix& b) {
        return matrix_ptr(OaFnMatrix::Sub(a, b));
    }, nb::arg("A"), nb::arg("B"), nb::rv_policy::take_ownership);

    m.def("Mul", [](const OaMatrix& a, const OaMatrix& b) {
        return matrix_ptr(OaFnMatrix::Mul(a, b));
    }, nb::arg("A"), nb::arg("B"), nb::rv_policy::take_ownership);

    m.def("Div", [](const OaMatrix& a, const OaMatrix& b) {
        return matrix_ptr(OaFnMatrix::Div(a, b));
    }, nb::arg("A"), nb::arg("B"), nb::rv_policy::take_ownership);

    m.def("Argmax", &OaFnMatrix::Argmax, nb::arg("A"), nb::arg("Dim") = -1);

    m.def("Scalar", &OaFnMatrix::Scalar, nb::arg("Mat"));

    // ═════════════════════════════════════════════════════════════════════════
    // Host readback
    // ═════════════════════════════════════════════════════════════════════════

    m.def("CopyToHost", [](const OaMatrix& mat) -> nb::list {
        const auto copyTyped = [&mat]<typename T>() -> nb::list {
            std::vector<T> host(static_cast<size_t>(mat.NumElements()));
            throw_if_error(OaFnMatrix::CopyToHost(mat, host.data(), host.size() * sizeof(T)));
            nb::list result;
            for (const T value : host) {
                result.append(value);
            }
            return result;
        };

        switch (mat.GetDtype()) {
            case OaScalarType::Float32: return copyTyped.template operator()<OaF32>();
            case OaScalarType::Float64: return copyTyped.template operator()<OaF64>();
            case OaScalarType::Int8:    return copyTyped.template operator()<OaI8>();
            case OaScalarType::Int16:   return copyTyped.template operator()<OaI16>();
            case OaScalarType::Int32:   return copyTyped.template operator()<OaI32>();
            case OaScalarType::Int64:   return copyTyped.template operator()<OaI64>();
            case OaScalarType::UInt8:
            case OaScalarType::Bool:    return copyTyped.template operator()<OaU8>();
            case OaScalarType::UInt16:  return copyTyped.template operator()<OaU16>();
            case OaScalarType::UInt32:  return copyTyped.template operator()<OaU32>();
            case OaScalarType::UInt64:  return copyTyped.template operator()<OaU64>();
            default:
                throw nb::type_error(
                    "CopyToHost does not yet decode this storage dtype; cast the matrix to "
                    "Float32 or an integer dtype before readback");
        }
    }, nb::arg("Mat"), "Copy a device matrix to a Python list without reinterpreting its dtype");

    m.def("CopyToHost2D", [](const OaMatrix& mat, OaI64 rows, OaI64 cols) -> std::vector<std::vector<float>> {
        std::vector<float> flat(static_cast<size_t>(mat.NumElements()));
        throw_if_error(OaFnMatrix::CopyToHost(mat, flat.data(), flat.size() * sizeof(float)));
        std::vector<std::vector<float>> result;
        result.reserve(static_cast<size_t>(rows));
        for (OaI64 r = 0; r < rows; ++r) {
            std::vector<float> row;
            row.reserve(static_cast<size_t>(cols));
            for (OaI64 c = 0; c < cols; ++c) {
                row.push_back(flat[static_cast<size_t>(r * cols + c)]);
            }
            result.push_back(std::move(row));
        }
        return result;
    }, nb::arg("Mat"), nb::arg("Rows"), nb::arg("Cols"), "Copy device matrix to host as a 2D list of float32 values");

    // ═════════════════════════════════════════════════════════════════════════
    // More OaFnMatrix ops (generated in C++; manual bindings here)
    // ═════════════════════════════════════════════════════════════════════════

    m.def("Tanh", [](const OaMatrix& a) {
        return matrix_ptr(OaFnMatrix::Tanh(a));
    }, nb::arg("A"), nb::rv_policy::take_ownership);

    m.def("Relu", [](const OaMatrix& a) {
        return matrix_ptr(OaFnMatrix::Relu(a));
    }, nb::arg("A"), nb::rv_policy::take_ownership);

    m.def("Gelu", [](const OaMatrix& a) {
        return matrix_ptr(OaFnMatrix::Gelu(a));
    }, nb::arg("A"), nb::rv_policy::take_ownership);

    m.def("Silu", [](const OaMatrix& a) {
        return matrix_ptr(OaFnMatrix::Silu(a));
    }, nb::arg("A"), nb::rv_policy::take_ownership);

    m.def("Sigmoid", [](const OaMatrix& a) {
        return matrix_ptr(OaFnMatrix::Sigmoid(a));
    }, nb::arg("A"), nb::rv_policy::take_ownership);

    m.def("Log", [](const OaMatrix& a) {
        return matrix_ptr(OaFnMatrix::Log(a));
    }, nb::arg("A"), nb::rv_policy::take_ownership);

    m.def("Exp", [](const OaMatrix& a) {
        return matrix_ptr(OaFnMatrix::Exp(a));
    }, nb::arg("A"), nb::rv_policy::take_ownership);

    m.def("Sqrt", [](const OaMatrix& a) {
        return matrix_ptr(OaFnMatrix::Sqrt(a));
    }, nb::arg("A"), nb::rv_policy::take_ownership);

    m.def("Pow", [](const OaMatrix& a, OaF32 exponent) {
        return matrix_ptr(OaFnMatrix::Pow(a, exponent));
    }, nb::arg("A"), nb::arg("Exponent"), nb::rv_policy::take_ownership);

    m.def("Reshape", [](const OaMatrix& a, const std::vector<OaI64>& dims) {
        return matrix_ptr(OaFnMatrix::Reshape(a, shape_from_vector(dims)));
    }, nb::arg("A"), nb::arg("Shape"), nb::rv_policy::take_ownership);

    m.def("Reshape", [](const OaMatrix& a, OaI64 d0, OaI64 d1) {
        return matrix_ptr(OaFnMatrix::Reshape(a, OaMatrixShape{d0, d1}));
    }, nb::arg("A"), nb::arg("D0"), nb::arg("D1"), nb::rv_policy::take_ownership);

    m.def("Reshape", [](const OaMatrix& a, OaI64 d0) {
        return matrix_ptr(OaFnMatrix::Reshape(a, OaMatrixShape{d0}));
    }, nb::arg("A"), nb::arg("D0"), nb::rv_policy::take_ownership);

    m.def("Transpose", [](const OaMatrix& a, OaI32 dim0, OaI32 dim1) {
        return matrix_ptr(OaFnMatrix::Transpose(a, dim0, dim1));
    }, nb::arg("A"), nb::arg("Dim0"), nb::arg("Dim1"), nb::rv_policy::take_ownership);

    m.def("Scale", [](const OaMatrix& a, OaF32 scalar) {
        return matrix_ptr(OaFnMatrix::Scale(a, scalar));
    }, nb::arg("A"), nb::arg("Scalar"), nb::rv_policy::take_ownership);

    m.def("AddScalar", [](const OaMatrix& a, OaF32 scalar) {
        return matrix_ptr(OaFnMatrix::AddScalar(a, scalar));
    }, nb::arg("A"), nb::arg("Scalar"), nb::rv_policy::take_ownership);

    m.def("SubScalar", [](const OaMatrix& a, OaF32 scalar) {
        return matrix_ptr(OaFnMatrix::SubScalar(a, scalar));
    }, nb::arg("A"), nb::arg("Scalar"), nb::rv_policy::take_ownership);

    m.def("DivScalar", [](const OaMatrix& a, OaF32 scalar) {
        return matrix_ptr(OaFnMatrix::DivScalar(a, scalar));
    }, nb::arg("A"), nb::arg("Scalar"), nb::rv_policy::take_ownership);

    m.def("Gather", [](const OaMatrix& a, const OaMatrix& indices) {
        return matrix_ptr(OaFnMatrix::Gather(a, indices));
    }, nb::arg("A"), nb::arg("Indices"), nb::rv_policy::take_ownership);

    m.def("Slice", [](const OaMatrix& a, OaI32 dim, OaI64 start, OaI64 end) {
        return matrix_ptr(OaFnMatrix::Slice(a, dim, start, end));
    }, nb::arg("A"), nb::arg("Dim"), nb::arg("Start"), nb::arg("End"), nb::rv_policy::take_ownership);

    m.def("Cast", [](const OaMatrix& a, OaScalarType dtype) {
        return matrix_ptr(OaFnMatrix::Cast(a, dtype));
    }, nb::arg("A"), nb::arg("Dtype"), nb::rv_policy::take_ownership);

    m.def("Detach", [](const OaMatrix& a) {
        return matrix_ptr(OaFnMatrix::Detach(a));
    }, nb::arg("A"), nb::rv_policy::take_ownership);

    m.def("CausalMask", [](OaI64 seq_len) {
        return matrix_ptr(OaFnMatrix::CausalMask(seq_len));
    }, nb::arg("SeqLen"), nb::rv_policy::take_ownership);

    m.def("CausalMask", [](const OaMatrix& scores) {
        return matrix_ptr(OaFnMatrix::CausalMask(scores));
    }, nb::arg("Scores"), nb::rv_policy::take_ownership);

    m.def("FromBytes", [](const std::vector<uint8_t>& data, const std::vector<OaI64>& dims, OaScalarType dtype) {
        (void)PythonEngine();
        return matrix_ptr(OaFnMatrix::FromBytes(
            OaSpan<const OaU8>(data.data(), data.size()),
            shape_from_vector(dims), dtype));
    }, nb::arg("Data"), nb::arg("Shape"), nb::arg("Dtype"),
      nb::rv_policy::take_ownership, "Create matrix from host byte data");

    m.def("FromBytes", [](const std::vector<uint8_t>& data, OaI64 d0, OaI64 d1, OaScalarType dtype) {
        (void)PythonEngine();
        return matrix_ptr(OaFnMatrix::FromBytes(
            OaSpan<const OaU8>(data.data(), data.size()),
            OaMatrixShape{d0, d1}, dtype));
    }, nb::arg("Data"), nb::arg("D0"), nb::arg("D1"), nb::arg("Dtype"),
      nb::rv_policy::take_ownership, "Create 2D matrix from host byte data");

    m.def("FromBytes", [](const std::vector<uint8_t>& data, OaI64 d0, OaScalarType dtype) {
        (void)PythonEngine();
        auto mat = OaFnMatrix::FromBytes(
            OaSpan<const OaU8>(data.data(), data.size()),
            OaMatrixShape{d0}, dtype);
        return matrix_ptr(std::move(mat));
    }, nb::arg("Data"), nb::arg("D0"), nb::arg("Dtype"),
      nb::rv_policy::take_ownership, "Create 1D matrix from host byte data");

    m.def("FromInt32", [](const std::vector<OaI32>& data, const std::vector<OaI64>& dims, OaScalarType dtype) {
        (void)PythonEngine();
        return matrix_ptr(OaFnMatrix::FromInt32(
            OaSpan<const OaI32>(data.data(), data.size()),
            shape_from_vector(dims), dtype));
    }, nb::arg("Data"), nb::arg("Shape"), nb::arg("Dtype") = OaScalarType::Int32,
      nb::rv_policy::take_ownership, "Create matrix from host int32 data");

    // FromFloats — upload host float data as a Float32 matrix. Fills the gap left by
    // FromBytes (raw byte reinterpret) and FromInt32 (int upload): there was no
    // first-class way to get arbitrary float *values* onto the device — only the
    // Full constant. Integer inputs (Scale/matmul on a UInt8 matrix) silently
    // produce garbage, so any float feature tensor (e.g. normalized image pixels)
    // must come in as Float32 via this path.
    m.def("FromFloats", [](const std::vector<float>& data, const std::vector<OaI64>& dims) {
        (void)PythonEngine();
        return matrix_ptr(OaFnMatrix::FromBytes(
            OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(data.data()),
                               data.size() * sizeof(float)),
            shape_from_vector(dims), OaScalarType::Float32));
    }, nb::arg("Data"), nb::arg("Shape"), nb::rv_policy::take_ownership,
      "Create a Float32 matrix from host float data");

    m.def("FromFloats", [](const std::vector<float>& data, OaI64 d0, OaI64 d1) {
        (void)PythonEngine();
        return matrix_ptr(OaFnMatrix::FromBytes(
            OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(data.data()),
                               data.size() * sizeof(float)),
            OaMatrixShape{d0, d1}, OaScalarType::Float32));
    }, nb::arg("Data"), nb::arg("D0"), nb::arg("D1"), nb::rv_policy::take_ownership,
      "Create a 2D Float32 matrix from host float data");

    m.def("FromFloats", [](const std::vector<float>& data, OaI64 d0) {
        (void)PythonEngine();
        return matrix_ptr(OaFnMatrix::FromBytes(
            OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(data.data()),
                               data.size() * sizeof(float)),
            OaMatrixShape{d0}, OaScalarType::Float32));
    }, nb::arg("Data"), nb::arg("D0"), nb::rv_policy::take_ownership,
      "Create a 1D Float32 matrix from host float data");

}
