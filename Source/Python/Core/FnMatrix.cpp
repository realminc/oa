// OA Python bindings — functional matrix operations and host transfer.
#include "../Binding.h"

#include <Oa/Core/FnMatrix.h>
#include <Oa/Ml/FnMatrix.h>

void BindCoreFnMatrix(nb::module_& m) {
    // ═════════════════════════════════════════════════════════════════════════
    // OaFnMatrix core ops
    // ═════════════════════════════════════════════════════════════════════════

    m.def("MatMulNt", [](const OaMatrix& a, const OaMatrix& b, OaContextMatMulPrecision precision) {
        return matrix_ptr(OaFnMatrix::MatMulNt(a, b, precision));
    }, nb::arg("a"), nb::arg("b"), nb::arg("precision") = OaContextMatMulPrecision::Auto, nb::rv_policy::take_ownership,
      "Matrix multiplication: C = A @ B^T (weight convention). B is stored transposed as [N,K]. For standard A@B use Bmm.");

    m.def("Add", [](const OaMatrix& a, const OaMatrix& b) {
        return matrix_ptr(OaFnMatrix::Add(a, b));
    }, nb::arg("a"), nb::arg("b"), nb::rv_policy::take_ownership);

    m.def("Sub", [](const OaMatrix& a, const OaMatrix& b) {
        return matrix_ptr(OaFnMatrix::Sub(a, b));
    }, nb::arg("a"), nb::arg("b"), nb::rv_policy::take_ownership);

    m.def("Mul", [](const OaMatrix& a, const OaMatrix& b) {
        return matrix_ptr(OaFnMatrix::Mul(a, b));
    }, nb::arg("a"), nb::arg("b"), nb::rv_policy::take_ownership);

    m.def("Div", [](const OaMatrix& a, const OaMatrix& b) {
        return matrix_ptr(OaFnMatrix::Div(a, b));
    }, nb::arg("a"), nb::arg("b"), nb::rv_policy::take_ownership);

    m.def("Sum", [](const OaMatrix& a, OaI32 dim) {
        return matrix_ptr(OaFnMatrix::Sum(a, dim));
    }, nb::arg("a"), nb::arg("dim") = -1, nb::rv_policy::take_ownership);

    m.def("Mean", [](const OaMatrix& a, OaI32 dim) {
        return matrix_ptr(OaFnMatrix::Mean(a, dim));
    }, nb::arg("a"), nb::arg("dim") = -1, nb::rv_policy::take_ownership);

    m.def("Max", [](const OaMatrix& a, OaI32 dim) {
        return matrix_ptr(OaFnMatrix::Max(a, dim));
    }, nb::arg("a"), nb::arg("dim") = -1, nb::rv_policy::take_ownership);

    m.def("Argmax", &OaFnMatrix::Argmax, nb::arg("a"), nb::arg("dim") = -1);

    m.def("Scalar", &OaFnMatrix::Scalar, nb::arg("mat"));

    // ═════════════════════════════════════════════════════════════════════════
    // Host readback
    // ═════════════════════════════════════════════════════════════════════════

    m.def("CopyToHost", [](const OaMatrix& mat) -> nb::object {
        if (mat.GetDtype() == OaScalarType::UInt8) {
            std::vector<uint8_t> host(static_cast<size_t>(mat.NumElements()));
            throw_if_error(OaFnMatrix::CopyToHost(mat, host.data(), host.size() * sizeof(uint8_t)));
            return nb::cast(host);
        } else {
            std::vector<float> host(static_cast<size_t>(mat.NumElements()));
            throw_if_error(OaFnMatrix::CopyToHost(mat, host.data(), host.size() * sizeof(float)));
            return nb::cast(host);
        }
    }, nb::arg("mat"), "Copy device matrix to host as a list (dtype-aware: uint8 or float32)");

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
    }, nb::arg("mat"), nb::arg("rows"), nb::arg("cols"), "Copy device matrix to host as a 2D list of float32 values");

    // ═════════════════════════════════════════════════════════════════════════
    // More OaFnMatrix ops (generated in C++; manual bindings here)
    // ═════════════════════════════════════════════════════════════════════════

    m.def("Tanh", [](const OaMatrix& a) {
        return matrix_ptr(OaFnMatrix::Tanh(a));
    }, nb::arg("a"), nb::rv_policy::take_ownership);

    m.def("Relu", [](const OaMatrix& a) {
        return matrix_ptr(OaFnMatrix::Relu(a));
    }, nb::arg("a"), nb::rv_policy::take_ownership);

    m.def("Gelu", [](const OaMatrix& a) {
        return matrix_ptr(OaFnMatrix::Gelu(a));
    }, nb::arg("a"), nb::rv_policy::take_ownership);

    m.def("Silu", [](const OaMatrix& a) {
        return matrix_ptr(OaFnMatrix::Silu(a));
    }, nb::arg("a"), nb::rv_policy::take_ownership);

    m.def("Sigmoid", [](const OaMatrix& a) {
        return matrix_ptr(OaFnMatrix::Sigmoid(a));
    }, nb::arg("a"), nb::rv_policy::take_ownership);

    m.def("Log", [](const OaMatrix& a) {
        return matrix_ptr(OaFnMatrix::Log(a));
    }, nb::arg("a"), nb::rv_policy::take_ownership);

    m.def("Exp", [](const OaMatrix& a) {
        return matrix_ptr(OaFnMatrix::Exp(a));
    }, nb::arg("a"), nb::rv_policy::take_ownership);

    m.def("Sqrt", [](const OaMatrix& a) {
        return matrix_ptr(OaFnMatrix::Sqrt(a));
    }, nb::arg("a"), nb::rv_policy::take_ownership);

    m.def("Pow", [](const OaMatrix& a, OaF32 exponent) {
        return matrix_ptr(OaFnMatrix::Pow(a, exponent));
    }, nb::arg("a"), nb::arg("exponent"), nb::rv_policy::take_ownership);

    m.def("Softmax", [](const OaMatrix& a, OaI32 dim) {
        return matrix_ptr(OaFnMatrix::Softmax(a, dim));
    }, nb::arg("a"), nb::arg("dim") = -1, nb::rv_policy::take_ownership);

    m.def("LogSoftmax", [](const OaMatrix& a, OaI32 dim) {
        return matrix_ptr(OaFnMatrix::LogSoftmax(a, dim));
    }, nb::arg("a"), nb::arg("dim") = -1, nb::rv_policy::take_ownership);

    m.def("Reshape", [](const OaMatrix& a, const std::vector<OaI64>& dims) {
        return matrix_ptr(OaFnMatrix::Reshape(a, shape_from_vector(dims)));
    }, nb::arg("a"), nb::arg("shape"), nb::rv_policy::take_ownership);

    m.def("Reshape", [](const OaMatrix& a, OaI64 d0, OaI64 d1) {
        return matrix_ptr(OaFnMatrix::Reshape(a, OaMatrixShape{d0, d1}));
    }, nb::arg("a"), nb::arg("d0"), nb::arg("d1"), nb::rv_policy::take_ownership);

    m.def("Reshape", [](const OaMatrix& a, OaI64 d0) {
        return matrix_ptr(OaFnMatrix::Reshape(a, OaMatrixShape{d0}));
    }, nb::arg("a"), nb::arg("d0"), nb::rv_policy::take_ownership);

    m.def("Transpose", [](const OaMatrix& a, OaI32 dim0, OaI32 dim1) {
        return matrix_ptr(OaFnMatrix::Transpose(a, dim0, dim1));
    }, nb::arg("a"), nb::arg("dim0"), nb::arg("dim1"), nb::rv_policy::take_ownership);

    m.def("Scale", [](const OaMatrix& a, OaF32 scalar) {
        return matrix_ptr(OaFnMatrix::Scale(a, scalar));
    }, nb::arg("a"), nb::arg("scalar"), nb::rv_policy::take_ownership);

    m.def("AddScalar", [](const OaMatrix& a, OaF32 scalar) {
        return matrix_ptr(OaFnMatrix::AddScalar(a, scalar));
    }, nb::arg("a"), nb::arg("scalar"), nb::rv_policy::take_ownership);

    m.def("SubScalar", [](const OaMatrix& a, OaF32 scalar) {
        return matrix_ptr(OaFnMatrix::SubScalar(a, scalar));
    }, nb::arg("a"), nb::arg("scalar"), nb::rv_policy::take_ownership);

    m.def("DivScalar", [](const OaMatrix& a, OaF32 scalar) {
        return matrix_ptr(OaFnMatrix::DivScalar(a, scalar));
    }, nb::arg("a"), nb::arg("scalar"), nb::rv_policy::take_ownership);

    m.def("Gather", [](const OaMatrix& a, const OaMatrix& indices) {
        return matrix_ptr(OaFnMatrix::Gather(a, indices));
    }, nb::arg("a"), nb::arg("indices"), nb::rv_policy::take_ownership);

    m.def("Slice", [](const OaMatrix& a, OaI32 dim, OaI64 start, OaI64 end) {
        return matrix_ptr(OaFnMatrix::Slice(a, dim, start, end));
    }, nb::arg("a"), nb::arg("dim"), nb::arg("start"), nb::arg("end"), nb::rv_policy::take_ownership);

    m.def("Cast", [](const OaMatrix& a, OaScalarType dtype) {
        return matrix_ptr(OaFnMatrix::Cast(a, dtype));
    }, nb::arg("a"), nb::arg("dtype"), nb::rv_policy::take_ownership);

    m.def("Detach", [](const OaMatrix& a) {
        return matrix_ptr(OaFnMatrix::Detach(a));
    }, nb::arg("a"), nb::rv_policy::take_ownership);

    m.def("CausalMask", [](OaI64 seq_len) {
        return matrix_ptr(OaFnMatrix::CausalMask(seq_len));
    }, nb::arg("seq_len"), nb::rv_policy::take_ownership);

    m.def("CausalMask", [](const OaMatrix& scores) {
        return matrix_ptr(OaFnMatrix::CausalMask(scores));
    }, nb::arg("scores"), nb::rv_policy::take_ownership);

    m.def("FromBytes", [](const std::vector<uint8_t>& data, const std::vector<OaI64>& dims, OaScalarType dtype) {
        return matrix_ptr(OaFnMatrix::FromBytes(
            OaSpan<const OaU8>(data.data(), data.size()),
            shape_from_vector(dims), dtype));
    }, nb::arg("data"), nb::arg("shape"), nb::arg("dtype"),
      nb::rv_policy::take_ownership, "Create matrix from host byte data");

    m.def("FromBytes", [](const std::vector<uint8_t>& data, OaI64 d0, OaI64 d1, OaScalarType dtype) {
        return matrix_ptr(OaFnMatrix::FromBytes(
            OaSpan<const OaU8>(data.data(), data.size()),
            OaMatrixShape{d0, d1}, dtype));
    }, nb::arg("data"), nb::arg("d0"), nb::arg("d1"), nb::arg("dtype"),
      nb::rv_policy::take_ownership, "Create 2D matrix from host byte data");

    m.def("FromBytes", [](const std::vector<uint8_t>& data, OaI64 d0, OaScalarType dtype) {
        auto mat = OaFnMatrix::FromBytes(
            OaSpan<const OaU8>(data.data(), data.size()),
            OaMatrixShape{d0}, dtype);
        return matrix_ptr(std::move(mat));
    }, nb::arg("data"), nb::arg("d0"), nb::arg("dtype"),
      nb::rv_policy::take_ownership, "Create 1D matrix from host byte data");

    m.def("FromInt32", [](const std::vector<OaI32>& data, const std::vector<OaI64>& dims, OaScalarType dtype) {
        return matrix_ptr(OaFnMatrix::FromInt32(
            OaSpan<const OaI32>(data.data(), data.size()),
            shape_from_vector(dims), dtype));
    }, nb::arg("data"), nb::arg("shape"), nb::arg("dtype") = OaScalarType::Int32,
      nb::rv_policy::take_ownership, "Create matrix from host int32 data");

    // FromFloats — upload host float data as a Float32 matrix. Fills the gap left by
    // FromBytes (raw byte reinterpret) and FromInt32 (int upload): there was no
    // first-class way to get arbitrary float *values* onto the device — only the
    // Full constant. Integer inputs (Scale/matmul on a UInt8 matrix) silently
    // produce garbage, so any float feature tensor (e.g. normalized image pixels)
    // must come in as Float32 via this path.
    m.def("FromFloats", [](const std::vector<float>& data, const std::vector<OaI64>& dims) {
        return matrix_ptr(OaFnMatrix::FromBytes(
            OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(data.data()),
                               data.size() * sizeof(float)),
            shape_from_vector(dims), OaScalarType::Float32));
    }, nb::arg("data"), nb::arg("shape"), nb::rv_policy::take_ownership,
      "Create a Float32 matrix from host float data");

    m.def("FromFloats", [](const std::vector<float>& data, OaI64 d0, OaI64 d1) {
        return matrix_ptr(OaFnMatrix::FromBytes(
            OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(data.data()),
                               data.size() * sizeof(float)),
            OaMatrixShape{d0, d1}, OaScalarType::Float32));
    }, nb::arg("data"), nb::arg("d0"), nb::arg("d1"), nb::rv_policy::take_ownership,
      "Create a 2D Float32 matrix from host float data");

    m.def("FromFloats", [](const std::vector<float>& data, OaI64 d0) {
        return matrix_ptr(OaFnMatrix::FromBytes(
            OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(data.data()),
                               data.size() * sizeof(float)),
            OaMatrixShape{d0}, OaScalarType::Float32));
    }, nb::arg("data"), nb::arg("d0"), nb::rv_policy::take_ownership,
      "Create a 1D Float32 matrix from host float data");

}
