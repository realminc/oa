// OA Python native module — shared helpers + per-module registration hooks.
//
// Each source module registers into its matching Python submodule. All bindings
// remain in one private extension so nanobind has one cross-module type registry.
#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

#include <Oa/Core/MatrixShape.h>
#include <Oa/Core/Types.h>
#include <Oa/Core/Matrix.h>
#include <Oa/Core/Status.h>

#include <vector>
#include <string>
#include <stdexcept>
#include <utility>

namespace nb = nanobind;

class OaEngine;
class OaImage;
struct OaViewerConfig;

// ─── Shared helpers (header-inline so every binding TU shares one definition) ─

inline void throw_if_error(const OaStatus& status) {
    if (!status.IsOk()) {
        throw std::runtime_error(status.ToString().c_str());
    }
}

inline OaMatrixShape shape_from_vector(const std::vector<OaI64>& dims) {
    if (dims.size() > static_cast<size_t>(OA_MAX_TENSOR_DIMS)) {
        throw std::runtime_error("OaMatrixShape rank exceeds OA_MAX_TENSOR_DIMS");
    }

    OaMatrixShape shape;
    shape.Rank = static_cast<OaI32>(dims.size());
    for (OaI32 i = 0; i < shape.Rank; ++i) {
        shape[i] = dims[static_cast<size_t>(i)];
    }
    return shape;
}

inline std::vector<OaI64> shape_to_vector(const OaMatrixShape& shape) {
    std::vector<OaI64> dims;
    dims.reserve(static_cast<size_t>(shape.Rank));
    for (OaI32 i = 0; i < shape.Rank; ++i) {
        dims.push_back(shape[i]);
    }
    return dims;
}

inline OaMatrix* matrix_ptr(OaMatrix&& matrix) {
    return new OaMatrix(std::move(matrix));
}

// Accept OA paths, strings, pathlib.Path, and any other Python os.PathLike.
// PyOS_FSPath is the canonical Python protocol boundary; OA keeps OaPath as the
// native public value rather than leaking pathlib into user-facing examples.
inline OaPath path_from_python(nb::handle value) {
    PyObject* raw_path = PyOS_FSPath(value.ptr());
    if (raw_path == nullptr) {
        throw nb::python_error();
    }
    nb::object owned_path = nb::steal<nb::object>(nb::handle(raw_path));

    if (PyUnicode_Check(raw_path)) {
        Py_ssize_t size = 0;
        const char* data = PyUnicode_AsUTF8AndSize(raw_path, &size);
        if (data == nullptr) {
            throw nb::python_error();
        }
        std::string path(data, static_cast<size_t>(size));
        if (path.find('\0') != std::string::npos) {
            PyErr_SetString(PyExc_ValueError, "path contains an embedded null");
            throw nb::python_error();
        }
        return OaPath(std::move(path));
    }
    if (PyBytes_Check(raw_path)) {
        std::string path(
            PyBytes_AS_STRING(raw_path),
            static_cast<size_t>(PyBytes_GET_SIZE(raw_path)));
        if (path.find('\0') != std::string::npos) {
            PyErr_SetString(PyExc_ValueError, "path contains an embedded null");
            throw nb::python_error();
        }
        return OaPath(std::move(path));
    }

    PyErr_SetString(PyExc_TypeError, "__fspath__ must return str or bytes");
    throw nb::python_error();
}

// Process-scoped engine owned by Runtime/Runtime.cpp. Domain bindings use this
// instead of exposing OaEngine or raw Vulkan handles to Python.
[[nodiscard]] OaEngine& PythonEngine();
// Blocking Viewer sessions release the GIL. These helpers retain the runtime
// host lock so explicit shutdown cannot destroy the borrowed engine mid-call.
[[nodiscard]] OaStatus PythonViewerShow(
    const OaMatrix& image,
    const OaViewerConfig& config);
[[nodiscard]] OaStatus PythonViewerShow(
    const OaImage& image,
    const OaViewerConfig& config);
// Register a process-finalization callback after native object registration.
// It runs after Python-owned OA objects are released but before C++ static
// destructors and loader teardown.
void RegisterPythonRuntimeExitHook();

// ─── Per-module registration (Core.cpp / Runtime.cpp / Ml.cpp / Audio.cpp) ───

void BindCore(nb::module_& m);
void BindCoreType(nb::module_& m);
void BindCoreFilesystem(nb::module_& m);
void BindCoreFactory(nb::module_& m);
void BindCoreFnMatrix(nb::module_& m);
void BindCoreBackward(nb::module_& m);

void BindRuntime(nb::module_& m);

void BindMl(nb::module_& m);
void BindMlModule(nb::module_& m);
void BindMlNn(nb::module_& m);
void BindMlLoss(nb::module_& m);
void BindMlAutograd(nb::module_& m);
void BindMlOptim(nb::module_& m);
void BindMlTraining(nb::module_& m);
void BindMlNlp(nb::module_& m);
void BindMlRl(nb::module_& m);

void BindAudio(nb::module_& m);
void BindAudioType(nb::module_& m);
void BindAudioCodec(nb::module_& m);
void BindAudioFn(nb::module_& m);

void BindVision(nb::module_& m);
void BindVisionType(nb::module_& m);
void BindVisionImage(nb::module_& m);
void BindVisionDetection(nb::module_& m);
void BindVisionCodec(nb::module_& m);
void BindVisionVideo(nb::module_& m);

void BindPlot(nb::module_& m);
void BindViewer(nb::module_& m);

void BindCrypto(nb::module_& m);
void BindCryptoHash(nb::module_& m);
void BindCryptoSign(nb::module_& m);
void BindCryptoFnHash(nb::module_& m);
