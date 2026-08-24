// OA Python native module — shared helpers + per-module registration hooks.
//
// Each source module registers into its matching Python submodule. All bindings
// remain in one private extension so nanobind has one cross-module type registry.
#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

#include <oa/core/matrixShape.h>
#include <oa/core/types.h>
#include <oa/core/matrix.h>
#include <oa/core/status.h>

#include <vector>
#include <string>
#include <stdexcept>
#include <utility>

namespace nb = nanobind;

namespace oa {
class Engine;
class Image;
struct ViewerConfig;
}

// ─── Shared helpers (header-inline so every binding TU shares one definition) ─

inline void throwIfError(const oa::Status& status) {
    if (!status.isOk()) {
        throw std::runtime_error(status.toString().cStr());
    }
}

inline oa::MatrixShape shapeFromVector(const std::vector<oa::I64>& dims) {
    if (dims.size() > static_cast<size_t>(OA_MAX_TENSOR_DIMS)) {
        throw std::runtime_error("oa::MatrixShape rank exceeds OA_MAX_TENSOR_DIMS");
    }

    oa::MatrixShape shape;
    shape.rank = static_cast<oa::I32>(dims.size());
    for (oa::I32 i = 0; i < shape.rank; ++i) {
        shape[i] = dims[static_cast<size_t>(i)];
    }
    return shape;
}

inline std::vector<oa::I64> shapeToVector(const oa::MatrixShape& shape) {
    std::vector<oa::I64> dims;
    dims.reserve(static_cast<size_t>(shape.rank));
    for (oa::I32 i = 0; i < shape.rank; ++i) {
        dims.push_back(shape[i]);
    }
    return dims;
}

inline oa::Matrix* matrixPtr(oa::Matrix&& matrix) {
    return new oa::Matrix(std::move(matrix));
}

// Accept OA paths, strings, pathlib.path, and any other Python os.pathLike.
// PyOS_FSPath is the canonical Python protocol boundary; OA keeps oa::Path as the
// native public value rather than leaking pathlib into user-facing examples.
inline oa::Path pathFromPython(nb::handle value) {
    PyObject* rawPath = PyOS_FSPath(value.ptr());
    if (rawPath == nullptr) {
        throw nb::python_error();
    }
    nb::object ownedPath = nb::steal<nb::object>(nb::handle(rawPath));

    if (PyUnicode_Check(rawPath)) {
        Py_ssize_t size = 0;
        const char* data = PyUnicode_AsUTF8AndSize(rawPath, &size);
        if (data == nullptr) {
            throw nb::python_error();
        }
        std::string path(data, static_cast<size_t>(size));
        if (path.find('\0') != std::string::npos) {
            PyErr_SetString(PyExc_ValueError, "path contains an embedded null");
            throw nb::python_error();
        }
        return oa::Path(std::move(path));
    }
    if (PyBytes_Check(rawPath)) {
        std::string path(
            PyBytes_AS_STRING(rawPath),
            static_cast<size_t>(PyBytes_GET_SIZE(rawPath)));
        if (path.find('\0') != std::string::npos) {
            PyErr_SetString(PyExc_ValueError, "path contains an embedded null");
            throw nb::python_error();
        }
        return oa::Path(std::move(path));
    }

    PyErr_SetString(PyExc_TypeError, "__fspath__ must return str or bytes");
    throw nb::python_error();
}

// Process-scoped engine owned by Runtime/Runtime.cpp. Domain bindings use this
// instead of exposing oa::Engine or raw Vulkan handles to Python.
[[nodiscard]] oa::Engine& pythonEngine();
// Blocking Viewer sessions release the GIL. These helpers retain the runtime
// host lock so explicit shutdown cannot destroy the borrowed engine mid-call.
[[nodiscard]] oa::Status pythonViewerShow(
    const oa::Matrix& image,
    const oa::ViewerConfig& config);
[[nodiscard]] oa::Status pythonViewerShow(
    const oa::Image& image,
    const oa::ViewerConfig& config);
// Register a process-finalization callback after native object registration.
// It runs after Python-owned OA objects are released but before C++ static
// destructors and loader teardown.
void registerPythonRuntimeExitHook();

// ─── Per-module registration (Core.cpp / Runtime.cpp / Ml.cpp / Audio.cpp) ───

void bindCore(nb::module_& m, nb::module_& inFnMatrix);
void bindCoreType(nb::module_& m);
void bindCoreFilesystem(nb::module_& m);
void bindCoreFactory(nb::module_& m);
void bindCoreFnMatrix(nb::module_& m);
void bindCoreBackward(nb::module_& m);

void bindRuntime(nb::module_& m);

void bindMl(
    nb::module_& m,
    nb::module_& inFnMatrix,
    nb::module_& inFnLoss,
    nb::module_& inFnAutograd,
    nb::module_& inFnMetric,
    nb::module_& inFnAdvantage,
    nb::module_& inFnEnvironment,
    nb::module_& inFnPolicy);
void bindMlFnMatrix(nb::module_& m);
void bindMlModule(nb::module_& m);
void bindMlNn(nb::module_& m);
void bindMlLoss(nb::module_& m);
void bindMlAutograd(nb::module_& m, nb::module_& inFnAutograd);
void bindMlOptim(nb::module_& m);
void bindTraining(nb::module_& m);
void bindMlMetric(nb::module_& inFnMetric);
void bindMlNlp(nb::module_& m);
void bindMlReinforcement(
	nb::module_& m,
	nb::module_& inFnAdvantage,
	nb::module_& inFnEnvironment,
	nb::module_& inFnPolicy);

void bindAudio(nb::module_& m, nb::module_& inFnAudio);
void bindAudioType(nb::module_& m);
void bindAudioSession(nb::module_& m);
void bindAudioCodec(nb::module_& inFnAudio);
void bindAudioFn(nb::module_& m);

void bindVision(
    nb::module_& m,
    nb::module_& inFnImage,
    nb::module_& inFnDetection);
void bindVisionType(nb::module_& m);
void bindVisionImage(nb::module_& m, nb::module_& inFnImage);
void bindVisionDetection(nb::module_& m, nb::module_& inFnDetection);
void bindVisionCodec(nb::module_& m, nb::module_& inFnImage);
void bindVisionVideo(nb::module_& m);

void bindPlot(nb::module_& m);
void bindViewer(nb::module_& m);

void bindCrypto(nb::module_& m, nb::module_& inFnHash);
void bindCryptoHash(nb::module_& m);
void bindCryptoSign(nb::module_& m);
void bindCryptoFnHash(nb::module_& m);
