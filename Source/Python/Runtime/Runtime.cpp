// OA Python Bindings — Runtime (Oa/Runtime/*): engine init/teardown, OaContext.
#include "../Binding.h"

#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Core/Log.h>

#include <Oa/Ui/Viewer.h>
#include <Oa/Ui/ViewerPlatform.h>

#include <mutex>

// Process-scoped engine / context lifetime. The engine is pinned (non-movable):
// Create returns an owning OaUniquePtr, which we hold directly.
static OaViewerPlatformLease gViewerPlatform;
static OaUniquePtr<OaEngine> gEngine;
static std::mutex gEngineMutex;
static bool gPresentationCapable = false;
static bool gExitHookRegistered = false;

static void ShutdownPythonRuntimeAtExit() {
    std::lock_guard lock(gEngineMutex);
    gEngine.reset();
    gPresentationCapable = false;
    gViewerPlatform.Release();
}

void RegisterPythonRuntimeExitHook() {
    std::lock_guard lock(gEngineMutex);
    if (gExitHookRegistered) {
        return;
    }
    if (Py_AtExit(&ShutdownPythonRuntimeAtExit) != 0) {
        throw std::runtime_error(
            "OA could not register its native Python runtime shutdown hook");
    }
    gExitHookRegistered = true;
}

static bool InitPythonEngineLocked() {
    if (gEngine != nullptr) {
        return true;
    }

    OaEngineConfig cfg;
    cfg.AppName = "oa_python";
    cfg.RegisterAsGlobal = true;
    // Python promises lazy first use. Eagerly building every embedded pipeline
    // makes a small script pay for unrelated ML, audio, and crypto kernels and
    // is especially pathological under GPU-assisted/validation instrumentation.
    cfg.PreloadEmbeddedPipelines = false;

    const OaStatus platformStatus = gViewerPlatform.Acquire(&cfg);
    const bool requestedPresentation = platformStatus.IsOk();
    if (not requestedPresentation) {
        OA_LOG_WARN(OaLogComponent::App,
            "OA Python presentation unavailable; requesting compute-only engine: %s",
            platformStatus.ToString().c_str());
    }

    auto result = OaEngine::Create(cfg);
    if (not result.IsOk() and requestedPresentation) {
        OA_LOG_WARN(OaLogComponent::App,
            "OA Python presentation-capable engine creation failed; "
            "retrying compute-only: %s",
            result.GetStatus().ToString().c_str());
        gViewerPlatform.Release();
        OaEngineConfig computeCfg;
        computeCfg.AppName = "oa_python";
        computeCfg.PresentationMode = OaPresentationMode::None;
        computeCfg.RegisterAsGlobal = true;
        computeCfg.PreloadEmbeddedPipelines = false;
        result = OaEngine::Create(computeCfg);
    }
    if (!result.IsOk()) {
        return false;
    }
    gEngine = std::move(result).GetValue();
    gPresentationCapable =
        requestedPresentation and gViewerPlatform.IsAcquired()
        and gEngine->HasGraphics();
    return true;
}

OaEngine& PythonEngine() {
    std::lock_guard lock(gEngineMutex);
    if (!InitPythonEngineLocked()) {
        throw std::runtime_error(
            "OA could not create a Vulkan compute engine for the first "
            "device-backed Python operation");
    }
    return *gEngine;
}

OaStatus PythonViewerShow(
    const OaMatrix& image,
    const OaViewerConfig& config) {
    std::lock_guard lock(gEngineMutex);
    if (!InitPythonEngineLocked()) {
        return OaStatus::Error(
            OaStatusCode::Unavailable,
            "OA could not create a Vulkan engine for OaViewer");
    }
    return OaViewer::Show(gEngine->GetContext(), image, config);
}

OaStatus PythonViewerShow(
    const OaImage& image,
    const OaViewerConfig& config) {
    std::lock_guard lock(gEngineMutex);
    if (!InitPythonEngineLocked()) {
        return OaStatus::Error(
            OaStatusCode::Unavailable,
            "OA could not create a Vulkan engine for OaViewer");
    }
    return OaViewer::Show(gEngine->GetContext(), image, config);
}

void BindRuntime(nb::module_& m) {
    // ═════════════════════════════════════════════════════════════════════════
    // Engine init / teardown
    // ═════════════════════════════════════════════════════════════════════════

    m.def("OaInitComputeEngine", []() -> bool {
        std::lock_guard lock(gEngineMutex);
        return InitPythonEngineLocked();
    }, "Eagerly initialize OA's process-scoped compute engine. Device-backed operations initialize it lazily.");

    m.def("OaShutdownComputeEngine", []() {
        std::lock_guard lock(gEngineMutex);
        gEngine.reset();
        gPresentationCapable = false;
        gViewerPlatform.Release();
    }, "Shutdown OA compute engine and release resources.");

    m.def("_OaPythonEngineInitialized", []() -> bool {
        std::lock_guard lock(gEngineMutex);
        return gEngine != nullptr;
    }, "Internal lifecycle probe used by Python import and first-use tests.");

    m.def("_OaPythonEnginePresentationCapable", []() -> bool {
        std::lock_guard lock(gEngineMutex);
        return gEngine != nullptr and gPresentationCapable;
    }, "Internal capability probe used by Viewer tests.");

    // ═════════════════════════════════════════════════════════════════════════
    // OaContext (default context access + manual Execute/Sync)
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<OaContext>(m, "OaContext")
        .def("Execute", [](OaContext& self) {
            throw_if_error(self.Execute());
        }, "Execute all recorded GPU operations")
        .def("Sync", [](OaContext& self) {
            throw_if_error(self.Sync());
        }, "Synchronize (wait for GPU completion)");

    m.def("OaContextGetDefault", []() -> OaContext& {
        return PythonEngine().GetContext();
    }, nb::rv_policy::reference, "Get the thread-local default context");

    // ═════════════════════════════════════════════════════════════════════════
    // Context helpers (manual Execute/Sync; Python-level context manager in
    // oa/__init__.py wraps these)
    // ═════════════════════════════════════════════════════════════════════════

}
