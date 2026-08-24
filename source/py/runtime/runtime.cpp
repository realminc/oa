// OA Python bindings — runtime (oa/runtime/*): engine lifecycle.
#include "../binding.h"

#include <oa/runtime/engine.h>
#include <oa/core/log.h>

#include <oa/ui/viewer.h>
#include <oa/ui/viewerPlatform.h>

#include <mutex>

// Process-scoped engine / context lifetime. The engine is pinned (non-movable):
// Create returns an owning oa::UniquePtr, which we hold directly.
static oa::ViewerPlatformLease gViewerPlatform;
static oa::UniquePtr<oa::Engine> gEngine;
static std::mutex gEngineMutex;
static bool gPresentationCapable = false;
static bool gExitHookRegistered = false;

static void shutdownPythonRuntimeAtExit() {
    std::lock_guard lock(gEngineMutex);
    gEngine.reset();
    gPresentationCapable = false;
    gViewerPlatform.release();
}

void registerPythonRuntimeExitHook() {
    std::lock_guard lock(gEngineMutex);
    if (gExitHookRegistered) {
        return;
    }
    if (Py_AtExit(&shutdownPythonRuntimeAtExit) != 0) {
        throw std::runtime_error(
            "OA could not register its native Python runtime shutdown hook");
    }
    gExitHookRegistered = true;
}

static bool initPythonEngineLocked() {
    if (gEngine != nullptr) {
        return true;
    }

    oa::EngineConfig cfg;
    cfg.appName = "oa_python";
    cfg.selectForThread = true;
    // Python promises lazy first use. Eagerly building every embedded pipeline
    // makes a small script pay for unrelated ML, audio, and crypto kernels and
    // is especially pathological under GPU-assisted/validation instrumentation.
    cfg.preloadEmbeddedPipelines = false;

    const oa::Status platformStatus = gViewerPlatform.acquire(&cfg);
    const bool requestedPresentation = platformStatus.isOk();
    if (not requestedPresentation) {
        OaLogWarn(oa::LogComponent::Python,
            "OA Python presentation unavailable; requesting compute-only engine: %s",
            platformStatus.toString().cStr());
    }

    auto result = oa::Engine::create(cfg);
    if (not result.isOk() and requestedPresentation) {
        OaLogWarn(oa::LogComponent::Python,
            "OA Python presentation-capable engine creation failed; "
            "retrying compute-only: %s",
            result.getStatus().toString().cStr());
        gViewerPlatform.release();
        oa::EngineConfig computeCfg;
        computeCfg.appName = "oa_python";
        computeCfg.presentationMode = oa::PresentationMode::None;
        computeCfg.selectForThread = true;
        computeCfg.preloadEmbeddedPipelines = false;
        result = oa::Engine::create(computeCfg);
    }
    if (!result.isOk()) {
        return false;
    }
    gEngine = std::move(result).getValue();
    gPresentationCapable =
        requestedPresentation and gViewerPlatform.isAcquired()
        and gEngine->hasGraphics();
    return true;
}

oa::Engine& pythonEngine() {
    std::lock_guard lock(gEngineMutex);
    if (!initPythonEngineLocked()) {
        throw std::runtime_error(
            "OA could not create a Vulkan compute engine for the first "
            "device-backed Python operation");
    }
    return *gEngine;
}

oa::Status pythonViewerShow(
    const oa::Matrix& image,
    const oa::ViewerConfig& config) {
    std::lock_guard lock(gEngineMutex);
    if (!initPythonEngineLocked()) {
        return oa::Status::error(
            oa::StatusCode::Unavailable,
            "OA could not create a Vulkan engine for oa::Viewer");
    }
    return oa::Viewer::show(*gEngine, image, config);
}

oa::Status pythonViewerShow(
    const oa::Image& image,
    const oa::ViewerConfig& config) {
    std::lock_guard lock(gEngineMutex);
    if (!initPythonEngineLocked()) {
        return oa::Status::error(
            oa::StatusCode::Unavailable,
            "OA could not create a Vulkan engine for oa::Viewer");
    }
    return oa::Viewer::show(*gEngine, image, config);
}

oa::Status pythonViewerPreview(
    const oa::String& path,
    const oa::ViewerConfig& config) {
    std::lock_guard lock(gEngineMutex);
    if (!initPythonEngineLocked()) {
        return oa::Status::error(
            oa::StatusCode::Unavailable,
            "OA could not create a Vulkan engine for oa::Viewer");
    }
    oa::ViewerConfig resolved = config;
    resolved.path = path;
    oa::Viewer viewer(resolved);
    return viewer.run(*gEngine);
}

void bindRuntime(nb::module_& m) {
    // ═════════════════════════════════════════════════════════════════════════
    // Engine init / teardown
    // ═════════════════════════════════════════════════════════════════════════

    m.def("initComputeEngine", []() -> bool {
        std::lock_guard lock(gEngineMutex);
        return initPythonEngineLocked();
    }, "Eagerly initialize OA's process-scoped compute engine. Device-backed operations initialize it lazily.");

    m.def("shutdownComputeEngine", []() {
        std::lock_guard lock(gEngineMutex);
        gEngine.reset();
        gPresentationCapable = false;
        gViewerPlatform.release();
    }, "Shutdown OA compute engine and release resources.");

    m.def("_pythonEngineInitialized", []() -> bool {
        std::lock_guard lock(gEngineMutex);
        return gEngine != nullptr;
    }, "Internal lifecycle probe used by Python import and first-use tests.");

    m.def("_pythonEnginePresentationCapable", []() -> bool {
        std::lock_guard lock(gEngineMutex);
        return gEngine != nullptr and gPresentationCapable;
    }, "Internal capability probe used by Viewer tests.");

}
