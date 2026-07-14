// OA Python Bindings — Runtime (Oa/Runtime/*): engine init/teardown, OaContext.
#include "../Binding.h"

#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/RuntimeGlobal.h>

// Process-scoped engine / context lifetime. The engine is pinned (non-movable):
// Create returns an owning OaUniquePtr, which we hold directly.
static OaUniquePtr<OaComputeEngine> gEngine;
static OaContext* gContext = nullptr;

OaComputeEngine& PythonComputeEngine() {
    if (gEngine == nullptr) {
        throw std::runtime_error(
            "OA compute engine is not initialized; call "
            "oa.runtime.OaInitComputeEngine() first");
    }
    return *gEngine;
}

void BindRuntime(nb::module_& m) {
    // ═════════════════════════════════════════════════════════════════════════
    // Engine init / teardown
    // ═════════════════════════════════════════════════════════════════════════

    m.def("OaInitComputeEngine", []() -> bool {
        if (gContext != nullptr) {
            return true; // Already initialized
        }
        OaEngineConfig cfg;
        cfg.AppName = "oa_python";
        cfg.PresentationMode = OaPresentationMode::None;
        cfg.RegisterAsGlobal = true;

        auto result = OaComputeEngine::Create(cfg);
        if (!result.IsOk()) {
            return false;
        }
        gEngine = std::move(result).GetValue();
        OaRuntimeGlobal::SetRuntime(gEngine.get());

        gContext = OaContext::Create(gEngine.get());
        OaContext::SetDefault(gContext);
        return true;
    }, "Initialize OA compute engine and default context. Must be called before any GPU operations.");

    m.def("OaShutdownComputeEngine", []() {
        if (gContext != nullptr) {
            delete gContext;
            gContext = nullptr;
        }
        gEngine.reset();
    }, "Shutdown OA compute engine and release resources.");

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

    m.def("OaContext_GetDefault", []() -> OaContext& {
        return OaContext::GetDefault();
    }, nb::rv_policy::reference, "Get the thread-local default context");

    // ═════════════════════════════════════════════════════════════════════════
    // Context helpers (manual Execute/Sync; Python-level context manager in
    // oa/__init__.py wraps these)
    // ═════════════════════════════════════════════════════════════════════════

}
