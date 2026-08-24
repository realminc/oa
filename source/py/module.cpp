// OA private native Python module. The public API lives under the pure-Python
// `oa` package; `_oa` only owns native type registration and implementation.
#include "binding.h"

NB_MODULE(_oa, m) {
    m.doc() = "Private native implementation for the OA Python package";

    auto core = m.def_submodule("core", "OA tensors and functional operations");
    auto runtime = m.def_submodule("runtime", "OA Vulkan runtime and execution contexts");
    auto ml = m.def_submodule("ml", "OA machine-learning modules and training");
    auto audio = m.def_submodule("audio", "OA audio codecs and GPU signal operations");
    auto crypto = m.def_submodule("crypto", "OA cryptography and GPU hashing");
    auto vision = m.def_submodule("vision", "OA image processing and Vulkan Video");
    auto ui = m.def_submodule("ui", "OA viewer and interactive presentation sessions");
    auto plot = m.def_submodule("plot", "OA metric plots and evaluation figures");
    auto fnMatrix = m.def_submodule(
        "FnMatrix", "Native Python view of C++ namespace oa::FnMatrix");
    auto fnLoss = m.def_submodule(
        "FnLoss", "Native Python view of C++ namespace oa::FnLoss");
    auto fnAutograd = m.def_submodule(
        "FnAutograd", "Native Python view of C++ namespace oa::FnAutograd");
    auto fnMetric = m.def_submodule(
        "FnMetric", "Native Python view of C++ namespace oa::FnMetric");
    auto fnAdvantage = m.def_submodule(
        "FnAdvantage", "Native Python view of C++ namespace oa::FnAdvantage");
    auto fnEnvironment = m.def_submodule(
        "FnEnvironment", "Native Python view of C++ namespace oa::FnEnvironment");
    auto fnPolicy = m.def_submodule(
        "FnPolicy", "Native Python view of C++ namespace oa::FnPolicy");
    auto fnAudio = m.def_submodule(
        "FnAudio", "Native Python view of C++ namespace oa::FnAudio");
    auto fnImage = m.def_submodule(
        "FnImage", "Native Python view of C++ namespace oa::FnImage");
    auto fnDetection = m.def_submodule(
        "FnDetection", "Native Python view of C++ namespace oa::FnDetection");

    // Registration order matters: shared matrix and enum types must exist before
    // higher-level modules refer to them.
    bindCore(core, fnMatrix);
    bindRuntime(runtime);
    bindMl(
        ml, fnMatrix, fnLoss, fnAutograd, fnMetric,
        fnAdvantage, fnEnvironment, fnPolicy);
    bindAudio(audio, fnAudio);
    bindVision(vision, fnImage, fnDetection);
    bindViewer(ui);
    bindPlot(plot);
#ifdef OA_BUILD_CRYPTO
    auto fnHash = m.def_submodule(
        "FnHash", "Native Python view of C++ namespace oa::FnHash");
    bindCrypto(crypto, fnHash);
#else
    crypto.attr("available") = false;
#endif

    registerPythonRuntimeExitHook();
}
