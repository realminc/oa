// OA private native Python module. The public API lives under the pure-Python
// `oa` package; `_oa` only owns native type registration and implementation.
#include "Binding.h"

NB_MODULE(_oa, m) {
    m.doc() = "Private native implementation for the OA Python package";

    auto core = m.def_submodule("core", "OA tensors and functional operations");
    auto runtime = m.def_submodule("runtime", "OA Vulkan runtime and execution contexts");
    auto ml = m.def_submodule("ml", "OA machine-learning modules and training");
    auto audio = m.def_submodule("audio", "OA audio codecs and GPU signal operations");
    auto crypto = m.def_submodule("crypto", "OA cryptography and GPU hashing");
    auto vision = m.def_submodule("vision", "OA image processing and Vulkan Video");
    auto plot = m.def_submodule("plot", "OA metric plots and evaluation figures");

    // Registration order matters: shared matrix and enum types must exist before
    // higher-level modules refer to them.
    BindCore(core);
    BindRuntime(runtime);
    BindMl(ml);
    BindAudio(audio);
    BindVision(vision);
    BindPlot(plot);
#ifdef OA_BUILD_CRYPTO
    BindCrypto(crypto);
#else
    crypto.attr("available") = false;
#endif
}
