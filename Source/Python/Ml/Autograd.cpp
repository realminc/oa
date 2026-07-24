// OA Python bindings — autograd state and gradient tape.
#include "../Binding.h"

#include <Oa/Ml/Autograd.h>

void BindMlAutograd(nb::module_& m) {
    // ═════════════════════════════════════════════════════════════════════════
    // OaGradientTape
    // ═════════════════════════════════════════════════════════════════════════

    m.def("IsEnabled", []() { return OaFnAutograd::IsEnabled(); });
    m.def("GradEnabled", []() { return OaFnAutograd::IsEnabled(); });
    m.def("SetEnabled", [](bool v) { OaFnAutograd::SetEnabled(v); });
    m.def("SetGradEnabled", [](bool v) { OaFnAutograd::SetEnabled(v); });

    nb::class_<OaGradientTape>(m, "GradientTape", nb::is_final())
        .def(nb::init<>(), "Create autograd tape (RAII: enables gradient tracking in constructor, restores in destructor)")
        .def("Backward", &OaGradientTape::Backward, nb::arg("Root"),
             "Reverse-mode autodiff from scalar loss root")
        .def("Close", &OaGradientTape::Close,
             "Restore the gradient-enabled state captured by this tape")
        .def("__enter__", [](OaGradientTape& self) -> OaGradientTape& { return self; }, nb::rv_policy::reference)
        .def("__exit__", [](OaGradientTape& self, nb::args) {
            self.Close();
        });
}
