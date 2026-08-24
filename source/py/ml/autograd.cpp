// OA Python bindings — autograd state and gradient tape.
#include "../binding.h"

#include <oa/ml/autograd.h>

void bindMlAutograd(nb::module_& m, nb::module_& inFnAutograd) {
    // ═════════════════════════════════════════════════════════════════════════
    // oa::GradientTape
    // ═════════════════════════════════════════════════════════════════════════

    inFnAutograd.def("isEnabled", []() {
        return oa::FnAutograd::isEnabled();
    });
    inFnAutograd.def("setEnabled", [](bool inEnabled) {
        oa::FnAutograd::setEnabled(inEnabled);
    }, nb::arg("enabled"));

    nb::class_<oa::GradientTape>(m, "GradientTape", nb::is_final())
        .def(nb::init<>(), "Create autograd tape (RAII: enables gradient tracking in constructor, restores in destructor)")
        .def("backward", [](oa::GradientTape& self, const oa::Matrix& inRoot) {
            throwIfError(self.tryBackward(inRoot));
        }, nb::arg("root"), "Reverse-mode autodiff from scalar loss root")
        .def("close", &oa::GradientTape::close,
             "Restore the gradient-enabled state captured by this tape")
        .def("__enter__", [](oa::GradientTape& self) -> oa::GradientTape& { return self; }, nb::rv_policy::reference)
        .def("__exit__", [](oa::GradientTape& self, nb::args) {
            self.close();
        });
}
