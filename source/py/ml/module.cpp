// OA Python bindings — ML parameter and module base types.
#include "../binding.h"

#include <oa/ml/module.h>
#include <oa/ml/nn.h>
#include <oa/ml/optim.h>

#include <string>

void bindMlModule(nb::module_& m) {
    nb::enum_<oa::Activation>(m, "Activation")
        .value("None", oa::Activation::None)
        .value("Relu", oa::Activation::Relu)
        .value("Gelu", oa::Activation::Gelu)
        .value("Silu", oa::Activation::Silu);

    // ═════════════════════════════════════════════════════════════════════════
    // oa::Parameter
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<oa::Parameter>(m, "Parameter")
        .def_prop_ro("name", [](const oa::Parameter& p) { return p.name.cStr(); })
        .def_prop_rw("data",
            [](oa::Parameter& p) -> oa::Matrix& { return p.data; },
            [](oa::Parameter& p, const oa::Matrix& m) { p.data = m; },
            nb::rv_policy::reference_internal)
        // Grad routes to the single source of truth on Data's autograd meta — there
        // is no separate snapshot field. The getter returns a handle sharing the live
        // GPU buffer; the setter writes THROUGH to that same buffer (for hand-wired
        // backward passes), so it can never desync from what the optimizer reads.
        .def_prop_rw("grad",
            [](const oa::Parameter& p) { return p.grad(); },
            [](oa::Parameter& p, const oa::Matrix& g) { p.grad() = g; })
        .def_rw("requiresGrad", &oa::Parameter::requiresGrad);

    // ═════════════════════════════════════════════════════════════════════════
    // oa::Module
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<oa::Module>(m, "Module")
        .def("forward", [](oa::Module& self, const oa::Matrix& input) {
            return matrixPtr(self.forward(input));
        }, nb::arg("input"), nb::rv_policy::take_ownership)
        .def("parameters", [](oa::Module& self) -> std::vector<oa::Parameter*> {
            auto& params = self.parameters();
            std::vector<oa::Parameter*> result;
            result.reserve(params.size());
            for (auto& p : params) result.push_back(&p);
            return result;
        }, nb::rv_policy::reference_internal,
           "Get direct parameters only. Use AllParameterPtrs for nested modules.")
        .def("allParameterPtrs", [](oa::Module& self) -> std::vector<oa::Parameter*> {
            auto params = self.allParameterPtrs();
            std::vector<oa::Parameter*> result;
            result.reserve(params.size());
            for (auto* p : params) result.push_back(p);
            return result;
        }, nb::rv_policy::reference_internal,
           "Get recursive trainable parameter pointers.")
        .def("numParameters", &oa::Module::numParameters)
        .def("setName", [](oa::Module& self, const std::string& name) { self.setName(name.c_str()); })
        .def("getName", [](const oa::Module& self) { return self.getName().cStr(); })
        // Persistence — dotted-path tree walk over registered params/children into an
        // .oam file. Works on any module (a leaf like oa::Linear, or a nested oa::Rnn/oa::Gru),
        // so a composed model persists by round-tripping its submodules.
        .def("save", [](const oa::Module& self, nb::handle path) {
            const auto native = pathFromPython(path);
            throwIfError(self.save(pythonEngine(), native.string()));
        }, nb::arg("path"), "Serialize module parameters to an .oam file")
        .def("save", [](const oa::Module& self, nb::handle path,
                        const oa::Optimizer& optimizer) {
            const auto native = pathFromPython(path);
            throwIfError(self.save(pythonEngine(), native.string(), optimizer));
        }, nb::arg("path"), nb::arg("optimizer"),
           "Serialize module parameters and optimizer state to an .oam file")
        .def("load", [](oa::Module& self, nb::handle path) {
            const auto native = pathFromPython(path);
            throwIfError(self.load(pythonEngine(), native.string()));
        }, nb::arg("path"), "Load module parameters from an .oam file")
        .def("load", [](oa::Module& self, nb::handle path,
                        oa::Optimizer& optimizer) {
            const auto native = pathFromPython(path);
            throwIfError(self.load(pythonEngine(), native.string(), optimizer));
        }, nb::arg("path"), nb::arg("optimizer"),
           "Load module parameters and optimizer state from an .oam file");
}
