// OA Python bindings — ML parameter and module base types.
#include "../Binding.h"

#include <Oa/Ml/Module.h>
#include <Oa/Ml/Nn.h>

#include <string>

void BindMlModule(nb::module_& m) {
    nb::enum_<OaActivation>(m, "OaActivation")
        .value("None", OaActivation::None)
        .value("Relu", OaActivation::Relu)
        .value("Gelu", OaActivation::Gelu);

    // ═════════════════════════════════════════════════════════════════════════
    // OaParameter
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<OaParameter>(m, "OaParameter")
        .def_prop_ro("Name", [](const OaParameter& p) { return p.Name.c_str(); })
        .def_prop_rw("Data",
            [](OaParameter& p) -> OaMatrix& { return p.Data; },
            [](OaParameter& p, const OaMatrix& m) { p.Data = m; },
            nb::rv_policy::reference_internal)
        // Grad routes to the single source of truth on Data's autograd meta — there
        // is no separate snapshot field. The getter returns a handle sharing the live
        // GPU buffer; the setter writes THROUGH to that same buffer (for hand-wired
        // backward passes), so it can never desync from what the optimizer reads.
        .def_prop_rw("Grad",
            [](const OaParameter& p) { return p.Grad(); },
            [](OaParameter& p, const OaMatrix& g) { p.Grad() = g; })
        .def_rw("RequiresGrad", &OaParameter::RequiresGrad);

    // ═════════════════════════════════════════════════════════════════════════
    // OaModule
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<OaModule>(m, "OaModule")
        .def("Forward", [](OaModule& self, const OaMatrix& input) {
            return matrix_ptr(self.Forward(input));
        }, nb::arg("input"), nb::rv_policy::take_ownership)
        .def("Parameters", [](OaModule& self) -> std::vector<OaParameter*> {
            auto& params = self.Parameters();
            std::vector<OaParameter*> result;
            result.reserve(params.Size());
            for (auto& p : params) result.push_back(&p);
            return result;
        }, nb::rv_policy::reference_internal,
           "Get direct parameters only. Use AllParameterPtrs for nested modules.")
        .def("AllParameterPtrs", [](OaModule& self) -> std::vector<OaParameter*> {
            auto params = self.AllParameterPtrs();
            std::vector<OaParameter*> result;
            result.reserve(params.Size());
            for (auto* p : params) result.push_back(p);
            return result;
        }, nb::rv_policy::reference_internal,
           "Get recursive trainable parameter pointers.")
        .def("NumParameters", &OaModule::NumParameters)
        .def("SetName", [](OaModule& self, const std::string& name) { self.SetName(name.c_str()); })
        .def("GetName", [](const OaModule& self) { return self.GetName().c_str(); })
        // Persistence — dotted-path tree walk over registered params/children into an
        // .oam file. Works on any module (a leaf like OaLinear, or a nested OaRnn/OaGru),
        // so a composed model persists by round-tripping its submodules.
        .def("Save", [](const OaModule& self, const std::string& path) {
            throw_if_error(self.Save(OaString(path.c_str())));
        }, nb::arg("path"), "Serialize module parameters to an .oam file")
        .def("Load", [](OaModule& self, const std::string& path) {
            throw_if_error(self.Load(OaString(path.c_str())));
        }, nb::arg("path"), "Load module parameters from an .oam file");
}
