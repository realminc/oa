// OA Python bindings — optimizers.
#include "../binding.h"

#include <oa/ml/module.h>
#include <oa/ml/optim.h>

void bindMlOptim(nb::module_& m) {
    // oa::Optimizer (base class)
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<oa::Optimizer>(m, "Optimizer")
        .def("step", &oa::Optimizer::step)
        .def("zeroGrad", &oa::Optimizer::zeroGrad)
        .def("setLr", &oa::Optimizer::setLr, nb::arg("lr"))
        .def("lr", &oa::Optimizer::lr)
        .def("getLr", &oa::Optimizer::getLr)
        .def("getStep", &oa::Optimizer::getStep);

    nb::class_<oa::OptimizerNoOp, oa::Optimizer>(m, "OptimizerNoOp")
        .def(nb::init<>());

    nb::class_<oa::Sgd, oa::Optimizer>(m, "Sgd")
        .def("__init__", [](oa::Sgd* self, nb::list params, oa::F32 lr, oa::F32 momentum,
                            oa::F32 weightDecay) {
            oa::Vec<oa::Parameter*> ptrs;
            for (auto item : params) {
                ptrs.pushBack(nb::cast<oa::Parameter*>(item));
            }
            new (self) oa::Sgd(oa::Span<oa::Parameter*>(ptrs.data(), ptrs.size()),
                lr, momentum, weightDecay);
        }, nb::arg("params"), nb::arg("lr") = 1e-2f, nb::arg("momentum") = 0.0f,
           nb::arg("weightDecay") = 0.0f)
        .def("step", &oa::Sgd::step)
        .def("zeroGrad", &oa::Sgd::zeroGrad);

    nb::class_<oa::Adam, oa::Optimizer>(m, "Adam")
        .def("__init__", [](oa::Adam* self, nb::list params, oa::F32 lr, oa::F32 beta1,
                            oa::F32 beta2, oa::F32 eps) {
            oa::Vec<oa::Parameter*> ptrs;
            for (auto item : params) {
                ptrs.pushBack(nb::cast<oa::Parameter*>(item));
            }
            new (self) oa::Adam(oa::Span<oa::Parameter*>(ptrs.data(), ptrs.size()),
                lr, beta1, beta2, eps);
        }, nb::arg("params"), nb::arg("lr") = 1e-3f, nb::arg("beta1") = 0.9f,
           nb::arg("beta2") = 0.999f, nb::arg("eps") = 1e-8f)
        .def("step", &oa::Adam::step)
        .def("zeroGrad", &oa::Adam::zeroGrad);

    // ═════════════════════════════════════════════════════════════════════════
    // oa::AdamW
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<oa::AdamW, oa::Optimizer>(m, "AdamW")
        .def("__init__", [](oa::AdamW* self, nb::list params, oa::F32 lr, oa::F32 beta1,
                            oa::F32 beta2, oa::F32 eps, oa::F32 weightDecay) {
            oa::Vec<oa::Parameter*> ptrs;
            for (auto item : params) {
                ptrs.pushBack(nb::cast<oa::Parameter*>(item));
            }
            new (self) oa::AdamW(oa::Span<oa::Parameter*>(ptrs.data(), ptrs.size()),
                lr, beta1, beta2, eps, weightDecay);
        }, nb::arg("params"), nb::arg("lr") = 1e-3f, nb::arg("beta1") = 0.9f,
           nb::arg("beta2") = 0.999f, nb::arg("eps") = 1e-8f,
           nb::arg("weightDecay") = 0.01f,
           "AdamW optimizer (decoupled weight decay)")
        .def("step", &oa::AdamW::step, "Apply one optimizer step (update weights)")
        .def("zeroGrad", &oa::AdamW::zeroGrad, "Zero all parameter gradients");

    // ═════════════════════════════════════════════════════════════════════════
    // oa::Muon
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<oa::Muon, oa::Optimizer>(m, "Muon")
        .def("__init__", [](oa::Muon* self, nb::list params, oa::F32 lr, oa::F32 beta,
                            oa::F32 weightDecay, oa::F32 eps, oa::I32 ns5Iters) {
            oa::Vec<oa::Parameter*> ptrs;
            for (auto item : params) {
                ptrs.pushBack(nb::cast<oa::Parameter*>(item));
            }
            new (self) oa::Muon(oa::Span<oa::Parameter*>(ptrs.data(), ptrs.size()),
                lr, beta, weightDecay, eps, ns5Iters);
        }, nb::arg("params"), nb::arg("lr") = 1e-3f, nb::arg("beta") = 0.95f,
           nb::arg("weightDecay") = 0.1f, nb::arg("eps") = 1e-7f, nb::arg("ns5Iters") = 5,
           "Muon optimizer (2D hidden matrices)")
        .def("step", &oa::Muon::step)
        .def("zeroGrad", &oa::Muon::zeroGrad);

    // ═════════════════════════════════════════════════════════════════════════
    // ═════════════════════════════════════════════════════════════════════════

}
