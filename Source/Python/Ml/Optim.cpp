// OA Python bindings — optimizers and optimizer composition.
#include "../Binding.h"

#include <Oa/Ml/Module.h>
#include <Oa/Ml/Optim.h>
#include <Oa/Ml/OptimUtil.h>

void BindMlOptim(nb::module_& m) {
    // ═════════════════════════════════════════════════════════════════════════
    // OaOptimizer (base class)
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<OaOptimizer>(m, "OaOptimizer")
        .def("Step", &OaOptimizer::Step)
        .def("ZeroGrad", &OaOptimizer::ZeroGrad)
        .def("SetLr", &OaOptimizer::SetLr, nb::arg("Lr"))
        .def("Lr", &OaOptimizer::Lr)
        .def("GetLr", &OaOptimizer::GetLr)
        .def("GetStep", &OaOptimizer::GetStep);

    nb::class_<OaOptimizerNoOp, OaOptimizer>(m, "OaOptimizerNoOp")
        .def(nb::init<>());

    nb::class_<OaSGD, OaOptimizer>(m, "OaSGD")
        .def("__init__", [](OaSGD* self, nb::list params, OaF32 lr, OaF32 momentum,
                            OaF32 weight_decay) {
            OaVec<OaParameter*> ptrs;
            for (auto item : params) {
                ptrs.PushBack(nb::cast<OaParameter*>(item));
            }
            new (self) OaSGD(OaSpan<OaParameter*>(ptrs.Data(), ptrs.Size()),
                lr, momentum, weight_decay);
        }, nb::arg("Params"), nb::arg("Lr") = 1e-2f, nb::arg("Momentum") = 0.0f,
           nb::arg("WeightDecay") = 0.0f)
        .def("Step", &OaSGD::Step)
        .def("ZeroGrad", &OaSGD::ZeroGrad);

    nb::class_<OaAdam, OaOptimizer>(m, "OaAdam")
        .def("__init__", [](OaAdam* self, nb::list params, OaF32 lr, OaF32 beta1,
                            OaF32 beta2, OaF32 eps) {
            OaVec<OaParameter*> ptrs;
            for (auto item : params) {
                ptrs.PushBack(nb::cast<OaParameter*>(item));
            }
            new (self) OaAdam(OaSpan<OaParameter*>(ptrs.Data(), ptrs.Size()),
                lr, beta1, beta2, eps);
        }, nb::arg("Params"), nb::arg("Lr") = 1e-3f, nb::arg("Beta1") = 0.9f,
           nb::arg("Beta2") = 0.999f, nb::arg("Eps") = 1e-8f)
        .def("Step", &OaAdam::Step)
        .def("ZeroGrad", &OaAdam::ZeroGrad);

    // ═════════════════════════════════════════════════════════════════════════
    // OaAdamW
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<OaAdamW, OaOptimizer>(m, "OaAdamW")
        .def("__init__", [](OaAdamW* self, nb::list params, OaF32 lr, OaF32 beta1,
                            OaF32 beta2, OaF32 eps, OaF32 weight_decay) {
            OaVec<OaParameter*> ptrs;
            for (auto item : params) {
                ptrs.PushBack(nb::cast<OaParameter*>(item));
            }
            new (self) OaAdamW(OaSpan<OaParameter*>(ptrs.Data(), ptrs.Size()),
                lr, beta1, beta2, eps, weight_decay);
        }, nb::arg("Params"), nb::arg("Lr") = 1e-3f, nb::arg("Beta1") = 0.9f,
           nb::arg("Beta2") = 0.999f, nb::arg("Eps") = 1e-8f,
           nb::arg("WeightDecay") = 0.01f,
           "AdamW optimizer (decoupled weight decay)")
        .def("Step", &OaAdamW::Step, "Apply one optimizer step (update weights)")
        .def("ZeroGrad", &OaAdamW::ZeroGrad, "Zero all parameter gradients");

    // ═════════════════════════════════════════════════════════════════════════
    // OaMuon
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<OaMuon, OaOptimizer>(m, "OaMuon")
        .def("__init__", [](OaMuon* self, nb::list params, OaF32 lr, OaF32 beta,
                            OaF32 weight_decay, OaF32 eps, OaI32 ns5_iters) {
            OaVec<OaParameter*> ptrs;
            for (auto item : params) {
                ptrs.PushBack(nb::cast<OaParameter*>(item));
            }
            new (self) OaMuon(OaSpan<OaParameter*>(ptrs.Data(), ptrs.Size()),
                lr, beta, weight_decay, eps, ns5_iters);
        }, nb::arg("Params"), nb::arg("Lr") = 1e-3f, nb::arg("Beta") = 0.95f,
           nb::arg("WeightDecay") = 0.1f, nb::arg("Eps") = 1e-7f, nb::arg("Ns5Iters") = 5,
           "Muon optimizer (2D hidden matrices)")
        .def("Step", &OaMuon::Step)
        .def("ZeroGrad", &OaMuon::ZeroGrad);

    // ═════════════════════════════════════════════════════════════════════════
    // OaOptimizerComposite + Muon+AdamW factory
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<OaOptimizerComposite, OaOptimizer>(m, "OaOptimizerComposite")
        .def(nb::init<>())
        .def("Step", &OaOptimizerComposite::Step)
        .def("ZeroGrad", &OaOptimizerComposite::ZeroGrad);

    nb::class_<OaMuonAdamWConfig>(m, "OaMuonAdamWConfig")
        .def(nb::init<>())
        .def_rw("MuonLr", &OaMuonAdamWConfig::MuonLr)
        .def_rw("AdamWLr", &OaMuonAdamWConfig::AdamWLr)
        .def_rw("MuonBeta", &OaMuonAdamWConfig::MuonBeta)
        .def_rw("MuonWeightDecay", &OaMuonAdamWConfig::MuonWeightDecay)
        .def_rw("MuonEps", &OaMuonAdamWConfig::MuonEps)
        .def_rw("MuonNs5Iters", &OaMuonAdamWConfig::MuonNs5Iters)
        .def_rw("AdamWBeta1", &OaMuonAdamWConfig::AdamWBeta1)
        .def_rw("AdamWBeta2", &OaMuonAdamWConfig::AdamWBeta2)
        .def_rw("AdamWEps", &OaMuonAdamWConfig::AdamWEps)
        .def_rw("AdamWWeightDecay", &OaMuonAdamWConfig::AdamWWeightDecay);

    m.def("MakeMuonAdamWOptimizer",
        [](OaModule& model, const OaMuonAdamWConfig& cfg) {
            return MakeMuonAdamWOptimizer(model, cfg).Release();
        },
        nb::arg("Model"), nb::arg("Config") = OaMuonAdamWConfig(),
        nb::rv_policy::take_ownership,
        "Build official Muon+AdamW composite (Muon on 2D body, AdamW on embed/head/1D)");
}
