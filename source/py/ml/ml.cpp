// OA Python bindings — ML registration order.
#include "../binding.h"

void bindMl(
    nb::module_& m,
    nb::module_& inFnMatrix,
    nb::module_& inFnLoss,
    nb::module_& inFnAutograd,
    nb::module_& inFnMetric,
    nb::module_& inFnAdvantage,
    nb::module_& inFnEnvironment,
    nb::module_& inFnPolicy) {
    bindMlFnMatrix(inFnMatrix);
    bindMlModule(m);
    bindMlNn(m);
    bindMlAutograd(m, inFnAutograd);
    bindMlOptim(m);
    bindTraining(m);
    bindMlMetric(inFnMetric);
    bindMlNlp(m);
    bindMlReinforcement(
        m, inFnAdvantage, inFnEnvironment, inFnPolicy);
    // RL owns the PPO/DQN/SAC configuration and structured-result types used
    // by the generated oa::FnLoss registrations, including their Python defaults.
    bindMlLoss(inFnLoss);
}
