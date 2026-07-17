// OA Python bindings — ML registration order.
#include "../Binding.h"

void BindMl(nb::module_& m) {
    BindMlModule(m);
    BindMlNn(m);
    BindMlLoss(m);
    BindMlAutograd(m);
    BindMlOptim(m);
    BindMlTraining(m);
    BindMlRl(m);
}
