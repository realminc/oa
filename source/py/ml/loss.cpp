// OA Python bindings — loss functions.
#include "../binding.h"

#include <oa/ml/fnLoss.h>
#include <oa/ml/fnLoss.h>

void bindMlLoss(nb::module_& m) {
    // ═════════════════════════════════════════════════════════════════════════
    // oa::FnLoss
    // ═════════════════════════════════════════════════════════════════════════

#include "fnLossOps.gen.inl"
}
