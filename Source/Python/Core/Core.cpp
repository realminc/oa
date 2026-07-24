// OA Python bindings — Core registration order.
#include "../Binding.h"

void BindCore(nb::module_& m) {
    BindCoreType(m);
    BindCoreFilesystem(m);
    BindCoreFactory(m);
    BindCoreFnMatrix(m);
    BindCoreBackward(m);
}
