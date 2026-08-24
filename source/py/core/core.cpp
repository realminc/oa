// OA Python bindings — Core registration order.
#include "../binding.h"

void bindCore(nb::module_& m, nb::module_& inFnMatrix) {
    bindCoreType(m);
    bindCoreFilesystem(m);
    bindCoreFactory(inFnMatrix);
    bindCoreFnMatrix(inFnMatrix);
    bindCoreBackward(inFnMatrix);
}
