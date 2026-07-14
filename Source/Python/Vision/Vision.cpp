// OA Python bindings — Vision registration in dependency order.
#include "../Binding.h"

void BindVision(nb::module_& m) {
    BindVisionType(m);
    BindVisionImage(m);
    BindVisionCodec(m);
    BindVisionVideo(m);
}
