// OA Python bindings — Audio registration order.
#include "../Binding.h"

void BindAudio(nb::module_& m) {
    BindAudioType(m);
    BindAudioCodec(m);
    BindAudioFn(m);
}
