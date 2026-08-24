// OA Python bindings — Audio registration order.
#include "../binding.h"

void bindAudio(nb::module_& m, nb::module_& inFnAudio) {
    bindAudioType(m);
    bindAudioSession(m);
    bindAudioCodec(inFnAudio);
    bindAudioFn(inFnAudio);
}
