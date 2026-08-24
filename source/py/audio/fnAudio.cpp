// OA Python bindings — GPU audio transform and signal operations.
#include "../binding.h"

#include <oa/audio/fnAudio.h>

void bindAudioFn(nb::module_& m) {
	// Signatures, PascalCase arguments, defaults, return ownership, and docs
	// come from the same schemas as the C++ audio operation contracts.
#include "fnAudioOps.gen.inl"
}
