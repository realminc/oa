#pragma once

#include <oa/ml/transferWeights.h>

namespace oa {

// Register the exact openai/clip-vit-large-patch14 text-tower translator used
// by oa::Alm. Idempotent: repeated calls are harmless.
[[nodiscard]] Status registerClipTextModelTranslator();

} // namespace oa
