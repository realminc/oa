#pragma once

#include <Oa/Ml/TransferWeights.h>

// Register the exact openai/clip-vit-large-patch14 text-tower adapter used by
// OaAlm. Idempotent: repeated calls are harmless.
[[nodiscard]] OaStatus OaRegisterClipTextWeightAdapter();
