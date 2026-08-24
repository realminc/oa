#pragma once

#include <oa/core/status.h>

namespace oa {

class Engine;

// Installs the SDK's concrete environment kernels into exactly one engine.
// The engine remains the sole owner; repeated calls are idempotent and no
// process-global provider or search path is introduced.
[[nodiscard]] oa::Status ensureEnvironmentKernelPack(oa::Engine& inEngine);

} // namespace oa
