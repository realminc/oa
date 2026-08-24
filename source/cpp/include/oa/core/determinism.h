// OA determinism Mode — Runtime control for numeric behavior
// Provides Fast/Stable/Deterministic modes for compute operations

#pragma once

#include <oa/core/types.h>

namespace oa {

// get current determinism mode from environment variable
// environment variable: OA_DETERMINISM_MODE (Fast/Stable/Deterministic)
// Default: Stable for training contexts, Fast for inference
[[nodiscard]] DeterminismMode getDeterminismMode();

// set determinism mode programmatically (overrides environment variable)
void setDeterminismMode(DeterminismMode inMode);

// Check if current mode is fast (vendor math, non-deterministic reductions)
[[nodiscard]] bool isFastMode();

// Check if current mode is stable (FP32 accumulators, deterministic where possible)
[[nodiscard]] bool isStableMode();

// Check if current mode is deterministic (fixed reduction order, no race-dependent atomics)
[[nodiscard]] bool isDeterministicMode();

} // namespace oa
