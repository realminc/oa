// OA Determinism Mode — Runtime control for numeric behavior
// Provides Fast/Stable/Deterministic modes for compute operations

#pragma once

#include <Oa/Core/Types.h>

// Get current determinism mode from environment variable
// Environment variable: OA_DETERMINISM_MODE (Fast/Stable/Deterministic)
// Default: Stable for training contexts, Fast for inference
[[nodiscard]] OaDeterminismMode OaGetDeterminismMode();

// Set determinism mode programmatically (overrides environment variable)
void OaSetDeterminismMode(OaDeterminismMode InMode);

// Check if current mode is Fast (vendor math, non-deterministic reductions)
[[nodiscard]] bool OaIsFastMode();

// Check if current mode is Stable (FP32 accumulators, deterministic where possible)
[[nodiscard]] bool OaIsStableMode();

// Check if current mode is Deterministic (fixed reduction order, no race-dependent atomics)
[[nodiscard]] bool OaIsDeterministicMode();
