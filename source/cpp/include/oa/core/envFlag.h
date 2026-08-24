// EnvFlag — Single-source environment variable convention for OA.
//
// Two reading patterns:
//   1. bool toggle:    EnvFlag::isSet("OA_DISABLE_COOPMAT")
//                      Returns true for any non-empty value that is NOT one of
//                      "0", "false", "no", "off" (case-insensitive).
//   2. String override: EnvFlag::getString("OA_FORCE_PRECISION", "FP32")
//                       Returns the env value if set+non-empty, else the default.
//
// Recognized OA env knobs (canonical public list):
//
//   Disable toggles (route around a code path)
//     OA_DISABLE_COOPMAT                Skip CoopMat extension enable + route to scalar.
//     OA_DISABLE_COOPMAT2               Skip VK_NV_cooperative_matrix2 enable.
//     OA_DISABLE_BF16                   Force FP32; skip VK_KHR_shader_bfloat16 enable.
//     OA_DISABLE_PERSISTENT_LOOP        Force single-step submission.
//     OA_DISABLE_INTEGER_DOT_PRODUCT    Skip VK_KHR_shader_integer_dot_product enable.
//     OA_DISABLE_GRU_SCAN               Use decomposed GRU cells instead of the fused scan.
//     OA_DISABLE_GEMM_ROUTE_CACHE       Skip measured GEMM route-cache replay.
//     OA_DISABLE_NARROW_ROW_KERNELS     Skip narrow LayerNorm/softmax schedules.
//     OA_DISABLE_TILED_BMM              Skip the shared-memory BMM schedule.
//     OA_DISABLE_LINEAR_PARAM_ROWS32    Skip the narrow Linear parameter-adjoint schedule.
//
//   Force overrides (string or numeric override)
//     OA_FORCE_PRECISION=FP32|BF16|FP16     Override EngineConfig::precision.
//     OA_FORCE_COOPMAT=1                    Bypass vendor-trust blacklist.
//     OA_FORCE_COOPVEC=1                    Bypass CoopVec NVIDIA-only routing gate.
//     OA_FORCE_DEVICE_INDEX=N               Override device pick.
//     OA_SHADER_LOAD_THREADS=0|1|N          shader preload workers: 0=automatic
//                                           (serial warm / physical cores cold),
//                                           1=serial, N=explicit worker count.
//
//   Diagnostic logs (opt-in extra output, all cost-free when off)
//     OA_VK_VALIDATION=1               Enable vulkan validation for this run.
//     OA_LOG_GEMM_ROUTER=1              Per-call runtime GEMM routing decision log.
//     OA_LOG_PIPELINE_LOAD=1            Per-shader load timing + status.
//     OA_LOG_BARRIERS=1                 Barrier-count summary per graph compile.
//     OA_LOG_CONTEXT_GRAPH=N            log the first N Context graphs before
//                                       execution, including node shaders/groups.
//     OA_GRAPH_REPORT=path|1             Write the first captured training
//                                       program as deterministic JSON. Value 1
//                                       uses var/report/training_graph.json.
//     OA_LOG_COOPMAT_SHAPES=1           log enumerated coopmat shapes at device init.
//     OA_LOG_NUMERIC_DEVIATIONS=1       Per-test summary of max-observed deviation
//                                       per tolerance tier (planned —
//                                       requires per-EXPECT_NEAR_* harness wiring).

#pragma once

#include <oa/core/types.h>

namespace oa {

class EnvFlag {
public:
	// bool toggle: true iff the env var is set AND its value is NOT one of
	// {"", "0", "false", "no", "off"} (case-insensitive).
	[[nodiscard]] static bool isSet(const char* inName);

	// String override: returns env value if set+non-empty, else returns
	// inDefault as an oa::String. Empty value falls back to inDefault.
	[[nodiscard]] static oa::String getString(const char* inName, const char* inDefault = "");

	// Integer override: returns parsed env value if set+non-empty+parsable,
	// else returns inDefault. Parsing accepts decimal only.
	[[nodiscard]] static oa::I64 getInt(const char* inName, oa::I64 inDefault = 0);

	// Programmatic override — sets the env value to "1" if not already set.
	// Returns false if the env var was already set externally (caller's
	// signal that user-supplied env wins over oa::EngineConfig::numericMode).
	// Used internally by applyNumericMode; not the API to expose to users.
	static bool setIfUnset(const char* inName, const char* inValue);
};

// process-wide numerical execution policy. Core owns the vocabulary and its
// environment mapping; Runtime consumes it through oa::EngineConfig.
enum class NumericMode : oa::U8 {
	Fast          = 0,
	Stable        = 1,
	Deterministic = 2,
};

// translate an NumericMode to the equivalent env-knob state for the rest of
// the runtime. Called once at engine init from oa::Engine::Create.
//
// Numeric-mode mapping:
//   Fast           no-op
//   Stable         OA_FORCE_PRECISION=FP32, OA_DISABLE_COOPMAT=1
//   Deterministic  + OA_DISABLE_PERSISTENT_LOOP=1
//
// env vars set externally by the user always win (SetIfUnset checks first).
// Logs to OA at INFO each var the call actually touched.
void applyNumericMode(NumericMode inMode);

} // namespace oa
