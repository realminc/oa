// OaEnvFlag — Single-source environment variable convention for OA.
//
// Two reading patterns:
//   1. Bool toggle:    OaEnvFlag::IsSet("OA_DISABLE_COOPMAT")
//                      Returns true for any non-empty value that is NOT one of
//                      "0", "false", "no", "off" (case-insensitive).
//   2. String override: OaEnvFlag::GetString("OA_FORCE_PRECISION", "FP32")
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
//
//   Force overrides (string or numeric override)
//     OA_FORCE_PRECISION=FP32|BF16|FP16     Override OaEngineDesc::Precision.
//     OA_FORCE_COOPMAT=1                    Bypass vendor-trust blacklist.
//     OA_FORCE_COOPVEC=1                    Bypass CoopVec NVIDIA-only routing gate.
//     OA_FORCE_DEVICE_INDEX=N               Override device pick.
//     OA_SHADER_LOAD_THREADS=0|1|N          Shader preload workers: 0=automatic
//                                           (serial warm / physical cores cold),
//                                           1=serial, N=explicit worker count.
//
//   Diagnostic logs (opt-in extra output, all cost-free when off)
//     OA_LOG_GEMM_ROUTER=1              Per-call runtime GEMM routing decision log.
//     OA_LOG_PIPELINE_LOAD=1            Per-shader load timing + status.
//     OA_LOG_BARRIERS=1                 Barrier-count summary per graph compile.
//     OA_LOG_CONTEXT_GRAPH=N            Log the first N OaContext graphs before
//                                       execution, including node shaders/groups.
//     OA_GRAPH_REPORT=path|1             Write the first captured training
//                                       program as deterministic JSON. Value 1
//                                       uses var/report/training_graph.json.
//     OA_LOG_COOPMAT_SHAPES=1           Log enumerated coopmat shapes at device init.
//     OA_LOG_NUMERIC_DEVIATIONS=1       Per-test summary of max-observed deviation
//                                       per tolerance tier (planned —
//                                       requires per-EXPECT_NEAR_* harness wiring).

#pragma once

#include <Oa/Core/Types.h>

class OaEnvFlag {
public:
	// Bool toggle: true iff the env var is set AND its value is NOT one of
	// {"", "0", "false", "no", "off"} (case-insensitive).
	[[nodiscard]] static bool IsSet(const char* InName);

	// String override: returns env value if set+non-empty, else returns
	// InDefault as an OaString. Empty value falls back to InDefault.
	[[nodiscard]] static OaString GetString(const char* InName, const char* InDefault = "");

	// Integer override: returns parsed env value if set+non-empty+parsable,
	// else returns InDefault. Parsing accepts decimal only.
	[[nodiscard]] static OaI64 GetInt(const char* InName, OaI64 InDefault = 0);

	// Programmatic override — sets the env value to "1" if not already set.
	// Returns false if the env var was already set externally (caller's
	// signal that user-supplied env wins over OaEngineConfig::NumericMode).
	// Used internally by OaApplyNumericMode; not the API to expose to users.
	static bool SetIfUnset(const char* InName, const char* InValue);
};

// Forward decl — defined in Runtime/Engine.h. Kept here to avoid pulling the
// full Engine.h into every Core consumer of OaEnvFlag.
enum class OaNumericMode : OaU8;

// Translate an OaNumericMode to the equivalent env-knob state for the rest of
// the runtime. Called once at engine init from OaEngine::Create.
//
// Numeric-mode mapping:
//   Fast           no-op
//   Stable         OA_FORCE_PRECISION=FP32, OA_DISABLE_COOPMAT=1
//   Deterministic  + OA_DISABLE_PERSISTENT_LOOP=1
//
// Env vars set externally by the user always win (SetIfUnset checks first).
// Logs to OA at INFO each var the call actually touched.
void OaApplyNumericMode(OaNumericMode InMode);
