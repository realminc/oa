from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY = (
	REPO_ROOT / "source" / "cpp" / "lib" / "oa" / "runtime"
	/ "kernelRegistry.h"
)
SCHEMA_DIR = Path(__file__).resolve().parent / "schema"
DEFAULT_OUTPUT = REPO_ROOT / "build" / "gen" / "fn"
LIVE_SOURCE_ROOT = REPO_ROOT / "source"

VALID_KINDS = {
	"binary",
	"unary",
	"unary_scalar",
	"nullary_scalar",
	"reduce_full",
	"session_command",
}
VALID_BODIES = {"auto", "manual_session", "bias_add_broadcast", "cpp_expr", "cpu_util"}
VALID_FORMULAS_PREFIX = (
	"manual", "manual:",
	"auto_elemwise", "auto_unary", "auto_binary_", "auto_unary_scalar_", "auto_bwd_",
	"none",
)
VALID_DISPATCH_WORKGROUPS = {"elemwise", "output_elemwise"}
VALID_OUTPUT_SHAPES = {"match_input", "scalar_1d", "last_dim_half"}
VALID_OUTPUT_DTYPES = {
	"match_input",
	"Bool",
	"UInt8",
	"UInt16",
	"UInt32",
	"UInt64",
	"Int8",
	"Int16",
	"Int32",
	"Int64",
	"Float16",
	"BFloat16",
	"Float32",
	"Float64",
	"Complex64",
	"Complex128",
}
SLANG_EMIT_KINDS = {"binary", "unary", "unary_scalar", "nullary_scalar"}
AUTO_BODY_KINDS = {"binary", "unary", "unary_scalar", "nullary_scalar", "reduce_full"}

# Schema-v2 semantic operation contracts. These describe the operation before
# kernel selection and are intentionally independent of Vulkan/runtime details.
VALID_CONTRACT_VALUE_KINDS = {
	"matrix", "image", "audio", "video_frame", "quant_matrix"
}
VALID_CONTRACT_SHAPE_RULES = {"match_input", "broadcast", "matmul_nt", "explicit"}
VALID_CONTRACT_DTYPE_RULES = {"match_input", "promote_float", "explicit"}
VALID_CONTRACT_EFFECTS = {"read_inputs", "write_outputs"}
VALID_CONTRACT_DIFFERENTIATION = {"none", "reverse"}
VALID_CONTRACT_LOWERING = {"dispatch", "gemm"}
VALID_CONTRACT_CONTROL_FLOW = {"straight_line", "conditional", "loop"}
VALID_CONTRACT_ATTRIBUTE_KINDS = {
	"boolean",
	"signed_integer",
	"unsigned_integer",
	"float",
	"string",
	"shape",
	"enum",
}

# Private neural-network graph-planner metadata. These labels classify one
# schema-owned semantic operation for `oa::Dnn` pattern matching; they are not
# public API, kernel identities, or executable-route promises.
VALID_DNN_ROLES = {
	"add",
	"bias_add",
	"flash_attention_causal",
	"gated_multiply",
	"gelu",
	"grouped_gemm",
	"matmul",
	"multiply",
	"relu",
	"residual_rms_norm",
	"rms_norm",
	"silu",
}
VALID_DNN_EPILOGUES = {
	"none",
	"bias",
	"bias_relu",
	"bias_gelu",
	"bias_silu",
}

# Schema-owned fixed-kernel metadata. operation generation only emits identities for
# built-in OA kernels mechanically tied to an operation schema.
VALID_KERNEL_PREFIXES = {"Ml", "Crypto", "Vision", "Ui", "Audio", "Render"}
VALID_KERNEL_CATEGORIES = {
	"Ml", "Math", "Crypto", "Vision", "Ui", "Audio", "Render"
}

# Architectural classification of schema rows. Only operation/composite/session
# rows are semantic operations; views, kernels, aliases, and host observations
# have different ownership and must not inflate operation-contract coverage.
VALID_OPERATION_SURFACES = {
	"public_operation",
	"public_cpp_operation",
	"stable_composite",
	"value_view",
	"internal_operation",
	"kernel",
	"session_command",
	"alias",
	"host_observation",
	"host_utility",
}

# Stateful commands are described separately from stateless operation
# contracts. The owning session defines its detailed state machine; the schema
# freezes the visible transition, effects, and completion boundary.
VALID_SESSION_EFFECTS = {
	"read_state",
	"write_state",
	"device_work",
	"host_input",
	"host_output",
}
VALID_SESSION_COMPLETIONS = {
	"recorded",
	"output_event",
	"synchronous",
}

# Domain-specific namespace mapping (default — schemas may override via `namespace`).
# Keys are lowercase to match on-disk schema directory names (audio/, vision/, etc.).
DOMAIN_NAMESPACE = {
	"core": "oa::FnMatrix",
	"ml": "oa::FnMatrix",
	"audio": "oa::FnAudio",
	"vision": "oa::FnImage",
	"ui": "oa::FnMatrix",
	"crypto": "oa::FnHash",
	"data": "oa::FnDataset",
	"render": "oa::FnCamera",
	"runtime": "oa::FnTexture",
}

# (Domain, namespace) → extra public header to include in generated .cpp.
# Multiple domains (core/ml/ui) all extend the oa::FnMatrix namespace via their
# own fnMatrix.h header — each carries the declarations for its own schemas.
NAMESPACE_HEADER = {
	("core",   "oa::FnMatrix"): None,
	("ml",     "oa::FnMatrix"): "oa/ml/fnMatrix.h",
	("ml",     "oa::FnLoss"):   "oa/ml/fnLoss.h",
	("ml",     "oa::FnAdvantage"): "oa/ml/advantage.h",
	("ml",     "oa::FnEnvironment"): "oa/ml/fnEnvironment.h",
	("ml",     "oa::FnPolicy"): "oa/ml/policy.h",
	("ml",     "oa::PolicyEvaluator"): "oa/ml/policyEvaluator.h",
	("ui",     "oa::FnMatrix"): "oa/ui/fnMatrix.h",
	("vision", "oa::FnImage"):  "oa/vision/fnImage.h",
	("vision", "oa::FnDetection"): "oa/vision/fnDetection.h",
	("vision", "oa::FnVideo"):  "oa/vision/fnVideo.h",
	("audio",  "oa::FnAudio"):  "oa/audio/fnAudio.h",
	("crypto", "oa::FnHash"):   "oa/crypto/fnHash.h",
	("data",   "oa::FnDataset"): "oa/data/fnDataset.h",
	("render", "oa::FnCamera"): "oa/render/fnCamera.h",
	("render", "oa::FnMesh"):   "oa/render/fnMesh.h",
	("runtime", "oa::FnTexture"): "oa/runtime/texture.h",
}

DOMAIN_FILE_PREFIX = {
	"core": "FnMatrix",
	"ml": "FnMatrix",
	"audio": "FnAudio",
	"vision": "FnImage",
	"ui": "FnMatrix",
	"crypto": "FnHash",
	"data": "FnDataset",
	"render": "FnCamera",
	"runtime": "FnTexture",
}

DOMAIN_SUBDIR = {
	"core": "FnMatrix",
	"ml": "FnMatrix",
	"audio": "FnAudio",
	"vision": "FnImage",
	"ui": "FnMatrix",
	"crypto": "use_file_category",
	"data": "FnDataset",
	"render": "use_file_category",
	"runtime": "FnTexture",
}

# Domain-specific subdir for Loss namespace (separate from DOMAIN_SUBDIR)
LOSS_SUBDIR = "Loss"
