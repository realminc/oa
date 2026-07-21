from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY = REPO_ROOT / "Source" / "Public" / "Oa" / "Core" / "KernelRegistry.h"
SCHEMA_DIR = Path(__file__).resolve().parents[1] / "Schema"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "Output"
LIVE_SOURCE_ROOT = REPO_ROOT / "Source"

VALID_KINDS = {"binary", "unary", "unary_scalar", "nullary_scalar", "reduce_full"}
VALID_BODIES = {"auto", "manual_context", "bias_add_broadcast", "cpp_expr", "cpu_util"}
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
VALID_CONTRACT_VALUE_KINDS = {"matrix", "image", "audio_buffer", "video_frame"}
VALID_CONTRACT_SHAPE_RULES = {"match_input", "broadcast", "matmul_nt", "explicit"}
VALID_CONTRACT_DTYPE_RULES = {"match_input", "promote_float"}
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

# Domain-specific namespace mapping (default — schemas may override via `namespace`)
DOMAIN_NAMESPACE = {
	"Core": "OaFnMatrix",
	"Ml": "OaFnMatrix",
	"Audio": "OaFnAudio",
	"Vision": "OaFnImage",
	"Ui": "OaFnMatrix",
	"Crypto": "OaFnHash",
}

# (Domain, namespace) → extra public header to include in generated .cpp.
# Multiple domains (Core/Ml/Ui) all extend the OaFnMatrix namespace via their
# own FnMatrix.h header — each carries the declarations for its own schemas.
NAMESPACE_HEADER = {
	("Core",   "OaFnMatrix"): None,
	("Ml",     "OaFnMatrix"): "Oa/Ml/FnMatrix.h",
	("Ml",     "OaFnLoss"):   "Oa/Ml/FnLoss.h",
	("Ui",     "OaFnMatrix"): "Oa/Ui/FnMatrix.h",
	("Vision", "OaFnImage"):  "Oa/Vision/FnImage.h",
	("Vision", "OaFnVideo"):  "Oa/Vision/FnVideo.h",
	("Audio",  "OaFnAudio"):  "Oa/Audio/FnAudio.h",
	("Crypto", "OaFnHash"):   "Oa/Crypto/FnHash.h",
}

DOMAIN_FILE_PREFIX = {
	"Core": "FnMatrix",
	"Ml": "FnMatrix",
	"Audio": "FnAudio",
	"Vision": "FnImage",
	"Ui": "FnMatrix",
	"Crypto": "Fn",
}

DOMAIN_SUBDIR = {
	"Core": "FnMatrix",
	"Ml": "FnMatrix",
	"Audio": "FnAudio",
	"Vision": "FnImage",
	"Ui": "Matrix",
	"Crypto": "use_file_category",
}

# Domain-specific subdir for Loss namespace (separate from DOMAIN_SUBDIR)
LOSS_SUBDIR = "FnLoss"
