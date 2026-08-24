"""
Configuration for OA type generation.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent.parent
SCHEMA_DIR = Path(__file__).resolve().parent / "schema"
DEFAULT_OUTPUT = REPO_ROOT / "build" / "gen" / "type"
LIVE_SOURCE_ROOT = REPO_ROOT

# Domain-specific output paths
# Generated header goes to source/cpp/include/ (public), source/cpp/lib/ (private)
DOMAIN_OUTPUT_PATHS = {
	"core": {
		"header": "source/cpp/include/oa/core/type.gen.h",
		"cpp": "source/cpp/lib/oa/core/type.gen.cpp",
	},
	"ml": {
		"header": "source/cpp/include/oa/ml/type.gen.h",
		"cpp": "source/cpp/lib/oa/ml/type.gen.cpp",
	},
	"vision": {
		"header": "source/cpp/include/oa/vision/type.gen.h",
		"cpp": "source/cpp/lib/oa/vision/type.gen.cpp",
	},
	"ui": {
		"header": "source/cpp/include/oa/ui/type.gen.h",
		"cpp": "source/cpp/lib/oa/ui/type.gen.cpp",
	},
	"runtime": {
		"header": "source/cpp/lib/oa/runtime/type.gen.h",
		"cpp": "source/cpp/lib/oa/runtime/type.gen.cpp",
	},
	"audio": {
		"header": "source/cpp/include/oa/audio/type.gen.h",
		"cpp": "source/cpp/lib/oa/audio/type.gen.cpp",
	},
	"render": {
		"header": "source/cpp/include/oa/render/type.gen.h",
		"cpp": "source/cpp/lib/oa/render/type.gen.cpp",
	},
}
