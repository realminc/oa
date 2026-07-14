"""
Configuration for OaTypeAutogen.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent.parent
SCHEMA_DIR = REPO_ROOT / "Tools" / "TypeAutogen" / "Schema"
DEFAULT_OUTPUT = REPO_ROOT / "Tools" / "TypeAutogen" / "Output"
LIVE_SOURCE_ROOT = REPO_ROOT

# Domain-specific output paths
# Generated header goes to Public/ (for includes), cpp goes to Private/
DOMAIN_OUTPUT_PATHS = {
	"Core": {
		"header": "Source/Public/Oa/Core/Type.gen.h",
		"cpp": "Source/Private/Oa/Core/Type.gen.cpp",
	},
	"Ml": {
		"header": "Source/Public/Oa/Ml/Type.gen.h",
		"cpp": "Source/Private/Oa/Ml/Type.gen.cpp",
	},
	"Vision": {
		"header": "Source/Public/Oa/Vision/Type.gen.h",
		"cpp": "Source/Private/Oa/Vision/Type.gen.cpp",
	},
	"Ui": {
		"header": "Source/Public/Oa/Ui/Type.gen.h",
		"cpp": "Source/Private/Oa/Ui/Type.gen.cpp",
	},
	"Runtime": {
		"header": "Source/Public/Oa/Runtime/Type.gen.h",
		"cpp": "Source/Private/Oa/Runtime/Type.gen.cpp",
	},
	"Audio": {
		"header": "Source/Public/Oa/Audio/Type.gen.h",
		"cpp": "Source/Private/Oa/Audio/Type.gen.cpp",
	},
	"Render": {
		"header": "Source/Public/Oa/Render/Type.gen.h",
		"cpp": "Source/Private/Oa/Render/Type.gen.cpp",
	},
	"Animation": {
		"header": "Source/Public/Oa/Animation/Type.gen.h",
		"cpp": "Source/Private/Oa/Animation/Type.gen.cpp",
	},
}
