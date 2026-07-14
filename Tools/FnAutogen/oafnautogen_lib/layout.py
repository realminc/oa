from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import DOMAIN_FILE_PREFIX, DOMAIN_NAMESPACE, DOMAIN_SUBDIR, REPO_ROOT


@dataclass(frozen=True)
class SchemaLayout:
	domain: str
	namespace: str
	file_prefix: str
	cpp_subdir: str
	header_path: Path
	cpp_path: Path
	autograd_header_path: Path
	test_path: Path
	emit_header: bool = True
	emit_cpp: bool = True


def infer_domain(schema_path: Path) -> str:
	return schema_path.parent.name if schema_path.parent.name in DOMAIN_NAMESPACE else "Core"


def infer_cpp_subdir(domain: str, file_category: str) -> str:
	subdir_rule = DOMAIN_SUBDIR.get(domain, "Matrix")
	if subdir_rule != "use_file_category":
		return subdir_rule
	if file_category.startswith("Hash"):
		return "Hash"
	if file_category.startswith("Sign"):
		return "Sign"
	return file_category


def build_schema_layout(
	schema_path: Path,
	data: dict,
	file_category: str,
	out_root: Path,
	*,
	live: bool,
	emit_header: bool = True,
	emit_cpp: bool = True,
) -> SchemaLayout:
	domain = infer_domain(schema_path)
	namespace = data.get("namespace", DOMAIN_NAMESPACE.get(domain, "OaFnMatrix"))
	file_prefix = data.get("file_prefix", DOMAIN_FILE_PREFIX.get(domain, "FnMatrix"))
	cpp_subdir = data.get("cpp_subdir", infer_cpp_subdir(domain, file_category))
	# Use category subdirectories for Ml/FnMatrix domain (e.g., /FnMatrix/Activation/FnMatrixActivation.gen.h)
	# Use per-function subdirectories for Ml/FnLoss domain (e.g., /FnLoss/Mse/FnLossMse.gen.h)
	# Use category subdirectories for Core/FnMatrix domain (e.g., /FnMatrix/Blas/FnMatrixBlas.gen.h)
	# Use category subdirectories for Audio/FnAudio domain (e.g., /FnAudio/Signal/FnAudioSignal.gen.h)
	# Use category subdirectories for Vision/FnImage domain (e.g., /FnImage/Color/FnImageColor.gen.h)
	# Use category subdirectories for Vision/FnVideo domain (e.g., /FnVideo/Codec/FnVideoCodec.gen.h)
	if domain == "Ml" and cpp_subdir == "FnMatrix":
		category_subdir = file_category
	elif domain == "Ml" and cpp_subdir == "FnLoss":
		# For Loss, we'll use per-function subdirectories - handled in oafnautogen.py
		category_subdir = ""
	elif domain == "Core" and cpp_subdir == "FnMatrix":
		category_subdir = file_category
	elif domain == "Audio" and cpp_subdir == "FnAudio":
		category_subdir = file_category
	elif domain == "Vision" and cpp_subdir == "FnImage":
		category_subdir = file_category
	elif domain == "Vision" and cpp_subdir == "FnVideo":
		category_subdir = file_category
	else:
		category_subdir = ""
	h_path = out_root / "Private" / "Oa" / domain / cpp_subdir / category_subdir / f"{file_prefix}{file_category}.gen.h"
	cpp_path = out_root / "Private" / "Oa" / domain / cpp_subdir / category_subdir / f"{file_prefix}{file_category}.gen.cpp"
	# Autograd generated headers mirror the cpp_subdir domain but strip the "Fn" prefix
	# e.g. FnMatrix -> Matrix, FnLoss -> Loss, FnOptim -> Optim
	autograd_subdir = cpp_subdir.removeprefix("Fn") if cpp_subdir.startswith("Fn") else cpp_subdir
	autograd_h_path = out_root / "Private" / "Oa" / "Ml" / "Autograd" / autograd_subdir / f"Autograd{file_category}.gen.h"
	test_root = REPO_ROOT / "Test" if live else out_root / "Test"
	# Mirror source structure: Test/{Domain}/{Subdir}/{Category}/Test{Prefix}{Category}.gen.cpp
	test_path = test_root / domain / cpp_subdir / category_subdir / f"Test{file_prefix}{file_category}.gen.cpp"
	return SchemaLayout(
		domain=domain,
		namespace=namespace,
		file_prefix=file_prefix,
		cpp_subdir=cpp_subdir,
		header_path=h_path,
		cpp_path=cpp_path,
		autograd_header_path=autograd_h_path,
		test_path=test_path,
		emit_header=emit_header,
		emit_cpp=emit_cpp,
	)
