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
	emit_autograd: bool = False


def infer_domain(schema_path: Path) -> str:
	# Schema directories are lowercase (audio/, vision/, core/, etc.).
	# DOMAIN_NAMESPACE keys are also lowercase since the directory rename.
	name = schema_path.parent.name.lower()
	return name if name in DOMAIN_NAMESPACE else "core"


def camel_file_stem(name: str) -> str:
	return name[:1].lower() + name[1:]


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
	emit_autograd: bool = False,
) -> SchemaLayout:
	domain = infer_domain(schema_path)
	namespace = data.get("namespace", DOMAIN_NAMESPACE.get(domain, "oa::FnMatrix"))
	file_prefix = data.get("file_prefix", DOMAIN_FILE_PREFIX.get(domain, "Matrix"))
	cpp_subdir = data.get("cpp_subdir", infer_cpp_subdir(domain, file_category))
	# Generated internals mirror the Fn family and operation category while all
	# physical directories remain lowercase.
	if domain == "ml" and cpp_subdir == "FnMatrix":
		category_subdir = file_category
	elif domain == "ml" and cpp_subdir == "FnLoss":
		# For Loss, we'll use per-function subdirectories - handled in oafnautogen.py
		category_subdir = ""
	elif domain == "core" and cpp_subdir == "FnMatrix":
		category_subdir = file_category
	elif domain == "audio" and cpp_subdir == "FnAudio":
		category_subdir = file_category
	elif domain == "vision" and cpp_subdir == "FnImage":
		category_subdir = file_category
	elif domain == "vision" and cpp_subdir == "FnVideo":
		category_subdir = file_category
	else:
		category_subdir = ""
	# Per-category generated fragments are generator internals. A single
	# package-stable declaration fragment is emitted beside the public umbrella.
	cpp_dir = cpp_subdir.lower()
	category_dir = category_subdir.lower()
	file_stem = camel_file_stem(f"{file_prefix}{file_category}")
	h_path = out_root / "cpp" / "lib" / "oa" / domain.lower() / cpp_dir / category_dir / f"{file_stem}.gen.h"
	cpp_path = out_root / "cpp" / "lib" / "oa" / domain.lower() / cpp_dir / category_dir / f"{file_stem}.gen.cpp"
	# Gradient nodes are organized by semantic value family, not by the public
	# Fn namespace. Matrix operations therefore share the existing matrix node
	# directory with handwritten matrix gradients.
	autograd_subdir = "matrix" if cpp_subdir == "FnMatrix" else cpp_subdir
	autograd_h_path = (
		out_root / "cpp" / "lib" / "oa" / domain.lower() / "autograd"
		/ autograd_subdir.lower()
		/ f"{camel_file_stem(f'Autograd{file_category}')}.gen.h"
	)
	test_root = REPO_ROOT / "test" / "cpp" if live else out_root / "test" / "cpp"
	# Mirror source structure: test/cpp/{domain}/{subdir}/{category}/test{Prefix}{Category}.gen.cpp
	test_path = (
		test_root / domain.lower() / cpp_dir / category_dir
		/ f"{camel_file_stem(f'Test{file_prefix}{file_category}')}.gen.cpp"
	)
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
		emit_autograd=emit_autograd,
	)
