from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tools.gen.fn.config import DOMAIN_FILE_PREFIX, DOMAIN_NAMESPACE, DOMAIN_SUBDIR, REPO_ROOT


@dataclass(frozen=True)
class SchemaLayout:
	domain: str
	namespace: str
	filePrefix: str
	cppSubdir: str
	headerPath: Path
	cppPath: Path
	autogradHeaderPath: Path
	testPath: Path
	emitHeader: bool = True
	emitCpp: bool = True
	emitAutograd: bool = False


def inferDomain(schemaPath: Path) -> str:
 # Schema directories are lowercase (audio/, vision/, core/, etc.).
 # DOMAIN_NAMESPACE keys are also lowercase since the directory rename.
	name = schemaPath.parent.name.lower()
	return name if name in DOMAIN_NAMESPACE else "core"


def camelFileStem(name: str) -> str:
	return name[:1].lower() + name[1:]


def inferCppSubdir(domain: str, fileCategory: str) -> str:
	subdirRule = DOMAIN_SUBDIR.get(domain, "Matrix")
	if subdirRule != "use_file_category":
		return subdirRule
	if fileCategory.startswith("Hash"):
		return "Hash"
	if fileCategory.startswith("Sign"):
		return "Sign"
	return fileCategory


def buildSchemaLayout(
 schemaPath: Path,
 data: dict,
 fileCategory: str,
 outRoot: Path,
 *,
 live: bool,
 emitHeader: bool = True,
 emitCpp: bool = True,
 emitAutograd: bool = False,
) -> SchemaLayout:
	domain = inferDomain(schemaPath)
	namespace = data.get("namespace", DOMAIN_NAMESPACE.get(domain, "oa::FnMatrix"))
	filePrefix = data.get("file_prefix", DOMAIN_FILE_PREFIX.get(domain, "Matrix"))
	cppSubdir = data.get("cpp_subdir", inferCppSubdir(domain, fileCategory))
	# Generated internals mirror the Fn family and operation category while all
	# physical directories remain lowercase.
	if domain == "ml" and cppSubdir == "FnMatrix":
		categorySubdir = fileCategory
	elif domain == "ml" and cppSubdir == "FnLoss":
	 # Loss may use per-function subdirectories; generate.py owns that split.
		categorySubdir = ""
	elif domain == "core" and cppSubdir == "FnMatrix":
		categorySubdir = fileCategory
	elif domain == "audio" and cppSubdir == "FnAudio":
		categorySubdir = fileCategory
	elif domain == "vision" and cppSubdir == "FnImage":
		categorySubdir = fileCategory
	elif domain == "vision" and cppSubdir == "FnVideo":
		categorySubdir = fileCategory
	else:
		categorySubdir = ""
 # Per-category generated fragments are generator internals. A single
 # package-stable declaration fragment is emitted beside the public umbrella.
	cppDir = cppSubdir.lower()
	categoryDir = categorySubdir.lower()
	fileStem = camelFileStem(f"{filePrefix}{fileCategory}")
	hPath = outRoot / "cpp" / "lib" / "oa" / domain.lower() / cppDir / categoryDir / f"{fileStem}.gen.h"
	cppPath = outRoot / "cpp" / "lib" / "oa" / domain.lower() / cppDir / categoryDir / f"{fileStem}.gen.cpp"
	# Gradient nodes are organized by semantic value family, not by the public
	# Fn namespace. Matrix operations therefore share the existing matrix node
	# directory with handwritten matrix gradients.
	autogradSubdir = "matrix" if cppSubdir == "FnMatrix" else cppSubdir
	autogradHPath = (
	 outRoot / "cpp" / "lib" / "oa" / domain.lower() / "autograd"
	 / autogradSubdir.lower()
	 / f"{camelFileStem(f'Autograd{fileCategory}')}.gen.h"
	)
	testRoot = REPO_ROOT / "test" / "cpp" if live else outRoot / "test" / "cpp"
	# Mirror source structure: test/cpp/{domain}/{subdir}/{category}/test{Prefix}{Category}.gen.cpp
	testPath = (
	 testRoot / domain.lower() / cppDir / categoryDir
	 / f"{camelFileStem(f'Test{filePrefix}{fileCategory}')}.gen.cpp"
	)
	return SchemaLayout(
	 domain=domain,
	 namespace=namespace,
	 filePrefix=filePrefix,
	 cppSubdir=cppSubdir,
	 headerPath=hPath,
	 cppPath=cppPath,
	 autogradHeaderPath=autogradHPath,
	 testPath=testPath,
	 emitHeader=emitHeader,
	 emitCpp=emitCpp,
	 emitAutograd=emitAutograd,
	)
