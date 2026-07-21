#!/usr/bin/env python3
"""
check_autogen_drift — the regen-safety gate for OA's code generators.

Run this BEFORE any `--live` regen. It runs every generator (Type / Fn / Nn /
OaTile)
into a throwaway scratch tree and diffs the result against the committed
`.gen.*` files in Source/, Test/, and the internal operation reference when that
documentation tree is present. Any difference means the working tree and the
generators have diverged, which is the *only* way a `--live` regen can silently
destroy work:

  * a .gen.* file was hand-patched but the schema was never updated  → regen
    REVERTS the patch.
  * the schema/generator changed but the tree was never regenerated  → regen
    REWRITES the file (stale tree).

Either way you want to know and reconcile (fold the fix into the schema, or
regen on purpose) — not discover it after the fact.

Exit codes:
  0  clean — `--live` is safe, tree == generator output.
  1  drift — files differ; nothing was written. Resolve before regenerating.

Usage:
  python3 Tools/check_autogen_drift.py            # summary
  python3 Tools/check_autogen_drift.py --verbose  # unified diffs for each file

Stdlib only. The generators are run via the live source root so cmake/source
paths are repo-relative and identical to a real `--live` run.
"""
from __future__ import annotations

import argparse
import difflib
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# (script, scratch-subdir). Each generator mirrors the repo layout under its
# --out root: Fn/Nn write Private/… + Test/…; Type writes Source/….
GENERATORS = [
	("Tools/TypeAutogen/oatypeautogen.py", "ty"),
	("Tools/FnAutogen/oafnautogen.py", "fn"),
	("Tools/NnAutogen/oannautogen.py", "nn"),
	("Tools/OaTile/oatile.py", "tile"),
]

# The release tool adds this marker to sanitized public snapshots. Only marked
# trees may omit generated documentation; generated source and test artifacts
# remain mandatory everywhere.
PUBLIC_SNAPSHOT_MARKER = REPO_ROOT / ".oa-public-snapshot"
PUBLIC_SNAPSHOT_MARKER_CONTENT = b"OA sanitized public source snapshot\n"
PUBLIC_OMITTED_OPERATION_REFERENCE = (
	Path("Docs") / "Internal" / "Architecture" / "OaOperations.gen.md"
)


def tree_target(rel: Path) -> Path:
	"""Map a generated scratch-relative path to its committed counterpart."""
	parts = rel.parts
	if not parts:
		return REPO_ROOT / rel
	# Type generator roots at Source/…; Fn/Nn root at Private|Public|Test/…
	if parts[0] == "Source":
		return REPO_ROOT / rel
	if parts[0] in ("Private", "Public", "Python"):
		return REPO_ROOT / "Source" / rel
	return REPO_ROOT / rel  # Test/… and anything else is repo-relative


def norm(path: Path) -> list[str]:
	return path.read_bytes().replace(b"\r\n", b"\n").decode("utf-8", "replace").splitlines()


def is_sanitized_public_snapshot() -> bool:
	try:
		return (
			PUBLIC_SNAPSHOT_MARKER.read_bytes()
			== PUBLIC_SNAPSHOT_MARKER_CONTENT
		)
	except OSError:
		return False


def intentionally_omitted_public_doc(target: Path) -> bool:
	if not is_sanitized_public_snapshot():
		return False
	try:
		rel = target.relative_to(REPO_ROOT)
	except ValueError:
		return False
	return rel == PUBLIC_OMITTED_OPERATION_REFERENCE and not target.exists()


def run_generators(scratch: Path) -> int:
	rc = 0
	for script, sub in GENERATORS:
		out = scratch / sub
		out.mkdir(parents=True, exist_ok=True)
		# Force UTF-8 so the scratch run can never introduce its own encoding
		# noise on a Windows console (the generators write utf-8 regardless).
		res = subprocess.run(
			[sys.executable, script, "--out", str(out)],
			cwd=REPO_ROOT, capture_output=True, text=True,
		)
		if res.returncode != 0:
			sys.stderr.write(f"[check] generator failed: {script}\n{res.stdout}\n{res.stderr}\n")
			rc |= res.returncode
	return rc


def main() -> int:
	ap = argparse.ArgumentParser(description="Gate that the tree matches generator output before --live regen.")
	ap.add_argument("--verbose", action="store_true", help="Print a unified diff for every drifted file.")
	args = ap.parse_args()

	with tempfile.TemporaryDirectory(prefix="oa_autogen_check_") as tmp:
		scratch = Path(tmp)
		if run_generators(scratch) != 0:
			return 1

		# Regeneration must preserve timestamps when every emitted byte is
		# unchanged. Otherwise an idempotent schema check invalidates Ninja's
		# dependency graph and can rebuild the complete shader/archive surface.
		first_mtimes = {
			path: path.stat().st_mtime_ns
			for path in scratch.rglob("*")
			if path.is_file()
		}
		time.sleep(0.02)
		if run_generators(scratch) != 0:
			return 1
		mtime_drift = [
			path.relative_to(scratch)
			for path, first_mtime in first_mtimes.items()
			if not path.exists() or path.stat().st_mtime_ns != first_mtime
		]
		if mtime_drift:
			print(f"[check] NON-IDEMPOTENT — {len(mtime_drift)} unchanged generated file(s) were rewritten:")
			for path in sorted(mtime_drift):
				print(f"          ! {path.as_posix()}")
			return 1

		drift: list[tuple[Path, list[str], list[str]]] = []
		missing: list[Path] = []  # generator would CREATE (no tree counterpart)
		omitted_public_docs: list[Path] = []
		omission_contract_error = False
		expected_targets: set[Path] = set()
		checked = 0
		for _, sub in GENERATORS:
			root = scratch / sub
			for gen_file in root.rglob("*"):
				if not gen_file.is_file() or ".gen." not in gen_file.name:
					continue
				rel = gen_file.relative_to(root)
				tgt = tree_target(rel)
				expected_targets.add(tgt)
				# FnAutogen never writes <Name>.gen.slang when a hand-written
				# <Name>.slang sits beside it in the live tree — so live regen can
				# neither create nor clobber it. The scratch run can't see that
				# sibling; honor the same rule here (whether or not it exists).
				if gen_file.name.endswith(".gen.slang"):
					manual = tgt.parent / gen_file.name.replace(".gen.slang", ".slang")
					if manual.exists():
						continue
				if not tgt.exists() and intentionally_omitted_public_doc(tgt):
					omitted_public_docs.append(rel)
					continue
				checked += 1
				if not tgt.exists():
					missing.append(rel)
					continue
				a, b = norm(tgt), norm(gen_file)
				if a != b:
					drift.append((rel, b, a))

		# A scratch-only comparison cannot see files that a generator stopped
		# emitting. Catch those stale artifacts explicitly. Limit this to files
		# carrying FnAutogen's exact ownership marker so hand-maintained umbrella
		# headers and outputs from other generators are never guessed at.
		fn_marker = "AUTO-GENERATED by Tools/FnAutogen/oafnautogen.py — DO NOT EDIT."
		fn_markers = {
			f"// {fn_marker}",
			f"<!-- {fn_marker} -->",
		}
		stale: list[Path] = []
		for base in (REPO_ROOT / "Source", REPO_ROOT / "Test", REPO_ROOT / "Docs"):
			for candidate in base.rglob("*.gen.*"):
				if not candidate.is_file() or candidate in expected_targets:
					continue
				try:
					first_line = candidate.open(encoding="utf-8").readline().rstrip("\r\n")
				except (OSError, UnicodeError):
					continue
				if first_line in fn_markers:
					stale.append(candidate.relative_to(REPO_ROOT))

		if is_sanitized_public_snapshot():
			expected_omissions = [PUBLIC_OMITTED_OPERATION_REFERENCE]
			if sorted(omitted_public_docs) != expected_omissions:
				omission_contract_error = True

	print(f"[check] compared {checked} generated .gen.* files")
	if omitted_public_docs:
		print(
			f"[check] skipped {len(omitted_public_docs)} documentation output(s) "
			"intentionally absent from the sanitized public tree:"
		)
		for rel in sorted(omitted_public_docs):
			print(f"          ~ {rel.as_posix()}")
	if missing:
		print(f"[check] {len(missing)} file(s) the generator would CREATE (not in tree):")
		for rel in sorted(missing):
			print(f"          + {rel.as_posix()}")
	if stale:
		print(f"[check] {len(stale)} stale FnAutogen file(s) are no longer emitted:")
		for rel in sorted(stale):
			print(f"          - {rel.as_posix()}")
	if omission_contract_error:
		print("[check] sanitized-snapshot omission contract mismatch:")
		print(f"          expected: {PUBLIC_OMITTED_OPERATION_REFERENCE.as_posix()}")
		observed = ", ".join(path.as_posix() for path in sorted(omitted_public_docs))
		print(f"          observed: {observed or '<none>'}")
	if not drift:
		if missing or stale or omission_contract_error:
			return 1
		print("[check] CLEAN — tree matches generator output. --live regen is safe.")
		return 0

	print(f"[check] DRIFT — {len(drift)} file(s) differ from generator output.")
	print("        A --live regen would overwrite these. Reconcile (fold the manual")
	print("        change into the schema, or regen on purpose) before regenerating.\n")
	for rel, gen_lines, tree_lines in sorted(drift, key=lambda d: d[0].as_posix()):
		print(f"  DRIFT  {rel.as_posix()}")
		if args.verbose:
			for line in difflib.unified_diff(gen_lines, tree_lines,
			                                 fromfile="generator", tofile="tree",
			                                 lineterm="", n=1):
				print(f"    {line}")
			print()
	return 1


if __name__ == "__main__":
	sys.exit(main())
