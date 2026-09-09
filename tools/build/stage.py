#!/usr/bin/env python3
"""Stage OA's runnable Cargo targets without copying Cargo's build tree."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BIN_ROOT = ROOT / "bin"


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser()
	parser.add_argument("--profile", choices=("debug", "release"))
	parser.add_argument("--target", action="append", default=[])
	parser.add_argument(
		"--tests",
		action="store_true",
		help="build and stage only the declared Rust integration-test executables",
	)
	parser.add_argument("--clean", action="store_true")
	return parser.parse_args()


def clean(profile: str | None) -> None:
	destination = BIN_ROOT / profile if profile else BIN_ROOT
	if destination.exists():
		shutil.rmtree(destination)
		print(f"removed {destination.relative_to(ROOT)}")


def cargo_metadata() -> dict[str, object]:
	completed = subprocess.run(
		("cargo", "metadata", "--format-version=1", "--no-deps"),
		cwd=ROOT,
		check=True,
		capture_output=True,
		text=True,
	)
	return json.loads(completed.stdout)


def destination_directory(source: Path, kind: str) -> Path:
	relative = source.relative_to(ROOT)
	parts = relative.parts
	if len(parts) >= 4 and parts[:2] == ("sdk", "rs"):
		return Path("sdk", *parts[2:-1])
	if kind == "example":
		return Path("examples")
	return Path()


def test_destination(source: Path, executable_suffix: str) -> Path:
	relative = source.relative_to(ROOT)
	parts = relative.parts
	if len(parts) != 3 or parts[:2] != ("test", "rs"):
		raise ValueError(f"Rust test suite must live directly under test/rs: {relative}")
	stem = source.stem
	module = stem.removeprefix("test_")
	return Path("test", module, f"{stem}{executable_suffix}")


def cargo_test_executables(profile: str, package_id: str) -> dict[str, Path]:
	command = [
		"cargo",
		"test",
		"--no-run",
		"--all-features",
		"--message-format=json",
	]
	if profile == "release":
		command.insert(2, "--release")
	completed = subprocess.run(
		command,
		cwd=ROOT,
		check=True,
		stdout=subprocess.PIPE,
		text=True,
	)
	executables: dict[str, Path] = {}
	for line in completed.stdout.splitlines():
		message = json.loads(line)
		target = message.get("target", {})
		executable = message.get("executable")
		if (
			message.get("reason") != "compiler-artifact"
			or message.get("package_id") != package_id
			or "test" not in target.get("kind", [])
			or executable is None
		):
			continue
		executables[target["name"]] = Path(executable)
	return executables


def stage(profile: str, selected_targets: set[str], include_tests: bool) -> None:
	metadata = cargo_metadata()
	manifest_path = str(ROOT / "Cargo.toml")
	package = next(
		package
		for package in metadata["packages"]
		if package["manifest_path"] == manifest_path
	)
	target_directory = Path(metadata["target_directory"])
	executable_suffix = ".exe" if sys.platform == "win32" else ""
	staged: set[str] = set()

	if not include_tests:
		for target in package["targets"]:
			kind = next((kind for kind in target["kind"] if kind in {"bin", "example"}), None)
			name = target["name"]
			if kind is None or selected_targets and name not in selected_targets:
				continue

			source_directory = target_directory / profile
			if kind == "example":
				source_directory /= "examples"
			source = source_directory / f"{name}{executable_suffix}"
			if not source.is_file():
				raise FileNotFoundError(
					f"Cargo executable is missing: {source}; build target {name!r} first"
				)

			destination = (
				BIN_ROOT
				/ profile
				/ destination_directory(Path(target["src_path"]), kind)
				/ source.name
			)
			destination.parent.mkdir(parents=True, exist_ok=True)
			shutil.copy2(source, destination)
			staged.add(name)
			print(f"staged {destination.relative_to(ROOT)}")

	if include_tests:
		test_executables = cargo_test_executables(profile, package["id"])
		for target in package["targets"]:
			name = target["name"]
			if "test" not in target["kind"] or selected_targets and name not in selected_targets:
				continue
			source = test_executables.get(name)
			if source is None or not source.is_file():
				raise FileNotFoundError(f"Cargo test executable is missing for target {name!r}")
			destination = (
				BIN_ROOT
				/ profile
				/ test_destination(Path(target["src_path"]), executable_suffix)
			)
			destination.parent.mkdir(parents=True, exist_ok=True)
			shutil.copy2(source, destination)
			staged.add(name)
			print(f"staged {destination.relative_to(ROOT)}")

	missing = selected_targets - staged
	if missing:
		raise ValueError(f"unknown or non-runnable Cargo target(s): {', '.join(sorted(missing))}")


def main() -> None:
	args = parse_args()
	if args.clean:
		clean(args.profile)
		return
	if args.profile is None:
		raise ValueError("--profile is required when staging executables")
	stage(args.profile, set(args.target), args.tests)


if __name__ == "__main__":
	main()
