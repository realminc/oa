#!/usr/bin/env python3
"""Build and stage one OA profile, optionally installing the Python extension."""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
import venv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VENV = ROOT / ".venv"


def run(*command: str, cwd: Path = ROOT, env: dict[str, str] | None = None) -> None:
	print(f"+ {' '.join(command)}", flush=True)
	subprocess.run(command, cwd=cwd, env=env, check=True)


def stop_staged_processes(profile: str) -> None:
	"""Stop only this user's processes executing from the selected staged tree."""
	process_root = Path("/proc")
	if not process_root.is_dir():
		return
	staged_root = (ROOT / "bin" / profile).resolve()
	pids: list[int] = []
	for entry in process_root.iterdir():
		if not entry.name.isdecimal():
			continue
		try:
			if entry.stat().st_uid != os.getuid():
				continue
			executable = os.readlink(entry / "exe").removesuffix(" (deleted)")
			Path(executable).relative_to(staged_root)
			pid = int(entry.name)
			if pid != os.getpid():
				pids.append(pid)
		except (OSError, ValueError):
			continue
	for pid in pids:
		try:
			print(f"stopping staged {profile} process {pid}", flush=True)
			os.kill(pid, signal.SIGTERM)
		except OSError:
			pass
	def still_running(pid: int) -> bool:
		try:
			state = (process_root / str(pid) / "stat").read_text().rsplit(") ", 1)[1][0]
			return state not in {"Z", "X"}
		except (OSError, IndexError):
			return False
	deadline = time.monotonic() + 3.0
	while pids and time.monotonic() < deadline:
		pids = [pid for pid in pids if still_running(pid)]
		if pids:
			time.sleep(0.1)
	for pid in pids:
		try:
			# Recheck the executable before a forceful signal in case the PID changed.
			executable = os.readlink(process_root / str(pid) / "exe").removesuffix(" (deleted)")
			Path(executable).relative_to(staged_root)
			os.kill(pid, signal.SIGKILL)
		except (OSError, ValueError):
			pass


def venv_python() -> Path:
	return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def install_python(profile: str, clean: bool) -> None:
	if clean and VENV.exists():
		shutil.rmtree(VENV)
	if not venv_python().is_file():
		print(f"creating {VENV.relative_to(ROOT)}", flush=True)
		venv.EnvBuilder(with_pip=True).create(VENV)
	python = str(venv_python())
	maturin = VENV / ("Scripts/maturin.exe" if os.name == "nt" else "bin/maturin")
	patchelf = VENV / ("Scripts/patchelf.exe" if os.name == "nt" else "bin/patchelf")
	if not maturin.is_file() or (sys.platform == "linux" and not patchelf.is_file()):
		package = "maturin[patchelf]>=1.9,<2" if sys.platform == "linux" else "maturin>=1.9,<2"
		run(python, "-m", "pip", "install", package)
	env = os.environ.copy()
	env["VIRTUAL_ENV"] = str(VENV)
	env["PATH"] = f"{maturin.parent}{os.pathsep}{env.get('PATH', '')}"
	command = [str(maturin), "develop"]
	if profile == "release":
		command.append("--release")
	run(*command, cwd=ROOT, env=env)
	# Import in a fresh process so an already-loaded extension cannot mask drift.
	run(
		python, "-c",
		"from importlib.metadata import version\n"
		"import oa\n"
		"from oa import _native\n"
		"installed = version('oapython')\n"
		"if _native.__version__ != installed:\n"
		"    raise RuntimeError(f'OA native version {_native.__version__} differs from installed {installed}')\n"
		"print(f'OA Python {installed}: native imports verified')\n",
		env=env,
	)


def build(profile: str, clean: bool, python: bool, python_only: bool) -> None:
	if python_only:
		install_python(profile, clean)
		return
	stop_staged_processes(profile)
	if clean:
		run("cargo", "clean", "--profile", "dev" if profile == "debug" else "release")
		run(sys.executable, "tool/build/stage.py", "--clean", "--profile", profile)
	command = ["cargo", "build", "--all-targets", "--all-features"]
	if profile == "release":
		command.insert(2, "--release")
	run(*command)
	if not clean:
		run(sys.executable, "tool/build/stage.py", "--clean", "--profile", profile)
	run(sys.executable, "tool/build/stage.py", "--profile", profile)
	run(sys.executable, "tool/build/stage.py", "--profile", profile, "--tests")
	if python:
		install_python(profile, clean)


def clean_all() -> None:
	for profile in ("debug", "release"):
		stop_staged_processes(profile)
	run("cargo", "clean")
	run(sys.executable, "tool/build/stage.py", "--clean")
	if VENV.exists():
		shutil.rmtree(VENV)
		print(f"removed {VENV.relative_to(ROOT)}", flush=True)


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--profile", choices=("debug", "release"))
	parser.add_argument("--clean", action="store_true")
	parser.add_argument("--clean-all", action="store_true", help="remove both build profiles and the Python venv")
	parser.add_argument("--python", action="store_true", help="create a venv and install OA Python")
	parser.add_argument("--python-only", action="store_true", help="install only OA Python")
	args = parser.parse_args()
	if args.python and args.python_only:
		parser.error("--python and --python-only are mutually exclusive")
	if args.clean_all:
		if args.profile or args.clean or args.python or args.python_only:
			parser.error("--clean-all cannot be combined with other options")
		clean_all()
		return
	if args.profile is None:
		parser.error("--profile is required for a build")
	build(args.profile, args.clean, args.python, args.python_only)


if __name__ == "__main__":
	main()
